#include <float.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <time.h>

#include "ray.h"
#include "camera.h"
#include "scene.h"
#include "rnd.h"
#include "options.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

#include "cudautils.h"
#include "metrics.h"

#include "glwindow.h"

#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays

const int MaxBlockWidth = 32;
const int MaxBlockHeight = 2; // block width is 32

typedef unsigned long long ull;

__device__ __constant__ float d_colormap[256 * 3];
__device__ __constant__ bvh_node d_nodes[2048];

texture<float4> t_bvh;
texture<float> t_spheres;
float* d_bvh_buf;
float* d_spheres_buf;
vec3* d_fb;
int* d_colors;
unsigned int* d_cuda_render_buffer;

// Camera controls
camera* cam = NULL;
float c_theta = 80 * kPI / 180;
float c_phi = 45 * kPI / 180;
float c_relative_dist = 1.0f;
const float c_rotation_speed = 1 * kPI / 180;
const float c_zoom_speed = 1.0f / 100;
bool camera_updated = false;

struct render_params {
    vec3* fb;
    int leaf_offset;
    unsigned int width;
    unsigned int height;
    unsigned int spp;
    unsigned int maxActivePaths;

    int* colors;

    int maxBounces;
    float lightRadius;
    vec3 lightColor;

    vec3 skyColor;
};

typedef enum pathstate {
    DONE,           // nothing more to do for this path
    SCATTER,        // path need to traverse the BVH tree
    NO_HIT,         // path didn't hit any primitive
    HIT,            // path hit a primitive
    SHADOW,         // path hit a primitive and generated a shadow ray
    HIT_AND_LIGHT  // path hit a primitive and its shadow ray didn't hit any primitive
} pathstate;

struct paths {
    ull* next_sample; // used by init() to track next sample to fetch
    ull* numsamples_perpixel; // how many samples have been traced per pixel so far

    // pixel_id of active paths currently being traced by the renderer, it's a subset of all_sample_pool
    unsigned int* active_paths;
    unsigned int* next_path; // used by hit_bvh() to track next path to fetch and trace

    ray* r;
    ray* shadow;
    rand_state* state;
    vec3* attentuation;
    vec3* emitted;
    unsigned short* bounce;
    pathstate* pstate;
    int* hit_id;
    vec3* hit_normal;
    float* hit_t;

    metrics m;
};

// Renderer
unsigned int r_iteration = 0;
unsigned int r_num_pixels = 0;
paths* r_paths = NULL;

void setup_paths(paths& p, int nx, int ny, int ns, unsigned int maxActivePaths) {
    // at any given moment only kMaxActivePaths at most are active at the same time
    const unsigned num_paths = maxActivePaths;
    checkCudaErrors(cudaMalloc((void**)& p.r, num_paths * sizeof(ray)));
    checkCudaErrors(cudaMalloc((void**)& p.shadow, num_paths * sizeof(ray)));
    checkCudaErrors(cudaMalloc((void**)& p.state, num_paths * sizeof(rand_state)));
    checkCudaErrors(cudaMalloc((void**)& p.attentuation, num_paths * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)& p.emitted, num_paths * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)& p.bounce, num_paths * sizeof(unsigned short)));
    checkCudaErrors(cudaMalloc((void**)& p.pstate, num_paths * sizeof(pathstate)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_id, num_paths * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_normal, num_paths * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_t, num_paths * sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)& p.active_paths, num_paths * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)& p.next_path, sizeof(unsigned int)));

    checkCudaErrors(cudaMalloc((void**)& p.next_sample, sizeof(ull)));
    checkCudaErrors(cudaMemset((void*)p.next_sample, 0, sizeof(ull)));
    p.m.allocateDeviceMem();

    checkCudaErrors(cudaMalloc((void**)&p.numsamples_perpixel, nx * ny * sizeof(ull)));
    checkCudaErrors(cudaMemset((void*)p.numsamples_perpixel, 0, nx * ny * sizeof(ull)));
}

void free_paths(const paths& p) {
    checkCudaErrors(cudaFree(p.r));
    checkCudaErrors(cudaFree(p.shadow));
    checkCudaErrors(cudaFree(p.state));
    checkCudaErrors(cudaFree(p.attentuation));
    checkCudaErrors(cudaFree(p.emitted));
    checkCudaErrors(cudaFree(p.bounce));
    checkCudaErrors(cudaFree(p.pstate));
    checkCudaErrors(cudaFree(p.hit_id));
    checkCudaErrors(cudaFree(p.hit_normal));
    checkCudaErrors(cudaFree(p.hit_t));
    checkCudaErrors(cudaFree(p.next_sample));

    checkCudaErrors(cudaFree(p.active_paths));
    checkCudaErrors(cudaFree(p.next_path));

    p.m.freeDeviceMem();
    checkCudaErrors(cudaFree(p.numsamples_perpixel));
}

__global__ void fetch_samples(const render_params params, paths p, bool first, const camera cam) {
    // kMaxActivePaths threads are started to fetch the samples from all_sample_pool and initialize the paths
    // to keep things simple a block contains a single warp so that we only need to keep a single shared nextSample per block

    const unsigned int pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid == 0)
        p.next_path[0] = 0;
    p.m.reset(pid, first);
    __syncthreads();

    if (pid >= params.maxActivePaths)
        return;

    rand_state state;
    pathstate pstate;
    if (first) {
        // this is the very first init, all paths are marked terminated, and we don't have a valid random state yet
        state = (wang_hash(pid) * 336343633) | 1;
        pstate = DONE;
    } else {
        state = p.state[pid];
        pstate = p.pstate[pid];
    }

    // generate all terminated paths
    const bool          terminated     = pstate == DONE;
    const unsigned int  maskTerminated = __ballot_sync(__activemask(), terminated);
    const int           numTerminated  = __popc(maskTerminated);
    const int           idxTerminated  = __popc(maskTerminated & ((1u << threadIdx.x) - 1));

    __shared__ volatile ull nextSample;

    if (terminated) {
        // first terminated lane increments next_sample
        if (idxTerminated == 0)
            nextSample = atomicAdd(p.next_sample, numTerminated);

        // compute sample this lane is going to fetch
        const ull sample_id = nextSample + idxTerminated;
        //const ull max_samples = (ull)params.width * (ull)params.height * (ull)params.spp;
        //if (sample_id >= max_samples)
        //    return; // no more samples to fetch

        // retrieve pixel_id corresponding to current path
        const unsigned int pixel_id = (sample_id / params.spp) % (params.width * params.height);
        p.active_paths[pid] = pixel_id;
        atomicAdd(p.numsamples_perpixel + pixel_id, 1);

        // compute pixel coordinates
        const unsigned int x = pixel_id % params.width;
        const unsigned int y = pixel_id / params.width;

        // generate camera ray
        float u = float(x + random_float(state)) / float(params.width);
        float v = float(y + random_float(state)) / float(params.height);
        p.r[pid] = cam.get_ray(u, v, state);
        p.state[pid] = state;
        p.attentuation[pid] = vec3(1, 1, 1);
        p.bounce[pid] = 0;
        p.pstate[pid] = SCATTER;
    }
}

#define IDX_SENTINEL    0
#define IS_DONE(idx)    (idx == IDX_SENTINEL)
#define IS_LEAF(idx)    (idx >= params.leaf_offset)

#define BIT_DONE        3
#define BIT_MASK        3
#define BIT_PARENT      0
#define BIT_LEFT        1
#define BIT_RIGHT       2

__device__ void pop_bitstack(unsigned long long& bitstack, int& idx) {
    const int m = (__ffsll(bitstack) - 1) / 2;
    bitstack >>= (m << 1);
    idx >>= m;

    if (bitstack == BIT_DONE) {
        idx = IDX_SENTINEL;
    }
    else {
        // idx could point to left or right child regardless of sibling we need to go to
        idx = (idx >> 1) << 1; // make sure idx always points to left sibling
        idx += (bitstack & BIT_MASK) - 1; // move idx to the sibling stored in bitstack
        bitstack = bitstack & (~BIT_MASK); // set bitstack to parent, so we can backtrack
    }
}

__global__ void trace_scattered(const render_params params, paths p) {
    // a limited number of threads are started to operate on active_paths

    unsigned int pid = 0; // currently traced path
    ray r; // corresponding ray

    // bvh traversal state
    int idx = IDX_SENTINEL;
    bool found;
    float closest;
    hit_record rec;

    unsigned long long bitstack;

    // Initialize persistent threads.
    // given that each block is 32 thread wide, we can use threadIdx.x as a warpId
    __shared__ volatile int nextPathArray[MaxBlockHeight]; // Current ray index in global buffer.
    __shared__ volatile bool noMorePaths[MaxBlockHeight]; // true when no more paths are available to fetch

    // Persistent threads: fetch and process rays in a loop.

    while (true) {
        const int tidx = threadIdx.x;
        volatile int& pathBase = nextPathArray[threadIdx.y];
        volatile bool& noMoreP = noMorePaths[threadIdx.y];
        pathstate pstate;

        // identify which lanes are done
        const bool          terminated      = IS_DONE(idx);
        const unsigned int  maskTerminated  = __ballot_sync(__activemask(), terminated);
        const int           numTerminated   = __popc(maskTerminated);
        const int           idxTerminated   = __popc(maskTerminated & ((1u << tidx) - 1));

        if (terminated) {
            // first terminated lane updates the base ray index
            if (idxTerminated == 0) {
                pathBase = atomicAdd(p.next_path, numTerminated);
                noMoreP = (pathBase + numTerminated) >= params.maxActivePaths;
            }

            pid = pathBase + idxTerminated;
            if (pid >= params.maxActivePaths) {
                return;
            }

            found = false; // always reset found to avoid writing hit information for terminated paths
            // setup ray if path not already terminated
            pstate = p.pstate[pid];
            if (pstate == SCATTER) {
                // Fetch ray
                r = p.r[pid];

                // idx is already set to IDX_SENTINEL, but make sure we set found to false
                idx = 1;
                closest = FLT_MAX;
                bitstack = BIT_DONE;
            }
        }

        // traversal
        while (!IS_DONE(idx)) {
            //p.m.lanes_cnt.increment(tidx);

            // we already intersected ray with idx node, now we need to load its children and intersect the ray with them
            if (!IS_LEAF(idx)) {
                // load left, right nodes
                bvh_node left, right;
                const int idx2 = idx * 2; // we are going to load and intersect children of idx
                if (idx2 < 2048) {
                    left = d_nodes[idx2];
                    right = d_nodes[idx2 + 1];
                }
                else {
                    // each spot in the texture holds two children, that's why we devide the relative texture index by 2
                    unsigned int tex_idx = ((idx2 - 2048) >> 1) * 3;
                    float4 a = tex1Dfetch(t_bvh, tex_idx++);
                    float4 b = tex1Dfetch(t_bvh, tex_idx++);
                    float4 c = tex1Dfetch(t_bvh, tex_idx++);
                    left = bvh_node(a.x, a.y, a.z, a.w, b.x, b.y);
                    right = bvh_node(b.z, b.w, c.x, c.y, c.z, c.w);
                }

                const float left_t = hit_bbox(left, r, closest);
                const bool traverse_left = left_t < FLT_MAX;
                const float right_t = hit_bbox(right, r, closest);
                const bool traverse_right = right_t < FLT_MAX;

                const bool swap = right_t < left_t; // right child is closer

                if (traverse_left || traverse_right) {
                    idx = idx2 + swap; // intersect closer node next
                    if (traverse_left && traverse_right) // push farther node into the stack
                        bitstack = (bitstack << 2) + (swap ? BIT_LEFT : BIT_RIGHT);
                    else // push parent bit to the stack to backtrack later
                        bitstack = (bitstack << 2) + BIT_PARENT;
                }
                else {
                    pop_bitstack(bitstack, idx);
                }
            } else {
                int m = (idx - params.leaf_offset) * lane_size_float;
                #pragma unroll
                for (int i = 0; i < lane_size_spheres; i++) {
                    float x = tex1Dfetch(t_spheres, m++);
                    float y = tex1Dfetch(t_spheres, m++);
                    float z = tex1Dfetch(t_spheres, m++);
                    vec3 center(x, y, z);
                    if (hit_point(center, r, 0.001f, closest, rec)) {
                        found = true;
                        closest = rec.t;
                        rec.idx = (idx - params.leaf_offset) * lane_size_spheres + i;
                    }
                }

                if (found) // exit traversal once we find an intersection in any leaf
                    idx = IDX_SENTINEL;
                else
                    pop_bitstack(bitstack, idx);
            }

            // some lanes may have already exited the loop, if not enough active thread are left, exit the loop
            if (!noMoreP && __popc(__activemask()) < DYNAMIC_FETCH_THRESHOLD)
                break;
        }

        if (pstate == SCATTER && IS_DONE(idx)) {
            if (found) {
                // finished traversing bvh
                p.hit_id[pid] = rec.idx;
                p.hit_normal[pid] = rec.n;
                p.hit_t[pid] = rec.t;
                p.pstate[pid] = HIT;
            } else {
                p.pstate[pid] = NO_HIT;
            }
        }
    }
}

// generate shadow rays for all non terminated rays with intersections
__global__ void generate_shadow_rays(const render_params params, paths p) {

    const vec3 light_center(5000, 0, 0);
    const float light_radius = params.lightRadius;
    const vec3 light_color = params.lightColor;

    // kMaxActivePaths threads update all p.num_active_paths
    const unsigned int pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid == 0)
        p.next_path[0] = 0;
    __syncthreads();

    if (pid >= params.maxActivePaths)
        return;

    // if the path has no intersection, which includes terminated paths, do nothing
    if (p.pstate[pid] != HIT)
        return;

    const ray r = p.r[pid];
    const float hit_t = p.hit_t[pid];
    const vec3 hit_p = r.point_at_parameter(hit_t);
    const vec3 hit_n = p.hit_normal[pid];
    rand_state state = p.state[pid];

    // create a random direction towards the light
    // coord system for sampling
    const vec3 sw = unit_vector(light_center - hit_p);
    const vec3 su = unit_vector(cross(fabs(sw.x()) > 0.01f ? vec3(0, 1, 0) : vec3(1, 0, 0), sw));
    const vec3 sv = cross(sw, su);

    // sample sphere by solid angle
    const float cosAMax = sqrt(1.0f - light_radius * light_radius / (hit_p - light_center).squared_length());
    const float eps1 = random_float(state);
    const float eps2 = random_float(state);
    const float cosA = 1.0f - eps1 + eps1 * cosAMax;
    const float sinA = sqrt(1.0f - cosA * cosA);
    const float phi = 2 * kPI * eps2;
    const vec3 l = unit_vector(su * cosf(phi) * sinA + sv * sinf(phi) * sinA + sw * cosA);

    p.state[pid] = state;
    const float dotl = dot(l, hit_n);
    if (dotl <= 0)
        return;

    const float omega = 2 * kPI * (1.0f - cosAMax);
    p.shadow[pid] = ray(hit_p, l);
    p.emitted[pid] = light_color * dotl * omega / kPI;
    p.pstate[pid] = SHADOW;
}

// traces all paths that have FLAG_HAS_SHADOW set, sets FLAG_SHADOW_HIT to true if there is a hit
__global__ void trace_shadows(const render_params params, paths p) {
    // a limited number of threads are started to operate on active_paths

    unsigned int pid = 0; // currently traced path
    ray r; // corresponding ray

    // bvh traversal state
    int idx = IDX_SENTINEL;
    bool found = false;
    hit_record rec;

    unsigned long long bitstack;

    // Initialize persistent threads.
    // given that each block is 32 thread wide, we can use threadIdx.x as a warpId
    __shared__ volatile int nextPathArray[MaxBlockHeight]; // Current ray index in global buffer.

    // Persistent threads: fetch and process rays in a loop.

    while (true) {
        const int tidx = threadIdx.x;
        volatile int& pathBase = nextPathArray[threadIdx.y];
        pathstate pstate;

        // identify which lanes are done
        const bool          terminated = IS_DONE(idx);
        const unsigned int  maskTerminated = __ballot_sync(__activemask(), terminated);
        const int           numTerminated = __popc(maskTerminated);
        const int           idxTerminated = __popc(maskTerminated & ((1u << tidx) - 1));

        if (terminated) {
            // first terminated lane updates the base ray index
            if (idxTerminated == 0)
                pathBase = atomicAdd(p.next_path, numTerminated);

            pid = pathBase + idxTerminated;
            if (pid >= params.maxActivePaths)
                return;

            // setup ray if path has a shadow ray
            pstate = p.pstate[pid];
            if (pstate == SHADOW) {
                // Fetch ray
                r = p.shadow[pid];

                // idx is already set to IDX_SENTINEL, but make sure we set found to false
                found = false;
                idx = 1;
                bitstack = BIT_DONE;
            }
        }

        // traversal
        while (!IS_DONE(idx)) {
            // we already intersected ray with idx node, now we need to load its children and intersect the ray with them
            if (!IS_LEAF(idx)) {
                // load left, right nodes
                bvh_node left, right;
                const int idx2 = idx * 2; // we are going to load and intersect children of idx
                if (idx2 < 2048) {
                    left = d_nodes[idx2];
                    right = d_nodes[idx2 + 1];
                }
                else {
                    // each spot in the texture holds two children, that's why we devide the relative texture index by 2
                    unsigned int tex_idx = ((idx2 - 2048) >> 1) * 3;
                    float4 a = tex1Dfetch(t_bvh, tex_idx++);
                    float4 b = tex1Dfetch(t_bvh, tex_idx++);
                    float4 c = tex1Dfetch(t_bvh, tex_idx++);
                    left = bvh_node(a.x, a.y, a.z, a.w, b.x, b.y);
                    right = bvh_node(b.z, b.w, c.x, c.y, c.z, c.w);
                }

                const float left_t = hit_bbox(left, r, FLT_MAX);
                const bool traverse_left = left_t < FLT_MAX;
                const float right_t = hit_bbox(right, r, FLT_MAX);
                const bool traverse_right = right_t < FLT_MAX;

                const bool swap = right_t < left_t; // right child is closer

                if (traverse_left || traverse_right) {
                    idx = idx2 + swap; // intersect closer node next
                    if (traverse_left && traverse_right) // push farther node into the stack
                        bitstack = (bitstack << 2) + (swap ? BIT_LEFT : BIT_RIGHT);
                    else // push parent bit to the stack to backtrack later
                        bitstack = (bitstack << 2) + BIT_PARENT;
                }
                else {
                    pop_bitstack(bitstack, idx);
                }
            }
            else {
                int m = (idx - params.leaf_offset) * lane_size_float;
                #pragma unroll
                for (int i = 0; i < lane_size_spheres && !found; i++) {
                    float x = tex1Dfetch(t_spheres, m++);
                    float y = tex1Dfetch(t_spheres, m++);
                    float z = tex1Dfetch(t_spheres, m++);
                    vec3 center(x, y, z);
                    found = hit_point(center, r, 0.001f, FLT_MAX, rec);
                }

                if (found) // exit traversal once we find an intersection in any leaf
                    idx = IDX_SENTINEL;
                else
                    pop_bitstack(bitstack, idx);
            }

            // some lanes may have already exited the loop, if not enough active thread are left, exit the loop
            if (__popc(__activemask()) < DYNAMIC_FETCH_THRESHOLD) {
                break;
            }
        }

        if (pstate == SHADOW)
            p.pstate[pid] = found ? HIT : HIT_AND_LIGHT;
    }
}

// http://chilliant.blogspot.com.au/2012/08/srgb-approximations-for-hlsl.html
__host__ __device__ uint32_t LinearToSRGB(float x)
{
    x = max(x, 0.0f);
    x = max(1.055f * powf(x, 0.416666667f) - 0.055f, 0.0f);
    uint32_t u = min((uint32_t)(x * 255.9f), 255u);
    return u;
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
    r = LinearToSRGB(r);
    g = LinearToSRGB(g);
    b = LinearToSRGB(b);
    return (int(b) << 16) | (int(g) << 8) | int(r);
}

__global__ void copyToUintBuffer(const render_params params, ull* numsamples_perpixel, unsigned int* uint_render_buffer) {
    const unsigned int pixel_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixel_id >= (params.width * params.height))
        return;
    const vec3 pixel = params.fb[pixel_id];
    const ull spp = numsamples_perpixel[pixel_id];
    uint_render_buffer[pixel_id] = rgbToInt(pixel.r() / spp, pixel.g() / spp, pixel.b() / spp);
}

// for all non terminated rays, accounts for shadow hit, compute scattered ray and resets the flag
__global__ void update(const render_params params, paths p) {
    const vec3 sky_emissive = params.skyColor;

    // kMaxActivePaths threads update all p.num_active_paths
    const unsigned int pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= params.maxActivePaths)
        return;

    // is the path already done ?
    pathstate pstate = p.pstate[pid];
    if (pstate == DONE)
        return; // yup, done and already taken care of
    unsigned short bounce = p.bounce[pid];

    // did the ray hit a primitive ?
    if (pstate == HIT || pstate == HIT_AND_LIGHT) {
        // update path attenuation
        const int hit_id = p.hit_id[pid];
        int clr_idx = params.colors[hit_id] * 3;
        const vec3 albedo = vec3(d_colormap[clr_idx++], d_colormap[clr_idx++], d_colormap[clr_idx++]);
        
        vec3 attenuation = p.attentuation[pid] * albedo;
        p.attentuation[pid] = attenuation;

        // account for light contribution if no shadow hit
        if (pstate == HIT_AND_LIGHT) {
            const vec3 incoming = p.emitted[pid] * attenuation;
            const unsigned int pixel_id = p.active_paths[pid];
            atomicAdd(params.fb[pixel_id].e, incoming.e[0]);
            atomicAdd(params.fb[pixel_id].e + 1, incoming.e[1]);
            atomicAdd(params.fb[pixel_id].e + 2, incoming.e[2]);
        }

        // scatter ray, only if we didn't reach kMaxBounces
        bounce++;
        if (bounce < params.maxBounces) {
            const ray r = p.r[pid];
            const float hit_t = p.hit_t[pid];
            const vec3 hit_p = r.point_at_parameter(hit_t);

            const vec3 hit_n = p.hit_normal[pid];
            rand_state state = p.state[pid];
            const vec3 target = hit_n + random_in_unit_sphere(state);

            p.r[pid] = ray(hit_p, target);
            p.state[pid] = state;
            pstate = SCATTER;
        } else {
            pstate = DONE;
        }
    }
    else {
        if (bounce > 0) {
            const vec3 incoming = p.attentuation[pid] * sky_emissive;
            const unsigned int pixel_id = p.active_paths[pid];
            atomicAdd(params.fb[pixel_id].e, incoming.e[0]);
            atomicAdd(params.fb[pixel_id].e + 1, incoming.e[1]);
            atomicAdd(params.fb[pixel_id].e + 2, incoming.e[2]);
        }
        pstate = DONE;
    }

    p.pstate[pid] = pstate;
    p.bounce[pid] = bounce;
}

__global__ void print_metrics(metrics m, unsigned int iteration, unsigned int maxActivePaths, float elapsedSeconds, bool last) {
    m.print(iteration, elapsedSeconds, last);
}

void copySceneToDevice(const scene& sc, int** d_colors) {
    // copy the first 2048 nodes to constant memory
    const int const_size = min(2048, sc.bvh_size);
    checkCudaErrors(cudaMemcpyToSymbol(d_nodes, sc.bvh, const_size * sizeof(bvh_node)));

    // copy remaining nodes to global memory
    int remaining = sc.bvh_size - const_size;
    if (remaining > 0) {
        // declare and allocate memory
        const int buf_size_bytes = remaining * 6 * sizeof(float);
        checkCudaErrors(cudaMalloc(&d_bvh_buf, buf_size_bytes));
        checkCudaErrors(cudaMemcpy(d_bvh_buf, (void*)(sc.bvh + const_size), buf_size_bytes, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaBindTexture(NULL, t_bvh, (void*)d_bvh_buf, buf_size_bytes));
    }

    // copying spheres to texture memory
    const int spheres_size_float = lane_size_float * (sc.spheres_size / lane_size_spheres);

    // copy the spheres in array of floats
    // do it after we build the BVH as it would have moved the spheres around
    float* floats = new float[spheres_size_float];
    int* colors = new int[sc.spheres_size];
    int idx = 0;
    int i = 0;
    while (i < sc.spheres_size) {
        for (int j = 0; j < lane_size_spheres; j++, i++) {
            floats[idx++] = sc.spheres[i].center.x();
            floats[idx++] = sc.spheres[i].center.y();
            floats[idx++] = sc.spheres[i].center.z();
            colors[i] = sc.spheres[i].color;
        }
        idx += lane_padding_float; // padding
    }
    assert(idx == scene_size_float);

    checkCudaErrors(cudaMalloc((void**)d_colors, sc.spheres_size * sizeof(int)));
    checkCudaErrors(cudaMemcpy(*d_colors, colors, sc.spheres_size * sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)& d_spheres_buf, spheres_size_float * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_spheres_buf, floats, spheres_size_float * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaBindTexture(NULL, t_spheres, (void*)d_spheres_buf, spheres_size_float * sizeof(float)));

    delete[] floats;
    delete[] colors;
}

void releaseScene(int* d_colors) {
    // destroy texture object
    checkCudaErrors(cudaUnbindTexture(t_bvh));
    checkCudaErrors(cudaUnbindTexture(t_spheres));
    checkCudaErrors(cudaFree(d_bvh_buf));
    checkCudaErrors(cudaFree(d_spheres_buf));
    checkCudaErrors(cudaFree(d_colors));
}

void setup_camera(int nx, int ny, float dist) {
    vec3 lookfrom(dist, dist, dist);
    vec3 lookat(0, 0, 0);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.1;
    cam = new camera(lookfrom,
        lookat,
        vec3(0, 1, 0),
        30.0,
        float(nx) / float(ny),
        aperture,
        dist_to_focus);
    cam->look_from(c_theta, c_phi, c_relative_dist);
}

void write_image(const char* output_file, const vec3 *fb, const int nx, const int ny, const int ns) {
    unsigned int* idata = new unsigned int[nx * ny];
    checkCudaErrors(cudaMemcpy(idata, d_cuda_render_buffer, nx * ny * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    char* data = new char[nx * ny * 3];
    int idx = 0;
    for (size_t i = 0; i < nx * ny; i++) {
        unsigned int pixel = idata[i];
        data[idx++] = pixel & 0xFF;
        data[idx++] = (pixel & 0xFF00) >> 8;
        data[idx++] = (pixel & 0xFF0000) >> 16;
    }
    stbi_write_png(output_file, nx, ny, 3, (void*)data, nx * 3);

    delete[] idata;
    delete[] data;
}

int cmpfunc(const void * a, const void * b) {
    if (*(double*)a > *(double*)b)
        return 1;
    else if (*(double*)a < *(double*)b)
        return -1;
    else
        return 0;
}

void initCuda(const options opt) {
    int num_pixels = opt.nx * opt.ny;
    size_t fb_size = num_pixels * sizeof(vec3);
    checkCudaErrors(cudaMalloc((void**)&d_fb, fb_size));
    checkCudaErrors(cudaMemset(d_fb, 0, fb_size));
}

void loadColormap(const char* filename) {
    vector<vector<float>> data = parse2DCsvFile(filename);
    float* colormap = new float[data.size() * 3];
    int idx = 0;
    for (auto l : data) {
        colormap[idx++] = (float)l[0];
        colormap[idx++] = (float)l[1];
        colormap[idx++] = (float)l[2];
    }

    // copy colors to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(d_colormap, colormap, 256 * 3 * sizeof(float)));
    delete[] colormap;
}

void saveImage(const options opt, const char* filename) {
    int num_pixels = opt.nx * opt.ny;
    size_t fb_size = num_pixels * sizeof(vec3);
    vec3* h_fb = new vec3[fb_size];
    checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
    write_image(filename, h_fb, opt.nx, opt.ny, opt.ns);
    delete[] h_fb;
}

int loadScene(const options opt) {
    scene sc;
    if (!opt.binary) {
        load_from_csv(opt.input, sc);
        store_to_binary(strcat(opt.input, ".bin"), sc);
    }
    else {
        load_from_binary(opt.input, sc);
    }
    copySceneToDevice(sc, &d_colors);
    sc.release();

    return sc.bvh_size;
}

render_params setupRenderParams(options opt, int bvh_size) {
    render_params params;
    params.fb = d_fb;
    params.leaf_offset = bvh_size / 2;
    params.colors = d_colors;
    params.width = opt.nx;
    params.height = opt.ny;
    params.spp = opt.ns;
    params.maxActivePaths = opt.maxActivePaths;
    return params;
}

void renderIteration(const options& opt, const render_params& params, const paths& p, const camera& cam, unsigned int iteration, bool lightEnabled) {

    // init kMaxActivePaths using equal number of threads
    {
        const int threads = 32; // 1 warp per block
        const int blocks = (opt.maxActivePaths + threads - 1) / threads;
        fetch_samples <<< blocks, threads >>> (params, p, iteration == 0, cam);
        checkCudaErrors(cudaGetLastError());
    }

    // traverse bvh
    {
        dim3 blocks(6400 * 2, 1);
        dim3 threads(MaxBlockWidth, MaxBlockHeight);
        trace_scattered <<< blocks, threads >>> (params, p);
        checkCudaErrors(cudaGetLastError());
    }

    // generate shadow rays
    if (lightEnabled)
    {
        const int threads = 128;
        const int blocks = (opt.maxActivePaths + threads - 1) / threads;
        generate_shadow_rays <<< blocks, threads >>> (params, p);
        checkCudaErrors(cudaGetLastError());
    }

    // trace shadow rays
    if (lightEnabled)
    {
        dim3 blocks(6400 * 2, 1);
        dim3 threads(MaxBlockWidth, MaxBlockHeight);
        trace_shadows <<< blocks, threads >>> (params, p);
        checkCudaErrors(cudaGetLastError());
    }

    // update paths accounting for intersection and light contribution
    {
        const int threads = 128;
        const int blocks = (opt.maxActivePaths + threads - 1) / threads;
        update <<< blocks, threads >>> (params, p);
        checkCudaErrors(cudaGetLastError());
    }
}

void resetRenderer() {
    r_iteration = 0;
    checkCudaErrors(cudaMemset(d_fb, 0, r_num_pixels * sizeof(vec3)));
    checkCudaErrors(cudaMemset((void*)r_paths->next_sample, 0, sizeof(ull)));
    checkCudaErrors(cudaMemset((void*)r_paths->numsamples_perpixel, 0, r_num_pixels * sizeof(ull)));
}

void render(const options& opt, render_params& params, const paths& p, camera& cam) {
    static GuiParams guiParams;
    bool guiChanged = true;
    bool lightEnabled = true;
    clock_t start = clock();
    cudaProfilerStart();

    while (!pollWindowEvents()) {
        if (camera_updated || guiChanged) {
            cam.look_from(c_theta, c_phi, c_relative_dist);
            resetRenderer();
            camera_updated = false;
            guiChanged = false;

            params.maxBounces = guiParams.maxBounces;
            params.lightRadius = guiParams.lightRadius;
            params.lightColor = vec3(guiParams.lightColor[0], guiParams.lightColor[1], guiParams.lightColor[2]) * guiParams.lightIntensity;
            lightEnabled = guiParams.lightIntensity > 0;
            params.skyColor = vec3(guiParams.skyColor[0], guiParams.skyColor[1], guiParams.skyColor[2]) * guiParams.skyIntensity;
        }

        renderIteration(opt, params, p, cam, r_iteration, lightEnabled);

        {
            const int threads = 128;
            const int blocks = (params.width * params.height + threads - 1) / threads;
            copyToUintBuffer <<< blocks, threads >>> (params, p.numsamples_perpixel, d_cuda_render_buffer);

            updateWindow(guiParams, guiChanged);
        }

        // print metrics
        if (opt.verbose) {
            print_metrics <<< 1, 1 >>> (p.m, r_iteration, opt.maxActivePaths, (float)(clock() - start) / CLOCKS_PER_SEC, false);
            checkCudaErrors(cudaGetLastError());
        }

        r_iteration++;
    }
    cudaProfilerStop();

    print_metrics <<< 1, 1 >>> (p.m, r_iteration, opt.maxActivePaths, (float)(clock() - start) / CLOCKS_PER_SEC, true);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaDeviceSynchronize());
}

void mouseMove(int dx, int dy, int mouse_btn) {
    if (mouse_btn == MOUSE_LEFT) {
        c_theta += -dy * c_rotation_speed;
        //if (theta < delta) theta = delta;
        //if (theta > (kPI/2 - delta)) theta = kPI/2 - delta;
        c_phi += -dx * c_rotation_speed;
    }
    else {
        // drag with right button changes camera distance
        // only x movement is taken into account
        c_relative_dist += dx * c_zoom_speed;
        if (c_relative_dist < 0.1f)
            c_relative_dist = 0.1f;
    }
    camera_updated = true;
}

int main(int argc, char** argv) {
    options opt;
    if (!parse_args(argc, argv, opt))
        return -1;

    initWindow(argc, argv, opt.nx, opt.ny, &d_cuda_render_buffer);
    registerMouseMoveFunc(mouseMove);

    initCuda(opt);
    loadColormap(opt.colormap);

    r_num_pixels = opt.nx * opt.ny;

    const int bvh_size = loadScene(opt);

    setup_camera(opt.nx, opt.ny, opt.dist);

    render_params params = setupRenderParams(opt, bvh_size);

    r_paths = new paths();
    setup_paths(*r_paths, opt.nx, opt.ny, opt.ns, opt.maxActivePaths);

    cout << "started renderer\n" << std::flush;

    render(opt, params, *r_paths, *cam);

    char imagename[100];
    sprintf(imagename, "%s_%dx%dx%d_%d_bvh.png", opt.input, opt.nx, opt.ny, opt.ns, opt.dist);
    saveImage(opt, imagename);

    // clean up
    destroyWindow();

    delete cam;
    free_paths(*r_paths);
    delete r_paths;
    releaseScene(d_colors);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(params.fb));

    cudaDeviceReset();
}