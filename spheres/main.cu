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

typedef unsigned long long ull;

// ---- constants ----

#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays

const int TRAVERSAL_MAX_BLOCK_WIDTH = 32;
const int TRAVERSAL_MAX_BLOCK_HEIGHT = 2; // block width is 32

const unsigned int PREVIEW_WIDTH = 512;
const unsigned int PREVIEW_HEIGHT = 512;

// ---- constant memory ----

__device__ __constant__ float d_colormap[256 * 3];
__device__ __constant__ bvh_node d_nodes[2048];

// ---- kernel parameters ----

typedef enum _pathstate {
    DONE,           // nothing more to do for this path
    SCATTER,        // path need to traverse the BVH tree
    NO_HIT,         // path didn't hit any primitive
    HIT,            // path hit a primitive
    SHADOW,         // path hit a primitive and generated a shadow ray
    HIT_AND_LIGHT  // path hit a primitive and its shadow ray didn't hit any primitive
} pathstate;

// Structure of Arrays that has all the informations we need per path
// we have one value per path in each array for a total of maxActivePaths
struct paths {
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

    unsigned int* pixelId; // active paths currently processed by the renderer
};

// texture variable need to be global as they are device only
// just keep anything texture related global
texture<float4> t_bvh;
texture<float> t_spheres;
float* d_bvh_buf;
float* d_spheres_buf;

int* d_colors; // scene related

class RenderContext {
private:
    // these params are affected by preview mode
    const unsigned int _width;
    const unsigned int _height;

public:
    bool preview = false;

    paths paths;

    ull* next_sample; // used by init() to track next sample to fetch
    ull* numsamples_perpixel; // how many samples have been traced per pixel so far

    unsigned int* next_path; // used by hit_bvh() to track next path to fetch and trace

    vec3* output_buffer;
    metrics m;

    unsigned int iteration = 0;
    
    const int leaf_offset; // bvh related, everything else is either in constant or texture memory and thus global
    int maxBounces;

    unsigned int spp;
    unsigned int maxActivePaths;

    int numPrimitivesPerLeaf;

    int* colors;

    float lightRadius;
    vec3 lightColor;

    vec3 skyColor;

    __host__ __device__ unsigned int numPixels() {
        return width() * height();
    }

    RenderContext(int width, int height, int _spp, int _maxActivePaths, int _leafOffset, int ppl, int *_colors) :_width(width), _height(height), spp(_spp), 
            maxActivePaths(_maxActivePaths), leaf_offset(_leafOffset), numPrimitivesPerLeaf(ppl), colors(_colors) {
        checkCudaErrors(cudaMalloc((void**)&next_path, sizeof(unsigned int)));
        checkCudaErrors(cudaMalloc((void**)&next_sample, sizeof(ull)));
        checkCudaErrors(cudaMalloc((void**)&numsamples_perpixel, numPixels() * sizeof(ull)));

        checkCudaErrors(cudaMalloc((void**)&output_buffer, numPixels() * sizeof(vec3)));

        m.allocateDeviceMem();

        resetRenderer();
    }

    void freeDeviceMem() {
        checkCudaErrors(cudaFree(next_sample));
        checkCudaErrors(cudaFree(next_path));
        checkCudaErrors(cudaFree(numsamples_perpixel));
        checkCudaErrors(cudaFree(output_buffer));

        m.freeDeviceMem();
    }

    void resetRenderer() {
        iteration = 0;
        checkCudaErrors(cudaMemset(output_buffer, 0, numPixels() * sizeof(vec3)));
        checkCudaErrors(cudaMemset((void*)next_sample, 0, sizeof(ull)));
        checkCudaErrors(cudaMemset((void*)numsamples_perpixel, 0, numPixels() * sizeof(ull)));
        checkCudaErrors(cudaMemset(output_buffer, 0, numPixels() * sizeof(vec3)));
    }

    __host__ __device__ int width() {
        return preview ? PREVIEW_WIDTH : _width;
    }

    __host__ __device__ int height() {
        return preview ? PREVIEW_HEIGHT : _height;
    }
};

// ---- UI ----

GuiParams guiParams;

CudaGLContext* render_context;
CudaGLContext* preview_context;
bool r_preview = false;

// ---- Camera controls ----

camera* cam = NULL;
const float c_zoom_speed = 1.0f / 100;
bool camera_updated = false;

void setup_paths(paths& p, unsigned int maxActivePaths) {
    // at any given moment only kMaxActivePaths at most are active at the same time
    checkCudaErrors(cudaMalloc((void**)& p.r, maxActivePaths * sizeof(ray)));
    checkCudaErrors(cudaMalloc((void**)& p.shadow, maxActivePaths * sizeof(ray)));
    checkCudaErrors(cudaMalloc((void**)& p.state, maxActivePaths * sizeof(rand_state)));
    checkCudaErrors(cudaMalloc((void**)& p.attentuation, maxActivePaths * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)& p.emitted, maxActivePaths * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)& p.bounce, maxActivePaths * sizeof(unsigned short)));
    checkCudaErrors(cudaMalloc((void**)& p.pstate, maxActivePaths * sizeof(pathstate)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_id, maxActivePaths * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_normal, maxActivePaths * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_t, maxActivePaths * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)& p.pixelId, maxActivePaths * sizeof(unsigned int)));
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
    checkCudaErrors(cudaFree(p.pixelId));
}

__global__ void fetch_samples(RenderContext context, bool first, const camera cam) {
    // kMaxActivePaths threads are started to fetch the samples from all_sample_pool and initialize the paths
    // to keep things simple a block contains a single warp so that we only need to keep a single shared nextSample per block
    paths& p = context.paths;

    const unsigned int pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid == 0)
        context.next_path[0] = 0;
    context.m.reset(pid, first);
    __syncthreads();

    if (pid >= context.maxActivePaths)
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
            nextSample = atomicAdd(context.next_sample, numTerminated);

        // compute sample this lane is going to fetch
        const ull sample_id = nextSample + idxTerminated;
        //const ull max_samples = (ull)params.width * (ull)params.height * (ull)params.spp;
        //if (sample_id >= max_samples)
        //    return; // no more samples to fetch

        // retrieve pixel_id corresponding to current path
        const unsigned int pixel_id = (sample_id / context.spp) % context.numPixels();
        p.pixelId[pid] = pixel_id;
        atomicAdd(context.numsamples_perpixel + pixel_id, 1);

        // compute pixel coordinates
        const unsigned int x = pixel_id % context.width();
        const unsigned int y = pixel_id / context.width();

        // generate camera ray
        float u = float(x + random_float(state)) / float(context.width());
        float v = float(y + random_float(state)) / float(context.height());
        p.r[pid] = cam.get_ray(u, v, state);
        p.state[pid] = state;
        p.attentuation[pid] = vec3(1, 1, 1);
        p.bounce[pid] = 0;
        p.pstate[pid] = SCATTER;
    }
}

#define IDX_SENTINEL    0
#define IS_DONE(idx)    (idx == IDX_SENTINEL)
#define IS_LEAF(idx)    (idx >= context.leaf_offset)

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

__global__ void trace_scattered(RenderContext context) {
    paths& p = context.paths;

    // a limited number of threads are started to operate on the active paths (paths.pixelId)

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
    __shared__ volatile int nextPathArray[TRAVERSAL_MAX_BLOCK_HEIGHT]; // Current ray index in global buffer.
    __shared__ volatile bool noMorePaths[TRAVERSAL_MAX_BLOCK_HEIGHT]; // true when no more paths are available to fetch

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
                pathBase = atomicAdd(context.next_path, numTerminated);
                noMoreP = (pathBase + numTerminated) >= context.maxActivePaths;
            }

            pid = pathBase + idxTerminated;
            if (pid >= context.maxActivePaths) {
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
                int m = (idx - context.leaf_offset) * context.numPrimitivesPerLeaf * 3; // index to the first float in the leaf primitives
                #pragma unroll
                for (int i = 0; i < context.numPrimitivesPerLeaf; i++) {
                    float x = tex1Dfetch(t_spheres, m++);
                    if (isinf(x))
                        break; // we reached the end of the primitives buffer

                    float y = tex1Dfetch(t_spheres, m++);
                    float z = tex1Dfetch(t_spheres, m++);
                    vec3 center(x, y, z);
                    if (hit_point(center, r, 0.001f, closest, rec)) {
                        found = true;
                        closest = rec.t;
                        rec.idx = (idx - context.leaf_offset) * context.numPrimitivesPerLeaf + i;
                    }
                }

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
__global__ void generate_shadow_rays(RenderContext context) {
    paths& p = context.paths;

    const vec3 light_center(5000, 0, 0);
    const float light_radius = context.lightRadius;
    const vec3 light_color = context.lightColor;

    // kMaxActivePaths threads update all p.num_active_paths
    const unsigned int pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid == 0)
        context.next_path[0] = 0;
    __syncthreads();

    if (pid >= context.maxActivePaths)
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
__global__ void trace_shadows(RenderContext context) {
    paths& p = context.paths;

    // a limited number of threads are started to operate on the active paths (paths.pixelId)

    unsigned int pid = 0; // currently traced path
    ray r; // corresponding ray

    // bvh traversal state
    int idx = IDX_SENTINEL;
    bool found = false;
    hit_record rec;

    unsigned long long bitstack;

    // Initialize persistent threads.
    // given that each block is 32 thread wide, we can use threadIdx.x as a warpId
    __shared__ volatile int nextPathArray[TRAVERSAL_MAX_BLOCK_HEIGHT]; // Current ray index in global buffer.

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
                pathBase = atomicAdd(context.next_path, numTerminated);

            pid = pathBase + idxTerminated;
            if (pid >= context.maxActivePaths)
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
                int m = (idx - context.leaf_offset) * context.numPrimitivesPerLeaf * 3;
                #pragma unroll
                for (int i = 0; i < context.numPrimitivesPerLeaf && !found; i++) {
                    float x = tex1Dfetch(t_spheres, m++);
                    if (isinf(x))
                        break; // we reached the end of the primitives buffer

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

__global__ void copyToUintBuffer(RenderContext context, unsigned int* uint_render_buffer) {
    const unsigned int pixel_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixel_id >= context.numPixels())
        return;
    const vec3 pixel = context.output_buffer[pixel_id];
    const ull spp = context.numsamples_perpixel[pixel_id];
    uint_render_buffer[pixel_id] = rgbToInt(pixel.r() / spp, pixel.g() / spp, pixel.b() / spp);
}

// for all non terminated rays, accounts for shadow hit, compute scattered ray and resets the flag
__global__ void update(RenderContext context) {
    paths& p = context.paths;

    const vec3 sky_emissive = context.skyColor;

    // kMaxActivePaths threads update all p.num_active_paths
    const unsigned int pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= context.maxActivePaths)
        return;

    // is the path already done ?
    pathstate pstate = p.pstate[pid];
    if (pstate == DONE)
        return; // yup, done and already taken care of
    unsigned short bounce = p.bounce[pid];

    // did the ray hit a primitive ?
    if (context.preview) {
        // in preview mode, just use the primitive color
        if (pstate == HIT) {
            const int hit_id = p.hit_id[pid];
            int clr_idx = context.colors[hit_id] * 3;
            const vec3 albedo = vec3(d_colormap[clr_idx++], d_colormap[clr_idx++], d_colormap[clr_idx++]);

            const unsigned int pixel_id = p.pixelId[pid];
            atomicAdd(context.output_buffer[pixel_id].e, albedo.e[0]);
            atomicAdd(context.output_buffer[pixel_id].e + 1, albedo.e[1]);
            atomicAdd(context.output_buffer[pixel_id].e + 2, albedo.e[2]);
        }
        // in preview mode, do not bounce
        p.pstate[pid] = DONE;
    } else if (pstate == HIT || pstate == HIT_AND_LIGHT) {
        // update path attenuation
        const int hit_id = p.hit_id[pid];
        int clr_idx = context.colors[hit_id] * 3;
        const vec3 albedo = vec3(d_colormap[clr_idx++], d_colormap[clr_idx++], d_colormap[clr_idx++]);
        
        vec3 attenuation = p.attentuation[pid] * albedo;
        p.attentuation[pid] = attenuation;

        // account for light contribution if no shadow hit
        if (pstate == HIT_AND_LIGHT) {
            const vec3 incoming = p.emitted[pid] * attenuation;
            const unsigned int pixel_id = p.pixelId[pid];
            atomicAdd(context.output_buffer[pixel_id].e, incoming.e[0]);
            atomicAdd(context.output_buffer[pixel_id].e + 1, incoming.e[1]);
            atomicAdd(context.output_buffer[pixel_id].e + 2, incoming.e[2]);
        }

        // scatter ray, only if we didn't reach kMaxBounces
        bounce++;
        if (bounce < context.maxBounces) {
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
            const unsigned int pixel_id = p.pixelId[pid];
            atomicAdd(context.output_buffer[pixel_id].e, incoming.e[0]);
            atomicAdd(context.output_buffer[pixel_id].e + 1, incoming.e[1]);
            atomicAdd(context.output_buffer[pixel_id].e + 2, incoming.e[2]);
        }
        pstate = DONE;
    }

    p.pstate[pid] = pstate;
    p.bounce[pid] = bounce;
}

__global__ void print_metrics(RenderContext context, float elapsedSeconds, bool last) {
    context.m.print(context.iteration, elapsedSeconds, last);
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
    const int spheres_size_float = sc.spheres_size * 3;

    // copy the spheres in array of floats
    // do it after we build the BVH as it would have moved the spheres around
    float* floats = new float[spheres_size_float];
    int* colors = new int[sc.spheres_size];
    for (int i = 0, idx = 0; i < sc.spheres_size; i++) {
        const sphere s = sc.spheres[i];
        floats[idx++] = s.center.x();
        floats[idx++] = s.center.y();
        floats[idx++] = s.center.z();
        colors[i] = s.color;
    }

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
    cam->update();
}

void write_image(const char* output_file, CudaGLContext *context) {
    const unsigned int numPixels = context->t_height * context->t_width;
    unsigned int* idata = new unsigned int[numPixels];
    checkCudaErrors(cudaMemcpy(idata, context->cuda_dev_render_buffer, numPixels * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    char* data = new char[numPixels * 3];
    // revert y-axis to keep image consistent with window display
    int idx = 0;
    for (int y = context->t_height - 1; y >= 0; y--) {
        for (int x = 0; x < context->t_width; x++) {
            unsigned int pixel = idata[y * context->t_width + x];
            data[idx++] = pixel & 0xFF;
            data[idx++] = (pixel & 0xFF00) >> 8;
            data[idx++] = (pixel & 0xFF0000) >> 16;
        }
    }
    stbi_write_png(output_file, context->t_width, context->t_height, 3, (void*)data, context->t_width * 3);

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

int loadScene(const options opt) {
    scene sc;
    if (!opt.binary) {
        load_from_csv(opt.input, sc, opt.numPrimitivesPerLeaf);
        store_to_binary(strcat(opt.input, ".bin"), sc);
    }
    else {
        load_from_binary(opt.input, sc);
    }
    copySceneToDevice(sc, &d_colors);
    sc.release();

    return sc.bvh_size;
}

void renderIteration(const RenderContext& context, const camera& cam, bool lightEnabled) {

    // init kMaxActivePaths using equal number of threads
    {
        const int threads = 32; // 1 warp per block
        const int blocks = (context.maxActivePaths + threads - 1) / threads;
        fetch_samples <<< blocks, threads >>> (context, context.iteration == 0, cam);
        checkCudaErrors(cudaGetLastError());
    }

    // traverse bvh
    {
        dim3 blocks(6400 * 2, 1);
        dim3 threads(TRAVERSAL_MAX_BLOCK_WIDTH, TRAVERSAL_MAX_BLOCK_HEIGHT);
        trace_scattered <<< blocks, threads >>> (context);
        checkCudaErrors(cudaGetLastError());
    }

    // generate shadow rays
    if (lightEnabled)
    {
        const int threads = 128;
        const int blocks = (context.maxActivePaths + threads - 1) / threads;
        generate_shadow_rays <<< blocks, threads >>> (context);
        checkCudaErrors(cudaGetLastError());
    }

    // trace shadow rays
    if (lightEnabled)
    {
        dim3 blocks(6400 * 2, 1);
        dim3 threads(TRAVERSAL_MAX_BLOCK_WIDTH, TRAVERSAL_MAX_BLOCK_HEIGHT);
        trace_shadows <<< blocks, threads >>> (context);
        checkCudaErrors(cudaGetLastError());
    }

    // update paths accounting for intersection and light contribution
    {
        const int threads = 128;
        const int blocks = (context.maxActivePaths + threads - 1) / threads;
        update <<< blocks, threads >>> (context);
        checkCudaErrors(cudaGetLastError());
    }
}

void render(RenderContext& context, camera& cam, bool verbose) {
    bool guiChanged = true;
    bool lightEnabled = true;
    clock_t start = clock();
    cudaProfilerStart();
    bool render = true;

    while (!pollWindowEvents()) {
        if (camera_updated || guiChanged || context.preview != r_preview) {
            context.preview = r_preview;
            cam.update();
            context.resetRenderer();
            camera_updated = false;
            guiChanged = false;
            render = true;

            context.lightRadius = guiParams.lightRadius;
            context.lightColor = vec3(guiParams.lightColor[0], guiParams.lightColor[1], guiParams.lightColor[2]) * guiParams.lightIntensity;
            lightEnabled = !r_preview && guiParams.lightIntensity > 0;
            context.skyColor = vec3(guiParams.skyColor[0], guiParams.skyColor[1], guiParams.skyColor[2]) * guiParams.skyIntensity;
        }

        if (render) {
            renderIteration(context, cam, lightEnabled);

            {
                const int threads = 128;
                const int blocks = (context.numPixels() + threads - 1) / threads;
                CudaGLContext* glContext = context.preview ? preview_context : render_context;
                copyToUintBuffer <<< blocks, threads >>> (context, (unsigned int*)glContext->cuda_dev_render_buffer);

                updateWindow(glContext, guiParams, guiChanged);
            }

            // print metrics
            if (verbose) {
                print_metrics <<< 1, 1 >>> (context, (float)(clock() - start) / CLOCKS_PER_SEC, false);
                checkCudaErrors(cudaGetLastError());
            }

            context.iteration++;
            if (!context.preview && context.iteration > 0 && !(context.iteration % 100))
                write_image("temp_save.png", render_context);
            
            if (context.preview)
                render = !render;
        }
    }
    cudaProfilerStop();

    print_metrics <<< 1, 1 >>> (context, (float)(clock() - start) / CLOCKS_PER_SEC, true);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaDeviceSynchronize());
}

void mouseMove(int dx, int dy, int mouse_btn) {
    camera_updated = true;
    if (mouse_btn == MOUSE_LEFT) {
        cam->yDelta = -dx;
        cam->xDelta = -dy;
        r_preview = true;
    }
    else if (mouse_btn == MOUSE_RIGHT) {
        // drag with right button changes camera distance
        // only x movement is taken into account
        cam->relative_dist = max(0.1f, cam->relative_dist + dx * c_zoom_speed);
        r_preview = true;
    }
    else {
        camera_updated = false;
        r_preview = false;
    }
}

int main(int argc, char** argv) {
    options opt;
    if (!parse_args(argc, argv, opt))
        return -1;

    initWindow();
    render_context = new CudaGLContext(opt.nx, opt.ny);
    preview_context = new CudaGLContext(512, 512);

    registerMouseMoveFunc(mouseMove);

    loadColormap(opt.colormap);

    const int bvh_size = loadScene(opt);

    setup_camera(opt.nx, opt.ny, opt.dist);

    RenderContext renderContext(opt.nx, opt.ny, opt.ns, opt.maxActivePaths, bvh_size/2, opt.numPrimitivesPerLeaf, d_colors);

    setup_paths(renderContext.paths, opt.maxActivePaths);

    cout << "started renderer\n" << std::flush;

    render(renderContext, *cam, opt.verbose);

    char imagename[100];
    sprintf(imagename, "%s_%dx%dx%d_%d_bvh.png", opt.input, opt.nx, opt.ny, opt.ns, opt.dist);
    write_image(imagename, render_context);

    // clean up
    destroyWindow();

    delete cam;
    
    free_paths(renderContext.paths);
    renderContext.freeDeviceMem();

    releaseScene(d_colors);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    
    delete render_context;
    delete preview_context;

    cudaDeviceReset();
}