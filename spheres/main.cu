#include <float.h>
#include <cuda_profiler_api.h>

#include "ray.h"
#include "camera.h"
#include "scene.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays

const int MaxBlockWidth = 32;
const int MaxBlockHeight = 2; // block width is 32
const int kMaxBounces = 10;

const int nx = 1200;
const int ny = 1200;

struct render_params {
    vec3* fb;
    scene sc;
    unsigned int width;
    unsigned int height;
    unsigned int spp;
    unsigned int maxActivePaths;
};

struct paths {
    unsigned long long* next_sample; // used by init() to track next sample to fetch

    // pixel_id of active paths currently being traced by the renderer, it's a subset of all_sample_pool
    unsigned int* active_paths;
    unsigned int* next_path; // used by hit_bvh() to track next path to fetch and trace

    ray* r;
    rand_state* state;
    vec3* attentuation;
    unsigned short* bounce;
    int* hit_id;
    vec3* hit_normal;
    float* hit_t;

    unsigned int* metric_num_active_paths;
};

void setup_paths(paths& p, int nx, int ny, int ns, unsigned int maxActivePaths) {
    // at any given moment only kMaxActivePaths at most are active at the same time
    const unsigned num_paths = maxActivePaths;
    checkCudaErrors(cudaMalloc((void**)& p.r, num_paths * sizeof(ray)));
    checkCudaErrors(cudaMalloc((void**)& p.state, num_paths * sizeof(rand_state)));
    checkCudaErrors(cudaMalloc((void**)& p.attentuation, num_paths * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)& p.bounce, num_paths * sizeof(unsigned short)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_id, num_paths * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_normal, num_paths * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_t, num_paths * sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)& p.active_paths, num_paths * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)& p.next_path, sizeof(unsigned int)));

    checkCudaErrors(cudaMalloc((void**)& p.next_sample, sizeof(unsigned long)));
    checkCudaErrors(cudaMemset((void*)p.next_sample, 0, sizeof(unsigned long)));
    checkCudaErrors(cudaMalloc((void**)& p.metric_num_active_paths, sizeof(unsigned int)));
}

void free_paths(const paths& p) {
    checkCudaErrors(cudaFree(p.r));
    checkCudaErrors(cudaFree(p.state));
    checkCudaErrors(cudaFree(p.attentuation));
    checkCudaErrors(cudaFree(p.bounce));
    checkCudaErrors(cudaFree(p.hit_id));
    checkCudaErrors(cudaFree(p.hit_normal));
    checkCudaErrors(cudaFree(p.hit_t));
    checkCudaErrors(cudaFree(p.next_sample));

    checkCudaErrors(cudaFree(p.active_paths));
    checkCudaErrors(cudaFree(p.next_path));

    checkCudaErrors(cudaFree(p.metric_num_active_paths));
}

__global__ void init(const render_params params, paths p, bool first, const camera cam) {
    // kMaxActivePaths threads are started to fetch the samples from all_sample_pool and initialize the paths
    // to keep things simple a block contains a single warp so that we only need to keep a single shared nextSample per block

    const unsigned int pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid == 0) {
        p.metric_num_active_paths[0] = 0;
        p.next_path[0] = 0;
    }

    if (pid >= params.maxActivePaths)
        return;

    rand_state state;
    unsigned int bounce;
    if (first) {
        // this is the very first init, all paths are marked terminated, and we don't have a valid random state yet
        state = (wang_hash(pid) * 336343633) | 1;
        bounce = kMaxBounces;
    } else {
        state = p.state[pid];
        bounce = p.bounce[pid];
    }

    // generate all terminated paths
    const bool          terminated     = bounce == kMaxBounces;
    const unsigned int  maskTerminated = __ballot_sync(__activemask(), terminated);
    const int           numTerminated  = __popc(maskTerminated);
    const int           idxTerminated  = __popc(maskTerminated & ((1u << threadIdx.x) - 1));

    __shared__ volatile unsigned long long nextSample;

    if (terminated) {
        // first terminated lane increments next_sample
        if (idxTerminated == 0)
            nextSample = atomicAdd(p.next_sample, numTerminated);

        // compute sample this lane is going to fetch
        const unsigned long long sample_id = nextSample + idxTerminated;
        const unsigned long long num_all_samples = ((unsigned long long) params.width) * params.height * params.spp;
        if (sample_id >= num_all_samples)
            return; // no more samples to fetch

        // retrieve pixel_id corresponding to current path
        const unsigned int pixel_id = sample_id / params.spp;
        p.active_paths[pid] = pixel_id;

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
    }

    // path still active or has just been generated
    //TODO each warp uses activemask() to count number active lanes then just 1st active lane need to update the metric
    atomicAdd(p.metric_num_active_paths, 1);
}

#define IDX_SENTINEL    0
#define IS_DONE(idx)    (idx == IDX_SENTINEL)
#define IS_LEAF(idx)    (idx >= sc.count)

#define BIT_MASK        3
#define BIT_PARENT      3
#define BIT_LEFT        1
#define BIT_RIGHT       2

__device__ void pop_bitstack(unsigned long long& bitstack, int& idx) {
    // TODO we may be able to combine this logic with main traversal loop to simplify this
    while ((bitstack & BIT_MASK) == BIT_PARENT) {
        // pop one level out of the stack
        bitstack = bitstack >> 2;
        idx = idx >> 1;
    }

    if (bitstack == 0) {
        idx = IDX_SENTINEL;
    }
    else {
        // idx could point to left or right child regardless of sibling we need to go to
        idx = (idx >> 1) << 1; // make sure idx always points to left sibling
        idx += (bitstack & BIT_MASK) - 1; // move idx to the sibling stored in bitstack
        bitstack = bitstack | BIT_PARENT; // set bitstack to parent, so we can backtrack
    }
}

__global__ void hit_bvh(const render_params params, paths p) {
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

    // Persistent threads: fetch and process rays in a loop.

    while (true) {
        const int tidx = threadIdx.x;
        volatile int& pathBase = nextPathArray[threadIdx.y];

        // identify which lanes are done
        const bool          terminated      = IS_DONE(idx);
        const unsigned int  maskTerminated  = __ballot_sync(__activemask(), terminated);
        const int           numTerminated   = __popc(maskTerminated);
        const int           idxTerminated   = __popc(maskTerminated & ((1u << tidx) - 1));

        if (terminated) {
            // first terminated lane updates the base ray index
            if (idxTerminated == 0)
                pathBase = atomicAdd(p.next_path, numTerminated);

            pid = pathBase + idxTerminated;
            if (pid >= params.maxActivePaths)
                return;

            // setup ray if path not already terminated
            if (p.bounce[pid] < kMaxBounces) {
                // Fetch ray
                r = p.r[pid];


                // idx is already set to IDX_SENTINEL, but make sure we set found to false
                found = false;

                // setup traversal if ray intersects root node
                //if (hit_bbox(d_nodes[1], r, FLT_MAX) < FLT_MAX) {
                    idx = 1;
                    closest = FLT_MAX;
                    bitstack = 0;
                //}
            }
        }

        // traversal
        const scene& sc = params.sc;
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
                int m = (idx - sc.count) * lane_size_float;
                #pragma unroll
                for (int i = 0; i < lane_size_spheres; i++) {
                    vec3 center(sc.spheres[m++], sc.spheres[m++], sc.spheres[m++]);
                    if (hit_point(center, r, 0.001f, closest, rec)) {
                        found = true;
                        closest = rec.t;
                        rec.idx = (idx - sc.count) * lane_size_spheres + i;
                    }
                }

                pop_bitstack(bitstack, idx);
            }

            // some lanes may have already exited the loop, if not enough active thread are left, exit the loop
            if (__popc(__activemask()) < DYNAMIC_FETCH_THRESHOLD)
                break;
        }

        if (IS_DONE(idx)) {
            // finished traversing bvh
            if (found) {
                p.hit_id[pid] = rec.idx;
                p.hit_normal[pid] = rec.n;
                p.hit_t[pid] = rec.t;
            }
            else {
                p.hit_id[pid] = -1;
            }
        }
    }
}

__global__ void update(const render_params params, paths p) {

    const vec3 light_center(5000, 0, 0);
    const float light_radius = 500;
    const float light_emissive = 100;
    const float sky_emissive = .2f;

    // kMaxActivePaths threads update all p.num_active_paths
    const unsigned int pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= params.maxActivePaths)
        return;

    // is the path already done ?
    unsigned int bounce = p.bounce[pid];
    if (bounce == kMaxBounces)
        return; // yup, done and already taken care of

    // did the ray hit a primitive ?
    const int hit_id = p.hit_id[pid];
    if (hit_id >= 0) {
        // update path attenuation
        int clr_idx = params.sc.colors[hit_id] * 3;
        const vec3 albedo = vec3(d_colormap[clr_idx++], d_colormap[clr_idx++], d_colormap[clr_idx++]);
        p.attentuation[pid] *= albedo;

        // scatter ray, only if we didn't reach kMaxBounces
        bounce++;
        if (bounce < kMaxBounces) {
            const ray r = p.r[pid];
            const float hit_t = p.hit_t[pid];
            const vec3 hit_p = r.point_at_parameter(hit_t);

            const vec3 hit_n = p.hit_normal[pid];
            rand_state state = p.state[pid];
            const vec3 target = hit_n + random_in_unit_sphere(state);

            p.r[pid] = ray(hit_p, target);
            p.state[pid] = state;
        }
    }
    else {
        // primary rays (bounce = 0) return black
        if (bounce > 0) {
            const vec3 attenuation = p.attentuation[pid];
            vec3 incoming;
            if (hit_light(light_center, light_radius, p.r[pid], 0.001f, FLT_MAX))
                incoming = attenuation * light_emissive;
            else
                incoming = attenuation * sky_emissive;

            const unsigned int pixel_id = p.active_paths[pid];
            atomicAdd(params.fb[pixel_id].e, incoming.e[0]);
            atomicAdd(params.fb[pixel_id].e + 1, incoming.e[1]);
            atomicAdd(params.fb[pixel_id].e + 2, incoming.e[2]);
        }
        
        bounce = kMaxBounces; // mark path as terminated
    }

    p.bounce[pid] = bounce;
}

__global__ void print_metrics(paths p, unsigned int iteration, unsigned int maxActivePaths) {
    unsigned int metric_num_active_paths = p.metric_num_active_paths[0];
    unsigned int ratio = 100.0 * metric_num_active_paths / maxActivePaths;
    printf("iteration %4d: metric_num_active_paths = %d (%2d%%)\n", iteration, metric_num_active_paths, ratio);
}

float rand(unsigned int &state) {
    state = (214013 * state + 2531011);
    return (float)((state >> 16) & 0x7FFF) / 32767;
}

#define RND (rand(rand_state))

camera setup_camera(int nx, int ny, float dist) {
    vec3 lookfrom(dist, dist, dist);
    vec3 lookat(0, 0, 0);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.1;
    return camera(lookfrom,
        lookat,
        vec3(0, 1, 0),
        30.0,
        float(nx) / float(ny),
        aperture,
        dist_to_focus);
}

// http://chilliant.blogspot.com.au/2012/08/srgb-approximations-for-hlsl.html
static uint32_t LinearToSRGB(float x)
{
    x = max(x, 0.0f);
    x = max(1.055f * powf(x, 0.416666667f) - 0.055f, 0.0f);
    uint32_t u = min((uint32_t)(x * 255.9f), 255u);
    return u;
}

void write_image(const char* output_file, const vec3 *fb, const int nx, const int ny, const int ns) {
    char *data = new char[nx * ny * 3];
    int idx = 0;
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            data[idx++] = LinearToSRGB(fb[pixel_index].r() / ns);
            data[idx++] = LinearToSRGB(fb[pixel_index].g() / ns);
            data[idx++] = LinearToSRGB(fb[pixel_index].b() / ns);
        }
    }
    stbi_write_png(output_file, nx, ny, 3, (void*)data, nx * 3);
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

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "usage spheres file_name [num_samples=1] [camera_dist=100] [maxActivePaths=1M] [numBouncesPerIter=4] [colormap=viridis.csv] [verbose]";
        exit(-1);
    }
    char* input = argv[1];
    const int ns = (argc > 2) ? strtol(argv[2], NULL, 10) : 1;
    const int dist = (argc > 3) ? strtof(argv[3], NULL) : 100;
    const int maxActivePaths = (argc > 4) ? strtol(argv[4], NULL, 10) : (1024 * 1024);
    const int numBouncesPerIter = (argc > 5) ? strtol(argv[5], NULL, 10) : 4;
    const char* colormap = (argc > 6) ? argv[6] : "viridis.csv";
    const bool verbose = (argc > 7 && !strcmp(argv[7], "verbose"));

    const bool is_csv = strncmp(input + strlen(input) - 4, ".csv", 4) == 0;
    
    cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel, maxActivePaths = " << maxActivePaths << ", numBouncesPerIter = " << numBouncesPerIter << "\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *d_fb;
    checkCudaErrors(cudaMalloc((void **)&d_fb, fb_size));
    checkCudaErrors(cudaMemset(d_fb, 0, fb_size));


    // load colormap
    vector<vector<float>> data = parse2DCsvFile(colormap);
    std::cout << "colormap contains " << data.size() << " points\n";
    float *_viridis_data = new float[data.size() * 3];
    int idx = 0;
    for (auto l : data) {
        _viridis_data[idx++] = (float)l[0];
        _viridis_data[idx++] = (float)l[1];
        _viridis_data[idx++] = (float)l[2];
    }
    // setup scene
    scene sc;
    setup_scene(input, sc, is_csv, _viridis_data);
    delete[] _viridis_data;
    _viridis_data = NULL;

    camera cam = setup_camera(nx, ny, dist);
    vec3* h_fb = new vec3[fb_size];

    render_params params;
    params.fb = d_fb;
    params.sc = sc;
    params.width = nx;
    params.height = ny;
    params.spp = ns;
    params.maxActivePaths = maxActivePaths;

    paths p;
    setup_paths(p, nx, ny, ns, maxActivePaths);

    cout << "started renderer\n" << std::flush;
    clock_t start = clock();
    cudaProfilerStart();

    unsigned int iteration = 0;
    while (true) {

        // init kMaxActivePaths using equal number of threads
        {
            const int threads = 32; // 1 warp per block
            const int blocks = (maxActivePaths + threads - 1) / threads;
            init << <blocks, threads >> > (params, p, iteration == 0, cam);
            checkCudaErrors(cudaGetLastError());
        }

        // check if not all paths terminated
        // we don't want to check the metric after each bounce, we do it every numBouncesPerIter iterations
        if (iteration > 0 && (iteration % numBouncesPerIter) == 0) {
            unsigned int num_active_paths;
            checkCudaErrors(cudaMemcpy((void*)& num_active_paths, (void*)p.metric_num_active_paths, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            if (num_active_paths == 0) {
                break;
            }
        }

        // traverse bvh
        {
            dim3 blocks(6400 * 2, 1);
            dim3 threads(MaxBlockWidth, MaxBlockHeight);
            hit_bvh << <blocks, threads >> > (params, p);
            checkCudaErrors(cudaGetLastError());
        }
        // update kMaxActivePaths using equal number of threads
        {
            const int threads = 128;
            const int blocks = (maxActivePaths + threads - 1) / threads;
            update << <blocks, threads >> > (params, p);
            checkCudaErrors(cudaGetLastError());
        }
        // print metrics
        if (verbose) {
            print_metrics << <1, 1 >> > (p, iteration, maxActivePaths);
            checkCudaErrors(cudaGetLastError());
        }
        checkCudaErrors(cudaDeviceSynchronize());

        iteration++;
    }
    cudaProfilerStop();

    checkCudaErrors(cudaDeviceSynchronize());
    cerr << "rendered " << (params.width* params.height* params.spp) << " samples in " << (float)(clock() - start) / CLOCKS_PER_SEC << " seconds.\r";

    // Output FB as Image
    checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
    char file_name[100];
    sprintf(file_name, "%s_%dx%dx%d_%d_bvh.png", input, nx, ny, ns, dist);
    write_image(file_name, h_fb, nx, ny, ns);
    delete[] h_fb;
    h_fb = NULL;

    // clean up
    free_paths(p);
    releaseScene(sc);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(params.fb));

    cudaDeviceReset();
}