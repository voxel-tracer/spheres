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
};

struct paths {
    // pixel_id of all samples that need to be traced by the renderer
    unsigned int* all_sample_pool;
    unsigned int num_all_samples;
    int* next_sample;

    ray* r;
    rand_state* state;
    vec3* attentuation;
    unsigned short* bounce;
    int* hit_id;
    vec3* hit_normal;
    float* hit_t;
};

void setup_paths(paths& p, int nx, int ny, int ns) {
    const unsigned num_paths = nx * ny * ns;
    checkCudaErrors(cudaMalloc((void**)& p.r, num_paths * sizeof(ray)));
    checkCudaErrors(cudaMalloc((void**)& p.state, num_paths * sizeof(rand_state)));
    checkCudaErrors(cudaMalloc((void**)& p.attentuation, num_paths * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)& p.bounce, num_paths * sizeof(unsigned short)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_id, num_paths * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_normal, num_paths * sizeof(vec3)));
    checkCudaErrors(cudaMalloc((void**)& p.hit_t, num_paths * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)& p.all_sample_pool, num_paths * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)& p.next_sample, sizeof(int)));

    p.num_all_samples = num_paths;

    // init path_id on host, this way we have more control about how to layout the samples in memory
    {
        unsigned int* ids = new unsigned int[num_paths];
        int idx = 0;
        for (size_t y = 0; y < ny; y++)
            for (size_t x = 0; x < nx; x++)
                for (size_t spp = 0; spp < ns; spp++)
                    ids[idx++] = y * nx + x;
        checkCudaErrors(cudaMemcpy(p.all_sample_pool, ids, num_paths * sizeof(unsigned int), cudaMemcpyHostToDevice));
        delete[] ids;
    }
}

void free_paths(const paths& p) {
    checkCudaErrors(cudaFree(p.r));
    checkCudaErrors(cudaFree(p.state));
    checkCudaErrors(cudaFree(p.attentuation));
    checkCudaErrors(cudaFree(p.bounce));
    checkCudaErrors(cudaFree(p.hit_id));
    checkCudaErrors(cudaFree(p.hit_normal));
    checkCudaErrors(cudaFree(p.hit_t));
    checkCudaErrors(cudaFree(p.all_sample_pool));
    checkCudaErrors(cudaFree(p.next_sample));
}

__global__ void init(const render_params params, paths p, int frame, const camera cam) {
    const unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int pid = ty * params.width + tx;
    if (pid >= p.num_all_samples)
        return;

    // initialize random state
    rand_state state = ((wang_hash(pid) + frame * 101141101) * 336343633) | 1;

    // retrieve pixel_id corresponding to current path
    const unsigned int pixel_id = p.all_sample_pool[pid];

    // compute sample coordinates
    const unsigned int x = pixel_id % params.width;
    const unsigned int y = pixel_id / params.width;
    float u = float(x + random_float(state)) / float(params.width);
    float v = float(y + random_float(state)) / float(params.height);

    // generate camera ray
    p.r[pid] = cam.get_ray(u, v, state);
    p.state[pid] = state;
    p.attentuation[pid] = vec3(1, 1, 1);
    p.bounce[pid] = 0;
}

__global__ void hit_bvh(const render_params params, paths p) {
    unsigned int pid = 0; // currently traced path
    ray r; // corresponding ray

    // bvh traversal state
    bool down = false;
    int idx = 1;
    bool found = false;
    float closest = FLT_MAX;
    hit_record rec;

    unsigned int move_bit_stack = 0;
    int lvl = 0;

    // Initialize persistent threads.
    // given that each block is 32 thread wide, we can use threadIdx.x as a warpId
    __shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.

    // Persistent threads: fetch and process rays in a loop.

    while (true) {
        const int tidx = threadIdx.x;
        volatile int& rayBase = nextRayArray[threadIdx.y];

        // identify which lanes are done
        const bool          terminated      = idx == 1 && !down;
        const unsigned int  maskTerminated  = __ballot_sync(__activemask(), terminated);
        const int           numTerminated   = __popc(maskTerminated);
        const int           idxTerminated   = __popc(maskTerminated & ((1u << tidx) - 1));

        if (terminated) {
            // first terminated lane updates the base ray index
            if (idxTerminated == 0)
                rayBase = atomicAdd(p.next_sample, numTerminated);

            pid = rayBase + idxTerminated;
            if (pid >= p.num_all_samples)
                return;

            // setup ray if path not already terminated
            if (p.bounce[pid] < kMaxBounces) {
                // Fetch ray
                r = p.r[pid];

                // setup traversal
                down = true;
                idx = 1;
                found = false;
                closest = FLT_MAX;
                move_bit_stack = 0;
                lvl = 0;
            }
        }

        while (true) {
            if (down) {
                bvh_node node;
                if (idx < 2048)
                    node = d_nodes[idx];
                else {
                    unsigned int tex_idx = (idx - 2048) * 3;
                    float2 a = tex1Dfetch(t_bvh, tex_idx++);
                    float2 b = tex1Dfetch(t_bvh, tex_idx++);
                    float2 c = tex1Dfetch(t_bvh, tex_idx++);

                    node = bvh_node(a.x, a.y, b.x, b.y, c.x, c.y);
                }

                if (hit_bbox(node, r, 0.001f, closest)) {
                    const scene& sc = params.sc;
                    if (idx >= sc.count) { // leaf node
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
                        down = false;
                    }
                    else {
                        // current -> left
                        const int move_left = signbit(r.direction()[node.split_axis()]);
                        move_bit_stack &= ~(1 << lvl); // clear previous bit
                        move_bit_stack |= move_left << lvl;
                        idx = idx * 2 + move_left;
                        lvl++;
                    }
                }
                else {
                    down = false;
                }
            }
            else if (idx == 1) {
                break;
            }
            else {
                const int move_left = (move_bit_stack >> (lvl - 1)) & 1;
                const int left_idx = move_left;
                if ((idx % 2) == left_idx) { // left -> right
                    idx += -2 * left_idx + 1; // node = node.sibling
                    down = true;
                }
                else { // right -> parent
                    lvl--;
                    idx = idx / 2; // node = node.parent
                }
            }

            // some lanes may have already exited the loop, if not enough active thread are left, exit the loop
            if (__popc(__activemask()) < DYNAMIC_FETCH_THRESHOLD)
                break;
        }

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

__global__ void update(const render_params params, paths p) {
    const unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int pid = ty * params.width + tx;
    if (pid >= p.num_all_samples)
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
            const float sky_emissive = .2f;
            vec3 incoming = p.attentuation[pid] * sky_emissive;
            const unsigned int pixel_id = p.all_sample_pool[pid];
            atomicAdd(params.fb[pixel_id].e, incoming.e[0]);
            atomicAdd(params.fb[pixel_id].e + 1, incoming.e[1]);
            atomicAdd(params.fb[pixel_id].e + 2, incoming.e[2]);
        }
        
        bounce = kMaxBounces; // mark path as terminated
    }

    p.bounce[pid] = bounce;
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
        cerr << "usage spheres file_name [num_samples=1] [num_runs=1] [camera_dist=100] [colormap=viridis.csv]";
        exit(-1);
    }
    char* input = argv[1];
    const int ns = (argc > 2) ? strtol(argv[2], NULL, 10) : 1;
    int nr = (argc > 3) ? strtol(argv[3], NULL, 10) : 1;
    if (nr == 0) nr = INT_MAX;
    const int dist = (argc > 4) ? strtof(argv[4], NULL) : 100;
    const char* colormap = (argc > 5) ? argv[5] : "viridis.csv";

    const bool is_csv = strncmp(input + strlen(input) - 4, ".csv", 4) == 0;
    
    cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel\n";

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

    paths p;
    setup_paths(p, nx, ny, ns);

    double render_time = 0;
    for (int r = 0, frame = 0; r < nr; r++, frame ++) {
        // Render our buffer
        clock_t start;
        start = clock();

        cudaProfilerStart();
        // init paths
        {
            dim3 threads(128);
            dim3 blocks((p.num_all_samples + 127) / threads.x);
            init <<<blocks, threads >>> (params, p, frame, cam);
            checkCudaErrors(cudaGetLastError());
        }
        for (size_t bounce = 0; bounce < kMaxBounces; bounce++) {
            // reset pool counter
            checkCudaErrors(cudaMemset(p.next_sample, 0, sizeof(int)));

            // traverse bvh
            {
                // visual profiler shows that we can only run 40 warps per SM, and I have 5 SMs so I can run a total of
                // 5*40*32 = 6400. Run twice as much to give the device enough thread to hide memory latency
                dim3 blocks(6400 * 2, 1);
                dim3 threads(MaxBlockWidth, MaxBlockHeight);
                hit_bvh << <blocks, threads >> > (params, p);
                checkCudaErrors(cudaGetLastError());
            }
            // update paths
            {
                dim3 threads(128);
                dim3 blocks((p.num_all_samples + 127) / threads.x);
                update << <blocks, threads >> > (params, p);
                checkCudaErrors(cudaGetLastError());
            }
        }
        checkCudaErrors(cudaDeviceSynchronize());
        render_time += clock() - start;
        cerr << "rendered " << (frame + ns) << " samples in " << render_time / CLOCKS_PER_SEC << " seconds.\r";

        // save temp output
        checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
        cudaProfilerStop();
        write_image("inprogress.png", h_fb, nx, ny, frame + ns);
    }

    // Output FB as Image
    checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
    char file_name[100];
    sprintf(file_name, "%s_%dx%dx%d_%d_bvh.png", input, nx, ny, ns*nr, dist);
    write_image(file_name, h_fb, nx, ny, ns*nr);
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