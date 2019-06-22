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
    unsigned int* rayIds;
    unsigned int numRays;
    int* warpCounter;
    vec3* fb;
    scene sc;
    unsigned int width;
    unsigned int height;
};

__global__ void render(const render_params params, int frame, const camera cam) {

    const vec3 light_center(5000, 0, 0);
    const float light_radius = 500;
    const float light_emissive = 100;
    const float sky_emissive = .2f;

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    rand_state   state = ((wang_hash(j * params.width + i) + frame * 101141101) * 336343633) | 1;
    unsigned int bounce = kMaxBounces;
    vec3         attenuation;
    vec3         incoming;
    ray          r;
    int          rayidx;
    unsigned int pixel_id;

    // Initialize persistent threads.
    // given that each block is 32 thread wide, we can use threadIdx.x as a warpId
    __shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.

    // Persistent threads: fetch and process rays in a loop.

    while (true) {
        const int tidx = threadIdx.x;
        volatile int& rayBase = nextRayArray[threadIdx.y];

        // Fetch new rays from the global pool using lane 0.

        const bool terminated               = bounce == kMaxBounces;
        const unsigned int  maskTerminated  = __ballot_sync(__activemask(), terminated);
        const int           numTerminated   = __popc(maskTerminated);
        const int           idxTerminated   = __popc(maskTerminated & ((1u << tidx) - 1));

        if (terminated) {
            if (idxTerminated == 0)
                rayBase = atomicAdd(params.warpCounter, numTerminated);

            rayidx = rayBase + idxTerminated;
            if (rayidx >= params.numRays)
                return;

            // Fetch pixel_id
            pixel_id = params.rayIds[rayidx];
            const int x = pixel_id % params.width;
            const int y = pixel_id / params.width;

            // generate ray
            float u = float(x + random_float(state)) / float(params.width);
            float v = float(y + random_float(state)) / float(params.height);
            r = cam.get_ray(u, v, state);

            // setup traversal
            bounce = 0;
            attenuation = vec3(1, 1, 1);
            incoming = vec3(0, 0, 0);
        }

        while (bounce < kMaxBounces) {
            hit_record rec;
            if (hit_bvh(params.sc, r, 0.001f, FLT_MAX, rec)) {
                const vec3 p = r.point_at_parameter(rec.t);
                vec3 target = rec.n + random_in_unit_sphere(state);

                int clr_idx = params.sc.colors[rec.idx] * 3;
                vec3 albedo = vec3(d_colormap[clr_idx++], d_colormap[clr_idx++], d_colormap[clr_idx++]);

                attenuation *= albedo;
                r = ray(p, target);
                bounce++;
            }
            else {
                // primary rays (bounce = 0) return black
                if (bounce > 0) {
                    if (hit_light(light_center, light_radius, r, 0.001f, FLT_MAX))
                        incoming = attenuation * light_emissive;
                    else
                        incoming = attenuation * sky_emissive;
                }
                bounce = kMaxBounces; // mark the lane as terminated
                break;
            }

            // some lanes may have already exited the loop, if not enough active thread are left, exit the loop
            if (__popc(__activemask()) < DYNAMIC_FETCH_THRESHOLD)
                break;
        }

        if (bounce == kMaxBounces) {
            // only works when spp=1
            // otherwise we should do this atomically
            atomicAdd(params.fb[pixel_id].e, incoming.e[0]);
            atomicAdd(params.fb[pixel_id].e+1, incoming.e[1]);
            atomicAdd(params.fb[pixel_id].e+2, incoming.e[2]);
        }
    }

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

void setup_pixels(int nx, int ny, int ns, unsigned int** d_pixels) {
    checkCudaErrors(cudaMalloc((void**)d_pixels, nx * ny * ns * sizeof(unsigned int)));
    unsigned int* pixels = new unsigned int[nx * ny * ns];
    int idx = 0;
    for (size_t y = 0; y < ny; y++)
        for (size_t x = 0; x < nx; x++)
            for (size_t spp = 0; spp < ns; spp++)
                pixels[idx++] = y * nx + x;
    
    checkCudaErrors(cudaMemcpy(*d_pixels, pixels, nx * ns * ny * sizeof(unsigned int), cudaMemcpyHostToDevice));
    delete[] pixels;
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
    unsigned int* d_pixels;
    int* d_warpCount;
    checkCudaErrors(cudaMalloc((void **)&d_fb, fb_size));
    checkCudaErrors(cudaMemset(d_fb, 0, fb_size));

    checkCudaErrors(cudaMalloc((void**)&d_warpCount, sizeof(int)));

    setup_pixels(nx, ny, ns, &d_pixels);

    // load colormap
    vector<vector<float>> data = parse2DCsvFile(colormap);
    cout << "colormap contains " << data.size() << " points\n";
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
    params.rayIds = d_pixels;
    params.numRays = nx * ny * ns;
    params.warpCounter = d_warpCount;
    params.fb = d_fb;
    params.sc = sc;
    params.width = nx;
    params.height = ny;

    double render_time = 0;
    for (int r = 0, frame = 0; r < nr; r++, frame += ns) {
        // Render our buffer
        clock_t start;
        start = clock();
        // visual profiler shows that we can only run 40 warps per SM, and I have 5 SMs so I can run a total of
        // 5*40*32 = 6400
        dim3 blocks(6400, 1);
        dim3 threads(MaxBlockWidth, MaxBlockHeight);
        checkCudaErrors(cudaMemset(params.warpCounter, 0, sizeof(int)));
        cudaProfilerStart();
        render << <blocks, threads >> > (params, frame, cam);
        checkCudaErrors(cudaGetLastError());
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
    releaseScene(sc);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(params.fb));
    checkCudaErrors(cudaFree(params.warpCounter));

    cudaDeviceReset();
}