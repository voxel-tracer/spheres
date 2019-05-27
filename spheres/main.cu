#include <float.h>

#include "ray.h"
#include "camera.h"
#include "scene.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int kMaxBounces = 10;
const int nx = 1200;
const int ny = 1200;

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, const scene s, rand_state& rand_state) {
    vec3 light_center(5000, 0, 0);
    float light_radius = 500;
    float light_emissive = 100;
    float sky_emissive = .2f;

    ray cur_ray = r;
    vec3 attenuation = vec3(1, 1, 1);
    vec3 incoming = vec3(0, 0, 0);
    for (int i = 0; i < kMaxBounces; i++) {
        hit_record rec;
        if (hit_bvh(s, cur_ray, 0.001f, FLT_MAX, rec)) {
            const vec3 p = cur_ray.point_at_parameter(rec.t);
            vec3 target = rec.n + random_in_unit_sphere(rand_state);

            int clr_idx = s.colors[rec.idx] * 3;
            vec3 albedo = vec3(d_colormap[clr_idx++], d_colormap[clr_idx++], d_colormap[clr_idx++]);

            // explicit light sampling

            // create a random direction towards sphere
            
            // coord system for sampling: sw, su, sv
            vec3 sw = unit_vector(light_center - p);
            vec3 su = unit_vector(cross(fabs(sw.x()) > 0.01f ? vec3(0, 1, 0) : vec3(1, 0, 0), sw));
            vec3 sv = cross(sw, su);
            
            // sample sphere by solid angle
            float cosAMax = sqrtf(1.0f - light_radius * light_radius / (p - light_center).squared_length());
            float eps1 = random_float(rand_state), eps2 = random_float(rand_state);
            float cosA = 1.0f - eps1 + eps1 * cosAMax;
            float sinA = sqrtf(1.0f - cosA * cosA);
            float phi = 2 * kPI * eps2;
            vec3 l = unit_vector(su * cosf(phi) * sinA + sv * sin(phi) * sinA + sw * cosA);

            // shoot shadow ray
            if (!shadow_bvh(s, ray(p, l), 0.001f, FLT_MAX)) {
                float omega = 2 * kPI * (1 - cosAMax);

                vec3 rdir = cur_ray.direction();
                vec3 nl = dot(rec.n, rdir) < 0 ? rec.n : -rec.n;
                incoming += attenuation * (albedo * light_emissive) * (fmaxf(0.0f, dot(l, nl)) * omega / kPI);
            }

            attenuation *= albedo;
            cur_ray = ray(p, target);
        }
        else if (i == 0) { // primary ray didn't hit anything
            break; // black background
        }
        else {
            return incoming + attenuation * sky_emissive;
        }
    }
    return incoming; // exceeded recursion
}

__global__ void render(vec3 *fb, const scene sc, int max_x, int max_y, int ns, int frame, const camera cam) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    rand_state state = ((wang_hash(pixel_index) + frame * 101141101) * 336343633) | 1;
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + random_float(state)) / float(max_x);
        float v = float(j + random_float(state)) / float(max_y);
        ray r = cam.get_ray(u, v, state);
        col += color(r, sc, state);
    }
    fb[pixel_index] += col;
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
    const int tx = 8;
    const int ty = 8;
    int nr = (argc > 3) ? strtol(argv[3], NULL, 10) : 1;
    if (nr == 0) nr = INT_MAX;
    const int dist = (argc > 4) ? strtof(argv[4], NULL) : 100;
    const char* colormap = (argc > 5) ? argv[5] : "viridis.csv";

    const bool is_csv = strncmp(input + strlen(input) - 4, ".csv", 4) == 0;
    
    cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *d_fb;
    checkCudaErrors(cudaMalloc((void **)&d_fb, fb_size));
    checkCudaErrors(cudaMemset(d_fb, 0, fb_size));

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

    double render_time = 0;
    for (int r = 0, frame = 0; r < nr; r++, frame += ns) {
        // Render our buffer
        clock_t start;
        start = clock();
        dim3 blocks(nx / tx + 1, ny / ty + 1);
        dim3 threads(tx, ty);
        render << <blocks, threads >> >(d_fb, sc, nx, ny, ns, frame, cam);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        render_time += clock() - start;
        cerr << "rendered " << (frame + ns) << " samples in " << render_time / CLOCKS_PER_SEC << " seconds.\r";

        // save temp output
        checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
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
    checkCudaErrors(cudaFree(d_fb));

    cudaDeviceReset();
}