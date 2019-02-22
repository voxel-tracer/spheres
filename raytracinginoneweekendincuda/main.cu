#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << cudaGetErrorString(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

const int kSphereCount = 22 * 22 + 1 + 3;

__device__ __constant__ sphere d_spheres[kSphereCount];
__device__ __constant__ material d_materials[kSphereCount];

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, rand_state& rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if (hit_spheres(d_spheres, kSphereCount, cur_ray, 0.001f, FLT_MAX, rec)) {
            if (!scatter(d_spheres[rec.hit_idx], d_materials[rec.hit_idx], cur_ray, cur_attenuation, rec, rand_state)) {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            float t = 0.5f*(cur_ray.direction().y() + 1.0f);
            vec3 c = (1.0f - t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, const camera cam) {
    // 1D blocks
    const int rIdx = (threadIdx.x + blockIdx.x*blockDim.x);
    if (rIdx >= (max_x*max_y*ns)) return;
    rand_state state = (wang_hash(rIdx) * 336343633) | 1;
    const int y = rIdx / (max_x*ns);
    const int x = (rIdx % max_x) / ns;
    float u = float(x + random_float(state)) / float(max_x);
    float v = float(y + random_float(state)) / float(max_y);
    ray r = cam.get_ray(u, v, state);
    fb[rIdx] = color(r, state);
}

float rand(unsigned int &state) {
    state = (214013 * state + 2531011);
    return (float)((state >> 16) & 0x7FFF) / 32767;
}

#define RND (rand(rand_state))

void setup_scene(sphere **h_spheres, material **h_materials) {
    sphere* spheres = new sphere[kSphereCount];
    material* materials = new material[kSphereCount];

    unsigned int rand_state = 0;

    materials[0] = material(material::Lambertian, vec3(0.5, 0.5, 0.5), 0, 0);
    spheres[0] = sphere(vec3(0, -1000.0, -1), 1000);
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = RND;
            vec3 center(a + RND, 0.2, b + RND);
            if (choose_mat < 0.8f) {
                materials[i] = material(material::Lambertian, vec3(RND*RND, RND*RND, RND*RND), 0, 0);
                spheres[i++] = sphere(center, 0.2);
            }
            else if (choose_mat < 0.95f) {
                materials[i] = material(material::Metal, vec3(0.5f*(1.0f + RND), 0.5f*(1.0f + RND), 0.5f*(1.0f + RND)), 0.5f*RND, 0);
                spheres[i++] = sphere(center, 0.2);
            }
            else {
                materials[i] = material(material::Dielectric, vec3(), 0, 1.5);
                spheres[i++] = sphere(center, 0.2);
            }
        }
    }
    materials[i] = material(material::Dielectric, vec3(), 0, 1.5);
    spheres[i++] = sphere(vec3(0, 1, 0), 1.0);
    materials[i] = material(material::Lambertian, vec3(0.4, 0.2, 0.1), 0, 0);
    spheres[i++] = sphere(vec3(-4, 1, 0), 1.0);
    materials[i] = material(material::Metal, vec3(0.7, 0.6, 0.5), 0, 0);
    spheres[i++] = sphere(vec3(4, 1, 0), 1.0);

    *h_spheres = spheres;
    *h_materials = materials;
}

camera setup_camera(int nx, int ny) {
    vec3 lookfrom(13, 2, 3);
    vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0; (lookfrom - lookat).length();
    float aperture = 0.1;
    return camera(lookfrom,
        lookat,
        vec3(0, 1, 0),
        30.0,
        float(nx) / float(ny),
        aperture,
        dist_to_focus);
}

void write_image(const char* output_file, const vec3 *fb, const int nx, const int ny, const int ns) {
    char *data = new char[nx * ny * 3];
    int idx = 0;
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = (j * nx + i) * ns;
            vec3 col(0, 0, 0);
            for (int s = 0; s < ns; s++)
                col += fb[pixel_index++];
            col /= ns;

            data[idx++] = int(255.99*sqrtf(col.r()));
            data[idx++] = int(255.99*sqrtf(col.g()));
            data[idx++] = int(255.99*sqrtf(col.b()));
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
    const int nx = 1200;
    const int ny = 800;
    const int ns = (argc > 1) ? strtol(argv[1], NULL, 10) : 1;
    int block_size = 64;
    const int nr = (argc > 2) ? strtol(argv[2], NULL, 10) : 1;
    
    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << block_size << " blocks.\n";

    int num_pixels = nx * ny * ns;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *d_fb;
    checkCudaErrors(cudaMalloc((void **)&d_fb, fb_size));

    // setup scene
    sphere* h_spheres;
    material* h_materials;
    setup_scene(&h_spheres, &h_materials);

    // copy the scene to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(d_spheres, h_spheres, kSphereCount * sizeof(sphere)));
    checkCudaErrors(cudaMemcpyToSymbol(d_materials, h_materials, kSphereCount * sizeof(material)));

    camera cam = setup_camera(nx, ny);

    double *runs = new double[nr];
    for (int r = 0; r < nr; r++) {
        clock_t start, stop;
        start = clock();
        // Render our buffer
        dim3 blocks((nx*ny*ns) / block_size + 1);
        dim3 threads(block_size);
        render << <blocks, threads >> >(d_fb, nx, ny, ns, cam);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        stop = clock();
        runs[r] = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "took " << runs[r] << " seconds.\n";
    }
    if (nr > 1) {
        // compute median
        std::qsort(runs, nr, sizeof(double), cmpfunc);
        std::cerr << "median run took " << runs[nr / 2] << " seconds.\n";
    }
    delete[] runs;
    runs = NULL;

    // Output FB as Image
    vec3* h_fb = new vec3[fb_size];
    checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
    write_image("output.png", h_fb, nx, ny, ns);
    delete[] h_fb;
    h_fb = NULL;

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_fb));

    cudaDeviceReset();
}