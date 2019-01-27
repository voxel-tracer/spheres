#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable_list **world, rand_state *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if (hit_hitable_list(*world, cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (scatter(*rec.mat_ptr, cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(rand_state *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, rand_state *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable_list **world, rand_state *rnd_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    rand_state local_rand_state = rnd_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rnd_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(sphere **d_spheres, material **d_materials, hitable_list **d_world, camera **d_camera, int nx, int ny, rand_state *rnd_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        rand_state local_rand_state = *rnd_state;
        d_materials[0] = new material(material::Lambertian, vec3(0.5, 0.5, 0.5), 0, 0);
        d_spheres[0] = new sphere(vec3(0, -1000.0, -1), 1000);
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_materials[i] = new material(material::Lambertian, vec3(RND*RND, RND*RND, RND*RND), 0, 0);
                    d_spheres[i++] = new sphere(center, 0.2);
                }
                else if (choose_mat < 0.95f) {
                    d_materials[i] = new material(material::Metal, vec3(0.5f*(1.0f + RND), 0.5f*(1.0f + RND), 0.5f*(1.0f + RND)), 0.5f*RND, 0);
                    d_spheres[i++] = new sphere(center, 0.2);
                }
                else {
                    d_materials[i] = new material(material::Dielectric, vec3(), 0, 1.5);
                    d_spheres[i++] = new sphere(center, 0.2);
                }
            }
        }
        d_materials[i] = new material(material::Dielectric, vec3(), 0, 1.5);
        d_spheres[i++] = new sphere(vec3(0, 1, 0), 1.0);
        d_materials[i] = new material(material::Lambertian, vec3(0.4, 0.2, 0.1), 0, 0);
        d_spheres[i++] = new sphere(vec3(-4, 1, 0), 1.0);
        d_materials[i] = new material(material::Metal, vec3(0.7, 0.6, 0.5), 0, 0);
        d_spheres[i++] = new sphere(vec3(4, 1, 0), 1.0);
        *rnd_state = local_rand_state;
        *d_world = new hitable_list(d_spheres, d_materials, 22 * 22 + 1 + 3);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            30.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
    }
}

__global__ void free_world(sphere **d_spheres, material **d_materials, hitable_list **d_world, camera **d_camera) {
    for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
        delete d_materials[i];
        delete d_spheres[i];
    }
    delete *d_world;
    delete *d_camera;
}

void write_image(const char* output_file, const vec3 *fb, const int nx, const int ny) {
    char *data = new char[nx * ny * 3];
    int idx = 0;
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            data[idx++] = int(255.99*fb[pixel_index].r());
            data[idx++] = int(255.99*fb[pixel_index].g());
            data[idx++] = int(255.99*fb[pixel_index].b());
        }
    }
    stbi_write_png(output_file, nx, ny, 3, (void*)data, nx * 3);
    delete[] data;
}

int main() {
    int nx = 1200;
    int ny = 800;
    int ns = 10;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *d_fb;
    checkCudaErrors(cudaMalloc((void **)&d_fb, fb_size));

    // allocate random state
    rand_state *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(rand_state)));
    rand_state *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(rand_state)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init << <1, 1 >> >(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    sphere **d_spheres;
    material **d_materials;
    int num_hitables = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **)&d_spheres, num_hitables * sizeof(sphere *)));
    checkCudaErrors(cudaMalloc((void **)&d_materials, num_hitables * sizeof(material *)));
    hitable_list **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable_list *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world <<<1, 1 >>>(d_spheres, d_materials, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> >(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render << <blocks, threads >> >(d_fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    vec3* h_fb = new vec3[fb_size];
    checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
    write_image("output.png", h_fb, nx, ny);
    delete[] h_fb;
    h_fb = NULL;

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world <<<1, 1 >>>(d_spheres, d_materials, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_spheres));
    checkCudaErrors(cudaFree(d_materials));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_fb));

    cudaDeviceReset();
}