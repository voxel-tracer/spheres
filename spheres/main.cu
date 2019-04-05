#include <float.h>

#include "ray.h"
#include "camera.h"
#include "scene.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, const scene s, rand_state& rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1, 1, 1);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if (hit_bvh(s, cur_ray, 0.001f, FLT_MAX, rec)) {
            const vec3 p = cur_ray.point_at_parameter(rec.t);
            vec3 target = rec.n + random_in_unit_sphere(rand_state);
            cur_ray = ray(p, target);
            cur_attenuation *= vec3(.05f, .05f, .35f);
        }
        else if (i == 0) {
            break; // black background
        }
        else {
            float t = 0.5f*(cur_ray.direction().y() + 1.0f);
            vec3 c = (1.0f - t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void render(vec3 *fb, const scene sc, int max_x, int max_y, int ns, const camera cam) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    rand_state state = (wang_hash(pixel_index) * 336343633) | 1;
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + random_float(state)) / float(max_x);
        float v = float(j + random_float(state)) / float(max_y);
        ray r = cam.get_ray(u, v, state);
        col += color(r, sc, state);
    }
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
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
        cerr << "usage spheres file_name [num_samples=1] [num_runs=1] [camera_dist=100]";
        exit(-1);
    }
    const char* input = argv[1];
    const int nx = 1200;
    const int ny = 800;
    const int ns = (argc > 2) ? strtol(argv[2], NULL, 10) : 1;
    const int tx = 8;
    const int ty = 8;
    const int nr = (argc > 3) ? strtol(argv[3], NULL, 10) : 1;
    const int dist = (argc > 4) ? strtof(argv[4], NULL) : 100;
    
    cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *d_fb;
    checkCudaErrors(cudaMalloc((void **)&d_fb, fb_size));

    // setup scene
    scene sc;
    setup_scene(input, sc);

    camera cam = setup_camera(nx, ny, dist);

    double *runs = new double[nr];
    for (int r = 0; r < nr; r++) {
        clock_t start, stop;
        start = clock();
        // Render our buffer
        dim3 blocks(nx / tx + 1, ny / ty + 1);
        dim3 threads(tx, ty);
        render << <blocks, threads >> >(d_fb, sc, nx, ny, ns, cam);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        stop = clock();
        runs[r] = ((double)(stop - start)) / CLOCKS_PER_SEC;
        cerr << "took " << runs[r] << " seconds.\n";
    }
    if (nr > 1) {
        // compute median
        qsort(runs, nr, sizeof(double), cmpfunc);
        cerr << "median run took " << runs[nr / 2] << " seconds.\n";
    }
    delete[] runs;
    runs = NULL;

    // Output FB as Image
    vec3* h_fb = new vec3[fb_size];
    checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
    char file_name[100];
    sprintf(file_name, "%s_%dx%dx%d_%d_bvh.png", input, nx, ny, ns, dist);
    write_image(file_name, h_fb, nx, ny);
    delete[] h_fb;
    h_fb = NULL;

    // clean up
    releaseScene(sc);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_fb));

    cudaDeviceReset();
}