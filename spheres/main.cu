#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>

#include <vector>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "camera.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        cerr << "CUDA error = " << cudaGetErrorString(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

const int kSphereCount = 5001;

__device__ __constant__ sphere d_spheres[kSphereCount];

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
            const vec3 normal = d_spheres[rec.hit_idx].normal(rec.p);
            vec3 target = normal + random_in_unit_sphere(rand_state);
            cur_ray = ray(rec.p, target);
            cur_attenuation *= vec3(.35f, .35f, .35f);
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
        col += color(r, state);
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


/**
* Reads csv file into table, exported as a vector of vector of doubles.
* @param inputFileName input file name (full path).
* @return data as vector of vector of doubles.
*
* code adapted from https://waterprogramming.wordpress.com/2017/08/20/reading-csv-files-in-c/
*/
vector<vector<float>> parse2DCsvFile(string inputFileName) {

    vector<vector<float> > data;
    ifstream inputFile(inputFileName);
    int l = 0;

    while (inputFile) {
        l++;
        string s;
        if (!getline(inputFile, s)) break;
        if (s[0] != '#') {
            istringstream ss(s);
            vector<float> record;

            while (ss) {
                string line;
                if (!getline(ss, line, ','))
                    break;
                try {
                    record.push_back(stof(line));
                }
                catch (const std::invalid_argument e) {
                    cout << "NaN found in file " << inputFileName << " line " << l
                        << endl;
                    e.what();
                }
            }

            data.push_back(record);
        }
    }

    if (!inputFile.eof()) {
        cerr << "Could not read file " << inputFileName << "\n";
        exit(99);
    }

    return data;
}

sphere* setup_scene() {
    sphere* spheres = new sphere[kSphereCount];

    cerr << "Loading spheres from disk\n";
    vector<vector<float>> data = parse2DCsvFile("s5k.csv");
    int i = 0;
    for (auto l : data) {
        vec3 center(l[2], l[3], l[4]);
        spheres[i++] = sphere(center);
    }

    cout << "scene uses " << kSphereCount * sizeof(vec3) << " bytes";

    return spheres;
}

camera setup_camera(int nx, int ny) {
    vec3 lookfrom(100, 100, 100);
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
    const int nx = 1200;
    const int ny = 800;
    const int ns = (argc > 1) ? strtol(argv[1], NULL, 10) : 1;
    const int tx = 8;
    const int ty = 8;
    const int nr = (argc > 2) ? strtol(argv[2], NULL, 10) : 1;
    
    cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *d_fb;
    checkCudaErrors(cudaMalloc((void **)&d_fb, fb_size));

    // setup scene
    sphere* h_spheres = setup_scene();

    // copy the scene to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(d_spheres, h_spheres, kSphereCount * sizeof(sphere)));

    camera cam = setup_camera(nx, ny);

    double *runs = new double[nr];
    for (int r = 0; r < nr; r++) {
        clock_t start, stop;
        start = clock();
        // Render our buffer
        dim3 blocks(nx / tx + 1, ny / ty + 1);
        dim3 threads(tx, ty);
        render << <blocks, threads >> >(d_fb, nx, ny, ns, cam);
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
    write_image("output.png", h_fb, nx, ny);
    delete[] h_fb;
    h_fb = NULL;

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_fb));

    cudaDeviceReset();
}