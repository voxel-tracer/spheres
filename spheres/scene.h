#pragma once

#include <cuda_runtime.h>

#include "vec3.h"
#include "sphere.h"

#include <time.h>

//#undef NDEBUG
#include <cassert>

#include "constants.h"
#include "bvh.h"
#include "utils.h"

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

// step 1: allocate memory for the constant part
__device__ __constant__ bvh_node d_nodes[2048];
__device__ __constant__ float d_colormap[256 * 3];

texture<float4> t_bvh;
texture<float> t_spheres;
float* d_bvh_buf;
float* d_spheres_buf;

struct scene {
    sphere* spheres;
    int spheres_size;

    bvh_node* bvh;
    int bvh_size;
};

void store_to_binary(const char *output, const scene& sc) {
    fstream out(output, ios::out | ios::binary);
    out.write((char*)& sc.spheres_size, sizeof(int));
    out.write((char*)sc.spheres, sizeof(sphere) * sc.spheres_size);
    out.write((char*)& sc.bvh_size, sizeof(int));
    out.write((char*)sc.bvh, sizeof(bvh_node) * sc.bvh_size);
    out.close();
}

void load_from_binary(const char *input, scene& sc) {
    fstream in(input, ios::in | ios::binary);
    in.read((char*)& sc.spheres_size, sizeof(int));
    sc.spheres = new sphere[sc.spheres_size];
    in.read((char*)sc.spheres, sizeof(sphere) * sc.spheres_size);

    in.read((char*)& sc.bvh_size, sizeof(int));
    sc.bvh = new bvh_node[sc.bvh_size];
    in.read((char*)sc.bvh, sizeof(bvh_node) * sc.bvh_size);
}

void load_from_csv(const char *input, scene& sc) {
    vector<vector<float>> data = parse2DCsvFile(input);
    // make sure we only load N such that (N/lane_size_spheres) is a multiple of 2
    int size = data.size();
    size /= 10;
    sc.spheres_size = powf(2, (int)(log2f((float)size))) * 10;
    sc.spheres = new sphere[sc.spheres_size];

    int max_gen = 0;
    int i = 0;
    for (auto l : data) {
        int parent = (int)l[1];
        int gen = 1 + (parent > 0 ? sc.spheres[parent - 1].color : 0);
        max_gen = max(gen, max_gen);
        sc.spheres[i++] = sphere(vec3(l[2], l[3], l[4]), gen);
        if (i == sc.spheres_size)
            break;
    }

    // normalize color idx such that max_gen = 256
    float normalizer = 255.0f / max_gen;
    for (int i = 0; i < sc.spheres_size; i++) {
        int gen = sc.spheres[i].color;
        sc.spheres[i].color = (int) (gen * normalizer);
    }

    sc.bvh = build_bvh(sc.spheres, sc.spheres_size, sc.bvh_size);
}

void setup_scene(char *input, bool csv, float *colormap, int **d_colors, int& bvh_size) {
    scene sc;

    if (csv) {
        load_from_csv(input, sc);
        store_to_binary(strcat(input, ".bin"), sc);
    }
    else {
        load_from_binary(input, sc);
    }

    bvh_size = sc.bvh_size;

    // once we build the tree, copy the first 2048 nodes to constant memory
    const int const_size = min(2048, sc.bvh_size);
    checkCudaErrors(cudaMemcpyToSymbol(d_nodes, sc.bvh, const_size * sizeof(bvh_node)));

    // copy colors to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(d_colormap, colormap, 256 * 3 * sizeof(float)));

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
    float *floats = new float[spheres_size_float];
    int *colors = new int[sc.spheres_size];
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

    checkCudaErrors(cudaMalloc((void **) d_colors, sc.spheres_size * sizeof(int)));
    checkCudaErrors(cudaMemcpy(*d_colors, colors, sc.spheres_size * sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)& d_spheres_buf, spheres_size_float * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_spheres_buf, floats, spheres_size_float * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaBindTexture(NULL, t_spheres, (void*)d_spheres_buf, spheres_size_float * sizeof(float)));

    delete[] sc.bvh;
    delete[] floats;
    delete[] sc.spheres;
    delete[] colors;
}

void releaseScene(int *d_colors) {
    // destroy texture object
    checkCudaErrors(cudaUnbindTexture(t_bvh));
    checkCudaErrors(cudaUnbindTexture(t_spheres));
    checkCudaErrors(cudaFree(d_bvh_buf));
    checkCudaErrors(cudaFree(d_spheres_buf));
    checkCudaErrors(cudaFree(d_colors));
}