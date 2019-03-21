#pragma once

#include "vec3.h"
#include "sphere.h"

#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>

#undef NDEBUG
#include <cassert>

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

/**
non-leaf nodes represent bounding box of all spheres that are inside it
leaf nodes represent 2 spheres
*/
struct bvh_node {
    __host__ bvh_node() {}
    __host__ bvh_node(const vec3& A, const vec3& B) :a(A), b(B) {}

    __device__ vec3 min() const { return a; }
    __device__ vec3 max() const { return b; }
    __device__ vec3 left() const { return a; }
    __device__ vec3 right() const { return b; }

    vec3 a;
    vec3 b;
};

struct scene {
    float* spheres;
    bvh_node* bvh;
    int count;
};

unsigned int g_state = 0;

float drand48() {
    g_state = (214013 * g_state + 2531011);
    return (float)((g_state >> 16) & 0x7FFF) / 32767;
}

int box_x_compare(const void* a, const void* b) {
    return (*(sphere*)a).center[0] - (*(sphere*)b).center[0];
}

int box_y_compare(const void* a, const void* b) {
    return (*(sphere*)a).center[1] - (*(sphere*)b).center[1];
}

int box_z_compare(const void* a, const void* b) {
    return (*(sphere*)a).center[2] - (*(sphere*)b).center[2];
}

vec3 minof(const sphere *l, int n) {
    vec3 min = l[0].center;
    for (int i = 1; i < n; i++) {
        for (int a = 0; a < 3; a++)
            min[a] = fminf(min[a], l[i].center[a]);
    }
    return min;
}

vec3 maxof(const sphere *l, int n) {
    vec3 max = l[0].center;
    for (int i = 1; i < n; i++) {
        for (int a = 0; a < 3; a++)
            max[a] = fmaxf(max[a], l[i].center[a]);
    }
    return max;
}

void build_bvh(bvh_node *nodes, int idx, sphere *l, int n) {
    assert(n >= 10);
    nodes[idx] = bvh_node(minof(l, n), maxof(l, n));

    if (n > 10) {
        int axis = int(3 * drand48());
        if (axis == 0)
            qsort(l, n, sizeof(sphere), box_x_compare);
        else if (axis == 1)
            qsort(l, n, sizeof(sphere), box_y_compare);
        else
            qsort(l, n, sizeof(sphere), box_z_compare);

        build_bvh(nodes, idx * 2, l, n / 2);
        build_bvh(nodes, idx * 2 + 1, l + n / 2, n / 2);
    }
}

void build_bvh(sphere *l, int n, scene& sc) {
    cout << " building BVH...";
    clock_t start, stop;
    start = clock();

    sc.count = n / 10;
    cout << " of size " << 2 * sc.count;
    bvh_node* nodes = new bvh_node[2 * sc.count];
    build_bvh(nodes, 1, l, n);

    // copy bvh to device
    int bvh_size = 2 * sc.count * sizeof(bvh_node);
    cout << " uses " << bvh_size << " bytes...";

    stop = clock();
    cout << "done in " << ((double)(stop - start)) / CLOCKS_PER_SEC << "s" << endl;

    checkCudaErrors(cudaMalloc((void **)&sc.bvh, bvh_size));
    checkCudaErrors(cudaMemcpy(sc.bvh, nodes, bvh_size, cudaMemcpyHostToDevice));

    delete[] nodes;
}

void setup_scene(const char *input, scene &sc) {
    cout << "Loading spheres from disk";
    vector<vector<float>> data = parse2DCsvFile(input);
    int n = data.size();
    cout << ", loaded " << n << " spheres";
    sphere* spheres = new sphere[n];
    int i = 0;
    for (auto l : data) {
        vec3 center(l[2], l[3], l[4]);
        spheres[i++] = sphere(center);
    }
    
    const int scene_size = 32 * (n / 10); // store every 10 spheres in 32 floats (one mem lane)
    cout << ", uses " << (scene_size * sizeof(float)) << " bytes" << endl;

    build_bvh(spheres, n, sc);

    // copy the spheres in array of floats
    // do it after we build the BVH as it would have moved the spheres around
    float *floats = new float[scene_size];
    int idx = 0;
    i = 0;
    while (i < n) {
        for (int j = 0; j < 10; j++, i++) {
            floats[idx++] = spheres[i].center.x();
            floats[idx++] = spheres[i].center.y();
            floats[idx++] = spheres[i].center.z();
        }
        idx += 2; // padding
    }
    assert(idx == scene_size);

    checkCudaErrors(cudaMalloc((void **)&sc.spheres, scene_size * sizeof(float)));
    checkCudaErrors(cudaMemcpy(sc.spheres, floats, scene_size * sizeof(float), cudaMemcpyHostToDevice));

    delete[] floats;
    delete[] spheres;
}

__device__ bool hit_bbox(const bvh_node& node, const ray& r, float t_min, float t_max) {
    for (int a = 0; a < 3; a++) {
        float invD = 1.0f / r.direction()[a];
        float t0 = (node.min()[a] - 1 - r.origin()[a])*invD;
        float t1 = (node.max()[a] + 1 - r.origin()[a])*invD;
        if (invD < 0.0f) {
            float tmp = t0; t0 = t1; t1 = tmp;
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min)
            return false;
    }

    return true;
}

__device__ bool hit_bvh(const scene& sc, const ray& r, float t_min, float t_max, hit_record &rec) {
    
    bool down = true;
    int idx = 1;
    bool found = false;
    float closest = t_max;
    bvh_node node = sc.bvh[1];

    while (true) {
        if (down) {
            if (idx >= sc.count) { // node is leaf
                if (hit_bbox(node, r, t_min, closest)) {
                    // compute index of first sphere's float
                    int m = (idx - sc.count) * 32;
                    for (int i = 0; i < 10; i++) {
                        vec3 center(sc.spheres[m++], sc.spheres[m++], sc.spheres[m++]);
                        if (hit_sphere(center, r, t_min, closest, rec)) {
                            found = true;
                            closest = rec.t;
                        }
                    }
                }
                down = false;
            }
            else if (hit_bbox(node, r, t_min, closest)) {
                idx = idx * 2; // node = node.left
                node = sc.bvh[idx];
                // keep going down
            }
            else {
                down = false;
            }
        }
        else if (idx == 1) {
            break;
        }
        else if ((idx % 2) == 0) { // node is left
            idx++; // node = node.sibling
            node = sc.bvh[idx];
            down = true;
        }
        else {
            idx = idx / 2; // node = node.parent
            node = sc.bvh[idx];
        }
    }

    return found;
}