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

const unsigned int lane_size_float = 128 / sizeof(float);
const unsigned int lane_size_spheres = lane_size_float / 3;
const unsigned int lane_padding_float = lane_size_float - lane_size_spheres * 3;

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

    __host__ __device__ unsigned int split_axis() const { return max_component(b - a); }

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
    assert(n >= lane_size_spheres);
    nodes[idx] = bvh_node(minof(l, n), maxof(l, n));

    if (n > lane_size_spheres) {
        const unsigned int axis = nodes[idx].split_axis();
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

    sc.count = n / lane_size_spheres;
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
    
    const int scene_size_float = lane_size_float * (n / lane_size_spheres);
    cout << ", uses " << (scene_size_float * sizeof(float)) << " bytes" << endl;

    build_bvh(spheres, n, sc);

    cout << "copying spheres to device" << endl;
    cout.flush();

    // copy the spheres in array of floats
    // do it after we build the BVH as it would have moved the spheres around
    float *floats = new float[scene_size_float];
    int idx = 0;
    i = 0;
    while (i < n) {
        for (int j = 0; j < lane_size_spheres; j++, i++) {
            floats[idx++] = spheres[i].center.x();
            floats[idx++] = spheres[i].center.y();
            floats[idx++] = spheres[i].center.z();
        }
        idx += lane_padding_float; // padding
    }
    assert(idx == scene_size_float);

    checkCudaErrors(cudaMalloc((void **)&sc.spheres, scene_size_float * sizeof(float)));
    checkCudaErrors(cudaMemcpy(sc.spheres, floats, scene_size_float * sizeof(float), cudaMemcpyHostToDevice));

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

__device__ __forceinline__ float get_left(const vec3& d, const int axis) {
    if (axis == 0)
        return (d.x() >= 0) ? 0 : 1;
    else if (axis==1)
        return (d.y() >= 0) ? 0 : 1;
    return (d.z() >= 0) ? 0 : 1;
}

__device__ __forceinline__ float get_right(const vec3& d, const int axis) {
    if (axis == 0)
        return (d.x() >= 0) ? 1 : -1;
    else if (axis == 1)
        return (d.y() >= 0) ? 1 : -1;
    return (d.z() >= 0) ? 1 : -1;
}

__device__ bool hit_bvh(const scene& sc, const ray& r, float t_min, float t_max, hit_record &rec) {
    
    bool down = true;
    int idx = 1;
    bool found = false;
    float closest = t_max;
    bvh_node node = sc.bvh[1];

    // precompute move increments
    //int move_left[3], move_right[3];
    //for (unsigned int i = 0; i < 3; i++) {
    //    move_left[i] = (r.direction()[i] >= 0) ? 0 : 1;
    //    move_right[i] = (r.direction()[i] >= 0) ? 1 : -1;
    //}

    while (true) {
        if (down) {
            if (hit_bbox(node, r, t_min, closest)) {
                if (idx >= sc.count) { // leaf node
                    int m = (idx - sc.count) * lane_size_float;
                    #pragma unroll
                    for (int i = 0; i < lane_size_spheres; i++) {
                        vec3 center(sc.spheres[m++], sc.spheres[m++], sc.spheres[m++]);
                        if (hit_sphere(center, r, t_min, closest, rec)) {
                            found = true;
                            closest = rec.t;
                        }
                    }
                    down = false;
                }
                else {
                    // keep going down
                    idx = idx * 2 + get_left(r.direction(), node.split_axis()); // node = node.left
                    node = sc.bvh[idx];
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
            // let's read the parent again
            node = sc.bvh[idx / 2];
            if ((idx % 2) == get_left(r.direction(), node.split_axis())) { // go to sibling
                idx += get_right(r.direction(), node.split_axis()); // node = node.sibling
                node = sc.bvh[idx];
                down = true;
            }
            else { // go up
                idx = idx / 2; // node = node.parent
                //node = sc.bvh[idx]; // we already read parent node
            }
        }
    }

    return found;
}
