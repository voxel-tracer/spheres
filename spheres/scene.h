#pragma once

#include "vec3.h"
#include "sphere.h"

#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>

//#undef NDEBUG
#include <cassert>

//#define COUNT_BVH

using namespace std;

const unsigned int lane_size_float = 64 / sizeof(float);
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

// step 1: allocate memory for the constant part
__device__ __constant__ bvh_node d_nodes[2048];
__device__ __constant__ float d_colormap[256 * 3];

struct scene {
    float* spheres;
    bvh_node* bvh;
    int count;
    int *colors;

#ifdef COUNT_BVH
    unsigned int *counters;
#endif
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

bvh_node* build_bvh(sphere *l, int n, int &count) {
    cout << " building BVH...";
    clock_t start, stop;
    start = clock();

    count = n / lane_size_spheres;
    cout << " of size " << 2 * count << endl;
    bvh_node* nodes = new bvh_node[2 * count];
    build_bvh(nodes, 1, l, n);

    stop = clock();
    cout << "done in " << ((double)(stop - start)) / CLOCKS_PER_SEC << "s" << endl;

    return nodes;
}

void store_to_binary(const char *output, const bvh_node *nodes, const sphere *spheres, int num_spheres, int num_nodes) {
    cout << "Saving scene to binary file" << endl;
    fstream out(output, ios::out | ios::binary);
    out.write((char*)&num_spheres, sizeof(num_spheres));
    out.write((char*)spheres, sizeof(sphere)*num_spheres);
    out.write((char*)&num_nodes, sizeof(num_nodes));
    out.write((char*)nodes, sizeof(bvh_node)*num_nodes);
    out.close();
}

void load_from_binary(const char *input, sphere **spheres, bvh_node **nodes, int &num_spheres, int &num_nodes) {
    cout << "Loading scene from disk";
    fstream in(input, ios::in | ios::binary);
    in.read((char*)&num_spheres, sizeof(num_spheres));
    *spheres = new sphere[num_spheres];
    in.read((char*)(*spheres), sizeof(sphere)*num_spheres);

    in.read((char*)&num_nodes, sizeof(num_nodes));
    *nodes = new bvh_node[num_nodes];
    in.read((char*)(*nodes), sizeof(bvh_node)*num_nodes);
    cout << ", loaded " << num_spheres << " spheres, and " << num_nodes << " bvh nodes" << endl;
}

void load_from_csv(const char *input, sphere **spheres, bvh_node **nodes, int &num_spheres, int &num_nodes) {
    cout << "Loading spheres from disk";
    vector<vector<float>> data = parse2DCsvFile(input);
    // make sure we only load N such that (N/lane_size_spheres) is a multiple of 2
    int size = data.size();
    size /= 10;
    num_spheres = powf(2, (int)(log2f((float)size))) * 10;
    cout << ", loaded " << num_spheres << " spheres" << endl;
    *spheres = new sphere[num_spheres];
    int max_gen = 0;
    int i = 0;
    for (auto l : data) {
        int parent = (int)l[1];
        int gen = 1 + (parent > 0 ? (*spheres)[parent - 1].color : 0);
        max_gen = max(gen, max_gen);
        (*spheres)[i++] = sphere(vec3(l[2], l[3], l[4]), gen);
        if (i == num_spheres)
            break;
    }

    // normalize color idx such that max_gen = 256
    float normalizer = 255.0f / max_gen;
    for (int i = 0; i < num_spheres; i++) {
        int gen = (*spheres)[i].color;
        (*spheres)[i].color = (int) (gen * normalizer);
    }

    int half_num_nodes;
    *nodes = build_bvh(*spheres, num_spheres, half_num_nodes);
    num_nodes = half_num_nodes * 2;
}

void setup_scene(const char *input, scene &sc, bool csv, float *colormap) {
    int num_spheres, num_nodes;
    bvh_node *nodes;
    sphere *spheres;

    if (csv) {
        load_from_csv(input, &spheres, &nodes, num_spheres, num_nodes);
        store_to_binary((std::string(input) + ".bin").c_str(), nodes, spheres, num_spheres, num_nodes);
    }
    else {
        load_from_binary((std::string(input) + ".bin").c_str(), &spheres, &nodes, num_spheres, num_nodes);
    }
    sc.count = num_nodes / 2;

    // once we build the tree, copy the first 2048 nodes to constant memory
    const int const_size = min(2048, num_nodes);
    checkCudaErrors(cudaMemcpyToSymbol(d_nodes, nodes, const_size * sizeof(bvh_node)));
    checkCudaErrors(cudaMemcpyToSymbol(d_colormap, colormap, 256 * 3 * sizeof(float)));
    cout << "copied " << const_size << " nodes to constant memory." << endl;

    // copy remaining nodes to global memory
    int bvh_size = num_nodes - const_size;
    if (bvh_size > 0) {
        checkCudaErrors(cudaMalloc((void **)&sc.bvh, bvh_size * sizeof(bvh_node)));
        checkCudaErrors(cudaMemcpy(sc.bvh, nodes + const_size, bvh_size * sizeof(bvh_node), cudaMemcpyHostToDevice));
        cout << "copied " << bvh_size << " nodes to global memory." << endl;
    }
    else {
        sc.bvh = NULL;
    }

    delete[] nodes;

    cout << "copying spheres to device" << endl;
    cout.flush();

    const int scene_size_float = lane_size_float * (num_spheres / lane_size_spheres);

    // copy the spheres in array of floats
    // do it after we build the BVH as it would have moved the spheres around
    float *floats = new float[scene_size_float];
    int *colors = new int[num_spheres];
    int idx = 0;
    int i = 0;
    while (i < num_spheres) {
        for (int j = 0; j < lane_size_spheres; j++, i++) {
            floats[idx++] = spheres[i].center.x();
            floats[idx++] = spheres[i].center.y();
            floats[idx++] = spheres[i].center.z();
            colors[i] = spheres[i].color;
        }
        idx += lane_padding_float; // padding
    }
    assert(idx == scene_size_float);

    checkCudaErrors(cudaMalloc((void **)&sc.colors, num_spheres * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&sc.spheres, scene_size_float * sizeof(float)));
    checkCudaErrors(cudaMemcpy(sc.colors, colors, num_spheres * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(sc.spheres, floats, scene_size_float * sizeof(float), cudaMemcpyHostToDevice));

#ifdef COUNT_BVH
    checkCudaErrors(cudaMalloc((void **)&sc.counters, 32 * sizeof(long)));
    checkCudaErrors(cudaMemset(sc.counters, 0, 32 * sizeof(long)));
#endif

    delete[] floats;
    delete[] spheres;
    delete[] colors;
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

void releaseScene(scene& sc) {
#ifdef COUNT_BVH
    unsigned int h_counters[32];
    unsigned long total = 0;
    checkCudaErrors(cudaMemcpy(h_counters, sc.counters, 32 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cout << "counters collected per level:" << endl;
    for (size_t i = 0; i < 32; i++) {
        cout << " " << i << ": " << h_counters[i] << endl;
        total += h_counters[i];
    }
    cout << endl << "total " << total << endl;
    

    checkCudaErrors(cudaFree(sc.counters));
#endif
    if (sc.bvh != NULL) {
        checkCudaErrors(cudaFree(sc.bvh));
    }
    checkCudaErrors(cudaFree(sc.spheres));
    checkCudaErrors(cudaFree(sc.colors));
}

__device__ bool hit_bvh(const scene& sc, const ray& r, float t_min, float t_max, hit_record &rec) {

    bool down = true;
    int idx = 1;
    bool found = false;
    float closest = t_max;

    unsigned int move_bit_stack = 0;
    int lvl = 0;

#ifdef COUNT_BVH
    atomicAdd(sc.counters, 1); // count how many rays reached this method
#endif

    while (true) {
        if (down) {
            bvh_node node = (idx < 2048) ? d_nodes[idx] : sc.bvh[idx - 2048];
#ifdef COUNT_BVH
            atomicAdd(sc.counters + lvl + 1, 1);
#endif
            if (hit_bbox(node, r, t_min, closest)) {
                if (idx >= sc.count) { // leaf node
#ifdef COUNT_BVH
                    atomicAdd(sc.counters + lvl + 2, 1);
#endif
                    int m = (idx - sc.count) * lane_size_float;
                    #pragma unroll
                    for (int i = 0; i < lane_size_spheres; i++) {
                        vec3 center(sc.spheres[m++], sc.spheres[m++], sc.spheres[m++]);
                        if (hit_point(center, r, t_min, closest, rec)) {
                            found = true;
                            closest = rec.t;
                            rec.idx = (idx - sc.count)*lane_size_spheres + i;
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
    }

    return found;
}
