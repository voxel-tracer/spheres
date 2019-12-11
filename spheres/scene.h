#pragma once

#include <fstream>

#include "sphere.h"
#include "bvh.h"
#include "utils.h"

struct scene {
    sphere* spheres;
    int spheres_size;

    bvh_node* bvh;
    int bvh_size;

    void release() {
        delete[] spheres;
        delete[] bvh;
    }
};

void store_to_binary(const char *output, const scene& sc) {
    std::fstream out(output, std::ios::out | std::ios::binary);
    out.write((char*)& sc.spheres_size, sizeof(int));
    out.write((char*)sc.spheres, sizeof(sphere) * sc.spheres_size);
    out.write((char*)& sc.bvh_size, sizeof(int));
    out.write((char*)sc.bvh, sizeof(bvh_node) * sc.bvh_size);
    out.close();
}

void load_from_binary(const char *input, scene& sc) {
    std::fstream in(input, std::ios::in | std::ios::binary);
    in.read((char*)& sc.spheres_size, sizeof(int));
    sc.spheres = new sphere[sc.spheres_size];
    in.read((char*)sc.spheres, sizeof(sphere) * sc.spheres_size);

    in.read((char*)& sc.bvh_size, sizeof(int));
    sc.bvh = new bvh_node[sc.bvh_size];
    in.read((char*)sc.bvh, sizeof(bvh_node) * sc.bvh_size);
}

void load_from_csv(const char *input, scene& sc) {
    std::vector<std::vector<float>> data = parse2DCsvFile(input);
    // make sure we only load N such that (N/lane_size_spheres) is a multiple of 2
    int size = data.size();
    size /= lane_size_spheres;
    sc.spheres_size = powf(2, (int)(log2f((float)size))) * lane_size_spheres;
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
