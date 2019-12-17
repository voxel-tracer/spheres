#pragma once

#include <fstream>

#include "sphere.h"
#include "bvh.h"
#include "utils.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

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

void load_from_csv(const char *input, scene& sc, int numPrimitivesPerLeaf) {
    std::vector<std::vector<float>> data = parse2DCsvFile(input);
    int size = data.size();
    const bool addMarker = (size % numPrimitivesPerLeaf) > 0;
    sc.spheres_size = addMarker ? size + 1 : size; // add room for the end marker
    sc.spheres = new sphere[sc.spheres_size];

    int max_gen = 0;
    int i = 0;
    for (auto l : data) {
        int parent = (int)l[1];
        int gen = 1 + (parent > 0 ? sc.spheres[parent - 1].color : 0);
        max_gen = max(gen, max_gen);
        sc.spheres[i++] = sphere(vec3(l[2], l[3], l[4]), gen);
    }

    // normalize color idx such that max_gen = 256
    float normalizer = 255.0f / max_gen;
    for (int i = 0; i < size; i++) {
        int gen = sc.spheres[i].color;
        sc.spheres[i].color = (int) (gen * normalizer);
    }

    sc.bvh = build_bvh(sc.spheres, size, numPrimitivesPerLeaf, sc.bvh_size);

    if (addMarker)
        sc.spheres[size] = sphere(vec3(INFINITY, INFINITY, INFINITY), 0);
}

struct elevationData {
    const unsigned char* data;
    const unsigned int width;
    const unsigned int height;

    elevationData(unsigned char* d, unsigned int w, unsigned int h) : data(d), width(w), height(h) {}

    unsigned char elevation(int x, int y) const {
        if (x < 0 || x >= width) return 255;
        if (y < 0 || y >= height)return 255;
        return data[y * width + x];
    }

    unsigned char minNeighbors(int x, int y) const {
        unsigned char minN = 255;
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                minN = min(minN, elevation(x + i, y + j));
        return minN;
    }
};

void load_elevation(const char* input, scene& sc, int numPrimitivesPerLeaf) {
    // load heightmap image
    int image_x, image_y, image_n;
    int image_desired_channels = 1; // grayscale
    unsigned char* data = stbi_load(input, &image_x, &image_y, &image_n, image_desired_channels);
    const elevationData eData(data, image_x, image_y);

    // count total number of voxels to allocate the correct number of spheres
    int size = 0;
    for (int y = 0, idx = 0; y < eData.height; y++)
        for (int x = 0; x < eData.width; x++, idx++)
            size += (eData.elevation(x, y) - eData.minNeighbors(x, y) + 1);

    const bool addMarker = (size % numPrimitivesPerLeaf) > 0;
    sc.spheres_size = addMarker ? size + 1 : size; // add room for the end marker
    sc.spheres = new sphere[sc.spheres_size];

    // go through all pixels and generate enough primitives to fill out the column
    // center the whole model around (0, 0, 0) => translate all primitives by (-image_x/2, 0, -image_y/2)
    int max_e = 0;
    for (int y = 0, out=0; y < eData.height; y++) {
        for (int x = 0; x < eData.width; x++) {
            const unsigned int e = eData.elevation(x, y);
            const unsigned char minN = eData.minNeighbors(x, y);
            max_e = max(max_e, e);
            for (int z = minN; z <= e;z++)
                sc.spheres[out++] = sphere(vec3(x + 0.5f - image_x / 2, y + 0.5f - image_y / 2, z + 0.5f), e);
        }
    }

    // normalize color idx such that max_gen = 256
    float normalizer = 255.0f / max_e;
    for (int i = 0; i < size; i++) {
        int gen = sc.spheres[i].color;
        sc.spheres[i].color = (int)(gen * normalizer);
    }

    stbi_image_free(data);

    sc.bvh = build_bvh(sc.spheres, size, numPrimitivesPerLeaf, sc.bvh_size);

    if (addMarker)
        sc.spheres[size] = sphere(vec3(INFINITY, INFINITY, INFINITY), 0);
}