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

struct MapBoxTerrainRGB {
    const unsigned char* data;
    const unsigned int width;
    const unsigned int height;
    const float minAll;
    const float maxAll;
    const unsigned char _scale;

    MapBoxTerrainRGB(unsigned char* d, unsigned int w, unsigned int h, unsigned char scale) : data(d), width(w), height(h), _scale(scale), minAll(_minAll()), maxAll(_maxAll()) {}

    // scale all elevations to be [0, 256[ => e = round((e - minAll)*255/(maxAll-minAll))
    unsigned char elevation(int x, int y) const {
        float e = elevationRaw(x, y);
        return (unsigned char)roundf((e - minAll) * _scale / (maxAll - minAll));
    }

    float elevationRaw(int x, int y) const {
        if (x < 0 || x >= width) return FLT_MAX;
        if (y < 0 || y >= height)return FLT_MAX;

        const unsigned int r = data[(y * width + x) * 3];
        const unsigned int g = data[(y * width + x) * 3 + 1];
        const unsigned int b = data[(y * width + x) * 3 + 2];

        return -10000 + ((r * 256 * 256 + g * 256 + b) * 0.1f);
    }

    float _minAll() const {
        float min = FLT_MAX;
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                min = fminf(min, elevationRaw(x, y));
            }
        }
        return min;
    }

    float _maxAll() const {
        float max = -FLT_MAX;
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                max = fmaxf(max, elevationRaw(x, y));
            }
        }
        return max;
    }

    unsigned char minNeighbors(int x, int y) const {
        unsigned char minN = UCHAR_MAX;
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                minN = min(minN, elevation(x + i, y + j));
        return minN;
    }
};

void loadSatelliteImage(const char* input, vec3** data, unsigned int& size) {
    int image_x, image_y, image_n;
    unsigned char* _data = stbi_load(input, &image_x, &image_y, &image_n, 3);

    size = image_x * image_y;
    *data = new vec3[size];
    for (int i = 0, idx = 0; i < size; i++) {
        float r = _data[idx++] / 255.0f;
        float g = _data[idx++] / 255.0f;
        float b = _data[idx++] / 255.0f;
        (*data)[i] = vec3(r, g, b);
    }

    stbi_image_free(_data);
}

void loadMapBoxTerrainRGB(const char* input, scene& sc, int numPrimitivesPerLeaf) {
    // elevation is stored as RGB
    int imageX, imageY, imageN;
    unsigned char* data = stbi_load(input, &imageX, &imageY, &imageN, 3);
    const MapBoxTerrainRGB terrain(data, imageX, imageY, 64);


    // count total number of voxels to allocate the correct number of primitives
    int size = 0;
    for (int y = 0, idx = 0; y < terrain.height; y++)
        for (int x = 0; x < terrain.width; x++, idx++)
            size += (terrain.elevation(x, y) - terrain.minNeighbors(x, y) + 1);

    const bool addMarker = (size % numPrimitivesPerLeaf) > 0;
    sc.spheres_size = addMarker ? size + 1 : size; // add room for the end marker
    sc.spheres = new sphere[sc.spheres_size];

    // go through all pixels and generate enough primitives to fill out the column
    // center the whole model around (0, 0, 0) => translate all primitives by (-image_x/2, 0, -image_y/2)
    int max_e = 0;
    for (int y = 0, out = 0; y < terrain.height; y++) {
        for (int x = 0; x < terrain.width; x++) {
            const unsigned int e = terrain.elevation(x, y);
            const unsigned char minN = terrain.minNeighbors(x, y);
            max_e = max(max_e, e);
            for (int z = minN; z <= e; z++)
                sc.spheres[out++] = sphere(vec3(x + 0.5f - imageX / 2, z + 0.5f, y + 0.5f - imageY / 2), y * terrain.width + x);
        }
    }

    stbi_image_free(data);

    sc.bvh = build_bvh(sc.spheres, size, numPrimitivesPerLeaf, sc.bvh_size);

    if (addMarker)
        sc.spheres[size] = sphere(vec3(INFINITY, INFINITY, INFINITY), 0);
}

void load_elevation(const char* input, scene& sc, int numPrimitivesPerLeaf) {
    // load heightmap image
    int image_x, image_y, image_n;
    int image_desired_channels = 1; // grayscale
    unsigned char* data = stbi_load(input, &image_x, &image_y, &image_n, image_desired_channels);
    const elevationData eData(data, image_x, image_y);

    // count total number of voxels to allocate the correct number of primitives
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
                sc.spheres[out++] = sphere(vec3(x + 0.5f - image_x / 2, z + 0.5f, y + 0.5f - image_y / 2), y * eData.width + x);
        }
    }

    // normalize color idx such that max_gen = 256
    //float normalizer = 255.0f / max_e;
    //for (int i = 0; i < size; i++) {
    //    int gen = sc.spheres[i].color;
    //    sc.spheres[i].color = (int)(gen * normalizer);
    //}

    stbi_image_free(data);

    sc.bvh = build_bvh(sc.spheres, size, numPrimitivesPerLeaf, sc.bvh_size);

    if (addMarker)
        sc.spheres[size] = sphere(vec3(INFINITY, INFINITY, INFINITY), 0);
}