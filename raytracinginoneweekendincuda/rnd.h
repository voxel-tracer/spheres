#pragma once

#include "vec3.h"
#include "math_functions.h"

typedef unsigned int rand_state;

#define kPI 3.1415926f

__device__ unsigned int xor_shift_32(rand_state& state)
{
    unsigned int x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 15;
    state = x;
    return x;
}

__device__ float random_float(rand_state& state)
{
    return (xor_shift_32(state) & 0xFFFFFF) / 16777216.0f;
}

__device__ vec3 random_in_unit_disk(rand_state& state)
{
    const float a = random_float(state) * 2.0f * kPI;
    const float u = sqrtf(random_float(state));
    float c, s;
    sincosf(a, &s, &c);
    return vec3(u*c, u*s, 0);
}

__device__ vec3 random_in_unit_sphere(rand_state& state) {
    float z = random_float(state) * 2.0f - 1.0f;
    float t = random_float(state) * 2.0f * kPI;
    float r = sqrtf(fmaxf(0.0, 1.0f - z * z));
    float c, s;
    sincosf(t, &s, &c);
    vec3 res = vec3(r*c, r*s, z);
    res *= cbrtf(random_float(state));
    return res;
}

/*
* based off http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
*/
__device__ rand_state wang_hash(rand_state seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}
