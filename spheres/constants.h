#pragma once

#define kPI 3.1415926f

const unsigned int lane_size_float = 64 / sizeof(float);
const unsigned int lane_size_spheres = lane_size_float / 3;
const unsigned int lane_padding_float = lane_size_float - lane_size_spheres * 3;
