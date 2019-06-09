#ifndef SPHEREH
#define SPHEREH

#include "ray.h"

struct hit_record
{
    float t;
    vec3 n;
    int idx;
};

struct sphere  {
        sphere() {}
        sphere(vec3 cen, int clr) : center(cen), color(clr) {};

        __device__ vec3 normal(const vec3& p) const { return p - center; }

        vec3 center;
        int color;
};

__device__ bool hit_point(const vec3& center, const ray& r, float t_min, float t_max, hit_record& rec) {
    vec3 oc = r.origin() - center;
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - 1;
    float discriminant = b*b - c;
    if (discriminant > 0) {
        const float dsqrt = sqrt(discriminant);
        float temp = -b - dsqrt;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            vec3 p = r.point_at_parameter(temp);
            rec.n = p - center;
            return true;
        }
        temp = -b + dsqrt;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            vec3 p = r.point_at_parameter(temp);
            rec.n = p - center;
            return true;
        }
    }
    return false;
}

__device__ bool hit_light(const vec3& center, const float radius, const ray& r, float t_min, float t_max) {
    vec3 oc = r.origin() - center;
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - c;
    if (discriminant > 0) {
        const float dsqrt = sqrt(discriminant);
        float temp = -b - dsqrt;
        if (temp < t_max && temp > t_min) {
            return true;
        }
        temp = -b + dsqrt;
        if (temp < t_max && temp > t_min) {
            return true;
        }
    }
    return false;
}

#endif
