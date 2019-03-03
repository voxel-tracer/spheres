#ifndef SPHEREH
#define SPHEREH

#include "ray.h"

struct hit_record
{
    float t;
    vec3 p;
    int hit_idx;
};

struct sphere  {
        sphere() {}
        sphere(vec3 cen) : center(cen) {};

        __device__ vec3 normal(const vec3& p) const { return p - center; }

        vec3 center;
};

__device__ bool hit_sphere(const sphere& s, const ray& r, float t_min, float t_max, hit_record& rec) {
    const vec3 center = s.center;
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - 1;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            return true;
        }
    }
    return false;
}

__device__ bool hit_spheres(const sphere* spheres, const int numSpheres, const ray& r, float t_min, float t_max, hit_record& rec) {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < numSpheres; i++) {
        if (hit_sphere(spheres[i], r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
            rec.hit_idx = i;
        }
    }
    return hit_anything;
}

#endif
