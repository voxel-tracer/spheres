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

struct scene {
    sphere* spheres;
    int count;

    vec3 min;
    vec3 max;
};

__device__ bool hit_bbox(const vec3& min, const vec3& max, const ray& r, float t_min, float t_max, hit_record& rec) {
    for (int a = 0; a < 3; a++) {
        float invD = 1.0f / r.direction()[a];
        float t0 = (min[a] - r.origin()[a])*invD;
        float t1 = (max[a] - r.origin()[a])*invD;
        if (invD < 0.0f) {
            float tmp = t0; t0 = t1; t1 = tmp;
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min)
            return false;
    }

    rec.t = t_min;
    rec.p = r.point_at_parameter(t_min);
    return true;
}

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

__device__ bool hit_spheres(const scene& sc, const ray& r, float t_min, float t_max, hit_record& rec) {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    //if (!hit_bbox(sc.min, sc.max, r, t_min, t_max, temp_rec))
    //    return false;

    for (int i = 0; i < sc.count; i++) {
        const sphere s = sc.spheres[i];
        if (hit_sphere(s, r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
            rec.hit_idx = i;
        }
    }
    return hit_anything;
}

#endif
