#ifndef HITABLELISTH
#define HITABLELISTH

#include "sphere.h"

struct hitable_list {
        __device__ hitable_list() {}
        __device__ hitable_list(sphere *s, material *m, int n) { spheres = s; materials = m;  num_spheres = n; }

        sphere *spheres;
        material *materials;
        int num_spheres;
};

__device__ bool hit_hitable_list(const hitable_list* list, const ray& r, float t_min, float t_max, hit_record& rec) {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;
        for (int i = 0; i < list->num_spheres; i++) {
            if (hit_sphere(&(list->spheres[i]), r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
                rec.hit_idx = i;
            }
        }
        return hit_anything;
}

#endif
