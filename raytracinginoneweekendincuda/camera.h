#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "ray.h"

#define kPI 3.1415926f

__device__ vec3 random_in_unit_disk(rand_state *local_rand_state) {
    const float a = curand_uniform(local_rand_state)*2.0f*kPI;
    const float u = sqrtf(curand_uniform(local_rand_state));
    return vec3(u*cosf(a), u*sinf(a), 0);
}

class camera {
public:
    camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        float theta = vfov*kPI/180.0f;
        float half_height = tan(theta/2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
        horizontal = 2.0f*half_width*focus_dist*u;
        vertical = 2.0f*half_height*focus_dist*v;
    }
    __device__ ray get_ray(float s, float t, rand_state *local_rand_state) const {
        vec3 rd = lens_radius*random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};

#endif
