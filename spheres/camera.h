#ifndef CAMERAH
#define CAMERAH

#include <host_defines.h>
#include "ray.h"
#include "rnd.h"
#include "constants.h"
#include "math_3d.h"

class camera {
public:
    camera(vec3 lookfrom, vec3 _lookat, vec3 _vup, float vfov, float aspect, float aperture, float _focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2;
        float theta = vfov * kPI / 180;
        half_height = tan(theta / 2);
        half_width = aspect * half_height;
        lookat = _lookat;
        focus_dist = _focus_dist;
        radial_distance = (lookfrom - lookat).length();
        init(lookfrom, _vup);
    }

    void init(vec3 lookfrom, vec3 vup) {
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2 * half_width * focus_dist * u;
        vertical = 2 * half_height * focus_dist * v;
    }

    void update() {
        // compute delta angles in radians
        const float xAngleRad = xDelta * kPI / 180;
        const float yAngleRad = yDelta * kPI / 180;
        // use current viewMat to transform rotation axis
        const vec3_t xAxis = m4_mul_dir(viewMat, v3(1, 0, 0));
        const vec3_t yAxis = m4_mul_dir(viewMat, v3(0, 1, 0));
        // compute delta transformation matrix
        const mat4_t mx = m4_rotation(xAngleRad, xAxis);
        const mat4_t my = m4_rotation(yAngleRad, yAxis);
        const mat4_t m = m4_mul(my, mx);
        // update viewMat
        viewMat = m4_mul(m, viewMat);
        // compute new camera's lookFrom, vUp
        vec3_t lookFrom = m4_mul_pos(viewMat, v3(0, 0, relative_dist * radial_distance));
        vec3_t vUp = m4_mul_dir(viewMat, v3(0, 1, 0));
        init(vec3(lookFrom.x, lookFrom.y, lookFrom.z), vec3(vUp.x, vUp.y, vUp.z));

        xDelta = yDelta = 0;
    }

    __device__ ray get_ray(float s, float t, rand_state& local_rand_state) const {
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
    }

    vec3 lookat;
    vec3 origin;
    float half_width;
    float half_height;
    float focus_dist;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;

    float relative_dist = 1.0f;
    int xDelta = 0;
    int yDelta = 0;
private:
    float radial_distance;
    mat4_t viewMat = m4_identity();
};

#endif
