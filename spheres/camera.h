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

    void init(vec3 lookfrom, vec3 _vup) {
        vUp = _vup;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vUp, w));
        v = cross(w, u);

        origin = lookat + w * radial_distance * relative_dist;
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2 * half_width * focus_dist * u;
        vertical = 2 * half_height * focus_dist * v;
    }

    void update() {
        // compute delta angles in radians
        const float xAngleRad = xDelta * kPI / 180;
        const float yAngleRad = yDelta * kPI / 180;

        //xAngleRad rotates around vUp
        const mat4_t mx = m4_rotation(xAngleRad, v3(vUp[0], vUp[1], vUp[2]));
        // yAngleRad rotates around u
        // u is a unit vector that points to the side of the camera
        const mat4_t my = m4_rotation(yAngleRad, v3(u[0], u[1], u[2]));
        // full transformation
        const mat4_t m = m4_mul(my, mx);

        // compute new camera's lookFrom, vUp
        vec3_t lookFrom = m4_mul_pos(m, v3(origin[0], origin[1], origin[2]));
        vec3_t _vUp = m4_mul_dir(m, v3(vUp[0], vUp[1], vUp[2]));
        init(vec3(lookFrom.x, lookFrom.y, lookFrom.z), vec3(_vUp.x, _vUp.y, _vUp.z));

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
    vec3 vUp;

};

#endif
