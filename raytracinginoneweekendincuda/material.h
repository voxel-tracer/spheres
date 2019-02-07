#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "ray.h"
#include "sphere.h"


__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(rand_state *local_rand_state) {
    float z = curand_uniform(local_rand_state) * 2.0f - 1.0f;
    float t = curand_uniform(local_rand_state) * 2.0f * kPI;
    float r = sqrtf(fmaxf(0.0, 1.0f - z * z));
    float x = r * cosf(t);
    float y = r * sinf(t);
    vec3 res = vec3(x, y, z);
    res *= cbrtf(curand_uniform(local_rand_state));
    return res;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
     return v - 2.0f*dot(v,n)*n;
}

struct material {
    enum Type { Lambertian, Metal, Dielectric };
    Type type;
    vec3 albedo;
    float fuzz;
    float ref_idx;

    material() {}
    material(Type _type, vec3 _albedo, float _fuzz, float _ref_idx) :type(_type), albedo(_albedo), fuzz(_fuzz), ref_idx(_ref_idx) {}
};

__device__ bool scatter(const sphere& s, const material& mat, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state *local_rand_state) {
    switch (mat.type) {
    case material::Lambertian:
    {
        const vec3 normal = s.normal(rec.p);
        vec3 target = rec.p + normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.p, target - rec.p);
        attenuation = mat.albedo;
        return true;
    }

    case material::Metal:
    {
        const vec3 normal = s.normal(rec.p);
        vec3 reflected = reflect(unit_vector(r_in.direction()), normal);
        scattered = ray(rec.p, reflected + mat.fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = mat.albedo;
        return (dot(scattered.direction(), normal) > 0.0f);
    }

    case material::Dielectric:
    {
        const vec3 normal = s.normal(rec.p);
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), normal);
        float ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), normal) > 0.0f) {
            outward_normal = -normal;
            ni_over_nt = mat.ref_idx;
            cosine = dot(r_in.direction(), normal) / r_in.direction().length();
            cosine = sqrt(1.0f - mat.ref_idx * mat.ref_idx*(1 - cosine * cosine));
        }
        else {
            outward_normal = normal;
            ni_over_nt = 1.0f / mat.ref_idx;
            cosine = -dot(r_in.direction(), normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, mat.ref_idx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    default:
        return false;
    }
}

#endif
