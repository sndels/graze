#include "material.hu"

#include <cfloat>

namespace {
    __device__ float schlick(float cosine, float refIdx)
    {
        float r0 = (1.f - refIdx) / (1.f + refIdx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow(1.f - cosine, 5.f);
    }
}

__device__ Lambertian::Lambertian(const Vec3& albedo) :
    _albedo(albedo)
{ }

__device__ bool Lambertian::scatter(const Ray& r, const Hit& hit, Vec3* attenuation, Ray* scattered, curandStatePhilox4_32_10_t* randState) const
{
    *scattered = Ray{
        hit.p,
        normalize(hit.n + randomDir(randState)),
        0.001f,
        FLT_MAX
    };
    *attenuation = _albedo;
    return true;
}

__device__ Metal::Metal(const Vec3& albedo, const float roughness) :
    _albedo(albedo),
    _roughness(min(roughness, 1.f))
{ }

__device__ bool Metal::scatter(const Ray& r, const Hit& hit, Vec3* attenuation, Ray* scattered, curandStatePhilox4_32_10_t* randState) const
{
    *scattered = Ray{
        hit.p,
        normalize(reflect(r.d, hit.n) + _roughness * randomDir(randState)),
        0.001f,
        FLT_MAX
    };
    *attenuation = _albedo;
    return dot(scattered->d, hit.n) > 0;
}

__device__ Dielectric::Dielectric(const float refIdx) :
    _refIdx(refIdx)
{ }

__device__ bool Dielectric::scatter(const Ray& r, const Hit& hit, Vec3* attenuation, Ray* scattered, curandStatePhilox4_32_10_t* randState) const
{
    const Vec3 reflected = reflect(r.d, hit.n);
    *attenuation = Vec3{1.f};

    Vec3 outN;
    float cosine;
    float niPerNt;
    if (dot(r.d, hit.n) > 0) {
        outN = -hit.n;
        niPerNt = _refIdx;
        cosine = _refIdx * dot(r.d, hit.n);
    } else {
        outN = hit.n;
        niPerNt = 1.f / _refIdx;
        cosine = -dot(r.d, hit.n);
    }

    Vec3 refracted;
    float pReflect;
    if (refract(r.d, outN, niPerNt, &refracted))
        pReflect = schlick(cosine, _refIdx);
    else 
        pReflect = 1.f;

    *scattered = Ray{
        hit.p,
        curand_uniform(randState) < pReflect ? reflected : refracted,
        0.001f,
        FLT_MAX
    };
    return true;
}
