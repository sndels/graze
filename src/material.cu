#include "material.hu"

#include <cfloat>

namespace {
    // NoV is the angle between the direction the ray is coming from and the interface
    // normal towards the source medium
    // ni, nt are the incoming and outgoing indices of refraction, respectively
    __device__ float schlickApprox(const float NoV, const float ni, const float nt)
    {
        float r0 = (ni - nt) / (ni + nt);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow(1.f - NoV, 5.f);
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
    // TODO: Less hacky rough reflections
    *scattered = Ray{
        hit.p,
        normalize(reflect(r.d, hit.n) + _roughness * randomDir(randState)),
        0.001f,
        FLT_MAX
    };
    *attenuation = _albedo;
    // Don't scatter on rays falling under "horizon"
    return dot(scattered->d, hit.n) > 0;
}

__device__ Dielectric::Dielectric(const float refIdx) :
    _refIdx(refIdx)
{ }

__device__ bool Dielectric::scatter(const Ray& r, const Hit& hit, Vec3* attenuation, Ray* scattered, curandStatePhilox4_32_10_t* randState) const
{
    // Glass so no absorption, always scatter
    *attenuation = Vec3{1.f};

    Vec3 rn; // Surface normal towards source medium
    float ni;
    float nt;
    if (dot(hit.n, -r.d) < 0.f) {
        // From object
        rn = -hit.n;
        ni = _refIdx;
        nt = 1.f; // Only dielectric/air -refraction is supported
    } else {
        // To object
        rn = hit.n;
        ni = 1.f; // Only air/dielectric-refraction is supported
        nt = _refIdx;
    }

    const Vec3 reflected = reflect(r.d, rn);

    // Get refraction and Fresnel factor
    Vec3 refracted;
    float r0;
    if (refract(r.d, rn, ni, nt, &refracted))
        r0 = schlickApprox(dot(rn, -r.d), ni, nt);
    else 
        r0 = 1.f;

    // Distribute reflected and refracted rays according to Fresnel factor
    *scattered = Ray{
        hit.p,
        curand_uniform(randState) < r0 ? reflected : refracted,
        0.001f,
        FLT_MAX
    };
    return true;
}
