#include "material.hu"

#include <cfloat>

__device__ Lambertian::Lambertian(const Vec3& albedo) :
    albedo(albedo)
{ }

__device__ bool Lambertian::scatter(const Ray& r, const Hit& hit, Vec3* attenuation, Ray* scattered, curandStatePhilox4_32_10_t* randState) const
{
    *scattered = Ray{
        hit.p,
        normalize(hit.n + randomDir(randState)),
        0.001f,
        FLT_MAX
    };
    *attenuation = albedo;
    return true;
}

__device__ Metal::Metal(const Vec3& albedo, const float roughness) :
    albedo(albedo),
    roughness(min(roughness, 1.f))
{ }

__device__ bool Metal::scatter(const Ray& r, const Hit& hit, Vec3* attenuation, Ray* scattered, curandStatePhilox4_32_10_t* randState) const
{
    *scattered = Ray{
        hit.p,
        normalize(reflect(r.d, hit.n) + roughness * randomDir(randState)),
        0.001f,
        FLT_MAX
    };
    *attenuation = albedo;
    return dot(scattered->d, hit.n) > 0;
}
