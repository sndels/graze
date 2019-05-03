#include "ray.hu"

__device__ Ray::Ray() :
    o(0.f),
    d(0.f),
    t_min(0.f),
    t_max(0.f)
{ }

__device__ Ray::Ray(const Vec3& o, const Vec3& d, const float t_min, const float t_max) :
    o(o),
    d(d),
    t_min(t_min),
    t_max(t_max)
{ }

Vec3 __device__ Ray::point(const float t) const
{
    return o + t * d;
}
