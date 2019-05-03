#include "ray.hu"

__device__ Ray::Ray(const Vec3& o, const Vec3& d) :
    o(o),
    d(d)
{ }

Vec3 __device__ Ray::point(const float t) const
{
    return o + t * d;
}
