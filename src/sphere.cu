#include "sphere.hu"

__device__ Sphere::Sphere(const Vec3& center, const float radius, Material* material) :
    _center{center},
    _radius{radius},
    _material{material}
{ }

__device__ bool Sphere::intersect(Ray* r, Hit* hit) const
{
    const Vec3 oc = r->o - _center;
    const float a = lenSq(r->d);
    const float b = 2.f * dot(oc, r->d);
    const float c = lenSq(oc) - _radius * _radius;
    const float d = b * b - 4.f * a * c;
    // Check for hit
    if (d > 0.f) {
        // Get the two hit locations, t0 < t1
        float t0 = (-b - sqrt(d)) / (2.f * a);
        float t1 = (-b + sqrt(d)) / (2.f * a);
        if (t0 > t1) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        // Check if sphere is outside ray range
        if (t0 > r->tMax || t1 <= r->tMin)
            return false;

        float t = t0;
        // Check if we are inside
        if (t <= r->tMin) {
            t = t1;
            // Check if outgoing hit is beyond ray range
            if (t > r->tMax)
                return false;
        }
        r->tMax = t;
        hit->t = t;
        hit->p = r->point(t);
        hit->n = (hit->p - _center) / _radius;
        hit->material = _material;
        return true;
    }
    return false;
}
