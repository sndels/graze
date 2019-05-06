#include "sphere.hu"

__device__ Sphere::Sphere(const Vec3& center, const float radius, Material* material) :
    _center0{center},
    _center1{center},
    _time0{0.f},
    _time1{1.f},
    _radius{radius},
    _material{material}
{ }

__device__ Sphere::Sphere(const Vec3& center0, const Vec3& center1, const float time0, const float time1, const float radius, Material* material) :
    _center0{center0},
    _center1{center1},
    _time0{time0},
    _time1{time1},
    _radius{radius},
    _material{material}
{ }

__device__ bool Sphere::intersect(Ray* r, Hit* hit) const
{
    const Vec3 center = lerp(_center0, _center1, (r->time - _time0) / (_time1 - _time0));
    const Vec3 oc = r->o - center;
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
        hit->n = (hit->p - center) / _radius;
        hit->material = _material;
        return true;
    }
    return false;
}
