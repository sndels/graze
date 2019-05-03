#include "sphere.hu"

__device__ Sphere::Sphere(const Vec3& center, const float radius) :
    _center(center),
    _radius(radius)
{ }

__device__ bool Sphere::intersect(Ray* r, Hit* hit) const
{
    const Vec3 oc = r->o - _center;
    const float a = lenSq(r->d);
    const float b = 2.f * dot(oc, r->d);
    const float c = lenSq(oc) - _radius * _radius;
    const float d = b * b - 4.f * a * c;
    if (d > 0.f) {
        float t0 = (-b - sqrt(d)) / (2.f * a);
        float t1 = (-b + sqrt(d)) / (2.f * a);
        if (t0 > t1) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        if (t0 > r->t_max || t1 <= r->t_min)
            return false;
        float t = t0;
        if (t <= r->t_min) {
            t = t1;
            if (t > r->t_max)
                return false;
        }
        r->t_max = t;
        hit->t = t;
        hit->p = r->point(t);
        hit->n = (hit->p - _center) / _radius;
        return true;
    }
    return false;
}
