#include "intersectable.hu"

#include <cfloat>

__device__ IntersectableList::~IntersectableList()
{
    for (int i = 0; i < _intersectables.size(); ++i)
        delete _intersectables[i];
}

__device__ void IntersectableList::add(Intersectable* i)
{
    _intersectables.push_back(i);
}

__device__ bool IntersectableList::intersect(Ray* r, Hit* hit) const
{
    bool hitSomething = false;
    for (int i = 0; i < _intersectables.size(); ++i)
        hitSomething |= _intersectables[i]->intersect(r, hit);

    return hitSomething;
}

__device__ AABB IntersectableList::aabb(const float t0, const float t1) const
{
    AABB aabb{Vec3{FLT_MAX}, Vec3{-FLT_MAX}};
    for (int i = 0; i < _intersectables.size(); ++i)
        aabb.merge(_intersectables[i]->aabb(t0, t1));
    return aabb;
}
