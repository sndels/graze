#include "intersectable.hu"

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
