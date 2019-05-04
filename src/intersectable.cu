#include "intersectable.hu"

__device__ IntersectableList::IntersectableList(Intersectable** intersectables, const int numIntersectables) :
    intersectables{intersectables},
    numIntersectables{numIntersectables}
{ }

__device__ bool IntersectableList::intersect(Ray* r, Hit* hit) const
{
    bool hitSomething = false;
    for (int i = 0; i < numIntersectables; ++i)
        hitSomething |= intersectables[i]->intersect(r, hit);

    return hitSomething;
}
