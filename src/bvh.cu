#include "bvh.hu"

#include <utility>

namespace {
    const int leafSize = 50;

    template<typename T, typename Predicate>
    __device__ void sort(T* start, T* end, Predicate p)
    {
        // Insertion sort
        for (auto i = start + 1; i < end; ++i) {
            for (auto j = i; j > start && p(*j, *(j - 1)); --j) {
                T tmp = *j;
                *j = *(j - 1);
                *(j - 1) = tmp;
            }
        }
    }

    __device__ Intersectable** splitMedian(Intersectable** start, Intersectable** end, const float t0, const float t1)
    {
        // Find out bounds of the centroids of contained bounds
        AABB centroid{Vec3{FLT_MAX}, Vec3{-FLT_MAX}};
        for (auto i = start; i < end; ++i) {
            const Vec3 center = (*i)->aabb(t0, t1).center;
            centroid.merge(AABB{center, center});
        }

        // Splitting won't work if centroids overlap
        auto mid = end;
        if (centroid.v0 != centroid.v1) {
            Vec3 dim = centroid.v1 - centroid.v0;
            mid = start + (end - start) / 2;
            if (true ||(dim.x > dim.y && dim.x > dim.z)) {
                sort(start, end, [&](const Intersectable* a, const Intersectable* b){
                    return a->aabb(t0, t1).center.x < b->aabb(t0, t1).center.x;
                });
            } else if (dim.y > dim.x && dim.y > dim.z) {
                sort(start, end, [&](const Intersectable* a, const Intersectable* b){
                    return a->aabb(t0, t1).center.y < b->aabb(t0, t1).center.y;
                });
            } else {
                sort(start, end, [&](const Intersectable* a, const Intersectable* b){
                    return a->aabb(t0, t1).center.z < b->aabb(t0, t1).center.z;
                });
            }
        }
        return mid;
    }
}

__device__ BVHNode::BVHNode(Intersectable** start, Intersectable** end) :
    aabb{AABB{Vec3{FLT_MAX}, Vec3{-FLT_MAX}}},
    left{nullptr},
    right{nullptr},
    start{start},
    end{end}
{ }

__device__ bool BVHNode::intersect(Ray* r, Hit* hit) const
{
    bool hitSomething = false;
    if (aabb.intersect(r)) {
        if (left) {
            hitSomething = left->intersect(r, hit);
            if (right)
                hitSomething |= right->intersect(r, hit);
            return hitSomething;
        } else {
            for (auto i = start; i < end; ++i)
                hitSomething |= (*i)->intersect(r, hit);
        }
    }
    return hitSomething;
}

__device__ BVH::BVH(DeviceVector<Intersectable*>&& intersectables, const float t0, const float t1) :
    _intersectables{std::move(intersectables)}
{
    _root = new BVHNode{
        &_intersectables[0],
        &_intersectables.back() + 1
    };
    DeviceVector<BVHNode*> stack;
    stack.push_back(_root);
    while (!stack.empty()) {
        auto node = stack.back();
        // No need to keep node in stack as it will be populated here
        stack.pop_back();

        for (auto i = node->start; i < node->end; ++i)
            node->aabb.merge((*i)->aabb(t0, t1));

        // Split children as necessary (and viable)
        if (node->end - node->start > leafSize) {
            auto mid = splitMedian(node->start, node->end, t0, t1);
            if (mid != node->end) {
                node->left = new BVHNode{node->start, mid};
                node->right = new BVHNode{mid, node->end};
                stack.push_back(node->left);
                stack.push_back(node->right);
            }
        }
    }
}

__device__ BVH::~BVH()
{
    DeviceVector<BVHNode*> stack;
    stack.push_back(_root);
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        stack.push_back(node->left);
        stack.push_back(node->right);
        delete node;
    }
}

__device__ bool BVH::intersect(Ray* r, Hit* hit) const
{
    return _root->intersect(r, hit);
}

__device__ AABB BVH::aabb(const float, const float) const
{
    return _root->aabb;
}
