#include "camera.hu"

#include <cfloat>

Camera::Camera(const Vec3& eye, const Vec3& target, const Vec3& up, const float fov, const float ar) :
    _eye(eye)
{
    const float theta = fov * M_PI / 180.f;
    const float halfHeight = tan(theta / 2.f);
    const float halfWidth = ar * halfHeight;
    const Vec3 w = normalize(eye - target);
    const Vec3 u = normalize(cross(up, w));
    const Vec3 v = cross(w, u);
    _lowerLeft = eye - halfWidth * u - halfHeight * v - w;
    _horizontal = 2.f * halfWidth * u;
    _vertical = 2.f * halfHeight * v;
}

__device__ Ray Camera::ray(const float s, const float t) const
{
    return Ray{
        _eye,
        _lowerLeft + s * _horizontal + t * _vertical - _eye,
        0.f,
        FLT_MAX
    };
}
