#include "camera.hu"

#include <cfloat>

Camera::Camera(const Vec3& eye, const Vec3& target, const Vec3& up, const float fov, const float ar, const float aperture, const float focusDist) :
    _eye(eye),
    _lensRadius(aperture / 2.f)
{
    const float theta = fov * M_PI / 180.f;
    const float halfHeight = tan(theta / 2.f);
    const float halfWidth = ar * halfHeight;
    _w = normalize(eye - target);
    _u = normalize(cross(up, _w));
    _v = cross(_w, _u);
    _lowerLeft = eye - focusDist * (halfWidth * _u + halfHeight * _v + _w);
    _horizontal = 2.f * halfWidth * focusDist * _u;
    _vertical = 2.f * halfHeight * focusDist * _v;
}

__device__ Ray Camera::ray(const float s, const float t, curandStatePhilox4_32_10_t* randState) const
{
    const Vec3 d = _lensRadius * randomDir(randState);
    const Vec3 offset = _u * d.x + _v * d.y;
    return Ray{
        _eye + offset,
        normalize(_lowerLeft + s * _horizontal + t * _vertical - _eye - offset),
        0.f,
        FLT_MAX
    };
}
