#include "camera.hu"

#include <cfloat>

Camera::Camera(const CameraSettings& settings, const float ar) :
    _eye{settings.eye},
    _lensRadius{settings.aperture / 2.f},
    _time0{settings.time0},
    _time1{settings.time1}
{
    const Vec3 up{0.f, 1.f, 0.f};
    // TODO: comment
    const float theta = settings.fov * M_PI / 180.f;
    const float halfHeight = tan(theta / 2.f);
    const float halfWidth = ar * halfHeight;
    _w = normalize(settings.eye - settings.target);
    _u = normalize(cross(up, _w));
    _v = cross(_w, _u);
    _lowerLeft = settings.eye - settings.focalLength * (halfWidth * _u + halfHeight * _v + _w);
    _horizontal = 2.f * halfWidth * settings.focalLength * _u;
    _vertical = 2.f * halfHeight * settings.focalLength * _v;
}

__device__ Ray Camera::ray(const float s, const float t, curandStatePhilox4_32_10_t* randState) const
{
    // TODO: comment
    const Vec3 d = _lensRadius * randomDir(randState);
    const Vec3 offset = _u * d.x + _v * d.y;
    return Ray{
        _eye + offset,
        normalize(_lowerLeft + s * _horizontal + t * _vertical - _eye - offset),
        _time0 + curand_uniform(randState) * (_time1 - _time0),
        0.f,
        FLT_MAX
    };
}
