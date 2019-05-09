#include "texture.hu"

#include "perlin.hu"

__device__ ConstantTexture::ConstantTexture(const Vec3& color) :
    _color{color}
{ }

__device__ Vec3 ConstantTexture::value(const float u, const float v, const Vec3& p) const
{
    return _color;
}

__device__ CheckerTexture::CheckerTexture(Texture* odd, Texture* even) :
    _odd{odd},
    _even{even}
{ }

__device__ CheckerTexture::~CheckerTexture()
{
    delete _odd;
    delete _even;
}

__device__ Vec3 CheckerTexture::value(const float u, const float v, const Vec3& p) const
{
    float sines = sin(10.f * p.x) * sin(10.f * p.y) * sin(10.f * p.z);
    if (sines < 0.f)
        return _odd->value(u, v, p);
    else
        return _even->value(u, v, p);
}

__device__ NoiseTexture::NoiseTexture(const float scale) :
    _scale{scale}
{ }

__device__ Vec3 NoiseTexture::value(const float u, const float v, const Vec3& p) const
{
    return 0.5f * (1.f + sin(_scale * p.x + 10.f * Perlin::turbulence(_scale * p, 7)));
}
