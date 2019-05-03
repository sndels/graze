#include "renderer.hu"

#include "cuda_helpers.hu"
#include "ray.hu"

namespace {
    struct Camera {
        Vec3 lower_left;
        Vec3 horizontal;
        Vec3 vertical;
    };

    __device__ bool hit_sphere(const Vec3& center, const float radius, const Ray& r)
    {
        Vec3 oc = r.o - center;
        float a = lenSq(r.d);
        float b = 2.f * dot(oc, r.d);
        float c = lenSq(oc) - radius * radius;
        float discriminant = b * b - 4.f * a * c;
        return discriminant > 0.f;
    }

    __device__ Vec3 trace(const Ray& r)
    {
        if (hit_sphere(Vec3{0.f, 0.f, -1.f}, 0.5, r))
            return Vec3{1.f, 0.f, 0.f};

        float t = 0.5f * (r.d.y + 1.0f);
        return (1.f - t) * Vec3{1.f} + t * Vec3{0.5f, 0.7f, 1.f};
    }

    __global__ void cuRender(Film::Surface surface, Camera cam)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= surface.width || y >= surface.height)
            return;
        const int pxI = y * surface.width + x;
        const float u = float(x) / surface.width;
        const float v = float(y) / surface.height;

        Ray r{
            {0.f, 0.f, 0.f},
            normalize(cam.lower_left + u * cam.horizontal + v * cam.vertical)
        };
        surface.fb[pxI] = trace(r);
    }
}

void render(Film* film)
{
    const auto& surface = film->surface();
    Camera cam{
        {-2.f, -1.f, -1.f},
        {4.f, 0.f, 0.f},
        {0.f, 2.f, 0.f}
    };

    const uint32_t tx = 8;
    const uint32_t ty = 8;
    const dim3 blocks{
        surface.width / tx + 1,
        surface.height / ty + 1
    };
    const dim3 threads{tx, ty};
    // This passes a copy of surface but the contained pointer won't be modified, only data
    cuRender<<<blocks, threads>>>(surface, cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    film->setDirty();
}
