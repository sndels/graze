#include "renderer.hu"

#include <cfloat>

#include "cuda_helpers.hu"
#include "ray.hu"

namespace {
    struct Camera {
        Vec3 lower_left;
        Vec3 horizontal;
        Vec3 vertical;
    };

    __device__ Vec3 trace(Ray r, Intersectable** scene)
    {
        Hit hit;
        if ((*scene)->intersect(&r, &hit))
            return 0.5f * (hit.n + 1.f);

        float t = 0.5f * (r.d.y + 1.0f);
        return (1.f - t) * Vec3{1.f} + t * Vec3{0.5f, 0.7f, 1.f};
    }

    __global__ void cuRender(Film::Surface surface, Camera cam, Intersectable** scene)
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
            normalize(cam.lower_left + u * cam.horizontal + v * cam.vertical),
            0.f,
            FLT_MAX
        };
        surface.fb[pxI] = trace(r, scene);
    }
}

void render(Film* film, Intersectable** scene)
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
    cuRender<<<blocks, threads>>>(surface, cam, scene);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    film->setDirty();
}
