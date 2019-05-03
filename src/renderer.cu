#include "renderer.hu"

#include "cuda_helpers.hu"
#include "ray.hu"

namespace {
    __device__ Vec3 trace(const Ray& r)
    {
        float t = 0.5f * (r.d.y + 1.0f);
        return (1.f - t) * Vec3{1.f} + t * Vec3{0.5f, 0.7f, 1.f};
    }

    __global__ void cuRender(Film::Surface surface)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= surface.width || y >= surface.height)
            return;
        const int pxI = y * surface.width + x;
        const float u = float(x) / surface.width;
        const float v = float(y) / surface.height;

        Vec3 lower_left{-2.f, -1.f, -1.f};
        Vec3 horizontal{4.f, 0.f, 0.f};
        Vec3 vertical{0.f, 2.f, 0.f};
        Ray r{
            {0.f, 0.f, 0.f},
            normalize(lower_left + u * horizontal + v * vertical)
        };
        surface.fb[pxI] = trace(r);
    }
}

void render(Film* film)
{
    const auto& surface = film->surface();
    const uint32_t tx = 8;
    const uint32_t ty = 8;
    const dim3 blocks{
        surface.width / tx + 1,
        surface.height / ty + 1
    };
    const dim3 threads{tx, ty};
    // This passes a copy of surface but the contained pointer won't be modified, only data
    cuRender<<<blocks, threads>>>(surface);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    film->setDirty();
}
