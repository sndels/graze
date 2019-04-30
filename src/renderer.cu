#include "renderer.hu"

#include "cuda_helpers.hu"

namespace {
    __global__ void cuRender(Film::Surface surface)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= surface.width || y >= surface.height)
            return;
        const int pxI = (y * surface.width + x) * 3;

        surface.fb[pxI] = float(x) / surface.width;
        surface.fb[pxI + 1] = float(y) / surface.height;
        surface.fb[pxI + 2] = 0.2;
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
