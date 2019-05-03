#include "renderer.hu"

#include <cfloat>

#include "cuda_helpers.hu"
#include "ray.hu"
#include "material.hu"

namespace {
    struct Camera {
        Vec3 lower_left;
        Vec3 horizontal;
        Vec3 vertical;
    };

    __device__ Vec3 trace(Ray r, Intersectable** scene, curandStatePhilox4_32_10_t* randState)
    {
        Hit hit;
        Vec3 atten{1.f};
        for (int bounce = 0; bounce < 50; ++bounce) {
            if ((*scene)->intersect(&r, &hit)) {
                Ray scattered;
                Vec3 attenuation;
                if (hit.material->scatter(r, hit, &attenuation, &scattered, randState)) {
                    atten *= attenuation;
                    r = scattered;
                } else {
                    atten = Vec3{0.f};
                    break;
                }
            } else
                break;
        }

        float t = 0.5f * (r.d.y + 1.0f);
        Vec3 color = (1.f - t) * Vec3{1.f} + t * Vec3{0.5f, 0.7f, 1.f};
        return atten * color;
    }

    __global__ void cuRender(Film::Surface surface, Camera cam, Intersectable** scene)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= surface.width || y >= surface.height)
            return;
        const int pxI = y * surface.width + x;

        // Use Philox because XORWOW init is slow and threw error 700 with large sequences
        // Suggested in https://devtalk.nvidia.com/default/topic/1028057/curand_init-sequence-number-problem/
        curandStatePhilox4_32_10_t localRand;
        curand_init(1337, pxI, 0, &localRand);

        Vec3 color;
        for (int s = 0; s < surface.samples; ++s) {
            const float u = float(x + curand_uniform(&localRand)) / surface.width;
            const float v = float(y + curand_uniform(&localRand)) / surface.height;

            Ray r{
                {0.f, 0.f, 0.f},
                normalize(cam.lower_left + u * cam.horizontal + v * cam.vertical),
                0.f,
                FLT_MAX
            };
            color += trace(r, scene, &localRand);
        }
        surface.fb[pxI] = pow(color / surface.samples, 1.f / 2.2f);
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
