#include "renderer.hu"

#include <cfloat>

#include "camera.hu"
#include "cuda_helpers.hu"
#include "ray.hu"
#include "material.hu"

namespace {
    __device__ Vec3 trace(const Ray& startRay, const Intersectable* const* scene, curandStatePhilox4_32_10_t* randState)
    {
        Hit hit;
        Vec3 atten{1.f};
        Ray r = startRay;
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

        // Light from sky
        float t = 0.5f * (r.d.y + 1.0f);
        Vec3 color = (1.f - t) * Vec3{1.f} + t * Vec3{0.5f, 0.7f, 1.f};
        return atten * color;
    }

    // Pass surface by value (copy) since only contents of fb are modified
    __global__ void cuRender(const Film::Surface surface, Camera cam, const Intersectable* const* scene)
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
            // Simple jitter on pixel location
            const float u = float(x + curand_uniform(&localRand)) / surface.width;
            const float v = float(y + curand_uniform(&localRand)) / surface.height;

            const Ray r = cam.ray(u, v, &localRand);
            color += trace(r, scene, &localRand);
        }
        surface.fb[pxI] = pow(color / surface.samples, 1.f / 2.2f);
    }
}

void render(const CameraSettings& cameraSettings, Film* film, const Intersectable* const* scene)
{
    const auto& surface = film->surface();
    const Camera cam{
        cameraSettings.eye,
        cameraSettings.target,
        Vec3{0.f, 1.f, 0.f},
        cameraSettings.fov,
        float(surface.width) / surface.height,
        cameraSettings.aperture,
        cameraSettings.focalLength
    };

    const dim3 threads{8, 8};
    const dim3 blocks{
        surface.width / threads.x + 1,
        surface.height / threads.y + 1
    };
    cuRender<<<blocks, threads>>>(surface, cam, scene);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    film->setDirty();
}
