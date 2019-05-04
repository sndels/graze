#include "stdio.h"

#include "cuda_helpers.hu"
#include "film.hu"
#include "gui.hu"
#include "material.hu"
#include "renderer.hu"
#include "sphere.hu"
#include "timer.hpp"
#include "window.hpp"

namespace {
    // This a loose upper bound
    const uint32_t numSpheres = 500;

    // Init on gpu to use abstract base class
    __global__ void init_scene(Material** materials, Intersectable** intersectables, Intersectable** scene)
    {
        materials[0] = new Lambertian{Vec3{0.5f, 0.5f, 0.5f}};
        intersectables[0] = new Sphere{
            Vec3{0.f, -1000.f, 0.f},
            1000.f,
            materials[0]
        };

        int i = 1;
        curandState randState;
        curand_init(1337, 0, 0, &randState);
        for (int a = -11; a < 11; ++a) {
            for (int b = -11; b < 11; ++b) {
                const Vec3 center{
                    a + 0.9f * curand_uniform(&randState),
                    0.2f,
                    b + 0.9f * curand_uniform(&randState)
                };
                if (len(center - Vec3{4.f, 0.2f, 0.f}) > 0.9f) {
                    const float chooseMat = curand_uniform(&randState);
                    if (chooseMat < 0.8f) {
                        materials[i] = new Lambertian{
                            Vec3{
                                curand_uniform(&randState) * curand_uniform(&randState),
                                curand_uniform(&randState) * curand_uniform(&randState),
                                curand_uniform(&randState) * curand_uniform(&randState)
                            }
                        };
                    } else if (chooseMat < 0.95f) {
                        materials[i] = new Metal{
                            0.5f * (1.f - Vec3{
                                curand_uniform(&randState),
                                curand_uniform(&randState),
                                curand_uniform(&randState)
                            }),
                            0.5f * curand_uniform(&randState)
                        };
                    } else
                        materials[i] = new Dielectric{1.5f};

                    intersectables[i] = new Sphere{
                        center,
                        0.2f,
                        materials[i]
                    };
                    ++i;
                }
            }
        }

        materials[i] = new Dielectric{1.5f};
        intersectables[i] = new Sphere{
            Vec3{0.f, 1.f, 0.f},
            1.f,
            materials[i]
        };
        materials[++i] = new Lambertian{Vec3{0.4f, 0.2f, 0.1f}};
        intersectables[i] = new Sphere{
            Vec3{-4.f, 1.f, 0.f},
            1.f,
            materials[i]
        };
        materials[++i] = new Metal{Vec3{0.7f, 0.6f, 0.5f}, 0.f};
        intersectables[i] = new Sphere{
            Vec3{4.f, 1.f, 0.f},
            1.f,
            materials[i]
        };

        *scene = new IntersectableList(intersectables, ++i);
    }

    __global__ void free_scene(Material** materials, Intersectable** intersectables, Intersectable** scene)
    {
        for (int i = 0; i < reinterpret_cast<IntersectableList*>(*scene)->numIntersectables; ++i) {
            delete materials[i];
            delete intersectables[i];
        }
        delete *scene;
    }
}

int main()
{
    Window window{1280, 720};
    if (!window.init())
        return -1;

    GUI gui{window.ptr()};
    Film film{gui.filmSettings()};
    Timer timer;

    timer.reset();
    Material** materials;
    Intersectable** intersectables;
    Intersectable** scene;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&materials), (numSpheres + 1) * sizeof(Material*)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&intersectables), (numSpheres + 1) * sizeof(Intersectable*)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&scene), sizeof(Intersectable*)));
    init_scene<<<1, 1>>>(materials, intersectables, scene);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("Scene built in %.3fs!\n", timer.seconds());


    // Run the main loop
    while (window.open()) {
        window.startFrame();
        gui.startFrame();

        // Prepare GL
        glClear(GL_COLOR_BUFFER_BIT);

        if (window.startRender() || gui.startRender()) {
            film.updateSettings(gui.filmSettings());

            printf("Initiating render!\n");
            timer.reset();
            render(gui.cameraSettings(), &film, scene);
            printf("Done in %.3fs!\n", timer.seconds());
        }

        film.display(window.width(), window.height());
        gui.endFrame();
        window.endFrame();
    }

    checkCudaErrors(cudaDeviceSynchronize());
    free_scene<<<1, 1>>>(materials, intersectables, scene);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(materials));
    checkCudaErrors(cudaFree(intersectables));
    checkCudaErrors(cudaFree(scene));
    film.destroy();
    gui.destroy();
    window.destroy();
    return 0;
}
