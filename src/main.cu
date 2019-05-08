#include <cstdio>
#include <utility>

#include "bvh.hu"
#include "cuda_helpers.hu"
#include "device_vector.hu"
#include "film.hu"
#include "gui.hu"
#include "material.hu"
#include "renderer.hu"
#include "sphere.hu"
#include "timer.hpp"
#include "window.hpp"

namespace {
    // Init on gpu to use abstract base class
    __global__ void init_scene(DeviceVector<Material*>** materials, Intersectable** scene)
    {
        if ((blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
           (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
            *materials = new DeviceVector<Material*>;
            DeviceVector<Intersectable*> intersectables;
            (*materials)->push_back(new Lambertian{Vec3{0.5f, 0.5f, 0.5f}});
            intersectables.push_back(new Sphere{
                Vec3{0.f, -1000.f, 0.f},
                1000.f,
                (*materials)->back()
            });

            curandState randState;
            curand_init(1337, 0, 0, &randState);
            for (int a = -11; a < 11; ++a) {
                for (int b = -11; b < 11; ++b) {
                    const Vec3 center0{
                        a + 0.9f * curand_uniform(&randState),
                        0.2f,
                        b + 0.9f * curand_uniform(&randState)
                    };
                    Vec3 center1 = center0;
                    if (len(center0 - Vec3{4.f, 0.2f, 0.f}) > 0.9f) {
                        const float chooseMat = curand_uniform(&randState);
                        if (chooseMat < 0.8f) {
                            center1 += Vec3{0.f, 0.5f * curand_uniform(&randState), 0.f};
                            (*materials)->push_back(new Lambertian{
                                Vec3{
                                    curand_uniform(&randState) * curand_uniform(&randState),
                                    curand_uniform(&randState) * curand_uniform(&randState),
                                    curand_uniform(&randState) * curand_uniform(&randState)
                                }
                            });
                        } else if (chooseMat < 0.95f) {
                            (*materials)->push_back(new Metal{
                                0.5f * (1.f -
                                        Vec3{
                                            curand_uniform(&randState),
                                            curand_uniform(&randState),
                                            curand_uniform(&randState)
                                        }),
                                0.5f * curand_uniform(&randState)
                            });
                        } else
                            (*materials)->push_back(new Dielectric{1.5f});

                        intersectables.push_back(new Sphere{
                            center0,
                            center1,
                            0.f,
                            1.f,
                            0.2f,
                            (*materials)->back()
                        });
                    }
                }
            }

            (*materials)->push_back(new Dielectric{1.5f});
            intersectables.push_back(new Sphere{
                Vec3{0.f, 1.f, 0.f},
                1.f,
                (*materials)->back()
            });
            (*materials)->push_back(new Lambertian{Vec3{0.4f, 0.2f, 0.1f}});
            intersectables.push_back(new Sphere{
                Vec3{-4.f, 1.f, 0.f},
                1.f,
                (*materials)->back()
            });
            (*materials)->push_back(new Metal{Vec3{0.7f, 0.6f, 0.5f}, 0.f});
            intersectables.push_back(new Sphere{
                Vec3{4.f, 1.f, 0.f},
                1.f,
                (*materials)->back()
            });

            *scene = new BVH(std::move(intersectables), 0.f, 1.f);
        }
    }

    __global__ void free_scene(DeviceVector<Material*>** materials, Intersectable** scene)
    {
        if ((blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&
           (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
            for (int i = 0; i < (**materials).size(); ++i)
                delete (**materials)[i];
            delete *scene;
        }
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
    printf("Building scene..."); fflush(stdout);
    DeviceVector<Material*>** materials;
    Intersectable** scene;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&materials), sizeof(DeviceVector<Material*>*)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&scene), sizeof(Intersectable*)));
    // Scene needs to be initialized on GPU since it uses abstract classes
    init_scene<<<1, 1>>>(materials, scene);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf(" took %.3fs!\n", timer.seconds());


    // Run the main loop
    while (window.open()) {
        window.startFrame();
        gui.startFrame();

        // Prepare GL
        glClear(GL_COLOR_BUFFER_BIT);

        if (window.shouldRender() || gui.shouldRender()) {
            film.updateSettings(gui.filmSettings());

            printf("Initiating render..."); fflush(stdout);
            timer.reset();
            render(gui.cameraSettings(), &film, scene);
            printf(" took %.3fs!\n", timer.seconds());
        }

        film.display(window.width(), window.height());
        gui.endFrame();
        window.endFrame();
    }

    checkCudaErrors(cudaDeviceSynchronize());
    // Scene needs to be freed up on GPU as it was initialized there
    free_scene<<<1, 1>>>(materials, scene);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(materials));
    checkCudaErrors(cudaFree(scene));

    film.destroy();
    gui.destroy();
    window.destroy();
    return 0;
}
