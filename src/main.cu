#include "stdio.h"

#include "cuda_helpers.hu"
#include "film.hu"
#include "gui.hpp"
#include "material.hu"
#include "renderer.hu"
#include "sphere.hu"
#include "timer.hpp"
#include "window.hpp"

namespace {
    // Init on gpu to use abstract base class
    __global__ void init_scene(Intersectable** intersectables, Intersectable** scene)
    {
        intersectables[0] = new Sphere{
            Vec3{0.f, 0.f, -1.f},
            0.5f,
            new Lambertian{Vec3{0.1f, 0.2f, 0.5f}}
        };
        intersectables[1] = new Sphere{
            Vec3{0.f, -100.5f, -1.f},
            100.f,
            new Lambertian{Vec3{0.8f, 0.8f, 0.f}}
        };
        intersectables[2] = new Sphere{
            Vec3{1.f, 0.f, -1.f},
            0.5f,
            new Metal{Vec3{0.8f, 0.6f, 0.3f}, 0.3f}
        };
        intersectables[3] = new Sphere{
            Vec3{-1.f, 0.f, -1.f},
            0.5f,
            new Dielectric{1.5f}
        };
        intersectables[4] = new Sphere{
            Vec3{-1.f, 0.f, -1.f},
            -0.45f, // Flip normal to make hollow sphere
            new Dielectric{1.5f}
        };
        *scene = new IntersectableList(intersectables, 5);
    }

    __global__ void free_scene(Intersectable** intersectables, Intersectable** scene)
    {
        for (int i = 0; i < 5; ++i) {
            // TODO: Not generic if other types are added
            delete reinterpret_cast<Sphere*>(intersectables[i])->material;
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
    Intersectable** intersectables;
    Intersectable** scene;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&intersectables), 5 * sizeof(Intersectable*)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&scene), sizeof(Intersectable*)));
    init_scene<<<1, 1>>>(intersectables, scene);
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
            render(&film, scene);
            printf("Done in %.3fs!\n", timer.seconds());
        }

        film.display(window.width(), window.height());
        gui.endFrame();
        window.endFrame();
    }

    checkCudaErrors(cudaDeviceSynchronize());
    free_scene<<<1, 1>>>(intersectables, scene);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(intersectables));
    checkCudaErrors(cudaFree(scene));
    film.destroy();
    gui.destroy();
    window.destroy();
    return 0;
}
