#include "stdio.h"

#include "cuda_helpers.hu"
#include "film.hu"
#include "gui.hpp"
#include "renderer.hu"
#include "sphere.hu"
#include "timer.hpp"
#include "window.hpp"

namespace {
    // Init on gpu to use abstract base class
    __global__ void init_scene(Intersectable** intersectables, Intersectable** scene)
    {
        intersectables[0] = new Sphere{Vec3{0.f, 0.f, -1.f}, 0.5f};
        intersectables[1] = new Sphere{Vec3{0.f, -100.5f, -1.f}, 100.f};
        *scene = new IntersectableList(intersectables, 2);
    }

    __global__ void free_scene(Intersectable** intersectables, Intersectable** scene)
    {
        delete intersectables[0];
        delete intersectables[1];
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
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&intersectables), 2 * sizeof(Intersectable*)));
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
