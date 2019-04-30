#include "stdio.h"

#include "film.hu"
#include "gui.hpp"
#include "renderer.hu"
#include "timer.hpp"
#include "window.hpp"

int main()
{
    Window window{1280, 720};
    if (!window.init())
        return -1;

    GUI gui{window.ptr()};
    Film film{gui.filmSettings()};
    Timer timer;

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
            render(&film);
            printf("Done in %.3fs!\n", timer.seconds());
        }

        film.display(window.width(), window.height());
        gui.endFrame();
        window.endFrame();
    }

    film.destroy();
    gui.destroy();
    window.destroy();
    return 0;
}
