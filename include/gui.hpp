#ifndef GRAZE_GUI_HPP
#define GRAZE_GUI_HPP

#include <GLFW/glfw3.h>

struct FilmSettings
{
    // These need to be back-to-back as they are written to as int[2]
    uint32_t width = 800;
    uint32_t height = 400;
};

class GUI
{
public:
    GUI(GLFWwindow* window);
    // Needs to happen before window is destroyed
    void destroy();

    const FilmSettings& filmSettings() const;
    bool startRender() const;

    void startFrame();
    void endFrame();

private:
    FilmSettings _filmSettings;
    bool _startRender = false;
};

#endif // GRAZE_GUI_HPP
