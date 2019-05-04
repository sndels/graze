#ifndef GRAZE_WINDOW_HPP
#define GRAZE_WINDOW_HPP

#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

class Window
{
public:
    Window(uint32_t w, uint32_t h);

    Window(const Window& other) = delete;
    const Window& operator=(const Window& other) = delete;
    
    bool init();
    void destroy();

    bool open() const;
    GLFWwindow* ptr() const;
    uint32_t width() const;
    uint32_t height() const;
    bool shouldRender() const;

    void startFrame();
    void endFrame() const;

    static void errorCallback(int error, const char* description);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void cursorCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void keyCallback(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods);
    static void charCallback(GLFWwindow* window, unsigned int c);

private:
    GLFWwindow* _window;
    uint32_t _w, _h;
    bool _shouldRender;
};

#endif // GRAZE_WINDOW_HPP
