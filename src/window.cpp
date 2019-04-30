#include "window.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <stdio.h>

Window::Window(uint32_t w, uint32_t h) :
    _w(w),
    _h(h),
    _startRender(false)
{ }

bool Window::init() {
    // Init GLFW-context
    glfwSetErrorCallback(errorCallback);
    if (!glfwInit())
        return false;

    // Set desired context hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create the window
    _window = glfwCreateWindow(_w, _h, "graze", NULL, NULL);
    if (!_window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(_window);

    // Register callbacks, these will also handle calling ImGui's corresponding callbacks
    glfwSetWindowUserPointer(_window, (void*)this);
    glfwSetFramebufferSizeCallback(_window, framebufferSizeCallback);
    glfwSetCursorPosCallback(_window, cursorCallback);
    glfwSetMouseButtonCallback(_window, mouseButtonCallback);
    glfwSetScrollCallback(_window, scrollCallback);
    glfwSetKeyCallback(_window, keyCallback);
    glfwSetCharCallback(_window, charCallback);

    // Set vsync on
    glfwSwapInterval(1);

    // Init GL
    gl3wInit();
    glClearColor(0.f, 0.f, 0.f, 1.f);

    // Check that GL is happy
    GLenum err = glGetError();
    if(err != GL_NO_ERROR) {
        glfwDestroyWindow(_window);
        glfwTerminate();
        fprintf(stderr, "Error initializing GL!\n");
        fprintf(stderr, "Code: %d\n", err);
        return false;
    }
    return true;
}

void Window::destroy()
{
    glfwDestroyWindow(_window);
    glfwTerminate();
}

bool Window::open() const
{
    return !glfwWindowShouldClose(_window);
}

GLFWwindow* Window::ptr() const
{
    return _window;
}

uint32_t Window::width() const
{
    return _w;
}

uint32_t Window::height() const
{
    return _h;
}

bool Window::startRender() const
{
    return _startRender;
}

void Window::startFrame()
{
    _startRender = false;
    glfwPollEvents();
}

void Window::endFrame() const
{
    glfwSwapBuffers(_window);
}

void Window::errorCallback(int error, const char* description)
{
    fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

void Window::framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    Window* thisPtr = (Window*)glfwGetWindowUserPointer(window);
    thisPtr->_w = width;
    thisPtr->_h = height;
    glViewport(0, 0, width, height);
}

void Window::cursorCallback(GLFWwindow*, double xpos, double ypos)
{
    (void) xpos;
    (void) ypos;
}

void Window::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    // ImGui should handle any events on top of its windows
    if (ImGui::IsMouseHoveringAnyWindow())
        ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    else
        ;
}

void Window::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    // ImGui should handle any events on top of its windows
    if (ImGui::IsMouseHoveringAnyWindow())
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    else
        ;
}

void Window::keyCallback(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods)
{
    Window* thisPtr = (Window*)glfwGetWindowUserPointer(window);
    // Skip key events when e.g. editing an input field
    if (!ImGui::IsAnyItemActive()) {
        if (action == GLFW_PRESS) {
            switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_ENTER:
                thisPtr->_startRender = true;
                break;
            default: break;
            }
        }
    }
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
}

void Window::charCallback(GLFWwindow* window, unsigned int c)
{
    ImGui_ImplGlfw_CharCallback(window, c);
}
