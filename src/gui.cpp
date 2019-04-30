#include "gui.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace {
    const ImVec2 windowPos{10.f, 10.f};
    const ImVec2 windowSize{300.f, 244.f};
    const float floatInputWidth = 50.f;
    const float float2InputWidth = 121.f;
    const float float3InputWidth = 221.f;
}

GUI::GUI(GLFWwindow* window)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, false);
    ImGui_ImplOpenGL3_Init("#version 150");
    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowBorderSize = 0.f;
    style.WindowRounding = 0.f;
}

void GUI::destroy()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

const FilmSettings& GUI::filmSettings() const
{
    return _filmSettings;
}

bool GUI::startRender() const
{
    return _startRender;
}

void GUI::startFrame()
{
    // Start ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(windowPos, ImGuiSetCond_Always);
    ImGui::SetNextWindowSize(windowSize, ImGuiSetCond_Always);
    ImGui::Begin("Such ui");

    if (ImGui::CollapsingHeader("Film", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushItemWidth(float2InputWidth);
        // Use hex characters to enforce unsigned input
        // FilmSettings has width and height back-to-back
        ImGui::InputInt2("resolution", (int*)&_filmSettings.width, ImGuiInputTextFlags_CharsHexadecimal);
        ImGui::PopItemWidth();
    }

    ImGui::Spacing();
    _startRender = ImGui::Button("Render"); ImGui::SameLine();
    ImGui::End();
}

void GUI::endFrame()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
