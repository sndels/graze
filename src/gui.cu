#include "gui.hu"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace {
    const ImVec2 windowPos{10.f, 10.f};
    const ImVec2 windowSize{300.f, 265.f};
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

const CameraSettings& GUI::cameraSettings() const
{
    return _cameraSettings;
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
        // FilmSettings has width and height back-to-back
        ImGui::InputScalarN("resolution", ImGuiDataType_U32, &_filmSettings.width, 2);
        ImGui::InputScalar("samples", ImGuiDataType_U32, &_filmSettings.samples);
        ImGui::PopItemWidth();
    }
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushItemWidth(float3InputWidth);
        ImGui::DragFloat3("eye", (float*)&_cameraSettings.eye, 0.1f, -FLT_MAX, FLT_MAX, "%.1f");
        ImGui::DragFloat3("target", (float*)&_cameraSettings.target, 0.1f, -FLT_MAX, FLT_MAX, "%.1f");
        ImGui::PopItemWidth();
        ImGui::PushItemWidth(floatInputWidth);
        ImGui::DragFloat("focalLength", &_cameraSettings.focalLength, 0.1f, 0.f, FLT_MAX, "%.1f");
        ImGui::DragFloat("aperture", &_cameraSettings.aperture, 0.1f, 0.f, 100.f, "%.1f");
        ImGui::DragFloat("fov", &_cameraSettings.fov, 0.1f, 0.1f, 179.9f, "%.1f");
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
