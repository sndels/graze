cmake_minimum_required(VERSION 3.4)
project(graze LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)

# Build for Turing
list(APPEND CMAKE_CUDA_FLAGS "-gencode=arch=compute_75,code=sm_75")

# Set up sub-builds and sources
add_subdirectory(ext)
add_subdirectory(include)
add_subdirectory(src)

# Set up project targets
add_executable(graze ${GRAZE_SOURCES} ${GRAZE_INCLUDES})
target_compile_features(graze
    PRIVATE
    cxx_std_14
)
set_target_properties(graze
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
# TODO: pedantic
target_include_directories(graze
    PRIVATE
    ${GRAZE_INCLUDE_DIR}
)
target_link_libraries(graze
    PRIVATE
    libgl3w
    glfw
    imgui
)
