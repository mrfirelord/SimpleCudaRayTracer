cmake_minimum_required(VERSION 3.30)
project(SimpleCudaRayTracer CUDA)

set(CMAKE_CUDA_STANDARD 20)

# Download and build raylib from source
include(FetchContent)

FetchContent_Declare(
        raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG 5.0
)

# Configure raylib options
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(raylib)

# Add executable
add_executable(SimpleCudaRayTracer main.cu)

# Set CUDA properties
set_target_properties(SimpleCudaRayTracer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Link raylib - now built with same compiler
target_link_libraries(SimpleCudaRayTracer PRIVATE raylib)

# Optional: Set output directory
set_target_properties(SimpleCudaRayTracer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)