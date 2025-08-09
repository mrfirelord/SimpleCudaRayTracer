cmake_minimum_required(VERSION 3.30)
project(untitled3 CUDA)

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
add_executable(untitled3 main.cu)

# Set CUDA properties
set_target_properties(untitled3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Link raylib - now built with same compiler
target_link_libraries(untitled3 PRIVATE raylib)

# Optional: Set output directory
set_target_properties(untitled3 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)