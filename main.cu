#include <raylib.h>
#include <cuda_runtime.h>

#include "rtweekend.cu"
#include <iostream>
#include "hittable_list.cu"
#include "sphere.cu"
#include "camera.cu"

#define RAYLIB_STATIC

void printDeviceInfo();

int main() {
    printDeviceInfo();

    // Camera

    const rt_in_one_weekend::Camera camera(3024, 16.0 / 9.0, 10U);
    rt_in_one_weekend::Camera *dCamera;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dCamera), sizeof(rt_in_one_weekend::Camera)));
    CHECK_CUDA(cudaMemcpy(dCamera, &camera, sizeof(rt_in_one_weekend::Camera), cudaMemcpyHostToDevice));

    // World

    rt_in_one_weekend::HittableList hWorld;
    hWorld.add(rt_in_one_weekend::Sphere(rt_in_one_weekend::Point3(0, 0, -1), 0.5));
    hWorld.add(rt_in_one_weekend::Sphere(rt_in_one_weekend::Point3(0, -100.5, -1), 100));

    rt_in_one_weekend::HittableList *dWorld;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dWorld), sizeof(rt_in_one_weekend::HittableList)));
    CHECK_CUDA(cudaMemcpy(dWorld, &hWorld, sizeof(rt_in_one_weekend::HittableList), cudaMemcpyHostToDevice));

    // Output pixels

    auto *hPixels = static_cast<unsigned char *>(malloc(camera.imageSize * 4));

    unsigned char *dPixels;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dPixels), camera.imageSize * 4 * sizeof(unsigned char)));


    dim3 blockSize(8, 8);
    dim3 gridSize((camera.imageWidth + blockSize.x - 1) / blockSize.x,
                  (camera.imageHeight + blockSize.y - 1) / blockSize.y);

    InitWindow(camera.imageWidth, camera.imageHeight, "CUDA + raylib Demo");

    // Create raylib texture
    const Image image = GenImageColor(camera.imageWidth, camera.imageHeight, BLACK);
    const Texture2D texture = LoadTextureFromImage(image);
    UnloadImage(image);

    while (!WindowShouldClose()) {
        writeColor<<<gridSize, blockSize>>>(*dWorld, *dCamera, dPixels);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(hPixels, dPixels, camera.imageSize * 4, cudaMemcpyDeviceToHost));
        UpdateTexture(texture, hPixels);

        BeginDrawing();
        ClearBackground(WHITE);
        DrawTexture(texture, 0, 0, WHITE);
        DrawFPS(10, camera.imageHeight - 30);
        EndDrawing();
    }

    UnloadTexture(texture);
    free(hPixels);
    cudaFree(dPixels);
    cudaFree(dWorld);
    CloseWindow();

    return 0;
}

void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);

        std::cout << "\n=== Device " << i << ": " << prop.name << " ===" << std::endl;

        // Memory limitations
        std::cout << "Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "Constant memory: " << prop.totalConstMem / 1024 << " KB" << std::endl;

        // Compute limitations
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: (" << prop.maxThreadsDim[0] << ", "
                << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "Max grid dimensions: (" << prop.maxGridSize[0] << ", "
                << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;

        // Hardware limitations
        std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;
        std::cout << "Warp size: " << prop.warpSize << std::endl;
        std::cout << "Registers per block: " << prop.regsPerBlock << std::endl;
        std::cout << "Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "Memory clock rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;

        // Other useful limits
        std::cout << "Max texture 1D size: " << prop.maxTexture1D << std::endl;
        std::cout << "Max texture 2D size: (" << prop.maxTexture2D[0] << ", "
                << prop.maxTexture2D[1] << ")" << std::endl;
        std::cout << "Can map host memory: " << (prop.canMapHostMemory ? "Yes" : "No") << std::endl;
        std::cout << "Unified addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
        std::cout << "======" << std::endl;
    }
}
