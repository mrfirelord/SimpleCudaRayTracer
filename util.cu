#include <cuda_runtime.h>
#include "Vec3.cu"

#include "hittable.cu"
#include "hittable_list.cu"
#include "sphere.cu"

namespace rt_in_one_weekend {
    using Color = Vec3;

    __device__ Color rayColor(const Ray &ray, const HittableList &world) {
        HitRecord rec;
        if (world.hit(ray, Interval(0, infinity), rec))
            return 0.5 * (rec.normal + Color(1, 1, 1));

        const Vec3 unitDirection = unitVector(ray.direction());
        const auto a = 0.5 * (unitDirection.y() + 1.0);
        return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
    }

    __global__ void write_color(unsigned char *pixels,
                                const unsigned int width,
                                const unsigned int height,
                                const HittableList &world,
                                const Vec3 cameraCenter,
                                const Vec3 pixel00Loc,
                                const Vec3 pixelDeltaU,
                                const Vec3 pixelDeltaV) {
        // printf("%d", world.count);
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            const auto pixelCenter = pixel00Loc + x * pixelDeltaU + y * pixelDeltaV;
            const auto rayDirection = pixelCenter - cameraCenter;
            const Ray ray(cameraCenter, rayDirection);
            const Color pixelColor = rayColor(ray, world);

            const unsigned int idx = y * width + x;

            const auto r = pixelColor.x();
            const auto g = pixelColor.y();
            const auto b = pixelColor.z();

            // Translate the [0,1] component values to the byte range [0,255].
            const int rByte = static_cast<int>(255.999 * r);
            const int gByte = static_cast<int>(255.999 * g);
            const int bByte = static_cast<int>(255.999 * b);

            // const char *s = "" + rByte + gByte + bByte;
            // printf("[%d;%d;%d] %f;%f;%f \n", x, y, idx, r, g, b);

            const unsigned int outputPosition = idx * 4;
            pixels[outputPosition + 0] = static_cast<unsigned char>(rByte); // R
            pixels[outputPosition + 1] = static_cast<unsigned char>(gByte); // G
            pixels[outputPosition + 2] = static_cast<unsigned char>(bByte); // B
            pixels[outputPosition + 3] = 255; // A
        }
    }
}
