#ifndef CAMERA_H
#define CAMERA_H

#include "Vec3.cu"

namespace rt_in_one_weekend {
    class Camera {
    public:
        int imageWidth;
        int imageHeight;
        int imageSize;
        double aspectRatio;
        double focalLength = 1.0;
        Point3 center;
        Vec3 pixel00Loc;
        Vec3 pixelDeltaU;
        Vec3 pixelDeltaV;

    public:
        Camera(const int imageWidth, const double aspectRatio): imageWidth(imageWidth), aspectRatio(aspectRatio) {
            imageHeight = static_cast<int>(imageWidth / aspectRatio);
            imageSize = imageWidth * imageHeight;

            constexpr auto viewportHeight = 2.0;
            const auto viewportWidth = viewportHeight * (static_cast<double>(imageWidth) / imageHeight);
            center = Point3(0, 0, 0);

            // Calculate the vectors across the horizontal and down the vertical viewport edges.
            const auto viewportU = Vec3(viewportWidth, 0, 0);
            const auto viewportV = Vec3(0, -viewportHeight, 0);

            // Calculate the horizontal and vertical delta vectors from pixel to pixel.
            pixelDeltaU = viewportU / imageWidth;
            pixelDeltaV = viewportV / imageHeight;

            // Calculate the location of the upper left pixel.
            const auto viewportUpperLeft = center - Vec3(0, 0, focalLength) - viewportU / 2 - viewportV / 2;
            pixel00Loc = viewportUpperLeft + 0.5 * (pixelDeltaU + pixelDeltaV);
        }
    };

    using Color = Vec3;

    __device__ Color rayColor(const Ray &ray, const HittableList &world) {
        HitRecord rec;
        if (world.hit(ray, Interval(0, infinity), rec))
            return 0.5 * (rec.normal + Color(1, 1, 1));

        const Vec3 unitDirection = unitVector(ray.direction());
        const auto a = 0.5 * (unitDirection.y() + 1.0);
        return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
    }

    __global__ void write_color(const HittableList &world, const Camera &camera, unsigned char *pixels) {
        // printf("%d", world.count);
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < camera.imageWidth && y < camera.imageHeight) {
            const auto pixelCenter = camera.pixel00Loc + x * camera.pixelDeltaU + y * camera.pixelDeltaV;
            const auto rayDirection = pixelCenter - camera.center;
            const Ray ray(camera.center, rayDirection);
            const Color pixelColor = rayColor(ray, world);

            const unsigned int idx = y * camera.imageWidth + x;

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

#endif
