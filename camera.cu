#ifndef CAMERA_H
#define CAMERA_H

#include "Vec3.cu"
#include "interval.cu"

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
        int samplesPerPixel = 10; // Count of random samples for each pixel
        double pixelSamplesScale;

    public:
        Camera(const int imageWidth, const double aspectRatio,
               const unsigned int samplesPerPixel): imageWidth(imageWidth),
                                                    aspectRatio(aspectRatio),
                                                    samplesPerPixel(samplesPerPixel) {
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
            pixelSamplesScale = 1.0 / samplesPerPixel;
        }
    };

    using Color = Vec3;

    __device__ Vec3 sampleSquare() {
        // Use thread and block indices to create unique seed per thread
        unsigned int seed = blockIdx.x * blockDim.x + threadIdx.x + 
                           (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x;
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return Vec3(randomDouble(&seed) - 0.5, randomDouble(&seed) - 0.5, 0);
    }

    __device__ Ray getRay(const int x, const int y, const Camera &camera) {
        // Construct a camera ray originating from the origin and directed at randomly sampled
        // point around the pixel location i, j.

        const auto offset = sampleSquare();
        const auto pixelSample = camera.pixel00Loc
                                 + (x + offset.x()) * camera.pixelDeltaU
                                 + (y + offset.y()) * camera.pixelDeltaV;

        const auto rayOrigin = camera.center;
        const auto rayDirection = pixelSample - rayOrigin;

        return Ray(rayOrigin, rayDirection);
    }

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
            Color pixelColor = Color(0, 0, 0);
            for (int sample = 0; sample < camera.samplesPerPixel; sample++) {
                Ray r = getRay(x, y, camera);
                pixelColor += rayColor(r, world);
            }
            pixelColor = pixelColor * camera.pixelSamplesScale;

            const unsigned int idx = y * camera.imageWidth + x;

            const auto r = pixelColor.x();
            const auto g = pixelColor.y();
            const auto b = pixelColor.z();

            // Translate the [0,1] component values to the byte range [0,255].
            const Interval intensity(0.000, 0.999);
            int rByte = static_cast<int>(256 * intensity.clamp(r));
            int gByte = static_cast<int>(256 * intensity.clamp(g));
            int bByte = static_cast<int>(256 * intensity.clamp(b));

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
