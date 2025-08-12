#ifndef CAMERA_H
#define CAMERA_H

#include "Vec3.cu"
#include "interval.cu"
#include "material.cu"

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
        int samplesPerPixel = 50; // Count of random samples for each pixel
        double pixelSamplesScale;
        unsigned int maxDepth = 50; // Maximum number of ray bounces into scene
        double vfov; // Vertical view angle (field of view)

        Point3 lookfrom; // Point camera is looking from
        Point3 lookat; // Point camera is looking at
        Vec3 vup = Vec3(0, 1, 0); // Camera-relative "up" direction

        Vec3 u, v, w; // Camera frame basis vectors

    public:
        Camera(const int imageWidth,
               const double aspectRatio,
               const unsigned int samplesPerPixel,
               const double vfov,
               Point3 lookfrom,
               Point3 lookAt): imageWidth(imageWidth),
                               aspectRatio(aspectRatio),
                               samplesPerPixel(samplesPerPixel),
                               vfov(vfov),
                               lookfrom(lookfrom),
                               lookat(lookAt) {
            imageHeight = static_cast<int>(imageWidth / aspectRatio);
            imageSize = imageWidth * imageHeight;

            center = lookfrom;
            focalLength = (lookfrom - lookat).length();

            auto theta = degreesToRadians(vfov);
            auto h = std::tan(theta / 2);
            const auto viewportHeight = 2 * h * focalLength;
            const auto viewportWidth = viewportHeight * (static_cast<double>(imageWidth) / imageHeight);

            // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
            w = unitVector(lookfrom - lookat);
            u = unitVector(cross(vup, w));
            v = cross(w, u);

            // Calculate the vectors across the horizontal and down the vertical viewport edges.
            const auto viewportU = viewportWidth * u;
            const auto viewportV = viewportHeight * v;

            // Calculate the horizontal and vertical delta vectors from pixel to pixel.
            pixelDeltaU = viewportU / imageWidth;
            pixelDeltaV = viewportV / imageHeight;

            // Calculate the location of the upper left pixel.
            auto viewportUpperLeft = center - (focalLength * w) - viewportU / 2 - viewportV / 2;
            pixel00Loc = viewportUpperLeft + 0.5 * (pixelDeltaU + pixelDeltaV);
            pixelSamplesScale = 1.0 / samplesPerPixel;
        }
    };

    __device__ Vec3 sampleSquare(curandState *state) {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return Vec3(randomDouble(state) - 0.5, randomDouble(state) - 0.5, 0);
    }

    __device__ Ray getRay(const int x, const int y, const Camera &camera, curandState *state) {
        // Construct a camera ray originating from the origin and directed at randomly sampled
        // point around the pixel location i, j.

        const auto offset = sampleSquare(state);
        const auto pixelSample = camera.pixel00Loc
                                 + (x + offset.x()) * camera.pixelDeltaU
                                 + (y + offset.y()) * camera.pixelDeltaV;

        const auto rayOrigin = camera.center;
        const auto rayDirection = pixelSample - rayOrigin;

        return Ray(rayOrigin, rayDirection);
    }

    __device__ Color rayColor(Ray ray, const HittableList &world, unsigned int maxDepth, curandState *cuRandState) {
        auto finalColor = Color(0, 0, 0);
        auto accumulatedAttenuation = Color(1, 1, 1);

        for (unsigned int depth = 0; depth < maxDepth; depth++) {
            HitRecord rec;
            if (world.hit(ray, Interval(0.001, infinity), rec)) {
                Ray scattered;
                Color attenuation;
                if (rec.material->scatter(ray, rec, attenuation, scattered, cuRandState)) {
                    ray = scattered;
                    accumulatedAttenuation = accumulatedAttenuation * attenuation;
                } else {
                    finalColor = Color(0, 0, 0);
                    break;
                }
            } else {
                const Vec3 unitDirection = unitVector(ray.direction());
                auto a = 0.5 * (unitDirection.y() + 1.0);
                finalColor = (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
                break;
            }
        }

        return accumulatedAttenuation * finalColor;
    }

    C_DH inline double linearToGamma(double linearComponent) {
        return linearComponent > 0 ? sqrt(linearComponent) : 0.0;
    }

    __global__ void writeColor(
        const HittableList &world, const Camera &camera, unsigned char *pixels, curandState *randStates) {
        // printf("%d", world.count);
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < camera.imageWidth && y < camera.imageHeight) {
            const unsigned int idx = y * camera.imageWidth + x;
            curandState *state = &randStates[idx];

            Color pixelColor = Color(0, 0, 0);
            for (int sample = 0; sample < camera.samplesPerPixel; sample++) {
                Ray r = getRay(x, y, camera, state);
                pixelColor += rayColor(r, world, camera.maxDepth, state);
            }
            pixelColor = pixelColor * camera.pixelSamplesScale;

            const auto r = linearToGamma(pixelColor.x());
            const auto g = linearToGamma(pixelColor.y());
            const auto b = linearToGamma(pixelColor.z());

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

    __global__ void initCurand(curandState *state, unsigned long seed, int imageWidth, int imageHeight) {
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < imageWidth && y < imageHeight) {
            const unsigned int idx = y * imageWidth + x;
            curand_init(seed, idx, 0, &state[idx]);
        }
    }
}

#endif
