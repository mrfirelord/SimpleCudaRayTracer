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
        Point3 cameraCenter;
        Vec3 pixel00Loc;
        Vec3 viewportU;
        Vec3 viewportV;
        Vec3 pixelDeltaU;
        Vec3 pixelDeltaV;

    public:
        Camera(const int imageWidth, const double aspectRatio): imageWidth(imageWidth), aspectRatio(aspectRatio) {
            imageHeight = static_cast<int>(imageWidth / aspectRatio);
            imageSize = imageWidth * imageHeight;

            constexpr auto viewportHeight = 2.0;
            const auto viewportWidth = viewportHeight * (static_cast<double>(imageWidth) / imageHeight);
            cameraCenter = Point3(0, 0, 0);

            // Calculate the vectors across the horizontal and down the vertical viewport edges.
            viewportU = Vec3(viewportWidth, 0, 0);
            viewportV = Vec3(0, -viewportHeight, 0);

            // Calculate the horizontal and vertical delta vectors from pixel to pixel.
            pixelDeltaU = viewportU / imageWidth;
            pixelDeltaV = viewportV / imageHeight;

            // Calculate the location of the upper left pixel.
            const auto viewportUpperLeft = cameraCenter - Vec3(0, 0, focalLength) - viewportU / 2 - viewportV / 2;
            pixel00Loc = viewportUpperLeft + 0.5 * (pixelDeltaU + pixelDeltaV);
        }
    };
}

#endif
