#ifndef RAY_H
#define RAY_H

#include "Vec3.cu"

namespace rt_in_one_weekend {
    class Ray {
    public:
        __device__ __host__ Ray() = default;

        __device__ __host__ Ray(const Point3 &origin, const Vec3 &direction) : orig(origin), dir(direction) {
        }

        __device__ __host__ const Point3 &origin() const { return orig; }
        __device__ __host__ const Vec3 &direction() const { return dir; }

        __device__ __host__ Point3 at(const double t) const { return orig + t * dir; }

    private:
        Point3 orig;
        Vec3 dir;
    };
}

#endif
