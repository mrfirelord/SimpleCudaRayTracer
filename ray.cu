#ifndef RAY_H
#define RAY_H

#include "Vec3.cu"

namespace rt_in_one_weekend {
    class Ray {
    public:
        C_DH Ray() = default;

        C_DH Ray(const Point3 &origin, const Vec3 &direction) : orig(origin), dir(direction) {
        }

        C_DH const Point3 &origin() const { return orig; }
        C_DH const Vec3 &direction() const { return dir; }

        C_DH Point3 at(const double t) const { return orig + t * dir; }

    private:
        Point3 orig;
        Vec3 dir;
    };
}

#endif
