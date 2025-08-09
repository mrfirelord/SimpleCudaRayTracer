#ifndef HITTABLE_H
#define HITTABLE_H

#include "Ray.cu"

namespace rt_in_one_weekend {
    class HitRecord {
    public:
        Point3 p;
        Vec3 normal;
        double t;

        bool frontFace;

        __device__ __host__ void setFaceNormal(const Ray &r, const Vec3 &outwardNormal) {
            // Sets the hit record normal vector.
            // NOTE: the parameter `outward_normal` is assumed to have unit length.

            frontFace = dot(r.direction(), outwardNormal) < 0;
            normal = frontFace ? outwardNormal : -outwardNormal;
        }
    };

    class Hittable {
    public:
        __device__ __host__ virtual ~Hittable() = default;

        __device__ __host__ virtual bool hit(const Ray &ray, Interval ray_t, HitRecord &rec) const = 0;
    };
}
#endif
