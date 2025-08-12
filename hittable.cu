#ifndef HITTABLE_H
#define HITTABLE_H

#include "Ray.cu"

namespace rt_in_one_weekend {
    class Material;

    class HitRecord {
    public:
        Point3 p;
        Vec3 normal;
        Material *material;
        double t;

        bool frontFace;

        C_DH void setFaceNormal(const Ray &r, const Vec3 &outwardNormal) {
            // Sets the hit record normal vector.
            // NOTE: the parameter `outward_normal` is assumed to have unit length.

            frontFace = dot(r.direction(), outwardNormal) < 0;
            normal = frontFace ? outwardNormal : -outwardNormal;
        }
    };

    class Hittable {
    public:
        C_DH virtual ~Hittable() = default;

        C_DH virtual bool hit(const Ray &ray, Interval ray_t, HitRecord &rec) const = 0;
    };
}
#endif
