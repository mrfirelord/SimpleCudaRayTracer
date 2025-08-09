#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.cu"
#include "Vec3.cu"

namespace rt_in_one_weekend {
    class Sphere final : public Hittable {
    public:
        __device__ __host__ Sphere() : center(Point3(0, 0, 0)), radius(0) {
        }

        __device__ __host__ Sphere(const Point3 &center, double radius) : center(center), radius(std::fmax(0, radius)) {
        }

        __device__ __host__ bool
        hit(const Ray &r, const Interval interval, HitRecord &rec) const override {
            const Vec3 oc = center - r.origin();
            const auto a = r.direction().lengthSquared();
            const auto h = dot(r.direction(), oc);
            const auto c = oc.lengthSquared() - radius * radius;

            const auto discriminant = h * h - a * c;
            if (discriminant < 0)
                return false;

            const auto sqrtDiscriminant = std::sqrt(discriminant);

            // Find the nearest root that lies in the acceptable range.
            auto root = (h - sqrtDiscriminant) / a;
            if (!interval.surrounds(root)) {
                root = (h + sqrtDiscriminant) / a;
                if (!interval.surrounds(root))
                    return false;
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            const Vec3 outwardNormal = (rec.p - center) / radius;
            rec.setFaceNormal(r, outwardNormal);

            return true;
        }

    private:
        Point3 center;
        double radius;
    };
}

#endif
