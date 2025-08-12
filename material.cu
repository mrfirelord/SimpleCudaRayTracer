#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.cu"

namespace rt_in_one_weekend {
    class Material {
    public:
        C_DH virtual ~Material() = default;

        C_D virtual bool scatter(const Ray &r_in,
                                 const HitRecord &rec,
                                 Color &attenuation,
                                 Ray &scattered,
                                 curandState *cuRandState) const {
            return false;
        }
    };

    class Lambertian final : public Material {
    public:
        C_DH Lambertian(const Color &albedo) : albedo(albedo) {
        }

        C_D bool scatter(const Ray &incoming,
                         const HitRecord &rec,
                         Color &attenuation,
                         Ray &scattered,
                         curandState *cuRandState) const override {
            auto scatterDirection = rec.normal + randomUnitVector(cuRandState);

            // Catch degenerate scatter direction
            if (scatterDirection.nearZero())
                scatterDirection = rec.normal;

            scattered = Ray(rec.p, scatterDirection);
            attenuation = albedo;
            return true;
        }

    private:
        Color albedo;
    };

    class Metal final : public Material {
    public:
        C_DH Metal(const Color &albedo) : albedo(albedo) {
        }

        C_D bool scatter(const Ray &incoming,
                         const HitRecord &rec,
                         Color &attenuation,
                         Ray &scattered,
                         curandState *cuRandState) const override {
            const Vec3 reflected = reflect(incoming.direction(), rec.normal);
            scattered = Ray(rec.p, reflected);
            attenuation = albedo;
            return true;
        }

    private:
        Color albedo;
    };

    __global__ void initLambertian(Lambertian *material, Color albedo) {
        new(material) Lambertian(albedo);
    }

    __global__ void initMetal(Metal *material, Color albedo) {
        new(material) Metal(albedo);
    }
}

#endif
