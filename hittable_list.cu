#include "hittable.cu"

#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.cu"
#include "sphere.cu"

namespace rt_in_one_weekend {
    class HittableList final : public Hittable {
    public:
        static constexpr int MAX_OBJECTS = 100;
        Sphere objects[MAX_OBJECTS];
        int count;

        __device__ __host__ HittableList() : objects{}, count(0) {
        }

        __device__ __host__ void clear() { count = 0; }

        __device__ __host__ void add(const Sphere &object) {
            if (count < MAX_OBJECTS) {
                objects[count] = object;
                count++;
            }
        }

        __device__ __host__ bool
        hit(const Ray &ray, const Interval interval, HitRecord &rec) const override {
            HitRecord tempRec;
            bool hit_anything = false;
            auto closestSoFar = interval.max;

            for (int i = 0; i < count; i++) {
                if (objects[i].hit(ray, Interval(interval.min, closestSoFar), tempRec)) {
                    hit_anything = true;
                    closestSoFar = tempRec.t;
                    rec = tempRec;
                }
            }

            return hit_anything;
        }
    };
}

#endif
