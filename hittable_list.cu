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

        C_DH HittableList() : objects{}, count(0) {
        }

        C_DH void clear() { count = 0; }

        C_DH void add(const Sphere &object) {
            if (count < MAX_OBJECTS) {
                objects[count] = object;
                count++;
            }
        }

        C_DH bool
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
