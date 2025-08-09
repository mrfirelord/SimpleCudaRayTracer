#ifndef INTERVAL_H
#define INTERVAL_H

namespace rt_in_one_weekend {
    class Interval {
    public:
        double min, max;

        // Default interval is empty
        __device__ __host__ Interval() : min(+infinity), max(-infinity) {
        }

        __device__ __host__ Interval(double min, double max) : min(min), max(max) {
        }

        __device__ __host__ double size() const {
            return max - min;
        }

        __device__ __host__ bool contains(double x) const {
            return min <= x && x <= max;
        }

        __device__ __host__ bool surrounds(double x) const {
            return min < x && x < max;
        }

        static const Interval empty, universe;
    };

    const Interval Interval::empty = Interval(+infinity, -infinity);
    const Interval Interval::universe = Interval(-infinity, +infinity);
}
#endif
