#ifndef INTERVAL_H
#define INTERVAL_H

namespace rt_in_one_weekend {
    class Interval {
    public:
        double min, max;

        // Default interval is empty
        C_DH Interval() : min(+infinity), max(-infinity) {
        }

        C_DH Interval(double min, double max) : min(min), max(max) {
        }

        C_DH double size() const {
            return max - min;
        }

        C_DH bool contains(double x) const {
            return min <= x && x <= max;
        }

        C_DH bool surrounds(double x) const {
            return min < x && x < max;
        }

        C_DH double clamp(const double x) const {
            if (x < min) return min;
            if (x > max) return max;
            return x;
        }

        static const Interval empty, universe;
    };

    const Interval Interval::empty = Interval(+infinity, -infinity);
    const Interval Interval::universe = Interval(-infinity, +infinity);
}
#endif
