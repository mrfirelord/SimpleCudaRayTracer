#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

namespace rt_in_one_weekend {
    class Vec3 {
    public:
        double e[3];

        C_DH Vec3() : e{0, 0, 0} {
        }

        C_DH Vec3(double e0, double e1, double e2) : e{e0, e1, e2} {
        }

        C_DH double x() const { return e[0]; }
        C_DH double y() const { return e[1]; }
        C_DH double z() const { return e[2]; }

        C_DH Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
        C_DH double operator[](int i) const { return e[i]; }
        C_DH double &operator[](int i) { return e[i]; }

        C_DH Vec3 &operator+=(const Vec3 &v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        C_DH Vec3 &operator*=(double t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

        C_DH Vec3 &operator/=(double t) {
            return *this *= 1 / t;
        }

        C_DH double length() const {
            return sqrt(lengthSquared());
        }

        C_DH double lengthSquared() const {
            return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        }

        C_DH bool nearZero() const {
            // Return true if the vector is close to zero in all dimensions.
            auto s = 1e-8;
            return fabs(e[0]) < s && fabs(e[1]) < s && fabs(e[2]) < s;
        }

        C_D static Vec3 random(curandState *state) {
            return Vec3(randomDouble(state), randomDouble(state), randomDouble(state));
        }

        C_D static Vec3 random(curandState *state, const double min, const double max) {
            return Vec3(randomDouble(state, min, max), randomDouble(state, min, max), randomDouble(state, min, max));
        }
    };

    // Vector Utility Functions

    C_DH inline std::ostream &operator<<(std::ostream &out, const Vec3 &v) {
        return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
    }

    C_DH inline Vec3 operator+(const Vec3 &u, const Vec3 &v) {
        return Vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
    }

    C_DH inline Vec3 operator-(const Vec3 &u, const Vec3 &v) {
        return Vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
    }

    C_DH inline Vec3 operator*(const Vec3 &u, const Vec3 &v) {
        return Vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
    }

    C_DH inline Vec3 operator*(double t, const Vec3 &v) {
        return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
    }

    C_DH inline Vec3 operator*(const Vec3 &v, double t) {
        return t * v;
    }

    C_DH inline Vec3 operator/(const Vec3 &v, double t) {
        return (1 / t) * v;
    }

    C_DH inline double dot(const Vec3 &u, const Vec3 &v) {
        return u.e[0] * v.e[0]
               + u.e[1] * v.e[1]
               + u.e[2] * v.e[2];
    }

    C_DH inline Vec3 cross(const Vec3 &u, const Vec3 &v) {
        return Vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                    u.e[2] * v.e[0] - u.e[0] * v.e[2],
                    u.e[0] * v.e[1] - u.e[1] * v.e[0]);
    }

    C_DH inline Vec3 unitVector(const Vec3 &v) {
        return v / v.length();
    }

    C_D inline Vec3 randomUnitVector(curandState *state) {
        while (true) {
            auto p = Vec3::random(state, -1, 1);
            auto lengthSquared = p.lengthSquared();
            if (1e-160 < lengthSquared && lengthSquared <= 1)
                return p / sqrt(lengthSquared);
        }
    }

    C_D inline Vec3 randomOnHemisphere(const Vec3 &normal, curandState *state) {
        Vec3 onUnitSphere = randomUnitVector(state);
        if (dot(onUnitSphere, normal) > 0.0) // In the same hemisphere as the normal
            return onUnitSphere;
        return -onUnitSphere;
    }

    C_DH inline Vec3 reflect(const Vec3 &v, const Vec3 &n) {
        return v - 2 * dot(v, n) * n;
    }

    using Point3 = Vec3;
    using Color = Vec3;
}
#endif
