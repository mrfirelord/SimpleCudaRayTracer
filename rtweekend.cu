#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>


// C++ Std Usings

using std::make_shared;
using std::shared_ptr;

// Constants

#define CHECK_CUDA(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
exit(1); \
} \
} while(0)

namespace rt_in_one_weekend {
    __device__ constexpr double infinity = std::numeric_limits<double>::infinity();
    __device__ constexpr double pi = 3.1415926535897932385;

    // Utility Functions

    __device__ __host__ inline double degreesToRadians(const double degrees) { return degrees * pi / 180.0; }

    __device__ inline double randomDouble(unsigned int *seed) {
        // Linear congruential generator
        *seed = *seed * 1103515245 + 12345;
        return (*seed & 0x7fffffff) / static_cast<double>(0x7fffffff);
    }

    __device__ inline double randomDouble(unsigned int *seed, double min, double max) {
        return min + (max - min) * randomDouble(seed);
    }
}

// Common Headers

#include "Vec3.cu"
#include "interval.cu"
#include "ray.cu"

#endif
