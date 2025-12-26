#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <exception>

#define CHECK_COND(x) \
    do { \
        if (!(x)) { \
            std::cerr << "check failed, file=" \
                << __FILE__ << ", line=" \
                << __LINE__ << std::endl; \
            std::terminate(); \
        } \
    } while(false)

#define CHECK_HIP(x) \
    do { \
        hipError_t __err_code = (x); \
        if( __err_code != hipSuccess ) { \
            std::cerr << "call hip api failed, file=" \
                << __FILE__ << ", line=" \
                << __LINE__ << ", name=" \
                << hipGetErrorName(__err_code) \
                << std::endl; \
            std::terminate(); \
        } \
    } while(false)