#pragma once
// hip_check.h — HIP_CALL: throw on HIP errors. pybind11 turns the exception
// into a Python RuntimeError, so callers can catch it (the rms_macros.h
// version this replaces called exit(-1) and killed the process).

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#define HIP_CALL(cmd)                                                      \
    do {                                                                   \
        hipError_t _hip_err = (cmd);                                       \
        if (_hip_err != hipSuccess)                                        \
            throw std::runtime_error(std::string(#cmd) + ": " +            \
                                     hipGetErrorString(_hip_err) + " at " + \
                                     __FILE__ + ":" + std::to_string(__LINE__)); \
    } while (0)
