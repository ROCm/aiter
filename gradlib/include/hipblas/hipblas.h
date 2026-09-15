// Minimal hipblas.h shim for hipbsolgemm.cu. hipbsolgemm uses only hipblasLt*
// APIs; hipblas.h is required only transitively via hipblaslt.h. The one
// non-Lt symbol used is hipblasStatusToString (in the CHECK_HIPBLAS_ERROR macro).
#pragma once
#include <hipblas-common/hipblas-common.h>

#ifndef HIPBLAS_EXPORT
#define HIPBLAS_EXPORT __attribute__((visibility("default")))
#endif

inline HIPBLAS_EXPORT const char* hipblasStatusToString(hipblasStatus_t status)
{
    switch(status)
    {
    case HIPBLAS_STATUS_SUCCESS:          return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:  return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:     return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:    return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_MAPPING_ERROR:    return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:   return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:    return "HIPBLAS_STATUS_NOT_SUPPORTED";
    case HIPBLAS_STATUS_ARCH_MISMATCH:    return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR: return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
    case HIPBLAS_STATUS_INVALID_ENUM:     return "HIPBLAS_STATUS_INVALID_ENUM";
    case HIPBLAS_STATUS_UNKNOWN:          return "HIPBLAS_STATUS_UNKNOWN";
    default:                              return "HIPBLAS_STATUS_UNDEFINED";
    }
}
