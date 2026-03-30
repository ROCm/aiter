// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Generic C-ABI error bridging for aiter ctypes kernels.
//
// Problem:  extern "C" functions called via Python ctypes cannot propagate
//           C++ exceptions -- crossing the C ABI boundary is undefined
//           behaviour and typically kills the worker process.
//
// Solution: Store the last error in a thread-local string + return status code.
//           Python reads status first, then fetches error from TLS when needed.
//
// Design (same as CUDA Driver API / TVM FFI):
//   - Single extern "C" int entry point per kernel.
//   - Implementation uses AITER_CHECK / HIP_CALL / throw for errors.
//   - aiter_safe_call catches everything, stores in TLS, returns -1.
//   - All callers (C, C++, Python) use the same function and check return code.
//
// Usage in a .cu file:
//
//   #include "aiter_ctypes_error.h"
//   AITER_CTYPES_ERROR_DEF
//
//   AITER_CTYPES_DEFINE_ENTRYPOINT(
//       my_kernel,
//       (int a, float b),
//       (a, b))
//   {
//       AITER_CHECK(a > 0, "a must be positive, got ", a);
//       return 0;
//   }
//
// Python side (aiter/jit/core.py _ctypes_call) is already generic:
//   it probes every .so for aiter_get_last_error / aiter_clear_last_error
//   and raises RuntimeError automatically -- no per-kernel Python changes.
//
#pragma once
#include <string>
#include <utility>

// ---------------------------------------------------------------------------
// AITER_CTYPES_ERROR_DEF -- place once at file scope in ONE translation unit
// per shared object (.so) that is called via ctypes.
// Requires: aiter_hip_common.h included first (provides AITER_C_ITFS).
// Defines:
//   - thread_local g_aiter_last_error  (TLS error storage)
//   - extern "C" aiter_ctypes_abi_version (returns ABI version number)
//   - extern "C" aiter_get_last_error   (returns c_str or nullptr)
//   - extern "C" aiter_clear_last_error
// ---------------------------------------------------------------------------
#define AITER_CTYPES_ERROR_DEF                                                    \
    thread_local std::string g_aiter_last_error;                                \
                                                                                \
    AITER_C_ITFS int aiter_ctypes_abi_version()                                   \
    {                                                                           \
        return 2;                                                               \
    }                                                                           \
                                                                                \
    AITER_C_ITFS const char *aiter_get_last_error()                             \
    {                                                                           \
        return g_aiter_last_error.empty() ? nullptr                             \
                                          : g_aiter_last_error.c_str();         \
    }                                                                           \
                                                                                \
    AITER_C_ITFS void aiter_clear_last_error()                                  \
    {                                                                           \
        g_aiter_last_error.clear();                                             \
    }

// ---------------------------------------------------------------------------
// AITER_CTYPES_ERROR_DECL -- use in additional translation units within the
// same .so when AITER_CTYPES_ERROR_DEF is defined elsewhere.
// ---------------------------------------------------------------------------
#define AITER_CTYPES_ERROR_DECL                                                   \
    extern thread_local std::string g_aiter_last_error

// ---------------------------------------------------------------------------
// aiter_safe_call -- wraps a callable (typically a lambda) with try/catch.
// Catches all C++ exceptions, stores the message in TLS, returns -1.
// On success the callable should return 0.
// ---------------------------------------------------------------------------
template <typename Func>
inline int aiter_safe_call(std::string& tls_error, Func&& fn)
{
    tls_error.clear();
    try {
        return fn();
    } catch (const std::exception& e) {
        tls_error = e.what();
        return -1;
    } catch (...) {
        tls_error = "unknown C++ exception";
        return -1;
    }
}

// ---------------------------------------------------------------------------
// AITER_CTYPES_DEFINE_ENTRYPOINT - define one C-ABI entrypoint with hidden
// try/catch bridging:
//   - static int <func>_impl DECL_ARGS  (user writes body here)
//   - AITER_C_ITFS int <func> DECL_ARGS (generated wrapper)
//
// Example:
//   AITER_CTYPES_DEFINE_ENTRYPOINT(foo, (int x), (x)) { ...; return 0; }
// ---------------------------------------------------------------------------
#define AITER_CTYPES_DEFINE_ENTRYPOINT(func, DECL_ARGS, CALL_ARGS)                \
    static int func##_impl DECL_ARGS;                                            \
    AITER_C_ITFS int func DECL_ARGS                                              \
    {                                                                            \
        return aiter_safe_call(g_aiter_last_error, [&]() -> int {               \
            return func##_impl CALL_ARGS;                                        \
        });                                                                      \
    }                                                                            \
    static int func##_impl DECL_ARGS
