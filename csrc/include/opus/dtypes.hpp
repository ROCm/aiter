/***************************************************************************************************
 * OPUS, AI (O)(P)erator Micro(U) (S)TD
 *
 * MIT License
 * Copyright (C) 2025-2026 carlus.huang@amd.com
 *
 **************************************************************************************************/
#pragma once

// Scalar spellings of the half-precision types opus registers as dtypes.
//
// This is the single source of truth: opus.hpp feeds these to REGISTER_DTYPE,
// and headers that need the spellings in a translation unit that deliberately
// does not pull in all of opus.hpp alias them from here. Hardcoding __fp16 or
// _Float16 anywhere else silently breaks opus::cast on the compilers where the
// two differ.
//
// clang>=24 types the half operands of the fp16 matrix-core builtins
// (wmma_*_f16, mfma_*f16) and of raw_ptr_buffer_atomic_fadd_v2f16 as _Float16,
// and an __fp16 ext_vector no longer converts to a _Float16 one. Keeping
// __fp16 on clang 20-23 preserves their fp32-intermediate fp16 fma
// (v_fma_mixlo_f16) rather than shifting to a native half fma (v_fma_f16).

namespace opus::dtypes {

#if __clang_major__ >= 24
using bf16 = __bf16;
using fp16 = _Float16;
#elif __clang_major__ >= 20   // enable for rocm 7.0+
using bf16 = __bf16;
using fp16 = __fp16;
#else
using bf16 = unsigned short;
using fp16 = _Float16;
#endif

} // namespace opus::dtypes
