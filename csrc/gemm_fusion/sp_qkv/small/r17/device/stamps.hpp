// SPDX-License-Identifier: MIT
// SM_STAMPS diagnostic instrumentation (default OFF => every macro is empty; separate binary when ON).
// Clock: s_memrealtime (100 MHz). Same-GPU deltas only. Plain global stores by one thread per WG; no fences added.
#pragma once
#include "sm_abi_v2.h"
#ifndef SM_STAMPS
#define SM_STAMPS 0
#endif
#if SM_STAMPS
#define SM_STAMP_ON 1
// record pointer: kind_off = 0 for RAN WGs, SM_DBG_RAN_MAXWG for GEMM WGs. Replay slot from dbg[0] (advanced by last RAN WG).
#define SM_STAMP_REC(dbg, kind_off, wg) \
    ((dbg) ? ((dbg) + SM_DBG_HDR + ((((dbg)[0] % SM_DBG_SLOTS) * (SM_DBG_RAN_MAXWG + SM_DBG_GEMM_MAXWG)) + (kind_off) + (wg)) * SM_DBG_WORDS) : nullptr)
#define SM_STAMP(rec, idx) do { if (rec) { (rec)[(idx)] = __builtin_amdgcn_s_memrealtime(); } } while (0)
#define SM_STAMP_VAL(rec, idx, v) do { if (rec) { (rec)[(idx)] = static_cast<uint64_t>(v); } } while (0)
// HW_ID (s_getreg id 4, offset 0, size 32); UNVERIFIED encoding on gfx950 - only used in the diagnostic build.
#define SM_STAMP_HWID(rec, idx) do { if (rec) { (rec)[(idx)] = static_cast<uint64_t>(__builtin_amdgcn_s_getreg(4 | (31 << 11))); } } while (0)
// XCC_ID (HW_REG_XCC_ID = 20, bits 3:0 on gfx940+): encoding UNVERIFIED on this toolchain; stamps build only.
#define SM_STAMP_XCC(rec, idx) do { if (rec) { (rec)[(idx)] = static_cast<uint64_t>(__builtin_amdgcn_s_getreg(20 | (3 << 11))); } } while (0)
#else
#define SM_STAMP_XCC(rec, idx) do { } while (0)
#define SM_STAMP_ON 0
#define SM_STAMP_REC(dbg, kind_off, wg) (static_cast<uint64_t*>(nullptr))
#define SM_STAMP(rec, idx) do { } while (0)
#define SM_STAMP_VAL(rec, idx, v) do { } while (0)
#define SM_STAMP_HWID(rec, idx) do { } while (0)
#endif
