// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

void unload_decode_mla_stage2_asm_f51b00db_0d1d2d3d45678910(void);
void load_decode_mla_stage2_asm_f51b00db_0d1d2d3d45678910(void);
// tt-linker: decode_mla_stage2_asm_f51b00db_0d1d2d3d45678910:hipDeviceptr_t Mid_O, hipDeviceptr_t Mid_lse, hipDeviceptr_t O, hipDeviceptr_t kv_indptr, int32_t stride_mid_ob, int32_t stride_mid_oh, int32_t stride_mid_os, int32_t stride_obs, int32_t stride_oh, int32_t bs, int32_t nheads:16x512x512x64_warps4xstages2
hipError_t decode_mla_stage2_asm_f51b00db_0d1d2d3d45678910(hipStream_t stream, hipDeviceptr_t Mid_O, hipDeviceptr_t Mid_lse, hipDeviceptr_t O, hipDeviceptr_t kv_indptr, int32_t stride_mid_ob, int32_t stride_mid_oh, int32_t stride_mid_os, int32_t stride_obs, int32_t stride_oh, int32_t bs, int32_t nheads);