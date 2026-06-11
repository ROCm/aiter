#pragma once
#include <hip/hip_runtime.h>
#include "runner/params.hpp"

// Host launch wrappers. Each is defined in the SAME .cu as its __global__ so the
// <<<>>> launch stays intra-TU (JIT builds with -fno-gpu-rdc -> no cross-TU launch).
void launch_msk0_split(const FmhaFwdSplitParams& sp, dim3 grid, hipStream_t stream);
void launch_msk1_split(const FmhaFwdSplitParams& sp, dim3 grid, hipStream_t stream);
void launch_combine(const FmhaFwdCombineParams& cp, dim3 grid, hipStream_t stream);
