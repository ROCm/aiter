// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <climits>

static constexpr int nextPow2(unsigned int num)
{
  if (num <= 1)
    return 1;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

extern "C" __attribute__((visibility("default")))
int getPaddedM(int M, int N, int K, int gl) {
    int padded_m = M;
    // granularity level, gl = 0, Fine-grained search
    if (gl == 0) {
        // below one 16-row step the next power of two is the finer bucket: a
        // tuned M = 4 / 8 row, not M = 16, serves M = 3 / 5..7
        if(M > 0 && M < 16)
        {
            padded_m = nextPow2(M);
        }
        else if(M <= 256)
        {
            padded_m = (M + 15) / 16 * 16;
        }
        else if(M <= 1024)
        {
            padded_m = (M + 31) / 32 * 32;
        }
        else if(M <= 4096)
        {
            padded_m = (M + 63) / 64 * 64;
        }
        else
        {
            padded_m = (M + 127) / 128 * 128;
        }
    } else if (gl == 1) {
        if (M > 8192 && N > 4096) {
            padded_m = 8192;
        } else {
            padded_m = nextPow2(M);
        }
    }
    return padded_m;
}
