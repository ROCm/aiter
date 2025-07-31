
#include <cmath>
#include "gemm_common.h"

static constexpr int nextPow2(unsigned int num)
{
  if (num <= 1)
    return 1;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

int getPaddedM(int M, int N, int K, int gl) {
    int padded_m = M;
    // granularity level, gl = 0, Fine-grained search
    if (gl == 0) {
        if(M <= 256)
        {
            padded_m = (M + 15) / 16 * 16; // Round up to the next multiple of 16
        }
        else if(M <= 1024)
        {
            padded_m = (M + 31) / 32 * 32; // Round up to the next multiple of 32
        }
        else if(M <= 4096)
        {
            padded_m = (M + 63) / 64 * 64; // Round up to the next multiple of 64
        }
        else
        {
            padded_m = (M + 127) / 128 * 128; // Round up to the next multiple of 128
        }
    } else if (gl == 1) {
        padded_m = nextPow2(M);
    }
    return padded_m;
    
}