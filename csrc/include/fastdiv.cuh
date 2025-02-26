#include <cstdint>

struct uint_fastdiv {
    uint32_t d;
    uint32_t m;
    uint32_t s;
    uint32_t a;
  
    __host__ __device__ uint_fastdiv() : d(0), m(0), s(0), a(0) {}
  
    __host__ uint_fastdiv(uint32_t d) : d(d) {
      unsigned int p, nc, delta, q1, r1, q2, r2;
      a = 0;
      nc = unsigned(-1) - unsigned(-d) % d;
      p = 31;
      q1 = 0x80000000 / nc;
      r1 = 0x80000000 - q1 * nc;
      q2 = 0x7FFFFFFF / d;
      r2 = 0x7FFFFFFF - q2 * d;
      do {
        p++;
        if (r1 >= nc - r1) {
          q1 = 2 * q1 + 1;
          r1 = 2 * r1 - nc;
        } else {
          q1 = 2 * q1;
          r1 = 2 * r1;
        }
        if (r2 + 1 >= d - r2) {
          if (q2 >= 0x7FFFFFFF) a = 1;
          q2 = 2 * q2 + 1;
          r2 = 2 * r2 + 1 - d;
        } else {
          if (q2 >= 0x80000000) a = 1;
          q2 = 2 * q2;
          r2 = 2 * r2 + 1;
        }
        delta = d - 1 - r2;
      } while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));
      m = q2 + 1;
      s = p - 32;
    }
    __host__ __device__ __forceinline__ operator unsigned int() const { return d; }

    __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
      if (d == 1) {
        q = n;
      } else {
#ifdef __CUDA_ARCH__
      q = __umulhi(m, n);
#else
      q = (((unsigned long long)((long long)m * (long long)n)) >> 32);
#endif
      q += a * n;
      q >>= s;
    }
    r = n - q * d;
  }
};