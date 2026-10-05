// ============================================================================
//  gfx950 (MI350X / MI355X) hardware defect reproducer
//  Packed-FP32 src1 high-dword operand read returns 0 during a concurrent MFMA
//  ----------------------------------------------------------------------------
//  Contact/able to re-run on: AMD Instinct MI350X, ROCm 7.0, gfx950.
// ============================================================================
//
//  THE DEFECT
//  ----------
//  On gfx950, a packed-FP32 ALU instruction
//        V_PK_MUL_F32 / V_PK_ADD_F32 / V_PK_FMA_F32
//  whose src1 is a VGPR read with its HIGH dword selected for the LOW result
//  lane (op_sel bit0 = 1 on src1, = 0 on src0) returns ZERO for that operand
//  read, in hardware lanes 48-63, WHEN another wave resident on the same SIMD
//  is concurrently executing a double-rate XDL MFMA. The packed op then
//  computes  src0 * 0  (or  src0 + 0).
//
//  Characterised (see the accompanying write-up for the full experiment set):
//    * Cross-wave. A single wave NEVER reproduces it. It requires >=2 waves
//      co-resident on one SIMD and concurrent XDL MFMA activity.
//    * A read-datapath fault, not data corruption. The MFMA output and the
//      src1 register both read back correct; only the src1 high-dword *read*
//      inside the packed op is lost.
//    * The src1 (operand B) port specifically. The identical arithmetic with
//      the high-dword select placed on src0 (operand A) is always correct.
//      -> this is why commuting the operands is a valid software workaround.
//    * Independent of register numbers, of how src1 was produced, and of src1
//      readiness (a settled constant fails just as hard).
//    * Triggered only by the gfx950 double-rate XDL MFMAs
//      (32x32x16 / 16x16x32 f16 and bf16; 32x32x64 / 16x16x128 f8f6f4;
//       32x32x32 i8; and the AGPR-accumulator forms). Legacy MFMAs, DGEMM,
//      SGEMM, SMFMAC, DOT and transcendental ops do NOT trigger it.
//
//  WHY ONE LINE OF INLINE ASM
//  --------------------------
//  The aggressor MFMA uses the __builtin_amdgcn_mfma intrinsic (no asm).
//  The victim packed op is forced with a single inline-asm instruction ONLY
//  because the HIP-C compiler (hipcc, ROCm 7.0 LLVM) does not reliably emit the
//  vulnerable packed form from plain C++: in this context (a loop adjacent to
//  MFMA activity) its instruction selection scalarises/unpacks the packed
//  "swap multiply" (r.x = a.x*b.y; r.y = a.y*b.x) into scalar v_mul_f32, which is
//  not vulnerable, so the packed instruction never reaches the hardware. To
//  present the exact instruction under test deterministically and independent of
//  compiler version, we emit it directly. Everything else is ordinary C++/HIP.
//
//  BUILD / RUN
//  -----------
//    hipcc --offload-arch=gfx950 pk_mfma_crosswave.cpp -o pk_mfma_crosswave
//
//    ./pk_mfma_crosswave --op mul            # expect: thousands miscomputed
//    ./pk_mfma_crosswave --op add            # expect: thousands miscomputed
//    ./pk_mfma_crosswave --op fma            # expect: thousands miscomputed
//    ./pk_mfma_crosswave --op mul --safe     # commuted op -> expect: 0
//    ./pk_mfma_crosswave --op mul --single   # one wave     -> expect: 0
//
//  Each victim lane repeatedly computes a value that must equal 96.0 and keeps
//  a running minimum (so a transient wrong read cannot be masked by a later
//  correct one). Any stored value != 96.0 is a hardware miscompute.
// ============================================================================

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

typedef _Float16 h8  __attribute__((ext_vector_type(8)));
typedef float    f16 __attribute__((ext_vector_type(16)));

__device__ float g_sink;

// One 8-wave block + large LDS so exactly one block fits per CU; with 4 SIMDs/CU
// this places two waves on every SIMD (gfx950 has 160 KiB LDS/CU).
static constexpr int WAVES     = 8;
static constexpr int THREADS   = WAVES * 64;
static constexpr int LDS_BYTES = 40 * 1024;

enum Op { MUL = 0, ADD = 1, FMA = 2 };

// Pack two f32 into one 64-bit value (a VGPR pair at the ISA level).
__device__ __forceinline__ unsigned long long pack(float lo, float hi) {
  unsigned long long v; float t[2] = {lo, hi}; memcpy(&v, t, 8); return v;
}
__device__ __forceinline__ float low(unsigned long long v) {
  float t; memcpy(&t, &v, 4); return t;
}

// One vulnerable packed op (src1 = VGPR, high dword selected for the low lane).
// SAFE==true uses the commuted form (high-dword select on src0) -> expected clean.
template <int OP, bool SAFE>
__device__ __forceinline__ unsigned long long packed_op(unsigned long long a,
                                                        unsigned long long s,
                                                        unsigned long long c) {
  unsigned long long r;
  if (OP == MUL) {
    if (SAFE) asm volatile("v_pk_mul_f32 %0, %2, %1 op_sel:[1,0]\n" : "=v"(r) : "v"(a), "v"(s));
    else      asm volatile("v_pk_mul_f32 %0, %1, %2 op_sel:[0,1]\n" : "=v"(r) : "v"(a), "v"(s));
  } else if (OP == ADD) {
    if (SAFE) asm volatile("v_pk_add_f32 %0, %2, %1 op_sel:[1,0]\n" : "=v"(r) : "v"(a), "v"(s));
    else      asm volatile("v_pk_add_f32 %0, %1, %2 op_sel:[0,1]\n" : "=v"(r) : "v"(a), "v"(s));
  } else {
    if (SAFE) asm volatile("v_pk_fma_f32 %0, %3, %1, %2 op_sel:[1,0,0]\n" : "=v"(r) : "v"(a), "v"(c), "v"(s));
    else      asm volatile("v_pk_fma_f32 %0, %1, %3, %2 op_sel:[0,1,0]\n" : "=v"(r) : "v"(a), "v"(c), "v"(s));
  }
  return r;
}

// Operands chosen so the correct low result is exactly 96.0 for every op, and
// so that losing the src1 high-dword read (reading 0) changes it:
//   MUL: a=(96,96),  s=(0.5,1.0) -> 96 * s.hi(1.0)       = 96  ; fault -> 96*0   = 0
//   ADD: a=(90,90),  s=(0.0,6.0) -> 90 + s.hi(6.0)       = 96  ; fault -> 90+0   = 90
//   FMA: a=(96,96),  s=(0.5,1.0), c=(0,0)
//                                -> 96 * s.hi(1.0) + 0   = 96  ; fault -> 96*0+0 = 0
template <int OP> __device__ __forceinline__ unsigned long long opA() { return pack(96,96); }
template <> __device__ __forceinline__ unsigned long long opA<ADD>() { return pack(90,90); }
template <int OP> __device__ __forceinline__ unsigned long long opS() { return pack(0.5f,1.0f); }
template <> __device__ __forceinline__ unsigned long long opS<ADD>() { return pack(0.0f,6.0f); }

template <int OP, bool SAFE>
__global__ __launch_bounds__(THREADS, 1)
void kernel(float* __restrict__ out, int iters) {
  __shared__ float lds[LDS_BYTES / sizeof(float)];
  if (threadIdx.x == 0) lds[0] = out[0];
  __syncthreads();

  const int wave = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;

  if (wave < WAVES / 2) {                             // aggressor: MFMA only
    h8 a = {1,1,1,1,1,1,1,1}, b = {1,1,1,1,1,1,1,1};
    f16 c = {0};
    for (int i = 0; i < iters; ++i)                   // long MFMA stream (outlasts the victim)
      c = __builtin_amdgcn_mfma_f32_32x32x16_f16(a, b, c, 0, 0, 0);
    if (c[0] == -1.0f) g_sink = c[0];
    return;
  }

  const unsigned long long a = opA<OP>(), s = opS<OP>(), c = pack(0.0f, 0.0f);
  float acc = 96.0f;
  for (int i = 0; i < iters; ++i)                     // packed op each iteration, min-tracked
    acc = fminf(acc, low(packed_op<OP, SAFE>(a, s, c)));

  out[(blockIdx.x * (WAVES / 2) + (wave - WAVES / 2)) * 64 + lane] = acc;
}

template <int OP>
__global__ __launch_bounds__(64)
void single(float* __restrict__ out, int iters) {
  const int lane = threadIdx.x & 63;
  h8 ma = {1,1,1,1,1,1,1,1}, mb = {1,1,1,1,1,1,1,1};
  f16 mc = {0};
  const unsigned long long a = opA<OP>(), s = opS<OP>(), c = pack(0.0f, 0.0f);
  float acc = 96.0f;
  for (int i = 0; i < iters; ++i) {
    mc = __builtin_amdgcn_mfma_f32_32x32x16_f16(ma, mb, mc, 0, 0, 0);
    acc = fminf(acc, low(packed_op<OP, false>(a, s, c)));
  }
  if (mc[0] == -1.0f) g_sink = mc[0];
  out[lane] = acc;
}

static const char* es(hipError_t e) { return hipGetErrorString(e); }
#define CK(x) do{ hipError_t e=(x); if(e){ printf("HIP %s: %s\n",#x,es(e)); return 2; } }while(0)

template <int OP>
int launch(bool safe, bool one, int grid, int iters, int launches) {
  size_t n = one ? 64 : (size_t)grid * (WAVES / 2) * 64;
  float* d; CK(hipMalloc(&d, n * sizeof(float)));
  std::vector<float> h(n);
  long bad_total = 0; size_t seen = 0;
  for (int L = 0; L < launches; ++L) {
    CK(hipMemset(d, 0, n * sizeof(float)));
    if      (one)  single<OP><<<1, 64, 0, 0>>>(d, iters);
    else if (safe) kernel<OP, true> <<<grid, THREADS, 0, 0>>>(d, iters);
    else           kernel<OP, false><<<grid, THREADS, 0, 0>>>(d, iters);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(h.data(), d, n * sizeof(float), hipMemcpyDeviceToHost));
    long bad = 0; for (size_t i = 0; i < n; ++i) if (h[i] != 96.0f) ++bad;
    bad_total += bad; seen += n;
    printf("  launch %d: %ld / %zu wrong\n", L, bad, n);
  }
  const char* mode = one ? "SINGLE-WAVE" : (safe ? "SAFE-CONTROL" : "VULNERABLE");
  printf("%-12s : %ld / %zu victim lanes != 96.0  -> %s\n", mode, bad_total, seen,
         bad_total ? "REPRODUCED (hardware miscompute)" : "clean");
  CK(hipFree(d));
  return bad_total ? 1 : 0;
}

int main(int argc, char** argv) {
  bool safe = false, one = false; int grid = 2048, iters = 4000, launches = 6; std::string op = "mul";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if      (a == "--safe")   safe = true;
    else if (a == "--single") one  = true;
    else if (a == "--op")     op   = argv[++i];
    else if (a == "--grid")   grid = atoi(argv[++i]);
    else if (a == "--iters")  iters = atoi(argv[++i]);
    else if (a == "--launches") launches = atoi(argv[++i]);
  }
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
  printf("device: %s (%s), %d CUs | op=%s%s%s\n", p.name, p.gcnArchName,
         p.multiProcessorCount, op.c_str(), safe ? " [safe/commuted]" : "",
         one ? " [single-wave]" : "");
  if (one) { grid = 1; }
  if (op == "mul") return launch<MUL>(safe, one, grid, iters, launches);
  if (op == "add") return launch<ADD>(safe, one, grid, iters, launches);
  if (op == "fma") return launch<FMA>(safe, one, grid, iters, launches);
  printf("unknown --op %s (use mul|add|fma)\n", op.c_str());
  return 2;
}
