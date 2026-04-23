#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>

constexpr int HEAD_K_DIM = 128;
constexpr int HEAD_V_DIM = 128;
constexpr int NUM_K_HEADS = 16;
constexpr int NUM_V_HEADS = 32;

constexpr int WARP_SIZE = 64;
constexpr int NUM_WARPS = 4;
constexpr int BLOCK_THREADS = NUM_WARPS * WARP_SIZE;
constexpr int WARP_K_THREADS = 8;
constexpr int WARP_V_THREADS = WARP_SIZE / WARP_K_THREADS;
constexpr int VEC_K = 4;
constexpr int WARP_TILE_K = WARP_K_THREADS * VEC_K;
constexpr int K_ITERS = HEAD_K_DIM / WARP_TILE_K;

// BS=1 unroll experiment: -DGDN_EXPLICIT_K_UNROLL4=1 → #pragma unroll 4 on K_ITERS loops
// (default: #pragma unroll — full unroll heuristic; K_ITERS is 4 at compile time)
#if defined(GDN_EXPLICIT_K_UNROLL4) && GDN_EXPLICIT_K_UNROLL4
#define GDN_K_ITERS_UNROLL _Pragma("unroll 4")
#else
#define GDN_K_ITERS_UNROLL _Pragma("unroll")
#endif

__device__ __forceinline__ float as_float(int v) {
    union { int i; float f; } u; u.i = v; return u.f;
}
__device__ __forceinline__ int as_int(float v) {
    union { float f; int i; } u; u.f = v; return u.i;
}

// ds_swizzle Manual mode (rocwmma Ctrl::Manual<xormask,0,0x1f>): same cross-lane xor as
// __shfl_xor / ds_bpermute @ (lane^d)<<2; still LDS-backed (lgkmcnt) but often faster than ds_bpermute on CDNA3.
constexpr unsigned kDsSwizzleXor1 = (1u << 10) | 0x1fu; // 0x041f
constexpr unsigned kDsSwizzleXor2 = (2u << 10) | 0x1fu; // 0x081f
constexpr unsigned kDsSwizzleXor4 = (4u << 10) | 0x1fu; // 0x101f

template <unsigned SwPat>
__device__ __forceinline__ float ds_swizzle_xor_f32(float v) {
    union { float f; int i; } u;
    u.f = v;
    u.i = __builtin_amdgcn_ds_swizzle(u.i, SwPat);
    return u.f;
}

__device__ __forceinline__ float shfl_xor(float val, int offset) {
    int lane = threadIdx.x % WARP_SIZE;
    return as_float(__builtin_amdgcn_ds_bpermute((lane ^ offset) << 2, as_int(val)));
}
__device__ __forceinline__ float shfl_idx(float val, int src) {
    return as_float(__builtin_amdgcn_ds_bpermute(src << 2, as_int(val)));
}
__device__ __forceinline__ float reduce_sum_k(float val) {
    val += ds_swizzle_xor_f32<kDsSwizzleXor1>(val);
    val += ds_swizzle_xor_f32<kDsSwizzleXor2>(val);
    val += ds_swizzle_xor_f32<kDsSwizzleXor4>(val);
    return val;
}

// ============================================================
// K-lane xor reductions: ds_swizzle Manual xor1/xor2/xor4 (replaces ds_bpermute for fixed shuffles)
// Indexed shuffles (leader broadcast) still use ds_bpermute — see shfl_idx_*
// Signature keeps (a1,a2,a4) for call-site compatibility; xor shuffles ignore them.
// ============================================================
__device__ __forceinline__ void reduce_8way_asm(
    float& v0, float& v1, float& v2, float& v3,
    float& v4, float& v5, float& v6, float& v7,
    int a1, int a2, int a4)
{
    (void)a1;
    (void)a2;
    (void)a4;
    float t0, t1, t2, t3, t4, t5, t6, t7;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v0);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v1);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v2);
    t3 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v3);
    t4 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v4);
    t5 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v5);
    t6 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v6);
    t7 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v7);
    v0 += t0;
    v1 += t1;
    v2 += t2;
    v3 += t3;
    v4 += t4;
    v5 += t5;
    v6 += t6;
    v7 += t7;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v0);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v1);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v2);
    t3 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v3);
    t4 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v4);
    t5 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v5);
    t6 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v6);
    t7 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v7);
    v0 += t0;
    v1 += t1;
    v2 += t2;
    v3 += t3;
    v4 += t4;
    v5 += t5;
    v6 += t6;
    v7 += t7;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v0);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v1);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v2);
    t3 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v3);
    t4 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v4);
    t5 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v5);
    t6 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v6);
    t7 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v7);
    v0 += t0;
    v1 += t1;
    v2 += t2;
    v3 += t3;
    v4 += t4;
    v5 += t5;
    v6 += t6;
    v7 += t7;
}

__device__ __forceinline__ void reduce_2way_asm(
    float& v0, float& v1, int a1, int a2, int a4)
{
    (void)a1;
    (void)a2;
    (void)a4;
    v0 += ds_swizzle_xor_f32<kDsSwizzleXor1>(v0);
    v1 += ds_swizzle_xor_f32<kDsSwizzleXor1>(v1);
    v0 += ds_swizzle_xor_f32<kDsSwizzleXor2>(v0);
    v1 += ds_swizzle_xor_f32<kDsSwizzleXor2>(v1);
    v0 += ds_swizzle_xor_f32<kDsSwizzleXor4>(v0);
    v1 += ds_swizzle_xor_f32<kDsSwizzleXor4>(v1);
}

__device__ __forceinline__ void shfl_idx_4way_asm(
    float& d0, float& d1, float& d2, float& d3,
    float s0, float s1, float s2, float s3, int addr)
{
    asm volatile(
        "ds_bpermute_b32 %0, %8, %4\n"
        "ds_bpermute_b32 %1, %8, %5\n"
        "ds_bpermute_b32 %2, %8, %6\n"
        "ds_bpermute_b32 %3, %8, %7\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=&v"(d0), "=&v"(d1), "=&v"(d2), "=&v"(d3)
        : "v"(s0), "v"(s1), "v"(s2), "v"(s3), "v"(addr)
    );
}


__device__ __forceinline__ void reduce_1way_asm(float& v0, int a1, int a2, int a4)
{
    (void)a1;
    (void)a2;
    (void)a4;
    v0 += ds_swizzle_xor_f32<kDsSwizzleXor1>(v0);
    v0 += ds_swizzle_xor_f32<kDsSwizzleXor2>(v0);
    v0 += ds_swizzle_xor_f32<kDsSwizzleXor4>(v0);
}

__device__ __forceinline__ void reduce_4way_asm(
    float& v0, float& v1, float& v2, float& v3,
    int a1, int a2, int a4)
{
    (void)a1;
    (void)a2;
    (void)a4;
    float t0, t1, t2, t3;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v0);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v1);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v2);
    t3 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v3);
    v0 += t0;
    v1 += t1;
    v2 += t2;
    v3 += t3;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v0);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v1);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v2);
    t3 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v3);
    v0 += t0;
    v1 += t1;
    v2 += t2;
    v3 += t3;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v0);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v1);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v2);
    t3 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v3);
    v0 += t0;
    v1 += t1;
    v2 += t2;
    v3 += t3;
}

__device__ __forceinline__ void shfl_idx_2way_asm(
    float& d0, float& d1, float s0, float s1, int addr)
{
    asm volatile(
        "ds_bpermute_b32 %0, %4, %2\n"
        "ds_bpermute_b32 %1, %4, %3\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=&v"(d0), "=&v"(d1)
        : "v"(s0), "v"(s1), "v"(addr)
    );
}

__device__ __forceinline__ void shfl_idx_4way_bi(
    float& d0, float& d1, float& d2, float& d3,
    float s0, float s1, float s2, float s3, int addr)
{
    d0 = as_float(__builtin_amdgcn_ds_bpermute(addr, as_int(s0)));
    d1 = as_float(__builtin_amdgcn_ds_bpermute(addr, as_int(s1)));
    d2 = as_float(__builtin_amdgcn_ds_bpermute(addr, as_int(s2)));
    d3 = as_float(__builtin_amdgcn_ds_bpermute(addr, as_int(s3)));
}

__device__ __forceinline__ void shfl_idx_2way_bi(
    float& d0, float& d1, float s0, float s1, int addr)
{
    d0 = as_float(__builtin_amdgcn_ds_bpermute(addr, as_int(s0)));
    d1 = as_float(__builtin_amdgcn_ds_bpermute(addr, as_int(s1)));
}

__device__ __forceinline__ float shfl_idx_1way_bi(float s, int addr) {
    return as_float(__builtin_amdgcn_ds_bpermute(addr, as_int(s)));
}

// Fused 5-way xor reduce: dot_kq + res_hk[0..1] + res_hq[0..1] in 3 rounds (vs 6 for 1+4)
__device__ __forceinline__ void reduce_5way_asm(
    float& v0, float& v1, float& v2, float& v3, float& v4,
    int a1, int a2, int a4)
{
    (void)a1; (void)a2; (void)a4;
    float t0, t1, t2, t3, t4;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v0);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v1);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v2);
    t3 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v3);
    t4 = ds_swizzle_xor_f32<kDsSwizzleXor1>(v4);
    v0 += t0; v1 += t1; v2 += t2; v3 += t3; v4 += t4;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v0);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v1);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v2);
    t3 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v3);
    t4 = ds_swizzle_xor_f32<kDsSwizzleXor2>(v4);
    v0 += t0; v1 += t1; v2 += t2; v3 += t3; v4 += t4;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v0);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v1);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v2);
    t3 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v3);
    t4 = ds_swizzle_xor_f32<kDsSwizzleXor4>(v4);
    v0 += t0; v1 += t1; v2 += t2; v3 += t3; v4 += t4;
}

// ============================================================
// Fast math helpers using native AMD instructions
// ============================================================
__device__ __forceinline__ float fast_exp(float x) {
    // v_exp2_f32 is native on AMD, faster than v_exp_f32
    // exp(x) = exp2(x * log2(e)) = exp2(x * 1.4426950408889634)
    constexpr float LOG2E = 1.4426950408889634f;
    return exp2f(x * LOG2E);
}

__device__ __forceinline__ float fast_log1p(float x) {
    return log1pf(x);
}

// Fused reduce: 3-level butterfly (ds_swizzle xor1/xor2/xor4)
__device__ __forceinline__ void reduce_1way_fast_asm(float& v, int a1, int a2, int a4) {
    (void)a1;
    (void)a2;
    (void)a4;
    v += ds_swizzle_xor_f32<kDsSwizzleXor1>(v);
    v += ds_swizzle_xor_f32<kDsSwizzleXor2>(v);
    v += ds_swizzle_xor_f32<kDsSwizzleXor4>(v);
}

// Post-loop fused reduce for SQ1 + NVB=4 (V_ITERS=4): pack dot_kq + partial hk/hq into one 8-way + one 1-way.
// Pointers (not float(&)[4]): HIP may still type-check discarded if constexpr call sites — float[V_ITERS]
// with V_ITERS==1 must decay to float* without a reference mismatch.
__device__ __forceinline__ void reduce_postloop_viters4_fused_asm(
    float& dot_kq, float* __restrict__ res_hk, float* __restrict__ res_hq, int /*tid*/)
{
    float r0 = dot_kq;
    float r1 = res_hk[0], r2 = res_hk[1], r3 = res_hk[2], r4 = res_hk[3];
    float r5 = res_hq[0], r6 = res_hq[1], r7 = res_hq[2];
    reduce_8way_asm(r0, r1, r2, r3, r4, r5, r6, r7, 0, 0, 0);
    dot_kq = r0;
    res_hk[0] = r1;
    res_hk[1] = r2;
    res_hk[2] = r3;
    res_hk[3] = r4;
    res_hq[0] = r5;
    res_hq[1] = r6;
    res_hq[2] = r7;
    reduce_1way_fast_asm(res_hq[3], 0, 0, 0);
}

// Ultra-fast exp2 using inline ASM (single v_exp_f32 instruction after scaling)
__device__ __forceinline__ float asm_exp2(float x) {
    float r;
    asm volatile("v_exp_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}

// Ultra-fast log2 using inline ASM
__device__ __forceinline__ float asm_log2(float x) {
    float r;
    asm volatile("v_log_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}

// Ultra-fast rsqrt using inline ASM
__device__ __forceinline__ float asm_rsqrt(float x) {
    float r;
    asm volatile("v_rsq_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}

// Issue two v_rsq_f32 after L2 norm is final; overlap results with unrelated VALU.
__device__ __forceinline__ void asm_rsqrt_pair_issue(
    float sq_q_eps, float sq_k_eps, float& inv_rsq_q, float& inv_rsq_k) {
    asm volatile("v_rsq_f32 %0, %1" : "=v"(inv_rsq_q) : "v"(sq_q_eps));
    asm volatile("v_rsq_f32 %0, %1" : "=v"(inv_rsq_k) : "v"(sq_k_eps));
}

// Ultra-fast rcp using inline ASM
__device__ __forceinline__ float asm_rcp(float x) {
    float r;
    asm volatile("v_rcp_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}

// Fast exp using exp2: exp(x) = 2^(x * log2(e))
__device__ __forceinline__ float asm_fast_exp(float x) {
    constexpr float LOG2E = 1.4426950408889634f;
    return asm_exp2(x * LOG2E);
}

// Fast log1p: log(1+x) = log2(1+x) * log(2)
__device__ __forceinline__ float asm_fast_log1p(float x) {
    constexpr float LOG2 = 0.6931471805599453f;
    return asm_log2(1.0f + x) * LOG2;
}

// ── Builtin variants: same instructions but NOT volatile, so compiler can reorder freely ──
__device__ __forceinline__ float bi_exp2(float x)  { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ float bi_log2(float x)  { return __builtin_amdgcn_logf(x); }
__device__ __forceinline__ float bi_rcp(float x)   { return __builtin_amdgcn_rcpf(x); }
__device__ __forceinline__ float bi_rsqrt(float x) { return __builtin_amdgcn_rsqf(x); }
__device__ __forceinline__ float bi_fast_exp(float x) {
    return bi_exp2(x * 1.4426950408889634f);
}
__device__ __forceinline__ float bi_fast_log1p(float x) {
    return bi_log2(1.0f + x) * 0.6931471805599453f;
}

// FlyDSL-style: explicit exp2 path for gating (v_exp_f32 on log2 domain)
__device__ __forceinline__ float fast_exp_via_exp2(float x) {
    const float log2e = 1.4426950408889634f;
    return __builtin_amdgcn_exp2f(x * log2e);
}

// ln(1+x) via log2 domain (same as bi_fast_log1p; bi_log2 is AMDGPU log2 instruction)
__device__ __forceinline__ float fast_log1p_via_log2(float x) {
    const float ln2 = 0.6931471805599453f;
    return bi_log2(1.0f + x) * ln2;
}

// ============================================================
// Buffer load/store helpers using inline ASM
// On MI300X, buffer_load_dwordx4 uses SGPR-based descriptors
// with 32-bit VGPR offsets — more efficient than global_load
// which needs full 64-bit VGPR addresses per lane.
// ============================================================
typedef unsigned __attribute__((ext_vector_type(4))) buf_u32x4_t;

// GFX9 buffer descriptor flags: DST_SEL=XYZW, NUM_FORMAT=FLOAT, DATA_FORMAT=32, OOB_SELECT=3(raw)
#define BUF_RSRC_FLAGS 0x18027FAC

__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_buffer_rsrc(const void* ptr, int num_bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(ptr), (short)0, num_bytes, BUF_RSRC_FLAGS);
}

__device__ __forceinline__ void buf_load_f4(float &d0, float &d1, float &d2, float &d3,
                                            __amdgpu_buffer_rsrc_t rsrc, int byte_offset) {
    buf_u32x4_t raw = __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset, 0, 0);
    d0 = __uint_as_float(raw.x);
    d1 = __uint_as_float(raw.y);
    d2 = __uint_as_float(raw.z);
    d3 = __uint_as_float(raw.w);
}

__device__ __forceinline__ void buf_store_f4(float s0, float s1, float s2, float s3,
                                             __amdgpu_buffer_rsrc_t rsrc, int byte_offset) {
    buf_u32x4_t raw;
    raw.x = __float_as_uint(s0);
    raw.y = __float_as_uint(s1);
    raw.z = __float_as_uint(s2);
    raw.w = __float_as_uint(s3);
    __builtin_amdgcn_raw_buffer_store_b128(raw, rsrc, byte_offset, 0, 0);
}

// Non-temporal float4 store for state write-back (streaming path; bypasses L2)
__device__ __forceinline__ void store_state_f4_nt(float* __restrict__ p, float4 t) {
    typedef float v4f __attribute__((vector_size(16)));
    v4f vv = {t.x, t.y, t.z, t.w};
    __builtin_nontemporal_store(vv, reinterpret_cast<v4f*>(p));
}

// Compile-time switchable load/store helpers for state.
// Used by builtin body to opt into cache-bypass loads/stores per-kernel
// without touching call sites of the existing kernels.
template <bool NT>
__device__ __forceinline__ float4 load_state_f4_maybent(const float* __restrict__ p) {
    if constexpr (NT) {
        typedef float v4f __attribute__((vector_size(16)));
        v4f vv = __builtin_nontemporal_load(reinterpret_cast<const v4f*>(p));
        float4 r; r.x = vv[0]; r.y = vv[1]; r.z = vv[2]; r.w = vv[3];
        return r;
    } else {
        return *reinterpret_cast<const float4*>(p);
    }
}

template <bool NT>
__device__ __forceinline__ void store_state_f4_maybent(float* __restrict__ p, float4 t) {
    if constexpr (NT) {
        typedef float v4f __attribute__((vector_size(16)));
        v4f vv = {t.x, t.y, t.z, t.w};
        __builtin_nontemporal_store(vv, reinterpret_cast<v4f*>(p));
    } else {
        *reinterpret_cast<float4*>(p) = t;
    }
}

__device__ __forceinline__ float buf_load_f32(__amdgpu_buffer_rsrc_t rsrc, int byte_offset) {
    return __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(rsrc, byte_offset, 0, 0));
}

// ============================================================
// Advanced ASM: Overlapped butterfly reduce
// Issue all 3 levels of permute before waiting, then accumulate
// ============================================================
__device__ __forceinline__ void reduce_2way_overlapped_asm(
    float& v0, float& v1, int a1, int a2, int a4)
{
    float t0_1, t1_1, t0_2, t1_2, t0_3, t1_3;
    asm volatile(
        // Level 1: issue both permutes
        "ds_bpermute_b32 %0, %6, %4\n"
        "ds_bpermute_b32 %1, %6, %5\n"
        "s_waitcnt lgkmcnt(0)\n"
        "v_add_f32 %4, %4, %0\n"
        "v_add_f32 %5, %5, %1\n"
        // Level 2
        "ds_bpermute_b32 %2, %7, %4\n"
        "ds_bpermute_b32 %3, %7, %5\n"
        "s_waitcnt lgkmcnt(0)\n"
        "v_add_f32 %4, %4, %2\n"
        "v_add_f32 %5, %5, %3\n"
        // Level 3
        "ds_bpermute_b32 %0, %8, %4\n"
        "ds_bpermute_b32 %1, %8, %5\n"
        "s_waitcnt lgkmcnt(0)\n"
        "v_add_f32 %4, %4, %0\n"
        "v_add_f32 %5, %5, %1\n"
        : "=&v"(t0_1), "=&v"(t1_1), "=&v"(t0_2), "=&v"(t1_2),
          "+v"(v0), "+v"(v1)
        : "v"(a1), "v"(a2), "v"(a4)
    );
}

// ============================================================
// Advanced ASM: Explicit global_load_dwordx4 with vmcnt control
// ============================================================
__device__ __forceinline__ void asm_global_load_dwordx4(float4& dst, const float* ptr) {
    asm volatile(
        "global_load_dwordx4 %0, %1, off\n"
        : "=v"(dst)
        : "v"(ptr)
    );
}

// Wait for N global loads to complete (vmcnt counts down)
__device__ __forceinline__ void asm_waitcnt_vmcnt(int n) {
    if (n == 0) {
        asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    } else if (n == 1) {
        asm volatile("s_waitcnt vmcnt(1)\n" ::: "memory");
    } else if (n == 2) {
        asm volatile("s_waitcnt vmcnt(2)\n" ::: "memory");
    } else if (n == 3) {
        asm volatile("s_waitcnt vmcnt(3)\n" ::: "memory");
    }
}

template <int Vmcnt>
__device__ __forceinline__ void s_waitcnt_vmcnt_imm() {
    static_assert(Vmcnt >= 0 && Vmcnt <= 63, "vmcnt immediate out of range");
    asm volatile("s_waitcnt vmcnt(%0)\n" : : "n"(Vmcnt) : "memory");
}

// ============================================================
// Advanced ASM: Fused multiply-accumulate (v_fmac_f32)
// dst = dst + src0 * src1 (in-place accumulate)
// ============================================================
__device__ __forceinline__ void asm_fmac(float& dst, float src0, float src1) {
    asm volatile(
        "v_fmac_f32 %0, %1, %2\n"
        : "+v"(dst)
        : "v"(src0), "v"(src1)
    );
}

// BF16 pair dot in float: a0*b0 + a1*b1 + acc (math-equivalent to v_dot2_f32_bf16 when available).
// ROCm 7.2 / LLVM AMDGPU rejects inline "v_dot2_f32_bf16" for gfx942 (instruction is gfx950+ in this toolchain).
__device__ __forceinline__ float bf16_pair_dot_accum(uint32_t a, uint32_t b, float acc) {
    const __hip_bfloat16* pa = reinterpret_cast<const __hip_bfloat16*>(&a);
    const __hip_bfloat16* pb = reinterpret_cast<const __hip_bfloat16*>(&b);
    return __fmaf_rn(__bfloat162float(pa[0]), __bfloat162float(pb[0]),
                     __fmaf_rn(__bfloat162float(pa[1]), __bfloat162float(pb[1]), acc));
}

// ============================================================
// Advanced ASM: Dual-issue friendly compute
// Interleave independent FMAs to maximize VALU throughput
// ============================================================
template<int N>
__device__ __forceinline__ void asm_dot_interleaved(
    float (&acc)[N], const float (&a)[N], const float (&b)[N])
{
    // Issue all FMAs without dependencies between them
    #pragma unroll
    for (int i = 0; i < N; i++) {
        asm_fmac(acc[i], a[i], b[i]);
    }
}

// ============================================================
// BS=1 Hyper-optimized kernel: all advanced ASM techniques
// - Overlapped butterfly reduces
// - Explicit global_load with vmcnt control
// - Software pipelining
// - Dual-issue friendly FMA chains
// ============================================================
template <int NKH, int NVH, int NUM_V_BLOCKS>
__device__ __forceinline__ void gdn_decode_kernel_bs1_hyper_opt(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    const int pool_idx = indices[0];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int xor1 = (lane_id ^ 1) << 2;
    const int xor2 = (lane_id ^ 2) << 2;
    const int xor4 = (lane_id ^ 4) << 2;

    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;

    // ═══════════════════════════════════════════════════════════
    // PHASE 1: Issue ALL state loads (compiler generates global_load)
    // These will be in-flight while we compute gating
    // ═══════════════════════════════════════════════════════════
    float4 S_load[V_ITERS][K_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            S_load[vi][ki] = *reinterpret_cast<const float4*>(
                sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        }
    }

    // Pointers
    const auto* qb = query + hk_idx * HEAD_K_DIM;
    const auto* kb = key + hk_idx * HEAD_K_DIM;
    const auto* vb = value + hv_idx * HEAD_V_DIM;

    // ═══════════════════════════════════════════════════════════
    // PHASE 2: Gating computation (overlaps with state loads)
    // Using inline ASM transcendentals
    // ═══════════════════════════════════════════════════════════
    float a_val = __bfloat162float(a_input[hv_idx]);
    float dt_val = __bfloat162float(dt_bias[hv_idx]);
    float A_log_val = A_log[hv_idx];
    float b_val = __bfloat162float(b_input[hv_idx]);

    float x = a_val + dt_val;
    float sp = (x <= 20.0f) ? asm_fast_log1p(asm_fast_exp(x)) : x;
    float exp_g = asm_fast_exp(-asm_fast_exp(A_log_val) * sp);
    float beta = asm_rcp(1.0f + asm_fast_exp(-b_val));

    // ═══════════════════════════════════════════════════════════
    // PHASE 3: Q/K loads and L2 norms with overlapped reduce
    // ═══════════════════════════════════════════════════════════
    float Q_cache[K_ITERS][VEC_K];
    float K_cache[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;

    // Load and accumulate norms - use explicit FMA chaining
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            Q_cache[ki][j] = qv;
            K_cache[ki][j] = kv;
            // Use FMA for sq accumulation
            asm_fmac(sq_q, qv, qv);
            asm_fmac(sq_k, kv, kv);
        }
    }

    // Overlapped 2-way reduce for L2 norms
    reduce_2way_overlapped_asm(sq_q, sq_k, xor1, xor2, xor4);

    // ═══════════════════════════════════════════════════════════
    // PHASE 4: Unpack state (loads should be complete by now)
    // ═══════════════════════════════════════════════════════════
    float S[V_ITERS][K_ITERS][VEC_K];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            S[vi][ki][0] = S_load[vi][ki].x;
            S[vi][ki][1] = S_load[vi][ki].y;
            S[vi][ki][2] = S_load[vi][ki].z;
            S[vi][ki][3] = S_load[vi][ki].w;
        }
    }

    // Normalization
    const float inv_sq_q = asm_rsqrt(sq_q + 1e-6f);
    const float inv_sq_k = asm_rsqrt(sq_k + 1e-6f);
    const float qs = scale * inv_sq_q;
    const float ks = inv_sq_k;

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            Q_cache[ki][j] *= qs;
            K_cache[ki][j] *= ks;
        }
    }

    // ═══════════════════════════════════════════════════════════
    // PHASE 5: Main computation with interleaved FMAs
    // Structure to maximize VALU throughput
    // ═══════════════════════════════════════════════════════════
    float dot_kq = 0.0f;
    float res_hk[V_ITERS];
    float res_hq[V_ITERS];
    
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] = 0.0f;
        res_hq[vi] = 0.0f;
    }

    // Interleaved dot products - issue independent FMAs
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = Q_cache[ki][j];
            float kv = K_cache[ki][j];
            
            // These FMAs are independent across V_ITERS
            asm_fmac(dot_kq, kv, qv);
            
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                float sv = S[vi][ki][j];
                asm_fmac(res_hk[vi], sv, kv);
                asm_fmac(res_hq[vi], sv, qv);
            }
        }
    }

    // Reduce dot_kq
    reduce_1way_fast_asm(dot_kq, xor1, xor2, xor4);

    // Reduce res_hk and res_hq
    if constexpr (V_ITERS == 4) {
        reduce_8way_asm(res_hk[0], res_hk[1], res_hk[2], res_hk[3],
                       res_hq[0], res_hq[1], res_hq[2], res_hq[3], xor1, xor2, xor4);
    } else if constexpr (V_ITERS == 2) {
        reduce_4way_asm(res_hk[0], res_hk[1], res_hq[0], res_hq[1], xor1, xor2, xor4);
    } else {
        reduce_2way_asm(res_hk[0], res_hq[0], xor1, xor2, xor4);
    }

    // ═══════════════════════════════════════════════════════════
    // PHASE 6: Apply gating and delta
    // ═══════════════════════════════════════════════════════════
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] *= exp_g;
        res_hq[vi] *= exp_g;
    }

    float vn[V_ITERS];
    if constexpr (V_ITERS == 4) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        float tmp2 = (__bfloat162float(vb[global_v_base + 2 * WARP_GROUP_TILE_V]) - res_hk[2]) * beta;
        float tmp3 = (__bfloat162float(vb[global_v_base + 3 * WARP_GROUP_TILE_V]) - res_hk[3]) * beta;
        shfl_idx_4way_asm(vn[0], vn[1], vn[2], vn[3], tmp0, tmp1, tmp2, tmp3, leader_addr);
    } else if constexpr (V_ITERS == 2) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        shfl_idx_2way_asm(vn[0], vn[1], tmp0, tmp1, leader_addr);
    } else {
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk[0]) * beta;
        vn[0] = shfl_idx(tmp0, leader_addr >> 2);
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        asm_fmac(res_hq[vi], vn[vi], dot_kq);
    }

    // ═══════════════════════════════════════════════════════════
    // PHASE 7: Write output
    // ═══════════════════════════════════════════════════════════
    if (k_tid == 0) {
        const int64_t out_base = (int64_t)hv_idx * HEAD_V_DIM + global_v_base;
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // PHASE 8: State update with FMA chains
    // ═══════════════════════════════════════════════════════════
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t;
            // h_new = h_old * g + k * vn
            t.x = __fmaf_rn(K_cache[ki][0], vn[vi], S[vi][ki][0] * exp_g);
            t.y = __fmaf_rn(K_cache[ki][1], vn[vi], S[vi][ki][1] * exp_g);
            t.z = __fmaf_rn(K_cache[ki][2], vn[vi], S[vi][ki][2] * exp_g);
            t.w = __fmaf_rn(K_cache[ki][3], vn[vi], S[vi][ki][3] * exp_g);
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

// ============================================================
// BS=1 Extreme ASM kernel: pure inline ASM for all transcendentals
// - v_exp_f32, v_log_f32, v_rsq_f32 direct
// - Parallel load + compute overlap
// ============================================================
template <int NKH, int NVH, int NUM_V_BLOCKS>
__device__ __forceinline__ void gdn_decode_kernel_bs1_extreme_asm(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    const int pool_idx = indices[0];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int xor1 = (lane_id ^ 1) << 2;
    const int xor2 = (lane_id ^ 2) << 2;
    const int xor4 = (lane_id ^ 4) << 2;

    // State base
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;

    // Issue all loads upfront
    float4 S_load[V_ITERS][K_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            S_load[vi][ki] = *reinterpret_cast<const float4*>(
                sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        }
    }

    const auto* qb = query + hk_idx * HEAD_K_DIM;
    const auto* kb = key + hk_idx * HEAD_K_DIM;
    const auto* vb = value + hv_idx * HEAD_V_DIM;

    // Gating parameters
    float a_val = __bfloat162float(a_input[hv_idx]);
    float dt_val = __bfloat162float(dt_bias[hv_idx]);
    float A_log_val = A_log[hv_idx];
    float b_val = __bfloat162float(b_input[hv_idx]);

    // Compute gating using ASM transcendentals
    float x = a_val + dt_val;
    // softplus: sp = log(1 + exp(x)) if x <= 20, else x
    float sp = (x <= 20.0f) ? asm_fast_log1p(asm_fast_exp(x)) : x;
    // g = exp(-exp(A_log) * sp) 
    float exp_g = asm_fast_exp(-asm_fast_exp(A_log_val) * sp);
    // beta = sigmoid(b) = 1 / (1 + exp(-b))
    float beta = asm_rcp(1.0f + asm_fast_exp(-b_val));

    // Q/K loads and L2 norms
    float Q_cache[K_ITERS][VEC_K];
    float K_cache[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            Q_cache[ki][j] = qv;
            K_cache[ki][j] = kv;
            sq_q = __fmaf_rn(qv, qv, sq_q);
            sq_k = __fmaf_rn(kv, kv, sq_k);
        }
    }

    reduce_2way_asm(sq_q, sq_k, xor1, xor2, xor4);

    float inv_sq_q, inv_sq_k;
    asm_rsqrt_pair_issue(sq_q + 1e-6f, sq_k + 1e-6f, inv_sq_q, inv_sq_k);

    // Unpack state while v_rsq_f32 completes
    float S[V_ITERS][K_ITERS][VEC_K];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            S[vi][ki][0] = S_load[vi][ki].x;
            S[vi][ki][1] = S_load[vi][ki].y;
            S[vi][ki][2] = S_load[vi][ki].z;
            S[vi][ki][3] = S_load[vi][ki].w;
        }
    }

    const float qs = scale * inv_sq_q;
    const float ks = inv_sq_k;

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            Q_cache[ki][j] *= qs;
            K_cache[ki][j] *= ks;
        }
    }

    // Main computation
    float dot_kq = 0.0f;
    float res_hk[V_ITERS];
    float res_hq[V_ITERS];
    
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] = 0.0f;
        res_hq[vi] = 0.0f;
    }

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = Q_cache[ki][j];
            float kv = K_cache[ki][j];
            dot_kq = __fmaf_rn(kv, qv, dot_kq);
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] = __fmaf_rn(S[vi][ki][j], kv, res_hk[vi]);
                res_hq[vi] = __fmaf_rn(S[vi][ki][j], qv, res_hq[vi]);
            }
        }
    }

    reduce_1way_fast_asm(dot_kq, xor1, xor2, xor4);

    if constexpr (V_ITERS == 4) {
        reduce_8way_asm(res_hk[0], res_hk[1], res_hk[2], res_hk[3],
                       res_hq[0], res_hq[1], res_hq[2], res_hq[3], xor1, xor2, xor4);
    } else if constexpr (V_ITERS == 2) {
        reduce_4way_asm(res_hk[0], res_hk[1], res_hq[0], res_hq[1], xor1, xor2, xor4);
    } else {
        reduce_2way_asm(res_hk[0], res_hq[0], xor1, xor2, xor4);
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] *= exp_g;
        res_hq[vi] *= exp_g;
    }

    float vn[V_ITERS];
    if constexpr (V_ITERS == 4) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        float tmp2 = (__bfloat162float(vb[global_v_base + 2 * WARP_GROUP_TILE_V]) - res_hk[2]) * beta;
        float tmp3 = (__bfloat162float(vb[global_v_base + 3 * WARP_GROUP_TILE_V]) - res_hk[3]) * beta;
        shfl_idx_4way_asm(vn[0], vn[1], vn[2], vn[3], tmp0, tmp1, tmp2, tmp3, leader_addr);
    } else if constexpr (V_ITERS == 2) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        shfl_idx_2way_asm(vn[0], vn[1], tmp0, tmp1, leader_addr);
    } else {
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk[0]) * beta;
        vn[0] = shfl_idx(tmp0, leader_addr >> 2);
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hq[vi] = __fmaf_rn(vn[vi], dot_kq, res_hq[vi]);
    }

    if (k_tid == 0) {
        const int64_t out_base = (int64_t)hv_idx * HEAD_V_DIM + global_v_base;
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }
    }

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float kv = K_cache[ki][j];
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                S[vi][ki][j] = __fmaf_rn(kv, vn[vi], S[vi][ki][j] * exp_g);
            }
        }
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

// ============================================================
// BS=1 Ultra-optimized kernel: maximum latency hiding
// - All state loads issued first with async semantics
// - exp2 instead of exp (native instruction)
// - Fused reduce operations
// - Interleaved compute with memory ops
// ============================================================
template <int NKH, int NVH, int NUM_V_BLOCKS>
__device__ __forceinline__ void gdn_decode_kernel_bs1_ultra_opt(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    // BS=1 specialized: batch_idx always 0
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    const int pool_idx = indices[0];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int xor1 = (lane_id ^ 1) << 2;
    const int xor2 = (lane_id ^ 2) << 2;
    const int xor4 = (lane_id ^ 4) << 2;

    // ═══════════════════════════════════════════════════════════
    // PHASE 1: Issue ALL memory loads (state + Q + K + gating params)
    // ═══════════════════════════════════════════════════════════
    
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;

    // State loads - let compiler pipeline these
    float4 S_load[V_ITERS][K_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            S_load[vi][ki] = *reinterpret_cast<const float4*>(
                sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        }
    }

    // Q/K/V pointers (BS=1, so no batch offset needed)
    const auto* qb = query + hk_idx * HEAD_K_DIM;
    const auto* kb = key + hk_idx * HEAD_K_DIM;
    const auto* vb = value + hv_idx * HEAD_V_DIM;

    // Load gating parameters (only once per head)
    const float a_val = __bfloat162float(a_input[hv_idx]);
    const float dt_val = __bfloat162float(dt_bias[hv_idx]);
    const float A_log_val = A_log[hv_idx];
    const float b_val = __bfloat162float(b_input[hv_idx]);

    // ═══════════════════════════════════════════════════════════
    // PHASE 2: Compute gating using fast_exp (exp2)
    // ═══════════════════════════════════════════════════════════
    
    float x = a_val + dt_val;
    float sp = (x <= 20.0f) ? fast_log1p(fast_exp(x)) : x;
    float exp_g = fast_exp(-fast_exp(A_log_val) * sp);
    float beta = __frcp_rn(1.0f + fast_exp(-b_val));

    // ═══════════════════════════════════════════════════════════
    // PHASE 3: Load Q/K and compute L2 norms
    // ═══════════════════════════════════════════════════════════
    
    float Q_cache[K_ITERS][VEC_K];
    float K_cache[K_ITERS][VEC_K];
    float sq_q = 0.0f;
    float sq_k = 0.0f;

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            Q_cache[ki][j] = qv;
            K_cache[ki][j] = kv;
            sq_q = __fmaf_rn(qv, qv, sq_q);  // FMA for sq += qv*qv
            sq_k = __fmaf_rn(kv, kv, sq_k);
        }
    }

    // Reduce L2 norms across K threads
    reduce_2way_asm(sq_q, sq_k, xor1, xor2, xor4);

    const float sq_q_eps = sq_q + 1e-6f;
    const float sq_k_eps = sq_k + 1e-6f;
    float inv_rsq_q, inv_rsq_k;
    asm_rsqrt_pair_issue(sq_q_eps, sq_k_eps, inv_rsq_q, inv_rsq_k);

    // ═══════════════════════════════════════════════════════════
    // PHASE 4: Unpack state while v_rsq_f32 is in flight
    // ═══════════════════════════════════════════════════════════

    float S[V_ITERS][K_ITERS][VEC_K];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            S[vi][ki][0] = S_load[vi][ki].x;
            S[vi][ki][1] = S_load[vi][ki].y;
            S[vi][ki][2] = S_load[vi][ki].z;
            S[vi][ki][3] = S_load[vi][ki].w;
        }
    }

    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    // Normalize cached Q/K
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            Q_cache[ki][j] *= qs;
            K_cache[ki][j] *= ks;
        }
    }

    // ═══════════════════════════════════════════════════════════
    // PHASE 5: Main computation (fused dot products)
    // ═══════════════════════════════════════════════════════════
    
    float dot_kq = 0.0f;
    float res_hk[V_ITERS];
    float res_hq[V_ITERS];
    
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] = 0.0f;
        res_hq[vi] = 0.0f;
    }

    // Interleaved dot product computation with FMA
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = Q_cache[ki][j];
            float kv = K_cache[ki][j];
            dot_kq = __fmaf_rn(kv, qv, dot_kq);
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] = __fmaf_rn(S[vi][ki][j], kv, res_hk[vi]);
                res_hq[vi] = __fmaf_rn(S[vi][ki][j], qv, res_hq[vi]);
            }
        }
    }

    // Reduce dot_kq
    reduce_1way_fast_asm(dot_kq, xor1, xor2, xor4);

    // Reduce res_hk and res_hq
    if constexpr (V_ITERS == 4) {
        reduce_8way_asm(
            res_hk[0], res_hk[1], res_hk[2], res_hk[3],
            res_hq[0], res_hq[1], res_hq[2], res_hq[3],
            xor1, xor2, xor4);
    } else if constexpr (V_ITERS == 2) {
        reduce_4way_asm(res_hk[0], res_hk[1], res_hq[0], res_hq[1], xor1, xor2, xor4);
    } else {
        reduce_2way_asm(res_hk[0], res_hq[0], xor1, xor2, xor4);
    }

    // ═══════════════════════════════════════════════════════════
    // PHASE 6: Apply gating and compute delta
    // ═══════════════════════════════════════════════════════════
    
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] *= exp_g;
        res_hq[vi] *= exp_g;
    }

    float vn[V_ITERS];
    if constexpr (V_ITERS == 4) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        float tmp2 = (__bfloat162float(vb[global_v_base + 2 * WARP_GROUP_TILE_V]) - res_hk[2]) * beta;
        float tmp3 = (__bfloat162float(vb[global_v_base + 3 * WARP_GROUP_TILE_V]) - res_hk[3]) * beta;
        shfl_idx_4way_asm(vn[0], vn[1], vn[2], vn[3], tmp0, tmp1, tmp2, tmp3, leader_addr);
    } else if constexpr (V_ITERS == 2) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        shfl_idx_2way_asm(vn[0], vn[1], tmp0, tmp1, leader_addr);
    } else {
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk[0]) * beta;
        vn[0] = shfl_idx(tmp0, leader_addr >> 2);
    }

    // Final output computation with FMA
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hq[vi] = __fmaf_rn(vn[vi], dot_kq, res_hq[vi]);
    }

    // ═══════════════════════════════════════════════════════════
    // PHASE 7: Write output (only k_tid==0)
    // ═══════════════════════════════════════════════════════════
    
    if (k_tid == 0) {
        const int64_t out_base = (int64_t)hv_idx * HEAD_V_DIM + global_v_base;
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // PHASE 8: Update state using cached K values
    // ═══════════════════════════════════════════════════════════
    
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float kv = K_cache[ki][j];
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                S[vi][ki][j] = __fmaf_rn(kv, vn[vi], S[vi][ki][j] * exp_g);
            }
        }
    }

    // Write back state
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}


// ============================================================
// Optimized kernel body for BS=1: aggressive latency hiding
// - State loads issued early with delayed waitcnt
// - rsqrt latency hidden by overlapping gating computation
// - v_dot2_f32_bf16 for L2 norm
// - FUSE_POST_REDUCE: NVB=4 fused post-loop reduce (fewer xor rounds when enabled).
// ============================================================
template <int NKH, int NVH, int NUM_V_BLOCKS, bool VmWaitSched = false, int FIXED_BATCH_SIZE = 0,
          bool FUSE_POST_REDUCE = false>
__device__ __forceinline__ void gdn_decode_kernel_iasm_sq1_body_opt(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    const int* __restrict__            batch_perm,
    int batch_size, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if constexpr (int(FIXED_BATCH_SIZE) > 0) {
        __builtin_assume(batch_idx >= 0 && batch_idx < int(FIXED_BATCH_SIZE));
    } else {
        if (batch_idx >= batch_size) return;
    }
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;
    const int io_bid = batch_perm ? batch_perm[batch_idx] : batch_idx;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int xor1 = (lane_id ^ 1) << 2;
    const int xor2 = (lane_id ^ 2) << 2;
    const int xor4 = (lane_id ^ 4) << 2;

    // ── Phase 1: Issue ALL state loads early (no wait) ──
    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;

    // Issue loads - compiler will pipeline these
    float4 S_load[V_ITERS][K_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        GDN_K_ITERS_UNROLL
        for (int ki = 0; ki < K_ITERS; ki++) {
            S_load[vi][ki] = *reinterpret_cast<const float4*>(
                sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        }
    }

    // ── Phase 2: Compute gating (while state loads in flight) ──
    const int ab = io_bid * NVH + hv_idx;
    float a_val = __bfloat162float(a_input[ab]);
    float dt_val = __bfloat162float(dt_bias[hv_idx]);
    float x = a_val + dt_val;
    float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
    float A_log_val = A_log[hv_idx];
    float exp_g = __expf(-__expf(A_log_val) * sp);
    float b_val = __bfloat162float(b_input[ab]);
    float beta = __frcp_rn(1.0f + __expf(-b_val));

    if constexpr (VmWaitSched) {
        constexpr int N_STATE = V_ITERS * K_ITERS;
        constexpr int POST_GATE_VMCNT =
            (N_STATE >= 16) ? (N_STATE - 8) : (N_STATE >= 2 ? (N_STATE - 2) : 0);
        if constexpr (POST_GATE_VMCNT > 0)
            s_waitcnt_vmcnt_imm<POST_GATE_VMCNT>();
    }

    // ── Phase 3: Load Q/K, cache values, and compute L2 norm ──
    const auto* qb = query + (int64_t)io_bid * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key + (int64_t)io_bid * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)io_bid * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    // Cache Q/K values to avoid double bf16->f32 conversion
    float Q_cache[K_ITERS][VEC_K];
    float K_cache[K_ITERS][VEC_K];
    float sq_q = 0.0f;
    float sq_k = 0.0f;

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            Q_cache[ki][j] = qv;
            K_cache[ki][j] = kv;
            sq_q = __fmaf_rn(qv, qv, sq_q);  // FMA for sq accumulation
            sq_k = __fmaf_rn(kv, kv, sq_k);
        }
    }

    // ── Phase 4: Reduce L2 norms across warp ──
    reduce_2way_asm(sq_q, sq_k, xor1, xor2, xor4);

    float inv_rsq_q, inv_rsq_k;
    asm_rsqrt_pair_issue(sq_q + 1e-6f, sq_k + 1e-6f, inv_rsq_q, inv_rsq_k);

    if constexpr (VmWaitSched) {
        s_waitcnt_vmcnt_imm<0>();
    }

    // ── Unpack state loads while v_rsq_f32 completes ──
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        GDN_K_ITERS_UNROLL
        for (int ki = 0; ki < K_ITERS; ki++) {
            S[vi][ki][0] = S_load[vi][ki].x;
            S[vi][ki][1] = S_load[vi][ki].y;
            S[vi][ki][2] = S_load[vi][ki].z;
            S[vi][ki][3] = S_load[vi][ki].w;
        }
    }

    // ── Phase 5: Scale cached Q/K ──
    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    // Apply normalization to cached values
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            Q_cache[ki][j] *= qs;
            K_cache[ki][j] *= ks;
        }
    }

    // ── Phase 6: Main dot product loop (using cached values) ──
    float dot_kq = 0.0f;
    float res_hk[V_ITERS];
    float res_hq[V_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] = 0.0f;
        res_hq[vi] = 0.0f;
    }

    // Main dot product loop with FMA for better throughput
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = Q_cache[ki][j];
            float kv = K_cache[ki][j];
            dot_kq = __fmaf_rn(kv, qv, dot_kq);
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] = __fmaf_rn(S[vi][ki][j], kv, res_hk[vi]);
                res_hq[vi] = __fmaf_rn(S[vi][ki][j], qv, res_hq[vi]);
            }
        }
    }
    if constexpr (FUSE_POST_REDUCE && V_ITERS == 4) {
        reduce_postloop_viters4_fused_asm(dot_kq, res_hk, res_hq, tid);
    } else {
        reduce_1way_asm(dot_kq, xor1, xor2, xor4);

        if constexpr (V_ITERS == 4) {
            reduce_8way_asm(
                res_hk[0], res_hk[1], res_hk[2], res_hk[3],
                res_hq[0], res_hq[1], res_hq[2], res_hq[3],
                xor1, xor2, xor4);
        } else if constexpr (V_ITERS == 2) {
            reduce_4way_asm(res_hk[0], res_hk[1], res_hq[0], res_hq[1], xor1, xor2, xor4);
        } else {
            // V_ITERS == 1
            reduce_2way_asm(res_hk[0], res_hq[0], xor1, xor2, xor4);
        }
    }

    // ── Phase 7: Apply gating and delta update ──
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] *= exp_g;
        res_hq[vi] *= exp_g;
    }

    float vn[V_ITERS];
    if constexpr (V_ITERS == 4) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        float tmp2 = (__bfloat162float(vb[global_v_base + 2 * WARP_GROUP_TILE_V]) - res_hk[2]) * beta;
        float tmp3 = (__bfloat162float(vb[global_v_base + 3 * WARP_GROUP_TILE_V]) - res_hk[3]) * beta;
        shfl_idx_4way_asm(vn[0], vn[1], vn[2], vn[3], tmp0, tmp1, tmp2, tmp3, leader_addr);
    } else if constexpr (V_ITERS == 2) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        shfl_idx_2way_asm(vn[0], vn[1], tmp0, tmp1, leader_addr);
    } else {
        // V_ITERS == 1: use single shuffle
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk[0]) * beta;
        vn[0] = shfl_idx(tmp0, leader_addr >> 2);
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hq[vi] = __fmaf_rn(vn[vi], dot_kq, res_hq[vi]);
    }

    // ── Phase 8: Write output ──
    if (k_tid == 0) {
        const int64_t out_base = (int64_t)io_bid * NVH * HEAD_V_DIM
                               + hv_idx * HEAD_V_DIM + global_v_base;
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }
    }

    // ── Phase 9: Update and store state (using cached K values) ──
    // h_new = h_old * g + k * vn (use FMA: kv * vn + S * exp_g)
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float kv = K_cache[ki][j];  // Already normalized
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                S[vi][ki][j] = __fmaf_rn(kv, vn[vi], S[vi][ki][j] * exp_g);
            }
        }
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        GDN_K_ITERS_UNROLL
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

// Forwarding overload
template <int NKH, int NVH, int NUM_V_BLOCKS, bool VmWaitSched = false, int FIXED_BATCH_SIZE = 0,
          bool FUSE_POST_REDUCE = false>
__device__ __forceinline__ void gdn_decode_kernel_iasm_sq1_body_opt(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    gdn_decode_kernel_iasm_sq1_body_opt<NKH, NVH, NUM_V_BLOCKS, VmWaitSched, FIXED_BATCH_SIZE,
                                        FUSE_POST_REDUCE>(
        query, key, value, a_input, b_input, dt_bias, A_log, indices,
        state, output, (const int*)nullptr, batch_size, scale);
}

// ── dot2_opt: v_dot2_f32_bf16 for L2 + raw Q·K (NKH=2 NVH=8 NVB=2 path) ──
template <int NKH, int NVH, int NUM_V_BLOCKS>
__device__ __forceinline__ void gdn_decode_kernel_iasm_sq1_body_dot2_opt(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    const int* __restrict__            batch_perm,
    int batch_size, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;
    constexpr int PAIR_K = VEC_K / 2;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;
    const int io_bid = batch_perm ? batch_perm[batch_idx] : batch_idx;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int xor1 = (lane_id ^ 1) << 2;
    const int xor2 = (lane_id ^ 2) << 2;
    const int xor4 = (lane_id ^ 4) << 2;

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;

    float4 S_load[V_ITERS][K_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            S_load[vi][ki] = *reinterpret_cast<const float4*>(
                sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        }
    }

    const int ab = io_bid * NVH + hv_idx;
    float a_val = __bfloat162float(a_input[ab]);
    float dt_val = __bfloat162float(dt_bias[hv_idx]);
    float x = a_val + dt_val;
    float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
    float A_log_val = A_log[hv_idx];
    float exp_g = __expf(-__expf(A_log_val) * sp);
    float b_val = __bfloat162float(b_input[ab]);
    float beta = __frcp_rn(1.0f + __expf(-b_val));

    const auto* qb = query + (int64_t)io_bid * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key + (int64_t)io_bid * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)io_bid * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    uint32_t Qpk[K_ITERS][PAIR_K];
    uint32_t Kpk[K_ITERS][PAIR_K];
    float Q_cache[K_ITERS][VEC_K];
    float K_cache[K_ITERS][VEC_K];
    float sq_q = 0.0f;
    float sq_k = 0.0f;

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int jp = 0; jp < PAIR_K; jp++) {
            uint32_t pq = *reinterpret_cast<const uint32_t*>(qb + ko + jp * 2);
            uint32_t pk = *reinterpret_cast<const uint32_t*>(kb + ko + jp * 2);
            Qpk[ki][jp] = pq;
            Kpk[ki][jp] = pk;
            sq_q = bf16_pair_dot_accum(pq, pq, sq_q);
            sq_k = bf16_pair_dot_accum(pk, pk, sq_k);
        }
    }

    reduce_2way_asm(sq_q, sq_k, xor1, xor2, xor4);

    float inv_rsq_q, inv_rsq_k;
    asm_rsqrt_pair_issue(sq_q + 1e-6f, sq_k + 1e-6f, inv_rsq_q, inv_rsq_k);

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            S[vi][ki][0] = S_load[vi][ki].x;
            S[vi][ki][1] = S_load[vi][ki].y;
            S[vi][ki][2] = S_load[vi][ki].z;
            S[vi][ki][3] = S_load[vi][ki].w;
        }
    }

    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int jp = 0; jp < PAIR_K; jp++) {
            const __hip_bfloat16* qh = reinterpret_cast<const __hip_bfloat16*>(&Qpk[ki][jp]);
            const __hip_bfloat16* kh = reinterpret_cast<const __hip_bfloat16*>(&Kpk[ki][jp]);
            Q_cache[ki][jp * 2]     = qs * __bfloat162float(qh[0]);
            Q_cache[ki][jp * 2 + 1] = qs * __bfloat162float(qh[1]);
            K_cache[ki][jp * 2]     = ks * __bfloat162float(kh[0]);
            K_cache[ki][jp * 2 + 1] = ks * __bfloat162float(kh[1]);
        }
    }

    float res_hk[V_ITERS];
    float res_hq[V_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] = 0.0f;
        res_hq[vi] = 0.0f;
    }

    float raw_dot = 0.0f;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int jp = 0; jp < PAIR_K; jp++) {
            raw_dot = bf16_pair_dot_accum(Qpk[ki][jp], Kpk[ki][jp], raw_dot);
        }
    }
    reduce_1way_asm(raw_dot, xor1, xor2, xor4);
    const float dot_kq = qs * ks * raw_dot;

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = Q_cache[ki][j];
            float kv = K_cache[ki][j];
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] = __fmaf_rn(S[vi][ki][j], kv, res_hk[vi]);
                res_hq[vi] = __fmaf_rn(S[vi][ki][j], qv, res_hq[vi]);
            }
        }
    }

    if constexpr (V_ITERS == 4) {
        reduce_8way_asm(
            res_hk[0], res_hk[1], res_hk[2], res_hk[3],
            res_hq[0], res_hq[1], res_hq[2], res_hq[3],
            xor1, xor2, xor4);
    } else if constexpr (V_ITERS == 2) {
        reduce_4way_asm(res_hk[0], res_hk[1], res_hq[0], res_hq[1], xor1, xor2, xor4);
    } else {
        reduce_2way_asm(res_hk[0], res_hq[0], xor1, xor2, xor4);
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] *= exp_g;
        res_hq[vi] *= exp_g;
    }

    float vn[V_ITERS];
    if constexpr (V_ITERS == 4) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        float tmp2 = (__bfloat162float(vb[global_v_base + 2 * WARP_GROUP_TILE_V]) - res_hk[2]) * beta;
        float tmp3 = (__bfloat162float(vb[global_v_base + 3 * WARP_GROUP_TILE_V]) - res_hk[3]) * beta;
        shfl_idx_4way_asm(vn[0], vn[1], vn[2], vn[3], tmp0, tmp1, tmp2, tmp3, leader_addr);
    } else if constexpr (V_ITERS == 2) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        shfl_idx_2way_asm(vn[0], vn[1], tmp0, tmp1, leader_addr);
    } else {
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk[0]) * beta;
        vn[0] = shfl_idx(tmp0, leader_addr >> 2);
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hq[vi] = __fmaf_rn(vn[vi], dot_kq, res_hq[vi]);
    }

    if (k_tid == 0) {
        const int64_t out_base = (int64_t)io_bid * NVH * HEAD_V_DIM
                               + hv_idx * HEAD_V_DIM + global_v_base;
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }
    }

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float kv = K_cache[ki][j];
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                S[vi][ki][j] = __fmaf_rn(kv, vn[vi], S[vi][ki][j] * exp_g);
            }
        }
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

template <int NKH, int NVH, int NUM_V_BLOCKS>
__device__ __forceinline__ void gdn_decode_kernel_iasm_sq1_body_dot2_opt(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    gdn_decode_kernel_iasm_sq1_body_dot2_opt<NKH, NVH, NUM_V_BLOCKS>(
        query, key, value, a_input, b_input, dt_bias, A_log, indices,
        state, output, (const int*)nullptr, batch_size, scale);
}


// ============================================================
// TUNED kernel: Maximized ILP through restructured loops
// ============================================================
template <int NUM_V_BLOCKS, bool USE_QK_L2NORM>
__global__
__launch_bounds__(BLOCK_THREADS, 3)
__attribute__((amdgpu_waves_per_eu(3, 5)))
void gdn_decode_kernel_tuned(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, int seq_length, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NUM_V_HEADS;
    const int hv_idx = b_hv % NUM_V_HEADS;
    const int hk_idx = hv_idx / (NUM_V_HEADS / NUM_K_HEADS);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader = (lane_id / WARP_K_THREADS) * WARP_K_THREADS;

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NUM_V_HEADS * shs + hv_idx * shs;

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = *reinterpret_cast<const float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y; S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    const float r_A_log = A_log[hv_idx];
    const float r_dt_bias = __bfloat162float(dt_bias[hv_idx]);

    float k_reg[K_ITERS][VEC_K];

    for (int sq = 0; sq < seq_length; sq++) {
        const int ab = batch_idx * seq_length * NUM_V_HEADS + sq * NUM_V_HEADS + hv_idx;
        float x = __bfloat162float(a_input[ab]) + r_dt_bias;
        float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
        float exp_g = __expf(-__expf(r_A_log) * sp);
        float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b_input[ab])));

        const auto* qb = query + (int64_t)(batch_idx * seq_length + sq) * NUM_K_HEADS * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* kb = key + (int64_t)(batch_idx * seq_length + sq) * NUM_K_HEADS * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* vb = value + (int64_t)(batch_idx * seq_length + sq) * NUM_V_HEADS * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int ko = warp_k_vec_start + ki * WARP_TILE_K;
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                k_reg[ki][j] = __bfloat162float(kb[ko + j]);
        }

        float dot_kq = 0;
        float res_hk[V_ITERS], res_hq[V_ITERS], res_rv[V_ITERS];

        // ===== Phase A: q_reg scoped, max ILP =====
        {
            float q_reg[K_ITERS][VEC_K];
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++) {
                int ko = warp_k_vec_start + ki * WARP_TILE_K;
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    q_reg[ki][j] = __bfloat162float(qb[ko + j]);
            }

            if constexpr (USE_QK_L2NORM) {
                float sq_q = 0, sq_k = 0;
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        sq_q += q_reg[ki][j] * q_reg[ki][j];
                        sq_k += k_reg[ki][j] * k_reg[ki][j];
                    }
                sq_q = reduce_sum_k(sq_q); sq_k = reduce_sum_k(sq_k);
                float inv_rsq_q, inv_rsq_k;
                asm_rsqrt_pair_issue(sq_q + 1e-6f, sq_k + 1e-6f, inv_rsq_q, inv_rsq_k);
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++)
                    res_rv[vi] = __bfloat162float(vb[global_v_base + vi * WARP_GROUP_TILE_V]);
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++) {
                    res_hk[vi] = 0.0f;
                    res_hq[vi] = 0.0f;
                }
                float qs = scale * inv_rsq_q;
                float ks = inv_rsq_k;
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        q_reg[ki][j] *= qs;
                        k_reg[ki][j] *= ks;
                    }
            } else {
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++)
                        q_reg[ki][j] *= scale;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    dot_kq += k_reg[ki][j] * q_reg[ki][j];
            dot_kq = reduce_sum_k(dot_kq);

            if constexpr (!USE_QK_L2NORM) {
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++)
                    res_rv[vi] = __bfloat162float(vb[global_v_base + vi * WARP_GROUP_TILE_V]);
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++) {
                    res_hk[vi] = 0.0f;
                    res_hq[vi] = 0.0f;
                }
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++) {
                    float kv = k_reg[ki][j];
                    float qv = q_reg[ki][j];
                    #pragma unroll
                    for (int vi = 0; vi < V_ITERS; vi++) {
                        res_hk[vi] += S[vi][ki][j] * kv;
                        res_hq[vi] += S[vi][ki][j] * qv;
                    }
                }

            // Interleaved reduce: all values per round
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 1);
                res_hq[vi] += shfl_xor(res_hq[vi], 1);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 2);
                res_hq[vi] += shfl_xor(res_hq[vi], 2);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 4);
                res_hq[vi] += shfl_xor(res_hq[vi], 4);
            }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] *= exp_g;
                res_hq[vi] *= exp_g;
            }
        }
        // q_reg freed

        // ===== Phase B: interleaved state update =====
        // Batch all vn computations (interleaved shfl_idx)
        float vn[V_ITERS];
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            float tmp = (res_rv[vi] - res_hk[vi]) * beta;
            vn[vi] = shfl_idx(tmp, leader);
        }

        // Compute outputs, then single branch for all stores
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;

        if (k_tid == 0) {
            const int64_t out_base = (int64_t)(batch_idx * seq_length + sq) * NUM_V_HEADS * HEAD_V_DIM
                                     + hv_idx * HEAD_V_DIM + global_v_base;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }

        // Interleaved state update: ki,j outer, vi inner
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) {
                float kv = k_reg[ki][j];
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++)
                    S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
            }
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

// ============================================================
// IASM kernel: TUNED + inline ASM for reduce/shuffle
// ============================================================
template <int NUM_V_BLOCKS, bool USE_QK_L2NORM, int NKH, int NVH>
__global__
__launch_bounds__(BLOCK_THREADS, 3)
__attribute__((amdgpu_waves_per_eu(3, 5)))
void gdn_decode_kernel_iasm(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, int seq_length, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int xor1 = (lane_id ^ 1) << 2;
    const int xor2 = (lane_id ^ 2) << 2;
    const int xor4 = (lane_id ^ 4) << 2;

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = *reinterpret_cast<const float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y; S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    const float r_A_log = A_log[hv_idx];
    const float r_dt_bias = __bfloat162float(dt_bias[hv_idx]);

    float k_reg[K_ITERS][VEC_K];

    for (int sq = 0; sq < seq_length; sq++) {
        const int ab = batch_idx * seq_length * NVH + sq * NVH + hv_idx;
        float x = __bfloat162float(a_input[ab]) + r_dt_bias;
        float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
        float exp_g = __expf(-__expf(r_A_log) * sp);
        float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b_input[ab])));

        const auto* qb = query + (int64_t)(batch_idx * seq_length + sq) * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* kb = key + (int64_t)(batch_idx * seq_length + sq) * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* vb = value + (int64_t)(batch_idx * seq_length + sq) * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int ko = warp_k_vec_start + ki * WARP_TILE_K;
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                k_reg[ki][j] = __bfloat162float(kb[ko + j]);
        }

        float dot_kq = 0;
        float res_hk[V_ITERS], res_hq[V_ITERS], res_rv[V_ITERS];

        {
            float q_reg[K_ITERS][VEC_K];
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++) {
                int ko = warp_k_vec_start + ki * WARP_TILE_K;
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    q_reg[ki][j] = __bfloat162float(qb[ko + j]);
            }

            if constexpr (USE_QK_L2NORM) {
                float sq_q = 0, sq_k = 0;
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        sq_q += q_reg[ki][j] * q_reg[ki][j];
                        sq_k += k_reg[ki][j] * k_reg[ki][j];
                    }
                reduce_2way_asm(sq_q, sq_k, xor1, xor2, xor4);
                float inv_rsq_q, inv_rsq_k;
                asm_rsqrt_pair_issue(sq_q + 1e-6f, sq_k + 1e-6f, inv_rsq_q, inv_rsq_k);
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++)
                    res_rv[vi] = __bfloat162float(vb[global_v_base + vi * WARP_GROUP_TILE_V]);
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++) {
                    res_hk[vi] = 0.0f;
                    res_hq[vi] = 0.0f;
                }
                float qs = scale * inv_rsq_q;
                float ks = inv_rsq_k;
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        q_reg[ki][j] *= qs;
                        k_reg[ki][j] *= ks;
                    }
            } else {
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++)
                        q_reg[ki][j] *= scale;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    dot_kq += k_reg[ki][j] * q_reg[ki][j];
            dot_kq = reduce_sum_k(dot_kq);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                res_rv[vi] = __bfloat162float(vb[global_v_base + vi * WARP_GROUP_TILE_V]);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] = 0.0f;
                res_hq[vi] = 0.0f;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++) {
                    float kv = k_reg[ki][j];
                    float qv = q_reg[ki][j];
                    #pragma unroll
                    for (int vi = 0; vi < V_ITERS; vi++) {
                        res_hk[vi] += S[vi][ki][j] * kv;
                        res_hq[vi] += S[vi][ki][j] * qv;
                    }
                }

            if constexpr (V_ITERS == 4) {
                reduce_8way_asm(
                    res_hk[0], res_hk[1], res_hk[2], res_hk[3],
                    res_hq[0], res_hq[1], res_hq[2], res_hq[3],
                    xor1, xor2, xor4);
            } else {
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++) {
                    res_hk[vi] += shfl_xor(res_hk[vi], 1);
                    res_hq[vi] += shfl_xor(res_hq[vi], 1);
                }
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++) {
                    res_hk[vi] += shfl_xor(res_hk[vi], 2);
                    res_hq[vi] += shfl_xor(res_hq[vi], 2);
                }
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++) {
                    res_hk[vi] += shfl_xor(res_hk[vi], 4);
                    res_hq[vi] += shfl_xor(res_hq[vi], 4);
                }
            }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] *= exp_g;
                res_hq[vi] *= exp_g;
            }
        }

        float vn[V_ITERS];
        if constexpr (V_ITERS == 4) {
            float tmp0 = (res_rv[0] - res_hk[0]) * beta;
            float tmp1 = (res_rv[1] - res_hk[1]) * beta;
            float tmp2 = (res_rv[2] - res_hk[2]) * beta;
            float tmp3 = (res_rv[3] - res_hk[3]) * beta;
            shfl_idx_4way_asm(vn[0], vn[1], vn[2], vn[3],
                              tmp0, tmp1, tmp2, tmp3, leader_addr);
        } else {
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                float tmp = (res_rv[vi] - res_hk[vi]) * beta;
                vn[vi] = shfl_idx(tmp, leader_addr >> 2);
            }
        }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;

        if (k_tid == 0) {
            const int64_t out_base = (int64_t)(batch_idx * seq_length + sq) * NVH * HEAD_V_DIM
                                     + hv_idx * HEAD_V_DIM + global_v_base;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) {
                float kv = k_reg[ki][j];
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++)
                    S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
            }
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

// ============================================================
// Streaming version: reads state per-ki to reduce VGPR from 64 to ~16
// Trades register pressure for a second state read in update phase
// Target: VGPR ~100, waves/EU 5 (vs 126/4 in original)
// ============================================================
template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_iasm_sq1_body_streaming(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    const int* __restrict__            batch_perm,
    int batch_size, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv = blockIdx.x;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;
    const int io_bid = batch_perm ? batch_perm[batch_idx] : batch_idx;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int xor1 = (lane_id ^ 1) << 2;
    const int xor2 = (lane_id ^ 2) << 2;
    const int xor4 = (lane_id ^ 4) << 2;

    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;

    const int ab = io_bid * NVH + hv_idx;
    float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
    float exp_g = __expf(-__expf(A_log[hv_idx]) * sp);
    float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b_input[ab])));

    const auto* qb = query + (int64_t)io_bid * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key + (int64_t)io_bid * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)io_bid * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float sq_q = 0.0f;
    float sq_k = 0.0f;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }
    reduce_2way_asm(sq_q, sq_k, xor1, xor2, xor4);
    float inv_rsq_q, inv_rsq_k;
    asm_rsqrt_pair_issue(sq_q + 1e-6f, sq_k + 1e-6f, inv_rsq_q, inv_rsq_k);

    float dot_kq = 0.0f;
    float res_hk[V_ITERS];
    float res_hq[V_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] = 0.0f;
        res_hq[vi] = 0.0f;
    }

    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    // Forward pass: stream S per-ki instead of holding all S[4][4][4]
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        // Load S slice for this ki: S_slice[V_ITERS][VEC_K] = 16 floats
        float S_slice[V_ITERS][VEC_K];
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            int gv = global_v_base + vi * WARP_GROUP_TILE_V;
            float4 t = *reinterpret_cast<const float4*>(
                sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S_slice[vi][0] = t.x;
            S_slice[vi][1] = t.y;
            S_slice[vi][2] = t.z;
            S_slice[vi][3] = t.w;
        }
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]) * qs;
            float kv = __bfloat162float(kb[ko + j]) * ks;
            dot_kq += kv * qv;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += S_slice[vi][j] * kv;
                res_hq[vi] += S_slice[vi][j] * qv;
            }
        }
    }
    reduce_1way_asm(dot_kq, xor1, xor2, xor4);

    reduce_8way_asm(
        res_hk[0], res_hk[1], res_hk[2], res_hk[3],
        res_hq[0], res_hq[1], res_hq[2], res_hq[3],
        xor1, xor2, xor4);

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] *= exp_g;
        res_hq[vi] *= exp_g;
    }

    float vn[V_ITERS];
    {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        float tmp2 = (__bfloat162float(vb[global_v_base + 2 * WARP_GROUP_TILE_V]) - res_hk[2]) * beta;
        float tmp3 = (__bfloat162float(vb[global_v_base + 3 * WARP_GROUP_TILE_V]) - res_hk[3]) * beta;
        shfl_idx_4way_asm(vn[0], vn[1], vn[2], vn[3], tmp0, tmp1, tmp2, tmp3, leader_addr);
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;
    }

    if (k_tid == 0) {
        const int64_t out_base = (int64_t)io_bid * NVH * HEAD_V_DIM
                               + hv_idx * HEAD_V_DIM + global_v_base;
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }
    }

    // State update: re-read, update, write back per-ki (L1/L2 cached)
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            int gv = global_v_base + vi * WARP_GROUP_TILE_V;
            float4 t = *reinterpret_cast<const float4*>(
                sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) {
                float kv = __bfloat162float(kb[ko + j]) * ks;
                float s_old = (j == 0) ? t.x : (j == 1) ? t.y : (j == 2) ? t.z : t.w;
                float s_new = s_old * exp_g + kv * vn[vi];
                if (j == 0) t.x = s_new;
                else if (j == 1) t.y = s_new;
                else if (j == 2) t.z = s_new;
                else t.w = s_new;
            }
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

// NVB=4 streaming: V_ITERS=1, stream state per-ki + NT stores on write-back (BS=1 grid: batch*NVH*4)
template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_iasm_sq1_body_streaming_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    const int* __restrict__            batch_perm,
    int batch_size, float scale
) {
    constexpr int NUM_V_BLOCKS = 4;
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;
    static_assert(V_ITERS == 1, "NVB=4 streaming expects TILE_V==WARP_GROUP_TILE_V");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;
    const int io_bid = batch_perm ? batch_perm[batch_idx] : batch_idx;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int xor1 = (lane_id ^ 1) << 2;
    const int xor2 = (lane_id ^ 2) << 2;
    const int xor4 = (lane_id ^ 4) << 2;

    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;

    const auto* qb = query + (int64_t)io_bid * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key + (int64_t)io_bid * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)io_bid * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    const int ab = io_bid * NVH + hv_idx;
    float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    float sp = asm_fast_log1p(asm_fast_exp(x));
    float exp_g = asm_fast_exp(-asm_fast_exp(A_log[hv_idx]) * sp);
    float beta = asm_rcp(1.0f + asm_fast_exp(-__bfloat162float(b_input[ab])));

    float sq_q = 0.0f;
    float sq_k = 0.0f;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }
    reduce_2way_asm(sq_q, sq_k, xor1, xor2, xor4);
    float inv_rsq_q, inv_rsq_k;
    asm_rsqrt_pair_issue(sq_q + 1e-6f, sq_k + 1e-6f, inv_rsq_q, inv_rsq_k);

    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    float dot_kq = 0.0f;
    float res_hk0 = 0.0f;
    float res_hq0 = 0.0f;

    const int gv_fwd = global_v_base;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        float4 t = *reinterpret_cast<const float4*>(
            sb + gv_fwd * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        float S_slice[VEC_K] = {t.x, t.y, t.z, t.w};
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]) * qs;
            float kv = __bfloat162float(kb[ko + j]) * ks;
            dot_kq += kv * qv;
            res_hk0 += S_slice[j] * kv;
            res_hq0 += S_slice[j] * qv;
        }
    }
    reduce_1way_asm(dot_kq, xor1, xor2, xor4);
    reduce_1way_asm(res_hk0, xor1, xor2, xor4);
    reduce_1way_asm(res_hq0, xor1, xor2, xor4);

    res_hk0 *= exp_g;
    res_hq0 *= exp_g;

    float vn0;
    {
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk0) * beta;
        asm volatile(
            "ds_bpermute_b32 %0, %2, %1\n"
            "s_waitcnt lgkmcnt(0)\n"
            : "=&v"(vn0)
            : "v"(tmp0), "v"(leader_addr)
        );
    }

    res_hq0 = res_hq0 + vn0 * dot_kq;

    if (k_tid == 0) {
        const int64_t out_base = (int64_t)io_bid * NVH * HEAD_V_DIM
                               + hv_idx * HEAD_V_DIM + global_v_base;
        output[out_base] = __float2bfloat16(res_hq0);
    }

    const int gv_up = global_v_base;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        float4 t = *reinterpret_cast<const float4*>(
            sb + gv_up * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float kv = __bfloat162float(kb[ko + j]) * ks;
            float s_old = (j == 0) ? t.x : (j == 1) ? t.y : (j == 2) ? t.z : t.w;
            float s_new = s_old * exp_g + kv * vn0;
            if (j == 0) t.x = s_new;
            else if (j == 1) t.y = s_new;
            else if (j == 2) t.z = s_new;
            else t.w = s_new;
        }
        float* dst = sb + gv_up * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K;
        store_state_f4_nt(dst, t);
    }
}

template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_iasm_sq1_body_streaming_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    gdn_decode_kernel_iasm_sq1_body_streaming_nvb4<NKH, NVH>(
        query, key, value, a_input, b_input, dt_bias, A_log, indices,
        state, output, (const int*)nullptr, batch_size, scale);
}

// NVB=4 streaming + interleaved ds_swizzle (FlyDSL): issue 3 xor swizzles, run VALU while lgkm
// completes, then s_waitcnt lgkmcnt(0) before consuming results — matches ultra_body SWAP pattern.
template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_interleaved_body_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    const int* __restrict__            batch_perm,
    int batch_size, float scale
) {
    constexpr int NUM_V_BLOCKS = 4;
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;
    static_assert(V_ITERS == 1, "NVB=4 interleaved expects TILE_V==WARP_GROUP_TILE_V");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;
    const int io_bid = batch_perm ? batch_perm[batch_idx] : batch_idx;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int xor1 = (lane_id ^ 1) << 2;
    const int xor2 = (lane_id ^ 2) << 2;
    const int xor4 = (lane_id ^ 4) << 2;
    (void)xor1;
    (void)xor2;
    (void)xor4;

    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;

    const auto* qb = query + (int64_t)io_bid * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key + (int64_t)io_bid * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)io_bid * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    const int ab = io_bid * NVH + hv_idx;
    float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    float sp = asm_fast_log1p(asm_fast_exp(x));
    float exp_g = asm_fast_exp(-asm_fast_exp(A_log[hv_idx]) * sp);
    float beta = asm_rcp(1.0f + asm_fast_exp(-__bfloat162float(b_input[ab])));

    float sq_q = 0.0f;
    float sq_k = 0.0f;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }
    reduce_2way_asm(sq_q, sq_k, xor1, xor2, xor4);
    float inv_rsq_q, inv_rsq_k;
    asm_rsqrt_pair_issue(sq_q + 1e-6f, sq_k + 1e-6f, inv_rsq_q, inv_rsq_k);

    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    float dot_kq = 0.0f;
    float res_hk0 = 0.0f;
    float res_hq0 = 0.0f;

    const int gv_fwd = global_v_base;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        float4 t = *reinterpret_cast<const float4*>(
            sb + gv_fwd * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        float S_slice[VEC_K] = {t.x, t.y, t.z, t.w};
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]) * qs;
            float kv = __bfloat162float(kb[ko + j]) * ks;
            dot_kq += kv * qv;
            res_hk0 += S_slice[j] * kv;
            res_hq0 += S_slice[j] * qv;
        }
    }

    float t0, t1, t2;
    float k_pref[3][VEC_K];
    float beta_vbf;
    int64_t out_base;
    float* sb_row_base;

    // xor1: issue all three swizzles, overlap with VALU + vmem prefetch for state-update K rows
    asm volatile(
        "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,1)\n"
        "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,1)\n"
        "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,1)\n"
        : "=v"(t0), "=v"(t1), "=v"(t2)
        : "v"(dot_kq), "v"(res_hk0), "v"(res_hq0)
    );
    {
        float vbf_early = __bfloat162float(vb[global_v_base]);
        beta_vbf = vbf_early * beta;
        out_base = (int64_t)io_bid * NVH * HEAD_V_DIM
            + (int64_t)hv_idx * HEAD_V_DIM + (int64_t)global_v_base;
        sb_row_base = sb + (int64_t)gv_fwd * HEAD_K_DIM + warp_k_vec_start;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[0][j] = __bfloat162float(kb[warp_k_vec_start + j]) * ks;
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    dot_kq += t0;
    res_hk0 += t1;
    res_hq0 += t2;

    // xor2
    asm volatile(
        "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,2)\n"
        "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,2)\n"
        "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,2)\n"
        : "=v"(t0), "=v"(t1), "=v"(t2)
        : "v"(dot_kq), "v"(res_hk0), "v"(res_hq0)
    );
    {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[1][j] = __bfloat162float(kb[warp_k_vec_start + WARP_TILE_K + j]) * ks;
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    dot_kq += t0;
    res_hk0 += t1;
    res_hq0 += t2;

    // xor4
    asm volatile(
        "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,4)\n"
        "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,4)\n"
        "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,4)\n"
        : "=v"(t0), "=v"(t1), "=v"(t2)
        : "v"(dot_kq), "v"(res_hk0), "v"(res_hq0)
    );
    {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[2][j] = __bfloat162float(kb[warp_k_vec_start + 2 * WARP_TILE_K + j]) * ks;
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    dot_kq += t0;
    res_hk0 += t1;
    res_hq0 += t2;

    res_hk0 *= exp_g;
    res_hq0 *= exp_g;

    float vn0;
    {
        float tmp0 = beta_vbf - res_hk0 * beta;
        asm volatile(
            "ds_bpermute_b32 %0, %2, %1\n"
            "s_waitcnt lgkmcnt(0)\n"
            : "=&v"(vn0)
            : "v"(tmp0), "v"(leader_addr)
        );
    }

    res_hq0 = res_hq0 + vn0 * dot_kq;

    if (k_tid == 0) {
        output[out_base] = __float2bfloat16(res_hq0);
    }

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        float4 t = *reinterpret_cast<const float4*>(
            sb_row_base + ki * WARP_TILE_K);
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float kv = (ki < 3) ? k_pref[ki][j] : __bfloat162float(kb[ko + j]) * ks;
            float s_old = (j == 0) ? t.x : (j == 1) ? t.y : (j == 2) ? t.z : t.w;
            float s_new = s_old * exp_g + kv * vn0;
            if (j == 0) t.x = s_new;
            else if (j == 1) t.y = s_new;
            else if (j == 2) t.z = s_new;
            else t.w = s_new;
        }
        float* dst = sb_row_base + ki * WARP_TILE_K;
        store_state_f4_nt(dst, t);
    }
}

template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_interleaved_body_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    gdn_decode_kernel_interleaved_body_nvb4<NKH, NVH>(
        query, key, value, a_input, b_input, dt_bias, A_log, indices,
        state, output, (const int*)nullptr, batch_size, scale);
}

// Forwarding overload for streaming body: existing callers → nullptr
template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_iasm_sq1_body_streaming(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    gdn_decode_kernel_iasm_sq1_body_streaming<NKH, NVH>(
        query, key, value, a_input, b_input, dt_bias, A_log, indices,
        state, output, (const int*)nullptr, batch_size, scale);
}

// FIXED_BATCH_SIZE>0: BS is known at compile time (__builtin_assume for idx); omit runtime batch bound.
template <int NKH, int NVH, int NUM_V_BLOCKS, int FIXED_BATCH_SIZE = 0>
__device__ __forceinline__ void gdn_decode_kernel_iasm_sq1_body(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    const int* __restrict__            batch_perm,
    int batch_size, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if constexpr (FIXED_BATCH_SIZE > 0) {
        __builtin_assume(batch_idx >= 0 && batch_idx < FIXED_BATCH_SIZE);
    } else {
        if (batch_idx >= batch_size) return;
    }
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;
    const int io_bid = batch_perm ? batch_perm[batch_idx] : batch_idx;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int xor1 = (lane_id ^ 1) << 2;
    const int xor2 = (lane_id ^ 2) << 2;
    const int xor4 = (lane_id ^ 4) << 2;

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = *reinterpret_cast<const float4*>(
                sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S[vi][ki][0] = t.x;
            S[vi][ki][1] = t.y;
            S[vi][ki][2] = t.z;
            S[vi][ki][3] = t.w;
        }
    }

    const auto* qb = query + (int64_t)io_bid * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key + (int64_t)io_bid * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)io_bid * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    // OPT: Load q/k into register cache + fused L2 norm (single global read)
    float q_reg[K_ITERS][VEC_K];
    float k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f;
    float sq_k = 0.0f;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv;
            k_reg[ki][j] = kv;
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }

    // OPT: Fast ASM transcendentals for gating (branchless, native v_exp/v_log)
    const int ab = io_bid * NVH + hv_idx;
    float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    float sp = asm_fast_log1p(asm_fast_exp(x));
    float exp_g = asm_fast_exp(-asm_fast_exp(A_log[hv_idx]) * sp);
    float beta = asm_rcp(1.0f + asm_fast_exp(-__bfloat162float(b_input[ab])));

    reduce_2way_asm(sq_q, sq_k, xor1, xor2, xor4);
    float inv_rsq_q, inv_rsq_k;
    asm_rsqrt_pair_issue(sq_q + 1e-6f, sq_k + 1e-6f, inv_rsq_q, inv_rsq_k);

    // Scale q/k in register cache
    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs;
            k_reg[ki][j] *= ks;
        }
    }

    // Matvec from register cache (no second global q/k read)
    float dot_kq = 0.0f;
    float res_hk[V_ITERS];
    float res_hq[V_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] = 0.0f;
        res_hq[vi] = 0.0f;
    }

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = q_reg[ki][j];
            float kv = k_reg[ki][j];
            dot_kq += kv * qv;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += S[vi][ki][j] * kv;
                res_hq[vi] += S[vi][ki][j] * qv;
            }
        }
    }

    // OPT: Fused reduction (3 ds_swizzle rounds instead of 6)
    if constexpr (V_ITERS == 4) {
        reduce_postloop_viters4_fused_asm(dot_kq, res_hk, res_hq, tid);
    } else if constexpr (V_ITERS == 2) {
        reduce_5way_asm(dot_kq, res_hk[0], res_hk[1], res_hq[0], res_hq[1], xor1, xor2, xor4);
    } else {
        // V_ITERS == 1: fused 3-way reduce
        float t0, t1, t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] *= exp_g;
        res_hq[vi] *= exp_g;
    }

    float vn[V_ITERS];
    if constexpr (V_ITERS == 4) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        float tmp2 = (__bfloat162float(vb[global_v_base + 2 * WARP_GROUP_TILE_V]) - res_hk[2]) * beta;
        float tmp3 = (__bfloat162float(vb[global_v_base + 3 * WARP_GROUP_TILE_V]) - res_hk[3]) * beta;
        shfl_idx_4way_asm(vn[0], vn[1], vn[2], vn[3], tmp0, tmp1, tmp2, tmp3, leader_addr);
    } else if constexpr (V_ITERS >= 2) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        shfl_idx_2way_asm(vn[0], vn[1], tmp0, tmp1, leader_addr);
    } else {
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk[0]) * beta;
        asm volatile(
            "ds_bpermute_b32 %0, %2, %1\n"
            "s_waitcnt lgkmcnt(0)\n"
            : "=&v"(vn[0])
            : "v"(tmp0), "v"(leader_addr)
        );
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;
    }

    if (k_tid == 0) {
        const int64_t out_base = (int64_t)io_bid * NVH * HEAD_V_DIM
                               + hv_idx * HEAD_V_DIM + global_v_base;
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }
    }

    // OPT: State update from register cache (no global k reload)
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float kv = k_reg[ki][j];
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
            }
        }
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

// Forwarding overload: existing callers pass no batch_perm → nullptr
template <int NKH, int NVH, int NUM_V_BLOCKS, int FIXED_BATCH_SIZE = 0>
__device__ __forceinline__ void gdn_decode_kernel_iasm_sq1_body(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    gdn_decode_kernel_iasm_sq1_body<NKH, NVH, NUM_V_BLOCKS, FIXED_BATCH_SIZE>(
        query, key, value, a_input, b_input, dt_bias, A_log, indices,
        state, output, (const int*)nullptr, batch_size, scale);
}

// ============================================================
// Builtin body: uses __builtin_amdgcn_* instead of asm volatile
// for transcendentals. Compiler gets full scheduling freedom to
// overlap gating math with memory loads.
// Gating is computed BEFORE state loads to maximize overlap.
// Optional 4th template arg: fused interleaved output+state stores (NVB=4 fusestore sweep).
// ============================================================
template <int NKH, int NVH, int NUM_V_BLOCKS, bool BUILTIN_FUSESTORE = false,
          bool NT_STATE_LOAD = false, bool NT_STATE_STORE = false>
__device__ __forceinline__ void gdn_decode_kernel_builtin_body(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;
    constexpr bool FUSE_POST_STORES = BUILTIN_FUSESTORE;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    // Gating: builtins let compiler schedule these freely among loads
    const int ab = batch_idx * NVH + hv_idx;
    float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    float sp = bi_fast_log1p(bi_fast_exp(x));
    float exp_g = bi_fast_exp(-bi_fast_exp(A_log[hv_idx]) * sp);
    float beta = bi_rcp(1.0f + bi_fast_exp(-__bfloat162float(b_input[ab])));

    // State load
    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        GDN_K_ITERS_UNROLL
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = load_state_f4_maybent<NT_STATE_LOAD>(
                sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y;
            S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    // Q/K loads + fused L2 norm
    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key   + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv; k_reg[ki][j] = kv;
            sq_q += qv * qv; sq_k += kv * kv;
        }
    }

    // L2 norm reduction
    reduce_2way_asm(sq_q, sq_k,
        (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    float inv_rsq_q = bi_rsqrt(sq_q + 1e-6f);
    float inv_rsq_k = bi_rsqrt(sq_k + 1e-6f);

    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs; k_reg[ki][j] *= ks;
        }
    }

    // Matvec
    float dot_kq = 0.0f;
    float res_hk[V_ITERS], res_hq[V_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) { res_hk[vi] = 0.0f; res_hq[vi] = 0.0f; }
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = q_reg[ki][j], kv = k_reg[ki][j];
            dot_kq += kv * qv;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += S[vi][ki][j] * kv;
                res_hq[vi] += S[vi][ki][j] * qv;
            }
        }
    }

    // 3-way fused reduction
    if constexpr (V_ITERS == 4) {
        reduce_postloop_viters4_fused_asm(dot_kq, res_hk, res_hq, tid);
    } else if constexpr (V_ITERS == 2) {
        reduce_5way_asm(dot_kq, res_hk[0], res_hk[1], res_hq[0], res_hq[1],
            (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    } else {
        float t0, t1, t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] *= exp_g; res_hq[vi] *= exp_g;
    }

    float vn[V_ITERS];
    if constexpr (V_ITERS == 4) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0*WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1*WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        float tmp2 = (__bfloat162float(vb[global_v_base + 2*WARP_GROUP_TILE_V]) - res_hk[2]) * beta;
        float tmp3 = (__bfloat162float(vb[global_v_base + 3*WARP_GROUP_TILE_V]) - res_hk[3]) * beta;
        shfl_idx_4way_asm(vn[0], vn[1], vn[2], vn[3], tmp0, tmp1, tmp2, tmp3, leader_addr);
    } else if constexpr (V_ITERS >= 2) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0*WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1*WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        shfl_idx_2way_asm(vn[0], vn[1], tmp0, tmp1, leader_addr);
    } else {
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk[0]) * beta;
        asm volatile(
            "ds_bpermute_b32 %0, %2, %1\n"
            "s_waitcnt lgkmcnt(0)\n"
            : "=&v"(vn[0]) : "v"(tmp0), "v"(leader_addr));
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++)
        res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float kv = k_reg[ki][j];
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
        }
    }

    const int64_t out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM
                           + hv_idx * HEAD_V_DIM + global_v_base;

    if constexpr (FUSE_POST_STORES) {
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            if (k_tid == 0) {
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
            }
            int gv = global_v_base + vi * WARP_GROUP_TILE_V;
            GDN_K_ITERS_UNROLL
            for (int ki = 0; ki < K_ITERS; ki++) {
                float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
                store_state_f4_maybent<NT_STATE_STORE>(
                    sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K, t);
            }
        }
    } else {
        if (k_tid == 0) {
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            int gv = global_v_base + vi * WARP_GROUP_TILE_V;
            GDN_K_ITERS_UNROLL
            for (int ki = 0; ki < K_ITERS; ki++) {
                float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
                store_state_f4_maybent<NT_STATE_STORE>(
                    sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K, t);
            }
        }
    }
}

// ============================================================
// NVB=4 body: same structure as gdn_decode_kernel_builtin_body<*,*,4> (gating via bi_fast_*).
// ============================================================
template <int NKH, int NVH, bool BUILTIN_FUSESTORE = false>
__device__ __forceinline__ void gdn_decode_kernel_exp2_body_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int NUM_V_BLOCKS = 4;
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;
    constexpr bool FUSE_POST_STORES = BUILTIN_FUSESTORE;
    static_assert(V_ITERS == 1, "exp2_body_nvb4: NVB=4 fixed tile geometry uses V_ITERS==1");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int ab = batch_idx * NVH + hv_idx;
    float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    float sp = bi_fast_log1p(bi_fast_exp(x));
    float exp_g = bi_fast_exp(-bi_fast_exp(A_log[hv_idx]) * sp);
    float beta = bi_rcp(1.0f + bi_fast_exp(-__bfloat162float(b_input[ab])));

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        GDN_K_ITERS_UNROLL
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = *reinterpret_cast<const float4*>(
                sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y;
            S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key   + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv; k_reg[ki][j] = kv;
            sq_q += qv * qv; sq_k += kv * kv;
        }
    }

    reduce_2way_asm(sq_q, sq_k,
        (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    float inv_rsq_q = bi_rsqrt(sq_q + 1e-6f);
    float inv_rsq_k = bi_rsqrt(sq_k + 1e-6f);

    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs; k_reg[ki][j] *= ks;
        }
    }

    float dot_kq = 0.0f;
    float res_hk[V_ITERS], res_hq[V_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) { res_hk[vi] = 0.0f; res_hq[vi] = 0.0f; }
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = q_reg[ki][j], kv = k_reg[ki][j];
            dot_kq += kv * qv;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += S[vi][ki][j] * kv;
                res_hq[vi] += S[vi][ki][j] * qv;
            }
        }
    }

    {
        float t0, t1, t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] *= exp_g; res_hq[vi] *= exp_g;
    }

    float vn[V_ITERS];
    {
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk[0]) * beta;
        asm volatile(
            "ds_bpermute_b32 %0, %2, %1\n"
            "s_waitcnt lgkmcnt(0)\n"
            : "=&v"(vn[0]) : "v"(tmp0), "v"(leader_addr));
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++)
        res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float kv = k_reg[ki][j];
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
        }
    }

    const int64_t out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM
                           + hv_idx * HEAD_V_DIM + global_v_base;

    if constexpr (FUSE_POST_STORES) {
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            if (k_tid == 0) {
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
            }
            int gv = global_v_base + vi * WARP_GROUP_TILE_V;
            GDN_K_ITERS_UNROLL
            for (int ki = 0; ki < K_ITERS; ki++) {
                float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
                *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
            }
        }
    } else {
        if (k_tid == 0) {
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            int gv = global_v_base + vi * WARP_GROUP_TILE_V;
            GDN_K_ITERS_UNROLL
            for (int ki = 0; ki < K_ITERS; ki++) {
                float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
                *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
            }
        }
    }
}

// ============================================================
// Compact BS=1-friendly body for NVB=4:
// - Hard-specialized for V_ITERS==1 path (no vi arrays / no vi branches)
// - Keeps math equivalent to builtin_body while reducing index/control overhead
// ============================================================
template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_builtin_body_nvb4_compact(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int NUM_V_BLOCKS = 4;
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    static_assert(TILE_V == WARP_GROUP_TILE_V, "compact body expects V_ITERS==1 for NVB=4");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int ab = batch_idx * NVH + hv_idx;
    const float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    const float sp = bi_fast_log1p(bi_fast_exp(x));
    const float exp_g = bi_fast_exp(-bi_fast_exp(A_log[hv_idx]) * sp);
    const float beta = bi_rcp(1.0f + bi_fast_exp(-__bfloat162float(b_input[ab])));

    float S[K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = *reinterpret_cast<const float4*>(
            sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        S[ki][0] = t.x; S[ki][1] = t.y; S[ki][2] = t.z; S[ki][3] = t.w;
    }

    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key   + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv;
            k_reg[ki][j] = kv;
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }

    reduce_2way_asm(sq_q, sq_k, (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    float inv_rsq_q = bi_rsqrt(sq_q + 1e-6f);
    float inv_rsq_k = bi_rsqrt(sq_k + 1e-6f);
    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs;
            k_reg[ki][j] *= ks;
        }
    }

    float dot_kq = 0.0f;
    float res_hk = 0.0f;
    float res_hq = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = q_reg[ki][j];
            float kv = k_reg[ki][j];
            dot_kq += kv * qv;
            res_hk += S[ki][j] * kv;
            res_hq += S[ki][j] * qv;
        }
    }

    float t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(dot_kq);
    float t1 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hk);
    float t2 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(dot_kq);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hk);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(dot_kq);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hk);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;

    res_hk *= exp_g;
    res_hq *= exp_g;

    float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk) * beta;
    // Use as_int/as_float for proper type conversion with ds_bpermute
    float vn = as_float(__builtin_amdgcn_ds_bpermute(leader_addr, as_int(tmp0)));

    res_hq += vn * dot_kq;
    if (k_tid == 0) {
        const int64_t out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM + global_v_base;
        output[out_base] = __float2bfloat16(res_hq);
    }

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            S[ki][j] = S[ki][j] * exp_g + k_reg[ki][j] * vn;
        }
    }

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = {S[ki][0], S[ki][1], S[ki][2], S[ki][3]};
        *reinterpret_cast<float4*>(sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
    }
}

// ============================================================
// NVB=4: compact V_ITERS==1 layout + exp2/log2-domain gating + interleaved ds_swizzle
// (issue 3 xor swizzles per level, VALU overlap, then s_waitcnt) — variants 130–135.
// ============================================================
template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_exp2_compact_body_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int NUM_V_BLOCKS = 4;
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    static_assert(TILE_V == WARP_GROUP_TILE_V, "exp2_compact body expects V_ITERS==1 for NVB=4");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int ab = batch_idx * NVH + hv_idx;
    const float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    const float sp = fast_log1p_via_log2(fast_exp_via_exp2(x));
    const float exp_g = fast_exp_via_exp2(-fast_exp_via_exp2(A_log[hv_idx]) * sp);
    const float beta = bi_rcp(1.0f + fast_exp_via_exp2(-__bfloat162float(b_input[ab])));

    float S[K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = *reinterpret_cast<const float4*>(
            sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        S[ki][0] = t.x; S[ki][1] = t.y; S[ki][2] = t.z; S[ki][3] = t.w;
    }

    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key   + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv;
            k_reg[ki][j] = kv;
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }

    reduce_2way_asm(sq_q, sq_k, (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    float inv_rsq_q = bi_rsqrt(sq_q + 1e-6f);
    float inv_rsq_k = bi_rsqrt(sq_k + 1e-6f);
    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs;
            k_reg[ki][j] *= ks;
        }
    }

    float dot_kq = 0.0f;
    float res_hk = 0.0f;
    float res_hq = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = q_reg[ki][j];
            float kv = k_reg[ki][j];
            dot_kq += kv * qv;
            res_hk += S[ki][j] * kv;
            res_hq += S[ki][j] * qv;
        }
    }

    float t0, t1, t2;
    float k_pref[3][VEC_K];
    float beta_vbf;
    int64_t out_base;
    float* sb_row_base;
    const int gv_fwd = global_v_base;

    asm volatile(
        "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,1)\n"
        "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,1)\n"
        "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,1)\n"
        : "=v"(t0), "=v"(t1), "=v"(t2)
        : "v"(dot_kq), "v"(res_hk), "v"(res_hq)
    );
    {
        float vbf_early = __bfloat162float(vb[global_v_base]);
        beta_vbf = vbf_early * beta;
        out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM
            + (int64_t)hv_idx * HEAD_V_DIM + (int64_t)global_v_base;
        sb_row_base = sb + (int64_t)gv_fwd * HEAD_K_DIM + warp_k_vec_start;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[0][j] = __bfloat162float(kb[warp_k_vec_start + j]) * ks;
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    dot_kq += t0;
    res_hk += t1;
    res_hq += t2;

    asm volatile(
        "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,2)\n"
        "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,2)\n"
        "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,2)\n"
        : "=v"(t0), "=v"(t1), "=v"(t2)
        : "v"(dot_kq), "v"(res_hk), "v"(res_hq)
    );
    {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[1][j] = __bfloat162float(kb[warp_k_vec_start + WARP_TILE_K + j]) * ks;
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    dot_kq += t0;
    res_hk += t1;
    res_hq += t2;

    asm volatile(
        "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,4)\n"
        "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,4)\n"
        "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,4)\n"
        : "=v"(t0), "=v"(t1), "=v"(t2)
        : "v"(dot_kq), "v"(res_hk), "v"(res_hq)
    );
    {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[2][j] = __bfloat162float(kb[warp_k_vec_start + 2 * WARP_TILE_K + j]) * ks;
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    dot_kq += t0;
    res_hk += t1;
    res_hq += t2;

    res_hk *= exp_g;
    res_hq *= exp_g;

    float vn;
    {
        float tmp0 = beta_vbf - res_hk * beta;
        vn = as_float(__builtin_amdgcn_ds_bpermute(leader_addr, as_int(tmp0)));
    }

    res_hq += vn * dot_kq;
    if (k_tid == 0) {
        output[out_base] = __float2bfloat16(res_hq);
    }

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        float4 t = *reinterpret_cast<const float4*>(sb_row_base + ki * WARP_TILE_K);
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float kv = (ki < 3) ? k_pref[ki][j] : __bfloat162float(kb[ko + j]) * ks;
            float s_old = (j == 0) ? t.x : (j == 1) ? t.y : (j == 2) ? t.z : t.w;
            float s_new = s_old * exp_g + kv * vn;
            if (j == 0) t.x = s_new;
            else if (j == 1) t.y = s_new;
            else if (j == 2) t.z = s_new;
            else t.w = s_new;
        }
        float* dst = sb_row_base + ki * WARP_TILE_K;
        store_state_f4_nt(dst, t);
    }
}

// ============================================================
// NVB=4: same compact layout + exp2 gating as exp2_compact, but ds_swizzle via
// __builtin_amdgcn_ds_swizzle (ds_swizzle_xor_f32) and no explicit s_waitcnt —
// compiler schedules lgkm waits (variants 150–155).
// ============================================================
template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_nowait_body_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int NUM_V_BLOCKS = 4;
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    static_assert(TILE_V == WARP_GROUP_TILE_V, "nowait_body_nvb4 expects V_ITERS==1 for NVB=4");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int ab = batch_idx * NVH + hv_idx;
    const float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    const float sp = fast_log1p_via_log2(fast_exp_via_exp2(x));
    const float exp_g = fast_exp_via_exp2(-fast_exp_via_exp2(A_log[hv_idx]) * sp);
    const float beta = bi_rcp(1.0f + fast_exp_via_exp2(-__bfloat162float(b_input[ab])));

    float S[K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = *reinterpret_cast<const float4*>(
            sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        S[ki][0] = t.x; S[ki][1] = t.y; S[ki][2] = t.z; S[ki][3] = t.w;
    }

    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key   + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv;
            k_reg[ki][j] = kv;
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }

    reduce_2way_asm(sq_q, sq_k, (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    float inv_rsq_q = bi_rsqrt(sq_q + 1e-6f);
    float inv_rsq_k = bi_rsqrt(sq_k + 1e-6f);
    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs;
            k_reg[ki][j] *= ks;
        }
    }

    float dot_kq = 0.0f;
    float res_hk = 0.0f;
    float res_hq = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = q_reg[ki][j];
            float kv = k_reg[ki][j];
            dot_kq += kv * qv;
            res_hk += S[ki][j] * kv;
            res_hq += S[ki][j] * qv;
        }
    }

    float t0, t1, t2;
    float k_pref[3][VEC_K];
    float beta_vbf;
    int64_t out_base;
    float* sb_row_base;
    const int gv_fwd = global_v_base;

    t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(dot_kq);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hk);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hq);
    {
        float vbf_early = __bfloat162float(vb[global_v_base]);
        beta_vbf = vbf_early * beta;
        out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM
            + (int64_t)hv_idx * HEAD_V_DIM + (int64_t)global_v_base;
        sb_row_base = sb + (int64_t)gv_fwd * HEAD_K_DIM + warp_k_vec_start;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[0][j] = __bfloat162float(kb[warp_k_vec_start + j]) * ks;
    }
    dot_kq += t0;
    res_hk += t1;
    res_hq += t2;

    t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(dot_kq);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hk);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hq);
    {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[1][j] = __bfloat162float(kb[warp_k_vec_start + WARP_TILE_K + j]) * ks;
    }
    dot_kq += t0;
    res_hk += t1;
    res_hq += t2;

    t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(dot_kq);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hk);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hq);
    {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[2][j] = __bfloat162float(kb[warp_k_vec_start + 2 * WARP_TILE_K + j]) * ks;
    }
    dot_kq += t0;
    res_hk += t1;
    res_hq += t2;

    res_hk *= exp_g;
    res_hq *= exp_g;

    float vn;
    {
        float tmp0 = beta_vbf - res_hk * beta;
        vn = as_float(__builtin_amdgcn_ds_bpermute(leader_addr, as_int(tmp0)));
    }

    res_hq += vn * dot_kq;
    if (k_tid == 0) {
        output[out_base] = __float2bfloat16(res_hq);
    }

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        float4 t = *reinterpret_cast<const float4*>(sb_row_base + ki * WARP_TILE_K);
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float kv = (ki < 3) ? k_pref[ki][j] : __bfloat162float(kb[ko + j]) * ks;
            float s_old = (j == 0) ? t.x : (j == 1) ? t.y : (j == 2) ? t.z : t.w;
            float s_new = s_old * exp_g + kv * vn;
            if (j == 0) t.x = s_new;
            else if (j == 1) t.y = s_new;
            else if (j == 2) t.z = s_new;
            else t.w = s_new;
        }
        float* dst = sb_row_base + ki * WARP_TILE_K;
        store_state_f4_nt(dst, t);
    }
}

// Packed f32 (float2) dot/matvec: v_pk_mul_f32 / v_pk_add_f32 for k·q, S·k, S·q.
__device__ __forceinline__ void gdn_pk_mul_f32(float2& dst, float2 a, float2 b) {
    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(dst) : "v"(a), "v"(b));
}
__device__ __forceinline__ void gdn_pk_add_f32(float2& dst, float2 a, float2 b) {
    asm volatile("v_pk_add_f32 %0, %1, %2" : "=v"(dst) : "v"(a), "v"(b));
}

__device__ __forceinline__ void pk_fma_f32(float2& acc, float2 a, float2 b) {
    // v_pk_fma_f32 acc, a, b, acc  (acc = acc + a * b)
    asm volatile(
        "v_pk_fma_f32 %0, %1, %2, %0"
        : "+v"(acc)
        : "v"(a), "v"(b)
    );
}

// Pair (res_hk, res_hq): ds_swizzle per lane + v_pk_add_f32 warp xor-sum (variants 160–165).
template <unsigned SwPat>
__device__ __forceinline__ float2 ds_swizzle_xor_f32_float2(float2 v) {
    union { float2 f; uint2 u; } in;
    in.f = v;
    union { float2 f; uint2 u; } out;
    out.u.x = __builtin_amdgcn_ds_swizzle(in.u.x, SwPat);
    out.u.y = __builtin_amdgcn_ds_swizzle(in.u.y, SwPat);
    return out.f;
}

__device__ __forceinline__ void reduce_xor_warp_sum_f32_float2_pk(float2& v) {
    float2 t = ds_swizzle_xor_f32_float2<kDsSwizzleXor1>(v);
    gdn_pk_add_f32(v, v, t);
    t = ds_swizzle_xor_f32_float2<kDsSwizzleXor2>(v);
    gdn_pk_add_f32(v, v, t);
    t = ds_swizzle_xor_f32_float2<kDsSwizzleXor4>(v);
    gdn_pk_add_f32(v, v, t);
}

// ============================================================
// NVB=4 exp2 gating + fused K-reduction: dot_kq (3x ds_swizzle) + (res_hk,res_hq) pair (pk add).
// ============================================================
template <int NKH, int NVH, bool BUILTIN_FUSESTORE = false>
__device__ __forceinline__ void gdn_decode_kernel_fastreduce_body_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int NUM_V_BLOCKS = 4;
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;
    constexpr bool FUSE_POST_STORES = BUILTIN_FUSESTORE;
    static_assert(V_ITERS == 1, "fastreduce_body_nvb4: NVB=4 fixed tile geometry uses V_ITERS==1");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int ab = batch_idx * NVH + hv_idx;
    float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    float sp = bi_fast_log1p(bi_fast_exp(x));
    float exp_g = bi_fast_exp(-bi_fast_exp(A_log[hv_idx]) * sp);
    float beta = bi_rcp(1.0f + bi_fast_exp(-__bfloat162float(b_input[ab])));

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        GDN_K_ITERS_UNROLL
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = *reinterpret_cast<const float4*>(
                sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y;
            S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key   + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv; k_reg[ki][j] = kv;
            sq_q += qv * qv; sq_k += kv * kv;
        }
    }

    reduce_2way_asm(sq_q, sq_k,
        (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    float inv_rsq_q = bi_rsqrt(sq_q + 1e-6f);
    float inv_rsq_k = bi_rsqrt(sq_k + 1e-6f);

    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs; k_reg[ki][j] *= ks;
        }
    }

    float dot_kq = 0.0f;
    float res_hk[V_ITERS], res_hq[V_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) { res_hk[vi] = 0.0f; res_hq[vi] = 0.0f; }
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = q_reg[ki][j], kv = k_reg[ki][j];
            dot_kq += kv * qv;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += S[vi][ki][j] * kv;
                res_hq[vi] += S[vi][ki][j] * qv;
            }
        }
    }

    {
        float t0;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(dot_kq);
        dot_kq += t0;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(dot_kq);
        dot_kq += t0;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(dot_kq);
        dot_kq += t0;

        float2 res_h = {res_hk[0], res_hq[0]};
        reduce_xor_warp_sum_f32_float2_pk(res_h);
        res_hk[0] = res_h.x;
        res_hq[0] = res_h.y;
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] *= exp_g; res_hq[vi] *= exp_g;
    }

    float vn[V_ITERS];
    {
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk[0]) * beta;
        asm volatile(
            "ds_bpermute_b32 %0, %2, %1\n"
            "s_waitcnt lgkmcnt(0)\n"
            : "=&v"(vn[0]) : "v"(tmp0), "v"(leader_addr));
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++)
        res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float kv = k_reg[ki][j];
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
        }
    }

    const int64_t out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM
                           + hv_idx * HEAD_V_DIM + global_v_base;

    if constexpr (FUSE_POST_STORES) {
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            if (k_tid == 0) {
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
            }
            int gv = global_v_base + vi * WARP_GROUP_TILE_V;
            GDN_K_ITERS_UNROLL
            for (int ki = 0; ki < K_ITERS; ki++) {
                float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
                *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
            }
        }
    } else {
        if (k_tid == 0) {
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            int gv = global_v_base + vi * WARP_GROUP_TILE_V;
            GDN_K_ITERS_UNROLL
            for (int ki = 0; ki < K_ITERS; ki++) {
                float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
                *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
            }
        }
    }
}

// ============================================================
// NVB=4 compact path with packed f32 dot products (V_ITERS==1).
// ============================================================
template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_packed_body_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int NUM_V_BLOCKS = 4;
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    static_assert(TILE_V == WARP_GROUP_TILE_V, "packed nvb4 body expects V_ITERS==1 for NVB=4");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int ab = batch_idx * NVH + hv_idx;
    const float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    const float sp = bi_fast_log1p(bi_fast_exp(x));
    const float exp_g = bi_fast_exp(-bi_fast_exp(A_log[hv_idx]) * sp);
    const float beta = bi_rcp(1.0f + bi_fast_exp(-__bfloat162float(b_input[ab])));

    float S[K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = *reinterpret_cast<const float4*>(
            sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        S[ki][0] = t.x; S[ki][1] = t.y; S[ki][2] = t.z; S[ki][3] = t.w;
    }

    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key   + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv;
            k_reg[ki][j] = kv;
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }

    reduce_2way_asm(sq_q, sq_k, (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    float inv_rsq_q = bi_rsqrt(sq_q + 1e-6f);
    float inv_rsq_k = bi_rsqrt(sq_k + 1e-6f);
    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs;
            k_reg[ki][j] *= ks;
        }
    }

    float2 dot_kq_pk = {0.f, 0.f};
    float2 res_hk_pk = {0.f, 0.f};
    float2 res_hq_pk = {0.f, 0.f};
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float2 q01 = {q_reg[ki][0], q_reg[ki][1]};
        float2 q23 = {q_reg[ki][2], q_reg[ki][3]};
        float2 k01 = {k_reg[ki][0], k_reg[ki][1]};
        float2 k23 = {k_reg[ki][2], k_reg[ki][3]};
        float2 S01 = {S[ki][0], S[ki][1]};
        float2 S23 = {S[ki][2], S[ki][3]};
        float2 t_kq0, t_kq1, t_hk0, t_hk1, t_hq0, t_hq1;
        gdn_pk_mul_f32(t_kq0, k01, q01);
        gdn_pk_mul_f32(t_kq1, k23, q23);
        gdn_pk_mul_f32(t_hk0, S01, k01);
        gdn_pk_mul_f32(t_hk1, S23, k23);
        gdn_pk_mul_f32(t_hq0, S01, q01);
        gdn_pk_mul_f32(t_hq1, S23, q23);
        gdn_pk_add_f32(dot_kq_pk, dot_kq_pk, t_kq0);
        gdn_pk_add_f32(dot_kq_pk, dot_kq_pk, t_kq1);
        gdn_pk_add_f32(res_hk_pk, res_hk_pk, t_hk0);
        gdn_pk_add_f32(res_hk_pk, res_hk_pk, t_hk1);
        gdn_pk_add_f32(res_hq_pk, res_hq_pk, t_hq0);
        gdn_pk_add_f32(res_hq_pk, res_hq_pk, t_hq1);
    }

    float dot_kq = dot_kq_pk.x + dot_kq_pk.y;
    float res_hk = res_hk_pk.x + res_hk_pk.y;
    float res_hq = res_hq_pk.x + res_hq_pk.y;

    float t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(dot_kq);
    float t1 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hk);
    float t2 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(dot_kq);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hk);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(dot_kq);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hk);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;

    res_hk *= exp_g;
    res_hq *= exp_g;

    float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk) * beta;
    float vn = as_float(__builtin_amdgcn_ds_bpermute(leader_addr, as_int(tmp0)));

    res_hq += vn * dot_kq;
    if (k_tid == 0) {
        const int64_t out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM + global_v_base;
        output[out_base] = __float2bfloat16(res_hq);
    }

    const float2 vn2 = {vn, vn};
    const float2 eg2 = {exp_g, exp_g};
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float2 k01 = {k_reg[ki][0], k_reg[ki][1]};
        float2 k23 = {k_reg[ki][2], k_reg[ki][3]};
        float2 S01 = {S[ki][0], S[ki][1]};
        float2 S23 = {S[ki][2], S[ki][3]};
        float2 s_exp0, s_exp1, kv0, kv1, out0, out1;
        gdn_pk_mul_f32(s_exp0, S01, eg2);
        gdn_pk_mul_f32(s_exp1, S23, eg2);
        gdn_pk_mul_f32(kv0, k01, vn2);
        gdn_pk_mul_f32(kv1, k23, vn2);
        gdn_pk_add_f32(out0, s_exp0, kv0);
        gdn_pk_add_f32(out1, s_exp1, kv1);
        S[ki][0] = out0.x; S[ki][1] = out0.y;
        S[ki][2] = out1.x; S[ki][3] = out1.y;
    }

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = {S[ki][0], S[ki][1], S[ki][2], S[ki][3]};
        *reinterpret_cast<float4*>(sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
    }
}

// ============================================================
// NVB=4: exp2 gating + v_pk_fma_f32 dot/matvec + single-level xor swizzle (140–145).
// ============================================================
template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_pkfma_body_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int NUM_V_BLOCKS = 4;
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    static_assert(TILE_V == WARP_GROUP_TILE_V, "pkfma nvb4 body expects V_ITERS==1 for NVB=4");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int ab = batch_idx * NVH + hv_idx;
    const float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    const float sp = fast_log1p_via_log2(fast_exp_via_exp2(x));
    const float exp_g = fast_exp_via_exp2(-fast_exp_via_exp2(A_log[hv_idx]) * sp);
    const float beta = bi_rcp(1.0f + fast_exp_via_exp2(-__bfloat162float(b_input[ab])));

    float2 S01[K_ITERS], S23[K_ITERS];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = *reinterpret_cast<const float4*>(
            sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        S01[ki] = float2{t.x, t.y};
        S23[ki] = float2{t.z, t.w};
    }

    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key   + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv;
            k_reg[ki][j] = kv;
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }

    reduce_2way_asm(sq_q, sq_k, (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    float inv_rsq_q = bi_rsqrt(sq_q + 1e-6f);
    float inv_rsq_k = bi_rsqrt(sq_k + 1e-6f);
    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs;
            k_reg[ki][j] *= ks;
        }
    }

    float2 dot_kq_pk = {0.f, 0.f};
    float2 res_hk_pk = {0.f, 0.f};
    float2 res_hq_pk = {0.f, 0.f};
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float2 q01 = {q_reg[ki][0], q_reg[ki][1]};
        float2 q23 = {q_reg[ki][2], q_reg[ki][3]};
        float2 k01 = {k_reg[ki][0], k_reg[ki][1]};
        float2 k23 = {k_reg[ki][2], k_reg[ki][3]};
        pk_fma_f32(dot_kq_pk, k01, q01);
        pk_fma_f32(dot_kq_pk, k23, q23);
        pk_fma_f32(res_hk_pk, S01[ki], k01);
        pk_fma_f32(res_hk_pk, S23[ki], k23);
        pk_fma_f32(res_hq_pk, S01[ki], q01);
        pk_fma_f32(res_hq_pk, S23[ki], q23);
    }

    float dot_kq = dot_kq_pk.x + dot_kq_pk.y;
    float res_hk = res_hk_pk.x + res_hk_pk.y;
    float res_hq = res_hq_pk.x + res_hq_pk.y;

    float t0, t1, t2;
    float k_pref[3][VEC_K];
    float beta_vbf;
    int64_t out_base;
    float* sb_row_base;
    const int gv_fwd = global_v_base;

    asm volatile(
        "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,1)\n"
        "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,1)\n"
        "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,1)\n"
        : "=v"(t0), "=v"(t1), "=v"(t2)
        : "v"(dot_kq), "v"(res_hk), "v"(res_hq)
    );
    {
        float vbf_early = __bfloat162float(vb[global_v_base]);
        beta_vbf = vbf_early * beta;
        out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM
            + (int64_t)hv_idx * HEAD_V_DIM + (int64_t)global_v_base;
        sb_row_base = sb + (int64_t)gv_fwd * HEAD_K_DIM + warp_k_vec_start;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[0][j] = __bfloat162float(kb[warp_k_vec_start + j]) * ks;
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    dot_kq += t0;
    res_hk += t1;
    res_hq += t2;

    asm volatile(
        "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,2)\n"
        "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,2)\n"
        "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,2)\n"
        : "=v"(t0), "=v"(t1), "=v"(t2)
        : "v"(dot_kq), "v"(res_hk), "v"(res_hq)
    );
    {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[1][j] = __bfloat162float(kb[warp_k_vec_start + WARP_TILE_K + j]) * ks;
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    dot_kq += t0;
    res_hk += t1;
    res_hq += t2;

    asm volatile(
        "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,4)\n"
        "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,4)\n"
        "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,4)\n"
        : "=v"(t0), "=v"(t1), "=v"(t2)
        : "v"(dot_kq), "v"(res_hk), "v"(res_hq)
    );
    {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[2][j] = __bfloat162float(kb[warp_k_vec_start + 2 * WARP_TILE_K + j]) * ks;
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    dot_kq += t0;
    res_hk += t1;
    res_hq += t2;

    res_hk *= exp_g;
    res_hq *= exp_g;

    float vn;
    {
        float tmp0 = beta_vbf - res_hk * beta;
        vn = as_float(__builtin_amdgcn_ds_bpermute(leader_addr, as_int(tmp0)));
    }

    res_hq += vn * dot_kq;
    if (k_tid == 0) {
        output[out_base] = __float2bfloat16(res_hq);
    }

    const float2 vn2 = {vn, vn};
    const float2 eg2 = {exp_g, exp_g};
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        float4 t = *reinterpret_cast<const float4*>(sb_row_base + ki * WARP_TILE_K);
        float2 t01 = {t.x, t.y};
        float2 t23 = {t.z, t.w};
        float2 kv01, kv23;
        if (ki < 3) {
            kv01 = {k_pref[ki][0], k_pref[ki][1]};
            kv23 = {k_pref[ki][2], k_pref[ki][3]};
        } else {
            kv01 = {__bfloat162float(kb[ko + 0]) * ks, __bfloat162float(kb[ko + 1]) * ks};
            kv23 = {__bfloat162float(kb[ko + 2]) * ks, __bfloat162float(kb[ko + 3]) * ks};
        }
        float2 o01 = {0.f, 0.f};
        float2 o23 = {0.f, 0.f};
        pk_fma_f32(o01, t01, eg2);
        pk_fma_f32(o23, t23, eg2);
        pk_fma_f32(o01, kv01, vn2);
        pk_fma_f32(o23, kv23, vn2);
        t = {o01.x, o01.y, o23.x, o23.y};
        float* dst = sb_row_base + ki * WARP_TILE_K;
        store_state_f4_nt(dst, t);
    }
}

// ============================================================
// NVB=4: FlyDSL-style hand-ASM hot path (GDN_NVH8_NVB4_VARIANT 200–205).
// - Gating: v_exp_f32 / v_log_f32 / v_rcp_f32 on log2 domain (matches fast_*_via_exp2 path)
// - Dot / matvec: unrolled v_pk_fma_f32 chains
// - K-lane xor: ds_swizzle_b32 + interleaved VALU (C-issued loads) + s_waitcnt lgkmcnt(0)
// - Post: v_mul / ds_bpermute / v_fmac; state rows use v_pk_fma_f32 + NT store
// ============================================================
template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_pureasm_body_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int NUM_V_BLOCKS = 4;
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    static_assert(TILE_V == WARP_GROUP_TILE_V, "pureasm nvb4 body expects V_ITERS==1 for NVB=4");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int ab = batch_idx * NVH + hv_idx;
    // Gating: explicit v_exp_f32 / v_log_f32 path via asm_exp2 / asm_log2 / asm_rcp (matches pkfma numerics).
    constexpr float kLog2e = 1.4426950408889634f;
    constexpr float kLn2 = 0.6931471805599453f;
    const float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    const float ex = asm_exp2(x * kLog2e);
    const float sp = asm_log2(1.0f + ex) * kLn2;
    const float e_al = asm_exp2(A_log[hv_idx] * kLog2e);
    const float exp_g = asm_exp2(-(e_al * sp * kLog2e));
    const float beta = asm_rcp(1.0f + asm_exp2(-__bfloat162float(b_input[ab]) * kLog2e));

    float2 S01[K_ITERS], S23[K_ITERS];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = *reinterpret_cast<const float4*>(
            sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        S01[ki] = float2{t.x, t.y};
        S23[ki] = float2{t.z, t.w};
    }

    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key   + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv;
            k_reg[ki][j] = kv;
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }

    reduce_2way_asm(sq_q, sq_k, (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    float inv_rsq_q = bi_rsqrt(sq_q + 1e-6f);
    float inv_rsq_k = bi_rsqrt(sq_k + 1e-6f);
    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs;
            k_reg[ki][j] *= ks;
        }
    }

    float2 dot_kq_pk = {0.f, 0.f};
    float2 res_hk_pk = {0.f, 0.f};
    float2 res_hq_pk = {0.f, 0.f};

    // Unrolled v_pk_fma_f32 matvec/dot (K_ITERS==4)
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        float2 q01 = {q_reg[ki][0], q_reg[ki][1]};
        float2 q23 = {q_reg[ki][2], q_reg[ki][3]};
        float2 k01 = {k_reg[ki][0], k_reg[ki][1]};
        float2 k23 = {k_reg[ki][2], k_reg[ki][3]};
        asm volatile(
            "v_pk_fma_f32 %0, %3, %5, %0\n"
            "v_pk_fma_f32 %0, %4, %6, %0\n"
            "v_pk_fma_f32 %1, %7, %3, %1\n"
            "v_pk_fma_f32 %1, %8, %4, %1\n"
            "v_pk_fma_f32 %2, %7, %5, %2\n"
            "v_pk_fma_f32 %2, %8, %6, %2\n"
            : "+v"(dot_kq_pk), "+v"(res_hk_pk), "+v"(res_hq_pk)
            : "v"(k01), "v"(k23), "v"(q01), "v"(q23), "v"(S01[ki]), "v"(S23[ki]));
    }

    float dot_kq, res_hk, res_hq;
    asm volatile(
        "v_add_f32 %0, %3, %4\n"
        "v_add_f32 %1, %5, %6\n"
        "v_add_f32 %2, %7, %8\n"
        : "=&v"(dot_kq), "=&v"(res_hk), "=&v"(res_hq)
        : "v"(dot_kq_pk.x), "v"(dot_kq_pk.y), "v"(res_hk_pk.x), "v"(res_hk_pk.y),
          "v"(res_hq_pk.x), "v"(res_hq_pk.y));

    float sw0, sw1, sw2;
    float k_pref[3][VEC_K];
    float beta_vbf;
    int64_t out_base;
    float* sb_row_base;
    const int gv_fwd = global_v_base;

    asm volatile(
        "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,1)\n"
        "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,1)\n"
        "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,1)\n"
        : "=v"(sw0), "=v"(sw1), "=v"(sw2)
        : "v"(dot_kq), "v"(res_hk), "v"(res_hq));
    {
        float vbf_early = __bfloat162float(vb[global_v_base]);
        beta_vbf = vbf_early * beta;
        out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM
            + (int64_t)hv_idx * HEAD_V_DIM + (int64_t)global_v_base;
        sb_row_base = sb + (int64_t)gv_fwd * HEAD_K_DIM + warp_k_vec_start;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[0][j] = __bfloat162float(kb[warp_k_vec_start + j]) * ks;
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    dot_kq += sw0;
    res_hk += sw1;
    res_hq += sw2;

    asm volatile(
        "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,2)\n"
        "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,2)\n"
        "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,2)\n"
        : "=v"(sw0), "=v"(sw1), "=v"(sw2)
        : "v"(dot_kq), "v"(res_hk), "v"(res_hq));
    {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[1][j] = __bfloat162float(kb[warp_k_vec_start + WARP_TILE_K + j]) * ks;
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    dot_kq += sw0;
    res_hk += sw1;
    res_hq += sw2;

    asm volatile(
        "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,4)\n"
        "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,4)\n"
        "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,4)\n"
        : "=v"(sw0), "=v"(sw1), "=v"(sw2)
        : "v"(dot_kq), "v"(res_hk), "v"(res_hq));
    {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++)
            k_pref[2][j] = __bfloat162float(kb[warp_k_vec_start + 2 * WARP_TILE_K + j]) * ks;
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    dot_kq += sw0;
    res_hk += sw1;
    res_hq += sw2;

    asm volatile(
        "v_mul_f32 %0, %0, %2\n"
        "v_mul_f32 %1, %1, %2\n"
        : "+v"(res_hk), "+v"(res_hq) : "v"(exp_g));

    float vn;
    {
        float tmp0;
        asm volatile(
            "v_mul_f32 %0, %1, %2\n"
            "v_sub_f32 %0, %3, %0\n"
            : "=&v"(tmp0) : "v"(res_hk), "v"(beta), "v"(beta_vbf));
        asm volatile(
            "ds_bpermute_b32 %0, %2, %1\n"
            "s_waitcnt lgkmcnt(0)\n"
            : "=&v"(vn) : "v"(tmp0), "v"(leader_addr));
    }

    asm volatile("v_fmac_f32 %0, %1, %2\n" : "+v"(res_hq) : "v"(vn), "v"(dot_kq));

    if (k_tid == 0) {
        output[out_base] = __float2bfloat16(res_hq);
    }

    const float2 vn2 = {vn, vn};
    const float2 eg2 = {exp_g, exp_g};
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        float4 t = *reinterpret_cast<const float4*>(sb_row_base + ki * WARP_TILE_K);
        float2 t01 = {t.x, t.y};
        float2 t23 = {t.z, t.w};
        float2 kv01, kv23;
        if (ki < 3) {
            kv01 = {k_pref[ki][0], k_pref[ki][1]};
            kv23 = {k_pref[ki][2], k_pref[ki][3]};
        } else {
            kv01 = {__bfloat162float(kb[ko + 0]) * ks, __bfloat162float(kb[ko + 1]) * ks};
            kv23 = {__bfloat162float(kb[ko + 2]) * ks, __bfloat162float(kb[ko + 3]) * ks};
        }
        float2 o01 = {0.f, 0.f};
        float2 o23 = {0.f, 0.f};
        asm volatile(
            "v_pk_fma_f32 %0, %2, %4, %0\n"
            "v_pk_fma_f32 %1, %3, %4, %1\n"
            "v_pk_fma_f32 %0, %6, %5, %0\n"
            "v_pk_fma_f32 %1, %7, %5, %1\n"
            : "+v"(o01), "+v"(o23)
            : "v"(t01), "v"(t23), "v"(eg2), "v"(vn2), "v"(kv01), "v"(kv23));
        t = {o01.x, o01.y, o23.x, o23.y};
        float* dst = sb_row_base + ki * WARP_TILE_K;
        store_state_f4_nt(dst, t);
    }
}

// ============================================================
// Min-register NVB=4 body (V_ITERS==1): same math as compact, lower VGPR.
// - L2 partials: one ki-less pass (no q_reg/k_reg arrays)
// - Matvec: reload q/k per ki, scale on the fly (no cached q/k tiles)
// - State update: reload k per element (reuses live temps only)
// - 3-way warp reduce: single temp reused across xor rounds
// ============================================================
template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_minreg_body_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int NUM_V_BLOCKS = 4;
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    static_assert(TILE_V == WARP_GROUP_TILE_V, "minreg body expects V_ITERS==1 for NVB=4");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int ab = batch_idx * NVH + hv_idx;
    const float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    const float sp = bi_fast_log1p(bi_fast_exp(x));
    const float exp_g = bi_fast_exp(-bi_fast_exp(A_log[hv_idx]) * sp);
    const float beta = bi_rcp(1.0f + bi_fast_exp(-__bfloat162float(b_input[ab])));

    float S[K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = *reinterpret_cast<const float4*>(
            sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        S[ki][0] = t.x; S[ki][1] = t.y; S[ki][2] = t.z; S[ki][3] = t.w;
    }

    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key   + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float sq_q = 0.0f;
    float sq_k = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        const int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            const float qv = __bfloat162float(qb[ko + j]);
            const float kv = __bfloat162float(kb[ko + j]);
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }

    reduce_2way_asm(sq_q, sq_k, (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    const float inv_rsq_q = bi_rsqrt(sq_q + 1e-6f);
    const float inv_rsq_k = bi_rsqrt(sq_k + 1e-6f);
    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    float dot_kq = 0.0f;
    float res_hk = 0.0f;
    float res_hq = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        const int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            const float qv = __bfloat162float(qb[ko + j]) * qs;
            const float kv = __bfloat162float(kb[ko + j]) * ks;
            dot_kq += kv * qv;
            res_hk += S[ki][j] * kv;
            res_hq += S[ki][j] * qv;
        }
    }

    float t = ds_swizzle_xor_f32<kDsSwizzleXor1>(dot_kq);
    dot_kq += t;
    t = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hk);
    res_hk += t;
    t = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hq);
    res_hq += t;
    t = ds_swizzle_xor_f32<kDsSwizzleXor2>(dot_kq);
    dot_kq += t;
    t = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hk);
    res_hk += t;
    t = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hq);
    res_hq += t;
    t = ds_swizzle_xor_f32<kDsSwizzleXor4>(dot_kq);
    dot_kq += t;
    t = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hk);
    res_hk += t;
    t = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hq);
    res_hq += t;

    res_hk *= exp_g;
    res_hq *= exp_g;

    const float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk) * beta;
    const float vn = as_float(__builtin_amdgcn_ds_bpermute(leader_addr, as_int(tmp0)));

    res_hq += vn * dot_kq;
    if (k_tid == 0) {
        const int64_t out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM + global_v_base;
        output[out_base] = __float2bfloat16(res_hq);
    }

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        const int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            const float kv = __bfloat162float(kb[ko + j]) * ks;
            S[ki][j] = S[ki][j] * exp_g + kv * vn;
        }
    }

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 storev = {S[ki][0], S[ki][1], S[ki][2], S[ki][3]};
        *reinterpret_cast<float4*>(sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = storev;
    }
}

// ============================================================
// Stable body: zero asm-volatile — compiler has 100% scheduling freedom.
// Replaces shfl_idx asm volatile with __builtin_amdgcn_ds_bpermute.
// ============================================================
template <int NKH, int NVH, int NUM_V_BLOCKS>
__device__ __forceinline__ void gdn_decode_kernel_stable_body(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int ab = batch_idx * NVH + hv_idx;
    float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    float sp = bi_fast_log1p(bi_fast_exp(x));
    float exp_g = bi_fast_exp(-bi_fast_exp(A_log[hv_idx]) * sp);
    float beta = bi_rcp(1.0f + bi_fast_exp(-__bfloat162float(b_input[ab])));

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = *reinterpret_cast<const float4*>(
                sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y;
            S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key   + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv; k_reg[ki][j] = kv;
            sq_q += qv * qv; sq_k += kv * kv;
        }
    }

    reduce_2way_asm(sq_q, sq_k,
        (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    float inv_rsq_q = bi_rsqrt(sq_q + 1e-6f);
    float inv_rsq_k = bi_rsqrt(sq_k + 1e-6f);

    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs; k_reg[ki][j] *= ks;
        }
    }

    float dot_kq = 0.0f;
    float res_hk[V_ITERS], res_hq[V_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) { res_hk[vi] = 0.0f; res_hq[vi] = 0.0f; }
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = q_reg[ki][j], kv = k_reg[ki][j];
            dot_kq += kv * qv;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += S[vi][ki][j] * kv;
                res_hq[vi] += S[vi][ki][j] * qv;
            }
        }
    }

    if constexpr (V_ITERS == 4) {
        reduce_postloop_viters4_fused_asm(dot_kq, res_hk, res_hq, tid);
    } else if constexpr (V_ITERS == 2) {
        reduce_5way_asm(dot_kq, res_hk[0], res_hk[1], res_hq[0], res_hq[1],
            (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    } else {
        float t0, t1, t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] *= exp_g; res_hq[vi] *= exp_g;
    }

    float vn[V_ITERS];
    if constexpr (V_ITERS == 4) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0*WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1*WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        float tmp2 = (__bfloat162float(vb[global_v_base + 2*WARP_GROUP_TILE_V]) - res_hk[2]) * beta;
        float tmp3 = (__bfloat162float(vb[global_v_base + 3*WARP_GROUP_TILE_V]) - res_hk[3]) * beta;
        shfl_idx_4way_bi(vn[0], vn[1], vn[2], vn[3], tmp0, tmp1, tmp2, tmp3, leader_addr);
    } else if constexpr (V_ITERS >= 2) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0*WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1*WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        shfl_idx_2way_bi(vn[0], vn[1], tmp0, tmp1, leader_addr);
    } else {
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk[0]) * beta;
        vn[0] = shfl_idx_1way_bi(tmp0, leader_addr);
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++)
        res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;

    if (k_tid == 0) {
        const int64_t out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM
                               + hv_idx * HEAD_V_DIM + global_v_base;
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
    }

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float kv = k_reg[ki][j];
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
        }
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

__global__
__launch_bounds__(BLOCK_THREADS, 4)
__attribute__((amdgpu_waves_per_eu(4, 6)))
void gdn_decode_kernel_iasm_tp8_sq1(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    gdn_decode_kernel_iasm_sq1_body<2, 4, 2>(
        query, key, value, a_input, b_input, dt_bias, A_log, indices, state, output, batch_size, scale);
}

// Fused-norm kernel: defer q/k normalization AFTER matvec to eliminate
// norm-reduce → normalize → matvec serial dependency.
// Collapses 2-way norm reduce + 3-way matvec reduce into single 5-way reduce.
template <int NKH, int NVH, int NUM_V_BLOCKS>
__device__ __forceinline__ void gdn_decode_kernel_fused_norm_body(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int xor1 = (lane_id ^ 1) << 2;
    const int xor2 = (lane_id ^ 2) << 2;
    const int xor4 = (lane_id ^ 4) << 2;

    // ---- Phase 1: Issue ALL loads upfront ----
    // State loads
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    float S[V_ITERS][K_ITERS][VEC_K];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = *reinterpret_cast<const float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y; S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    // Q/K loads + norm accumulation (NO reduction yet - deferred)
    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv;
            k_reg[ki][j] = kv;
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }

    // Gating (overlaps with outstanding memory ops)
    const int ab = batch_idx * NVH + hv_idx;
    float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    float sp = asm_fast_log1p(asm_fast_exp(x));
    float exp_g = asm_fast_exp(-asm_fast_exp(A_log[hv_idx]) * sp);
    float beta = asm_rcp(1.0f + asm_fast_exp(-__bfloat162float(b_input[ab])));

    // ---- Phase 2: Matvec with UN-NORMALIZED q/k ----
    // No waiting for norm reduction! Matvec starts immediately after q/k load.
    float dot_kq = 0.0f;
    float res_hk[V_ITERS], res_hq[V_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] = 0.0f;
        res_hq[vi] = 0.0f;
    }

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = q_reg[ki][j];
            float kv = k_reg[ki][j];
            dot_kq += kv * qv;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += S[vi][ki][j] * kv;
                res_hq[vi] += S[vi][ki][j] * qv;
            }
        }
    }

    // ---- Phase 3: Single 5-way reduction (norm + matvec together) ----
    // Reduce sq_q, sq_k, dot_kq, res_hk, res_hq all at once
    if constexpr (V_ITERS == 1) {
        // 5-way reduce: sq_q, sq_k, dot_kq, res_hk[0], res_hq[0]
        reduce_5way_asm(sq_q, sq_k, dot_kq, res_hk[0], res_hq[0], xor1, xor2, xor4);
    } else {
        // V_ITERS>=2: need 7-way? Fall back to separate reductions
        reduce_2way_asm(sq_q, sq_k, xor1, xor2, xor4);
        reduce_5way_asm(dot_kq, res_hk[0], res_hk[1], res_hq[0], res_hq[1], xor1, xor2, xor4);
    }

    // ---- Phase 4: Apply deferred normalization to RESULTS ----
    float inv_rsq_q, inv_rsq_k;
    asm_rsqrt_pair_issue(sq_q + 1e-6f, sq_k + 1e-6f, inv_rsq_q, inv_rsq_k);
    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    // Scale matvec results by deferred normalization factors
    // dot_kq_normalized = dot_kq_raw * qs * ks
    dot_kq *= qs * ks;
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] *= ks * exp_g;
        res_hq[vi] *= qs * exp_g;
    }

    // ---- Phase 5: Compute vn, output, state update (same as original) ----
    float vn[V_ITERS];
    if constexpr (V_ITERS >= 2) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        shfl_idx_2way_asm(vn[0], vn[1], tmp0, tmp1, leader_addr);
    } else {
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk[0]) * beta;
        asm volatile(
            "ds_bpermute_b32 %0, %2, %1\n"
            "s_waitcnt lgkmcnt(0)\n"
            : "=&v"(vn[0])
            : "v"(tmp0), "v"(leader_addr)
        );
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;
    }

    if (k_tid == 0) {
        const int64_t out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM
                               + hv_idx * HEAD_V_DIM + global_v_base;
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }
    }

    // State update: S_new = S_old * exp_g + k_normalized * vn
    // Need to normalize k before state update
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) {
                S[vi][ki][j] = S[vi][ki][j] * exp_g + k_reg[ki][j] * (ks * vn[vi]);
            }
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

// Ultra-optimized kernel body: explicit FMA, minimal instruction count, buffer_load
// Optimized for latency-bound BS=1 scenario with only 16 blocks on 304 CUs
template <int NKH, int NVH, int NUM_V_BLOCKS>
__device__ __forceinline__ void gdn_decode_kernel_ultra_body(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    // State buffer descriptor
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    __amdgpu_buffer_rsrc_t state_rsrc = make_buffer_rsrc(sb, shs * 4);

    // Issue ALL state loads upfront to maximize latency hiding
    float S[V_ITERS][K_ITERS][VEC_K];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int byte_off = (gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) * 4;
            buf_load_f4(S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3],
                        state_rsrc, byte_off);
        }
    }

    // Load q/k, compute norms, and gating while state loads are in flight
    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K];
    float k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv;
            k_reg[ki][j] = kv;
            sq_q = __fmaf_rn(qv, qv, sq_q);
            sq_k = __fmaf_rn(kv, kv, sq_k);
        }
    }

    // Gating (overlap with q/k loads and norm computation)
    const int ab = batch_idx * NVH + hv_idx;
    float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    float sp = __logf(1.0f + __expf(x));
    float exp_g = __expf(-__expf(A_log[hv_idx]) * sp);
    float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b_input[ab])));

    // Norm reduction
    reduce_2way_asm(sq_q, sq_k,
                    (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    float inv_rsq_q, inv_rsq_k;
    asm_rsqrt_pair_issue(sq_q + 1e-6f, sq_k + 1e-6f, inv_rsq_q, inv_rsq_k);

    // Apply norms: use FMA where possible
    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs;
            k_reg[ki][j] *= ks;
        }
    }

    // Matvec with explicit FMA to minimize instruction count
    float dot_kq = 0.0f;
    float res_hk[V_ITERS], res_hq[V_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] = 0.0f;
        res_hq[vi] = 0.0f;
    }

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            const float qv = q_reg[ki][j];
            const float kv = k_reg[ki][j];
            dot_kq = __fmaf_rn(kv, qv, dot_kq);
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] = __fmaf_rn(S[vi][ki][j], kv, res_hk[vi]);
                res_hq[vi] = __fmaf_rn(S[vi][ki][j], qv, res_hq[vi]);
            }
        }
    }

    // 3-way reduction for V_ITERS=1 with overlapped swizzle+add
    if constexpr (V_ITERS >= 2) {
        reduce_5way_asm(dot_kq, res_hk[0], res_hk[1], res_hq[0], res_hq[1],
                        (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    } else {
        // Issue all 3 swizzles per round together, then accumulate
        float t0, t1, t2;
        int a1 = (lane_id ^ 1) << 2, a2 = (lane_id ^ 2) << 2, a4 = (lane_id ^ 4) << 2;
        // Round 1: xor1
        asm volatile(
            "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,1)\n"
            "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,1)\n"
            "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,1)\n"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(t0), "=v"(t1), "=v"(t2)
            : "v"(dot_kq), "v"(res_hk[0]), "v"(res_hq[0])
        );
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
        // Round 2: xor2
        asm volatile(
            "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,2)\n"
            "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,2)\n"
            "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,2)\n"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(t0), "=v"(t1), "=v"(t2)
            : "v"(dot_kq), "v"(res_hk[0]), "v"(res_hq[0])
        );
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
        // Round 3: xor4
        asm volatile(
            "ds_swizzle_b32 %0, %3 offset:swizzle(SWAP,4)\n"
            "ds_swizzle_b32 %1, %4 offset:swizzle(SWAP,4)\n"
            "ds_swizzle_b32 %2, %5 offset:swizzle(SWAP,4)\n"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(t0), "=v"(t1), "=v"(t2)
            : "v"(dot_kq), "v"(res_hk[0]), "v"(res_hq[0])
        );
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
    }

    // Apply gating + compute vn
    float vn[V_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] *= exp_g;
        res_hq[vi] *= exp_g;
    }

    if constexpr (V_ITERS >= 2) {
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        shfl_idx_2way_asm(vn[0], vn[1], tmp0, tmp1, leader_addr);
    } else {
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk[0]) * beta;
        asm volatile(
            "ds_bpermute_b32 %0, %2, %1\n"
            "s_waitcnt lgkmcnt(0)"
            : "=v"(vn[0]) : "v"(tmp0), "v"(leader_addr)
        );
    }

    // Output: res_hq + vn * dot_kq
    if (k_tid == 0) {
        const int64_t out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM
                               + hv_idx * HEAD_V_DIM + global_v_base;
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            output[out_base + vi * WARP_GROUP_TILE_V] =
                __float2bfloat16(__fmaf_rn(vn[vi], dot_kq, res_hq[vi]));
        }
    }

    // State update: S_new = S_old * exp_g + k * vn, store via buffer_store
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) {
                S[vi][ki][j] = __fmaf_rn(k_reg[ki][j], vn[vi], S[vi][ki][j] * exp_g);
            }
            int byte_off = (gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) * 4;
            buf_store_f4(S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3],
                         state_rsrc, byte_off);
        }
    }
}

// Buffer-load kernel body: uses buffer_load_dwordx4 for state (matches FlyDSL ISA)
template <int NKH, int NVH, int NUM_V_BLOCKS>
__device__ __forceinline__ void gdn_decode_kernel_bufload_body(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int xor1 = (lane_id ^ 1) << 2;
    const int xor2 = (lane_id ^ 2) << 2;
    const int xor4 = (lane_id ^ 4) << 2;

    // State buffer descriptor (buffer_load path)
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    __amdgpu_buffer_rsrc_t state_rsrc = make_buffer_rsrc(sb, shs * 4);

    // Load state via buffer_load_dwordx4
    float S[V_ITERS][K_ITERS][VEC_K];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int byte_off = (gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) * 4;
            buf_load_f4(S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3],
                        state_rsrc, byte_off);
        }
    }

    // Q/K buffer descriptors
    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    // Load q/k into register cache + fused L2 norm
    float q_reg[K_ITERS][VEC_K];
    float k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f;
    float sq_k = 0.0f;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv;
            k_reg[ki][j] = kv;
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }

    // Gating
    const int ab = batch_idx * NVH + hv_idx;
    float x = __bfloat162float(a_input[ab]) + __bfloat162float(dt_bias[hv_idx]);
    float sp = asm_fast_log1p(asm_fast_exp(x));
    float exp_g = asm_fast_exp(-asm_fast_exp(A_log[hv_idx]) * sp);
    float beta = asm_rcp(1.0f + asm_fast_exp(-__bfloat162float(b_input[ab])));

    reduce_2way_asm(sq_q, sq_k, xor1, xor2, xor4);
    float inv_rsq_q, inv_rsq_k;
    asm_rsqrt_pair_issue(sq_q + 1e-6f, sq_k + 1e-6f, inv_rsq_q, inv_rsq_k);

    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs;
            k_reg[ki][j] *= ks;
        }
    }

    // Matvec from register cache
    float dot_kq = 0.0f;
    float res_hk[V_ITERS];
    float res_hq[V_ITERS];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] = 0.0f;
        res_hq[vi] = 0.0f;
    }

    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = q_reg[ki][j];
            float kv = k_reg[ki][j];
            dot_kq += kv * qv;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += S[vi][ki][j] * kv;
                res_hq[vi] += S[vi][ki][j] * qv;
            }
        }
    }

    // Fused reduction
    if constexpr (V_ITERS >= 2) {
        reduce_5way_asm(dot_kq, res_hk[0], res_hk[1], res_hq[0], res_hq[1], xor1, xor2, xor4);
    } else {
        float t0, t1, t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
        t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(dot_kq);
        t1 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hk[0]);
        t2 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hq[0]);
        dot_kq += t0; res_hk[0] += t1; res_hq[0] += t2;
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hk[vi] *= exp_g;
        res_hq[vi] *= exp_g;
    }

    float vn[V_ITERS];
    if constexpr (V_ITERS >= 2) {
        float tmp0 = (__bfloat162float(vb[global_v_base + 0 * WARP_GROUP_TILE_V]) - res_hk[0]) * beta;
        float tmp1 = (__bfloat162float(vb[global_v_base + 1 * WARP_GROUP_TILE_V]) - res_hk[1]) * beta;
        shfl_idx_2way_asm(vn[0], vn[1], tmp0, tmp1, leader_addr);
    } else {
        float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk[0]) * beta;
        asm volatile(
            "ds_bpermute_b32 %0, %2, %1\n"
            "s_waitcnt lgkmcnt(0)\n"
            : "=&v"(vn[0])
            : "v"(tmp0), "v"(leader_addr)
        );
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;
    }

    if (k_tid == 0) {
        const int64_t out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM
                               + hv_idx * HEAD_V_DIM + global_v_base;
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }
    }

    // State update and store via buffer_store_dwordx4
    #pragma unroll
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float kv = k_reg[ki][j];
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
            }
        }
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int byte_off = (gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) * 4;
            buf_store_f4(S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3],
                         state_rsrc, byte_off);
        }
    }
}

// NVB=4 optimal: lb10 + VGPR48 — beats FlyDSL at BS=1
__global__
__launch_bounds__(BLOCK_THREADS, 10)
__attribute__((amdgpu_waves_per_eu(10, 10)))
__attribute__((amdgpu_num_vgpr(48)))
void gdn_decode_kernel_iasm_tp8_sq1_nvb4_lb10v48(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    gdn_decode_kernel_iasm_sq1_body<2, 4, 4>(
        query, key, value, a_input, b_input, dt_bias, A_log, indices, state, output, batch_size, scale);
}

__global__
__launch_bounds__(BLOCK_THREADS, 3)
__attribute__((amdgpu_waves_per_eu(3, 5)))
void gdn_decode_kernel_iasm_tp8_sq1_nvb1(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    gdn_decode_kernel_iasm_sq1_body<2, 4, 1>(
        query, key, value, a_input, b_input, dt_bias, A_log, indices, state, output, batch_size, scale);
}

// NVB=1 launch_bounds sweep variants
#define DEFINE_NVH8_NVB1_VARIANT(NAME, LB_WAVES, WPE_MIN, WPE_MAX) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB_WAVES) \
__attribute__((amdgpu_waves_per_eu(WPE_MIN, WPE_MAX))) \
void NAME( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__          A_log, \
    const int* __restrict__            indices, \
    float* __restrict__                state, \
    __hip_bfloat16* __restrict__       output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_iasm_sq1_body<2, 8, 1>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, state, output, batch_size, scale); \
}

DEFINE_NVH8_NVB1_VARIANT(gdn_decode_kernel_iasm_tp8_nvh8_sq1_nvb1,       3, 3, 5)  // baseline
DEFINE_NVH8_NVB1_VARIANT(gdn_decode_kernel_iasm_tp8_nvh8_sq1_nvb1_lb4,   4, 4, 6)  // high occupancy

#undef DEFINE_NVH8_NVB1_VARIANT

// ── NVH=8 NVB=4 production sweep: test VGPR cap vs no-cap ──
#define DEF_NVH8_NVB4_BASE(SUFFIX, LB, WPE_LO, WPE_HI) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
void gdn_decode_kernel_iasm_tp8_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_iasm_sq1_body<2, 8, 4>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

#define DEF_NVH8_NVB4_VGPR(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_iasm_tp8_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_iasm_sq1_body<2, 8, 4>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_BASE(_lb2, 2, 2, 2)
DEF_NVH8_NVB4_BASE(_lb4, 4, 4, 4)
DEF_NVH8_NVB4_VGPR(_v80_lb6, 6, 6, 6, 80)
DEF_NVH8_NVB4_VGPR(_v64_lb8, 8, 8, 8, 64)
DEF_NVH8_NVB4_VGPR(_v48_lb10, 10, 10, 10, 48)
// Round 2: fine-tune around v80_lb6 (best from Round 1)
DEF_NVH8_NVB4_VGPR(_v76_lb6, 6, 6, 6, 76)
DEF_NVH8_NVB4_VGPR(_v72_lb6, 6, 6, 6, 72)
DEF_NVH8_NVB4_VGPR(_v88_lb6, 6, 6, 6, 88)
DEF_NVH8_NVB4_VGPR(_v96_lb5, 5, 5, 5, 96)
DEF_NVH8_NVB4_VGPR(_v80_lb5, 5, 5, 5, 80)
DEF_NVH8_NVB4_VGPR(_v80_lb7, 7, 7, 7, 80)
DEF_NVH8_NVB4_BASE(_lb6_nocap, 6, 6, 6)

#undef DEF_NVH8_NVB4_BASE
#undef DEF_NVH8_NVB4_VGPR

// ── Builtin body variants: compiler-schedulable transcendentals ──
#define DEF_NVH8_NVB4_BUILTIN(SUFFIX, LB, WPE_LO, WPE_HI) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
void gdn_decode_kernel_builtin_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_builtin_body<2, 8, 4>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

#define DEF_NVH8_NVB4_BUILTIN_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_builtin_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_builtin_body<2, 8, 4>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_BUILTIN(_lb2, 2, 2, 2)
DEF_NVH8_NVB4_BUILTIN(_lb4, 4, 4, 4)
DEF_NVH8_NVB4_BUILTIN(_lb6, 6, 6, 6)
DEF_NVH8_NVB4_BUILTIN_V(_v88_lb6, 6, 6, 6, 88)
DEF_NVH8_NVB4_BUILTIN_V(_v80_lb6, 6, 6, 6, 80)
DEF_NVH8_NVB4_BUILTIN_V(_v96_lb5, 5, 5, 5, 96)
DEF_NVH8_NVB4_BUILTIN_V(_v64_lb8, 8, 8, 8, 64)
DEF_NVH8_NVB4_BUILTIN_V(_v44_lb12, 12, 12, 12, 44)
DEF_NVH8_NVB4_BUILTIN_V(_v56_lb9, 9, 9, 9, 56)
DEF_NVH8_NVB4_BUILTIN_V(_v52_lb10, 10, 10, 10, 52)
DEF_NVH8_NVB4_BUILTIN_V(_v60_lb9, 9, 9, 9, 60)
DEF_NVH8_NVB4_BUILTIN_V(_v48_lb10, 10, 10, 10, 48)
DEF_NVH8_NVB4_BUILTIN_V(_v40_lb14, 14, 14, 14, 40)
DEF_NVH8_NVB4_BUILTIN_V(_v36_lb16, 16, 16, 16, 36)
DEF_NVH8_NVB4_BUILTIN_V(_v112_lb2, 2, 2, 2, 112)
DEF_NVH8_NVB4_BUILTIN_V(_v128_lb2, 2, 2, 2, 128)

// ── BS=2 cache-bypass variants ──
// In `online` mode the GDN kernel sits between in_proj/conv (which thrash L2) and
// rmsnorm/out_proj (which read GDN output). For BS=2 the state working-set per
// XCD is ~256KB and gets evicted by companion ops between captured-graph replays.
// Mark state float4 load/store as nontemporal so they bypass L2 entirely (HBM
// streaming) and stop polluting the small lines that companion needs warm.
// The DATA LAYOUT is unchanged — only memory-policy bits differ.
#define DEF_NVH8_NVB4_BUILTIN_V_NT(SUFFIX, LB, WPE_LO, WPE_HI, VGPR, NT_LD, NT_ST) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_builtin_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_builtin_body<2, 8, 4, false, NT_LD, NT_ST>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_BUILTIN_V_NT(_v52_lb10_ntload,         10, 10, 10, 52, true,  false)
DEF_NVH8_NVB4_BUILTIN_V_NT(_v52_lb10_ntstore,        10, 10, 10, 52, false, true)
DEF_NVH8_NVB4_BUILTIN_V_NT(_v52_lb10_ntboth,         10, 10, 10, 52, true,  true)
DEF_NVH8_NVB4_BUILTIN_V_NT(_v48_lb10_ntload,         10, 10, 10, 48, true,  false)
DEF_NVH8_NVB4_BUILTIN_V_NT(_v48_lb10_ntboth,         10, 10, 10, 48, true,  true)
DEF_NVH8_NVB4_BUILTIN_V_NT(_v56_lb9_ntload,           9,  9,  9, 56, true,  false)
DEF_NVH8_NVB4_BUILTIN_V_NT(_v56_lb9_ntboth,           9,  9,  9, 56, true,  true)

#undef DEF_NVH8_NVB4_BUILTIN
#undef DEF_NVH8_NVB4_BUILTIN_V
#undef DEF_NVH8_NVB4_BUILTIN_V_NT

// ── exp2 gating path (explicit v_exp_f32 + log2-domain log1p), variants 110–115 ──
#define DEF_NVH8_NVB4_EXP2(SUFFIX, LB, WPE_LO, WPE_HI) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
void gdn_decode_kernel_exp2_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_exp2_body_nvb4<2, 8>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

#define DEF_NVH8_NVB4_EXP2_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_exp2_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_exp2_body_nvb4<2, 8>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_EXP2_V(_v52_lb10, 10, 10, 10, 52)
DEF_NVH8_NVB4_EXP2_V(_v48_lb10, 10, 10, 10, 48)
DEF_NVH8_NVB4_EXP2_V(_v56_lb9, 9, 9, 9, 56)
DEF_NVH8_NVB4_EXP2_V(_v60_lb9, 9, 9, 9, 60)
DEF_NVH8_NVB4_EXP2_V(_v44_lb12, 12, 12, 12, 44)
DEF_NVH8_NVB4_EXP2_V(_v64_lb8, 8, 8, 8, 64)

// exp2 compact + interleaved swizzle (GDN_NVH8_NVB4_VARIANT 130–135)
#define DEF_NVH8_NVB4_EXP2_COMPACT_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_exp2_compact_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_exp2_compact_body_nvb4<2, 8>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_EXP2_COMPACT_V(_v52_lb10, 10, 10, 10, 52)
DEF_NVH8_NVB4_EXP2_COMPACT_V(_v48_lb10, 10, 10, 10, 48)
DEF_NVH8_NVB4_EXP2_COMPACT_V(_v56_lb9, 9, 9, 9, 56)
DEF_NVH8_NVB4_EXP2_COMPACT_V(_v60_lb8, 8, 8, 8, 60)
DEF_NVH8_NVB4_EXP2_COMPACT_V(_v44_lb12, 12, 12, 12, 44)
DEF_NVH8_NVB4_EXP2_COMPACT_V(_v64_lb8, 8, 8, 8, 64)

#undef DEF_NVH8_NVB4_EXP2_COMPACT_V

// Builtin ds_swizzle + compiler-scheduled waits (GDN_NVH8_NVB4_VARIANT 150–155)
#define DEF_NVH8_NVB4_NOWAIT_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_nowait_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_nowait_body_nvb4<2, 8>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_NOWAIT_V(_v52_lb10, 10, 10, 10, 52)
DEF_NVH8_NVB4_NOWAIT_V(_v48_lb10, 10, 10, 10, 48)
DEF_NVH8_NVB4_NOWAIT_V(_v56_lb9, 9, 9, 9, 56)
DEF_NVH8_NVB4_NOWAIT_V(_v60_lb8, 8, 8, 8, 60)
DEF_NVH8_NVB4_NOWAIT_V(_v44_lb12, 12, 12, 12, 44)
DEF_NVH8_NVB4_NOWAIT_V(_v64_lb8, 8, 8, 8, 64)

#undef DEF_NVH8_NVB4_NOWAIT_V

// exp2 + paired res_hk/res_hq reduction (float2 ds_swizzle + v_pk_add) (GDN_NVH8_NVB4_VARIANT 160–165)
#define DEF_NVH8_NVB4_FASTREDUCE_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_fastreduce_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_fastreduce_body_nvb4<2, 8>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_FASTREDUCE_V(_v44_lb12, 12, 12, 12, 44)
DEF_NVH8_NVB4_FASTREDUCE_V(_v48_lb10, 10, 10, 10, 48)
DEF_NVH8_NVB4_FASTREDUCE_V(_v52_lb10, 10, 10, 10, 52)
DEF_NVH8_NVB4_FASTREDUCE_V(_v56_lb9, 9, 9, 9, 56)
DEF_NVH8_NVB4_FASTREDUCE_V(_v60_lb8, 8, 8, 8, 60)
DEF_NVH8_NVB4_FASTREDUCE_V(_v68_lb8, 8, 8, 8, 68)

#undef DEF_NVH8_NVB4_FASTREDUCE_V

#undef DEF_NVH8_NVB4_EXP2
#undef DEF_NVH8_NVB4_EXP2_V

#define DEF_NVH8_NVB4_BUILTIN_FUSESTORE(SUFFIX, LB, WPE_LO, WPE_HI) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
void gdn_decode_kernel_builtin_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_builtin_body<2, 8, 4, true>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

#define DEF_NVH8_NVB4_BUILTIN_FUSESTORE_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_builtin_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_builtin_body<2, 8, 4, true>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_BUILTIN_FUSESTORE_V(_v80_lb6_fusestore, 6, 6, 6, 80)
DEF_NVH8_NVB4_BUILTIN_FUSESTORE_V(_v64_lb8_fusestore, 8, 8, 8, 64)

#undef DEF_NVH8_NVB4_BUILTIN_FUSESTORE
#undef DEF_NVH8_NVB4_BUILTIN_FUSESTORE_V

// ── Compact NVB=4 body variants: specialized V_ITERS==1 fast path (BS=1 friendly) ──
#define DEF_NVH8_NVB4_COMPACT_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_builtin_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_builtin_body_nvb4_compact<2, 8>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_COMPACT_V(_compact_v52_lb10, 10, 10, 10, 52)
DEF_NVH8_NVB4_COMPACT_V(_compact_v48_lb10, 10, 10, 10, 48)
DEF_NVH8_NVB4_COMPACT_V(_compact_v44_lb12, 12, 12, 12, 44)
DEF_NVH8_NVB4_COMPACT_V(_compact_v56_lb9, 9, 9, 9, 56)
DEF_NVH8_NVB4_COMPACT_V(_compact_v60_lb9, 9, 9, 9, 60)
DEF_NVH8_NVB4_COMPACT_V(_compact_v64_lb8, 8, 8, 8, 64)
DEF_NVH8_NVB4_COMPACT_V(_compact_v40_lb14, 14, 14, 14, 40)
DEF_NVH8_NVB4_COMPACT_V(_compact_v36_lb16, 16, 16, 16, 36)

#undef DEF_NVH8_NVB4_COMPACT_V

// ── Min-register NVB=4 body: amdgpu_num_vgpr sweep (occupancy vs spill) ──
#define DEF_NVH8_NVB4_MINREG_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_builtin_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_minreg_body_nvb4<2, 8>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_MINREG_V(_minreg_v32_lb22, 22, 22, 22, 32)
DEF_NVH8_NVB4_MINREG_V(_minreg_v34_lb20, 20, 20, 20, 34)
DEF_NVH8_NVB4_MINREG_V(_minreg_v36_lb18, 18, 18, 18, 36)
DEF_NVH8_NVB4_MINREG_V(_minreg_v38_lb16, 16, 16, 16, 38)
DEF_NVH8_NVB4_MINREG_V(_minreg_v40_lb14, 14, 14, 14, 40)
DEF_NVH8_NVB4_MINREG_V(_minreg_v44_lb12, 12, 12, 12, 44)

#undef DEF_NVH8_NVB4_MINREG_V

// ============================================================
// Prefetch-optimized BS=1 body: compute gating while state loads
// - Keeps same memory access pattern as compact body
// - Computes gating parameters during memory load phase
// - Uses compiler-managed waits for correctness
// ============================================================
template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_prefetch_body_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
);

// Templated NT-capable variant of prefetch_body (combines V=4 scalar-first
// reordering with V=6 nontemporal cache-bypass on state load/store).
// Original prefetch_body_nvb4<NKH,NVH> remains the default (NT=false,false).
template <int NKH, int NVH, bool NT_LD, bool NT_ST>
__device__ __forceinline__ void gdn_decode_kernel_prefetch_body_nvb4_nt(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int NUM_V_BLOCKS = 4;
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    static_assert(TILE_V == WARP_GROUP_TILE_V, "prefetch_body_nt expects V_ITERS==1 for NVB=4");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int ab = batch_idx * NVH + hv_idx;

    // Scalar-first ordering: load gating scalars + compute SFU ops first so
    // their long-latency exp/log/rcp pipeline overlaps with the upcoming
    // state HBM load.
    const float r_a = __bfloat162float(a_input[ab]);
    const float r_dt = __bfloat162float(dt_bias[hv_idx]);
    const float r_b = __bfloat162float(b_input[ab]);
    const float r_A_log = A_log[hv_idx];

    const float x = r_a + r_dt;
    const float sp = bi_fast_log1p(bi_fast_exp(x));
    const float exp_g = bi_fast_exp(-bi_fast_exp(r_A_log) * sp);
    const float beta = bi_rcp(1.0f + bi_fast_exp(-r_b));

    // State load (nontemporal-capable).
    float S[K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = load_state_f4_maybent<NT_LD>(
            sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        S[ki][0] = t.x; S[ki][1] = t.y; S[ki][2] = t.z; S[ki][3] = t.w;
    }

    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key   + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv;
            k_reg[ki][j] = kv;
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }

    reduce_2way_asm(sq_q, sq_k, (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    float inv_rsq_q = bi_rsqrt(sq_q + 1e-6f);
    float inv_rsq_k = bi_rsqrt(sq_k + 1e-6f);
    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs;
            k_reg[ki][j] *= ks;
        }
    }

    float dot_kq = 0.0f;
    float res_hk = 0.0f;
    float res_hq = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = q_reg[ki][j];
            float kv = k_reg[ki][j];
            dot_kq += kv * qv;
            res_hk += S[ki][j] * kv;
            res_hq += S[ki][j] * qv;
        }
    }

    float t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(dot_kq);
    float t1 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hk);
    float t2 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(dot_kq);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hk);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(dot_kq);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hk);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;

    res_hk *= exp_g;
    res_hq *= exp_g;

    float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk) * beta;
    float vn = as_float(__builtin_amdgcn_ds_bpermute(leader_addr, as_int(tmp0)));

    res_hq += vn * dot_kq;
    if (k_tid == 0) {
        const int64_t out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM + global_v_base;
        output[out_base] = __float2bfloat16(res_hq);
    }

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            S[ki][j] = S[ki][j] * exp_g + k_reg[ki][j] * vn;
        }
    }

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = {S[ki][0], S[ki][1], S[ki][2], S[ki][3]};
        store_state_f4_maybent<NT_ST>(
            sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K, t);
    }
}

// Original (non-NT) prefetch_body_nvb4 — definition.
template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_prefetch_body_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int NUM_V_BLOCKS = 4;
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    static_assert(TILE_V == WARP_GROUP_TILE_V, "prefetch body expects V_ITERS==1 for NVB=4");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int ab = batch_idx * NVH + hv_idx;
    
    // Load gating scalars first (fast, L1 cached)
    const float r_a = __bfloat162float(a_input[ab]);
    const float r_dt = __bfloat162float(dt_bias[hv_idx]);
    const float r_b = __bfloat162float(b_input[ab]);
    const float r_A_log = A_log[hv_idx];
    
    // Compute gating (SFU ops) - can overlap with subsequent memory loads
    const float x = r_a + r_dt;
    const float sp = bi_fast_log1p(bi_fast_exp(x));
    const float exp_g = bi_fast_exp(-bi_fast_exp(r_A_log) * sp);
    const float beta = bi_rcp(1.0f + bi_fast_exp(-r_b));

    // Load state (same pattern as compact body)
    float S[K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = *reinterpret_cast<const float4*>(
            sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        S[ki][0] = t.x; S[ki][1] = t.y; S[ki][2] = t.z; S[ki][3] = t.w;
    }

    // Load Q/K (same pattern as compact body)
    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key   + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv;
            k_reg[ki][j] = kv;
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }

    // L2 norm reduction (same as compact body)
    reduce_2way_asm(sq_q, sq_k, (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    float inv_rsq_q = bi_rsqrt(sq_q + 1e-6f);
    float inv_rsq_k = bi_rsqrt(sq_k + 1e-6f);
    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs;
            k_reg[ki][j] *= ks;
        }
    }

    // Compute dot products (same as compact body)
    float dot_kq = 0.0f;
    float res_hk = 0.0f;
    float res_hq = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = q_reg[ki][j];
            float kv = k_reg[ki][j];
            dot_kq += kv * qv;
            res_hk += S[ki][j] * kv;
            res_hq += S[ki][j] * qv;
        }
    }

    // 3-way reduction (same as compact body)
    float t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(dot_kq);
    float t1 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hk);
    float t2 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(dot_kq);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hk);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(dot_kq);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hk);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;

    res_hk *= exp_g;
    res_hq *= exp_g;

    // Leader broadcast (same as compact body, with proper type conversion)
    float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk) * beta;
    float vn = as_float(__builtin_amdgcn_ds_bpermute(leader_addr, as_int(tmp0)));

    res_hq += vn * dot_kq;
    if (k_tid == 0) {
        const int64_t out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM + global_v_base;
        output[out_base] = __float2bfloat16(res_hq);
    }

    // State update (same as compact body)
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            S[ki][j] = S[ki][j] * exp_g + k_reg[ki][j] * vn;
        }
    }

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = {S[ki][0], S[ki][1], S[ki][2], S[ki][3]};
        *reinterpret_cast<float4*>(sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
    }
}

// Prefetch body kernel wrappers with different VGPR/LB configurations
#define DEF_NVH8_NVB4_PREFETCH_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_prefetch_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_prefetch_body_nvb4<2, 8>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_PREFETCH_V(_v52_lb10, 10, 10, 10, 52)
DEF_NVH8_NVB4_PREFETCH_V(_v48_lb10, 10, 10, 10, 48)
DEF_NVH8_NVB4_PREFETCH_V(_v56_lb9, 9, 9, 9, 56)
DEF_NVH8_NVB4_PREFETCH_V(_v60_lb8, 8, 8, 8, 60)
DEF_NVH8_NVB4_PREFETCH_V(_v44_lb12, 12, 12, 12, 44)
DEF_NVH8_NVB4_PREFETCH_V(_v40_lb14, 14, 14, 14, 40)

#undef DEF_NVH8_NVB4_PREFETCH_V

// Combined V=4 (scalar-first reorder) + V=6 (nontemporal load+store) wrappers.
#define DEF_NVH8_NVB4_PREFETCH_NT_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR, NT_LD, NT_ST) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_prefetch_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_prefetch_body_nvb4_nt<2, 8, NT_LD, NT_ST>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_PREFETCH_NT_V(_v52_lb10_ntload,  10, 10, 10, 52, true,  false)
DEF_NVH8_NVB4_PREFETCH_NT_V(_v52_lb10_ntstore, 10, 10, 10, 52, false, true)
DEF_NVH8_NVB4_PREFETCH_NT_V(_v52_lb10_ntboth,  10, 10, 10, 52, true,  true)

#undef DEF_NVH8_NVB4_PREFETCH_NT_V

// ============================================================
// L2-prefetch body for BS=1 (production cold-L2 mitigation).
//
// Motivation: in production sglang, between two consecutive same-layer
// GDN decode calls we run ~21 other layers (attention + MoE) which fully
// thrash the per-XCD 4 MB L2.  This makes the state-tensor load a cold
// HBM fetch (~600 cycles) and pushes BS=1 from microbench 5.1us up to
// ~5.8us in production trace.
//
// This body issues a "fake" state load via inline asm at kernel entry —
// before scalar loads, gating compute, and Q/K/V loads.  The load goes
// through L1→L2→HBM and warms L2 lines for the *real* state load that
// happens later in the body.  The dummy destination register has zero
// live range (compiler will allocate any free temp), and `s_waitcnt vmcnt`
// is intentionally NOT issued — the load runs in the background while
// the rest of the kernel does scalar/SFU/Q/K/V work.
//
// Net effect (target): bring production BS=1 closer to microbench 5.1us
// by hiding HBM→L2 latency behind ~30-50 cycles of setup work.
// ============================================================
template <int NKH, int NVH>
__device__ __forceinline__ void gdn_decode_kernel_l2pref_body_nvb4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    constexpr int NUM_V_BLOCKS = 4;
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    static_assert(TILE_V == WARP_GROUP_TILE_V, "l2pref body expects V_ITERS==1 for NVB=4");

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;

    // ── L2 PREFETCH: issue dummy state loads at kernel entry ──
    // K_ITERS = HEAD_K_DIM / WARP_TILE_K = 128 / 32 = 4 iters per thread.
    // Each iter is a float4 (16B), so per thread = 64B per WG = 16 KB,
    // exactly covering the WG's state working set.  We do not wait on
    // vmcnt — the load runs in parallel with scalar/gating/Q/K/V work
    // below.  The destination register pool_pf is dead-on-arrival (never
    // read), so it has zero live range.  `volatile` + memory clobber
    // prevents the compiler from eliminating the load.
    {
        const float* pf_base = sb + global_v_base * HEAD_K_DIM + warp_k_vec_start;
        float4 pf_buf;
        GDN_K_ITERS_UNROLL
        for (int ki = 0; ki < K_ITERS; ki++) {
            asm volatile(
                "global_load_dwordx4 %0, %1, off"
                : "=v"(pf_buf)
                : "v"(pf_base + ki * WARP_TILE_K)
                : "memory"
            );
        }
        // No s_waitcnt vmcnt — the load is in flight.
    }

    const int ab = batch_idx * NVH + hv_idx;

    // Scalar loads — these go to L1/L2 paths different from state HBM,
    // so they happen in parallel with the in-flight state prefetch above.
    const float r_a = __bfloat162float(a_input[ab]);
    const float r_dt = __bfloat162float(dt_bias[hv_idx]);
    const float r_b = __bfloat162float(b_input[ab]);
    const float r_A_log = A_log[hv_idx];

    // Gating compute (SFU ops) overlaps with in-flight state HBM→L2.
    const float x = r_a + r_dt;
    const float sp = bi_fast_log1p(bi_fast_exp(x));
    const float exp_g = bi_fast_exp(-bi_fast_exp(r_A_log) * sp);
    const float beta = bi_rcp(1.0f + bi_fast_exp(-r_b));

    // Real state load — should hit L2 now (warmed by prefetch above).
    float S[K_ITERS][VEC_K];
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = *reinterpret_cast<const float4*>(
            sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
        S[ki][0] = t.x; S[ki][1] = t.y; S[ki][2] = t.z; S[ki][3] = t.w;
    }

    const auto* qb = query + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* kb = key   + (int64_t)batch_idx * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
    const auto* vb = value + (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

    float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
    float sq_q = 0.0f, sq_k = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        int ko = warp_k_vec_start + ki * WARP_TILE_K;
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = __bfloat162float(qb[ko + j]);
            float kv = __bfloat162float(kb[ko + j]);
            q_reg[ki][j] = qv;
            k_reg[ki][j] = kv;
            sq_q += qv * qv;
            sq_k += kv * kv;
        }
    }

    reduce_2way_asm(sq_q, sq_k, (lane_id ^ 1) << 2, (lane_id ^ 2) << 2, (lane_id ^ 4) << 2);
    float inv_rsq_q = bi_rsqrt(sq_q + 1e-6f);
    float inv_rsq_k = bi_rsqrt(sq_k + 1e-6f);
    const float qs = scale * inv_rsq_q;
    const float ks = inv_rsq_k;

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            q_reg[ki][j] *= qs;
            k_reg[ki][j] *= ks;
        }
    }

    float dot_kq = 0.0f;
    float res_hk = 0.0f;
    float res_hq = 0.0f;
    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            float qv = q_reg[ki][j];
            float kv = k_reg[ki][j];
            dot_kq += kv * qv;
            res_hk += S[ki][j] * kv;
            res_hq += S[ki][j] * qv;
        }
    }

    float t0 = ds_swizzle_xor_f32<kDsSwizzleXor1>(dot_kq);
    float t1 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hk);
    float t2 = ds_swizzle_xor_f32<kDsSwizzleXor1>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor2>(dot_kq);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hk);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor2>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;
    t0 = ds_swizzle_xor_f32<kDsSwizzleXor4>(dot_kq);
    t1 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hk);
    t2 = ds_swizzle_xor_f32<kDsSwizzleXor4>(res_hq);
    dot_kq += t0; res_hk += t1; res_hq += t2;

    res_hk *= exp_g;
    res_hq *= exp_g;

    float tmp0 = (__bfloat162float(vb[global_v_base]) - res_hk) * beta;
    float vn = as_float(__builtin_amdgcn_ds_bpermute(leader_addr, as_int(tmp0)));

    res_hq += vn * dot_kq;
    if (k_tid == 0) {
        const int64_t out_base = (int64_t)batch_idx * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM + global_v_base;
        output[out_base] = __float2bfloat16(res_hq);
    }

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        #pragma unroll
        for (int j = 0; j < VEC_K; j++) {
            S[ki][j] = S[ki][j] * exp_g + k_reg[ki][j] * vn;
        }
    }

    GDN_K_ITERS_UNROLL
    for (int ki = 0; ki < K_ITERS; ki++) {
        float4 t = {S[ki][0], S[ki][1], S[ki][2], S[ki][3]};
        *reinterpret_cast<float4*>(sb + global_v_base * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
    }
}

// L2-prefetch body kernel wrappers
#define DEF_NVH8_NVB4_L2PREF_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_l2pref_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_l2pref_body_nvb4<2, 8>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_L2PREF_V(_v52_lb10, 10, 10, 10, 52)
DEF_NVH8_NVB4_L2PREF_V(_v48_lb10, 10, 10, 10, 48)
DEF_NVH8_NVB4_L2PREF_V(_v56_lb9,  9,  9,  9,  56)

#undef DEF_NVH8_NVB4_L2PREF_V

// NVB=4 streaming + interleaved ds_swizzle / lgkm overlap (GDN_NVH8_NVB4_VARIANT 100–105)
#define DEF_NVH8_NVB4_INTERLEAVED_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_interleaved_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_interleaved_body_nvb4<2, 8>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_INTERLEAVED_V(_v52_lb10, 10, 10, 10, 52)
DEF_NVH8_NVB4_INTERLEAVED_V(_v48_lb10, 10, 10, 10, 48)
DEF_NVH8_NVB4_INTERLEAVED_V(_v56_lb9, 9, 9, 9, 56)
DEF_NVH8_NVB4_INTERLEAVED_V(_v64_lb8, 8, 8, 8, 64)
DEF_NVH8_NVB4_INTERLEAVED_V(_v44_lb12, 12, 12, 12, 44)
DEF_NVH8_NVB4_INTERLEAVED_V(_v40_lb14, 14, 14, 14, 40)

#undef DEF_NVH8_NVB4_INTERLEAVED_V

// Packed f32 dot body (GDN_NVH8_NVB4_VARIANT 90–95): v_pk_mul / v_pk_add matvec
#define DEF_NVH8_NVB4_PACKED_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_packed_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_packed_body_nvb4<2, 8>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_PACKED_V(_v52_lb10, 10, 10, 10, 52)
DEF_NVH8_NVB4_PACKED_V(_v48_lb10, 10, 10, 10, 48)
DEF_NVH8_NVB4_PACKED_V(_v56_lb9, 9, 9, 9, 56)
DEF_NVH8_NVB4_PACKED_V(_v60_lb8, 8, 8, 8, 60)
DEF_NVH8_NVB4_PACKED_V(_v44_lb12, 12, 12, 12, 44)
DEF_NVH8_NVB4_PACKED_V(_v40_lb14, 14, 14, 14, 40)

#undef DEF_NVH8_NVB4_PACKED_V

// v_pk_fma_f32 dot/matvec + exp2 gating (GDN_NVH8_NVB4_VARIANT 140–145)
#define DEF_NVH8_NVB4_PKFMA_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_pkfma_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_pkfma_body_nvb4<2, 8>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_PKFMA_V(_v48_lb10, 10, 10, 10, 48)
DEF_NVH8_NVB4_PKFMA_V(_v52_lb10, 10, 10, 10, 52)
DEF_NVH8_NVB4_PKFMA_V(_v56_lb9, 9, 9, 9, 56)
DEF_NVH8_NVB4_PKFMA_V(_v60_lb8, 8, 8, 8, 60)
DEF_NVH8_NVB4_PKFMA_V(_v44_lb12, 12, 12, 12, 44)
DEF_NVH8_NVB4_PKFMA_V(_v40_lb14, 14, 14, 14, 40)

#undef DEF_NVH8_NVB4_PKFMA_V

// Hand-written ASM hot path (GDN_NVH8_NVB4_VARIANT 200–205)
#define DEF_NVH8_NVB4_PUREASM_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_pureasm_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_pureasm_body_nvb4<2, 8>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_PUREASM_V(_v40_lb14, 14, 14, 14, 40)
DEF_NVH8_NVB4_PUREASM_V(_v44_lb12, 12, 12, 12, 44)
DEF_NVH8_NVB4_PUREASM_V(_v48_lb10, 10, 10, 10, 48)
DEF_NVH8_NVB4_PUREASM_V(_v52_lb10, 10, 10, 10, 52)
DEF_NVH8_NVB4_PUREASM_V(_v56_lb9, 9, 9, 9, 56)
DEF_NVH8_NVB4_PUREASM_V(_v60_lb8, 8, 8, 8, 60)

#undef DEF_NVH8_NVB4_PUREASM_V

// ── Builtin body NVH=8 NVB=2: grid = batch * NVH * 2 (BS=1 → 16 WGs vs 32 for NVB=4) ──
#define DEF_NVH8_NVB2_BUILTIN(SUFFIX, LB, WPE_LO, WPE_HI) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
void gdn_decode_kernel_builtin_nvh8_nvb2##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_builtin_body<2, 8, 2>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

#define DEF_NVH8_NVB2_BUILTIN_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_builtin_nvh8_nvb2##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_builtin_body<2, 8, 2>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB2_BUILTIN_V(_v64_lb8, 8, 8, 8, 64)
DEF_NVH8_NVB2_BUILTIN_V(_v80_lb6, 6, 6, 6, 80)
DEF_NVH8_NVB2_BUILTIN(_lb3, 3, 3, 5)

#undef DEF_NVH8_NVB2_BUILTIN
#undef DEF_NVH8_NVB2_BUILTIN_V

#define DEF_NVH8_NVB4_STABLE_V(SUFFIX, LB, WPE_LO, WPE_HI, VGPR) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB) \
__attribute__((amdgpu_waves_per_eu(WPE_LO, WPE_HI))) \
__attribute__((amdgpu_num_vgpr(VGPR))) \
void gdn_decode_kernel_stable_nvh8_nvb4##SUFFIX( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__ A_log, \
    const int* __restrict__ indices, \
    float* __restrict__ state, \
    __hip_bfloat16* __restrict__ output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_stable_body<2, 8, 4>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, \
        state, output, batch_size, scale); \
}

DEF_NVH8_NVB4_STABLE_V(_v60_lb9, 9, 9, 9, 60)

#undef DEF_NVH8_NVB4_STABLE_V

// ── BS=2 / BS=16 fixed-grid helpers (template arg order: VmWait, FIXED_BS, FUSE_POST) ──
#define DEFINE_BS2_NVB4_SQ1(NAME, LB_WAVES, WPE_MIN, WPE_MAX) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB_WAVES) \
__attribute__((amdgpu_waves_per_eu(WPE_MIN, WPE_MAX))) \
void NAME( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__          A_log, \
    const int* __restrict__            indices, \
    float* __restrict__                state, \
    __hip_bfloat16* __restrict__       output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_iasm_sq1_body_opt<2, 8, 4, false, 0, false>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, state, output, \
        batch_size, scale); \
}

DEFINE_BS2_NVB4_SQ1(gdn_decode_kernel_iasm_tp8_nvh8_bs2_nvb4_lb2, 2, 2, 4)
DEFINE_BS2_NVB4_SQ1(gdn_decode_kernel_iasm_tp8_nvh8_bs2_nvb4_lb3, 3, 3, 5)
DEFINE_BS2_NVB4_SQ1(gdn_decode_kernel_iasm_tp8_nvh8_bs2_nvb4,       4, 4, 6)
DEFINE_BS2_NVB4_SQ1(gdn_decode_kernel_iasm_tp8_nvh8_bs2_nvb4_lb6,  6, 6, 8)

#undef DEFINE_BS2_NVB4_SQ1

#define DEFINE_BS2_NVB2_SQ1(NAME, LB_WAVES, WPE_MIN, WPE_MAX) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB_WAVES) \
__attribute__((amdgpu_waves_per_eu(WPE_MIN, WPE_MAX))) \
void NAME( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__          A_log, \
    const int* __restrict__            indices, \
    float* __restrict__                state, \
    __hip_bfloat16* __restrict__       output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_iasm_sq1_body_opt<2, 8, 2, false, 0, false>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, state, output, \
        batch_size, scale); \
}

DEFINE_BS2_NVB2_SQ1(gdn_decode_kernel_iasm_tp8_nvh8_bs2_nvb2_lb3, 3, 3, 5)
DEFINE_BS2_NVB2_SQ1(gdn_decode_kernel_iasm_tp8_nvh8_bs2_nvb2,       4, 4, 6)

#undef DEFINE_BS2_NVB2_SQ1

#define DEFINE_GDN_BS2_EXTREME(NVB, NAME, LB_WAVES, WPE_MIN, WPE_MAX) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB_WAVES) \
__attribute__((amdgpu_waves_per_eu(WPE_MIN, WPE_MAX))) \
void NAME( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__          A_log, \
    const int* __restrict__            indices, \
    float* __restrict__                state, \
    __hip_bfloat16* __restrict__       output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_iasm_sq1_body_opt<2, 8, NVB, false, 2, false>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, state, output, \
        batch_size, scale); \
}

DEFINE_GDN_BS2_EXTREME(4, gdn_decode_kernel_bs2_extreme, 3, 3, 5)
DEFINE_GDN_BS2_EXTREME(4, gdn_decode_kernel_bs2_extreme_lb2, 2, 2, 4)
DEFINE_GDN_BS2_EXTREME(4, gdn_decode_kernel_bs2_extreme_lb4, 4, 4, 6)
DEFINE_GDN_BS2_EXTREME(4, gdn_decode_kernel_bs2_extreme_lb6, 6, 6, 8)
DEFINE_GDN_BS2_EXTREME(2, gdn_decode_kernel_bs2_extreme_nvb2, 3, 3, 5)
DEFINE_GDN_BS2_EXTREME(2, gdn_decode_kernel_bs2_extreme_nvb2_lb2, 2, 2, 4)

#undef DEFINE_GDN_BS2_EXTREME

#define DEFINE_GDN_BS16_EXTREME(NAME, LB_WAVES, WPE_MIN, WPE_MAX) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB_WAVES) \
__attribute__((amdgpu_waves_per_eu(WPE_MIN, WPE_MAX))) \
void NAME( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__          A_log, \
    const int* __restrict__            indices, \
    float* __restrict__                state, \
    __hip_bfloat16* __restrict__       output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_iasm_sq1_body_opt<2, 8, 1, false, 16, false>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, state, output, \
        batch_size, scale); \
}

DEFINE_GDN_BS16_EXTREME(gdn_decode_kernel_bs16_extreme, 4, 4, 6)
DEFINE_GDN_BS16_EXTREME(gdn_decode_kernel_bs16_extreme_lb2, 2, 2, 4)
DEFINE_GDN_BS16_EXTREME(gdn_decode_kernel_bs16_extreme_lb3, 3, 3, 5)
DEFINE_GDN_BS16_EXTREME(gdn_decode_kernel_bs16_extreme_lb6, 6, 6, 8)
DEFINE_GDN_BS16_EXTREME(gdn_decode_kernel_bs16_extreme_lb1, 1, 1, 2)

#undef DEFINE_GDN_BS16_EXTREME

__global__
__launch_bounds__(BLOCK_THREADS, 3)
__attribute__((amdgpu_waves_per_eu(3, 5)))
void gdn_decode_kernel_iasm_tp8_nvh8_bs16_nvb1_lb3(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    gdn_decode_kernel_iasm_sq1_body_opt<2, 8, 1, false, 0, false>(
        query, key, value, a_input, b_input, dt_bias, A_log, indices, state, output, batch_size, scale);
}

// BS=16: sq1_body with FIXED_BATCH_SIZE=16 (__builtin_assume on batch_idx) + launch_bounds sweep
#define DEFINE_BS16_NVB1_SQ1_FIXED(NAME, LB_WAVES, WPE_MIN, WPE_MAX) \
__global__ \
__launch_bounds__(BLOCK_THREADS, LB_WAVES) \
__attribute__((amdgpu_waves_per_eu(WPE_MIN, WPE_MAX))) \
void NAME( \
    const __hip_bfloat16* __restrict__ query, \
    const __hip_bfloat16* __restrict__ key, \
    const __hip_bfloat16* __restrict__ value, \
    const __hip_bfloat16* __restrict__ a_input, \
    const __hip_bfloat16* __restrict__ b_input, \
    const __hip_bfloat16* __restrict__ dt_bias, \
    const float* __restrict__          A_log, \
    const int* __restrict__            indices, \
    float* __restrict__                state, \
    __hip_bfloat16* __restrict__       output, \
    int batch_size, float scale \
) { \
    gdn_decode_kernel_iasm_sq1_body<2, 8, 1, 16>( \
        query, key, value, a_input, b_input, dt_bias, A_log, indices, state, output, batch_size, scale); \
}

DEFINE_BS16_NVB1_SQ1_FIXED(gdn_decode_kernel_iasm_tp8_nvh8_bs16_nvb1_lb2_fixedbs, 2, 2, 4)
DEFINE_BS16_NVB1_SQ1_FIXED(gdn_decode_kernel_iasm_tp8_nvh8_bs16_nvb1_lb3_fixedbs, 3, 3, 5)
DEFINE_BS16_NVB1_SQ1_FIXED(gdn_decode_kernel_iasm_tp8_nvh8_bs16_nvb1_lb4_fixedbs, 4, 4, 6)
DEFINE_BS16_NVB1_SQ1_FIXED(gdn_decode_kernel_iasm_tp8_nvh8_bs16_nvb1_lb6_fixedbs, 6, 6, 8)

#undef DEFINE_BS16_NVB1_SQ1_FIXED

// ── Sorted variants: accept batch_perm for indirect I/O indexing ──
// indices must be pre-sorted; batch_perm maps sorted position → original batch position
__global__
__launch_bounds__(BLOCK_THREADS, 4)
__attribute__((amdgpu_waves_per_eu(4, 6)))
void gdn_decode_kernel_iasm_tp8_nvh8_sq1_sorted(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    const int* __restrict__            batch_perm,
    int batch_size, float scale
) {
    gdn_decode_kernel_iasm_sq1_body<2, 8, 2>(
        query, key, value, a_input, b_input, dt_bias, A_log, indices,
        state, output, batch_perm, batch_size, scale);
}

__global__
__launch_bounds__(BLOCK_THREADS, 4)
__attribute__((amdgpu_waves_per_eu(4, 6)))
void gdn_decode_kernel_iasm_tp8_nvh8_sq1_nvb1_lb4_sorted(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    const int* __restrict__            batch_perm,
    int batch_size, float scale
) {
    gdn_decode_kernel_iasm_sq1_body<2, 8, 1>(
        query, key, value, a_input, b_input, dt_bias, A_log, indices,
        state, output, batch_perm, batch_size, scale);
}

__global__
__launch_bounds__(BLOCK_THREADS, 3)
__attribute__((amdgpu_waves_per_eu(3, 5)))
void gdn_decode_kernel_iasm_full_sq1(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, float scale
) {
    gdn_decode_kernel_iasm_sq1_body<16, 32, 1>(
        query, key, value, a_input, b_input, dt_bias, A_log, indices, state, output, batch_size, scale);
}

// ============================================================
// Launchers
// ============================================================
extern "C" {

#define LAUNCH(KERNEL, NVB, L2) \
   hipLaunchKernelGGL(( KERNEL<NVB, L2>), dim3(grid), dim3(block), 0, stream,  \
        q, k, v, a, b, dt, al, idx, st, o, batch_size, seq_length, scale)

void launch_gdn_decode_tuned(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    hipStream_t stream
) {
    dim3 grid(batch_size * NUM_V_HEADS * num_v_blocks);
    dim3 block(BLOCK_THREADS);
    auto q = reinterpret_cast<const __hip_bfloat16*>(query);
    auto k = reinterpret_cast<const __hip_bfloat16*>(key);
    auto v = reinterpret_cast<const __hip_bfloat16*>(value);
    auto a = reinterpret_cast<const __hip_bfloat16*>(a_input);
    auto b = reinterpret_cast<const __hip_bfloat16*>(b_input);
    auto dt = reinterpret_cast<const __hip_bfloat16*>(dt_bias);
    auto al = reinterpret_cast<const float*>(A_log);
    auto idx = reinterpret_cast<const int*>(indices);
    auto st = reinterpret_cast<float*>(state);
    auto o = reinterpret_cast<__hip_bfloat16*>(output);

    if (num_v_blocks == 1) {
        if (use_qk_l2norm) LAUNCH(gdn_decode_kernel_tuned, 1, true);
        else               LAUNCH(gdn_decode_kernel_tuned, 1, false);
    } else {
        if (use_qk_l2norm) LAUNCH(gdn_decode_kernel_tuned, 2, true);
        else               LAUNCH(gdn_decode_kernel_tuned, 2, false);
    }
}

} // close extern "C" for template dispatch

template <int NKH, int NVH>
static void dispatch_iasm_generic(
    const __hip_bfloat16* q, const __hip_bfloat16* k, const __hip_bfloat16* v,
    const __hip_bfloat16* a, const __hip_bfloat16* b, const __hip_bfloat16* dt,
    const float* al, const int* idx, float* st, __hip_bfloat16* o,
    int batch_size, int seq_length, int num_v_blocks, bool use_qk_l2norm,
    float scale, hipStream_t stream
) {
    dim3 grid(batch_size * NVH * num_v_blocks);
    dim3 block(BLOCK_THREADS);
    #define LAUNCH_IASM(NVB, L2) \
        hipLaunchKernelGGL((gdn_decode_kernel_iasm<NVB, L2, NKH, NVH>), \
            dim3(grid), dim3(block), 0, stream, \
            q, k, v, a, b, dt, al, idx, st, o, batch_size, seq_length, scale)
    if (num_v_blocks == 1) {
        if (use_qk_l2norm) LAUNCH_IASM(1, true);
        else               LAUNCH_IASM(1, false);
    } else {
        if (use_qk_l2norm) LAUNCH_IASM(2, true);
        else               LAUNCH_IASM(2, false);
    }
    #undef LAUNCH_IASM
}

extern "C"
void launch_gdn_decode_iasm_generic(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
) {
    auto q = reinterpret_cast<const __hip_bfloat16*>(query);
    auto k = reinterpret_cast<const __hip_bfloat16*>(key);
    auto v = reinterpret_cast<const __hip_bfloat16*>(value);
    auto a = reinterpret_cast<const __hip_bfloat16*>(a_input);
    auto b = reinterpret_cast<const __hip_bfloat16*>(b_input);
    auto dt = reinterpret_cast<const __hip_bfloat16*>(dt_bias);
    auto al = reinterpret_cast<const float*>(A_log);
    auto idx = reinterpret_cast<const int*>(indices);
    auto st = reinterpret_cast<float*>(state);
    auto o = reinterpret_cast<__hip_bfloat16*>(output);

    if (num_k_heads == 16 && num_v_heads == 32)
        dispatch_iasm_generic<16,32>(q,k,v,a,b,dt,al,idx,st,o,batch_size,seq_length,num_v_blocks,use_qk_l2norm,scale,stream);
    else if (num_k_heads == 2 && num_v_heads == 4)
        dispatch_iasm_generic<2,4>(q,k,v,a,b,dt,al,idx,st,o,batch_size,seq_length,num_v_blocks,use_qk_l2norm,scale,stream);
    else if (num_k_heads == 4 && num_v_heads == 8)
        dispatch_iasm_generic<4,8>(q,k,v,a,b,dt,al,idx,st,o,batch_size,seq_length,num_v_blocks,use_qk_l2norm,scale,stream);
    else if (num_k_heads == 2 && num_v_heads == 8)
        dispatch_iasm_generic<2,8>(q,k,v,a,b,dt,al,idx,st,o,batch_size,seq_length,num_v_blocks,use_qk_l2norm,scale,stream);
    else if (num_k_heads == 8 && num_v_heads == 16)
        dispatch_iasm_generic<8,16>(q,k,v,a,b,dt,al,idx,st,o,batch_size,seq_length,num_v_blocks,use_qk_l2norm,scale,stream);
}

extern "C"
void launch_gdn_decode_iasm(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    const int* batch_perm,
    hipStream_t stream
) {
    auto q = reinterpret_cast<const __hip_bfloat16*>(query);
    auto k = reinterpret_cast<const __hip_bfloat16*>(key);
    auto v = reinterpret_cast<const __hip_bfloat16*>(value);
    auto a = reinterpret_cast<const __hip_bfloat16*>(a_input);
    auto b = reinterpret_cast<const __hip_bfloat16*>(b_input);
    auto dt = reinterpret_cast<const __hip_bfloat16*>(dt_bias);
    auto al = reinterpret_cast<const float*>(A_log);
    auto idx = reinterpret_cast<const int*>(indices);
    auto st = reinterpret_cast<float*>(state);
    auto o = reinterpret_cast<__hip_bfloat16*>(output);

    if (seq_length == 1 && num_v_blocks == 1 && use_qk_l2norm) {
        if (num_k_heads == 2 && num_v_heads == 4) {
            if (batch_size <= 1) {
                dim3 grid(batch_size * 4 * 4);
                dim3 block(BLOCK_THREADS);
                hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_sq1_nvb4_lb10v48, dim3(grid), dim3(block), 0, stream,
                    q, k, v, a, b, dt, al, idx, st, o, batch_size, scale);
            } else if (batch_size <= 16) {
                dim3 grid(batch_size * 4 * 2);
                dim3 block(BLOCK_THREADS);
                hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_sq1, dim3(grid), dim3(block), 0, stream,
                    q, k, v, a, b, dt, al, idx, st, o, batch_size, scale);
            } else {
                dim3 grid(batch_size * 4);
                dim3 block(BLOCK_THREADS);
                hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_sq1_nvb1, dim3(grid), dim3(block), 0, stream,
                    q, k, v, a, b, dt, al, idx, st, o, batch_size, scale);
            }
            return;
        }
        if (num_k_heads == 2 && num_v_heads == 8) {
            if (batch_perm) {
                if (batch_size <= 4) {
                    dim3 grid(batch_size * 8 * 2);
                    dim3 block(BLOCK_THREADS);
                    hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_nvh8_sq1_sorted, dim3(grid), dim3(block), 0, stream,
                        q, k, v, a, b, dt, al, idx, st, o, batch_perm, batch_size, scale);
                } else {
                    dim3 grid(batch_size * 8);
                    dim3 block(BLOCK_THREADS);
                    hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_nvh8_sq1_nvb1_lb4_sorted, dim3(grid), dim3(block), 0, stream,
                        q, k, v, a, b, dt, al, idx, st, o, batch_perm, batch_size, scale);
                }
            } else {
                if (batch_size == 2) {
                    // BS=2 cache-bypass dispatch.
                    // In `online` GDN sits between in_proj/conv (heavy L2 footprint) and
                    // rmsnorm/out_proj.  Without bypass, our 1MB state read/write thrashes
                    // L2 ↔ companion ops, costing ~0.5us/replay.  Marking state load+store
                    // nontemporal (HBM streaming) keeps L2 lines for companion and gives
                    // ~+8% over FlyDSL (vs ~6% baseline) without changing data layout.
                    //
                    // Default V=3 = _v52_lb10 + NT load + NT store (best stable choice;
                    // V=5/V=7 share identical median but show occasional tail outliers).
                    // V=0 falls back to the original cached-load kernel.
                    static int bs2v = -1;
                    if (bs2v < 0) {
                        const char* e = getenv("GDN_BS2_VARIANT");
                        bs2v = e ? atoi(e) : 3;
                    }
                    dim3 block(BLOCK_THREADS);
                    dim3 grid_nvb4(batch_size * 8 * 4);
                    switch (bs2v) {
                        case 0: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v52_lb10,         dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 1: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v52_lb10_ntload,  dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 2: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v52_lb10_ntstore, dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 4: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v48_lb10_ntload,  dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 5: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v48_lb10_ntboth,  dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 6: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v56_lb9_ntload,   dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 7: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v56_lb9_ntboth,   dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        default: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v52_lb10_ntboth, dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale);
                    }
                } else if (batch_size == 1) {
                    // BS=1 dedicated dispatch (split off from the BS<=4 path so
                    // experiments here don't perturb BS=3/4).  Production trace
                    // A/B (sglang Qwen3.5 397B, scheme #4):
                    //   V=0  builtin  v52_lb10            mean=5.83 std=0.13 us  ← default (beats FlyDSL 6.02us by +3.2%)
                    //   V=1  l2pref   v52_lb10            mean=5.95 std=0.16 us  (min 4.97 best, but adds 4 ASM-load
                    //                                      cycles that hurt the typical warm-L2 case)
                    //   V=2  l2pref   v48_lb10            (lower VGPR experiment)
                    //   V=3  l2pref   v56_lb9             (higher VGPR experiment)
                    //   V=4  prefetch v52_lb10            (compute-prefetch body, scalars-first ordering)
                    //   V=5  builtin  v52_lb10 + nt-load  (NT-load skips L2 cache fill; for cold-L2 ablation only)
                    //
                    // 2026-04 re-A/B with sglang_dev + Scheme #4 + new
                    // qkvzba_split→conv1d→GDN pattern: state SSM is in fact
                    // COLD when GDN starts (the in_proj GEMM dirties ~10MB and
                    // qkvzba_split scatters into another ~10MB, both fully
                    // thrashing the per-XCD 4MB L2 between consecutive same-
                    // layer GDN calls).  4-trial online microbench BS=1
                    // (median across 4 process-isolated trials):
                    //   V=0  builtin  v52_lb10                  med 5.58  (old default)
                    //   V=4  prefetch_body v52_lb10             med 5.39  (-3.4% — scalar-first ordering hides state HBM)
                    //   V=5  builtin  v52_lb10 ntload           med 5.59  (no effect)
                    //   V=6  builtin  v52_lb10 ntboth           med 5.57  (sensitive to compiler scheduling — unstable)
                    //   V=7  builtin  v52_lb10 ntstore          med 5.60  (no effect)
                    //   V=8  prefetch_body v52_lb10 ntboth      med 5.37  (-3.8%, all 4 trials = 5.37) ← new default
                    //   V=9  prefetch_body v52_lb10 ntload      med 5.56
                    //   V=10 prefetch_body v52_lb10 ntstore     med 5.40
                    // V=8 stacks two orthogonal levers: scalar/SFU compute
                    // before VMEM state load (overlaps gating exp/log/rcp with
                    // the cold HBM load) AND nontemporal load+store on state
                    // (skips L2-fill traffic since the next access to this slot
                    // is ~518us later after 21 other layers have flushed L2).
                    // It is also the most reproducible variant across rebuilds.
                    static int bs1v = -1;
                    if (bs1v < 0) {
                        const char* e = getenv("GDN_BS1_VARIANT");
                        bs1v = e ? atoi(e) : 8;
                    }
                    dim3 block(BLOCK_THREADS);
                    dim3 grid_nvb4(batch_size * 8 * 4);
                    switch (bs1v) {
                        case 1: hipLaunchKernelGGL(gdn_decode_kernel_l2pref_nvh8_nvb4_v52_lb10,    dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 2: hipLaunchKernelGGL(gdn_decode_kernel_l2pref_nvh8_nvb4_v48_lb10,    dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 3: hipLaunchKernelGGL(gdn_decode_kernel_l2pref_nvh8_nvb4_v56_lb9,     dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 4: hipLaunchKernelGGL(gdn_decode_kernel_prefetch_nvh8_nvb4_v52_lb10,  dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 5: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v52_lb10_ntload, dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 6: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v52_lb10_ntboth,  dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 7: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v52_lb10_ntstore, dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 8: hipLaunchKernelGGL(gdn_decode_kernel_prefetch_nvh8_nvb4_v52_lb10_ntboth,  dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 9: hipLaunchKernelGGL(gdn_decode_kernel_prefetch_nvh8_nvb4_v52_lb10_ntload,  dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 10: hipLaunchKernelGGL(gdn_decode_kernel_prefetch_nvh8_nvb4_v52_lb10_ntstore, dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        default: hipLaunchKernelGGL(gdn_decode_kernel_prefetch_nvh8_nvb4_v52_lb10_ntboth,  dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale);
                    }
                } else if (batch_size <= 4) {
                    static int bs4v = -1;
                    if (bs4v < 0) {
                        const char* e = getenv("GDN_BS4_VARIANT");
                        bs4v = e ? atoi(e) : 0;
                    }
                    dim3 block(BLOCK_THREADS);
                    dim3 grid_nvb4(batch_size * 8 * 4);
                    dim3 grid_nvb1(batch_size * 8);
                    switch (bs4v) {
                        case 1: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v48_lb10,    dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 2: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v56_lb9,     dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 3: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v44_lb12,    dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 4: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v60_lb9,     dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 5: hipLaunchKernelGGL(gdn_decode_kernel_pureasm_nvh8_nvb4_v52_lb10,    dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 6: hipLaunchKernelGGL(gdn_decode_kernel_nowait_nvh8_nvb4_v52_lb10,     dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 7: hipLaunchKernelGGL(gdn_decode_kernel_prefetch_nvh8_nvb4_v52_lb10,   dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 8: hipLaunchKernelGGL(gdn_decode_kernel_pkfma_nvh8_nvb4_v52_lb10,      dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 9: hipLaunchKernelGGL(gdn_decode_kernel_exp2_nvh8_nvb4_v52_lb10,       dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 10: hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_nvh8_sq1_nvb1_lb4,   dim3(grid_nvb1), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 11: hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_nvh8_sq1_nvb1,       dim3(grid_nvb1), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        // L2-prefetch variants (cold-L2 mitigation for production BS=1).
                        case 12: hipLaunchKernelGGL(gdn_decode_kernel_l2pref_nvh8_nvb4_v52_lb10,    dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 13: hipLaunchKernelGGL(gdn_decode_kernel_l2pref_nvh8_nvb4_v48_lb10,    dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        case 14: hipLaunchKernelGGL(gdn_decode_kernel_l2pref_nvh8_nvb4_v56_lb9,     dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                        default: hipLaunchKernelGGL(gdn_decode_kernel_builtin_nvh8_nvb4_v52_lb10,   dim3(grid_nvb4), block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale);
                    }
                } else {
                    dim3 grid(batch_size * 8);
                    dim3 block(BLOCK_THREADS);
                    if (batch_size == 16) {
                        static int bs16v = -1;
                        if (bs16v < 0) {
                            const char* e = getenv("GDN_BS16_VARIANT");
                            bs16v = e ? atoi(e) : 9;
                        }
                        switch (bs16v) {
                            case 1: hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_nvh8_sq1_nvb1,                grid, block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                            case 2: hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_nvh8_sq1_nvb1_lb4,            grid, block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                            case 3: hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_nvh8_bs16_nvb1_lb2_fixedbs,   grid, block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                            case 4: hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_nvh8_bs16_nvb1_lb3_fixedbs,   grid, block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                            case 5: hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_nvh8_bs16_nvb1_lb4_fixedbs,   grid, block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                            case 6: hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_nvh8_bs16_nvb1_lb6_fixedbs,   grid, block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                            case 7: hipLaunchKernelGGL(gdn_decode_kernel_bs16_extreme,                          grid, block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                            case 8: hipLaunchKernelGGL(gdn_decode_kernel_bs16_extreme_lb2,                      grid, block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                            case 9: hipLaunchKernelGGL(gdn_decode_kernel_bs16_extreme_lb3,                      grid, block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                            case 10: hipLaunchKernelGGL(gdn_decode_kernel_bs16_extreme_lb6,                     grid, block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale); break;
                            default: hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_nvh8_bs16_nvb1_lb3,          grid, block, 0, stream, q, k, v, a, b, dt, al, idx, st, o, batch_size, scale);
                        }
                    } else if (batch_size <= 14) {
                        hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_nvh8_sq1_nvb1, dim3(grid), dim3(block), 0, stream,
                            q, k, v, a, b, dt, al, idx, st, o, batch_size, scale);
                    } else {
                        hipLaunchKernelGGL(gdn_decode_kernel_iasm_tp8_nvh8_sq1_nvb1_lb4, dim3(grid), dim3(block), 0, stream,
                            q, k, v, a, b, dt, al, idx, st, o, batch_size, scale);
                    }
                }
            }
            return;
        }
        if (num_k_heads == 16 && num_v_heads == 32) {
            dim3 grid(batch_size * 32);
            dim3 block(BLOCK_THREADS);
            hipLaunchKernelGGL(gdn_decode_kernel_iasm_full_sq1, dim3(grid), dim3(block), 0, stream,
                q, k, v, a, b, dt, al, idx, st, o, batch_size, scale);
            return;
        }
    }

    launch_gdn_decode_iasm_generic(
        query, key, value, a_input, b_input, dt_bias, A_log, indices,
        state, output, batch_size, seq_length, num_v_blocks, use_qk_l2norm,
        scale, num_k_heads, num_v_heads, stream);
}

#undef LAUNCH

// ============================================================
// Sglang-compatible TUNED kernel: state layout [pool, HV, K, V]
// (Triton/sglang convention: K outer, V inner)
// ============================================================
template <int NUM_V_BLOCKS, bool USE_QK_L2NORM>
__global__
__launch_bounds__(BLOCK_THREADS, 3)
__attribute__((amdgpu_waves_per_eu(3, 5)))
void gdn_decode_kernel_tuned_kv(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const float* __restrict__          dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, int seq_length, float scale,
    int num_k_heads, int num_v_heads
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;
    const int gva_ratio = num_v_heads / num_k_heads;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / num_v_heads;
    const int hv_idx = b_hv % num_v_heads;
    const int hk_idx = hv_idx / gva_ratio;

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader = (lane_id / WARP_K_THREADS) * WARP_K_THREADS;

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_K_DIM * HEAD_V_DIM;
    float* sb = state + (int64_t)pool_idx * num_v_heads * shs + hv_idx * shs;

    // State load: [K, V] layout — sb[k_idx * HEAD_V_DIM + v_idx]
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                S[vi][ki][j] = sb[(warp_k_vec_start + ki * WARP_TILE_K + j) * HEAD_V_DIM + gv];
    }

    const float r_A_log = A_log[hv_idx];
    const float r_dt_bias = dt_bias[hv_idx];

    float k_reg[K_ITERS][VEC_K];

    for (int sq = 0; sq < seq_length; sq++) {
        const int ab = batch_idx * seq_length * num_v_heads + sq * num_v_heads + hv_idx;
        float x = __bfloat162float(a_input[ab]) + r_dt_bias;
        float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
        float exp_g = __expf(-__expf(r_A_log) * sp);
        float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b_input[ab])));

        const auto* qb = query + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* kb = key + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* vb = value + (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int ko = warp_k_vec_start + ki * WARP_TILE_K;
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                k_reg[ki][j] = __bfloat162float(kb[ko + j]);
        }

        float dot_kq = 0;
        float res_hk[V_ITERS], res_hq[V_ITERS], res_rv[V_ITERS];

        {
            float q_reg[K_ITERS][VEC_K];
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++) {
                int ko = warp_k_vec_start + ki * WARP_TILE_K;
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    q_reg[ki][j] = __bfloat162float(qb[ko + j]);
            }

            if constexpr (USE_QK_L2NORM) {
                float sq_q = 0, sq_k = 0;
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        sq_q += q_reg[ki][j] * q_reg[ki][j];
                        sq_k += k_reg[ki][j] * k_reg[ki][j];
                    }
                sq_q = reduce_sum_k(sq_q); sq_k = reduce_sum_k(sq_k);
                float qs = scale * __frsqrt_rn(sq_q + 1e-6f);
                float ks = __frsqrt_rn(sq_k + 1e-6f);
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        q_reg[ki][j] *= qs;
                        k_reg[ki][j] *= ks;
                    }
            } else {
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++)
                        q_reg[ki][j] *= scale;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    dot_kq += k_reg[ki][j] * q_reg[ki][j];
            dot_kq = reduce_sum_k(dot_kq);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                res_rv[vi] = __bfloat162float(vb[global_v_base + vi * WARP_GROUP_TILE_V]);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] = 0.0f;
                res_hq[vi] = 0.0f;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++) {
                    float kv = k_reg[ki][j];
                    float qv = q_reg[ki][j];
                    #pragma unroll
                    for (int vi = 0; vi < V_ITERS; vi++) {
                        res_hk[vi] += S[vi][ki][j] * kv;
                        res_hq[vi] += S[vi][ki][j] * qv;
                    }
                }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 1);
                res_hq[vi] += shfl_xor(res_hq[vi], 1);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 2);
                res_hq[vi] += shfl_xor(res_hq[vi], 2);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 4);
                res_hq[vi] += shfl_xor(res_hq[vi], 4);
            }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] *= exp_g;
                res_hq[vi] *= exp_g;
            }
        }

        float vn[V_ITERS];
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            float tmp = (res_rv[vi] - res_hk[vi]) * beta;
            vn[vi] = shfl_idx(tmp, leader);
        }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;

        if (k_tid == 0) {
            const int64_t out_base = (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM
                                     + hv_idx * HEAD_V_DIM + global_v_base;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) {
                float kv = k_reg[ki][j];
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++)
                    S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
            }
    }

    // State store: [K, V] layout
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                sb[(warp_k_vec_start + ki * WARP_TILE_K + j) * HEAD_V_DIM + gv] = S[vi][ki][j];
    }
}

extern "C" {

#define LAUNCH_KV(KERNEL, NVB, L2) \
   hipLaunchKernelGGL(( KERNEL<NVB, L2>), dim3(grid), dim3(block), 0, stream,  \
        q, k, v, a, b, dt, al, idx, st, o, batch_size, seq_length, scale, nkh, nvh)

void launch_gdn_decode_tuned_kv(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
) {
    dim3 grid(batch_size * num_v_heads * num_v_blocks);
    dim3 block(BLOCK_THREADS);
    auto q = reinterpret_cast<const __hip_bfloat16*>(query);
    auto k = reinterpret_cast<const __hip_bfloat16*>(key);
    auto v = reinterpret_cast<const __hip_bfloat16*>(value);
    auto a = reinterpret_cast<const __hip_bfloat16*>(a_input);
    auto b = reinterpret_cast<const __hip_bfloat16*>(b_input);
    auto dt = reinterpret_cast<const float*>(dt_bias);
    auto al = reinterpret_cast<const float*>(A_log);
    auto idx = reinterpret_cast<const int*>(indices);
    auto st = reinterpret_cast<float*>(state);
    auto o = reinterpret_cast<__hip_bfloat16*>(output);
    int nkh = num_k_heads, nvh = num_v_heads;

    if (num_v_blocks == 1) {
        if (use_qk_l2norm) LAUNCH_KV(gdn_decode_kernel_tuned_kv, 1, true);
        else               LAUNCH_KV(gdn_decode_kernel_tuned_kv, 1, false);
    } else {
        if (use_qk_l2norm) LAUNCH_KV(gdn_decode_kernel_tuned_kv, 2, true);
        else               LAUNCH_KV(gdn_decode_kernel_tuned_kv, 2, false);
    }
}

#undef LAUNCH_KV

} // extern "C"

// ============================================================
// TUNED kernel with [V, K] state layout (float4 vectorized) +
// runtime head counts + float32 dt_bias.
// For sglang integration: caller transposes state [K,V]→[V,K]
// before call and [V,K]→[K,V] after call.
// ============================================================
template <int NUM_V_BLOCKS, bool USE_QK_L2NORM>
__global__
__launch_bounds__(BLOCK_THREADS, 3)
__attribute__((amdgpu_waves_per_eu(3, 5)))
void gdn_decode_kernel_tuned_vk(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, int seq_length, float scale,
    int num_k_heads, int num_v_heads
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;
    const int gva_ratio = num_v_heads / num_k_heads;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / num_v_heads;
    const int hv_idx = b_hv % num_v_heads;
    const int hk_idx = hv_idx / gva_ratio;

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader = (lane_id / WARP_K_THREADS) * WARP_K_THREADS;

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * num_v_heads * shs + hv_idx * shs;

    // State load: [V, K] layout with float4 vectorized access
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = *reinterpret_cast<const float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y; S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    const float r_A_log = A_log[hv_idx];
    const float r_dt_bias = __bfloat162float(dt_bias[hv_idx]);

    float k_reg[K_ITERS][VEC_K];

    for (int sq = 0; sq < seq_length; sq++) {
        const int ab = batch_idx * seq_length * num_v_heads + sq * num_v_heads + hv_idx;
        float x = __bfloat162float(a_input[ab]) + r_dt_bias;
        float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
        float exp_g = __expf(-__expf(r_A_log) * sp);
        float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b_input[ab])));

        const auto* qb = query + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* kb = key + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* vb = value + (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int ko = warp_k_vec_start + ki * WARP_TILE_K;
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                k_reg[ki][j] = __bfloat162float(kb[ko + j]);
        }

        float dot_kq = 0;
        float res_hk[V_ITERS], res_hq[V_ITERS], res_rv[V_ITERS];

        {
            float q_reg[K_ITERS][VEC_K];
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++) {
                int ko = warp_k_vec_start + ki * WARP_TILE_K;
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    q_reg[ki][j] = __bfloat162float(qb[ko + j]);
            }

            if constexpr (USE_QK_L2NORM) {
                float sq_q = 0, sq_k = 0;
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        sq_q += q_reg[ki][j] * q_reg[ki][j];
                        sq_k += k_reg[ki][j] * k_reg[ki][j];
                    }
                sq_q = reduce_sum_k(sq_q); sq_k = reduce_sum_k(sq_k);
                float qs = scale * __frsqrt_rn(sq_q + 1e-6f);
                float ks = __frsqrt_rn(sq_k + 1e-6f);
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        q_reg[ki][j] *= qs;
                        k_reg[ki][j] *= ks;
                    }
            } else {
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++)
                        q_reg[ki][j] *= scale;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    dot_kq += k_reg[ki][j] * q_reg[ki][j];
            dot_kq = reduce_sum_k(dot_kq);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                res_rv[vi] = __bfloat162float(vb[global_v_base + vi * WARP_GROUP_TILE_V]);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] = 0.0f;
                res_hq[vi] = 0.0f;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++) {
                    float kv = k_reg[ki][j];
                    float qv = q_reg[ki][j];
                    #pragma unroll
                    for (int vi = 0; vi < V_ITERS; vi++) {
                        res_hk[vi] += S[vi][ki][j] * kv;
                        res_hq[vi] += S[vi][ki][j] * qv;
                    }
                }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 1);
                res_hq[vi] += shfl_xor(res_hq[vi], 1);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 2);
                res_hq[vi] += shfl_xor(res_hq[vi], 2);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 4);
                res_hq[vi] += shfl_xor(res_hq[vi], 4);
            }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] *= exp_g;
                res_hq[vi] *= exp_g;
            }
        }

        float vn[V_ITERS];
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            float tmp = (res_rv[vi] - res_hk[vi]) * beta;
            vn[vi] = shfl_idx(tmp, leader);
        }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;

        if (k_tid == 0) {
            const int64_t out_base = (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM
                                     + hv_idx * HEAD_V_DIM + global_v_base;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) {
                float kv = k_reg[ki][j];
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++)
                    S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
            }
    }

    // State store: [V, K] layout with float4
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

extern "C" {

#define LAUNCH_VK(KERNEL, NVB, L2) \
   hipLaunchKernelGGL(( KERNEL<NVB, L2>), dim3(grid), dim3(block), 0, stream,  \
        q, k, v, a, b, dt, al, idx, st, o, batch_size, seq_length, scale, nkh, nvh)

void launch_gdn_decode_tuned_vk(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
) {
    dim3 grid(batch_size * num_v_heads * num_v_blocks);
    dim3 block(BLOCK_THREADS);
    auto q = reinterpret_cast<const __hip_bfloat16*>(query);
    auto k = reinterpret_cast<const __hip_bfloat16*>(key);
    auto v = reinterpret_cast<const __hip_bfloat16*>(value);
    auto a = reinterpret_cast<const __hip_bfloat16*>(a_input);
    auto b = reinterpret_cast<const __hip_bfloat16*>(b_input);
    auto dt = reinterpret_cast<const __hip_bfloat16*>(dt_bias);
    auto al = reinterpret_cast<const float*>(A_log);
    auto idx = reinterpret_cast<const int*>(indices);
    auto st = reinterpret_cast<float*>(state);
    auto o = reinterpret_cast<__hip_bfloat16*>(output);
    int nkh = num_k_heads, nvh = num_v_heads;

    if (num_v_blocks == 1) {
        if (use_qk_l2norm) LAUNCH_VK(gdn_decode_kernel_tuned_vk, 1, true);
        else               LAUNCH_VK(gdn_decode_kernel_tuned_vk, 1, false);
    } else {
        if (use_qk_l2norm) LAUNCH_VK(gdn_decode_kernel_tuned_vk, 2, true);
        else               LAUNCH_VK(gdn_decode_kernel_tuned_vk, 2, false);
    }
}

#undef LAUNCH_VK

} // extern "C"

// ============================================================
// Fast in-place 128×128 float32 matrix transpose.
// One block per (matrix, row). Each block handles swaps for one
// row: swap mat[row][col] with mat[col][row] for col > row.
// 512 blocks for BS=1 HV=4 → good GPU utilization.
// ============================================================
constexpr int TRANS_THREADS = 128;

__global__
void state_transpose_128x128_kernel(
    float* __restrict__ state,
    const int* __restrict__ indices,
    int batch_size,
    int num_v_heads
) {
    constexpr int N = 128;
    const int global_idx = blockIdx.x;
    const int mat_idx = global_idx / N;
    const int row = global_idx % N;
    const int batch_idx = mat_idx / num_v_heads;
    const int hv_idx = mat_idx % num_v_heads;
    if (batch_idx >= batch_size) return;

    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    float* mat = state + (int64_t)pool_idx * num_v_heads * N * N + (int64_t)hv_idx * N * N;

    for (int col = row + 1 + threadIdx.x; col < N; col += TRANS_THREADS) {
        int idx1 = row * N + col;
        int idx2 = col * N + row;
        float tmp = mat[idx1];
        mat[idx1] = mat[idx2];
        mat[idx2] = tmp;
    }
}

extern "C" {

void launch_state_transpose(
    void* state, const void* indices,
    int batch_size, int num_v_heads,
    hipStream_t stream
) {
    dim3 grid(batch_size * num_v_heads * 128);
    dim3 block(TRANS_THREADS);
    hipLaunchKernelGGL(state_transpose_128x128_kernel,
        grid, block, 0, stream,
        reinterpret_cast<float*>(state),
        reinterpret_cast<const int*>(indices),
        batch_size, num_v_heads);
}

} // extern "C" (transpose)

// ============================================================
// Multi-layer in-place 128x128 transpose with per-slot layout
// gating. One launch handles ALL Qwen3.5 GDN layers (48) in one
// shot, eliminating the previous per-layer prologue+epilogue
// pattern that triggered 96 launches per prefill chunk.
// state_base layout: [num_layers, slots+1, num_v_heads, N, N]
// slot_layout: int8 array of size >= max(pool_idx)+1
//   value 0 == KV (FLA prefill layout)
//   value 1 == VK (HIP decode layout)
// Each block reads slot_layout[pool_idx] once; if it already
// matches target_layout, the block early-exits (no transpose
// performed). This makes the kernel idempotent and lets us
// always launch unconditionally without CPU sync.
// ============================================================
__global__
void state_transpose_128x128_multi_layer_kernel(
    float* __restrict__ state_base,
    const int* __restrict__ indices,
    const int8_t* __restrict__ slot_layout,
    int target_layout,
    int num_layers,
    int batch_size,
    int num_v_heads,
    int64_t layer_stride_floats
) {
    constexpr int N = 128;
    const int per_layer_blocks = batch_size * num_v_heads * N;
    const int gidx = blockIdx.x;
    const int layer_idx = gidx / per_layer_blocks;
    const int within = gidx - layer_idx * per_layer_blocks;
    const int mat_idx = within / N;
    const int row = within - mat_idx * N;
    const int batch_idx = mat_idx / num_v_heads;
    const int hv_idx = mat_idx - batch_idx * num_v_heads;
    if (layer_idx >= num_layers) return;
    if (batch_idx >= batch_size) return;

    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    if ((int)slot_layout[pool_idx] == target_layout) return;

    float* mat = state_base
        + (int64_t)layer_idx * layer_stride_floats
        + (int64_t)pool_idx * num_v_heads * N * N
        + (int64_t)hv_idx * N * N;

    for (int col = row + 1 + threadIdx.x; col < N; col += TRANS_THREADS) {
        int idx1 = row * N + col;
        int idx2 = col * N + row;
        float tmp = mat[idx1];
        mat[idx1] = mat[idx2];
        mat[idx2] = tmp;
    }
}

extern "C" {

void launch_state_transpose_multi_layer(
    void* state_base,
    const void* indices,
    const void* slot_layout,
    int target_layout,
    int num_layers,
    int batch_size,
    int num_v_heads,
    int64_t layer_stride_floats,
    hipStream_t stream
) {
    constexpr int N = 128;
    const unsigned grid_x =
        (unsigned)num_layers * (unsigned)batch_size * (unsigned)num_v_heads * (unsigned)N;
    dim3 grid(grid_x);
    dim3 block(TRANS_THREADS);
    hipLaunchKernelGGL(state_transpose_128x128_multi_layer_kernel,
        grid, block, 0, stream,
        reinterpret_cast<float*>(state_base),
        reinterpret_cast<const int*>(indices),
        reinterpret_cast<const int8_t*>(slot_layout),
        target_layout, num_layers, batch_size, num_v_heads,
        layer_stride_floats);
}

} // extern "C" (multi-layer transpose)

// ============================================================
// KV4 kernel: works directly on [K, V] state layout (sglang).
// float4 vectorized loads along V dimension (V is innermost
// in [K,V]). No transpose needed.
// Thread mapping: WARP_K_THREADS=8 (scalar K), WARP_V_THREADS=8
// Each V-thread loads VEC_V=4 consecutive V values via float4.
// ============================================================
constexpr int KV4_VEC_V = 4;
constexpr int KV4_WARP_TILE_V = WARP_V_THREADS * KV4_VEC_V;  // 8*4=32
constexpr int KV4_WARP_TILE_K = WARP_K_THREADS;               // 8 (scalar)
constexpr int KV4_K_ITERS = HEAD_K_DIM / KV4_WARP_TILE_K;     // 128/8=16

template <int NUM_V_BLOCKS, bool USE_QK_L2NORM>
__global__
__launch_bounds__(BLOCK_THREADS, 3)
__attribute__((amdgpu_waves_per_eu(3, 5)))
void gdn_decode_kernel_kv4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const float* __restrict__          dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, int seq_length, float scale,
    int num_k_heads, int num_v_heads
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * KV4_WARP_TILE_V;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;
    const int gva_ratio = num_v_heads / num_k_heads;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / num_v_heads;
    const int hv_idx = b_hv % num_v_heads;
    const int hk_idx = hv_idx / gva_ratio;

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int v_base = v_block_id * TILE_V + warp_id * KV4_WARP_TILE_V + v_tid * KV4_VEC_V;
    const int leader = (lane_id / WARP_K_THREADS) * WARP_K_THREADS;

    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * num_v_heads * shs + hv_idx * shs;

    float S[V_ITERS][KV4_K_ITERS][KV4_VEC_V];

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < KV4_K_ITERS; ki++) {
            int k_pos = k_tid + ki * KV4_WARP_TILE_K;
            float4 t = *reinterpret_cast<const float4*>(sb + k_pos * HEAD_V_DIM + gv);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y; S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    const float r_A_log = A_log[hv_idx];
    const float r_dt_bias = dt_bias[hv_idx];

    for (int sq = 0; sq < seq_length; sq++) {
        const int ab = batch_idx * seq_length * num_v_heads + sq * num_v_heads + hv_idx;
        float x = __bfloat162float(a_input[ab]) + r_dt_bias;
        float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
        float exp_g = __expf(-__expf(r_A_log) * sp);
        float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b_input[ab])));

        const auto* qb = query + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* kb = key + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* vb = value + (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

        float q_reg[KV4_K_ITERS], k_reg[KV4_K_ITERS];
        #pragma unroll
        for (int ki = 0; ki < KV4_K_ITERS; ki++) {
            int k_pos = k_tid + ki * KV4_WARP_TILE_K;
            q_reg[ki] = __bfloat162float(qb[k_pos]);
            k_reg[ki] = __bfloat162float(kb[k_pos]);
        }

        if constexpr (USE_QK_L2NORM) {
            float sq_q = 0, sq_k = 0;
            #pragma unroll
            for (int ki = 0; ki < KV4_K_ITERS; ki++) {
                sq_q += q_reg[ki] * q_reg[ki];
                sq_k += k_reg[ki] * k_reg[ki];
            }
            sq_q = reduce_sum_k(sq_q); sq_k = reduce_sum_k(sq_k);
            float qs = scale * __frsqrt_rn(sq_q + 1e-6f);
            float ks = __frsqrt_rn(sq_k + 1e-6f);
            #pragma unroll
            for (int ki = 0; ki < KV4_K_ITERS; ki++) {
                q_reg[ki] *= qs;
                k_reg[ki] *= ks;
            }
        } else {
            #pragma unroll
            for (int ki = 0; ki < KV4_K_ITERS; ki++)
                q_reg[ki] *= scale;
        }

        float dot_kq = 0;
        #pragma unroll
        for (int ki = 0; ki < KV4_K_ITERS; ki++)
            dot_kq += q_reg[ki] * k_reg[ki];
        dot_kq = reduce_sum_k(dot_kq);

        float res_rv[V_ITERS][KV4_VEC_V];
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            int gv = v_base + vi * WARP_GROUP_TILE_V;
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++)
                res_rv[vi][jv] = __bfloat162float(vb[gv + jv]);
        }

        float res_hk[V_ITERS][KV4_VEC_V];
        float res_hq[V_ITERS][KV4_VEC_V];

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++) {
                res_hk[vi][jv] = 0.0f;
                res_hq[vi][jv] = 0.0f;
            }

        #pragma unroll
        for (int ki = 0; ki < KV4_K_ITERS; ki++) {
            float kv = k_reg[ki];
            float qv = q_reg[ki];
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                #pragma unroll
                for (int jv = 0; jv < KV4_VEC_V; jv++) {
                    res_hk[vi][jv] += S[vi][ki][jv] * kv;
                    res_hq[vi][jv] += S[vi][ki][jv] * qv;
                }
        }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++) {
                res_hk[vi][jv] += shfl_xor(res_hk[vi][jv], 1);
                res_hq[vi][jv] += shfl_xor(res_hq[vi][jv], 1);
            }
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++) {
                res_hk[vi][jv] += shfl_xor(res_hk[vi][jv], 2);
                res_hq[vi][jv] += shfl_xor(res_hq[vi][jv], 2);
            }
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++) {
                res_hk[vi][jv] += shfl_xor(res_hk[vi][jv], 4);
                res_hq[vi][jv] += shfl_xor(res_hq[vi][jv], 4);
            }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++) {
                res_hk[vi][jv] *= exp_g;
                res_hq[vi][jv] *= exp_g;
            }

        float vn[V_ITERS][KV4_VEC_V];
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++) {
                float tmp = (res_rv[vi][jv] - res_hk[vi][jv]) * beta;
                vn[vi][jv] = shfl_idx(tmp, leader);
            }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++)
                res_hq[vi][jv] = res_hq[vi][jv] + vn[vi][jv] * dot_kq;

        if (k_tid == 0) {
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                int gv = v_base + vi * WARP_GROUP_TILE_V;
                const int64_t out_base = (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM
                                         + hv_idx * HEAD_V_DIM + gv;
                #pragma unroll
                for (int jv = 0; jv < KV4_VEC_V; jv++)
                    output[out_base + jv] = __float2bfloat16(res_hq[vi][jv]);
            }
        }

        #pragma unroll
        for (int ki = 0; ki < KV4_K_ITERS; ki++) {
            float kv = k_reg[ki];
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                #pragma unroll
                for (int jv = 0; jv < KV4_VEC_V; jv++)
                    S[vi][ki][jv] = S[vi][ki][jv] * exp_g + kv * vn[vi][jv];
        }
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < KV4_K_ITERS; ki++) {
            int k_pos = k_tid + ki * KV4_WARP_TILE_K;
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + k_pos * HEAD_V_DIM + gv) = t;
        }
    }
}

extern "C" {

#define LAUNCH_KV4(KERNEL, NVB, L2) \
   hipLaunchKernelGGL(( KERNEL<NVB, L2>), dim3(grid), dim3(block), 0, stream,  \
        q, k, v, a, b, dt, al, idx, st, o, batch_size, seq_length, scale, nkh, nvh)

void launch_gdn_decode_kv4(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
) {
    dim3 grid(batch_size * num_v_heads * num_v_blocks);
    dim3 block(BLOCK_THREADS);
    auto q = reinterpret_cast<const __hip_bfloat16*>(query);
    auto k = reinterpret_cast<const __hip_bfloat16*>(key);
    auto v = reinterpret_cast<const __hip_bfloat16*>(value);
    auto a = reinterpret_cast<const __hip_bfloat16*>(a_input);
    auto b = reinterpret_cast<const __hip_bfloat16*>(b_input);
    auto dt = reinterpret_cast<const float*>(dt_bias);
    auto al = reinterpret_cast<const float*>(A_log);
    auto idx = reinterpret_cast<const int*>(indices);
    auto st = reinterpret_cast<float*>(state);
    auto o = reinterpret_cast<__hip_bfloat16*>(output);
    int nkh = num_k_heads, nvh = num_v_heads;

    (void)num_v_blocks;
    if (use_qk_l2norm) LAUNCH_KV4(gdn_decode_kernel_kv4, 1, true);
    else               LAUNCH_KV4(gdn_decode_kernel_kv4, 1, false);
}

#undef LAUNCH_KV4

} // extern "C"

// ============================================================
// FUSED kernel: reads [K,V] state, transposes in LDS, computes
// using VK-style float4-along-K, writes back [K,V].
// Uses 64KB LDS (full MI300X budget). 1 block per CU max.
// 256 threads load 64KB cooperatively via float4.
// ============================================================
constexpr int LDS_K = HEAD_K_DIM;     // 128
constexpr int LDS_V = HEAD_V_DIM;     // 128
constexpr int LDS_STRIDE = LDS_K;     // 128 floats per V-row (no pad, exactly 64KB)

template <bool USE_QK_L2NORM>
__global__
__launch_bounds__(BLOCK_THREADS, 1)
void gdn_decode_kernel_fused(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const float* __restrict__          dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, int seq_length, float scale,
    int num_k_heads, int num_v_heads
) {
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;  // 4*8=32
    constexpr int V_ITERS = HEAD_V_DIM / WARP_GROUP_TILE_V;       // 128/32=4
    const int gva_ratio = num_v_heads / num_k_heads;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv = blockIdx.x;
    const int batch_idx = b_hv / num_v_heads;
    const int hv_idx = b_hv % num_v_heads;
    const int hk_idx = hv_idx / gva_ratio;

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = warp_id * WARP_V_THREADS + v_tid;
    const int leader = (lane_id / WARP_K_THREADS) * WARP_K_THREADS;

    // Global state pointer: [pool, HV, K, V] layout
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb_global = state + (int64_t)pool_idx * num_v_heads * shs + hv_idx * shs;

    // LDS: [V, K+pad] — transposed from global [K, V]
    // 128 * 129 * 4 = 66,048 bytes ≈ 64.5KB
    __shared__ float lds[LDS_V][LDS_STRIDE];

    // Cooperative load: global [K,V] → LDS [V, K]
    // Each thread loads multiple elements
    // Global: row k, col v → lds[v][k]
    constexpr int TOTAL_ELEMS = LDS_K * LDS_V;  // 16384
    constexpr int ELEMS_PER_THREAD = (TOTAL_ELEMS + BLOCK_THREADS - 1) / BLOCK_THREADS;  // 64

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int elem = tid + i * BLOCK_THREADS;
        if (elem < TOTAL_ELEMS) {
            int gk = elem / LDS_V;  // global K index
            int gv = elem % LDS_V;  // global V index
            lds[gv][gk] = sb_global[gk * HEAD_V_DIM + gv];
        }
    }
    __syncthreads();

    // Load from LDS in [V, K] order with float4 along K (like VK kernel)
    float S[V_ITERS][K_ITERS][VEC_K];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int k_base = warp_k_vec_start + ki * WARP_TILE_K;
            S[vi][ki][0] = lds[gv][k_base];
            S[vi][ki][1] = lds[gv][k_base + 1];
            S[vi][ki][2] = lds[gv][k_base + 2];
            S[vi][ki][3] = lds[gv][k_base + 3];
        }
    }
    // No more LDS needed for state loading; barrier before any writes
    __syncthreads();

    const float r_A_log = A_log[hv_idx];
    const float r_dt_bias = dt_bias[hv_idx];

    float k_reg[K_ITERS][VEC_K];

    for (int sq = 0; sq < seq_length; sq++) {
        const int ab = batch_idx * seq_length * num_v_heads + sq * num_v_heads + hv_idx;
        float x = __bfloat162float(a_input[ab]) + r_dt_bias;
        float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
        float exp_g = __expf(-__expf(r_A_log) * sp);
        float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b_input[ab])));

        const auto* qb = query + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* kb = key + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* vb = value + (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int ko = warp_k_vec_start + ki * WARP_TILE_K;
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                k_reg[ki][j] = __bfloat162float(kb[ko + j]);
        }

        float dot_kq = 0;
        float res_hk[V_ITERS], res_hq[V_ITERS], res_rv[V_ITERS];

        {
            float q_reg[K_ITERS][VEC_K];
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++) {
                int ko = warp_k_vec_start + ki * WARP_TILE_K;
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    q_reg[ki][j] = __bfloat162float(qb[ko + j]);
            }

            if constexpr (USE_QK_L2NORM) {
                float sq_q = 0, sq_k = 0;
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        sq_q += q_reg[ki][j] * q_reg[ki][j];
                        sq_k += k_reg[ki][j] * k_reg[ki][j];
                    }
                sq_q = reduce_sum_k(sq_q); sq_k = reduce_sum_k(sq_k);
                float qs = scale * __frsqrt_rn(sq_q + 1e-6f);
                float ks = __frsqrt_rn(sq_k + 1e-6f);
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        q_reg[ki][j] *= qs;
                        k_reg[ki][j] *= ks;
                    }
            } else {
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++)
                        q_reg[ki][j] *= scale;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    dot_kq += k_reg[ki][j] * q_reg[ki][j];
            dot_kq = reduce_sum_k(dot_kq);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                res_rv[vi] = __bfloat162float(vb[global_v_base + vi * WARP_GROUP_TILE_V]);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] = 0.0f;
                res_hq[vi] = 0.0f;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++) {
                    float kv = k_reg[ki][j];
                    float qv = q_reg[ki][j];
                    #pragma unroll
                    for (int vi = 0; vi < V_ITERS; vi++) {
                        res_hk[vi] += S[vi][ki][j] * kv;
                        res_hq[vi] += S[vi][ki][j] * qv;
                    }
                }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 1);
                res_hq[vi] += shfl_xor(res_hq[vi], 1);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 2);
                res_hq[vi] += shfl_xor(res_hq[vi], 2);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 4);
                res_hq[vi] += shfl_xor(res_hq[vi], 4);
            }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] *= exp_g;
                res_hq[vi] *= exp_g;
            }
        }

        float vn[V_ITERS];
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            float tmp = (res_rv[vi] - res_hk[vi]) * beta;
            vn[vi] = shfl_idx(tmp, leader);
        }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;

        if (k_tid == 0) {
            const int64_t out_base = (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM
                                     + hv_idx * HEAD_V_DIM + global_v_base;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) {
                float kv = k_reg[ki][j];
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++)
                    S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
            }
    }

    // State store: registers → LDS [V, K] → global [K, V]
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int k_base = warp_k_vec_start + ki * WARP_TILE_K;
            lds[gv][k_base]     = S[vi][ki][0];
            lds[gv][k_base + 1] = S[vi][ki][1];
            lds[gv][k_base + 2] = S[vi][ki][2];
            lds[gv][k_base + 3] = S[vi][ki][3];
        }
    }
    __syncthreads();

    // Cooperative store: LDS [V, K] → global [K, V]
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int elem = tid + i * BLOCK_THREADS;
        if (elem < TOTAL_ELEMS) {
            int gk = elem / LDS_V;
            int gv = elem % LDS_V;
            sb_global[gk * HEAD_V_DIM + gv] = lds[gv][gk];
        }
    }
}

extern "C" {

#define LAUNCH_FUSED(KERNEL, L2) \
   hipLaunchKernelGGL(( KERNEL<L2>), dim3(grid), dim3(block), 0, stream,  \
        q, k, v, a, b, dt, al, idx, st, o, batch_size, seq_length, scale, nkh, nvh)

void launch_gdn_decode_fused(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
) {
    dim3 grid(batch_size * num_v_heads);
    dim3 block(BLOCK_THREADS);
    auto q = reinterpret_cast<const __hip_bfloat16*>(query);
    auto k = reinterpret_cast<const __hip_bfloat16*>(key);
    auto v = reinterpret_cast<const __hip_bfloat16*>(value);
    auto a = reinterpret_cast<const __hip_bfloat16*>(a_input);
    auto b = reinterpret_cast<const __hip_bfloat16*>(b_input);
    auto dt = reinterpret_cast<const float*>(dt_bias);
    auto al = reinterpret_cast<const float*>(A_log);
    auto idx = reinterpret_cast<const int*>(indices);
    auto st = reinterpret_cast<float*>(state);
    auto o = reinterpret_cast<__hip_bfloat16*>(output);
    int nkh = num_k_heads, nvh = num_v_heads;

    if (use_qk_l2norm) LAUNCH_FUSED(gdn_decode_kernel_fused, true);
    else               LAUNCH_FUSED(gdn_decode_kernel_fused, false);
}

#undef LAUNCH_FUSED

} // extern "C"
