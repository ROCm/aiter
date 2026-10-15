// libhk_sparse_mla: sparse MLA (DSA) forward for gfx950 (MI350X / MI355X), HipKittens. See sparse_mla.cu for the design.
//
//   out[t][h][:] = softmax_k(q[t][h] . kv[idx[t][k]] * sm_scale) @ kv[idx[t][k]][:512]   (all heads share the KV row)
//
// Shapes: 32 heads, 512 latent dims (no rope part); q [T][32][512] fp8 e4m3, kv rows at kv + slot * kv_stride_bytes for
// slot < num_slots (first 512 bytes = latent, rows 16-byte aligned; other slots are masked), idx [T][topk] int32 (topk a multiple of 128, <= 2304; idx < 0 =
// masked), out [T][32][512] bf16. Small T splits each token's keys; the partials need hk_sparse_mla_workspace_bytes.
#pragma once
#include <hip/hip_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum HkSparseMlaStatus : int {
    HK_SPARSE_MLA_OK = 0,
    HK_SPARSE_MLA_BAD_SHAPE = -1,       // topk not a positive multiple of 128 <= MAX_TOPK, misaligned kv rows or bad num_slots
    HK_SPARSE_MLA_LAUNCH_FAILED = -2,
    HK_SPARSE_MLA_SMALL_WORKSPACE = -3,
};

// Number of key splits used for T tokens.
int hk_sparse_mla_splits(int T, int topk);
// fp32 partial workspace for T tokens (0 when no split is needed).
size_t hk_sparse_mla_workspace_bytes(int T, int topk);
// Returns an HkSparseMlaStatus; all work is enqueued on `stream` (graph-capturable).
int hk_sparse_mla_fwd(const void *q, const void *kv, const int *idx, void *out, int T, int topk, long kv_stride_bytes,
                      long num_slots, float sm_scale, void *workspace, size_t ws_bytes, hipStream_t stream);

#ifdef __cplusplus
}
#endif
