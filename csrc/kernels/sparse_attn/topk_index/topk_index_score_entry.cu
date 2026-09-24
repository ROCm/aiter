// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// ctypes entry point for the bf16-Q x fp8-K decode index-scoring pass.
// Guards only: it validates, then dispatches into the table in
// topk_index_score.hpp. Every unsupported input is refused here with a reason,
// never routed onto a neighbouring cell.

// No device code is instantiated in this entry TU.
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_hip_common.h"
#include "topk_index_score.hpp"

#include "aiter_ctypes_error.h"

#include <cmath> // std::isfinite for the sm_scale guard below

// This TU DEFINES the ctypes TLS error storage: these sources are their own
// module (module_topk_index_score), selected by -DOPUS_IDX_SCORE_OWN_MODULE.
#if defined(OPUS_IDX_SCORE_OWN_MODULE)
AITER_CTYPES_ERROR_DEF
#else
AITER_CTYPES_ERROR_DECL;
#endif

namespace {

// Arch gate. Cached PER DEVICE, not once per process: get_gpu_arch() reads the
// current device every call, so a single static would reuse the first device's
// verdict after a device switch.
inline bool opus_idx_score_arch_supported()
{
    int dev = 0;
    if(hipGetDevice(&dev) != hipSuccess)
        return false;
    constexpr int kMaxDevices = 64;
    // 0 = unknown, 1 = supported, 2 = refused. Racy only in the benign sense:
    // two threads computing the same answer for the same device write the same
    // value.
    static unsigned char cache[kMaxDevices] = {0};
    if(dev < 0 || dev >= kMaxDevices)
        return get_gpu_arch() == "gfx950";
    if(cache[dev] == 0)
        cache[dev] = (get_gpu_arch() == "gfx950") ? 1 : 2;
    return cache[dev] == 1;
}

} // namespace

#define OPUS_IDX_SCORE_PASS "topk_index_score_decode"

// Certification-mode bypass token. Lets a certification run reach a built but
// uncertified cell; anything other than the exact token is refused.
#define OPUS_IDX_SCORE_CERT_TOKEN 0x5339C0DE

// One arm per BUILT cell -- the fail-closed table. Reaching the end means the
// cell is not built, and that is a refusal, never a fallthrough.
#define OPUS_IDX_SCORE_DISPATCH(H, Q, A)                                        \
    if(num_idx_heads == (H) && query_len == (Q) && aux_k == (A))                \
    {                                                                           \
        aiter::sparse_attn::OPUS_IDX_SCORE_FN(H, Q, A)(a, num_chunks, stream);  \
        return;                                                                 \
    }

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(topk_index_score_decode,
                                    (size_t q_idx_ptr,
                                     size_t key_cache_idx_ptr,
                                     size_t score_ptr,
                                     size_t block_table_ptr,
                                     size_t seq_lens_ptr,
                                     long long q_numel,
                                     long long key_cache_numel,
                                     long long score_numel,
                                     long long block_table_numel,
                                     int batch,
                                     int num_chunks,
                                     int chunk_blocks,
                                     long long stride_q_n,
                                     long long stride_q_h,
                                     long long stride_ik_blk,
                                     long long stride_s_h,
                                     long long stride_s_b,
                                     long long stride_bt_b,
                                     float sm_scale,
                                     int block_size,
                                     int head_dim,
                                     int num_idx_heads,
                                     int query_len,
                                     int aux_k,
                                     int cert_bypass,
                                     hipStream_t stream),
                                    (q_idx_ptr,
                                     key_cache_idx_ptr,
                                     score_ptr,
                                     block_table_ptr,
                                     seq_lens_ptr,
                                     q_numel,
                                     key_cache_numel,
                                     score_numel,
                                     block_table_numel,
                                     batch,
                                     num_chunks,
                                     chunk_blocks,
                                     stride_q_n,
                                     stride_q_h,
                                     stride_ik_blk,
                                     stride_s_h,
                                     stride_s_b,
                                     stride_bt_b,
                                     sm_scale,
                                     block_size,
                                     head_dim,
                                     num_idx_heads,
                                     query_len,
                                     aux_k,
                                     cert_bypass,
                                     stream))
{
    // C1 -- architecture.
    AITER_CHECK(opus_idx_score_arch_supported(),
                OPUS_IDX_SCORE_PASS ": requires gfx950 architecture; the running device is ",
                get_gpu_arch());

    // C2 -- the shapes fixed for the whole build. head_dim is not a default:
    // invariant I5 (1.00 L1 accesses per 128 B line) is derived for 128-byte
    // rows, and the LDS staging layout is derived for head_dim 128.
    AITER_CHECK(head_dim == OPUS_IDX_HEAD_DIM,
                OPUS_IDX_SCORE_PASS ": head_dim ",
                head_dim,
                " is not built; the staging layout is derived for head_dim ",
                OPUS_IDX_HEAD_DIM);
    AITER_CHECK(block_size == OPUS_IDX_BLOCK_SIZE,
                OPUS_IDX_SCORE_PASS ": block_size ",
                block_size,
                " is not built; this build carries block_size ",
                OPUS_IDX_BLOCK_SIZE);

    // N13 -- an empty batch is a no-op, not an error, and must not reach a
    // launch (G4: no zero-extent grid may be captured). score is left untouched,
    // matching the incumbent.
    if(batch == 0)
        return;

    // C2b -- launch geometry. These are capture-time constants computed by the
    // caller (G1/G2); nothing here reads a device tensor.
    // D06 second half: positivity alone admitted num_chunks * chunk_blocks
    // products that overflow int inside the kernel's own indexing. The upper
    // bound is what the block table can actually address.
    AITER_CHECK(batch > 0 && num_chunks > 0 && chunk_blocks > 0 &&
                    (long long)num_chunks * (long long)chunk_blocks <= (1LL << 24),
                OPUS_IDX_SCORE_PASS ": batch/num_chunks/chunk_blocks must be positive, got ",
                batch,
                "/",
                num_chunks,
                "/",
                chunk_blocks);

    // Descriptor extents. These element counts become the buffer-descriptor
    // sizes, which is where the hardware bounds check comes from.
    AITER_CHECK(q_numel > 0 && key_cache_numel > 0 && score_numel > 0 && block_table_numel > 0,
                OPUS_IDX_SCORE_PASS ": tensor element counts must be positive, got q=",
                q_numel,
                " k=",
                key_cache_numel,
                " score=",
                score_numel,
                " block_table=",
                block_table_numel);

    // Beyond the required guard list: a non-finite sm_scale would propagate
    // into every score, and the caller almost certainly did not mean it.
    AITER_CHECK(std::isfinite(sm_scale),
                OPUS_IDX_SCORE_PASS ": sm_scale must be finite, got ",
                sm_scale);

    // The BLOCK_SIZE_N = 16 specialisation. Re-checked here even though layer P
    // rejects it: a Python routing bug must not be able to reach a launch.
    // Computed in long long so the check cannot itself overflow.
    AITER_CHECK(num_idx_heads > 0 && query_len > 0 &&
                    (long long)num_idx_heads * (long long)query_len <= 16LL,
                OPUS_IDX_SCORE_PASS ": num_idx_heads * query_len must be in [1, 16], got ",
                num_idx_heads,
                " * ",
                query_len);

    // C3b -- AUX_K is the one tuned axis and carries exactly two built values.
    AITER_CHECK(aux_k == 0 || aux_k == 3,
                OPUS_IDX_SCORE_PASS ": aux_k ",
                aux_k,
                " is not built; the tuned axis carries {0, 3}");

    // Certification mode, checked BEFORE N16 uses it. A non-zero value that is
    // not exactly the token means something tried to enable the bypass and got
    // it wrong -- fail hard rather than quietly treating it as "no bypass".
    AITER_CHECK(cert_bypass == 0 || cert_bypass == OPUS_IDX_SCORE_CERT_TOKEN,
                OPUS_IDX_SCORE_PASS ": cert_bypass must be 0 or the exact "
                                    "certification token; got ",
                cert_bypass);
    const bool cert_mode = (cert_bypass == OPUS_IDX_SCORE_CERT_TOKEN);

    // Built but not certified: the cell compiles, so the evidence gap is
    // visible and closable, but dispatch refuses it. Widening this needs
    // certification evidence produced by something that did not author it.
    AITER_CHECK(cert_mode ||
                    aiter::sparse_attn::opus_idx_score_cell_certified(num_idx_heads, query_len),
                OPUS_IDX_SCORE_PASS ": variant built but not certified -- (num_idx_heads=",
                num_idx_heads,
                ", query_len=",
                query_len,
                ") has no per-cell bitwise evidence at the accepted config; "
                "phase-1 Tier-2 evidence covers (1, 4) only");

    // Everything the kernel reads, assembled ONCE. num_chunks is grid geometry,
    // not kernel state, so it rides beside the struct rather than inside it.
    aiter::sparse_attn::opus_decode_score_args a{};
    a.q_ptr         = reinterpret_cast<const void*>(q_idx_ptr);
    a.ik_ptr        = reinterpret_cast<const void*>(key_cache_idx_ptr);
    a.score_ptr     = reinterpret_cast<float*>(score_ptr);
    a.bt_ptr        = reinterpret_cast<const int*>(block_table_ptr);
    a.seq_lens_ptr  = reinterpret_cast<const int*>(seq_lens_ptr);
    a.q_numel       = q_numel;
    a.ik_numel      = key_cache_numel;
    a.score_numel   = score_numel;
    a.bt_numel      = block_table_numel;
    a.batch         = batch;
    a.chunk_blocks  = chunk_blocks;
    a.stride_q_n    = stride_q_n;
    a.stride_q_h    = stride_q_h;
    a.stride_ik_blk = stride_ik_blk;
    a.stride_s_h    = stride_s_h;
    a.stride_s_b    = stride_s_b;
    a.stride_bt_b   = stride_bt_b;
    a.sm_scale      = sm_scale;

    // C4 -- the instantiation table. Fail closed on anything Python let through.
    OPUS_IDX_SCORE_TABLE(OPUS_IDX_SCORE_DISPATCH)

    AITER_CHECK(false,
                OPUS_IDX_SCORE_PASS ": no build for num_idx_heads=",
                num_idx_heads,
                " query_len=",
                query_len,
                " aux_k=",
                aux_k);
}
#endif // !__HIP_DEVICE_COMPILE__ && !__HIPCC_RTC__
