// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// topk_index_score_entry.cu -- the ctypes entry point for the bf16-Q x
// fp8-K decode lightning-indexer scoring path (Layer C of integration.md
// section 8.2). Guards only: this file dispatches into the table declared by
// topk_index_score.hpp and refuses everything else.
//
// It is a NEW entry point BESIDE pa_sparse_block_select_entry.cu, not a change
// to it. The incumbent's three entry points are not touched and keep their
// behaviour byte-for-byte (T1).
//
// GUARD ORDER (section 8.2: every input is rejected at exactly ONE layer -- the
// tightest that can express the rejection -- and no input reaches an undefined
// branch):
//
//   C1  arch == gfx950
//   C2  head_dim == 128, block_size == 128
//   C2b descriptor extents and launch geometry are usable
//   C3  H * Q <= 16
//   C3b aux_k in {0, 3}
//   N16 the (H, Q) cell is BUILT AND CERTIFIED -- else an explicit
//       "built but not certified" refusal, never a launch
//   C4  (H, Q, AUX_K) is in the instantiation table -- else AITER_CHECK(false)
//
// Redundancy with the Python layer is deliberate: defense in depth is the
// incumbent's own convention, and a Python routing bug must not be able to
// reach a launch.
//
// NOT here, by design: the dtype routing (N1/N2/N3/N17 -- fp8 vs bf16, and the
// e4m3fn vs e4m3fnuz distinction) needs tensor metadata and therefore belongs
// to Layer P in msa_block_select.py. This entry point receives raw addresses
// and cannot see a dtype; that is exactly why the Python predicate must be
// pinned to == float8_e4m3fn and not a loose is_fp8.

// aiter_ctypes_error.h needs aiter_detail from the headers this pulls in, so
// keep it first (blank line stops clang-format from sorting the two together).
#include "topk_index_score.hpp"

#include "aiter_ctypes_error.h"

#include <cmath> // std::isfinite for the sm_scale guard below

// TLS error storage placement depends on P3's UNRESOLVED module decision:
//
//   * same .so as the incumbent (module_msa_sparse_attention as it stands
//     today): pa_sparse_block_select_entry.cu already carries
//     AITER_CTYPES_ERROR_DEF, and a second DEF in the same .so is a duplicate
//     symbol -- so this TU must DECLARE, which is the default below;
//   * its own gfx950-only .so (the option that would also let the arch refusal
//     become a hard #error, and that the -mllvm -amdgpu-mfma-vgpr-form=1
//     per-source flag question may force anyway): this TU must DEFINE, which
//     the build selects with -DOPUS_IDX_SCORE_OWN_MODULE.
//
// Either way the wrong choice fails LOUDLY at link time -- duplicate symbol, or
// undefined g_aiter_last_error -- never silently. The decision is P3's and is
// not taken here.
// RESOLVED (2026-09-23): this path now has its OWN module,
// module_topk_index_score, so this TU DEFINES the TLS error storage. The build
// passes -DOPUS_IDX_SCORE_OWN_MODULE; without it the link fails with an
// undefined g_aiter_last_error, which is the loud failure the note below
// promised. The alternative branch stays for the case where these sources are
// ever folded back into a shared .so.
#if defined(OPUS_IDX_SCORE_OWN_MODULE)
AITER_CTYPES_ERROR_DEF
#else
AITER_CTYPES_ERROR_DECL;
#endif

namespace {

// C1. Same shape as the incumbent's sparse_arch_supported(), under its own name
// so the two TUs never collide.
//
// D05 (review, upheld at YELLOW): this cached the verdict in a function-local
// static while get_gpu_arch() reads hipGetDevice on EVERY call. A process that
// touched a gfx950 device first and then hipSetDevice'd to another architecture
// reused the first answer and admitted an unsupported device. The incumbent's
// pa_sparse_block_select_entry.cu has the same shape; that makes it a shared
// defect, not a licence.
//
// Cached PER DEVICE rather than not at all: get_gpu_arch() costs a
// hipGetDeviceProperties, which is not something to pay on every launch, and
// the device ordinal is what actually varies.
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

// N16 certification-mode bypass token (lead ruling, team-message-7dff0e55).
//
// Layer C must honour the same bypass Layer P honours: a bypass in one layer
// and a refusal in the next is the "passes one layer, unchecked by the next"
// hole section 8.2 forbids -- it would make the harness unusable AND leave a
// half-open gate, the worst of both.
//
// A TOKEN rather than a boolean, deliberately: a stray 1, a truthy garbage
// value, or an uninitialised field must NOT enable this. Any non-zero value
// that is not exactly the token is a HARD FAILURE, not a silent "no bypass" --
// something tried to enable certification mode and got it wrong, and that is
// worth stopping for.
//
// Mirrored by OPUS_CERT_BYPASS_TOKEN in aiter/ops/msa_opus_index_score.py; the
// two must be edited together.
#define OPUS_IDX_SCORE_CERT_TOKEN 0x5339C0DE

// C4. One arm per BUILT cell -- the fail-closed table. Reaching here means the
// cell passed N16 either by being certified or by an explicit certification-mode
// bypass; C4 itself is about what was COMPILED, which no mode waives. (This
// comment previously said "reaching here means the cell is dispatchable", which
// stopped being true the moment the bypass landed. Stale comments are how the
// 32-vs-64-bank error stayed invisible for eleven rounds.)
#define OPUS_IDX_SCORE_DISPATCH(H, Q, A)                                        \
    if(num_idx_heads == (H) && query_len == (Q) && aux_k == (A))                \
    {                                                                           \
        aiter::sparse_attn::OPUS_IDX_SCORE_FN(H, Q, A)(OPUS_IDX_SCORE_ARGS);    \
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

    // C2c -- descriptor extents. These element counts become the buffer
    // descriptors' extents, and those ARE the hardware bounds check the page
    // offset relies on (invariant I3: the offset rides in voffset precisely so
    // the check applies). A zero or negative extent would not fault -- it would
    // make every load return zero and produce silently wrong scores -- so it is
    // refused here rather than trusted.
    AITER_CHECK(q_numel > 0 && key_cache_numel > 0 && score_numel > 0 && block_table_numel > 0,
                OPUS_IDX_SCORE_PASS ": tensor element counts must be positive, got q=",
                q_numel,
                " k=",
                key_cache_numel,
                " score=",
                score_numel,
                " block_table=",
                block_table_numel);

    // ADDED GUARD, beyond the section 8.2 list -- flagged for the lead rather
    // than absorbed silently. sm_scale is multiplied by log2e and applied to
    // every dot; a NaN or infinite scale turns the whole score buffer into
    // NaN/inf, which the top-k consumer would then rank. No caller has a
    // defined meaning for a non-finite scale, so this cannot reject a
    // legitimate input -- but it is my addition, not the frozen contract's, and
    // it belongs in the adversarial test set either way.
    AITER_CHECK(std::isfinite(sm_scale),
                OPUS_IDX_SCORE_PASS ": sm_scale must be finite, got ",
                sm_scale);

    // C3 -- the BLOCK_SIZE_N = 16 specialisation. Re-checked here even though
    // Layer P2 already rejected it: a Python routing bug must not be able to
    // reach a launch.
    // D06 (review, upheld at YELLOW): num_idx_heads * query_len was computed in
    // signed int BEFORE being rejected, so 65536 * 65536 was undefined behaviour
    // inside the very check that exists to reject it. Widened to long long: the
    // product of two ints always fits, so the comparison is now total.
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

    // N16 -- built but not certified. Eight cells are BUILT so the evidence gap
    // is visible and closable; only cells with per-cell BITWISE evidence at
    // config 3 may be DISPATCHED. Refusing here rather than launching is the
    // whole point of the lead's scope ruling: a variant dispatched on the
    // "Class A by the envelope's own logic" argument would make the owed
    // evidence invisible. Widening this requires certification evidence and an
    // edit to opus_idx_score_cell_certified -- not an edit here.
    //
    // cert_mode waives THIS gate and nothing else: N16 is a dispatch-POLICY
    // gate (a statement about evidence), and the certification harness exists
    // to produce that evidence. Every gate above -- arch, the 128/128 build
    // shapes, the extents, H*Q, aux_k, and the built-cell table below -- has
    // already run and is NOT waivable. The dtype pins live in Layer P for the
    // same reason they always did: this entry cannot see a dtype.
    AITER_CHECK(cert_mode ||
                    aiter::sparse_attn::opus_idx_score_cell_certified(num_idx_heads, query_len),
                OPUS_IDX_SCORE_PASS ": variant built but not certified -- (num_idx_heads=",
                num_idx_heads,
                ", query_len=",
                query_len,
                ") has no per-cell bitwise evidence at the accepted config; "
                "phase-1 Tier-2 evidence covers (1, 4) only");

    const auto* q_idx         = reinterpret_cast<const void*>(q_idx_ptr);
    const auto* key_cache_idx = reinterpret_cast<const void*>(key_cache_idx_ptr);
    auto* score               = reinterpret_cast<float*>(score_ptr);
    const auto* block_table   = reinterpret_cast<const int*>(block_table_ptr);
    const auto* seq_lens      = reinterpret_cast<const int*>(seq_lens_ptr);

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
