import os

import torch

from aiter.ops.triton._triton_kernels.attention.unified_attention_sparse_mla import (
    UA_SPARSE_MLA_AUTOTUNE,
    _2d_csr_autotuner,
    _2d_topk_autotuner,
    _3d_csr_autotuner,
    _kernel_unified_attention_sparse_mla_2d,
    _kernel_unified_attention_sparse_mla_csr_3d,
    _kernel_unified_attention_sparse_mla_csr_reduce,
)


# Use 3D split-K path when CSR is used and max_sparse_len exceeds this.
# The 3D split adds work but parallelizes across the topk dimension which is
# essential when (num_tokens * num_query_heads / BLOCK_M) << num_CUs.
_SPLIT_K_THRESHOLD = int(
    os.environ.get("UNIFIED_ATTENTION_SPARSE_MLA_SPLIT_K_THRESHOLD", "1024")
)
# Force-disable 3D path for debugging.
_DISABLE_SPLIT_K = os.environ.get(
    "UNIFIED_ATTENTION_SPARSE_MLA_DISABLE_SPLIT_K", "0"
).lower() in ("1", "true", "yes", "on")
# Number of KV segments to split into. Choose to keep ~ all CUs busy.
_NUM_CU_HINT = int(os.environ.get("UNIFIED_ATTENTION_SPARSE_MLA_NUM_CU", "256"))


def _choose_num_segments(num_q_blocks: int, max_sparse_len: int, tile_size: int) -> int:
    """Pick NUM_SEGMENTS_PER_SEQ to roughly fill the GPU.

    We want num_q_blocks * NUM_SEGMENTS ~ num_CU * waves. Also constrained by
    NUM_SEGMENTS <= ceil(max_sparse_len / tile_size) since smaller is wasteful.
    """
    max_useful = (max_sparse_len + tile_size - 1) // tile_size
    if num_q_blocks <= 0:
        return 1
    desired = max(1, _NUM_CU_HINT // max(1, num_q_blocks))
    n = 1
    while n < desired:
        n *= 2
    n = min(n, max_useful)
    p = 1
    while p < n:
        p *= 2
    return max(1, min(p, 64))


def _coerce_scale(s, device):
    """Coerce Python scalars into 0-d float32 tensors on the right device.

    SGLang sometimes passes Python ``int``/``float`` for scale args; the
    kernel expects a tensor pointer. ``None`` is preserved (compile-time
    no-op via the ``Q_SCALE/K_SCALE/V_SCALE`` constexpr flags).
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return torch.tensor(s, dtype=torch.float32, device=device)
    return s


def unified_attention_sparse_mla(
    q,
    kv,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    topk_indices,
    block_table,
    kv_lora_rank,
    kv_indptr=None,
    kv_indices=None,
    max_sparse_len=None,
    q_scale=None,
    k_scale=None,
    v_scale=None,
):
    """
    Sparse MLA attention.

    Q:             [seq_len, NUM_HEADS, kv_lora_rank + rope_rank], bfloat16
    KV:            [num_blks, blk_size, 1, kv_lora_rank + rope_rank], bfloat16 or fp8_e4m3
    cu_seqlens_q:  [BATCH + 1], int32
    softmax_scale: float32
    topk_indices:  Optional [seq_len, TOP_K] int32
    kv_indptr:     Optional [seq_len + 1] int32
    kv_indices:    Optional [nnz] int32
    block_table:   [BATCH, MAX_NUM_BLOCKS_PER_BATCH] int32
    kv_lora_rank:  int
    q_scale:       Optional scalar tensor or python float for per-tensor FP8 Q scale
    k_scale:       Optional scalar tensor or python float for per-tensor FP8 K scale
    v_scale:       Optional scalar tensor or python float for per-tensor FP8 V scale

    Returns:
    out (in-place): [seq_len, NUM_HEADS, kv_lora_rank] bfloat16
    """

    use_csr = kv_indptr is not None or kv_indices is not None
    if use_csr:
        assert kv_indptr is not None and kv_indices is not None
        assert kv_indptr.shape[0] >= q.shape[0] + 1
        if max_sparse_len is None:
            row_lens = kv_indptr[1 : q.shape[0] + 1] - kv_indptr[: q.shape[0]]
            max_sparse_len = int(row_lens.max().item())
    else:
        assert topk_indices is not None

    q_scale_t = _coerce_scale(q_scale, q.device)
    k_scale_t = _coerce_scale(k_scale, q.device)
    v_scale_t = _coerce_scale(v_scale, q.device)
    Q_SCALE = q_scale_t is not None
    K_SCALE = k_scale_t is not None
    V_SCALE = v_scale_t is not None
    block_size = kv.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = 1
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]
    topk_count = topk_indices.shape[1] if topk_indices is not None else 0
    k = kv
    v = kv[..., :kv_lora_rank]

    BLOCK_M = 16
    total_num_q_blocks = q.shape[0] * (num_query_heads // BLOCK_M)
    ALL_DECODE = max_seqlen_q == 1

    ROPE_RANK = head_size - kv_lora_rank
    KV_LORA_RANK = kv_lora_rank

    DEFAULT_TILE_SIZE = 64
    DEFAULT_NUM_WARPS = 4
    DEFAULT_NUM_STAGES = 1
    DEFAULT_PRELOAD_V = False
    DEFAULT_WAVES_PER_EU = 1

    # Tuned defaults for the 2D CSR path (used when CSR is enabled but the GPU
    # is already filled — high batch * heads, so 3D split-K's extra parallelism
    # buys nothing). PRELOAD_V=True + waves_per_eu=2 sweep winners on
    # heads=128 batch=64 (GLM-5/DSA decode shape): 0.6147 -> 0.4405 ms (28%).
    DEFAULT_2D_CSR_TILE_SIZE = 64
    DEFAULT_2D_CSR_NUM_WARPS = 4
    DEFAULT_2D_CSR_NUM_STAGES = 1
    DEFAULT_2D_CSR_PRELOAD_V = True
    DEFAULT_2D_CSR_WAVES_PER_EU = 2

    # Tuned defaults for the 3D split-K path. Picked from autotune sweep on
    # GLM-5 decode shapes (heads=16, lora=512, rope=64, block=64, sk=8192,
    # top_k=2048) over batches 1..64. Best config: TILE_SIZE=32, PRELOAD_V=True,
    # num_warps=8, num_stages=2, waves_per_eu=2.
    DEFAULT_3D_TILE_SIZE = 32
    DEFAULT_3D_NUM_WARPS = 8
    DEFAULT_3D_NUM_STAGES = 2
    DEFAULT_3D_PRELOAD_V = True
    DEFAULT_3D_WAVES_PER_EU = 2

    if use_csr:
        effective_len = max_sparse_len
        use_split_k = (
            not _DISABLE_SPLIT_K
            and ALL_DECODE
            and total_num_q_blocks > 0
            and total_num_q_blocks < _NUM_CU_HINT
            and effective_len >= _SPLIT_K_THRESHOLD
        )

        if use_split_k:
            tile_for_seg = DEFAULT_TILE_SIZE
            num_segments = _choose_num_segments(
                total_num_q_blocks, effective_len, tile_for_seg
            )
        else:
            num_segments = 1

        if use_split_k and num_segments > 1:
            segm_output = torch.empty(
                (q.shape[0], num_query_heads, num_segments, KV_LORA_RANK),
                dtype=torch.float32,
                device=q.device,
            )
            segm_max = torch.empty(
                (q.shape[0], num_query_heads, num_segments),
                dtype=torch.float32,
                device=q.device,
            )
            segm_expsum = torch.empty(
                (q.shape[0], num_query_heads, num_segments),
                dtype=torch.float32,
                device=q.device,
            )

            kernel_kwargs = dict(
                segm_output_ptr=segm_output,
                segm_max_ptr=segm_max,
                segm_expsum_ptr=segm_expsum,
                query_ptr=q,
                key_cache_ptr=k,
                value_cache_ptr=v,
                kv_indptr_ptr=kv_indptr,
                kv_indices_ptr=kv_indices,
                scale=softmax_scale,
                q_scale=q_scale_t,
                k_scale=k_scale_t,
                v_scale=v_scale_t,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                query_stride_0=q.stride(0),
                query_stride_1=q.stride(1),
                BLOCK_SIZE=block_size,
                stride_k_cache_0=k.stride(0),
                stride_k_cache_1=k.stride(1),
                stride_k_cache_2=k.stride(2),
                stride_k_cache_3=k.stride(3),
                stride_v_cache_0=v.stride(0),
                stride_v_cache_1=v.stride(1),
                stride_v_cache_2=v.stride(2),
                stride_v_cache_3=v.stride(3),
                max_sparse_len=max_sparse_len,
                query_start_len_ptr=cu_seqlens_q,
                num_seqs=num_seqs,
                BLOCK_M=BLOCK_M,
                ROPE_RANK=ROPE_RANK,
                KV_LORA_RANK=KV_LORA_RANK,
                NUM_SEGMENTS_PER_SEQ=num_segments,
                segm_out_stride_tok=segm_output.stride(0),
                segm_out_stride_head=segm_output.stride(1),
                segm_stat_stride_tok=segm_max.stride(0),
                segm_stat_stride_head=segm_max.stride(1),
                ALL_DECODE=ALL_DECODE,
                Q_SCALE=Q_SCALE,
                K_SCALE=K_SCALE,
                V_SCALE=V_SCALE,
            )

            grid_3d = (total_num_q_blocks, num_segments)
            if UA_SPARSE_MLA_AUTOTUNE and _3d_csr_autotuner is not None:
                _3d_csr_autotuner[grid_3d](**kernel_kwargs)
            else:
                _kernel_unified_attention_sparse_mla_csr_3d[grid_3d](
                    **kernel_kwargs,
                    TILE_SIZE=DEFAULT_3D_TILE_SIZE,
                    PRELOAD_V=DEFAULT_3D_PRELOAD_V,
                    num_warps=DEFAULT_3D_NUM_WARPS,
                    num_stages=DEFAULT_3D_NUM_STAGES,
                    waves_per_eu=DEFAULT_3D_WAVES_PER_EU,
                )

            _kernel_unified_attention_sparse_mla_csr_reduce[
                (q.shape[0], num_query_heads)
            ](
                output_ptr=out,
                segm_output_ptr=segm_output,
                segm_max_ptr=segm_max,
                segm_expsum_ptr=segm_expsum,
                output_stride_0=out.stride(0),
                output_stride_1=out.stride(1),
                segm_out_stride_tok=segm_output.stride(0),
                segm_out_stride_head=segm_output.stride(1),
                segm_stat_stride_tok=segm_max.stride(0),
                segm_stat_stride_head=segm_max.stride(1),
                KV_LORA_RANK=KV_LORA_RANK,
                NUM_SEGMENTS_PER_SEQ=num_segments,
                num_warps=4,
                num_stages=1,
            )
            return

        # 2D CSR path (merged kernel; topk_indices_ptr unused under USE_CSR=True
        # — pass kv_indices as a placeholder pointer so Triton typechecks).
        kernel_kwargs = dict(
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            topk_indices_ptr=kv_indices,
            kv_indptr_ptr=kv_indptr,
            kv_indices_ptr=kv_indices,
            seq_lens_ptr=seqused_k,
            scale=softmax_scale,
            q_scale=q_scale_t,
            k_scale=k_scale_t,
            v_scale=v_scale_t,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            BLOCK_SIZE=block_size,
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            topk_count=0,
            max_sparse_len=max_sparse_len,
            query_start_len_ptr=cu_seqlens_q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            ROPE_RANK=ROPE_RANK,
            KV_LORA_RANK=KV_LORA_RANK,
            USE_CSR=True,
            ALL_DECODE=ALL_DECODE,
            Q_SCALE=Q_SCALE,
            K_SCALE=K_SCALE,
            V_SCALE=V_SCALE,
        )
        if UA_SPARSE_MLA_AUTOTUNE and _2d_csr_autotuner is not None:
            _2d_csr_autotuner[(total_num_q_blocks,)](**kernel_kwargs)
        else:
            _kernel_unified_attention_sparse_mla_2d[(total_num_q_blocks,)](
                **kernel_kwargs,
                TILE_SIZE=DEFAULT_2D_CSR_TILE_SIZE,
                PRELOAD_V=DEFAULT_2D_CSR_PRELOAD_V,
                num_warps=DEFAULT_2D_CSR_NUM_WARPS,
                num_stages=DEFAULT_2D_CSR_NUM_STAGES,
                waves_per_eu=DEFAULT_2D_CSR_WAVES_PER_EU,
            )
        return

    # Dense top-k path (merged kernel; kv_indptr_ptr/kv_indices_ptr unused under
    # USE_CSR=False — pass topk_indices as a placeholder pointer).
    kernel_kwargs = dict(
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        topk_indices_ptr=topk_indices,
        kv_indptr_ptr=topk_indices,
        kv_indices_ptr=topk_indices,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        q_scale=q_scale_t,
        k_scale=k_scale_t,
        v_scale=v_scale_t,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        topk_count=topk_count,
        max_sparse_len=0,
        query_start_len_ptr=cu_seqlens_q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        ROPE_RANK=ROPE_RANK,
        KV_LORA_RANK=KV_LORA_RANK,
        USE_CSR=False,
        ALL_DECODE=ALL_DECODE,
        Q_SCALE=Q_SCALE,
        K_SCALE=K_SCALE,
        V_SCALE=V_SCALE,
    )
    if UA_SPARSE_MLA_AUTOTUNE and _2d_topk_autotuner is not None:
        _2d_topk_autotuner[(total_num_q_blocks,)](**kernel_kwargs)
    else:
        _kernel_unified_attention_sparse_mla_2d[(total_num_q_blocks,)](
            **kernel_kwargs,
            TILE_SIZE=DEFAULT_TILE_SIZE,
            PRELOAD_V=DEFAULT_PRELOAD_V,
            num_warps=DEFAULT_NUM_WARPS,
            num_stages=DEFAULT_NUM_STAGES,
            waves_per_eu=DEFAULT_WAVES_PER_EU,
        )
