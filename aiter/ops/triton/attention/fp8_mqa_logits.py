import torch

from aiter.ops.triton._triton_kernels.attention.fp8_mqa_logits import (
    _fp8_mqa_logits_kernel,
)
import triton
from aiter.ops.triton.utils._triton import arch_info
from packaging.version import Version

arch = arch_info.get_arch()
if arch == "gfx950":
    from aiter.ops.triton._gluon_kernels.gfx950.attention.fp8_mqa_logits import (
        _gluon_fp8_mqa_logits_kernel,
    )
elif arch == "gfx1250":
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.fp8_mqa_logits import (
        _gluon_fp8_mqa_logits_kernel,
    )
else:
    _gluon_fp8_mqa_logits_kernel = None

TRITON_VERSION = Version(triton.__version__)
TRITON_BEYOND_37 = TRITON_VERSION > Version("3.7.0")


def fp8_mqa_logits(
    Q,
    KV,
    kv_scales,
    weights,
    cu_starts,
    cu_ends,
):
    """
    This function computes the logits to be used by a topk function for sparse attention.

    Q:           [seq_len, NUM_HEADS, HEAD_SIZE], dtype float8
    KV:          [seq_len_kv, HEAD_SIZE], dtype float8
    kv_scales:   [seq_len_kv], dtype float32
    weights:     [seq_len, NUM_HEADS], dtype float32
    cu_starts:   [seq_len], dtype int32, start indices
    cu_ends:     [seq_len], dtype int32, end indices

    Returns:
    logits:      [seq_len, seq_len_kv], dtype float32 (must be initialized to -inf, because of causal masking)
    """

    BLOCK_KV = 128
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]
    # TODO: Currently assuming num_heads and head_size is power of 2.
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."
    # Initialize with -inf because of causal masking
    logits = torch.full(
        (seq_len, seq_len_kv),
        fill_value=-float("inf"),
        dtype=torch.float32,
        device=Q.device,
    )

    use_gluon = arch in SUPPORTED_ARCHS
    stride_q_s, stride_q_h, stride_q_d = Q.stride()
    stride_kv_s, stride_kv_d = KV.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()
    if not use_gluon:
        # heuristic for MFMA instruction shape
        matrix_instr_nonkdim = 32
        if seq_len <= 1024:
            matrix_instr_nonkdim = 16

        _fp8_mqa_logits_kernel[(seq_len,)](
            Q_ptr=Q,
            KV_ptr=KV,
            kv_scales_ptr=kv_scales,
            weights_ptr=weights,
            cu_start_ptr=cu_starts,
            cu_end_ptr=cu_ends,
            logits_ptr=logits,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            NUM_HEADS=num_heads,
            HEAD_SIZE=head_size,
            stride_q_s=stride_q_s,
            stride_q_h=stride_q_h,
            stride_q_d=stride_q_d,
            stride_kv_s=stride_kv_s,
            stride_kv_d=stride_kv_d,
            stride_w_s=stride_w_s,
            stride_w_h=stride_w_h,
            stride_logits_s=stride_logits_s,
            stride_logits_k=stride_logits_k,
            BLOCK_KV=BLOCK_KV,
            num_warps=4,
            num_stages=2,
            waves_per_eu=2,
            matrix_instr_nonkdim=matrix_instr_nonkdim,
        )
    else:
        num_buffers = 2

        if arch == "gfx950":
            num_buffers = 2
            loop_variant = 0
            waves_per_eu = 3
            num_chains = 4 if TRITON_BEYOND_37 else 0
            num_warps = 1
            block_kv = 32
            other = {}
        else:
            loop_variant = 1
            waves_per_eu = 1
            num_chains = 8
            num_warps = 4
            block_kv = 128
            other = {"LOOP_VARIANT":loop_variant}

        # Buffer ops use a 32-bit byte offset (2 GiB resource descriptor cap).
        # Fall back to plain global load/store when a tensor exceeds that.
        BUFFER_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
        use_buffer_load = KV.numel() * KV.element_size() < BUFFER_LIMIT_BYTES
        use_buffer_store = logits.numel() * logits.element_size() < BUFFER_LIMIT_BYTES
        _gluon_fp8_mqa_logits_kernel[(seq_len,)](
            Q_ptr=Q,
            KV_ptr=KV,
            kv_scales_ptr=kv_scales,
            weights_ptr=weights,
            cu_start_ptr=cu_starts,
            cu_end_ptr=cu_ends,
            logits_ptr=logits,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            NUM_HEADS=num_heads,
            HEAD_SIZE=head_size,
            stride_q_s=stride_q_s,
            stride_q_h=stride_q_h,
            stride_q_d=stride_q_d,
            stride_kv_s=stride_kv_s,
            stride_kv_d=stride_kv_d,
            stride_w_s=stride_w_s,
            stride_w_h=stride_w_h,
            stride_logits_s=stride_logits_s,
            stride_logits_k=stride_logits_k,
            BLOCK_KV=block_kv,
            NUM_WARPS=num_warps,
            NUM_BUFFERS=num_buffers,
            NUM_CHAINS=num_chains,
            ARCH_NAME=arch,
            USE_BUFFER_LOAD=use_buffer_load,
            USE_BUFFER_STORE=use_buffer_store,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            **other,
        )

    return logits
