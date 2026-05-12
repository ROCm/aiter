"""
HIP/ASM GDN decode kernel.

Drop-in replacement for fused_sigmoid_gating_delta_rule_update (Triton)
in decode mode.  Uses inline-assembly optimized kernel with template-
specialised head dispatch (TP=8 and Full-Heads paths).

State layout: expects [pool, HV, V, K] (VK layout), so callers can
transpose recurrent state once before decode and avoid per-step transpose
overhead.

Kernel parameters are specialized for Qwen3.5:
  K_heads=16, V_heads=32, K=128, V=128, bf16.

pool_idx sorting for L2-cache-friendly state access is handled inside
the C++ extension (configurable via HIP_GDN_SORT_IDX_BS env var).
"""

from typing import Optional

import torch

_ext = None


def _load_extension():
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load
    import os

    src_dir = os.path.dirname(os.path.abspath(__file__))
    gpu_archs = os.environ.get("GPU_ARCHS") or os.environ.get("AITER_HIP_GDN_ARCH")
    arch = (gpu_archs.split(";")[0] if gpu_archs else "gfx942").strip()
    _ext = load(
        name="hip_gdn_decode_ext",
        sources=[
            os.path.join(src_dir, "gdn_decode_ext.cpp"),
            os.path.join(src_dir, "gdn_decode_kernel_hip.hip"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", f"--offload-arch={arch}", "-std=c++17"],
        verbose=False,
    )
    return _ext


def hip_fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
):
    """VK decode kernel (inline-ASM) — state must already be in [V,K] layout."""
    ext = _load_extension()

    B, T, H, K = q.shape
    HV = v.shape[2]

    if scale is None:
        scale = K ** -0.5

    N = B * T if cu_seqlens is None else len(cu_seqlens) - 1

    o = torch.empty_like(v)

    dt_bias_bf16 = dt_bias.to(torch.bfloat16) if dt_bias.dtype != torch.bfloat16 else dt_bias

    indices_int32 = (
        initial_state_indices.to(torch.int32)
        if initial_state_indices.dtype != torch.int32
        else initial_state_indices
    )

    batch_size = N
    seq_length = 1 if cu_seqlens is not None else T
    num_k_heads = H
    num_v_heads = HV

    ext.hip_gdn_decode_asm_inplace(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
        dt_bias_bf16,
        A_log.contiguous(),
        indices_int32,
        initial_state_source,
        o,
        batch_size,
        seq_length,
        1,  # num_v_blocks (auto-selected by launcher)
        use_qk_l2norm_in_kernel,
        scale,
        num_k_heads,
        num_v_heads,
    )

    return o
