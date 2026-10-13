import torch
import triton
from aiter.ops.triton._gluon_kernels.gfx1250.attention.mha_fp8 import _mha_fwd_subtiled


def get_mha_fwd_config(device: torch.device) -> dict:
    """Returns the default config for the fp8 MHA forward kernel."""
    return {
        "BLOCK_M": 256,
        "BLOCK_N": 128, 
        "NUM_BUFFERS": 2,
        "num_warps": 4,
    }


def mha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor | float,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
    o: torch.Tensor | None = None,
    sm_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    Multi-head attention forward with per-tensor fp8 (e4m3) Q, K and V.

    q, k, v:   [B, H, S, D] (layout "bhsd") dtype float8_e4m3fn. 
    q_scale, k_scale, v_scale: per-tensor dequant scales, fp32 scalars (python
               float or a 1-element / 0-dim tensor).
    o:         optional output buffer, same shape as q, dtype bfloat16. The
               kernel downcasts the fp32 accumulator on the way out, so
               bfloat16 is the only supported output dtype.
    sm_scale:  softmax scale; defaults to 1 / sqrt(D).
    causal:    not implemented by the kernel yet (the IS_CAUSAL constexpr is
               unused), so only causal=False is accepted.

    Returns:
    o:         same shape as q, dtype bfloat16.
    """
    for t, name in zip((q, k, v), ("q", "k", "v")):
        assert t.dtype == torch.float8_e4m3fn, f"{name} must be float8_e4m3fn"
        assert t.dim() == 4, f"{name} must be 4D, got {t.dim()}D"

    batch, num_heads, seqlen, head_dim = q.shape

    config = get_mha_fwd_config(q.device)
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    NUM_BUFFERS = config["NUM_BUFFERS"]

    head_dim_pow2 = triton.next_power_of_2(head_dim)
    # wmma_scaled runs a 16x16x128 fp8 instruction over the head dim.
    assert head_dim_pow2 >= 128, f"head_dim must be at least 128, got {head_dim}"
    # Only the final K/V tile is masked, so every tile the prologue prefetches
    # has to be a real one.
    assert triton.cdiv(seqlen, BLOCK_N) >= NUM_BUFFERS - 1, (
        f"seqlen {seqlen} is too short for {NUM_BUFFERS} pipeline buffers"
    )

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)

    def _as_scale(s, name):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32, device=q.device)
        assert s.numel() == 1, f"{name} must be a per-tensor (scalar) scale"
        return s.to(device=q.device, dtype=torch.float32)

    q_scale = _as_scale(q_scale, "q_scale")
    k_scale = _as_scale(k_scale, "k_scale")
    v_scale = _as_scale(v_scale, "v_scale")

    if o is None:
        o = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    else:
        assert o.shape == q.shape, "output must have the same shape as q"
        # The kernel stores acc.to(bfloat16) through a descriptor typed by o.
        assert o.dtype == torch.bfloat16, "output must be bfloat16"

    stride_q_z, stride_q_h, stride_q_m, stride_q_d = (q.stride(i) for i in range(4))
    stride_k_z, stride_k_h, stride_k_n, stride_k_d = (k.stride(i) for i in range(4))
    stride_v_z, stride_v_h, stride_v_n, stride_v_d = (v.stride(i) for i in range(4))
    stride_o_z, stride_o_h, stride_o_m, stride_o_d = (o.stride(i) for i in range(4))
    if seqlen % 2 == 0 and head_dim == head_dim_pow2:
        preshuffle_factor_k = 2
    else:
        preshuffle_factor_k = 1
    preshuffle_factor_v = 1
    stride_k_n *= preshuffle_factor_k
    stride_v_n *= preshuffle_factor_v

    grid = (triton.cdiv(seqlen, BLOCK_M), num_heads, batch)
    _mha_fwd_subtiled[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        q_scale_ptr=q_scale,
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        o_ptr=o,
        stride_q_z=stride_q_z,
        stride_q_h=stride_q_h,
        stride_q_m=stride_q_m,
        stride_q_d=stride_q_d,
        stride_k_z=stride_k_z,
        stride_k_h=stride_k_h,
        stride_k_n=stride_k_n,
        stride_k_d=stride_k_d,
        stride_v_z=stride_v_z,
        stride_v_h=stride_v_h,
        stride_v_n=stride_v_n,
        stride_v_d=stride_v_d,
        stride_o_z=stride_o_z,
        stride_o_h=stride_o_h,
        stride_o_m=stride_o_m,
        stride_o_d=stride_o_d,
        SM_SCALE=sm_scale,
        CAUSAL=causal,
        NUM_HEADS=num_heads,
        SEQLEN=seqlen,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=head_dim,
        HEAD_DIM_POW2=head_dim_pow2,
        NUM_BUFFERS=NUM_BUFFERS,
        EVEN_N=(seqlen % BLOCK_N == 0) and (triton.cdiv(seqlen, BLOCK_N) >= 3),
        PRESHUFFLE_FACTOR_K = preshuffle_factor_k,
        PRESHUFFLE_FACTOR_V = preshuffle_factor_v,
        num_warps=config["num_warps"],
        waves_per_eu=0,
    )

    return o
