# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from ..jit.core import compile_ops
from ..utility.dtypes import get_dtype_fp8
from typing import Optional


@compile_ops(
    "module_fused_qk_norm_rope_cache_quant_shuffle",
    fc_name="fused_qk_norm_rope_cache_quant_shuffle",
    develop=True,
)
def _fused_qk_norm_rope_cache_quant_shuffle_hip(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    qw: Tensor,
    kw: Tensor,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
    pos_ids: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,  # 4D [B,Hv,D,page] or 5D shuffle [B,Hv,page//x,D,x], x=16//elem_size
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
) -> None: ...


def fused_qk_norm_rope_cache_quant_shuffle(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    qw: Tensor,
    kw: Tensor,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
    pos_ids: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
) -> None:
    _fused_qk_norm_rope_cache_quant_shuffle_hip(
        q,
        k,
        v,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        qw,
        kw,
        cos_sin_cache,
        is_neox_style,
        pos_ids,
        k_cache,
        v_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


@compile_ops(
    "module_fused_qk_norm_rope_cache_quant_shuffle",
    fc_name="fused_qk_rmsnorm",
    develop=True,
)
def _fused_qk_rmsnorm_kernel(
    q: Tensor,
    q_weight: Tensor,
    q_eps: float,
    k: Tensor,
    k_weight: Tensor,
    k_eps: float,
    q_out: Tensor,
    k_out: Tensor,
) -> None: ...


@compile_ops(
    "module_fused_qk_norm_rope_cache_quant_shuffle",
    fc_name="minimax_qk_norm_rope",
    develop=True,
)
def _minimax_qk_norm_rope_kernel(
    qkv: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos_sin_cache: Tensor,
    positions: Tensor,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    is_neox_style: bool,
    q_out: Tensor,
    k_out: Tensor,
    v_out: Tensor,
) -> None: ...


def minimax_qk_norm_rope(
    qkv: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos_sin_cache: Tensor,
    positions: Tensor,
    *,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    is_neox_style: bool,
    q_out: Optional[Tensor] = None,
    k_out: Optional[Tensor] = None,
    v_out: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """MiniMax TP1 qkv split with full-vector q/k RMSNorm and RoPE.

    Unlike the generic qk-norm RoPE kernels, MiniMax normalizes Q across
    num_heads_q * head_dim and K across num_heads_k * head_dim.
    """
    num_tokens = qkv.size(0)
    q_size = num_heads_q * head_dim
    kv_size = num_heads_k * head_dim
    if q_out is None:
        q_out = torch.empty((num_tokens, q_size), dtype=qkv.dtype, device=qkv.device)
    if k_out is None:
        k_out = torch.empty((num_tokens, kv_size), dtype=qkv.dtype, device=qkv.device)
    if v_out is None:
        v_out = torch.empty((num_tokens, kv_size), dtype=qkv.dtype, device=qkv.device)

    _minimax_qk_norm_rope_kernel(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        num_heads_q,
        num_heads_k,
        head_dim,
        rotary_dim,
        eps,
        is_neox_style,
        q_out,
        k_out,
        v_out,
    )
    return q_out, k_out, v_out


_FUSED_QK_FALLBACK_M = 16384


def _fused_qk_rmsnorm(
    q_out: Optional[Tensor],
    q: Tensor,
    q_weight: Tensor,
    q_eps: float,
    k_out: Optional[Tensor],
    k: Tensor,
    k_weight: Tensor,
    k_eps: float,
) -> tuple[Tensor, Tensor]:
    if q_out is None:
        q_out = torch.empty_like(q, dtype=q.dtype, device=q.device)
    if k_out is None:
        k_out = torch.empty_like(k, dtype=k.dtype, device=k.device)

    m = q.size(0)
    if m >= _FUSED_QK_FALLBACK_M:
        from .rmsnorm import rmsnorm

        rmsnorm(q_out, q, q_weight, q_eps)
        rmsnorm(k_out, k, k_weight, k_eps)
    else:
        _fused_qk_rmsnorm_kernel(q, q_weight, q_eps, k, k_weight, k_eps, q_out, k_out)
    return q_out, k_out


@compile_ops("module_fused_qk_norm_rope_cache_quant_shuffle", develop=True)
def fused_qk_norm_rope_cache_block_quant_shuffle(
    qkv: Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    qw: Tensor,
    kw: Tensor,
    cos_sin_cache: Tensor,
    is_neox_style: bool,
    pos_ids: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    slot_mapping: Tensor,
    cu_q_len: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
    max_tokens_per_batch: int = 0,
) -> None: ...


@compile_ops("module_fused_qk_norm_rope_cache_quant_shuffle", develop=True)
def fused_qk_norm_rope_cache_pts_quant_shuffle(
    qkv: Tensor,
    qw: Tensor,
    kw: Tensor,
    cos_sin: Tensor,
    positions: Tensor,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    is_neox_style: bool,
    eps: float,
    q_out: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    slot_mapping: Tensor,
    per_tensor_k_scale: Tensor,
    per_tensor_v_scale: Tensor,
    k_out: Optional[Tensor],
    v_out: Optional[Tensor],
    return_kv: bool,
    use_shuffle_layout: bool,
    block_size: int,
    x: int,
    rotary_dim: int = 0,
) -> None: ...


@compile_ops("module_fused_qk_norm_rope_cache_quant_shuffle", develop=True)
def fused_qk_norm_rope_2way(
    q0: Tensor,
    k0: Tensor,
    q1: Tensor,
    k1: Tensor,
    w_q0: Tensor,
    w_k0: Tensor,
    w_q1: Tensor,
    w_k1: Tensor,
    cos_sin0: Tensor,
    cos_sin1: Tensor,
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
    out_q01: Tensor,
    out_k01: Tensor,
) -> None: ...


@compile_ops("module_fused_qk_norm_rope_cache_quant_shuffle", develop=True)
def fused_qk_norm_rope_1way(
    q: Tensor,
    k: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    cos_sin: Tensor,
    batch_size: int,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
    out_q: Tensor,
    out_k: Tensor,
) -> None:
    """Fused per-head RMSNorm + RoPE on q/k for the 1way (single-stream) layout.

    Dtype contract:
        q, k, w_q, w_k, out_q, out_k : torch.bfloat16 or torch.float16 (same dtype)
        cos_sin                      : torch.float32  (REQUIRED)

    cos_sin must be float32 to match the diffusers / qwen-image-edit reference
    (RoPE freqs are computed in fp32 there and the precision is consumed by the
    fp32 rope multiply). Passing bf16/fp16 cos_sin will raise inside the kernel.
    """
    ...


@compile_ops(
    "module_fused_qk_norm_rope_cache_quant_shuffle",
    fc_name="fused_qk_norm_rope_group_quant",
    develop=True,
)
def _fused_qk_norm_rope_group_quant_kernel(
    q: Tensor,  # [num_tokens, num_heads, head_dim]
    kv: Tensor,  # [num_tokens, (k_num_heads,) head_dim]
    k_rope_buff: Tensor,  # [num_tokens, (k_num_heads,) pe_dim] bf16 (RoPE'd K-PE)
    k_weight: Tensor,  # [head_dim] RMSNorm weights
    k_nope_scale_buff: Tensor,  # [num_tokens, (k_num_heads,) 512B] K nope+scale, token-contiguous
    q_nope_scale_buff: Tensor,  # [num_tokens, num_heads, head_dim] bf16 (full Q) OR fp8 (nope+scale)
    positions: Tensor,  # [num_tokens]
    cos_cache: Tensor,  # [max_position, rot_dim//2]
    sin_cache: Tensor,  # [max_position, rot_dim//2]
    eps: float,
    is_neox: bool,
    # q_weight: optional per-channel RMSNorm weight for Q [head_dim]. None = weightless (V4-Pro).
    q_weight: Optional[Tensor] = None,
    # q_scale: legacy separate Q scale (unused on the fp8 inline path; scale is written
    #   into q_nope_scale_buff at bytes [nope_dim : nope_dim+2*num_nope_groups), each tile-scale x2).
    q_scale: Optional[Tensor] = None,
    # quant_group_size: 1xG block-scale width for Q. Must be one of {32, 64, 128} and divide head_dim.
    # Ignored when q_nope_scale_buff is bf16.
    quant_group_size: int = 64,
    # scale_dtype: 'e8m0' (1-byte MX) or 'fp32' (4-byte). Ignored when q_nope_scale_buff is bf16.
    scale_dtype: str = "e8m0",
    # q_rope_buff: rotated Q-PE bf16 [num_tokens, num_heads, pe_dim]; required when Q is fp8
    #   (fp8 Q mirrors K: nope fp8 + inline dup e8m0 scale in q_nope_scale_buff, PE bf16 here). None for bf16 Q.
    q_rope_buff: Optional[Tensor] = None,
    # --- Optional fused SWA ring-cache write (decode-only) ---
    # swa_nope_scale_buff: ring [num_slots, cache_size, entry] mirroring k_nope_scale_buff
    #   (same nope fp8 + inline-scale layout/dtype). swa_rope_buff: ring [num_slots, cache_size, pe_dim] bf16.
    # state_slot_mapping: [bs] int32 per-seq ring slot. batch_id_per_token: [num_tokens] int32,
    #   token->seq (-1 = CG-pad, skipped). K row -> swa_*[slot, positions[t] % cache_size, :].
    # All four together or all None.
    swa_nope_scale_buff: Optional[Tensor] = None,
    swa_rope_buff: Optional[Tensor] = None,
    state_slot_mapping: Optional[Tensor] = None,
    batch_id_per_token: Optional[Tensor] = None,
) -> None: ...


def fused_qk_norm_rope_group_quant(
    q: Tensor,  # [num_tokens, num_heads, head_dim]
    kv: Tensor,  # [num_tokens, (num_kv_heads,) head_dim]
    k_weight: Tensor,  # [head_dim] RMSNorm weights
    positions: Tensor,  # [num_tokens] int64
    cos_cache: Tensor,  # [max_position, rot_dim//2]
    sin_cache: Tensor,  # [max_position, rot_dim//2]
    eps: float,
    is_neox: bool = False,
    # q_out_dtype controls whether Q is quantized: fp8 -> Q group-quant (q_scale produced);
    # bf16/fp16 -> Q stays unquantized. Default bf16. Ignored when an explicit `q_nope_scale_buff` is passed (dtype wins).
    q_out_dtype: torch.dtype = torch.bfloat16,
    q_nope_scale_buff: Optional[
        Tensor
    ] = None,  # bf16: [.,H,512] full rotated Q; fp8: [.,H,512] Q nope+scale. dtype decides quant. Alloc if None.
    q_rope_buff: Optional[
        Tensor
    ] = None,  # [num_tokens, num_heads, rot_dim] bf16 rotated Q-PE; only for fp8 Q. Alloc if None.
    k_nope_scale_buff: Optional[
        Tensor
    ] = None,  # [num_tokens, num_kv_heads, 512] fp8 K nope+scale; allocated (zeroed) if None
    k_rope_buff: Optional[
        Tensor
    ] = None,  # [num_tokens, num_kv_heads, rot_dim] bf16 rotated K-PE; allocated if None
    q_weight: Optional[
        Tensor
    ] = None,  # optional per-channel Q RMSNorm weight [head_dim]
    quant_group_size: int = 64,
    scale_dtype: str = "e8m0",
    # --- Optional fused SWA ring-cache write (decode-only) ---
    # swa_nope_scale_buff: ring [num_slots, cache_size, entry] mirroring k_nope_scale_buff
    #   (same nope fp8 + inline-scale layout/dtype). swa_rope_buff: ring [num_slots, cache_size, rot_dim] bf16.
    # state_slot_mapping: [bs] int32 per-seq ring slot. batch_id_per_token: [num_tokens] int32 (-1 = skip).
    # All four together or all None. Caller owns the ring buffers (persistent state cache).
    swa_nope_scale_buff: Optional[Tensor] = None,
    swa_rope_buff: Optional[Tensor] = None,
    state_slot_mapping: Optional[Tensor] = None,
    batch_id_per_token: Optional[Tensor] = None,
):
    """DeepSeek-V4 fused Q/K RMSNorm + RoPE + group-quant WITHOUT a paged KV cache.

    Thin, dense (token-contiguous) wrapper over the fused QK norm+RoPE+group-quant kernel.
    K NoPE is fp8 with a 1x64 e8m0 group scale; K RoPE (pe) stays bf16 (NOT quantized).
    No ``slot_mapping`` / paging.

    See also ``fused_kv_norm_rope_group_quant`` -- the K-only fast path for call
    sites that do NOT need Q (e.g. the V4-Pro Indexer's Compressor). It skips the
    Q wave entirely (~1.6x faster at large T) and is bit-exact with this function's
    K-side. Prefer it whenever you would pass a throwaway Q here.

    K output is split into the two buffers the v4 nm asm attention kernel reads:

    ``k_nope_scale_buff`` -- per (token, kv_head), 512 bytes (head_dim=512):
        [0   : 448)  K-nope fp8                                            (448 B)
        [448 : 462)  e8m0 scale, 2*(nope_dim/64)=14 B, each tile-scale x2  (s0,s0,..,s6,s6)
        [462 : 512)  pad (uninitialised -- never read by the asm reader)    (50 B)
      The asm reader reads each tile scale TWICE consecutively, hence the x2 duplication.

    ``k_rope_buff`` -- per (token, kv_head), rotated K-PE bf16 [rot_dim]    (128 B).

    Q output mirrors K when fp8 (``q_out_dtype=fp8``):
      ``q_nope_scale_buff`` -- [.,H,512]: Q nope fp8 + 14 dup e8m0 scale + pad.
      ``q_rope_buff``       -- rotated Q-PE bf16 [.,H,rot_dim] (Q-PE NOT quantized).
    For bf16 Q (default, DeepSeek-V4 / ATOM sparse_attn): ``q_nope_scale_buff`` is the
    full [.,H,512] bf16 rotated Q and ``q_rope_buff`` is None.

    Layout is always nope-first (V4); the kernel hardcodes it.
    Returns ``(q_nope_scale_buff, q_rope_buff_or_None, k_nope_scale_buff, k_rope_buff)``.
    """
    assert q.dim() == 3, "q must be [num_tokens, num_heads, head_dim]"
    num_tokens, num_heads, head_dim = q.shape
    num_kv_heads = kv.shape[1] if kv.dim() == 3 else 1
    rot_dim = cos_cache.shape[-1] * 2
    # k_nope_scale_buff / fp8 q_nope_scale_buff entry = 512 B (head_dim): nope + 2*(nope/G) dup
    # e8m0 scale + pad (14 B for G=64, 28 B for G=32).
    k_entry_bytes = head_dim

    from .. import dtypes

    q_is_fp8 = (
        q_nope_scale_buff.dtype if q_nope_scale_buff is not None else q_out_dtype
    ) == dtypes.fp8
    if q_is_fp8:
        # Only fp8 Q is group-quantised, and only over the NoPE region (head_dim - rot_dim);
        # the trailing rot_dim is RoPE'd bf16 and never quantised. bf16 Q ignores group size.
        assert quant_group_size in (
            32,
            64,
        ), f"quant_group_size must be one of {{32, 64}}, got {quant_group_size}"
        assert (
            head_dim - rot_dim
        ) % quant_group_size == 0, (
            "NoPE size (head_dim - rot_dim) must be divisible by quant_group_size"
        )
    if q_nope_scale_buff is None:
        # dtype = q_out_dtype covers both cases: when q_is_fp8 the buffer is None
        # so q_is_fp8 == (q_out_dtype == fp8), i.e. q_out_dtype is already fp8
        # (nope+scale, pad left uninitialised -- asm reader ignores it); otherwise
        # it is the plain [.,H,512] rotated bf16 Q.
        q_nope_scale_buff = torch.empty(
            (num_tokens, num_heads, head_dim), dtype=q_out_dtype, device=q.device
        )
    if q_is_fp8 and q_rope_buff is None:
        q_rope_buff = torch.empty(
            (num_tokens, num_heads, rot_dim), dtype=kv.dtype, device=q.device
        )
    if not q_is_fp8:
        q_rope_buff = (
            None  # bf16 Q: PE stays in q_nope_scale_buff, no separate rope buffer
        )
    if k_nope_scale_buff is None:
        # The kernel writes nope[0:nope) + 14 scale bytes; the trailing pad is
        # never read by the asm reader, so no zero-init is needed.
        k_nope_scale_buff = torch.empty(
            (num_tokens, num_kv_heads, k_entry_bytes), dtype=dtypes.fp8, device=q.device
        )
    if k_rope_buff is None:
        k_rope_buff = torch.empty(
            (num_tokens, num_kv_heads, rot_dim), dtype=kv.dtype, device=q.device
        )

    _fused_qk_norm_rope_group_quant_kernel(
        q,
        kv,
        k_rope_buff,
        k_weight,
        k_nope_scale_buff,
        q_nope_scale_buff,
        positions,
        cos_cache,
        sin_cache,
        eps,
        is_neox,
        q_weight=q_weight,
        quant_group_size=quant_group_size,
        scale_dtype=scale_dtype,
        q_rope_buff=q_rope_buff,
        swa_nope_scale_buff=swa_nope_scale_buff,
        swa_rope_buff=swa_rope_buff,
        state_slot_mapping=state_slot_mapping,
        batch_id_per_token=batch_id_per_token,
    )
    return q_nope_scale_buff, q_rope_buff, k_nope_scale_buff, k_rope_buff


@compile_ops(
    "module_fused_qk_norm_rope_cache_quant_shuffle",
    fc_name="fused_qk_norm_rope_2way_fp8_perhead_quant",
    develop=True,
)
def _fused_qk_norm_rope_2way_fp8_perhead_quant_kernel(
    q0: Tensor,
    k0: Tensor,
    q1: Tensor,
    k1: Tensor,
    w_q0: Tensor,
    w_k0: Tensor,
    w_q1: Tensor,
    w_k1: Tensor,
    cos_sin0: Tensor,
    cos_sin1: Tensor,
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
    q_fp8: Tensor,
    k_fp8: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    q_unquantized: Tensor,
    k_unquantized: Tensor,
) -> None: ...


def fused_qk_norm_rope_2way_fp8_perhead_quant(
    q0: Tensor,
    k0: Tensor,
    q1: Tensor,
    k1: Tensor,
    w_q0: Tensor,
    w_k0: Tensor,
    w_q1: Tensor,
    w_k1: Tensor,
    cos_sin0: Tensor,
    cos_sin1: Tensor,
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
    out_q01: Optional[Tensor] = None,
    out_k01: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Same as the pertensor variant, but with per-(batch, head) descales.

    Returns (q_fp8, k_fp8, q_descale, k_descale, q_bf16, k_bf16) where
    q_descale.shape == (batch_size, num_heads_q) and
    k_descale.shape == (batch_size, num_heads_k). These shapes match what
    CK FP8 flash attention accepts natively.
    """
    want_bf16 = out_q01 is not None or out_k01 is not None
    total_tokens = num_tokens0 + num_tokens1
    fp8_dtype = get_dtype_fp8()

    q_fp8 = torch.empty(
        (batch_size, total_tokens, num_heads_q, head_size),
        dtype=fp8_dtype,
        device=q0.device,
    )
    k_fp8 = torch.empty(
        (batch_size, total_tokens, num_heads_k, head_size),
        dtype=fp8_dtype,
        device=k0.device,
    )
    q_descale = torch.empty(
        (batch_size, num_heads_q), dtype=torch.float32, device=q0.device
    )
    k_descale = torch.empty(
        (batch_size, num_heads_k), dtype=torch.float32, device=k0.device
    )
    q_unquantized = (
        out_q01
        if out_q01 is not None
        else torch.empty(
            (batch_size, total_tokens, num_heads_q, head_size),
            dtype=q0.dtype,
            device=q0.device,
        )
    )
    k_unquantized = (
        out_k01
        if out_k01 is not None
        else torch.empty(
            (batch_size, total_tokens, num_heads_k, head_size),
            dtype=k0.dtype,
            device=k0.device,
        )
    )

    _fused_qk_norm_rope_2way_fp8_perhead_quant_kernel(
        q0,
        k0,
        q1,
        k1,
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        batch_size,
        num_tokens0,
        num_tokens1,
        num_heads_q,
        num_heads_k,
        head_size,
        is_interleaved,
        eps,
        q_fp8,
        k_fp8,
        q_descale,
        k_descale,
        q_unquantized,
        k_unquantized,
    )

    if not want_bf16:
        q_unquantized = torch.empty(0, dtype=q0.dtype, device=q0.device)
        k_unquantized = torch.empty(0, dtype=k0.dtype, device=k0.device)

    return q_fp8, k_fp8, q_descale, k_descale, q_unquantized, k_unquantized


@compile_ops(
    "module_fused_qk_norm_rope_cache_quant_shuffle",
    fc_name="fused_kv_norm_rope_group_quant",
    develop=True,
)
def _fused_kv_norm_rope_group_quant_kernel(
    kv: Tensor,  # [num_tokens, (NK=1,) head_dim]
    k_rope_buff: Tensor,  # paged rope cache [num_blocks, page_size, pe_dim] bf16 (MQA: NK=1)
    k_weight: Tensor,  # [head_dim] RMSNorm weights
    k_nope_scale_buff: Tensor,  # paged nope+scale cache [num_blocks, page_size, head_dim] fp8 (MQA: NK=1)
    positions: Tensor,  # [num_tokens]
    slot_mapping: Tensor,  # [num_tokens] int64 flat slot = block*page_size + offset
    cos_cache: Tensor,  # [max_position, rot_dim//2]
    sin_cache: Tensor,  # [max_position, rot_dim//2]
    eps: float,
    is_neox: bool,
    # quant_group_size / scale_dtype are accepted for API symmetry with the
    # QK entry. The kernel currently hardcodes G=64 and e8m0 (the V4-Pro shape).
    quant_group_size: int = 64,
    scale_dtype: str = "e8m0",
) -> None: ...


def fused_kv_norm_rope_group_quant(
    kv: Tensor,  # [num_tokens, head_dim] OR [num_tokens, num_kv_heads, head_dim]
    kv_weight: Tensor,  # [head_dim] RMSNorm weights (reuses K-side gamma)
    positions: Tensor,  # [num_tokens] int64
    slot_mapping: Tensor,  # [num_tokens] int64 flat slot = block*page_size + offset
    cos_cache: Tensor,  # [max_position, rot_dim//2]
    sin_cache: Tensor,  # [max_position, rot_dim//2]
    eps: float,
    k_nope_scale_buff: Tensor,  # paged [num_blocks, page_size, head_dim] fp8 (caller-owned KV cache, MQA)
    k_rope_buff: Tensor,  # paged [num_blocks, page_size, rot_dim] bf16 (caller-owned KV cache, MQA)
    is_neox: bool = False,
    quant_group_size: int = 64,
    scale_dtype: str = "e8m0",
) -> tuple[Tensor, Tensor]:
    """KV-only fused RMSNorm + GPT-J/NeoX RoPE + FP8 group-quant, scattered into
    a PAGED KV cache via ``slot_mapping``.

    RMSNorm the KV, apply RoPE on the PE tail (stays bf16), 1xG e8m0 FP8
    group-quant the NoPE part, and write each token into the paged cache at the
    flat slot ``slot_mapping[token] = physical_block*page_size + offset`` (the
    kernel splits it against the caches' ``[num_blocks, page_size, entry]``
    strides; MQA so there is no num_kv_heads dim). A negative slot skips the
    token. Bit-exact (per entry) with the K-side of ``fused_qk_norm_rope_group_quant``.

    Per-entry layout (V4 nm asm sparse-attn reader, nope-first):
        ``k_nope_scale_buff`` entry -- ``head_dim`` fp8 bytes: ``[0:nope_dim)``
            K-nope fp8, then ``2*nGroups`` e8m0 scale bytes (each tile-scale
            duplicated x2), then zero pad (must be pre-zeroed by the caller).
        ``k_rope_buff`` entry -- rotated K-PE bf16 ``[rot_dim]``.

    The two caches are caller-owned (sized by num_blocks*page_size) and REQUIRED
    -- this op writes into an existing paged cache, it does not allocate one.

    Supported (head_dim, rot_dim, group_size) shapes are defined by
    ``KV_K_ONLY_DISPATCH_TABLE`` in
    ``csrc/kernels/fused_qk_norm_rope_cache_quant.cu`` (default ``(512, 64, 64)``
    is V4-Pro). An unsupported shape raises from the kernel.

    Args:
        kv: ``[T, D]`` or ``[T, NK, D]`` bf16 (MQA: ``NK == 1``).
        kv_weight: ``[D]`` bf16 RMSNorm gamma.
        positions: ``[T]`` int64 RoPE positions.
        slot_mapping: ``[T]`` int64 flat destination slot per token.
        cos_cache, sin_cache: ``[max_pos, rot_dim//2]`` bf16 RoPE tables.
        eps: RMSNorm epsilon.
        k_nope_scale_buff: paged fp8 cache ``[num_blocks, page_size, D]`` (MQA);
            the trailing pad of each entry must read back zero (pre-zero it).
        k_rope_buff: paged bf16 cache ``[num_blocks, page_size, rot_dim]`` (MQA).
        is_neox: NeoX (half-split) vs GPT-J (adjacent-pair) PE rotation.
        quant_group_size: ``G`` for the NoPE e8m0 scale; must divide
            ``head_dim - rot_dim``. Default 64.
        scale_dtype: only ``"e8m0"`` supported.

    Returns:
        ``(k_nope_scale_buff, k_rope_buff)`` (the same caller tensors, written).
    """
    if kv.dim() == 2:
        kv_3d = kv.unsqueeze(1)
    elif kv.dim() == 3:
        kv_3d = kv
    else:
        raise ValueError(
            f"kv must be 2D [T,D] or 3D [T,NK,D]; got rank {kv.dim()} shape={tuple(kv.shape)}"
        )
    num_tokens, num_kv_heads, head_dim = kv_3d.shape
    rot_dim = cos_cache.shape[-1] * 2
    nope_dim = head_dim - rot_dim
    if nope_dim <= 0:
        raise ValueError(
            f"rot_dim ({rot_dim}) must be < head_dim ({head_dim}); cos_cache last-dim must be < head_dim/2"
        )
    if nope_dim % quant_group_size != 0:
        raise ValueError(
            f"(head_dim - rot_dim) = {nope_dim} must be divisible by quant_group_size={quant_group_size}"
        )

    _fused_kv_norm_rope_group_quant_kernel(
        kv_3d,
        k_rope_buff,
        kv_weight,
        k_nope_scale_buff,
        positions,
        slot_mapping,
        cos_cache,
        sin_cache,
        eps,
        is_neox,
        quant_group_size=quant_group_size,
        scale_dtype=scale_dtype,
    )
    return k_nope_scale_buff, k_rope_buff


@compile_ops(
    "module_fused_qk_norm_rope_cache_quant_shuffle",
    fc_name="v_2way_per_head_fp8_quant",
    develop=True,
)
def _v_2way_per_head_fp8_quant_kernel(
    v0: Tensor,
    v1: Tensor,
    v_fp8: Tensor,
    v_descale: Tensor,
) -> None: ...


def v_2way_per_head_fp8_quant(v0: Tensor, v1: Tensor) -> tuple[Tensor, Tensor]:
    """Per-(batch, head) FP8 quant for concatenated [v0, v1] without bf16 cat."""
    batch_size = v0.size(0)
    num_heads = v0.size(2)
    head_size = v0.size(3)
    total_tokens = v0.size(1) + v1.size(1)
    fp8_dtype = get_dtype_fp8()
    v_fp8 = torch.empty(
        (batch_size, total_tokens, num_heads, head_size),
        dtype=fp8_dtype,
        device=v0.device,
    )
    v_descale = torch.empty(
        (batch_size, num_heads), dtype=torch.float32, device=v0.device
    )
    _v_2way_per_head_fp8_quant_kernel(v0, v1, v_fp8, v_descale)
    return v_fp8, v_descale
