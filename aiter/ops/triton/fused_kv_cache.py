import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.fused_kv_cache import (
    _fused_qk_rope_cosine_cache_llama_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_qk_rope_cosine_cache_llama(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pos: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_neox: bool,
    flash_layout: bool,
    apply_scale: bool = True,
    offs: torch.Tensor = None,
    q_out: torch.Tensor = None,
):
    """
    Perform RoPE on q and k and along the last dimension and copy k and v in to key_cache and value_cache inplace

    Key parameters:
    - q: shape (T, QH, D).
    - k: shape (T_slot, KH, D).
    - v: shape (T_slot, KH, D).
    - if flash_layout:
    -     key_cache: shape (T_cache, block_size, KH, D).
    -     value_cache: shape (T_cache, block_size, KH, D).
    - else:
    -     key_cache: shape (T_cache, KH, D // x, block_size, x).
    -     value_cache: shape (T_cache, KH, D, block_size).
    - slot_mapping: shape (T_slot, ).

    T is the number of decode tokens, T_cahce * block_size is the max number of tokens of kv_cache
    QH must be multiple of KH

    Returns:
    - q_out: same shape as input q.
    - key_cache: same shape as input key_cache (inplace).
    - value_cache: same shape as input value_cache (inplace).
    """
    _LOGGER.info(
        f"FUSED_QK_ROPE_COSINE_CACHE_LLAMA: q={tuple(q.shape)} k={tuple(k.shape)} "
        + f"pos={tuple(pos.shape)} cos={tuple(cos.shape)} sin={tuple(sin.shape)} key_cache={tuple(key_cache.shape)} value_cache={tuple(value_cache.shape)} slot_mapping={tuple(slot_mapping.shape)}"
    )

    t, qh, d = q.shape
    tk, kh, dk = k.shape
    tv, vh, dv = v.shape
    if flash_layout:
        t_cache, block_size, kh_cache, dk_cache = key_cache.shape
        t_cache_v, block_size_v, vh_cache, dv_cache = value_cache.shape
    else:
        t_cache, kh_cache, dkx_cache, block_size, x_cache = key_cache.shape
        t_cache_v, vh_cache, dv_cache, block_size_v = value_cache.shape
    (t_slot,) = slot_mapping.shape

    assert (
        t == tk == tv and t_slot <= tk
    ), f"Number of tokens should be identical for q, kand v. The number of tokens of slot_mapping should no more than that of q, k and v, {t=} {tk=} {tv=} {t_slot=}"
    assert (
        block_size == block_size_v
    ), f"block size should be identical for key_cache, and value_cache {block_size} {block_size_v}"
    assert (
        kh == vh == kh_cache == vh_cache
    ), "KV head should be identical for k, v, key_cache, and value_cache"
    assert (
        t_cache == t_cache_v
    ), "Number of tokens should be identical for key_cache, and value_cache"
    if flash_layout:
        assert (
            d == dk == dv == dk_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"
    else:
        assert (
            d == dk == dv == dkx_cache * x_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"
        assert x_cache == triton.next_power_of_2(x_cache), "x_size should be power of 2"

    assert d == triton.next_power_of_2(d), "D dimension should be power of 2"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    d_freq = cos.shape[-1]
    assert (d_freq == d // 2) or (
        d_freq == d
    ), "cos/sin last dim should be the same or half of the qk last dim"
    reuse_freqs_front_part = d_freq == d // 2

    if q_out is None:
        q_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)

    n_pid = t * qh + (t_slot - t) * kh
    grid = (n_pid, 1, 1)
    _fused_qk_rope_cosine_cache_llama_kernel[grid](
        q,
        k,
        v,
        pos,
        cos,
        sin,
        offs,
        key_cache,
        value_cache,
        slot_mapping,
        q_out,
        t,
        t_slot,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        cos.stride(0),
        cos.stride(-1),
        *q_out.stride(),
        key_cache.stride(0) if not flash_layout else key_cache.stride(0),
        key_cache.stride(1) if not flash_layout else key_cache.stride(2),
        key_cache.stride(2) if not flash_layout else key_cache.stride(3),
        key_cache.stride(3) if not flash_layout else key_cache.stride(1),
        key_cache.stride(4) if not flash_layout else 0,
        value_cache.stride(0) if not flash_layout else value_cache.stride(0),
        value_cache.stride(1) if not flash_layout else value_cache.stride(2),
        value_cache.stride(2) if not flash_layout else value_cache.stride(3),
        value_cache.stride(3) if not flash_layout else value_cache.stride(1),
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        QH_PER_KH=qh // kh,
        QH=qh,
        KH=kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        BLOCK_D_pe=d,
        BLOCK_D_HALF_pe=d // 2,
        BLOCK_SIZE=block_size,
        X_SIZE=x_cache if not flash_layout else 0,
        FLASH_LAYOUT=flash_layout,
        HAVE_POS=(offs is not None),
        HAVE_K_SCALE=(k_scale is not None and apply_scale),
        HAVE_V_SCALE=(v_scale is not None and apply_scale),
        num_warps=1,
    )
    return q_out, key_cache, value_cache
