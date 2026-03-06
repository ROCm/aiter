import torch
import triton
from aiter.ops.triton._triton_kernels.rope.fused_qkv_split_qk_rope import (
    _fused_qkv_split_qk_rope_kernel,
)


def fused_qkv_split_qk_rope(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    qh: int,
    kvh: int,
    head_dim: int,
    is_neox: bool = True,
    offsets: torch.Tensor = None,
    reuse_freqs_front_part: bool = True,
    nope_first: bool = False,
):
    T = qkv.shape[0]
    q_size = qh * head_dim
    kv_size = kvh * head_dim

    assert qh >= kvh and qh % kvh == 0, "qh must be mutiple of kvh"

    q = torch.empty((qkv.shape[0], qh, head_dim), dtype=qkv.dtype, device=qkv.device)
    k = torch.empty((qkv.shape[0], kvh, head_dim), dtype=qkv.dtype, device=qkv.device)
    v = torch.empty((qkv.shape[0], kvh, head_dim), dtype=qkv.dtype, device=qkv.device)

    if cos.shape[-1] == head_dim // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif cos.shape[-1] == head_dim // 4:
        have_nope = True
    else:
        have_nope = False

    assert qkv.shape[-1] == q_size + 2 * kv_size, "Shape error"
    assert head_dim // ((2 if have_nope else 1)) == triton.next_power_of_2(
        head_dim // ((2 if have_nope else 1))
    ), "head_dim should be power of 2"

    if have_nope:
        BLOCK_D = head_dim // 2
        BLOCK_D_HALF = head_dim // 4
    else:
        BLOCK_D = head_dim
        BLOCK_D_HALF = head_dim // 2

    BLOCK_T = 32
    num_warps = 4
    waves_per_eu = 0
    grid = (triton.cdiv(T, BLOCK_T), qh, 1)

    _fused_qkv_split_qk_rope_kernel[grid](
        qkv,
        cos,
        sin,
        positions,
        offsets,
        q,
        k,
        v,
        T,
        *qkv.stride(),
        cos.stride(0),
        cos.stride(-1),
        *positions.stride(),
        *q.stride(),
        *k.stride(),
        HAVE_NOPE=have_nope,
        NOPE_FIRST=nope_first,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        HAVE_POS=(positions is not None),
        HAVE_OFFS=(offsets is not None),
        QH=qh,
        KVH=kvh,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        BLOCK_D_HALF=BLOCK_D_HALF,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    return q, k, v
