# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from typing import Literal

import torch
import triton
import triton.language as tl
from packaging.version import Version

from aiter.ops.triton._gluon_kernels.gfx950.attention.mha import (
    _attn_fwd as _gluon_attn_fwd,
)
from aiter.ops.triton._gluon_kernels.gfx950.attention.mha import (
    _get_config as _get_gluon_config,
)
from aiter.ops.triton._triton_kernels.attention.mha import _attn_fwd, _get_config
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2
from aiter.ops.triton.attention.mha_fused_bwd import flash_attn_fused_backward
from aiter.ops.triton.attention.mha_onekernel_bwd import flash_attn_onekernel_backward
from aiter.ops.triton.utils import types
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# ---------------------------------------------------------------------------
# Gluon backend capability helpers
#
# The Gluon backend is a forward-only flash-attention kernel and does NOT
# implement the full feature set of the default Triton MHA backend. These
# helpers are the single source of truth for "can the Gluon backend serve this
# call?" and are shared by the wrappers below (to reject unsupported calls) and
# the unit tests (to skip unsupported parametrizations).
# ---------------------------------------------------------------------------
_GLUON_SUPPORTED_ARCHS = ("gfx950",)
_TRITON_GE_36 = Version(triton.__version__) >= Version("3.6.0")

_USE_FUSED_BWD_KERNEL = False


def is_gluon_available() -> bool:
    """True when the Gluon MHA forward kernel can actually run on this device."""
    if not _TRITON_GE_36:
        return False
    arch = get_arch() or ""
    return any(supported in arch for supported in _GLUON_SUPPORTED_ARCHS)


_MHA_BACKENDS = ("triton", "gluon")


def _resolve_backend(
    backend: Literal["triton", "gluon"] | None,
) -> Literal["triton", "gluon"]:
    """None selects the default Triton backend; anything else must be known."""
    if backend is None:
        return "triton"
    if backend not in _MHA_BACKENDS:
        raise ValueError(
            f"Unknown MHA backend {backend!r}, expected one of {_MHA_BACKENDS}"
        )
    return backend


def mha_set_use_fused_bwd_kernel(value: bool):
    """
    Set whether to use fused backward kernel (with atomics) or one-kernel backward (without atomics).
    Fused backward is faster but doesn't support positional encoding.
    """
    global _USE_FUSED_BWD_KERNEL
    _USE_FUSED_BWD_KERNEL = value


_MHA_IMPL: Literal["default", "dao_ai"] = "default"


def mha_set_impl(impl: Literal["default", "dao_ai"]):
    """Set MHA forward implementation: 'default' (_attn_fwd) or 'dao_ai' (flash_attn_triton_amd)."""
    global _MHA_IMPL
    _MHA_IMPL = impl


_USE_INT64_STRIDES = True


def mha_set_use_int64_strides(value: bool):
    """Use 64-bit integer strides to prevent integer overflows with very large tensors."""
    global _USE_INT64_STRIDES
    _USE_INT64_STRIDES = value


_MHA_SWIZZLE_VALUES = ("default", "spatial")

_env_swizzle = os.environ.get("AITER_TRITON_MHA_SWIZZLE", "default")
if _env_swizzle not in _MHA_SWIZZLE_VALUES:
    raise ValueError(
        f"Invalid AITER_TRITON_MHA_SWIZZLE value: {_env_swizzle!r}. "
        f"Must be one of {_MHA_SWIZZLE_VALUES}."
    )
_MHA_SWIZZLE: Literal["default", "spatial"] = _env_swizzle
del _env_swizzle


def mha_set_swizzle(value: Literal["default", "spatial"]):
    """Set MHA workgroup swizzle mode.

    Args:
        value: ``"default"`` preserves the existing remap_xcd behaviour.
               ``"spatial"`` enables XCD-aware KV-head mapping for MHA and GQA.
    """
    if value not in _MHA_SWIZZLE_VALUES:
        raise ValueError(
            f"Invalid swizzle value: {value!r}. Must be one of {_MHA_SWIZZLE_VALUES}."
        )
    global _MHA_SWIZZLE
    _MHA_SWIZZLE = value


def _get_sliding_window_size(window_size: tuple[int, int]) -> int:
    return max(int(window_size[0]), 0)


def gluon_forward_unsupported_reason(
    *,
    dropout_p: float = 0.0,
    bias=None,
    alibi_slopes=None,
    return_lse: bool = False,
    return_attn_probs: bool = False,
    block_table=None,
):
    """Reason (str) why the Gluon forward backend can't serve this config, else None.

    Pass the feature flags for the support checks.
    """
    if not is_gluon_available():
        return (
            f"Gluon MHA backend requires one of {_GLUON_SUPPORTED_ARCHS} with "
            f"Triton>=3.6 (arch={get_arch()!r}, triton={triton.__version__})"
        )
    if dropout_p and dropout_p != 0.0:
        return "Gluon MHA backend does not support dropout"
    if bias is not None:
        return "Gluon MHA backend does not support attention bias"
    if alibi_slopes is not None:
        return "Gluon MHA backend does not support alibi slopes"
    if return_lse:
        return "Gluon MHA backend does not support returning LSE"
    if return_attn_probs:
        return "Gluon MHA backend does not support returning attention probabilities"
    if block_table is not None:
        return "Gluon MHA backend does not support paged KV (block_table)"
    return None


def _gluon_flash_attn_forward(
    q,
    k,
    v,
    causal=False,
    sm_scale=None,
    o=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    descale_q=None,
    descale_k=None,
    descale_v=None,
    sink=None,
    window_size=(-1, -1),
    config: dict[str, any] | None = None,
):
    """Validate + launch the Gluon forward kernel for both fixed-length (bshd)
    and varlen (thd) batches.

    Arguments:
        q, k, v: query / key / value tensors (layout per the mode above).
        causal: whether to apply a (bottom-right aligned) causal mask.
        sm_scale: QK^T scale. Defaults to 1 / sqrt(head_dim_qk).
        o: optional preallocated output, shaped like q but with V's head dim.
            Defaults to a fresh tensor (fp32 for fp8).
        cu_seqlens_q/cu_seqlens_k: (batch + 1,) int32 cumulative lengths (varlen).
        max_seqlen_q/max_seqlen_k: max sequence lengths in the batch (varlen).
        descale_q: (batch, num_q_heads) fp32 dequant scalars for q (fp8 only).
        descale_k/descale_v: (batch, num_k_heads) fp32 dequant scalars for k/v (fp8 only).
        sink: (num_q_heads,) attention sink logits, or None. Each one acts as an
            extra softmax column with no value vector.
        window_size: (left, right) local attention window. Only a left window is
            supported, so right must be -1.
    Return:
        o: same layout as q. Dtype is fp32 for fp8 inputs unless the caller passed
            an ``o`` of a different dtype, in which case that dtype is returned.
    """
    varlen = cu_seqlens_q is not None

    assert is_gluon_available(), (
        f"Gluon MHA backend requires one of {_GLUON_SUPPORTED_ARCHS} with "
        f"Triton>=3.6 (arch={get_arch()!r}, triton={triton.__version__})"
    )
    assert q.shape[-1] == k.shape[-1], "q/k head_dim mismatch"

    if int(window_size[1]) != -1:
        raise ValueError("window_size_right is not supported yet in the Gluon Backend")
    sliding_window = _get_sliding_window_size(window_size)

    IS_FP8 = types._is_fp8(q)
    FP8_MAX = torch.finfo(q.dtype).max if IS_FP8 else 0.0

    qk_head_dim = q.shape[-1]
    v_head_dim = v.shape[-1]
    pe_head_dim = qk_head_dim - v_head_dim
    if IS_FP8:
        assert (
            descale_q is not None and descale_k is not None and descale_v is not None
        ), "FP8 Gluon MHA requires descale_q/descale_k/descale_v"
        assert (
            q.dtype == types.e4m3_dtype
            and k.dtype == types.e4m3_dtype
            and v.dtype == types.e4m3_dtype
        ), (
            f"FP8 Gluon MHA only supports the e4m3 fp8 dtype ({types.e4m3_dtype}) "
            f"(got q={q.dtype}, k={k.dtype}, v={v.dtype})"
        )

    if config is None:
        config = _get_gluon_config(is_fp8=IS_FP8, has_pe=pe_head_dim > 0)
    config = dict(config)
    BLOCK_M = config.pop("BLOCK_M")
    BLOCK_N = config.pop("BLOCK_N")
    assert BLOCK_N % 32 == 0, "BLOCK_N must be a multiple of 32"
    # The scaled f8f6f4 MFMA is 32x32x64 (K=64), so the P@V contraction
    # (BLOCK_N) must be a multiple of 64.
    assert (
        not IS_FP8 or BLOCK_N % 64 == 0
    ), "FP8 Gluon MHA requires BLOCK_N to be a multiple of 64"

    if varlen:
        _, num_q_heads, _ = q.shape
        _, num_k_heads, _ = k.shape
        batch = cu_seqlens_q.numel() - 1
        seqlen_q = int(max_seqlen_q)
        seqlen_k = int(max_seqlen_k)
    else:
        batch, seqlen_q, num_q_heads, _ = q.shape
        _, seqlen_k, num_k_heads, _ = k.shape

    assert (
        num_q_heads % num_k_heads == 0
    ), "num_q_heads must be divisible by num_k_heads"
    assert (sink is None) or (
        sink.dim() == 1 and sink.shape[0] == num_q_heads
    ), "Sink must be 1D and have one element per query head."

    if sm_scale is None:
        sm_scale = qk_head_dim ** (-0.5)

    # Pad to a vectorizable head dim. Unreachable with positional encoding,
    # which requires unpadded power-of-2 head sizes.
    head_size_og = v_head_dim
    if head_size_og % 8 != 0:
        pad = 8 - head_size_og % 8
        q = torch.nn.functional.pad(q, [0, pad])
        k = torch.nn.functional.pad(k, [0, pad])
        v = torch.nn.functional.pad(v, [0, pad])
        v_head_dim = v.shape[-1]

    o_provided = o
    if IS_FP8 or o is None or v_head_dim != head_size_og:
        o_dtype = torch.float32 if IS_FP8 else q.dtype
        if pe_head_dim > 0:
            # Q/K carry the PE slice, the output only spans V's head dim.
            o = torch.empty(
                q.shape[:-1] + (v_head_dim,), dtype=o_dtype, device=q.device
            )
        else:
            o = torch.empty_like(q, dtype=o_dtype)

    if varlen:
        # (total_tokens, head, head_dim)
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    else:
        # (batch, head, seq, head_dim)
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

    min_pad = 64 if IS_FP8 else 16
    BLOCK_DMODEL_POW2 = max(triton.next_power_of_2(v_head_dim), min_pad)
    BLOCK_DMODEL_PE_POW2 = (
        0 if pe_head_dim == 0 else max(triton.next_power_of_2(pe_head_dim), 16)
    )
    assert (pe_head_dim == 0 and BLOCK_DMODEL_PE_POW2 == 0) or (
        v_head_dim == BLOCK_DMODEL_POW2 and pe_head_dim == BLOCK_DMODEL_PE_POW2
    ), "Positional encoding support requires NOPE and PE head sizes to be unpadded powers of 2."
    assert (not IS_FP8) or (
        IS_FP8 and pe_head_dim == 0
    ), "Positional encoding doesn't support FP8."
    # FP8 uses 32x32x64 (K=64) scaled MFMA, so the PE head size must be a multiple of 64.
    # TODO: Enable this assert if the assert above is ever lifted.
    # assert (not IS_FP8) or (
    #     pe_head_dim % 64 == 0
    # ), "FP8 positional encoding requires the PE head size to be a multiple of 64."
    head_stride_aligned_8 = (
        q_strides[1] % 8 == 0 and k_strides[1] % 8 == 0 and v_strides[1] % 8 == 0
    )

    grid = (batch * num_q_heads * triton.cdiv(seqlen_q, BLOCK_M), 1)

    _gluon_attn_fwd[grid](
        q,
        k,
        v,
        o,
        descale_q,
        descale_k,
        descale_v,
        sink,
        sm_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlen_q,
        seqlen_k,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        descale_q.stride(0) if descale_q is not None else 0,
        descale_k.stride(0) if descale_k is not None else 0,
        descale_v.stride(0) if descale_v is not None else 0,
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        IS_CAUSAL=causal,
        VARLEN=varlen,
        BATCH=batch,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=v_head_dim,
        BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
        BLOCK_DMODEL_PE=pe_head_dim,
        NUM_XCD=get_num_xcds(),
        USE_INT64_STRIDES=_USE_INT64_STRIDES,
        IS_FP8=IS_FP8,
        FP8_MAX=FP8_MAX,
        ENABLE_SINK=sink is not None,
        SLIDING_WINDOW=sliding_window,
        HEAD_STRIDE_ALIGNED_8=head_stride_aligned_8,
        **config,
    )

    if v_head_dim != head_size_og:
        o = o[..., :head_size_og]
    if o_provided is not None and o_provided.data_ptr() != o.data_ptr():
        o_provided.copy_(o)
        o = o_provided
    return o


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    return_lse: bool,  # Not used
    return_softmax: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
    descale_v: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    config: dict[str, any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, int]:

    if bias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")
    if _MHA_IMPL != "dao_ai" and window_size_right != -1:
        raise ValueError("window_size_right is not supported yet in the Triton Backend")
    sliding_window = max(window_size_left, 0)

    # Triton cannot specialize on numpy scalar types; ensure native Python int
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_k = int(max_seqlen_k)

    # FP8
    IS_FP8 = types._is_fp8(q)
    FP8_MAX: tl.constexpr = torch.finfo(q.dtype).max
    is_varlen = cu_seqlens_q is not None

    if IS_FP8:
        o = torch.zeros(
            (q.shape[:-1] + v.shape[-1:]), dtype=torch.float32, device=q.device
        )
    else:
        o = torch.zeros((q.shape[:-1] + v.shape[-1:]), dtype=q.dtype, device=q.device)
    if is_varlen:
        # Layout is thd.
        # q and k are [total_tokens, num_head, head_dim_qk].
        # v is [total_tokens, num_head, head_dim_v].
        batch, seqlen_q, num_q_heads = (
            len(cu_seqlens_q) - 1,
            max_seqlen_q,
            q.shape[1],
        )
        num_k_heads = k.shape[1]
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    else:
        # Layout is bshd.
        # q and k are [batch, seq_len, num_head, head_dim_qk].
        # v is [batch, seq_len, num_head, head_dim_v].
        batch, seqlen_q, num_q_heads = (int(x) for x in q.shape[:-1])
        num_k_heads = k.shape[2]
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

    qk_head_dim = q.shape[-1]
    v_head_dim = v.shape[-1]
    pe_head_dim = qk_head_dim - v_head_dim
    # padding for head_dim. Power of 2 or 16
    BLOCK_DMODEL_POW2 = max(triton.next_power_of_2(v_head_dim), 16)
    BLOCK_DMODEL_PE_POW2 = (
        0 if pe_head_dim == 0 else max(triton.next_power_of_2(pe_head_dim), 16)
    )
    assert (pe_head_dim == 0 and BLOCK_DMODEL_PE_POW2 == 0) or (
        v_head_dim == BLOCK_DMODEL_POW2 and pe_head_dim == BLOCK_DMODEL_PE_POW2
    ), "Positional encoding support requires NOPE and PE head sizes to be unpadded powers of 2."
    assert (not IS_FP8) or (
        IS_FP8 and pe_head_dim == 0
    ), "Positional encoding doesn't support FP8."

    assert (sink is None) or (
        sink is not None and sink.dim() == 1 and sink.shape[0] == num_q_heads
    ), "Sink must be 1D and have one element per query head."

    # softmax_lse [batch, num_q_heads, seqlen_q]
    if is_varlen:
        softmax_lse = torch.zeros(
            (q.shape[0], num_q_heads), device=q.device, dtype=torch.float32
        )
        stride_lse_z, stride_lse_h, stride_lse_m = (
            0,
            softmax_lse.stride(1),
            softmax_lse.stride(0),
        )
    else:
        softmax_lse = torch.zeros(
            (batch, num_q_heads, max_seqlen_q), device=q.device, dtype=torch.float32
        )
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    # exp_scores [batch, num_q_heads, seqlen_q, seqlen_k]
    enable_dropout = dropout_p > 0.0
    if enable_dropout:
        philox_seed = torch.randint(0, 0xFFFFFF, (1,))[
            0
        ].item()  # No specific reason to restrict range to 0xffffff
        philox_offset = torch.randint(0, 0xFFFFFF, (1,))[
            0
        ].item()  # Pass in an int, not Tensor
    else:
        philox_seed = 0
        philox_offset = 0
    if return_softmax or enable_dropout:
        s_dmask = torch.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
        dropout_mask = torch.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        s_dmask = None
        dropout_mask = None

    if _MHA_IMPL == "dao_ai":
        assert sink is None, "dao_ai impl does not support attention sink."
        assert (
            pe_head_dim == 0
        ), "dao_ai impl does not support positional encoding (pe_head_dim > 0)."
        assert (
            not IS_FP8
        ), "dao_ai impl does not support FP8. Use the default impl or FA3 path."
        if is_varlen:
            o, softmax_lse, s_dmask, _ = flash_attn_2.varlen_fwd(
                q,
                k,
                v,
                o,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_k=None,
                leftpad_k=None,
                block_table_=None,
                alibi_slopes=alibi_slopes,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                zero_tensors=False,
                causal=causal,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                softcap=0.0,
                return_softmax=return_softmax,
            )
        else:
            o, softmax_lse, s_dmask, _ = flash_attn_2.fwd(
                q,
                k,
                v,
                o,
                alibi_slopes,
                dropout_p,
                softmax_scale,
                causal,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                softcap=0.0,
                return_softmax=return_softmax,
            )
        # Verify softmax_lse shape contract:
        #   non-varlen: (batch, nheads_q, seqlen_q)
        #   varlen:     (nheads_q, total_q)  -- transposed vs default impl
        if is_varlen:
            assert softmax_lse.shape == (
                num_q_heads,
                q.shape[0],
            ), f"dao_ai varlen softmax_lse shape {softmax_lse.shape} != expected ({num_q_heads}, {q.shape[0]})"
        else:
            assert (
                softmax_lse.shape[0] == batch and softmax_lse.shape[1] == num_q_heads
            ), f"dao_ai softmax_lse shape {softmax_lse.shape} != expected (batch={batch}, nheads={num_q_heads}, ...)"
    else:
        if config is None:
            config = _get_config(
                enable_dropout, q.dtype, has_pe=pe_head_dim > 0, head_dim_v=v_head_dim
            )

        grid = lambda META: (
            batch * num_q_heads * triton.cdiv(seqlen_q, META["BLOCK_M"]),
        )

        _attn_fwd[grid](
            q,
            k,
            v,
            descale_q,
            descale_k,
            descale_v,
            o,
            alibi_slopes,
            s_dmask,
            dropout_mask,
            softmax_lse,
            sink,
            *q_strides,
            *k_strides,
            *v_strides,
            descale_q.stride(0) if descale_q is not None else 0,
            descale_k.stride(0) if descale_k is not None else 0,
            descale_v.stride(0) if descale_v is not None else 0,
            *o_strides,
            alibi_slopes.stride(0) if alibi_slopes is not None else 0,
            alibi_slopes.stride(1) if alibi_slopes is not None else 0,
            s_dmask.stride(0) if s_dmask is not None else 0,
            s_dmask.stride(1) if s_dmask is not None else 0,
            s_dmask.stride(2) if s_dmask is not None else 0,
            s_dmask.stride(3) if s_dmask is not None else 0,
            stride_lse_z if softmax_lse is not None else 0,
            stride_lse_h if softmax_lse is not None else 0,
            stride_lse_m if softmax_lse is not None else 0,
            softmax_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            dropout_p,
            philox_seed,
            philox_offset,
            SEQLEN_Q=max_seqlen_q,
            SEQLEN_K=max_seqlen_k,
            IS_CAUSAL=causal,
            NUM_Q_HEADS=num_q_heads,
            NUM_K_HEADS=num_k_heads,
            BLOCK_DMODEL=v_head_dim,
            BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE=pe_head_dim,
            RETURN_SCORES=return_softmax,
            ENABLE_DROPOUT=enable_dropout,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            VARLEN=is_varlen,
            BATCH=batch,
            NUM_XCD=get_num_xcds(),
            SWIZZLE=_MHA_SWIZZLE,
            USE_INT64_STRIDES=_USE_INT64_STRIDES,
            ENABLE_SINK=sink is not None,
            SLIDING_WINDOW=sliding_window,
            # Soundness precondition: only set when every Q/K/V head-axis
            # stride is a multiple of 8 elements. q_strides[1]/k_strides[1]/
            # v_strides[1] are the head-axis strides in both thd and bshd
            # layouts (see q_strides assembly above).
            HEAD_STRIDE_ALIGNED_8=(
                q_strides[1] % 8 == 0
                and k_strides[1] % 8 == 0
                and v_strides[1] % 8 == 0
            ),
            **config,
        )

    return o, softmax_lse, s_dmask, philox_seed, philox_offset


class _FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        sink,
        is_grad_enabled,
        config=None,
    ):
        is_grad = is_grad_enabled and any(
            x is not None and x.requires_grad for x in [q, k, v, sink]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = (
            _flash_attn_forward(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size_left=int(window_size[0]),
                window_size_right=int(window_size[1]),
                bias=bias,
                alibi_slopes=alibi_slopes,
                return_lse=return_lse,
                return_softmax=return_softmax and dropout_p > 0,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                sink=sink,
                config=config,
            )
        )

        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, sink)
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.bias = bias
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic

        out = out_padded[..., :head_size_og]
        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        q, k, v, out, softmax_lse, sink = ctx.saved_tensors
        bias = ctx.bias
        dbias = torch.empty_like(bias) if bias is not None else None
        dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
        dsink = (
            torch.zeros_like(sink, dtype=torch.float32) if sink is not None else None
        )
        head_size_v_og = do.size(3)
        do_padded = do
        if head_size_v_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_v_og % 8])
        sliding_window = _get_sliding_window_size(ctx.window_size)

        if _MHA_IMPL == "dao_ai":
            assert sink is None, "dao_ai impl does not support attention sink."
            flash_attn_2.bwd(
                do_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                ctx.alibi_slopes,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                window_size_left=ctx.window_size[0],
                window_size_right=ctx.window_size[1],
                softcap=0.0,
                deterministic=ctx.deterministic,
            )
        else:
            if _USE_FUSED_BWD_KERNEL:
                if sliding_window > 0:
                    raise ValueError(
                        "Fused backward doesn't support sliding window attention. "
                        "Disable fused backward or use the one-kernel backward."
                    )
                assert (
                    sink is None and dsink is None
                ), "Fused backward doesn't support sinks."
                flash_attn_fused_backward(
                    do_padded,
                    q,
                    k,
                    v,
                    out,
                    softmax_lse,
                    dq,
                    dk,
                    dv,
                    dbias,
                    ctx.softmax_scale,
                    ctx.alibi_slopes,
                    ctx.causal,
                    None,
                    None,
                    max_seqlen_q=q.shape[1],
                    max_seqlen_k=k.shape[1],
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset=ctx.philox_offset,
                    USE_INT64_STRIDES=_USE_INT64_STRIDES,
                )
            else:
                flash_attn_onekernel_backward(
                    do_padded,
                    q,
                    k,
                    v,
                    out,
                    softmax_lse,
                    dq,
                    dk,
                    dv,
                    dbias,
                    ctx.softmax_scale,
                    ctx.alibi_slopes,
                    ctx.causal,
                    None,
                    None,
                    max_seqlen_q=q.shape[1],
                    max_seqlen_k=k.shape[1],
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset=ctx.philox_offset,
                    USE_INT64_STRIDES=_USE_INT64_STRIDES,
                    sink=sink,
                    dsink=dsink,
                    sliding_window=sliding_window,
                )

        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return (
            dq,
            dk,
            dv,
            None,  # dropout_p
            None,  # softmax_scale
            None,  # causal
            None,  # window_size
            dbias,
            None,  # alibi_slopes
            None,  # deterministic
            None,  # return_lse
            None,  # return_softmax
            dsink,
            None,  # is_grad_enabled
            None,  # config
        )


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    bias=None,
    alibi_slopes=None,
    deterministic=True,
    return_lse=False,
    return_attn_probs=False,
    sink=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    config: dict[str, any] | None = None,
    backend: Literal["triton", "gluon"] | None = "triton",
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim_q).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        bias: (seqlen_q, seqlen_k)
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        sink: (nheads,), attention sink scores (one per Q head), or None
        q_descale, k_descale, v_descale: optional fp8 dequant scalars, honored only
            by the "gluon" backend when q/k/v are fp8. Shapes (batch, num_q_heads)
            for q and (batch, num_k_heads) for k/v; the output is fp32.
        backend: "triton" (default) or "gluon". The "gluon" backend runs the
            forward-only gfx950 Gluon kernel and supports the base feature set
            plus FP8, positional encoding, attention sink and a left sliding
            window (no dropout/bias/alibi, no right window, no LSE/softmax return
            and no backward pass). For FP8, pass pre-quantized fp8 q/k/v with
            q_descale/k_descale/v_descale.
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    backend = _resolve_backend(backend)
    _LOGGER.info(
        f"FLASH_ATTN [{backend}]:  q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)}"
    )

    if backend == "gluon":
        reason = gluon_forward_unsupported_reason(
            dropout_p=dropout_p,
            bias=bias,
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
        )
        assert reason is None, reason
        return _gluon_flash_attn_forward(
            q,
            k,
            v,
            causal=causal,
            sm_scale=softmax_scale,
            descale_q=q_descale,
            descale_k=k_descale,
            descale_v=v_descale,
            sink=sink,
            window_size=window_size,
            config=config,
        )

    return _FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        sink,
        torch.is_grad_enabled(),
        config,
    )


class _FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        block_table,
        out,
        sink,
        is_grad_enabled,
        config=None,
    ):
        is_grad = is_grad_enabled and any(
            x is not None and x.requires_grad for x in [q, k, v, sink]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = (
            _flash_attn_forward(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size_left=int(window_size[0]),
                window_size_right=int(window_size[1]),
                bias=bias,
                alibi_slopes=alibi_slopes,
                return_lse=return_lse,
                return_softmax=return_softmax and dropout_p > 0.0,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                sink=sink,
                config=config,
            )
        )
        if is_grad:
            ctx.save_for_backward(
                q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, sink
            )
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.bias = bias
            ctx.alibi_slopes = alibi_slopes
        out = out_padded[..., :head_size_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, sink = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
        bias = ctx.bias
        dbias = torch.empty_like(bias) if bias is not None else None
        dsink = (
            torch.zeros_like(sink, dtype=torch.float32) if sink is not None else None
        )
        head_size_og = do.size(2)
        do_padded = do
        if head_size_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_og % 8])
        sliding_window = _get_sliding_window_size(ctx.window_size)

        if _MHA_IMPL == "dao_ai":
            assert sink is None, "dao_ai impl does not support attention sink."
            flash_attn_2.varlen_bwd(
                do_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                cu_seqlens_q,
                cu_seqlens_k,
                ctx.alibi_slopes,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                dropout_p=ctx.dropout_p,
                softmax_scale=ctx.softmax_scale,
                zero_tensors=False,
                causal=ctx.causal,
                window_size_left=ctx.window_size[0],
                window_size_right=ctx.window_size[1],
                softcap=0.0,
                deterministic=False,
            )
        else:
            if _USE_FUSED_BWD_KERNEL:
                if sliding_window > 0:
                    raise ValueError(
                        "Fused backward doesn't support sliding window attention. "
                        "Disable fused backward or use the one-kernel backward."
                    )
                assert (
                    sink is None and dsink is None
                ), "Fused backward doesn't support sinks."
                flash_attn_fused_backward(
                    do_padded,
                    q,
                    k,
                    v,
                    out,
                    softmax_lse,
                    dq,
                    dk,
                    dv,
                    dbias,
                    ctx.softmax_scale,
                    ctx.alibi_slopes,
                    ctx.causal,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q=ctx.max_seqlen_q,
                    max_seqlen_k=ctx.max_seqlen_k,
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset=ctx.philox_offset,
                    USE_INT64_STRIDES=_USE_INT64_STRIDES,
                )
            else:
                flash_attn_onekernel_backward(
                    do_padded,
                    q,
                    k,
                    v,
                    out,
                    softmax_lse,
                    dq,
                    dk,
                    dv,
                    dbias,
                    ctx.softmax_scale,
                    ctx.alibi_slopes,
                    ctx.causal,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q=ctx.max_seqlen_q,
                    max_seqlen_k=ctx.max_seqlen_k,
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset=ctx.philox_offset,
                    USE_INT64_STRIDES=_USE_INT64_STRIDES,
                    sink=sink,
                    dsink=dsink,
                    sliding_window=sliding_window,
                )

        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return (
            dq,
            dk,
            dv,
            None,  # cu_seqlens_q,
            None,  # cu_seqlens_k
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            None,  # dropout_p
            None,  # softmax_scale
            None,  # causal
            None,  # window_size
            dbias,
            None,  # alibi_slopes
            None,  # deterministic
            None,  # return_lse
            None,  # return_softmax
            None,  # block_table
            None,  # out
            dsink,
            None,  # is_grad_enabled
            None,  # config
        )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    block_table=None,
    out=None,
    sink=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    config: dict[str, any] | None = None,
    backend: Literal["triton", "gluon"] | None = "triton",
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        bias: (seqlen_q, seqlen_k)
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        sink: (nheads,), attention sink scores (one per Q head), or None
        q_descale, k_descale, v_descale: optional fp8 dequant scalars, honored only
            by the "gluon" backend when q/k/v are fp8. Shapes (batch, num_q_heads)
            for q and (batch, num_k_heads) for k/v; the output is fp32.
        backend: "triton" (default) or "gluon". The "gluon" backend runs the
            forward-only gfx950 Gluon kernel and supports the base feature set
            plus FP8, positional encoding, attention sink and a left sliding
            window (no dropout/bias/alibi, no right window, no LSE/softmax return
            and no backward pass). For FP8, pass pre-quantized fp8 q/k/v with
            q_descale/k_descale/v_descale.
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    backend = _resolve_backend(backend)
    _LOGGER.info(
        f"FLASH_ATTN_VARLEN [{backend}]:  q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)}"
    )

    if backend == "gluon":
        reason = gluon_forward_unsupported_reason(
            dropout_p=dropout_p,
            bias=bias,
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
            block_table=block_table,
        )
        assert reason is None, reason
        return _gluon_flash_attn_forward(
            q,
            k,
            v,
            causal=causal,
            sm_scale=softmax_scale,
            o=out,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            descale_q=q_descale,
            descale_k=k_descale,
            descale_v=v_descale,
            sink=sink,
            window_size=window_size,
            config=config,
        )

    return _FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        block_table,
        out,
        sink,
        torch.is_grad_enabled(),
        config,
    )


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | int | None = None,
    softmax_scale: float | None = None,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    num_splits: int = 0,
    rotary_cos: torch.Tensor | None = None,
    rotary_sin: torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    rotary_interleaved: bool = True,
    return_softmax_lse: bool = False,
):
    """
    This mirrors the public flash_attn v2 interface for KV cache using the AMD Triton backend.

    Args:
        q: (batch, seqlen_q, nheads_q, headdim)
        k_cache / v_cache: Either contiguous (batch, seqlen_cache, nheads_k, headdim) or paged
            (num_blocks, page_block_size, nheads_k, headdim) when block_table provided.
        k, v: Optional incremental tokens to append in-place (appended logically after existing cache).
        cache_seqlens: int or (batch,) current valid lengths per batch entry.
        softmax_scale: Optional override; defaults to 1/sqrt(headdim).
        causal: Apply causal masking.
        window_size: (left, right) local attention window; (-1,-1) = full.
        softcap: (float) currently must be 0.0 (backend limitation).
        num_splits: 0 or 1 only (backend limitation >1).
        rotary_cos/rotary_sin: Optional rotary embeddings (applied if provided) - interleaving flag unused here.
        cache_batch_idx/cache_leftpad: Optional indexing / left padding metadata.
            block_table: Optional paging table mapping logical blocks for paged KV cache.
        alibi_slopes: (nheads,) or (batch,nheads) bias slopes (currently ignored if provided - placeholder).
        rotary_interleaved: Flag kept for parity (currently forwarded as True constant to backend which ignores it).
            return_softmax_lse: If True returns (out, lse) else out.

    Returns:
        out (and optionally softmax_lse): (batch, seqlen_q, nheads_q, headdim)
    """
    # Feature guards / normalization
    if softcap != 0.0:
        raise NotImplementedError(
            "softcap != 0 not supported in v2 KV cache backend yet"
        )
    if num_splits not in (0, 1):
        raise NotImplementedError(
            "num_splits > 1 not supported in v2 KV cache backend yet"
        )

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )

    # Contiguity (align last dim contiguous requirement similar to v3 path assumptions)
    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1

    out, softmax_lse = flash_attn_2.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        None,  # out tensor
        softmax_scale,
        causal,
        int(window_size[0]),
        int(window_size[1]),
        0.0,  # softcap (guarded)
        rotary_interleaved,
        num_splits,
    )
    return (out, softmax_lse) if return_softmax_lse else out
