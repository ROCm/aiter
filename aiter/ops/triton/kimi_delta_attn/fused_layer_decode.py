# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Kimi-K3 KDA *decode* layer (whole rank-local linear-attention block).

A single Gluon launch replaces the entire Kimi-K3 Delta-Attention decode step:
the QKVG projection, the beta/forget-A projection, forget-B, the beta sigmoid,
the causal convolution, the single-token gated-delta recurrence, the gated
RMSNorm, and the output (O) projection -- all fused. The op reads the raw
hidden state ``(m, 7168)`` and writes the O-projection *partial* ``(m, 7168)``;
the conv/recurrent state pools are updated in place and returned unchanged
(same storage).

Because the O matmul is fused but tensor-parallel, the returned ``(m, 7168)`` is
the rank-local partial -- the caller is responsible for the O-projection
all-reduce across the attention TP group, exactly as the native
``RowParallelLinear`` o_proj would do. This op deliberately stops at the kernel
boundary and does not reduce.

Dispatch routes by batch range to two general schedules: ``kda_layer_decode_m64``
for ``m <= 64`` and ``kda_layer_decode_m256`` for ``64 < m <= 256``. An on-device
tile-vs-tile bake-off across every real decode tier found these two dominate --
each is correct over its range and within a few microseconds of the per-tier
best -- while the remaining per-size Kimi-K3 kernel-pack variants are never
fastest and the general serial catch-all is 2-8x slower at large ``m``. Decode
batches are exact (single tokens, no ragged lengths), so the range split is
unambiguous.

Support is deliberately narrow and reported by
:func:`kimi_kda_fused_layer_decode_supported` rather than asserted, so callers
fall back to the native KDA decode path instead of crashing. Two hard gates:

* gfx950 only -- the tiles use CDNA4 buffer addressing and explicit MFMA
  register layouts.
* Triton >= 3.8 -- the tiles are written in the Gluon dialect. On Triton 3.7 (or
  any build without ``triton.experimental.gluon``) the gate returns ``False`` so
  the caller falls back; importing this module never raises, and the Gluon/HIP
  imports only fire once a tile is actually dispatched on a supported device.
"""

import functools
import re

import torch

# NB: the Gluon tile kernels are imported lazily inside the host launchers
# (appended at the end of this module), not at module load, so this module -- and
# in particular the :func:`kimi_kda_fused_layer_decode_supported` gate a caller
# probes -- stays importable and callable on any platform (older Triton,
# non-gfx950, CUDA).

_HIDDEN = 7168
_CORE = 1536
_HEAD_DIM = 128
_NUM_HEADS = 12
_MAX_M = 256
# The m256 schedule tiles the recurrence grid in groups of this many rows
# (``m // _M256_GROUP``), so it only covers an ``m`` that is a multiple of it;
# a non-multiple would silently skip its tail rows. Every real cuda-graph decode
# tier above 64 is a multiple of 8, but gate on it so an odd ``m`` falls back to
# native instead. (m64 tiles per-row and has no such requirement.)
_M256_GROUP = 8

# (m) -> tile key. An on-device tile-vs-tile bake-off across every real decode
# batch tier (see the bench) showed the two general schedules dominate: m64 is
# correct and near-optimal for the whole low range (m<=64, ~54us), and m256 is
# correct and optimal for the high range (m>64, 64..109us). The per-batch-size
# Artemis variants (m1/m2/m4/m32/m128) are never fastest at a real tier, and the
# m1_256 catch-all is a serial m=1-style kernel that is 2-8x slower at large m,
# so a dispatch keyed on exact m silently routes most tiers to it. Routing by
# range to {m64, m256} is both faster and simpler. Decode batches are exact
# (single tokens, no ragged lengths), so this range split is unambiguous.


@functools.lru_cache(maxsize=1)
def _arch_supported() -> tuple[bool, str]:
    """Probe the GPU arch lazily and fail closed if it cannot be read.

    ``get_arch`` is imported here, not at module load, so this module stays
    importable on a CPU-only / non-ROCm host; a probe failure (e.g. no
    ``rocminfo``) returns ``(False, reason)`` rather than raising.
    """
    try:
        from aiter.ops.triton.utils._triton.arch_info import get_arch

        arch = get_arch()
    except Exception as exc:  # noqa: BLE001 - any probe failure => unsupported
        return False, f"could not probe GPU arch ({exc})"
    if arch != "gfx950":
        return False, f"gfx950 only, got {arch}"
    return True, ""


@functools.lru_cache(maxsize=1)
def _gluon_supported() -> tuple[bool, str]:
    """Cached probe: can this Triton compile the tiles' Gluon dialect?

    gfx950 alone is not enough. The tiles use the Gluon dialect as it stands in
    Triton **3.8**; ROCm backported an *earlier, incompatible* Gluon into some
    3.7 builds where ``triton.experimental.gluon`` imports fine but the tiles
    fail to compile. So the gate is an explicit ``>= 3.8`` version check, not
    just an import probe -- a 3.7 build returns ``(False, reason)`` and the
    caller falls back cleanly.
    """
    try:
        import triton
    except ImportError as exc:
        return False, f"triton not importable ({exc})"
    version = triton.__version__
    matched = re.match(r"(\d+)\.(\d+)", version or "")
    if matched is None or (int(matched.group(1)), int(matched.group(2))) < (3, 8):
        return False, f"Triton >= 3.8 required for this Gluon dialect, got {version}"
    try:
        import triton.experimental.gluon
    except ImportError:
        return False, f"triton.experimental.gluon unavailable (Triton {version})"
    return True, ""


def _select_tile_key(m: int) -> str | None:
    """Return the tile key for a covered decode batch ``m``, else ``None``.

    Pure -- no Gluon import -- so the coverage decision is shared cheaply by the
    support predicate and the dispatcher.
    """
    if not (1 <= m <= _MAX_M):
        return None
    if m <= 64:
        return "m64"
    if m % _M256_GROUP != 0:
        return None
    return "m256"


def _load_tile(key: str):
    """Return the host launcher for ``key``.

    The launchers live at module scope (appended below); each imports its Gluon
    tile kernels lazily on first call (the ``triton.experimental.gluon`` + ROCm
    ``libdevice`` imports happen there, on the dispatch path, never at module
    load).
    """
    return {
        "m64": kda_layer_decode_m64,
        "m256": kda_layer_decode_m256,
    }[key]


def _row_major_ok(t: torch.Tensor, shape: tuple[int, ...]) -> bool:
    """2-D, unit inner stride, outer stride >= row width (the tiles' contract)."""
    return (
        t.ndim == 2
        and tuple(t.shape) == tuple(shape)
        and t.stride(1) == 1
        and t.stride(0) >= shape[1]
    )


def kimi_kda_fused_layer_decode_supported(
    x: torch.Tensor,
    qkvg_weight: torch.Tensor,
    beta_forget_weight: torch.Tensor,
    output_weight: torch.Tensor,
    forget_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
) -> tuple[bool, str]:
    """Report whether this decode call is covered, and if not, why.

    Returns ``(True, "")`` or ``(False, reason)``; the reason is meant to be
    logged once by the caller on its fallback path. Mirrors the Artemis
    ``_validate_kda_inputs`` metadata contract 1:1.
    """
    arch_ok, arch_reason = _arch_supported()
    if not arch_ok:
        return False, arch_reason

    gluon_ok, gluon_reason = _gluon_supported()
    if not gluon_ok:
        return False, gluon_reason

    if not isinstance(x, torch.Tensor) or not x.is_cuda or x.ndim != 2:
        return False, "hidden state must be a 2-D CUDA matrix"
    m = x.shape[0]
    if _select_tile_key(m) is None:
        return False, f"decode batch m={m} outside covered range [1, {_MAX_M}]"
    device = x.device

    if not isinstance(state, torch.Tensor) or state.ndim != 4 or state.shape[0] <= 0:
        return False, "temporal cache must be rank-4 with positive capacity"
    slots = state.shape[0]

    row_major = (
        ("hidden", x, (m, _HIDDEN)),
        ("QKVG weight", qkvg_weight, (6144, _HIDDEN)),
        ("beta/forget-A weight", beta_forget_weight, (144, _HIDDEN)),
        ("O weight", output_weight, (_HIDDEN, _CORE)),
        ("forget-B weight", forget_weight, (_CORE, _HEAD_DIM)),
    )
    for name, t, shape in row_major:
        if not isinstance(t, torch.Tensor) or t.device != device:
            return False, f"{name} must be a CUDA tensor on {device}"
        if t.dtype != torch.bfloat16:
            return False, f"{name} must be bf16, got {t.dtype}"
        if not _row_major_ok(t, shape):
            return False, f"{name} must be row-major {shape}, got {tuple(t.shape)}"

    contiguous = (
        ("convolution weights", conv_weight, (4608, 4), torch.float32),
        ("A_log", a_log, (_NUM_HEADS,), torch.float32),
        ("dt_bias", dt_bias, (_CORE,), torch.float32),
        ("gated RMSNorm weight", norm_weight, (_HEAD_DIM,), torch.bfloat16),
        ("convolution cache", conv_state, (slots, 3, 4608), torch.bfloat16),
        (
            "temporal cache",
            state,
            (slots, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM),
            torch.float32,
        ),
    )
    for name, t, shape, dtype in contiguous:
        if not isinstance(t, torch.Tensor) or t.device != device:
            return False, f"{name} must be a CUDA tensor on {device}"
        if t.dtype != dtype or tuple(t.shape) != shape or not t.is_contiguous():
            return False, f"{name} must be contiguous {dtype} {shape}"

    if not isinstance(state_indices, torch.Tensor) or state_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        return False, "cache indices must be int32/int64"
    if state_indices.device != device:
        return False, f"cache indices must be on {device}"
    if tuple(state_indices.shape) != (m,) or not state_indices.is_contiguous():
        return False, f"cache indices must be contiguous [{m}]"

    return True, ""


def kimi_kda_fused_layer_decode(
    x: torch.Tensor,
    qkvg_weight: torch.Tensor,
    beta_forget_weight: torch.Tensor,
    output_weight: torch.Tensor,
    forget_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    lower_bound: float = -5.0,
    norm_eps: float = 1e-5,
    output_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Kimi-K3 KDA decode layer (projections + conv + gated-delta
    recurrence + gated RMSNorm + O matmul) in a single Gluon launch.

    ``conv_state`` and ``state`` are updated in place. Returns
    ``(o_partial, conv_state, state)`` where ``o_partial`` is the rank-local
    ``(m, 7168)`` O-projection output (written into ``output_tensor`` when
    given); the caller must all-reduce it across the attention TP group.

    Narrow by design; guard with
    :func:`kimi_kda_fused_layer_decode_supported` and fall back to the native
    KDA decode path when it returns ``False``.
    """
    ok, reason = kimi_kda_fused_layer_decode_supported(
        x,
        qkvg_weight,
        beta_forget_weight,
        output_weight,
        forget_weight,
        conv_weight,
        a_log,
        dt_bias,
        norm_weight,
        conv_state,
        state,
        state_indices,
    )
    if not ok:
        raise ValueError(
            f"kimi_kda_fused_layer_decode does not support this call: {reason}"
        )

    if output_tensor is not None:
        expected = (x.shape[0], _HIDDEN)
        if (
            not isinstance(output_tensor, torch.Tensor)
            or tuple(output_tensor.shape) != expected
            or output_tensor.dtype != torch.bfloat16
            or output_tensor.device != x.device
            or not output_tensor.is_contiguous()
        ):
            raise ValueError(
                f"output_tensor must be a contiguous bf16 {expected} tensor on "
                f"{x.device}"
            )

    tile = _load_tile(_select_tile_key(x.shape[0]))
    return tile(
        x,
        qkvg_weight,
        beta_forget_weight,
        output_weight,
        forget_weight,
        conv_weight,
        a_log,
        dt_bias,
        norm_weight,
        conv_state,
        state,
        state_indices,
        lower_bound=lower_bound,
        norm_eps=norm_eps,
        output_tensor=output_tensor,
    )


# --- host launchers (moved from the Gluon tile so the tile module stays
# torch-free; each imports its tile kernels lazily on first call) ---
def _m256_fallback_linear(x, weight, output_tensor=None):
    import triton

    from aiter.ops.triton._gluon_kernels.gfx950.kimi_delta_attn.fused_kda_layer_decode import (
        _m256_fallback_gemm,
    )

    m, k = x.shape
    n = weight.shape[0]
    out = x.new_empty((m, n)) if output_tensor is None else output_tensor
    _m256_fallback_gemm[(triton.cdiv(n, 64), triton.cdiv(m, 16))](
        x,
        weight,
        out,
        m,
        n,
        k,
        *x.stride(),
        *weight.stride(),
        out.stride(0),
        num_warps=4,
    )
    return out


def kda_layer_decode_m256(
    x,
    qkvg_weight,
    beta_forget_weight,
    output_weight,
    forget_weight,
    conv_weight,
    a_log,
    dt_bias,
    norm_weight,
    conv_state,
    state,
    state_indices,
    *,
    lower_bound=-5.0,
    norm_eps=1e-5,
    output_tensor=None,
):

    import triton

    from aiter.ops.triton._gluon_kernels.gfx950.kimi_delta_attn.fused_kda_layer_decode import (
        _m256_kda_decode,
        _m256_output_m128,
        _m256_output_m256,
        _m256_output_ragged,
        _m256_project_qkvg_beta,
        _m256_project_ragged,
    )

    m = x.shape[0]
    group = 8
    q_write_through = m not in (128, 256)
    qkvg = x.new_empty((m, 6144))
    beta_forget = x.new_empty((m, 144))
    core = x.new_empty((m, 1536))
    aligned = (
        m in (128, 256)
        and x.stride(0) % 8 == 0
        and qkvg_weight.stride(0) % 8 == 0
        and beta_forget_weight.stride(0) % 8 == 0
        and output_weight.stride(0) % 8 == 0
    )
    ragged_aligned = (
        m >= 64
        and x.stride(0) % 8 == 0
        and qkvg_weight.stride(0) % 8 == 0
        and beta_forget_weight.stride(0) % 8 == 0
        and output_weight.stride(0) % 8 == 0
    )
    sort_rows = 128 <= m <= 512 and ragged_aligned
    order_dtype = torch.int64 if state_indices.dtype == torch.int32 else torch.int32
    row_order = (
        torch.empty((m,), device=x.device, dtype=order_dtype) if sort_rows else None
    )
    order_block = triton.next_power_of_2(m)
    order_log = order_block.bit_length() - 1
    if aligned:
        tile_n = 64
        tile_m = 128
        waves = 4
        pid_groups = 16
        beta_columns = 32
        beta_depth = 4
        programs = m // tile_m * (6144 // tile_n) + m // 64 * triton.cdiv(
            144, beta_columns
        )
        _m256_project_qkvg_beta[(programs + int(sort_rows),)](
            x,
            qkvg_weight,
            beta_forget_weight,
            qkvg,
            beta_forget,
            m,
            x.stride(0),
            qkvg_weight.stride(0),
            beta_forget_weight.stride(0),
            tile_n,
            waves,
            pid_groups,
            64,
            beta_columns,
            beta_depth,
            q_write_through,
            qkvg.stride(0),
            beta_forget.stride(0),
            state_indices,
            row_order,
            state_indices.stride(0),
            sort_rows,
            order_block,
            order_log,
            tile_m,
            2,
            3,
            128,
            num_warps=waves,
            waves_per_eu=2,
        )
    elif ragged_aligned:
        _m256_project_ragged[(triton.cdiv(m, 64) * 105 + int(sort_rows),)](
            x,
            qkvg_weight,
            beta_forget_weight,
            qkvg,
            beta_forget,
            m,
            x.stride(0),
            qkvg_weight.stride(0),
            beta_forget_weight.stride(0),
            qkvg.stride(0),
            beta_forget.stride(0),
            state_indices,
            row_order,
            state_indices.stride(0),
            sort_rows,
            order_block,
            order_log,
            num_warps=4,
            waves_per_eu=2,
        )
    else:
        _m256_fallback_linear(x, qkvg_weight, output_tensor=qkvg)
        _m256_fallback_linear(x, beta_forget_weight, output_tensor=beta_forget)
    decode = _m256_kda_decode
    decode[(12 * group, m // group)](
        qkvg,
        qkvg[:, 4608:],
        beta_forget,
        beta_forget[:, 128:140],
        forget_weight,
        conv_weight,
        a_log,
        dt_bias,
        norm_weight,
        conv_state,
        state,
        state_indices,
        core,
        12,
        qkvg.stride(0),
        qkvg.stride(0),
        beta_forget.stride(0),
        beta_forget.stride(0),
        forget_weight.stride(0),
        *conv_state.stride(),
        state.stride(0),
        state_indices.stride(0),
        lower_bound,
        norm_eps,
        group,
        m >= 192,
        8,
        True,
        core.stride(0),
        row_order,
        sort_rows,
        4,
        m >= 128,
        num_warps=8,
        enable_fp_fusion=False,
    )
    out = x.new_empty((m, 7168)) if output_tensor is None else output_tensor
    if aligned and m == 256:
        _m256_output_m256[(m // 128 * 112,)](
            core,
            output_weight,
            out,
            m,
            core.stride(0),
            output_weight.stride(0),
            out.stride(0),
            num_warps=4,
            waves_per_eu=2,
        )
    elif aligned:
        _m256_output_m128[(m // 64 * 112,)](
            core,
            output_weight,
            out,
            core.stride(0),
            output_weight.stride(0),
            out.stride(0),
            num_warps=4,
            waves_per_eu=2,
        )
    elif ragged_aligned:
        _m256_output_ragged[(triton.cdiv(m, 64) * 112,)](
            core,
            output_weight,
            out,
            m,
            core.stride(0),
            output_weight.stride(0),
            out.stride(0),
            num_warps=4,
        )
    else:
        _m256_fallback_linear(core, output_weight, output_tensor=out)
    return out, conv_state, state


def kda_layer_decode_m64(
    x,
    qkvg_weight,
    beta_forget_weight,
    output_weight,
    forget_weight,
    conv_weight,
    a_log,
    dt_bias,
    norm_weight,
    conv_state,
    state,
    state_indices,
    *,
    lower_bound=-5.0,
    norm_eps=1e-5,
    output_tensor=None,
):

    import triton

    from aiter.ops.triton._gluon_kernels.gfx950.kimi_delta_attn.fused_kda_layer_decode import (
        _m32_projection,
        _m64_recurrent_head,
    )

    m = x.shape[0]
    packed = torch.empty((m, 6336), device=x.device, dtype=torch.bfloat16)

    core = torch.empty((m, 1536), device=x.device, dtype=torch.bfloat16)
    out = (
        torch.empty((m, 7168), device=x.device, dtype=torch.bfloat16)
        if output_tensor is None
        else output_tensor
    )
    assert (
        out.shape == (m, 7168)
        and out.dtype == torch.bfloat16
        and out.device == x.device
        and out.is_contiguous()
    )
    rows = 64
    waves = 8
    input_panel = 256
    output_panel = 256
    _m32_projection[(197 * triton.cdiv(m, rows),)](
        x,
        qkvg_weight,
        beta_forget_weight,
        packed,
        m,
        7168,
        x.stride(0),
        qkvg_weight.stride(0),
        beta_forget_weight.stride(0),
        6336,
        True,
        BM=rows,
        NW=waves,
        BK=input_panel,
        num_warps=waves,
    )
    _m64_recurrent_head[(m * 12,)](
        packed,
        conv_weight,
        forget_weight,
        conv_state,
        state,
        state_indices,
        a_log,
        dt_bias,
        norm_weight,
        core,
        forget_weight.stride(0),
        state_indices.stride(0),
        *conv_state.stride(),
        state.stride(0),
        lower_bound,
        norm_eps,
        NW=8,
        M=m,
        GROUP=2,
        CORE_STRIDE=core.stride(0),
        NARROW_SLOT=(
            m > 32
            and state.shape[0] * state.stride(0) < 2**31
            and conv_state.shape[0] * conv_state.stride(0) < 2**31
        ),
        num_warps=8,
        enable_fp_fusion=False,
        waves_per_eu=1,
    )
    _m32_projection[(224 * triton.cdiv(m, rows),)](
        core,
        output_weight,
        output_weight,
        out,
        m,
        1536,
        core.stride(0),
        output_weight.stride(0),
        output_weight.stride(0),
        7168,
        False,
        BM=rows,
        NW=waves,
        BK=output_panel,
        num_warps=waves,
    )
    return out, conv_state, state
