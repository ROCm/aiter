# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 grouped MoE GEMM (a8w4 default / a4w4).

This module owns the FlyDSL grouped-GEMM path so the generic ``fused_moe``
dispatcher does not carry gfx1250-specific implementation details.

Optional **DeepGEMM-style contiguous M-tile** scheduler: set environment
``AITER_GROUPED_DEEPGEMM_CONTIGUOUS=1`` or CSV column ``grouped_contiguous_m=1``.
Block swizzle matches ``deep_gemm::Scheduler::get_swizzled_block_idx`` in
DeepGEMM's ``scheduler.cuh``; optional override ``AITER_DEEPGEMM_NUM_1D_BLOCKS=8|16``.
"""

import os
import csv
import functools

from typing import Optional

import torch

from aiter import ActivationType, QuantType, dtypes, logger
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode

# Opt-in switch for the gfx1250 FlyDSL grouped-GEMM path.
_TRUTHY_ENV = ("1", "true", "True", "yes", "YES")
_GROUPED_CONFIG_CACHE = {}
_WARNED_NAIVE_EPILOGUE = False
# Cache the contiguous uint8 view of static MoE weights so a non-contiguous
# weight is materialized at most once (not re-copied on every fused_moe call).
_GROUPED_WEIGHT_CACHE = {}


def _grouped_weight_uint8(w: torch.Tensor) -> torch.Tensor:
    """Return a contiguous uint8 view of a (static) MoE weight, cached by buffer.

    Weights don't change across decode/bench steps, so this turns a potential
    per-call ``.contiguous()`` D2D copy into a one-time cost. Keyed by
    ``data_ptr`` (weights stay alive, so no buffer-reuse aliasing in practice);
    the cache is bounded and cleared if it grows unexpectedly.
    """
    key = (w.data_ptr(), tuple(w.shape), tuple(w.stride()), str(w.dtype))
    cached = _GROUPED_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    out = (w if w.dtype == torch.uint8 else w.view(torch.uint8)).contiguous()
    if len(_GROUPED_WEIGHT_CACHE) > 64:
        _GROUPED_WEIGHT_CACHE.clear()
    _GROUPED_WEIGHT_CACHE[key] = out
    return out


def _as_bool(value, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip() in _TRUTHY_ENV


def _as_int(value, default: int | None) -> int | None:
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _dtype_name(dtype) -> str:
    if dtype is torch.bfloat16 or dtype == dtypes.bf16:
        return "torch.bfloat16"
    if dtype is torch.float16 or dtype == dtypes.fp16:
        return "torch.float16"
    return str(dtype)


def _enum_name(value) -> str:
    if hasattr(value, "name"):
        return f"{type(value).__name__}.{value.name}"
    return str(value)


def _load_grouped_config_rows():
    cfg_path = os.environ.get("AITER_CONFIG_GROUPED_FMOE")
    if not cfg_path:
        try:
            from aiter.jit.core import AITER_CONFIGS

            cfg_path = AITER_CONFIGS.AITER_CONFIG_GROUPED_FMOE_FILE
        except Exception:
            cfg_path = ""
    cached = _GROUPED_CONFIG_CACHE.get(cfg_path)
    if cached is not None:
        return cached
    rows = []
    for path in str(cfg_path).split(os.pathsep):
        if not path or not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    _GROUPED_CONFIG_CACHE[cfg_path] = rows
    return rows


def _nextPow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


_PADDED_M_TIERS = [32768, 131072]


def _get_padded_M(M):
    if M < _PADDED_M_TIERS[0]:
        return _nextPow2(M)
    for tier in reversed(_PADDED_M_TIERS):
        if M >= tier:
            return tier
    return _PADDED_M_TIERS[0]


@functools.lru_cache(maxsize=1024)
def _find_grouped_config(
    *,
    token_num: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    activation,
    dtype,
    q_dtype_a,
    q_dtype_w,
    quant_type,
    gate_mode,
):
    from aiter.jit.utils.chip_info import get_cu_num

    keys = {
        "gfx": str(get_gfx()),
        "cu_num": str(get_cu_num()),
        "token": str(int(token_num)),
        "model_dim": str(int(model_dim)),
        "inter_dim": str(int(inter_dim)),
        "expert": str(int(experts)),
        "topk": str(int(topk)),
        "act_type": _enum_name(activation),
        "dtype": _dtype_name(dtype),
        "q_dtype_a": str(q_dtype_a),
        "q_dtype_w": str(q_dtype_w),
        "q_type": _enum_name(quant_type),
        "gate_mode": _enum_name(gate_mode),
    }
    rows = _load_grouped_config_rows()

    # Hardware is locked by (gfx, cu_num): gfx (architecture) is always a hard
    # constraint, while cu_num can be relaxed as a fallback. Columns missing from
    # the CSV (e.g. older configs without a 'gfx' column) are skipped, so this
    # stays backward compatible with pre-gfx tuned files.
    def _matches(row, *, require_cu_num: bool):
        for k, v in keys.items():
            if k == "cu_num" and not require_cu_num:
                continue
            if row.get(k) and str(row.get(k)).strip() != v:
                return False
        return True

    matches = [row for row in rows if _matches(row, require_cu_num=True)]
    if not matches:
        matches = [row for row in rows if _matches(row, require_cu_num=False)]
    if not matches:
        if os.environ.get("AITER_GROUPED_DEBUG", "0") not in (
            "",
            "0",
            "false",
            "False",
        ):
            print(
                f"[grouped-gemm-debug] no grouped CSV config match for {keys}; "
                f"loaded_rows={len(rows)}",
                flush=True,
            )
        return None
    matches.sort(key=lambda r: float(r.get("us") or 0.0))
    return matches[0]


def _use_grouped_gemm_enabled() -> bool:
    """Runtime check for AITER_USE_GROUPED_GEMM so tests can toggle it."""
    return os.environ.get("AITER_USE_GROUPED_GEMM", "1") in _TRUTHY_ENV


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be > 0, got {alignment}")
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _make_contiguous_psum_layout(
    *,
    masked_m: torch.Tensor,
    rows_to_tokens: torch.Tensor,
    topids_to_rows: torch.Tensor,
    experts: int,
    max_m: int,
    tile_m: int,
    token_num: int,
    topk: int,
):
    """Build DeepGEMM-style psum layout: grouped_layout[e] = actual_end.

    contiguous_m is a static upper bound (no .item() sync), so this is safe
    during CUDAGraph capture. Padding rows are never read by GEMM/gather.
    """
    device = masked_m.device

    starts_t, psum_t, _ = contiguous_psum(masked_m, int(experts), int(tile_m))
    ub = int(token_num) * int(topk) + int(experts) * (int(tile_m) - 1)
    contiguous_m = max(int(tile_m), _align_up(ub, int(tile_m)))

    old_flat = topids_to_rows.reshape(-1)
    expert = torch.div(old_flat, int(max_m), rounding_mode="floor")
    slot = old_flat - expert * int(max_m)
    new_flat = starts_t[expert.to(torch.long)] + slot
    remapped_topids = new_flat.to(torch.int32).view_as(topids_to_rows)

    # Inverse map (contiguous row -> source token) via one scatter.
    remapped_rows = torch.full(
        (int(contiguous_m),), -1, device=device, dtype=torch.int32
    )
    src_tokens = rows_to_tokens[old_flat.to(torch.long)]
    remapped_rows[new_flat.to(torch.long)] = src_tokens

    return remapped_topids, remapped_rows, psum_t, int(contiguous_m)


def _grouped_a8w4_preshuffle_e8m0_scale(
    scale: torch.Tensor,
    warp_tile: int,
    scale_k_per_tile: int = 4,
) -> torch.Tensor:
    # Preshuffle row/k-scale axes; experts stay as the leading batch dim.
    scale = scale.view(torch.uint8).contiguous()
    E, _, k_scale = scale.shape
    wmma_rep = int(warp_tile) // 16
    k_groups = k_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // 4
    g = scale.view(E, -1, wmma_rep, 16, k_groups, k_wmma_steps, 4)
    g = g.permute(0, 1, 3, 4, 5, 2, 6).contiguous()
    return g.reshape(E, -1, k_groups * k_wmma_steps * wmma_rep * 4)


def _grouped_a8w4_prepare_scale_batch(
    scale: torch.Tensor,
    *,
    experts: int,
    rows: int,
    k_dim: int,
    warp_tile: int,
    tile_k: int,
    device: torch.device,
) -> torch.Tensor:
    scale_u8 = scale.view(torch.uint8).contiguous()
    raw_shape = (experts, rows, k_dim // 32)
    wmma_rep = int(warp_tile) // 16
    preshuffled_shape = (experts, rows // wmma_rep, (k_dim // 32) * wmma_rep)
    if tuple(scale_u8.shape) == preshuffled_shape:
        return scale_u8
    if tuple(scale_u8.shape) == (experts * rows, k_dim // 32):
        scale_u8 = scale_u8.view(raw_shape)
    elif tuple(scale_u8.shape) != raw_shape:
        raise ValueError(
            f"scale shape must be raw {raw_shape}, flat raw {(experts * rows, k_dim // 32)} "
            f"or preshuffled {preshuffled_shape}, got {tuple(scale_u8.shape)}"
        )
    scale_k_per_tile = int(tile_k) // 32
    return _grouped_a8w4_preshuffle_e8m0_scale(
        scale_u8, warp_tile=warp_tile, scale_k_per_tile=scale_k_per_tile
    ).to(device=device)


def _build_route_maps_naive(topk_ids: torch.Tensor, E: int, max_m: int):
    """Torch fallback for route -> grouped-row maps."""
    import torch.nn.functional as F

    device = topk_ids.device
    token_num, topk = topk_ids.shape
    flat_e = topk_ids.reshape(-1).to(torch.long)
    # slot = number of earlier routes to the same expert (token-major order).
    slot = F.one_hot(flat_e, E).cumsum(0).gather(1, flat_e[:, None]).squeeze(1) - 1
    topids_to_rows = (flat_e * max_m + slot).to(torch.int32)
    # Inverse map: grouped row -> source token (-1 for unused padding rows).
    rows_to_tokens = torch.full((E * max_m,), -1, dtype=torch.int32, device=device)
    src_tokens = torch.arange(
        token_num, device=device, dtype=torch.int32
    ).repeat_interleave(topk)
    rows_to_tokens[topids_to_rows.to(torch.long)] = src_tokens
    masked_m = torch.bincount(flat_e, minlength=E).to(torch.int32)
    return topids_to_rows.view(token_num, topk), rows_to_tokens, masked_m


def _maybe_grouped_gfx1250_a8w4_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    E: int,
    model_dim: int,
    inter_dim: int,
    dtype: torch.dtype,
    activation: ActivationType,
    quant_type: QuantType,
    q_dtype_a,
    q_dtype_w,
    isG1U1: bool,
    doweight_stage1: bool,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    expert_mask: Optional[torch.Tensor],
    hidden_pad: int,
    intermediate_pad: int,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    gate_mode: GateMode = GateMode.SEPARATED,
):
    def _grouped_dbg(msg: str, stacklevel: int = 1):
        if os.environ.get("AITER_GROUPED_DEBUG", "0") not in (
            "",
            "0",
            "false",
            "False",
        ):
            import inspect

            frame = inspect.stack()[stacklevel]
            print(
                f"[grouped-gemm-debug] {frame.filename}:{frame.lineno} {msg}",
                flush=True,
            )

    def _fmt(v):
        if isinstance(v, torch.Tensor):
            return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype})"
        return repr(v)

    _grouped_dbg(
        "inputs: "
        + ", ".join(
            f"{k}={_fmt(v)}"
            for k, v in [
                ("hidden_states", hidden_states),
                ("w1", w1),
                ("w2", w2),
                ("topk_weight", topk_weight),
                ("topk_ids", topk_ids),
                ("E", E),
                ("model_dim", model_dim),
                ("inter_dim", inter_dim),
                ("dtype", dtype),
                ("activation", activation),
                ("quant_type", quant_type),
                ("q_dtype_a", q_dtype_a),
                ("q_dtype_w", q_dtype_w),
                ("isG1U1", isG1U1),
                ("doweight_stage1", doweight_stage1),
                ("w1_scale", w1_scale),
                ("w2_scale", w2_scale),
                ("expert_mask", expert_mask),
                ("hidden_pad", hidden_pad),
                ("intermediate_pad", intermediate_pad),
                ("bias1", bias1),
                ("bias2", bias2),
                ("gate_mode", gate_mode),
            ]
        )
    )
    _grouped_dbg("enter grouped helper")
    # Main opt-in plus legacy kill switch.
    if not _use_grouped_gemm_enabled():
        _grouped_dbg("AITER_USE_GROUPED_GEMM not enabled; skip grouped mode")
        return None
    if os.environ.get("AITER_DISABLE_GROUPED_A8W4", "0") == "1":
        _grouped_dbg("AITER_DISABLE_GROUPED_A8W4 enabled; skip grouped mode")
        return None
    os.environ["AITER_LAST_FUSED_MOE_IMPL"] = "default"
    if expert_mask is not None or bias1 is not None or bias2 is not None:
        _grouped_dbg("bias1 and bias not none")
        # return None
    if hidden_pad != 0 or intermediate_pad != 0:
        hidden_pad = 0
        intermediate_pad = 0
        _grouped_dbg("haspad")
        # return None
    if not isG1U1 or quant_type != QuantType.per_1x32:
        _grouped_dbg("not g1u1 or not 1x32")
        return None
    if activation not in (ActivationType.Silu, ActivationType.Swiglu):
        _grouped_dbg("not silu or not swiglu")
        return None
    if gate_mode not in (GateMode.SEPARATED, GateMode.INTERLEAVE):
        _grouped_dbg(f"unsupported gate_mode={gate_mode}")
        return None
    # Default layout follows gate_mode; env override is for diagnostics.
    env_stage1_layout = (
        os.environ.get("AITER_GROUPED_STAGE1_WEIGHT_LAYOUT", "").strip().lower()
    )
    if env_stage1_layout:
        if env_stage1_layout not in ("gguu", "gugu"):
            raise ValueError(
                "AITER_GROUPED_STAGE1_WEIGHT_LAYOUT must be 'gguu' or 'gugu', "
                f"got {env_stage1_layout!r}"
            )
        stage1_weight_layout = env_stage1_layout
        _grouped_dbg(
            f"stage1_weight_layout overridden by env: {stage1_weight_layout!r}"
        )
    else:
        stage1_weight_layout = "gugu" if gate_mode == GateMode.INTERLEAVE else "gguu"
    # Log the stage1 gate/up layout used by the grouped kernel (debug only).
    logger.debug(
        "[MoE-GUMODE] gate_mode=%s -> stage1_weight_layout=%s (%s)",
        gate_mode.name,
        stage1_weight_layout,
        stage1_weight_layout.upper(),
    )
    is_grouped_a4w4 = q_dtype_a == dtypes.fp4x2 and q_dtype_w == dtypes.fp4x2
    is_grouped_a8w4 = q_dtype_a == dtypes.fp8 and (
        q_dtype_w == dtypes.fp4x2 or w1.dtype == torch.uint8
    )
    if not (is_grouped_a4w4 or is_grouped_a8w4):
        return None
    data_format = "fp4" if is_grouped_a4w4 else "a8w4"
    # Normalize uint8-viewed fp4 weights back to fp4x2 for CSV key matching.
    q_dtype_w_key = (
        dtypes.fp4x2
        if (q_dtype_w == dtypes.fp4x2 or w1.dtype == torch.uint8)
        else q_dtype_w
    )
    _grouped_dbg(f"eligible data_format={data_format}")
    if w1_scale is None or w2_scale is None:
        return None
    _gfx_env = ";".join(
        str(os.environ.get(k, "")).lower()
        for k in ("GPU_ARCHS", "TARGET_ARCH", "AITER_GPU_ARCHS")
    )
    _force_gfx1250 = os.environ.get("AITER_FORCE_GFX1250", "0") in _TRUTHY_ENV
    if get_gfx() != "gfx1250" and "gfx1250" not in _gfx_env and not _force_gfx1250:
        return None

    try:
        from aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250 import (
            compile_moe_grouped_gemm1_a8w4_masked,
            compile_moe_grouped_gemm2_a8w4_masked,
            compile_moe_grouped_gemm1_mxfp4_masked,
            compile_moe_grouped_gemm2_mxfp4_masked,
        )
    except Exception as vendored_exc:
        try:
            from kernels.moe_grouped_gemm_mxscale_gfx1250 import (
                compile_moe_grouped_gemm1_a8w4_masked,
                compile_moe_grouped_gemm2_a8w4_masked,
                compile_moe_grouped_gemm1_mxfp4_masked,
                compile_moe_grouped_gemm2_mxfp4_masked,
            )
        except Exception as exc:
            logger.warning(
                f"[grouped_a8w4] grouped FlyDSL import failed, fallback: "
                f"vendored={vendored_exc}; flydsl={exc}"
            )
            return None

    _grouped_dbg("imports done")
    device = hidden_states.device
    token_num, topk = topk_ids.shape
    tile_m, tile_n, tile_k = 64, 256, 256
    m_warp, n_warp = 1, 4
    num_buffers = 2
    split_k1 = 1
    split_k2 = 1
    grouped_contiguous_m = False
    cfg_row = _find_grouped_config(
        token_num=_get_padded_M(token_num),
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        activation=activation,
        dtype=dtype,
        q_dtype_a=q_dtype_a,
        q_dtype_w=q_dtype_w_key,
        quant_type=quant_type,
        gate_mode=gate_mode,
    )
    if cfg_row is not None:
        tile_m = _as_int(cfg_row.get("tile_m"), tile_m)
        n_warp = _as_int(cfg_row.get("n_warp"), n_warp)
        num_buffers = _as_int(cfg_row.get("num_buffers"), num_buffers)
        split_k1 = _as_int(cfg_row.get("split_k1"), split_k1)
        split_k2 = _as_int(cfg_row.get("split_k2"), split_k2)
        grouped_contiguous_m = _as_bool(
            cfg_row.get("grouped_contiguous_m"), grouped_contiguous_m
        )
        stage1_weight_layout = (
            cfg_row.get("stage1_weight_layout") or stage1_weight_layout
        )
        _grouped_dbg(f"using grouped CSV config: {cfg_row}")
    tile_n = int(n_warp) * 64
    tile_k = 256
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp

    if os.environ.get("AITER_GROUPED_DEEPGEMM_CONTIGUOUS", "0") in _TRUTHY_ENV:
        grouped_contiguous_m = True
    # Switch to DeepGEMM-style contiguous-M at large batches (env-overridable).
    _contig_token_threshold = _as_int(
        os.environ.get("AITER_GROUPED_CONTIGUOUS_TOKEN_THRESHOLD"), 512
    )
    if token_num > _contig_token_threshold:
        grouped_contiguous_m = True
        _grouped_dbg(
            f"token_num={token_num} > {_contig_token_threshold}; "
            "auto-enable contiguous M scheduler"
        )
    if grouped_contiguous_m:
        _grouped_dbg("DeepGEMM contiguous M scheduler enabled")

    # topk_ids is already an integer tensor; keep one flattened view for routing.
    flat_experts = topk_ids.reshape(-1)
    # [crash-probe] syncs are debug-only; gated by AITER_GROUPED_DEBUG.
    _grouped_sync_dbg = os.environ.get("AITER_GROUPED_DEBUG", "0") not in (
        "",
        "0",
        "false",
        "False",
    )
    # Expert-id range validation is a debug-only safety check: at decode sizes it
    # issues ~6 tiny launches/iter (lt+ge compare_scalar, two any() reductions)
    # plus a device->host sync from the `or` -- a real hotspot relative to the
    # tiny grouped work. Gate it behind AITER_GROUPED_DEBUG so production skips it
    # (topk_ids is already produced in-range by the router); set the env to 1 to
    # re-enable the check when diagnosing bad route ids. Skip entirely during
    # CUDAGraph capture (dynamic control flow / sync).
    if _grouped_sync_dbg:
        if torch.any(flat_experts < 0) or torch.any(flat_experts >= E):
            raise ValueError("grouped a8w4 path expects local expert ids in [0, E)")
    counts = None
    raw_max_m = _as_int(cfg_row.get("max_m"), token_num) if cfg_row else token_num
    _grouped_dbg(f"routing cfg_row={cfg_row} raw_max_m={raw_max_m}")
    max_m = max(
        warp_tile_m, ((raw_max_m + warp_tile_m - 1) // warp_tile_m) * warp_tile_m
    )
    _grouped_dbg(f"routing max_m={max_m}")

    # Build route maps once. The fast path uses the FlyDSL atomic-scatter kernel;
    # the naive path keeps a deterministic torch fallback for tests/debug.
    _use_naive = os.environ.get("AITER_GROUPED_GEMM_NAIVE", "0") == "1"
    # Per-expert counts are only consumed by the naive epilogues, the doweight
    # multiply, and the optional dump (masked_m drives the GEMM). Build it only on
    # the naive path so the fast path skips the bincount (two int reductions + a
    # host sync); the lazy fallbacks below recompute it if ever needed.
    if _use_naive:
        if counts is None:
            counts = torch.bincount(flat_experts.to(torch.long), minlength=E)
        topids_to_rows, rows_to_tokens, masked_m = _build_route_maps_naive(
            topk_ids, E, max_m
        )
        route_tokens = rows_to_tokens.view(E, max_m).to(torch.long)
    else:
        if doweight_stage1:
            raise NotImplementedError(
                "doweight_stage1 is only supported on the grouped NAIVE path; "
                "set AITER_GROUPED_GEMM_NAIVE=1"
            )

        topids_to_rows, rows_to_tokens, masked_m = build_route_maps(topk_ids, E, max_m)
    # Grouped row -> source token, (E, max_m); padding rows (-1) are never read
    # because the naive epilogues are bounded by per-expert counts.
    out_dtype_str = "bf16" if dtype == dtypes.bf16 else "f16"
    m_tile_prefix = None
    m_tile_map = None
    route_E = E
    route_max_m = max_m
    effective_grouped_contiguous_m = bool(grouped_contiguous_m)
    if not _use_naive and effective_grouped_contiguous_m:
        topids_to_rows, rows_to_tokens, m_tile_map, contiguous_m = (
            _make_contiguous_psum_layout(
                masked_m=masked_m,
                rows_to_tokens=rows_to_tokens,
                topids_to_rows=topids_to_rows,
                experts=E,
                max_m=max_m,
                tile_m=tile_m,
                token_num=token_num,
                topk=topk,
            )
        )
        route_E = 1
        route_max_m = int(contiguous_m)

    def _quantize_mxfp8_payload(x: torch.Tensor, last_dim: int):
        from aiter.ops.triton.quant import dynamic_mxfp8_quant

        y, scale = dynamic_mxfp8_quant(
            x.contiguous().view(-1, last_dim), quant_dtype=dtypes.fp8
        )
        payload = y.view(torch.uint8).contiguous().view(*x.shape)
        scale_u8 = (
            scale.view(*x.shape[:-1], last_dim // 32).view(torch.uint8).contiguous()
        )
        return payload, scale_u8

    if data_format == "fp4":
        # a1 fp4 quant: AITER_GROUPED_GEMM_NAIVE=1 uses the torch reference;
        # the fast path uses the HIP MXFP4 quant kernel so its e8m0 rounding
        # matches the production HIP quant contract.
        if _use_naive:
            from aiter.ops.quant import per_1x32_f4_quant as _a1_f4_quant
        else:
            from aiter.ops.quant import per_1x32_f4_quant_hip as _a1_f4_quant

        _grouped_dbg("start a1 fp4 quant")
        a1_quant, a1_scale_token = _a1_f4_quant(
            hidden_states, quant_dtype=dtypes.fp4x2, shuffle=False
        )
        _grouped_dbg("a1 fp4 quant done")
        a1_payload = a1_quant.view(torch.uint8).contiguous()
        a1_scale_token_u8 = a1_scale_token.view(torch.uint8).contiguous()
        grouped_a1 = torch.empty(
            (route_E, route_max_m, model_dim // 2), dtype=torch.uint8, device=device
        )
        # Only the naive path needs the row-major scale buffer; the fast path
        # gathers + preshuffles the scale in one fused kernel (no a1_scale_raw).
        a1_scale_raw = (
            torch.empty(
                (route_E, route_max_m, model_dim // 32),
                dtype=torch.uint8,
                device=device,
            )
            if _use_naive
            else None
        )
    else:
        # a8w4 stage1 input: per-block-32 MXFP8 quantization.
        a1_payload, a1_scale_token_u8 = _quantize_mxfp8_payload(
            hidden_states, model_dim
        )
        grouped_a1 = torch.empty(
            (route_E, route_max_m, model_dim), dtype=torch.uint8, device=device
        )
        # Padding rows decode with scale=1.0. Only the naive path needs the
        # row-major scale buffer; the fast path fuses gather + preshuffle.
        a1_scale_raw = (
            torch.empty(
                (route_E, route_max_m, model_dim // 32),
                dtype=torch.uint8,
                device=device,
            )
            if _use_naive
            else None
        )

    # Route-gather into the grouped per-expert layout.
    if not _use_naive:
        _grouped_dbg("start route gather (scatter-copy kernel)")
        # Payload route-gather only; the e8m0 scale is route-gathered AND
        # preshuffled into the WMMA layout in one fused pass below (no
        # intermediate row-major a1_scale_raw + separate permute).
        flydsl_moe_scatter_copy_token(
            a1_payload,
            None,
            rows_to_tokens,
            route_E,
            route_max_m,
            grouped_a1=grouped_a1,
        )
        grouped_a1_scale = flydsl_moe_scatter_preshuffle_scale(
            a1_scale_token_u8,
            rows_to_tokens,
            route_E,
            route_max_m,
            wmma_rep=warp_tile_m // 16,
            scale_k_per_tile=tile_k // 32,
        )
        _grouped_dbg("route gather + scale preshuffle done")
    else:
        _grouped_dbg("start route gather (naive)")
        # Naive torch route-gather.
        flat_routes = torch.arange(token_num * topk, device=device, dtype=torch.long)
        flat_tokens = flat_routes // topk
        flat_rows = topids_to_rows.reshape(-1).to(torch.long)
        grouped_a1.view(E * max_m, -1)[flat_rows] = a1_payload[flat_tokens]
        if a1_scale_token_u8 is not None:
            a1_scale_raw.view(E * max_m, -1)[flat_rows] = a1_scale_token_u8[flat_tokens]
        # Only the naive epilogue needs grouped row weights.
        route_weights = torch.empty((E, max_m), dtype=dtype, device=device)
        route_weights.view(-1)[topids_to_rows.reshape(-1)] = topk_weight.reshape(-1).to(
            route_weights.dtype
        )
        grouped_a1_scale = _grouped_a8w4_preshuffle_e8m0_scale(
            a1_scale_raw, warp_tile=warp_tile_m, scale_k_per_tile=tile_k // 32
        )
        _grouped_dbg("route gather done")

    grouped_w1 = _grouped_weight_uint8(w1)
    grouped_w2 = _grouped_weight_uint8(w2)
    _grouped_dbg("weight layout done")
    # Weight scales are already preshuffled per expert.
    _wmma_rep = warp_tile_n // 16
    grouped_w1_scale = w1_scale.reshape(
        E, (2 * inter_dim) // _wmma_rep, (model_dim // 32) * _wmma_rep
    )
    grouped_w2_scale = w2_scale.reshape(
        E, model_dim // _wmma_rep, (inter_dim // 32) * _wmma_rep
    )

    # grouped_a1_scale already produced above (fast or naive path).
    _grouped_dbg("scale layout done")

    grouped_a2 = torch.empty(
        (route_E, route_max_m, inter_dim), dtype=dtype, device=device
    )
    stage1_compiler = (
        compile_moe_grouped_gemm1_mxfp4_masked
        if data_format == "fp4"
        else compile_moe_grouped_gemm1_a8w4_masked
    )
    _grouped_dbg("start stage1 compile")
    stage1 = stage1_compiler(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        max_m=max_m,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        out_dtype=out_dtype_str,
        num_buffers=num_buffers,
        split_k=split_k1,
        expert_sched_mode=False,
        grouped_persistent_m=False,
        grouped_contiguous_m=effective_grouped_contiguous_m,
        persistent_workers=None,
        act="swiglu" if activation == ActivationType.Swiglu else "silu",
        stage1_weight_layout=stage1_weight_layout,
    )
    _grouped_dbg("stage1 compile done; start launch")
    _bias1_arg = bias1 if (bias1 is not None and bias1.numel() > 0) else None
    if _bias1_arg is not None and _bias1_arg.dtype != dtype:
        _bias1_arg = _bias1_arg.to(dtype)
    if _grouped_sync_dbg:
        torch.cuda.synchronize()
    _grouped_dbg(f"[crash-probe] before stage1 tokens={token_num} max_m={max_m} E={E}")
    stage1(
        grouped_a2,
        grouped_a1,
        grouped_w1,
        grouped_a1_scale,
        grouped_w1_scale,
        masked_m,
        max_m,
        inter_dim,
        model_dim,
        E,
        stream=torch.cuda.current_stream(),
        _m_tile_prefix=m_tile_prefix,
        _m_tile_map=m_tile_map,
        bias=_bias1_arg,
    )
    if _grouped_sync_dbg:
        torch.cuda.synchronize()
    _grouped_dbg("[crash-probe] after stage1 sync OK, unsort")
    _grouped_dbg("[crash-probe] after stage1 sync OK")

    # Optional single-token stage1 dump.
    _dump_a2 = os.environ.get("AITER_GROUPED_DUMP_A2", "0")
    if _dump_a2 not in ("", "0", "false", "False"):
        if token_num == 1:
            _routed_experts = topk_ids[0].to(torch.long)
            _a2_tt = grouped_a2[_routed_experts, 0].view(token_num, topk, inter_dim)
            print(
                f"[dump] grouped_a2 (num_token, topk, inter_dim)={tuple(_a2_tt.shape)}",
                flush=True,
            )
            for _k in range(topk):
                _row = _a2_tt[0, _k, :10].detach().cpu().tolist()
                print(
                    f"[dump]   topk={_k} expert={int(_routed_experts[_k])} "
                    f"first10={_row}",
                    flush=True,
                )
        else:
            _grouped_dbg(
                f"[dump] skip grouped_a2 dump: only num_token==1 supported "
                f"(got token_num={token_num})"
            )

    if doweight_stage1:
        # doweight_stage1 is only supported on the naive path.
        for e in range(E):
            n = int(counts[e].item())
            if n:
                grouped_a2[e, :n].mul_(route_weights[e, :n].view(-1, 1))

    if data_format == "fp4":
        # a2 fp4 quant: same NAIVE gating as a1 -- torch reference on NAIVE=1,
        # HIP MXFP4 quant on the fast path.
        if _use_naive:
            from aiter.ops.quant import per_1x32_f4_quant as _a2_f4_quant
        else:
            from aiter.ops.quant import per_1x32_f4_quant_hip as _a2_f4_quant

        _grouped_dbg("start a2 fp4 quant")
        a2_quant, a2_scale_token = _a2_f4_quant(
            grouped_a2.view(route_E * route_max_m, inter_dim),
            quant_dtype=dtypes.fp4x2,
            shuffle=False,
        )
        _grouped_dbg("a2 fp4 quant done")
        grouped_a2_payload = (
            a2_quant.view(torch.uint8)
            .contiguous()
            .view(route_E, route_max_m, inter_dim // 2)
        )
        a2_scale_raw = (
            a2_scale_token.view(torch.uint8)
            .contiguous()
            .view(route_E, route_max_m, inter_dim // 32)
        )
        if _grouped_sync_dbg:
            torch.cuda.synchronize()
        _grouped_dbg("[crash-probe] after a2 fp4 quant sync OK")
    else:
        # a8w4 stage2 input also needs per-block-32 MXFP8 scale; SiLU outputs
        # can exceed unit-scale fp8 and direct casts may encode NaNs.
        grouped_a2_payload, a2_scale_raw = _quantize_mxfp8_payload(
            grouped_a2, inter_dim
        )
    if _use_naive:
        grouped_a2_scale = _grouped_a8w4_preshuffle_e8m0_scale(
            a2_scale_raw, warp_tile=warp_tile_m, scale_k_per_tile=tile_k // 32
        )
    else:
        # a2_scale_raw is already grouped row-major (no route-gather), so use the
        # in-kernel preshuffle (gather-less fused version, like stage1).

        grouped_a2_scale = flydsl_moe_preshuffle_scale(
            a2_scale_raw,
            route_E,
            route_max_m,
            wmma_rep=warp_tile_m // 16,
            scale_k_per_tile=tile_k // 32,
        )
    _grouped_dbg("a2 scale layout done")
    grouped_out = torch.empty(
        (route_E, route_max_m, model_dim), dtype=dtype, device=device
    )
    stage2_compiler = (
        compile_moe_grouped_gemm2_mxfp4_masked
        if data_format == "fp4"
        else compile_moe_grouped_gemm2_a8w4_masked
    )
    _grouped_dbg("start stage2 compile")
    stage2 = stage2_compiler(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        max_m=max_m,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        out_dtype=out_dtype_str,
        num_buffers=num_buffers,
        split_k=split_k2,
        expert_sched_mode=False,
        grouped_persistent_m=False,
        grouped_contiguous_m=effective_grouped_contiguous_m,
        persistent_workers=None,
    )
    _grouped_dbg("stage2 compile done; start launch")
    _bias2_arg = bias2 if (bias2 is not None and bias2.numel() > 0) else None
    if _bias2_arg is not None and _bias2_arg.dtype != dtype:
        _bias2_arg = _bias2_arg.to(dtype)
    if _grouped_sync_dbg:
        torch.cuda.synchronize()
    _grouped_dbg(f"[crash-probe] before stage2 tokens={token_num} max_m={max_m} E={E}")
    stage2(
        grouped_out,
        grouped_a2_payload,
        grouped_w2,
        grouped_a2_scale,
        grouped_w2_scale,
        masked_m,
        max_m,
        model_dim,
        inter_dim,
        E,
        stream=torch.cuda.current_stream(),
        _m_tile_prefix=m_tile_prefix,
        _m_tile_map=m_tile_map,
        bias=_bias2_arg,
    )
    if _grouped_sync_dbg:
        torch.cuda.synchronize()
    _grouped_dbg("[crash-probe] after stage2 sync OK")
    if os.environ.get("MOE_DUMP_INTER", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    ):
        _dump_counts = (
            counts if counts is not None else torch.bincount(flat_experts, minlength=E)
        )
        _e0 = (
            int(torch.nonzero(_dump_counts > 0)[0].item())
            if (_dump_counts > 0).any()
            else 0
        )
        print(
            f"  aiter   grouped_out_stage2[e0={_e0},0,:10]="
            f"{grouped_out[_e0, 0].float()[:10].tolist()} (pre route-weight)",
            flush=True,
        )

    moe_out = torch.empty((token_num, model_dim), dtype=dtype, device=device)
    # Fast epilogue gathers/reduces grouped rows back to token order.
    if (not _use_naive) and dtype in (dtypes.bf16, dtypes.fp16):
        _grouped_dbg("start gather-reduce output")
        # Reuse the route map; the kernel accumulates in f32.
        gather_w = (
            torch.ones((token_num, topk), dtype=dtype, device=device)
            if doweight_stage1
            else topk_weight.to(dtype)
        )
        flydsl_moe_gather_reduce(grouped_out, topids_to_rows, gather_w, out=moe_out)
        _grouped_dbg("gather-reduce output done")
    else:
        _grouped_dbg("start scatter output")
        # Naive fallback: per-expert D2D loop (slow). Warn once outside capture.
        global _WARNED_NAIVE_EPILOGUE
        if not _WARNED_NAIVE_EPILOGUE:
            _WARNED_NAIVE_EPILOGUE = True
            logger.warning(
                "[grouped_a8w4] slow naive scatter epilogue: per-expert loop "
                "(E=%d) issues E D2D copies + E D2H syncs per call. Use dtype "
                "bf16/fp16 and unset AITER_GROUPED_GEMM_NAIVE to take the fused "
                "flydsl_moe_gather_reduce path.",
                E,
            )
        # Naive fallback epilogue.
        if counts is None:
            counts = torch.bincount(flat_experts, minlength=E)
        for e in range(E):
            n = int(counts[e].item())
            if n == 0:
                continue
            vals = grouped_out[e, :n]
            if not doweight_stage1:
                vals = vals * route_weights[e, :n].view(-1, 1)
            moe_out.index_add_(0, route_tokens[e, :n], vals)
        _grouped_dbg("scatter output done")
    impl_name = "grouped_a4w4" if data_format == "fp4" else "grouped_a8w4"
    os.environ["AITER_LAST_FUSED_MOE_IMPL"] = impl_name
    logger.debug(
        f"[{impl_name}] used grouped FlyDSL {data_format} path: tokens={token_num}, topk={topk}, E={E}, max_m={max_m}"
    )
    return moe_out


# --- Functions moved from moe_kernels.py for grouped gemm ---
@functools.cache
def _get_compiled_gather_reduce(model_dim: int, topk: int, out_dtype: str):
    """Compile and cache the one-pass MoE gather-reduce kernel."""
    from aiter.ops.flydsl.kernels.moe_gather_reduce import (
        build_moe_gather_reduce_module,
    )

    return build_moe_gather_reduce_module(model_dim, topk, out_dtype)


def build_topids_to_rows(
    topk_ids: torch.Tensor,  # (token_num, topk) local expert ids in [0, E)
    max_m: int,
    E: int,
) -> torch.Tensor:
    """Per-token gather map: ``topids_to_rows[t,k] = topk_ids[t,k]*max_m + slot``, where
    ``slot`` is token ``t``'s within-expert position in token-major route order
    (matching how the route-gather fills each expert). Returns (token_num, topk)
    int32.

    Argsort-free: the within-expert ``slot`` is a one-hot cumsum (running count
    per expert in route order). Build this once and share it with the
    route-gather (scatter-copy) step instead of recomputing.
    """
    import torch.nn.functional as F

    token_num, topk = topk_ids.shape
    flat_e = topk_ids.reshape(-1).to(torch.long)
    # slot[r] = (# earlier routes to the same expert) = running count - 1
    slot = F.one_hot(flat_e, E).cumsum(0).gather(1, flat_e[:, None]).squeeze(1) - 1
    return (flat_e * max_m + slot).view(token_num, topk).to(torch.int32)


@functools.cache
def _get_compiled_route_maps():
    """Compile and cache the atomic route -> grouped-row map kernel."""
    from aiter.ops.flydsl.kernels.moe_route_maps import build_moe_route_maps_module

    return build_moe_route_maps_module()


@functools.cache
def _get_compiled_contiguous_psum():
    """Compile and cache the contiguous M-tile prefix-sum kernel."""
    from aiter.ops.flydsl.kernels.moe_contiguous_psum import (
        build_moe_contiguous_psum_module,
    )

    return build_moe_contiguous_psum_module()


def contiguous_psum(masked_m: torch.Tensor, experts: int, tile_m: int):
    """Tile-aligned exclusive prefix sum over per-expert counts in one FlyDSL
    kernel (no torch.cumsum / rocprim scan / D2D temp copy).

    Returns ``(starts, psum, contiguous_m_t)``:
      starts        : (E,) int32  exclusive prefix sum of ceil-aligned counts
      psum          : (E,) int32  starts[e] + masked_m[e] (actual ends)
      contiguous_m_t: (1,) int32  max(tile_m, sum(aligned)); read ``.item()``
                      once on the host for the grouped-buffer shape.
    """
    device = masked_m.device
    experts = int(experts)
    masked_m_i32 = masked_m[:experts].to(torch.int32)
    starts = torch.empty(experts, dtype=torch.int32, device=device)
    psum = torch.empty(experts, dtype=torch.int32, device=device)
    contiguous_m_t = torch.empty(1, dtype=torch.int32, device=device)
    launch = _get_compiled_contiguous_psum()
    launch(
        masked_m_i32,
        starts,
        psum,
        contiguous_m_t,
        experts,
        int(tile_m),
        stream=torch.cuda.current_stream(),
    )
    return starts, psum, contiguous_m_t


def build_route_maps(topk_ids: torch.Tensor, E: int, max_m: int):
    """Per-token route maps via a single atomic-scatter kernel (SGLang-style),
    no host-side argsort / nonzero / one-hot. Returns
    ``(topids_to_rows, rows_to_tokens, masked_m)``:

      topids_to_rows : (token_num, topk) int32  -- route -> grouped row
                 = ``topk_ids[t,k]*max_m + slot`` (gather-reduce input)
      rows_to_tokens  : (E*max_m,)        int32  -- grouped row -> source token
                 (-1 for unused padding rows; scatter-copy input)
      masked_m        : (E,)              int32  -- rows routed to each expert
                 (== bincount(topk_ids), the per-expert GEMM mask)

    The within-expert ``slot`` is claimed by ``atomicAdd(1)`` on a per-expert
    counter initialized to 0; the kernel forms the grouped row in-place as
    ``slot + e*max_m`` (one int mul-add per thread, hidden behind the atomic).
    It writes both maps in one pass (topids_to_rows + its inverse
    rows_to_tokens), and the final counter value is exactly ``counts[e]`` -- so
    ``masked_m`` is the counter itself, no bincount and no host-side
    ``arange``/``clone``/``sub`` to build or strip an offset. Order within an
    expert is atomic-race order (nondeterministic) but self-consistent -- both
    maps come from the same run, and the grouped GEMM is order-agnostic per
    expert.
    """
    device = topk_ids.device
    token_num, topk = topk_ids.shape
    numel = token_num * topk
    topk_ids_i32 = topk_ids.reshape(-1).to(torch.int32).contiguous()
    # Per-expert counter starts at 0; the kernel applies the e*max_m offset, so
    # after the run this buffer holds counts[e] directly == masked_m.
    atomic_buffer = torch.zeros(E, dtype=torch.int32, device=device)
    topids_to_rows = torch.empty(numel, dtype=torch.int32, device=device)
    rows_to_tokens = torch.full((E * max_m,), -1, dtype=torch.int32, device=device)
    grid_blocks = (numel + 255) // 256
    launch = _get_compiled_route_maps()
    launch(
        topk_ids_i32,
        atomic_buffer,
        topids_to_rows,
        rows_to_tokens,
        numel,
        topk,
        max_m,
        grid_blocks,
        stream=torch.cuda.current_stream(),
    )
    # atomic_buffer[e] == counts[e] now; it is masked_m, no further math.
    masked_m = atomic_buffer
    return topids_to_rows.view(token_num, topk), rows_to_tokens, masked_m


def flydsl_moe_gather_reduce(
    grouped_out: torch.Tensor,  # (E, max_m, model_dim) bf16/f16
    topids_to_rows: torch.Tensor,  # (token_num, topk) int32 grouped flat rows
    gather_w: torch.Tensor,  # (token_num, topk) weight, bf16/f16 (== grouped_out dtype)
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-pass gather-reduce epilogue. Thin launcher over a *precomputed* gather
    map: ``out[t] = sum_k gather_w[t,k] * grouped_out_flat[topids_to_rows[t,k]]``.

    The caller builds ``topids_to_rows`` once (see ``build_topids_to_rows``,
    argsort-free) and may share it with the route-gather step; this wrapper does
    no host-side map building. ``grouped_out`` and ``gather_w`` must be bf16 or
    f16 (the kernel extends the weight to f32 internally for accumulation)."""
    E, max_m, model_dim = grouped_out.shape
    token_num, topk = topids_to_rows.shape
    device = grouped_out.device
    if grouped_out.dtype == torch.bfloat16:
        out_dtype = "bf16"
    elif grouped_out.dtype == torch.float16:
        out_dtype = "f16"
    else:
        raise ValueError(f"unsupported dtype {grouped_out.dtype}; need bf16/f16")

    # Caller passes topids_to_rows int32 and gather_w bf16/f16 (both contiguous).
    grouped_out_flat = grouped_out.contiguous().view(E * max_m, model_dim)
    if out is None:
        out = torch.empty(
            (token_num, model_dim), dtype=grouped_out.dtype, device=device
        )

    launch = _get_compiled_gather_reduce(model_dim, topk, out_dtype)
    launch(
        grouped_out_flat,
        topids_to_rows,
        gather_w,
        out,
        token_num,
        stream=torch.cuda.current_stream(),
    )
    return out


# ---------------------------------------------------------------------------
# MoE route-gather (scatter-copy) input layout
#
# Pre-stage1 step: copy each token's quantized payload (and per-token scale)
# from the flat per-token layout into the grouped per-expert layout::
#
#     for e in range(E):
#         toks = tokens routed to expert e        # n = counts[e]
#         grouped[e, :n] = a_payload[toks]
#
# ``flydsl_moe_scatter_copy_token`` does the heavy row copies in one kernel pass
# (one block per grouped row, gathered from its source token via a precomputed
# dst->src map) and fills route_tokens/route_weights with cheap host ops. The
# reference loop it is validated against lives in
# ``op_tests/test_moe_scatter_copy_token.py``.
# ---------------------------------------------------------------------------


@functools.cache
def _get_compiled_scatter_copy(row_bytes: int):
    """Compile and cache the one-pass row scatter-copy kernel (per row width)."""
    from aiter.ops.flydsl.kernels.moe_scatter_copy_token import (
        build_moe_scatter_copy_token_module,
    )

    return build_moe_scatter_copy_token_module(row_bytes)


def flydsl_moe_scatter_copy_token(
    a1_payload: torch.Tensor,  # (token_num, Wp) uint8
    a1_scale_token_u8: Optional[torch.Tensor],  # (token_num, Ws) uint8 or None
    rows_to_tokens: torch.Tensor,  # (E*max_m,) int32 grouped row -> token (-1 pad)
    E: int,
    max_m: int,
    grouped_a1: Optional[torch.Tensor] = None,  # (E, max_m, Wp) uint8 out
    a1_scale_raw: Optional[torch.Tensor] = None,  # (E, max_m, Ws) uint8 out
):
    """Copy each token's payload (and per-token scale) into the grouped layout,
    driven by ``rows_to_tokens`` (grouped row -> source token, -1 for padding)
    from ``build_route_maps``. Pure copy -- one kernel per tensor.

    route_tokens/route_weights are NOT produced here: they are needed only by the
    naive epilogue (built in that loop) and, for doweight_stage1, derived on
    demand by the caller from topk_weight + topids_to_rows.

    Output tensors may be passed in (the kernel writes only the mapped/valid
    rows, leaving any pre-existing padding untouched -- e.g. an a1_scale_raw
    pre-filled with 127). When omitted they are allocated zero-filled.

    Returns (grouped_a1, a1_scale_raw)."""
    device = a1_payload.device
    Wp = a1_payload.shape[1]
    num_dst = E * max_m

    if grouped_a1 is None:
        grouped_a1 = torch.zeros((E, max_m, Wp), dtype=torch.uint8, device=device)
    launch_p = _get_compiled_scatter_copy(Wp)
    launch_p(
        a1_payload.contiguous().view(-1, Wp),
        grouped_a1.view(num_dst, Wp),
        rows_to_tokens,
        num_dst,
        stream=torch.cuda.current_stream(),
    )

    if a1_scale_token_u8 is not None:
        Ws = a1_scale_token_u8.shape[1]
        if a1_scale_raw is None:
            a1_scale_raw = torch.zeros((E, max_m, Ws), dtype=torch.uint8, device=device)
        launch_s = _get_compiled_scatter_copy(Ws)
        launch_s(
            a1_scale_token_u8.contiguous().view(-1, Ws),
            a1_scale_raw.view(num_dst, Ws),
            rows_to_tokens,
            num_dst,
            stream=torch.cuda.current_stream(),
        )

    return grouped_a1, a1_scale_raw


@functools.cache
def _get_compiled_scatter_preshuffle_scale(
    row_bytes: int, wmma_rep: int, scale_k_per_tile: int, gather: bool = True
):
    """Compile and cache the WMMA-preshuffle scale kernel (with/without gather)."""
    from aiter.ops.flydsl.kernels.moe_scatter_copy_preshuffle_scale import (
        build_moe_scatter_copy_preshuffle_scale_module,
    )

    return build_moe_scatter_copy_preshuffle_scale_module(
        row_bytes, wmma_rep, scale_k_per_tile, gather=gather
    )


def flydsl_moe_scatter_preshuffle_scale(
    a1_scale_token_u8: torch.Tensor,  # (token_num, Ws) uint8
    rows_to_tokens: torch.Tensor,  # (E*max_m,) int32 grouped row -> token (-1 pad)
    E: int,
    max_m: int,
    *,
    wmma_rep: int,
    scale_k_per_tile: int,
    grouped_a1_scale: Optional[
        torch.Tensor
    ] = None,  # (E, max_m//wmma_rep, Ws*wmma_rep)
):
    """Route-gather each token's e8m0 scale row AND preshuffle it into the WMMA
    layout in a single kernel pass -- fusing ``flydsl_moe_scatter_copy_token``'s
    scale copy with ``_grouped_a8w4_preshuffle_e8m0_scale``.

    ``max_m`` must be a multiple of ``wmma_rep*16`` (the grouped path pads it to
    a multiple of ``warp_tile_m``). Padding rows (``rows_to_tokens == -1``) are
    left untouched -- the masked GEMM never reads them, matching the previous
    uninitialized ``a1_scale_raw`` behaviour. Returns ``grouped_a1_scale``."""
    device = a1_scale_token_u8.device
    Ws = a1_scale_token_u8.shape[1]
    rows_per_tile = wmma_rep * 16
    assert (
        max_m % rows_per_tile == 0
    ), f"max_m ({max_m}) must be a multiple of wmma_rep*16 ({rows_per_tile})"
    tiles_per_expert = max_m // rows_per_tile

    if grouped_a1_scale is None:
        grouped_a1_scale = torch.empty(
            (E, max_m // wmma_rep, Ws * wmma_rep), dtype=torch.uint8, device=device
        )

    launch = _get_compiled_scatter_preshuffle_scale(
        Ws, wmma_rep, scale_k_per_tile, True
    )
    launch(
        a1_scale_token_u8.contiguous().view(-1, Ws),
        grouped_a1_scale.view(E * (max_m // wmma_rep), Ws * wmma_rep),
        rows_to_tokens,
        max_m,
        E,
        tiles_per_expert,
        stream=torch.cuda.current_stream(),
    )
    return grouped_a1_scale


def flydsl_moe_preshuffle_scale(
    scale_grouped_u8: torch.Tensor,  # (E, max_m, Ws) or (E*max_m, Ws) uint8
    E: int,
    max_m: int,
    *,
    wmma_rep: int,
    scale_k_per_tile: int,
    out: Optional[torch.Tensor] = None,  # (E, max_m//wmma_rep, Ws*wmma_rep)
):
    """Preshuffle an already-grouped row-major e8m0 scale into the WMMA layout in
    one kernel pass -- the in-kernel equivalent of the torch
    ``_grouped_a8w4_preshuffle_e8m0_scale`` permute (used by stage2, where the
    scale is already grouped so no route-gather is needed). Returns ``out``."""
    device = scale_grouped_u8.device
    Ws = scale_grouped_u8.shape[-1]
    rows_per_tile = wmma_rep * 16
    assert (
        max_m % rows_per_tile == 0
    ), f"max_m ({max_m}) must be a multiple of wmma_rep*16 ({rows_per_tile})"
    tiles_per_expert = max_m // rows_per_tile

    if out is None:
        out = torch.empty(
            (E, max_m // wmma_rep, Ws * wmma_rep), dtype=torch.uint8, device=device
        )

    launch = _get_compiled_scatter_preshuffle_scale(
        Ws, wmma_rep, scale_k_per_tile, False
    )
    launch(
        scale_grouped_u8.contiguous().view(E * max_m, Ws),
        out.view(E * (max_m // wmma_rep), Ws * wmma_rep),
        max_m,
        E,
        tiles_per_expert,
        stream=torch.cuda.current_stream(),
    )
    return out
