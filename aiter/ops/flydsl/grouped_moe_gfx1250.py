# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 grouped MoE GEMM (a8w4 default / a4w4).

This module owns the FlyDSL grouped-GEMM path so the generic ``fused_moe``
dispatcher does not carry gfx1250-specific implementation details.
"""

import os
import csv

from typing import Optional

import torch

from aiter import ActivationType, QuantType, dtypes, logger
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode

# Opt-in switch for the gfx1250 FlyDSL grouped-GEMM path.
_TRUTHY_ENV = ("1", "true", "True", "yes", "YES")
_GROUPED_CONFIG_CACHE = {}


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


def _is_stream_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


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


# ---------------------------------------------------------------------------
# Weight (B) scale preshuffle -- "N4K8" layout (gfx1250 grouped GEMM).
#
# The weight scale folds a 64-row x 256-K super-block (4 WMMA N-tiles x 8 e8m0
# blocks = 2 WMMA-K steps) into one row so that each lane reads its complete
# WMMA scaleB operand (the 4 e8m0 blocks = one WMMA-K=128 step) with a single
# contiguous ds_load_b32, and op_sel selects the lane-half (see the WMMA scale
# contract in ffm/FlyDSL/tests/kernels/test_wmma_scale_sample.py):
#
#   view(E, N//64, 4, 16, k_scale//8, 2, 4)
#     .permute(0, 1, 4, 2, 5, 3, 6)        # [E, remain_n, remain_b, p, q, lane, r]
#     .reshape(E, N//64, k_scale*64)
#
# For a raw byte at (n, ks) (n in [0,N), ks in [0,k_scale)=[0,K//32)):
#   n  = super_row*64 + p*16 + lane      # p = (n//16)%4,  lane = n%16
#   ks = remain_b*8   + q*4 + r          # q = (ks//4)%2,  r    = ks%4
#   super_row = n // 64
#   col = remain_b*512 + p*128 + q*64 + lane*4 + r
#
# The 4 e8m0 blocks (r=0..3) of one WMMA-K step land CONTIGUOUS (one i32);
# p (N-tile) has stride 128, q (the 2nd WMMA-K step in the block) stride 64.
# A full ``remain_b`` block is 512 contiguous cols = 8 e8m0 = 256 K = one k-tile
# (tile_k=256), so the consumer descriptor loads it as one contiguous tile.
# ---------------------------------------------------------------------------
def _grouped_b_scale_preshuffle_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Preshuffle a raw per-expert weight e8m0 scale into the N4K8 B layout."""
    scale = scale.view(torch.uint8).contiguous()
    E, N, k_scale = scale.shape
    if N % 64 != 0:
        raise ValueError(f"B-scale rows must be divisible by 64, got {N}")
    if k_scale % 8 != 0:
        raise ValueError(
            f"B-scale k columns (K//32) must be divisible by 8 (K%256==0), "
            f"got {k_scale}"
        )
    remain_n = N // 64
    remain_b = k_scale // 8
    g = scale.view(E, remain_n, 4, 16, remain_b, 2, 4)
    g = g.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    return g.reshape(E, remain_n, k_scale * 64)


def _grouped_b_scale_prepare_batch(
    scale: torch.Tensor,
    *,
    experts: int,
    rows: int,
    k_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Accept raw / flat-raw / already-preshuffled weight scale -> new B layout."""
    scale_u8 = scale.view(torch.uint8).contiguous()
    k_scale = k_dim // 32
    raw_shape = (experts, rows, k_scale)
    preshuffled_shape = (experts, rows // 64, k_scale * 64)
    if tuple(scale_u8.shape) == preshuffled_shape:
        return scale_u8.to(device=device)
    if tuple(scale_u8.shape) == (experts * rows, k_scale):
        scale_u8 = scale_u8.view(raw_shape)
    elif tuple(scale_u8.shape) != raw_shape:
        raise ValueError(
            f"B-scale shape must be raw {raw_shape}, flat raw "
            f"{(experts * rows, k_scale)} or preshuffled {preshuffled_shape}, "
            f"got {tuple(scale_u8.shape)}"
        )
    return _grouped_b_scale_preshuffle_e8m0(scale_u8).to(device=device)


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
            _GroupedA8W4Config,
            _make_m_tile_prefix_map,
            compile_moe_grouped_gemm1_a8w4_masked,
            compile_moe_grouped_gemm2_a8w4_masked,
            compile_moe_grouped_gemm1_mxfp4_masked,
            compile_moe_grouped_gemm2_mxfp4_masked,
        )
    except Exception as vendored_exc:
        try:
            from kernels.moe_grouped_gemm_mxscale_gfx1250 import (
                _GroupedA8W4Config,
                _make_m_tile_prefix_map,
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
    grouped_persistent_m = False
    persistent_workers = None
    cfg_row = _find_grouped_config(
        token_num=token_num,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        activation=activation,
        dtype=dtype,
        q_dtype_a=q_dtype_a,
        q_dtype_w=q_dtype_w,
        quant_type=quant_type,
        gate_mode=gate_mode,
    )
    if cfg_row is not None:
        tile_m = _as_int(cfg_row.get("tile_m"), tile_m)
        n_warp = _as_int(cfg_row.get("n_warp"), n_warp)
        num_buffers = _as_int(cfg_row.get("num_buffers"), num_buffers)
        split_k1 = _as_int(cfg_row.get("split_k1"), split_k1)
        split_k2 = _as_int(cfg_row.get("split_k2"), split_k2)
        grouped_persistent_m = _as_bool(
            cfg_row.get("grouped_persistent_m"), grouped_persistent_m
        )
        persistent_workers = _as_int(cfg_row.get("persistent_workers"), None)
        stage1_weight_layout = (
            cfg_row.get("stage1_weight_layout") or stage1_weight_layout
        )
        _grouped_dbg(f"using grouped CSV config: {cfg_row}")
    tile_n = int(n_warp) * 64
    tile_k = 256
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp

    # topk_ids is already an integer tensor; keep one flattened view for routing.
    flat_experts = topk_ids.reshape(-1)
    _capturing = _is_stream_capturing()
    # Expert-id range validation is a debug-only safety check: at decode sizes it
    # issues ~6 tiny launches/iter (lt+ge compare_scalar, two any() reductions)
    # plus a device->host sync from the `or` -- a real hotspot relative to the
    # tiny grouped work. Gate it behind AITER_GROUPED_DEBUG so production skips it
    # (topk_ids is already produced in-range by the router); set the env to 1 to
    # re-enable the check when diagnosing bad route ids. Skip entirely during
    # CUDAGraph capture (dynamic control flow / sync).
    if not _capturing and os.environ.get("AITER_GROUPED_DEBUG", "0") not in (
        "",
        "0",
        "false",
        "False",
    ):
        if torch.any(flat_experts < 0) or torch.any(flat_experts >= E):
            raise ValueError("grouped a8w4 path expects local expert ids in [0, E)")
    # Default max_m comes directly from token_num. Exact per-call max_m requires
    # a device->host sync (`counts.max().item()`), so keep it behind an opt-in
    # switch for benchmarking/tuning.
    counts = None
    use_actual_max_m = (not _capturing) and os.environ.get(
        "AITER_GROUPED_USE_ACTUAL_MAX_M", "0"
    ) in _TRUTHY_ENV
    if use_actual_max_m:
        counts = torch.bincount(flat_experts.to(torch.long), minlength=E)
        raw_max_m = int(counts.max().item()) if counts.numel() else 0
    else:
        # Static upper bound: each token routes at most one row per expert, so no
        # expert can receive more than token_num rows. CUDAGraph buckets have
        # static token_num/topk; per-expert count <= token_num*topk.
        raw_max_m = token_num
    max_m = max(
        warp_tile_m, ((raw_max_m + warp_tile_m - 1) // warp_tile_m) * warp_tile_m
    )
    _grouped_dbg(f"routing max_m={max_m} actual={use_actual_max_m}")

    # Stage1 input prep. The fast (non-naive) path is the fused warp-per-route
    # kernel: build_route_maps + per_1x32 MX quant + scatter_copy + scale
    # preshuffle in a single launch (fp4 a4w4 / fp8 a8w4, quant_mode from
    # data_format). It is the *sole* router on this path, so its
    # topids_to_rows/masked_m feed both the GEMM and the gather-reduce. The naive
    # path (AITER_GROUPED_GEMM_NAIVE=1) keeps a deterministic torch fallback for
    # tests/debug and is the only path supporting doweight_stage1 / non-bf16.
    _use_naive = os.environ.get("AITER_GROUPED_GEMM_NAIVE", "0") == "1"
    fused_quant_mode = "fp4" if data_format == "fp4" else "fp8"
    # Both branches produce grouped_a1 / grouped_a1_scale / masked_m /
    # topids_to_rows. The naive path additionally builds counts / route_tokens /
    # route_weights for its torch epilogues.
    grouped_a1 = None
    grouped_a1_scale = None

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

    if _use_naive:
        # Deterministic torch fallback: route maps + per-token MX quant +
        # route-gather + scale preshuffle, all in torch. Also the only path
        # supporting doweight_stage1 / non-bf16 hidden.
        if counts is None:
            counts = torch.bincount(flat_experts.to(torch.long), minlength=E)
        topids_to_rows, rows_to_tokens, masked_m = _build_route_maps_naive(
            topk_ids, E, max_m
        )
        route_tokens = rows_to_tokens.view(E, max_m).to(torch.long)

        # Per-token MX quant (fp4 torch ref / a8w4 MXFP8).
        if data_format == "fp4":
            from aiter.ops.quant import per_1x32_f4_quant

            _grouped_dbg("start a1 fp4 quant (naive)")
            a1_quant, a1_scale_token = per_1x32_f4_quant(
                hidden_states, quant_dtype=dtypes.fp4x2, shuffle=False
            )
            _grouped_dbg("a1 fp4 quant done")
            a1_payload = a1_quant.view(torch.uint8).contiguous()
            a1_scale_token_u8 = a1_scale_token.view(torch.uint8).contiguous()
            grouped_a1 = torch.empty(
                (E, max_m, model_dim // 2), dtype=torch.uint8, device=device
            )
        else:
            a1_payload, a1_scale_token_u8 = _quantize_mxfp8_payload(
                hidden_states, model_dim
            )
            grouped_a1 = torch.empty(
                (E, max_m, model_dim), dtype=torch.uint8, device=device
            )
        # Padding rows decode with scale=1.0.
        a1_scale_raw = torch.empty(
            (E, max_m, model_dim // 32), dtype=torch.uint8, device=device
        )

        # Route-gather + scale preshuffle (torch).
        _grouped_dbg("start route gather (naive)")
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
    else:
        # Fast path: one fused warp-per-route kernel does build_route_maps +
        # per_1x32 MX quant + scatter_copy + scale preshuffle (fp4 a4w4 / fp8
        # a8w4). It is the *sole* router, feeding both the GEMM and the
        # gather-reduce. It supports only what the warp-per-route quant expresses
        # today; fall back to the naive path (AITER_GROUPED_GEMM_NAIVE=1)
        # otherwise.
        if doweight_stage1:
            raise NotImplementedError(
                "doweight_stage1 is only supported on the grouped NAIVE path; "
                "set AITER_GROUPED_GEMM_NAIVE=1"
            )
        if data_format not in ("fp4", "a8w4"):
            raise NotImplementedError(
                f"fused grouped stage1 prep: unsupported data_format "
                f"{data_format!r} (expected 'fp4' or 'a8w4')"
            )
        if hidden_states.dtype != dtypes.bf16:
            raise NotImplementedError(
                "fused grouped stage1 prep requires bf16 hidden_states "
                f"(got {hidden_states.dtype}); set AITER_GROUPED_GEMM_NAIVE=1"
            )
        from aiter.ops.flydsl.moe_kernels import (
            flydsl_moe_fused_route_quant_scatter,
        )

        _grouped_dbg(f"start fused route+quant+scatter ({fused_quant_mode})")
        (
            grouped_a1,
            grouped_a1_scale,
            masked_m,
            topids_to_rows,
        ) = flydsl_moe_fused_route_quant_scatter(
            hidden_states,
            topk_ids,
            E,
            max_m,
            wmma_rep=warp_tile_m // 16,
            quant_mode=fused_quant_mode,
        )
        rows_to_tokens = None
        _grouped_dbg("fused route+quant+scatter done")

    out_dtype_str = "bf16" if dtype == dtypes.bf16 else "f16"
    m_tile_prefix = None
    m_tile_map = None
    # Persistent-M needs m_tile_prefix/m_tile_map. Skip both when disabled by
    # CSV/config, and force flat-grid launch during HIP CUDAGraph capture.
    effective_grouped_persistent_m = bool(grouped_persistent_m) and not _capturing
    if not _use_naive and effective_grouped_persistent_m:
        _m_tile_cfg = _GroupedA8W4Config(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=E,
            max_m=max_m,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            m_warp=m_warp,
            n_warp=n_warp,
            num_buffers=num_buffers,
            waves_per_eu=None,
            out_dtype=out_dtype_str,
            use_tdm_store=True,
            inst_prefetch=False,
            wave_specialized_tdm=False,
            split_k=1,
            cluster_m=1,
            cluster_n=1,
            use_scale_opsel=False,
            expert_sched_mode=False,
            grouped_persistent_m=effective_grouped_persistent_m,
            persistent_workers=persistent_workers,
            data_format=data_format,
            act="swiglu" if activation == ActivationType.Swiglu else "silu",
            stage1_weight_layout=stage1_weight_layout,
        )
        m_tile_prefix, m_tile_map = _make_m_tile_prefix_map(masked_m, _m_tile_cfg)

    grouped_w1 = (w1 if w1.dtype == torch.uint8 else w1.view(torch.uint8)).contiguous()
    grouped_w2 = (w2 if w2.dtype == torch.uint8 else w2.view(torch.uint8)).contiguous()
    _grouped_dbg("weight layout done")
    # Weight scales are already preshuffled per expert (N4K8 B-scale layout:
    # rows N -> N//64 super-rows, k_scale cols -> k_scale*64 folded cols; see
    # _grouped_b_scale_preshuffle_e8m0).
    grouped_w1_scale = w1_scale.reshape(
        E, (2 * inter_dim) // 64, (model_dim // 32) * 64
    )
    grouped_w2_scale = w2_scale.reshape(E, model_dim // 64, (inter_dim // 32) * 64)

    # grouped_a1_scale is produced above: fused gather+preshuffle (fast path) or
    # scatter + _grouped_a8w4_preshuffle_e8m0_scale (naive path).
    _grouped_dbg("scale layout done")

    grouped_a2 = torch.empty((E, max_m, inter_dim), dtype=dtype, device=device)
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
        grouped_persistent_m=effective_grouped_persistent_m,
        persistent_workers=persistent_workers,
        act="swiglu" if activation == ActivationType.Swiglu else "silu",
        stage1_weight_layout=stage1_weight_layout,
    )
    _grouped_dbg("stage1 compile done; start launch")
    _bias1_arg = bias1 if (bias1 is not None and bias1.numel() > 0) else None
    if _bias1_arg is not None and _bias1_arg.dtype != dtype:
        _bias1_arg = _bias1_arg.to(dtype)
    if not _capturing:
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
    if not _capturing:
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
                f"[dump] grouped_a2 (num_token, topk, inter_dim)="
                f"{tuple(_a2_tt.shape)}",
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

    if _use_naive:
        # Deterministic torch fallback: per-row MX quant + torch preshuffle.
        if data_format == "fp4":
            from aiter.ops.quant import per_1x32_f4_quant as _a2_f4_quant

            _grouped_dbg("start a2 fp4 quant (naive)")
            a2_quant, a2_scale_token = _a2_f4_quant(
                grouped_a2.view(E * max_m, inter_dim),
                quant_dtype=dtypes.fp4x2,
                shuffle=False,
            )
            _grouped_dbg("a2 fp4 quant done")
            grouped_a2_payload = (
                a2_quant.view(torch.uint8).contiguous().view(E, max_m, inter_dim // 2)
            )
            a2_scale_raw = (
                a2_scale_token.view(torch.uint8)
                .contiguous()
                .view(E, max_m, inter_dim // 32)
            )
        else:
            # a8w4 stage2 input also needs per-block-32 MXFP8 scale; SiLU outputs
            # can exceed unit-scale fp8 and direct casts may encode NaNs.
            grouped_a2_payload, a2_scale_raw = _quantize_mxfp8_payload(
                grouped_a2, inter_dim
            )
        grouped_a2_scale = _grouped_a8w4_preshuffle_e8m0_scale(
            a2_scale_raw, warp_tile=warp_tile_m, scale_k_per_tile=tile_k // 32
        )
    else:
        # Fast path: one fused kernel does the grouped MX quant AND writes the
        # preshuffled e8m0 scale (the stage2 analog of the fused stage1 input
        # prep). grouped_a2 is already grouped row-major, so no route-gather.
        from aiter.ops.flydsl.moe_kernels import flydsl_moe_fused_quant_preshuffle

        _grouped_dbg("start a2 fused quant+preshuffle")
        grouped_a2_payload, grouped_a2_scale = flydsl_moe_fused_quant_preshuffle(
            grouped_a2,
            E,
            max_m,
            wmma_rep=warp_tile_m // 16,
            quant_mode=fused_quant_mode,
        )
        if not _capturing:
            torch.cuda.synchronize()
        _grouped_dbg("[crash-probe] after a2 fused quant+preshuffle sync OK")
    _grouped_dbg("a2 scale layout done")
    grouped_out = torch.empty((E, max_m, model_dim), dtype=dtype, device=device)
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
        grouped_persistent_m=effective_grouped_persistent_m,
        persistent_workers=persistent_workers,
    )
    _grouped_dbg("stage2 compile done; start launch")
    _bias2_arg = bias2 if (bias2 is not None and bias2.numel() > 0) else None
    if _bias2_arg is not None and _bias2_arg.dtype != dtype:
        _bias2_arg = _bias2_arg.to(dtype)
    if not _capturing:
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
    if not _capturing:
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
        from aiter.ops.flydsl.moe_kernels import flydsl_moe_gather_reduce

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
