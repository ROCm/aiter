# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 grouped MoE GEMM (a8w4 default / a4w4).

This module owns the FlyDSL grouped-GEMM path so the generic ``fused_moe``
dispatcher does not carry gfx1250-specific implementation details.
"""

import os

from typing import Optional

import torch

from aiter import ActivationType, QuantType, dtypes, logger
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode

# Opt-in switch for the gfx1250 FlyDSL grouped-GEMM path.
_TRUTHY_ENV = ("1", "true", "True", "yes", "YES")


def _use_grouped_gemm_enabled() -> bool:
    """Runtime check for AITER_USE_GROUPED_GEMM so tests can toggle it."""
    return os.environ.get("AITER_USE_GROUPED_GEMM", "1") in _TRUTHY_ENV



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
    src_tokens = (
        torch.arange(token_num, device=device, dtype=torch.int32)
        .repeat_interleave(topk)
    )
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
    # Log the stage1 gate/up layout used by the grouped kernel.
    logger.info(
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
    tile_m, tile_n, tile_k = 16, 256, 256
    m_warp, n_warp = 1, 4
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp

    # topk_ids is already an integer tensor; keep one flattened view for routing.
    flat_experts = topk_ids.reshape(-1)
    if torch.any(flat_experts < 0) or torch.any(flat_experts >= E):
        raise ValueError("grouped a8w4 path expects local expert ids in [0, E)")
    # Size each expert block for the actual busiest expert.
    counts = torch.bincount(flat_experts.to(torch.long), minlength=E)
    raw_max_m = int(counts.max().item()) if counts.numel() else 0
    max_m = raw_max_m
    max_m = max(warp_tile_m, ((max_m + warp_tile_m - 1) // warp_tile_m) * warp_tile_m)
    _grouped_dbg(f"routing counts done raw_max_m={raw_max_m} max_m={max_m}")

    # Build route maps once. The fast path uses the FlyDSL atomic-scatter kernel;
    # the naive path keeps a deterministic torch fallback for tests/debug.
    _use_naive = os.environ.get("AITER_GROUPED_GEMM_NAIVE", "0") == "1"
    # masked_m drives the GEMM; counts is only for fallback/debug code.
    if _use_naive:
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
        from aiter.ops.flydsl.moe_kernels import build_route_maps

        topids_to_rows, rows_to_tokens, masked_m = build_route_maps(
            topk_ids, E, max_m
        )
    # Grouped row -> source token, (E, max_m); padding rows (-1) are never read
    # because the naive epilogues are bounded by per-expert counts.

    if data_format == "fp4":
        # Use the reference MXFP4 quantization contract for a4w4 accuracy.
        from aiter.ops.quant import per_1x32_f4_quant as _a1_f4_quant

        _grouped_dbg("start a1 fp4 quant")
        a1_quant, a1_scale_token = _a1_f4_quant(
            hidden_states, quant_dtype=dtypes.fp4x2, shuffle=False
        )
        _grouped_dbg("a1 fp4 quant done")
        a1_payload = a1_quant.view(torch.uint8).contiguous()
        a1_scale_token_u8 = a1_scale_token.view(torch.uint8).contiguous()
        grouped_a1 = torch.empty(
            (E, max_m, model_dim // 2), dtype=torch.uint8, device=device
        )
        a1_scale_raw = torch.empty(
            (E, max_m, model_dim // 32), dtype=torch.uint8, device=device
        )
    else:
        # a8w4 stage1 input: per-block-32 MXFP8 quantization.
        from aiter.utility import fp4_utils as _aiter_fp4u

        BLOCK = 32
        DTYPE_MAX = 240.0
        a1_flat = hidden_states.contiguous().view(-1, model_dim).float()
        Mtok = a1_flat.shape[0]
        blk = a1_flat.view(-1, BLOCK)
        blk = torch.nan_to_num(blk, nan=0.0, posinf=0.0, neginf=0.0)
        max_abs = blk.abs().amax(dim=1)
        scale_e8m0 = _aiter_fp4u.f32_to_e8m0(max_abs / DTYPE_MAX)
        scale_f32 = _aiter_fp4u.e8m0_to_f32(scale_e8m0)
        scale_f32 = torch.nan_to_num(scale_f32, nan=1.0, posinf=1.0, neginf=1.0)
        scale_f32[scale_f32 == 0] = 1.0
        y_f32 = blk / scale_f32.unsqueeze(1)
        y_f32 = torch.clamp(y_f32, min=-DTYPE_MAX, max=DTYPE_MAX)
        a1_bytes = _aiter_fp4u._f32_to_floatx_unpacked(
            y_f32.contiguous().view(-1), 4, 3
        )
        a1_payload = a1_bytes.view(torch.uint8).contiguous().view(Mtok, model_dim)
        a1_scale_token_u8 = (
            scale_e8m0.view(Mtok, model_dim // BLOCK).view(torch.uint8).contiguous()
        )
        grouped_a1 = torch.empty(
            (E, max_m, model_dim), dtype=torch.uint8, device=device
        )
        # Padding rows decode with scale=1.0.
        a1_scale_raw = torch.empty(
            (E, max_m, model_dim // 32), dtype=torch.uint8, device=device
        )

    # Route-gather into the grouped per-expert layout.
    if not _use_naive:
        from aiter.ops.flydsl.moe_kernels import flydsl_moe_scatter_copy_token

        _grouped_dbg("start route gather (scatter-copy kernel)")
        flydsl_moe_scatter_copy_token(
            a1_payload,
            a1_scale_token_u8,
            rows_to_tokens,
            E,
            max_m,
            grouped_a1=grouped_a1,
            a1_scale_raw=a1_scale_raw,
        )
        _grouped_dbg("route gather done")
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
        route_weights.view(-1)[topids_to_rows.reshape(-1)] = (
            topk_weight.reshape(-1).to(route_weights.dtype)
        )
        _grouped_dbg("route gather done")

    grouped_w1 = (w1 if w1.dtype == torch.uint8 else w1.view(torch.uint8)).contiguous()
    grouped_w2 = (w2 if w2.dtype == torch.uint8 else w2.view(torch.uint8)).contiguous()
    _grouped_dbg("weight layout done")
    # Weight scales are already preshuffled per expert.
    _wmma_rep = warp_tile_n // 16
    grouped_w1_scale = w1_scale.reshape(
        E, (2 * inter_dim) // _wmma_rep, (model_dim // 32) * _wmma_rep
    )
    grouped_w2_scale = w2_scale.reshape(
        E, model_dim // _wmma_rep, (inter_dim // 32) * _wmma_rep
    )

    grouped_a1_scale = _grouped_a8w4_preshuffle_e8m0_scale(
        a1_scale_raw, warp_tile=warp_tile_m, scale_k_per_tile=tile_k // 32
    )
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
        out_dtype="bf16" if dtype == dtypes.bf16 else "f16",
        num_buffers=2,
        expert_sched_mode=False,
        act="swiglu" if activation == ActivationType.Swiglu else "silu",
        stage1_weight_layout=stage1_weight_layout,
    )
    _grouped_dbg("stage1 compile done; start launch")
    _bias1_arg = bias1 if (bias1 is not None and bias1.numel() > 0) else None
    if _bias1_arg is not None and _bias1_arg.dtype != dtype:
        _bias1_arg = _bias1_arg.to(dtype)
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
        bias=bias1.to(dtype) if bias1 is not None else None,
    )
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

    if data_format == "fp4":
        # Keep stage2 quantization on the same contract as stage1.
        from aiter.ops.quant import per_1x32_f4_quant as _a2_f4_quant

        _grouped_dbg("start a2 fp4 quant")
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
        torch.cuda.synchronize()
        _grouped_dbg("[crash-probe] after a2 fp4 quant sync OK")
    else:
        grouped_a2_payload = grouped_a2.to(dtypes.fp8).view(torch.uint8).contiguous()
        a2_scale_raw = torch.full(
            (E, max_m, inter_dim // 32), 127, dtype=torch.uint8, device=device
        )
    grouped_a2_scale = _grouped_a8w4_preshuffle_e8m0_scale(
        a2_scale_raw, warp_tile=warp_tile_m, scale_k_per_tile=tile_k // 32
    )
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
        out_dtype="bf16" if dtype == dtypes.bf16 else "f16",
        num_buffers=2,
        expert_sched_mode=False,
    )
    _grouped_dbg("stage2 compile done; start launch")
    _bias2_arg = bias2 if (bias2 is not None and bias2.numel() > 0) else None
    if _bias2_arg is not None and _bias2_arg.dtype != dtype:
        _bias2_arg = _bias2_arg.to(dtype)
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
        bias=bias2.to(dtype) if bias2 is not None else None,
    )
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
    logger.info(
        f"[{impl_name}] used grouped FlyDSL {data_format} path: tokens={token_num}, topk={topk}, E={E}, max_m={max_m}"
    )
    return moe_out

