# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 grouped MoE GEMM (a8w4 default / a4w4).

Self-contained FlyDSL grouped-GEMM path, decoupled from ``fused_moe.py``.
``fused_moe`` only calls ``_maybe_grouped_gfx1250_a8w4_moe`` (plus the small
``_is_gfx1250`` dispatch helpers); everything else here is private to the
grouped path. See the entry function's docstring for the phase layout.
"""

import os

import torch

from dataclasses import dataclass
from typing import Optional

from aiter import ActivationType, QuantType, dtypes, logger
from aiter.ops.flydsl.moe_common import GateMode
from aiter.jit.utils.chip_info import get_cu_num, get_gfx

# Opt-in switch for the gfx1250 FlyDSL MoE grouped-GEMM path. When unset (or "0"),
# the dispatcher skips ``_maybe_grouped_gfx1250_a8w4_moe`` entirely and falls back
# to the default 2-stage flow. Set ``AITER_USE_GROUPED_GEMM=1`` to enable the
# grouped-GEMM mode (still subject to the other eligibility checks inside the
# helper, e.g. dtype / activation / gfx1250 dispatch / FlyDSL availability).
_TRUTHY_ENV = ("1", "true", "True", "yes", "YES")


def _use_grouped_gemm_enabled() -> bool:
    """Runtime check for AITER_USE_GROUPED_GEMM so tests can toggle it."""
    return os.environ.get("AITER_USE_GROUPED_GEMM", "1") in _TRUTHY_ENV


def _is_gfx1250() -> bool:
    """True when running on (or explicitly targeting) gfx1250."""
    if get_gfx() == "gfx1250":
        return True
    env = ";".join(
        str(os.environ.get(k, "")).lower()
        for k in ("GPU_ARCHS", "TARGET_ARCH", "AITER_GPU_ARCHS")
    )
    return "gfx1250" in env or "1" in env


def _is_gfx1250_a8w4_dispatch(q_type, q_dtype_a, q_dtype_w) -> bool:
    """gfx1250 FlyDSL a8w4 dispatch: per_1x32 mxfp8 activations x mxfp4 weights."""
    return (
        q_type == QuantType.per_1x32
        and q_dtype_w == dtypes.fp4x2
        and q_dtype_a == dtypes.fp8
    )


def _grouped_a8w4_preshuffle_e8m0_scale(
    scale: torch.Tensor,
    warp_tile: int,
    scale_k_per_tile: int = 4,
) -> torch.Tensor:
    scale = scale.view(torch.uint8).contiguous()
    _, k_scale = scale.shape
    wmma_rep = int(warp_tile) // 16
    k_groups = k_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // 4
    g = scale.view(-1, wmma_rep, 16, k_groups, k_wmma_steps, 4)
    g = g.permute(0, 2, 3, 4, 1, 5).contiguous()
    return g.reshape(-1, k_groups * k_wmma_steps * wmma_rep * 4)


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
    return torch.stack(
        [
            _grouped_a8w4_preshuffle_e8m0_scale(
                scale_u8[e], warp_tile=warp_tile, scale_k_per_tile=scale_k_per_tile
            )
            for e in range(experts)
        ]
    ).to(device=device)


# =====================================================================
# gfx1250 grouped MoE GEMM (a8w4 default / a4w4) -- self-contained block.
# Entry point: _maybe_grouped_gfx1250_a8w4_moe (called from fused_moe_).
# Tile schedule: GroupedTileConfig / pick_grouped_tile_config.
# Scale pre-shuffle: _grouped_a8w4_prepare_scale_batch (above).
# =====================================================================
def _grouped_dbg(msg: str) -> None:
    """Lean opt-in trace for the grouped gfx1250 path (AITER_GROUPED_DEBUG=1)."""
    if os.environ.get("AITER_GROUPED_DEBUG", "0") not in ("", "0", "false", "False"):
        print(f"[grouped-gemm-debug] {msg}", flush=True)


@dataclass(frozen=True)
class GroupedTileConfig:
    """Tile schedule for the gfx1250 grouped MoE GEMM.

    ``warp_tile_n = tile_n // n_warp`` and ``tile_k`` determine the weight
    scale pre-shuffle layout, so they are pinned (32 / 128) to stay compatible
    with the caller-provided pre-shuffled weight scales. ``tile_m`` and
    ``tile_n`` (with ``n_warp = tile_n // 32``) are the free axes.
    """

    tile_m: int
    tile_n: int
    tile_k: int
    m_warp: int
    n_warp: int
    num_buffers: int

    @property
    def warp_tile_m(self) -> int:
        return self.tile_m // self.m_warp

    @property
    def warp_tile_n(self) -> int:
        return self.tile_n // self.n_warp


# Temporary heuristic tile table for the grouped path, preferred-first.
# Each entry is (tile_m, tile_n, n_warp); all keep warp_tile_n == 32 so the
# pre-shuffled weight scale layout (wmma_rep=2) is preserved without a kernel
# change. Ordered by measured stage1 GEMM throughput on gfx1250 (E=256,
# K=7168, I=256): tile_m=64 is the dominant lever (2-3x over tile_m=16), and
# (64,128,4) is the fastest within-contract config (e.g. ~283us vs ~826us for
# (16,64,2) at max_m=256). Smaller tiles are kept for tiny workloads that
# can't saturate the CUs with a big tile.
# NOTE: tile_n=256 with n_warp=4 (warp_tile_n=64) measured slightly faster
# still, but needs weight scales pre-shuffled for wmma_rep=4 -- enable it once
# the scale contract / kernel is updated (Phase 2).
_GROUPED_TILE_CANDIDATES = (
    (64, 128, 4),
    (32, 128, 4),
    (32, 64, 2),
    (16, 64, 2),
)
_GROUPED_TILE_MIN = (16, 64, 2)
_GROUPED_TILE_K = 128


def pick_grouped_tile_config(
    max_m: int,
    stage_n: int,
    experts: int,
    *,
    cu_num: Optional[int] = None,
) -> GroupedTileConfig:
    """Pick a grouped GEMM tile config (temporary, occupancy-driven heuristic).

    Principle: keep every CU busy first, then prefer the largest tile. We walk
    candidates from largest to smallest and take the first whose estimated
    work-tile count (``experts * ceil(max_m/tile_m) * ceil(stage_n/tile_n)``)
    is at least ``cu_num``; if none saturate we fall back to the smallest tile
    so tiny workloads still spread across CUs.
    """
    if cu_num is None:
        cu_num = get_cu_num()
    chosen = _GROUPED_TILE_MIN
    for tile_m, tile_n, n_warp in _GROUPED_TILE_CANDIDATES:
        m_tiles = experts * ((max_m + tile_m - 1) // tile_m)
        n_tiles = (stage_n + tile_n - 1) // tile_n
        if m_tiles * n_tiles >= cu_num:
            chosen = (tile_m, tile_n, n_warp)
            break
    tile_m, tile_n, n_warp = chosen
    num_buffers = 2 if max_m <= 32 else 3
    return GroupedTileConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=_GROUPED_TILE_K,
        m_warp=1,
        n_warp=n_warp,
        num_buffers=num_buffers,
    )


def _grouped_mxfp8_block_quant(x: torch.Tensor, block: int = 32):
    """Per-block-32 MXFP8 quant (e4m3fn bytes + e8m0 scale) for the a8w4 path.

    Matches the FlyDSL UT ``_per_1x32_fp8_quant`` (scale uses fnuz max=240 but
    the byte encoding is fn via ``_f32_to_floatx_unpacked``). Returns
    ``(payload_uint8 [same shape as x], scale_e8m0_uint8 [..., K//block])``.
    Using a real block scale (instead of a fixed 1.0) is required for SiLU,
    whose unclamped stage-1 output otherwise overflows fp8 -> inf/nan.
    """
    from aiter.utility import fp4_utils as _aiter_fp4u

    DTYPE_MAX = 240.0
    shape = tuple(x.shape)
    K = shape[-1]
    flat = x.contiguous().view(-1, K).float()
    blk = torch.nan_to_num(flat.view(-1, block), nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = blk.abs().amax(dim=1)
    scale_e8m0 = _aiter_fp4u.f32_to_e8m0(max_abs / DTYPE_MAX)
    scale_f32 = _aiter_fp4u.e8m0_to_f32(scale_e8m0)
    scale_f32 = torch.nan_to_num(scale_f32, nan=1.0, posinf=1.0, neginf=1.0)
    scale_f32[scale_f32 == 0] = 1.0
    y_f32 = torch.clamp(blk / scale_f32.unsqueeze(1), min=-DTYPE_MAX, max=DTYPE_MAX)
    payload = (
        _aiter_fp4u._f32_to_floatx_unpacked(y_f32.contiguous().view(-1), 4, 3)
        .view(torch.uint8)
        .contiguous()
        .view(*shape)
    )
    scale = (
        scale_e8m0.view(*shape[:-1], K // block).view(torch.uint8).contiguous()
    )
    return payload, scale


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
    """gfx1250 grouped MoE (a8w4 default / a4w4).

    Phase layout (see inline ``[small-op]`` / ``[GEMM phase N]`` banners):
      eligibility + tile selection
      -> [small-op]   routing metadata
      -> [small-op]   stage-1 activation quant
      -> [small-op]   route scatter into per-expert buffers
      -> [small-op]   weight + scale layout
      -> [GEMM 1]     gate+up projection (+ activation)
      -> [small-op]   doweight_stage1 (optional)
      -> [small-op]   stage-2 activation quant + scale layout
      -> [GEMM 2]     down projection
      -> [small-op]   scatter back to source-token order

    Only the two ``[GEMM phase N]`` blocks run FlyDSL kernels; everything
    tagged ``[small-op]`` is host-driven pre/post-processing.
    """
    _grouped_dbg("enter grouped helper")
    # Master opt-in switch: only enter the gfx1250 FlyDSL grouped-GEMM mode
    # when AITER_USE_GROUPED_GEMM is explicitly enabled. The legacy
    # AITER_DISABLE_GROUPED_A8W4 kill-switch is still honoured for users who
    # want to force-disable the path even after opting in.
    if not _use_grouped_gemm_enabled():
        _grouped_dbg("AITER_USE_GROUPED_GEMM not enabled; skip grouped mode")
        return None
    if os.environ.get("AITER_DISABLE_GROUPED_A8W4", "0") == "1":
        _grouped_dbg("AITER_DISABLE_GROUPED_A8W4 enabled; skip grouped mode")
        return None
    if hidden_pad != 0 or intermediate_pad != 0:
        hidden_pad = 0
        intermediate_pad = 0
    if not isG1U1 or quant_type != QuantType.per_1x32:
        _grouped_dbg("not g1u1 or not 1x32")
        return None
    if activation not in (ActivationType.Silu, ActivationType.Swiglu):
        _grouped_dbg("not silu or not swiglu")
        return None
    if gate_mode not in (GateMode.SEPARATED, GateMode.INTERLEAVE):
        _grouped_dbg(f"unsupported gate_mode={gate_mode}")
        return None
    # Default: derive stage1_weight_layout from gate_mode
    #   SEPARATED -> GGUU, INTERLEAVE -> GUGU
    # ``AITER_GROUPED_STAGE1_WEIGHT_LAYOUT={gguu,gugu}`` lets callers override
    # this when the physical weight layout doesn't match the gate_mode
    # convention (e.g. running gpt-oss GUGU weights through a SEPARATED
    # dispatch path for diagnostics, or feeding GGUU weights into INTERLEAVE).
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
    # G/U mode dump: gate_mode (SEPARATED/INTERLEAVE) -> stage1 weight layout.
    # GUGU = interleaved [gate0,up0,gate1,up1,...]; GGUU = concatenated [gate|up].
    logger.info(
        "[MoE-GUMODE] gate_mode=%s -> stage1_weight_layout=%s (%s)",
        gate_mode.name,
        stage1_weight_layout,
        stage1_weight_layout.upper(),
    )
    # gfx1250 defaults to a8w4 (mxfp8 activations x mxfp4 weights); a4w4
    # (mxfp4 x mxfp4) stays available when both operands are fp4x2.
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
    if not _is_gfx1250():
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

    flat_experts = topk_ids.reshape(-1).to(torch.long)
    if torch.any(flat_experts < 0) or torch.any(flat_experts >= E):
        raise ValueError("grouped a8w4 path expects local expert ids in [0, E)")
    counts = torch.bincount(flat_experts, minlength=E)
    raw_max_m = int(counts.max().item()) if counts.numel() else 0

    # Temporary occupancy-driven tile selection (see pick_grouped_tile_config).
    # stage1 N = 2*inter_dim is the dominant GEMM width; warp_tile_n stays 32
    # so the pre-shuffled weight scales remain valid (no kernel change).
    tile_cfg = pick_grouped_tile_config(max(raw_max_m, 1), 2 * inter_dim, E)
    tile_m, tile_n, tile_k = tile_cfg.tile_m, tile_cfg.tile_n, tile_cfg.tile_k
    m_warp, n_warp = tile_cfg.m_warp, tile_cfg.n_warp
    warp_tile_m, warp_tile_n = tile_cfg.warp_tile_m, tile_cfg.warp_tile_n
    max_m = max(
        warp_tile_m, ((raw_max_m + warp_tile_m - 1) // warp_tile_m) * warp_tile_m
    )
    _grouped_dbg(f"routing counts done max_m={max_m} tile={tile_cfg}")

    # ------------------------------------------------------------------
    # [small-op] routing metadata: per-route token id / weight, masked_m.
    # ------------------------------------------------------------------
    flat_routes = torch.arange(token_num * topk, device=device, dtype=torch.long)
    flat_tokens = flat_routes // topk
    flat_weights = topk_weight.reshape(-1).to(dtype)
    route_tokens = torch.empty((E, max_m), dtype=torch.long, device=device)
    route_weights = torch.empty((E, max_m), dtype=dtype, device=device)
    masked_m = counts.to(torch.int32).to(device=device)

    # ------------------------------------------------------------------
    # [small-op] stage-1 activation quant (per-1x32 mxfp4 / mxfp8).
    # ------------------------------------------------------------------
    if data_format == "fp4":
        from aiter.ops.quant import per_1x32_f4_quant

        a1_quant, a1_scale_token = per_1x32_f4_quant(
            hidden_states, quant_dtype=dtypes.fp4x2, shuffle=False
        )
        a1_payload = a1_quant.view(torch.uint8).contiguous()
        a1_scale_token_u8 = a1_scale_token.view(torch.uint8).contiguous()
        grouped_a1 = torch.zeros(
            (E, max_m, model_dim // 2), dtype=torch.uint8, device=device
        )
        a1_scale_raw = torch.full(
            (E, max_m, model_dim // 32), 127, dtype=torch.uint8, device=device
        )
    else:
        # a8w4 stage1 input: per-block-32 mxfp8 quantization (matches the
        # FlyDSL UT _per_1x32_fp8_quant). A direct ``hidden_states.to(fp8)``
        # + scale=1.0 would discard the activation magnitude.
        a1_payload, a1_scale_token_u8 = _grouped_mxfp8_block_quant(
            hidden_states.contiguous().view(-1, model_dim)
        )
        grouped_a1 = torch.zeros(
            (E, max_m, model_dim), dtype=torch.uint8, device=device
        )
        # Initial fill with byte=127 (=2^0=1.0) so empty rows in inactive
        # experts decode to scale=1.0 and don't trip the kernel.
        a1_scale_raw = torch.full(
            (E, max_m, model_dim // 32), 127, dtype=torch.uint8, device=device
        )

    # ------------------------------------------------------------------
    # [small-op] scatter routed tokens into per-expert (E, max_m, K) buffers.
    # ------------------------------------------------------------------
    _grouped_dbg("start route gather")
    for e in range(E):
        mask = flat_experts == e
        n = int(counts[e].item())
        if n == 0:
            continue
        toks = flat_tokens[mask]
        grouped_a1[e, :n].copy_(a1_payload[toks])
        if a1_scale_token_u8 is not None:
            a1_scale_raw[e, :n].copy_(a1_scale_token_u8[toks])
        route_tokens[e, :n].copy_(toks)
        route_weights[e, :n].copy_(flat_weights[mask])
    _grouped_dbg("route gather done")

    # ------------------------------------------------------------------
    # [small-op] weight bytes + scale layout for the masked grouped kernel.
    # ------------------------------------------------------------------
    grouped_w1 = (w1 if w1.dtype == torch.uint8 else w1.view(torch.uint8)).contiguous()
    grouped_w2 = (w2 if w2.dtype == torch.uint8 else w2.view(torch.uint8)).contiguous()
    _grouped_dbg("weight layout done")
    # Reshape flat rank-2 scales to the rank-3 shape the masked grouped kernel
    # expects: (E, rows//wmma_rep, (k/32)*wmma_rep) with wmma_rep = warp_tile_n//16.
    #   stage1 (w1): (E, 2*inter_dim//wmma_rep, (model_dim/32)*wmma_rep)
    #   stage2 (w2): (E, model_dim//wmma_rep,   (inter_dim/32)*wmma_rep)
    _wmma_rep = warp_tile_n // 16
    grouped_w1_scale = w1_scale.reshape(
        E, (2 * inter_dim) // _wmma_rep, (model_dim // 32) * _wmma_rep
    )
    grouped_w2_scale = w2_scale.reshape(
        E, model_dim // _wmma_rep, (inter_dim // 32) * _wmma_rep
    )

    grouped_a1_scale = torch.stack(
        [
            _grouped_a8w4_preshuffle_e8m0_scale(
                a1_scale_raw[e], warp_tile=warp_tile_m, scale_k_per_tile=tile_k // 32
            )
            for e in range(E)
        ]
    )
    _grouped_dbg("scale layout done")

    # ==================================================================
    # [GEMM phase 1] gate+up projection with fused activation (silu/swiglu).
    # ==================================================================
    grouped_a2 = torch.zeros((E, max_m, inter_dim), dtype=dtype, device=device)
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
        num_buffers=tile_cfg.num_buffers,
        expert_sched_mode=False,
        act="swiglu" if activation == ActivationType.Swiglu else "silu",
        stage1_weight_layout=stage1_weight_layout,
    )
    _grouped_dbg("stage1 compile done; start launch")
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
    _grouped_dbg("stage1 launch done")
    # ------------------------------------------------------------------
    # [small-op] optional stage-1 weighting (doweight_stage1).
    # ------------------------------------------------------------------
    if doweight_stage1:
        for e in range(E):
            n = int(counts[e].item())
            if n:
                grouped_a2[e, :n].mul_(route_weights[e, :n].view(-1, 1))

    # ------------------------------------------------------------------
    # [small-op] stage-2 activation quant + scale layout.
    # ------------------------------------------------------------------
    if data_format == "fp4":
        from aiter.ops.quant import per_1x32_f4_quant

        a2_quant, a2_scale_token = per_1x32_f4_quant(
            grouped_a2.view(E * max_m, inter_dim),
            quant_dtype=dtypes.fp4x2,
            shuffle=False,
        )
        grouped_a2_payload = (
            a2_quant.view(torch.uint8).contiguous().view(E, max_m, inter_dim // 2)
        )
        a2_scale_raw = (
            a2_scale_token.view(torch.uint8)
            .contiguous()
            .view(E, max_m, inter_dim // 32)
        )
    else:
        # a8w4 stage2 input: per-block-32 mxfp8 quant with a real e8m0 block
        # scale. A fixed scale=1.0 (the old shortcut) overflows fp8 for SiLU's
        # unclamped stage-1 output -> inf/nan; SwiGLU happened to survive only
        # because of its clamp.
        grouped_a2_payload, a2_scale_raw = _grouped_mxfp8_block_quant(grouped_a2)
    grouped_a2_scale = torch.stack(
        [
            _grouped_a8w4_preshuffle_e8m0_scale(
                a2_scale_raw[e], warp_tile=warp_tile_m, scale_k_per_tile=tile_k // 32
            )
            for e in range(E)
        ]
    )
    _grouped_dbg("a2 scale layout done")
    # ==================================================================
    # [GEMM phase 2] down projection (atomic-accumulate into output).
    # ==================================================================
    grouped_out = torch.zeros((E, max_m, model_dim), dtype=dtype, device=device)
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
        num_buffers=tile_cfg.num_buffers,
        expert_sched_mode=False,
    )
    _grouped_dbg("stage2 compile done; start launch")
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
    _grouped_dbg("stage2 launch done")

    # ------------------------------------------------------------------
    # [small-op] scatter per-expert results back to source-token order.
    # ------------------------------------------------------------------
    moe_out = torch.zeros((token_num, model_dim), dtype=dtype, device=device)
    _grouped_dbg("start scatter output")
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

