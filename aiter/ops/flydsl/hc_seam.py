# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Finished-FP32-gate hyper-connection seam on gfx950."""

import math
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.hyper_connection_gated_residual.common import _build_deferred_reduce_gate
from .kernels.hyper_connection_gated_residual.k1 import (
    _build_combine_only,
    _build_down_deferred_partial_pipe,
    flydsl_k1_combine_norm_down,
)
from .kernels.hyper_connection_gated_residual.k2 import flydsl_up_gate_mix_norm
from .kernels.hyper_connection_gated_residual.op import (
    _merged_pad,
    _validate_w_up,
    merge_gr_two_stage_weight,
)
from .kernels.hyper_connection_gated_residual.tuned import (
    SEAM_SKINNY_KNOBS,
    seam_config,
    seam_default_tile,
    seam_deferred_default,
    seam_deferred_k1_config,
    seam_deferred_k1_legal,
    seam_deferred_k2_config,
    seam_deferred_structural_default,
    seam_geometry_legal,
    seam_heuristic_config,
    seam_k2_branch_config,
    seam_k2_default,
    seam_k2_legal,
    seam_padded_extent,
)
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "HcSeamPreparedWeights",
    "hc_gated_combine",
    "hc_seam",
    "hc_seam_prepare_weights",
    "hc_seam_prepared",
]

_HIDDEN = 10240
_LOWRANK = 320
_HC = 4


def _require_gfx950():
    # get_gfx_runtime() takes no device and is cached, so no device context applies.
    arch = get_gfx_runtime()
    if arch != "gfx950":
        raise NotImplementedError("hc_seam requires gfx950")
    return arch


def _check_tensors(inputs, ref_name, device):
    """Shared device/dtype/shape/contiguity checks; every failure is a ValueError."""
    for name, tensor, shape, dtypes in inputs:
        if tensor is None:
            raise ValueError(f"{name} must be present")
        if tensor.device != device:
            raise ValueError(f"{name} must be on the same device as {ref_name}")
        if tensor.dtype not in dtypes:
            raise ValueError(f"{name} must have dtype in {dtypes}")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if name != "gate_prev" and not tensor.is_contiguous():
            raise ValueError(f"{name} must be row-major contiguous")


def _validate_combine_inputs(y, residual, gate_prev):
    if residual.ndim != 2 or residual.shape[1] != 10240:
        raise ValueError("residual must have shape [M,10240]")
    rows = residual.shape[0]
    if not 1 <= rows <= 256:
        raise ValueError("M must be in 1–256")
    if residual.device.type != "cuda":
        raise ValueError("residual must be on a GPU")
    _require_gfx950()
    _check_tensors(
        (
            ("residual", residual, (rows, 10240), (torch.bfloat16,)),
            ("y", y, (rows, 2560), (torch.bfloat16,)),
            ("gate_prev", gate_prev, (rows, 4), (torch.float32,)),
        ),
        "residual",
        residual.device,
    )


def hc_gated_combine(y, residual, gate_prev):
    """One-launch BF16(FP32-FMA(gate_prev, y, residual)), without normalization.

    BF16 contiguous y [M,2560] and residual [M,10240], FP32 gate_prev [M,4]
    (strided views supported), gfx950, M=1–256. Inputs are not mutated.
    """
    _validate_combine_inputs(y, residual, gate_prev)
    with torch.cuda.device(residual.device):
        out = torch.empty_like(residual)
        launch = _build_combine_only(
            4,
            2560,
            finished_gate=True,
            inj_row_stride=gate_prev.stride(0),
            inj_col_stride=gate_prev.stride(1),
        )
        # A gate view need not have a stride-one axis, unlike the automatic
        # dynamic-layout tensor adaptor. Static DLPack marshalling needs no copy.
        # The token extent is dynamic so one compiled kernel serves every M.
        gate_arg = flyc.from_dlpack(gate_prev).mark_shape_dynamic(0)
        _run_compiled(
            launch,
            residual,
            y,
            gate_arg,
            out,
            fx.Stream(torch.cuda.current_stream()),
        )
        return out


@dataclass(frozen=True)
class HcSeamPreparedWeights:
    """Seam weights packed once by :func:`hc_seam_prepare_weights`.

    The tensors are read-only by contract: ``merged`` and ``w_up_interleaved`` are owned by this object,
    ``norm_w``, ``w_down``, ``w_up`` and ``w_inject`` alias the caller's tensors.
    """

    norm_w: torch.Tensor  # [10240], BF16 or FP32, as the kernels read it
    merged: torch.Tensor  # [n_pad,10240] BF16: w_down, then w_inject_next, then zeros
    w_up_interleaved: torch.Tensor  # [160,4,16,320] BF16, owned
    w_up: torch.Tensor  # [10240,320] BF16
    w_down: torch.Tensor  # [320,10240] BF16, without inject columns
    w_inject: torch.Tensor | None  # [4,10240] BF16, ordinary exact-X weights
    has_next_gate: bool
    device: torch.device

    def __post_init__(self):
        # Hand-built or dataclasses.replace'd objects reach the kernels unchecked
        # otherwise: the deferred path reads w_down/w_inject, the 4-launch path merged.
        if self.has_next_gate != (self.w_inject is not None):
            raise ValueError("has_next_gate must match the presence of w_inject")
        if tuple(self.w_up_interleaved.shape) != (160, 4, 16, 320):
            raise ValueError("w_up_interleaved must have shape [160,4,16,320]")
        if tuple(self.w_down.shape) != (_LOWRANK, _HIDDEN):
            raise ValueError("w_down must have shape [320,10240]")
        if self.w_inject is not None and tuple(self.w_inject.shape) != (_HC, _HIDDEN):
            raise ValueError("w_inject must have shape [4,10240]")
        rows = _LOWRANK + _HC * self.has_next_gate
        if (
            self.merged.ndim != 2
            or self.merged.shape[0] < rows
            or (self.merged.shape[1] != _HIDDEN)
        ):
            raise ValueError("merged must have shape [>=n_rows,10240]")
        for name in ("norm_w", "merged", "w_up_interleaved", "w_up", "w_down"):
            if getattr(self, name).device != self.device:
                raise ValueError(f"{name} must be on the prepared device")


def hc_seam_prepare_weights(norm_w, w_down, w_up, w_inject_next=None):
    """Validate and pack the seam weights once; reuse across calls and M.

    Shapes and dtypes as in :func:`hc_seam`. w_inject_next=None prepares a
    seam without a next gate. The seam kernels read the norm weight as given
    (BF16 or FP32), so no per-call conversion remains.
    """
    if w_down.device.type != "cuda":
        raise ValueError("weights must be on a GPU")
    _require_gfx950()
    _validate_w_up(w_up, _HIDDEN, _LOWRANK, w_down.device)
    inputs = [
        ("norm_w", norm_w, (_HIDDEN,), (torch.bfloat16, torch.float32)),
        ("w_down", w_down, (_LOWRANK, _HIDDEN), (torch.bfloat16,)),
    ]
    if w_inject_next is not None:
        inputs.append(
            ("w_inject_next", w_inject_next, (_HC, _HIDDEN), (torch.bfloat16,))
        )
    _check_tensors(inputs, "w_down", w_down.device)
    with torch.cuda.device(w_down.device):
        merged = merge_gr_two_stage_weight(w_down, w_inject_next, _HC)
    return HcSeamPreparedWeights(
        norm_w=norm_w,
        merged=merged,
        w_up=w_up,
        w_up_interleaved=w_up.view(4, 160, 16, 320).permute(1, 0, 2, 3).contiguous(),
        w_down=w_down,
        w_inject=w_inject_next,
        has_next_gate=w_inject_next is not None,
        device=w_down.device,
    )


def hc_seam_prepared(
    y,
    residual,
    gate_prev,
    prepared,
    eps,
    *,
    row_tile=None,
    k1_row_tile=None,
    k2_row_tile=None,
    k1_n_tile=None,
    k1_m_waves=None,
    k1_n_waves=None,
    k2_n_tile=None,
    k2_m_waves=None,
    k2_n_waves=None,
    padded_extent=None,
    fallback=None,
    split_k=None,
    deferred=False,
    k2_branch_mode=None,
    k2_slp=None,
    k1_a_batch=None,
    k1_grid_swap=None,
    k2_grid_swap=None,
    k1_block_k=None,
    k1_stages=None,
    k1_slp=None,
):
    """:func:`hc_seam` with weights packed by :func:`hc_seam_prepare_weights`.

    Same returns as hc_seam, with no per-call weight fill or copy. M <= 3 uses
    two skinny kernels plus launch-local counter zeroing. Admitted row ranges
    use three deferred launches; other rows use four pre-normalized launches.
    The next gate is produced iff ``prepared.has_next_gate``.
    Optional geometry overrides use None for tuned/heuristic selection:
    row_tile in {16,32,64} (shorthand for k1_row_tile and k2_row_tile, which
    may differ; the padded extent is a multiple of the larger), K1/K2 N tiles and M/N wave counts, padded_extent
    (a tile multiple >= M), fallback in {"splitk","decouple","skinny"},
    and split_k (a divisor of 160). Tiled overrides disable automatic skinny
    selection; fallback="skinny" forces its measured geometry through M=8.
    Overrides are checked before compilation.
    k2_branch_mode selects the LDS branchwise path or register-only combined
    and separate branch GEMMs; None picks per M (tuned.seam_k2_default: separate
    at small M, branchwise above the crossover; forced geometry uses
    branchwise); k2_slp selects LLVM SLP vectorization for combined/separate K2
    and is ignored by branchwise K2 and the skinny path. Up weights
    use one-time 16-channel panels for both register-only schedules.
    deferred=True selects three launches with a per-M K1 geometry,
    branch-aligned 256-channel chunks and cooperative exact-X gate programs.
    Deferred K1 overrides use k1_row_tile, k1_n_tile, k1_m_waves,
    k1_n_waves, k1_block_k in {64,128}, k1_stages in {2,3}, and k1_slp.
    Pipeline overrides require explicit deferred=True. K2 geometry and padded_extent overrides remain available.
    k1_a_batch in {0,1} issues all A-group loads before transforms;
    k1_grid_swap and k2_grid_swap in {0,1} permute workgroup axes only.
    Structural knobs retain the dispatched geometry and require explicit
    deferred=True.
    None (default) lets the selector decide: only when the three-launch path
    is chosen by default (tuned.seam_deferred_structural_default, on for
    64 <= M <= 128 with combine and next gate); every other call, including explicit deferred=True,
    resolves None to 0. Explicit 0/1 is always honored.
    To force the
    four-launch path in the deferred range, use fallback="splitk". A False
    deferred value allows default selection; True explicitly forces deferral.
    """
    if type(deferred) is not bool:
        raise ValueError("deferred must be an explicit boolean")
    for name, value in (
        ("row_tile", row_tile),
        ("k1_row_tile", k1_row_tile),
        ("k2_row_tile", k2_row_tile),
    ):
        if value is not None and value not in (16, 32, 64):
            raise ValueError(f"{name} must be 16, 32 or 64")
    if fallback not in (None, "splitk", "decouple", "skinny"):
        raise ValueError("fallback must be splitk, decouple or skinny")
    structural = (k1_a_batch, k1_grid_swap, k2_grid_swap)
    explicit_structural = [v for v in structural if v is not None]
    if any(type(v) is not int or v not in (0, 1) for v in explicit_structural):
        raise ValueError("structural knobs must be in {0,1}")
    if any(explicit_structural) and (deferred is not True or fallback is not None):
        raise ValueError(
            "structural knobs require explicit deferred=True without fallback"
        )
    pipeline_k1 = any(v is not None for v in (k1_block_k, k1_stages, k1_slp))
    if pipeline_k1 and (not deferred or fallback is not None):
        raise ValueError(
            "K1 pipeline overrides require explicit deferred=True without fallback"
        )
    if (y is None) != (gate_prev is None):
        raise ValueError("y and gate_prev must both be present or both None")
    if residual.ndim != 2 or residual.shape[1] != _HIDDEN:
        raise ValueError("residual must have shape [M,10240]")
    rows = residual.shape[0]
    if not 1 <= rows <= 256:
        raise ValueError("M must be in 1–256")
    if residual.device.type != "cuda":
        raise ValueError("residual must be on a GPU")
    if residual.device != prepared.device:
        raise ValueError("residual must be on the same device as the prepared weights")
    arch = _require_gfx950()
    # Hand-built weights (e.g. a caller-padded 336-row merged tensor) have no
    # tuned rows and would fail late with a generic geometry error.
    if prepared.merged.shape[0] != _merged_pad(_LOWRANK + _HC * prepared.has_next_gate):
        raise ValueError(
            "prepared.merged has an unsupported row count; construct the weights "
            "with hc_seam_prepare_weights"
        )
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    inputs = [("residual", residual, (rows, _HIDDEN), (torch.bfloat16,))]
    if y is not None:
        inputs.extend(
            [
                ("y", y, (rows, 2560), (torch.bfloat16,)),
                ("gate_prev", gate_prev, (rows, 4), (torch.float32,)),
            ]
        )
    _check_tensors(inputs, "residual", residual.device)
    with torch.cuda.device(residual.device):
        block_output = y
        gate = gate_prev
        # contiguous() does not canonicalize singleton row strides.
        if gate is not None and gate.stride() != (4, 1):
            gate = gate.clone(memory_format=torch.contiguous_format)
        default_deferred = False
        no_geometry_override = all(
            v is None
            for v in (
                row_tile,
                k1_row_tile,
                k2_row_tile,
                k1_n_tile,
                k1_m_waves,
                k1_n_waves,
                k2_n_tile,
                k2_m_waves,
                k2_n_waves,
                padded_extent,
                split_k,
                k2_branch_mode,
            )
        )
        # Explicit geometry/fallback controls keep the four-launch path. k2_slp is
        # not one of them: it only tunes the tiled K2 and never changes dispatch.
        if not deferred and fallback is None and no_geometry_override:
            deferred = seam_deferred_default(rows)
            default_deferred = deferred
        sel = (
            seam_deferred_structural_default(
                rows, y is not None, prepared.has_next_gate
            )
            if default_deferred
            else {"k1_a_batch": 0, "k1_grid_swap": 0, "k2_grid_swap": 0}
        )
        if k1_a_batch is None:
            k1_a_batch = sel["k1_a_batch"]
        if k1_grid_swap is None:
            k1_grid_swap = sel["k1_grid_swap"]
        if k2_grid_swap is None:
            k2_grid_swap = sel["k2_grid_swap"]
        skinny_default = (
            rows <= SEAM_SKINNY_KNOBS["threshold"]["default"]
            and fallback is None
            and not deferred
            and no_geometry_override
        )
        if fallback == "skinny" or skinny_default:
            from .kernels.hyper_connection_gated_residual.k1 import (
                flydsl_k1k2_skinny_decode,
            )
            from .kernels.hyper_connection_gated_residual.tuned import (
                seam_skinny_config,
            )

            cfg = seam_skinny_config(rows)
            if cfg is None or deferred:
                raise ValueError("skinny requires M <= 8 and pre-normalized semantics")
            return flydsl_k1k2_skinny_decode(
                residual,
                y,
                gate,
                prepared.norm_w,
                prepared.w_up,
                prepared.merged,
                _LOWRANK,
                _HC,
                eps,
                prepared.has_next_gate,
                stream=torch.cuda.current_stream(),
                finished_gate=True,
                **cfg,
            )
        # row_tile is shorthand for both stages; a stage-specific tile wins.
        bm1 = k1_row_tile if k1_row_tile is not None and not deferred else row_tile
        bm2 = k2_row_tile if k2_row_tile is not None else row_tile
        default_tile = seam_default_tile(rows)
        k1_kw = {}
        k2_kw = {}
        forced = (
            bm1 is not None or bm2 is not None or fallback in ("splitk", "decouple")
        )
        default_candidate = None
        if k2_branch_mode is None:
            if deferred and not forced and fallback is None:
                k2_branch_mode, default_candidate = seam_deferred_k2_config(rows)
            elif forced:
                k2_branch_mode = "branchwise"
            else:
                k2_branch_mode, default_candidate = seam_k2_default(rows)
        if forced:
            bm1 = bm1 if bm1 is not None else default_tile
            bm2 = bm2 if bm2 is not None else default_tile
            tile = max(bm1, bm2)
            padded = ((rows + tile - 1) // tile) * tile
            # A single M wave owns BM=16; larger tiles retain whole-atom waves.
            mw = k1_m_waves if k1_m_waves is not None else (2 if bm1 == 64 else 1)
            nw = k1_n_waves if k1_n_waves is not None else (1 if bm1 == 16 else 2)
            bn = (
                k1_n_tile
                if k1_n_tile is not None
                else (64 if prepared.merged.shape[0] == 320 else 96)
            )
            sk = split_k if split_k is not None else (32 if padded <= 64 else 20)
            if fallback == "decouple":
                sk = 1
            k1_kw = {
                "use_tuned": False,
                "block_m": bm1,
                "sk_block_m": bm1,
                "dn_block_m": bm1,
                "dn_block_n": bn,
                "dn_m_waves": mw,
                "dn_n_waves": nw,
                "m_waves": mw,
                "n_waves": nw,
                "block_k": 64,
                "stages": 2,
                "split_k": sk,
            }
            k2_kw = {
                "block_m": bm2,
                "block_n": (
                    k2_n_tile if k2_n_tile is not None else (32 if bm2 == 16 else 16)
                ),
                "m_waves": k2_m_waves if k2_m_waves is not None else 1,
                "n_waves": k2_n_waves if k2_n_waves is not None else 1,
            }
        else:
            tile = default_tile
            # Deferred K1/K2 never read a table row; only branchwise K2 needs
            # a geometry, and the heuristic one is always legal for it.
            config = (
                seam_heuristic_config(rows, prepared.merged.shape[0])
                if deferred
                else seam_config(
                    arch,
                    rows,
                    y is not None,
                    prepared.has_next_gate,
                    prepared.merged.shape[0],
                )
            )
            k1_kw, k2_kw = config["k1"], config["k2"]
            candidate = (
                dict(default_candidate)
                if default_candidate is not None
                else seam_k2_branch_config(rows, k2_branch_mode)
            )
            selected_slp = candidate.pop("slp", True)
            k2_kw.update(candidate)
            tile = seam_padded_extent(1, {"k1": k1_kw, "k2": k2_kw})
            for name, value in (
                ("dn_block_n", k1_n_tile),
                ("dn_m_waves", k1_m_waves),
                ("dn_n_waves", k1_n_waves),
                ("split_k", split_k),
            ):
                if value is not None:
                    k1_kw[name] = value
                    if name == "dn_m_waves":
                        k1_kw["m_waves"] = value
                    if name == "dn_n_waves":
                        k1_kw["n_waves"] = value
            for name, value in (
                ("block_n", k2_n_tile),
                ("m_waves", k2_m_waves),
                ("n_waves", k2_n_waves),
            ):
                if value is not None:
                    k2_kw[name] = value
            padded = ((rows + tile - 1) // tile) * tile
        if k2_grid_swap:
            k2_kw["grid_swap"] = k2_grid_swap
        if padded_extent is not None:
            if deferred:
                if (
                    not isinstance(padded_extent, int)
                    or padded_extent < rows
                    or padded_extent % 64
                ):
                    raise ValueError(
                        "padded_extent must be a multiple of 64 >= M on the deferred path"
                    )
            elif (
                not isinstance(padded_extent, int)
                or padded_extent < rows
                or padded_extent % tile
            ):
                raise ValueError(
                    "padded_extent must be a multiple of max(BM1, BM2) >= M"
                )
            padded = padded_extent
        if k2_branch_mode not in ("branchwise", "combined", "separate"):
            raise ValueError("k2_branch_mode must be branchwise, combined or separate")
        if k2_branch_mode == "combined" and k2_n_waves is None:
            k2_kw["n_waves"] = 1
        k2_kw["branch_mode"] = k2_branch_mode
        # Validate both stages atomically, before the prologue can launch.
        if deferred:
            deferred_k1 = dict(seam_deferred_k1_config(rows))
            for name, value in (
                ("block_m", k1_row_tile),
                ("block_n", k1_n_tile),
                ("m_waves", k1_m_waves),
                ("n_waves", k1_n_waves),
                ("block_k", k1_block_k),
                ("stages", k1_stages),
                ("slp", k1_slp),
            ):
                if value is not None:
                    deferred_k1[name] = value
            if k1_a_batch:
                deferred_k1["k1_a_batch"] = k1_a_batch
            if k1_grid_swap:
                deferred_k1["k1_grid_swap"] = k1_grid_swap
            # The deferred launches pad to 64; validate K2 at the extent it runs.
            # Every K1/K2 row tile (16/32/64) divides it.
            padded = padded_extent or ((rows + 63) // 64) * 64
            if not (
                seam_deferred_k1_legal(deferred_k1) and seam_k2_legal(k2_kw, padded)
            ):
                raise ValueError("illegal deferred seam geometry")
        elif not seam_geometry_legal(
            {"k1": k1_kw, "k2": k2_kw}, prepared.merged.shape[0], padded
        ):
            raise ValueError(
                "illegal seam geometry (MMA, async work, split-K or K2 vector-work divisibility)"
            )
        if deferred and any(v is not None for v in (split_k, fallback, row_tile)):
            raise ValueError(
                "deferred does not support split_k, fallback or shared row_tile overrides"
            )
        rrms = residual.new_empty((padded, 4), dtype=torch.float32)
        gate_next = (
            residual.new_empty((rows, 4), dtype=torch.float32)
            if prepared.has_next_gate
            else None
        )
        if deferred:
            r_new = torch.empty_like(residual) if y is not None else residual
            chunk_width = deferred_k1.get("chunk_width", 256)
            splits = 10240 // chunk_width
            partial = residual.new_empty((splits * padded, 320), dtype=torch.float32)
            stats = residual.new_empty((splits * rows,), dtype=torch.float32)
            packed = residual.new_empty((padded, 320))
            stream = fx.Stream(torch.cuda.current_stream())
            _run_compiled(
                _build_down_deferred_partial_pipe(
                    **deferred_k1,
                    has_combine=y is not None,
                    w_bf16=prepared.norm_w.dtype == torch.bfloat16,
                ),
                residual,
                block_output if y is not None else residual,
                gate if gate is not None else residual,
                prepared.norm_w,
                prepared.w_down,
                r_new,
                partial,
                stats,
                stream,
            )
            _run_compiled(
                _build_deferred_reduce_gate(
                    prepared.has_next_gate,
                    prepared.norm_w.dtype == torch.bfloat16,
                    eps,
                    chunk_width=chunk_width,
                ),
                partial,
                stats,
                r_new,
                prepared.norm_w,
                prepared.w_inject if prepared.has_next_gate else residual,
                packed,
                rrms,
                gate_next if gate_next is not None else rrms,
                stream,
            )
        else:
            r_new, packed = flydsl_k1_combine_norm_down(
                residual,
                block_output if y is not None else None,
                gate if y is not None else None,
                prepared.norm_w,
                prepared.merged,
                _LOWRANK,
                _HC,
                eps,
                rrms_out=rrms,
                gemm_pad=padded,
                finished_gate=True,
                gate_out=gate_next,
                **k1_kw,
            )
        mixed = flydsl_up_gate_mix_norm(
            packed[:, :_LOWRANK],
            # K2 bounds its residual reads, stores and grid by the real rows.
            r_new[:rows],
            rrms,
            prepared.norm_w,
            (
                prepared.w_up
                if k2_branch_mode == "branchwise"
                else prepared.w_up_interleaved
            ),
            _HC,
            slp=(
                k2_slp if k2_slp is not None else (selected_slp if not forced else True)
            ),
            bounded_rows=True,
            **k2_kw,
        )
        return (
            r_new[:rows] if y is not None else residual,
            mixed[:rows],
            gate_next[:rows] if gate_next is not None else None,
        )


def hc_seam(y, residual, gate_prev, norm_w, w_down, w_up, w_inject_next, eps):
    """Combine, normalize and mix with finished gates.

    BF16 row-major inputs: residual [M,10240], y [M,2560], norm_w [10240],
    w_down [320,10240], w_up [10240,320], w_inject_next [4,10240]. Norm may
    also be FP32. gate_prev is FP32 [M,4], including strided views. M is 1–256.
    y and gate_prev must both be present or both None. No combine returns the
    original residual object without a copy, including non-aligned rows.
    No w_inject_next returns no next gate.
    Returns (residual_new BF16, mixed BF16 [M,2560], gate_next FP32 [M,4] | None).
    Packs the weights on every call (about 14 us of fill and copy), so it is
    slow in a loop: prepare once with hc_seam_prepare_weights and call
    hc_seam_prepared.
    """
    # Validate residual first so its errors precede weight-side ones.
    if residual.device.type != "cuda":
        raise ValueError("residual must be on a GPU")
    prepared = hc_seam_prepare_weights(norm_w, w_down, w_up, w_inject_next)
    return hc_seam_prepared(y, residual, gate_prev, prepared, eps)
