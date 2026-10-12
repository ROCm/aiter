# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host wrapper for the FlyDSL delayed mHC seam with RMSNorm (gfx942), two launches.

Same arguments and returns as the Triton seam
``aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm``. A fused pass post-mixes the
streams, stages the un-normalised collapse in ``layer_input`` and publishes the split-K gate
projection and both sums of squares; a finish pass folds them, forms the gates and rescales
``layer_input`` in place. See ``aiter/ops/flydsl/kernels/mhc_seam.py`` for the kernels.
"""

import functools

import torch

__all__ = ["MHC_SEAM_DEFAULTS", "flydsl_mhc_seam", "get_mhc_seam_config"]

_GFX = ("gfx942",)
_N = 4
# Last-level (Infinity) cache, which decides whether the staged collapse stays resident.
_MALL_BYTES = 256 * 1024 * 1024
# Split-K of the fused pass: the largest legal split that keeps the grid within
# _SPLIT_TARGET_BLOCKS workgroups or _MAX_SPLITS splits, whichever allows more, preferring one
# whose last round of workgroups is at least _MIN_ROUND_FILL full -- a nearly empty last round
# costs most of a full one.
_MIN_ROUND_FILL = 0.73
_SPLIT_TARGET_BLOCKS = 512
_MAX_SPLITS = 8

# knob defaults merged under every policy decision and user override
MHC_SEAM_DEFAULTS = {
    "BLOCK_M": 16,  # tokens per fused-pass workgroup
    "CHUNK_C": 256,  # hidden columns per chunk of the fused pass's sweep
    "SPLITS": 1,  # hidden segments (split-K) of the fused pass
    "WAVES_PER_EU": 2,  # occupancy hint of the fused pass
    "FINISH_WAVES_PER_EU": 2,  # occupancy hint of the finish pass
    "FAST_FP_MATH": True,
    "FAST_NARROW": True,  # bf16 rounding in integer ops; required by FAST_FP_MATH
    # nt hint on the residual / sublayer_out loads; the two stores are always streamed
    "CACHE_NT": False,
    # nt hint on the staged collapse's store and its reload; None: _stage_nt
    "STAGE_NT": None,
    "GATE_STEP": None,  # finish pass gives the gates a column step; None: kernels.gate_step
}

# Launch table from a sweep on 304-CU gfx942 parts at H = 5120, one row per token bucket:
# (largest T, BLOCK_M, CHUNK_C, CACHE_NT). Past the last row its config holds. SPLITS is not
# tabled: get_mhc_seam_config derives it from the grid, which reproduces the sweep's choices up
# to 4096 tokens. CACHE_NT pays once the seam's traffic no longer fits the MALL.
_TABLE = (
    (1, 16, 128, False),
    (32, 16, 128, False),
    (128, 32, 128, False),
    (384, 32, 64, False),
    (512, 32, 64, False),
    (1024, 32, 64, False),
    (2048, 64, 64, False),
    (4096, 64, 64, False),
    (8192, 64, 64, True),
    (16384, 64, 64, True),
    (32768, 64, 64, True),
)
# (BLOCK_M, CHUNK_C) tried, in order, when a row's tile is illegal for H
_FALLBACK_TILES = tuple((bm, cc) for cc in (64, 128, 256) for bm in (32, 64, 16))


def _require(cond: bool, msg: str, exc: type = ValueError) -> None:
    """Input validation that survives ``python -O`` (unlike ``assert``)."""
    if not cond:
        raise exc(f"[flydsl_mhc_seam] {msg}")


@functools.lru_cache(maxsize=16)
def _arch(device_index: int) -> str:
    """Memoized: ``get_device_properties`` is a per-call host cost of tens of us."""
    return torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0]


@functools.lru_cache(maxsize=1)
def _cu_num() -> int:
    """Memoized: ``get_cu_num`` shells out to rocminfo; it honours the ``CU_NUM`` override."""
    from aiter.jit.utils.chip_info import get_cu_num

    return get_cu_num()


@functools.lru_cache(maxsize=16)
def _lds_bytes(arch: str) -> int:
    from aiter.jit.utils.chip_info import get_lds_capacity_bytes

    return get_lds_capacity_bytes(arch)


def _check_config(T: int, H: int, cfg: dict, arch: str) -> None:
    from aiter.ops.flydsl.kernels.mhc_seam import check_config

    try:
        check_config(
            T,
            H,
            _N,
            cfg["BLOCK_M"],
            cfg["CHUNK_C"],
            cfg["SPLITS"],
            lds_bytes=_lds_bytes(arch),
            fast_fp_math=cfg["FAST_FP_MATH"],
            fast_narrow=cfg["FAST_NARROW"],
        )
    except ValueError as e:
        raise ValueError(f"[flydsl_mhc_seam] {e}") from None


def _stage_nt(T: int, H: int) -> bool:
    """Whether the staged collapse's round trip overruns the last-level cache.

    It is the one streamed buffer whose writer and reader are the same call: the fused pass
    writes ``T * H`` bf16 and the finish pass reads back exactly those lines, so while it fits
    it never reaches HBM. At ``_N + 1`` times smaller than the read set ``CACHE_NT`` is tuned
    against, it fits where that set does not.
    """
    return T * H * 2 > _MALL_BYTES


def _round_fill(grid: int, cu_num: int) -> float:
    """Fraction of the workgroup slots a grid's rounds keep busy, at one workgroup per CU; its
    last round decides it."""
    return grid / (-(-grid // cu_num) * cu_num)


def get_mhc_seam_config(T: int, H: int, arch: str, cu_num: int) -> dict:
    """Launch policy keyed on (arch, cu_num) and T.

    * ``BLOCK_M``, ``CHUNK_C`` and ``CACHE_NT`` from the first ``_TABLE`` row whose bucket holds
      ``T`` (the last row beyond it).
    * Another ``H``: the row's tile if it is legal, else the first legal one in
      ``_FALLBACK_TILES``.
    * ``SPLITS``: the largest divisor of the chunk count that keeps the fused grid within
      ``_SPLIT_TARGET_BLOCKS`` workgroups or ``_MAX_SPLITS`` splits, whichever allows more,
      preferring those whose last round is at least ``_MIN_ROUND_FILL`` full.
    * ``GATE_STEP`` from ``kernels.mhc_seam.gate_step``, which is what the table's sweep found
      (on up to 4096 tokens at 304 CUs, off from 8192).
    * ``STAGE_NT`` from ``_stage_nt``: the staged collapse's round trip stays cached while it
      fits the last-level cache, whatever ``CACHE_NT`` says.
    """
    from aiter.ops.flydsl.kernels.mhc_seam import check_config, gate_step

    _require(arch in _GFX, f"unsupported arch {arch}", RuntimeError)
    lds = _lds_bytes(arch)
    row = next((r for r in _TABLE if T <= r[0]), _TABLE[-1])
    _, bm, cc, cache_nt = row
    for tile in ((bm, cc), *_FALLBACK_TILES):
        try:
            check_config(T, H, _N, *tile, 1, lds_bytes=lds)
        except ValueError:
            continue
        bm, cc = tile
        break
    else:
        raise ValueError(f"[flydsl_mhc_seam] no legal tile for H={H}")
    chunks = H // cc
    blocks = -(-T // bm)
    target = max(_SPLIT_TARGET_BLOCKS, blocks * _MAX_SPLITS)
    fitting = [
        s for s in range(1, chunks + 1) if chunks % s == 0 and blocks * s <= target
    ]
    full = [s for s in fitting if _round_fill(blocks * s, cu_num) >= _MIN_ROUND_FILL]
    splits = max(full or fitting)
    cfg = dict(
        MHC_SEAM_DEFAULTS, BLOCK_M=bm, CHUNK_C=cc, SPLITS=splits, CACHE_NT=cache_nt
    )
    cfg["GATE_STEP"] = gate_step(T, H, cu_num)
    cfg["STAGE_NT"] = _stage_nt(T, H)
    return cfg


def flydsl_mhc_seam(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None,
    sublayer_out: torch.Tensor | None = None,
    post_layer_mix: torch.Tensor | None = None,
    comb_res_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    *,
    residual_out: torch.Tensor | None = None,
    config: dict | None = None,
):
    """Delayed mHC seam with RMSNorm, two launches.

    Returns ``(residual_out, post_mix (T, 4, 1), comb_mix (T, 4, 4), layer_input (T, H) bf16,
    next_pre_mix (T, 4))``; the gates are fp32. ``pre_mix=None`` is the identity pre gate (the
    collapse is stream 0). With no post-mix (``sublayer_out=None``) the residual is returned as
    is and ``residual_out`` must not be passed; otherwise it is allocated when not given.

    Invalid inputs raise ``ValueError`` / ``TypeError`` (not ``assert``). The scratch is
    allocated per call from the caching allocator, so CUDA-graph capture needs no warm-up beyond
    the kernels' first compile.
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels import mhc_seam as k
    from aiter.ops.flydsl.kernels.kernels_common import ceildiv
    from aiter.ops.flydsl.kernels.tensor_shim import (
        _run_compiled,
        ptr_arg,
    )

    _require(residual.dim() == 3, "residual must be (T, 4, H)")
    _require(residual.dtype == torch.bfloat16, "residual must be bf16", TypeError)
    _require(residual.is_contiguous(), "residual must be contiguous")
    T, n, H = residual.shape
    _require(n == _N, f"the seam kernels are specialised for hc_mult=4, got {n}")
    _require(
        T * _N * H <= k.MAX_FLAT_ELEMENTS,
        "residual exceeds the int32 element offsets the kernels use",
    )
    _require(
        fn.shape == (24, _N * H), f"fn must be (24, {_N * H}), got {tuple(fn.shape)}"
    )
    _require(fn.dtype == torch.float32, "fn must be fp32", TypeError)
    _require(fn.is_contiguous(), "fn must be contiguous")
    _require(
        hc_scale.shape == (3,) and hc_base.shape == (24,),
        "hc_scale (3,) and hc_base (24,)",
    )
    _require(
        hc_scale.dtype == hc_base.dtype == torch.float32,
        "hc_scale and hc_base must be fp32",
        TypeError,
    )
    _require(
        hc_scale.is_contiguous() and hc_base.is_contiguous(),
        "hc_scale / hc_base must be contiguous",
    )
    _require(
        norm_weight is not None and norm_weight.shape == (H,),
        f"norm_weight must be ({H},)",
    )
    _require(norm_weight.dtype == torch.bfloat16, "norm_weight must be bf16", TypeError)
    _require(norm_weight.is_contiguous(), "norm_weight must be contiguous")
    _require(int(sinkhorn_repeat) >= 1, "sinkhorn_repeat must be >= 1")
    device = residual.device
    arch = _arch(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    _require(arch in _GFX, f"supports {_GFX}, got {arch}", RuntimeError)

    has_post = sublayer_out is not None
    if has_post:
        _require(
            post_layer_mix is not None and comb_res_mix is not None,
            "sublayer_out needs post_layer_mix and comb_res_mix",
        )
        _require(sublayer_out.shape == (T, H), f"sublayer_out must be ({T}, {H})")
        _require(
            sublayer_out.dtype == torch.bfloat16, "sublayer_out must be bf16", TypeError
        )
        _require(sublayer_out.is_contiguous(), "sublayer_out must be contiguous")
        _require(
            post_layer_mix.numel() == T * _N and post_layer_mix.is_contiguous(),
            f"post_layer_mix must be {T * _N} contiguous values",
        )
        _require(
            comb_res_mix.numel() == T * _N * _N and comb_res_mix.is_contiguous(),
            f"comb_res_mix must be {T * _N * _N} contiguous values",
        )
        _require(
            post_layer_mix.dtype == comb_res_mix.dtype == torch.float32,
            "post_layer_mix and comb_res_mix must be fp32",
            TypeError,
        )
        if residual_out is None:
            residual_out = torch.empty_like(residual)
        _require(
            residual_out.shape == residual.shape
            and residual_out.dtype == torch.bfloat16,
            "residual_out must match residual (shape, bf16)",
        )
        _require(residual_out.is_contiguous(), "residual_out must be contiguous")
    else:
        _require(residual_out is None, "no post-mix: the residual is returned as is")
        # Never written: without the post block the fused pass skips the residual store.
        residual_out = residual

    # The entry seam's identity gate is a build flag; the kernel writes it and reads no tensor,
    # so `pre_mix` stays None and the argument goes in as a null pointer rather than a buffer
    # allocated per call to be ignored.
    identity_pre = pre_mix is None
    if not identity_pre:
        _require(
            pre_mix.numel() == T * _N and pre_mix.dtype == torch.float32,
            f"pre_mix must be {T * _N} fp32 values",
        )
        _require(pre_mix.is_contiguous(), "pre_mix must be contiguous")
    for t in (
        fn,
        hc_scale,
        hc_base,
        norm_weight,
        pre_mix,
        sublayer_out,
        post_layer_mix,
        comb_res_mix,
        residual_out,
    ):
        _require(
            t is None or t.device == device, "all tensors must be on the same device"
        )

    post_out = torch.empty(T, _N, 1, dtype=torch.float32, device=device)
    comb_out = torch.empty(T, _N, _N, dtype=torch.float32, device=device)
    next_pre = torch.empty(T, _N, dtype=torch.float32, device=device)
    # also the fused pass's stage: the finish pass rescales it in place
    layer_input = torch.empty(T, H, dtype=torch.bfloat16, device=device)
    if T == 0:
        return residual_out, post_out, comb_out, layer_input, next_pre

    if config is None:
        cfg = get_mhc_seam_config(T, H, arch, _cu_num())
    else:
        unknown = sorted(set(config) - set(MHC_SEAM_DEFAULTS))
        if unknown:
            raise ValueError(f"[flydsl_mhc_seam] unknown config keys: {unknown}")
        cfg = dict(MHC_SEAM_DEFAULTS)
        cfg.update(config)
    _check_config(T, H, cfg, arch)
    gate_only = cfg["GATE_STEP"]
    if gate_only is None:
        gate_only = k.gate_step(T, H, _cu_num())
    stage_nt = cfg["STAGE_NT"]
    if stage_nt is None:
        stage_nt = _stage_nt(T, H)

    bm, splits = cfg["BLOCK_M"], cfg["SPLITS"]
    fused = k.compile_mhc_seam_fused(
        streams=_N,
        hidden=H,
        has_post=has_post,
        identity_pre=identity_pre,
        block_m=bm,
        chunk_c=cfg["CHUNK_C"],
        splits=splits,
        fast_fp_math=cfg["FAST_FP_MATH"],
        fast_narrow=cfg["FAST_NARROW"],
        load_nt=cfg["CACHE_NT"],
        store_nt=True,
        stage_nt=bool(stage_nt),
        waves_per_eu=cfg["WAVES_PER_EU"],
    )
    finish = k.compile_mhc_seam_finish(
        streams=_N,
        hidden=H,
        block_m=bm,
        splits=splits,
        rms_eps=float(rms_eps),
        norm_eps=float(norm_eps),
        hc_pre_eps=float(hc_pre_eps),
        hc_sinkhorn_eps=float(hc_sinkhorn_eps),
        hc_post_mult_value=float(hc_post_mult_value),
        sinkhorn_repeat=int(sinkhorn_repeat),
        fast_fp_math=cfg["FAST_FP_MATH"],
        fast_narrow=cfg["FAST_NARROW"],
        norm_load_nt=bool(stage_nt),
        norm_store_nt=True,
        waves_per_eu=cfg["FINISH_WAVES_PER_EU"],
        gate_only=bool(gate_only),
    )

    part = torch.empty(
        k.part_elements(T, splits, bm), dtype=torch.float32, device=device
    )
    sums = torch.empty(k.sums_elements(T, splits), dtype=torch.float32, device=device)
    stream = torch.cuda.current_stream(device)
    bf16, f32 = fx.BFloat16, fx.Float32
    # The flags the kernels were built with keep them off the null pointers.
    stage_p = ptr_arg(layer_input, bf16)
    part_p, sums_p = ptr_arg(part, f32), ptr_arg(sums, f32)
    _run_compiled(
        fused,
        ptr_arg(residual, bf16),
        ptr_arg(sublayer_out, bf16) if has_post else flyc.from_c_void_p(bf16, 0),
        ptr_arg(post_layer_mix, f32) if has_post else flyc.from_c_void_p(f32, 0),
        ptr_arg(comb_res_mix, f32) if has_post else flyc.from_c_void_p(f32, 0),
        flyc.from_c_void_p(f32, 0) if identity_pre else ptr_arg(pre_mix, f32),
        ptr_arg(fn, f32),
        ptr_arg(residual_out, bf16),
        stage_p,
        part_p,
        sums_p,
        T,
        ceildiv(T, bm),
        stream,
    )
    _run_compiled(
        finish,
        part_p,
        sums_p,
        ptr_arg(hc_scale, f32),
        ptr_arg(hc_base, f32),
        stage_p,
        ptr_arg(norm_weight, bf16),
        stage_p,
        ptr_arg(post_out, f32),
        ptr_arg(comb_out, f32),
        ptr_arg(next_pre, f32),
        T,
        k.finish_blocks(T),
        stream,
    )
    return residual_out, post_out, comb_out, layer_input, next_pre
