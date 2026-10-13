# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host wrapper for the FlyDSL delayed mHC seam with RMSNorm (gfx942), two launches, or three
with the separate RMSNorm.

Same arguments and returns as the Triton seam
``aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm`` for contiguous inputs: the
Triton seam copies a strided one, this wrapper rejects it. A fused pass post-mixes the
streams, stages the un-normalised collapse in ``layer_input`` and publishes the split-K gate
projection and both sums of squares; a finish pass folds them, forms the gates and rescales
``layer_input`` in place. With ``NORM="separate"`` the kernels of
``aiter/ops/flydsl/kernels/mhc_seam_nonorm.py`` produce the un-normalised collapse and
``aiter.rms_norm`` applies the RMSNorm after it. See ``aiter/ops/flydsl/kernels/mhc_seam.py``
for the fused-norm kernels.
"""

import functools
import weakref

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
    # "fused": the finish pass applies the RMSNorm; "separate": aiter.rms_norm after the seam
    "NORM": "fused",
    # nt hint on the sublayer_out load of the NORM="separate" kernels; None: CACHE_NT
    "CACHE_NT_X": None,
}
_NORMS = ("fused", "separate")

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
# The NORM="separate" path's table, same sweep: (largest T, BLOCK_M, CHUNK_C, CACHE_NT,
# CACHE_NT_X), for T above _SEPARATE_NORM_ABOVE. Up to there the fused norm is faster: the
# separate RMSNorm costs a launch the short call cannot hide. SPLITS follows the same rule.
_SEPARATE_NORM_ABOVE = 128
_SEPARATE_NORM_TABLE = (
    (384, 16, 256, True, False),
    (1024, 16, 128, False, True),
    (2048, 32, 128, False, True),
    (32768, 32, 128, True, False),
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
    _require(
        cfg["NORM"] in _NORMS,
        f"NORM must be one of {_NORMS}, got {cfg['NORM']!r}",
    )
    if cfg["NORM"] == "separate":
        from aiter.ops.flydsl.kernels.mhc_seam_nonorm import check_config
    else:
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


_FN_CACHE: dict = {}


def _pack_fn_nonorm(fn: torch.Tensor, capturing: bool) -> torch.Tensor:
    """fn (24, 4H) fp32 -> the no-norm fused pass's bf16 [hi; lo] rows in MFMA B lane order,
    cached per weight tensor.

    Keyed on the tensor object (dropped when it is freed, so a new weight at a recycled
    address never hits a stale pack) and checked against its address and version counter, so
    an in-place update repacks. A repack writes into the existing pack, whose address a
    captured graph holds, so one eager call after an in-place update brings captured graphs
    up to date too. Packing only runs eagerly: during capture it would be recorded, not
    computed. The host waits for the pack once, so a call on another stream cannot read it
    half written.
    """
    key = id(fn)
    hit = _FN_CACHE.get(key)
    stamp = (fn.data_ptr(), fn._version)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    _require(
        not capturing,
        "fn is not packed for NORM='separate' during CUDA-graph capture: run one eager "
        "call with the same fn before capturing, and again after updating it in place",
        RuntimeError,
    )
    hi = fn.to(torch.bfloat16)
    lo = (fn - hi.float()).to(torch.bfloat16)
    rows = torch.cat([hi, lo])  # row-product n = 24 * limb + row
    # (n, k) -> (k // 16, n // 16, k % 16 // 4, n % 16, k % 4); see kernels/mhc_seam_nonorm.py
    packed = rows.view(3, 16, -1, 4, 4).permute(2, 0, 3, 1, 4).contiguous()
    if hit is None:
        weakref.finalize(fn, _FN_CACHE.pop, key, None)
    else:
        # A fresh buffer would leave captured graphs on the old one.
        _require(
            hit[1].shape == packed.shape,
            "fn changed shape in place; the pack captured graphs hold cannot follow it",
            RuntimeError,
        )
        packed = hit[1].copy_(packed)
    torch.cuda.current_stream(fn.device).synchronize()
    _FN_CACHE[key] = (stamp, packed)
    return packed


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
    from aiter.ops.flydsl.kernels.kernels_common import ceildiv

    return grid / (ceildiv(grid, cu_num) * cu_num)


def _splits(T: int, H: int, bm: int, cc: int, cu_num: int) -> int:
    from aiter.ops.flydsl.kernels.kernels_common import ceildiv

    chunks = H // cc
    blocks = ceildiv(T, bm)
    target = max(_SPLIT_TARGET_BLOCKS, blocks * _MAX_SPLITS)
    fitting = [
        s for s in range(1, chunks + 1) if chunks % s == 0 and blocks * s <= target
    ]
    full = [s for s in fitting if _round_fill(blocks * s, cu_num) >= _MIN_ROUND_FILL]
    return max(full or fitting)


def _mode(has_post: bool, identity_pre: bool) -> str:
    """The policy's mode of a call. The API's fourth combination, no post-mix with the
    identity pre gate, takes the fused-norm policy like ``no_post``."""
    if has_post:
        return "identity_pre" if identity_pre else "post"
    return "no_post_identity_pre" if identity_pre else "no_post"


def get_mhc_seam_config(
    T: int, H: int, arch: str, cu_num: int, mode: str = "post", *, norm=None
) -> dict:
    """Launch policy keyed on (arch, cu_num), H, T and the seam's mode (``post``, ``no_post``,
    ``identity_pre``; see ``_mode``).

    * ``NORM="separate"`` above ``_SEPARATE_NORM_ABOVE`` tokens in ``post`` mode, the only one
      the no-norm kernels serve, from ``_separate_norm_policy``; ``_fused_norm_policy``
      otherwise.
    * ``norm`` restricts the policy to one path, ``"fused"`` or ``"separate"``.
    * ``GATE_STEP`` and ``STAGE_NT`` of the fused-norm path from their rules (see
      ``_fused_norm_policy``).

    Returns a copy of the memoized decision (``_policy``), so the caller may change it.
    """
    return dict(_policy(T, H, arch, cu_num, mode, norm))


@functools.lru_cache(maxsize=512)
def _policy(T: int, H: int, arch: str, cu_num: int, mode: str, norm) -> dict:
    """``get_mhc_seam_config``'s decision, memoized: the table lookups, the split search and
    the legality checks are per-call host cost otherwise. Every result passes
    ``_check_config``, so a call on the policy skips it. Callers must not mutate the result.
    """
    from aiter.ops.flydsl.kernels.mhc_seam import MAX_FLAT_ELEMENTS, gate_step

    _require(arch in _GFX, f"unsupported arch {arch}", RuntimeError)
    _require(
        T * _N * H <= MAX_FLAT_ELEMENTS,
        "residual exceeds the int32 element offsets the kernels use",
    )
    _require(norm in (None, *_NORMS), f"norm must be one of {_NORMS}, got {norm!r}")
    _require(
        norm != "separate" or mode == "post",
        f"NORM='separate' serves the post mode only, got {mode!r}",
    )

    cfg = None
    if cfg is None and (
        norm == "separate"
        or (norm is None and mode == "post" and T > _SEPARATE_NORM_ABOVE)
    ):
        cfg = _separate_norm_policy(T, H, arch, cu_num)
        _require(
            cfg is not None or norm != "separate",
            f"no legal NORM='separate' tile for H={H}",
        )

    if cfg is None:
        cfg = _fused_norm_policy(T, H, arch, cu_num)
    if cfg["NORM"] == "fused":
        cfg["GATE_STEP"] = gate_step(T, H, cu_num)
        cfg["STAGE_NT"] = _stage_nt(T, H)
    return cfg


def _separate_norm_policy(T: int, H: int, arch: str, cu_num: int) -> dict | None:
    """In-code policy of the separate-norm path: ``BLOCK_M``, ``CHUNK_C``, ``CACHE_NT`` and
    ``CACHE_NT_X`` from the ``_SEPARATE_NORM_TABLE`` row whose bucket holds ``T``, ``SPLITS``
    by ``_splits``. ``None`` when the row's tile is illegal for ``H``."""
    from aiter.ops.flydsl.kernels.mhc_seam_nonorm import check_config

    row = next((r for r in _SEPARATE_NORM_TABLE if T <= r[0]), _SEPARATE_NORM_TABLE[-1])
    _, bm, cc, cache_nt, cache_nt_x = row
    try:
        check_config(T, H, _N, bm, cc, 1, lds_bytes=_lds_bytes(arch))
    except ValueError:
        return None
    return dict(
        MHC_SEAM_DEFAULTS,
        NORM="separate",
        BLOCK_M=bm,
        CHUNK_C=cc,
        SPLITS=_splits(T, H, bm, cc, cu_num),
        CACHE_NT=cache_nt,
        CACHE_NT_X=cache_nt_x,
    )


def _fused_norm_policy(T: int, H: int, arch: str, cu_num: int) -> dict:
    """In-code policy of the fused-norm path.

    * ``BLOCK_M``, ``CHUNK_C`` and ``CACHE_NT`` from the first ``_TABLE`` row whose bucket holds
      ``T`` (the last row beyond it).
    * Another ``H``: the row's tile if it is legal, else the first legal one in
      ``_FALLBACK_TILES``.
    * ``SPLITS``: the largest divisor of the chunk count that keeps the fused grid within
      ``_SPLIT_TARGET_BLOCKS`` workgroups or ``_MAX_SPLITS`` splits, whichever allows more,
      preferring those whose last round is at least ``_MIN_ROUND_FILL`` full.
    * ``GATE_STEP`` (set by ``get_mhc_seam_config``) from ``kernels.mhc_seam.gate_step``,
      which is what the table's sweep found (on up to 4096 tokens at 304 CUs, off from 8192).
    * ``STAGE_NT`` (likewise) from ``_stage_nt``: the staged collapse's round trip stays cached
      while it fits the last-level cache, whatever ``CACHE_NT`` says.
    """
    from aiter.ops.flydsl.kernels.mhc_seam import check_config

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
    return dict(
        MHC_SEAM_DEFAULTS,
        BLOCK_M=bm,
        CHUNK_C=cc,
        SPLITS=_splits(T, H, bm, cc, cu_num),
        CACHE_NT=cache_nt,
    )


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
    """Delayed mHC seam with RMSNorm, two launches, or three with the separate RMSNorm.

    Returns ``(residual_out, post_mix (T, 4, 1), comb_mix (T, 4, 4), layer_input (T, H) bf16,
    next_pre_mix (T, 4))``; the gates are fp32. ``pre_mix=None`` is the identity pre gate (the
    collapse is stream 0). With no post-mix (``sublayer_out=None``) the residual is returned as
    is and ``residual_out`` must not be passed; otherwise it is allocated when not given.

    Above 128 tokens with the post-mix and a carried ``pre_mix``, the policy runs the no-norm
    kernels and ``aiter.rms_norm`` instead (``NORM="separate"``).

    Invalid inputs raise ``ValueError`` / ``TypeError`` (not ``assert``). The scratch is
    allocated per call from the caching allocator, so CUDA-graph capture of the fused-norm path
    needs no warm-up beyond the kernels' first compile. The separate path packs ``fn`` once per
    weight and refuses to pack during capture, so it needs one eager call with the same ``fn``
    before capturing, and one after updating ``fn`` in place before replaying graphs.
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
        k.PAD_ROWS * fn.shape[1] * fn.element_size() < 2**31,
        "fn exceeds the 2 GiB buffer descriptor offset window",
    )
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
        # Read only below, so the memoized dict is used as is.
        cfg = _policy(T, H, arch, _cu_num(), _mode(has_post, identity_pre), None)
    else:
        unknown = sorted(set(config) - set(MHC_SEAM_DEFAULTS))
        if unknown:
            raise ValueError(f"[flydsl_mhc_seam] unknown config keys: {unknown}")
        cfg = dict(MHC_SEAM_DEFAULTS)
        cfg.update(config)
        _check_config(T, H, cfg, arch)

    bm, splits = cfg["BLOCK_M"], cfg["SPLITS"]
    stream = torch.cuda.current_stream(device)
    bf16, f32 = fx.BFloat16, fx.Float32
    if cfg["NORM"] == "separate":
        # The no-norm kernels write the bf16 collapse and the gates; aiter.rms_norm
        # normalises the collapse after them.
        from aiter.ops.flydsl.kernels import mhc_seam_nonorm as nn
        from aiter.ops.rmsnorm import rms_norm

        _require(
            has_post and not identity_pre,
            "NORM='separate' needs the post-mix and a carried pre_mix",
        )
        load_nt_x = cfg["CACHE_NT_X"]
        if load_nt_x is None:
            load_nt_x = cfg["CACHE_NT"]
        fused = nn.compile_mhc_seam_fused_nonorm(
            hidden=H,
            block_m=bm,
            chunk_c=cfg["CHUNK_C"],
            splits=splits,
            load_nt=cfg["CACHE_NT"],
            load_nt_x=bool(load_nt_x),
            store_nt=True,
            fast_fp_math=cfg["FAST_FP_MATH"],
            fast_narrow=cfg["FAST_NARROW"],
            waves_per_eu=cfg["WAVES_PER_EU"],
        )
        finish = nn.compile_mhc_seam_finish_nonorm(
            hidden=H,
            splits=splits,
            rms_eps=float(rms_eps),
            hc_pre_eps=float(hc_pre_eps),
            hc_sinkhorn_eps=float(hc_sinkhorn_eps),
            hc_post_mult_value=float(hc_post_mult_value),
            sinkhorn_repeat=int(sinkhorn_repeat),
            fast_fp_math=cfg["FAST_FP_MATH"],
        )
        packed = _pack_fn_nonorm(fn, torch.cuda.is_current_stream_capturing())
        part = torch.empty(
            nn.part_elements(T, splits), dtype=torch.float32, device=device
        )
        part_p = ptr_arg(part, f32)
        _run_compiled(
            fused,
            ptr_arg(residual, bf16),
            ptr_arg(sublayer_out, bf16),
            ptr_arg(post_layer_mix, f32),
            ptr_arg(comb_res_mix, f32),
            ptr_arg(pre_mix, f32),
            ptr_arg(packed, bf16),
            ptr_arg(residual_out, bf16),
            ptr_arg(layer_input, bf16),
            part_p,
            T,
            ceildiv(T, bm),
            stream,
        )
        _run_compiled(
            finish,
            part_p,
            ptr_arg(hc_scale, f32),
            ptr_arg(hc_base, f32),
            ptr_arg(post_out, f32),
            ptr_arg(comb_out, f32),
            ptr_arg(next_pre, f32),
            T,
            nn.finish_blocks(T),
            stream,
        )
        layer_input = rms_norm(layer_input, norm_weight, float(norm_eps))
        return residual_out, post_out, comb_out, layer_input, next_pre

    gate_only = cfg["GATE_STEP"]
    if gate_only is None:
        gate_only = k.gate_step(T, H, _cu_num())
    stage_nt = cfg["STAGE_NT"]
    if stage_nt is None:
        stage_nt = _stage_nt(T, H)

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
