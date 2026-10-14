# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL delayed mHC seam with RMSNorm (gfx942).

The fused pass post-mixes the residual streams, stages the collapse, and computes
split-K gate projections. The finish pass forms the gates and applies RMSNorm.
With NORM="separate", aiter.rms_norm normalizes the collapse in a third launch.
"""

import functools
import weakref
from collections.abc import Callable

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_cu_num, get_lds_capacity_bytes
from aiter.ops.flydsl.kernels import mhc_seam, mhc_seam_nonorm
from aiter.ops.flydsl.kernels.kernels_common import ceildiv
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg
from aiter.ops.rmsnorm import rms_norm

__all__ = [
    "MHC_SEAM_DEFAULTS",
    "flydsl_mhc_seam",
    "flydsl_mhc_seam_pack_fn",
    "get_mhc_seam_config",
]

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

# Defaults for both the launch policy and explicit config overrides.
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

# Tuned on 304-CU gfx942 at H=5120: (max T, BLOCK_M, CHUNK_C, CACHE_NT).
# One row per token bucket; the last row also covers larger T.
# SPLITS is derived from the grid. CACHE_NT helps once traffic exceeds the MALL.
# BLOCK_M stays 32 above 1024 tokens: the accumulator is as wide as the limb stack, which a
# 64-token tile cannot afford.
_FUSED_NORM_TABLE = (
    (1, 16, 128, False),
    (32, 16, 128, False),
    (128, 32, 128, False),
    (384, 32, 64, False),
    (512, 32, 64, False),
    (1024, 32, 64, False),
    (2048, 32, 64, False),
    (4096, 32, 64, False),
    (8192, 32, 64, True),
    (16384, 32, 64, True),
    (32768, 32, 64, True),
)
# Same sweep: (max T, BLOCK_M, CHUNK_C, CACHE_NT, CACHE_NT_X).
# Below the threshold, fused norm avoids the cost of a separate RMSNorm launch.
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
    return get_cu_num()


@functools.lru_cache(maxsize=16)
def _lds_bytes(arch: str) -> int:
    return get_lds_capacity_bytes(arch)


def _check_config(T: int, H: int, cfg: dict, arch: str) -> None:
    _require(
        cfg["NORM"] in _NORMS,
        f"NORM must be one of {_NORMS}, got {cfg['NORM']!r}",
    )
    kernels = mhc_seam_nonorm if cfg["NORM"] == "separate" else mhc_seam
    try:
        kernels.check_config(
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
    """Stream the staged bf16 collapse only when it exceeds the last-level cache."""
    return T * H * 2 > _MALL_BYTES


def _round_fill(grid: int, cu_num: int) -> float:
    """Fraction of occupied workgroup slots across rounds, at one workgroup per CU."""
    return grid / (ceildiv(grid, cu_num) * cu_num)


def _splits(T: int, H: int, bm: int, cc: int, cu_num: int) -> int:
    chunks = H // cc
    blocks = ceildiv(T, bm)
    target = max(_SPLIT_TARGET_BLOCKS, blocks * _MAX_SPLITS)
    fitting = [
        s for s in range(1, chunks + 1) if chunks % s == 0 and blocks * s <= target
    ]
    full = [s for s in fitting if _round_fill(blocks * s, cu_num) >= _MIN_ROUND_FILL]
    return max(full or fitting)


def _mode(has_post: bool, identity_pre: bool) -> str:
    """Only post-mix with a carried pre gate supports separate RMSNorm."""
    if has_post:
        return "identity_pre" if identity_pre else "post"
    return "no_post_identity_pre" if identity_pre else "no_post"


def get_mhc_seam_config(
    T: int,
    H: int,
    arch: str,
    cu_num: int,
    mode: str = "post",
    *,
    norm: str | None = None,
) -> dict:
    """Return a mutable copy of the launch config for this shape, device, and mode.

    Modes are ``post``, ``no_post``, ``identity_pre``, and ``no_post_identity_pre``.
    Post mode prefers separate RMSNorm above 128 tokens when a legal tile exists;
    other modes use fused RMSNorm. Set ``norm`` to force ``"fused"`` or ``"separate"``.
    """
    return dict(_policy(T, H, arch, cu_num, mode, norm))


@functools.lru_cache(maxsize=512)
def _policy(T: int, H: int, arch: str, cu_num: int, mode: str, norm) -> dict:
    """Cache the validated policy config. Callers must not mutate the result."""
    _require(arch in _GFX, f"unsupported arch {arch}", RuntimeError)
    _require(
        T * _N * H <= mhc_seam.MAX_FLAT_ELEMENTS,
        "residual exceeds the int32 element offsets the kernels use",
    )
    _require(norm in (None, *_NORMS), f"norm must be one of {_NORMS}, got {norm!r}")
    _require(
        norm != "separate" or mode == "post",
        f"NORM='separate' serves the post mode only, got {mode!r}",
    )
    cfg = None
    if norm == "separate" or (
        norm is None and mode == "post" and T > _SEPARATE_NORM_ABOVE
    ):
        cfg = _separate_norm_policy(T, H, arch, cu_num)
        _require(
            cfg is not None or norm != "separate",
            f"no legal NORM='separate' tile for H={H}",
        )
    if cfg is None:
        cfg = _fused_norm_policy(T, H, arch, cu_num)
    return _derive_knobs(cfg, T, H, cu_num)


def _derive_knobs(cfg: dict, T: int, H: int, cu_num: int) -> dict:
    """Resolve automatic knobs in place, preserving explicit values."""
    if cfg["NORM"] == "fused":
        if cfg["GATE_STEP"] is None:
            cfg["GATE_STEP"] = mhc_seam.gate_step(T, H, cu_num)
        if cfg["STAGE_NT"] is None:
            cfg["STAGE_NT"] = _stage_nt(T, H)
    elif cfg["CACHE_NT_X"] is None:
        cfg["CACHE_NT_X"] = cfg["CACHE_NT"]
    return cfg


def _config_key(config: dict | None) -> tuple | None:
    """Snapshot scalar overrides in key order; None selects the automatic policy."""
    if config is None:
        return None
    unknown = sorted(set(config) - set(MHC_SEAM_DEFAULTS))
    _require(not unknown, f"unknown config keys: {unknown}")
    return tuple(sorted(config.items()))


def _resolve_config(
    T: int,
    H: int,
    arch: str,
    cu_num: int,
    mode: str,
    config: dict | None,
) -> dict:
    """Use the policy, or merge explicit overrides over defaults and validate them."""
    if config is None:
        return _policy(T, H, arch, cu_num, mode, None)
    cfg = {**MHC_SEAM_DEFAULTS, **config}
    _check_config(T, H, cfg, arch)
    _require(
        cfg["NORM"] != "separate" or mode == "post",
        "NORM='separate' needs the post-mix and a carried pre_mix",
    )
    return _derive_knobs(cfg, T, H, cu_num)


def _separate_norm_policy(
    T: int,
    H: int,
    arch: str,
    cu_num: int,
) -> dict | None:
    """Use the separate-norm table, or return None if its tile is illegal for H."""
    row = next((r for r in _SEPARATE_NORM_TABLE if T <= r[0]), _SEPARATE_NORM_TABLE[-1])
    _, bm, cc, cache_nt, cache_nt_x = row
    try:
        mhc_seam_nonorm.check_config(T, H, _N, bm, cc, 1, lds_bytes=_lds_bytes(arch))
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
    """Use the fused-norm table, falling back to the first legal tile for H."""
    lds = _lds_bytes(arch)
    row = next((r for r in _FUSED_NORM_TABLE if T <= r[0]), _FUSED_NORM_TABLE[-1])
    _, bm, cc, cache_nt = row
    for tile in ((bm, cc), *_FALLBACK_TILES):
        try:
            mhc_seam.check_config(T, H, _N, *tile, 1, lds_bytes=lds)
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


def _validate_inputs(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None,
    sublayer_out: torch.Tensor | None,
    post_layer_mix: torch.Tensor | None,
    comb_res_mix: torch.Tensor | None,
    norm_weight: torch.Tensor | None,
    residual_out: torch.Tensor | None,
    fn_packed: torch.Tensor | None,
) -> tuple[int, int, str]:
    """Validate inputs and return the token count, hidden size, and GPU arch."""
    _require(residual.dim() == 3, "residual must be (T, 4, H)")
    _require(residual.dtype == torch.bfloat16, "residual must be bf16", TypeError)
    _require(residual.is_contiguous(), "residual must be contiguous")
    T, n, H = residual.shape
    _require(n == _N, f"the seam kernels are specialised for hc_mult=4, got {n}")
    _require(
        T * _N * H <= mhc_seam.MAX_FLAT_ELEMENTS,
        "residual exceeds the int32 element offsets the kernels use",
    )
    _require(
        fn.shape == (24, _N * H), f"fn must be (24, {_N * H}), got {tuple(fn.shape)}"
    )
    _require(fn.dtype == torch.float32, "fn must be fp32", TypeError)
    _require(fn.is_contiguous(), "fn must be contiguous")
    _require(
        mhc_seam.PACK_ROWS * fn.shape[1] < 2**31,
        "the packed fn exceeds the int32 element offsets the kernels use",
    )
    if fn_packed is not None:
        _require(
            fn_packed.shape == _packed_fn_shape(H),
            "fn_packed must come from flydsl_mhc_seam_pack_fn(fn)",
        )
        _require(fn_packed.dtype == torch.bfloat16, "fn_packed must be bf16", TypeError)
        _require(fn_packed.is_contiguous(), "fn_packed must be contiguous")
    else:
        _require(
            not fn.is_inference(),
            "an inference-mode fn has no version counter to detect updates with: "
            "pass fn_packed=flydsl_mhc_seam_pack_fn(fn)",
            RuntimeError,
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

    if sublayer_out is not None:
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
        if residual_out is not None:
            _require(
                residual_out.shape == residual.shape
                and residual_out.dtype == torch.bfloat16,
                "residual_out must match residual (shape, bf16)",
            )
            _require(residual_out.is_contiguous(), "residual_out must be contiguous")
    else:
        _require(residual_out is None, "no post-mix: the residual is returned as is")

    if pre_mix is not None:
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
        fn_packed,
    ):
        _require(
            t is None or t.device == device, "all tensors must be on the same device"
        )
    return T, H, arch


def _ptr(t: torch.Tensor | None, dtype):
    """Pass a null pointer for optional inputs the kernel does not read."""
    return flyc.from_c_void_p(dtype, 0) if t is None else ptr_arg(t, dtype)


def _packed_fn_shape(H: int) -> tuple[int, ...]:
    return (_N * H // 16, mhc_seam.N_TILES, 4, mhc_seam.MFMA_MN, 4)


def flydsl_mhc_seam_pack_fn(fn: torch.Tensor) -> torch.Tensor:
    """fn (24, 4H) fp32 -> bf16 [hi; lo] limbs in MFMA B lane order, for ``fn_packed=``.

    Pack when the weight is loaded, and again into the same buffer when it is reloaded.
    """
    _require(fn.dim() == 2 and fn.shape[0] == 24, "fn must be (24, 4H)")
    _require(fn.shape[1] % 16 == 0, "fn's 4H columns must be a multiple of 16")
    _require(fn.dtype == torch.float32, "fn must be fp32", TypeError)
    hi = fn.to(torch.bfloat16)
    lo = (fn - hi.float()).to(torch.bfloat16)
    rows = torch.cat([hi, lo])  # row-product n = 24 * limb + row
    # (n, k) -> (k // 16, n // 16, k % 16 // 4, n % 16, k % 4); see kernels/mhc_seam.py
    return (
        rows.view(mhc_seam.N_TILES, mhc_seam.MFMA_MN, -1, 4, 4)
        .permute(2, 0, 3, 1, 4)
        .contiguous()
    )


_FN_CACHE: dict = {}


def _pack_fn(fn: torch.Tensor, capturing: bool) -> torch.Tensor:
    """``flydsl_mhc_seam_pack_fn`` cached per weight tensor, for calls without ``fn_packed``.

    Keyed on the tensor object and stamped with its address and version counter, so an
    in-place update through ``fn`` repacks; a write through ``fn.data`` changes neither and
    is missed. A repack writes into the buffer captured graphs hold. Packing is eager only,
    and the host waits for it so no other stream reads it half written.
    """
    key = id(fn)
    hit = _FN_CACHE.get(key)
    stamp = (fn.data_ptr(), fn._version)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    _require(
        not capturing,
        "fn is not packed during CUDA-graph capture: pass fn_packed, "
        "or run one eager call with the same fn before capturing and again after "
        "updating it in place",
        RuntimeError,
    )
    packed = flydsl_mhc_seam_pack_fn(fn)
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


def _make_fused_norm_runner(
    T: int,
    cfg: dict,
    fused: Callable,
    finish: Callable,
    has_post: bool,
) -> Callable:
    """Bind the launch geometry; allocate scratch and form pointers on each call."""
    bm, splits = cfg["BLOCK_M"], cfg["SPLITS"]
    part_size = mhc_seam.part_elements(T, splits, bm)
    sums_size = mhc_seam.sums_elements(T, splits)
    fused_blocks = ceildiv(T, bm)
    finish_blocks = mhc_seam.finish_blocks(T)

    def run(
        *,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        pre_mix: torch.Tensor | None,
        sublayer_out: torch.Tensor | None,
        post_layer_mix: torch.Tensor | None,
        comb_res_mix: torch.Tensor | None,
        norm_weight: torch.Tensor,
        residual_out: torch.Tensor,
        post_out: torch.Tensor,
        comb_out: torch.Tensor,
        layer_input: torch.Tensor,
        next_pre: torch.Tensor,
        stream: torch.cuda.Stream,
        fn_packed: torch.Tensor | None,
    ) -> torch.Tensor:
        bf16, f32 = fx.BFloat16, fx.Float32
        packed = fn_packed
        if packed is None:
            packed = _pack_fn(fn, torch.cuda.is_current_stream_capturing())
        part = torch.empty(part_size, dtype=torch.float32, device=residual.device)
        sums = torch.empty(sums_size, dtype=torch.float32, device=residual.device)
        part_p, sums_p = ptr_arg(part, f32), ptr_arg(sums, f32)
        stage_p = ptr_arg(layer_input, bf16)
        _run_compiled(
            fused,
            ptr_arg(residual, bf16),
            _ptr(sublayer_out, bf16),
            _ptr(post_layer_mix if has_post else None, f32),
            _ptr(comb_res_mix if has_post else None, f32),
            _ptr(pre_mix, f32),
            ptr_arg(packed, bf16),
            ptr_arg(residual_out, bf16),
            stage_p,
            part_p,
            sums_p,
            T,
            fused_blocks,
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
            finish_blocks,
            stream,
        )
        return layer_input

    return run


def _make_separate_norm_runner(
    T: int,
    cfg: dict,
    fused: Callable,
    finish: Callable,
    norm_eps: float,
) -> Callable:
    """Bind the no-norm kernels and apply RMSNorm after their two launches."""
    part_size = mhc_seam_nonorm.part_elements(T, cfg["SPLITS"])
    fused_blocks = ceildiv(T, cfg["BLOCK_M"])
    finish_blocks = mhc_seam_nonorm.finish_blocks(T)

    def run(
        *,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        pre_mix: torch.Tensor | None,
        sublayer_out: torch.Tensor | None,
        post_layer_mix: torch.Tensor | None,
        comb_res_mix: torch.Tensor | None,
        norm_weight: torch.Tensor,
        residual_out: torch.Tensor,
        post_out: torch.Tensor,
        comb_out: torch.Tensor,
        layer_input: torch.Tensor,
        next_pre: torch.Tensor,
        stream: torch.cuda.Stream,
        fn_packed: torch.Tensor | None,
    ) -> torch.Tensor:
        bf16, f32 = fx.BFloat16, fx.Float32
        packed = fn_packed
        if packed is None:
            packed = _pack_fn(fn, torch.cuda.is_current_stream_capturing())
        part = torch.empty(part_size, dtype=torch.float32, device=residual.device)
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
            fused_blocks,
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
            finish_blocks,
            stream,
        )
        return rms_norm(layer_input, norm_weight, norm_eps)

    return run


@functools.lru_cache(maxsize=512)
def _compile_launcher(
    *,
    T: int,
    H: int,
    arch: str,
    cu_num: int,
    config_key: tuple | None,
    has_post: bool,
    identity_pre: bool,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
) -> Callable:
    """Cache a tensor-level runner by shape, device properties, mode, and config.

    The runner retains only the compiled kernels and scalar launch metadata.
    Tensors, scratch, and streams belong to each invocation.
    """
    cfg = _resolve_config(
        T,
        H,
        arch,
        cu_num,
        _mode(has_post, identity_pre),
        None if config_key is None else dict(config_key),
    )
    fused_kwargs = {
        "hidden": H,
        "block_m": cfg["BLOCK_M"],
        "chunk_c": cfg["CHUNK_C"],
        "splits": cfg["SPLITS"],
        "fast_fp_math": cfg["FAST_FP_MATH"],
        "fast_narrow": cfg["FAST_NARROW"],
        "load_nt": cfg["CACHE_NT"],
        "store_nt": True,
        "waves_per_eu": cfg["WAVES_PER_EU"],
    }
    finish_kwargs = {
        "hidden": H,
        "splits": cfg["SPLITS"],
        "rms_eps": rms_eps,
        "hc_pre_eps": hc_pre_eps,
        "hc_sinkhorn_eps": hc_sinkhorn_eps,
        "hc_post_mult_value": hc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
        "fast_fp_math": cfg["FAST_FP_MATH"],
    }
    if cfg["NORM"] == "separate":
        fused = mhc_seam_nonorm.compile_mhc_seam_fused_nonorm(
            **fused_kwargs,
            load_nt_x=bool(cfg["CACHE_NT_X"]),
        )
        finish = mhc_seam_nonorm.compile_mhc_seam_finish_nonorm(**finish_kwargs)
        return _make_separate_norm_runner(T, cfg, fused, finish, norm_eps)

    fused = mhc_seam.compile_mhc_seam_fused(
        **fused_kwargs,
        streams=_N,
        has_post=has_post,
        identity_pre=identity_pre,
        stage_nt=bool(cfg["STAGE_NT"]),
    )
    finish = mhc_seam.compile_mhc_seam_finish(
        **finish_kwargs,
        streams=_N,
        block_m=cfg["BLOCK_M"],
        norm_eps=norm_eps,
        fast_narrow=cfg["FAST_NARROW"],
        norm_load_nt=bool(cfg["STAGE_NT"]),
        norm_store_nt=True,
        waves_per_eu=cfg["FINISH_WAVES_PER_EU"],
        gate_only=bool(cfg["GATE_STEP"]),
    )
    return _make_fused_norm_runner(T, cfg, fused, finish, has_post)


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
    fn_packed: torch.Tensor | None = None,
):
    """Delayed mHC seam with RMSNorm for contiguous inputs.

    Returns ``(residual_out, post_mix, comb_mix, layer_input, next_pre_mix)`` with
    shapes ``(T, 4, H)``, ``(T, 4, 1)``, ``(T, 4, 4)``, ``(T, H)``, and ``(T, 4)``.
    Residual and layer input are bf16; gates are fp32. ``pre_mix=None`` collapses
    stream 0. Without ``sublayer_out``, the residual is returned unchanged and
    ``residual_out`` must be None; otherwise it is allocated when not supplied.

    With ``config=None``, use the launch policy. Explicit configs override
    ``MHC_SEAM_DEFAULTS``. Fused RMSNorm takes two launches; separate RMSNorm takes
    three and is preferred above 128 tokens with post-mix and a carried pre gate.

    Scratch is allocated per call. Compile before CUDA-graph capture. Pass
    ``fn_packed=flydsl_mhc_seam_pack_fn(fn)``, repacked into the same buffer when
    ``fn`` is reloaded; it is required for an inference-mode ``fn``. Without it, ``fn``
    is packed on the first eager call, which must precede capture, and repacked after
    an in-place update through ``fn`` but not through ``fn.data``.
    """
    T, H, arch = _validate_inputs(
        residual,
        fn,
        hc_scale,
        hc_base,
        sinkhorn_repeat,
        pre_mix,
        sublayer_out,
        post_layer_mix,
        comb_res_mix,
        norm_weight,
        residual_out,
        fn_packed,
    )
    has_post = sublayer_out is not None
    identity_pre = pre_mix is None
    device = residual.device

    # The no-post kernel skips the residual store, so that output aliases the input.
    if not has_post:
        residual_out = residual
    elif residual_out is None:
        residual_out = torch.empty_like(residual)
    post_out = torch.empty(T, _N, 1, dtype=torch.float32, device=device)
    comb_out = torch.empty(T, _N, _N, dtype=torch.float32, device=device)
    next_pre = torch.empty(T, _N, dtype=torch.float32, device=device)
    layer_input = torch.empty(T, H, dtype=torch.bfloat16, device=device)
    if T == 0:
        return residual_out, post_out, comb_out, layer_input, next_pre

    launcher = _compile_launcher(
        T=T,
        H=H,
        arch=arch,
        cu_num=_cu_num(),
        config_key=_config_key(config),
        has_post=has_post,
        identity_pre=identity_pre,
        rms_eps=float(rms_eps),
        hc_pre_eps=float(hc_pre_eps),
        hc_sinkhorn_eps=float(hc_sinkhorn_eps),
        hc_post_mult_value=float(hc_post_mult_value),
        sinkhorn_repeat=int(sinkhorn_repeat),
        norm_eps=float(norm_eps),
    )
    layer_input = launcher(
        residual=residual,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        pre_mix=pre_mix,
        sublayer_out=sublayer_out,
        post_layer_mix=post_layer_mix,
        comb_res_mix=comb_res_mix,
        norm_weight=norm_weight,
        residual_out=residual_out,
        post_out=post_out,
        comb_out=comb_out,
        layer_input=layer_input,
        next_pre=next_pre,
        stream=torch.cuda.current_stream(device),
        fn_packed=fn_packed,
    )
    return residual_out, post_out, comb_out, layer_input, next_pre
