# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE kernel management: naming, compilation, and high-level API."""

import functools
import os
import re

from typing import Dict, Optional
from aiter.jit.utils.chip_info import get_gfx
from aiter.utility import dtypes

import flydsl.compiler as flyc
import torch

_KERNEL_PARAMS: Dict[str, Dict] = {}
_DIRECT_STAGE2_LAST_WORKSPACE: Optional[torch.Tensor] = None

_SUFFIX_RE = re.compile(r"(?P<fq>_fq)?(?:_sbm(?P<sbm>\d+))?$")
_KERNEL_NAME_RE = re.compile(
    r"^flydsl_moe(?P<stage>[12])_a(?P<a>.+?)_w(?P<b>.+?)_"
    r"(?P<out>bf16|f16)_t(?P<tm>\d+)x(?P<tn>\d+)x(?P<tk>\d+)"
    r"(?:_(?P<rest>.*))?$"
)
_DIRECT_KERNEL_NAME_RE = re.compile(
    r"^flydsl_moe2_direct_a(?P<a>.+?)_w(?P<b>.+?)_"
    r"(?P<out>bf16|f16)_t(?P<tm>\d+)x(?P<tn>\d+)x(?P<tk>\d+)"
    r"(?:_(?P<rest>.*))?$"
)
_DIRECT_STAGE1_KERNEL_NAME_RE = re.compile(
    r"^flydsl_moe1_direct_a(?P<a>.+?)_w(?P<b>.+?)_"
    r"(?P<out>bf16|f16)_t(?P<tm>\d+)x(?P<tn>\d+)x(?P<tk>\d+)"
    r"(?:_(?P<rest>.*))?$"
)

_MFMA16_ALIASES = {"16", "16x16", "16x16x128", "mfma16", "mfma16k128"}
_MFMA32_ALIASES = {"32", "32x32", "32x32x64", "mfma32", "mfma32k64"}
_LDS_LIMIT_BYTES_BY_GFX = {
    "gfx942": 64 * 1024,
    "gfx950": 160 * 1024,
}


def _current_gfx() -> str:
    return str(get_gfx()).split(":", 1)[0].lower()


def _expand_per_tensor_scale(scale: Optional[torch.Tensor], rows: int, cols: int):
    """Expand compact per-tensor/per-expert scales to FlyDSL row-scale layout.

    Used only by the legacy (non-fused-init) fallback paths now.  In the fused
    init path the broadcast is folded into the init kernel and this function
    is bypassed.  We intentionally do not cache the expanded result: the old
    ``id(tensor)``-keyed cache made correctness depend on caller-side object
    reuse and tangled with weakref finalizers, and once both stages run
    through fused init the cache covers almost no GPU work anyway.
    """
    if scale is None:
        return None
    flat = scale.view(-1)
    if flat.numel() == 1:
        return flat.expand(rows).contiguous()
    if flat.numel() == rows:
        return flat
    if cols > 1 and flat.numel() * cols == rows:
        return flat.view(-1, 1).expand(-1, cols).contiguous().view(-1)
    return flat


def _resolve_a_scale_for_fused_init(
    a_scale: Optional[torch.Tensor], rows: int, dev: torch.device
):
    """Resolve a_scale into ``(flat_a, needs_fused_expand)`` for fused init.

    For per-tensor a_scale (numel==1) we allocate a fresh ``(rows,)`` f32
    buffer and signal that the fused kernel must populate it via scalar
    broadcast. For already-flat (numel==rows) we return the source as-is.
    For unusual layouts we fall back to the legacy expand so the GEMM sees
    the same flat buffer it always has.
    """
    if a_scale is None:
        return torch.empty(0, device=dev), False
    flat = a_scale.view(-1)
    n = flat.numel()
    if n == 1:
        # Fused init will write `a_scale[0]` into all `rows` slots.
        return (
            torch.empty(rows, dtype=flat.dtype, device=flat.device),
            True,
        )
    if n == rows:
        return flat, False
    # Unusual layout (e.g. per-1x32). Legacy path handles it; no fusion.
    fallback = _expand_per_tensor_scale(a_scale, rows, 1)
    if fallback is None:
        fallback = torch.empty(0, device=dev)
    return fallback, False


def _resolve_w_scale_for_fused_init(
    w_scale: Optional[torch.Tensor], rows: int, cols: int, dev: torch.device
):
    """Resolve w_scale into ``(flat_w, needs_fused_expand)`` for fused init.

    Always allocates a fresh buffer on the per-expert broadcast path and
    asks the fused init kernel to do the broadcast.  No host-side cache:
    the broadcast write is folded into the kernel that has to launch for
    the zero-fill anyway, so reusing a pre-expanded buffer across calls
    saves no measurable GPU time and the ``id()``-keyed cache it required
    was a correctness hazard.
    """
    if w_scale is None:
        return torch.empty(0, device=dev), False
    flat = w_scale.view(-1)
    n = flat.numel()
    if n == rows:
        return flat, False
    if n == 1:
        # Rare: per-tensor weight scale.  Reuse legacy scalar-broadcast path
        # rather than wiring a third broadcast variant into the fused kernel.
        return flat.expand(rows).contiguous(), False
    if cols > 1 and n * cols == rows:
        return torch.empty(rows, dtype=flat.dtype, device=flat.device), True
    # Unknown layout - keep flat (matches legacy fallback behavior).
    return flat, False


def _stage2_mfma_variant_tag(tile_k: int, a_dtype: str, b_dtype: str) -> str:
    """Return the FP4 stage2 MFMA variant tag needed in kernel names."""
    if a_dtype != "fp4" or b_dtype != "fp4":
        return ""
    if int(tile_k) == 128:
        return "mfma32k64"

    variant = os.environ.get("FLIR_MOE_STAGE2_MFMA", "16x16x128").strip().lower()
    if not variant or variant in _MFMA16_ALIASES:
        return ""
    if variant in _MFMA32_ALIASES:
        return "mfma32k64"
    raise ValueError(
        "FLIR_MOE_STAGE2_MFMA must be '16x16x128' or '32x32x64', "
        f"got {variant!r}"
    )


def flydsl_kernel_name(
    stage: int,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    mode: str = "",
    sort_block_m: int = 0,
    fuse_fp4_quant: bool = False,
) -> str:
    """Construct kernel name: ``flydsl_moe{stage}_a{a}_w{b}_{out}_t{M}x{N}x{K}[_{mode}][_fq][_sbm{S}]``."""
    name = f"flydsl_moe{stage}_a{a_dtype}_w{b_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
    if mode:
        name += f"_{mode}"
    if fuse_fp4_quant:
        name += "_fq"
    if sort_block_m > 0 and sort_block_m != tile_m:
        name += f"_sbm{sort_block_m}"
    return name


def _parse_flydsl_kernel_name(name: str) -> Optional[Dict]:
    direct_stage1_match = _DIRECT_STAGE1_KERNEL_NAME_RE.match(name)
    if direct_stage1_match is not None:
        params: Dict = {
            "stage": 1,
            "direct": True,
            "a_dtype": direct_stage1_match.group("a"),
            "b_dtype": direct_stage1_match.group("b"),
            "out_dtype": direct_stage1_match.group("out"),
            "tile_m": int(direct_stage1_match.group("tm")),
            "tile_n": int(direct_stage1_match.group("tn")),
            "tile_k": int(direct_stage1_match.group("tk")),
            "MPerBlock": int(direct_stage1_match.group("tm")),
            "use_async_copy": False,
            "waves_per_eu": 0,
            "b_nt": 2,
            "k_batch": 1,
            "gate_only": False,
            "fuse_fp4_quant": False,
            "routes_per_block": 1,
            "num_waves": 0,
            # Default split-K mode: "atomic" (legacy single-buffer atomic-fadd).
            # Only honored when k_batch>1.  See compile_moe_gemm1_direct_smallm.
            "splitk_mode": "atomic",
        }
        for token in (direct_stage1_match.group("rest") or "").split("_"):
            if not token:
                continue
            if token.startswith("w") and token[1:].isdigit():
                params["waves_per_eu"] = int(token[1:])
            elif token.startswith("bnt") and token[3:].isdigit():
                params["b_nt"] = int(token[3:])
            elif token.startswith("rpb") and token[3:].isdigit():
                params["routes_per_block"] = int(token[3:])
            elif token.startswith("nw") and token[2:].isdigit():
                params["num_waves"] = int(token[2:])
            elif token.startswith("kb") and token[2:].isdigit():
                # Split-K factor along model_dim; see compile_moe_gemm1_direct_smallm.
                params["k_batch"] = int(token[2:])
            elif token == "red":
                # Split-K reduce-mode tag: per-WG kb-slice plain-stores + host
                # kb-axis sum.  Default ("atomic") keeps the legacy
                # single-buffer atomic-fadd path.
                params["splitk_mode"] = "reduce"
            else:
                return None
        return params

    direct_match = _DIRECT_KERNEL_NAME_RE.match(name)
    if direct_match is not None:
        params: Dict = {
            "stage": 2,
            "direct": True,
            "a_dtype": direct_match.group("a"),
            "b_dtype": direct_match.group("b"),
            "out_dtype": direct_match.group("out"),
            "tile_m": int(direct_match.group("tm")),
            "tile_n": int(direct_match.group("tn")),
            "tile_k": int(direct_match.group("tk")),
            "MPerBlock": int(direct_match.group("tm")),
            "mode": "direct",
            "use_async_copy": False,
            "waves_per_eu": 0,
            "b_nt": 2,
            "split_reduce": False,
            "sort_block_m": 0,
            "persist": False,
        }
        for token in (direct_match.group("rest") or "").split("_"):
            if not token:
                continue
            if token.startswith("w") and token[1:].isdigit():
                params["waves_per_eu"] = int(token[1:])
            elif token.startswith("bnt") and token[3:].isdigit():
                params["b_nt"] = int(token[3:])
            elif token in ("sr", "splitreduce"):
                params["split_reduce"] = True
            else:
                return None
        return params

    match = _KERNEL_NAME_RE.match(name)
    if match is None:
        return None

    stage = int(match.group("stage"))
    a_dtype = match.group("a")
    b_dtype = match.group("b")
    params: Dict = {
        "stage": stage,
        "a_dtype": a_dtype,
        "b_dtype": b_dtype,
        "out_dtype": match.group("out"),
        "tile_m": int(match.group("tm")),
        "tile_n": int(match.group("tn")),
        "tile_k": int(match.group("tk")),
        "MPerBlock": int(match.group("tm")),
        "use_async_copy": False,
        "waves_per_eu": 1 if b_dtype == "fp4" else 0,
        "b_nt": 2,
    }
    if stage == 1:
        # n_per_wave defaults to 32 (legacy behavior); see compile_moe_gemm1.
        # splitk_mode defaults to "atomic" (legacy); only honored when k_batch>1
        # and only relevant for non-fp4. Parser sets it whenever the "_red"
        # suffix is present (regardless of stage) so name<->params round-trips.
        params.update(
            {"k_batch": 1, "gate_only": False, "n_per_wave": 32, "splitk_mode": "atomic"}
        )
    else:
        # Stage2 also supports n_per_wave (default 32 = legacy) and k_batch
        # (split-K, default 1 = no split); see compile_moe_gemm2.
        params.update(
            {
                "mode": "atomic",
                "sort_block_m": 0,
                "persist": False,
                "n_per_wave": 32,
                "k_batch": 1,
            }
        )

    for token in (match.group("rest") or "").split("_"):
        if not token:
            continue
        if token == "async":
            params["use_async_copy"] = True
        elif stage == 2 and token in ("atomic", "reduce"):
            params["mode"] = token
        elif token in ("mfma16k128", "mfma32k64"):
            params["mfma_variant"] = token
        elif token.startswith("w") and token[1:].isdigit():
            params["waves_per_eu"] = int(token[1:])
        elif token.startswith("bnt") and token[3:].isdigit():
            params["b_nt"] = int(token[3:])
        elif token.startswith("kb") and token[2:].isdigit():
            # Split-K factor; valid for stage1 (already wired) and stage2
            # (newly wired via compile_moe_gemm2.k_batch).
            params["k_batch"] = int(token[2:])
        elif stage == 1 and token == "red":
            # Stage1 split-K reduce-mode tag: per-WG kb-slice plain-stores
            # + host-side kb-axis sum.  Default ("atomic") keeps the legacy
            # single-buffer atomic-fadd path.  See compile_moe_gemm1.splitk_mode.
            params["splitk_mode"] = "reduce"
        elif stage == 1 and token == "go":
            params["gate_only"] = True
        elif stage == 1 and token == "fq":
            params["fuse_fp4_quant"] = True
        elif stage == 2 and token == "persist":
            params["persist"] = True
        elif token.startswith("sbm") and token[3:].isdigit():
            params["sort_block_m"] = int(token[3:])
        elif token == "fq":
            params["fuse_fp4_quant"] = True
        elif token.startswith("n") and token[1:].isdigit():
            # _n<N> overrides N-cols-per-wave for both stages (codegen knob,
            # see compile_moe_gemm{1,2}.n_per_wave). Default 32 (legacy); 16
            # doubles `num_waves` and halves `num_acc_n` -- helps small tile_m.
            params["n_per_wave"] = int(token[1:])
        else:
            return None
    return params


def get_flydsl_kernel_params(name: str) -> Optional[Dict]:
    """Lookup kernel params by name. Strips ``_fq`` / ``_sbm{N}`` suffixes transparently."""
    params = _KERNEL_PARAMS.get(name)
    if params is not None:
        return params
    m = _SUFFIX_RE.search(name)
    if m and m.group(0):
        base_name = name[: m.start()]
        params = _KERNEL_PARAMS.get(base_name)
        if params is not None:
            extra: Dict = {}
            if m.group("fq"):
                extra["fuse_fp4_quant"] = True
            if m.group("sbm") is not None:
                extra["sort_block_m"] = int(m.group("sbm"))
            return {**params, **extra}
    return _parse_flydsl_kernel_name(name)


def _x_load_supported(
    a_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    use_async_copy: bool,
    n_per_wave: int = 32,
) -> bool:
    """Mirror X-load compile-time constraints before creating tune tasks.

    Halving ``n_per_wave`` doubles ``num_waves`` (=tile_n/n_per_wave) and thus
    ``total_threads``, which halves ``bytes_per_thread_x``. The dword (4-byte)
    indexed load mapping in compile_moe_gemm{1,2} requires
    ``bytes_per_thread_x % 4 == 0`` — a constraint that bites earliest for fp8
    with small tile_k. Bake the actual ``n_per_wave`` into this filter so the
    enumerator never produces candidates that can't compile.
    """
    elem_bytes = 2 if a_dtype in ("fp16", "bf16", "int4_bf16") else 1
    total_threads = (tile_n // n_per_wave) * 64
    bytes_x_per_tile = tile_m * tile_k * elem_bytes
    if bytes_x_per_tile % total_threads != 0:
        return False

    bytes_per_thread_x = bytes_x_per_tile // total_threads
    if a_dtype in ("fp16", "bf16", "int4_bf16"):
        return bytes_per_thread_x % 16 == 0
    if use_async_copy:
        return bytes_per_thread_x % 16 == 0
    return bytes_per_thread_x % 4 == 0


@functools.lru_cache(maxsize=1)
def _device_lds_limit_bytes() -> int:
    """Return the supported gfx target's per-workgroup LDS limit in bytes."""
    gfx = _current_gfx()
    if gfx not in _LDS_LIMIT_BYTES_BY_GFX:
        raise RuntimeError(f"FlyDSL MoE LDS filtering does not support {gfx!r}.")
    return _LDS_LIMIT_BYTES_BY_GFX[gfx]


def _async_copy_candidates() -> list[bool]:
    gfx = _current_gfx()
    if gfx == "gfx942":
        #TODO: gfx942 not support buffer load from global memory to LDS directly, so we only support false
        return [False]
    if gfx == "gfx950":
        return [False, True]
    raise RuntimeError(f"FlyDSL MoE async-copy enumeration does not support {gfx!r}.")


def _lds_within_limit(
    a_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    waves_per_eu: int,
    *,
    use_cshuffle_epilog: bool = True,
    lds_limit_bytes: Optional[int] = None,
) -> bool:
    """Filter candidates that exceed the supported gfx target's LDS limit."""
    if lds_limit_bytes is None:
        lds_limit_bytes = _device_lds_limit_bytes()
    elem_bytes = 2 if a_dtype in ("fp16", "bf16", "int4_bf16") else 1
    lds_x_bytes = 2 * int(tile_m) * int(tile_k) * int(elem_bytes)
    lds_out_bytes = 2 * int(tile_m) * int(tile_n) if use_cshuffle_epilog else 0
    lds_tid_bytes = int(tile_m) * 4
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes) + lds_tid_bytes
    if waves_per_eu >= 1:
        # Match compile_moe_gemm{1,2}: waves_per_eu reserves minimum LDS to
        # influence occupancy. The reservation budget is per-CU LDS / arch,
        # i.e. 64 KB on gfx942 and 160 KB on gfx950 -- using a hard-coded
        # 160 KB here would over-filter on gfx942 (rejecting candidates that
        # actually fit) AND mis-match the compile-time padding in
        # `kernels/moe_gemm_2stage.py::compile_moe_gemm{1,2}`.
        min_lds = lds_limit_bytes // (waves_per_eu + 1) + 1
        lds_total_bytes = max(lds_total_bytes, min_lds)
    return lds_total_bytes <= lds_limit_bytes


def stage1_kb_candidates_for_m(M: int) -> list[int]:
    """Recommended split-K (`k_batch`) candidates for stage1 at a given M.

    Empirical plateau analysis from rocprofv3 sweeps under Hunyuan-V3 shape
    `model_dim=4096, inter_dim=192, topk=9, fp8 per-Tensor`; tile
    `t16x64x256_n16`.  Authoritative source: the WARMUP=5 ITERS=30 sweep
    at `lm_eval_logs/moe_splitk_reduce_compare_stage1/COMPARE.txt` (per-cell
    SE ~0.1-0.3 us, 2sigma-significance-filtered verdicts).  An earlier
    WARMUP=2 ITERS=3 sweep produced numbers that were ~15 % uniformly biased
    by JIT warmup / GPU thermal state; ignore those if you find them.

    Best per M, clean v3 numbers (atomic-mode unless noted):

        M=1   ->  kb=8  (7.91 us  vs 18.89 us kb=1   -> 2.39x)
        M=2   ->  kb=8  (13.61 us vs 19.59 us kb=1   -> 1.44x)
        M=4   ->  kb=8  (21.84 us vs 24.56 us kb=1   -> 1.12x)
        M=8   ->  kb=8  (36.16 us vs 37.83 us kb=1   -> 1.05x; reduce/atomic tied)
        M=16  ->  kb=1  (38.39 us; no split-K variant beats baseline)
        M=32  ->  kb=1  (58.99 us; no split-K variant beats baseline)
        M=64  ->  kb=1  (70.62 us; reduce-mode kb=2 is 2nd at 76.53 us,
                         the one cell where reduce beats atomic by >2sigma)

    Atomic vs reduce split-K mode is a 2-significant-cells-out-of-21
    decision (M=64 kb=2 favors reduce, M=64 kb=8 favors atomic); for every
    other (M, kb) the two modes are tied within noise.  The autotuner still
    enumerates both modes (see `get_flydsl_stage1_kernels`) so per-shape
    differences are picked up automatically.

    Implication: keep a wide kb range only for M <= 8 where split-K really
    helps; for M >= 16 the autotuner just wastes time on kb>1 candidates,
    so we narrow to [1].
    """
    if M <= 1:
        return [1, 2, 4, 8]
    if M <= 2:
        return [1, 2, 4, 8]
    if M <= 4:
        return [1, 2, 4, 8]
    if M <= 8:
        return [1, 2, 4, 8]
    # M >= 16: every split-K variant was strictly worse than kb=1 in the
    # clean v3 sweep (best kb>1 was within +1.05x of kb=1 only at M=64
    # with reduce-mode kb=2 = 76.53 us vs kb=1 = 70.62 us, i.e. an 8%
    # regression).  No point asking the autotuner to try them.
    return [1]


def get_flydsl_stage1_kernels(
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    model_dim: Optional[int] = None,
    n_dim: Optional[int] = None,
    m_for_kb_filter: Optional[int] = None,
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage1 configs.

    When ``m_for_kb_filter`` is provided, restrict the enumerated split-K
    candidates to ``stage1_kb_candidates_for_m(m_for_kb_filter)``.  This
    lets the tuner avoid generating obviously-bad ``_kb*`` variants at
    large M (autotune scales down significantly).
    """
    kernels = {}
    is_fp4 = b_dtype == "fp4"

    tile_ns = [32, 64, 128, 256] if is_fp4 else [32, 64, 128, 256]
    tile_ks = [256] if is_fp4 else [128, 256]
    tile_ms = [16, 32, 64, 128] if is_fp4 else [16, 32, 64]
    waves_per_eus = [1, 2, 3, 4] if is_fp4 else [0, 1, 2, 3, 4]
    # Non-fp4 split-K (the new compile_moe_gemm1 codegen path) requires an
    # EVEN number of K tiles per WG and at least 2, so the largest useful
    # k_batch is bounded by model_dim/(tile_k*2).  For Hunyuan-V3
    # (model_dim=4096, tile_k in {128, 256}) k_batch up to 8 stays valid.
    # The fp4 path keeps its existing wider sweep.
    k_batches = [1, 2, 4, 8, 16] if is_fp4 else [1, 2, 4, 8]
    if m_for_kb_filter is not None and not is_fp4:
        # Cap split-K candidates to the empirical plateau at this M.  Always
        # keep kb=1 in the candidate list so the no-split baseline survives.
        _allowed_kb = set(stage1_kb_candidates_for_m(int(m_for_kb_filter)))
        _allowed_kb.add(1)
        k_batches = [k for k in k_batches if k in _allowed_kb]
    b_nts = [0, 2] if is_fp4 else [0, 2]
    async_copies = _async_copy_candidates()
    for tm in tile_ms:
        stage1_tile_ns = tile_ns
        if is_fp4:
            if tm in [16, 32]:
                stage1_tile_ns = [32, 64, 128]
            else:
                stage1_tile_ns = [64, 128, 256]
        for tn in stage1_tile_ns:
            if n_dim is not None and n_dim % tn != 0:
                continue
            use_cshuffle_epilog = None if is_fp4 or tn % 128 == 0 else False
            for tk in tile_ks:
                if model_dim is not None and model_dim % tk != 0:
                    continue
                for async_copy in async_copies:
                    if not _x_load_supported(a_dtype, tm, tn, tk, async_copy):
                        continue
                    for wpe in waves_per_eus:
                        if (
                            not is_fp4
                            and not _lds_within_limit(
                                a_dtype,
                                tm,
                                tn,
                                tk,
                                wpe,
                                use_cshuffle_epilog=(
                                    True
                                    if use_cshuffle_epilog is None
                                    else bool(use_cshuffle_epilog)
                                ),
                            )
                        ):
                            continue
                        for kb in k_batches if wpe == 3 else [1]:
                            if is_fp4:
                                if tk == 512 and kb == 16:
                                    continue
                                if tn // 32 * 64 > 256:
                                    continue
                            # Split-K stage1 requires:
                            #   model_dim % k_batch == 0 and
                            #   (model_dim // k_batch) % tile_k == 0
                            # The non-fp4 codegen path additionally requires
                            #   tiles_per_batch = (model_dim/k_batch)/tile_k
                            #   to be EVEN and >= 2 (the kernel main loop
                            #   assumes a fixed 2-tile tail).  When model_dim
                            #   is provided by the tuner, skip combinations
                            #   that can never compile for that shape.
                            if model_dim is not None and kb > 1:
                                if model_dim % kb != 0:
                                    continue
                                if (model_dim // kb) % tk != 0:
                                    continue
                                if not is_fp4:
                                    _tiles_per_batch = (model_dim // kb) // tk
                                    if (
                                        _tiles_per_batch < 2
                                        or _tiles_per_batch % 2 != 0
                                    ):
                                        continue
                            gate_onlys = [False, True] if kb > 1 and is_fp4 else [False]
                            # Stage1 split-K mode: atomic (legacy) for kb>=1,
                            # plus a reduce-mode variant for kb>1 on the non-fp4
                            # codegen path. The reduce variant uses identical
                            # tile/wpe/bnt/npw settings but a per-WG kb-slice
                            # plain-store + host kb-axis sum, eliminating the
                            # atomic-fadd contention at the cost of kb x more
                            # tmp memory. See compile_moe_gemm1.splitk_mode.
                            if kb > 1 and not is_fp4:
                                splitk_modes = ["atomic", "reduce"]
                            else:
                                splitk_modes = ["atomic"]
                            for bnt in b_nts:
                                for go in gate_onlys:
                                    npw_candidates = [32]
                                    if (
                                        not is_fp4
                                        and tm <= 16
                                        and tn % 16 == 0
                                        and (tn // 16) * 64 <= 1024
                                        and _x_load_supported(
                                            a_dtype, tm, tn, tk, async_copy,
                                            n_per_wave=16,
                                        )
                                    ):
                                        npw_candidates.append(16)
                                    for npw in npw_candidates:
                                        if tn % npw != 0:
                                            continue
                                        for skmode in splitk_modes:
                                            name = flydsl_kernel_name(
                                                1, a_dtype, b_dtype, out_dtype, tm, tn, tk
                                            )
                                            if async_copy:
                                                name += "_async"
                                            if is_fp4 and wpe != 1:
                                                name += f"_w{wpe}"
                                            elif not is_fp4 and wpe > 0:
                                                name += f"_w{wpe}"
                                            if kb != 1:
                                                name += f"_kb{kb}"
                                            if bnt != 2:
                                                name += f"_bnt{bnt}"
                                            if go:
                                                name += "_go"
                                            if npw != 32:
                                                name += f"_n{npw}"
                                            if skmode == "reduce":
                                                name += "_red"
                                            kernels[name] = {
                                                "stage": 1,
                                                "a_dtype": a_dtype,
                                                "b_dtype": b_dtype,
                                                "out_dtype": out_dtype,
                                                "tile_m": tm,
                                                "tile_n": tn,
                                                "tile_k": tk,
                                                "MPerBlock": tm,
                                                "use_cshuffle_epilog": use_cshuffle_epilog,
                                                "use_async_copy": async_copy,
                                                "waves_per_eu": wpe,
                                                "k_batch": kb,
                                                "b_nt": bnt,
                                                "gate_only": go,
                                                "n_per_wave": npw,
                                                "splitk_mode": skmode,
                                            }

    if (
        a_dtype == "fp8"
        and b_dtype == "fp8"
        and out_dtype == "bf16"
        and model_dim is not None
        and n_dim is not None
        and model_dim % 64 == 0
    ):
        # Direct stage1 supports the SAME split-K codegen path as the standard
        # kernel (compile_moe_gemm1_direct_smallm.k_batch / splitk_mode).
        # Apply the same kb shortlist and the same m-based filter so the tuner
        # doesn't queue obviously-bad kb>1 variants at large M.  Direct kernel
        # uses a simple per-tile K loop without 2-tile tail unrolling, so the
        # standard kernel's "EVEN tile count per WG" restriction does NOT
        # apply here -- any K-tile count >= 1 is fine.
        _direct_kb_pool = [1, 2, 4, 8]
        if m_for_kb_filter is not None:
            _allowed_kb = set(stage1_kb_candidates_for_m(int(m_for_kb_filter)))
            _allowed_kb.add(1)
            _direct_kb_pool = [k for k in _direct_kb_pool if k in _allowed_kb]
        for tn in (16, 32, 64, 96, 192):
            if n_dim % tn != 0:
                continue
            for tk in (64, 128, 256, 512):
                if model_dim % tk != 0:
                    continue
                num_waves_candidates = [0]
                if tn in (32, 64):
                    num_waves_candidates.append(2)
                elif tn == 96:
                    num_waves_candidates.extend([2, 3])
                elif tn == 192:
                    num_waves_candidates.append(3)
                for num_waves in num_waves_candidates:
                    for kb in _direct_kb_pool:
                        if kb > 1:
                            if model_dim % kb != 0:
                                continue
                            _k_per_batch = model_dim // kb
                            if _k_per_batch % tk != 0:
                                continue
                            if (_k_per_batch // tk) < 1:
                                continue
                        # Reduce mode only meaningful with kb>1; atomic
                        # covers both kb==1 (degenerate, no atomics issued)
                        # and kb>1 (legacy single-buffer atomic-fadd).
                        splitk_modes = ["atomic", "reduce"] if kb > 1 else ["atomic"]
                        for skmode in splitk_modes:
                            name = (
                                f"flydsl_moe1_direct_a{a_dtype}_w{b_dtype}"
                                f"_{out_dtype}_t16x{tn}x{tk}"
                            )
                            if num_waves:
                                name += f"_nw{num_waves}"
                            if kb != 1:
                                name += f"_kb{kb}"
                            # IMPORTANT: keep "_red" at the very end so the
                            # parser picks it up as a standalone token
                            # regardless of which other knobs are present.
                            if skmode == "reduce":
                                name += "_red"
                            kernels[name] = {
                                "stage": 1,
                                "direct": True,
                                "a_dtype": a_dtype,
                                "b_dtype": b_dtype,
                                "out_dtype": out_dtype,
                                "tile_m": 16,
                                "tile_n": tn,
                                "tile_k": tk,
                                "MPerBlock": 16,
                                "use_async_copy": False,
                                "waves_per_eu": 0,
                                "k_batch": kb,
                                "b_nt": 2,
                                "gate_only": False,
                                "routes_per_block": 1,
                                "num_waves": num_waves,
                                "splitk_mode": skmode,
                            }
    return kernels


def get_flydsl_stage2_kernels(
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    k_dim: Optional[int] = None,
    n_dim: Optional[int] = None,
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage2 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"
    tile_ns = [128, 256] if is_fp4 else [64, 128, 256]
    tile_ks = [128, 256] if is_fp4 else [64, 128, 256]
    tile_ms = [16, 32, 64, 128] if is_fp4 else [16, 32, 64, 128]
    modes = ["atomic", "reduce"]
    waves_per_eus = [0] if is_fp4 else [0, 1, 2, 3, 4]
    b_nts = [0] if is_fp4 else [0, 2]
    async_copies = _async_copy_candidates()

    for tm in tile_ms:
        for tn in tile_ns:
            if n_dim is not None and n_dim % tn != 0:
                continue
            for tk in tile_ks:
                if k_dim is not None and k_dim % tk != 0:
                    continue
                mfma_variant_tag = _stage2_mfma_variant_tag(tk, a_dtype, b_dtype)
                if is_fp4 and tk == 128:
                    # tile_k=128 uses the dedicated MFMA32/K64 stage2 path.
                    # Keep the candidate set deliberately small until layout
                    # coverage is broadened.
                    if a_dtype != "fp4" or tm not in (32, 64) or tn != 128:
                        continue
                for mode in modes:
                    for async_copy in async_copies:
                        if is_fp4 and tk == 128 and async_copy:
                            continue
                        if not _x_load_supported(a_dtype, tm, tn, tk, async_copy):
                            continue
                        for wpe in waves_per_eus:
                            if not is_fp4 and not _lds_within_limit(
                                a_dtype, tm, tn, tk, wpe
                            ):
                                continue
                            for bnt in b_nts:
                                npw_candidates = [32]
                                if (
                                    not is_fp4
                                    and tm <= 16
                                    and tn % 16 == 0
                                    and (tn // 16) * 64 <= 1024
                                    and _x_load_supported(
                                        a_dtype, tm, tn, tk, async_copy,
                                        n_per_wave=16,
                                    )
                                ):
                                    npw_candidates.append(16)
                                for npw in npw_candidates:
                                    if tn % npw != 0:
                                        continue
                                    if not is_fp4 and mode == "atomic":
                                        kb_candidates = [1, 2, 3, 4, 6, 8]
                                    else:
                                        kb_candidates = [1]
                                    for kb in kb_candidates:
                                        if kb > 1 and k_dim is not None:
                                            if k_dim % kb != 0:
                                                continue
                                            if (k_dim // kb) % tk != 0:
                                                continue
                                        base_name = flydsl_kernel_name(
                                            2,
                                            a_dtype,
                                            b_dtype,
                                            out_dtype,
                                            tm,
                                            tn,
                                            tk,
                                            mode,
                                        )
                                        if async_copy:
                                            base_name += "_async"
                                        if mfma_variant_tag:
                                            base_name += f"_{mfma_variant_tag}"
                                        if is_fp4 and wpe != 1:
                                            base_name += f"_w{wpe}"
                                        elif not is_fp4 and wpe > 0:
                                            base_name += f"_w{wpe}"
                                        if bnt != 2:
                                            base_name += f"_bnt{bnt}"
                                        if npw != 32:
                                            base_name += f"_n{npw}"
                                        if kb != 1:
                                            base_name += f"_kb{kb}"
                                        base_params = {
                                            "stage": 2,
                                            "a_dtype": a_dtype,
                                            "b_dtype": b_dtype,
                                            "out_dtype": out_dtype,
                                            "tile_m": tm,
                                            "tile_n": tn,
                                            "tile_k": tk,
                                            "mode": mode,
                                            "MPerBlock": tm,
                                            "use_async_copy": async_copy,
                                            "waves_per_eu": wpe,
                                            "b_nt": bnt,
                                            "n_per_wave": npw,
                                            "k_batch": kb,
                                        }
                                        if mfma_variant_tag:
                                            base_params["mfma_variant"] = mfma_variant_tag
                                        kernels[base_name] = base_params
    if (
        a_dtype == "fp8"
        and b_dtype == "fp8"
        and out_dtype == "bf16"
        and k_dim is not None
        and n_dim is not None
        and k_dim % 64 == 0
    ):
        for tn in (32, 64, 128, 256):
            if n_dim % tn != 0:
                continue
            for split_reduce in (False, True):
                name = f"flydsl_moe2_direct_a{a_dtype}_w{b_dtype}_{out_dtype}_t16x{tn}x64"
                if split_reduce:
                    name += "_sr"
                kernels[name] = {
                    "stage": 2,
                    "direct": True,
                    "a_dtype": a_dtype,
                    "b_dtype": b_dtype,
                    "out_dtype": out_dtype,
                    "tile_m": 16,
                    "tile_n": tn,
                    "tile_k": 64,
                    "MPerBlock": 16,
                    "mode": "direct",
                    "split_reduce": split_reduce,
                    "use_async_copy": False,
                    "waves_per_eu": 0,
                    "b_nt": 2,
                    "sort_block_m": 0,
                    "persist": False,
                }
    return kernels


def _register_all_configs():
    """Pre-populate _KERNEL_PARAMS with all supported configs at import time."""
    for a in ("fp8", "fp4", "fp16"):
        for b in ("fp4",):
            for out in ("bf16", "f16"):
                _KERNEL_PARAMS.update(get_flydsl_stage1_kernels(a, b, out))
                _KERNEL_PARAMS.update(get_flydsl_stage2_kernels(a, b, out))
    for a in ("fp8", "int8"):
        for b in (a,):
            for out in ("bf16", "f16"):
                _KERNEL_PARAMS.update(get_flydsl_stage1_kernels(a, b, out))
                _KERNEL_PARAMS.update(get_flydsl_stage2_kernels(a, b, out))


_register_all_configs()


def compile_flydsl_moe_stage1(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    act: str = "silu",
    use_g1u1: bool = True,
    persist_m: int = 1,
    fuse_fp4_quant: bool = False,
    fuse_sort_scale: bool = False,
    use_async_copy: bool = False,
    use_cshuffle_epilog: Optional[bool] = None,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    gate_only: bool = False,
    n_per_wave: int = 32,
    splitk_mode: str = "atomic",
):
    """Compile stage1 kernel (cached via underlying lru_cache).

    ``splitk_mode`` (``"atomic"`` | ``"reduce"``) selects how the K-axis
    partials are merged when ``k_batch > 1``; ignored otherwise.  See
    ``compile_moe_gemm1`` (kernels/moe_gemm_2stage.py) for the codegen
    semantics and ``flydsl_moe_stage1`` for the matching tmp-buffer layout
    + post-pass that the reduce mode requires.
    """
    if b_dtype == "fp4":
        # NOTE: n_per_wave knob is only wired into the bf16/fp8/int8 codegen
        # (kernels/moe_gemm_2stage.py). The mixed_moe_gemm_2stage (fp4) path
        # still uses the legacy `num_waves = tile_n // 32` derivation; reject
        # non-default values rather than silently ignore them.
        if n_per_wave != 32:
            raise ValueError(
                "compile_flydsl_moe_stage1: n_per_wave override is only "
                f"supported for non-fp4 b_dtype, got b_dtype={b_dtype!r} "
                f"with n_per_wave={n_per_wave}"
            )
        if str(splitk_mode) != "atomic":
            raise ValueError(
                "compile_flydsl_moe_stage1: splitk_mode override is only "
                f"supported for non-fp4 b_dtype, got b_dtype={b_dtype!r} "
                f"with splitk_mode={splitk_mode!r}"
            )
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1

        return compile_mixed_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            act=act,
            use_g1u1=use_g1u1,
            persist_m=persist_m,
            fuse_fp4_quant=fuse_fp4_quant,
            fuse_sort_scale=fuse_sort_scale,
            use_async_copy=use_async_copy,
            k_batch=k_batch,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
            gate_only=gate_only,
        )
    else:
        from .kernels.moe_gemm_2stage import compile_moe_gemm1

        return compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            in_dtype=a_dtype,
            out_dtype=out_dtype,
            act=act,
            use_g1u1=use_g1u1,
            use_cshuffle_epilog=use_cshuffle_epilog,
            use_async_copy=use_async_copy,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
            n_per_wave=n_per_wave,
            k_batch=k_batch,
            splitk_mode=splitk_mode,
        )


def compile_flydsl_moe_stage2(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    accumulate: bool = True,
    persist_m: int = 1,
    sort_block_m: int = 0,
    use_async_copy: bool = False,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    mfma_variant: Optional[str] = None,
    direct: bool = False,
    a_scale_scalar: bool = False,
    w_scale_per_expert: bool = False,
    split_reduce: bool = False,
    n_per_wave: int = 32,
    k_batch: int = 1,
):
    """Compile stage2 kernel (cached via underlying lru_cache)."""
    if direct:
        if a_dtype != "fp8" or b_dtype != "fp8":
            raise ValueError("direct small-M stage2 currently supports only fp8/fp8")
        from .kernels.moe_gemm_2stage_direct import compile_moe_gemm2_direct_smallm

        return compile_moe_gemm2_direct_smallm(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            in_dtype=a_dtype,
            out_dtype=out_dtype,
            a_scale_scalar=a_scale_scalar,
            w_scale_per_expert=w_scale_per_expert,
            split_reduce=split_reduce,
        )
    if b_dtype == "fp4":
        # NOTE: n_per_wave / k_batch knobs are only wired into the bf16/fp8/int8
        # codegen (kernels/moe_gemm_2stage.py). The mixed_moe_gemm_2stage (fp4)
        # path still uses the legacy `num_waves = tile_n // 32` derivation and a
        # single Z-grid block; reject non-default values rather than silently
        # ignore them.
        if n_per_wave != 32:
            raise ValueError(
                "compile_flydsl_moe_stage2: n_per_wave override is only "
                f"supported for non-fp4 b_dtype, got b_dtype={b_dtype!r} "
                f"with n_per_wave={n_per_wave}"
            )
        if int(k_batch) != 1:
            raise ValueError(
                "compile_flydsl_moe_stage2: k_batch (split-K) override is only "
                f"supported for non-fp4 b_dtype, got b_dtype={b_dtype!r} "
                f"with k_batch={k_batch}"
            )
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

        return compile_mixed_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            accumulate=accumulate,
            persist_m=persist_m,
            sort_block_m=sort_block_m,
            use_async_copy=use_async_copy,
            mfma_variant=mfma_variant,
        )
    else:
        from .kernels.moe_gemm_2stage import compile_moe_gemm2

        return compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=a_dtype,
            out_dtype=out_dtype,
            accumulate=accumulate,
            use_async_copy=use_async_copy,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
            n_per_wave=n_per_wave,
            k_batch=k_batch,
        )


# Private helpers


_DLPACK_SAFE = (torch.uint8, torch.float16, torch.bfloat16, torch.float32)


def _view_safe(t: torch.Tensor) -> torch.Tensor:
    """View as uint8 if dtype is not dlpack-safe, otherwise return as-is."""
    return (
        t.view(torch.uint8)
        if t is not None and t.numel() > 0 and t.dtype not in _DLPACK_SAFE
        else t
    )


def _s1_args_fp4(
    out,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    out_scale_sorted,
    token_num,
    n_in,
    k_in,
    size_expert_ids_in,
    dev,
):
    empty_f32 = torch.empty(0, device=dev, dtype=torch.float32)
    return (
        _view_safe(out),
        _view_safe(a),
        _view_safe(w),
        _view_safe(a_scale),
        _view_safe(w_scale),
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        empty_f32,
        out_scale_sorted,
        token_num,
        n_in,
        k_in,
        size_expert_ids_in,
        torch.cuda.current_stream(),
    )


def _s1_args_std(
    out,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    size_expert_ids_in,
):
    return (
        out,
        a,
        w,
        a_scale,
        w_scale,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        token_num,
        n_in,
        k_in,
        size_expert_ids_in,
        torch.cuda.current_stream(),
    )


def _s2_args_fp4(
    target,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    blocks,
    dev,
):
    empty_f32 = torch.empty(0, device=dev, dtype=torch.float32)
    return (
        _view_safe(target),
        _view_safe(a),
        _view_safe(w),
        _view_safe(a_scale),
        _view_safe(w_scale),
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        empty_f32,
        token_num,
        n_in,
        k_in,
        blocks,
        torch.cuda.current_stream(),
    )


def _s2_args_std(
    target,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    blocks,
):
    return (
        target,
        a,
        w,
        a_scale,
        w_scale,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        token_num,
        n_in,
        k_in,
        blocks,
        torch.cuda.current_stream(),
    )


def _s2_direct_args_std(
    out,
    a,
    w,
    a_scale,
    w_scale,
    topk_ids,
    topk_weights,
    token_num,
    n_in,
    k_in,
):
    return (
        _view_safe(out),
        _view_safe(a),
        _view_safe(w),
        _view_safe(a_scale),
        _view_safe(w_scale),
        topk_ids,
        topk_weights,
        token_num,
        n_in,
        k_in,
        torch.cuda.current_stream(),
    )


def _s1_direct_args_std(
    out,
    a,
    w,
    a_scale,
    w_scale,
    topk_ids,
    token_num,
    n_in,
    k_in,
):
    return (
        _view_safe(out),
        _view_safe(a),
        _view_safe(w),
        _view_safe(a_scale),
        _view_safe(w_scale),
        topk_ids,
        token_num,
        n_in,
        k_in,
        torch.cuda.current_stream(),
    )


def _direct_stage2_workspace(
    out: torch.Tensor,
    token_num: int,
    topk: int,
    model_dim: int,
) -> torch.Tensor:
    global _DIRECT_STAGE2_LAST_WORKSPACE
    shape = (int(token_num) * int(topk), int(model_dim))
    workspace = _DIRECT_STAGE2_LAST_WORKSPACE
    if (
        workspace is None
        or workspace.shape != shape
        or workspace.dtype != out.dtype
        or workspace.device != out.device
    ):
        workspace = torch.empty(shape, dtype=out.dtype, device=out.device)
        _DIRECT_STAGE2_LAST_WORKSPACE = workspace
    return workspace


def _run_compiled(exe, args):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.
    """
    # flydsl>=0.1.2 exposes flydsl.compiler.compile; older versions (e.g.
    # 0.1.1.dev409) only provide jit wrappers and execute via exe(*args).
    if not hasattr(flyc, "compile"):
        exe(*args)
        return

    cf = getattr(exe, "_aiter_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._aiter_cf = cf
    else:
        cf(*args)


@functools.cache
def _get_compiled_silu_fq(inter_dim: int, topk: int):
    """Compile and cache the fused silu_and_mul + mxfp4 quant + scale-sort kernel."""
    from aiter.ops.flydsl.kernels.silu_and_mul_fq import build_silu_and_mul_fq_module

    return build_silu_and_mul_fq_module(inter_dim, topk)


# Public API


def flydsl_moe_stage1(
    a: torch.Tensor,
    w1: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 256,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    act: str = "silu",
    use_g1u1: Optional[bool] = None,
    w1_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
    persist_m: int = 0,
    fuse_fp4_quant: bool = False,
    fuse_sort_scale: bool = False,
    use_async_copy: bool = False,
    use_cshuffle_epilog: Optional[bool] = None,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    gate_only: bool = False,
    n_per_wave: int = 32,
    splitk_mode: str = "atomic",
):
    """Fused MOE stage1 GEMM.

    a: (token_num, model_dim)
    w1:
      - g1u1: (E, 2*inter_dim, model_dim) pre-shuffled
      - g1u0: (E, inter_dim, model_dim) pre-shuffled
    For fp4 stage1, `w1`/`w1_scale` must use the same preshuffle layout as
    `shuffle_weight(..., (16, 16))` and `e8m0_shuffle(...)`.

    When fuse_sort_scale=True, the kernel writes e8m0 scales in sorted tiled
    layout directly, avoiding a separate moe_mxfp4_sort call.

    When k_batch>1 (split-K), the kernel writes partial sums via atomics into
    a zeroed buffer, then runs post-activation:
      - g1u1: reduce gate/up partials with activation+mul
      - g1u0: reduce proj partials then apply unary activation

    When gate_only=True (requires use_g1u1 and k_batch>1), each workgroup computes only
    one B-tile stream (no gate/up interleaving).  The grid X doubles so
    that by_n naturally covers both gate and up regions.

    Returns:
        Basic:                      out
        fuse_sort_scale:            (out, out_scale_sorted)
    """
    token_num = a.shape[0]
    E = w1.shape[0]

    if use_g1u1 is None:
        logical_out_inter_dim = None
        if out is not None:
            logical_out_inter_dim = out.shape[-1] * (2 if fuse_fp4_quant else 1)
        if logical_out_inter_dim is not None:
            if w1.shape[1] == 2 * logical_out_inter_dim:
                use_g1u1 = True
                inter_dim = logical_out_inter_dim
            elif w1.shape[1] == logical_out_inter_dim:
                use_g1u1 = False
                inter_dim = logical_out_inter_dim
            else:
                raise ValueError(
                    f"Unable to infer g1u mode from w1.shape={tuple(w1.shape)} "
                    f"and out.shape={tuple(out.shape)}"
                )
        else:
            # Preserve the historical direct-call behavior: when the caller does
            # not provide enough shape information, assume the legacy g1u1 path.
            if (w1.shape[1] % 2) != 0:
                raise ValueError(
                    f"Unable to infer g1u1 inter_dim from odd w1.shape[1]={w1.shape[1]}"
                )
            use_g1u1 = True
            inter_dim = w1.shape[1] // 2
    else:
        use_g1u1 = bool(use_g1u1)
        if use_g1u1:
            if (w1.shape[1] % 2) != 0:
                raise ValueError(
                    f"g1u1 stage1 expects w1.shape[1] to be even, got {w1.shape[1]}"
                )
            inter_dim = w1.shape[1] // 2
        else:
            inter_dim = w1.shape[1]

    if act == "swiglu" and not use_g1u1:
        raise ValueError("swiglu stage1 requires use_g1u1=True")

    model_dim = a.shape[1]

    if a_dtype == "fp4":
        model_dim = model_dim * 2

    torch_out_dtype = (
        dtypes.fp4x2
        if fuse_fp4_quant
        else dtypes.bf16 if out_dtype == "bf16" else dtypes.fp16
    )
    _is_splitk = k_batch > 1
    # Non-fp4 stage1 split-K (the codegen path added in compile_moe_gemm1) writes
    # f32 partials atomically into a (M*topk, 2*inter_dim) tmp buffer; the host
    # post-pass converts back to the requested out dtype via PyTorch silu+mul.
    _is_splitk_std = _is_splitk and (b_dtype != "fp4")
    if str(splitk_mode) not in ("atomic", "reduce"):
        raise ValueError(
            f"flydsl_moe_stage1: splitk_mode must be 'atomic' or 'reduce', "
            f"got {splitk_mode!r}"
        )
    # Reduce only makes sense for the non-fp4 split-K codegen path; for fp4 or
    # kb==1 we silently fall back to the legacy single-buffer layout.
    _splitk_std_reduce = _is_splitk_std and str(splitk_mode) == "reduce"

    dev = a.device
    _splitk_fq = _is_splitk and fuse_fp4_quant

    _splitk_out_cols = inter_dim * (2 if use_g1u1 else 1)
    if _splitk_fq and not use_g1u1:
        raise ValueError("split-K fused fp4 quant currently requires use_g1u1=True")
    if _splitk_fq and act not in ("silu", "swiglu"):
        raise ValueError("split-K fused fp4 quant only supports silu/swiglu stage1")
    if _is_splitk_std:
        if not use_g1u1:
            raise ValueError(
                "non-fp4 stage1 split-K (k_batch>1) currently requires use_g1u1=True "
                "(the codegen path stores gate/up as separate halves of the tmp buffer)"
            )
        if act not in ("silu",):
            raise ValueError(
                f"non-fp4 stage1 split-K only supports act='silu' for now, got {act!r}"
            )
        if sorted_weights is not None:
            raise ValueError(
                "non-fp4 stage1 split-K is incompatible with doweight_stage1 "
                "(per-token weight must be applied AFTER the K reduction; the "
                "silu_and_mul post-pass does not multiply by tw)"
            )

    if out is None:
        if fuse_fp4_quant:
            out = torch.empty(
                (token_num, topk, inter_dim // 2), dtype=torch_out_dtype, device=dev
            )
        else:
            out = torch.empty(
                (token_num, topk, inter_dim), dtype=torch_out_dtype, device=dev
            )

    # ------------------------------------------------------------------
    # Stage1 init: tmp_out zero + flat_a_scale broadcast + flat_w_scale
    # broadcast. The non-fp4 split-K path fuses these three helpers into a
    # single Triton launch (see aiter.ops.flydsl._fused_post). When the fused
    # path is disabled (env or fp4 codegen path) we fall back to torch.zeros
    # + two separate _expand_per_tensor_scale launches.
    # ------------------------------------------------------------------
    from aiter.ops.flydsl._fused_post import (
        fused_init as _fused_init,
        is_init_disabled as _fused_init_disabled,
    )

    _use_fused_init = _is_splitk_std and not _fused_init_disabled()

    if _is_splitk_std:
        # f32 partials.  Atomic mode: single (M*topk, 2*inter_dim) buffer that
        # every WG atomic-fadd's into.  Reduce mode: (k_batch, M*topk, 2*inter)
        # so each WG plain-stores into its own kb-slice; the host post-pass
        # then sums across the leading kb axis.  Both layouts put gate at
        # columns [0, inter_dim) and up at [inter_dim, 2*inter_dim) within
        # each row so the silu_and_mul fold is identical.
        # zeros() (or the fused init kernel below) is required for both modes:
        # atomic mode needs a zeroed accumulator, reduce mode needs zeros for
        # any (kb, valid-token) slice the GEMM skipped (e.g. invalid blocks)
        # so the kb-axis sum stays correct.
        if _splitk_std_reduce:
            _tmp_out_shape = (k_batch, token_num, topk, 2 * inter_dim)
        else:
            _tmp_out_shape = (token_num, topk, 2 * inter_dim)
        if _use_fused_init:
            tmp_out = torch.empty(_tmp_out_shape, dtype=torch.float32, device=dev)
        else:
            tmp_out = torch.zeros(_tmp_out_shape, dtype=torch.float32, device=dev)
    elif _is_splitk:
        torch_tmp_out_dtype = dtypes.bf16 if out_dtype == "bf16" else dtypes.fp16
        tmp_out = torch.zeros(
            (token_num, topk, _splitk_out_cols), dtype=torch_tmp_out_dtype, device=dev
        )
    else:
        tmp_out = None

    if _use_fused_init:
        # Single fused launch covering all three helpers (zero tmp_out +
        # scalar a_scale broadcast + per-expert w_scale broadcast). The
        # resolve helpers always allocate a fresh broadcast buffer for the
        # per-expert path and signal the kernel to fill it; the host-side
        # cache that used to short-circuit w-expand is gone.
        flat_a_scale, _need_a = _resolve_a_scale_for_fused_init(
            a1_scale, token_num, dev
        )
        flat_w_scale, _need_w = _resolve_w_scale_for_fused_init(
            w1_scale, E * w1.shape[1], w1.shape[1], dev
        )
        _fused_init(
            tmp_out=tmp_out,
            flat_a=flat_a_scale if _need_a else None,
            a_src=a1_scale if _need_a else None,
            flat_w=flat_w_scale if _need_w else None,
            w_src=w1_scale if _need_w else None,
            w_cols=w1.shape[1] if _need_w else 1,
        )
    else:
        flat_a_scale = _expand_per_tensor_scale(a1_scale, token_num, 1)
        if flat_a_scale is None:
            flat_a_scale = torch.empty(0, device=dev)
        flat_w_scale = _expand_per_tensor_scale(
            w1_scale, E * w1.shape[1], w1.shape[1]
        )
        if flat_w_scale is None:
            flat_w_scale = torch.empty(0, device=dev)
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )

    _need_quant = fuse_fp4_quant or _splitk_fq
    _need_sort = _need_quant and (fuse_sort_scale or _splitk_fq)

    # `_sort_block_m` MUST equal the `block_size` that moe_sorting was called
    # with. The kernel's grid_y is derived from `_dense_blks`, and one WG
    # handles exactly one sort-block of `block_size` rows. If we pick a value
    # > block_size, the floor-division in `_dense_blks` undercounts the actual
    # valid blocks, the kernel skips the trailing blocks, and their output
    # rows in stage1_out stay UNINITIALIZED (random garbage from the caching
    # allocator). This used to be hardcoded to max(32, tile_m) which silently
    # over-allocates the scale buffer below but BREAKS correctness whenever
    # tile_m < 32 (e.g. tile_m=16) because in production / tuner the contract
    # is moe_sorting.block_size == tile_m.
    _sort_block_m = tile_m
    _all_blks = sorted_expert_ids.shape[0]
    _dense_blks = (
        min(token_num * topk * _sort_block_m, sorted_token_ids.shape[0])
        // _sort_block_m
    )
    _grid_y = min(_dense_blks, _all_blks)

    _persist_m = persist_m if persist_m > 0 else 1

    # Allocate sorted-scale buffer with padding for tiled layout
    scale_cols = inter_dim // 32
    sorted_size = max(
        sorted_token_ids.shape[0], sorted_expert_ids.shape[0] * _sort_block_m
    )
    padded_rows = (sorted_size + 255) // 256 * 256
    padded_cols = (scale_cols + 7) // 8 * 8
    out_scale_sorted_flat = (
        torch.empty(padded_rows * padded_cols, dtype=torch.uint8, device=dev)
        if _need_sort
        else torch.empty(0, dtype=torch.uint8, device=dev)
    )

    # split-K GEMM kernel does not fuse quant; the fused silu_and_mul_fq kernel
    # handles activation + quant + scale-sort after the GEMM completes.
    _gemm_fq = fuse_fp4_quant and not _is_splitk
    _gemm_fss = fuse_sort_scale and not _is_splitk

    _kernel_out = tmp_out if _is_splitk else out
    is_fp4 = b_dtype == "fp4"
    _n_in = inter_dim * (2 if (is_fp4 and use_g1u1) else 1)
    # _n_in is used by kernel launch to compute gx.
    # Keep historical behavior:
    # - use_g1u1=True (gated): fp4 path uses 2*inter_dim, non-fp4 uses inter_dim
    # - use_g1u1=False  (non-gated): always inter_dim
    if use_g1u1:
        _n_in = inter_dim * 2 if is_fp4 else inter_dim
    else:
        _n_in = inter_dim
    _k_in = model_dim

    if is_fp4:
        args = _s1_args_fp4(
            _kernel_out.view(-1),
            a.view(-1),
            w1.view(-1),
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            out_scale_sorted_flat.view(-1),
            token_num,
            _n_in,
            _k_in,
            _grid_y,
            dev,
        )
    else:
        args = _s1_args_std(
            _kernel_out.view(-1),
            a.view(-1),
            w1.view(-1),
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            token_num,
            _n_in,
            _k_in,
            _grid_y,
        )

    exe = compile_flydsl_moe_stage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        act=act,
        use_g1u1=use_g1u1,
        persist_m=_persist_m,
        fuse_fp4_quant=_gemm_fq,
        fuse_sort_scale=_gemm_fss,
        use_async_copy=use_async_copy,
        use_cshuffle_epilog=False
        if use_cshuffle_epilog is None and tile_n % 128 != 0
        else use_cshuffle_epilog,
        k_batch=k_batch,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        gate_only=gate_only,
        n_per_wave=n_per_wave,
        splitk_mode=splitk_mode,
    )
    _run_compiled(exe, args)

    if _splitk_fq:
        _silu_fq = _get_compiled_silu_fq(inter_dim, topk)
        num_sorted_rows = sorted_token_ids.shape[0]
        _run_compiled(
            _silu_fq,
            (
                tmp_out.view(-1, inter_dim * 2),
                out.view(-1).view(torch.uint8),
                out_scale_sorted_flat,
                sorted_token_ids,
                num_valid_ids,
                token_num,
                num_sorted_rows,
                torch.cuda.current_stream(),
            ),
        )
    elif _is_splitk_std:
        # Non-fp4 split-K codegen path: tmp_out is f32.
        #   atomic mode: tmp_out shape (M, topk, 2*inter_dim) already reduced
        #                across kb in-place by the GEMM atomic-fadds.
        #   reduce mode: tmp_out shape (kb, M, topk, 2*inter_dim); sum across
        #                kb first so the gate/up slicing fed to silu_and_mul
        #                stays byte-identical to the atomic path.
        # The reduced (M*topk, 2*inter_dim) f32 buffer is then fed to the
        # fused `silu_and_mul` aiter kernel which takes f32 in / bf16|fp16
        # out in a single dispatch (silu(gate)*up + cast). This is the same
        # kernel the legacy non-_std split-K path below uses; the previous
        # PyTorch `(silu(gate) * up).to(out.dtype)` fallback launched 2-3
        # elementwise kernels per stage1 call instead of 1 fused one.
        # ``_is_splitk_std`` is already constrained to act=='silu'
        # (see compile_flydsl_moe_stage1); guard anyway in case that
        # constraint is relaxed to gelu in the future.
        from aiter.ops.flydsl._fused_post import (
            fused_kb_sum_silu_and_mul,
            is_disabled as _fused_post_disabled,
        )

        if _splitk_std_reduce:
            if _fused_post_disabled():
                # Fallback: explicit sum + aiter silu_and_mul (2 kernels).
                from aiter.ops.activation import silu_and_mul

                reduced = tmp_out.sum(dim=0)
                silu_and_mul(
                    out.view(-1, inter_dim),
                    reduced.view(-1, 2 * inter_dim),
                )
            else:
                # Fused triton: kb-sum + silu_and_mul in a single kernel,
                # no f32 reduced-buffer round-trip through HBM.
                # tmp_out is (kb, M, topk, 2*inter_dim) f32 contiguous; collapse
                # the (M, topk) axes so the kernel sees (kb, rows, 2*inter_dim).
                fused_kb_sum_silu_and_mul(
                    out.view(-1, inter_dim),
                    tmp_out.view(k_batch, -1, 2 * inter_dim),
                    inter_dim=inter_dim,
                )
        else:
            # Atomic mode: GEMM already reduced across kb in-place via
            # atomic-fadd, so tmp_out shape is (M, topk, 2*inter_dim) and we
            # just need silu_and_mul. The aiter HIP kernel is already a single
            # well-tuned launch; no win from re-implementing it in Triton here.
            from aiter.ops.activation import silu_and_mul

            silu_and_mul(
                out.view(-1, inter_dim),
                tmp_out.view(-1, 2 * inter_dim),
            )
    elif _is_splitk:
        if use_g1u1:
            if act == "gelu":
                from aiter.ops.activation import gelu_and_mul

                gelu_and_mul(out.view(-1, inter_dim), tmp_out.view(-1, inter_dim * 2))
            else:
                from aiter.ops.activation import silu_and_mul

                silu_and_mul(out.view(-1, inter_dim), tmp_out.view(-1, inter_dim * 2))
        else:
            # g1u0 split-K (fp4 path only): single-stream activation, no mul.
            # We deliberately do NOT use aiter's silu_and_mul / gelu_and_mul
            # here because those kernels require paired `[..., 2*d]` input
            # (gate||up) and there is no plain `silu(input)` / `gelu_tanh(input)`
            # host function exposed in aiter today (the internal
            # aiter::silu_kernel<T> + activation_kernel_vec template exist but
            # are only wrapped as `gelu_fast`, which uses a Pade approximation
            # that diverges from torch.nn.functional.gelu(approximate='tanh')).
            # Adding the missing bindings is a follow-up; for now use torch
            # directly on the native bf16/fp16 tmp_out -- torch.{silu,gelu}
            # compute internally in f32 anyway and return the same dtype, so
            # the explicit `.to(float32)` round-trip the legacy code did is
            # pure overhead (2 extra cast kernels per call).
            tmp_view = tmp_out.view(-1, inter_dim)
            if act == "gelu":
                out.copy_(
                    torch.nn.functional.gelu(tmp_view, approximate="tanh")
                    .view_as(out)
                )
            else:
                out.copy_(
                    torch.nn.functional.silu(tmp_view).view_as(out)
                )

    if fuse_fp4_quant:
        from aiter.utility.dtypes import fp8_e8m0

        out_scale_sorted = out_scale_sorted_flat.view(fp8_e8m0).view(
            padded_rows, padded_cols
        )
        return out, out_scale_sorted

    return out


def flydsl_moe_stage2(
    inter_states: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 128,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    mode: str = "atomic",
    w2_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
    sort_block_m: int = 0,
    persist: Optional[bool] = None,
    use_async_copy: bool = False,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    mfma_variant: Optional[str] = None,
    n_per_wave: int = 32,
    k_batch: int = 1,
) -> torch.Tensor:
    """Down-projection GEMM (MOE stage2). Supports atomic/reduce modes.

    a: (token_num, topk, inter_dim), w1: (E, model_dim, inter_dim) pre-shuffled.
    Returns (token_num, model_dim).

    sort_block_m: block_size used by moe_sorting / stage1. When 0 (default),
        assumed equal to tile_m. When set, stage2 can use a different tile_m
        from sorting/stage1.
    persist: if True, use persistent round-robin mode (grid_y=cu_num);
        if False, use legacy persist_m mode; if None, auto-select.

    k_batch: split-K factor along inter_dim (only honored in atomic mode and
        for non-fp4 b_dtype).  Default 1 = no split.  When k_batch>1, the
        kernel launches with a grid Z = k_batch and each WG processes only
        ``inter_dim/k_batch`` of the K reduction; partials are merged via the
        SAME atomic-add already used for topk accumulation.  Valid choices
        depend on tile_k and inter_dim divisibility.
    """

    token_num = inter_states.shape[0]
    E = w2.shape[0]
    model_dim = w2.shape[1]
    inter_dim = inter_states.shape[2]

    accumulate = mode != "reduce"

    if a_dtype == "fp4":
        inter_dim = inter_dim * 2

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    dev = inter_states.device

    # ------------------------------------------------------------------
    # Stage2 init: out zero-fill (atomic mode only, see flydsl_moe_stage2
    # contract) + flat_a_scale broadcast + flat_w_scale broadcast.  We fuse
    # all three into the same Triton kernel that stage1 uses.  When the
    # caller hands us an ``out`` we never touch it (their contract); when
    # the env kill switch is set we fall back to torch.zeros + two separate
    # _expand_per_tensor_scale launches like the historical path.
    # ------------------------------------------------------------------
    from aiter.ops.flydsl._fused_post import (
        fused_init as _fused_init,
        is_init_disabled as _fused_init_disabled,
    )

    _use_fused_init = not _fused_init_disabled()
    _alloc_out = out is None

    if _use_fused_init:
        if _alloc_out:
            out = torch.empty(
                (token_num, model_dim), dtype=torch_out_dtype, device=dev
            )
        flat_a_scale, _need_a = _resolve_a_scale_for_fused_init(
            a2_scale, token_num * topk, dev
        )
        flat_w_scale, _need_w = _resolve_w_scale_for_fused_init(
            w2_scale, E * model_dim, model_dim, dev
        )
        # Only zero `out` when we both allocated it AND the kernel needs a
        # zeroed accumulator (atomic mode).  Reduce mode writes its result
        # directly so an empty out is fine.
        _zero_out = _alloc_out and accumulate
        _fused_init(
            tmp_out=out if _zero_out else None,
            flat_a=flat_a_scale if _need_a else None,
            a_src=a2_scale if _need_a else None,
            flat_w=flat_w_scale if _need_w else None,
            w_src=w2_scale if _need_w else None,
            w_cols=model_dim if _need_w else 1,
        )
    else:
        if _alloc_out:
            alloc_fn = torch.zeros if accumulate else torch.empty
            out = alloc_fn(
                (token_num, model_dim), dtype=torch_out_dtype, device=dev
            )
        flat_a_scale = _expand_per_tensor_scale(a2_scale, token_num * topk, 1)
        if flat_a_scale is None:
            flat_a_scale = torch.empty(0, device=dev)
        flat_w_scale = _expand_per_tensor_scale(w2_scale, E * model_dim, model_dim)
        if flat_w_scale is None:
            flat_w_scale = torch.empty(0, device=dev)
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(sorted_token_ids.shape, dtype=torch.float32, device=dev)
    )

    _sbm = sort_block_m if sort_block_m > 0 else tile_m
    if _sbm == tile_m:
        m_blocks = min(sorted_expert_ids.shape[0], token_num * topk)
    else:
        total_sorted = sorted_expert_ids.shape[0] * _sbm
        m_blocks = (total_sorted + tile_m - 1) // tile_m
    # if persist is True:
    #     _persist_m = -1
    # elif persist is False:
    #     # _persist_m = 4 if m_blocks > 256 else 1
    #     _persist_m = 1  # _persist_m = 1 is better for g1u0
    # else:
    #     _persist_m = -1 if m_blocks > 256 else 1
    _persist_m = 1

    is_fp4 = b_dtype == "fp4"
    _n_in = model_dim
    _k_in = inter_dim

    target = out
    if not accumulate:
        target = torch.empty(
            (token_num * topk * model_dim,),
            device=out.device,
            dtype=out.dtype,
        )

    if is_fp4:
        args = _s2_args_fp4(
            target,
            inter_states,
            w2,
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            token_num,
            _n_in,
            _k_in,
            m_blocks,
            dev,
        )
    else:
        args = _s2_args_std(
            target,
            inter_states,
            w2,
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            token_num,
            _n_in,
            _k_in,
            m_blocks,
        )

    exe = compile_flydsl_moe_stage2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        accumulate=accumulate,
        persist_m=_persist_m,
        sort_block_m=sort_block_m,
        use_async_copy=use_async_copy,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        mfma_variant=mfma_variant,
        n_per_wave=n_per_wave,
        k_batch=k_batch,
    )
    _run_compiled(exe, args)

    if not accumulate:
        # Stage2 reduce mode: collapse topk axis. The fused Triton kernel
        # shaves the separate torch.sum dispatch (~2us at small M) and uses
        # the shared reduce autotune grid so large-M / large-model_dim
        # workloads also benefit (probes show ~11% over the prior heuristic
        # at int8 large M).  Kill-switch ``AITER_FLYDSL_FUSED_TOPK_OFF=1``
        # falls back to torch.sum.
        from aiter.ops.flydsl._fused_post import (
            fused_topk_sum as _fused_topk_sum,
            is_topk_sum_disabled as _fused_topk_disabled,
        )

        if _fused_topk_disabled():
            torch.sum(target.view(token_num, topk, model_dim), dim=1, out=out)
        else:
            _fused_topk_sum(
                out,
                target,
                token_num=token_num,
                topk=topk,
                model_dim=model_dim,
            )

    return out


def flydsl_moe_stage2_direct(
    inter_states: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 64,
    a_dtype: str = "fp8",
    b_dtype: str = "fp8",
    out_dtype: str = "bf16",
    w2_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    split_reduce: bool = False,
) -> torch.Tensor:
    """Small-M direct down-projection.

    Computes one token/N-tile per workgroup and reduces topk in-kernel:
    out[token, n] = sum_k A2[token, k, :] @ W2[topk_ids[token, k], n, :]
                    * a2_scale[token, k] * w2_scale[expert, n] * topk_weight[token, k]
    """

    if a_dtype != "fp8" or b_dtype != "fp8":
        raise ValueError("flydsl_moe_stage2_direct currently supports only fp8/fp8")
    if out_dtype != "bf16":
        raise ValueError("flydsl_moe_stage2_direct currently supports only bf16 output")

    token_num = inter_states.shape[0]
    E = w2.shape[0]
    model_dim = w2.shape[1]
    inter_dim = inter_states.shape[2]
    torch_out_dtype = torch.bfloat16
    if out is None:
        out = torch.empty(
            (token_num, model_dim), dtype=torch_out_dtype, device=inter_states.device
        )
    target = (
        _direct_stage2_workspace(out, token_num, topk, model_dim) if split_reduce else out
    )

    flat_a2_scale = a2_scale.view(-1) if a2_scale is not None else None
    a_scale_scalar = flat_a2_scale is not None and flat_a2_scale.numel() == 1
    flat_a_scale = (
        flat_a2_scale
        if a_scale_scalar
        else _expand_per_tensor_scale(a2_scale, token_num * topk, 1)
    )
    if flat_a_scale is None:
        raise ValueError("direct fp8 stage2 requires a2_scale")
    flat_w2_scale = w2_scale.view(-1) if w2_scale is not None else None
    w_scale_per_expert = flat_w2_scale is not None and flat_w2_scale.numel() == E
    flat_w_scale = (
        flat_w2_scale
        if w_scale_per_expert
        else _expand_per_tensor_scale(w2_scale, E * model_dim, model_dim)
    )
    if flat_w_scale is None:
        raise ValueError("direct fp8 stage2 requires w2_scale")

    stream = torch.cuda.current_stream()
    if split_reduce:
        from .kernels.moe_gemm_2stage_direct import compile_moe_gemm2_direct_smallm

        exe = compile_moe_gemm2_direct_smallm(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=E,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            in_dtype=a_dtype,
            out_dtype=out_dtype,
            a_scale_scalar=a_scale_scalar,
            w_scale_per_expert=w_scale_per_expert,
            split_reduce=True,
        )
    else:
        exe = compile_flydsl_moe_stage2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=E,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=True,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            direct=True,
            a_scale_scalar=a_scale_scalar,
            w_scale_per_expert=w_scale_per_expert,
            split_reduce=False,
        )
    if split_reduce:
        args = (
            _view_safe(out),
            _view_safe(target),
            _view_safe(inter_states),
            _view_safe(w2),
            _view_safe(flat_a_scale),
            _view_safe(flat_w_scale),
            topk_ids,
            topk_weights,
            token_num,
            model_dim,
            inter_dim,
            stream,
        )
    else:
        args = _s2_direct_args_std(
            target,
            inter_states,
            w2,
            flat_a_scale,
            flat_w_scale,
            topk_ids,
            topk_weights,
            token_num,
            model_dim,
            inter_dim,
        )
    _run_compiled(exe, args)
    return out


def flydsl_moe_stage1_direct(
    a: torch.Tensor,
    w1: torch.Tensor,
    topk_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 64,
    a_dtype: str = "fp8",
    b_dtype: str = "fp8",
    out_dtype: str = "bf16",
    w1_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    routes_per_block: int = 1,
    num_waves: int = 0,
    k_batch: int = 1,
    splitk_mode: str = "atomic",
) -> torch.Tensor:
    """Small-M direct stage1 for fp8/fp8 silu(gate)*up.

    ``k_batch`` enables split-K along the model_dim axis (mirrors the standard
    ``compile_flydsl_moe_stage1`` split-K codegen path).  When ``k_batch > 1``:
      * The kernel writes f32 pre-activation gate/up partials into a tmp
        buffer instead of the final bf16 silu(gate)*up.  Buffer layout:
          ``atomic`` -> (token_num, topk, 2*inter_dim) f32, atomic-fadd merged
          ``reduce`` -> (k_batch, token_num, topk, 2*inter_dim) f32, plain
                        stores per WG and host-side kb-axis sum.
      * The host post-pass runs aiter's fused ``silu_and_mul`` to fold the
        2-stream f32 buffer into the requested bf16 ``out`` in one dispatch.
      * ``k_batch == 1`` retains the legacy single-shot path with bf16 store
        and in-kernel silu(gate)*up.
    """

    if a_dtype != "fp8" or b_dtype != "fp8":
        raise ValueError("flydsl_moe_stage1_direct currently supports only fp8/fp8")
    if out_dtype != "bf16":
        raise ValueError("flydsl_moe_stage1_direct currently supports only bf16 output")
    if str(splitk_mode) not in ("atomic", "reduce"):
        raise ValueError(
            f"flydsl_moe_stage1_direct: splitk_mode must be 'atomic' or "
            f"'reduce', got {splitk_mode!r}"
        )

    token_num = a.shape[0]
    model_dim = a.shape[1]
    E = w1.shape[0]
    if w1.shape[1] % 2 != 0:
        raise ValueError("direct stage1 expects g1u1 W1 with shape[1] == 2*inter_dim")
    inter_dim = w1.shape[1] // 2
    if out is None:
        out = torch.empty(
            (token_num, topk, inter_dim), dtype=torch.bfloat16, device=a.device
        )

    flat_a1_scale = a1_scale.view(-1) if a1_scale is not None else None
    a_scale_scalar = flat_a1_scale is not None and flat_a1_scale.numel() == 1
    flat_a_scale = (
        flat_a1_scale
        if a_scale_scalar
        else _expand_per_tensor_scale(a1_scale, token_num, 1)
    )
    if flat_a_scale is None:
        raise ValueError("direct fp8 stage1 requires a1_scale")

    flat_w1_scale = w1_scale.view(-1) if w1_scale is not None else None
    w_scale_per_expert = flat_w1_scale is not None and flat_w1_scale.numel() == E
    flat_w_scale = (
        flat_w1_scale
        if w_scale_per_expert
        else _expand_per_tensor_scale(w1_scale, E * 2 * inter_dim, 2 * inter_dim)
    )
    if flat_w_scale is None:
        raise ValueError("direct fp8 stage1 requires w1_scale")

    from .kernels.moe_gemm_2stage_direct import compile_moe_gemm1_direct_smallm

    exe = compile_moe_gemm1_direct_smallm(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=a_dtype,
        out_dtype=out_dtype,
        a_scale_scalar=a_scale_scalar,
        w_scale_per_expert=w_scale_per_expert,
        routes_per_block=routes_per_block,
        num_waves_override=num_waves,
        k_batch=int(k_batch),
        splitk_mode=str(splitk_mode),
    )

    _is_splitk = int(k_batch) > 1
    _splitk_reduce = _is_splitk and str(splitk_mode) == "reduce"
    if _is_splitk:
        # f32 partial-sum tmp buffer.  zeros() is REQUIRED for both modes:
        #   atomic : the kernel atomic-fadd's into this buffer; needs a
        #            zero initial value so the kb-way reduction is correct.
        #   reduce : invalid (skipped) WGs leave their kb-slice untouched;
        #            the host sum across kb must see zeros for those.
        # We intentionally stay on ``torch.zeros`` here: micro-benchmarks
        # (HY3_BKC/scripts/_bench_direct_fused_init.py) show torch's HIP
        # zero-fill kernel runs at ~7us across the 6-860KB direct-stage1
        # shape range, while the Triton ``fused_init`` path costs ~19us
        # (~12us pure Python/dispatch overhead).  Patch 0004's fused_init
        # wins on the standard kernel only because it amortises three
        # helpers (zero + a-broadcast + w-broadcast) into one launch; the
        # direct kernel reads scalar a-scale and per-expert w-scale via
        # buffer_load, so there is no broadcast to fuse and the lone
        # zero-fill is faster as a stock torch op.
        if _splitk_reduce:
            tmp_out = torch.zeros(
                (int(k_batch), token_num, topk, 2 * inter_dim),
                dtype=torch.float32,
                device=a.device,
            )
        else:
            tmp_out = torch.zeros(
                (token_num, topk, 2 * inter_dim),
                dtype=torch.float32,
                device=a.device,
            )

        args = _s1_direct_args_std(
            tmp_out,
            a,
            w1,
            flat_a_scale,
            flat_w_scale,
            topk_ids,
            token_num,
            inter_dim,
            model_dim,
        )
        _run_compiled(exe, args)

        # Reduce mode post-pass: kb-axis sum + silu_and_mul.  The fused Triton
        # kernel collapses these two launches into one and skips the f32
        # ``reduced`` materialisation through HBM (saves ~5-8us at small M
        # vs the legacy ``tmp_out.sum(dim=0) + silu_and_mul`` two-kernel
        # path; see aiter.ops.flydsl._fused_post for the calibration data).
        # Atomic mode already merged kb in-kernel via atomic-fadd, so it
        # only needs the single-launch aiter silu_and_mul.
        # Env kill-switch ``AITER_FLYDSL_FUSED_SILU_OFF=1`` falls back.
        from aiter.ops.flydsl._fused_post import (
            fused_kb_sum_silu_and_mul as _fused_kb_silu,
            is_disabled as _fused_post_disabled,
        )

        if _splitk_reduce and not _fused_post_disabled():
            _fused_kb_silu(
                out.view(-1, inter_dim),
                tmp_out.view(int(k_batch), -1, 2 * inter_dim),
                inter_dim=inter_dim,
            )
        else:
            from aiter.ops.activation import silu_and_mul

            # Reduce-mode fallback: explicit sum + silu_and_mul (2 kernels).
            # Atomic mode: tmp_out is already (M, topk, 2*inter) f32 after
            # in-kernel atomic-fadd; just silu_and_mul.
            reduced = tmp_out.sum(dim=0) if _splitk_reduce else tmp_out
            silu_and_mul(
                out.view(-1, inter_dim),
                reduced.view(-1, 2 * inter_dim),
            )
        return out

    args = _s1_direct_args_std(
        out,
        a,
        w1,
        flat_a_scale,
        flat_w_scale,
        topk_ids,
        token_num,
        inter_dim,
        model_dim,
    )
    _run_compiled(exe, args)
    return out
