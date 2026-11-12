# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""NVFP4 routed-expert MoE for gfx950 (MiniMax-M3 and other SwiGLU-OAI shapes).

The kernels (csrc/kernels/nvfp4_moe/moe_nvfp4_kernels.hip) decode NVFP4 weights
to f16 in registers (exact), apply the E4M3 block scale and the per-expert
global multiplier, and run f16 MFMA on f16 or bf16 -> f16 activations (no
activation quant). Launches per MoE: fused sort + cast + zeroing, stage 1
(gate/up + SwiGLU-OAI), stage 2 (down, weighted sum over the top-k slots). The
stage-2 kernel selects the sum: atomics into the zeroed output (bf16 output,
_outbf16 kernels) or into a zeroed f32 workspace cast to the f16 output once
(one more launch), both not bitwise deterministic; or (_reduce kernels, any
output dtype) plain f32 stores of each (token, slot) row and a top-k reduce
kernel that sums them in slot order (deterministic, no zeroing). The
E4M3 scales of each weight tensor are stored once, grouped by GS K-tiles
(scale_gs of its K).

Fallback path (T * topk above MAX_SORT_ROWS, more than MAX_SORT_EXPERTS experts,
or fallback=True): AITER's opus moe_sorting (sort + output zeroing, any T) and
stage-1 kernels that read the bf16 activations and cast them to f16 themselves
(the _inbf16 kernels, same rounding: stage-1 output bit-identical to the main
path; f16 activations use f16 stage-1 kernels). Same weights and scale layouts
as the main path.

Kernel choice: the tuned CSV, else the default rule (moe_nvfp4_dispatch.py).

Usage:
    w = NvFp4MoeWeights(w13, sc13, w2, sc2, g13, g2)  # once per layer
    y = moe_nvfp4(w, x, topk_ids, topk_w)            # per step

moe_nvfp4 launches on the current stream, allocates through the torch
allocator and does no host sync, so it is CUDA-graph capturable. The one-time
setup runs outside the step: the JIT build of module_moe_nvfp4 in
NvFp4MoeWeights (weight-processing time), the kernel tables of a (shape, topk,
dtype) key on its first moe_nvfp4 call (warmup; with the JIT build of its CSV
kernels outside the module).
"""

import contextlib
import functools
import os
import threading

import torch
from torch import Tensor

from aiter import logger

from ..jit.core import (
    AITER_CONFIG_MOE_NVFP4,
    AITER_CSRC_DIR,
    AITER_REBUILD,
    JIT_LIB_EXT,
    _needs_arch_rebuild,
    build_module,
    compile_ops,
    get_args_of_build,
    get_user_jit_dir,
)
from ..jit.utils.chip_info import get_cu_num, get_gfx_runtime
from ..jit.utils.moe_nvfp4_common import (
    MAX_SORT_EXPERTS,
    MAX_SORT_ROWS,
    parse_kernel,
    scale_gs,
)
from .moe_nvfp4_dispatch import default_rows, read_dispatch_csv, uses_fallback
from .moe_sorting_opus import (
    moe_sorting_opus_fwd,
    moe_sorting_opus_get_workspace_size,
)
from .shuffle import shuffle_scale_nvfp4_moe, shuffle_weight

MD_NAME = "module_moe_nvfp4"
# Largest grid y (one row per sorted block of BM rows) HIP launches.
MAX_GRID_Y = 65536


@compile_ops(MD_NAME, fc_name="moe_nvfp4_sort", ffi_type="ctypes")
def _moe_nvfp4_sort(
    x: Tensor,
    x_f16: Tensor,
    out: Tensor,
    topk_ids: Tensor,
    topk_w: Tensor,
    sorted_ids: Tensor,
    sorted_w: Tensor,
    sorted_eids: Tensor,
    num_valid: Tensor,
    num_experts: int,
    block_m: int,
    nblk: int,
) -> None: ...


@compile_ops(MD_NAME, fc_name="moe_nvfp4_reduce", ffi_type="ctypes")
def _moe_nvfp4_reduce(ws: Tensor, out: Tensor, topk: int) -> None: ...


# The entries below are bound per JIT module by _module_ops: module_moe_nvfp4 or the
# module of one kernel (_jit_module).


# Index of kernel name in the module's kernel table, -1 if it has none
def _moe_nvfp4_cfg_index(name: str) -> int: ...


# Field _KERNEL_FIELDS[k] of kernel i
def _moe_nvfp4_cfg_field(i: int, k: int) -> int: ...


# 1 if kernel i runs reduction length K, else 0
def _moe_nvfp4_cfg_fits(i: int, K: int) -> int: ...


def _moe_nvfp4_gemm(
    kernel: int,
    x: Tensor,
    w: Tensor,
    w_sc: Tensor,
    inv_gs: Tensor,
    sorted_ids: Tensor,
    sorted_w: Tensor,
    sorted_eids: Tensor,
    num_valid: Tensor,
    out: Tensor,
    T: int,
    topk: int,
    K: int,
    ntiles: int,
    n_out: int,
    grid_x: int,
    grid_y: int,
) -> None: ...


@functools.cache
def _module_ops(md: str) -> dict:
    """ctypes callers of the per-module entries of JIT module md, by stub name."""
    stubs = (_moe_nvfp4_cfg_index, _moe_nvfp4_cfg_field, _moe_nvfp4_cfg_fits)
    stubs += (_moe_nvfp4_gemm,)
    return {
        f.__name__: compile_ops(md, fc_name=f.__name__[1:], ffi_type="ctypes")(f)
        for f in stubs
    }


# Kernel fields read by _moe_nvfp4_cfg_field, in CfgField order (moe_nvfp4_entry.h)
_KERNEL_FIELDS = (
    "s1", "m_tiles", "n_tiles", "waves", "gs", "inbf16", "outbf16", "reduce"
)  # fmt: skip

_LOCK = threading.Lock()
# Tuned CSV rows (read_dispatch_csv), loaded by prepare
_DISPATCH: dict[tuple, tuple] | None = None
# keys of the default-rule picks already logged by _pick
_WARNED: set[tuple] = set()


def kernel_stage(name: str) -> int:
    """Stage (1 or 2) of a kernel, from its nvfp4_moe_s1_ / nvfp4_moe_s2_ prefix."""
    for stage in (1, 2):
        if name.startswith(f"nvfp4_moe_s{stage}_"):
            return stage
    raise ValueError(f"NVFP4 MoE kernel name {name!r} has no stage prefix")


@contextlib.contextmanager
def _gfx950_archs():
    # The kernels are gfx950-only: pin a multi-arch GPU_ARCHS to gfx950 while
    # a first call JIT-builds a module.
    archs = os.environ.get("GPU_ARCHS")
    pin = archs is not None and archs not in ("native", "gfx950")
    if pin:
        os.environ["GPU_ARCHS"] = "gfx950"
    try:
        yield
    finally:
        if pin:
            os.environ["GPU_ARCHS"] = archs


def _kernel_entry(md: str, name: str) -> dict | None:
    """Kernel table entry of kernel name in JIT module md (its launch and K-fit
    callers and launch fields), None if md has no such kernel."""
    ops = _module_ops(md)
    with _gfx950_archs():
        i = ops["_moe_nvfp4_cfg_index"](name)
    if i < 0:
        return None
    fields = {
        f: ops["_moe_nvfp4_cfg_field"](i, k) for k, f in enumerate(_KERNEL_FIELDS)
    }
    stage = kernel_stage(name)
    if stage != (1 if fields.pop("s1") else 2):
        raise RuntimeError(f"NVFP4 MoE kernel {name} is built for the other stage")
    return {
        "idx": i,
        "md": md,
        "name": name,
        "stage": stage,
        "gemm": ops["_moe_nvfp4_gemm"],
        "fits": functools.partial(ops["_moe_nvfp4_cfg_fits"], i),
        **fields,
    }


def _jit_module(name: str) -> str:
    """JIT module of one kernel outside module_moe_nvfp4 (module_moe_nvfp4 sources,
    instances of that kernel only), built on first use, then cached on disk."""
    md = f"module_moe_nvfp4_k_{name}"
    so = os.path.join(get_user_jit_dir(), f"{md}{JIT_LIB_EXT}")
    if AITER_REBUILD or not os.path.exists(so) or _needs_arch_rebuild(md):
        d = get_args_of_build(MD_NAME)
        gen = f"{AITER_CSRC_DIR}/kernels/nvfp4_moe/gen_instances.py"
        with _gfx950_archs():
            build_module(
                md,
                d["srcs"],
                d["flags_extra_cc"],
                d["flags_extra_hip"],
                [f"{gen} --working_path {{}} --names {name}"],
                d["extra_include"],
                d["extra_ldflags"],
                d["verbose"],
                d["is_python_module"],
                d["is_standalone"],
                True,
                d.get("third_party", []),
            )
    return md


@functools.cache
def _main_kernel(name: str) -> dict | None:
    return _kernel_entry(MD_NAME, name)


@functools.cache
def _kernel(name: str) -> dict:
    """Kernel table entry of a kernel of module_moe_nvfp4, else of its own module."""
    parse_kernel(name)  # a bad name: ValueError before any build
    kernel = _main_kernel(name) or _kernel_entry(_jit_module(name), name)
    if kernel is None:
        raise RuntimeError(f"NVFP4 MoE: kernel {name} missing from its JIT module")
    return kernel


def prepare(device: torch.device | int | None = None) -> None:
    """Check that the GPU is gfx950, build/load module_moe_nvfp4 and read the tuned
    CSV. Call outside graph capture."""
    global _DISPATCH
    with _LOCK:
        if _DISPATCH is not None:
            return
        arch = get_gfx_runtime()
        if arch != "gfx950":
            raise RuntimeError(f"NVFP4 MoE kernels need gfx950, got {arch}")
        _kernel_entry(MD_NAME, "")  # JIT build / load (no kernel is named "")
        dispatch = read_dispatch_csv(AITER_CONFIG_MOE_NVFP4, get_cu_num())
        with torch.cuda.device(device):
            # JIT build / load of the fallback path's sort, outside capture
            moe_sorting_opus_get_workspace_size(1, 1, 1)
        _DISPATCH = dispatch


def _n_group(kernel):
    """N / 16 output tiles of a workgroup (stage 1 n_tiles: half gate, half up)."""
    tiles = kernel["n_tiles"] // 2 if kernel["stage"] == 1 else kernel["n_tiles"]
    return tiles * kernel["waves"]


def _round8(n):
    return (n + 7) // 8 * 8


def _first_fit(names: list, K: int, N: int):
    """The first kernel of names in module_moe_nvfp4 that runs reduction length K and
    whose N group divides the N / 16 output tiles, else None."""
    for n in names:
        kernel = _main_kernel(n)
        if kernel and (N // 16) % _n_group(kernel) == 0 and kernel["fits"](K):
            return kernel
    return None


def _check_shape(H: int, I: int) -> None:
    """Raises ValueError for a shape without kernels: H or I not a multiple of 128,
    an odd stage-1 K-tile count or an odd stage-2 K-tile count other than 3."""
    if H % 128 or I % 128:
        raise ValueError(f"NVFP4 MoE: H={H} and I={I} must be multiples of 128")
    if H // 128 % 2:
        raise ValueError(f"NVFP4 MoE: H={H}: no stage-1 kernels for odd H / 128")
    if I // 128 % 2 and I // 128 != 3:
        raise ValueError(f"NVFP4 MoE: I={I}: odd I / 128 is supported for 3 only")


def _check_row(key: tuple, t: int, kernel1: dict, kernel2: dict) -> None:
    """Raises ValueError for a tuned CSV row whose kernels cannot run its shape:
    another BM per stage (one sort serves both), a scale group size other than
    scale_gs of the stage K (the weight layout), an N group that does not divide
    the N / 16 output tiles, or a kernel that does not run the stage K."""
    H, I = key[:2]
    if kernel1["m_tiles"] != kernel2["m_tiles"]:
        raise ValueError(
            f"NVFP4 MoE dispatch {key} token={t}: stage-1 BM"
            f" {16 * kernel1['m_tiles']} != stage-2 BM {16 * kernel2['m_tiles']}"
        )
    for kernel, K, N in ((kernel1, H, I), (kernel2, I, H)):
        bad = None
        if kernel["gs"] != scale_gs(K // 128):
            gs = scale_gs(K // 128)
            bad = f"scale GS {kernel['gs']}, the K={K} weights have GS {gs}"
        elif (N // 16) % _n_group(kernel):
            bad = f"N group {16 * _n_group(kernel)} does not divide N={N}"
        elif not kernel["fits"](K):
            bad = f"does not run K={K}"
        if bad:
            raise ValueError(
                f"NVFP4 MoE dispatch {key} token={t}: {kernel['name']}: {bad}"
            )


def _default_table(H: int, I: int, x_dtype: str, out_dtype: str, fallback: bool):
    """default_rows with the first fitting kernel per stage; a row without kernels
    for both stages is dropped."""
    rows = []
    for t, s1, s2 in default_rows(H, I, x_dtype, out_dtype, fallback):
        kernel1, kernel2 = _first_fit(s1, H, I), _first_fit(s2, I, H)
        if kernel1 and kernel2:
            rows.append((t, kernel1, kernel2))
    if not rows:
        raise ValueError(f"NVFP4 MoE: no kernels for I={I}, H={H}")
    return tuple(rows)


@functools.cache
def _tables(H: int, I: int, E: int, topk: int, dtype: str) -> dict:
    """{fallback: (CSV rows, default-rule rows)} of a key, (T, stage-1 kernel,
    stage-2 kernel) rows. Both paths are resolved on the first call of a key. CSV
    rows: the tuned CSV rows of the key on the path of their own T (_check_row)."""
    _check_shape(H, I)
    key = (H, I, E, topk, dtype)
    out = {}
    for fb in (False, True):
        rows = []
        for t, c1, c2 in _DISPATCH.get(key, ()):
            if uses_fallback(t, topk, E) == fb:
                kernel1, kernel2 = _kernel(c1), _kernel(c2)
                _check_row(key, t, kernel1, kernel2)
                rows.append((t, kernel1, kernel2))
        out[fb] = (tuple(rows), _default_table(H, I, dtype, dtype, fb))
    return out


def _pick(W, T: int, topk: int, dtype: str, fallback: bool) -> tuple[dict, dict]:
    """(stage-1, stage-2) kernel table entries: the first CSV row of the path with
    T <= its T, else the first default-rule row with T <= its T, else its last row.
    A default-rule pick is logged once per shape, path and next-power-of-two T."""
    csv_rows, rule = _tables(W.H, W.I, W.E, topk, dtype)[fallback]
    for t, kernel1, kernel2 in csv_rows:
        if T <= t:
            return kernel1, kernel2
    kernel1, kernel2 = next(((c1, c2) for t, c1, c2 in rule if T <= t), rule[-1][1:])
    key = (W.H, W.I, W.E, topk, dtype, fallback, 1 << (T - 1).bit_length())
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(
            f"[moe_nvfp4] no tuned row for T={T} (bucket {key[-1]}; H={W.H} I={W.I}"
            f" E={W.E} topk={topk} dtype={dtype}"
            f" path={'fallback' if fallback else 'main'}, tuned up to"
            f" T={csv_rows[-1][0] if csv_rows else 'none'}); using default rule:"
            f" s1={kernel1['name']} s2={kernel2['name']}"
        )
    return kernel1, kernel2


class NvFp4MoeWeights:
    """One MoE layer in the F16 kernel layout.

    The weight scales of each tensor are in the GS layout of its K-tile count
    (scale_gs: one layout per tensor, read by the main and the fallback path).

    Args:
        w13: packed fp4 uint8 [E, 2I, H/2] (gate rows then up rows).
        sc13: E4M3 block scales [E, 2I, H/16] (any 1-byte dtype).
        w2: packed fp4 uint8 [E, H, I/2].
        sc2: E4M3 block scales [E, H, I/16].
        g13: fp32 [E, 2] multipliers (1 / weight_global_scale of gate, up).
        g2: fp32 [E] multipliers (1 / weight_global_scale of down).
        shuffled: the weights are in the shuffle_weight(w, (16, 32)) layout and the
            scales in the shuffle_scale_nvfp4_moe layout (wrapped, no copy).

    Raises:
        ValueError: a shape without kernels (_check_shape): H or I not a
            multiple of 128, H / 128 odd, or I / 128 odd and not 3.
    """

    def __init__(self, w13, sc13, w2, sc2, g13, g2, shuffled=False):
        prepare(w13.device)
        self.E, N13, H2 = w13.shape
        self.I, self.H = N13 // 2, H2 * 2
        _check_shape(self.H, self.I)
        self.g13 = g13.float().contiguous()
        self.g2 = g2.float().contiguous()
        if shuffled:
            self.w13, self.w2, self.s13, self.s2 = w13, w2, sc13, sc2
            return
        self.w13 = shuffle_weight(w13, (16, 32))
        self.w2 = shuffle_weight(w2, (16, 32))
        self.s13 = shuffle_scale_nvfp4_moe(sc13)
        self.s2 = shuffle_scale_nvfp4_moe(sc2)


def _fused_moe_nvfp4(req) -> Tensor:
    """fused_moe full_impl of fp4x2 weights with w1/w2_global_scale (FusedMoeRequest
    req): its shuffled weights and scales wrapped as NvFp4MoeWeights."""
    from aiter import ActivationType

    unsupported = None
    if req.expert_mask is not None:
        unsupported = "expert_mask (EP)"
    elif req.bias1 is not None or req.bias2 is not None:
        unsupported = "per-expert bias"
    elif req.doweight_stage1:
        unsupported = "doweight_stage1=True"
    elif req.a1_scale is not None or req.a2_scale is not None:
        unsupported = "prequantized activations"
    elif req.num_local_tokens is not None:
        unsupported = "num_local_tokens"
    elif req.hidden_pad or req.intermediate_pad:
        unsupported = "hidden/intermediate padding"
    elif req.gate_mode != "separated":
        unsupported = f"gate mode {req.gate_mode!r}"
    elif req.activation != ActivationType.Swiglu:
        unsupported = f"activation {req.activation}"
    elif req.swiglu_limit not in (None, 7.0):
        unsupported = f"swiglu_limit {req.swiglu_limit} (the kernels clamp at 7.0)"
    elif req.dtype != req.hidden_states.dtype:
        unsupported = f"output dtype {req.dtype} != {req.hidden_states.dtype}"
    if unsupported is not None:
        raise ValueError(f"NVFP4 fused_moe: {unsupported} is not supported")
    if req.w1_global_scale is None or req.w2_global_scale is None:
        raise ValueError("NVFP4 fused_moe needs w1_global_scale and w2_global_scale")
    E = req.w1.shape[0]
    for name, w, sc in (("w1", req.w1, req.w1_scale), ("w2", req.w2, req.w2_scale)):
        N, K = w.shape[1], w.shape[2] * 2
        if not getattr(w, "is_shuffled", False):
            raise ValueError(f"NVFP4 fused_moe: {name} is not shuffled (is_shuffled)")
        if (
            sc is None
            or sc.element_size() != 1
            or sc.shape[0] != E
            or sc.numel() != E * N * ((K // 16 + 15) // 16 * 16)
        ):
            raise ValueError(
                f"NVFP4 fused_moe: {name}_scale is not 1-byte E4M3 in the"
                f" shuffle_scale_nvfp4_moe layout of [{E}, {N}, {K // 16}]"
            )
    W = NvFp4MoeWeights(
        req.w1,
        req.w1_scale,
        req.w2,
        req.w2_scale,
        req.w1_global_scale,
        req.w2_global_scale,
        shuffled=True,
    )
    return moe_nvfp4(W, req.hidden_states, req.topk_ids, req.topk_weight)


def _gemm(kernel, sort, T, topk, E, x, w, w_sc, inv_gs, out, K, ntiles, n_out):
    """One GEMM stage of kernel on the sorted rows (sort: the four sort buffers)."""
    BM = 16 * kernel["m_tiles"]
    n = T * topk
    kernel["gemm"](
        kernel["idx"], x, w, w_sc, inv_gs, *sort, out, T, topk, K, ntiles, n_out,
        n_out // 16 // _n_group(kernel), _round8(min(E, n) + (n + BM - 1) // BM),
    )  # fmt: skip


def moe_nvfp4(
    W: NvFp4MoeWeights,
    x: Tensor,
    topk_ids: Tensor,
    topk_w: Tensor,
    fallback: bool | None = None,
) -> Tensor:
    """Routed-expert sum (no routed scaling).

    Args:
        W: layer weights from NvFp4MoeWeights.
        x: bf16 or f16 [T, H] activations.
        topk_ids: [T, topk] expert ids (any integer dtype).
        topk_w: [T, topk] routing weights (any float dtype).
        fallback: None: the fallback path iff uses_fallback(T, topk, E); True /
            False: force it on / off.

    Returns:
        [T, H] in the x dtype.

    Raises:
        ValueError: x not bf16 or f16 [T, W.H], topk above 127 (the slot field
            of the sorted ids), fallback=False for a shape the fused sort cannot
            take, a shape without default-rule kernels, or a shape over the
            kernel limits (32-bit byte offsets: T * H and T * topk * I
            below 2^30, T * H below 2^29 for f16 x with stage-2 atomics; at
            most MAX_GRID_Y sorted blocks).
    """
    T, H = x.shape
    topk = topk_ids.shape[1]
    bf16 = x.dtype == torch.bfloat16
    if not (bf16 or x.dtype == torch.float16) or H != W.H:
        raise ValueError(
            f"NVFP4 MoE: x must be bf16 or f16 [T, {W.H}], got {x.dtype} {H}"
        )
    if topk > 127:
        raise ValueError(f"NVFP4 MoE: topk={topk} above 127")
    if fallback is None:
        fallback = uses_fallback(T, topk, W.E)
    elif not fallback and uses_fallback(T, topk, W.E):
        raise ValueError(
            f"NVFP4 MoE main path supports T * topk <= {MAX_SORT_ROWS} and"
            f" E <= {MAX_SORT_EXPERTS}"
        )
    kernel1, kernel2 = _pick(W, T, topk, "bf16" if bf16 else "f16", fallback)
    BM = 16 * kernel1["m_tiles"]
    dev = x.device
    x = x.contiguous()
    n = T * topk
    nblk = min(W.E, n) + (n + BM - 1) // BM
    reduce = kernel2["reduce"]
    # 32-bit byte offsets of both stages' X and the stage-2 atomics target (_reduce
    # rows: 64-bit), grid y (sorted blocks)
    if (
        max(T * H * (1 if bf16 or reduce else 2), n * W.I) * 2 >= 1 << 31
        or _round8(nblk) > MAX_GRID_Y
    ):
        raise ValueError(f"NVFP4 MoE: T={T} topk={topk} exceeds the kernel limits")
    out = torch.empty(T, H, dtype=x.dtype, device=dev)
    # stage-2 target: the (token, slot) f32 rows of the top-k reduce, the bf16 output
    # (atomics) or an f32 workspace (f16 output: f16 atomics overflow on partial sums)
    if reduce:
        acc = torch.empty(n, H, dtype=torch.float32, device=dev)
    else:
        acc = out if bf16 else torch.empty(T, H, dtype=torch.float32, device=dev)
    # zeroed by the sort (atomics targets only)
    zero = torch.empty(T, 0, dtype=x.dtype, device=dev) if reduce else acc
    x_inter = torch.empty(n, W.I, dtype=torch.float16, device=dev)
    ids = topk_ids.to(torch.int32).contiguous()
    tw = topk_w.float().contiguous()
    if fallback:
        # The opus sort fills up to n + E * BM - topk rows; the kernels read
        # sorted_eids for every grid row (nblk rounded up to 8).
        nblk = max(_round8(nblk), (n + W.E * BM - topk + BM - 1) // BM)
        sorted_ids = torch.empty(nblk * BM, dtype=torch.int32, device=dev)
        sorted_w = torch.empty(nblk * BM, dtype=torch.float32, device=dev)
        sorted_eids = torch.empty(nblk, dtype=torch.int32, device=dev)
        num_valid = torch.empty(2, dtype=torch.int32, device=dev)
        ws_bytes = moe_sorting_opus_get_workspace_size(T, W.E, topk, 0, dev.index)
        ws = torch.empty(ws_bytes, dtype=torch.uint8, device=dev) if ws_bytes else None
        moe_sorting_opus_fwd(
            ids, tw, sorted_ids, sorted_w, sorted_eids, num_valid, zero, W.E, BM,
            None, None, ws, 0,
        )  # fmt: skip
        x1 = x
    else:
        sorted_ids = torch.empty(nblk * BM, dtype=torch.int32, device=dev)
        sorted_w = torch.empty(nblk * BM, dtype=torch.float32, device=dev)
        sorted_eids = torch.empty(_round8(nblk), dtype=torch.int32, device=dev)
        num_valid = torch.empty(1, dtype=torch.int32, device=dev)
        # f16 x: no cast (empty x_f16)
        x_f16 = torch.empty(T * H if bf16 else 0, dtype=torch.float16, device=dev)
        _moe_nvfp4_sort(
            x, x_f16, zero, ids, tw, sorted_ids, sorted_w, sorted_eids, num_valid,
            W.E, BM, nblk,
        )  # fmt: skip
        x1 = x_f16 if bf16 else x
    sort = (sorted_ids, sorted_w, sorted_eids, num_valid)
    # stage 1: x [T, H] -> x_inter [T * topk, I], stage 2: x_inter -> acc
    _gemm(kernel1, sort, T, topk, W.E, x1, W.w13, W.s13, W.g13, x_inter,
          K=W.H, ntiles=2 * W.I // 16, n_out=W.I)  # fmt: skip
    _gemm(kernel2, sort, T, topk, W.E, x_inter, W.w2, W.s2, W.g2, acc,
          K=W.I, ntiles=W.H // 16, n_out=W.H)  # fmt: skip
    if reduce:
        _moe_nvfp4_reduce(acc, out, topk)
        return out
    return out if bf16 else out.copy_(acc)
