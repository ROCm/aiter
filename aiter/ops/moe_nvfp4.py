# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""NVFP4 routed-expert MoE for gfx950 (MiniMax-M3 and other SwiGLU-OAI shapes).

The kernels (csrc/kernels/nvfp4_moe/moe_nvfp4_kernels.hip) decode NVFP4 weights
to f16 in registers (exact), apply the E4M3 block scale and the per-expert
global multiplier, and run f16 MFMA on f16 or bf16 -> f16 activations (no
activation quant). Three launches per MoE: fused sort + cast + output zeroing,
stage 1 (gate/up + SwiGLU-OAI), stage 2 (down, weighted atomic reduce into the
output, so results are not bitwise deterministic). f16 x: stage 2 reduces into
a zeroed f32 workspace, cast to the f16 output once (two more launches). The
E4M3 scales of each weight tensor are stored once, grouped by GS K-tiles
(scale_gs of its K).

Fallback path (T * topk above MAX_SORT_ROWS, more than MAX_SORT_EXPERTS experts,
or fallback=True): AITER's opus moe_sorting (sort + output zeroing, any T) and
stage-1 kernels that read the bf16 activations and cast them to f16 themselves
(the _inbf16 kernels, same rounding: stage-1 output bit-identical to the main
path; f16 activations use f16 stage-1 kernels). Same weights and scale layouts
as the main path.

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
import csv
import functools
import os
import re
import threading

import torch
from torch import Tensor

from aiter import logger

from ..jit.core import (
    AITER_CSRC_DIR,
    AITER_REBUILD,
    AITER_ROOT_DIR,
    compile_ops,
    rm_module,
)
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.moe_nvfp4_common import (
    MAX_SORT_EXPERTS,
    MAX_SORT_ROWS,
    default_rows,
    parse_kernel,
    uses_fallback,
)
from .moe_sorting_opus import (
    moe_sorting_opus_fwd,
    moe_sorting_opus_get_workspace_size,
)
from .shuffle import shuffle_scale_nvfp4_moe, shuffle_weight

MD_NAME = "module_moe_nvfp4"
# Rows: cu_num,H,I,E,topk,x_dtype,out_dtype,T,stage1,stage2 (dtypes bf16 / f16). A
# call takes the first row of its exact key and path (uses_fallback of the row T)
# with T >= its T; above the last row or without rows, the default rule
# (aiter/jit/utils/moe_nvfp4_common.py). Kernel names: kernel_name in that file
# (the name is the config; nvfp4_moe_s1_ / nvfp4_moe_s2_ prefix:
# the stage), _inbf16 stage 1 for bf16 x on the fallback path, _outbf16 stage 2 for
# bf16 output. Only the rows of the device's cu_num are used. A kernel outside
# module_moe_nvfp4 is JIT-built in its own module (_jit_kernel).
DISPATCH_CSV = os.getenv(
    "AITER_CONFIG_MOE_NVFP4",
    f"{AITER_ROOT_DIR}/aiter/configs/moe_nvfp4_dispatch.csv",
)
# Largest grid y (one row per sorted block of BM rows) HIP launches.
MAX_GRID_Y = 65536


@compile_ops(MD_NAME, fc_name="moe_nvfp4_num_cfgs", develop=True)
def _moe_nvfp4_num_cfgs() -> int: ...


@compile_ops(MD_NAME, fc_name="moe_nvfp4_cfg_name", develop=True)
def _moe_nvfp4_cfg_name(i: int) -> str: ...


# Field _KERNEL_FIELDS[k] of kernel i
@compile_ops(MD_NAME, fc_name="moe_nvfp4_cfg_field", develop=True)
def _moe_nvfp4_cfg_field(i: int, k: int) -> int: ...


# Whether kernel i runs reduction length K
@compile_ops(MD_NAME, fc_name="moe_nvfp4_cfg_fits", develop=True)
def _moe_nvfp4_cfg_fits(i: int, K: int) -> bool: ...


@compile_ops(MD_NAME, fc_name="moe_nvfp4_sort", develop=True)
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


@compile_ops(MD_NAME, fc_name="moe_nvfp4_gemm", develop=True)
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


# Kernel fields read by _moe_nvfp4_cfg_field, in CfgField order (moe_nvfp4_entry.h)
_KERNEL_FIELDS = ("s1", "m_tiles", "n_tiles", "waves", "gs", "inbf16", "outbf16")

_LOCK = threading.Lock()
# kernel name -> kernel index and launch fields
_KERNELS: dict[str, dict] | None = None
_DEVICES: set[int] = set()
# DISPATCH_CSV rows (_read_dispatch_csv), loaded by prepare
_DISPATCH: dict[tuple, tuple] = {}
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


def _kernel_entry(i: int, build: dict | None = None) -> dict:
    """Kernel table entry of kernel i of module_moe_nvfp4, or of the module of the
    custom_build_args build."""
    fields = {
        f: _moe_nvfp4_cfg_field(i, k, custom_build_args=build)
        for k, f in enumerate(_KERNEL_FIELDS)
    }
    name = _moe_nvfp4_cfg_name(i, custom_build_args=build)
    stage = kernel_stage(name)
    if stage != (1 if fields.pop("s1") else 2):
        raise RuntimeError(f"NVFP4 MoE kernel {name} is built for the other stage")
    return {"idx": i, "build": build, "name": name, "stage": stage, **fields}


def _build_kernels() -> dict[str, dict]:
    with _gfx950_archs():
        n = _moe_nvfp4_num_cfgs()
    out = {}
    for i in range(n):
        kernel = _kernel_entry(i)
        if kernel["name"] in out:
            raise RuntimeError(f"duplicate NVFP4 MoE kernel {kernel['name']}")
        out[kernel["name"]] = kernel
    return out


@functools.cache
def _jit_kernel(name: str) -> dict:
    """Kernel table entry of a kernel outside module_moe_nvfp4, from its own JIT
    module (built on first use, then cached on disk)."""
    parse_kernel(name)  # a bad name: ValueError before any build
    md = f"module_moe_nvfp4_k_{name}"
    if AITER_REBUILD:
        # compile_ops rebuilds only the default module (module_moe_nvfp4)
        rm_module(md)
    gen = f"{AITER_CSRC_DIR}/kernels/nvfp4_moe/gen_instances.py"
    cmd = f"{gen} --working_path {{}} --names {name}"
    with _gfx950_archs():
        return _kernel_entry(0, {"md_name": md, "blob_gen_cmd": [cmd]})


def _kernel(name: str) -> dict:
    return _KERNELS.get(name) or _jit_kernel(name)


def _read_dispatch_csv(path: str, cu_num: int) -> dict[tuple, tuple]:
    """DISPATCH_CSV rows of cu_num: {(H, I, E, topk, x_dtype, out_dtype): ((T,
    stage-1, stage-2), ...)}. Raises ValueError for a kernel name that does not
    match the row dtypes and path (_inbf16 stage 1 iff bf16 x on the fallback
    path, _outbf16 stage 2 iff bf16 output)."""
    out: dict[tuple, list] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["cu_num"]) != cu_num:
                continue
            key = (int(r["H"]), int(r["I"]), int(r["E"]), int(r["topk"]))
            key += (r["x_dtype"], r["out_dtype"])
            t, s1, s2 = int(r["T"]), r["stage1"], r["stage2"]
            fb = uses_fallback(t, key[3], key[2])
            ok1 = s1.endswith("_inbf16") == (fb and key[4] == "bf16")
            ok2 = s2.endswith("_outbf16") == (key[5] == "bf16")
            if not (ok1 and ok2):
                raise ValueError(
                    f"{path}: {key} T={t}: kernel names do not match the dtypes"
                    f" and path ({'fallback' if fb else 'main'})"
                )
            out.setdefault(key, []).append((t, s1, s2))
    return {k: tuple(sorted(v)) for k, v in out.items()}


def prepare(
    device: torch.device | int | None = None,
) -> dict[str, dict]:
    """Check that ``device`` is gfx950, build/load the module and return the
    kernel table, keyed by kernel name. Call outside graph capture."""
    global _KERNELS, _DISPATCH
    if isinstance(device, torch.device):
        device = device.index
    if device is None:
        device = torch.cuda.current_device()
    dev = torch.device("cuda", device)
    with _LOCK:
        if dev.index not in _DEVICES:
            arch = torch.cuda.get_device_properties(dev).gcnArchName.split(":")[0]
            if arch != "gfx950":
                raise RuntimeError(f"NVFP4 MoE kernels need gfx950, got {arch}")
            _DEVICES.add(dev.index)
        if _KERNELS is None:
            with torch.cuda.device(dev):
                kernels = _build_kernels()
            dispatch = _read_dispatch_csv(DISPATCH_CSV, get_cu_num())
            for table in dispatch.values():
                _check_dispatch(table)
            with torch.cuda.device(dev):
                # JIT build / load of the fallback path's sort, outside capture
                moe_sorting_opus_get_workspace_size(1, 1, 1)
            _DISPATCH = dispatch
            _KERNELS = kernels
    return _KERNELS


def _check_dispatch(table: tuple) -> None:
    """Kernel names ([a-z0-9_]) with the stage prefix of their column (rows run as
    written; a kernel outside the module is JIT-built by _tables)."""
    for _, c1, c2 in table:
        for stage, name in ((1, c1), (2, c2)):
            if not re.fullmatch(r"[a-z0-9_]+", name) or kernel_stage(name) != stage:
                raise ValueError(f"NVFP4 MoE dispatch: {name!r} is not stage {stage}")


def _n_group(kernel):
    """N / 16 output tiles of a workgroup (stage 1 n_tiles: half gate, half up)."""
    tiles = kernel["n_tiles"] // 2 if kernel["stage"] == 1 else kernel["n_tiles"]
    return tiles * kernel["waves"]


def _round8(n):
    return (n + 7) // 8 * 8


def _first_fit(names: list, K: int, N: int):
    """The first built kernel of names that runs reduction length K and whose N group
    divides the N / 16 output tiles, else None."""
    for n in names:
        kernel = _KERNELS.get(n)
        if kernel is None:
            continue
        if (N // 16) % _n_group(kernel) == 0 and _moe_nvfp4_cfg_fits(kernel["idx"], K):
            return n
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


def _default_table(H: int, I: int, x_dtype: str, out_dtype: str, fallback: bool):
    """default_rows with the first fitting kernel per stage; a row without kernels
    for both stages is dropped."""
    rows = []
    for t, s1, s2 in default_rows(H, I, x_dtype, out_dtype, fallback):
        c1, c2 = _first_fit(s1, H, I), _first_fit(s2, I, H)
        if c1 and c2:
            rows.append((t, _KERNELS[c1], _KERNELS[c2]))
    if not rows:
        raise ValueError(f"NVFP4 MoE: no kernels for I={I}, H={H}")
    return tuple(rows)


@functools.cache
def _tables(H: int, I: int, E: int, topk: int, x_dtype: str, out_dtype: str) -> dict:
    """{fallback: (CSV rows, default-rule rows)} of a key, (T, stage-1 kernel,
    stage-2 kernel) rows. Both paths are resolved on the first call of a key. CSV
    rows: the DISPATCH_CSV rows of the key on the path of their own T."""
    _check_shape(H, I)
    out = {}
    for fb in (False, True):
        rows = []
        for t, c1, c2 in _DISPATCH.get((H, I, E, topk, x_dtype, out_dtype), ()):
            if uses_fallback(t, topk, E) == fb:
                rows.append((t, _kernel(c1), _kernel(c2)))
        out[fb] = (tuple(rows), _default_table(H, I, x_dtype, out_dtype, fb))
    return out


def _pick(W, T: int, topk: int, x_dtype: str, fallback: bool) -> tuple[dict, dict]:
    """(stage-1, stage-2) kernel table entries: the first CSV row of the path with
    T <= its T, else the first default-rule row with T <= its T, else its last row.
    A default-rule pick is logged once per shape, path and next-power-of-two T."""
    csv_rows, rule = _tables(W.H, W.I, W.E, topk, x_dtype, x_dtype)[fallback]
    for t, kernel1, kernel2 in csv_rows:
        if T <= t:
            return kernel1, kernel2
    kernel1, kernel2 = next(((c1, c2) for t, c1, c2 in rule if T <= t), rule[-1][1:])
    key = (W.H, W.I, W.E, topk, x_dtype, fallback, 1 << (T - 1).bit_length())
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(
            f"[moe_nvfp4] no tuned row for T={T} (bucket {key[-1]}; H={W.H} I={W.I}"
            f" E={W.E} topk={topk} x={x_dtype} out={x_dtype}"
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
    """One GEMM stage of kernel on the sorted rows (sort: the sort buffers by name)."""
    BM = 16 * kernel["m_tiles"]
    n = T * topk
    _moe_nvfp4_gemm(
        kernel=kernel["idx"],
        x=x,
        w=w,
        w_sc=w_sc,
        inv_gs=inv_gs,
        **sort,
        out=out,
        T=T,
        topk=topk,
        K=K,
        ntiles=ntiles,
        n_out=n_out,
        grid_x=n_out // 16 // _n_group(kernel),
        grid_y=_round8(min(E, n) + (n + BM - 1) // BM),
        custom_build_args=kernel["build"],
    )


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
            below 2^30, T * H below 2^29 for f16 x; at most MAX_GRID_Y sorted
            blocks).
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
    # 32-bit byte offsets of both stages' X and the f32 output, grid y (sorted blocks)
    if (
        max(T * H * (1 if bf16 else 2), n * W.I) * 2 >= 1 << 31
        or _round8(nblk) > MAX_GRID_Y
    ):
        raise ValueError(f"NVFP4 MoE: T={T} topk={topk} exceeds the kernel limits")
    out = torch.empty(T, H, dtype=x.dtype, device=dev)
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
            ids, tw, sorted_ids, sorted_w, sorted_eids, num_valid, out, W.E, BM,
            None, None, ws, 0,
        )  # fmt: skip
        x1 = x
    else:
        sorted_ids = torch.empty(nblk * BM, dtype=torch.int32, device=dev)
        sorted_w = torch.empty(nblk * BM, dtype=torch.float32, device=dev)
        sorted_eids = torch.empty(_round8(nblk), dtype=torch.int32, device=dev)
        num_valid = torch.empty(1, dtype=torch.int32, device=dev)
        # f16 x: the sort only zeroes the output (empty x_f16)
        x_f16 = torch.empty(T * H if bf16 else 0, dtype=torch.float16, device=dev)
        _moe_nvfp4_sort(
            x, x_f16, out, ids, tw, sorted_ids, sorted_w, sorted_eids, num_valid,
            W.E, BM, nblk,
        )  # fmt: skip
        x1 = x_f16 if bf16 else x
    sort = {
        "sorted_ids": sorted_ids,
        "sorted_w": sorted_w,
        "sorted_eids": sorted_eids,
        "num_valid": num_valid,
    }
    # stage 1: x [T, H] -> x_inter [T * topk, I], stage 2: x_inter -> out [T, H]
    _gemm(kernel1, sort, T, topk, W.E, x1, W.w13, W.s13, W.g13, x_inter,
          K=W.H, ntiles=2 * W.I // 16, n_out=W.I)  # fmt: skip
    # f16 x: the expert sum in f32 (f16 atomics overflow on partial sums)
    acc = out if bf16 else torch.zeros(T, H, dtype=torch.float32, device=dev)
    _gemm(kernel2, sort, T, topk, W.E, x_inter, W.w2, W.s2, W.g2, acc,
          K=W.I, ntiles=W.H // 16, n_out=W.H)  # fmt: skip
    return out if bf16 else out.copy_(acc)
