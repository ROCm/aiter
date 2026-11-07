# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""NVFP4 routed-expert MoE for gfx950 (MiniMax-M3 and other SwiGLU-OAI shapes).

The kernels (csrc/kernels/nvfp4_moe/moe_nvfp4_kernels.hip) decode NVFP4 weights
to f16 in registers (exact), apply the E4M3 block scale and the per-expert
global multiplier, and run f16 MFMA on bf16 -> f16 activations (no activation
quant). Three launches per MoE: fused sort + cast + output zeroing, stage 1
(gate/up + SwiGLU-OAI), stage 2 (down, weighted atomic reduce into the bf16
output, so results are not bitwise deterministic). The E4M3 scales of each
weight tensor are stored once, grouped by GS K-tiles (scale_gs of its K).

Fallback path (T * topk above MAX_SORT_ROWS, more than MAX_SORT_EXPERTS experts,
or fallback=True): AITER's opus moe_sorting (sort + output zeroing, any T) and
stage-1 kernels that read the bf16 activations and cast them to f16 themselves
(the _xbf16 kernels, same rounding: stage-1 output bit-identical to the main
path). Same weights and scale layouts as the main path.

Usage:
    w = NvFp4MoeWeights(w13, sc13, w2, sc2, g13, g2)  # once per layer
    y = moe_nvfp4(w, x, topk_ids, topk_w)            # per step

moe_nvfp4 launches on the current stream, allocates through the torch
allocator and does no host sync, so it is CUDA-graph capturable. The one-time
setup (JIT build of module_moe_nvfp4, kernel table) runs in NvFp4MoeWeights,
i.e. at weight-processing time.
"""

import csv
import os
import threading

import torch
from torch import Tensor

from ..jit.core import AITER_ROOT_DIR, compile_ops

MD_NAME = "module_moe_nvfp4"
# Per-shape (H, I, E) dispatch rows: H,I,E,path,T,stage1,stage2 (path main or
# fallback; same row rule as TP4_DISPATCH).
DISPATCH_CSV = os.getenv(
    "AITER_CONFIG_MOE_NVFP4",
    f"{AITER_ROOT_DIR}/aiter/configs/moe_nvfp4_dispatch.csv",
)
# Largest T * topk and number of experts the fused sort kernel handles
# (SORT_THREADS * SORT_PER_THREAD, SORT_MAX_EXPERTS).
MAX_SORT_ROWS = 4096
MAX_SORT_EXPERTS = 512
# Largest grid y (one row per sorted block of BM rows) HIP launches.
MAX_GRID_Y = 65536

# Measured TP4 (E=128, I=768) best (stage1, stage2) kernels by T (main path:
# the fused sort casts X to f16 for the stage-1 kernels). A T between two rows
# uses the next larger row; a T above the last row uses the last row. Kernel
# names: the grammar of parse_cfg in moe_nvfp4_common.h (the name is the config;
# nvfp4_moe_s1_ / nvfp4_moe_s2_ prefix: the stage). All of them read the weight
# scales in the SG layout (GS 6). Used for H 6144 (the stage-1 kernels need it)
# and I 768 unless DISPATCH_CSV has the shape.
TP4_DISPATCH = (
    (
        1,
        "nvfp4_moe_s1_16x32_xreg_ksplit4_wbuf4x128_kdim6144_gs6_upfront_tight_occ4",
        "nvfp4_moe_s2_16x32_xreg_wbuf4x128_kdim768_gs6_upfront_tight_occ3",
    ),
    (
        2,
        "nvfp4_moe_s1_16x64_xreg_ksplit8_wbuf3x128_kdim6144_gs6_upfront_tight_occ3",
        "nvfp4_moe_s2_16x64_xstream_waves4_wbuf3x128_xbuf4_kdim768_gs6_upfront_occ4",
    ),
    (
        3,
        "nvfp4_moe_s1_16x96_xreg_ksplit8_wbuf2x128_kdim6144_gs6_upfront_tight",
        "nvfp4_moe_s2_16x64_xstream_waves4_wbuf2x128_xbuf5_kdim768_gs6_upfront_occ2",
    ),
    (
        4,
        "nvfp4_moe_s1_16x96_xreg_ksplit8_wbuf2x128_kdim6144_gs6_upfront_tight",
        "nvfp4_moe_s2_16x128_xstream_waves4_wbuf2x128_xbuf2_kdim768_gs6_upfront_occ2",
    ),
    (
        6,
        "nvfp4_moe_s1_16x32_xreg_ksplit4_wbuf4x128_kdim6144_gs6_upfront_tight_occ4",
        "nvfp4_moe_s2_16x64_xstream_waves4_wbuf3x128_xbuf4_kdim768_gs6_upfront_occ4",
    ),
    (
        32,
        "nvfp4_moe_s1_16x64_xreg_ksplit8_wbuf2x128_kdim6144_gs6_upfront_tight_occ3",
        "nvfp4_moe_s2_16x128_xres_waves4_wbuf2x128_kdim768_gs6_upfront_tight_occ5_staged",
    ),
    (
        48,
        "nvfp4_moe_s1_16x64_xreg_ksplit8_wbuf2x128_kdim6144_gs6_tight_occ3",
        "nvfp4_moe_s2_16x128_xstream_waves4_wbuf3x128_xbuf6_kdim768_gs6_tight",
    ),
    (
        64,
        "nvfp4_moe_s1_32x64_xreg_ksplit4_wbuf3x128_kdim6144_gs6_tight",
        "nvfp4_moe_s2_32x128_xstream_waves4_wbuf3x128_xbuf2_kdim768_gs6",
    ),
    (
        96,
        "nvfp4_moe_s1_16x64_xreg_ksplit4_wbuf2x256_kdim6144_gs6_upfront_tight",
        "nvfp4_moe_s2_16x128_xres_waves4_wbuf3x256_kdim768_gs6_upfront",
    ),
    (
        128,
        "nvfp4_moe_s1_32x128_xstream_waves4_wbuf2x128_xbuf3_kdim6144_gs6_upfront_tight_occ3",
        "nvfp4_moe_s2_32x256_xstream_waves4_wbuf2x128_xbuf2_kdim768_gs6_tight_occ6",
    ),
    (
        640,
        "nvfp4_moe_s1_32x256_xdma_waves8_wbuf2x128_xbuf3_gs6_xcdremap",
        "nvfp4_moe_s2_32x256_xstream_waves4_wbuf3x128_xbuf2_kdim768_gs6_upfront_tight",
    ),
    (
        1024,
        "nvfp4_moe_s1_32x256_xdma_waves8_wbuf2x128_xbuf3_gs6_xcdremap",
        "nvfp4_moe_s2_32x256_xres_waves4_wbuf3x256_kdim768_gs6_upfront_occ3_staged",
    ),
)

# H 6144, I 384 (KT 3) shapes without DISPATCH_CSV rows: the earlier TP4 rows, stage 2
# as their KT-3 twins (odd_kt_name).
KT3_ROWS = (
    (
        1,
        "nvfp4_moe_s1_16x32_xreg_ksplit4_wbuf2x256_kdim6144_gs6_upfront_tight",
        "nvfp4_moe_s2_16x128_xres_waves4_wbuf3x256_kdim768_gs6_upfront",
    ),
    (
        2,
        "nvfp4_moe_s1_16x64_xreg_ksplit4_wbuf2x256_kdim6144_gs6_upfront_tight",
        "nvfp4_moe_s2_16x128_xres_waves4_wbuf3x256_kdim768_gs6_upfront",
    ),
    (
        8,
        "nvfp4_moe_s1_16x32_xreg_ksplit4_wbuf2x256_kdim6144_gs6_upfront_tight",
        "nvfp4_moe_s2_16x128_xres_waves4_wbuf3x256_kdim768_gs6_upfront",
    ),
    (
        120,
        "nvfp4_moe_s1_16x64_xreg_ksplit4_wbuf2x256_kdim6144_gs6_upfront_tight",
        "nvfp4_moe_s2_16x128_xres_waves4_wbuf3x256_kdim768_gs6_upfront",
    ),
    (
        1024,
        "nvfp4_moe_s1_32x256_xdma_waves8_wbuf2x128_xbuf3_gs6_xcdremap",
        "nvfp4_moe_s2_32x256_xres_waves4_wbuf3x256_kdim768_gs6_upfront_occ3_staged",
    ),
)


def xbf16_twin(name: str) -> str:
    """bf16-X (_xbf16) twin of a stage-1 kernel: same tile and math (stage-1 output
    bit-identical); an X_DMA twin stages 3 X buffers instead of 2 (its bf16 X register
    ring needs them)."""
    if name.endswith("_xbf16"):
        return name
    if "_xdma_" in name:
        name = name.replace("_xbuf2_", "_xbuf3_")
    return name + "_xbf16"


def _fallback(rows: tuple, s2_2048: str) -> tuple:
    """Fallback path rows (bf16 activations into stage 1, AITER sort), same row rule:
    up to T 1024 the _xbf16 twins of the main rows; above, a 16-wave 32x512 tile (3 N
    blocks: half the X loads and casts per token; measured 0.93-0.98x of the 32x256
    f16 kernel at T 4096-16384)."""
    return tuple((t, xbf16_twin(c1), c2) for t, c1, c2 in rows) + (
        (
            2048,
            "nvfp4_moe_s1_32x512_xdma_waves16_wbuf2x128_xbuf3_gs6_xcdremap_xbf16",
            s2_2048,
        ),
    )


TP4_FALLBACK_DISPATCH = _fallback(TP4_DISPATCH, KT3_ROWS[-1][2])
KT3_FALLBACK_ROWS = _fallback(KT3_ROWS, KT3_ROWS[-1][2])


@compile_ops(MD_NAME, develop=True)
def moe_nvfp4_num_cfgs() -> int: ...


@compile_ops(MD_NAME, develop=True)
def moe_nvfp4_sort(
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


@compile_ops(MD_NAME, develop=True)
def moe_nvfp4_gemm(
    cfg: int,
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


_LOCK = threading.Lock()
# kernel name -> kernel index and launch fields
_CFGS: dict[str, dict] | None = None
_DEVICES: set[int] = set()
# (pybind module, torch -> aiter_tensor_t converter, raw current stream getter)
# for the hot path: the same calls the develop=True wrappers above make,
# minus their per-call custom-op dispatch (about 11 us of host time each).
_FAST = None
# module_moe_sorting_opus (pybind module) for the fallback path, loaded by prepare
_SORT = None
# DISPATCH_CSV rows (_read_dispatch_csv), loaded by prepare
_DISPATCH: dict[tuple, dict[bool, tuple]] = {}


def kernel_stage(name: str) -> int:
    """Stage (1 or 2) of a kernel, from its nvfp4_moe_s1_ / nvfp4_moe_s2_ prefix."""
    for stage in (1, 2):
        if name.startswith(f"nvfp4_moe_s{stage}_"):
            return stage
    raise ValueError(f"NVFP4 MoE kernel name {name!r} has no stage prefix")


def _build_cfgs() -> dict[str, dict]:
    # The kernels are gfx950-only: pin a multi-arch GPU_ARCHS to gfx950 while
    # the first call JIT-builds the module.
    archs = os.environ.get("GPU_ARCHS")
    pin = archs is not None and archs not in ("native", "gfx950")
    if pin:
        os.environ["GPU_ARCHS"] = "gfx950"
    try:
        n = moe_nvfp4_num_cfgs()
    finally:
        if pin:
            os.environ["GPU_ARCHS"] = archs
    from ..jit.core import get_module

    mod = get_module(MD_NAME)
    out = {}
    for i in range(n):
        s1, mt, nt, wn, gs, xb = (mod.moe_nvfp4_cfg_field(i, k) for k in range(6))
        name = mod.moe_nvfp4_cfg_name(i)
        stage = kernel_stage(name)
        if stage != (1 if s1 else 2):
            raise RuntimeError(f"NVFP4 MoE kernel {name} is built for the other stage")
        if name in out:
            raise RuntimeError(f"duplicate NVFP4 MoE kernel {name}")
        out[name] = {
            "idx": i,
            "stage": stage,
            "mt": mt,
            "nt": nt,
            "wn": wn,
            "gs": gs,
            "xb": bool(xb),
        }
    return out


def _read_dispatch_csv(path: str) -> dict[tuple, dict[bool, tuple]]:
    """DISPATCH_CSV rows: {(H, I, E): {fallback: ((T, stage-1, stage-2), ...)}}."""
    out: dict[tuple, dict[bool, list]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["path"] not in ("main", "fallback"):
                raise ValueError(f"{path}: bad path {r['path']!r}")
            key = (int(r["H"]), int(r["I"]), int(r["E"]))
            rows = out.setdefault(key, {False: [], True: []})
            rows[r["path"] == "fallback"].append(
                (int(r["T"]), r["stage1"], r["stage2"])
            )
    return {k: {fb: tuple(sorted(v)) for fb, v in t.items()} for k, t in out.items()}


def prepare(
    device: torch.device | int | None = None,
) -> dict[str, dict]:
    """Check that ``device`` is gfx950, build/load the module and return the
    kernel table, keyed by kernel name. Call outside graph capture."""
    global _CFGS, _DISPATCH, _FAST, _SORT
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
        if _CFGS is None:
            with torch.cuda.device(dev):
                cfgs = _build_cfgs()
            for table, xb in ((TP4_DISPATCH, False), (TP4_FALLBACK_DISPATCH, True)):
                _check_dispatch(cfgs, table, xb, 6, 6)
            dispatch = _read_dispatch_csv(DISPATCH_CSV)
            for (H, I, _), tables in dispatch.items():
                for fb, table in tables.items():
                    gs = (scale_gs(H // 128), scale_gs(I // 128))
                    _check_dispatch(cfgs, table, fb, *gs)
            _DISPATCH = dispatch
            _CFGS = cfgs
        if _FAST is None:
            from ..jit.core import _pybind_develop_hooks, get_module

            convert, _, raw_stream, _ = _pybind_develop_hooks()
            _FAST = (get_module(MD_NAME), convert, raw_stream)
        if _SORT is None:
            from ..jit.core import get_module
            from .moe_sorting_opus import moe_sorting_opus_get_workspace_size

            with torch.cuda.device(dev):
                moe_sorting_opus_get_workspace_size(1, 1, 1)  # JIT build / load
            _SORT = get_module("module_moe_sorting_opus")
    return _CFGS


def _check_dispatch(
    cfgs: dict[str, dict], table: tuple, xb: bool, gs1: int, gs2: int
) -> None:
    """Stage prefixes, scale group sizes, stage-1 X types and per-row BM (the
    fallback path needs bf16-X stage-1 kernels; the main path takes either)."""
    for _, c1, c2 in table:
        for stage, name, gs in ((1, c1, gs1), (2, c2, gs2)):
            if name not in cfgs:
                raise RuntimeError(f"NVFP4 MoE dispatch: no kernel {name}")
            if cfgs[name]["stage"] != stage:
                raise RuntimeError(f"NVFP4 MoE dispatch: {name} is not stage {stage}")
            if cfgs[name]["gs"] != gs:
                raise RuntimeError(f"kernel {name} does not read GS-{gs} scales")
        if xb and not cfgs[c1]["xb"]:
            raise RuntimeError(f"kernel {c1}: bf16 X {cfgs[c1]['xb']}, need {xb}")
        if cfgs[c1]["mt"] != cfgs[c2]["mt"]:
            raise RuntimeError(f"kernels {c1} and {c2} differ in block M")


def pick_cfg(T: int, fallback: bool = False) -> tuple[str, str]:
    """(stage-1, stage-2) TP4 kernel names for T tokens on the main or fallback
    path."""
    table = TP4_FALLBACK_DISPATCH if fallback else TP4_DISPATCH
    for t, c1, c2 in table:
        if T <= t:
            return c1, c2
    return table[-1][1], table[-1][2]


def odd_kt_name(name: str, kt: int) -> str:
    """KT-kt twin of a TP4 stage-2 kernel (odd kt): same tile, one K-tile per
    weight chunk (wbuf3x256 -> wbuf3x128), the GS-4 scales of KT + 1 (built
    for kt 3 only)."""
    return name.replace("_wbuf3x256_kdim768_gs6_", f"_wbuf3x128_kdim{128 * kt}_gs4_")


def scale_gs(kt: int) -> int:
    """Scale group size GS (K-tiles per group) of a weight tensor with kt K-tiles:
    6 if 6 divides kt, else 4 if kt <= 16 and 4 divides kt, else 2 (pair). An odd
    kt takes the GS of kt + 1 (the scales of the missing K-tile are zero)."""
    kt += kt % 2
    if kt % 6 == 0:
        return 6
    return 4 if kt <= 16 and kt % 4 == 0 else 2


def _generic_table(fallback: bool, kt13: int, kt2: int) -> tuple:
    """One row of K-generic kernels for a shape without dispatch rows: X_REG with
    no K split (stage 1 16x32, stage 2 16x16) reading the shape's scale layouts;
    stage 2 at odd KT 3: the 16x128 KT-3 xres kernel."""
    sfx = {2: "", 4: "_gs4", 6: "_gs6"}
    c1 = "nvfp4_moe_s1_16x32_xreg_wbuf2x256" + sfx[scale_gs(kt13)]
    c1 += "_xbf16" if fallback else ""
    if kt2 % 2:
        c2 = "nvfp4_moe_s2_16x128_xres_waves4_wbuf3x256_kdim768_gs6_upfront"
        c2 = odd_kt_name(c2, kt2)
    else:
        c2 = "nvfp4_moe_s2_16x16_xreg_wbuf2x256" + sfx[scale_gs(kt2)]
    return ((1, c1, c2),)


def _shape_table(fallback: bool, H: int, I: int, E: int) -> tuple:
    """(T, stage-1 cfg, stage-2 cfg) rows for one weight shape.

    DISPATCH_CSV rows of (H, I, E) if any; else the TP4 table for H 6144 and I
    768, KT3_ROWS for H 6144 and I 384 (stage 2: odd_kt_name twins); else one
    generic row.
    Rows whose N group does not divide the shape's N tiles are dropped (the
    previous row then covers them).

    Raises:
        ValueError: H or I not a multiple of 128, an odd stage-1 K-tile count, an
            odd stage-2 K-tile count other than 3, or no kernel fits the shape.
    """
    kt13, kt2 = H // 128, I // 128
    if H % 128 or I % 128:
        raise ValueError(f"NVFP4 MoE: H={H} and I={I} must be multiples of 128")
    if kt13 % 2:
        raise ValueError(f"NVFP4 MoE: H={H}: no stage-1 kernels for odd H / 128")
    if kt2 % 2 and kt2 != 3:
        raise ValueError(f"NVFP4 MoE: I={I}: odd I / 128 is supported for 3 only")
    if (H, I, E) in _DISPATCH:
        table = _DISPATCH[(H, I, E)][fallback]
    elif H == 6144 and kt2 == 6:
        table = TP4_FALLBACK_DISPATCH if fallback else TP4_DISPATCH
    elif H == 6144 and kt2 == 3:
        table = KT3_FALLBACK_ROWS if fallback else KT3_ROWS
        table = tuple((t, c1, odd_kt_name(c2, kt2)) for t, c1, c2 in table)
    else:
        table = _generic_table(fallback, kt13, kt2)
    gs = (scale_gs(kt13), scale_gs(kt2))
    _check_dispatch(_CFGS, table, fallback, *gs)
    rows = []
    for t, c1, c2 in table:
        cf1, cf2 = _CFGS[c1], _CFGS[c2]
        if (I // 16) % (cf1["nt"] // 2 * cf1["wn"]) or (H // 16) % (
            cf2["nt"] * cf2["wn"]
        ):
            continue
        rows.append((t, cf1, cf2))
    if not rows:
        raise ValueError(f"NVFP4 MoE: no kernels for I={I}, H={H}")
    return tuple(rows)


# ------------------------------------------------------------------ layouts
def tile_w(packed: Tensor) -> Tensor:
    E, N, K2 = packed.shape
    return (
        packed.view(E, N // 16, 16, K2 // 64, 4, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
    )


def tile_sc_nv(sc: Tensor) -> Tensor:
    """[E, N, K/16] -> pair layout [E][NT][KTP][64][4 B]; an odd K-tile count
    gets zero scales for the missing half of the last pair."""
    if sc.shape[2] % 16:
        sc = torch.nn.functional.pad(sc, (0, 16 - sc.shape[2] % 16))
    E, N, KS = sc.shape
    return (
        sc.view(E, N // 16, 16, KS // 16, 2, 4, 2)
        .permute(0, 1, 3, 5, 2, 4, 6)
        .contiguous()
    )


def gs_sc(ts: Tensor, gs: int) -> Tensor:
    """Pair layout [E][NT][KTP][64][4 B] -> GS layout [E][NT][KTP/(GS/2)][64][2 GS B]
    (GS 2: the pair layout itself, GS 6: the SG layout)."""
    E, NT = ts.shape[:2]
    gps = gs // 2
    return ts.reshape(E, NT, -1, gps, 64, 4).permute(0, 1, 2, 4, 3, 5).contiguous()


def layout_sc(sc: Tensor) -> Tensor:
    """E4M3 scales [E, N, K/16] (uint8) in the GS layout scale_gs(K / 128)."""
    return gs_sc(tile_sc_nv(sc), scale_gs((sc.shape[2] + 7) // 8))


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

    Raises:
        ValueError: a shape without kernels (see _shape_table): H or I not a
            multiple of 128, H / 128 odd, or I / 128 odd and not 3.
    """

    def __init__(self, w13, sc13, w2, sc2, g13, g2):
        prepare(w13.device)
        self.E, N13, H2 = w13.shape
        self.I, self.H = N13 // 2, H2 * 2
        self._tables = {
            fb: _shape_table(fb, self.H, self.I, self.E) for fb in (False, True)
        }
        self.g13 = g13.float().contiguous()
        self.g2 = g2.float().contiguous()
        sc13 = sc13.view(torch.uint8)
        sc2 = sc2.view(torch.uint8)
        self.w13, self.w2 = tile_w(w13), tile_w(w2)
        self.s13, self.s2 = layout_sc(sc13), layout_sc(sc2)
        # aiter_tensor_t views of the persistent tensors for the hot path
        cv = _FAST[1]
        self._a = {
            k: cv(getattr(self, k)) for k in ("w13", "w2", "s13", "s2", "g13", "g2")
        }

    def pick(self, T: int, fallback: bool) -> tuple[dict, dict]:
        """(stage-1, stage-2) kernel table entries for T tokens."""
        table = self._tables[fallback]
        for t, cf1, cf2 in table:
            if T <= t:
                return cf1, cf2
        return table[-1][1], table[-1][2]


def _grid(cfg, T, topk, n_out, E):
    BM = 16 * cfg["mt"]
    gy = min(E, T * topk) + (T * topk + BM - 1) // BM
    gy = (gy + 7) // 8 * 8
    if cfg["stage"] == 1:
        gx = (n_out // 16) // (cfg["nt"] // 2 * cfg["wn"])
    else:
        gx = (n_out // 16) // (cfg["nt"] * cfg["wn"])
    return gx, gy


def uses_fallback(T: int, topk: int, E: int) -> bool:
    """Whether moe_nvfp4 takes the fallback path by default for this shape."""
    return T * topk > MAX_SORT_ROWS or E > MAX_SORT_EXPERTS


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
        x: bf16 [T, H] activations.
        topk_ids: [T, topk] expert ids (any integer dtype).
        topk_w: [T, topk] routing weights (any float dtype).
        fallback: None: the fallback path iff uses_fallback(T, topk, E); True /
            False: force it on / off.

    Returns:
        bf16 [T, H].

    Raises:
        ValueError: fallback=False for a shape the fused sort cannot take, or a
            shape over the kernel limits (32-bit X byte offsets: T * H and
            T * topk * I below 2^30; at most MAX_GRID_Y sorted blocks).
    """
    T, H = x.shape
    topk = topk_ids.shape[1]
    if fallback is None:
        fallback = uses_fallback(T, topk, W.E)
    elif not fallback and uses_fallback(T, topk, W.E):
        raise ValueError(
            f"NVFP4 MoE main path supports T * topk <= {MAX_SORT_ROWS} and"
            f" E <= {MAX_SORT_EXPERTS}"
        )
    cf1, cf2 = W.pick(T, fallback)
    mod, cv, raw_stream = _FAST
    BM = 16 * cf1["mt"]
    dev = x.device
    x = x.contiguous()
    n = T * topk
    nblk = min(W.E, n) + (n + BM - 1) // BM
    # 32-bit byte offsets of both stages' X, grid y (sorted blocks)
    if max(T * H, n * W.I) * 2 >= 1 << 31 or (nblk + 7) // 8 * 8 > MAX_GRID_Y:
        raise ValueError(f"NVFP4 MoE: T={T} topk={topk} exceeds the kernel limits")
    out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    x_inter = torch.empty(n, W.I, dtype=torch.float16, device=dev)
    ids = topk_ids.to(torch.int32).contiguous()
    tw = topk_w.float().contiguous()
    if fallback:
        # The opus sort fills up to n + E * BM - topk rows; the kernels read
        # sorted_eids for every grid row (nblk rounded up to 8).
        nblk = max((nblk + 7) // 8 * 8, (n + W.E * BM - topk + BM - 1) // BM)
        sids = torch.empty(nblk * BM, dtype=torch.int32, device=dev)
        sw = torch.empty(nblk * BM, dtype=torch.float32, device=dev)
        seids = torch.empty(nblk, dtype=torch.int32, device=dev)
        nv = torch.empty(2, dtype=torch.int32, device=dev)
        ws_bytes = _SORT.moe_sorting_opus_get_workspace_size(T, W.E, topk, 0, dev.index)
        ws = torch.empty(ws_bytes, dtype=torch.uint8, device=dev) if ws_bytes else None
        _SORT._set_current_hip_stream(raw_stream(dev.index))
        _SORT.moe_sorting_opus_fwd(cv(ids), cv(tw), cv(sids), cv(sw), cv(seids), cv(nv),
                                   cv(out), W.E, BM, None, None,
                                   None if ws is None else cv(ws), 0)  # fmt: skip
        x1_ = cv(x)
    else:
        sids = torch.empty(nblk * BM, dtype=torch.int32, device=dev)
        sw = torch.empty(nblk * BM, dtype=torch.float32, device=dev)
        seids = torch.empty((nblk + 7) // 8 * 8, dtype=torch.int32, device=dev)
        nv = torch.empty(1, dtype=torch.int32, device=dev)
        # bf16-X stage 1: the sort only zeroes the output (empty x_f16: no cast)
        x_f16 = torch.empty(0 if cf1["xb"] else T * H, dtype=torch.float16, device=dev)
        x1_ = cv(x if cf1["xb"] else x_f16)
        xf_ = cv(x_f16)
    sort = (cv(sids), cv(sw), cv(seids), cv(nv))
    out_, x_inter_ = cv(out), cv(x_inter)
    wa = W._a
    mod._set_current_hip_stream(raw_stream(dev.index))
    if not fallback:
        mod.moe_nvfp4_sort(cv(x), xf_, out_, cv(ids), cv(tw), *sort, W.E, BM, nblk)
    gx, gy = _grid(cf1, T, topk, W.I, W.E)
    mod.moe_nvfp4_gemm(cf1["idx"], x1_, wa["w13"], wa["s13"], wa["g13"], *sort,
                       x_inter_, T, topk, W.H, 2 * W.I // 16, W.I,
                       gx, gy)  # fmt: skip
    gx, gy = _grid(cf2, T, topk, W.H, W.E)
    mod.moe_nvfp4_gemm(cf2["idx"], x_inter_, wa["w2"], wa["s2"], wa["g2"], *sort,
                       out_, T, topk, W.I, W.H // 16, W.H,
                       gx, gy)  # fmt: skip
    return out
