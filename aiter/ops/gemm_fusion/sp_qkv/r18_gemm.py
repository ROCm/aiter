# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""R18 route B (large src "r18"): the GEMM half of the plan: the large/r17 FlyDSL preshuffle FP8 GEMM with the destination-major BF16 epilogue, which writes S[dst][row][F] (golden field map, IK
replicated for the indexer) with the slab stride = ITS launch M rows (the plan launches it with M = m_real, so srows = m_real); the comm half is large/r18-put's lg_put_launch_v1 (put kernel +
r13's quiet_ready), see large_rt.LargeContext.put_launch.

large/r17/flydsl_dstmap_gemm.py is a FlyDSL (Python) kernel: it is compiled by FlyDSL on first use, not by hipcc, so it is loaded from its file (it lives outside the aiter package) and compiled
per (mode, tile) at attach, OUTSIDE graph capture. Launching it is an ordinary stream launch (capturable); nothing here allocates or synchronises per call except the per-m_real S view (cached).

The FlyDSL / torch imports are lazy: the module imports on CPU-only hosts and the tests inject fakes.
"""

from __future__ import annotations

import importlib.util
import os
import sys

from . import jit

K = 6144
TILE_N = 192  # the default tile_n of the r18 tile configs (a config may carry its own)
MAX_WAVE_TILES = 256  # one wave: the MI355X has 256 CUs
N_OF = {
    "dense": 9216,
    "indexer": 9856,
    "reuse": 9216,
}  # raw GEMM columns of the mode's weight (reuse = the dense build)
F_OF = {
    "dense": 2304,
    "indexer": 2560,
    "reuse": 2304,
}  # packed columns per destination in S
DST_MAP = {"dense": "dense", "indexer": "indexer", "reuse": "dense"}
R17_FILE = os.path.join(jit.CSRC, "large", jit.R17_SRC, "flydsl_dstmap_gemm.py")


def n_tiles(M: int, N: int, tile_m: int, tile_n: int = TILE_N) -> int:
    """Work groups of one launch: ceil(M / tile_m) * ceil(N / tile_n)."""
    return -(-M // tile_m) * -(-N // tile_n)


def tile_for(
    M: int, N: int, tiles: dict | None = None, table: dict | None = None
) -> int:
    """The tile of a launch (M rows, N columns): (1) the MEASURED override table jit.R18_TILE_TABLE (per N, ascending (m_max, tile id): the first entry with M <= m_max; Evidence's sweep), used when
    `tiles` is the default table and the table has the N and an entry covering M; otherwise (2) the wave rule over the tile table (jit.R18_TILES, data: add sweep winners there): among the tiles whose
    tile count ceil(M / tile_m) * ceil(N / tile_n) fits ONE wave (<= 256 work groups) the SMALLEST tile_m; when none fits, the fewest waves (ceil(tiles / 256)), ties -> the smallest tile_m (then the
    smallest id). Pass table={} for the pure rule. A plan pins its tile with the `tile` param instead.
    """
    if tiles is None:
        tiles = jit.R18_TILES
        table = jit.R18_TILE_TABLE if table is None else table
        for m_max, t in table.get(N, ()):
            if M <= m_max:
                if t not in tiles:
                    raise ValueError(
                        f"R18_TILE_TABLE names tile {t} which is not in R18_TILES"
                    )
                return t
    if not tiles:
        raise ValueError("no r18 tiles configured")
    cnt = {
        t: n_tiles(M, N, c["tile_m"], c.get("tile_n", TILE_N)) for t, c in tiles.items()
    }
    fit = [t for t in tiles if cnt[t] <= MAX_WAVE_TILES]
    if fit:
        return min(fit, key=lambda t: (tiles[t]["tile_m"], t))
    return min(
        tiles, key=lambda t: (-(-cnt[t] // MAX_WAVE_TILES), tiles[t]["tile_m"], t)
    )


def load_module(path: str = R17_FILE):
    """Import the vendored large/r17 kernel file (cached in sys.modules as `flydsl_dstmap_gemm`)."""
    mod = sys.modules.get("flydsl_dstmap_gemm")
    if mod is not None and getattr(mod, "__file__", None) == path:
        return mod
    spec = importlib.util.spec_from_file_location("flydsl_dstmap_gemm", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flydsl_dstmap_gemm"] = mod
    spec.loader.exec_module(mod)
    return mod


def _flydsl():
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    return fx, _run_compiled


class R18Gemm:
    """The FlyDSL dst-map GEMM of ONE mode: compiled executables per tile, the two empty placeholder tensors the kernel signature wants (split-K semaphore, bias), and the launch."""

    def __init__(self, mode: str, *, module=None, flydsl=None):
        if mode not in N_OF:
            raise ValueError(f"mode {mode!r}")
        self.mode, self.N, self.F, self.dmap = (
            mode,
            N_OF[mode],
            F_OF[mode],
            DST_MAP[mode],
        )
        self._module, self._fly = module, flydsl
        self._exe: dict[int, object] = {}
        self._sem = self._bias = None

    def _mod(self):
        if self._module is None:
            self._module = load_module()
        return self._module

    def _fx(self):
        if self._fly is None:
            self._fly = _flydsl()
        return self._fly

    def exe(self, tile: int):
        """Compile (once) the executable of `tile` for this mode's N and dst_map; FlyDSL compiles at the first call of compile_preshuffle_gemm / the first launch: warm() runs it before capture."""
        e = self._exe.get(tile)
        if e is None:
            cfg = jit.R18_TILES[tile]
            e = self._exe[tile] = self._mod().compile_preshuffle_gemm(
                N=self.N,
                K=K,
                in_dtype="fp8",
                out_dtype="bf16",
                split_k=1,
                dst_map=self.dmap,
                **cfg,
            )
        return e

    def warm(self, tiles, device="cuda"):
        """Compile every tile a launch could use and create the placeholder tensors: call at attach (prepare), outside capture."""
        import torch

        for t in sorted(set(tiles)):
            self.exe(t)
        if self._sem is None:
            self._sem = torch.empty(0, dtype=torch.int32, device=device)
            self._bias = torch.empty(0, dtype=torch.bfloat16, device=device)

    def launch(self, tile: int, s_flat, x, w, x_scale, w_scale, m: int, stream=None):
        """Enqueue the GEMM on `stream` (a torch stream; None = the current one): s_flat = the flat bf16 [4 * m * F] view of the S window, x fp8 [m, K] contiguous, w the (16,16)-preshuffled fp8
        weight (>= N rows), the scales fp32 flat. S slab d is written at d * m * F: srows = m. Returns nothing (completion = the kernel boundary).
        """
        import torch

        fx, run_compiled = self._fx()
        st = fx.Stream(stream if stream is not None else torch.cuda.current_stream())
        xq8 = x.view(torch.int8).view(-1)
        wq8 = w.view(torch.int8).view(-1)
        run_compiled(
            self.exe(tile),
            s_flat,
            s_flat,  # arg_c and arg_out are both S (arg_c only builds the unused default-store views)
            self._sem,
            xq8,
            wq8,
            x_scale.view(-1),
            w_scale.view(-1),
            self._bias,
            m,
            self.N,
            st,
        )
