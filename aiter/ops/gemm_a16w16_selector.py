# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Trained fallback kernel selector for A16W16 GEMM shapes without a tuned row (prototype).

Off by default. With ``AITER_GEMM_A16W16_SELECTOR=1`` (the artifact shipped for this gfx and
CU count) or ``=<path to an artifact>``, ``get_GEMM_A16W16_config`` asks the selector before
today's default chain whenever the tuned CSV has no row for the shape. Per new shape:

1. Eligibility: which kernels of the artifact's fixed candidate list the A16W16 tuner would
   offer for (M, N, K), with the tuner's own rules (FlyDSL pruner and parameter checks, ASM
   split-K and grid limits, Opus per-kernel helpers, the skinny shape rule; torch and Triton
   always).
2. Features from the shape and each candidate's static kernel config.
3. Scores from the artifact's LambdaRank trees, evaluated in NumPy; the highest-scoring
   eligible candidate is returned as a tuned-CSV-style row.

``AITER_GEMM_A16W16_SELECTOR_TOPK=k`` (default 1) times the top k once on the GPU at the
shape's first call and keeps the fastest; ``AITER_GEMM_A16W16_SELECTOR_CHECK=1`` checks each
timed candidate's result against an fp32 reference and skips a failing one. Neither runs
during CUDA-graph capture, where the top 1 is used. The timed calls are replayed from a
captured HIP graph, so launch overhead is excluded as it is in a graph-captured model
(``AITER_GEMM_A16W16_SELECTOR_TIMER=events`` times eager calls queued behind a GPU sleep).
``AITER_GEMM_A16W16_SELECTOR_LOG=<csv>`` appends one row per decided shape.

Decided shapes persist in a pick cache (``AITER_GEMM_A16W16_SELECTOR_CACHE``: default
``<aiter JIT dir>/gemm_a16w16_selector_picks.csv``, ``0`` turns it off). Its first line
stamps the artifact, aiter commit, ROCm and torch; any change empties the file. A cached
pick is reused when it was timed with at least the current top k and check setting.

Scope: the artifact's gfx and CU count, bf16 in and out, no bias, no scaleAB, no preshuffle.
"""

from __future__ import annotations

import csv
import functools
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

from aiter import logger

ENV = "AITER_GEMM_A16W16_SELECTOR"
ENV_TOPK = "AITER_GEMM_A16W16_SELECTOR_TOPK"
ENV_CHECK = "AITER_GEMM_A16W16_SELECTOR_CHECK"
ENV_LOG = "AITER_GEMM_A16W16_SELECTOR_LOG"
ENV_TIMER = "AITER_GEMM_A16W16_SELECTOR_TIMER"
ENV_CACHE = "AITER_GEMM_A16W16_SELECTOR_CACHE"
ARTIFACT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "gemm_a16w16_selector",
)
BF16 = "torch.bfloat16"
# Shape-independent splits of the FlyDSL pruner (block sizes of the policy grid).
_FLY_BM = (16, 32, 48, 64, 80, 96, 128, 256)
_FLY_BN = (16, 32, 64, 80, 96, 128, 256)
_FLY_BK = (64, 128, 256)
_FLY_LARGE = dict(
    block_m=256, block_n=256, block_k=64, stages=2, split_k=1, m_waves=2, n_waves=4, k_waves=1
)
_TIMED_CALLS = 20
_CACHE_BYTES = 256 << 20


def enabled() -> bool:
    return os.environ.get(ENV, "0") not in ("", "0")


def _ceil_div(a, b):
    return -(-a // b)


class Forest:
    """LightGBM trees as flat arrays; every (row, tree) takes exactly max-depth steps.

    Leaves point to themselves, so no per-row bookkeeping is needed. Matches LightGBM's raw
    score, including its missing-value and categorical rules, for float64 inputs.
    """

    def __init__(self, trees):
        offs = np.cumsum([0] + [len(t[0]) for t in trees])
        self.roots = offs[:-1].astype(np.int64)
        feat, thr, dleft, miss, left, right, value, mask, iscat = ([] for _ in range(9))
        for o, (f, th, cats, dl, ms, lf, rt, v) in zip(offs[:-1], trees):
            feat += f
            thr += th
            dleft += dl
            miss += ms
            left += [x + o if x >= 0 else -1 for x in lf]
            right += [x + o if x >= 0 else -1 for x in rt]
            value += v
            for c in cats:
                if c is not None and max(c) >= 64:
                    raise ValueError("categorical split on a code >= 64")
                iscat.append(c is not None)
                mask.append(sum(1 << int(x) for x in c) if c is not None else 0)
        feat = np.array(feat, np.int64)
        leaf = feat < 0
        idx = np.arange(len(feat))
        self.featc = np.where(leaf, 0, feat)
        self.thr = np.where(leaf, np.inf, np.array(thr, np.float64))
        self.dleft = np.array(dleft, bool)
        self.miss = np.where(leaf, 0, np.array(miss, np.int8)).astype(np.int8)
        self.left = np.where(leaf, idx, np.array(left, np.int64))
        self.right = np.where(leaf, idx, np.array(right, np.int64))
        self.value = np.array(value, np.float64)
        self.mask = np.array(mask, np.uint64)
        self.iscat = np.array(iscat, bool) & ~leaf
        self.any_cat = bool(self.iscat.any())
        self.any_miss = bool((self.miss > 0).any())
        depth = np.zeros(len(feat), np.int64)
        for r in self.roots:
            stack = [(int(r), 0)]
            while stack:
                i, d = stack.pop()
                depth[i] = d
                if not leaf[i]:
                    stack += [(int(self.left[i]), d + 1), (int(self.right[i]), d + 1)]
        self.steps = int(depth.max())

    def predict(self, X):
        n, T = len(X), len(self.roots)
        node = np.tile(self.roots, n)
        base = np.repeat(np.arange(n) * X.shape[1], T)
        Xf = np.ascontiguousarray(X, np.float64).ravel()
        one = np.uint64(1)
        for _ in range(self.steps):
            v = Xf[base + self.featc[node]]
            nan = v != v
            v0 = np.where(nan, 0.0, v)
            go = v0 <= self.thr[node]
            if self.any_miss:
                ms = self.miss[node]
                dflt = (nan & (ms == 2)) | ((ms == 1) & (np.abs(v0) <= 1e-35))
                go = np.where(dflt, self.dleft[node], go)
            if self.any_cat:
                c = np.flatnonzero(self.iscat[node])
                if c.size:
                    nd, vc = node[c], v0[c]
                    vi = np.clip(vc, 0, 63).astype(np.uint64)
                    bit = ((self.mask[nd] >> vi) & one).astype(bool) & (vc >= 0) & (vc < 64)
                    nan_left = (self.miss[nd] != 2) & ((self.mask[nd] & one) > 0)
                    go[c] = np.where(nan[c], nan_left, bit)
            node = np.where(go, self.left[node], self.right[node])
        return self.value[node].reshape(n, T).sum(1)


class Eligibility:
    """Which candidates the A16W16 tuner would offer for (M, N, K), bf16 in and out, no bias.

    FlyDSL: the tuner's config pruner and parameter checks in closed form over the fixed
    list; ASM: _get_asm_tasks' alignment, split-K, subK and grid rules plus the launch's
    N % tileN; Opus: the tuner's per-kernel helpers, limited to the compiled kids; skinny:
    is_skinny_default_shape; torch and Triton: always.
    """

    def __init__(self, cands, cu_num, props, is_skinny):
        from aiter.ops.flydsl.gemm_kernels import get_flydsl_hgemm_kernel_params
        from aiter.ops.flydsl.kernels import gemm_a16w16_gfx950 as fk
        from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950_utils import (
            GFX950_DMA_BYTES,
            SPLIT_K_SEMAPHORE_MAX_LEN,
        )
        from aiter.jit.utils.chip_info import get_lds_capacity_bytes
        from aiter.ops.gemm_op_a16w16 import ASM_SPLITK_MAX_GRID

        self.cu, self.props, self.is_skinny = cu_num, props, is_skinny
        lib = [c["libtype"] for c in cands]
        self.n = len(cands)
        self.fly = np.array([i for i, x in enumerate(lib) if x == "flydsl"], np.int64)
        cfgs = [get_flydsl_hgemm_kernel_params(cands[i]["kernelName"]) for i in self.fly]
        if any(c is None for c in cfgs):
            raise ValueError("a FlyDSL candidate is not in this aiter's kernel catalog")
        keys = ("block_m", "block_n", "block_k", "stages", "split_k", "m_waves", "n_waves",
                "k_waves", "group_m")
        f = {k: np.array([c[k] for c in cfgs], np.int64) for k in keys}
        f["hti"] = np.array([c["use_half_tile_interleaved"] for c in cfgs], np.int64)
        self.f = f
        ok, vec, thr = [], [], []
        for c in cfgs:
            try:
                p = fk.make_gemm_a16w16_gfx950_param(
                    in_dtype_id=fk.GEMM_A16W16_DTYPE_BF16,
                    out_dtype_id=fk.GEMM_A16W16_DTYPE_BF16,
                    a_is_transposed=False,
                    b_is_transposed=True,
                    has_bias=False,
                    use_half_tile_interleaved=c["use_half_tile_interleaved"],
                    **{k: c[k] for k in keys},
                )
                ok.append(True)
                vec.append(p.cshuffle_r2g_vec_size)
                thr.append(p.block_threads)
            except Exception:  # noqa: BLE001  rejected by the shape-free parameter check
                ok.append(False)
                vec.append(1)
                thr.append(1)
        vec, thr = np.array(vec, np.int64), np.array(thr, np.int64)
        bm, bn, bk, st = f["block_m"], f["block_n"], f["block_k"], f["stages"]
        mw, nw, kw = f["m_waves"], f["n_waves"], f["k_waves"]
        mma_ok = (f["hti"] == 1) | ((bm // mw // 16 <= 4) & (bn // nw // 16 <= 4))
        self.static_ok = np.array(ok, bool) & mma_ok
        self.large_cfg = np.all([f[k] == v for k, v in _FLY_LARGE.items()], axis=0) & (f["hti"] == 1)
        self.vec, self.semax = vec, SPLIT_K_SEMAPHORE_MAX_LEN
        self.dma = GFX950_DMA_BYTES // 2  # bf16 elements per DMA transfer
        self.sk_div_ok = (bm * bn) % (thr * vec) == 0
        lds_cap = get_lds_capacity_bytes()
        wpc = props.max_threads_per_multi_processor // props.warp_size

        def resident(waves, kwv):
            lds = np.maximum(st * (bm + bn) * bk * 2, kwv * bm * bn * 2)
            return np.minimum(wpc // waves, lds_cap // lds) * waves

        self.waves, self.res = mw * nw * kw, resident(mw * nw * kw, kw)
        self.waves1, self.res1 = mw * nw, resident(mw * nw, 1)
        self.kw2 = kw == 2

        self.asm_grid = ASM_SPLITK_MAX_GRID
        self.asm = []
        asm_idx = [i for i, x in enumerate(lib) if x == "asm"]
        if asm_idx:
            import pandas as pd
            from aiter.jit.core import get_asm_dir

            table = pd.read_csv(f"{get_asm_dir()}/bf16gemm/bf16gemm_fp32bf16.csv")
            by_name = {r.knl_name: r for r in table.itertuples(index=False)}
            for i in asm_idx:
                r = by_name[cands[i]["kernelName"]]
                self.asm.append((i, int(cands[i]["splitK"]), int(r.tileM), int(r.tileN),
                                 bool(r.splitK), int(r.subK), int(r.bPreshuffle)))

        self.opus, self.opus_tm = [], None
        opus_idx = [i for i, x in enumerate(lib) if x == "opus"]
        if opus_idx:
            try:
                self.opus_tm = _opus_tune_module()
                built = _opus_built_kids()
                for i in opus_idx:
                    if int(cands[i]["solidx"]) in built:
                        self.opus.append((i, int(cands[i]["solidx"]), int(cands[i]["splitK"])))
                skipped = sorted({int(cands[i]["solidx"]) for i in opus_idx} - built)
                if skipped:
                    logger.info(f"[gemm_a16w16_selector] Opus kids {skipped} are not compiled "
                                "in the Opus module; not offered")
            except Exception as e:  # noqa: BLE001  no csrc/ in this install: Opus never offered
                logger.warning(f"[gemm_a16w16_selector] Opus candidates disabled: {e}")
        self.skinny = np.array([i for i, x in enumerate(lib) if x == "skinny"], np.int64)
        self.always = np.array([i for i, x in enumerate(lib) if x in ("torch", "triton")], np.int64)

    def flydsl(self, m, n, k):
        f, ncu = self.f, self.props.multi_processor_count
        bm, bn, bk, st, sk = (f[x] for x in ("block_m", "block_n", "block_k", "stages", "split_k"))
        gm, hti = f["group_m"], f["hti"]
        sks = [1] + [s for s in range(2, 10) if k % s == 0]
        pm, pn = _ceil_div(m, bm) * bm, _ceil_div(n, bn) * bn
        pk = _ceil_div(k // sk, bk) * bk * sk
        iou = (m * n * k) / (pm * pn * pk).astype(np.float64)
        min_pm = min(_ceil_div(m, b) * b for b in _FLY_BM)
        min_pn = min(_ceil_div(n, b) * b for b in _FLY_BN)
        min_pk = min(_ceil_div(k // s, b) * b * s for b in _FLY_BK for s in sks)
        keep_ratio = max(0.625, 1 - m / 160, 1 - 120 / m)
        min_iou = m * n * k / (min_pm * min_pn * min_pk) * keep_ratio
        tiles = _ceil_div(m, bm) * _ceil_div(n, bn)
        max_tiles = max(ncu, _ceil_div(m, 256) * _ceil_div(n, 256)) * 4
        max_sk = np.where(tiles >= ncu, 1, _ceil_div(2 * ncu, tiles))
        kept = (np.isin(sk, sks) & (iou >= min_iou) & (tiles <= max_tiles) & (sk <= max_sk)
                & ((gm == 0) | ((tiles >= ncu) & (tiles % 8 == 0))))
        occ = np.minimum(np.minimum(8, self.res), tiles * sk * self.waves / ncu)
        occ1 = np.minimum(np.minimum(8, self.res1), tiles * sk * self.waves1 / ncu)
        kept &= ~self.kw2 | (occ > occ1 + 1e-9)
        if m >= 4096 and n >= 4096 and k >= 4096:
            kept &= self.large_cfg
        kept &= self.static_ok
        wk = _ceil_div(_ceil_div(k, sk), self.dma) * self.dma
        last = k - (sk - 1) * wk
        wt, lt = wk // bk, last // bk
        no_tail = ((wk % bk == 0) & (last > 0) & (last % bk == 0) & (wt >= st - 1) & (lt >= st - 1)
                   & ((hti == 0) | ((wt >= 2) & (wt % 2 == 0) & (lt >= 2) & (lt % 2 == 0))))
        kept &= no_tail & (n % self.vec == 0) & (k % self.dma == 0)
        kept &= (sk == 1) | ((tiles <= self.semax) & self.sk_div_ok)
        return kept

    def __call__(self, m, n, k):
        out = np.zeros(self.n, bool)
        out[self.fly] = self.flydsl(m, n, k)
        if k % 64 == 0 and n % 64 == 0:
            for i, ks, tile_m, tile_n, split_flag, subk, bpre in self.asm:
                if bpre != 0:
                    continue
                if split_flag:
                    tiles = _ceil_div(m, tile_m) * _ceil_div(n, tile_n)
                    hi = min(int(self.cu / tiles) if tiles < self.cu else 4, 16)
                    lo = 2
                else:
                    lo = hi = 1
                grid_ok = ks == 1 or _ceil_div(n, tile_n) * _ceil_div(m, tile_m) <= self.asm_grid
                # The launch rejects N % tileN != 0; the tuner checks only N % 64.
                out[i] = lo <= ks <= hi and k / ks >= subk and grid_ok and n % tile_n == 0
        tm = self.opus_tm
        if self.opus:
            kids = tm.candidate_kids_for_shape(m, n, k, False, self.cu)
            for i, kid, ks in self.opus:
                inst = tm.kernels_list[kid]
                if kid not in kids or tm.kid_rejects_shape(inst, m, n, k) or tm.kid_rejects_bias(inst, False):
                    continue
                sks = (tm.candidate_splitK(m, n, k, 1, self.cu, inst)
                       if kid in tm.SPLITK_KIDS else [0])
                out[i] = ks in sks
        if self.skinny.size and self.is_skinny(m, n, k, BF16, self.cu):
            out[self.skinny] = True
        out[self.always] = True
        return out


def _opus_tune_module():
    """csrc/opus_gemm/opus_gemm_tune.py, imported the way the A16W16 tuner imports it."""
    import aiter

    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(aiter.__file__))), "csrc",
                     "opus_gemm")
    if not os.path.isfile(os.path.join(d, "opus_gemm_tune.py")):
        raise ImportError(f"{d}/opus_gemm_tune.py not found")
    if d not in sys.path:
        sys.path.insert(0, d)
    import opus_gemm_tune

    return opus_gemm_tune


def _opus_built_kids():
    """Kids compiled into the installed Opus module: the subset-compile sidecar when its
    receipt matches the .so, else the arch's default compile floor (always built). A tuned
    row's kid is compiled because the build reads the tuned CSVs; a selector pick is not."""
    from opus_gemm_common import _opus_sidecar_path, default_compiled_kids_for_arch

    from aiter.jit import core
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.jit.utils.jit_cache import compiled_kids_are_current

    sidecar = _opus_sidecar_path()
    artifact = os.path.join(core.get_user_jit_dir(), "module_deepgemm_opus.so")
    if compiled_kids_are_current(sidecar, artifact, ()):
        with open(sidecar) as f:
            return {int(x) for x in json.load(f)}
    return set(default_compiled_kids_for_arch({get_gfx().lower()}))


class Selector:
    def __init__(self, artifact, props=None, is_skinny=None):
        meta = artifact["meta"]
        self.meta = meta
        self.cu_num = int(meta["cu_num"])
        self.features = artifact["features"]
        self.cands = artifact["candidates"]
        self.ids = [c["id"] for c in self.cands]
        self.index = {c: i for i, c in enumerate(self.ids)}
        self.cache = None
        self.static = {k: np.array([np.nan if v is None else v for v in vals], np.float64)
                       for k, vals in artifact["static"].items()}
        self.forest = Forest(artifact["trees"])
        if props is None:
            import torch

            p = torch.cuda.get_device_properties(torch.cuda.current_device())
            props = SimpleNamespace(multi_processor_count=p.multi_processor_count,
                                    max_threads_per_multi_processor=p.max_threads_per_multi_processor,
                                    warp_size=p.warp_size)
        if is_skinny is None:
            from aiter.tuned_gemm import is_skinny_default_shape as is_skinny
        self.eligible = Eligibility(self.cands, self.cu_num, props, is_skinny)

    @classmethod
    def load(cls, path, **kw):
        with open(path) as f:
            return cls(json.load(f), **kw)

    def feature_rows(self, idx, m, n, k):
        """Feature matrix (rows = candidates idx) as the trees saw it in training: float32."""
        st = {key: v[idx] for key, v in self.static.items()}
        cu = self.cu_num
        tm, tn, tk = st["g_tile_m"], st["g_tile_n"], st["g_tile_k"]
        sk, wt = st["g_split_k"], st["g_waves_total"]
        c = {"log_m": math.log2(m), "log_n": math.log2(n), "log_k": math.log2(k),
             "log_mn": math.log2(m * n), "log_mnk": math.log2(m * n * k),
             "log_intensity": math.log2(m * n * k / (m * k + n * k + m * n)),
             "log_n_over_k": math.log2(n / k)}
        tz = lambda v: min(math.log2(v & -v), 12)
        c["tz_m"], c["tz_n"], c["tz_k"] = tz(m), tz(n), tz(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            gm, gn = np.ceil(m / tm), np.ceil(n / tn)
            tiles = gm * gn
            ctas = tiles * sk
            ks = k / sk
            kit = np.ceil(ks / tk)
            c["g_log_tiles"], c["g_log_ctas"] = np.log2(tiles), np.log2(ctas)
            c["g_ctas_per_cu"] = ctas / cu
            c["g_wave_eff"] = ctas / (np.ceil(ctas / cu) * cu)
            c["g_wavefronts_per_cu"] = ctas * wt / cu
            c["g_pad_eff_m"], c["g_pad_eff_n"] = m / (gm * tm), n / (gn * tn)
            c["g_pad_eff_mn"] = m * n / (gm * tm * gn * tn)
            c["g_pad_eff_k"] = ks / (kit * tk)
            c["g_log_k_per_split"] = np.where(np.isfinite(tm), np.log2(ks), np.nan)
            c["g_k_iters"] = kit
            sk1 = np.where(sk > 0, sk, 1)
            c["log_cta_work"] = np.log2(tm * tn * (k / sk1))
            ctas1 = np.ceil(m / tm) * np.ceil(n / tn) * sk1
            c["last_wave_fill"] = np.where(np.isfinite(ctas1), (ctas1 - 1) % cu / cu + 1 / cu, np.nan)
        cols = [np.broadcast_to(np.asarray(c[f] if f in c else st[f], np.float64), (len(idx),))
                for f in self.features]
        return np.column_stack(cols).astype(np.float32).astype(np.float64)

    def rank(self, m, n, k):
        """Eligible candidate indices, best first, and their scores."""
        idx = np.flatnonzero(self.eligible(m, n, k))
        s = self.forest.predict(self.feature_rows(idx, m, n, k))
        order = np.argsort(-s, kind="stable")
        return idx[order], s[order]

    def config(self, i):
        c = self.cands[i]
        return {"libtype": c["libtype"], "solidx": int(c["solidx"]), "splitK": int(c["splitK"]),
                "kernelName": c["kernelName"], "selector": c["id"]}


def _artifact_path(gfx, cu_num):
    v = os.environ.get(ENV, "0")
    return v if v != "1" else os.path.join(ARTIFACT_DIR, f"{gfx}_cu{cu_num}.json")


def _aiter_commit():
    import subprocess

    import aiter

    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(aiter.__file__)), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:  # noqa: BLE001  not a checkout
        import importlib.metadata as md

        try:
            return md.version("amd-aiter")
        except md.PackageNotFoundError:
            return "unknown"


class PickCache:
    """Decided shapes on disk, one CSV per node, shared by every process under a file lock.

    The first line stamps what a pick depends on; a file with another stamp is emptied."""

    COLS = ("gfx", "cu_num", "M", "N", "K", "dtype", "otype", "pick", "rank", "eligible",
            "timed", "check", "timer", "timed_us")

    def __init__(self, path, stamp):
        self.path, self.rows = path, {}
        self.line = "# " + json.dumps(stamp, sort_keys=True)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self._lock():
            first = ""
            if os.path.exists(path):
                with open(path, newline="") as f:
                    first = f.readline().rstrip("\n")
                    if first == self.line:
                        for r in csv.DictReader(f):
                            self.rows[self.key(r)] = r
            if first != self.line:
                if first:
                    logger.info(f"[gemm_a16w16_selector] pick cache {path} has another stamp; "
                                "emptied")
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "w", newline="") as f:
                    f.write(self.line + "\n")
                    csv.writer(f).writerow(self.COLS)
                os.replace(tmp, path)

    @staticmethod
    def key(r):
        return tuple(str(r[c]) for c in ("gfx", "cu_num", "M", "N", "K", "dtype", "otype"))

    def _lock(self):
        import contextlib
        import fcntl

        @contextlib.contextmanager
        def held():
            with open(self.path + ".lock", "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

        return held()

    def get(self, key, topk, check, timer):
        """A row decided at least as thoroughly as (topk, check) now ask for, or None."""
        r = self.rows.get(tuple(map(str, key)))
        if r is None:
            return None
        need = topk if (topk > 1 or check) else 0
        if int(r["timed"]) < need or int(r["check"]) < int(check):
            return None
        return r if not need or r["timer"] == timer else None

    def put(self, row):
        row = {c: row.get(c, "") for c in self.COLS}
        old = self.rows.get(self.key(row))
        if old is not None and int(old["timed"]) > int(row["timed"]):
            return
        with self._lock():
            with open(self.path, newline="") as f:
                if f.readline().rstrip("\n") != self.line:
                    return
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow([row[c] for c in self.COLS])
        self.rows[self.key(row)] = {c: str(v) for c, v in row.items()}


def _pick_cache(artifact_path):
    v = os.environ.get(ENV_CACHE, "")
    if v == "0":
        return None
    import hashlib

    import torch

    from aiter.jit.core import get_user_jit_dir

    path = v or os.path.join(get_user_jit_dir(), "gemm_a16w16_selector_picks.csv")
    with open(artifact_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    stamp = {"format": 1, "artifact": sha, "aiter": _aiter_commit(),
             "rocm": torch.version.hip, "torch": torch.__version__}
    try:
        return PickCache(path, stamp)
    except OSError as e:
        logger.warning(f"[gemm_a16w16_selector] pick cache {path} unusable ({e}); not cached")
        return None


@functools.lru_cache(maxsize=1)
def get_selector(gfx, cu_num):
    path = _artifact_path(gfx, cu_num)
    if not os.path.isfile(path):
        logger.warning(f"[gemm_a16w16_selector] no artifact at {path}; using the default chain")
        return None
    t0 = time.perf_counter()
    sel = Selector.load(path)
    if sel.meta["gfx"] != gfx or int(sel.meta["cu_num"]) != cu_num:
        logger.warning(f"[gemm_a16w16_selector] {path} is for {sel.meta['gfx']} with "
                       f"{sel.meta['cu_num']} CUs, not {gfx} with {cu_num}; using the default chain")
        return None
    sel.cache = _pick_cache(path)
    logger.info(f"[gemm_a16w16_selector] loaded {path} ({len(sel.cands)} candidates, "
                f"{len(sel.forest.roots)} trees) in {(time.perf_counter() - t0) * 1e3:.0f} ms"
                + (f"; pick cache {sel.cache.path} ({len(sel.cache.rows)} shapes)"
                   if sel.cache else ""))
    return sel


def _events_us(cfg, copies, run):
    """Median µs of _TIMED_CALLS eager calls, all queued behind a GPU sleep so launch
    overhead does not reach the GPU timeline."""
    import torch

    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(_TIMED_CALLS)]
    torch.cuda.synchronize()
    torch.cuda._sleep(20_000_000)
    for j, (s, e) in enumerate(ev):
        x, w = copies[j % len(copies)]
        s.record()
        run(cfg, x, w)
        e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1e3 for s, e in ev)
    return t[_TIMED_CALLS // 2]


def _graph_us(cfg, copies, run, replays=5):
    """Mean µs per call of _TIMED_CALLS back-to-back calls captured in one HIP graph
    (median over replays), or None when the config cannot be captured."""
    import torch

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            for j in range(_TIMED_CALLS):
                x, w = copies[j % len(copies)]
                run(cfg, x, w)
    except Exception as e:  # noqa: BLE001
        logger.info(f"[gemm_a16w16_selector] {cfg['selector']} not graph-capturable "
                    f"({type(e).__name__}); timing eager calls")
        torch.cuda.synchronize()
        return None
    g.replay()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(replays)]
    torch.cuda.synchronize()
    torch.cuda._sleep(20_000_000)
    for s, e in ev:
        s.record()
        g.replay()
        e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1e3 / _TIMED_CALLS for s, e in ev)
    g.reset()
    return t[replays // 2]


def _trial(configs, m, n, k, run, check, timer="graph"):
    """Time each config once on random inputs (rotated copies, a warm-up pass over them,
    then _TIMED_CALLS calls, replayed from a HIP graph or eager); with check, drop a config
    whose result is off. Returns (index of the fastest passing config or None, per-config
    µs or None)."""
    import torch

    dev = torch.cuda.current_device()
    a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
    set_bytes = 2 * (m * k + n * k + m * n)
    ncopy = max(1, min(_TIMED_CALLS, math.ceil(2 * _CACHE_BYTES / set_bytes)))
    copies = [(a.clone(), b.clone()) for _ in range(ncopy)]
    ref = torch.nn.functional.linear(a.float(), b.float()) if check else None
    us = []
    for cfg in configs:
        try:
            out = run(cfg, a, b)
            if check:
                tol = 5e-2
                bad = (out.float() - ref).abs() > tol + tol * ref.abs()
                if bad.float().mean().item() > 0.05:
                    us.append(None)
                    continue
            for x, w in copies:
                run(cfg, x, w)
            t = _graph_us(cfg, copies, run) if timer == "graph" else None
            us.append(t if t is not None else _events_us(cfg, copies, run))
        except Exception as e:  # noqa: BLE001  a failing kernel is skipped, as in the tuner
            logger.warning(f"[gemm_a16w16_selector] {cfg['selector']} failed on "
                           f"({m}, {n}, {k}): {type(e).__name__}: {str(e)[:120]}")
            us.append(None)
    del copies, a, b, ref
    ok = [i for i, u in enumerate(us) if u is not None]
    return (min(ok, key=lambda i: us[i]) if ok else None), us


def select(M, N, K, bias, dtype, otype, scaleAB, bpreshuffle, gfx, cu_num, run):
    """A tuned-CSV-style row for an untuned shape, or None to use the default chain.

    run(config, a, b) runs one config on (a, b) and returns the output (for timing)."""
    if bias or scaleAB or bpreshuffle or str(dtype) != BF16 or str(otype) != BF16:
        return None
    sel = get_selector(gfx, cu_num)
    if sel is None:
        return None
    topk = max(1, int(os.environ.get(ENV_TOPK, "1")))
    check = os.environ.get(ENV_CHECK, "0") not in ("", "0")
    timer = "events" if os.environ.get(ENV_TIMER, "graph") == "events" else "graph"
    key = (gfx, cu_num, M, N, K, str(dtype), str(otype))
    hit = sel.cache.get(key, topk, check, timer) if sel.cache else None
    if hit is not None and (hit["pick"] == "" or hit["pick"] in sel.index):
        cfg = sel.config(sel.index[hit["pick"]]) if hit["pick"] else None
        logger.info(f"[gemm_a16w16_selector] M:{M}, N:{N}, K:{K} untuned: "
                    f"{hit['pick'] or 'default chain'} (pick cache)")
        _log(M, N, K, hit["pick"], hit["rank"], hit["eligible"], 0, 0, hit["timed_us"], "",
             "cache", hit["timer"])
        return cfg
    t0 = time.perf_counter()
    order, scores = sel.rank(M, N, K)
    rank_ms = (time.perf_counter() - t0) * 1e3
    if not len(order):
        return None
    kept, us, trial_ms = 0, [], 0.0
    import torch

    timed = (topk > 1 or check) and not torch.cuda.is_current_stream_capturing()
    if timed:
        cfgs = [sel.config(i) for i in order[:topk]]
        t1 = time.perf_counter()
        kept, us = _trial(cfgs, M, N, K, run, check, timer)
        trial_ms = (time.perf_counter() - t1) * 1e3
    us_txt = ";".join("" if u is None else f"{u:.2f}" for u in us)
    top5 = ";".join(sel.ids[i] for i in order[:5])
    row = dict(zip(PickCache.COLS[:7], key), eligible=len(order), timed=len(us) if timed else 0,
               check=int(check and timed), timer=timer if timed else "", timed_us=us_txt)
    if kept is None:
        logger.warning(f"[gemm_a16w16_selector] no timed candidate passed for ({M}, {N}, {K}); "
                       "using the default chain")
        if sel.cache:
            sel.cache.put(dict(row, pick="", rank=""))
        _log(M, N, K, "", "", len(order), rank_ms, trial_ms, us_txt, top5, "timed", timer)
        return None
    cfg = sel.config(order[kept])
    logger.info(f"[gemm_a16w16_selector] M:{M}, N:{N}, K:{K} untuned: {cfg['selector']} "
                f"(rank {kept + 1} of {len(order)} eligible, {rank_ms:.1f} ms"
                + (f", timed top {len(us)} in {trial_ms:.0f} ms" if us else "") + ")")
    if sel.cache:
        sel.cache.put(dict(row, pick=cfg["selector"], rank=kept + 1))
    _log(M, N, K, cfg["selector"], kept + 1, len(order), rank_ms, trial_ms, us_txt, top5,
         "timed" if timed else "ranked", timer if timed else "")
    return cfg


def _log(M, N, K, pick, rank, eligible, rank_ms, trial_ms, timed_us, top5, source, timer):
    path = os.environ.get(ENV_LOG)
    if not path:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["M", "N", "K", "pick", "rank", "eligible", "rank_ms", "trial_ms",
                        "timed_us", "top5", "source", "timer"])
        w.writerow([M, N, K, pick, rank, eligible, round(float(rank_ms), 3),
                    round(float(trial_ms), 1), timed_us, top5, source, timer])
