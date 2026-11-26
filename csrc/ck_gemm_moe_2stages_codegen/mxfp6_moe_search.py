# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel search of the FlyDSL MXFP6 MoE (aiter.ops.flydsl.flydsl_mxfp6_moe) for one
call shape: what `gemm_moe_tune.py --mxfp6-flydsl` runs per row of its untuned file.

The baseline is the heuristic's choice (mxfp6_moe_policy.default_config). Searched
from it:
  routing tile  128 / 256 rows: the router and both stages
  h layout      a 6-bit h between the stages, tile-step-major / row-major
  per stage     the tile with its LDS ring depth (stage 1 with 8-bit activations also
                128-row half tiles over 256-row routing tiles), the tile group size,
                non-temporal weight loads, padded rows (stage 1, 6-bit activations),
                the L2 warm (stage 2, tile-step h)
Not searched: the weight slab width (fixed when the layer is prepared), the router,
the row quantizer and the combine.

A shape is timed on one random call of it, of a fixed seed (SearchOptions.seed), as
the other MoE tuners do, in two steps:
  1. propose  per (routing tile, h layout), each stage alone on the routed buffers,
              from the heuristic's kernel for that pair: a coordinate descent over
              the dimensions above, or every combination (SearchOptions.full). A
              candidate is timed only after its output equals the baseline kernel's
              bit for bit.
  2. rank     the whole op of every pair of a proposed and a baseline stage, per
              (routing tile, h layout), each timed only after its output equals the
              heuristic's bit for bit. The result is the fastest pair; the
              heuristic's own pair is one of them, so it is the result where nothing
              found is faster.

Which pair is fastest also depends on which experts the tokens are routed to, which
the shape does not fix: SearchOptions.draws ranks on that many routings of the call
instead of the one.

Times are back-to-back calls after warm-up, by GPU events. Weights are random and
the routing uniformly random, so tune the token counts you serve.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from aiter.jit.utils.chip_info import get_cu_num
from aiter.ops.flydsl import mxfp6_moe_kernels as mk
from aiter.ops.flydsl import mxfp6_moe_policy as policy
from aiter.ops.flydsl import mxfp6_moe_quant as q
from aiter.ops.flydsl.mxfp6_moe import (
    Mxfp6MoeWeights,
    flydsl_mxfp6_moe,
    prepare_mxfp6_moe_weights,
)
from aiter.ops.flydsl.mxfp6_moe_policy import Mxfp6MoeConfig, Mxfp6StageConfig

ROUTE_M = (128, 256)
TILE_N = (128, 256)
RINGS = (2, 3)
GROUP_M = (1, 2, 4, 8, 16)
WARM = (0, 1, 2, 4, 8)
# a move of the descent must beat the stage's best by this much
STEP_GAIN = 0.005


@dataclass(frozen=True)
class SearchOptions:
    """full: every combination per stage instead of the coordinate descent (passes
    rounds of it). warmup: untimed calls before a timing; a timing is the median of
    repeats means of back-to-back calls, each of at least min_ms and at most
    max_iters calls. draws: routings of the call the pairs are ranked on, the first
    the call's own. seed: of the weights, the call and the further routings."""

    full: bool = False
    passes: int = 2
    warmup: int = 5
    max_iters: int = 101
    repeats: int = 3
    min_ms: float = 10.0
    draws: int = 1
    seed: int = 2026


@dataclass
class ShapeResult:
    """One shape's search.

    best: the fastest pair of kernels found; heuristic: the heuristic's. us /
    us_heuristic: their whole op, the mean over the draws; us1 / us2: best's stages
    alone, on the draw they were searched on. candidates: stage kernels tried;
    pairs: pairs ranked; profile: one row per stage kernel tried."""

    heuristic: Mxfp6MoeConfig
    best: Mxfp6MoeConfig
    us: float
    us_heuristic: float
    us1: float
    us2: float
    candidates: int
    pairs: int
    profile: list[dict] = field(default_factory=list)


def fastest(us: dict[Mxfp6MoeConfig, list[float]]) -> Mxfp6MoeConfig:
    """The pair with the smallest total time over the draws (us: per pair, its time
    on each draw); of equal ones, the first."""
    return min(us, key=lambda config: sum(us[config]))


def stage_dims(stage1: bool, six: bool, tile_step: bool) -> list[str]:
    """The dimensions of one stage's search."""
    dims = ["tile", "group_m", "nt_weights"]
    if stage1 and six:
        dims.append("pad_rows")
    if not stage1 and tile_step:
        dims.append("warm")
    return dims


def stage_variants(
    base: Mxfp6StageConfig,
    dim: str,
    *,
    stage1: bool,
    six: bool,
    route_m: int,
    n: int,
    slab: int,
    a_bytes: int,
) -> list[Mxfp6StageConfig]:
    """base with dim set to each of its values, base's own included.

    n: the stage's columns; slab: the width its weights are prepared in (a tile
    covers whole slabs); a_bytes: the stage's activation rows (the padded copy of
    4/3 of them must stay below the kernel's 32-bit offsets)."""
    replace = dataclasses.replace
    if dim == "tile":
        rows = [route_m]
        if stage1 and not six and route_m == 256:
            rows.append(route_m // 2)  # half tiles, as the heuristic uses them
        return [
            replace(base, tile_m=m, tile_n=tn, ring=ring, half_tiles=m < route_m)
            for m, tn, ring in itertools.product(rows, TILE_N, RINGS)
            # whole weight slabs; 256 x 256 on 8-bit rows fits two LDS stages only
            if not (tn % slab or n % tn or (not six and m * tn * ring > 256 * 256 * 2))
        ]
    if dim == "group_m":
        return [replace(base, group_m=v) for v in GROUP_M]
    if dim == "nt_weights":
        return [replace(base, nt_weights=v) for v in (False, True)]
    if dim == "pad_rows":
        fits = a_bytes // 3 * 4 + 4096 <= 2**32
        return [
            replace(base, pad_rows=v) for v in ((False, True) if fits else (False,))
        ]
    if dim == "warm":
        return [replace(base, warm=v) for v in WARM]
    raise ValueError(f"no search dimension {dim!r}")


class Timer:
    """us per call: median over repeats of the mean of back-to-back calls."""

    def __init__(self, options: SearchOptions):
        self.options = options

    def __call__(self, fn: Callable[[], object]) -> float:
        o = self.options
        for _ in range(o.warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        one_us = max(start.elapsed_time(end) * 1e3, 1.0)
        iters = int(min(o.max_iters, max(5, 1e3 * o.min_ms / one_us)))
        samples = []
        for _ in range(o.repeats):
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1e3 / iters)
        return statistics.median(samples)


@dataclass
class Case:
    """A layer of random weights, prepared, and one call's inputs."""

    weights: Mxfp6MoeWeights
    x: torch.Tensor
    ids: torch.Tensor
    w: torch.Tensor

    @property
    def tokens(self) -> int:
        return self.ids.shape[0]

    @property
    def topk(self) -> int:
        return self.ids.shape[1]

    def heuristic(self, **kw) -> Mxfp6MoeConfig:
        """The heuristic's kernels of this call; kw: default_config's route_m /
        h_layout."""
        st = self.weights
        return policy.default_config(
            tokens=self.tokens,
            topk=self.topk,
            experts=st.experts,
            model_dim=st.model_dim,
            inter_dim=st.inter_dim,
            a_fmt=st.a_fmt,
            slab_n1=st.slab_n1,
            slab_n2=st.slab_n2,
            w1_bytes=st.w1_bytes,
            w2_bytes=st.w2_bytes,
            cu_num=get_cu_num(),
            **kw,
        )

    def run(self, config: Mxfp6MoeConfig) -> torch.Tensor:
        return flydsl_mxfp6_moe(self.x, self.weights, self.w, self.ids, config=config)

    def redraw(self, seed: int) -> Case:
        """This call with another routing draw (the rows are kept: their values do
        not change the schedule)."""
        ids, w = draw_routing(self.weights, self.tokens, self.topk, seed)
        return Case(weights=self.weights, x=self.x, ids=ids, w=w)


def make_layer(
    model_dim: int,
    inter_dim: int,
    experts: int,
    a_fmt: str,
    w_fmt: str,
    seed: int,
    device: str = "cuda",
) -> Mxfp6MoeWeights:
    """A prepared layer of random weights (the fp32 copies are not kept)."""
    g = torch.Generator(device=device).manual_seed(seed)
    bf16 = torch.bfloat16
    w1 = torch.randn(
        experts, 2 * inter_dim, model_dim, generator=g, device=device, dtype=bf16
    )
    w2 = torch.randn(
        experts, model_dim, inter_dim, generator=g, device=device, dtype=bf16
    )
    w1.mul_(1 / math.sqrt(model_dim))
    w2.mul_(1 / math.sqrt(inter_dim))
    return prepare_mxfp6_moe_weights(w1, w2, a_fmt=a_fmt, w_fmt=w_fmt)


def draw_routing(
    weights: Mxfp6MoeWeights, tokens: int, topk: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """One routing draw: (ids [tokens, topk] int32, topk distinct uniformly random
    experts per token; w [tokens, topk] fp32, routing weights that sum to 1)."""
    device = weights.w1_frags.device
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.rand(tokens, weights.experts, generator=g, device=device).argsort(-1)
    w = torch.rand(tokens, topk, generator=g, device=device) + 0.5
    return ids[:, :topk].int().contiguous(), (w / w.sum(-1, keepdim=True)).float()


def make_case(weights: Mxfp6MoeWeights, tokens: int, topk: int, seed: int) -> Case:
    """One call on the layer: random rows and the routing draw of seed."""
    device = weights.w1_frags.device
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(tokens, weights.model_dim, generator=g, device=device)
    ids, w = draw_routing(weights, tokens, topk, seed)
    return Case(weights=weights, x=x.to(torch.bfloat16), ids=ids, w=w)


class _Combo:
    """The routed buffers of one (routing tile, h layout) and the outputs of the
    heuristic's kernels for it, which every candidate of a stage must reproduce."""

    def __init__(self, case: Case, start: Mxfp6MoeConfig):
        st = case.weights
        self.case, self.start = case, start
        H, I, af = st.model_dim, st.inter_dim, st.a_fmt
        self.bits = q.FORMATS[af][3]
        self.six = self.bits == 6
        self.tile_step = self.six and start.h_layout == "tile_step"
        device = case.x.device
        self.ridx, _, self.texp, rows = mk.flydsl_mxfp6_moe_route(
            case.ids, st.experts, start.route_m
        )
        self.xa, self.xs = mk.flydsl_mxfp6_moe_quant_rows(case.x, af)
        self.h_row_bytes = I * self.bits // 8

        # zeroed: a kernel writes the used tiles only; the buffers are compared whole
        def h():
            return (
                torch.zeros(rows, self.h_row_bytes, dtype=torch.uint8, device=device),
                torch.zeros(rows, q.kq(I), 4, dtype=torch.int32, device=device),
            )

        def o():
            return torch.zeros(rows, H, dtype=torch.bfloat16, device=device)

        (self.ha_ref, self.hs_ref), (self.ha, self.hs) = h(), h()
        self.o_ref, self.o = o(), o()
        self.stage1(start.stage1, self.ha_ref, self.hs_ref)()
        self.stage2(start.stage2, self.o_ref)()
        torch.cuda.synchronize()

    def _shared(self):
        st = self.case.weights
        return {
            "ridx": self.ridx,
            "texp": self.texp,
            "model_dim": st.model_dim,
            "inter_dim": st.inter_dim,
            "a_fmt": st.a_fmt,
            "w_fmt": st.w_fmt,
            "h_layout": self.start.h_layout,
        }

    def stage1(self, config, h_rows=None, h_scales=None):
        st = self.case.weights
        kw = {
            "x_rows": self.xa,
            "x_scales": self.xs,
            "w1_frags": st.w1_frags,
            "w1_scales": st.w1_scales,
            "h_rows": self.ha if h_rows is None else h_rows,
            "h_scales": self.hs if h_scales is None else h_scales,
            "config": config,
            **self._shared(),
        }
        return lambda: mk.flydsl_mxfp6_moe_stage1(**kw)

    def stage2(self, config, out=None):
        st = self.case.weights
        kw = {
            "h_rows": self.ha_ref,
            "h_scales": self.hs_ref,
            "w2_frags": st.w2_frags,
            "w2_scales": st.w2_scales,
            "out": self.o if out is None else out,
            "config": config,
            **self._shared(),
        }
        return lambda: mk.flydsl_mxfp6_moe_stage2(**kw)

    def clear(self, stage1: bool) -> None:
        """Zero the stage's candidate buffers: what the last candidate wrote must not
        pass for the next one's output."""
        for buffer in (self.ha, self.hs) if stage1 else (self.o,):
            buffer.zero_()

    def same(self, stage1: bool) -> bool:
        """The candidate buffers hold the baseline kernel's output, bit for bit."""
        if stage1:
            return torch.equal(self.ha, self.ha_ref) and torch.equal(
                self.hs, self.hs_ref
            )
        return torch.equal(self.o.view(torch.int16), self.o_ref.view(torch.int16))


def device_lost(error: BaseException) -> bool:
    """An error after which no kernel of this process can be trusted."""
    text = str(error)
    return type(error).__name__ in ("AcceleratorError", "GpuContextLost") or any(
        mark in text for mark in ("HIP error", "hipError", "illegal memory access")
    )


def moe_work(
    tokens: int, model_dim: int, inter_dim: int, experts: int, topk: int, a_fmt: str
) -> tuple[int, int]:
    """(FLOPs, bytes) of one call: stage 1 is a [tokens * topk, 2 x inter, model] GEMM
    and stage 2 a [tokens * topk, model, inter] one; the bytes are the 6-bit weights,
    x and y, and per routed row the quantized h and the bf16 stage-2 row, each
    written and read."""
    a_bits = q.FORMATS[a_fmt][3]
    nbytes = (
        experts * 3 * model_dim * inter_dim * 6 // 8
        + tokens * model_dim * 2 * 2
        + tokens * topk * (inter_dim * a_bits // 8 + model_dim * 2) * 2
    )
    return 6 * tokens * topk * model_dim * inter_dim, nbytes


def _search_stage(combo, stage1, timer, options, log, profile):
    """(best kernel, us of every kernel tried: inf where it was rejected) of one
    stage of combo."""
    st = combo.case.weights
    start = combo.start.stage1 if stage1 else combo.start.stage2
    run = combo.stage1 if stage1 else combo.stage2
    space = {
        "stage1": stage1,
        "six": combo.six,
        "route_m": combo.start.route_m,
        "n": 2 * st.inter_dim if stage1 else st.model_dim,
        "slab": st.slab_n1 if stage1 else st.slab_n2,
        "a_bytes": combo.xa.nbytes,
    }
    seen: dict[Mxfp6StageConfig, float] = {}

    def evaluate(config):
        if config in seen:
            return seen[config]
        status, us = "ok", math.inf
        try:
            fn = run(config)
            combo.clear(stage1)
            fn()
            torch.cuda.synchronize()
            if combo.same(stage1):
                us = timer(fn)
            else:
                status = "output differs from the heuristic's kernel"
        except Exception as e:  # a kernel the builder refuses is skipped
            if device_lost(e):
                raise
            status = f"{type(e).__name__}: {(str(e).splitlines() or [''])[0][:160]}"
        seen[config] = us
        name = policy.stage_kernel_name(stage1, config)
        profile.append(
            {
                "block_m": combo.start.route_m,
                "h_layout": combo.start.h_layout,
                "stage": 1 if stage1 else 2,
                "kernelName": name,
                "us": round(us, 4) if math.isfinite(us) else "",
                "status": status,
            }
        )
        log(f"    {name}: " + (f"{us:.1f} us" if math.isfinite(us) else status))
        return us

    best, best_us = start, evaluate(start)
    if not math.isfinite(best_us):
        name = policy.stage_kernel_name(stage1, start)
        raise RuntimeError(f"the heuristic's own kernel failed: {name}")
    dims = stage_dims(stage1, combo.six, combo.tile_step)
    if options.full:
        pool = {start}
        for dim in dims:
            pool = {v for c in pool for v in stage_variants(c, dim, **space)}
        for config in sorted(pool, key=lambda c: policy.stage_kernel_name(stage1, c)):
            us = evaluate(config)
            if us < best_us * (1 - STEP_GAIN):
                best, best_us = config, us
        return best, seen
    for _ in range(options.passes):
        moved = False
        for dim in dims:
            for config in stage_variants(best, dim, **space):
                us = evaluate(config)
                if us < best_us * (1 - STEP_GAIN):
                    best, best_us, moved = config, us, True
        if not moved:
            break
    return best, seen


def _pair_name(config: Mxfp6MoeConfig) -> str:
    names = policy.row_results(config)
    return (
        f"{config.route_m}-row tiles, h {config.h_layout}, {names['kernelName1']} + "
        f"{names['kernelName2']}"
    )


def search_shape(
    case: Case, options: SearchOptions, log: Callable[[str], None] = print
) -> ShapeResult:
    """Search the kernels of case's call; see the module docstring."""
    timer = Timer(options)
    heuristic = case.heuristic()
    six = q.FORMATS[case.weights.a_fmt][3] == 6
    layouts = policy.H_LAYOUTS if six else policy.H_LAYOUTS[:1]
    pairs, profile, stage_us = [heuristic], [], {}
    # 1. propose: each stage alone
    for route_m, h_layout in itertools.product(ROUTE_M, layouts):
        start = case.heuristic(route_m=route_m, h_layout=h_layout)
        log(f"  routing tile {route_m}, h layout {h_layout}")
        combo = _Combo(case, start)
        stage1, seen1 = _search_stage(combo, True, timer, options, log, profile)
        stage2, seen2 = _search_stage(combo, False, timer, options, log, profile)
        stage_us[route_m, h_layout] = (seen1, seen2)
        del combo
        torch.cuda.empty_cache()
        # the stages' bests need not add up: every pair with a baseline stage too
        pairs += [
            dataclasses.replace(start, stage1=first, stage2=second)
            for first in (start.stage1, stage1)
            for second in (start.stage2, stage2)
        ]
    pairs = list(dict.fromkeys(pairs))
    # 2. rank: the whole op, on that call and draws - 1 more routings of it
    us: dict[Mxfp6MoeConfig, list[float]] = {config: [] for config in pairs}
    for d in range(options.draws):
        draw = case.redraw(options.seed + d) if d else case
        expect = draw.run(heuristic).view(torch.int16).clone()
        for config in list(us)[:: -1 if d % 2 else 1]:
            if not torch.equal(draw.run(config).view(torch.int16), expect):
                log(f"    dropped, output differs on draw {d}: {_pair_name(config)}")
                del us[config]
                continue
            us[config].append(timer(lambda config=config, draw=draw: draw.run(config)))
    best = fastest(us)
    for config in sorted(us, key=lambda config: sum(us[config])):
        log(f"    whole op, {_pair_name(config)}: {statistics.mean(us[config]):.1f} us")
    seen1, seen2 = stage_us[best.route_m, best.h_layout]
    return ShapeResult(
        heuristic=heuristic,
        best=best,
        us=statistics.mean(us[best]),
        us_heuristic=statistics.mean(us[heuristic]),
        us1=seen1[best.stage1],
        us2=seen2[best.stage2],
        candidates=sum(len(a) + len(b) for a, b in stage_us.values()),
        pairs=len(us),
        profile=profile,
    )
