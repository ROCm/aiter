# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py

from itertools import product
from pathlib import Path
import dataclasses
import os
import statistics
import triton.profiler as proton
import torch
import argparse
import csv
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.moe_op_gemm_a16w4 import (
    moe_gemm_a16w4,
    moe_gemm_torch,
    _get_config,
)
from aiter.ops.triton.utils.shuffle import shuffle_scale_moe
from aiter.ops.triton.utils._triton.arch_info import get_arch
import tempfile
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, upcast_from_mxfp

# Config dict forwarded to every moe_gemm_a16w4 call (set from --backend /
# --gluon-stage in main). None -> let the op's `auto` config pick per shape.
_MOE_CONFIG = None


def parse_profile(profile_path, useful_op_regex, reps):
    """
    construct a PerfRecord from a (proton) profile path and a regex for useful operations
    """
    from triton.profiler import viewer

    gf, _, _, _ = viewer.read(profile_path)

    # aggregate "useful" flops + bytes
    useful = gf.filter(
        f"MATCH ('*', c) WHERE c.'name' =~ '{useful_op_regex}' AND c IS LEAF"
    ).dataframe
    bytes_ = int(useful["bytes"].sum())
    flops = int(
        sum(useful[[c for c in ["flops8", "flops16"] if c in useful.columns]].sum())
    )

    # take all ops (incl. "not useful" ones) when computing total time
    allops = gf.filter("MATCH ('*', c) WHERE c IS LEAF").dataframe
    total_time_ns = allops["time (ns)"].sum()
    kernel_time_ns = useful["time (ns)"].sum()
    return {
        "total_time_ns": total_time_ns,
        "kernel_time_ns": kernel_time_ns,
        "flops": flops,
        "bytes": bytes_,
        "reps": reps,
    }


def _default_swizzle():
    """Arch default for --swizzle auto: CDNA4 on the MI350 series (gfx950), none
    on the MI450 series (gfx1250, whose gluon fast path needs compact scales)."""
    arch = get_arch()
    if arch == "gfx950":
        return "CDNA4_SCALE"
    if arch == "gfx1250":
        return None
    return None


def check_and_shuffle_scales(scale, N, K, swizzle):
    """Apply the requested MX-scale swizzle, or leave the scale compact.

    `swizzle` is one of None / "CDNA4_SCALE" / "GFX1250_SCALE". Swizzling needs
    N % 32 == 0 and K % (32*8) == 0; shapes that don't satisfy that stay compact
    (returns None as the layout). Compact scales are the only layout the gluon
    backend accepts; swizzled scales force the triton backend."""
    if swizzle is None or not (N % 32 == 0 and K % (32 * 8) == 0):
        return scale, None
    shuffle_arch = "gfx950" if swizzle == "CDNA4_SCALE" else "gfx1250"
    scale, layout = shuffle_scale_moe(
        scale,
        arch=shuffle_arch,
        preshuffle_factor=32,
        scale_kwidth=8,
        return_layout=True,
    )
    assert layout == swizzle, (layout, swizzle)
    return scale, layout


def quantize(x, dtype):
    if dtype == "bf16":
        x = x.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return x, None
    elif dtype == "fp8":
        scale = x.abs().max().item() / 448.0
        fp8e4_dtype = (
            torch.float8_e4m3fn if get_arch() != "gfx942" else torch.float8_e4m3fnuz
        )
        x = x.to(fp8e4_dtype)
        return x, scale
    elif dtype == "mx8":
        fp8e4_dtype = (
            torch.float8_e4m3fn if get_arch() != "gfx942" else torch.float8_e4m3fnuz
        )
        x, scale = downcast_to_mxfp(x, fp8e4_dtype, axis=1)
        return x, scale
    else:
        assert dtype == "mx4", f"{dtype=}"
        x, scale = downcast_to_mxfp(x.to(torch.bfloat16), torch.uint8, axis=1)
        return x, scale


def make_routing(batch, n_expts_tot, n_expts_act, routing_mode, skew, dev, seed=None):
    """Build (routing_data, gather_indx, scatter_indx) for one shape.

    ``uniform``  -> independent random logits, so expert load is ~uniform
                    (many experts active), matching the old bench behaviour.
    ``skewed``   -> add a per-expert bias so a subset of experts dominates the
                    top-k selection, mimicking the concentrated load seen in
                    real captured routing (far fewer experts active).

    When ``seed`` is not None the logits are drawn from a dedicated generator
    keyed on ``(seed, batch)``, so the routing (and thus the active-expert count
    that drives latency) is identical for a given M across process runs and is
    independent of any other RNG consumption (e.g. weight init). This is what
    makes A/B comparisons (e.g. --gluon-stage 1 vs 2) fair. With seed=None the
    global RNG is used, reproducing the old non-deterministic behaviour.
    """
    gen = None
    if seed is not None:
        # 100003 is a prime multiplier so distinct M values never collide on the
        # same sub-seed; the generator lives on the routing device.
        gen = torch.Generator(device=dev).manual_seed(int(seed) * 100003 + int(batch))
    logits = torch.randn((batch, n_expts_tot), device=dev, generator=gen)
    if routing_mode == "skewed":
        expert_bias = torch.randn((n_expts_tot,), device=dev, generator=gen) * skew
        logits = logits + expert_bias[None, :]
    return routing(logits, n_expts_act)


def dump_routing_blocks(batch, routing_data):
    """Print, per launched M-block, the expert id and non-padding token count.

    Decodes routing_data.expt_data.block_pid_map exactly as the kernel does:
    val==-1 is an idle block; else expt_id = val & 0xFFFF, block_id = val >> 16,
    and tokens = min(block_m, hist[expt_id] - block_id*block_m).
    """
    block_m = int(routing_data.block_m)
    ed = routing_data.expt_data
    hist = ed.hist.tolist()
    bpm = ed.block_pid_map.tolist()
    active = []
    for pid, val in enumerate(bpm):
        if val == -1:
            continue
        expt = val & 0xFFFF
        blk = (val >> 16) & 0xFFFF
        active.append((pid, expt, blk, min(block_m, hist[expt] - blk * block_m)))
    idle = len(bpm) - len(active)
    print(
        f"  [blocks] M={batch} block_m={block_m} grid_m={len(bpm)} "
        f"active={len(active)} idle={idle} "
        f"experts_with_work={len({e for _, e, _, _ in active})}"
    )
    print(f"    {'pid':>5} {'expert':>7} {'blk_in_expert':>13} {'tokens':>7}")
    for pid, expt, blk, tok in active:
        print(f"    {pid:>5} {expt:>7} {blk:>13} {tok:>7}")


def op_bytes_flops(batch, dim1, dim2, n_expts_act, n_active, gemm="both"):
    """Analytic byte/flop model for the MoE layer (mxfp4 weights). `gemm` selects
    GEMM1 (gate/up), GEMM2 (down) or the full layer ("both").

    Weight traffic scales with the number of actually active experts, so uniform
    vs skewed routing shows up in TBPS.
    """
    R = batch * n_expts_act
    half = dim2 // 2
    # mxfp4 weights: 0.5 B/elem + 1 B e8m0 scale per 32 elems along K
    w1_bytes = n_active * (dim1 * dim2 * 0.5 + (dim1 // 32) * dim2)
    w2_bytes = n_active * (half * dim1 * 0.5 + (half // 32) * dim1)
    # bf16 activations
    x_bytes = batch * dim1 * 2  # GEMM1 input
    interm_bytes = R * half * 2  # GEMM1 write / GEMM2 read (one direction)
    out_bytes = batch * dim1 * 2  # GEMM2 output
    flops1 = 2 * R * dim1 * dim2
    flops2 = 2 * R * half * dim1
    if gemm == "gemm1":
        return int(w1_bytes + x_bytes + interm_bytes), int(flops1)
    if gemm == "gemm2":
        return int(w2_bytes + interm_bytes + out_bytes), int(flops2)
    total_bytes = w1_bytes + w2_bytes + x_bytes + 2 * interm_bytes + out_bytes
    flops = flops1 + flops2
    return int(total_bytes), int(flops)


def time_with_events(fn, warmup, iters):
    """Repeated-run CUDA-event timing (same method as the capture replay).

    Returns a list of per-call latencies in milliseconds.
    """
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()
    starts, ends = [], []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        starts.append(s)
        ends.append(e)
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _measure(fn, timing, warmup, iters):
    """Latency stats (ms) via repeated CUDA events or triton's do_bench.

    Returns {avg_ms, min_ms, max_ms, median_ms}. do_bench manages its own
    warmup/rep; the events path uses the `warmup`/`iters` counts.
    """
    if timing == "do_bench":
        import triton.testing

        med, lo, hi = triton.testing.do_bench(fn, quantiles=[0.5, 0.0, 1.0])
        return {"avg_ms": med, "min_ms": lo, "max_ms": hi, "median_ms": med}
    times = time_with_events(fn, warmup, iters)
    return {
        "avg_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "median_ms": statistics.median(times),
    }


# --- launched-grid capture ------------------------------------------------
# Patch KernelInterface.__getitem__ (shared by the triton & gluon kernels) to
# record each launch's decomposed grid (grid_m x grid_n, from the named kernel
# args) while capture is on -- so the bench can report the real grid per shape.
_GRID_CAP = {"on": False, "grids": []}


def _install_grid_capture():
    from triton.runtime.jit import KernelInterface

    if getattr(KernelInterface.__getitem__, "_grid_cap", False):
        return
    orig = KernelInterface.__getitem__

    def patched(self, grid):
        launcher = orig(self, grid)
        if not _GRID_CAP["on"]:
            return launcher
        arg_names = getattr(self, "arg_names", None)
        name = getattr(self, "__name__", None) or type(self).__name__
        total = grid[0] if isinstance(grid, (tuple, list)) and grid else grid

        def wrap(*a, **k):
            gm = gn = None
            if arg_names:
                nd = dict(zip(arg_names, a))
                nd.update(k)
                gm, gn = nd.get("grid_m"), nd.get("grid_n")
            _GRID_CAP["grids"].append((name, total, gm, gn))
            return launcher(*a, **k)

        return wrap

    patched._grid_cap = True
    KernelInterface.__getitem__ = patched


def _capture_grids(fn):
    """Run fn once with capture on; return 'gemm=64x48 gemm=64x48 reduce=16'."""
    _install_grid_capture()
    _GRID_CAP["grids"] = []
    _GRID_CAP["on"] = True
    try:
        fn()
    finally:
        _GRID_CAP["on"] = False
    parts = []
    for name, total, gm, gn in _GRID_CAP["grids"]:
        low = name.lower()
        if "gemm" in low:
            parts.append(
                f"gemm={gm}x{gn}"
                if gm is not None and gn is not None
                else f"gemm={total}"
            )
        elif "reduce" in low:
            parts.append(f"reduce={total}")
    return " ".join(parts) if parts else "-"


# --- autotuning -----------------------------------------------------------
# Search space for --tune: the Cartesian product of these per-parameter lists.
# block_m is fixed by routing; backend and (optionally) gluon stage are pinned by
# --backend / --gluon-stage. The product can be large, so tuning is slow.
_TUNE_SPACE = {
    "block_n": [64, 128, 256],
    "block_k": [32, 64, 128, 256, 512],
    "num_warps": [1, 2, 4, 8],
    "num_stages": [1, 2],
    "group_m": [1, 4],
    "xcd_swizzle": [1, 2, 8],
    "waves_per_eu": [0, 1, 2, 3, 4],
}
# Extra gluon-only knob (the pipeline stage), swept unless --gluon-stage pins it.
_TUNE_SPACE_GLUON = {"num_buffers": [1, 2, 3]}

_TUNE_FMT_KEYS = [
    "backend",
    "num_buffers",
    "block_n",
    "block_k",
    "num_warps",
    "num_stages",
    "group_m",
    "xcd_swizzle",
    "waves_per_eu",
]


def _fmt_cfg(cfg):
    return "{" + ", ".join(f"{k}={cfg[k]}" for k in _TUNE_FMT_KEYS if k in cfg) + "}"


def _tune_candidates(backend, pinned):
    """Cartesian product of _TUNE_SPACE (plus the gluon stage unless pinned), each
    merged with `pinned` (which fixes backend and optionally num_buffers)."""
    space = dict(_TUNE_SPACE)
    if backend == "gluon" and "num_buffers" not in pinned:
        space["num_buffers"] = _TUNE_SPACE_GLUON["num_buffers"]
    keys = list(space)
    cands = []
    for combo in product(*(space[k] for k in keys)):
        cfg = dict(pinned)
        cfg.update(dict(zip(keys, combo)))
        cands.append(cfg)
    return cands


def _load_assert_close():
    """Reuse assert_close from the unit test (op_tests/.../test_moe_gemm_a16w4.py)
    for the tuned-config correctness check."""
    import importlib.util

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "triton_tests",
        "moe",
        "test_moe_gemm_a16w4.py",
    )
    spec = importlib.util.spec_from_file_location("_moe_a16w4_unittest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.assert_close


def _make_is_close(ref):
    """Predicate checking a kernel output against `ref` with the unit test's
    assert_close tolerances (maxtol=4e-1, rmstol=4e-2). Its stdout (the mismatch
    dump on failure) is suppressed."""
    import contextlib
    import io

    assert_close = _load_assert_close()

    def is_close(out):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                assert_close(ref, out, maxtol=4e-1, rmstol=4e-2, verbose=False)
            return True
        except AssertionError:
            return False

    return is_close


def _perf_str(ms, flops, byts):
    """TFLOPS/TBPS string in the same format as the benchmark output."""
    if not ms:
        return ""
    tflops = flops / (ms * 1e6) * 1e-3
    tbps = byts / (ms * 1e6) * 1e-3
    return f"TFLOPS: {tflops:#.4g} | TBPS: {tbps:.2f}"


def _eval_config(make_fn, make_time_fn, cfg, is_close, timing, warmup, iters):
    """Evaluate one config -> (min_ms, status): 'ok' (correct), 'wrong' (ran but
    incorrect vs reference) or 'fail' (didn't compile/launch, e.g. LDS over
    budget). Correctness uses make_fn (the target GEMM alone); timing uses
    make_time_fn (the target GEMM alone for isolated tuning, or the full fused
    layer with only this GEMM's config varied for fused tuning)."""
    try:
        out = make_fn(cfg)()
    except Exception:
        return None, "fail"
    if is_close is not None and not is_close(out):
        return None, "wrong"
    try:
        return _measure(make_time_fn(cfg), timing, warmup, iters)["min_ms"], "ok"
    except Exception:
        return None, "fail"


def _tune_one(
    label,
    make_fn,
    make_time_fn,
    backend,
    pinned,
    current_cfg,
    is_close,
    flops,
    byts,
    timing,
    warmup,
    iters,
    gap,
    cand,
):
    cands = _tune_candidates(backend, pinned)
    print(f"\n[tune] {label}: sweeping {len(cands)} configs (pinned {pinned}) ...")
    scored = []
    n_wrong = n_fail = 0
    for c in cands:
        t, st = _eval_config(make_fn, make_time_fn, c, is_close, timing, warmup, iters)
        if st == "ok":
            scored.append((t, c))
        elif st == "wrong":
            n_wrong += 1
        else:
            n_fail += 1

    cur_t, cur_st = _eval_config(
        make_fn, make_time_fn, current_cfg, is_close, timing, warmup, iters
    )
    cur_s = f"{cur_t:.4f} ms" if cur_t is not None else "n/a"
    print(
        f"[tune] {label}: current config -> {cur_s} [{cur_st}] "
        f"{_perf_str(cur_t, flops, byts)}  {_fmt_cfg(current_cfg)}"
    )
    print(
        f"[tune] {label}: {len(scored)} correct, {n_wrong} wrong, {n_fail} failed "
        f"(of {len(cands)})"
    )
    if not scored:
        print(f"[tune] {label}: no correct config found")
        return
    scored.sort(key=lambda x: x[0])
    best = scored[0][0]
    picked = [(t, c) for (t, c) in scored if t <= best * (1 + gap / 100.0)][:cand]
    print(
        f"[tune] {label}: top {len(picked)} verified-correct configs within "
        f"{gap:g}% of best, fastest first:"
    )
    for t, c in picked:
        print(
            f"       {t:.4f} ms  ({t / best - 1:+.1%})  "
            f"{_perf_str(t, flops, byts)}  {_fmt_cfg(c)}"
        )


def _golden_ref(
    x, w, w_scale, bias, rdata, gidx, sidx, gammas, apply_swiglu, golden_cpu, out_dev
):
    """Reference (golden) output via moe_gemm_torch, used for the tuned-config
    correctness check. Per-expert upcast avoids the int32 overflow of a bulk
    upcast on big shapes. With --golden-cpu the upcast lands on CPU and the
    reference matmul runs on CPU (freeing GPU memory for the kernel on large
    MiniMax-M3 shapes); the small result is moved back to out_dev to compare."""
    golden_dev = "cpu" if golden_cpu else out_dev
    w_ref = torch.stack(
        [
            upcast_from_mxfp(w[e], w_scale[e], torch.bfloat16, axis=0).to(golden_dev)
            for e in range(w.shape[0])
        ]
    )
    if not golden_cpu:
        return moe_gemm_torch(x, w_ref, bias, rdata, gidx, sidx, gammas, apply_swiglu)
    # moe_gemm_torch follows x.device; move all golden inputs (incl. the routing
    # histogram) to CPU, then bring the result back for the device-matched compare.
    rdata_g = dataclasses.replace(
        rdata, expt_hist=None if rdata.expt_hist is None else rdata.expt_hist.cpu()
    )
    ref = moe_gemm_torch(
        x.cpu(),
        w_ref,
        bias.cpu(),
        rdata_g,
        None if gidx is None else gidx.cpu(),
        None if sidx is None else sidx.cpu(),
        None if gammas is None else gammas.cpu(),
        apply_swiglu,
    )
    return ref.to(out_dev)


def tune_shape(
    batch,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype,
    w_dtype,
    TP,
    routing_mode,
    skew,
    seed,
    gluon_stage,
    gap,
    cand,
    warmup,
    iters,
    gemm="both",
    timing="events",
    fused=False,
    golden_cpu=False,
):
    """Autotune GEMM1 (gate/up) and/or GEMM2 (down) for one shape (`gemm` selects
    which). When `fused`, each candidate is timed inside the full two-GEMM layer
    with only the tuned GEMM's config varied (the other stays at its resolved
    config), so cache behavior matches production; otherwise the GEMM is timed in
    isolation. Correctness is always checked per-GEMM against moe_gemm_torch (on
    CPU when `golden_cpu`, freeing GPU memory for the kernel on large shapes)."""
    dev = "cuda:0"
    if seed is not None:
        torch.manual_seed(int(seed))
    w1 = torch.randn((n_expts_tot, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot, dim2 // TP // 2, dim1), device=dev)
    b1 = torch.randn((n_expts_tot, dim2 // TP), device=dev)
    b2 = torch.randn((n_expts_tot, dim1), device=dev)
    w1, w1_scale = quantize(w1, w_dtype)
    w2, w2_scale = quantize(w2, w_dtype)
    # Tune against compact (non-swizzled) scales: the common denominator both
    # backends accept (gluon rejects swizzled scales).
    sw1 = sw2 = None
    xdt = torch.bfloat16 if x_dtype == "bf16" else torch.float16
    x = torch.randn((batch, dim1), dtype=xdt, device=dev)
    rdata, gidx, sidx = make_routing(
        batch, n_expts_tot, n_expts_act, routing_mode, skew, dev, seed=seed
    )
    gammas = rdata.gate_scal
    n_active = int((rdata.expt_hist > 0).sum())
    bytes1, flops1 = op_bytes_flops(
        batch, dim1, dim2 // TP, n_expts_act, n_active, gemm="gemm1"
    )
    bytes2, flops2 = op_bytes_flops(
        batch, dim1, dim2 // TP, n_expts_act, n_active, gemm="gemm2"
    )
    # Fused timing ranks by whole-layer time, so report whole-layer flops/bytes.
    bytes_all, flops_all = op_bytes_flops(
        batch, dim1, dim2 // TP, n_expts_act, n_active, gemm="both"
    )
    print(
        f"\n===== tune shape: M={batch} dim1={dim1} dim2={dim2} "
        f"E={n_expts_tot}/{n_expts_act} block_m={rdata.block_m} "
        f"({'fused' if fused else 'isolated'} timing) ====="
    )

    def _pin(backend):
        p = {"backend": backend}
        if gluon_stage is not None and backend == "gluon":
            p["num_buffers"] = gluon_stage
        return p

    # GEMM1: gate/up (gather + swiglu). N=dim2/TP, K=dim1.
    def make_gemm1(cfg):
        return lambda: moe_gemm_a16w4(
            x,
            w1,
            None,
            w1_scale,
            None,
            None,
            b1,
            rdata,
            gather_indx=gidx,
            swizzle_mx_scale=sw1,
            out_dtype=xdt,
            apply_swiglu=True,
            config=cfg,
        )

    # Resolved current config for each GEMM (both needed up front so fused timing
    # can hold the non-tuned GEMM fixed).
    cur1, _ = _get_config(
        rdata, gidx.shape[0], dim2 // TP, dim1, config=_MOE_CONFIG, swizzle_mx_scale=sw1
    )
    cur2, _ = _get_config(
        rdata,
        gidx.shape[0],
        dim1,
        dim2 // TP // 2,
        config=_MOE_CONFIG,
        swizzle_mx_scale=sw2,
    )

    def make_layer(g1_cfg, g2_cfg):
        # Full fused two-GEMM layer with per-GEMM configs (fused-mode timing).
        def run():
            im = moe_gemm_a16w4(
                x,
                w1,
                None,
                w1_scale,
                None,
                None,
                b1,
                rdata,
                gather_indx=gidx,
                swizzle_mx_scale=sw1,
                out_dtype=xdt,
                apply_swiglu=True,
                config=g1_cfg,
            )
            return moe_gemm_a16w4(
                im,
                w2,
                None,
                w2_scale,
                None,
                None,
                b2,
                rdata,
                scatter_indx=sidx,
                gammas=gammas,
                swizzle_mx_scale=sw2,
                out_dtype=xdt,
                config=g2_cfg,
            )

        return run

    if gemm in ("both", "gemm1"):
        ref1 = _golden_ref(
            x, w1, w1_scale, b1, rdata, gidx, None, None, True, golden_cpu, dev
        )
        time_fn = (lambda cfg: make_layer(cfg, cur2)) if fused else make_gemm1
        fl, by = (flops_all, bytes_all) if fused else (flops1, bytes1)
        _tune_one(
            f"M={batch} GEMM1 gate/up (N={dim2 // TP} K={dim1})",
            make_gemm1,
            time_fn,
            cur1["backend"],
            _pin(cur1["backend"]),
            cur1,
            _make_is_close(ref1),
            fl,
            by,
            timing,
            warmup,
            iters,
            gap,
            cand,
        )

    if gemm not in ("both", "gemm2"):
        return

    # GEMM2: down (scatter). N=dim1, K=dim2/TP/2. Correctness uses a fixed interm
    # (from cur1); fused timing recomputes it inside make_layer each iteration.
    interm = make_gemm1(cur1)()

    def make_gemm2(cfg):
        return lambda: moe_gemm_a16w4(
            interm,
            w2,
            None,
            w2_scale,
            None,
            None,
            b2,
            rdata,
            scatter_indx=sidx,
            gammas=gammas,
            swizzle_mx_scale=sw2,
            out_dtype=xdt,
            config=cfg,
        )

    ref2 = _golden_ref(
        interm, w2, w2_scale, b2, rdata, None, sidx, gammas, False, golden_cpu, dev
    )
    time_fn = (lambda cfg: make_layer(cur1, cfg)) if fused else make_gemm2
    fl, by = (flops_all, bytes_all) if fused else (flops2, bytes2)
    _tune_one(
        f"M={batch} GEMM2 down (N={dim1} K={dim2 // TP // 2})",
        make_gemm2,
        time_fn,
        cur2["backend"],
        _pin(cur2["backend"]),
        cur2,
        _make_is_close(ref2),
        fl,
        by,
        timing,
        warmup,
        iters,
        gap,
        cand,
    )


def bench_mlp_single_weight_init(
    batch,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype,
    w_dtype,
    TP,
    op_regex,
    timing="events",
    routing_mode="uniform",
    skew=4.0,
    warmup=5,
    iters=20,
    dump_blocks=False,
    seed=None,
    gemm="both",
    swizzle=None,
):
    rank = 0
    dev = f"cuda:{rank}"

    # Make weight/activation init reproducible too when a seed is given. Weight
    # *values* don't affect latency, but this keeps the whole run deterministic.
    if seed is not None:
        torch.manual_seed(int(seed))

    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"

    # -- init data --
    # weights
    w1 = torch.randn((n_expts_tot, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot, dim2 // TP // 2, dim1), device=dev)
    # biases
    b1 = torch.randn((n_expts_tot, dim2 // TP), device=dev)
    b2 = torch.randn((n_expts_tot, dim1), device=dev)

    # -- numerics --
    w1, w1_scale = quantize(w1, w_dtype)
    w2, w2_scale = quantize(w2, w_dtype)
    w1_scale, swizzle_mx_scale1 = check_and_shuffle_scales(
        w1_scale, dim2 // TP, dim1, swizzle
    )
    w2_scale, swizzle_mx_scale2 = check_and_shuffle_scales(
        w2_scale, dim1, dim2 // TP // 2, swizzle
    )

    x_dtype_torch = torch.bfloat16 if x_dtype == "bf16" else torch.float16
    x = torch.randn((batch, dim1), dtype=x_dtype_torch, device=dev)

    # routing computed once and held fixed across timed iters, exactly like the
    # capture replay (which replays fixed real gather/scatter indices).
    rdata, gather_indx, scatter_indx = make_routing(
        batch, n_expts_tot, n_expts_act, routing_mode, skew, dev, seed=seed
    )
    gammas = rdata.gate_scal
    n_active = int((rdata.expt_hist > 0).sum())

    if dump_blocks:
        dump_routing_blocks(batch, rdata)

    # GEMM1 (gate/up): gather + swiglu -> intermediate (M*topk, dim2/2). Output is
    # stashed so GEMM2 can run on the real intermediate.
    interm_holder = {}

    def run_gemm1():
        out = moe_gemm_a16w4(
            x,
            w1,
            None,
            w1_scale,
            None,
            None,
            b1,
            rdata,
            gather_indx=gather_indx,
            swizzle_mx_scale=swizzle_mx_scale1,
            out_dtype=x_dtype_torch,
            apply_swiglu=True,
            config=_MOE_CONFIG,
        )
        interm_holder["interm"] = out
        return out

    # GEMM2 (down): scatter-reduce with router weights, no swiglu -> (M, dim1),
    # matching the production fused_experts down-projection.
    def run_gemm2():
        return moe_gemm_a16w4(
            interm_holder["interm"],
            w2,
            None,
            w2_scale,
            None,
            None,
            b2,
            rdata,
            scatter_indx=scatter_indx,
            gammas=gammas,
            swizzle_mx_scale=swizzle_mx_scale2,
            out_dtype=x_dtype_torch,
            config=_MOE_CONFIG,
        )

    def run_layer():
        run_gemm1()
        return run_gemm2()

    # Select what to time: GEMM1 alone, GEMM2 alone (on the real intermediate) or
    # the full layer.
    if gemm == "gemm1":
        run_fn = run_gemm1
    elif gemm == "gemm2":
        run_gemm1()  # populate the intermediate once (not timed)
        run_fn = run_gemm2
    else:
        run_fn = run_layer

    total_bytes, flops = op_bytes_flops(
        batch, dim1, dim2 // TP, n_expts_act, n_active, gemm=gemm
    )

    # capture the actual launched grid(s) once
    launched_grid = _capture_grids(run_fn)

    if timing == "proton":
        reps = 100
        fpath = Path(tempfile.mktemp())
        proton.start(str(fpath), hook="triton")
        for _ in range(reps):
            run_fn()
        proton.finalize()
        perf = parse_profile(
            fpath.with_suffix(".hatchet"), useful_op_regex=op_regex, reps=reps
        )
        perf["launched_grid"] = launched_grid
        return perf

    # events / do_bench timing
    stats = _measure(run_fn, timing, warmup, iters)
    mean_ms = stats["avg_ms"]
    return {
        "total_time_ns": mean_ms * 1e6,
        "kernel_time_ns": mean_ms * 1e6,
        "flops": flops,
        "bytes": total_bytes,
        "reps": 1,
        **stats,
        "active_experts": n_active,
        "launched_grid": launched_grid,
    }


def bench_mlp(
    batch,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype,
    w_dtype,
    TP,
    op_regex,
    num_weight_inits=1,
    timing="events",
    routing_mode="uniform",
    skew=4.0,
    warmup=5,
    iters=20,
    dump_blocks=False,
    seed=None,
    gemm="both",
    swizzle=None,
):
    all_results = []
    for init_idx in range(num_weight_inits):
        # Distinct-but-reproducible seed per weight init so multiple inits still
        # sample different routing while staying stable across process runs.
        init_seed = None if seed is None else int(seed) + init_idx
        result = bench_mlp_single_weight_init(
            batch,
            dim1,
            dim2,
            n_expts_tot,
            n_expts_act,
            x_dtype,
            w_dtype,
            TP,
            op_regex,
            timing=timing,
            routing_mode=routing_mode,
            skew=skew,
            warmup=warmup,
            iters=iters,
            dump_blocks=dump_blocks,
            seed=init_seed,
            gemm=gemm,
            swizzle=swizzle,
        )
        all_results.append(result)

    num_runs = len(all_results)
    aggregated = {
        "total_time_ns": sum(r["total_time_ns"] for r in all_results) / num_runs,
        "kernel_time_ns": sum(r["kernel_time_ns"] for r in all_results) / num_runs,
        "flops": sum(r["flops"] for r in all_results) / num_runs,
        "bytes": sum(r["bytes"] for r in all_results) / num_runs,
        "reps": all_results[0]["reps"],
    }
    # average the optional (events-mode) latency fields when present
    for k in ["avg_ms", "min_ms", "max_ms", "median_ms", "active_experts"]:
        if k in all_results[0]:
            aggregated[k] = sum(r[k] for r in all_results) / num_runs
    # launched_grid is a string (same across inits) -- carry the first
    if "launched_grid" in all_results[0]:
        aggregated["launched_grid"] = all_results[0]["launched_grid"]

    return aggregated


def roofline_mlp(
    batch_sizes,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype,
    w_dtype,
    TP,
    op_regex,
    name="",
    num_weight_inits=1,
    timing="events",
    routing_mode="uniform",
    skew=4.0,
    warmup=5,
    iters=20,
    dump_blocks=False,
    seed=None,
    gemm="both",
    swizzle=None,
):
    # Avoid creating an empty directory named like the output CSV stem.
    out_dir = Path("logs") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if gemm == "both" else f"-{gemm}"
    out_path = out_dir / f"{x_dtype}x-{w_dtype}w-TP{TP}-{routing_mode}{suffix}.csv"

    # Sweep batch sizes, benchmark each, print a summary line and write a CSV.
    results: list[tuple[int, dict[str, int | float]]] = []
    print("=========================================")
    print(f"{out_path}  (gemm={gemm})...")
    print("=========================================")

    for batch in batch_sizes:
        perf = bench_mlp(
            batch,
            dim1,
            dim2,
            n_expts_tot,
            n_expts_act,
            x_dtype,
            w_dtype,
            TP,
            op_regex,
            num_weight_inits,
            timing=timing,
            routing_mode=routing_mode,
            skew=skew,
            warmup=warmup,
            iters=iters,
            dump_blocks=dump_blocks,
            seed=seed,
            gemm=gemm,
            swizzle=swizzle,
        )
        results.append((batch, perf))

        kt = perf["kernel_time_ns"] or float("nan")
        tflops = perf["flops"] / kt * 1e-3
        tbps = perf["bytes"] / kt * 1e-3
        kernel_latency_us = perf["kernel_time_ns"] / 1e3 / perf["reps"]
        # events mode carries explicit latency stats; proton mode does not
        if "avg_ms" in perf:
            lat = (
                f"avg: {perf['avg_ms']:.4f} ms | min: {perf['min_ms']:.4f} ms | "
                f"active experts: {perf.get('active_experts', '-')}"
            )
        else:
            lat = f"Kernel latency (us): {kernel_latency_us:.2f}"
        print(
            f"batch: {batch:6d} | {lat} | "
            f"TFLOPS: {tflops:#.4g} | TBPS: {tbps:.2f} | "
            f"grid[{perf.get('launched_grid', '-')}]"
        )

    # write CSV
    fieldnames = [
        "batch",
        "avg_ms",
        "min_ms",
        "max_ms",
        "median_ms",
        "active_experts",
        "launched_grid",
        "kernel_latency_us",
        "tflops",
        "tbps",
        "total_time_ns",
        "kernel_time_ns",
        "flops",
        "bytes",
        "reps",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for batch, perf in results:
            kt = perf["kernel_time_ns"] or float("nan")
            w.writerow(
                {
                    "batch": batch,
                    "avg_ms": perf.get("avg_ms", ""),
                    "min_ms": perf.get("min_ms", ""),
                    "max_ms": perf.get("max_ms", ""),
                    "median_ms": perf.get("median_ms", ""),
                    "active_experts": perf.get("active_experts", ""),
                    "launched_grid": perf.get("launched_grid", ""),
                    "kernel_latency_us": perf["kernel_time_ns"] / 1e3 / perf["reps"],
                    "tflops": perf["flops"] / kt * 1e-3,
                    "tbps": perf["bytes"] / kt * 1e-3,
                    "total_time_ns": perf["total_time_ns"],
                    "kernel_time_ns": perf["kernel_time_ns"],
                    "flops": perf["flops"],
                    "bytes": perf["bytes"],
                    "reps": perf["reps"],
                }
            )


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="Benchmark MoE")

    parser.add_argument(
        "--M",
        type=int,
        nargs="+",
        default=None,
        help="MoE batch sizes M (one or more integers). "
        "If not set, a predermined list of values will be used.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        metavar=("DIM"),
        help="Input feature dimensions of MoE layers. Must be two integers.",
    )
    parser.add_argument(
        "--experts",
        type=int,
        nargs="+",
        metavar=("DIM"),
        help="Number of total and active experts in [total experts, active experts] order.",
    )
    parser.add_argument(
        "--op-regex",
        type=str,
        default=".*moe_gemm.*",
        help="Regex to find perf for specific operation by its kernel name.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "triton", "gluon"],
        help="moe_gemm_a16w4 backend to benchmark. 'gluon' targets the tuned "
        "gfx1250 gluon kernel; 'auto' (default) lets the op pick per-shape.",
    )
    parser.add_argument(
        "--num-weight-inits",
        type=int,
        default=1,
        help="Number of different weight initializations to run for more stable results (default: 1). "
        "Each initialization runs 100 iterations. Use higher values (e.g., 10) for more stable benchmarks.",
    )
    parser.add_argument(
        "--timing",
        type=str,
        default="events",
        choices=["events", "proton", "do_bench"],
        help="Timing method. 'events' (default): repeated CUDA-event runs, no "
        "profiler; 'do_bench': triton testing.do_bench (L2-flushing, time-based); "
        "'proton': triton proton profiler (needs librocprofiler-sdk). Tuning uses "
        "do_bench/events (proton falls back to events).",
    )
    parser.add_argument(
        "--routing",
        type=str,
        default="uniform",
        choices=["uniform", "skewed"],
        help="Synthetic routing distribution. 'uniform' (default) spreads tokens "
        "over many experts; 'skewed' biases a subset of experts to mimic the "
        "concentrated load of real captured routing (fewer experts active).",
    )
    parser.add_argument(
        "--skew",
        type=float,
        default=4.0,
        help="Strength of per-expert bias when --routing skewed (larger => more "
        "concentrated, fewer active experts). Default: 4.0.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations per shape for --timing events (default: 5).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Timed iterations per shape for --timing events (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix the routing RNG seed so the active-expert distribution (which "
        "drives latency) is identical per M across process runs, making A/B "
        "comparisons (e.g. --gluon-stage 1 vs 2) fair. Routing is keyed on "
        "(seed, M) via a dedicated generator; weight init uses torch.manual_seed"
        "(seed). Default (None) reproduces the old non-deterministic behaviour.",
    )
    parser.add_argument(
        "--dump-blocks",
        action="store_true",
        help="For each shape, print the per-launched-M-block routing layout "
        "(expert id + non-padding token count) decoded from block_pid_map.",
    )
    parser.add_argument(
        "--gluon-stage",
        type=int,
        default=None,
        help="Force the gluon pipeline stage (= NUM_BUFFERS) for every shape: "
        "1 = single-buffer (stage-1), 2 = LDS-prefetch double-buffer (stage-2), "
        "3 = triple-buffer (stage-3). Default (None) uses the per-shape tuned "
        "value from get_kernel_config_gluon. Capped at cdiv(K, block_k) at launch.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Autotune instead of benchmarking: sweep the config search space "
        "(_TUNE_SPACE) for GEMM1 and GEMM2 of every M and report the fastest "
        "configs. --backend / --gluon-stage pin the backend / stage tuned.",
    )
    parser.add_argument(
        "--tune-gap-percent",
        type=float,
        default=5.0,
        help="When tuning, only return configs within this percent of the best "
        "config's time (default: 5).",
    )
    parser.add_argument(
        "--tune-fused",
        action="store_true",
        help="Tune each GEMM inside the full two-GEMM layer (only the tuned GEMM's "
        "config varies; the other stays at its resolved config), so cache effects "
        "match production. Default is isolated single-GEMM timing, which can "
        "over-promise vs the fused layer.",
    )
    parser.add_argument(
        "--golden-cpu",
        action="store_true",
        help="When tuning, compute the correctness reference (golden) on CPU "
        "instead of GPU (per-expert upcast still runs on GPU then moves to CPU), "
        "freeing GPU memory for the kernel on large MiniMax-M3 shapes.",
    )
    parser.add_argument(
        "--tune-cand-number",
        type=int,
        default=3,
        help="When tuning, return at most this many configs per shape, fastest "
        "first (default: 3).",
    )
    parser.add_argument(
        "--gemm",
        choices=["both", "gemm1", "gemm2"],
        default="both",
        help="Which projection to benchmark/tune: 'gemm1' = gate/up (gather + "
        "swiglu), 'gemm2' = down (scatter, no swiglu), 'both' = full two-GEMM "
        "layer (default). 'gemm1'/'gemm2' are handled separately.",
    )
    parser.add_argument(
        "--swizzle",
        choices=["auto", "none", "cdna4", "gfx1250"],
        default="auto",
        help="MX weight-scale layout for benchmarking. 'auto' (default) picks per "
        "arch: CDNA4 on the MI350 series (gfx950), none/compact on the MI450 "
        "series (gfx1250). 'none' keeps compact scales (required for the gluon "
        "backend); 'cdna4'/'gfx1250' force that swizzle (which forces the triton "
        "backend, since gluon accepts only compact scales). Applies to "
        "benchmarking; --tune always uses compact scales.",
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)

    # Build the config dict forwarded to every moe_gemm_a16w4 call. "auto" leaves
    # the backend unpinned (the op's `auto` config picks per shape); "triton" /
    # "gluon" pin the backend, and --gluon-stage pins the gluon pipeline stage.
    global _MOE_CONFIG
    cli_config = {}
    if parsed_args.backend != "auto":
        cli_config["backend"] = parsed_args.backend
    if parsed_args.gluon_stage is not None:
        cli_config["num_buffers"] = parsed_args.gluon_stage
    _MOE_CONFIG = cli_config or None
    print(f"moe_gemm_a16w4 config: {_MOE_CONFIG}")

    gemm_sel = parsed_args.gemm  # "both" | "gemm1" | "gemm2"

    # Resolve the MX-scale swizzle: "auto" -> arch default, else the named layout.
    _SWIZZLE_MAP = {
        "none": None,
        "cdna4": "CDNA4_SCALE",
        "gfx1250": "GFX1250_SCALE",
    }
    if parsed_args.swizzle == "auto":
        swizzle = _default_swizzle()
    else:
        swizzle = _SWIZZLE_MAP[parsed_args.swizzle]
    print(f"MX scale swizzle: {swizzle} (--swizzle {parsed_args.swizzle})")

    dim1, dim2 = parsed_args.shape
    total_experts, active_experts = parsed_args.experts
    if parsed_args.M is None:
        batch_sizes_moe = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 8192]
    else:
        batch_sizes_moe = parsed_args.M

    quantized_dtypes = ["bf16", "mx4"]

    if parsed_args.tune:
        for batch in batch_sizes_moe:
            tune_shape(
                batch,
                dim1,
                dim2,
                total_experts,
                active_experts,
                quantized_dtypes[0],
                quantized_dtypes[1],
                TP=1,
                routing_mode=parsed_args.routing,
                skew=parsed_args.skew,
                seed=parsed_args.seed,
                gluon_stage=parsed_args.gluon_stage,
                gap=parsed_args.tune_gap_percent,
                cand=parsed_args.tune_cand_number,
                warmup=parsed_args.warmup,
                iters=parsed_args.iters,
                gemm=gemm_sel,
                timing=parsed_args.timing,
                fused=parsed_args.tune_fused,
                golden_cpu=parsed_args.golden_cpu,
            )
        return

    roofline_mlp(
        batch_sizes_moe,
        dim1,
        dim2,
        total_experts,
        active_experts,
        quantized_dtypes[0],
        quantized_dtypes[1],
        TP=1,
        op_regex=parsed_args.op_regex,
        name=f"gpt-oss-x2-{parsed_args.backend}",
        num_weight_inits=parsed_args.num_weight_inits,
        timing=parsed_args.timing,
        routing_mode=parsed_args.routing,
        skew=parsed_args.skew,
        warmup=parsed_args.warmup,
        iters=parsed_args.iters,
        dump_blocks=parsed_args.dump_blocks,
        seed=parsed_args.seed,
        gemm=gemm_sel,
        swizzle=swizzle,
    )


if __name__ == "__main__":
    main()
