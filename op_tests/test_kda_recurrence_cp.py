# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``aiter.ops.kda.kda_recurrence`` chunkwise context parallelism, ranks emulated in one process.

The packed batch (global ``cu_seqlens``) is split into ``world`` contiguous
intervals with ``build_kda_cp_context``. Candidates:

* ``nocp``: the whole batch in one ``kda_recurrence`` call.
* ``cp_autograd``: one thread per rank through ``kda_recurrence(cp_context=)``
  (in ``megatron`` mode through the Megatron backend ``chunk_kda``, with
  Megatron's keywords), forward and ``loss.backward()``, the all-gathers
  served by an in-process group. CPU only: CUDA autograd runs every backward
  on one device thread, so rank threads blocked in the all-gather would never
  meet.
* ``cp_staged``: the same per-rank stages (``kda_cp_forward_local`` ->
  all-gather -> ``kda_cp_forward_finish``, ``kda_cp_backward_packet`` ->
  all-gather -> ``kda_cp_backward_finish`` -> local bwd) driven rank by rank.

Everything is graded against autograd through the token-by-token fp32
reference (``op_tests/triton_tests/utils/kda_ref.py``) on the unsplit batch:
the output, the final state at each sequence's last token, and the gradients
of q, k, v, g, beta (rank shards concatenated) and of A_log, dt_bias (summed
over ranks, as the CP-DP gradient all-reduce does). The loss weights both the
outputs and those final states, so final-state gradients cross ranks too.

``-k torch_ref`` runs the same reference as the local kernels, which isolates
the CP algebra (fp32, tight tolerance) and needs no GPU; ``triton`` and
``flydsl`` run aiter's chunk kernels (bf16 criteria, see ``CRITERIA``). The
column ``cp_staged vs nocp rel_l2`` is the error CP adds over the same
kernels. Every case runs and the tables print before the run fails on any
case. Correctness only: kernel timing is in ``test_kda_chunk_fwd.py`` /
``test_kda_chunk_bwd.py``.
"""

import argparse
import bisect
import importlib.util
import itertools
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter.ops.kda import (
    KdaKernels,
    KdaOptions,
    build_kda_cp_context,
    chunk_kda,
    get_kda_kernels,
    kda_cp_backward_finish,
    kda_cp_backward_packet,
    kda_cp_boundary,
    kda_cp_forward_finish,
    kda_cp_forward_local,
    kda_recurrence,
)
from aiter.test_common import benchmark, checkAllclose

# Loaded by path: the op_tests.triton_tests package init needs a GPU driver,
# and the torch_ref sweep runs without one.
_spec = importlib.util.spec_from_file_location(
    "kda_ref",
    os.path.join(os.path.dirname(__file__), "triton_tests", "utils", "kda_ref.py"),
)
_kda_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kda_ref)
chunk_kda_ref, l2norm_ref = _kda_ref.chunk_kda_ref, _kda_ref.l2norm_ref

LOWER_BOUND = -5.0
MODES = {
    # Megatron #6872's call: q/k L2-normalized by the pre-KDA fusion, raw gate
    # with the bounded in-kernel activation, post-sigmoid fp32 beta.
    "megatron": {
        "use_gate_in_kernel": True,
        "safe_gate": True,
        "lower_bound": LOWER_BOUND,
    },
    # Every activation in kernel (the Lumen#35 setting).
    "kimi": {
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "safe_gate": True,
        "lower_bound": LOWER_BOUND,
    },
    # Precomputed log decay, unbounded.
    "plain": {},
}
GRADS = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
# fp32 (torch_ref): every element within rtol/atol. bf16 kernels against the
# fp32 reference: relative L2 error per tensor (the 1e-2 of
# test_chunk_delta_attn_bwd.py) plus at most 1e-3 of the elements outside
# rtol/atol (the ratio test_kda_chunk_fwd.py allows). A zero-outlier bound
# does not hold in bf16: with q/k normalized outside the kernel (megatron
# mode) dk is not damped by the l2norm backward, and an element that cancels
# large bf16-rounded terms lands outside any fixed per-element tolerance.
CRITERIA = {
    torch.float32: {
        "rel_l2": 1e-5,
        "rtol": 1e-4,
        "atol": 1e-5,
        "outliers": 0.0,
        "sum_terms": 1e-6,
    },
    torch.bfloat16: {
        "rel_l2": 1e-2,
        "rtol": 3e-2,
        "atol": 3e-2,
        "outliers": 1e-3,
        "sum_terms": 1e-3,
    },
}


def _ref_fwd(q, k, v, g, beta, **kw):
    with torch.no_grad():
        return chunk_kda_ref(q, k, v, g, beta, **kw)


def _ref_bwd(do, dht, q, k, v, g, beta, *, A_log, dt_bias, initial_state, **kw):
    xs = [
        None if t is None else t.detach().requires_grad_()
        for t in (q, k, v, g, beta, A_log, dt_bias, initial_state)
    ]
    with torch.enable_grad():
        o, ht = chunk_kda_ref(
            *xs[:5],
            A_log=xs[5],
            dt_bias=xs[6],
            initial_state=xs[7],
            output_final_state=True,
            **kw,
        )
        outs, gouts = [o], [do.to(o.dtype)]
        if dht is not None:
            outs.append(ht)
            gouts.append(dht.float())
        live = [t for t in xs if t is not None]
        grads = iter(torch.autograd.grad(outs, live, gouts, allow_unused=True))
    return tuple(None if t is None else next(grads) for t in xs)


TORCH_REF_KERNELS = KdaKernels("torch_ref", _ref_fwd, _ref_bwd)


def _kernels(name):
    return TORCH_REF_KERNELS if name == "torch_ref" else get_kda_kernels(name)


class _SharedGather:
    """In-process all-gather: one slot per rank, a barrier on either side."""

    def __init__(self, world):
        self.world = world
        self.slots = [None] * world
        self.barrier = threading.Barrier(world, timeout=600)


class _EmulatedRank:
    """The ``rank()`` / ``size()`` / ``all_gather_into_tensor`` group interface."""

    def __init__(self, shared, rank):
        self.shared, self._rank = shared, rank

    def rank(self):
        return self._rank

    def size(self):
        return self.shared.world

    def all_gather_into_tensor(self, out, inp):
        s = self.shared
        s.slots[self._rank] = inp
        s.barrier.wait()
        out.copy_(torch.cat(s.slots, dim=0))
        s.barrier.wait()


def make_inputs(T, H, HV, D, dtype, device, mode, layout, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape, dt=dtype):
        return torch.randn(*shape, generator=gen).to(device=device, dtype=dt)

    x = {
        "q": r(1, T, H, D),
        "k": r(1, T, H, D),
        "v": r(1, T, HV, D),
        "g": r(1, T, HV, D),
        "beta": r(1, T, HV, dt=torch.float32),
        "A_log": r(HV, dt=torch.float32) * 0.5,
        "dt_bias": r(HV * D, dt=torch.float32) * 0.5,
    }
    opts = MODES[mode]
    if not opts.get("use_qk_l2norm_in_kernel"):
        x["q"], x["k"] = (l2norm_ref(x[n]).to(dtype) for n in ("q", "k"))
    if not opts.get("use_beta_sigmoid_in_kernel"):
        x["beta"] = torch.sigmoid(x["beta"])
    if not opts.get("use_gate_in_kernel"):
        x["g"] = F.logsigmoid(x["g"].float() + 3).to(dtype)
        del x["A_log"], x["dt_bias"]
    if layout == "one_seq":
        bounds = [0, T]
    elif layout == "aligned":
        bounds = [0, T // 2, T]
    else:  # "packed": uneven sequences, one shorter than a rank interval
        bounds = [0, T * 37 // 256, T * 150 // 256, T * 161 // 256, T]
    w_o = r(1, T, HV, D, dt=torch.float32)
    w_h = r(len(bounds) - 1, HV, D, D, dt=torch.float32)
    return x, bounds, w_o, w_h


def _leaf(t):
    return None if t is None else t.detach().clone().requires_grad_()


def reference(x, bounds, w_o, w_h, opts, h0=None):
    xs = {n: _leaf(t) for n, t in x.items()}
    h0 = _leaf(h0)
    cu = torch.tensor(bounds, dtype=torch.int32, device=x["q"].device)
    o, ht = chunk_kda_ref(
        xs["q"],
        xs["k"],
        xs["v"],
        xs["g"],
        xs["beta"],
        A_log=xs.get("A_log"),
        dt_bias=xs.get("dt_bias"),
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
        **opts,
    )
    ((o.float() * w_o).sum() + (ht * w_h).sum()).backward()
    grads = {n: t.grad for n, t in xs.items()}
    if h0 is not None:
        grads["h0"] = h0.grad
    return o.detach(), ht.detach(), grads


def _rank_slice(rank, world, T):
    part = T // world
    return slice(rank * part, (rank + 1) * part)


def _ends_here(bounds, ctx, rank, world, T):
    """Local sequence index -> global sequence index, for sequences ending on this rank."""
    sl = _rank_slice(rank, world, T)
    first = bisect.bisect_right(bounds, sl.start) - 1
    n_local = ctx.cu_seqlens_cpu.numel() - 1
    return {i: first + i for i in range(n_local) if bounds[first + i + 1] <= sl.stop}


def run_nocp(x, bounds, w_o, w_h, opts, kernels, h0=None):
    xs = {n: _leaf(t) for n, t in x.items()}
    h0 = _leaf(h0)
    cu = torch.tensor(bounds, dtype=torch.int32, device=x["q"].device)
    o, ht = kda_recurrence(
        xs["q"],
        xs["k"],
        xs["v"],
        xs["g"],
        xs["beta"],
        A_log=xs.get("A_log"),
        dt_bias=xs.get("dt_bias"),
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
        backend=kernels,
        **opts,
    )
    ((o.float() * w_o).sum() + (ht * w_h).sum()).backward()
    grads = {n: t.grad for n, t in xs.items()}
    if h0 is not None:
        grads["h0"] = h0.grad
    return o.detach(), ht.detach(), grads


def _gather_results(x, bounds, world, per_rank):
    """Concatenate rank outputs / token grads, sum parameter grads, pick true final states."""
    o = torch.cat([p["o"] for p in per_rank], dim=1)
    ht = [None] * (len(bounds) - 1)
    for p in per_rank:
        for i, n in p["ends"].items():
            ht[n] = p["ht"][i]
    grads = {}
    for n in x:
        gs = [p["grads"][n] for p in per_rank]
        grads[n] = (
            torch.cat(gs, dim=1) if n in ("q", "k", "v", "g", "beta") else sum(gs)
        )
    return o, torch.stack(ht), grads


def run_cp_autograd(x, bounds, w_o, w_h, opts, kernels, world, via_megatron=False):
    T = x["q"].shape[1]
    shared = _SharedGather(world)
    cu = torch.tensor(bounds, dtype=torch.int32, device=x["q"].device)
    per_rank = [None] * world

    def rank_main(rank):
        try:
            ctx = build_kda_cp_context(cu, _EmulatedRank(shared, rank))
            sl = _rank_slice(rank, world, T)
            xs = {
                n: _leaf(t[:, sl] if n in ("q", "k", "v", "g", "beta") else t)
                for n, t in x.items()
            }
            if via_megatron:
                # Megatron's kda.py call, keyword for keyword; the global
                # cu_seqlens it passes must give way to the context's.
                o, ht = chunk_kda(
                    q=xs["q"],
                    k=xs["k"],
                    v=xs["v"],
                    g=xs["g"],
                    beta=xs["beta"],
                    A_log=xs["A_log"],
                    dt_bias=xs["dt_bias"],
                    initial_state=None,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=False,
                    use_gate_in_kernel=True,
                    safe_gate=True,
                    lower_bound=LOWER_BOUND,
                    cu_seqlens=cu,
                    cp_context=ctx,
                    backend=kernels,
                )
            else:
                o, ht = kda_recurrence(
                    xs["q"],
                    xs["k"],
                    xs["v"],
                    xs["g"],
                    xs["beta"],
                    A_log=xs.get("A_log"),
                    dt_bias=xs.get("dt_bias"),
                    output_final_state=True,
                    cp_context=ctx,
                    backend=kernels,
                    **opts,
                )
            ends = _ends_here(bounds, ctx, rank, world, T)
            loss = (o.float() * w_o[:, sl]).sum()
            for i, n in ends.items():
                loss = loss + (ht[i] * w_h[n]).sum()
            loss.backward()
            per_rank[rank] = {
                "o": o.detach(),
                "ht": ht.detach(),
                "ends": ends,
                "grads": {n: t.grad for n, t in xs.items()},
            }
        finally:
            # A failed rank releases its peers from the all-gather barrier.
            if per_rank[rank] is None:
                shared.barrier.abort()

    with ThreadPoolExecutor(max_workers=world) as pool:
        futures = [pool.submit(rank_main, r) for r in range(world)]
    errors = [f.exception() for f in futures if f.exception() is not None]
    root = [e for e in errors if not isinstance(e, threading.BrokenBarrierError)]
    if errors:
        raise (root or errors)[0]
    return _gather_results(x, bounds, world, per_rank)


def run_cp_staged(x, bounds, w_o, w_h, opts, kernels, world):
    T, K = x["q"].shape[1], x["q"].shape[-1]
    shared = _SharedGather(world)
    cu = torch.tensor(bounds, dtype=torch.int32, device=x["q"].device)
    kopts = KdaOptions(scale=K**-0.5, **opts)
    ranks = []
    for r in range(world):
        ctx = build_kda_cp_context(cu, _EmulatedRank(shared, r))
        sl = _rank_slice(r, world, T)
        xs = {
            n: (t[:, sl] if n in ("q", "k", "v", "g", "beta") else t).contiguous()
            for n, t in x.items()
        }
        bnd = kda_cp_boundary(ctx, sl.stop - sl.start)
        ranks.append({"ctx": ctx, "sl": sl, "x": xs, "bnd": bnd})

    def args(p):
        xs = p["x"]
        return (xs["q"], xs["k"], xs["v"], xs["g"], xs["beta"])

    for p in ranks:
        p["fwd"] = kda_cp_forward_local(
            *args(p),
            p["x"].get("A_log"),
            p["x"].get("dt_bias"),
            cu_seqlens=p["ctx"].cu_seqlens,
            bnd=p["bnd"],
            kernels=kernels,
            opts=kopts,
        )
    gathered = torch.stack([p["fwd"][-1] for p in ranks])
    for p in ranks:
        o, ht, o_first, m_first, _ = p["fwd"]
        p["o"], p["ht"], p["h0"] = kda_cp_forward_finish(
            o, ht, o_first, m_first, gathered, bnd=p["bnd"]
        )
        p["o_first"], p["m_first"] = o_first, m_first
        p["ends"] = _ends_here(bounds, p["ctx"], p["bnd"].rank, world, T)
        p["do"] = w_o[:, p["sl"]].to(o.dtype)
        dht = torch.zeros_like(p["ht"])
        for i, n in p["ends"].items():
            dht[i] = w_h[n]
        p["dht"] = dht
    gathered = torch.stack(
        [
            kda_cp_backward_packet(
                p["do"], p["dht"], p["o_first"], p["m_first"], key_dim=K, bnd=p["bnd"]
            )
            for p in ranks
        ]
    )
    for p in ranks:
        dht = kda_cp_backward_finish(
            p["dht"], gathered, num_seqs=p["bnd"].num_seqs, bnd=p["bnd"]
        )
        g = kernels.bwd(
            p["do"],
            dht,
            *args(p),
            A_log=p["x"].get("A_log"),
            dt_bias=p["x"].get("dt_bias"),
            initial_state=p["h0"],
            cu_seqlens=p["ctx"].cu_seqlens,
            **kopts.kwargs(),
        )
        p["grads"] = {n: gi for n, gi in zip(GRADS, g) if n in p["x"]}
    return _gather_results(x, bounds, world, ranks)


def _rel_l2(ref, out):
    return (
        (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)
    ).item()


def sum_terms(x, ref_grads, HV, D):
    """``sum |dg * (g + dt_bias)|`` per head: the summands of dA_log, for its bound."""
    if "A_log" not in x:
        return None
    z = x["g"].float() + x["dt_bias"].float().view(HV, D)
    return (ref_grads["g"].float() * z).abs().sum((0, 1, 3))


def grade(name, ref, out, dtype, terms=None):
    """Per-tensor verdicts against the reference; returns ``(row, failures)``, never raises.

    Every tensor but dA_log needs ``rel_l2`` and the outlier fraction within
    ``CRITERIA``. dA_log, a per-head sum over every token and channel that
    largely cancels, is bounded per head by ``sum_terms`` times the size of its
    summands.
    """
    crit = CRITERIA[dtype]
    (o_ref, ht_ref, g_ref), (o, ht, grads) = ref, out
    checks = [("o", o_ref, o), ("ht", ht_ref, ht)]
    checks += [(f"d{n}", g_ref[n], grads[n]) for n in g_ref]
    row, failures = {}, []
    worst_name, worst_score = "", -1.0
    worst_rel = worst_out = 0.0
    for label, a, b in checks:
        a, b = a.float(), b.float()
        rel = _rel_l2(a, b)
        frac = checkAllclose(
            a,
            b,
            rtol=crit["rtol"],
            atol=crit["atol"],
            tol_err_ratio=crit["outliers"],
            msg=f"{name}: {label}",
            printLog=False,
        )
        if label == "dA_log" and terms is not None:
            score = ((b - a).abs() / (crit["sum_terms"] * terms)).max().item()
            ok = score <= 1.0
        else:
            score = max(rel / crit["rel_l2"], frac / max(crit["outliers"], 1e-12))
            ok = rel <= crit["rel_l2"] and frac <= crit["outliers"]
            worst_rel, worst_out = max(worst_rel, rel), max(worst_out, frac)
        if score > worst_score:
            worst_name, worst_score = label, score
        if not ok:
            failures.append(
                f"{name} {label}: rel_l2 {rel:.2e}, outliers {frac:.2e}, score {score:.2f}"
            )
    row[f"{name} rel_l2"] = worst_rel
    row[f"{name} outliers"] = worst_out
    row[f"{name} worst"] = f"{worst_name} {worst_score:.2f}"
    row[f"{name} ok"] = not failures
    return row, failures


@benchmark()
def test_kda_cp(backend, mode, layout, world, T, H, HV, D, dtype):
    device = "cpu" if backend == "torch_ref" else "cuda"
    kernels = _kernels(backend)
    opts = MODES[mode]
    x, bounds, w_o, w_h = make_inputs(T, H, HV, D, dtype, device, mode, layout)
    ref = reference(x, bounds, w_o, w_h, opts)
    candidates = {
        "nocp": lambda: run_nocp(x, bounds, w_o, w_h, opts, kernels),
        "cp_staged": lambda: run_cp_staged(x, bounds, w_o, w_h, opts, kernels, world),
    }
    if device == "cpu":
        candidates["cp_autograd"] = lambda: run_cp_autograd(
            x, bounds, w_o, w_h, opts, kernels, world, via_megatron=mode == "megatron"
        )
    terms = sum_terms(x, ref[2], HV, D)
    ret, failures, outs = {"bounds": str(bounds)}, [], {}
    for name, fn in candidates.items():
        outs[name] = fn()
        row, fails = grade(name, ref, outs[name], dtype, terms)
        ret.update(row)
        failures += fails
    # Informational: the error CP adds on top of the same kernels without CP.
    ret["cp_staged vs nocp rel_l2"] = max(
        _rel_l2(a, b)
        for a, b in zip(
            [outs["nocp"][0], outs["nocp"][1], *outs["nocp"][2].values()],
            [
                outs["cp_staged"][0],
                outs["cp_staged"][1],
                *outs["cp_staged"][2].values(),
            ],
        )
    )
    ret["failures"] = "; ".join(failures)
    return ret


@benchmark()
def test_kda_initial_state(backend, mode, T, H, HV, D, dtype):
    """Non-CP ``initial_state`` / final-state gradients (``dh0``, ``dht``)."""
    device = "cpu" if backend == "torch_ref" else "cuda"
    kernels = _kernels(backend)
    opts = MODES[mode]
    x, bounds, w_o, w_h = make_inputs(T, H, HV, D, dtype, device, mode, "packed")
    gen = torch.Generator(device="cpu").manual_seed(7)
    h0 = torch.randn(len(bounds) - 1, HV, D, D, generator=gen).to(device) * 0.1
    ref = reference(x, bounds, w_o, w_h, opts, h0=h0)
    out = run_nocp(x, bounds, w_o, w_h, opts, kernels, h0=h0)
    ret, failures = grade("nocp", ref, out, dtype, sum_terms(x, ref[2], HV, D))
    ret["dh0 rel_l2"] = _rel_l2(ref[2]["h0"], out[2]["h0"])
    ret["failures"] = "; ".join(failures)
    return ret


def summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="KDA recurrence chunkwise CP, emulated ranks",
    )
    parser.add_argument(
        "-k",
        "--backend",
        nargs="*",
        default=None,
        choices=["torch_ref", "triton", "flydsl", "auto"],
        help="Local kernels. Default: torch_ref without a GPU, else triton.",
    )
    parser.add_argument(
        "-m", "--mode", nargs="*", default=list(MODES), choices=list(MODES)
    )
    parser.add_argument(
        "-l",
        "--layout",
        nargs="*",
        default=["one_seq", "packed", "aligned"],
        choices=["one_seq", "packed", "aligned"],
    )
    parser.add_argument("-w", "--world", type=int, nargs="*", default=[1, 2, 4, 8])
    parser.add_argument(
        "-s",
        "--shape",
        type=lambda s: tuple(int(v) for v in s.split(",")),
        nargs="*",
        default=None,
        help="T,H,HV,D. Default: 256,2,2,16 and 256,2,4,16 (GVA) for torch_ref, "
        "1024,4,4,128 on a GPU.",
    )
    args = parser.parse_args()
    has_gpu = torch.cuda.is_available()
    backends = args.backend or (["triton"] if has_gpu else ["torch_ref"])
    if not has_gpu and any(b != "torch_ref" for b in backends):
        aiter.logger.warning("no GPU: running torch_ref only")
        backends = ["torch_ref"]
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // 2))

    failures = []
    for backend in backends:
        dtype = torch.float32 if backend == "torch_ref" else torch.bfloat16
        shapes = args.shape or (
            [(256, 2, 2, 16), (256, 2, 4, 16)]
            if backend == "torch_ref"
            else [(1024, 4, 4, 128)]
        )
        rows = [
            test_kda_cp(backend, mode, layout, world, T, H, HV, D, dtype)
            for mode, layout, world, (T, H, HV, D) in itertools.product(
                args.mode, args.layout, args.world, shapes
            )
        ]
        summarize(f"KDA chunkwise CP [{backend}]", rows)
        failures += [r["failures"] for r in rows if r["failures"]]
        rows = [
            test_kda_initial_state(backend, mode, T, H, HV, D, dtype)
            for mode, (T, H, HV, D) in itertools.product(args.mode, shapes)
        ]
        summarize(f"KDA initial/final state [{backend}]", rows)
        failures += [r["failures"] for r in rows if r["failures"]]
    assert not failures, f"{len(failures)} failing cases:\n" + "\n".join(failures)


if __name__ == "__main__":
    main()
