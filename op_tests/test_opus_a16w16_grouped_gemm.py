# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 grouped GEMMs of a MoE grouped linear layer (OPUS gfx950 4-wave grouped kids), DeepGEMM interfaces.

Per expert g of one rank (rows of g = the tokens routed to it, M_e of them):
    fwd    Y_g  = X_g W[g]^T     m_grouped_bf16_gemm_nt_contiguous   W [G, N, K]
    dgrad  dX_g = dY_g W[g]      m_grouped_bf16_gemm_nn_contiguous   W [G, N, K] read as [G, K', N']
    wgrad  dW[g] += dY_g^T X_g   k_grouped_bf16_gemm_tn_contiguous   fp32 main_grad [G, N, K]
The padded contiguous layout: every group padded to get_mk_alignment_for_contiguous_layout() rows,
padding rows hold NaN (never read into a valid result). masked: inference decode, a [G, max_m, K].

Shapes (-c / --config): per-rank G and tokens ΣM of the training runs
    glm-*  GLM-5.3 (#37): 256 experts top-8, EP8 -> G = 32, ΣM = S
           fc1 (gate+up) fwd N x K = 4096 x 6144, fc2 6144 x 2048
    k3-*   Kimi-K3 LatentMoE (#35): 896 experts top-16 in the 3584 latent; EP8 -> G = 112, ΣM = 2S;
           single GPU G = 896, ΣM = 16S. fc1 (w1+w3) 6144 x 3584, fc2 (w2) 3584 x 3072
dgrad swaps N and K of fwd; wgrad's output is fwd's W [G, N, K] with K = M_e.
Token distributions (-t / --dist), mean M_e = ΣM / G:
    balanced   every group the mean
    deepgemm   mean x U(0.7, 1.3), every fourth group empty
    zipf       Zipf over a shuffled group order, max ~4x the mean, 10% of the groups empty
    one_group  every token in one group
    tiny       M_e uniform in [1, 63]
Edge cases (-p edge): shapes off the tile grid (N not a multiple of 8 / of a tile), more than 64 and
more than 1024 groups, K-grouped bf16 / fp32 / fp32 += with and without a split, and the K-grouped
layout check. Repeated launches (-p repeat): the same call --launches times, back to back in bursts
(nothing waits between the launches of a burst), every output bit-equal to the first, checked one
(the plan kernels: a stale read gives a group the wrong rows).
Baselines: triton gmm / ptgmm / nptgmm (compact layout, no padding), SonicMoE grouped_gemm (#6483,
its default triton backend; wgrad: fp32 written, no accumulation), hipBLASLt grouped mm, and
hipblaslt_dense: the dense hipBLASLt GEMM (torch.mm) of the same FLOPs (M = ΣM; wgrad K = ΣM), the
upper reference.
"TFLOPS" counts the real rows, "pad TFLOPS" (opus) the padded ones.
"""

import argparse
import functools
import itertools
import random

import pandas as pd
import torch
import triton

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.gradlib import hipb_grouped_mm
from aiter.ops.opus import (
    get_mk_alignment_for_contiguous_layout,
    k_grouped_bf16_gemm_tn_contiguous,
    m_grouped_bf16_gemm_nn_contiguous,
    m_grouped_bf16_gemm_nt_contiguous,
    m_grouped_bf16_gemm_nt_masked,
)
from aiter.ops.sonicmoe import grouped_gemm as sonicmoe_grouped_gemm
from aiter.ops.triton.gemm.grouped.gmm import gmm, nptgmm, ptgmm
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]

# name: (G, ΣM per rank, {gemm: fwd (N, K)})
_GLM = {"fc1": (4096, 6144), "fc2": (6144, 2048)}
_K3 = {"fc1": (6144, 3584), "fc2": (3584, 3072)}
CONFIGS = {
    "glm-S2048": (32, 2048, _GLM),
    "glm-S8192": (32, 8192, _GLM),
    "glm-S69632": (32, 69632, _GLM),
    "k3-S8192-ep8": (112, 2 * 8192, _K3),
    "k3-S32768-ep8": (112, 2 * 32768, _K3),
    "k3-S8192-1gpu": (896, 16 * 8192, _K3),
}


def group_rows(num_groups, total, dist):
    """Rows per group (M_e) of one token distribution with total ΣM ~ total."""
    mean = total / num_groups
    if dist == "balanced":
        return [total // num_groups] * num_groups
    if dist == "deepgemm":
        return [
            0 if g % 4 == 3 else max(1, int(mean * random.uniform(0.7, 1.3)))
            for g in range(num_groups)
        ]
    if dist == "zipf":
        # weights rank^-s over the 90% non-empty groups, s set so that the largest is 4x the mean
        live = num_groups - num_groups // 10
        lo, hi = 0.0, 4.0
        for _ in range(50):
            s = (lo + hi) / 2
            w = [r**-s for r in range(1, live + 1)]
            lo, hi = (s, hi) if num_groups * w[0] / sum(w) < 4 else (lo, s)
        rows = [int(total * x / sum(w)) for x in w] + [0] * (num_groups - live)
        random.shuffle(rows)
        return rows
    if dist == "one_group":
        rows = [0] * num_groups
        rows[random.randrange(num_groups)] = total
        return rows
    if dist == "tiny":
        return [random.randint(1, 63) for _ in range(num_groups)]
    raise ValueError(dist)


def padded_layout(rows):
    """Padded contiguous layout: (start of each group, total rows, grouped_layout)."""
    align = get_mk_alignment_for_contiguous_layout()
    starts, total = [], 0
    for m in rows:
        starts.append(total)
        total += (m + align - 1) // align * align
    grouped_layout = torch.full((total,), -1, dtype=dtypes.i32)
    for g, (s, m) in enumerate(zip(starts, rows)):
        grouped_layout[s : s + m] = g
    return starts, total, grouped_layout


def to_padded(x, rows, starts, total):
    """The compact rows x [ΣM, D] in the padded layout, padding rows NaN."""
    out = torch.full((total, x.shape[1]), float("nan"), dtype=x.dtype)
    c = 0
    for s, m in zip(starts, rows):
        out[s : s + m] = x[c : c + m]
        c += m
    return out


@functools.cache
def hipb_available():
    """hipBLASLt grouped mm builds only against a hipBLASLt with GroupedGemm::setMaxWorkspaceBytes."""
    try:
        a = torch.zeros(1, 64, dtype=dtypes.bf16)
        hipb_grouped_mm(a, torch.zeros(1, 64, 64, dtype=dtypes.bf16), offsets([1]), a)
        return True
    except (
        RuntimeError,
        ImportError,
    ) as e:  # build failed (here or in a concurrent process)
        aiter.logger.warning(
            "hipb_grouped_mm unavailable, skipping it: %s", str(e)[:200]
        )
        return False


@functools.cache
def sonicmoe_available():
    """The SonicMoE triton kernel does not compile with every triton release."""
    try:
        a = torch.zeros(1, 64, dtype=dtypes.bf16)
        sonicmoe_grouped_gemm(
            a, torch.zeros(1, 64, 64, dtype=dtypes.bf16), offsets([1]), out=a.clone()
        )
        return True
    except triton.CompilationError as e:
        aiter.logger.warning(
            "sonicmoe grouped_gemm unavailable, skipping it: %s", str(e)[:200]
        )
        return False


def mixed_rows(m):
    """Rows the mixed-height kernel runs for a group of m rows: the 256 / 192 / 128-row tiles of least
    cost (100 / 91 / 65) holding them (mixed_counts in the pipeline)."""
    best = None
    for a2 in range(2):
        for a1 in range(4):
            a0 = max(0, -(-(m - 192 * a1 - 128 * a2) // 256))
            cost = 100 * a0 + 91 * a1 + 65 * a2
            if best is None or cost < best[0]:
                best = (cost, 256 * a0 + 192 * a1 + 128 * a2)
    return best[1] if m else 0


def offsets(rows):
    return torch.tensor([0, *itertools.accumulate(rows)], dtype=torch.int64)


def run_torch(a, b, layout, dtype=dtypes.fp32):
    """Reference of one group, computed in dtype: nt a @ b^T, nn a @ b, tn a^T @ b."""
    a, b = a.to(dtype), b.to(dtype)
    if layout == "nt":
        return a @ b.t()
    return a @ b if layout == "nn" else a.t() @ b


def check_groups(name, out_of, refs):
    """checkAllclose over every group (the reference one group at a time, with its tolerance);
    the worst mismatch fraction and the largest absolute error."""
    err, max_abs = 0.0, 0.0
    for g, ref, tol in refs():
        ref, out = ref.to(dtypes.fp32), out_of(g).to(dtypes.fp32)
        e = checkAllclose(
            ref, out, rtol=tol, atol=tol, msg=f"{name}: group {g} ", printLog=False
        )
        assert e == 0, f"{name}: group {g}: {e:.2%} of elements outside rtol/atol"
        err = max(err, e)
        if ref.numel():
            max_abs = max(max_abs, (out - ref).abs().max().item())
    return err, max_abs


def record(ret, name, us, flops, nbytes, err):
    ret[f"{name} us"] = us
    ret[f"{name} TFLOPS"] = flops / us / 1e6
    ret[f"{name} TB/s"] = nbytes / us / 1e6
    ret[f"{name} err"] = err


@benchmark()
def test_m_grouped(config, gemm, pass_, dist):
    """fwd (nt) / dgrad (nn) of one config, the padded contiguous layout."""
    G, total, gemms = CONFIGS[config]
    N, K = gemms[gemm]
    layout = "nt" if pass_ == "fwd" else "nn"
    if layout == "nn":  # dgrad: dX [ΣM, K_in] = dY [ΣM, N_out] W[g] [N_out, K_in]
        N, K = K, N
    rows = group_rows(G, total, dist)
    starts, m_pad, grouped_layout = padded_layout(rows)
    m_sum = sum(rows)
    a_compact = torch.randn(m_sum, K, dtype=dtypes.bf16)
    a = to_padded(a_compact, rows, starts, m_pad)
    # W [G, N_out, K_in] of fwd; dgrad reads it as b [G, K, N]
    b = torch.randn((G, N, K) if layout == "nt" else (G, K, N), dtype=dtypes.bf16)
    b *= K**-0.5
    d = torch.empty(m_pad, N, dtype=dtypes.bf16)
    d_compact = torch.empty(m_sum, N, dtype=dtypes.bf16)
    valid = grouped_layout >= 0

    def refs():
        c = 0
        for g, m in enumerate(rows):
            yield g, run_torch(a_compact[c : c + m], b[g], layout), 1e-2
            c += m

    compact_rows = list(itertools.accumulate([0, *rows]))
    opus = (
        m_grouped_bf16_gemm_nt_contiguous
        if layout == "nt"
        else m_grouped_bf16_gemm_nn_contiguous
    )
    # name: (output buffer, launch, group g of the output)
    candidates = {
        # the DeepGEMM call: grouped_layout only, the plan kernel counts the rows of each group
        "opus": (
            d,
            lambda: opus(a, b, d, grouped_layout),
            lambda g: d[starts[g] : starts[g] + rows[g]],
        ),
        # gmm's rhs is [G, K, N]: nt passes W as a transposed view
        "triton_gmm": (
            d_compact,
            lambda: gmm(
                a_compact,
                b.transpose(1, 2) if layout == "nt" else b,
                torch.tensor(rows, dtype=dtypes.i32),
                existing_out=d_compact,
            ),
            lambda g: d_compact[compact_rows[g] : compact_rows[g + 1]],
        ),
    }
    cu_seqlens = offsets(rows)
    # SonicMoE takes b [G, K, N]: fwd passes W as a transposed view
    b_sonic = b.transpose(1, 2) if layout == "nt" else b
    if sonicmoe_available():
        candidates["sonicmoe"] = (
            d_compact,
            lambda: sonicmoe_grouped_gemm(
                a_compact, b_sonic, cu_seqlens, out=d_compact
            ),
            lambda g: d_compact[compact_rows[g] : compact_rows[g + 1]],
        )
    if hipb_available():
        # hipBLASLt takes b [G, N, K]: dgrad needs a transposed copy of W (made outside the timing)
        b_hipb = b if layout == "nt" else b.transpose(1, 2).contiguous()
        candidates["hipb_grouped_mm"] = (
            d_compact,
            lambda: hipb_grouped_mm(a_compact, b_hipb, cu_seqlens, d_compact),
            lambda g: d_compact[compact_rows[g] : compact_rows[g + 1]],
        )

    flops = 2 * m_sum * N * K
    nbytes = (m_sum * K + G * N * K + m_sum * N) * a.element_size()
    ret = {
        "gfx": get_gfx_runtime(),
        "G": G,
        "ΣM": m_sum,
        "M pad": m_pad,
        "N": N,
        "K": K,
    }
    for name, (out, fn, out_of) in candidates.items():
        _, us = run_perftest(fn)
        out.fill_(float("nan"))  # an unwritten valid row fails the comparison
        fn()
        err, _ = check_groups(name, out_of, refs)
        record(ret, name, us, flops, nbytes, err)
        if name == "opus":
            # the rows the kernel runs: each group's mixed-height tiles
            run_rows = sum(mixed_rows(m) for m in rows)
            ret["opus pad TFLOPS"] = 2 * run_rows * N * K / us / 1e6
            assert not torch.isnan(d[valid]).any()

    # the dense hipBLASLt GEMM of the same FLOPs (M = ΣM, the first group's B)
    w = b[0] if layout == "nt" else b[0].t()  # [N, K] (nn: an N-contiguous view)
    y = torch.empty(m_sum, N, dtype=dtypes.bf16)
    _, us = run_perftest(lambda: torch.mm(a_compact, w.t(), out=y))
    y.fill_(float("nan"))
    torch.mm(a_compact, w.t(), out=y)
    err = checkAllclose(
        run_torch(a_compact, w, "nt"),
        y.to(dtypes.fp32),
        msg="hipblaslt_dense",
        printLog=False,
    )
    assert err == 0, f"hipblaslt_dense: {err:.2%} of elements outside rtol/atol"
    record(ret, "hipblaslt_dense", us, flops, (m_sum * K + N * K + m_sum * N) * 2, err)
    return ret


@benchmark()
def test_k_grouped(config, gemm, dist):
    """wgrad: dW[g] (+)= dY_g^T X_g, dY [ΣM, N], X [ΣM, K], dW [G, N, K] (W of fwd)."""
    G, total, gemms = CONFIGS[config]
    N, K = gemms[gemm]
    rows = group_rows(G, total, dist)
    starts, m_pad, _ = padded_layout(rows)
    m_sum = sum(rows)
    # scaled so the sums are O(1) at the mean M_e
    scale = max(m_sum / G, 1) ** -0.25
    dy_compact = (torch.randn(m_sum, N) * scale).to(dtypes.bf16)
    x_compact = (torch.randn(m_sum, K) * scale).to(dtypes.bf16)
    dy = to_padded(dy_compact, rows, starts, m_pad)  # padding rows NaN: never read
    x = to_padded(x_compact, rows, starts, m_pad)
    ks = torch.tensor(rows, dtype=dtypes.i32)
    compact_rows = list(itertools.accumulate([0, *rows]))

    # the initial main_grad: distinct per element and group, computed (a copy would double the memory)
    pattern = torch.arange(N * K, dtype=dtypes.fp32).view(N, K)

    def init(g):
        return torch.sin(pattern * (1e-3 * (g + 1)))

    def init_all():
        for g in range(G):
            main_grad[g] = init(g)

    # fp32 dW sums K_g = M_e products in fp32: against an fp64 reference the largest error grows
    # ~1e-8 K_g at these magnitudes (one sequential fp32 sum; measured 1.5e-4 at K_g = 16384, and the
    # torch fp32 matmul is twice that, too coarse a reference), two accumulations double it.
    def refs(accumulate=False, f32=False):
        for g in range(G):
            c0, c1 = compact_rows[g], compact_rows[g + 1]
            ref = run_torch(dy_compact[c0:c1], x_compact[c0:c1], "tn", torch.float64)
            if accumulate:  # fp32 main_grad: init + 2 ref
                yield g, init(g) + 2 * ref, 1e-4 + 4e-8 * rows[g]
            elif f32:  # fp32 dW written once
                yield g, ref, 1e-4 + 2e-8 * rows[g]
            else:  # bf16 dW
                yield g, ref, 1e-2

    main_grad = torch.empty(G, N, K, dtype=dtypes.fp32)  # fp32 outputs share one buffer
    dw = torch.empty(G, N, K, dtype=dtypes.bf16)  # bf16 outputs
    # name: (setup before the checked run, launch, output, refs)
    # opus: main_grad += dY^T X (Megatron's fused gradient accumulation), checked over two runs
    # from a non-constant main_grad: init + 2 ref; empty groups must keep init.
    candidates = {
        "opus": (
            init_all,
            lambda: k_grouped_bf16_gemm_tn_contiguous(
                dy, x, main_grad, ks, c=main_grad
            ),
            main_grad,
            lambda: refs(True),
        ),
        # fp32 dW written, no accumulation (the SonicMoE semantics)
        "opus_f32": (
            lambda: main_grad.fill_(float("nan")),
            lambda: k_grouped_bf16_gemm_tn_contiguous(dy, x, main_grad, ks),
            main_grad,
            lambda: refs(f32=True),
        ),
        # bf16 dW written (empty groups: 0)
        "opus_bf16": (
            lambda: dw.fill_(float("nan")),
            lambda: k_grouped_bf16_gemm_tn_contiguous(dy, x, dw, ks),
            dw,
            refs,
        ),
        # same accumulate semantics as opus
        "triton_ptgmm": (
            init_all,
            lambda: ptgmm(
                dy_compact, x_compact, ks, None, dtypes.fp32, main_grad, True
            ),
            main_grad,
            lambda: refs(True),
        ),
        "triton_nptgmm": (
            init_all,
            lambda: nptgmm(
                dy_compact, x_compact, ks, None, dtypes.fp32, main_grad, True
            ),
            main_grad,
            lambda: refs(True),
        ),
    }

    # SonicMoE wgrad: fp32 dW written (no accumulation into a main_grad)
    if sonicmoe_available():
        candidates["sonicmoe"] = (
            lambda: main_grad.fill_(float("nan")),
            lambda: sonicmoe_grouped_gemm(
                dy_compact,
                x_compact,
                offsets(rows),
                out=main_grad,
                A_is_transposed=True,
                out_dtype=dtypes.fp32,
            ),
            main_grad,
            lambda: refs(f32=True),
        )
    if hipb_available():
        cu_seqlens = offsets(rows)
        candidates["hipb_grouped_mm"] = (  # bf16 dW written
            lambda: dw.fill_(float("nan")),
            lambda: hipb_grouped_mm(dy_compact, x_compact, cu_seqlens, dw, True),
            dw,
            refs,
        )

    flops = 2 * m_sum * N * K
    in_bytes = m_sum * (N + K) * 2
    ret = {
        "gfx": get_gfx_runtime(),
        "G": G,
        "ΣM": m_sum,
        "M pad": m_pad,
        "N": N,
        "K": K,
    }
    for name, (setup, fn, out, refs_of) in candidates.items():
        _, us = run_perftest(fn)
        accumulates = name in ("opus", "triton_ptgmm", "triton_nptgmm")
        setup()
        fn()
        if accumulates:  # the second accumulation
            fn()
        err, max_abs = check_groups(name, out.__getitem__, refs_of)
        # read-modify-write of fp32 main_grad, one fp32 write, or one bf16 write
        out_bytes = G * N * K * (8 if accumulates else out.element_size())
        record(ret, name, us, flops, in_bytes + out_bytes, err)
        ret[f"{name} max_abs"] = max_abs
        if name == "opus":
            # the rows the kernel runs: each group's rounded up to 64-row K tiles, at least 2
            k_rows = sum(64 * max(2, -(-m // 64)) for m in rows if m)
            ret["opus pad TFLOPS"] = 2 * k_rows * N * K / us / 1e6
    candidates = main_grad = None  # free the fp32 buffer for the dense run

    # the dense hipBLASLt TN GEMM of the same FLOPs (K = ΣM, down to a multiple of 64): bf16
    # C = dY^T X (a transposed view of the compact rows)
    kd = m_sum // 64 * 64
    if kd >= 128:
        dyt, xt = dy_compact[:kd].t(), x_compact[:kd]
        _, us = run_perftest(lambda: torch.mm(dyt, xt, out=dw[0]))
        dw[0].fill_(float("nan"))
        torch.mm(dyt, xt, out=dw[0])
        err = checkAllclose(
            run_torch(dy_compact[:kd], x_compact[:kd], "tn"),
            dw[0].to(dtypes.fp32),
            msg="hipblaslt_dense",
            printLog=False,
        )
        assert err == 0, f"hipblaslt_dense: {err:.2%} of elements outside rtol/atol"
        record(
            ret,
            "hipblaslt_dense",
            us,
            2 * kd * N * K,
            kd * (N + K) * 2 + N * K * 2,
            err,
        )
    return ret


@benchmark()
def test_grouped_linear(config, gemm, dist):
    """fwd + dgrad + wgrad chained as an autograd Function vs torch group by group (M3 parity)."""
    G, total, gemms = CONFIGS[config]
    N, K = gemms[gemm]
    rows = group_rows(G, total, dist)
    starts, m_pad, grouped_layout = padded_layout(rows)
    ks = torch.tensor(rows, dtype=dtypes.i32)
    x_compact = torch.randn(sum(rows), K, dtype=dtypes.bf16)
    w_ref = (torch.randn(G, N, K) * K**-0.5).to(dtypes.bf16)
    dy_compact = torch.randn(sum(rows), N, dtype=dtypes.bf16)

    # opus: padded rows, fp32 main_grad accumulated in place
    x = to_padded(x_compact, rows, starts, m_pad).requires_grad_()
    w = w_ref.clone().requires_grad_()
    main_grad = torch.zeros(G, N, K, dtype=dtypes.fp32)

    dy = to_padded(dy_compact, rows, starts, m_pad)

    def step():
        y = GroupedLinear.apply(x, w, grouped_layout, ks, main_grad)
        y.backward(dy)
        return y

    _, us = run_perftest(step, num_iters=11, num_warmup=1)
    main_grad.zero_()
    x.grad = None
    y = step()

    # torch, group by group: Y = X W^T and dX = dY W in fp32 (bf16 outputs), dW = dY^T X in fp64
    # (fp32 main_grad: tolerance as in test_k_grouped, inputs here unscaled, ~10x the error)
    compact_rows = list(itertools.accumulate([0, *rows]))
    ret = {"gfx": get_gfx_runtime(), "G": G, "ΣM": sum(rows), "N": N, "K": K}
    for name, out_of, refs in (
        (
            "Y",
            lambda g: y.detach()[starts[g] : starts[g] + rows[g]],
            lambda: (
                (g, run_torch(x_compact[c0:c1], w_ref[g], "nt"), 1e-2)
                for g, (c0, c1) in enumerate(itertools.pairwise(compact_rows))
            ),
        ),
        (
            "dX",
            lambda g: x.grad[starts[g] : starts[g] + rows[g]],
            lambda: (
                (g, run_torch(dy_compact[c0:c1], w_ref[g], "nn"), 1e-2)
                for g, (c0, c1) in enumerate(itertools.pairwise(compact_rows))
            ),
        ),
        (
            "dW",
            main_grad.__getitem__,
            lambda: (
                (
                    g,
                    run_torch(dy_compact[c0:c1], x_compact[c0:c1], "tn", torch.float64),
                    1e-3 + 2e-7 * (c1 - c0),
                )
                for g, (c0, c1) in enumerate(itertools.pairwise(compact_rows))
            ),
        ),
    ):
        ret[f"{name} err"], ret[f"{name} max_abs"] = check_groups(name, out_of, refs)
    ret["fwd+bwd us"] = us
    ret["TFLOPS"] = 3 * 2 * sum(rows) * N * K / us / 1e6
    return ret


class GroupedLinear(torch.autograd.Function):
    """A grouped linear layer on the padded contiguous layout, the three OPUS grouped GEMMs:
    Y = X W[g]^T, dX = dY W[g], main_grad[g] += dY_g^T X_g (fp32, Megatron's fused
    gradient accumulation; the weight's grad is then None)."""

    @staticmethod
    def forward(ctx, x, w, grouped_layout, ks, main_grad):
        y = torch.empty(x.shape[0], w.shape[1], dtype=x.dtype)
        m_grouped_bf16_gemm_nt_contiguous(x, w, y, grouped_layout)
        ctx.save_for_backward(x, w, grouped_layout, ks, main_grad)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, grouped_layout, ks, main_grad = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        # W [G, N, K] is the [G, K', N'] b of nn: no transpose
        m_grouped_bf16_gemm_nn_contiguous(dy, w, dx, grouped_layout)
        k_grouped_bf16_gemm_tn_contiguous(dy, x, main_grad, ks, c=main_grad)
        return dx, None, None, None, None


@benchmark()
def test_masked(num_groups, max_m, expected_m, n, k):
    ms = [
        min(max_m, m)
        for m in group_rows(num_groups, num_groups * expected_m, "deepgemm")
    ]
    a = torch.randn(num_groups, max_m, k, dtype=dtypes.bf16)
    b = torch.randn(num_groups, n, k, dtype=dtypes.bf16) * k**-0.5
    masked_m = torch.tensor(ms, dtype=dtypes.i32)
    valid = torch.arange(max_m)[None, :] < masked_m[:, None]
    ref = torch.cat([run_torch(a[g, :m], b[g], "nt") for g, m in enumerate(ms)])
    a_compact = a[valid]
    group_sizes = masked_m.clone()

    d = torch.full((num_groups, max_m, n), float("nan"), dtype=dtypes.bf16)
    d_compact = torch.full((sum(ms), n), float("nan"), dtype=dtypes.bf16)
    candidates = {
        "opus": (
            lambda: m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m),
            lambda: d[valid],
        ),
        # The compact rows of the same groups.
        "triton_gmm": (
            lambda: gmm(
                a_compact, b.transpose(1, 2), group_sizes, existing_out=d_compact
            ),
            lambda: d_compact,
        ),
    }

    flops = 2 * sum(ms) * n * k
    nbytes = (sum(ms) * k + num_groups * n * k + sum(ms) * n) * a.element_size()
    ret = {"gfx": get_gfx_runtime()}
    for name, (fn, result) in candidates.items():
        _, us = run_perftest(fn)
        err = checkAllclose(
            ref, result().to(dtypes.fp32), rtol=1e-2, atol=1e-2, msg=f"{name}: masked"
        )
        assert err == 0, f"{name}: {err:.2%} of elements outside rtol/atol"
        record(ret, name, us, flops, nbytes, err)
    return ret


def k_tolerance(mode, m):
    """Tolerance of one K-grouped group of m rows (see test_k_grouped)."""
    return {"bf16": 1e-2, "f32": 1e-4 + 2e-8 * m, "f32_acc": 1e-4 + 4e-8 * m}[mode]


def m_grouped_case(layout, G, N, K, rows):
    """Inputs of one contiguous M-grouped call: (launch(out = output), output, valid rows, group g
    of an output, refs)."""
    starts, m_pad, grouped_layout = padded_layout(rows)
    a_compact = torch.randn(sum(rows), K, dtype=dtypes.bf16)
    a = to_padded(a_compact, rows, starts, m_pad)
    b = torch.randn((G, N, K) if layout == "nt" else (G, K, N), dtype=dtypes.bf16)
    b *= K**-0.5
    d = torch.empty(m_pad, N, dtype=dtypes.bf16)
    opus = (
        m_grouped_bf16_gemm_nt_contiguous
        if layout == "nt"
        else m_grouped_bf16_gemm_nn_contiguous
    )
    compact_rows = list(itertools.accumulate([0, *rows]))

    def refs():
        for g, (c0, c1) in enumerate(itertools.pairwise(compact_rows)):
            yield g, run_torch(a_compact[c0:c1], b[g], layout), 1e-2

    return (
        lambda out=d: opus(a, b, out, grouped_layout),
        d,
        grouped_layout >= 0,
        lambda g, out=d: out[starts[g] : starts[g] + rows[g]],
        refs,
    )


def k_grouped_case(G, N, K, rows, mode, seg_len, check_layout=False):
    """Inputs of one K-grouped call: (setup, launch(out = output), output, refs). mode: bf16 / f32 /
    f32_acc (main_grad += over two launches from a non-constant main_grad)."""
    starts, m_pad, _ = padded_layout(rows)
    m_sum = sum(rows)
    scale = max(m_sum / G, 1) ** -0.25
    dy_compact = (torch.randn(m_sum, N) * scale).to(dtypes.bf16)
    x_compact = (torch.randn(m_sum, K) * scale).to(dtypes.bf16)
    dy = to_padded(dy_compact, rows, starts, m_pad)
    x = to_padded(x_compact, rows, starts, m_pad)
    ks = torch.tensor(rows, dtype=dtypes.i32)
    out = torch.empty(G, N, K, dtype=dtypes.bf16 if mode == "bf16" else dtypes.fp32)
    pattern = torch.arange(N * K, dtype=dtypes.fp32).view(N, K)
    compact_rows = list(itertools.accumulate([0, *rows]))

    def init(g):
        return torch.sin(pattern * (1e-3 * (g + 1)))

    def setup():
        if mode == "f32_acc":
            for g in range(G):
                out[g] = init(g)
        else:
            out.fill_(float("nan"))

    def refs():
        for g, (c0, c1) in enumerate(itertools.pairwise(compact_rows)):
            ref = run_torch(dy_compact[c0:c1], x_compact[c0:c1], "tn", torch.float64)
            if mode == "f32_acc":
                ref = init(g) + 2 * ref
            yield g, ref, k_tolerance(mode, rows[g])

    def launch(out=out):
        k_grouped_bf16_gemm_tn_contiguous(
            dy,
            x,
            out,
            ks,
            c=out if mode == "f32_acc" else None,
            seg_len=seg_len,
            check_layout=check_layout,
        )

    return setup, launch, out, refs


@benchmark()
def test_m_grouped_edge(layout, G, N, K, dist, mean):
    """Contiguous nt / nn off the tile grid: a ragged last N tile (nt: N not a multiple of 8),
    more than 64 groups (the group prefix sums cross waves)."""
    rows = group_rows(G, G * mean, dist)
    launch, d, valid, out_of, refs = m_grouped_case(layout, G, N, K, rows)
    d.fill_(float("nan"))  # an unwritten valid row fails the comparison
    launch()
    err, max_abs = check_groups(f"m_grouped {layout}", out_of, refs)
    assert not torch.isnan(d[valid]).any()
    return {"gfx": get_gfx_runtime(), "ΣM": sum(rows), "err": err, "max_abs": max_abs}


@benchmark()
def test_k_grouped_edge(G, N, K, dist, mean, mode, seg_len):
    """K-grouped off the tile grid, bf16 / fp32 / fp32 += output; seg_len: None the default plan,
    0 no split, > 0 groups longer than it split into segments that add in order."""
    rows = group_rows(G, G * mean, dist)
    setup, launch, out, refs = k_grouped_case(G, N, K, rows, mode, seg_len)
    setup()
    launch()
    if mode == "f32_acc":  # the second accumulation
        launch()
    err, max_abs = check_groups(f"k_grouped {mode}", out.__getitem__, refs)
    return {
        "gfx": get_gfx_runtime(),
        "ΣM": sum(rows),
        "max M_e": max(rows),
        "segments": sum(-(-m // seg_len) for m in rows) if seg_len else len(rows),
        "err": err,
        "max_abs": max_abs,
    }


REPEAT_BURST = (
    8  # launches of test_repeat queued back to back before any output is read
)


@benchmark()
def test_repeat(kind, G, N, K, dist, mean, seg_len, launches):
    """The same call `launches` times, REPEAT_BURST of them back to back into separate outputs
    (nothing waits between them: the plan kernel of one launch runs right behind the main kernel
    of the one before). Every output must be bit-equal to the first one, which is checked against
    torch. Both plan kernels run once per launch; a group given the wrong rows (or none) by a
    stale read shows as a mismatched launch."""
    rows = group_rows(G, G * mean, dist)
    if kind in ("m_nt", "m_nn"):
        launch, out, valid, out_of, refs = m_grouped_case(kind[2:], G, N, K, rows)
        result = lambda o: o[valid]
        name = f"m_grouped {kind[2:]}"
    else:
        # check_layout=False: the layout check reads grouped_layout, which waits for the device
        _, launch, out, refs = k_grouped_case(
            G, N, K, rows, kind[2:], seg_len, check_layout=False
        )
        out_of, result, name = out.__getitem__, lambda o: o, f"k_grouped {kind[2:]}"
    out.fill_(float("nan"))
    launch()
    err, _ = check_groups(name, out_of, refs)
    first = result(out).clone()
    outs = [out] + [torch.empty_like(out) for _ in range(REPEAT_BURST - 1)]
    mismatched = 0
    for done in range(0, launches, REPEAT_BURST):
        burst = outs[: launches - done]
        for o in burst:
            o.fill_(float("nan"))
        for o in burst:
            launch(o)
        mismatched += sum(not torch.equal(result(o), first) for o in burst)
    assert mismatched == 0, f"{name}: {mismatched} of {launches} launches differ"
    return {
        "gfx": get_gfx_runtime(),
        "ΣM": sum(rows),
        "err": err,
        "mismatched launches": mismatched,
    }


def test_k_grouped_layout_check():
    """The K-grouped layout check: groups start at multiples of 128 rows, the last one unpadded."""
    N, K = 128, 192
    align = get_mk_alignment_for_contiguous_layout()

    def call(rows, total, **kwargs):
        dy = torch.randn(total, N, dtype=dtypes.bf16)
        x = torch.randn(total, K, dtype=dtypes.bf16)
        dw = torch.full((len(rows), N, K), float("nan"), dtype=dtypes.fp32)
        ks = torch.tensor(rows, dtype=dtypes.i32)
        k_grouped_bf16_gemm_tn_contiguous(dy, x, dw, ks, **kwargs)
        return dy, x, dw

    # accepted: the last group unpadded (one group; two groups, the second at row 128)
    for rows, total in (([100], 100), ([100, 100], align + 100), ([0, 100], 100)):
        dy, x, dw = call(rows, total, check_layout=True)
        start = 0
        for g, m in enumerate(rows):
            ref = run_torch(dy[start : start + m], x[start : start + m], "tn")
            err = checkAllclose(
                ref, dw[g], rtol=1e-3, atol=1e-3, msg=f"layout {rows}: group {g} "
            )
            assert err == 0, f"layout {rows}: group {g}: {err:.2%} outside rtol/atol"
            start += -(-m // align) * align
    # rejected: packed groups (group 1 would be read from row 128, past the 200 rows)
    for rows, total in (([100, 100], 200), ([100, 100], align + 99), ([129], 128)):
        try:
            call(rows, total, check_layout=True)
        except ValueError as e:
            assert "multiple of 128" in str(e), e
        else:
            raise AssertionError(f"layout {rows} in {total} rows was not rejected")
    # rejected by the launcher: a group of 32767 full segments (nseg << 16 would overflow)
    try:
        call([32767 * align], 32767 * align, seg_len=align, check_layout=False)
    except RuntimeError as e:
        assert "grid too large" in str(e), e
    else:
        raise AssertionError("32767 segments in one group were not rejected")
    aiter.logger.info("k_grouped layout check: passed")


def summarize(name, rows):
    if rows:
        aiter.logger.info(
            "%s summary (markdown):\n%s",
            name,
            pd.DataFrame(rows).to_markdown(index=False),
        )


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "opus grouped GEMM unsupported on %s; skipping", get_gfx_runtime()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        nargs="*",
        choices=list(CONFIGS),
        # k3-S8192-ep8: G B / fp32 dW of fc1 pass 2^31 elements (64-bit group offsets)
        default=["glm-S8192", "k3-S8192-ep8"],
        help="per-rank training config (G, ΣM), see the module docstring",
    )
    parser.add_argument(
        "-g",
        "--gemm",
        type=str,
        nargs="*",
        choices=["fc1", "fc2"],
        default=["fc1", "fc2"],
    )
    parser.add_argument(
        "-p",
        "--pass",
        dest="passes",
        type=str,
        nargs="*",
        choices=["edge", "repeat", "fwd", "dgrad", "wgrad", "linear"],
        default=["edge", "repeat", "fwd", "dgrad", "wgrad", "linear"],
        help="linear: the three chained in an autograd Function vs torch\n"
        "edge: shapes off the tile grid, > 64 groups, split / no split, the layout check\n"
        "repeat: --launches back-to-back launches, every output bit-equal to the first",
    )
    parser.add_argument(
        "--launches",
        type=int,
        default=1000,
        help="launches of each -p repeat case",
    )
    parser.add_argument(
        "-t",
        "--dist",
        type=str,
        nargs="*",
        choices=["balanced", "deepgemm", "zipf", "one_group", "tiny"],
        default=["balanced", "deepgemm", "zipf", "one_group", "tiny"],
    )
    parser.add_argument(
        "-m",
        "--masked",
        type=dtypes.str2tuple,
        nargs="*",
        # DeepGEMM's masked sets (max_m 4096)
        default=[(32, 4096, 192), (6, 4096, 1024), (32, 4096, 20), (6, 4096, 20)],
        help="masked (num_groups, max_m, expected_m), e.g.: -m 32,4096,192",
    )
    parser.add_argument(
        "-s",
        "--nk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(6144, 7168), (4096, 2048)],
        help="masked (N, K) pairs, e.g.: -s 4096,7168",
    )
    args = parser.parse_args()

    def sweep(fn, *axes):
        rows = []
        for case in itertools.product(*axes):
            random.seed(0)
            torch.manual_seed(0)
            rows.append(fn(*case))
            torch.cuda.empty_cache()
        return rows

    def run_cases(fn, cases):
        rows = []
        for case in cases:
            random.seed(0)
            torch.manual_seed(0)
            rows.append(fn(*case))
        return rows

    if "edge" in args.passes:
        # (layout, G, N, K, dist, mean M_e): nn needs N % 8 == 0 (the row pitch of b)
        summarize(
            "grouped fwd / dgrad edge",
            run_cases(
                test_m_grouped_edge,
                [
                    ("nt", 3, 100, 128, "deepgemm", 150),
                    ("nt", 70, 257, 192, "tiny", 32),
                    ("nt", 130, 250, 128, "zipf", 200),
                    ("nt", 5, 385, 320, "one_group", 300),
                    ("nn", 70, 200, 192, "tiny", 32),
                    ("nn", 130, 328, 128, "zipf", 200),
                    ("nn", 1, 128, 128, "balanced", 1),
                ],
            ),
        )
        # (G, N, K, dist, mean M_e, mode, seg_len)
        summarize(
            "grouped wgrad edge",
            run_cases(
                test_k_grouped_edge,
                [
                    (G, N, K, dist, mean, mode, seg_len)
                    for G, N, K, dist, mean in (
                        (3, 200, 328, "deepgemm", 700),
                        (70, 136, 72, "tiny", 32),
                        (130, 264, 128, "zipf", 300),
                        (1100, 128, 128, "deepgemm", 40),
                    )
                    for mode, seg_len in (
                        ("bf16", None),
                        ("f32", None),
                        ("f32", 0),
                        ("f32", 128),
                        ("f32_acc", 0),
                        ("f32_acc", 256),
                    )
                ],
            ),
        )
        test_k_grouped_layout_check()
    if "repeat" in args.passes:
        # (kind, G, N, K, dist, mean M_e, seg_len); k3 EP8 has 112 groups per rank (fc1 6144 x 3584:
        # the M-grouped plan kernel without its lgkmcnt waits loses a group in ~2% of these launches);
        # 1100 groups: the K-grouped plan carries its prefix sums over a second chunk of 1024 groups
        summarize(
            "grouped repeated launches",
            run_cases(
                test_repeat,
                [
                    (*case, args.launches)
                    for case in (
                        ("m_nt", 112, 6144, 3584, "tiny", 32, None),
                        ("m_nn", 112, 3584, 6144, "tiny", 32, None),
                        ("k_bf16", 112, 1024, 1024, "tiny", 32, None),
                        ("k_f32", 112, 1024, 1024, "tiny", 32, 0),
                        ("k_f32", 112, 1024, 1024, "deepgemm", 300, 128),
                        ("k_f32", 1100, 256, 256, "deepgemm", 150, 128),
                        ("k_bf16", 1100, 256, 256, "tiny", 32, None),
                    )
                ],
            ),
        )

    m_passes = [p for p in args.passes if p in ("fwd", "dgrad")]
    summarize(
        "grouped fwd / dgrad",
        sweep(test_m_grouped, args.config, args.gemm, m_passes, args.dist),
    )
    if "wgrad" in args.passes:
        summarize(
            "grouped wgrad", sweep(test_k_grouped, args.config, args.gemm, args.dist)
        )
    if "linear" in args.passes:
        summarize(
            "grouped linear",
            sweep(test_grouped_linear, args.config, args.gemm, args.dist),
        )
    random.seed(0)
    torch.manual_seed(0)
    summarize(
        "grouped masked",
        [
            test_masked(g, max_m, em, n, k)
            for (g, max_m, em), (n, k) in itertools.product(args.masked, args.nk)
        ],
    )


if __name__ == "__main__":
    main()
