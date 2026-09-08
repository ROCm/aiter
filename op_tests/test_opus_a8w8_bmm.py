# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression + perf sweep for the opus fp8 e8m0 mxscale flatmm split-K BMM.

Covers the mmajor DeepSeek-V4 wo_a path: O/Y are [M, G, *] (transposed views of
batch-major [G, M, *]); wo_a + w_scale stay batch-major. Activation scale is
per-token e8m0 (GROUP_M=1), weight scale is 128x128-block e8m0. Candidates are
kid 0 (always-runnable baseline) and the public dispatch path; the reference is
a dequantized fp32 einsum. Per-kid perf comparison / winner selection lives in
``csrc/opus_gemm/opus_bmm_mxscale_tune.py``.

``--check-m-align`` runs a different check instead of the sweep: an every-kid
guard that OpusGemmInstance.m_align still matches launcher behaviour (see
``check_m_align``). It is kept out of the sweep because it deliberately provokes
launch failures and needs no timing.

Usage:
    python3 op_tests/test_opus_a8w8_bmm.py
    python3 op_tests/test_opus_a8w8_bmm.py -s 512,1024,4096 -g 2 -d bf16
    python3 op_tests/test_opus_a8w8_bmm.py --check-m-align
"""

import argparse
import itertools
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.batched_gemm_op_a8w8 import lookup_mxscale_bmm_config
from aiter.ops.opus.bmm_op import _opus_bmm_a8w8_mxscale_raw, bmm_a8w8_mxscale_opus
from aiter.ops.shuffle import (
    shuffle_scale_a,
    shuffle_scale_b,
    shuffle_scale_mxsk_mpack,
    shuffle_weight,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]  # fp8 e8m0 mxscale flatmm is gfx950-only
GROUP = 128  # GROUP_N == GROUP_K == 128; GROUP_M == 1 (per-token)
_DT = {"fp32": dtypes.fp32, "bf16": dtypes.bf16}


def _preshuffled_kids():
    """Kids whose B must be shuffle_weight(w, layout=(16, 16)).

    Imported lazily for the same reason as _align_kids: the helpers here are
    reused by scripts that must not need the codegen package on sys.path.
    """
    from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_bmm_kernel_lists

    return {
        int(kid)
        for fam in a8w8_mxscale_bmm_kernel_lists
        for kid, inst in fam.items()
        if inst.needs_preshuffled_b
    }


_SCALE_LAYOUT_KIDS = {}


def _scale_layout_kids():
    """kid -> (shuffle_scale ``sub``, mpacked-SFA args) for kids wanting either.

    Imported lazily for the same reason as _preshuffled_kids.
    """
    if not _SCALE_LAYOUT_KIDS:
        from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_bmm_kernel_lists

        _SCALE_LAYOUT_KIDS.update(
            {
                int(kid): (inst.needs_shuffle_scale, inst.needs_mpacked_sfa)
                for fam in a8w8_mxscale_bmm_kernel_lists
                for kid, inst in fam.items()
                if inst.needs_shuffle_scale or inst.needs_mpacked_sfa
            }
        )
    return _SCALE_LAYOUT_KIDS


def _scale_picker(xs_mx, ws_mx, n, k):
    """Return kid -> the (A, B) scale buffers that kid reads, relaying out once.

    Same hazard as the weight, one step worse: a shuffle_scale kid handed the
    plain arrays reads the wrong elements, and an mpacked-SFA kid reads past the
    end of them, which arrives as a GPU memory fault rather than as a bad number.
    Either layout is fixed once per model, so it follows the kid.
    """
    cache = {}

    def pick(kid):
        key = _scale_layout_kids().get(int(kid), (None, None))
        if key not in cache:
            sub, mpack = key
            g, _, sk = xs_mx.shape
            if sub:
                # stride(0) zeroed: the shuffle_scale layout folds the row into
                # its own addressing, so the kernel takes only the per-batch slab
                # from stride(1) and would otherwise add a bogus row offset.
                slab = shuffle_scale_a(xs_mx, k, sub)
                sfa = slab.as_strided(
                    (xs_mx.shape[1], g, slab.shape[1]), (0, slab.shape[1], 1)
                )
                # Kept 3-D with the N-block axis in the middle so stride(0) is the
                # per-batch slab, the only term the shuffle_scale path reads.
                sfb = shuffle_scale_b(ws_mx, n, k).view(g, n // 128, -1)
            elif mpack:
                sfa = (
                    shuffle_scale_mxsk_mpack(xs_mx, *mpack)
                    .view(g, -1, sk)
                    .transpose(0, 1)
                )
                sfb = ws_mx
            else:
                sfa, sfb = xs_mx.transpose(0, 1), ws_mx
            cache[key] = (sfa, sfb)
        return cache[key]

    return pick


def _weight_picker(W_mx):
    """Return kid -> the B buffer that kid reads, shuffling at most once.

    A preshuffled kid handed the row-major buffer does not raise, it just
    computes the wrong answer against the same reference, so the choice cannot
    be left to the caller.
    """
    preshuffled = _preshuffled_kids()
    cache = {}

    def pick(kid):
        if kid not in preshuffled:
            return W_mx
        if "sh" not in cache:
            cache["sh"] = shuffle_weight(W_mx, layout=(16, 16))
        return cache["sh"]

    return pick


def _to_e8m0_scale(scale):
    # Round scale up to a power of two so quantized fp8 values stay in range.
    e = torch.ceil(torch.log2(scale.to(dtypes.fp32))).to(torch.int32) + 127
    e = torch.clamp(e, 0, 255).to(torch.uint8)
    scale_pow2 = torch.exp2(e.to(dtypes.fp32) - 127.0)
    return e, scale_pow2


def _quant_per_token_e8m0(x_bf16):
    """[G,M,K] bf16 -> fp8 + e8m0 x_scale [G,M,K/128] + fp32 scale."""
    G, M, K = x_bf16.shape
    xb = x_bf16.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    raw = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
    e8m0, scale = _to_e8m0_scale(raw)
    q = (xb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return q.view(G, M, K), e8m0.squeeze(-1), scale.squeeze(-1)


def _quant_block_e8m0(w_bf16):
    """[G,N,K] bf16 -> fp8 + e8m0 w_scale [G,N/128,K/128] + fp32 scale."""
    G, N, K = w_bf16.shape
    wb = w_bf16.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    raw = wb.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-8) / 448.0
    e8m0, scale = _to_e8m0_scale(raw)
    q = (wb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return (
        q.view(G, N, K),
        e8m0.view(G, N // GROUP, K // GROUP),
        scale.view(G, N // GROUP, K // GROUP),
    )


def run_torch(O_fp8, W_fp8, x_scale, w_scale):
    """Reference: dequant fp8 -> fp32 einsum -> [G,M,N]. Not timed."""
    G, M, K = O_fp8.shape
    N = W_fp8.shape[1]
    act = O_fp8.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    act = (act * x_scale.unsqueeze(-1)).view(G, M, K)
    W = W_fp8.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    W = (W * w_scale.view(G, N // GROUP, 1, K // GROUP, 1)).view(G, N, K)
    return torch.einsum("gmk,gnk->gmn", act, W).to(dtypes.fp32)


def _block_varied(shape, k):
    """Signed random tensor whose per-128-K-block magnitude spans several powers
    of two, so the e8m0 128-block scales cover many exponents.

    ``rand()/10`` (non-negative, near-uniform) is what let the shipped kid312/313
    tileN COM_REP_N>1 kernels pass this test at ~0.007 rel while silently
    transposing output column groups: a pure column permutation over symmetric
    positive columns barely moves any element, and the collapsed single block
    scale hides scale-application bugs. Signed data makes swapped columns
    uncorrelated (~100% element mismatch), and the varied amplitude exercises
    real per-block scales -- together they turn this test into a real guard."""
    x = torch.randn(shape, dtype=dtypes.fp32)
    amp = torch.exp2(torch.randint(-4, 4, (k // GROUP,), device=x.device).float())
    x = x * amp.repeat_interleave(GROUP)
    return x.to(dtypes.bf16)


@benchmark()
def test_mxscale_bmm(g, m, n, k, dtype):
    ydt = _DT[dtype]
    # Canonical batch-major tensors, then feed the kernel transposed (mmajor)
    # views exactly like the DSV4 wo_a call does (zero-copy, no contiguous copy).
    O_bf16 = _block_varied((g, m, k), k)
    W_bf16 = _block_varied((g, n, k), k)
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(O_bf16)
    W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(W_bf16)

    O_in = O_mx.transpose(0, 1)  # [m,g,k] view
    xs_in = xs_mx.transpose(0, 1)  # [m,g,k/128] view
    ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32).transpose(0, 1)  # [m,g,n]
    y_shape = (m, g, n)

    weight_for = _weight_picker(W_mx)

    def _call(kid):
        Y = torch.empty(y_shape, dtype=ydt)
        _opus_bmm_a8w8_mxscale_raw(O_in, weight_for(kid), Y, xs_in, ws_mx, 1, kid)
        return Y

    # Correctness-focused: kid 0 (k32 fused) is a fixed baseline with no
    # tile-alignment requirement (always runnable), plus the public dispatch path
    # end to end. Per-kid perf comparison / winner selection lives in
    # csrc/opus_gemm/opus_bmm_mxscale_tune.py, not here.
    candidates = {"kid0_k32_fused": (lambda: _call(0), ref)}

    # Public backend-neutral entry: no kernelId -> per-(g,m,n,k) tuned-CSV
    # lookup + heuristic fallback + libtype backend routing. Exercises the
    # whole aiter.batched_gemm_a8w8_mxscale -> bmm_a8w8_mxscale_opus path end
    # to end (not the raw binding).
    candidates["auto (batched_gemm_a8w8_mxscale)"] = (
        lambda: aiter.batched_gemm_a8w8_mxscale(O_in, W_mx, xs_in, ws_mx, dtype=ydt),
        ref,
    )

    flops = 2.0 * g * m * n * k
    # fp8 A + fp8 W + e8m0 scales (uint8) + output.
    nbytes = (
        g * m * k
        + g * n * k
        + g * m * (k // GROUP)
        + g * (n // GROUP) * (k // GROUP)
        + m * g * n * torch.empty((), dtype=ydt).element_size()
    )

    ret = {"gfx": get_gfx()}
    for name, (fn, fn_ref) in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            fn_ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"mxscale_bmm {name} g={g} m={m} n={n} k={k}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_mxscale_bmm_batch_first(g, m, n, k, dtype):
    """Batch-leading (batch-major) round trip.

    The caller's natural DSV4 buffers are batch-major: batch is the *first*
    (outermost-in-memory) dimension -- activation/output are [G, M, *], weight
    is [G, N, K]. They are handed to the kernel as zero-copy [M, G, *]
    transposed views (dim0=M, dim1=batch), and the result is written straight
    back into a batch-major [G, M, N] buffer through its [M, G, N] view.

    This is the stride path the dropped ``_mmajor`` suffix used to over-claim:
    the batch axis sits at an arbitrary (here outermost) memory position while
    only K (inputs) and N (output) stay contiguous. Same tuned CSV / heuristic
    entries must serve it. Correctness is checked in the caller's native
    [G, M, N] order.
    """
    ydt = _DT[dtype]
    O_bf16 = _block_varied((g, m, k), k)
    W_bf16 = _block_varied((g, n, k), k)
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(O_bf16)
    W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(W_bf16)

    O_in = O_mx.transpose(0, 1)  # [m, g, k] view (K contiguous)
    xs_in = xs_mx.transpose(0, 1)  # [m, g, k/128] view
    ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32)  # [g, m, n] batch-major

    weight_for = _weight_picker(W_mx)

    def _call_raw(kid):
        # Batch-major output buffer; hand the kernel its [m, g, n] view so the
        # store lands at Y.stride(1) (batch) = m*n (outermost), N contiguous.
        Yb = torch.empty((g, m, n), dtype=ydt)
        _opus_bmm_a8w8_mxscale_raw(
            O_in, weight_for(kid), Yb.transpose(0, 1), xs_in, ws_mx, 1, kid
        )
        return Yb  # [g, m, n]

    def _call_auto():
        # Same tuned-CSV lookup the public entry does, but writing into a
        # caller-owned batch-major buffer -- which the guarded public entry no
        # longer exposes (it returns fresh token-major), so drive the opus
        # backend directly with the looked-up kid + the batch-major out= view.
        Yb = torch.empty((g, m, n), dtype=ydt)
        cfg = lookup_mxscale_bmm_config(g, m, n, k)
        bmm_a8w8_mxscale_opus(
            O_in,
            W_mx,
            xs_in,
            ws_mx,
            out=Yb.transpose(0, 1),
            dtype=ydt,
            kernelId=int(cfg["kernelId"]) if cfg is not None else None,
            splitK=int(cfg["splitK"]) if cfg is not None else None,
        )
        return Yb

    # Correctness-focused: kid 0 (always runnable) as the batch-major baseline,
    # plus the backend dispatch path writing into the batch-major buffer via
    # out=. Per-kid perf sweep lives in csrc/opus_gemm/opus_bmm_mxscale_tune.py.
    candidates = {"kid0_k32_fused": (lambda: _call_raw(0), ref)}
    candidates["auto (bmm_a8w8_mxscale_opus)"] = (_call_auto, ref)

    flops = 2.0 * g * m * n * k
    # fp8 A + fp8 W + e8m0 scales (uint8) + output.
    nbytes = (
        g * m * k
        + g * n * k
        + g * m * (k // GROUP)
        + g * (n // GROUP) * (k // GROUP)
        + m * g * n * torch.empty((), dtype=ydt).element_size()
    )

    ret = {"gfx": get_gfx()}
    for name, (fn, fn_ref) in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            fn_ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"mxscale_bmm_batch_first {name} g={g} m={m} n={n} k={k}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# --- tileN column-map regression guard ------------------------------------
# kid312/313 are COM_REP_N>1 kernels that previously transposed output column
# groups. Keep them out of the narrow perf table, but always exercise both
# output layouts with signed, varied-block-scale data so the bug cannot
# silently return.
#
# kid388/389/390 are here for the adjacent reason: they reach the T_M=1 / T_N=2
# grid because they asked for it (TILE_N_) rather than because B_M == 16 left
# them no choice, so they are the first to run that column map with an M axis
# spanning more than one tile. The shape is deliberately not tile-aligned (100
# rows over a 64-row tile) because an aligned one hides the M mask, which is the
# other thing a grid change can break. Plain-scale, so this checks the grid
# alone; the shuffled layout has its own guards.
_TILEN_REGRESSION_CASES = (
    # (G, M, N, K),        kids sharing that shape
    ((2, 16, 128, 1024), (312, 313)),
    ((2, 100, 128, 1024), (388, 389, 390)),
)
_TILEN_REGRESSION_ERR_TOL = 0.003


def check_tilen_column_map():
    """Check tileN-grid kids' column mapping for token- and batch-major output."""
    failures = []
    checked = 0

    for (g, m, n, k), kids in _TILEN_REGRESSION_CASES:
        O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(_block_varied((g, m, k), k))
        W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(_block_varied((g, n, k), k))
        O_in = O_mx.transpose(0, 1)
        xs_in = xs_mx.transpose(0, 1)
        ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32).transpose(0, 1)
        # Must go through the picker: kid388/389/390 are bpreshuffle kids, and a
        # preshuffled kid handed the row-major buffer does not raise, it just
        # returns a wrong answer. kid312/313 predate this and are row-major, so
        # the guard used to pass W_mx directly and was correct only by accident.
        weight_for = _weight_picker(W_mx)

        for kid in kids:
            for layout in ("token-major", "batch-major"):
                if layout == "token-major":
                    out = torch.full((m, g, n), float("nan"), dtype=dtypes.bf16)
                else:
                    out = torch.full(
                        (g, m, n), float("nan"), dtype=dtypes.bf16
                    ).transpose(0, 1)
                _opus_bmm_a8w8_mxscale_raw(
                    O_in, weight_for(kid), out, xs_in, ws_mx, 1, kid
                )
                torch.cuda.synchronize()
                delta = (out.to(dtypes.fp32) - ref).abs()
                rows = delta.flatten(1).mean(1) / (ref.abs().flatten(1).mean(1) + 1e-9)
                err = rows.max().item()
                checked += 1
                if not (err <= _TILEN_REGRESSION_ERR_TOL):
                    failures.append(
                        f"kid {kid} g{g}/m{m}/n{n}/k{k} {layout}: "
                        f"worst row rel err {err:.4f} > {_TILEN_REGRESSION_ERR_TOL}"
                    )

    expected = sum(len(kids) for _, kids in _TILEN_REGRESSION_CASES) * 2
    assert checked == expected, f"guard ran {checked} of {expected} cases"
    assert not failures, "tileN column-map regression:\n  " + "\n  ".join(failures)
    return checked


# --- shuffle_scale index guard ---------------------------------------------
# Re-derives, on the host, the closed forms the pipelines use to find a lane's
# A-scale dword and its op_sel byte, and checks them against the layout
# shuffle_scale_a actually writes.
#
# Worth having as a separate check because the on-GPU twin test cannot localise a
# failure here: a wrong slot or a wrong bit both come back as "the numbers differ"
# on whichever kid happens to be built, and the shipped kids only cover some of
# the grids the formula has to serve. This enumerates every
# (B_M, T_M, B_K, row, wave, lane, im, ik) instead, so an off-by-one shows up as
# the tile that produced it.
#
# The geometry is transcribed from opus_sf_shuf_geom in full and `sub` is read
# from the header rather than written down, because both of these guards have
# been silently vacuous before.
_SHUF_W_M = 16


def _shuf_sub():
    """The shipped layout's ``sub``, read from the traits header.

    Never a literal here -- see the note above. Imported lazily for the same
    reason as _align_kids: the helpers in this file are reused by scripts that
    must not need the codegen package on sys.path.
    """
    from csrc.opus_gemm.opus_gemm_common import _opus_sf_shuf_sub

    return _opus_sf_shuf_sub()


class _SfShufGeom:
    """``opus_sf_shuf_geom``, transcribed.

    The three regimes are what the guards exist to keep apart, and they are
    disjoint by construction (asserted in the header, re-asserted here):

    * ``PAIRED``   -- SF_MB <= SUB, a lane's consecutive M subtiles share a dword.
    * ``WIDE``     -- SF_MB == 2*SUB, they do not; the lane's own two rows land in
      different subtiles of the *same* dword pair, so either the wave dimension
      supplies the byte (``WAVE_PAIR``, compile-time) or a runtime shift does
      (``WIDE_SPLIT``).
    * ``SUBTILE_TILE`` -- B_M < 2*SUB, the tile is narrower than the pair, so which
      half it sits in is a uniform runtime bit.

    ``MP_RUNTIME`` is the union of the two runtime cases and they never coincide.
    """

    def __init__(self, b_m, t_m, com_rep_k, sub, wave_pair=False, w_m=_SHUF_W_M):
        self.SUB, self.MB, self.B_M, self.T_M, self.W_M = sub, t_m * w_m, b_m, t_m, w_m
        assert com_rep_k in (1, 2, 4), com_rep_k
        self.COM_REP_M, self.COM_REP_K = b_m // self.MB, com_rep_k
        self.PAIRED = self.MB <= sub and sub % self.MB == 0
        self.WIDE = self.MB == 2 * sub
        self.WAVE_PAIR = wave_pair and self.WIDE and self.COM_REP_M % 2 == 0
        self.WIDE_SPLIT = self.WIDE and not self.WAVE_PAIR
        self.OK = (self.PAIRED or self.WIDE) and b_m % self.MB == 0
        self.NL_SLOTS = (
            1 if (not self.OK or self.WIDE) else min(sub // self.MB, self.COM_REP_M)
        )
        self.N1_BLOCKS = -(-b_m // (2 * sub))
        self.N1_STEP = 2 if self.WAVE_PAIR else 1
        self.A_SLOTS = (
            self.N1_BLOCKS // 2 if self.WAVE_PAIR else self.N1_BLOCKS * self.NL_SLOTS
        )
        self.SUBTILE_TILE = b_m < 2 * sub
        # op_sel is two bits, so one dword's two K blocks are all it can address.
        self.KD = com_rep_k // 2 if com_rep_k >= 2 else 1
        self.A_SLOTS_K = self.A_SLOTS * self.KD
        self.MP_RUNTIME = self.SUBTILE_TILE or self.WIDE_SPLIT
        if self.OK:
            assert self.A_SLOTS >= 1
            assert not (self.SUBTILE_TILE and self.WIDE), "regimes must be disjoint"
            assert self.KD * 2 >= com_rep_k

    # --- the traits' compile-time selectors --------------------------------
    def slot_of(self, im):
        if self.WAVE_PAIR:
            return im >> 1
        n1o = (im * self.MB) // (2 * self.SUB)
        nlo = ((im * self.MB) % self.SUB) // self.MB
        return n1o * self.NL_SLOTS + nlo

    def mb_bit_of(self, im):
        return (im & 1) if self.WAVE_PAIR else (((im * self.MB) // self.SUB) & 1)

    def slot_of_k(self, im, ik):
        return self.slot_of(im) * self.KD + (ik >> 1)

    # --- the pipeline's address / fold / fragment row ----------------------
    def word0(self, row, wave, lane, k1_max):
        """shuf_a_word0 + shuf_r_word, and the runtime byte shift beside it."""
        r_lane = wave * self.W_M + lane % self.W_M
        if self.WAVE_PAIR:
            return (
                row // (2 * self.SUB) + wave
            ) * k1_max * self.SUB + lane % self.W_M, 0
        if self.WIDE_SPLIT:
            base = (row // (2 * self.SUB)) * k1_max * self.SUB + r_lane % self.SUB
            return base, ((r_lane // self.SUB) & 1) << 3
        base = (
            (row // (2 * self.SUB)) * k1_max * self.SUB
            + ((row % self.SUB) if self.SUBTILE_TILE else 0)
            + r_lane
        )
        shift = (((row // self.SUB) & 1) << 3) if self.SUBTILE_TILE else 0
        return base, shift

    def a_row(self, row, wave, lane, im):
        """make_layout_ra_mxsk: the row the A fragment for subtile im holds."""
        t_m, w_m = self.T_M, self.W_M
        if self.WAVE_PAIR:
            return (
                row
                + (im // t_m) * t_m * t_m * w_m
                + wave * t_m * w_m
                + (im % t_m) * w_m
                + lane % w_m
            )
        return row + im * t_m * w_m + wave * w_m + lane % w_m


def _shuf_ref(m, kb, sub, k1):
    """shuffle_scale_a's map -- the ground truth the producer emits.

    row = n1*(2*sub) + np*sub + nl
    byte index = (((n1*K1 + k1)*sub + nl)*2 + kp)*2 + np
    """
    n1, np, nl = m // (2 * sub), (m // sub) & 1, m % sub
    return (n1 * k1 + kb // 2) * sub + nl, ((kb & 1) << 1) | np


# Every (tile, B_K) the shuffled kids span, plus the T_M=1/T_M=2 pairs at the same
# B_M so both regimes are exercised at each width. wave_pair mirrors the traits:
# the wave8 header requests it, flatmm_splitk keeps the default false -- so both
# arms of the WIDE split are enumerated rather than only the shipped one.
_SHUF_TILES = tuple(
    (b_m, t_m, b_k, wp)
    for b_m in (16, 32, 64, 128, 256)
    for t_m in (1, 2)
    for b_k in (128, 256, 512)
    for wp in (False, True)
    if b_m % (t_m * _SHUF_W_M) == 0
)


def _shuf_walk(sub, ksc, on_read, mutate=None):
    """Drive the transcribed chain over every tile and every lane it has.

    ``on_read(geom, ctx, word, byte, m, kb)`` is called once per (im, ik) with the
    dword and byte the kernel would name; ``mutate`` falsifies one selector, which
    is how the negative controls are built. Returns (n_reads, n_tiles, slots_seen).
    """
    k1 = (ksc + 1) // 2
    reads = tiles = 0
    dense = []
    for b_m, t_m, b_k, wp in _SHUF_TILES:
        com_rep_k = b_k // 128
        g = _SfShufGeom(b_m, t_m, com_rep_k, sub, wave_pair=wp)
        if not g.OK or ksc % com_rep_k:
            continue
        tiles += 1
        seen = set()
        for row in range(0, 4 * b_m, b_m):
            for wave in range(t_m):
                for lane in range(g.W_M):
                    base, shift = g.word0(row, wave, lane, k1)
                    for kt in range(ksc // com_rep_k):
                        # COM_REP_K == 1 addresses half-words, so its low K bit
                        # rides in the byte rather than the dword index.
                        kk1 = kt * g.KD if com_rep_k >= 2 else (kt >> 1)
                        kp = 0 if com_rep_k >= 2 else (kt & 1)
                        for im in range(g.COM_REP_M):
                            for ik in range(com_rep_k):
                                slot = g.slot_of_k(im, ik)
                                seen.add(slot)
                                b = (
                                    (slot // g.KD) * g.N1_STEP
                                    if g.WAVE_PAIR
                                    else (slot // g.KD) // g.NL_SLOTS
                                )
                                nl = 0 if g.WAVE_PAIR else (slot // g.KD) % g.NL_SLOTS
                                word = (
                                    base + (b * k1 + kk1 + (ik >> 1)) * sub + nl * g.MB
                                )
                                byte = ((ik & 1) << 1) | g.mb_bit_of(im)
                                byte += shift // 8 + (2 * kp if com_rep_k == 1 else 0)
                                if mutate:
                                    word, byte = mutate(g, im, ik, word, byte)
                                m = g.a_row(row, wave, lane, im)
                                kb = kt * com_rep_k + ik
                                on_read(
                                    g,
                                    (b_m, t_m, b_k, wp, row, wave, lane, kt, im, ik),
                                    word,
                                    byte,
                                    m,
                                    kb,
                                )
                                reads += 1
        dense.append((b_m, t_m, b_k, wp, sorted(seen), g.A_SLOTS_K))
    return reads, tiles, dense


def check_shuffle_scale_index():
    """Enumerate the transcribed index chain against shuffle_scale_a's map.

    Pure algebra: no tensor, no GPU. Checks that the (dword, byte) the kernel
    would name for a lane's A fragment is the one the layout assigns to the row
    that fragment actually holds -- which is the property the on-GPU twin test can
    only observe as "the numbers differ".

    Also asserts the slot set is *dense* (= 0..A_SLOTS_K-1). A chain that is
    injective but sparse still reads correct bytes while holding registers it
    never names, so bit-exactness alone would not catch it.
    """
    sub = _shuf_sub()
    ksc = 16  # 8 dword-pairs of K: enough for COM_REP_K in {1,2,4} to all divide
    k1 = (ksc + 1) // 2
    failures = []

    def check(g, ctx, word, byte, m, kb):
        rw, rb = _shuf_ref(m, kb, sub, k1)
        if (word, byte) != (rw, rb):
            failures.append(f"{ctx}: named ({word},{byte}), layout says ({rw},{rb})")

    reads, tiles, dense = _shuf_walk(sub, ksc, check)
    for b_m, t_m, b_k, wp, seen, a_slots_k in dense:
        if seen != list(range(a_slots_k)):
            failures.append(
                f"B_M={b_m} T_M={t_m} B_K={b_k} wave_pair={wp}: slots {seen} "
                f"!= dense 0..{a_slots_k - 1}"
            )
    assert not failures, "shuffle_scale index guard:\n  " + "\n  ".join(
        sorted(set(failures))[:20]
    )
    assert reads and tiles, "index guard enumerated nothing"

    # --- negative controls: a gate that cannot fail proves nothing ---------
    # Each falsifies exactly one selector the chain depends on. They must fire at
    # the *shipped* sub, not at some sub where the regime is unreachable.
    controls = {
        # the M-side chain: register slot, and op_sel's low bit
        "slot shifted by one dword": lambda g, im, ik, w, b: (w + g.SUB, b),
        "M byte bit pinned to 0": lambda g, im, ik, w, b: (w, b & ~1),
        # the K-side chain (2b): op_sel's high bit, and the KD register index that
        # gives a B_K=512 tile its second dword
        "K byte bit pinned to 0": lambda g, im, ik, w, b: (w, b & ~2),
        "K dword index dropped": lambda g, im, ik, w, b: (w - (ik >> 1) * g.SUB, b),
        # the WIDE/SUBTILE runtime fold, which is what sub=16 added
        "runtime byte shift dropped": lambda g, im, ik, w, b: (
            (w, b - 1) if g.MP_RUNTIME else (w, b)
        ),
    }
    inert = []
    for name, mut in controls.items():
        fired = []
        _shuf_walk(
            sub,
            ksc,
            lambda g, ctx, word, byte, m, kb, fired=fired: (
                fired.append(1) if (word, byte) != _shuf_ref(m, kb, sub, k1) else None
            ),
            mutate=mut,
        )
        if not fired:
            inert.append(name)
    assert not inert, (
        f"index guard negative controls did not fire: {inert} -- the guard cannot "
        f"detect a fault in that selector, so its green result says nothing about it"
    )
    return reads


def check_shuffle_scale_bytes():
    """Check the enumerated byte address against shuffle_scale_a's actual output.

    The index guard proves the addressing is self-consistent with the documented
    map; this one proves the map is what the function writes, which is the half no
    amount of algebra can settle. Covers M=100 so the row pad and the K pad are
    both live, and both parities of the SUBTILE/WIDE runtime bit.
    """
    sub = _shuf_sub()
    pad = 0x7F  # E8M0 == 1.0; every lane reads its scale byte unmasked
    failures = []
    checked = 0

    # Two independent fillings: a permutation error that lands on an equal value
    # under one linear hash will not survive the other.
    for mult in (37, 89):
        for g_n, rows, k in ((1, 128, 512), (2, 100, 384)):
            ksc = k // 128
            k1 = (ksc + 1) // 2
            mpad = -(-rows // (2 * sub)) * (2 * sub)
            idx = torch.arange(g_n * rows * ksc, dtype=torch.int32)
            plain = idx.mul(mult).remainder(126).add(1).to(torch.uint8)
            plain = plain.view(g_n, rows, ksc)
            words = shuffle_scale_a(plain, k, sub).view(g_n, -1).view(torch.int32)
            if words.shape[1] * 4 != mpad * 2 * k1:
                failures.append(
                    f"g={g_n} M={rows} K={k}: slab {words.shape[1] * 4} B != "
                    f"Mpad*2*K1 = {mpad * 2 * k1} B"
                )
                continue

            for gi in range(g_n):

                def check(
                    g,
                    ctx,
                    word,
                    byte,
                    m,
                    kb,
                    gi=gi,
                    plain=plain,
                    words=words,
                    rows=rows,
                    ksc=ksc,
                    k=k,
                    mpad=mpad,
                    k1=k1,
                ):
                    nonlocal checked
                    if m >= mpad or kb >= 2 * k1 or word >= words.shape[1]:
                        return
                    got = (int(words[gi, word]) >> (8 * byte)) & 0xFF
                    # Past M or past Ksc is pad. This pins shuffle_scale_a's own
                    # contract, which fills 0x7F; it is NOT a requirement on a
                    # producer -- the consumer masks the row pad, which is why
                    # the quant kernel may allocate with torch.empty.
                    want = int(plain[gi, m, kb]) if (m < rows and kb < ksc) else pad
                    checked += 1
                    if got != want:
                        failures.append(
                            f"g{gi} M={rows} {ctx}: word[{word}] byte {byte} = "
                            f"{got}, want {want}"
                        )

                _shuf_walk(sub, ksc, check)

    assert not failures, "shuffle_scale byte guard:\n  " + "\n  ".join(
        sorted(set(failures))[:20]
    )
    assert checked, "byte guard enumerated nothing"
    return checked


# --- m_align guard ---------------------------------------------------------
# Straddles every tile boundary in the family (B_M is 16/32/64/128/256) and every
# declared m_align (1 / B_M / 2*B_M), with aligned and unaligned M on both sides.
_ALIGN_MS = [1, 17, 48, 64, 96, 127, 128, 129, 200, 255, 256, 512]
_ALIGN_G = 2
# (N, K) candidates: the second entry serves the k1024-only pipeline kids.
_ALIGN_SHAPES = [(1024, 4096), (1024, 1024)]
_ALIGN_ERR_TOL = 0.003  # e8m0 quant floor is ~0.0014; same gate the tuner uses


def _align_kids():
    # Imported here so the sweep and the many scripts reusing the helpers above
    # never need the codegen package on sys.path.
    from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_bmm_kernel_lists

    return {
        int(kid): inst
        for fam in a8w8_mxscale_bmm_kernel_lists
        for kid, inst in fam.items()
    }


_ALIGN_INPUTS = {}


def _align_inputs(m, n, k):
    """Quantized inputs + fp32 reference for one shape, shared across kids."""
    key = (m, n, k)
    if key not in _ALIGN_INPUTS:
        g = _ALIGN_G
        O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(_block_varied((g, m, k), k))
        W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(_block_varied((g, n, k), k))
        ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32).transpose(0, 1)
        _ALIGN_INPUTS[key] = (
            O_mx.transpose(0, 1),
            W_mx,
            shuffle_weight(W_mx, layout=(16, 16)),
            # The pickers are cached with the buffers they relay out, so each
            # layout is built once per shape rather than once per kid and M.
            _scale_picker(xs_mx, ws_mx, n, k),
            ref,
        )
    return _ALIGN_INPUTS[key]


def _align_run(kid, inst, m, n, k):
    """Return (ok, rel_err). ok False means the launcher refused the shape."""
    O_in, W_mx, W_sh, scale_for, ref = _align_inputs(m, n, k)
    # A preshuffled kid reading the row-major buffer does not fail, it just
    # returns a wrong answer, so the weight has to follow the kid -- and so must
    # the scales.
    W_in = W_sh if inst.needs_preshuffled_b else W_mx
    xs_in, ws_mx = scale_for(kid)
    # NaN-filled so a row the kernel never writes shows up as nan, not as a
    # plausible value that a mean error would dilute.
    Y = torch.full((m, _ALIGN_G, n), float("nan"), dtype=dtypes.bf16)
    try:
        _opus_bmm_a8w8_mxscale_raw(O_in, W_in, Y, xs_in, ws_mx, 1, kid)
        torch.cuda.synchronize()
    except RuntimeError:
        # The launcher's AITER_CHECK on M surfaces here. Deliberately not a
        # blanket except: a harness bug must fail loudly, not read as a refusal.
        return False, 0.0
    d = (Y.to(dtypes.fp32) - ref).abs()
    # Per-row, not global: one wrong row out of a long M barely moves the mean.
    rows = d.flatten(1).mean(1) / (ref.abs().flatten(1).mean(1) + 1e-9)
    return True, rows.max().item()


def _align_pick_shape(kid, inst):
    """First (N, K) this kid accepts at an aligned M, or None if it accepts none."""
    for n, k in _ALIGN_SHAPES:
        if n % inst.B_N or k % inst.B_K:
            continue
        if _align_run(kid, inst, max(inst.m_align, inst.B_M), n, k)[0]:
            return n, k
    return None


def check_m_align():
    """Assert OpusGemmInstance.m_align matches what each mxscale BMM kid does.

    m_align says which M values a kid's launcher accepts (1 == it masks a partial
    M tile). Both the runtime's padded-M lookup (aiter/ops/opus/bmm_op.py) and a
    tuner's candidate filter act on it, so a wrong value is not merely cosmetic:
    too strict hides the fastest kernel from tuning (kid326 lost ~9% at the DSV4
    wo_a decode shapes that way, while the runtime dispatched it at those very
    M), too loose makes both propose a kid whose launcher throws.

    For every kid this checks the declaration against observed behaviour: at an M
    the declaration accepts, the launch must succeed and match the dequantized
    fp32 reference; at an M it rejects, the launch must raise. Kids are never
    silently skipped -- an unrunnable kid is reported.

    Each kid gets the B buffer and the scale layout its family reads, so the
    reference check is a real correctness gate for the preshuffled and
    rearranged-scale kids too, and not an m_align check riding on a wrong answer.
    """
    kids = _align_kids()
    failures, unrunnable = [], []

    for kid, inst in sorted(kids.items()):
        shape = _align_pick_shape(kid, inst)
        if shape is None:
            unrunnable.append(kid)
            continue
        n, k = shape
        align = inst.m_align
        for m in _ALIGN_MS:
            ok, err = _align_run(kid, inst, m, n, k)
            if m % align == 0:
                if not ok:
                    failures.append(f"kid {kid}: m_align={align} but M={m} rejected")
                elif not (err <= _ALIGN_ERR_TOL):
                    failures.append(
                        f"kid {kid}: M={m} accepted but worst row rel err "
                        f"{err:.4f} > {_ALIGN_ERR_TOL}"
                    )
            elif ok:
                failures.append(
                    f"kid {kid}: m_align={align} claims M={m} unusable, "
                    f"but it ran (worst row rel err {err:.4f}) -- m_align too strict"
                )

    assert not unrunnable, (
        f"kids that ran on no test shape: {unrunnable}; extend _ALIGN_SHAPES so "
        f"the guard keeps covering them"
    )
    assert not failures, "m_align disagrees with the launcher:\n  " + "\n  ".join(
        failures
    )
    return len(kids)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "opus mxscale flatmm BMM unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="opus fp8 e8m0 mxscale flatmm split-K BMM test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        nargs="*",
        default=["bf16"],
        choices=["bf16", "fp32"],
        help="output dtype(s) to sweep (default: bf16)",
    )
    parser.add_argument(
        "-g",
        "--groups",
        type=int,
        nargs="*",
        default=[2, 8],
        help="batch group counts to sweep (DSV4 wo_a G; default: 2)",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (1, 1024, 4096),
            (16, 1024, 4096),
            (128, 1024, 4096),
            (256, 1024, 4096),
            (512, 1024, 4096),
            (8192, 1024, 4096),
            (16384, 1024, 4096),
        ],
        help="(m,n,k) shapes to sweep",
    )
    parser.add_argument(
        "--check-m-align",
        action="store_true",
        help="run the every-kid m_align guard instead of the perf sweep",
    )
    args = parser.parse_args()

    n_tilen_checks = check_tilen_column_map()
    aiter.logger.info(
        "tileN column mapping passed for %d kid/layout combinations", n_tilen_checks
    )

    # Both are pure host arithmetic (~0.4 s) and they catch the one class of bug
    # a single tile shape hides: an off-by-one in the shuffled scale index that
    # only shows up at a T_M or B_M this run does not happen to dispatch.
    n_sf_idx = check_shuffle_scale_index()
    n_sf_byte = check_shuffle_scale_bytes()
    aiter.logger.info(
        "shuffle_scale index consistent over %d (tile,row,lane,im) combinations, "
        "and names the right byte for %d scales",
        n_sf_idx,
        n_sf_byte,
    )

    if args.check_m_align:
        try:
            n_kids = check_m_align()
        except AssertionError as exc:
            aiter.logger.error("m_align guard FAILED: %s", exc)
            sys.exit(1)
        aiter.logger.info(
            "m_align matches launcher behaviour for all %d mxscale BMM kids", n_kids
        )
        return

    for dtype in args.dtype:
        df = []
        df_bf = []
        for g, (m, n, k) in itertools.product(args.groups, args.mnk):
            df.append(test_mxscale_bmm(g, m, n, k, dtype))
            df_bf.append(test_mxscale_bmm_batch_first(g, m, n, k, dtype))
        aiter.logger.info(
            "opus mxscale flatmm BMM summary (dtype=%s):\n%s",
            dtype,
            pd.DataFrame(df).to_markdown(index=False),
        )
        aiter.logger.info(
            "opus mxscale flatmm BMM batch-first (batch-major) summary "
            "(dtype=%s):\n%s",
            dtype,
            pd.DataFrame(df_bf).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
