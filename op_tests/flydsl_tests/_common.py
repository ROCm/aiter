# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared helpers for the fp8 unified-attention test files.

Underscore-prefixed so pytest does not collect this as a test module.

Hoists only the pieces that are byte-identical (module quirks aside) across
the fp8-attention test files in this directory and in
op_tests/test_unified_attention_flydsl.py. Page-pool builders and the various
reference attention implementations are deliberately NOT here -- they differ
across files (varlen vs paged vs ragged layouts, causal vs decode-only,
sinks/ragged/mixed variants) and unifying them risks silently changing which
inputs a test actually exercises.

`per_tensor_quant` in aiter.ops.quant is NOT a drop-in replacement for `q8`:
it omits the `clamp(min=1e-4)` scale floor the hand-rolled copies apply, so it
produces a materially different scale for near-zero tensors (verified
2026-08-11: 3.9e-9 vs 2.2e-7 on a 1e-6-scale random tensor). `q8` below keeps
the hand-rolled implementation for that reason.

Clear ~/.flydsl/cache before trusting a run after editing a kernel helper
class: the JIT cache key walks the launcher's closure for function
dependencies and does not resolve methods reached through instance
attributes, so an edit to a helper method hits a stale binary under an
unchanged key. Editing the kernel body directly does invalidate the key.
"""

from __future__ import annotations

import torch

DEV = "cuda"


def q8(x):
    """Quantize to fp8 e4m3 with a per-tensor scale, returning (fp8, descale)."""
    s = x.abs().amax().clamp(min=1e-4) / 448.0
    return (x / s).to(torch.float8_e4m3fn), s.reshape(1).float().to(DEV)


def build_fp8_gfx950():
    """The vendored dense/paged/varlen fp8 attention module builder."""
    from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx950 import (
        build_flash_attn_dualwave_swp_fp8_module as build,
    )

    return build


def build_fp8_gfx950_with_ws_elems():
    """Same builder plus the split-K workspace sizing helper."""
    from aiter.ops.flydsl.kernels.flash_attn_dualwave_common import (
        dualwave_splitk_workspace_elems as ws_elems,
    )
    from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx950 import (
        build_flash_attn_dualwave_swp_fp8_module as build,
    )

    return build, ws_elems


def dense_fp8_reference(qf, qs, kf, ks, vf, vs, causal, d):
    """Dense (non-paged, non-varlen) fp32 reference from dequantized fp8 QKV."""
    q, k, v = qf.float() * qs, kf.float() * ks, vf.float() * vs
    _b, s, h, _ = q.shape
    rep = h // k.shape[2]
    kt = k.repeat_interleave(rep, 2).transpose(1, 2)
    vt = v.repeat_interleave(rep, 2).transpose(1, 2)
    att = q.transpose(1, 2) @ kt.transpose(-1, -2) / (d**0.5)
    if causal:
        m = torch.triu(torch.ones(s, s, device=DEV, dtype=torch.bool), 1)
        att = att.masked_fill(m, float("-inf"))
    out = (att.softmax(-1) @ vt).transpose(1, 2)
    del att
    return out


def assert_attn_close(err, cos, bad=0, max_err=1e-1, min_cos=0.99):
    """Standing numeric gate: bad-row count, relative/max error, cosine.

    Takes already-computed scalars so every call site keeps its own metric
    computation (some reduce bad-rows over different dims, some don't compute
    bad-rows at all -- `bad` defaults to 0, which is a no-op check for those).
    Tolerances are parameters so each call site keeps its exact threshold.
    """
    assert bad == 0, f"{bad} rows over threshold (max err {err:.4g})"
    assert err < max_err, f"max err {err:.4g}"
    assert cos > min_cos, f"cosine {cos:.6f}"
