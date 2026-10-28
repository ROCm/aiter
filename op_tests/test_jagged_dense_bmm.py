# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import itertools
import os
import tempfile
from unittest import mock

import flydsl.compiler as flyc
import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx, get_num_xcds
from aiter.ops.flydsl import jagged_dense_bmm_dispatch as jdbba_dispatch
from aiter.ops.flydsl.jagged_dense_bmm_dispatch import (
    SUPPORTED_GFX,
    _load_dispatch_table,
    analytic_config,
    jagged_dense_bmm_dispatched,
    resolve_config,
    tile_bucket,
)
from aiter.ops.flydsl.kernels.jagged_dense_bmm_gen import (
    _LAUNCHER_CACHE,
    jagged_dense_bmm,
    jdbba_arch_traits,
    smem_bytes,
)
from aiter.ops.flydsl.kernels.jagged_dense_bmm_gen import (
    BLOCK_M as _BLOCK_M,
)
from aiter.test_common import (
    assertAllclose,
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

SEED = 1234
# Host selection sweep. These names are mocked; the GPU sweep below stays
# on the live card and on SUPPORTED_GFX.
_ARCH_SECTION_CASES = ("gfx942", "gfx950", "gfx1250", "unknown")
_ARCH_SHAPE = (120, 256, 256, 7680)
# cu_num stamped on the measuring part: full MI300X, and this MI350.
# Both parts expose 8 XCDs. A 4-XCD or 1-XCD partition has no row.
_MEASURED_CU = {"gfx942": 304, "gfx950": 256}
_MEASURED_XCDS = 8
_TRAITS = {
    "gfx942": {
        "mfma_k": 16,
        "lds_bytes": 64 * 1024,
        "k_perm_shape": (4, 4, 2),
        "k_slice_nested": True,
    },
    "gfx950": {
        "mfma_k": 32,
        "lds_bytes": 160 * 1024,
        "k_perm_shape": (8, 4),
        "k_slice_nested": False,
    },
}

try:
    from generative_recommenders.ops.triton.triton_jagged import (
        triton_jagged_dense_bmm_add_fwd,
    )

    _HAS_TRITON = True
except ModuleNotFoundError as _exc:  # pragma: no cover
    _HAS_TRITON = False
    _TRITON_ERR = _exc


def _make_seq_offsets(B, Mi, regime, seed=SEED):
    if regime == "uniform":
        return torch.arange(0, (B + 1) * Mi, Mi, dtype=torch.int32, device="cuda")
    g = torch.Generator().manual_seed(seed)
    t = (Mi * (torch.rand(B, generator=g, device="cpu") ** 4)).floor().to(torch.int64)
    t[: max(1, B // 5)] = 0
    t[-1] = Mi
    if B > 1:
        t[-2] = int(0.9 * Mi)
    off = torch.zeros(B + 1, dtype=torch.int32, device="cpu")
    off[1:] = torch.cumsum(t, 0).to(torch.int32)
    return off.cuda()


def run_torch(jagged, dense, bias, seq_offsets, N, dtype=dtypes.bf16):
    L = jagged.shape[0]
    out = torch.zeros((L, N), dtype=dtype, device="cuda")
    so = seq_offsets.cpu().tolist()
    for b in range(dense.shape[0]):
        s, e = so[b], so[b + 1]
        if e > s:
            out[s:e] = (
                jagged[s:e].to(dtypes.fp32) @ dense[b].to(dtypes.fp32)
                + bias[b].to(dtypes.fp32)[None, :]
            ).to(dtype)
    return out


def _build_inputs(B, D, Kout, Mi, regime, dtype, seq_offsets=None, pad_rows=True):
    N, K = Kout, D
    if seq_offsets is None:
        seq_offsets = _make_seq_offsets(B, Mi, regime)
    L = int(seq_offsets[-1].item())
    torch.manual_seed(0)
    jagged = torch.randn(max(L, 1), K, dtype=dtype, device="cuda")
    dense = torch.randn(B, K, N, dtype=dtype, device="cuda")
    bias = torch.randn(B, N, dtype=dtype, device="cuda")

    dense_tall = dense.transpose(1, 2).reshape(B * N, K).contiguous()
    bias_flat = bias.reshape(B * N).contiguous()
    # The kernel writes exactly the packed rows. Padding is only for callers
    # that still pass a tail; the reuse test uses the exact (L, N) buffer.
    extra = _BLOCK_M if pad_rows else 0
    out = torch.zeros(L + extra, N, dtype=dtype, device="cuda")
    tA = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
    return {
        "jagged": jagged,
        "dense": dense,
        "bias": bias,
        "seq_offsets": seq_offsets,
        "dense_tall": dense_tall,
        "bias_flat": bias_flat,
        "out": out,
        "tA": tA,
        "tC": tC,
        "L": L,
        "N": N,
        "K": K,
    }


@benchmark()
def test_jdbba(B, D, Kout, Mi, regime, dtype):
    d = _build_inputs(B, D, Kout, Mi, regime, dtype)
    L, N, K = d["L"], d["N"], d["K"]
    uniform = regime == "uniform"
    st = torch.cuda.current_stream()

    ref = run_torch(d["jagged"], d["dense"], d["bias"], d["seq_offsets"], N, dtype)

    # Skew rebuilds TILE_MAP inside the timed call, so the row includes prep.
    def _flydsl():
        jagged_dense_bmm_dispatched(
            d["tC"],
            d["tA"],
            d["dense_tall"],
            d["bias_flat"],
            d["seq_offsets"],
            B,
            Mi,
            stream=st,
            uniform_seqlen=uniform,
        )
        return d["out"][:L]

    candidates = {"flydsl": _flydsl}
    if _HAS_TRITON:
        so64 = d["seq_offsets"].to(torch.int64)
        candidates["triton"] = lambda: triton_jagged_dense_bmm_add_fwd(
            Mi, so64, d["jagged"], d["dense"], d["bias"]
        )[0][:L]

    flops = 2 * L * K * N
    nbytes = (L * K + B * K * N + B * N + L * N) * d["jagged"].element_size()

    ret = {"gfx": get_gfx(), "L": L}
    for name, fn in candidates.items():
        got, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            got.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: jagged_dense_bmm ({regime})",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_jdbba_block_k(B, D, Kout, Mi, block_k, dtype):
    d = _build_inputs(B, D, Kout, Mi, "uniform", dtype)
    L, N, K = d["L"], d["N"], d["K"]
    st = torch.cuda.current_stream()
    ref = run_torch(d["jagged"], d["dense"], d["bias"], d["seq_offsets"], N, dtype)

    def _flydsl():
        jagged_dense_bmm(
            d["tC"],
            d["tA"],
            d["dense_tall"],
            d["bias_flat"],
            d["seq_offsets"],
            n_groups=B,
            max_seq_len=Mi,
            stream=st,
            uniform_seqlen=True,
            block_k=block_k,
        )
        return d["out"][:L]

    got, us = run_perftest(_flydsl)
    err = checkAllclose(
        ref.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg=f"block_k={block_k} uniform",
    )
    flops = 2 * L * K * N
    nbytes = (L * K + B * K * N + B * N + L * N) * d["jagged"].element_size()
    return {
        "gfx": get_gfx(),
        "L": L,
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
    }


@benchmark()
def test_jdbba_skew_varying_L(B, D, Kout, Mi, seed, dtype):
    d = _build_inputs(
        B,
        D,
        Kout,
        Mi,
        "skew",
        dtype,
        seq_offsets=_make_seq_offsets(B, Mi, "skew", seed=seed),
    )
    L, N, K = d["L"], d["N"], d["K"]
    st = torch.cuda.current_stream()
    ref = run_torch(d["jagged"], d["dense"], d["bias"], d["seq_offsets"], N, dtype)

    def _flydsl():
        jagged_dense_bmm_dispatched(
            d["tC"],
            d["tA"],
            d["dense_tall"],
            d["bias_flat"],
            d["seq_offsets"],
            B,
            Mi,
            stream=st,
            uniform_seqlen=False,
        )
        return d["out"][:L]

    got, us = run_perftest(_flydsl)
    err = checkAllclose(
        ref.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg=f"skew varying L seed={seed}",
    )
    flops = 2 * L * K * N
    nbytes = (L * K + B * K * N + B * N + L * N) * d["jagged"].element_size()
    return {
        "gfx": get_gfx(),
        "L": L,
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
    }


@benchmark()
def test_jdbba_dispatch(B, D, Kout, Mi):
    winners = _load_dispatch_table().get("winners") or {}
    key = (D, Kout, tile_bucket(B, Mi, Kout))
    cfg = resolve_config(n_groups=B, reduction_k=D, output_n=Kout, max_seq_len=Mi)
    in_table = key in winners
    if in_table:
        w = winners[key]
        routing_ok = (
            cfg["xcd_c"] == w["xcd_c"]
            and cfg["xcd_w"] == w["xcd_w"]
            and cfg["threads"] == w["threads"]
            and cfg["block_k"] == w["block_k"]
        )
    else:
        routing_ok = True

    d = _build_inputs(B, D, Kout, Mi, "uniform", dtypes.bf16)
    L, N, K = d["L"], d["N"], d["K"]
    st = torch.cuda.current_stream()
    ref = run_torch(d["jagged"], d["dense"], d["bias"], d["seq_offsets"], N)

    def _flydsl():
        jagged_dense_bmm_dispatched(
            d["tC"],
            d["tA"],
            d["dense_tall"],
            d["bias_flat"],
            d["seq_offsets"],
            B,
            Mi,
            stream=st,
            uniform_seqlen=True,
        )
        return d["out"][:L]

    got, us = run_perftest(_flydsl)
    err = checkAllclose(
        ref.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg=f"dispatch B{B} D{D} K{Kout} Mi{Mi}",
    )
    flops = 2 * L * K * N
    nbytes = (L * K + B * K * N + B * N + L * N) * d["jagged"].element_size()
    return {
        "gfx": get_gfx(),
        "in_table": in_table,
        "xcd_c": cfg["xcd_c"],
        "xcd_w": cfg["xcd_w"],
        "routing_ok": routing_ok,
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
    }


def _resegment_same_L(seq_offsets):
    """Rewrite group bounds in place without changing the total length."""
    ends = seq_offsets.detach().to("cpu")
    lengths = ends[1:] - ends[:-1]
    rev = torch.flip(lengths, dims=[0])
    if torch.equal(rev, lengths):
        rev = torch.roll(lengths, 1)
    if torch.equal(rev, lengths):
        raise AssertionError("could not build a distinct same-L segmentation")
    out = torch.zeros_like(ends)
    out[1:] = torch.cumsum(rev, 0)
    if int(out[-1]) != int(ends[-1]):
        raise AssertionError("same-L rewrite changed the total length")
    seq_offsets.copy_(out.to(device=seq_offsets.device, dtype=seq_offsets.dtype))


def _shorten_offsets(seq_offsets):
    """Rewrite group bounds in place so the packed length shrinks."""
    ends = seq_offsets.detach().to("cpu")
    lengths = ends[1:] - ends[:-1]
    short = torch.div(lengths, 2, rounding_mode="floor")
    if int(short.sum()) >= int(lengths.sum()):
        positive = (lengths > 0).nonzero()
        if positive.numel() == 0:
            raise AssertionError("skew offsets have no rows to shorten")
        short = lengths.clone()
        short[int(positive[-1])] = 0
    if int(short.sum()) >= int(lengths.sum()):
        raise AssertionError("shorter rewrite did not shrink L")
    out = torch.zeros_like(ends)
    out[1:] = torch.cumsum(short, 0)
    seq_offsets.copy_(out.to(device=seq_offsets.device, dtype=seq_offsets.dtype))


def _reuse_pair(B, Mi, regime):
    if regime == "uniform":
        # New group count and a partial M tile, same N/K/XCD/block.
        return B + 16, Mi + _BLOCK_M // 2
    # Same B keeps the skew XCD choice; a larger envelope changes L.
    return B, Mi * 2


def _occupied_tiles(L, n_groups):
    return (L + _BLOCK_M - 1) // _BLOCK_M + n_groups


@benchmark()
def test_jdbba_launcher_reuse(B, D, Kout, Mi, regime, dtype):
    """Second call keeps the compiled launcher and matches the reference.

    Uniform varies n_groups and max_seq_len with the tile, XCD, and block
    held fixed. Skew varies L (total_occ_tiles) at the same N, K, and batch,
    so the XCD choice does not change.
    """
    B2, Mi2 = _reuse_pair(B, Mi, regime)
    uniform = regime == "uniform"
    st = torch.cuda.current_stream()
    first = _build_inputs(B, D, Kout, Mi, regime, dtype, pad_rows=False)
    second = _build_inputs(B2, D, Kout, Mi2, regime, dtype, pad_rows=False)
    if uniform:
        bm1 = (Mi + _BLOCK_M - 1) // _BLOCK_M
        bm2 = (Mi2 + _BLOCK_M - 1) // _BLOCK_M
        if (B2, bm2) == (B, bm1):
            raise AssertionError("uniform reuse pair did not change the grid")
    elif _occupied_tiles(second["L"], B2) == _occupied_tiles(first["L"], B):
        raise AssertionError("skew reuse pair did not change occupied tiles")

    def _call(inputs, n_groups, max_seq_len):
        if uniform:
            jagged_dense_bmm(
                inputs["tC"],
                inputs["tA"],
                inputs["dense_tall"],
                inputs["bias_flat"],
                inputs["seq_offsets"],
                n_groups=n_groups,
                max_seq_len=max_seq_len,
                stream=st,
                uniform_seqlen=True,
                xcd_c=32,
                xcd_w=8,
                block_k=64,
                threads=256,
            )
        else:
            jagged_dense_bmm_dispatched(
                inputs["tC"],
                inputs["tA"],
                inputs["dense_tall"],
                inputs["bias_flat"],
                inputs["seq_offsets"],
                n_groups,
                max_seq_len,
                stream=st,
                uniform_seqlen=False,
                block_k=64,
            )
        return inputs["out"]

    def _check(inputs, n_groups, max_seq_len, msg):
        ref = run_torch(
            inputs["jagged"],
            inputs["dense"],
            inputs["bias"],
            inputs["seq_offsets"],
            inputs["N"],
            dtype,
        )
        got = _call(inputs, n_groups, max_seq_len)
        return assertAllclose(
            ref.to(dtypes.fp32),
            got.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=msg,
        )

    _check(first, B, Mi, f"launcher reuse prime ({regime})")
    n_launchers = len(_LAUNCHER_CACHE)
    ref2 = run_torch(
        second["jagged"],
        second["dense"],
        second["bias"],
        second["seq_offsets"],
        second["N"],
        dtype,
    )
    got = _call(second, B2, Mi2)
    if len(_LAUNCHER_CACHE) != n_launchers:
        raise AssertionError(
            f"launcher cache grew {n_launchers} -> {len(_LAUNCHER_CACHE)} "
            f"for a new {regime} length"
        )
    assertAllclose(
        ref2.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg=f"launcher reuse ({regime})",
    )

    L2, N2, K2 = second["L"], second["N"], second["K"]
    flops = 2 * L2 * K2 * N2
    nbytes = (L2 * K2 + B2 * K2 * N2 + B2 * N2 + L2 * N2) * second[
        "jagged"
    ].element_size()

    def _second():
        return _call(second, B2, Mi2)

    got, us = run_perftest(_second)
    err = checkAllclose(
        ref2.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg=f"launcher reuse timed ({regime})",
    )
    return {
        "gfx": get_gfx(),
        "B2": B2,
        "Mi2": Mi2,
        "L": L2,
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
        "new_launchers": len(_LAUNCHER_CACHE) - n_launchers,
    }


def _skew_once(B, D, Kout, Mi, dtype, seq_offsets, stream, msg):
    d = _build_inputs(
        B, D, Kout, Mi, "skew", dtype, seq_offsets=seq_offsets, pad_rows=False
    )
    ref = run_torch(d["jagged"], d["dense"], d["bias"], d["seq_offsets"], d["N"], dtype)

    def _flydsl():
        jagged_dense_bmm_dispatched(
            d["tC"],
            d["tA"],
            d["dense_tall"],
            d["bias_flat"],
            d["seq_offsets"],
            B,
            Mi,
            stream=stream,
            uniform_seqlen=False,
        )
        return d["out"]

    got = _flydsl()
    err = assertAllclose(
        ref.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg=msg,
    )
    return d, ref, err


@benchmark()
def test_jdbba_tile_map_inplace(B, D, Kout, Mi, dtype):
    """In-place seq_offsets updates must rebuild TILE_MAP.

    The same storage is rewritten twice: once with a new segmentation of the
    same packed length, then with a shorter length. C is exactly (L, N). An
    explicit side stream, distinct from the current one, must still see the
    TILE_MAP prep before the GEMM.
    """
    seq = _make_seq_offsets(B, Mi, "skew")
    st = torch.cuda.current_stream()
    _skew_once(B, D, Kout, Mi, dtype, seq, st, "tile map initial")

    _resegment_same_L(seq)
    _, _, same_err = _skew_once(
        B, D, Kout, Mi, dtype, seq, st, "tile map same-L rewrite"
    )

    _shorten_offsets(seq)
    shorter, ref_short, _ = _skew_once(
        B, D, Kout, Mi, dtype, seq, st, "tile map shorter L"
    )
    L2, N2, K2 = shorter["L"], shorter["N"], shorter["K"]

    # Inputs are ready on the current stream. A large fill then occupies that
    # stream. The op is launched on a different stream, and only that stream
    # is synchronized before the check. Prep queued on the current stream
    # would still be behind the fill, and the GEMM would read an unfinished map.
    side_seq = _make_seq_offsets(B, Mi, "skew", seed=SEED + 7)
    side_inputs = _build_inputs(
        B,
        D,
        Kout,
        Mi,
        "skew",
        dtype,
        seq_offsets=side_seq,
        pad_rows=False,
    )
    side_ref = run_torch(
        side_inputs["jagged"],
        side_inputs["dense"],
        side_inputs["bias"],
        side_inputs["seq_offsets"],
        side_inputs["N"],
        dtype,
    )
    torch.cuda.current_stream().synchronize()
    stall = torch.empty(16 * 1024 * 1024, dtype=torch.int32, device="cuda")
    stall.fill_(1)
    side = torch.cuda.Stream()
    jagged_dense_bmm_dispatched(
        side_inputs["tC"],
        side_inputs["tA"],
        side_inputs["dense_tall"],
        side_inputs["bias_flat"],
        side_inputs["seq_offsets"],
        B,
        Mi,
        stream=side,
        uniform_seqlen=False,
    )
    side.synchronize()
    assertAllclose(
        side_ref.to(dtypes.fp32),
        side_inputs["out"].to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="tile map side stream",
    )
    torch.cuda.current_stream().synchronize()

    def _shorter():
        jagged_dense_bmm_dispatched(
            shorter["tC"],
            shorter["tA"],
            shorter["dense_tall"],
            shorter["bias_flat"],
            shorter["seq_offsets"],
            B,
            Mi,
            stream=st,
            uniform_seqlen=False,
        )
        return shorter["out"]

    got, us = run_perftest(_shorter)
    err = checkAllclose(
        ref_short.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="tile map shorter L timed",
    )
    flops = 2 * L2 * K2 * N2
    nbytes = (L2 * K2 + B * K2 * N2 + B * N2 + L2 * N2) * shorter[
        "jagged"
    ].element_size()
    return {
        "gfx": get_gfx(),
        "L": L2,
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
        "same_L err": same_err,
    }


def _clear_dispatch_cache():
    jdbba_dispatch._DISPATCH_TABLES.clear()
    AITER_CONFIGS.get_config_file.cache_clear()


def _arch_winner_key():
    B, D, Kout, Mi = _ARCH_SHAPE
    return (D, Kout, tile_bucket(B, Mi, Kout))


def _resolve_arch_shape():
    B, D, Kout, Mi = _ARCH_SHAPE
    return resolve_config(n_groups=B, reduction_k=D, output_n=Kout, max_seq_len=Mi)


def _wrapper_refuses(arch):
    """Public entry raises before it reads tensors or compiles."""
    try:
        jagged_dense_bmm_dispatched(None, None, None, None, None, 1, 1)
    except RuntimeError as exc:
        text = str(exc)
        return arch in text and "supports" in text
    return False


def _check_arch_traits(arch):
    """The arch record owns the measured MFMA, K mode, and LDS budget."""
    traits = jdbba_arch_traits(arch)
    expect = _TRAITS[arch]
    if (
        traits.mfma_k != expect["mfma_k"]
        or traits.lds_bytes != expect["lds_bytes"]
        or traits.k_perm_shape != expect["k_perm_shape"]
        or traits.k_slice_nested != expect["k_slice_nested"]
        or traits.swizzle != (3, 3, 3)
    ):
        raise AssertionError(f"{arch} traits {traits} diverged from {expect}")
    # 128x256 bf16, two A stages: 128*256*2*2 = 131072 bytes.
    wide = smem_bytes(128, 128, 256)
    if arch == "gfx942" and wide <= traits.lds_bytes:
        raise AssertionError(f"{arch} accepted a {wide}-byte tile")
    if arch == "gfx950" and wide > traits.lds_bytes:
        raise AssertionError(f"{arch} rejected a {wide}-byte tile")
    if smem_bytes(128, 128, 128) > traits.lds_bytes:
        raise AssertionError(f"{arch} rejected the measured block_k 128 tile")


def _check_shipped_section(arch):
    _clear_dispatch_cache()
    table = _load_dispatch_table()
    selected = table.get("gfx") or ""
    if arch in SUPPORTED_GFX:
        if selected != arch:
            raise AssertionError(f"{arch} loaded section {selected!r}")
        cfg = _resolve_arch_shape()
        winner = table["winners"][_arch_winner_key()]
        if (
            cfg["xcd_c"] != winner["xcd_c"]
            or cfg["xcd_w"] != winner["xcd_w"]
            or cfg["threads"] != winner["threads"]
            or cfg["block_k"] != winner["block_k"]
        ):
            raise AssertionError(f"{arch} config {cfg} diverged from {winner}")
        B, D, Kout, Mi = _ARCH_SHAPE
        near = resolve_config(
            n_groups=128, reduction_k=D, output_n=Kout, max_seq_len=Mi
        )
        if near["xcd_c"] != winner["xcd_c"] or near["threads"] != winner["threads"]:
            raise AssertionError(f"{arch} B=128 missed the B={B} tile bucket")
        miss = resolve_config(n_groups=4, reduction_k=D, output_n=Kout, max_seq_len=128)
        expect = analytic_config(
            uniform_seqlen=True,
            n_groups=4,
            reduction_k=D,
            num_xcds=int(jdbba_dispatch.get_num_xcds()),
        )
        if miss["xcd_w"] != expect["xcd_w"] or miss["threads"] != expect["threads"]:
            raise AssertionError(f"{arch} empty bucket used {miss}")
        jdbba_dispatch._require_supported_arch()
        return selected, 0
    if selected or table["winners"]:
        raise AssertionError(f"{arch} loaded section {selected!r}")
    if not _wrapper_refuses(arch):
        raise AssertionError(f"{arch} did not raise before compile")
    return selected, 1


def _check_multisection_override(arch):
    """A CSV with two archs yields the runtime arch's row only."""
    _k, _n, bucket = _arch_winner_key()
    rows = {
        "gfx942": {"xcd_c": 32, "xcd_w": 16, "threads": 512},
        "gfx950": {"xcd_c": 7, "xcd_w": 3, "threads": 128},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "jagged_dense_bmm_tuned.csv")
        with open(path, "w", encoding="utf-8") as fh:
            lines = [
                (
                    "gfx,cu_num,num_xcds,regime,K,N,tile_bucket,"
                    "xcd_c,xcd_w,threads,block_k,us\n"
                )
            ]
            lines.extend(
                f"{gfx},{_MEASURED_CU[gfx]},{_MEASURED_XCDS},uniform,{_k},{_n},{bucket},"
                f"{cfg['xcd_c']},{cfg['xcd_w']},{cfg['threads']},64,1\n"
                for gfx, cfg in rows.items()
            )
            # Same shape on another XCD count must not replace the live row.
            lines.extend(
                f"{gfx},{_MEASURED_CU[gfx]},4,uniform,{_k},{_n},{bucket},9,9,"
                f"{cfg['threads']},64,1\n"
                for gfx, cfg in rows.items()
            )
            fh.writelines(lines)
        os.environ["AITER_CONFIG_JAGGED_DENSE_BMM"] = path
        # The retired env must not select a table.
        os.environ["FLYDSL_JAGGED_DENSE_BMM_DISPATCH_JSON"] = path
        _clear_dispatch_cache()
        table = _load_dispatch_table()
        selected = table.get("gfx") or ""
        if arch in rows:
            expect = rows[arch]
            cfg = _resolve_arch_shape()
            if (
                selected != arch
                or cfg["xcd_c"] != expect["xcd_c"]
                or cfg["xcd_w"] != expect["xcd_w"]
                or cfg["threads"] != expect["threads"]
            ):
                raise AssertionError(
                    f"{arch} override selected {selected!r} "
                    f"xcd_c={cfg['xcd_c']} xcd_w={cfg['xcd_w']} threads={cfg['threads']}"
                )
        elif selected or table["winners"]:
            raise AssertionError(f"{arch} override loaded section {selected!r}")
        os.environ.pop("AITER_CONFIG_JAGGED_DENSE_BMM", None)
        os.environ.pop("FLYDSL_JAGGED_DENSE_BMM_DISPATCH_JSON", None)
        _clear_dispatch_cache()
    return selected


def _check_xcd_partition(arch, num_xcds):
    """A partition without its own row uses analytic_config, not the 8-XCD winner."""
    with mock.patch.object(jdbba_dispatch, "get_num_xcds", return_value=num_xcds):
        _clear_dispatch_cache()
        table = _load_dispatch_table()
        if table["winners"]:
            raise AssertionError(
                f"{arch} num_xcds={num_xcds} loaded {len(table['winners'])} rows"
            )
        B, D, Kout, Mi = _ARCH_SHAPE
        cfg = resolve_config(n_groups=B, reduction_k=D, output_n=Kout, max_seq_len=Mi)
        expect = analytic_config(
            uniform_seqlen=True, n_groups=B, reduction_k=D, num_xcds=num_xcds
        )
        if cfg["xcd_c"] != expect["xcd_c"] or cfg["xcd_w"] != expect["xcd_w"]:
            raise AssertionError(f"{arch} num_xcds={num_xcds} config {cfg} != {expect}")
        if num_xcds == 1 and (cfg["xcd_c"] != 1 or cfg["xcd_w"] != 1):
            raise AssertionError(f"{arch} 1 XCD did not use identity {cfg}")
        skew = resolve_config(
            n_groups=B,
            reduction_k=D,
            output_n=Kout,
            max_seq_len=Mi,
            uniform_seqlen=False,
        )
        skew_expect = analytic_config(
            uniform_seqlen=False, n_groups=B, reduction_k=D, num_xcds=num_xcds
        )
        if (
            skew["xcd_c"] != skew_expect["xcd_c"]
            or skew["xcd_w"] != skew_expect["xcd_w"]
            or skew["block_k"] != skew_expect["block_k"]
        ):
            raise AssertionError(f"{arch} num_xcds={num_xcds} skew {skew}")
    _clear_dispatch_cache()


@benchmark()
def test_jdbba_arch_section(arch):
    """Which tuned rows the runtime arch owns, and whether launch is refused.

    Host selection only: the row records the section and the pre-compile refusal.
    """
    spoof = "gfx950" if arch == "gfx942" else "gfx942"
    os.environ["FLYDSL_JAGGED_DENSE_BMM_ARCH"] = spoof
    mfma_k = ""
    lds_kb = ""
    if arch in SUPPORTED_GFX:
        _check_arch_traits(arch)
        traits = jdbba_arch_traits(arch)
        mfma_k = traits.mfma_k
        lds_kb = traits.lds_bytes // 1024
    prev_config = os.environ.pop("AITER_CONFIG_JAGGED_DENSE_BMM", None)
    try:
        with (
            mock.patch.object(jdbba_dispatch, "get_gfx_runtime", return_value=arch),
            mock.patch.object(
                jdbba_dispatch, "get_cu_num", return_value=_MEASURED_CU.get(arch, 0)
            ),
            mock.patch.object(
                jdbba_dispatch, "get_num_xcds", return_value=_MEASURED_XCDS
            ),
        ):
            selected, raised = _check_shipped_section(arch)
            override = _check_multisection_override(arch)
            _check_xcd_partition(arch, 4)
            _check_xcd_partition(arch, 1)
        os.environ.pop("AITER_CONFIG_JAGGED_DENSE_BMM", None)
        if prev_config is not None:
            os.environ["AITER_CONFIG_JAGGED_DENSE_BMM"] = prev_config
        if arch == get_gfx():
            _clear_dispatch_cache()
            if arch in SUPPORTED_GFX:
                if get_cu_num() != _MEASURED_CU[arch]:
                    raise AssertionError(
                        f"live cu_num {get_cu_num()} has no {arch} winners"
                    )
                if get_num_xcds() != _MEASURED_XCDS:
                    raise AssertionError(
                        f"live num_xcds {get_num_xcds()} has no {arch} winners"
                    )
                live_traits = jdbba_arch_traits(get_gfx())
                if (
                    live_traits.arch != arch
                    or live_traits.mfma_k != _TRAITS[arch]["mfma_k"]
                ):
                    raise AssertionError(f"live traits {live_traits}")
                live = _load_dispatch_table().get("gfx") or ""
                if live != arch:
                    raise AssertionError(f"live {get_gfx()} loaded section {live!r}")
                jdbba_dispatch._require_supported_arch()
            elif not _wrapper_refuses(get_gfx()):
                raise AssertionError(f"live {get_gfx()} did not raise before compile")
    finally:
        os.environ.pop("FLYDSL_JAGGED_DENSE_BMM_ARCH", None)
        os.environ.pop("FLYDSL_JAGGED_DENSE_BMM_DISPATCH_JSON", None)
        if prev_config is None:
            os.environ.pop("AITER_CONFIG_JAGGED_DENSE_BMM", None)
        else:
            os.environ["AITER_CONFIG_JAGGED_DENSE_BMM"] = prev_config
        _clear_dispatch_cache()
    return {
        "gfx": get_gfx(),
        "selected": selected,
        "override": override,
        "raised": raised,
        "xcd4": "analytic",
        "xcd1": "identity",
        "mfma_k": mfma_k,
        "lds_kb": lds_kb,
        "err": 0,
    }


def _summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def _expand_shapes(shapes, batch, parser):
    """Yield (B, D, Kout, Mi). --batch, when given, supplies/overrides B."""
    out = []
    for shape in shapes:
        if not isinstance(shape, tuple):
            shape = (shape,)
        if len(shape) == 4:
            head, rest = (shape[0],), shape[1:]
        elif len(shape) == 3:
            head, rest = (), shape
        else:
            parser.error(f"shape must be B,D,Kout,Mi or D,Kout,Mi; got {shape}")
        if batch:
            out.extend((b, *rest) for b in batch)
        elif head:
            out.append((*head, *rest))
        else:
            parser.error(f"shape {shape} omits B; pass -b/--batch")
    return list(dict.fromkeys(out))


def main():
    # Runs on every card, including ones the GPU sweep skips. Selection is
    # mocked, so this does not compile the gfx942 kernel on the live GPU.
    rows = []
    for (arch,) in itertools.product(_ARCH_SECTION_CASES):
        rows.append(test_jdbba_arch_section(arch))
    _summarize("jagged_dense_bmm arch section", rows)

    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("jagged_dense_bmm unsupported on %s; skipping", get_gfx())
        return
    if not _HAS_TRITON:
        aiter.logger.warning(
            "generative-recommenders Triton baseline unavailable (%s); "
            "running flydsl-only",
            _TRITON_ERR,
        )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="e.g.: -d bf16",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=None,
        help="B (n_groups). Supplies B for 3-tuple -s, or overrides B in 4-tuple -s.\n"
        "e.g.: -b 120 1024 -s 256,256,7680",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (120, 256, 256, 7680),
            (120, 512, 512, 7680),
            (1024, 256, 256, 7680),
            (1024, 512, 512, 7680),
            (64, 512, 1024, 8192),
        ],
        help="B,D,Kout,Mi or D,Kout,Mi (requires -b).\n"
        "e.g.: -s 64,512,1024,8192  or  -s 256,256,7680",
    )
    parser.add_argument(
        "-r",
        "--regime",
        type=str,
        choices=["uniform", "skew"],
        nargs="*",
        default=["uniform", "skew"],
        help="e.g.: -r uniform",
    )
    parser.add_argument(
        "-ds",
        "--dispatch-shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (120, 256, 256, 7680),
            (120, 512, 512, 7680),
            (1024, 256, 256, 7680),
            (1024, 512, 512, 7680),
            (120, 384, 384, 7680),
        ],
        help="e.g.: -ds 120,384,384,7680",
    )
    parser.add_argument(
        "-k",
        "--block-k",
        type=int,
        nargs="*",
        default=[64, 128],
        help="e.g.: -k 64 128",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[111, 222],
        help="e.g.: --seeds 111 222",
    )
    parser.add_argument(
        "-rs",
        "--regression-shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(64, 256, 256, 2048)],
        help="B,D,Kout,Mi for block_k and skew-varying-L. e.g.: -rs 64,256,256,2048",
    )
    args = parser.parse_args()
    shapes = _expand_shapes(args.shapes, args.batch, parser)
    for shape in args.regression_shapes:
        if not isinstance(shape, tuple) or len(shape) != 4:
            parser.error(f"regression shape must be B,D,Kout,Mi; got {shape}")

    for dtype in args.dtype:
        rows = []
        for regime, (B, D, Kout, Mi) in itertools.product(args.regime, shapes):
            rows.append(test_jdbba(B, D, Kout, Mi, regime, dtype))
        _summarize("jagged_dense_bmm", rows)

    rows = [
        test_jdbba_dispatch(B, D, Kout, Mi) for (B, D, Kout, Mi) in args.dispatch_shapes
    ]
    _summarize("jagged_dense_bmm dispatch routing", rows)

    rows = []
    for dtype, (B, D, Kout, Mi), block_k in itertools.product(
        args.dtype, args.regression_shapes, args.block_k
    ):
        rows.append(test_jdbba_block_k(B, D, Kout, Mi, block_k, dtype))
    _summarize("jagged_dense_bmm block_k", rows)

    rows = []
    for dtype, (B, D, Kout, Mi), seed in itertools.product(
        args.dtype, args.regression_shapes, args.seeds
    ):
        rows.append(test_jdbba_skew_varying_L(B, D, Kout, Mi, seed, dtype))
    _summarize("jagged_dense_bmm skew varying L", rows)

    rows = []
    for dtype, regime, (B, D, Kout, Mi) in itertools.product(
        args.dtype, args.regime, args.regression_shapes
    ):
        rows.append(test_jdbba_launcher_reuse(B, D, Kout, Mi, regime, dtype))
    _summarize("jagged_dense_bmm launcher reuse", rows)

    rows = [
        test_jdbba_tile_map_inplace(B, D, Kout, Mi, dtype)
        for dtype, (B, D, Kout, Mi) in itertools.product(
            args.dtype, args.regression_shapes
        )
    ]
    _summarize("jagged_dense_bmm tile map inplace", rows)


if __name__ == "__main__":
    main()
