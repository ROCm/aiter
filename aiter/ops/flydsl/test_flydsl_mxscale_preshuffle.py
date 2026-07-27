# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the FlyDSL MXFP4/MXFP8 preshuffle GEMM (gfx950 MFMA).

Covers A4W4 (MXFP4 A x MXFP4 B) and A8W8 (MXFP8 A x MXFP8 B), per-1x32 E8M0
microscale folded into the scaled 16x16x128 MFMA, plus an A8W8 coarse-blockscale
case (A 1x128, B 128x128) broadcast caller-side to per-1x32. All quant/shuffle/
dequant reuse aiter's own helpers (aiter.ops.quant / aiter.ops.shuffle /
aiter.utility.fp4_utils) — no kernel-specific reference port is needed.

Usage:
    python aiter/ops/flydsl/test_flydsl_mxscale_preshuffle.py
    pytest -q aiter/ops/flydsl/test_flydsl_mxscale_preshuffle.py
"""

from __future__ import annotations

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL tests.", allow_module_level=True
    )

import torch.nn.functional as F
from flydsl.runtime.device import get_rocm_arch

from aiter import dtypes
from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
    flydsl_mxscale_preshuffle_gemm,
)
from aiter.ops.quant import per_1x32_f4_quant, per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight
from aiter.test_common import checkAllclose
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

_SHAPES = [
    (64, 8192, 8192, 64, 128, 128),
    (32, 8192, 8192, 32, 128, 256),
]

# in-kernel blockscale also covers tile_n=256 (n_pairs=2 -> B-load dedup path)
# and K=384 (K%256!=0 -> exercises the padded-chunk compact layout, tile_k=128).
_BLOCKSCALE_SHAPES = _SHAPES + [
    (32, 8192, 8192, 32, 256, 256),
    (32, 2048, 384, 32, 128, 128),
]

# (M, N, K, tile_m, tile_n, tile_k, split_k)
_SPLITK_SHAPES = [
    (8, 2048, 7168, 32, 128, 256, 2),
    (8, 2048, 7168, 32, 128, 256, 4),
    (1, 2048, 7168, 32, 128, 128, 2),
    (16, 4096, 8192, 32, 128, 256, 8),
]


def _skip_if_not_gfx950():
    arch = str(get_rocm_arch())
    if arch != "gfx950":
        pytest.skip(f"MXFP preshuffle GEMM requires gfx950, got {arch}")


def _rand_ab(M, N, K, dev):
    Ma, Na = (M + 31) // 32 * 32, (N + 31) // 32 * 32
    a_f = torch.zeros(Ma, K, device=dev)
    b_f = torch.zeros(Na, K, device=dev)
    a_f[:M] = torch.randn(M, K, device=dev)
    b_f[:N] = torch.randn(N, K, device=dev)
    return a_f, b_f


@pytest.mark.parametrize("M, N, K, tile_m, tile_n, tile_k", _SHAPES)
def test_a4w4(M, N, K, tile_m, tile_n, tile_k):
    """MXFP4 A x MXFP4 B."""
    _skip_if_not_gfx950()
    dev = torch.device("cuda")
    a_f, b_f = _rand_ab(M, N, K, dev)

    a_q, sa = per_1x32_f4_quant(a_f, quant_dtype=dtypes.fp4x2)
    b_q, sb = per_1x32_f4_quant(b_f, quant_dtype=dtypes.fp4x2)
    a_codes, b_codes = a_q[:M], b_q[:N]
    b_shuf = shuffle_weight(b_codes, layout=(16, 16))
    scale_a = shuffle_scale_a16w4(sa, 1, False)
    scale_b = shuffle_scale_a16w4(sb, 1, False)

    a_deq = fp4_utils.mxfp4_to_f32(a_codes) * fp4_utils.e8m0_to_f32(
        sa[:M].repeat_interleave(32, dim=1)
    )
    b_deq = fp4_utils.mxfp4_to_f32(b_codes) * fp4_utils.e8m0_to_f32(
        sb[:N].repeat_interleave(32, dim=1)
    )
    c_ref = F.linear(a_deq.to(torch.float32), b_deq.to(torch.float32)).to(
        torch.bfloat16
    )

    out = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    flydsl_mxscale_preshuffle_gemm(
        a_codes,
        b_shuf,
        scale_a,
        scale_b,
        out,
        a_dtype="fp4",
        b_dtype="fp4",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )
    torch.cuda.synchronize()
    err = checkAllclose(
        c_ref, out, rtol=1e-2, atol=1e-2, msg="a4w4", catastrophic_check=True
    )
    assert err < 0.01, f"a4w4 mismatch ratio={err:.4f}"


@pytest.mark.parametrize("M, N, K, tile_m, tile_n, tile_k", _SHAPES)
def test_a8w8(M, N, K, tile_m, tile_n, tile_k):
    """MXFP8 (E4M3) A x MXFP8 (E4M3) B."""
    _skip_if_not_gfx950()
    dev = torch.device("cuda")
    a_f, b_f = _rand_ab(M, N, K, dev)

    a_q, sa = per_1x32_f8_scale_f8_quant(
        a_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    b_q, sb = per_1x32_f8_scale_f8_quant(
        b_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    a_codes, b_codes = a_q[:M], b_q[:N]
    b_shuf = shuffle_weight(b_codes, layout=(16, 16))
    scale_a = shuffle_scale_a16w4(sa, 1, False)
    scale_b = shuffle_scale_a16w4(sb, 1, False)

    a_deq = a_codes.float() * fp4_utils.e8m0_to_f32(sa[:M].repeat_interleave(32, dim=1))
    b_deq = b_codes.float() * fp4_utils.e8m0_to_f32(sb[:N].repeat_interleave(32, dim=1))
    c_ref = F.linear(a_deq.to(torch.float32), b_deq.to(torch.float32)).to(
        torch.bfloat16
    )

    out = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    flydsl_mxscale_preshuffle_gemm(
        a_codes,
        b_shuf,
        scale_a,
        scale_b,
        out,
        a_dtype="fp8",
        b_dtype="fp8",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )
    torch.cuda.synchronize()
    err = checkAllclose(
        c_ref, out, rtol=1e-2, atol=1e-2, msg="a8w8", catastrophic_check=True
    )
    assert err < 0.01, f"a8w8 mismatch ratio={err:.4f}"


@pytest.mark.parametrize("M, N, K, tile_m, tile_n, tile_k, split_k", _SPLITK_SHAPES)
def test_a8w8_splitk(M, N, K, tile_m, tile_n, tile_k, split_k):
    """MXFP8 A x MXFP8 B with split-K (fp32 partial slabs + reduce): checks both
    the dequant reference and that split-K matches the single-launch result."""
    _skip_if_not_gfx950()
    dev = torch.device("cuda")
    a_f, b_f = _rand_ab(M, N, K, dev)

    a_q, sa = per_1x32_f8_scale_f8_quant(
        a_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    b_q, sb = per_1x32_f8_scale_f8_quant(
        b_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    a_codes, b_codes = a_q[:M], b_q[:N]
    b_shuf = shuffle_weight(b_codes, layout=(16, 16))
    scale_a = shuffle_scale_a16w4(sa, 1, False)
    scale_b = shuffle_scale_a16w4(sb, 1, False)

    a_deq = a_codes.float() * fp4_utils.e8m0_to_f32(sa[:M].repeat_interleave(32, dim=1))
    b_deq = b_codes.float() * fp4_utils.e8m0_to_f32(sb[:N].repeat_interleave(32, dim=1))
    c_ref = F.linear(a_deq.to(torch.float32), b_deq.to(torch.float32)).to(
        torch.bfloat16
    )

    def _run(sk):
        out = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
        flydsl_mxscale_preshuffle_gemm(
            a_codes,
            b_shuf,
            scale_a,
            scale_b,
            out,
            a_dtype="fp8",
            b_dtype="fp8",
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            split_k=sk,
        )
        torch.cuda.synchronize()
        return out

    out_sk = _run(split_k)
    out_1 = _run(1)
    err = checkAllclose(
        c_ref,
        out_sk,
        rtol=1e-2,
        atol=1e-2,
        msg=f"a8w8 split_k={split_k}",
        catastrophic_check=True,
    )
    assert err < 0.01, f"a8w8 split_k={split_k} mismatch ratio={err:.4f}"
    # split-K sums fp32 partials -> should closely match the single-launch result.
    err_self = checkAllclose(
        out_1,
        out_sk,
        rtol=1e-2,
        atol=1e-2,
        msg=f"a8w8 split_k={split_k} vs split_k=1",
        catastrophic_check=True,
    )
    assert err_self < 0.01, f"a8w8 split_k={split_k} vs split_k=1 ratio={err_self:.4f}"


@pytest.mark.parametrize("M, N, K, tile_m, tile_n, tile_k", _SHAPES)
def test_a8w8_blockscale(M, N, K, tile_m, tile_n, tile_k):
    """MXFP8 A x MXFP8 B with coarse blockscale (A 1x128 per-token, B 128x128
    per-block).

    Blockscale support is entirely caller-side: broadcast each block value to per-1x32 (A: x4 along K;
    B: x128 along N + x4 along K), shuffle, then call the plain op. This documents
    the pattern for feeding blockscale scales into the 1x32 scaled-MFMA path.
    """
    _skip_if_not_gfx950()
    dev = torch.device("cuda")
    a_f, b_f = _rand_ab(M, N, K, dev)

    a_q, sa = per_1x32_f8_scale_f8_quant(
        a_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    b_q, sb = per_1x32_f8_scale_f8_quant(
        b_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    a_codes, b_codes = a_q[:M], b_q[:N]
    b_shuf = shuffle_weight(b_codes, layout=(16, 16))

    # Coarsen per-1x32 E8M0 -> 1x128 (A) / 128x128 (B) by max over each block, so
    # codes and the block scale stay consistent (max exponent never underflows).
    sa_u = sa.view(torch.uint8)
    sb_u = sb.view(torch.uint8)
    Ma = sa_u.shape[0]
    assert N % 128 == 0, "blockscale test needs N % 128 == 0"
    sa_128 = sa_u.view(Ma, K // 128, 4).amax(dim=2).contiguous()
    sb_128 = sb_u[:N].view(N // 128, 128, K // 128, 4).amax(dim=(1, 3)).contiguous()

    # Caller-side broadcast: coarse blockscale -> per-1x32, then shuffle as usual.
    scale_a = shuffle_scale_a16w4(sa_128.repeat_interleave(4, dim=1), 1, False)
    scale_b = shuffle_scale_a16w4(
        sb_128.repeat_interleave(128, dim=0).repeat_interleave(4, dim=1), 1, False
    )

    a_deq = a_codes.float() * fp4_utils.e8m0_to_f32(
        sa_128[:M].repeat_interleave(128, dim=1)
    )
    b_deq = b_codes.float() * fp4_utils.e8m0_to_f32(
        sb_128.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    )
    c_ref = F.linear(a_deq.to(torch.float32), b_deq.to(torch.float32)).to(
        torch.bfloat16
    )

    out = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    flydsl_mxscale_preshuffle_gemm(
        a_codes,
        b_shuf,
        scale_a,
        scale_b,
        out,
        a_dtype="fp8",
        b_dtype="fp8",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )
    torch.cuda.synchronize()
    err = checkAllclose(
        c_ref, out, rtol=1e-2, atol=1e-2, msg="a8w8 blockscale", catastrophic_check=True
    )
    assert err < 0.01, f"a8w8 blockscale mismatch ratio={err:.4f}"


@pytest.mark.parametrize("M, N, K, tile_m, tile_n, tile_k", _BLOCKSCALE_SHAPES)
def test_a8w8_blockscale_inkernel(M, N, K, tile_m, tile_n, tile_k):
    """MXFP8 A x MXFP8 B coarse blockscale (A 1x128, B 128x128) via the in-kernel
    broadcast path (blockscale=True, compact scales — no per-1x32 buffer).

    Must byte-match the host-expand oracle (same values fed as full per-1x32) and
    the dequant reference."""
    _skip_if_not_gfx950()
    dev = torch.device("cuda")
    a_f, b_f = _rand_ab(M, N, K, dev)

    a_q, sa = per_1x32_f8_scale_f8_quant(
        a_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    b_q, sb = per_1x32_f8_scale_f8_quant(
        b_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    a_codes, b_codes = a_q[:M], b_q[:N]
    b_shuf = shuffle_weight(b_codes, layout=(16, 16))

    sa_u, sb_u = sa.view(torch.uint8), sb.view(torch.uint8)
    Ma = sa_u.shape[0]
    assert N % 128 == 0, "blockscale test needs N % 128 == 0"
    sa_128 = sa_u.view(Ma, K // 128, 4).amax(dim=2).contiguous()
    sb_128 = sb_u[:N].view(N // 128, 128, K // 128, 4).amax(dim=(1, 3)).contiguous()

    # Oracle: same block-constant values expanded to full per-1x32 through plain op.
    scale_a = shuffle_scale_a16w4(sa_128.repeat_interleave(4, dim=1), 1, False)
    scale_b = shuffle_scale_a16w4(
        sb_128.repeat_interleave(128, dim=0).repeat_interleave(4, dim=1), 1, False
    )
    out_oracle = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    flydsl_mxscale_preshuffle_gemm(
        a_codes,
        b_shuf,
        scale_a,
        scale_b,
        out_oracle,
        a_dtype="fp8",
        b_dtype="fp8",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )
    torch.cuda.synchronize()

    # In-kernel broadcast: compact coarse scales + blockscale=True.
    out = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    flydsl_mxscale_preshuffle_gemm(
        a_codes,
        b_shuf,
        sa_128,
        sb_128,
        out,
        a_dtype="fp8",
        b_dtype="fp8",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        blockscale=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_oracle), "in-kernel blockscale != host-expand oracle"

    a_deq = a_codes.float() * fp4_utils.e8m0_to_f32(
        sa_128[:M].repeat_interleave(128, dim=1)
    )
    b_deq = b_codes.float() * fp4_utils.e8m0_to_f32(
        sb_128.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    )
    c_ref = F.linear(a_deq.to(torch.float32), b_deq.to(torch.float32)).to(
        torch.bfloat16
    )
    err = checkAllclose(
        c_ref,
        out,
        rtol=1e-2,
        atol=1e-2,
        msg="a8w8 blockscale in-kernel",
        catastrophic_check=True,
    )
    assert err < 0.01, f"a8w8 blockscale in-kernel mismatch ratio={err:.4f}"


def _verify_tuned_shape(
    M,
    N,
    K,
    a_dtype,
    b_dtype,
    cfg,
    dev,
    rtol=1e-2,
    atol=1e-2,
    n_chunk=8192,
    blockscale=False,
):

    def _run(scale_a, scale_b, bs):
        out = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
        flydsl_mxscale_preshuffle_gemm(
            a_codes,
            b_shuf,
            scale_a,
            scale_b,
            out,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            tile_m=cfg["tile_m"],
            tile_n=cfg["tile_n"],
            tile_k=cfg["tile_k"],
            waves_per_eu=cfg["waves_per_eu"],
            xcd_swizzle=cfg["xcd_swizzle"],
            split_k=cfg["split_k"],
            blockscale=bs,
        )
        torch.cuda.synchronize()
        return out

    a_f, b_f = _rand_ab(M, N, K, dev)
    a_q, sa = per_1x32_f8_scale_f8_quant(
        a_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    b_q, sb = per_1x32_f8_scale_f8_quant(
        b_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    del a_f, b_f
    a_codes, b_codes = a_q[:M], b_q[:N]
    b_shuf = shuffle_weight(b_codes, layout=(16, 16))

    if not blockscale:
        # MX per-1x32: run once, compare to per-1x32 dequant reference.
        out = _run(
            shuffle_scale_a16w4(sa, 1, False),
            shuffle_scale_a16w4(sb, 1, False),
            False,
        )
        a_deq = a_codes.float() * fp4_utils.e8m0_to_f32(
            sa[:M].repeat_interleave(32, dim=1)
        )
        worst = 0.0
        for n0 in range(0, N, n_chunk):
            n1 = min(N, n0 + n_chunk)
            b_deq = b_codes[n0:n1].float() * fp4_utils.e8m0_to_f32(
                sb[n0:n1].repeat_interleave(32, dim=1)
            )
            ref = F.linear(a_deq, b_deq).to(torch.bfloat16)
            worst = max(
                worst,
                float(
                    checkAllclose(
                        ref,
                        out[:, n0:n1],
                        rtol=rtol,
                        atol=atol,
                        printLog=False,
                        catastrophic_check=True,
                    )
                ),
            )
            del b_deq, ref
        return worst

    # blockscale (a8w8): compact coarse scales (A 1x128, B 128x128), run the
    # in-kernel broadcast path once and compare to the block-constant dequant
    # reference (same style as op_tests test_gemm_a8w8_blockscale.py). The
    # stronger in-kernel==host-expand byte check lives in test_a8w8_blockscale_inkernel.
    if a_dtype != "fp8" or b_dtype != "fp8":
        raise ValueError("blockscale verify supports fp8/fp8 (a8w8) only")
    if N % 128 != 0:
        raise ValueError(f"blockscale verify needs N%128==0, got N={N}")
    sau, sbu = sa.view(torch.uint8), sb.view(torch.uint8)
    Ma = sau.shape[0]
    sa_128 = sau.view(Ma, K // 128, 4).amax(dim=2).contiguous()
    sb_128 = sbu[:N].view(N // 128, 128, K // 128, 4).amax(dim=(1, 3)).contiguous()

    out = _run(sa_128, sb_128, True)
    a_deq = a_codes.float() * fp4_utils.e8m0_to_f32(
        sa_128[:M].repeat_interleave(128, dim=1)
    )
    worst = 0.0
    for n0 in range(0, N, n_chunk):
        n1 = min(N, n0 + n_chunk)
        blk0, blk1 = n0 // 128, (n1 + 127) // 128
        sb_rows = sb_128[blk0:blk1].repeat_interleave(128, dim=0)[: n1 - n0]
        b_deq = b_codes[n0:n1].float() * fp4_utils.e8m0_to_f32(
            sb_rows.repeat_interleave(128, dim=1)
        )
        ref = F.linear(a_deq, b_deq).to(torch.bfloat16)
        worst = max(
            worst,
            float(
                checkAllclose(
                    ref,
                    out[:, n0:n1],
                    rtol=rtol,
                    atol=atol,
                    printLog=False,
                    catastrophic_check=True,
                )
            ),
        )
        del b_deq, ref
    return worst


def _verify_tuned_csv(
    csv_path, limit=None, m_cap=None, fail_thr=0.01, blockscale=False
):
    """Sweep every shape in the tuned CSV, run it with its recorded config, and
    check accuracy. Prints a per-shape line + a summary; returns exit-style int
    (0 = all good). blockscale=True verifies the in-kernel blockscale path
    (in-kernel == host-expand oracle + dequant reference) with the same config."""
    import pandas as pd

    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
        parse_kernel_name,
    )

    _skip_if_not_gfx950()
    dev = torch.device("cuda")
    df = pd.read_csv(csv_path).drop_duplicates(["M", "N", "K", "a_dtype", "b_dtype"])
    if limit:
        df = df.head(int(limit))
    total = len(df)
    passed = failed = errored = 0
    fails = []
    mode = "blockscale" if blockscale else "MX per-1x32"
    print(f"Verifying {total} tuned shapes ({mode}) from {csv_path}\n" + "=" * 78)
    for i, row in enumerate(df.itertuples(), 1):
        M, N, K = int(row.M), int(row.N), int(row.K)
        a_dtype, b_dtype = str(row.a_dtype), str(row.b_dtype)
        cfg = parse_kernel_name(str(row.kernelName))
        if cfg is None:
            print(f"[{i}/{total}] {M}x{N}x{K} SKIP (unparsable {row.kernelName!r})")
            errored += 1
            fails.append((M, N, K, "unparsable kernelName"))
            continue
        run_m = min(M, m_cap) if m_cap else M
        tag = f"{M}x{N}x{K} {a_dtype}/{b_dtype} " + (
            f"{cfg['tile_m']}x{cfg['tile_n']}x{cfg['tile_k']} "
            f"w{cfg['waves_per_eu']}_x{cfg['xcd_swizzle']}_sk{cfg['split_k']}"
        )
        try:
            err = _verify_tuned_shape(
                run_m, N, K, a_dtype, b_dtype, cfg, dev, blockscale=blockscale
            )
        # a config that crashes is itself a failure
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{total}] {tag} ERROR: {exc}")
            errored += 1
            fails.append((M, N, K, f"exception: {exc}"))
            torch.cuda.empty_cache()
            continue
        # free this shape's tensors before the next iteration (long sweep)
        torch.cuda.empty_cache()
        ok = err < fail_thr
        passed += ok
        failed += not ok
        note = f" (m_cap={run_m})" if run_m != M else ""
        print(
            f"[{i}/{total}] {tag}{note} err={err:.4g} "
            + ("PASS" if ok else "\033[31mFAIL\033[0m")
        )
        if not ok:
            fails.append((M, N, K, f"mismatch ratio={err:.4g}"))
    print("=" * 78)
    print(f"passed={passed}  failed={failed}  errored={errored}  total={total}")
    if fails:
        print("\nShapes with issues:")
        for M, N, K, why in fails:
            print(f"  {M}x{N}x{K}: {why}")
    return 0 if (failed == 0 and errored == 0) else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        help="tuned CSV to verify every shape (with its recorded config)",
    )
    ap.add_argument("--limit", type=int, default=None, help="only first N shapes")
    ap.add_argument(
        "--m-cap",
        type=int,
        default=None,
        help="cap M rows (same config) to bound runtime on very tall shapes",
    )
    ap.add_argument(
        "--fail-thr", type=float, default=0.01, help="mismatch-ratio fail threshold"
    )
    ap.add_argument(
        "--blockscale",
        action="store_true",
        help="verify the in-kernel blockscale path (a8w8, N%%128==0) per config",
    )
    args = ap.parse_args()

    if args.csv:
        raise SystemExit(
            _verify_tuned_csv(
                args.csv,
                limit=args.limit,
                m_cap=args.m_cap,
                fail_thr=args.fail_thr,
                blockscale=args.blockscale,
            )
        )

    for shp in _SHAPES:
        test_a4w4(*shp)
        test_a8w8(*shp)
        test_a8w8_blockscale(*shp)
        test_a8w8_blockscale_inkernel(*shp)
        print(f"OK {shp}")
    for shp in _SPLITK_SHAPES:
        test_a8w8_splitk(*shp)
        print(f"OK split-K {shp}")
