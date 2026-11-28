# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import functools
import hashlib
import json
import os

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_ASM_DIR, AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.gemm_op_a4w4 import _FLYDSL_AVAILABLE, gemm_a4w4
from aiter.ops.shuffle import shuffle_mxfp8fp4_scale, shuffle_weight
from aiter.test_common import (
    assertAllclose,
    benchmark,
    checkAllclose,
    perftest,
    run_perftest,
)
from aiter.utility import fp4_utils

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
SCALE_GROUP_SIZE = 32
SUPPORTED_GFX = ["gfx950", "gfx1250"]
# gfx1250 M the tuned table lacks, with whether the kernel each must reach is the compute
# one (every tuned compute row fills whole M tiles). 1 is below the smallest row; 17 and 257
# leave a partial M tile of a general kernel, 2049 one of the compute kernel. 129 and 255 are
# below the 256-row compute tile of N=36864 and reach it through the padded M, as one partial
# tile. Together they reach the masked-M stores and partial-tile cluster reduction of both
# kernel families. The last three leave 19 / 33 / 19 M tiles on compute kernels with cluster_m
# 2 / 4 / 2: cluster_m shrinks to the largest power-of-two divisor of the tile count (1 / 1 / 1).
OFF_GRID_SHAPES = {
    (1, 7168, 16384): False,
    (17, 7168, 16384): False,
    (257, 7168, 16384): False,
    (2049, 7168, 16384): True,
    (129, 36864, 7168): True,
    (255, 36864, 7168): True,
    (4609, 3584, 7168): True,
    (8321, 3584, 7168): True,
    (4609, 7168, 3584): True,
}
# (M, N, K, kernelName, other kernelName): two compute kernels that do not fit the instruction cache
# together. Each prefetches its own code, so it must run as fast right after the other as back to back:
# alternating vs back to back is 0.94-1.06 with the prefetch and 1.5-2.2 without. One launch holds three
# kernels, picked by the K-tiles left after the full revolutions (0, 2 or any count); the K below leave
# 0, 2 and 3. Other K add a data-side cost of up to 1.2 that the prefetch does not touch.
SWITCH_CASES = tuple(
    (
        4096,
        3584,
        k,
        "flydsl_a4w4_bpreshuffle_compute_wmma_t256x256x256_mw2_nw2_nb4_sk1_cm2_cn2",
        "flydsl_a4w4_bpreshuffle_compute_wmma_t256x256x256_mw2_nw2_nb4_sk1_cm4_cn2",
    )
    for k in (7168, 3584, 7936)
)
SWITCH_RATIO_MAX = 1.15
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 30)


def run_torch(x, w, x_scales, w_scales, dtype):
    m, _k = x.shape
    n, _k = w.shape
    # First convert the x and w inputs to f32.
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, out):
    return aiter.gemm_a4w4_blockscale(x, weight, x_scale, w_scale, out)


@perftest()
def run_triton(x, w, x_scales, w_scales, out, dtype=dtypes.bf16):
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    gemm_afp4wfp4(x, w, x_scales, w_scales, dtype, out)
    return out


@perftest()
def run_gemm_asm(
    x,
    weightshuffle,
    x_scale,
    w_scale,
    out,
    kernelName="",
    bias=None,
    dtype=dtypes.bf16,
    bpreshuffle=True,
    log2_k_split=None,
):
    # if log2_k_split is not None and log2_k_split > 0:
    #     out_reset = torch.zeros(
    #         (out.shape[0] + 31) // 32 * 32, out.shape[1], dtype=dtype
    #     )
    #     out = out_reset

    aiter.gemm_a4w4_asm(
        x,
        weightshuffle,
        x_scale,
        w_scale,
        out,
        kernelName,
        bias,
        bpreshuffle=bpreshuffle,
        log2_k_split=log2_k_split,
    )
    return out


PS_KERNEL = "_ZN5aiter45f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256_psE"


def ps_supported(x, w, x_scales, w_scales, out, tile=256):
    # mirrors ps_supported() in csrc/py_itfs_cu/asm_gemm_a4w4.cu (B preshuffled, no bias term)
    M, N, K = x.shape[0], w.shape[0], x.shape[1] * 2
    return (
        K % 64 == 0
        and N % 16 == 0
        and x.stride(0) % 16 == 0
        and w.stride(0) * 2 == K
        and x_scales.shape[0] >= (M + tile - 1) // tile * tile
        and w_scales.shape[0] >= (N + tile - 1) // tile * tile
        and M * x.stride(0) < 2**32
        and N * w.stride(0) < 2**32
        and tile * out.stride(0) * 2 < 2**31
    )


def quantize(dtype, M, N, K):
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    wshuffle = shuffle_weight(w, layout=(16, 16))
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)
    a = run_torch(x, w, x_scales, w_scales, dtype)
    return a, x, wshuffle, x_scales_shuffle, w_scales_shuffle


def _gfx1250_operands(dtype, M, N, K):
    """The operands of gemm_a4w4 as ATOM calls it on gfx1250: the weight and the
    activation scale in the 32x4 layout, the activation quantized straight into it.
    Returns xq, xs, wq, ws and the torch reference."""
    quant = aiter.get_hip_quant(aiter.QuantType.per_1x32)
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    # The row-major quant is the reference operand; every layout below holds these bytes.
    xq_raw, xs_raw = quant(x, quant_dtype=dtypes.fp4x2, shuffle=False)
    wq_raw, ws_raw = quant(w, quant_dtype=dtypes.fp4x2, shuffle=False)
    ref = run_torch(
        xq_raw, wq_raw, xs_raw.view(torch.uint8), ws_raw.view(torch.uint8), dtype
    )

    wq = shuffle_weight(wq_raw, layout=(16, 16))
    ws = shuffle_mxfp8fp4_scale(ws_raw)
    xq, xs = quant(x, quant_dtype=dtypes.fp4x2, scale_layout_m32k4=True)
    return xq, xs, wq, ws, ref


def _test_gemm_gfx1250(dtype, M, N, K):
    """gemm_a4w4 as ATOM calls it on gfx1250. ``tuned`` is the public op, so it runs
    the backend the tuned CSV (or the FlyDSL fallback) picks for this shape."""
    from aiter.ops.gemm_op_a4w4 import _a4w4_mxscale_config, gemm_mxfp4_asm

    xq, xs, wq, ws, ref = _gfx1250_operands(dtype, M, N, K)
    candidates = {
        "tuned": lambda: gemm_a4w4(xq, wq, xs, ws, dtype=dtype),
        "asm": lambda: gemm_mxfp4_asm(xq, wq, xs, ws, dtype=dtype, a_preshuffle=False),
    }
    config = _a4w4_mxscale_config(M, N, K) or ("asm", "")
    # An asm pick, or the other kernel family, would hide the coverage an off-grid M is for.
    # Without FlyDSL there is no family to check: both candidates still verify ASM.
    want_compute = OFF_GRID_SHAPES.get((M, N, K))
    if want_compute is not None and _FLYDSL_AVAILABLE:
        # Importing aiter.ops.flydsl needs flydsl itself, so only a FlyDSL run may import it.
        from aiter.ops.flydsl.a4w4_mxscale_bpreshuffle_gemm_gfx1250 import (
            parse_a4w4_kernel_name,
        )

        kernel = parse_a4w4_kernel_name(config[1]) if config[0] == "flydsl" else None
        assert kernel and kernel["compute_bound"] == want_compute, (
            f"M={M} N={N} K={K}: expected a FlyDSL "
            f"{'compute' if want_compute else 'general'} kernel, got {config[1] or 'asm'}"
        )
    ret = {
        "gfx": get_gfx(),
        "tuned kernel": config[1] if config[0] == "flydsl" else "asm",
    }
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = assertAllclose(
            ref, out, msg=f"{name}: a4w4 M={M} N={N} K={K}", catastrophic_check=True
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = M * N * K * 2 / us / 1e6
        ret[f"{name} TB/s"] = (
            (xq.nbytes + wq.nbytes + xs.nbytes + ws.nbytes + out.nbytes) / us / 1e6
        )
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_gemm(dtype, M, N, K):
    if get_gfx() == "gfx1250":
        return _test_gemm_gfx1250(dtype, M, N, K)
    ret = {}
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    wshuffle = shuffle_weight(w, layout=(16, 16))
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)
    a = run_torch(x, w, x_scales, w_scales, dtype)
    from aiter.ops.gemm_op_a4w4 import _a4w4_fallback_config, get_GEMM_config

    out_shape = ((M + 31) // 32 * 32, N)
    candidates = {
        "tuned": lambda: gemm_a4w4(
            x, wshuffle, x_scales_shuffle, w_scales_shuffle, dtype=dtype
        ),
    }
    config = get_GEMM_config(M, N, K)
    ret = {
        "gfx": get_gfx(),
        "configured kernel": config["kernelName"] if config else "asm",
    }
    if config and str(config["kernelName"]).startswith(("triton_", "flydsl_")):
        fallback = _a4w4_fallback_config(config)
        if fallback["kernelName"] and not fallback["kernelName"].startswith("_ZN"):
            candidates["baseline"] = lambda: aiter.gemm_a4w4_blockscale(
                x,
                wshuffle,
                x_scales_shuffle,
                w_scales_shuffle,
                torch.empty(out_shape, dtype=dtype, device=x.device),
                kernelName=fallback["kernelName"],
                splitK=fallback["splitK"],
            )[:M]
        else:
            candidates["baseline"] = lambda: aiter.gemm_a4w4_asm(
                x,
                wshuffle,
                x_scales_shuffle,
                w_scales_shuffle,
                torch.empty(out_shape, dtype=dtype, device=x.device),
                fallback["kernelName"],
                bpreshuffle=True,
                log2_k_split=fallback["splitK"],
            )[:M]
    for name, fn in candidates.items():
        result, us = run_perftest(fn)
        err = checkAllclose(
            a.float(), result.float(), msg=f"{name}: a4w4", catastrophic_check=True
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = M * N * K * 2 / us / 1e6
        ret[f"{name} TB/s"] = (
            (
                x.nbytes
                + w.nbytes
                + x_scales_shuffle.nbytes
                + w_scales_shuffle.nbytes
                + result.nbytes
            )
            / us
            / 1e6
        )
        ret[f"{name} err"] = err

    out = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    if ps_supported(x, wshuffle, x_scales_shuffle, w_scales_shuffle, out):
        d, us = run_gemm_asm(
            x, wshuffle, x_scales_shuffle, w_scales_shuffle, out, PS_KERNEL
        )
        err = checkAllclose(a, d[:M], msg="asm 256x256 ps", catastrophic_check=True)
        ret["us 256x256_ps"] = us
        ret["err 256x256_ps"] = err
        # a bias with beta 0 adds nothing
        bias = torch.randn(M, N, dtype=dtypes.fp32)
        d, _ = run_gemm_asm(
            x, wshuffle, x_scales_shuffle, w_scales_shuffle, out, PS_KERNEL, bias
        )
        ret["err 256x256_ps bias"] = checkAllclose(
            a, d[:M], msg="asm 256x256 ps, bias, beta 0", catastrophic_check=True
        )

    return ret


def _asm_f4_kernels():
    """Rows of this arch's asm kernel table, the table the launcher reads,
    that take preshuffled B and are not persistent (ps=0)."""
    path = os.path.join(
        AITER_ASM_DIR, get_gfx(), "f4gemm", "f4gemm_bf16_per1x32Fp4.csv"
    )
    if not os.path.isfile(path):
        return []
    table = pd.read_csv(path, comment="#")
    rows = table[(table["bpreshuffle"] == 1) & (table["ps"] == 0)]
    return list(rows.itertuples(index=False))


def _quant_rows(rows, k, chunk=65536):
    """Random MXFP4 rows, quantized in chunks so no bf16 copy of the whole
    tensor is needed. Returns data, plain scales and shuffled scales."""
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    q, s, s_sh = [], [], []
    for r0 in range(0, rows, chunk):
        t = torch.randn((min(chunk, rows - r0), k), dtype=dtypes.bf16)
        _, scales = quant_func(t, shuffle=False)
        tq, scales_sh = quant_func(t, shuffle=True)
        q.append(tq)
        s.append(scales.view(torch.uint8)[: t.shape[0]])
        s_sh.append(scales_sh[: t.shape[0]])
    return torch.cat(q), torch.cat(s), torch.cat(s_sh)


def _check_asm_f4_exact(
    x, w, x_scales, w_scales, x_scales_sh, w_scales_sh, out, name, case
):
    """Run one asm kernel by name; every element of out must match the
    reference."""
    out.fill_(float("nan"))
    wshuffle = shuffle_weight(w, layout=(16, 16))
    aiter.gemm_a4w4_asm(
        x, wshuffle, x_scales_sh, w_scales_sh, out, name, log2_k_split=0
    )
    m, n = out.shape
    rb = max(1, (1 << 28) // max(n, x.shape[1] * 2))
    bad = 0
    for r0 in range(0, m, rb):
        rows = slice(r0, r0 + rb)
        ref = run_torch(x[rows], w, x_scales[rows], w_scales, dtypes.bf16).float()
        got = out[rows].float()
        bad += int((~torch.isclose(got, ref, rtol=1e-2, atol=1e-2)).sum())
    assert bad == 0, f"a4w4 asm {name} {case}: {bad} of {m * n} elements wrong"
    aiter.logger.info(f"a4w4 asm {name} {case}: exact")


def check_asm_4gib_operands():
    """A or out of 4 GiB or more must be exact, for every kernel that
    _asm_f4_kernels() selects.

    Each operand's buffer descriptor holds a 32-bit size; the kernels give
    every tile its own 64-bit base.
    """
    kernels = _asm_f4_kernels()
    for operand, m, n, k in (
        ("out", 20192, 106496, 512),
        ("A", (1 << 33) // 8192 + 256, 512, 8192),
    ):
        x, x_scales, x_scales_sh = _quant_rows(m, k)
        w, w_scales, w_scales_sh = _quant_rows(n, k)
        out = torch.empty((m, n), dtype=dtypes.bf16)
        case = f"{operand} > 4 GiB {(m, n, k)}"
        for kn in kernels:
            _check_asm_f4_exact(
                x,
                w,
                x_scales,
                w_scales,
                x_scales_sh,
                w_scales_sh,
                out,
                kn.knl_name,
                case,
            )
        del x, w, out


def check_asm_last_tile_at_allocation_end():
    """A ends an allocation and the last M tile has 1 row, at small K.

    The kernels lower each tile's descriptor size by a fixed step per K step
    they prefetch, also past the end of K; it must stop at zero there. If it
    wrapped, those prefetches would read past the end of A, which faults
    (aborting the process) when A ends the allocation.
    """
    # A fresh 64 MiB block is a whole caching-allocator segment, so the memory
    # after it is normally unmapped; empty the cache so it is not cut from a
    # larger one.
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    torch.cuda.empty_cache()
    buf = torch.empty(64 << 20, dtype=torch.uint8)
    n = 512
    for kn in _asm_f4_kernels():
        m = kn.tile_M + 1
        for k in (256, 512):
            xb = torch.randn((m, k), dtype=dtypes.bf16)
            wb = torch.randn((n, k), dtype=dtypes.bf16)
            x, x_scales_sh = quant_func(xb, shuffle=True)
            w, w_scales_sh = quant_func(wb, shuffle=True)
            x_scales = quant_func(xb, shuffle=False)[1].view(torch.uint8)
            w_scales = quant_func(wb, shuffle=False)[1].view(torch.uint8)
            nbytes = x.numel() * x.element_size()
            xa = buf[buf.numel() - nbytes :].view(x.dtype).view(x.shape)
            xa.copy_(x)
            out = torch.empty((m, n), dtype=dtypes.bf16)
            case = f"A at allocation end {(m, n, k)}"
            _check_asm_f4_exact(
                xa,
                w,
                x_scales,
                w_scales,
                x_scales_sh,
                w_scales_sh,
                out,
                kn.knl_name,
                case,
            )


@benchmark()
def test_gemm_splitk(dtype, M, N, K, log2_k_split):
    ret = {}
    a, x, wshuffle, x_scales_shuffle, w_scales_shuffle = quantize(dtype, M, N, K)
    for name in ["_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E", PS_KERNEL]:
        tag = "256x256_ps" if name == PS_KERNEL else "256x256"
        out = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
        d, us = run_gemm_asm(
            x,
            wshuffle,
            x_scales_shuffle,
            w_scales_shuffle,
            out,
            name,
            log2_k_split=log2_k_split,
        )
        err = checkAllclose(
            a,
            d[:M],
            msg=f"asm {tag} log2_k_split={log2_k_split}",
            catastrophic_check=True,
        )
        ret[f"us {tag}"] = us
        ret[f"err {tag}"] = err
    return ret


@benchmark()
def test_compute_switch_penalty(
    M, N, K, kernel_name, other_kernel_name, dtype=dtypes.bf16
):
    """Two compute kernels run alternately must cost what they cost back to back,
    and each must still produce its own correct result."""
    from aiter.ops.flydsl.a4w4_mxscale_bpreshuffle_gemm_gfx1250 import (
        run_gemm_a4w4_mxscale_bpreshuffle_gfx1250,
    )

    xq, xs, wq, ws, ref = _gfx1250_operands(dtype, M, N, K)
    out = torch.empty((M, N), dtype=dtype)
    names = (kernel_name, other_kernel_name)
    fns = [
        functools.partial(
            run_gemm_a4w4_mxscale_bpreshuffle_gfx1250, xq, wq, xs, ws, out, name
        )
        for name in names
    ]
    # The alternating run overwrites the output, so each kernel is checked on its own first.
    for name, fn in zip(names, fns):
        out.fill_(float("nan"))
        fn()
        assertAllclose(
            ref, out, msg=f"{name}: a4w4 M={M} N={N} K={K}", catastrophic_check=True
        )

    def alternate():
        fns[0]()
        fns[1]()

    _, us_a = run_perftest(fns[0])
    _, us_b = run_perftest(fns[1])
    _, us_ab = run_perftest(alternate)
    assert us_ab <= SWITCH_RATIO_MAX * (us_a + us_b), (
        f"{kernel_name} then {other_kernel_name}: {us_ab:.1f} us alternating vs "
        f"{us_a:.1f} + {us_b:.1f} us back to back"
    )
    return {
        "gfx": get_gfx(),
        "compute us": us_a,
        "compute TFLOPS": 2 * M * N * K / us_a / 1e6,
        "other compute us": us_b,
        "alternating us": us_ab,
    }


# ---- gfx950 OPUS a4w4, layout="plain": row-major inputs (Triton gemm_afp4wfp4) ----
# One case per kid class of the a4w4 plain family, each on M/N tails (M not a tile
# multiple, N % 8 but not % 32) unless the kid needs aligned M/N, through every entry:
# opus_gemm (2D), opus_bmm (3D scales), gemm_afp4wfp4_opus (batched, 2D scales with
# rows > M so the per-batch scale stride is not M * K/32).
#   (entry, batch, M, N, K, kid, split_k)
PLAIN_KID_CASES = [
    *(("opus_gemm", 1, 1000, 1032, 1024, kid, 0) for kid in range(100, 106)),
    *(("opus_gemm", 1, 1000, 1032, 1024, kid, 0) for kid in range(110, 116)),
    ("opus_gemm", 1, 512, 512, 1024, 130, 0),  # nob: M, N multiples of 256
    *(("opus_gemm", 1, 33, 264, 2048, kid, 0) for kid in range(500, 505)),
    ("opus_gemm", 1, 7, 264, 4096, 500, 4),  # split-K through the fp32 workspace
    ("opus_gemm", 1, 50, 136, 2048, 503, 2),
    ("opus_bmm", 3, 200, 136, 512, 102, 0),
    ("opus_bmm", 2, 40, 520, 1024, 501, 2),
    ("gemm_afp4wfp4_opus", 2, 300, 264, 512, 112, 0),
    ("gemm_afp4wfp4_opus", 3, 17, 72, 1024, 504, 3),
    # Odd K-tile counts >= 3 (K / 256 = 3, 5): the 2x2 pipeline's extra even step and
    # its last, half-used scale panel; the small-tile ring's half-filled last step.
    *(
        ("opus_gemm", 1, 1000, 1032, k, kid, 0)
        for k in (768, 1280)
        for kid in (*range(100, 106), *range(110, 116))
    ),
    *(("opus_gemm", 1, 512, 512, k, 130, 0) for k in (768, 1280)),
    # More tiles than CUs (256): a persistent block runs several tiles, so the next
    # tile's K tile 0 / scale panel 0 are prefetched behind an odd K-tile count.
    ("opus_gemm", 1, 4200, 4104, 768, 100, 0),
    ("opus_gemm", 1, 2200, 2056, 1280, 103, 0),
    ("opus_gemm", 1, 4352, 4352, 768, 130, 0),
    *(
        ("opus_gemm", 1, 33, 264, k, kid, 0)
        for k in (768, 1280)
        for kid in range(500, 505)
    ),
    ("opus_gemm", 1, 7, 264, 1280, 500, 2),
    ("opus_bmm", 3, 200, 136, 768, 102, 0),
    ("gemm_afp4wfp4_opus", 2, 300, 264, 1280, 112, 0),
]
PLAIN_SCALE_PAD = 3  # extra scale rows per batch for the 2D batched scales


def _quant_plain(B, rows, K, dtype, pad=0):
    """Row-major fp4x2 [B, rows, K/2] and e8m0 scales [B, rows + pad, K/32]."""
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    q, sc = quant_func(torch.randn((B * rows, K), dtype=dtype), shuffle=False)
    sc = sc.view(torch.uint8).view(B, rows, K // SCALE_GROUP_SIZE)
    if pad:
        filler = torch.full((B, pad, K // SCALE_GROUP_SIZE), 127, dtype=torch.uint8)
        sc = torch.cat([sc, filler], dim=1)
    return q.view(dtypes.fp4x2).view(B, rows, K // 2), sc.contiguous()


def _ref_plain(xq, wq, xs, ws, dtype):
    return torch.stack(
        [run_torch(xq[b], wq[b], xs[b], ws[b], dtype) for b in range(xq.shape[0])]
    )


def _check_plain(name, ref, out):
    assert not out.isnan().any(), f"{name}: unwritten (NaN) output elements"
    err = checkAllclose(
        ref.float(), out.float(), msg=f"{name}: a4w4 plain", catastrophic_check=True
    )
    # No allowance: the M / N tails these cases exist for are under 1% of the elements.
    assert err == 0, f"{name}: {err:.2%} of elements outside rtol/atol"
    return err


@benchmark()
def test_gemm_plain_kid(entry, B, M, N, K, kid, split_k):
    """One exact a4w4 plain kid through one public entry, against the torch reference."""
    from aiter.ops.opus import opus_bmm, opus_gemm
    from aiter.ops.opus.gemm_op_a4w4 import gemm_afp4wfp4_opus

    dtype = dtypes.bf16
    pad = PLAIN_SCALE_PAD if entry == "gemm_afp4wfp4_opus" else 0
    xq, xs = _quant_plain(B, M, K, dtype, pad)
    wq, ws = _quant_plain(B, N, K, dtype, pad)
    ref = _ref_plain(xq, wq, xs, ws, dtype)
    # NaN sentinel: an output element the kernel never writes fails the check.
    y = torch.full((B, M, N), float("nan"), dtype=dtype)
    kw = {"kid": kid, "family": "a4w4", "layout": "plain", "split_k": split_k}
    if entry == "opus_gemm":
        fn = lambda: opus_gemm(xq[0], wq[0], y[0], x_scale=xs[0], w_scale=ws[0], **kw)
    elif entry == "opus_bmm":
        fn = lambda: opus_bmm(xq, wq, y, x_scale=xs, w_scale=ws, **kw)
    else:
        fn = lambda: gemm_afp4wfp4_opus(
            xq,
            wq,
            xs.view(B * (M + pad), -1),
            ws.view(B * (N + pad), -1),
            y=y,
            kernelId=kid,
            splitK=split_k,
        )
    _, us = run_perftest(fn)
    err = _check_plain(f"{entry} kid {kid}", ref, y)
    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": B * M * N * K * 2 / us / 1e6,
        "err": err,
    }


@benchmark()
def test_gemm_plain(dtype, M, N, K):
    """Triton gemm_afp4wfp4 and its OPUS counterpart (heuristic kid) on row-major inputs."""
    from aiter.ops.opus.gemm_op_a4w4 import gemm_afp4wfp4_opus
    from aiter.ops.opus.policy import a4w4_heuristic_kid
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    xq, xs = _quant_plain(1, M, K, dtype)
    wq, ws = _quant_plain(1, N, K, dtype)
    ref = _ref_plain(xq, wq, xs, ws, dtype)[0]
    x, w, x_scales, w_scales = xq[0], wq[0], xs[0], ws[0]
    # Triton takes the fp4x2 operands as uint8 (older Triton has no float4 dtype).
    x8, w8 = x.view(torch.uint8), w.view(torch.uint8)
    candidates = {
        "triton": lambda: gemm_afp4wfp4(x8, w8, x_scales, w_scales, dtype),
        "opus": lambda: gemm_afp4wfp4_opus(x, w, x_scales, w_scales, dtype),
    }
    nbytes = x.nbytes + w.nbytes + x_scales.nbytes + w_scales.nbytes + M * N * 2
    ret = {
        "gfx": get_gfx(),
        "opus kid": a4w4_heuristic_kid(M, N, K, 1, gfx=get_gfx(), cu_num=get_cu_num()),
    }
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = M * N * K * 2 / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = _check_plain(name, ref, out)
    return ret


def check_plain_large_offset(kid=100, M=65792, N=32768, K=256):
    """A pre-compiled kid on a Y of > 2^31 elements: (M - 256) * N = 2^31, so the
    last tile row's offset overflows 32-bit arithmetic. Only the first and last
    rows are compared against the reference (the full one would need 8 GiB)."""
    from aiter.ops.opus import opus_gemm

    dtype = dtypes.bf16
    xq, xs = _quant_plain(1, M, K, dtype)
    wq, ws = _quant_plain(1, N, K, dtype)
    y = torch.full((M, N), float("nan"), dtype=dtype)
    opus_gemm(xq[0], wq[0], y, kid=kid, family="a4w4", x_scale=xs[0], w_scale=ws[0])
    for rows in (slice(0, 4), slice(M - 4, M)):
        ref = run_torch(xq[0][rows], wq[0], xs[0][rows], ws[0], dtype)
        _check_plain(f"kid {kid} rows {rows.start}:{rows.stop} of M={M}", ref, y[rows])
    del y


def check_plain_co_images():
    """The committed a4w4 .co images are the ones gen_co/gfx950/a4w4/build_info.json
    records: same kids and entry symbols as co_kernels_gfx950_a4w4.json, same sha256."""
    from csrc.opus_gemm.opus_gemm_common import (
        A4W4_CO_DIR,
        A4W4_CO_JSON,
        a4w4_co_kernels_declared,
        a4w4_co_kernels_list,
    )

    image_dir = os.path.join(os.path.dirname(A4W4_CO_JSON), A4W4_CO_DIR)
    with open(os.path.join(image_dir, "build_info.json")) as f:
        built = {entry["kid"]: entry for entry in json.load(f)["kernels"]}
    declared = {kid: k.name for kid, k in a4w4_co_kernels_declared.items()}
    assert {kid: entry["symbol"] for kid, entry in built.items()} == declared
    assert set(a4w4_co_kernels_list) == set(declared), "a declared .co image is missing"
    for kid, entry in built.items():
        with open(os.path.join(image_dir, f"{entry['symbol']}.co"), "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        assert sha256 == entry["co_sha256"], (
            f"kid {kid}: {entry['symbol']}.co does not match build_info.json; "
            "rebuild with gen_co/build_co.py --arch gfx950 --family a4w4"
        )


def tuned_shapes():
    """CSV-selected preshuffle shapes on gfx950; all tuned shapes on gfx1250."""
    gfx950 = get_gfx() == "gfx950"
    try:
        table = pd.read_csv(
            AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_FILE
            if gfx950
            else AITER_CONFIGS.AITER_CONFIG_GEMM_A4W4_MXSCALE_BPRESHUFFLE_FILE
        )
    except OSError:
        return []  # no tuned shapes on this install
    if "gfx" not in table.columns:
        return []  # legacy overrides have no qualified per-architecture rows
    table = table[(table["gfx"] == get_gfx()) & (table["cu_num"] == get_cu_num())]
    if gfx950:
        table = table[
            table["kernelName"].str.startswith(("triton_", "flydsl_"), na=False)
        ]
    return sorted(set(table[["M", "N", "K"]].itertuples(index=False, name=None)))


def main():
    # Whole-op arch gate: gate here, not inside the @benchmark fn (that would emit an
    # args-only row).
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("a4w4 GEMM unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        choices=[dtypes.d_dtypes["bf16"]],
        metavar="{bf16}",
        default=[dtypes.d_dtypes["bf16"]],
        help="""Data type.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-mnk",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            # pure_compute
            (256, 2048, 8192),
            (2048, 8192, 8192),
            (16384, 16384, 16384),
            (32768, 106496, 16384),
            (32768, 16384, 53248),
            (32768, 18432, 16384),
            (32768, 16384, 16384),
            (128, 106496, 16384),
            (128, 16384, 53248),
            (128, 18432, 16384),
            (128, 16384, 16384),
            (64, 106496, 16384),
            (64, 16384, 53248),
            (64, 18432, 16384),
            (64, 16384, 16384),
            (64, 106496, 16384),
            (32, 106496, 16384),
            (32, 16384, 53248),
            (32, 18432, 16384),
            (32, 16384, 16384),
            # qkv_proj
            (1, 1280, 8192),
            (64, 1280, 8192),
            (127, 1280, 8192),
            (129, 1280, 8192),
            (65, 1280, 8192),
            (32, 1280, 8192),
            (64, 1280, 8192),
            (128, 1280, 8192),
            (192, 1280, 8192),
            (256, 1280, 8192),
            (320, 1280, 8192),
            (512, 1280, 8192),
            (1024, 1280, 8192),
            (2048, 1280, 8192),
            (4096, 1280, 8192),
            (8192, 1280, 8192),
            # attn_out
            (1, 8192, 1024),
            (32, 8192, 1024),
            (64, 8192, 1024),
            (128, 8192, 1024),
            (192, 8192, 1024),
            (256, 8192, 1024),
            (320, 8192, 1024),
            (512, 8192, 1024),
            (1024, 8192, 1024),
            (2048, 8192, 1024),
            (4096, 8192, 1024),
            (8192, 8192, 1024),
            (16384, 8192, 1024),
            # tune
            (1552, 8192, 8192),
            (1664, 8192, 8192),
            (1792, 8192, 8192),
            (1920, 8192, 8192),
            (3072, 8192, 8192),
            (1552, 10240, 8192),
            (1664, 10240, 8192),
            (1792, 10240, 8192),
            (1920, 10240, 8192),
            (3072, 10240, 8192),
            (1552, 57344, 8192),
            (1664, 57344, 8192),
            (1792, 57344, 8192),
            (1920, 57344, 8192),
            (3072, 57344, 8192),
            (1552, 8192, 28672),
            (1664, 8192, 28672),
            (1792, 8192, 28672),
            (1920, 8192, 28672),
            (3072, 8192, 28672),
        ],
        help="""Shape of mnk.
    e.g. -mnk 1280,8192,1024""",
    )
    parser.add_argument(
        "-pmnk",
        "--plain_shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (m, n, k)
            for n, k in ((2048, 7168), (3072, 6144), (4096, 4096), (4096, 8192))
            for m in (1, 8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
        ],
        help="""gfx950: shapes of the row-major (layout="plain") sweep, OPUS vs Triton
    gemm_afp4wfp4.
    e.g. -pmnk 1024,4096,4096""",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="""CSV file containing M, N, K columns (one shape per row); overrides -mnk.
    On gfx1250, with neither -mnk nor --csv, every shape of the tuned CSV runs, plus the
    off-grid M it lacks.
    e.g.: --csv aiter/configs/model_configs/dsr1_a4w4_mxscale_bpreshuffle_tuned_gemm.csv""",
    )
    parser.add_argument(
        "-mnk_splitk",
        "--shape_splitk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (256, 256, 8192),
            (128, 128, 16384),
            (256, 256, 16384),
            (256, 512, 8192),
            (512, 512, 8192),
            (2048, 2048, 8192),
        ],
        help="""Shapes of mnk for split K (log2_k_split 1..3) with the 256x256 kernels.
    e.g. -mnk_splitk 256,256,8192""",
    )
    args = parser.parse_args()

    gfx1250 = get_gfx() == "gfx1250"
    if gfx1250 and _FLYDSL_AVAILABLE:
        switch_rows = [test_compute_switch_penalty(*case) for case in SWITCH_CASES]
        aiter.logger.info(
            "a4w4 compute kernels run alternately vs back to back (markdown):\n%s",
            pd.DataFrame(switch_rows).to_markdown(index=False),
        )
    if args.csv is not None:
        if not os.path.exists(args.csv):
            raise FileNotFoundError(f"CSV file not found: {args.csv}")
        shapes_df = pd.read_csv(args.csv)
        print(f"Loaded {len(shapes_df)} shapes from {args.csv}", flush=True)
        shapes = [
            (int(m), int(n), int(k))
            for m, n, k in shapes_df[["M", "N", "K"]].itertuples(index=False)
        ]
    elif gfx1250 and args.shape is parser.get_default("shape"):
        # The built-in -mnk list is gfx950's; here the tuned table's shapes run.
        shapes = tuned_shapes()
        if not shapes:
            aiter.logger.warning(
                "no tuned a4w4 shapes for gfx1250 cu_num=%s; pass -mnk or --csv",
                get_cu_num(),
            )
            return
        shapes.extend(OFF_GRID_SHAPES)
    else:
        shapes = args.shape
        if not gfx1250 and args.shape is parser.get_default("shape"):
            shapes = list(dict.fromkeys([*shapes, *tuned_shapes()]))
    df = []
    for dtype in args.dtype:
        for m, n, k in shapes:
            ret = test_gemm(dtype, m, n, k)
            df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("gemm_a4w4 summary (markdown):\n%s", df_md)
    if (
        get_gfx() == "gfx950"
        and args.csv is None
        and args.shape is parser.get_default("shape")
    ):
        check_asm_last_tile_at_allocation_end()
        check_asm_4gib_operands()

    if gfx1250:
        return  # the split-K test runs the gfx950 256x256 kernels
    df = []
    for dtype in args.dtype:
        for m, n, k in args.shape_splitk:
            for log2_k_split in (1, 2, 3):
                df.append(test_gemm_splitk(dtype, m, n, k, log2_k_split))
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("gemm_a4w4 split K summary (markdown):\n%s", df_md)

    if get_gfx() == "gfx950":  # the OPUS a4w4 plain family is gfx950-only
        check_plain_co_images()
        df = pd.DataFrame([test_gemm_plain_kid(*case) for case in PLAIN_KID_CASES])
        aiter.logger.info(
            "gemm_a4w4 plain kids summary (markdown):\n%s", df.to_markdown(index=False)
        )
        check_plain_large_offset()
        df = pd.DataFrame(
            [
                test_gemm_plain(dtype, m, n, k)
                for dtype in args.dtype
                for m, n, k in args.plain_shape
            ]
        )
        aiter.logger.info(
            "gemm_a4w4 plain summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
