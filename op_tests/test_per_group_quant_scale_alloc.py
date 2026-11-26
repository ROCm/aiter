# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scale allocation of the MX-shuffled group-32 e8m0 quant.

``per_group_quant_hip(..., group_size=32, scale_type=fp8_e8m0, transpose_scale=True)``
writes the scale in 32-row x 8-group tiles (``mx_scale_shuffle_idx``), so the kernel
touches up to ``pad32(rows) * pad8(K // 32)`` bytes. The public op used to allocate
only ``rows * K // 32`` and the kernel wrote past the end of it whenever
``rows % 32 != 0`` (or ``K // 32 % 8 != 0``). The op now allocates the padded
buffer and returns the ``(*shape[:-1], K // 32)`` view of it; the kernel rejects a
shorter buffer.
"""

import argparse
import itertools
import resource
import subprocess
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.quant import dynamic_per_group_scaled_quant, per_group_quant_hip
from aiter.test_common import benchmark, run_perftest
from aiter.utility.fp4_utils import f32_to_mx_e8m0_scale
from aiter.utility.mx_types import MxDtypeInt

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]
GUARD = 8192  # larger than any overrun of the shapes below (about 6 KB at K=6144)
GUARD_BYTE = 0xAB
G = 32


def pad(v, a):
    return (v + a - 1) // a * a


def mx32_indices(rows, scale_n):
    """Flat byte index of every (row, group) in the MX-shuffled scale."""
    p = pad(scale_n, 8)
    x = torch.arange(rows)[:, None]
    y = torch.arange(scale_n)[None, :]
    return (
        (x // 32 * p) * 32
        + (y // 8) * 256
        + (y % 4) * 64
        + (x % 16) * 4
        + (y % 8) // 4 * 2
        + (x % 32) // 16
    )


def run_torch(x):
    """Reference scale bytes [rows, K // 32] and fp8 codes of MXFP8 1x32 quant."""
    m, k = x.shape
    xf = x.float().view(m, k // G, G)
    fnuz = get_gfx_runtime() == "gfx942"
    mx = MxDtypeInt.FP8_E4M3_FNUZ if fnuz else MxDtypeInt.FP8_E4M3
    amax = xf.abs().amax(-1).clamp(min=1e-10)
    ref_scale = f32_to_mx_e8m0_scale(amax, dtype=mx).view(torch.uint8)
    xs = xf / (ref_scale.to(torch.int32) - 127).float().exp2()[..., None]
    fp8 = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8).max
    codes = xs.clamp(-fp8_max, fp8_max).to(fp8).view(torch.uint8).view(m, k)
    return codes, ref_scale


def storage_bytes(t):
    """The whole allocation behind ``t`` as a flat uint8 tensor."""
    st = t.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(
        st, t.storage_offset(), (st.nbytes() - t.storage_offset(),)
    )


def guarded_kernel_call(x, numel):
    """Direct kernel call with `numel` scale bytes followed by a canary tail."""
    m, k = x.shape
    y = torch.empty(m, k, dtype=dtypes.fp8)
    buf = torch.full((numel + GUARD,), GUARD_BYTE, dtype=torch.uint8)
    dynamic_per_group_scaled_quant(
        y, x, buf[:numel].view(dtypes.fp8_e8m0), G, shuffle_scale=True
    )
    torch.cuda.synchronize()
    return y, buf


_CHILD = """
import sys, torch
from aiter import dtypes
from aiter.ops.quant import dynamic_per_group_scaled_quant
m, k, dt = int(sys.argv[1]), int(sys.argv[2]), getattr(torch, sys.argv[3])
x = torch.randn(m, k, dtype=dt, device="cuda")
y = torch.empty(m, k, dtype=dtypes.fp8, device="cuda")
s = torch.empty(m * (k // 32), dtype=torch.uint8, device="cuda").view(dtypes.fp8_e8m0)
dynamic_per_group_scaled_quant(y, x, s, 32, shuffle_scale=True)
torch.cuda.synchronize()
"""


def short_buffer_aborts(m, k, h_dtype):
    """True when the kernel refuses (aborts on) a rows * K / 32 byte scale."""
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, str(m), str(k), str(h_dtype).split(".")[-1]],
        check=False,
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)),
        capture_output=True,
        text=True,
    )
    return r.returncode != 0 and "pad32(rows) * pad8(cols / 32)" in r.stdout + r.stderr


@benchmark()
def test_scale_alloc(m, k, h_dtype):
    torch.manual_seed(m * 10007 + k)
    x = torch.randn((m, k), dtype=h_dtype)
    scale_n = k // G
    need = pad(m, 32) * pad(scale_n, 8)

    # 1. The public op: shape, dtype, layout and backing size.
    (y, s), us = run_perftest(
        per_group_quant_hip,
        x,
        quant_dtype=dtypes.fp8,
        group_size=G,
        transpose_scale=True,
        scale_type=dtypes.fp8_e8m0,
    )
    assert s.shape == (m, scale_n) and s.dtype == dtypes.fp8_e8m0, (s.shape, s.dtype)
    assert s.is_contiguous() and s.stride() == (scale_n, 1), s.stride()
    have = s.untyped_storage().nbytes() - s.storage_offset()
    assert have >= need, f"scale backed by {have} bytes, the kernel writes up to {need}"

    # 2. The kernel on exactly the padded size stays inside it ...
    y2, buf = guarded_kernel_call(x, need)
    assert torch.all(buf[need:] == GUARD_BYTE).item(), f"{m=} {k=}: wrote past {need}"

    # 3. ... and the public op's bytes are the kernel's, and the torch reference's.
    ref_codes, ref_scale = run_torch(x)
    idx = mx32_indices(m, scale_n)
    got_pub = storage_bytes(s)[idx]
    got_ker = buf[:need][idx]
    assert torch.equal(got_pub, got_ker), f"{m=} {k=}: public scale != kernel scale"
    assert torch.equal(got_pub, ref_scale), f"{m=} {k=}: scale != torch reference"
    assert torch.equal(y.view(torch.uint8), y2.view(torch.uint8)), "codes differ"
    assert torch.equal(y.view(torch.uint8), ref_codes), f"{m=} {k=}: codes != torch"

    # 4. A caller that hands the kernel a tight rows * K / 32 buffer must be refused
    # rather than overrun, whenever that is too small. AITER_CHECK aborts the process,
    # so the call runs in a child (without a core file).
    if m * scale_n < need:
        assert short_buffer_aborts(m, k, h_dtype), f"{m=} {k=}: short scale accepted"
    return {"gfx": get_gfx_runtime(), "extra bytes": need - m * scale_n, "us": us}


@benchmark()
def test_scale_alloc_nd(shape, h_dtype):
    """A batched input: rows are the flattened leading dims."""
    x = torch.randn(shape, dtype=h_dtype)
    k = shape[-1]
    m = x.numel() // k
    y, s = per_group_quant_hip(
        x,
        quant_dtype=dtypes.fp8,
        group_size=G,
        transpose_scale=True,
        scale_type=dtypes.fp8_e8m0,
    )
    assert s.shape == (*shape[:-1], k // G) and y.shape == x.shape
    assert s.untyped_storage().nbytes() >= pad(m, 32) * pad(k // G, 8)
    ref_codes, ref_scale = run_torch(x.view(m, k))
    got = storage_bytes(s)[mx32_indices(m, k // G)]
    assert torch.equal(got, ref_scale) and torch.equal(
        y.view(torch.uint8).view(m, k), ref_codes
    )
    return {"gfx": get_gfx_runtime(), "rows": m}


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("quant scale alloc unsupported on %s", get_gfx_runtime())
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
        default=[dtypes.d_dtypes["bf16"]],
        help="Data type. e.g.: -d bf16",
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="*",
        default=[1, 5, 8, 16, 17, 24, 33, 40, 48, 64, 96, 100, 128],
        help="Rows. e.g.: -m 33",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        nargs="*",
        # 288 and 160 are 9 and 5 groups: the column count is not a multiple of 8, so the scale is
        # padded in columns as well as rows (at M=64 in columns only)
        default=[2048, 6144, 2304, 1536, 288, 160],
        help="Columns, a multiple of 32. e.g.: -k 2048",
    )
    args = parser.parse_args()

    for h_dtype in args.dtype:
        df = [
            test_scale_alloc(m, k, h_dtype)
            for m, k in itertools.product(args.m, args.k)
        ]
        aiter.logger.info(
            "quant scale alloc summary (markdown):\n%s",
            pd.DataFrame(df).to_markdown(index=False),
        )
        df = [
            test_scale_alloc_nd(shape, h_dtype)
            for shape in [
                (2, 5, 2048),
                (3, 11, 2304),
                (4, 8, 1536),
                (2, 32, 288),
                (3, 7, 160),
            ]
        ]
        aiter.logger.info(
            "quant scale alloc nd summary (markdown):\n%s",
            pd.DataFrame(df).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
