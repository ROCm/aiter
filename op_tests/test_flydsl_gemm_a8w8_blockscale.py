# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and resources of the FlyDSL 8-wave FP8 blockscale GEMM (gfx950).

Calls the kernel directly, without tuned-CSV dispatch. The correctness cases
cover K=256, partial M and N tiles, N below one tile and plain B's N % 16 == 8,
each in at least one B layout and output dtype. C sits at the front of a
sentinel-filled buffer, so the test also catches unwritten elements and stores
past the end of C. One launcher is reused across M, as the M-free compile
cache does in serving. The resource test compiles without a GPU and requires
zero scratch at the 8-wave register budget.

Run:
    python op_tests/test_flydsl_gemm_a8w8_blockscale.py
"""

import os
import re
import subprocess
import sys
import textwrap

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.gemm_a8w8_blockscale import run_gemm_a8w8_blockscale
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_common import (
    VALIDATED_CU_NUM,
    kernel_fits_shape,
    kernelInstance,
)
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_gfx_runtime() != "gfx950"
    or torch.cuda.get_device_properties(0).multi_processor_count != VALIDATED_CU_NUM,
    reason="FlyDSL blockscale GEMM requires a 256-CU gfx950 (MI350X/MI355X)",
)

BF16, FP16 = torch.bfloat16, torch.float16

# (M, N, K, preshuffle_b, out dtype). The tile is 256x256x128.
CASES = [
    (256, 512, 256, False, BF16),  # whole tiles, single-iteration K loop
    (256, 512, 256, True, FP16),
    (257, 576, 512, False, BF16),  # one-row M tail, 64-column N tail
    (257, 576, 512, True, BF16),
    (1, 208, 1024, True, BF16),  # one row; N below one tile and one scale block
    (1, 208, 1024, False, FP16),
    (1000, 1344, 2048, False, FP16),  # M and N tails together
    (1000, 1344, 2048, True, BF16),
    (300, 200, 512, False, BF16),  # N % 16 == 8, plain B only
]

# About one output rounding step: half a BF16 ulp, one FP16 ulp.
RTOL = {BF16: 2**-8, FP16: 2**-10}


def make_inputs(m, n, k):
    g = torch.Generator(device="cuda").manual_seed(m * 7 + n * 3 + k)
    x = (torch.randn(m, k, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    w = (torch.randn(n, k, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    x_scale = torch.rand(m, k // 128, device="cuda", generator=g) * 0.02 + 0.001
    w_scale = torch.rand(-(-n // 128), k // 128, device="cuda", generator=g) * 0.02
    return x, w, x_scale, w_scale + 0.001


def reference(x, w, x_scale, w_scale):
    n = w.shape[0]
    a = x.float() * x_scale.repeat_interleave(128, 1)
    b = w.float() * w_scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return a @ b.T


def run_and_check(m, n, k, preshuffle_b, out_dtype):
    x, w, x_scale, w_scale = make_inputs(m, n, k)
    ref = reference(x, w, x_scale, w_scale)

    if preshuffle_b:
        wq = shuffle_weight(w, layout=(16, 16))
        # Preshuffled B takes x_scale as [K/128, M] memory under an (M, K/128) view.
        xs = x_scale.transpose(0, 1).contiguous().view(m, k // 128)
    else:
        wq, xs = w, x_scale

    guard = 256 * n
    buf = torch.full((m * n + guard,), float("nan"), dtype=out_dtype, device="cuda")
    out = buf[: m * n].view(m, n)
    name = kernelInstance(preshuffle_b).name
    run_gemm_a8w8_blockscale(x, wq, xs, w_scale, out, name, preshuffle_b)
    torch.cuda.synchronize()

    assert buf[m * n :].isnan().all(), "kernel stored past the end of C"
    torch.testing.assert_close(
        out.float(),
        ref,
        rtol=RTOL[out_dtype],
        atol=1e-3 * ref.abs().max().item(),
    )


@pytest.mark.parametrize("m,n,k,preshuffle_b,out_dtype", CASES)
def test_flydsl_gemm_a8w8_blockscale(m, n, k, preshuffle_b, out_dtype):
    run_and_check(m, n, k, preshuffle_b, out_dtype)


def test_flydsl_gemm_a8w8_blockscale_reuses_launcher_across_m():
    # M is a runtime argument: one (N, K, layout, dtype) launcher serves every M.
    for m in (256, 257, 1):
        run_and_check(m, 576, 512, False, BF16)


@pytest.mark.parametrize("cu_num", [128, 192])
def test_flydsl_gemm_a8w8_blockscale_rejects_unvalidated_cu_count(cu_num):
    ki = kernelInstance(False)
    assert kernel_fits_shape(ki, 256, 512, 256, "gfx950", VALIDATED_CU_NUM)
    assert not kernel_fits_shape(ki, 256, 512, 256, "gfx950", cu_num)


# Compiles each config with the GPUs hidden and dumps its final ISA.
_DUMP_SCRIPT = textwrap.dedent("""
    import os, sys, torch
    import flydsl.expr as fx
    from aiter.ops.flydsl.gemm_a8w8_blockscale import _compile_gemm
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_common import (
        kernelInstance,
    )

    root, m = sys.argv[1], 512
    for k in (256, 512):
        for ps in (False, True):
            for dt in (torch.bfloat16, torch.float16):
                os.environ["FLYDSL_DUMP_DIR"] = os.path.join(root, f"k{k}_ps{ps}_{dt}")
                n = 3392
                launch = _compile_gemm(n, k, kernelInstance(ps).name, 0, dt)
                launch(
                    torch.empty(m, k, dtype=torch.int8),
                    torch.empty(n, k, dtype=torch.int8),
                    torch.empty(m, n, dtype=dt),
                    torch.empty(k // 128 * m, dtype=torch.float32),
                    torch.empty(-(-n // 128) * (k // 128), dtype=torch.float32),
                    m,
                    fx.Stream(None),
                )
    """)


def test_flydsl_gemm_a8w8_blockscale_resources(tmp_path):
    env = dict(
        os.environ,
        FLYDSL_GPU_ARCH="gfx950",
        FLYDSL_RUNTIME_ENABLE_CACHE="0",
        FLYDSL_DEBUG_DUMP_ASM="1",
        COMPILE_ONLY="1",
        HIP_VISIBLE_DEVICES="99",
        CUDA_VISIBLE_DEVICES="99",
        ROCR_VISIBLE_DEVICES="99",
    )
    subprocess.run(
        [sys.executable, "-c", _DUMP_SCRIPT, str(tmp_path)], env=env, check=True
    )
    configs = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert len(configs) == 8
    for config in configs:
        (isa,) = config.rglob("*final_isa.s")
        meta = {
            key: int(value)
            for key, value in re.findall(
                r"^\s+\.(\w+):\s+(\d+)$", isa.read_text(), re.MULTILINE
            )
        }
        # An 8-wave workgroup puts two waves on each SIMD, leaving each wave at
        # most 256 of the SIMD's 512 VGPRs.
        assert meta["vgpr_count"] <= 256, (config.name, meta)
        assert meta["private_segment_fixed_size"] == 0, (config.name, meta)
        assert meta["vgpr_spill_count"] == 0, (config.name, meta)
        assert meta["sgpr_spill_count"] == 0, (config.name, meta)
        assert meta["group_segment_fixed_size"] <= 160 * 1024, (config.name, meta)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
