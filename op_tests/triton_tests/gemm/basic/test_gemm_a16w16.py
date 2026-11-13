# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import re

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.basic.gemm_a16w16 import _is_gluon_available, gemm_a16w16
from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
from op_tests.triton_tests.utils.types import str_to_torch_dtype


def is_gluon_supported():
    """gluon a16w16 kernels are only available on supported archs (gfx1250)."""
    return _is_gluon_available()


def _skip_unsupported_gluon(
    backend, kernel_type, M, N, K, layout="TN", activation=None
):
    """Skip gluon cases that no gluon kernel on this architecture covers."""
    if backend != "gluon":
        return
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if "gfx950" in (get_arch() or ""):
        # gfx950 has two gluon a16w16 kernels, compute_bound and streamk, TN layout.
        if kernel_type not in ("compute_bound", "streamk"):
            pytest.skip("gfx950 gluon a16w16 has only compute_bound and streamk")
        if layout != "TN":
            pytest.skip("gfx950 gluon a16w16 supports the TN layout only")
        if kernel_type == "streamk":
            from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
                unsupported_reason,
            )
        else:
            from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16 import (
                unsupported_reason,
            )

        reason = unsupported_reason(M, N, K, activation=activation)
        if reason:
            pytest.skip(f"gfx950 gluon a16w16: {reason}")
        return
    if not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")


def _skip_if_triton_on_gfx1250(backend):
    """gfx1250 only ships gluon-format a16w16 configs, so the triton backend has
    no usable config there; skip the triton backend on gfx1250."""
    if backend != "triton":
        return
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if "gfx1250" in (get_arch() or ""):
        pytest.skip("triton backend has no gfx1250 a16w16 config (gluon-only arch)")


def generate_gemm_a16w16_inputs(M, N, K, dtype, layout="TN", output=True, bias=False):
    torch.manual_seed(0)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    # TN is default layout
    if layout[0] == "T":
        x = torch.randn((M, K), dtype=dtype, device="cuda")
    else:
        x = torch.randn((K, M), dtype=dtype, device="cuda").T

    if layout[1] == "T":
        weight = torch.randn((K, N), dtype=dtype, device="cuda").T
    else:
        weight = torch.randn((N, K), dtype=dtype, device="cuda")

    bias_tensor = None
    if bias:
        bias_tensor = torch.empty((N), dtype=dtype, device="cuda")

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda")
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return x, weight, bias_tensor, out_dtype, y


def get_x_vals():
    x_vals = [(1, 1, 1)]  # minimal case
    x_vals += [(3, 5, 2)]  # irregular shape
    x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
    x_vals += [(2**i, 256, 7168) for i in range(5, 9)]  # DSR1 router GEMM
    # GPT-OSS-120B attention projections
    x_vals += [(2**i, 5120, 2880) for i in range(5, 9)]  # GPTOSS QKV input projection
    x_vals += [(2048, 5120, 2880)]  # GPTOSS QKV @ M=2048
    x_vals += [(8192, 5120, 2880)]  # GPTOSS QKV @ M=8192
    x_vals += [(2**i, 2880, 4096) for i in range(5, 9)]  # output projection
    x_vals += [(2**i, 128, 2880) for i in range(5, 9)]  # Router GEMM
    x_vals += [(v, 106496, 16384) for v in (256, 4096)]  # LL3 405B FC1
    return x_vals


# Test plain BF16 GEMMs - the most common types.
@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_gemm_a16_w16(M: int, N: int, K: int, backend, kernel_type):
    if backend == "triton" and kernel_type != "bandwidth_bound":
        pytest.skip("kernel_type only applies to the gluon backend")
    _skip_unsupported_gluon(backend, kernel_type, M, N, K)
    _skip_if_triton_on_gfx1250(backend)

    x, w, _, _out_dtype, _y = generate_gemm_a16w16_inputs(
        M,
        N,
        K,
        dtype=torch.bfloat16,
        output=False,
    )

    torch_out = F.linear(x, w, bias=None)

    triton_out = gemm_a16w16(x, w, backend=backend, kernel_type=kernel_type)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-2)


# Smaller set for testing activations, setting the output tensor and dtype
def get_fewer_x_vals():
    x_vals = [(16, 1024, 1024)]
    x_vals += [(128, 8192, 512)]
    x_vals += [(256, 512, 8192)]
    x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in (1, 5, 8)]
    return x_vals


# A smaller set of shapes that tests fused activations, different dtypes
# and output tensor arg. We don't want the larger set above to test
# all these combinations.
@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "silu"])
@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_gemm_a16_w16_activation(
    M: int, N: int, K: int, dtype, output, activation, backend, kernel_type
):
    if backend == "triton" and kernel_type != "bandwidth_bound":
        pytest.skip("kernel_type only applies to the gluon backend")
    _skip_unsupported_gluon(backend, kernel_type, M, N, K, activation=activation)
    _skip_if_triton_on_gfx1250(backend)

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M,
        N,
        K,
        dtype,
        output=output,
    )

    torch_out = F.linear(x, w, bias=None)
    if activation == "gelu":
        torch_out = F.gelu(torch_out)
    elif activation == "gelu_tanh":
        torch_out = F.gelu(torch_out, approximate="tanh")
    elif activation == "silu":
        torch_out = F.silu(torch_out)

    triton_out = gemm_a16w16(
        x,
        w,
        None,
        out_dtype,
        y,
        activation=activation,
        backend=backend,
        kernel_type=kernel_type,
    )

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("layout", ["TT", "NN", "NT"])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_gemm_a16_w16_layout(M: int, N: int, K: int, layout, backend, kernel_type):
    if backend == "triton" and kernel_type != "bandwidth_bound":
        pytest.skip("kernel_type only applies to the gluon backend")
    _skip_unsupported_gluon(backend, kernel_type, M, N, K, layout=layout)
    _skip_if_triton_on_gfx1250(backend)

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, layout=layout, output=False
    )

    torch_out = F.linear(x, w, bias=None)

    triton_out = gemm_a16w16(
        x, w, None, out_dtype, y, backend=backend, kernel_type=kernel_type
    )

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("output", [True, False])
def test_gemm_a16_w16_atomic(M: int, N: int, K: int, output):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _, _out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=output
    )

    torch_out = F.linear(x, w, bias=None)

    # Accumulation in bf16/fp16 leads to precision loss, cast y to fp32 to prevent that
    if output:
        y = y.to(torch.float32).zero_()
        triton_out = gemm_a16w16_atomic(x, w, torch.float32, y).to(torch.bfloat16)
    else:
        triton_out = gemm_a16w16_atomic(x, w, dtype=torch.float32).to(torch.bfloat16)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("layout", ["TT", "NN", "NT"])
def test_gemm_a16_w16_atomic_layout(M: int, N: int, K: int, layout):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _, _out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, layout=layout, output=True
    )

    torch_out = F.linear(x, w, bias=None)

    y = y.to(torch.float32).zero_()
    triton_out = gemm_a16w16_atomic(x, w, torch.float32, y).to(torch.bfloat16)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("strided", [False, True])
def test_gemm_a16_w16_atomic_accumulate(M: int, N: int, K: int, strided):
    """accumulate=True adds X @ W^T into y in place, like torch.addmm_."""
    torch.cuda.empty_cache()

    x, w, _, _out_dtype, _y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=False
    )
    base = torch.randn((M, 2 * N if strided else N), dtype=torch.float32, device="cuda")
    y = base[:, :N]
    torch_out = torch.addmm(y, x.float(), w.float().T)

    triton_out = gemm_a16w16_atomic(x, w, torch.float32, y, accumulate=True)

    assert triton_out.data_ptr() == y.data_ptr()
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("layout", ["TN", "TT"])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_gemm_a16w16_persistent_output(M: int, N: int, K: int, layout, output, backend):
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    torch.cuda.empty_cache()

    x, w, _, _out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, layout=layout, output=output
    )

    torch_out = F.linear(x, w, bias=None)

    if output:
        triton_out = gemm_a16w16(
            x, w, dtype=torch.bfloat16, y=y, backend=backend, persistent=True
        )
    else:
        triton_out = gemm_a16w16(
            x, w, dtype=torch.bfloat16, backend=backend, persistent=True
        )

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


# The gfx950 gluon compute_bound kernel (scheduled by Triton's MFMA scheduler) at compute-bound shapes,
# both dtypes, with and without a pre-allocated output.
@pytest.mark.parametrize(
    "M, N, K",
    [(4096, 4096, 8192), (8192, 8192, 8192), (4096, 8192, 4096), (256, 1024, 1024)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("bias", [False, True])
def test_gemm_a16_w16_gluon_compute_bound(M: int, N: int, K: int, dtype, output, bias):
    _skip_unsupported_gluon("gluon", "compute_bound", M, N, K)
    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(M, N, K, dtype, output=output)
    b = torch.randn((N,), dtype=dtype, device="cuda") if bias else None
    torch_out = F.linear(x, w, bias=b)
    out = gemm_a16w16(
        x, w, b, out_dtype, y, backend="gluon", kernel_type="compute_bound"
    )
    torch.testing.assert_close(out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "M, N, K",
    [
        (1000, 2048, 512),  # ragged M
        (512, 1016, 512),  # ragged N (a multiple of 8, not of any tile)
        (256, 256, 1032),  # ragged K, one tail chunk
        (256, 256, 1000),  # ragged K, two tail chunks, K not a multiple of 16
        (475, 1016, 1000),  # all three ragged
        (100, 72, 264),  # smaller than every tile
        (57, 19360, 4096),  # a production shape with a ragged M
    ],
)
@pytest.mark.parametrize("tile", [None, (256, 256), (64, 64)])
@pytest.mark.parametrize("bias", [False, True])
def test_gemm_a16_w16_gluon_compute_bound_ragged(M, N, K, tile, bias):
    """Ragged M and N (clamped loads, masked stores) and ragged K (the tail
    consumed first) give the right result, with the config's tile and forced
    ones; the output is pre-filled with NaN so an unwritten element shows."""
    _skip_unsupported_gluon("gluon", "compute_bound", M, N, K)
    x, w, _, out_dtype, _ = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=False
    )
    b = torch.randn((N,), dtype=torch.bfloat16, device="cuda") if bias else None
    y = torch.full((M, N), float("nan"), dtype=torch.bfloat16, device="cuda")
    torch_out = F.linear(x, w, bias=b)
    config = None if tile is None else {"BLOCK_M": tile[0], "BLOCK_N": tile[1]}
    out = gemm_a16w16(
        x,
        w,
        b,
        out_dtype,
        y,
        backend="gluon",
        kernel_type="compute_bound",
        config=config,
    )
    assert not torch.isnan(out).any()
    torch.testing.assert_close(out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "tile",
    [
        (256, 256),
        (256, 128),
        (128, 256),
        (128, 128),
        (256, 64),
        (64, 256),
        (128, 64),
        (64, 128),
        (64, 64),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
def test_gemm_a16_w16_gluon_compute_bound_tiles(tile, bias):
    """Every tile gives the right result, and the compiled kernels carry the MFMA
    scheduler's sched.barriers: a Triton that accepted schedule_hint but ignored
    it would still pass the numerical check, about 15% slower."""
    M, N, K = 1024, 1024, 512
    _skip_unsupported_gluon("gluon", "compute_bound", M, N, K)
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16 import (
        _gemm_a16w16_compute_bound_kernel,
    )

    x, w, _, out_dtype, _ = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=False
    )
    b = torch.randn((N,), dtype=torch.bfloat16, device="cuda") if bias else None
    torch_out = F.linear(x, w, bias=b)
    _gemm_a16w16_compute_bound_kernel.device_caches.clear()  # inspect this compile only
    out = gemm_a16w16(
        x,
        w,
        b,
        out_dtype,
        None,
        backend="gluon",
        kernel_type="compute_bound",
        config={"BLOCK_M": tile[0], "BLOCK_N": tile[1]},
    )
    torch.testing.assert_close(out, torch_out, atol=1e-1, rtol=1e-2)
    # device_caches[device] = (kernel_cache, key_cache, target, backend, binder)
    compiled = [
        kernel
        for entry in _gemm_a16w16_compute_bound_kernel.device_caches.values()
        for kernel in entry[0].values()
    ]
    assert compiled, "the gluon kernel was not compiled"
    for kernel in compiled:
        assert "llvm.amdgcn.sched.barrier" in kernel.asm["llir"]
        assert "sched_barrier" in kernel.asm["amdgcn"]


def test_gemm_a16_w16_gluon_pick_gemm_tile():
    """The default gluon config picks by the wave cost model (workgroup waves over
    per-tile throughput); a tuned config names its tile."""
    from aiter.ops.triton.utils.gemm_config_utils import pick_gemm_tile

    tiles = [
        [256, 256],
        [256, 128],
        [128, 256],
        [128, 128],
        [256, 64],
        [64, 256],
        [128, 64],
        [64, 128],
        [64, 64],
    ]
    default = {
        "TILES": tiles,
        "TILE_TFLOPS": [1583, 1386, 1388, 1162, 1040, 1009, 877, 871, 603],
        "TAIL_WAVE_COST": 0.5,
    }
    # Exactly one wave of the largest tile.
    assert pick_gemm_tile(default, 4096, 4096, 256) == (256, 256)
    # Half a wave of 256x256 loses to one full wave of 256x128.
    assert pick_gemm_tile(default, 4096, 2048, 256) == (256, 128)
    assert pick_gemm_tile(default, 2048, 2048, 256) == (128, 128)
    assert pick_gemm_tile(default, 1024, 1024, 256) == (64, 64)
    # 91% of a wave of 256x256 beats two waves of 256x128 (ragged M).
    assert pick_gemm_tile(default, 7226, 2048, 256) == (256, 256)
    # A tiny M: the tile that pads M least, not the one that fills the GPU.
    assert pick_gemm_tile(default, 57, 19360, 256) == (64, 128)
    assert pick_gemm_tile(default, 100, 4096, 256) == (64, 64)
    # Without throughputs: the largest tile that fills the GPU once.
    legacy = {"TILES": tiles, "WORKGROUPS_PER_CU": 1}
    assert pick_gemm_tile(legacy, 4096, 2048, 256) == (256, 128)
    assert pick_gemm_tile(legacy, 256, 256, 256) == (64, 64)
    assert pick_gemm_tile({"BLOCK_M": 128, "BLOCK_N": 256}, 4096, 4096, 256) == (
        128,
        256,
    )


@pytest.mark.parametrize(
    "kernel_name", ["gluon_compute_bound_256x256", "gluon_compute_bound_128x64", None]
)
def test_gemm_a16_w16_gluon_tuned_gemm(kernel_name):
    """tuned_gemm's gluon libtype runs the tile named by the tuned row, or the
    config's pick when the row has no name."""
    M, N, K = 1024, 2048, 512
    _skip_unsupported_gluon("gluon", "compute_bound", M, N, K)
    from aiter.tuned_gemm import gluon_gemm

    x, w, _, out_dtype, _ = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=False
    )
    out = gluon_gemm(x, w, 0, None, out_dtype, config={"kernelName": kernel_name})
    torch.testing.assert_close(out, F.linear(x, w), atol=1e-1, rtol=1e-2)


def _streamk_plan(policy, fixup="owner", tail_split_k=None, exchange="wt"):
    return {
        "STREAMK_POLICY": policy,
        "STREAMK_FIXUP": fixup,
        "TAIL_SPLIT_K": tail_split_k,
        "STREAMK_EXCHANGE": exchange,
    }


def _run_streamk(
    M, N, K, config, dtype=torch.bfloat16, bias=False, persistent_partials=True
):
    """The gfx950 gluon Stream-K kernel against F.linear, through the public
    wrapper, into an output pre-filled with NaN so an unwritten element shows.
    Skips a config the kernel rejects for this shape on this device (the tail tile
    count depends on the CU count)."""
    _skip_unsupported_gluon("gluon", "streamk", M, N, K)
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
        unsupported_reason,
        with_default_tile,
    )

    reason = unsupported_reason(M, N, K, config=with_default_tile(config))
    if reason:
        pytest.skip(f"gfx950 gluon a16w16 streamk: {reason}")
    x, w, _, _, _ = generate_gemm_a16w16_inputs(M, N, K, dtype, output=False)
    b = torch.randn((N,), dtype=dtype, device="cuda") if bias else None
    y = torch.full((M, N), float("nan"), dtype=dtype, device="cuda")
    out = gemm_a16w16(
        x,
        w,
        b,
        dtype,
        y,
        backend="gluon",
        kernel_type="streamk",
        config=config,
        persistent_partials=persistent_partials,
    )
    assert not torch.isnan(out).any(), "the Stream-K kernel left elements unwritten"
    if config and config.get("STREAMK_PARTIALS") == "16bit":
        # A 16-bit partial is rounded relative to itself, not to the sum: partials of
        # about sqrt(K) (randn inputs) that cancel leave their rounding on a small
        # output. 2e-2 sqrt(K) is about twice the worst case, still far below a wrong
        # or missing partial (about sqrt(K / n)).
        atol = 2e-2 * K**0.5
        torch.testing.assert_close(out, F.linear(x, w, bias=b), atol=atol, rtol=1e-2)
    else:
        torch.testing.assert_close(out, F.linear(x, w, bias=b), atol=1e-1, rtol=1e-2)


# Shapes in units of a 128x128 tile with K = 512 (four K-pairs), sized for 256 CUs:
# 16x16 tiles are one full wave, 17x16 leave a tail of 16 tiles after it, 16x28 a
# tail of 192 (more than half a wave), and 3x11 = 33 tiles fall short of one wave.
# On another CU count a case its config cannot run is skipped.
_STREAMK_CASES = {
    "dp": ((2048, 2048, 512), _streamk_plan("dp")),
    "owner": ((2176, 2048, 512), _streamk_plan("one_tile", "owner", 4)),
    "rs": ((2176, 2048, 512), _streamk_plan("one_tile", "rs", 4)),
    "l2": ((2176, 2048, 512), _streamk_plan("one_tile", "rs", 4, "l2")),
    "two_tile": ((2176, 2048, 512), _streamk_plan("two_tile")),
    "spread": ((2048, 3584, 512), _streamk_plan("spread")),
    "default_plan": ((2176, 2048, 512), {}),
    "compact_owner": ((384, 1408, 512), _streamk_plan("one_tile", "owner", 2)),
    # Extra readers: the programs the tail leaves idle read slices too.
    "compact_rs": ((384, 1408, 512), _streamk_plan("one_tile", "rs", 2)),
    "compact_l2": ((384, 1408, 512), _streamk_plan("one_tile", "rs", 4, "l2")),
    "dp_single": ((128, 128, 512), _streamk_plan("dp")),
    # Ragged M, N and K: clamped loads, masked stores, the K tail run first.
    "ragged_rs": ((2160, 2032, 496), _streamk_plan("one_tile", "rs", 2)),
    "ragged_n_owner": ((2176, 2032, 512), _streamk_plan("one_tile", "owner", 4)),
    "ragged_wave_dp": ((368, 1392, 496), _streamk_plan("dp")),
    # K_RES > BLOCK_K: the last pair's first K-step is masked too.
    "kres_gt_rs": ((2176, 2048, 432), _streamk_plan("one_tile", "rs", 4)),
    # Sub-tile tail: a tail of 16 cut into 64x64 sub-tiles (Q = 4), a tail of 96 into
    # 64x128 (Q = 2), and a ragged tail row whose lower 64x64 sub-tiles are wholly
    # outside C.
    "dp_sub": ((2176, 2048, 512), _streamk_plan("dp_sub")),
    "dp_sub_q2": ((2816, 2048, 512), _streamk_plan("dp_sub")),
    "ragged_dp_sub": ((2096, 2032, 496), _streamk_plan("dp_sub")),
}


@pytest.mark.parametrize("case", list(_STREAMK_CASES))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_gemm_a16_w16_gluon_streamk(case, dtype, bias):
    """Every Stream-K policy and ending (dp, dp_sub, one_tile with the owner and the
    reduce-scatter over wt and l2, two_tile, spread, the default plan), below one
    wave and after it, on aligned and ragged shapes."""
    (M, N, K), plan = _STREAMK_CASES[case]
    config = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, **plan}
    _run_streamk(M, N, K, config, dtype=dtype, bias=bias)


@pytest.mark.parametrize(
    "tile", [(256, 256, 64), (128, 256, 64), (64, 256, 64), (64, 64, 128)]
)
@pytest.mark.parametrize("ragged", [False, True])
def test_gemm_a16_w16_gluon_streamk_tiles(tile, ragged):
    """Other tiles, with the default plan, on a shape with a Stream-K tail."""
    bm, bn, bk = tile
    M, N, K = 17 * bm, 16 * bn, 8 * bk
    if ragged:
        M, N, K = M - bm // 8, N - bn // 8, K - 16
    _run_streamk(M, N, K, {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, bias=True)


@pytest.mark.parametrize("tail_split_k", [4, 8])
@pytest.mark.parametrize("partials", ["fp32", "16bit"])
def test_gemm_a16_w16_gluon_streamk_owner_256x256(tail_split_k, partials):
    """The 256x256 owner unrolls its peer loop up to split 4 (owner_unroll_max) and
    reads its peers in a run-time loop past it, ragged, with bias."""
    M, N, K = 17 * 256 - 32, 16 * 256 - 40, 2048 - 16
    config = {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 4}
    config.update(
        _streamk_plan("one_tile", "owner", tail_split_k), STREAMK_PARTIALS=partials
    )
    _run_streamk(M, N, K, config, bias=True)


def test_gemm_a16_w16_gluon_streamk_default_tile():
    """Without a config the kernel runs DEFAULT_TILE with the default plan."""
    _run_streamk(1000, 2048, 1024, None)


def test_gemm_a16_w16_gluon_streamk_candidates():
    """Every config the tuner times for a shape runs correctly, and its kernelName
    round-trips to the same config through tuned_gemm."""
    M, N, K = 1024, 1024, 512
    _skip_unsupported_gluon("gluon", "streamk", M, N, K)
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
        kernel_name,
        tuning_candidates,
    )
    from aiter.tuned_gemm import parse_gluon_kernel_name

    candidates = tuning_candidates(M, N, K)
    assert candidates, "no Stream-K candidate for a shape every tile fits"
    names = [kernel_name(c) for c in candidates]
    assert len(set(names)) == len(names), "two candidates share a kernelName"
    x, w, _, _, _ = generate_gemm_a16w16_inputs(M, N, K, torch.bfloat16, output=False)
    torch_out = F.linear(x, w)
    for config, name in zip(candidates, names):
        assert parse_gluon_kernel_name(name) == ("streamk", config), name
        out = gemm_a16w16(
            x,
            w,
            None,
            torch.bfloat16,
            None,
            backend="gluon",
            kernel_type="streamk",
            config=config,
        )
        torch.testing.assert_close(out, torch_out, atol=1e-1, rtol=1e-2, msg=name)


@pytest.mark.parametrize(
    "kernel_name",
    [
        "gluon_streamk_256x256x64_g4_dp",
        "gluon_streamk_128x128x64_g4_one_tile_owner_k2_wt",
        "gluon_streamk_128x128x64_g4_one_tile_rs_k2_l2",
    ],
)
def test_gemm_a16_w16_gluon_streamk_tuned_gemm(kernel_name):
    """tuned_gemm's gluon libtype runs the Stream-K config a tuned row names."""
    M, N, K = 1024, 2048, 512
    _skip_unsupported_gluon("gluon", "streamk", M, N, K)
    from aiter.tuned_gemm import gluon_gemm

    x, w, _, out_dtype, _ = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=False
    )
    out = gluon_gemm(x, w, 0, None, out_dtype, config={"kernelName": kernel_name})
    torch.testing.assert_close(out, F.linear(x, w), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "tile, tiles, tail",
    [
        ((256, 256), 260, (64, 64)),  # S = 4: Q = 16
        ((256, 256), 288, (64, 128)),  # S = 32: Q = 8, the smaller TAIL_BLOCK_M
        ((256, 256), 320, (128, 128)),  # S = 64: Q = 4, square
        ((256, 256), 384, (128, 256)),  # S = 128: Q = 2
        ((128, 256), 352, (128, 128)),  # S = 96: Q = 2, square over 64x256
        ((256, 256), 388, None),  # S = 132: more than half a wave
        ((256, 256), 512, None),  # no leftover
        ((256, 256), 100, None),  # no full wave
        ((64, 64), 260, None),  # no smaller sub-tile
    ],
)
def test_gemm_a16_w16_gluon_streamk_subtile_tail(tile, tiles, tail):
    """subtile_tail: the largest Q with S Q <= P, then the most square sub-tile,
    then the least padding, then the smaller TAIL_BLOCK_M (P = 256, four rows of
    tiles)."""
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
        subtile_tail,
    )

    bm, bn = tile
    M, N = 4 * bm, tiles // 4 * bn
    assert subtile_tail(M, N, (bm, bn, 64), 256) == tail


@pytest.mark.parametrize(
    "K, plan",
    [
        (512, "dp_sub"),
        (4096, "dp_sub"),
        (4160, "one_tile"),  # past SUBTILE_MAX_K: a K split
    ],
)
def test_gemm_a16_w16_gluon_streamk_default_plan_subtile(K, plan):
    """The default plan takes dp_sub after a full wave with at most half a wave left
    and K <= SUBTILE_MAX_K (P = 256, 256x256 tiles, a tail of 32)."""
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
        streamk_plan,
    )

    assert streamk_plan(1024, 72 * 256, K, (256, 256, 64), 256)[0] == plan
    # More than half a wave left: no sub-tile tail at any K.
    assert streamk_plan(1024, 112 * 256, 512, (256, 256, 64), 256)[0] != "dp_sub"


@pytest.mark.parametrize("M", [2176, 2816])
def test_gemm_a16_w16_gluon_streamk_dp_sub_follows_m(M):
    """A tuned dp_sub row derives its sub-tile from the actual M (tails of 16 and 96
    128x128 tiles on 256 CUs), and its kernelName round-trips."""
    N, K = 2048, 512
    _skip_unsupported_gluon("gluon", "streamk", M, N, K)
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
        kernel_name,
        tuning_candidates,
    )
    from aiter.tuned_gemm import gluon_gemm, parse_gluon_kernel_name

    name = "gluon_streamk_128x128x64_g4_dp_sub"
    subs = [c for c in tuning_candidates(M, N, K) if c["STREAMK_POLICY"] == "dp_sub"]
    if not subs:
        pytest.skip("no sub-tile tail on this CU count")
    assert name in [kernel_name(c) for c in subs]
    assert parse_gluon_kernel_name(name)[1]["STREAMK_POLICY"] == "dp_sub"
    x, w, _, out_dtype, _ = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=False
    )
    out = gluon_gemm(x, w, 0, None, out_dtype, config={"kernelName": name})
    torch.testing.assert_close(out, F.linear(x, w), atol=1e-1, rtol=1e-2)


def test_gemm_a16_w16_gluon_streamk_fit_tuned_config():
    """A tuned plan is kept where it fits the shape and otherwise replaced by the
    tile alone (the default plan)."""
    tuned_M, N, K = 2176, 2048, 512  # 17x16 128x128 tiles: a tail of 16 on 256 CUs
    _skip_unsupported_gluon("gluon", "streamk", tuned_M, N, K)
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
        fit_tuned_config,
        unsupported_reason,
    )

    tile = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4}
    config = {**tile, **_streamk_plan("one_tile", "owner", 4)}
    reason = unsupported_reason(tuned_M, N, K, config=config)
    if reason:
        pytest.skip(f"gfx950 gluon a16w16 streamk: {reason}")
    assert fit_tuned_config(config, tuned_M, N, K) is config
    assert fit_tuned_config(config, 2100, N, K) is config
    # 16x16 tiles: one full wave, no tail for one_tile.
    assert fit_tuned_config(config, 2048, N, K) == tile


@pytest.mark.parametrize(
    "kernel_name, M, N",
    [
        # Tuned at M = 2176 (a tail of 16 tiles); M = 2048 leaves none.
        ("gluon_streamk_128x128x64_g4_one_tile_owner_k4_wt", 2048, 2048),
        ("gluon_streamk_128x128x64_g4_one_tile_rs_k4_l2", 2048, 2048),
        # Tuned at M = 2048 (a tail of 192); M = 2560 leaves 48, too few for spread.
        ("gluon_streamk_128x128x64_g4_spread", 2560, 3584),
        # Tuned at M = 2176 (a tail of 16); M = 2048 leaves none for a sub-tile tail.
        ("gluon_streamk_128x128x64_g4_dp_sub", 2048, 2048),
    ],
)
def test_gemm_a16_w16_gluon_streamk_tuned_gemm_other_m(kernel_name, M, N):
    """A tuned row reached through the padded-M lookup runs at an M its plan cannot
    run, with the row's tile and the default plan."""
    K = 512
    _skip_unsupported_gluon("gluon", "streamk", M, N, K)
    from aiter.tuned_gemm import gluon_gemm

    x, w, _, out_dtype, _ = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=False
    )
    out = gluon_gemm(x, w, 0, None, out_dtype, config={"kernelName": kernel_name})
    torch.testing.assert_close(out, F.linear(x, w), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("case", ["owner", "rs", "l2", "compact_rs"])
@pytest.mark.parametrize("persistent_partials", [True, False])
def test_gemm_a16_w16_gluon_streamk_persistent_partials(case, persistent_partials):
    """Per-call and persistent partials, with launches of two tiles interleaved so
    the shared persistent buffer is sliced for each and the flags are reused."""
    (M, N, K), plan = _STREAMK_CASES[case]
    config = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, **plan}
    big = {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}
    for shape, cfg in [((M, N, K), config), ((4352, 4096, 512), big)] * 2:
        _run_streamk(*shape, cfg, persistent_partials=persistent_partials)


_P16_CASES = [
    "owner",
    "rs",
    "l2",
    "two_tile",
    "spread",
    "compact_owner",
    "compact_rs",
    "compact_l2",
    "ragged_rs",
    "ragged_n_owner",
    "kres_gt_rs",
]


@pytest.mark.parametrize("case", _P16_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_gemm_a16_w16_gluon_streamk_16bit_partials(case, dtype, bias):
    """Every policy and ending that stores partials, with 16-bit partials."""
    (M, N, K), plan = _STREAMK_CASES[case]
    config = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, **plan}
    _run_streamk(
        M, N, K, dict(config, STREAMK_PARTIALS="16bit"), dtype=dtype, bias=bias
    )


@pytest.mark.parametrize(
    "tile",
    [
        (256, 256, 64),  # quadrants of 16 blocks, paired 32 columns apart
        (256, 128, 64),
        (128, 64, 64),  # one block wide: paired 32 rows apart
        (256, 64, 64),
        (64, 64, 64),  # one block: tl with tr, bl with br
        (64, 64, 128),
    ],
)
@pytest.mark.parametrize("fixup, tail_split_k", [("owner", 4), ("rs", 2), ("rs", 8)])
@pytest.mark.parametrize("ragged", [False, True])
def test_gemm_a16_w16_gluon_streamk_16bit_partials_tiles(
    tile, fixup, tail_split_k, ragged
):
    """16-bit partials at every pairing of blocks, on a shape with a tail of 16 tiles
    after one wave (256 CUs)."""
    bm, bn, bk = tile
    M, N, K = 17 * bm, 16 * bn, 16 * bk
    if ragged:
        M, N, K = M - bm // 8, N - bn // 8, K - 16
    config = {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_SIZE_M": 4}
    config.update(
        _streamk_plan("one_tile", fixup, tail_split_k), STREAMK_PARTIALS="16bit"
    )
    _run_streamk(M, N, K, config, bias=True)


def test_gemm_a16_w16_gluon_streamk_16bit_partials_config():
    """16-bit partials: only plans that store partials take them, a tile-only config
    keeps them only when the default plan stores partials, kernelNames round-trip,
    and the tuner adds them only when asked."""
    M, N, K = 2176, 2048, 4096
    _skip_unsupported_gluon("gluon", "streamk", M, N, K)
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
        StreamKConfig,
        complete_config,
        kernel_name,
        parse_kernel_name,
        tuning_candidates,
    )

    tile = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4}
    for policy in ("dp", "dp_sub"):
        with pytest.raises(ValueError, match="stores no partials"):
            StreamKConfig.from_dict(
                {**tile, **_streamk_plan(policy), "STREAMK_PARTIALS": "16bit"}
            )
    with pytest.raises(ValueError, match="STREAMK_PARTIALS"):
        StreamKConfig.from_dict(
            {**tile, **_streamk_plan("dp"), "STREAMK_PARTIALS": "bf16"}
        )
    config = {
        **tile,
        **_streamk_plan("one_tile", "rs", 8, "l2"),
        "STREAMK_PARTIALS": "16bit",
    }
    name = kernel_name(config)
    assert name == "gluon_streamk_128x128x64_g4_one_tile_rs_k8_l2_p16"
    assert parse_kernel_name(name) == config
    assert kernel_name({**config, "STREAMK_PARTIALS": "fp32"}) == name[: -len("_p16")]
    # The default plan at 256 CUs: a K split at K = 8192 (a tail of 16), dp_sub at 512.
    split = complete_config(
        {**tile, "STREAMK_PARTIALS": "16bit"}, 2176, 2048, 8192, 256
    )
    assert split["STREAMK_POLICY"] == "one_tile"
    assert split["STREAMK_PARTIALS"] == "16bit"
    sub = complete_config({**tile, "STREAMK_PARTIALS": "16bit"}, 2176, 2048, 512, 256)
    assert sub["STREAMK_POLICY"] == "dp_sub" and sub["STREAMK_PARTIALS"] == "fp32"

    plain = [kernel_name(c) for c in tuning_candidates(M, N, K)]
    both = [kernel_name(c) for c in tuning_candidates(M, N, K, partials_16bit=True)]
    assert not any(n.endswith("_p16") for n in plain)
    assert set(plain) < set(both) and len(set(both)) == len(both)
    extra = [n for n in both if n not in plain]
    assert extra and all(n.endswith("_p16") for n in extra)
    assert all(parse_kernel_name(n)["STREAMK_POLICY"] != "dp" for n in extra)


def test_gemm_a16_w16_gluon_streamk_reduce_scatter_trip():
    """A reduce-scatter trip reads at most the tile's tail_split_k partials, and a
    reader's slice is a whole number of groups."""
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
        TileSpec,
        reduce_scatter_config,
    )

    for bm, bn in ((64, 64), (128, 64), (128, 128), (256, 256)):
        tile = TileSpec(bm, bn, 64)
        for k in (2, 3, 4, 6, 8, 16):
            for p16 in (False, True):
                rs = reduce_scatter_config(tile, k, p16=p16)
                assert rs.partials_per_trip <= k, (tile, k, p16, rs)
                assert (tile.blocks // rs.readers) % rs.blocks_per_group == 0
    # 64x64 at k = 4: 4 fp32 readers of one block, each partial one load, so one trip
    # of 4 loads rather than 16 with 12 masked off.
    assert reduce_scatter_config(TileSpec(64, 64, 64), 4).partials_per_trip == 4


def test_gemm_a16_w16_gluon_streamk_16bit_partials_tuned_gemm():
    """tuned_gemm runs a tuned 16-bit-partials row."""
    M, N, K = 2176, 2048, 512
    _skip_unsupported_gluon("gluon", "streamk", M, N, K)
    from aiter.tuned_gemm import gluon_gemm

    x, w, _, out_dtype, _ = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=False
    )
    name = "gluon_streamk_128x128x64_g4_one_tile_rs_k4_wt_p16"
    out = gluon_gemm(x, w, 0, None, out_dtype, config={"kernelName": name})
    torch.testing.assert_close(out, F.linear(x, w), atol=2e-2 * K**0.5, rtol=1e-2)


def _streamk_compiled(M, N, K, config, **run_kwargs):
    """The one Stream-K kernel compiled for a config (run through _run_streamk)."""
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
        _gemm_a16w16_streamk_kernel,
    )

    _gemm_a16w16_streamk_kernel.device_caches.clear()  # inspect this compile only
    _run_streamk(M, N, K, config, **run_kwargs)
    compiled = [
        kernel
        for entry in _gemm_a16w16_streamk_kernel.device_caches.values()
        for kernel in entry[0].values()
    ]
    assert len(compiled) == 1
    return compiled[0]


def _streamk_memory_ops(M, N, K, config):
    """The narrow (below 16-byte) buffer / global accesses in the compiled Stream-K
    kernel for a config."""
    ops = re.findall(
        r"^\s*((?:buffer|global)_(?:load|store)_\w+)",
        _streamk_compiled(M, N, K, config).asm["amdgcn"],
        re.M,
    )
    return [op for op in ops if not op.endswith(("x4", "b128"))]


@pytest.mark.parametrize("tile", [(256, 256, 64), (128, 64, 64), (64, 64, 64)])
@pytest.mark.parametrize("fixup", ["owner", "rs"])
@pytest.mark.parametrize("n_less", [0, 8])
def test_gemm_a16_w16_gluon_streamk_16bit_partials_width(tile, fixup, n_less):
    """16-bit partials are stored and read 16 bytes per lane: the 16-bit build has
    no more narrow accesses than the fp32 build of the same plan. An N of 8 mod 16
    (n_less 8), whose row stride the launch cannot prove 16-byte aligned, stores C no
    narrower than 16 bytes either."""
    bm, bn, bk = tile
    M, N, K = 17 * bm, 16 * bn - n_less, 16 * bk
    config = {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_SIZE_M": 4}
    config.update(_streamk_plan("one_tile", fixup, 4))
    fp32 = _streamk_memory_ops(M, N, K, config)
    p16 = _streamk_memory_ops(M, N, K, dict(config, STREAMK_PARTIALS="16bit"))
    assert len(p16) <= len(fp32), (fp32, p16)
    assert not any(op.endswith("_short") for op in fp32 + p16), (fp32, p16)


@pytest.mark.parametrize(
    "dtype, ir_type", [(torch.float16, "f16"), (torch.bfloat16, "bf16")]
)
def test_gemm_a16_w16_gluon_streamk_16bit_partials_dtype(dtype, ir_type):
    """16-bit partials take C's type: fp16 partials for an fp16 GEMM, not bf16."""
    M, N, K = 17 * 128, 16 * 128, 512
    config = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4}
    config.update(_streamk_plan("one_tile", "owner", 4), STREAMK_PARTIALS="16bit")
    kernel = _streamk_compiled(M, N, K, config, dtype=dtype, bias=True)
    p_type = re.search(r"%p_ptr: !tt\.ptr<(\w+)>", kernel.asm["ttgir"])
    assert p_type is not None and p_type.group(1) == ir_type, p_type


@pytest.mark.parametrize("tile", [(256, 256, 64), (64, 64, 64)])
def test_gemm_a16_w16_gluon_streamk_schedule(tile):
    """The compiled Stream-K kernel carries the MFMA scheduler's sched.barriers.
    K is a compile-time constant here, so K must be long enough that LLVM keeps
    the K loop (at 8 K-steps it unrolls it whole and the barriers go)."""
    M, N, K = 2048, 2048, 4096
    _skip_unsupported_gluon("gluon", "streamk", M, N, K)
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
        _gemm_a16w16_streamk_kernel,
    )

    bm, bn, bk = tile
    _gemm_a16w16_streamk_kernel.device_caches.clear()  # inspect this compile only
    _run_streamk(M, N, K, {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk})
    # device_caches[device] = (kernel_cache, key_cache, target, backend, binder)
    compiled = [
        kernel
        for entry in _gemm_a16w16_streamk_kernel.device_caches.values()
        for kernel in entry[0].values()
    ]
    assert compiled, "the gluon Stream-K kernel was not compiled"
    for kernel in compiled:
        assert "llvm.amdgcn.sched.barrier" in kernel.asm["llir"]
        assert "sched_barrier" in kernel.asm["amdgcn"]


# Non-power-of-two tiles (uneven halves): (M, N, K), tile, plan, sized for 256 CUs in
# units of the tile as above (17 x 16 tiles: a tail of 16 after one wave).
_UNEVEN_CASES = {
    # Ragged M, N and K: the extra blocks' clamped loads and masked stores.
    "dp": ((17 * 224 - 24, 16 * 256 - 40, 1008), (224, 256, 64), _streamk_plan("dp")),
    # 64x64 sub-tiles of a 192x256 tile, and 96x192 sub-tiles of a 192x192 tile (a
    # tail of 96).
    "dp_sub": ((17 * 192, 16 * 256, 1024), (192, 256, 64), _streamk_plan("dp_sub")),
    "dp_sub_uneven": (
        (22 * 192 - 40, 16 * 192 - 24, 1008),
        (192, 192, 64),
        _streamk_plan("dp_sub"),
    ),
    "owner": (
        (17 * 96, 16 * 96, 2048),
        (96, 96, 64),
        _streamk_plan("one_tile", "owner", 4),
    ),
    # Seven unrolled peers: 16-bit partials batch 8 loads, four batches.
    "owner_k8": (
        (17 * 224, 16 * 256, 2048),
        (224, 256, 64),
        _streamk_plan("one_tile", "owner", 8),
    ),
    "rs": (
        (17 * 224, 16 * 256, 4096),
        (224, 256, 64),
        _streamk_plan("one_tile", "rs", 8),
    ),
    # Below one wave: 32 tiles, so 4 idle programs per tile read slices too.
    "compact_rs": (
        (8 * 160 - 8, 4 * 224 - 16, 4096),
        (160, 224, 64),
        _streamk_plan("one_tile", "rs", 4),
    ),
    "l2": (
        (17 * 192, 16 * 128, 4096),
        (192, 128, 64),
        _streamk_plan("one_tile", "rs", 4, "l2"),
    ),
    "two_tile": ((16 * 224, 18 * 192, 2048), (224, 192, 64), _streamk_plan("two_tile")),
    "spread": ((10 * 224, 20 * 192, 2048), (224, 192, 64), _streamk_plan("spread")),
    # K_RES > BLOCK_K, BLOCK_K 128 at 96x160.
    "kres_gt_rs": (
        (17 * 160, 16 * 96, 432),
        (160, 96, 64),
        _streamk_plan("one_tile", "rs", 4),
    ),
    "bk128_owner": (
        (17 * 96, 16 * 160, 2048),
        (96, 160, 128),
        _streamk_plan("one_tile", "owner", 2),
    ),
    # Sides of 320: halves of 160 = 128 + 32, ragged M, N and K.
    "dp_320": (
        (17 * 320 - 24, 16 * 192 - 40, 1008),
        (320, 192, 64),
        _streamk_plan("dp"),
    ),
    "rs_320": (
        (17 * 192 - 8, 16 * 320 - 16, 4080),
        (192, 320, 64),
        _streamk_plan("one_tile", "rs", 8),
    ),
}


def _uneven_config(case, **extra):
    shape, (bm, bn, bk), plan = _UNEVEN_CASES[case]
    config = {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_SIZE_M": 4, **plan}
    return shape, dict(config, **extra)


@pytest.mark.parametrize("case", list(_UNEVEN_CASES))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_gemm_a16_w16_gluon_streamk_uneven(case, dtype, bias):
    """Non-power-of-two tiles on every Stream-K policy and ending."""
    shape, config = _uneven_config(case)
    _run_streamk(*shape, config, dtype=dtype, bias=bias)


@pytest.mark.parametrize(
    "case, tile",
    [
        ("owner", (96, 64, 64)),  # two one-block blocks paired across quadrants
        ("owner", (96, 96, 64)),  # one one-block block alone, 8 bytes per lane
        ("rs", (96, 96, 64)),
        ("rs", (160, 224, 64)),
        ("rs", (96, 64, 64)),
        ("rs", (224, 256, 64)),
        ("owner", (192, 128, 64)),
        ("spread", (224, 192, 64)),
        ("l2", (192, 128, 64)),
        ("rs", (192, 320, 64)),
        ("owner", (320, 192, 64)),
        ("owner_k8", (224, 256, 64)),
        ("owner_k8", (192, 320, 64)),
    ],
)
def test_gemm_a16_w16_gluon_streamk_uneven_16bit_partials(case, tile):
    """16-bit partials on uneven tiles, ragged, with bias."""
    (M, N, K), config = _uneven_config(case, STREAMK_PARTIALS="16bit")
    bm, bn, bk = tile
    scale_m, scale_n = M // config["BLOCK_M"], N // config["BLOCK_N"]
    config.update(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
    _run_streamk(scale_m * bm - 8, scale_n * bn - 16, K - 16, config, bias=True)


def test_gemm_a16_w16_gluon_streamk_uneven_geometry():
    """The tile menu, its halves and its rules."""
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
        TUNING_TILES,
        StreamKConfig,
        TileSpec,
        menu_tiles,
        uneven_first,
    )

    # The first half: as even as two valid halves (a power of two + 0, 32 or 64)
    # allow, the larger first; 0 for a power of two.
    assert [uneven_first(s) for s in (64, 96, 128, 160, 192, 224, 256, 288, 320)] == [
        0, 64, 0, 96, 96, 128, 0, 160, 160,
    ]  # fmt: skip
    assert TileSpec(224, 256, 64).uneven and not TileSpec(256, 128, 64).uneven
    for bad in [
        (288, 256, 64),
        (320, 320, 64),
        (352, 64, 64),
        (160, 160, 128),
    ]:
        with pytest.raises(ValueError):
            TileSpec(*bad)
    assert TileSpec(96, 160, 128).block_k == 128  # BLOCK_M + BLOCK_N <= 256
    menu = menu_tiles()
    assert set(TUNING_TILES) <= set(menu) and len(menu) == 86
    assert all(bm * bn <= 256 * 256 for bm, bn, _ in menu)
    # 16-bit partials on any menu tile: an odd number of one-block blocks keeps the
    # last one alone.
    plan = _streamk_plan("one_tile", "owner", 2)
    tile = {"BLOCK_M": 96, "BLOCK_N": 96, "BLOCK_K": 64, "GROUP_SIZE_M": 4}
    StreamKConfig.from_dict({**tile, **plan, "STREAMK_PARTIALS": "16bit"})


def test_gemm_a16_w16_gluon_streamk_tuning_tiles():
    """The tuner's --gluon_streamk_tiles values (tuning_tiles)."""
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
        menu_tiles,
        tuning_tiles,
    )

    assert tuning_tiles(None) is None and tuning_tiles("default") is None
    assert tuning_tiles("menu") == menu_tiles()
    assert tuning_tiles("192x320x64, 320x192x64") == [(192, 320, 64), (320, 192, 64)]
    for bad in ["288x256x64", "320x192x128", "192x320", "192x320xk"]:
        with pytest.raises(ValueError):
            tuning_tiles(bad)


@pytest.mark.parametrize(
    "tile, tiles, tail",
    [
        ((192, 256), 272, (64, 64)),  # S = 16: Q = 12
        ((224, 256), 272, (224, 64)),  # 224 has no smaller divisor in the menu
        ((192, 192), 352, (96, 192)),  # S = 96: Q = 2, an uneven sub-tile
        ((192, 256), 352, (192, 128)),  # Q = 2: the most square
        ((160, 160), 300, None),  # 160 has no smaller divisor in the menu
    ],
)
def test_gemm_a16_w16_gluon_streamk_uneven_subtile_tail(tile, tiles, tail):
    """subtile_tail with uneven tiles: sub-tile sides from the menu dividing the
    tile's (P = 256, four rows of tiles)."""
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
        subtile_tail,
    )

    bm, bn = tile
    assert subtile_tail(4 * bm, tiles // 4 * bn, (bm, bn, 64), 256) == tail


def test_gemm_a16_w16_gluon_streamk_uneven_candidates():
    """tuning_candidates(tiles=) times every plan at any menu tile, with kernelNames
    that round-trip through tuned_gemm; the tuner's own candidates stay
    TUNING_TILES', and a tile with a lone 32x32 block gets 16-bit plans too."""
    M, N, K = 17 * 224, 16 * 256, 1024
    _skip_unsupported_gluon("gluon", "streamk", M, N, K)
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk import (
        TUNING_TILES,
        kernel_name,
        tuning_candidates,
        tuning_tiles,
    )
    from aiter.tuned_gemm import parse_gluon_kernel_name

    stock = tuning_candidates(M, N, K)
    assert stock == tuning_candidates(M, N, K, tiles=TUNING_TILES)
    assert {(c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"]) for c in stock} <= set(
        TUNING_TILES
    )
    uneven = tuning_candidates(M, N, K, tiles=[(224, 256, 64)])
    assert uneven and all(c["BLOCK_M"] == 224 for c in uneven)
    wide = tuning_candidates(17 * 192, 16 * 320, K, tiles=tuning_tiles("192x320x64"))
    assert wide and all(c["BLOCK_N"] == 320 for c in wide)
    for config in uneven + wide:
        assert parse_gluon_kernel_name(kernel_name(config)) == ("streamk", config)
    odd = tuning_candidates(17 * 96, 16 * 96, K, True, tiles=[(96, 96, 64)])
    assert any(c.get("STREAMK_PARTIALS") == "16bit" for c in odd)


def test_gemm_a16_w16_gluon_streamk_uneven_tuned_gemm():
    """tuned_gemm runs a tuned row with an uneven tile."""
    M, N, K = 17 * 224, 16 * 256, 1024
    _skip_unsupported_gluon("gluon", "streamk", M, N, K)
    from aiter.tuned_gemm import gluon_gemm

    x, w, _, out_dtype, _ = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=False
    )
    name = "gluon_streamk_224x256x64_g4_one_tile_rs_k8_wt"
    out = gluon_gemm(x, w, 0, None, out_dtype, config={"kernelName": name})
    torch.testing.assert_close(out, F.linear(x, w), atol=1e-1, rtol=1e-2)
