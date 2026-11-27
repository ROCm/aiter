# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import functools
import importlib
import sys

import pytest
import torch
import triton

from aiter.ops.shuffle import shuffle_scale, shuffle_weight
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
    _get_config as _get_afp4wfp4_config,
)
from aiter.ops.triton.gemm.basic import gemm_afp4wfp4 as gemm_afp4wfp4_module
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    gemm_afp4wfp4 as triton_gemm_afp4wfp4,
)
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    gemm_afp4wfp4_preshuffle,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.triton_tests.gemm.gemm_shapes import get_shapes

DEVICE_ARCH = arch_info.get_arch()

# Mirrors the arch gate in gemm_afp4wfp4_(): archs with software-emulated FP4
# run only the unshuffled Triton path.
UNSHUFFLED_TRITON_ONLY = arch_info.is_sw_emulated_fp4_avail()

pytestmark = pytest.mark.skipif(
    not arch_info.is_any_fp4_avail(),
    reason="MXFP4 not supported on this architecture",
)


# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_inputs(
    M,
    N,
    K,
    dtype,
    layout="TN",
    output=True,
    shuffle_weight_fg=False,
    shuffle_scales_fg=False,
):
    if shuffle_weight_fg:
        assert (
            shuffle_scales_fg
        ), "weight shuffling is only supported with scale shuffling"

    torch.manual_seed(5)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    if layout[0] == "T":
        # 34 is two packed e2m1 values 0010 which is 1.0.
        x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device="cuda")
        x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device="cuda")
    else:
        x_low = torch.randint(0, 16, (K // 2, M), dtype=torch.uint8, device="cuda").T
        x_high = torch.randint(0, 16, (K // 2, M), dtype=torch.uint8, device="cuda").T

    if layout[1] == "N":
        w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
        w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    else:
        w_low = torch.randint(0, 16, (K // 2, N), dtype=torch.uint8, device="cuda").T
        w_high = torch.randint(0, 16, (K // 2, N), dtype=torch.uint8, device="cuda").T

    x = x_high << 4 | x_low

    w = w_low | w_high << 4
    # Scale of 1.0 in e8m0, bias 127.
    M_pad = (M + 255) // 256 * 256
    x_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, M_pad), dtype=torch.uint8, device="cuda"
    )
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    x_scales = x_scales.T
    w_scales = w_scales.T
    if shuffle_scales_fg:
        # Arch-independent aiter.ops.shuffle.shuffle_scale layout (shared with
        # the CK/asm GEMMs): 32-row stripes of 8 k-groups, returned flat as
        # (pad256(rows), pad8(K//32)). The kernels index one row per 32-row
        # stripe, so view it as (pad256(rows)//32, pad8(K//32)*32) -- taken off
        # the shuffled tensor's own shape, which is padded on both dims.
        # M < 32 stays un-shuffled (M, K//32) row-major.
        if M >= 32:
            xs = shuffle_scale(x_scales[:M])
            x_scales_shuffled = xs.view(-1, xs.shape[1] * 32)
        else:
            x_scales_shuffled = x_scales[:M].contiguous()
        ws = shuffle_scale(w_scales)
        w_scales_shuffled = ws.view(-1, ws.shape[1] * 32)
    else:
        x_scales_shuffled = x_scales[:M]
        w_scales_shuffled = w_scales

    if shuffle_weight_fg:
        # aiter.ops.shuffle.shuffle_weight (layout=(16, 16)) returns the (N, K)
        # shuffled weight, byte-identical on both arches; reshape to the
        # (N//16, K*16) layout the kernel consumes
        w_shuffed = shuffle_weight(w).reshape(w.shape[0] // 16, w.shape[1] * 16)
    else:
        w_shuffed = w

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda")
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return (
        x,
        w,
        w_shuffed,
        x_scales[:M],
        w_scales,
        x_scales_shuffled,
        w_scales_shuffled,
        out_dtype,
        y,
    )


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def run_torch(x, w, x_scales, w_scales, dtype):
    # First convert the x and w inputs to f32.
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)


@pytest.mark.parametrize("M, N, K", get_shapes(k_mult=32))
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("shuffle_weight_scales", [True, False])
@pytest.mark.parametrize("skip_reduce", [True, False])
@pytest.mark.parametrize("impl", ["triton", "gluon"])
def test_gemm_afp4_wfp4(
    M: int,
    N: int,
    K: int,
    output,
    shuffle_weight_scales,
    skip_reduce,
    impl,
):
    if impl == "gluon" and not arch_info.is_gluon_avail():
        pytest.skip("Gluon implementation is not supported on this GPU.")
    dtype = torch.bfloat16
    # TODO(brunomazzotti): Fix gluon instr shape then enable gluon tests conditionally on 950
    if impl == "gluon":
        pytest.skip("Gluon tests temporarily disabled.")

    if impl == "gluon" and shuffle_weight_scales:
        pytest.skip("Gluon kernel does not have a preshuffled implementation.")

    if UNSHUFFLED_TRITON_ONLY and (impl != "triton" or shuffle_weight_scales):
        pytest.skip(f"Only the unshuffled Triton path is supported on {DEVICE_ARCH}.")

    if shuffle_weight_scales:
        if N % 32 > 0:
            pytest.skip(
                f"N = {N} is not divisible by 32, skip this test for preshuffled weight/scales tests"
            )
        elif K % 256 > 0:
            pytest.skip(
                f"K = {K} is not divisible by 256, skip this test for preshuffled weight/scales tests"
            )

    (
        x,
        w,
        w_triton,
        x_scales,
        w_scales,
        x_scales_triton,
        w_scales_triton,
        _out_dtype,
        y,
    ) = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        dtype,
        layout="TN",
        output=output,
        shuffle_scales_fg=shuffle_weight_scales,
        shuffle_weight_fg=shuffle_weight_scales,
    )

    torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)

    if shuffle_weight_scales:
        triton_out = gemm_afp4wfp4_preshuffle(
            x,
            w_triton,
            x_scales_triton,
            w_scales_triton,
            dtype,
            y,
            skip_reduce=skip_reduce,
        )
    else:
        if impl == "triton":
            fn = triton_gemm_afp4wfp4
        elif impl == "gluon":
            fn = functools.partial(triton_gemm_afp4wfp4, backend="gluon")
        else:
            raise ValueError(f"Unknown implementation: {impl}")
        triton_out = fn(
            x,
            w_triton,
            x_scales_triton,
            w_scales_triton,
            dtype,
            y,
            skip_reduce=skip_reduce,
        )

    if triton_out.dim() == 3:
        triton_out = triton_out.sum(dim=0).to(dtype)

    if UNSHUFFLED_TRITON_ONLY:
        # The emulated dot_scaled result can differ from the reference by one
        # ULP of the output dtype after rounding.
        triton.testing.assert_close(
            torch_out, triton_out, atol=1e-2, rtol=torch.finfo(dtype).eps
        )
    else:
        triton.testing.assert_close(torch_out, triton_out)


@pytest.mark.parametrize("M, N, K", [(1, 10240, 8192), (64, 8192, 28672)])
def test_gemm_afp4_wfp4_preshuffle_splitk(M: int, N: int, K: int):
    if UNSHUFFLED_TRITON_ONLY:
        pytest.skip(f"Preshuffled MXFP4 GEMM is not supported on {DEVICE_ARCH}.")
    dtype = torch.bfloat16
    (
        x,
        w,
        w_triton,
        x_scales,
        w_scales,
        x_scales_triton,
        w_scales_triton,
        _out_dtype,
        y,
    ) = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        dtype,
        layout="TN",
        output=True,
        shuffle_scales_fg=True,
        shuffle_weight_fg=True,
    )

    # _get_config doubles K itself, so pass packed bytes as the wrapper does.
    config, _ = _get_afp4wfp4_config(M, N, K // 2, True, backend="triton")
    config = dict(config)
    config["NUM_KSPLIT"] = 4

    triton_out = gemm_afp4wfp4_preshuffle(
        x,
        w_triton,
        x_scales_triton,
        w_scales_triton,
        dtype,
        y,
        config=config,
    )

    torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)

    triton.testing.assert_close(torch_out, triton_out)


@pytest.mark.parametrize("M, N, K", get_shapes(k_mult=32))
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", ["TN"])  # "NN", "NT"
@pytest.mark.parametrize("output", [True, False])
def test_gemm_mxfp4_preshuffled_gfx1250(
    M: int,
    N: int,
    K: int,
    dtype,
    layout,
    output,
):
    if DEVICE_ARCH not in ("gfx1250", "gfx1250-strict"):
        pytest.skip("Preshuffled gfx1250 kernel only supported on gfx1250")

    if N % 32 > 0:
        pytest.skip(
            f"N = {N} is not divisible by 32, skip this test for preshuffled weight/scales tests"
        )
    if K % 256 > 0:
        pytest.skip(
            f"K = {K} is not divisible by 256, skip this test for preshuffled weight/scales tests"
        )

    (
        x,
        w,
        w_preshuf,
        x_scales,
        w_scales,
        x_scales_shuffled,
        w_scales_shuffled,
        _out_dtype,
        y,
    ) = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        dtype,
        layout=layout,
        output=output,
        shuffle_scales_fg=True,
        shuffle_weight_fg=True,
    )

    torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)

    triton_out = gemm_afp4wfp4_preshuffle(
        x,
        w_preshuf,
        x_scales_shuffled,
        w_scales_shuffled,
        dtype,
        y if y is not None else torch.empty_like(torch_out),
    )

    triton.testing.assert_close(torch_out, triton_out)


# gfx950 Gluon ping-pong kernel (config kernel_type "compute_bound"). The shapes have
# partial M and N tiles, one K step and odd numbers of K steps.
PINGPONG_SHAPES = [
    (16, 256, 256),
    (128, 3072, 1536),
    (512, 3072, 1536),
    (256, 256, 512),
    (512, 768, 1024),
    (1024, 1024, 4096),
    (256, 256, 768),
    (256, 256, 1280),
    (272, 528, 1024),
    (48, 304, 2816),
]


class _LaunchCount:
    """Counts the launches of a Gluon or Triton kernel and runs the kernel."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.count = 0

    def __getitem__(self, grid):
        self.count += 1
        return self.kernel[grid]


def _install_launch_counters(monkeypatch):
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import (
        gemm_afp4wfp4 as kernel_module,
    )

    kernel_map = kernel_module._KERNEL_MAP
    preshuffle_map = kernel_module._PRESHUFFLE_KERNEL_MAP
    launches = {
        "default": _LaunchCount(kernel_map["bandwidth_bound"]),
        "pingpong": _LaunchCount(kernel_map["compute_bound"]),
        "preshuffle_pingpong": _LaunchCount(preshuffle_map["compute_bound"]),
        "triton_preshuffle": _LaunchCount(
            gemm_afp4wfp4_module._triton_gemm_afp4wfp4_preshuffle_kernel
        ),
    }
    monkeypatch.setitem(kernel_map, "bandwidth_bound", launches["default"])
    monkeypatch.setitem(kernel_map, "compute_bound", launches["pingpong"])
    monkeypatch.setitem(
        preshuffle_map, "compute_bound", launches["preshuffle_pingpong"]
    )
    monkeypatch.setattr(
        gemm_afp4wfp4_module,
        "_triton_gemm_afp4wfp4_preshuffle_kernel",
        launches["triton_preshuffle"],
    )
    return launches


@pytest.fixture
def gluon_launches(monkeypatch):
    if DEVICE_ARCH != "gfx950":
        pytest.skip("The gfx950 Gluon kernels need gfx950.")
    return _install_launch_counters(monkeypatch)


def _skip_unless_pingpong():
    # The same check as the wrapper's dispatch: arch, Triton build and XCD count.
    reason = gemm_afp4wfp4_module._pingpong_env_unsupported()
    if reason is not None:
        pytest.skip(f"The ping-pong kernel cannot run here: {reason}.")


def _pingpong_inputs(M, N, K):
    x, w, _, x_scales, w_scales, _, _, _, _ = generate_gemm_afp4wfp4_inputs(
        M, N, K, torch.bfloat16, output=False
    )
    config, _ = _get_afp4wfp4_config(M, N, K // 2, backend="gluon")
    config["kernel_type"] = "compute_bound"
    return x, w, x_scales, w_scales, config


@pytest.mark.parametrize("M, N, K", PINGPONG_SHAPES)
def test_gemm_afp4wfp4_gluon_pingpong(M, N, K, gluon_launches):
    _skip_unless_pingpong()
    x, w, x_scales, w_scales, config = _pingpong_inputs(M, N, K)
    torch_out = run_torch(x, w, x_scales, w_scales, torch.bfloat16)

    # A NaN-filled y with a padded row stride: every element must be written in place.
    y_pad = torch.full((M, N + 16), float("nan"), dtype=torch.bfloat16, device="cuda")
    y = y_pad[:, :N]
    out = triton_gemm_afp4wfp4(
        x, w, x_scales, w_scales, torch.bfloat16, y, config=config, backend="gluon"
    )
    again = triton_gemm_afp4wfp4(
        x, w, x_scales, w_scales, torch.bfloat16, config=config, backend="gluon"
    )

    assert gluon_launches["pingpong"].count == 2, "the ping-pong kernel did not run"
    assert gluon_launches["default"].count == 0
    assert out.data_ptr() == y.data_ptr()
    assert not torch.isnan(out).any(), "output has unwritten elements"
    assert torch.isnan(y_pad[:, N:]).all(), "the kernel wrote outside y"
    torch.testing.assert_close(out, torch_out, atol=1e-2, rtol=0)
    assert torch.equal(out, again), "two runs differ"


@pytest.mark.parametrize(
    "M, N, K, case",
    [
        (256, 256, 288, "K % 256 != 0"),
        (304, 512, 512, "fp16 output"),
        (256, 256, 512, "row-major scales"),
        (300, 512, 512, "M % 16 != 0"),
    ],
)
def test_gemm_afp4wfp4_gluon_pingpong_fallback(M, N, K, case, gluon_launches):
    _skip_unless_pingpong()
    x, w, x_scales, w_scales, config = _pingpong_inputs(M, N, K)
    dtype = torch.float16 if case == "fp16 output" else torch.bfloat16
    if case == "row-major scales":
        x_scales = x_scales.contiguous()
    torch_out = run_torch(x, w, x_scales, w_scales, dtype)

    out = triton_gemm_afp4wfp4(
        x, w, x_scales, w_scales, dtype, config=config, backend="gluon"
    )

    assert gluon_launches["pingpong"].count == 0
    assert gluon_launches["default"].count == 1
    torch.testing.assert_close(out, torch_out, atol=1e-2, rtol=0)


def test_gemm_afp4wfp4_pingpong_selection(monkeypatch):
    _skip_unless_pingpong()
    M, N, K = 256, 256, 512
    x, w, x_scales, w_scales, config = _pingpong_inputs(M, N, K)

    def reason(**changes):
        args = {
            "x": x,
            "w": w,
            "x_scales": x_scales,
            "w_scales": w_scales,
            "y": None,
            "dtype": torch.bfloat16,
            "config": config,
        }
        args.update(changes)
        return gemm_afp4wfp4_module._pingpong_unsupported(**args)

    padded_x = torch.zeros((M, K // 2 + 16), dtype=torch.uint8, device="cuda")
    misaligned_x = torch.zeros(M * K // 2 + 16, dtype=torch.uint8, device="cuda")
    misaligned_x = misaligned_x[1 : 1 + M * K // 2].view(M, K // 2)

    assert reason() is None
    assert reason(dtype=torch.float16) == "output dtype or shape"
    assert (
        reason(x_scales=x_scales.contiguous()) == "scales not contiguous along M and N"
    )
    assert reason(x=padded_x[:, : K // 2]) == "x or w not contiguous"
    assert reason(x=misaligned_x) == "alignment"
    assert reason(config=dict(config, NUM_KSPLIT=2)) == "config"
    assert reason(config=dict(config, BLOCK_SIZE_M=128)) == "config"
    assert reason(config=dict(config, num_warps=4)) == "config"

    monkeypatch.setattr(gemm_afp4wfp4_module, "get_num_xcds", lambda default=8: 4)
    assert reason() == "partition mode"
    monkeypatch.setattr(gemm_afp4wfp4_module, "_pingpong_compiler_ok", lambda: False)
    assert reason() == "Triton build not validated"
    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx1250")
    assert reason() == "arch"


def test_gemm_afp4wfp4_gluon_default_unchanged(gluon_launches):
    if DEVICE_ARCH != "gfx950":
        pytest.skip("The gfx950 Gluon kernels need gfx950.")
    M, N, K = 256, 256, 512
    x, w, x_scales, w_scales, config = _pingpong_inputs(M, N, K)
    torch_out = run_torch(x, w, x_scales, w_scales, torch.bfloat16)

    # No kernel_type: the default Gluon kernel. kernel_type with backend="triton": Triton.
    gluon_out = triton_gemm_afp4wfp4(
        x, w, x_scales, w_scales, torch.bfloat16, backend="gluon"
    )
    triton_out = triton_gemm_afp4wfp4(
        x, w, x_scales, w_scales, torch.bfloat16, config=config, backend="triton"
    )

    assert gluon_launches["default"].count == 1
    assert gluon_launches["pingpong"].count == 0
    torch.testing.assert_close(gluon_out, torch_out, atol=1e-2, rtol=0)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=0)


def test_gemm_afp4wfp4_gluon_pingpong_compile(gluon_launches):
    _skip_unless_pingpong()
    M, N, K = 256, 256, 512
    x, w, x_scales, w_scales, config = _pingpong_inputs(M, N, K)
    fn = functools.partial(
        triton_gemm_afp4wfp4, dtype=torch.bfloat16, config=config, backend="gluon"
    )

    # The wrapper serializes its config with json.dumps, which is a graph break, so the
    # test does not use fullgraph=True.
    eager = fn(x, w, x_scales, w_scales)
    compiled = torch.compile(fn)(x, w, x_scales, w_scales)

    assert gluon_launches["pingpong"].count == 2
    assert torch.equal(compiled, eager)


def test_gemm_afp4wfp4_gluon_unvalidated_triton_build(monkeypatch):
    # On a Triton build without warp_pipeline_stage (Triton 3.6), the kernel module
    # must import, and the compute_bound dispatch must use the default Gluon kernel.
    if DEVICE_ARCH != "gfx950":
        pytest.skip("The gfx950 Gluon kernels need gfx950.")
    name = "aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_afp4wfp4"
    # Import the real module first, so that monkeypatch restores it after the test.
    importlib.import_module(name)
    package = importlib.import_module(name.rsplit(".", 1)[0])
    amd = importlib.import_module("triton.experimental.gluon.language.amd")
    monkeypatch.delattr(amd, "warp_pipeline_stage", raising=False)
    monkeypatch.setattr(gemm_afp4wfp4_module, "_pingpong_compiler_ok", lambda: False)
    monkeypatch.delitem(sys.modules, name)
    monkeypatch.delattr(package, name.rsplit(".", 1)[1])
    kernel_module = importlib.import_module(name)
    launches = _install_launch_counters(monkeypatch)
    x, w, x_scales, w_scales, config = _pingpong_inputs(256, 256, 512)

    out = triton_gemm_afp4wfp4(
        x, w, x_scales, w_scales, torch.bfloat16, config=config, backend="gluon"
    )

    assert kernel_module.warp_pipeline_stage is None
    assert launches["default"].count == 1
    assert launches["pingpong"].count == 0
    torch.testing.assert_close(
        out, run_torch(x, w, x_scales, w_scales, torch.bfloat16), atol=1e-2, rtol=0
    )


def test_gemm_afp4wfp4_pingpong_unknown_xcd_count(monkeypatch, gluon_launches):
    _skip_unless_pingpong()

    def no_chip_info(default=8):  # device_info.get_num_xcds without chip_info
        if default is None:
            raise RuntimeError("chip_info cannot be imported")
        return default

    monkeypatch.setattr(gemm_afp4wfp4_module, "get_num_xcds", no_chip_info)
    with pytest.raises(pytest.skip.Exception):
        _skip_unless_pingpong()
    x, w, x_scales, w_scales, config = _pingpong_inputs(256, 256, 512)

    out = triton_gemm_afp4wfp4(
        x, w, x_scales, w_scales, torch.bfloat16, config=config, backend="gluon"
    )

    assert gluon_launches["pingpong"].count == 0
    assert gluon_launches["default"].count == 1
    torch.testing.assert_close(
        out, run_torch(x, w, x_scales, w_scales, torch.bfloat16), atol=1e-2, rtol=0
    )


# gfx950 Gluon preshuffled ping-pong kernel (gemm_afp4wfp4_preshuffle with
# backend="gluon" and kernel_type "compute_bound"). N is a multiple of 16. The
# shapes have partial M and N tiles, one K step and odd numbers of K steps.
PRESHUFFLE_PINGPONG_SHAPES = [
    (32, 16, 512),
    (32, 48, 768),
    (48, 304, 2816),
    (128, 3072, 1536),
    (256, 256, 256),
    (256, 256, 1280),
    (272, 528, 1024),
    (300, 512, 512),
    (512, 768, 1024),
    (2048, 1040, 2048),
]


def _preshuffle_inputs(M, N, K):
    x, w, w_ps, x_scales, w_scales, xs_ps, ws_ps, _, _ = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        torch.bfloat16,
        output=False,
        shuffle_weight_fg=True,
        shuffle_scales_fg=True,
    )
    return x, w, w_ps, x_scales, w_scales, xs_ps, ws_ps


def _preshuffle_pingpong_config():
    # A gfx950 gluon config (M <= 4096, N = 4096, K = 8192) with kernel_type
    # "compute_bound", as an explicit config for other shapes.
    config, _ = _get_afp4wfp4_config(4096, 4096, 8192 // 2, True, backend="gluon")
    config["kernel_type"] = "compute_bound"
    return config


def _assert_only_triton_preshuffle(launches):
    assert launches["triton_preshuffle"].count == 1
    assert launches["preshuffle_pingpong"].count == 0
    assert launches["default"].count == 0
    assert launches["pingpong"].count == 0


@pytest.mark.parametrize("M, N, K", PRESHUFFLE_PINGPONG_SHAPES)
def test_gemm_afp4wfp4_preshuffle_gluon_pingpong(M, N, K, gluon_launches):
    _skip_unless_pingpong()
    x, w, w_ps, x_scales, w_scales, xs_ps, ws_ps = _preshuffle_inputs(M, N, K)
    config = _preshuffle_pingpong_config()
    torch_out = run_torch(x, w, x_scales, w_scales, torch.bfloat16)

    # A NaN-filled y with padded rows and row stride: every element must be
    # written in place, and nothing outside y.
    y_pad = torch.full(
        (M + 16, N + 16), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    y = y_pad[:M, :N]
    out = gemm_afp4wfp4_preshuffle(
        x, w_ps, xs_ps, ws_ps, torch.bfloat16, y, config=config, backend="gluon"
    )
    again = gemm_afp4wfp4_preshuffle(
        x, w_ps, xs_ps, ws_ps, torch.bfloat16, config=config, backend="gluon"
    )

    assert gluon_launches["preshuffle_pingpong"].count == 2
    assert gluon_launches["triton_preshuffle"].count == 0
    assert gluon_launches["default"].count == 0
    assert gluon_launches["pingpong"].count == 0
    assert out.data_ptr() == y.data_ptr()
    assert not torch.isnan(out).any(), "output has unwritten elements"
    assert torch.isnan(y_pad[M:]).all(), "the kernel wrote outside y"
    assert torch.isnan(y_pad[:, N:]).all(), "the kernel wrote outside y"
    torch.testing.assert_close(out, torch_out, atol=1e-2, rtol=0)
    assert torch.equal(out, again), "two runs differ"

    if M % 16 == 0:
        # The unshuffled ping-pong kernel gives the same bits: the fp32 sums of
        # these inputs are exact, so the K order of the MFMAs does not matter.
        _, _, _, _, plain_config = _pingpong_inputs(M, N, K)
        plain = triton_gemm_afp4wfp4(
            x,
            w,
            x_scales,
            w_scales,
            torch.bfloat16,
            config=plain_config,
            backend="gluon",
        )
        assert gluon_launches["pingpong"].count == 1
        assert torch.equal(out, plain)


def test_gemm_afp4wfp4_preshuffle_gluon_pingpong_compile(gluon_launches):
    _skip_unless_pingpong()
    M, N, K = 256, 256, 512
    x, _, w_ps, _, _, xs_ps, ws_ps = _preshuffle_inputs(M, N, K)
    fn = functools.partial(
        gemm_afp4wfp4_preshuffle,
        dtype=torch.bfloat16,
        config=_preshuffle_pingpong_config(),
        backend="gluon",
    )

    # Inductor cannot compile a Gluon kernel, so the launch runs eagerly after a
    # graph break.
    eager = fn(x, w_ps, xs_ps, ws_ps)
    compiled = torch.compile(fn)(x, w_ps, xs_ps, ws_ps)

    assert gluon_launches["preshuffle_pingpong"].count == 2
    assert torch.equal(compiled, eager)


@pytest.mark.parametrize(
    "M, N, K, case",
    [
        (16, 256, 512, "M < 32"),
        (256, 256, 768, "fp16 output"),
        (256, 256, 512, "bandwidth_bound"),
        (256, 256, 512, "x not contiguous"),
        (256, 256, 512, "Triton build not validated"),
        (256, 256, 512, "partition mode"),
    ],
)
def test_gemm_afp4wfp4_preshuffle_gluon_fallback(
    M, N, K, case, gluon_launches, monkeypatch
):
    _skip_unless_pingpong()
    x, w, w_ps, x_scales, w_scales, xs_ps, ws_ps = _preshuffle_inputs(M, N, K)
    dtype = torch.float16 if case == "fp16 output" else torch.bfloat16
    config = _preshuffle_pingpong_config()
    if case == "bandwidth_bound":
        # The bandwidth_bound entry of a gfx950 gluon config file.
        config, _ = _get_afp4wfp4_config(2048, 4096, 8192 // 2, True, backend="gluon")
        assert config["kernel_type"] == "bandwidth_bound"
    elif case == "x not contiguous":
        x_pad = torch.zeros((M, K // 2 + 16), dtype=torch.uint8, device="cuda")
        x_pad[:, : K // 2] = x
        x = x_pad[:, : K // 2]
    elif case == "Triton build not validated":
        monkeypatch.setattr(
            gemm_afp4wfp4_module, "_pingpong_compiler_ok", lambda: False
        )
    elif case == "partition mode":
        monkeypatch.setattr(gemm_afp4wfp4_module, "get_num_xcds", lambda default=8: 4)
    torch_out = run_torch(x, w, x_scales, w_scales, dtype)

    out = gemm_afp4wfp4_preshuffle(
        x, w_ps, xs_ps, ws_ps, dtype, config=config, backend="gluon"
    )

    # The fallback is the triton kernel with its own config: the default path.
    _assert_only_triton_preshuffle(gluon_launches)
    default = gemm_afp4wfp4_preshuffle(x, w_ps, xs_ps, ws_ps, dtype)
    assert torch.equal(out, default)
    torch.testing.assert_close(out, torch_out, atol=1e-2, rtol=0)


@pytest.mark.parametrize(
    "M, N, K, kernel_type",
    [
        (2048, 4096, 8192, "bandwidth_bound"),
        (2049, 4096, 8192, "compute_bound"),
        (4096, 4096, 8192, "compute_bound"),
        (4097, 4096, 8192, "bandwidth_bound"),
        (4096, 8192, 8192, "bandwidth_bound"),
        (4097, 8192, 8192, "compute_bound"),
        (8192, 8192, 8192, "compute_bound"),
        (8193, 8192, 8192, "bandwidth_bound"),
        (8192, 16384, 65536, "bandwidth_bound"),
        (8193, 16384, 65536, "compute_bound"),
        (16384, 16384, 65536, "compute_bound"),
        (8192, 8192, 16384, "bandwidth_bound"),
        (16, 8192, 8192, "bandwidth_bound"),
    ],
)
def test_gemm_afp4wfp4_preshuffle_gluon_config(M, N, K, kernel_type):
    # The gfx950 gluon configs select the ping-pong kernel only for the measured
    # shapes and M ranges. Other shapes use the triton kernel.
    if DEVICE_ARCH != "gfx950":
        pytest.skip("The gfx950 Gluon kernels need gfx950.")
    config, _ = _get_afp4wfp4_config(M, N, K // 2, True, backend="gluon")

    assert config["kernel_type"] == kernel_type
    if kernel_type == "compute_bound":
        assert config == _preshuffle_pingpong_config()


@pytest.mark.parametrize(
    "M, N, K, pingpong",
    [
        (2048, 4096, 8192, False),
        (4096, 4096, 8192, True),
        (8192, 8192, 8192, True),
    ],
)
def test_gemm_afp4wfp4_preshuffle_gluon_config_dispatch(
    M, N, K, pingpong, gluon_launches
):
    _skip_unless_pingpong()
    x, w, w_ps, x_scales, w_scales, xs_ps, ws_ps = _preshuffle_inputs(M, N, K)
    torch_out = run_torch(x, w, x_scales, w_scales, torch.bfloat16)

    out = gemm_afp4wfp4_preshuffle(
        x, w_ps, xs_ps, ws_ps, torch.bfloat16, backend="gluon"
    )

    assert gluon_launches["preshuffle_pingpong"].count == int(pingpong)
    assert gluon_launches["triton_preshuffle"].count == int(not pingpong)
    assert gluon_launches["default"].count == 0
    assert gluon_launches["pingpong"].count == 0
    torch.testing.assert_close(out, torch_out, atol=1e-2, rtol=0)


def test_gemm_afp4wfp4_preshuffle_default_backend_unchanged(
    gluon_launches, monkeypatch
):
    # Without backend="gluon", gfx950 runs the triton kernel, also for a shape
    # that a gluon config gives to the ping-pong kernel.
    M, N, K = 4096, 4096, 8192
    x, w, w_ps, x_scales, w_scales, xs_ps, ws_ps = _preshuffle_inputs(M, N, K)
    # The gluon branch checks the ping-pong limits first, also on a Triton build
    # that cannot run the ping-pong kernel.
    monkeypatch.setattr(
        gemm_afp4wfp4_module,
        "_preshuffle_pingpong_unsupported",
        lambda *args, **kwargs: pytest.fail("the default backend took the gluon path"),
    )

    out = gemm_afp4wfp4_preshuffle(x, w_ps, xs_ps, ws_ps, torch.bfloat16)

    _assert_only_triton_preshuffle(gluon_launches)
    triton_out = gemm_afp4wfp4_preshuffle(
        x, w_ps, xs_ps, ws_ps, torch.bfloat16, backend="triton"
    )
    assert torch.equal(out, triton_out)
    torch.testing.assert_close(
        out, run_torch(x, w, x_scales, w_scales, torch.bfloat16), atol=1e-2, rtol=0
    )


@pytest.mark.parametrize("pad", [0x00, 0xFF])
def test_gemm_afp4wfp4_preshuffle_gluon_scale_padding(pad, gluon_launches):
    # The scales of rows past M and N are padding. A poisoned pad (0xFF is NaN
    # in E8M0) must not change the output.
    _skip_unless_pingpong()
    M, N, K = 272, 528, 1024
    x, _, w_ps, x_scales, w_scales, xs_ps, ws_ps = _preshuffle_inputs(M, N, K)
    config = _preshuffle_pingpong_config()

    def padded_scales(scales, value):
        # shuffle_scale of the scales padded to 256 rows with value.
        rows = scales.shape[0]
        padded = torch.full(
            ((rows + 255) // 256 * 256, K // 32),
            value,
            dtype=torch.uint8,
            device="cuda",
        )
        padded[:rows] = scales
        shuffled = shuffle_scale(padded)
        return shuffled.view(-1, shuffled.shape[1] * 32)

    # shuffle_scale pads with 0x7F (E8M0 1.0).
    assert torch.equal(padded_scales(x_scales, 0x7F), xs_ps)
    assert torch.equal(padded_scales(w_scales, 0x7F), ws_ps)
    expected = gemm_afp4wfp4_preshuffle(
        x, w_ps, xs_ps, ws_ps, torch.bfloat16, config=config, backend="gluon"
    )
    out = gemm_afp4wfp4_preshuffle(
        x,
        w_ps,
        padded_scales(x_scales, pad),
        padded_scales(w_scales, pad),
        torch.bfloat16,
        config=config,
        backend="gluon",
    )

    assert gluon_launches["preshuffle_pingpong"].count == 2
    assert not torch.isnan(out).any()
    assert torch.equal(out, expected)


def test_gemm_afp4wfp4_preshuffle_pingpong_selection(monkeypatch):
    _skip_unless_pingpong()
    M, N, K = 256, 256, 512
    x, w, w_ps, x_scales, _, xs_ps, ws_ps = _preshuffle_inputs(M, N, K)
    config = _preshuffle_pingpong_config()

    def reason(**changes):
        args = {
            "x": x,
            "w": w_ps,
            "x_scales": xs_ps,
            "w_scales": ws_ps,
            "y": None,
            "dtype": torch.bfloat16,
            "config": config,
        }
        args.update(changes)
        return gemm_afp4wfp4_module._preshuffle_pingpong_unsupported(**args)

    padded_x = torch.zeros((M, K // 2 + 16), dtype=torch.uint8, device="cuda")
    misaligned_x = torch.zeros(M * K // 2 + 16, dtype=torch.uint8, device="cuda")
    misaligned_x = misaligned_x[1 : 1 + M * K // 2].view(M, K // 2)
    y_pad = torch.empty((M, N + 8), dtype=torch.bfloat16, device="cuda")

    assert reason() is None
    assert reason(x=x[:16], x_scales=x_scales[:16]) == "M < 32"
    assert reason(dtype=torch.float16) == "output dtype or shape"
    assert reason(y=y_pad[:, :N]) == "outer stride"
    assert reason(y=y_pad[:, : N - 16]) == "output dtype or shape"
    assert reason(w=w) == "input shape"  # an unshuffled weight
    assert reason(x_scales=x_scales) == "input shape"  # unshuffled scales
    assert reason(x=padded_x[:, : K // 2]) == "x or w not contiguous"
    assert reason(w_scales=ws_ps.T.contiguous().T) == "scales not contiguous along K"
    assert reason(x=misaligned_x) == "alignment"
    assert reason(x=x.view(torch.int16)) == "input dtype"
    # Too few scale stripes, a scale row stride that is not a multiple of 16 and
    # a y that is not contiguous along N.
    assert reason(x_scales=xs_ps[:7]) == "input shape"
    assert reason(w_scales=ws_ps[:7]) == "input shape"
    scales_pad = torch.zeros((8, K + 8), dtype=torch.uint8, device="cuda")
    assert reason(x_scales=scales_pad[:, :K]) == "outer stride"
    # One scale stripe with a row stride that does not fit in int32.
    one_stripe = torch.empty_strided(
        (1, K), (2**31, 1), dtype=torch.uint8, device="cuda"
    )
    assert reason(x=x[:32], x_scales=one_stripe) == "outer stride"
    # An output or a tensor of 2 GiB or more does not fit in the 32-bit buffer
    # offsets of the kernel.
    big_m, big_n = 16384, 65536
    big_inputs = {
        "x": torch.empty((big_m, K // 2), dtype=torch.uint8, device="cuda"),
        "w": torch.empty((big_n // 16, 8 * K), dtype=torch.uint8, device="cuda"),
        "x_scales": torch.empty((big_m // 32, K), dtype=torch.uint8, device="cuda"),
        "w_scales": torch.empty((big_n // 32, K), dtype=torch.uint8, device="cuda"),
    }
    assert reason(**big_inputs) == "output above 2 GiB"
    big = torch.empty(2**30, dtype=torch.bfloat16, device="cuda")
    assert reason(y=big[: M * N].view(M, N)) == "tensor above 2 GiB"
    del big
    y_t = torch.empty((N, M), dtype=torch.bfloat16, device="cuda").T
    assert reason(y=y_t) == "y inner stride"
    assert reason(config=dict(config, NUM_KSPLIT=2)) == "config"
    assert reason(config=dict(config, BLOCK_SIZE_M=128)) == "config"
    assert reason(config=dict(config, num_warps=4)) == "config"

    monkeypatch.setattr(gemm_afp4wfp4_module, "get_num_xcds", lambda default=8: 4)
    assert reason() == "partition mode"
    monkeypatch.setattr(gemm_afp4wfp4_module, "_pingpong_compiler_ok", lambda: False)
    assert reason() == "Triton build not validated"
    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx1250")
    assert reason() == "arch"


def test_gemm_afp4wfp4_preshuffle_backend_errors(monkeypatch):
    if DEVICE_ARCH != "gfx950":
        pytest.skip("The gfx950 Gluon kernels need gfx950.")
    x, _, w_ps, _, _, xs_ps, ws_ps = _preshuffle_inputs(256, 256, 512)
    config = _preshuffle_pingpong_config()

    with pytest.raises(AssertionError, match="Unknown backend"):
        gemm_afp4wfp4_preshuffle(x, w_ps, xs_ps, ws_ps, backend="cuda")
    with pytest.raises(AssertionError, match="Unknown kernel_type"):
        gemm_afp4wfp4_preshuffle(
            x,
            w_ps,
            xs_ps,
            ws_ps,
            config=dict(config, kernel_type="compute-bound"),
            backend="gluon",
        )
    # On gfx1250 this wrapper runs only the gluon kernel.
    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx1250")
    with pytest.raises(AssertionError, match="runs only the gluon kernel"):
        gemm_afp4wfp4_preshuffle(x, w_ps, xs_ps, ws_ps, backend="triton")
