# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One function per tunable kernel, registered with @kernel(...), which only carries metadata.

The tuning scripts only make, prune and select configs. Everything about running and timing a
kernel (inputs, the launch with a config, derived keys, the output check) is in its benchmark,
op_tests/op_benchmarks/triton/bench_<op>.py, as make_bench_case(). Every kernel function has the
same layout: shape values -> make_case -> optional should_skip -> return (make_case, should_skip).

  make_case(kwargs_rot_bufs)  the benchmark's make_bench_case for this shape and backend. It
                              returns (launch, rot_bufs, flops, mem, check); harness.py times
                              launch over rot_bufs and runs check before a config can win.
  should_skip(config)         True for a candidate the kernel would reject: its own asserts, or a
                              buffer count its wrapper clamps so the candidate only repeats another
                              one. None when the kernel has no rule of its own.

K is the logical element count everywhere, as the input generators take it: fp4 tensors hold
K // 2 bytes and the config files of fp4 families are named by the logical K.
torch / aiter / op_tests are imported inside the functions only.

To add a kernel, give its benchmark a make_bench_case (see bench_gemm_a16w16.py), copy the
nearest function below, and register it with its config family and where its gluon path exists:

    @kernel("GEMM-FOO", gluon_archs=("gfx1250",), gluon_default_archs=("gfx1250",))
    def gemm_foo(shape, backend):
        M, N, K = shape["M"], shape["N"], shape["K"]
        make_case = bench_case("bench_gemm_foo", M, N, K, "TN", backend)

        def should_skip(config):
            return config["BLOCK_SIZE_K"] % 64 != 0  # whatever the kernel asserts

        return make_case, should_skip
"""

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from _utils import cdiv, specialized_filename

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 6)))
KERNELS = {}


@dataclass(frozen=True)
class KernelSpec:
    name: str
    # the kernel function: (shape, backend) -> (make_case, should_skip)
    setup: Callable
    # the config family, exactly as the kernel's _get_config passes it
    config_name: str
    # ("B", "M", "N", "K") for batched kernels
    dims: tuple = ("M", "N", "K")
    # bits per element of A and B as they sit in LDS (scales not counted)
    bits: tuple = (16, 16)
    # archs where backend="gluon" can run at all
    gluon_archs: tuple = ()
    # archs where the wrapper picks gluon when no backend is given
    gluon_default_archs: tuple = ()
    # False: the wrapper takes no backend= and picks by arch
    backend_kwarg: bool = True
    # M bounds the kernel's _get_config passes (None: the file's M_BOUNDS or the standard list)
    bounds: tuple | None = None
    # DEFAULT.json keys the gluon wrapper never reads
    gluon_ignored_keys: tuple = ()
    # gluon-only candidate lists, for keys that mean something else there
    gluon_candidates: dict = field(default_factory=dict)
    # (config) -> (ctas_m, ctas_n): how the kernel splits BLOCK_SIZE_M/N over a CTA cluster
    cta_split: Callable | None = None

    def default_backend(self, arch):
        return "gluon" if arch in self.gluon_default_archs else "triton"

    def supports(self, arch, backend):
        if not self.backend_kwarg:
            return backend == self.default_backend(arch)
        return backend == "triton" or arch in self.gluon_archs


def kernel(config_name, **metadata):
    """Register the decorated kernel function under its own name."""

    def register(setup):
        KERNELS[setup.__name__] = KernelSpec(
            setup.__name__, setup, config_name, **metadata
        )
        return setup

    return register


def get_spec(name):
    if name not in KERNELS:
        raise argparse.ArgumentTypeError(
            f"Unknown kernel {name!r}; choose from {list(KERNELS)}"
        )
    return KERNELS[name]


def ensure_repo_on_path():  # op_tests is not an installed package
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def check_backend(spec, arch, backend):
    if not spec.supports(arch, backend):
        raise SystemExit(f"{spec.name}: backend '{backend}' is not available on {arch}")


def resolve_installed(spec, shape, backend):
    """The installed config for this shape (raw JSON bucket) and is_tuned, through the real loader."""
    from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

    return get_gemm_config(
        spec.config_name,
        shape["M"],
        shape["N"],
        shape["K"],
        bounds=spec.bounds,
        backend=backend,
        B=shape.get("B"),
    )


def seed_table(spec, backend, shape_nk):
    """(path, table) to install into: this shape's own file, else a new DEFAULT_FALLBACK table."""
    from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

    config_dir = resolve_config_dir("gemm", spec.config_name, backend=backend)
    path = f"{config_dir}/{specialized_filename(spec.config_name, shape_nk)}"
    table = load_config_json(path, required=False)
    if table is not None:
        return path, table
    default_path = f"{config_dir}/DEFAULT.json"
    default = load_config_json(default_path, required=False)
    if default is None:
        raise SystemExit(f"Required config file doesn't exist: {default_path}")
    return default_path, {
        "DEFAULT_FALLBACK": True,
        **{k: v for k, v in default.items() if k == "M_BOUNDS"},
    }


def family_bounds(spec, backend, shape_nk):
    """The M bounds the loader walks for this shape: the kernel's explicit bounds, else the
    served file's M_BOUNDS, else the standard list."""
    from aiter.ops.triton.utils.gemm_config_utils import STANDARD_M_BOUNDS

    if spec.bounds:
        return tuple(spec.bounds)
    return tuple(
        seed_table(spec, backend, shape_nk)[1].get("M_BOUNDS") or STANDARD_M_BOUNDS
    )


def bench_case(bench, *args, **kwargs):
    """make_case of a kernel: make_bench_case of op_tests/op_benchmarks/triton/<bench>.py."""

    def make_case(kwargs_rot_bufs):
        import importlib

        module = importlib.import_module(f"op_tests.op_benchmarks.triton.{bench}")
        return module.make_bench_case(kwargs_rot_bufs, *args, **kwargs)

    return make_case


@kernel(
    "BATCHED_GEMM-A16W16",
    dims=("B", "M", "N", "K"),
    gluon_archs=("gfx1250",),
    gluon_default_archs=("gfx1250",),
    gluon_ignored_keys=("GROUP_SIZE_M", "matrix_instr_nonkdim"),
)
def batched_gemm_bf16(shape, backend):
    B, M, N, K = shape["B"], shape["M"], shape["N"], shape["K"]
    make_case = bench_case("bench_batched_gemm_bf16", B, M, N, K, "TN", backend)

    def should_skip(config):
        if backend != "gluon":
            return False
        block_k = config["BLOCK_SIZE_K"]
        num_buffers = config.get("NUM_BUFFERS", 2)
        k_tiles = cdiv(cdiv(K, config.get("NUM_KSPLIT", 1)), block_k)
        if block_k % 32:
            return True  # 16x16x32 instructions
        if config.get("kernel_type") == "compute_bound":
            if num_buffers < 2 or k_tiles < 4:
                return True  # compute_bound needs a 2-deep pipeline and 4 K tiles
            return num_buffers > k_tiles - 2  # the wrapper clamps it there anyway
        return num_buffers > k_tiles  # the wrapper clamps it there anyway

    return make_case, should_skip


@kernel(
    "GEMM-A16W16",
    gluon_archs=("gfx1250",),
    gluon_default_archs=("gfx1250",),
    gluon_ignored_keys=("persistent",),
)
def gemm_a16w16(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]
    make_case = bench_case("bench_gemm_a16w16", M, N, K, "TN", backend)

    def should_skip(config):
        if backend != "gluon":
            return False
        block_k = config["BLOCK_K"]
        num_buffers = config.get("NUM_BUFFERS", 2)
        k_tiles = cdiv(K, block_k)
        if block_k % 32:
            return True  # 16x16x32 instructions
        if config.get("kernel_type") == "compute_bound":
            if num_buffers < 2 or k_tiles < 4:
                return True  # compute_bound needs a 2-deep pipeline and 4 K tiles
            return num_buffers > k_tiles - 2  # the wrapper clamps it there anyway
        return num_buffers > k_tiles  # the wrapper clamps it there anyway

    return make_case, should_skip


@kernel(
    "GEMM-A16W16-PERSISTENT", gluon_archs=("gfx1250",), gluon_default_archs=("gfx1250",)
)
def gemm_a16w16_persistent(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]
    make_case = bench_case("bench_gemm_a16w16", M, N, K, "TN", backend, persistent=True)

    def should_skip(config):
        if backend == "triton":
            return config.get("NUM_KSPLIT", 1) != 1  # the wrapper forces NUM_KSPLIT=1
        block_k = config["BLOCK_K"]
        num_buffers = config.get("NUM_BUFFERS", 2)
        k_tiles = cdiv(cdiv(K, config.get("NUM_KSPLIT", 1)), block_k)
        if block_k % 32:
            return True  # 16x16x32 instructions
        if num_buffers < 2:
            return True  # the wrapper raises it to 2 anyway
        return num_buffers > k_tiles + 1  # the wrapper clamps it there anyway

    return make_case, should_skip


@kernel("GEMM-A16W16-ATOMIC", backend_kwarg=False)
def gemm_a16w16_atomic(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]
    return bench_case("bench_gemm_a16w16", M, N, K, "TN", backend, atomic=True), None


@kernel("GEMM-A16W16-gated", backend_kwarg=False, bounds=(64, 128, 256, 512, 2048))
def gemm_a16w16_gated(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]
    return bench_case("bench_gemm_a16w16_gated", M, N, K, "TN"), None


@kernel("GEMM-A16W8_BLOCKSCALE", bits=(16, 8), backend_kwarg=False)
def gemm_a16w8_blockscale(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]
    make_case = bench_case("bench_gemm_a16w8_blockscale", M, N, K, "TN")

    def should_skip(config):
        return config["BLOCK_SIZE_K"] != 128  # the scale block is 128 wide

    return make_case, should_skip


@kernel("GEMM-A16W8_BLOCKSCALE_PRESHUFFLED", bits=(16, 8), backend_kwarg=False)
def gemm_a16w8_blockscale_preshuffle(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]
    make_case = bench_case(
        "bench_gemm_a16w8_blockscale", M, N, K, "TN", preshuffle=True
    )

    def should_skip(config):
        return config["BLOCK_SIZE_K"] != 128  # the scale block is 128 wide

    return make_case, should_skip


@kernel("GEMM-A16WFP4", bits=(16, 4), backend_kwarg=False)
def gemm_a16wfp4(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]
    return bench_case("bench_gemm_a16wfp4", M, N, K, "TN", False), None


# The kernel dequantizes the int4 weight tile to 16 bits before it goes to LDS.
@kernel("GEMM-A16W4", bits=(16, 16), backend_kwarg=False)
def gemm_a16w4(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]

    def make_case(kwargs_rot_bufs):
        import torch

        from op_tests.op_benchmarks.triton.bench_gemm_a16w4 import make_bench_case

        # Group 128; a K that is not a multiple of 128 gets per-channel scales.
        return make_bench_case(kwargs_rot_bufs, M, N, K, torch.bfloat16, 128)

    def should_skip(config):
        # tl.dot needs blocks of 16 or more in each dimension.
        block = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")
        return min(config[key] for key in block) < 16

    return make_case, should_skip


@kernel("GEMM-A8W8", bits=(8, 8), gluon_archs=("gfx950",))
def gemm_a8w8(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]
    return bench_case("bench_gemm_a8w8", M, N, K, "TN", False, backend), None


@kernel(
    "GEMM-A8W8_BLOCKSCALE",
    bits=(8, 8),
    gluon_archs=("gfx950", "gfx1250"),
    gluon_default_archs=("gfx1250",),
    gluon_candidates={
        "num_stages": [2, 3, 4, 6]
    },  # gfx950 gluon reads it as NUM_STAGES, at least 2
)
def gemm_a8w8_blockscale(shape, backend):
    from aiter.ops.triton.utils._triton import arch_info

    arch = arch_info.get_arch()
    M, N, K = shape["M"], shape["N"], shape["K"]
    make_case = bench_case("bench_gemm_a8w8_blockscale", M, N, K, "TN", False, backend)

    def should_skip(config):
        if config["BLOCK_SIZE_K"] != 128:
            return True  # the scale block is 128 wide
        if (
            backend == "gluon" and arch == "gfx950"
        ):  # that kernel has three fixed tiles, 4 warps
            tile = (config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"])
            return (
                tile not in {(64, 128), (128, 128), (128, 256)}
                or config["num_warps"] != 4
            )
        return False

    return make_case, should_skip


@kernel(
    "GEMM-A8W8_BLOCKSCALE_PRESHUFFLED",
    bits=(8, 8),
    gluon_archs=("gfx1250",),
    gluon_default_archs=("gfx1250",),
)
def gemm_a8w8_blockscale_preshuffle(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]
    make_case = bench_case("bench_gemm_a8w8_blockscale", M, N, K, "TN", True, backend)

    def should_skip(config):
        return config["BLOCK_SIZE_K"] != 128  # the scale block is 128 wide

    return make_case, should_skip


@kernel("GEMM-A8W8_PER_TOKEN_SCALE", bits=(8, 8), backend_kwarg=False)
def gemm_a8w8_per_token_scale(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]
    return bench_case("bench_gemm_a8w8_per_token_scale", M, N, K, "TN"), None


@kernel("GEMM-A8WFP4", bits=(8, 4), backend_kwarg=False)
def gemm_a8wfp4(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]
    return bench_case("bench_gemm_a8wfp4", M, N, K, "TN"), None


@kernel("GEMM-AFP4WFP4", bits=(4, 4), gluon_archs=("gfx950",))
def gemm_afp4wfp4(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]
    make_case = bench_case("bench_gemm_afp4wfp4", M, N, K, "TN", False, backend)

    def should_skip(config):
        if backend != "triton":
            return False
        return (
            config["BLOCK_SIZE_K"] < 128
        )  # the triton wrapper raises it to 128 anyway

    return make_case, should_skip


# gemm_afp4wfp4_pre_quant is gemm_a16wfp4 with atomic_add=True, so it reads GEMM-A16WFP4 too.
@kernel("GEMM-A16WFP4", bits=(16, 4), backend_kwarg=False)
def gemm_afp4wfp4_pre_quant_atomic(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]
    return bench_case("bench_gemm_a16wfp4", M, N, K, "TN", True), None


def mxfp4_cta_split(config):
    """num_ctas makes BLOCK_SIZE_M/N a cluster tile; each CTA computes its share of it."""
    if config.get("num_ctas", 1) == 1:
        return 1, 1
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_mxfp4 import (
        cluster_shape,
    )

    return cluster_shape(
        config["num_ctas"], config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"]
    )


@kernel(
    "GEMM-AFP4WFP4_PRESHUFFLED",
    bits=(4, 4),
    cta_split=mxfp4_cta_split,
    gluon_archs=("gfx1250",),
    gluon_default_archs=("gfx1250",),
    backend_kwarg=False,
    bounds=(4, 8, 16, 31, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
)
def gemm_afp4wfp4_preshuffle(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]
    # the preshuffle wrapper takes no backend: it picks it by arch
    make_case = bench_case("bench_gemm_afp4wfp4", M, N, K, "TN", True, None)

    def should_skip(config):
        block_m, block_n, block_k = (
            config["BLOCK_SIZE_M"],
            config["BLOCK_SIZE_N"],
            config["BLOCK_SIZE_K"],
        )
        ctas_m, ctas_n = mxfp4_cta_split(config)
        cta_m, cta_n = (
            block_m // ctas_m,
            block_n // ctas_n,
        )  # one CTA's share of the tile
        if (M < 32) != (block_m <= 16):
            return True  # x_scales are un-shuffled below M=32: BLOCK_SIZE_M <= 16 there, >= 32 above
        if M >= 32 and (cta_m < 32 or cta_m % 32):
            return True  # preshuffled scales come in 32-row stripes per CTA
        if cta_n % 32:
            return True  # w_scales come in 32-row stripes per CTA
        if backend == "triton":
            return block_k < 128  # the triton wrapper raises it to 128 anyway
        if block_k % 256 or K % block_k:
            return True  # the TDM kernel needs 256-wide K tiles that divide K
        if config["num_warps"] not in (2, 4, 8):
            return True  # the only warp layouts the kernel has
        k_tiles = K // block_k
        return config["NUM_BUFFERS"] > k_tiles  # the wrapper clamps it there anyway

    return make_case, should_skip


@kernel(
    "GEMM-AFP8WFP8_PRESHUFFLED",
    bits=(8, 8),
    gluon_archs=("gfx1250",),
    gluon_default_archs=("gfx1250",),
)
def gemm_afp8wfp8_preshuffle(shape, backend):
    M, N, K = shape["M"], shape["N"], shape["K"]

    def make_case(kwargs_rot_bufs):
        import torch

        from op_tests.op_benchmarks.triton.bench_gemm_afp8wfp8 import make_bench_case

        # 1x128 x scales, transposed as per_group_quant_hip(transpose_scale=True) makes them
        return make_bench_case(
            kwargs_rot_bufs, M, N, K, torch.bfloat16, True, backend, 128, True
        )

    def should_skip(config):
        if backend != "gluon":
            return False
        block_k = config["BLOCK_SIZE_K"]
        num_buffers = config["NUM_BUFFERS"]
        k_tiles = cdiv(cdiv(K, config.get("NUM_KSPLIT", 1)), block_k)
        if block_k % 128:
            return True  # 128-wide K tiles
        if config["BLOCK_SIZE_N"] % 16:
            return True  # the preshuffled B descriptor is indexed in N // 16
        if num_buffers < 2:
            return True  # the wrapper raises it to 2 anyway
        if num_buffers > k_tiles + 1:
            return True  # the wrapper clamps it there anyway
        clusters = config.get("CTAS_M", 1) * config.get("CTAS_N", 1) > 1
        return (
            clusters and config.get("kernel_type") != "bandwidth_bound"
        )  # clusters only there

    return make_case, should_skip
