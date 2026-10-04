# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One function per tunable kernel, registered with @kernel(...), which only carries metadata.

Every kernel function has the same layout: imports -> inputs -> call -> optional should_skip
-> return (call, inputs, should_skip).

  call(config, *inputs)  one launch of the public wrapper. config is a fresh deep copy per
                         launch (the wrappers mutate it), or None, which makes the wrapper use
                         the installed config (that is the baseline). Output resets and derived
                         keys belong here. The tensors are explicit arguments because the
                         benchmark rotates cold copies of them.
  inputs                 the tensors call takes; outputs are preallocated so nothing is
                         allocated per launch.
  should_skip(config)    True for a candidate the kernel would reject: its own asserts, or a
                         buffer count its wrapper clamps so the candidate only repeats another
                         one. None when the kernel has no rule of its own.

K is the logical element count everywhere, as the input generators take it: fp4 tensors hold
K // 2 bytes and the config files of fp4 families are named by the logical K.
torch / aiter / op_tests are imported inside the functions only.

To add a kernel, copy the nearest function below, change its inputs and launch, and register
it with its config family and where its gluon path exists:

    @kernel("GEMM-FOO", gluon_archs=("gfx1250",), gluon_default_archs=("gfx1250",))
    def gemm_foo(shape, backend):
        import torch

        from aiter.ops.triton.gemm.basic.gemm_foo import gemm_foo as gemm
        from op_tests.triton_tests.gemm.basic.test_gemm_foo import generate_gemm_foo_inputs

        M, N, K = shape["M"], shape["N"], shape["K"]
        x, w, y = generate_gemm_foo_inputs(M, N, K, torch.bfloat16, output=True)

        def call(config, x, w, y):
            return gemm(x, w, dtype=torch.bfloat16, y=y, config=config, backend=backend)

        def should_skip(config):
            return config["BLOCK_SIZE_K"] % 64 != 0  # whatever the kernel asserts

        return call, (x, w, y), should_skip
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
    # the kernel function: (shape, backend) -> (call, inputs, should_skip)
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
    # (num_warps) -> (warps_m, warps_n): how the gluon kernel spreads one CTA's tile over its
    # warps, for the register check; None (or a None result) leaves that check off
    warp_split: Callable | None = None

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


def flops(shape):
    return 2.0 * shape.get("B", 1) * shape["M"] * shape["N"] * shape["K"]


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


def add_splitk_block_size(config, K):
    """Mutates config: the triton a16w16-style kernels read SPLITK_BLOCK_SIZE, which the JSON lacks.

    Nothing to add for the baseline (config is None: the wrapper loads the installed config).
    """
    if config is not None and "NUM_KSPLIT" in config:
        config["SPLITK_BLOCK_SIZE"] = cdiv(K, config["NUM_KSPLIT"])
    return config


@kernel(
    "BATCHED_GEMM-A16W16",
    dims=("B", "M", "N", "K"),
    gluon_archs=("gfx1250",),
    gluon_default_archs=("gfx1250",),
    gluon_ignored_keys=("GROUP_SIZE_M", "matrix_instr_nonkdim"),
)
def batched_gemm_bf16(shape, backend):
    import torch

    from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import (
        batched_gemm_bf16 as gemm,
    )
    from op_tests.triton_tests.gemm.batched.test_batched_gemm_bf16 import (
        generate_batched_gemm_a16w16_inputs,
    )

    B, M, N, K = shape["B"], shape["M"], shape["N"], shape["K"]
    x, w, bias, y = generate_batched_gemm_a16w16_inputs(
        B, M, N, K, torch.bfloat16, output=True
    )

    def call(config, x, w, bias, y):
        if backend == "triton":
            add_splitk_block_size(config, K)
        return gemm(
            x, w, bias=bias, dtype=torch.bfloat16, YQ=y, config=config, backend=backend
        )

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

    return call, (x, w, bias, y), should_skip


@kernel(
    "GEMM-A16W16",
    gluon_archs=("gfx1250",),
    gluon_default_archs=("gfx1250",),
    gluon_ignored_keys=("persistent",),
)
def gemm_a16w16(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as gemm
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
        generate_gemm_a16w16_inputs,
    )

    M, N, K = shape["M"], shape["N"], shape["K"]
    x, w, bias, _, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=True, bias=True
    )

    def call(config, x, w, bias, y):
        if backend == "triton":
            add_splitk_block_size(config, K)
        return gemm(
            x, w, bias=bias, dtype=torch.bfloat16, y=y, config=config, backend=backend
        )

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

    return call, (x, w, bias, y), should_skip


@kernel(
    "GEMM-A16W16-PERSISTENT", gluon_archs=("gfx1250",), gluon_default_archs=("gfx1250",)
)
def gemm_a16w16_persistent(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16 as gemm
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
        generate_gemm_a16w16_inputs,
    )

    M, N, K = shape["M"], shape["N"], shape["K"]
    x, w, bias, _, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=True, bias=True
    )

    def call(config, x, w, bias, y):
        if backend == "triton":
            add_splitk_block_size(config, K)
        return gemm(
            x,
            w,
            bias=bias,
            dtype=torch.bfloat16,
            y=y,
            config=config,
            backend=backend,
            persistent=True,
        )

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

    return call, (x, w, bias, y), should_skip


@kernel("GEMM-A16W16-ATOMIC", backend_kwarg=False)
def gemm_a16w16_atomic(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import (
        gemm_a16w16_atomic as gemm,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
        generate_gemm_a16w16_inputs,
    )

    M, N, K = shape["M"], shape["N"], shape["K"]
    x, w, _, _, y = generate_gemm_a16w16_inputs(M, N, K, torch.bfloat16, output=True)

    def call(config, x, w, y):
        y.zero_()  # the kernel adds its split-K partials into y with atomics
        add_splitk_block_size(config, K)
        return gemm(x, w, dtype=torch.bfloat16, y=y, config=config)

    return call, (x, w, y), None


@kernel("GEMM-A16W16-gated", backend_kwarg=False, bounds=(64, 128, 256, 512, 2048))
def gemm_a16w16_gated(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16w16_gated import gemm_a16w16_gated as gemm
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w16_gated import (
        generate_gemm_a16w16_gated_inputs,
    )

    M, N, K = shape["M"], shape["N"], shape["K"]
    x, w, _, y = generate_gemm_a16w16_gated_inputs(M, N, K, torch.bfloat16, output=True)

    def call(config, x, w, y):
        return gemm(x, w, dtype=torch.bfloat16, y=y, config=config)

    return call, (x, w, y), None


@kernel("GEMM-A16W8_BLOCKSCALE", bits=(16, 8), backend_kwarg=False)
def gemm_a16w8_blockscale(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import (
        gemm_a16w8_blockscale as gemm,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w8_blockscale import (
        generate_gemm_a16w8_blockscale_inputs,
    )

    M, N, K = shape["M"], shape["N"], shape["K"]
    x, _, w, w_scale, y = generate_gemm_a16w8_blockscale_inputs(
        M, N, K, 128, 128, dtype=torch.bfloat16, output=True, shuffle=False
    )

    def call(config, x, w, w_scale, y):
        return gemm(
            x, w, w_scale, dtype=torch.bfloat16, y=y, prequant=False, config=config
        )

    def should_skip(config):
        return config["BLOCK_SIZE_K"] != 128  # the scale block is 128 wide

    return call, (x, w, w_scale, y), should_skip


@kernel("GEMM-A16W8_BLOCKSCALE_PRESHUFFLED", bits=(16, 8), backend_kwarg=False)
def gemm_a16w8_blockscale_preshuffle(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import (
        gemm_a16w8_blockscale_preshuffle as gemm,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a16w8_blockscale import (
        generate_gemm_a16w8_blockscale_inputs,
    )

    M, N, K = shape["M"], shape["N"], shape["K"]
    x, _, w, w_scale, y = generate_gemm_a16w8_blockscale_inputs(
        M, N, K, 128, 128, dtype=torch.bfloat16, output=True, shuffle=True
    )

    def call(config, x, w, w_scale, y):
        return gemm(
            x, w, w_scale, dtype=torch.bfloat16, y=y, prequant=False, config=config
        )

    def should_skip(config):
        return config["BLOCK_SIZE_K"] != 128  # the scale block is 128 wide

    return call, (x, w, w_scale, y), should_skip


@kernel("GEMM-A16WFP4", bits=(16, 4), backend_kwarg=False)
def gemm_a16wfp4(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4 as gemm
    from op_tests.triton_tests.gemm.basic.test_gemm_a16wfp4 import (
        generate_gemm_a16wfp4_inputs,
    )

    M, N, K = shape["M"], shape["N"], shape["K"]
    x, w, _, _, w_scales, _, y = generate_gemm_a16wfp4_inputs(
        M,
        N,
        K,
        output=True,
        atomic_add=False,
        dtype=torch.bfloat16,
        layout="TN",
        shuffle=False,
    )

    def call(config, x, w, w_scales, y):
        return gemm(
            x, w, w_scales, atomic_add=False, dtype=torch.bfloat16, y=y, config=config
        )

    return call, (x, w, w_scales, y), None


@kernel("GEMM-A8W8", bits=(8, 8), gluon_archs=("gfx950",))
def gemm_a8w8(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as gemm
    from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
    from aiter.ops.triton.utils.types import get_fp8_dtypes
    from op_tests.triton_tests.gemm.basic.test_gemm_a8w8 import (
        generate_gemm_a8w8_inputs,
    )

    M, N, K = shape["M"], shape["N"], shape["K"]
    _, fp8 = get_fp8_dtypes()
    x, _, w, x_scale, w_scale, _, y = generate_gemm_a8w8_inputs(
        M, N, K, in_dtype=fp8, out_dtype=torch.bfloat16, layout="TN", output=True
    )

    def call(config, x, w, x_scale, w_scale, y):
        if config is not None and backend == "triton":
            compute_splitk_params(
                config, K
            )  # mutates: derives SPLITK_BLOCK_SIZE, may shrink BK
        return gemm(
            x,
            w,
            x_scale,
            w_scale,
            bias=None,
            dtype=torch.bfloat16,
            y=y,
            config=config,
            backend=backend,
        )

    return call, (x, w, x_scale, w_scale, y), None


@kernel(
    "GEMM-A8W8_BLOCKSCALE",
    bits=(8, 8),
    gluon_archs=("gfx950", "gfx1250"),
    gluon_default_archs=("gfx1250",),
    gluon_candidates={
        "num_stages": [2, 3, 4, 6]
    },  # gfx1250 gluon reads it as NUM_BUFFERS
)
def gemm_a8w8_blockscale(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale as gemm,
    )
    from aiter.ops.triton.utils._triton import arch_info
    from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
        generate_gemm_a8w8_blockscale_inputs,
    )

    arch = arch_info.get_arch()
    M, N, K = shape["M"], shape["N"], shape["K"]
    x, _, w, _, x_scale, w_scale, y = generate_gemm_a8w8_blockscale_inputs(
        M, N, K, 128, 128, dtype=torch.bfloat16, layout="TN", output=True, shuffle=False
    )

    def call(config, x, w, x_scale, w_scale, y):
        return gemm(
            x,
            w,
            x_scale,
            w_scale,
            dtype=torch.bfloat16,
            y=y,
            config=config,
            backend=backend,
        )

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

    return call, (x, w, x_scale, w_scale, y), should_skip


@kernel(
    "GEMM-A8W8_BLOCKSCALE_PRESHUFFLED",
    bits=(8, 8),
    gluon_archs=("gfx1250",),
    gluon_default_archs=("gfx1250",),
    gluon_candidates={
        "num_stages": [2, 3, 4, 6]
    },  # gfx1250 gluon reads it as NUM_BUFFERS
)
def gemm_a8w8_blockscale_preshuffle(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale_preshuffle as gemm,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
        generate_gemm_a8w8_blockscale_inputs,
    )

    M, N, K = shape["M"], shape["N"], shape["K"]
    x, _, w, _, x_scale, w_scale, y = generate_gemm_a8w8_blockscale_inputs(
        M, N, K, 128, 128, dtype=torch.bfloat16, layout="TN", output=True, shuffle=True
    )

    def call(config, x, w, x_scale, w_scale, y):
        return gemm(
            x,
            w,
            x_scale,
            w_scale,
            dtype=torch.bfloat16,
            y=y,
            config=config,
            backend=backend,
        )

    def should_skip(config):
        return config["BLOCK_SIZE_K"] != 128  # the scale block is 128 wide

    return call, (x, w, x_scale, w_scale, y), should_skip


@kernel("GEMM-A8W8_PER_TOKEN_SCALE", bits=(8, 8), backend_kwarg=False)
def gemm_a8w8_per_token_scale(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8w8_per_token_scale import (
        gemm_a8w8_per_token_scale as gemm,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_per_token_scale import (
        generate_gemm_a8w8_per_token_scale_inputs,
    )

    M, N, K = shape["M"], shape["N"], shape["K"]
    x, w, x_scale, w_scale, y = generate_gemm_a8w8_per_token_scale_inputs(
        M, N, K, dtype=torch.bfloat16, layout="TN", output=True
    )

    def call(config, x, w, x_scale, w_scale, y):
        return gemm(x, w, x_scale, w_scale, dtype=torch.bfloat16, y=y, config=config)

    return call, (x, w, x_scale, w_scale, y), None


@kernel("GEMM-A8WFP4", bits=(8, 4), backend_kwarg=False)
def gemm_a8wfp4(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4 as gemm
    from aiter.ops.triton.utils.types import get_fp8_dtypes
    from op_tests.triton_tests.gemm.basic.test_gemm_a8wfp4 import (
        generate_gemm_a8wfp4_inputs,
    )

    M, N, K = shape["M"], shape["N"], shape["K"]
    _, fp8 = get_fp8_dtypes()
    x, w, x_scales, w_scales, _, _, y = generate_gemm_a8wfp4_inputs(
        M, N, K, fp8, torch.float16, layout="TN", output=True
    )

    def call(config, x, w, x_scales, w_scales, y):
        add_splitk_block_size(config, K)
        return gemm(x, w, y, x_scales, w_scales, dtype=torch.float16, config=config)

    return call, (x, w, x_scales, w_scales, y), None


@kernel("GEMM-AFP4WFP4", bits=(4, 4), gluon_archs=("gfx950",))
def gemm_afp4wfp4(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4 as gemm
    from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
        generate_gemm_afp4wfp4_inputs,
    )

    M, N, K = shape["M"], shape["N"], shape["K"]
    x, _, w, _, _, x_scales, w_scales, _, y = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        torch.bfloat16,
        output=True,
        shuffle_scales_fg=False,
        shuffle_weight_fg=False,
    )

    def call(config, x, w, x_scales, w_scales, y):
        return gemm(
            x,
            w,
            x_scales,
            w_scales,
            dtype=torch.bfloat16,
            y=y,
            config=config,
            backend=backend,
        )

    def should_skip(config):
        if backend != "triton":
            return False
        return (
            config["BLOCK_SIZE_K"] < 128
        )  # the triton wrapper raises it to 128 anyway

    return call, (x, w, x_scales, w_scales, y), should_skip


@kernel("GEMM-A16WFP4", bits=(16, 4), backend_kwarg=False)
def gemm_afp4wfp4_pre_quant_atomic(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4_pre_quant_atomic import (
        gemm_afp4wfp4_pre_quant as gemm,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_a16wfp4 import (
        generate_gemm_a16wfp4_inputs,
    )

    M, N, K = shape["M"], shape["N"], shape["K"]
    x, w, _, _, w_scales, _, y = generate_gemm_a16wfp4_inputs(
        M,
        N,
        K,
        output=True,
        atomic_add=True,
        dtype=torch.float32,
        layout="TN",
        shuffle=False,
    )

    def call(config, x, w, w_scales, y):
        y.zero_()  # the kernel adds its split-K partials into y with atomics
        return gemm(x, w, w_scales, dtype=torch.float32, y=y, config=config)

    return call, (x, w, w_scales, y), None


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


def mxfp4_warp_split(num_warps):
    """Warps along M and N: the warp_bases of get_gemm_afp4wfp4_preshuffle_layouts."""
    return {2: (2, 1), 4: (2, 2), 8: (4, 2)}.get(num_warps)


@kernel(
    "GEMM-AFP4WFP4_PRESHUFFLED",
    bits=(4, 4),
    cta_split=mxfp4_cta_split,
    warp_split=mxfp4_warp_split,
    gluon_archs=("gfx1250",),
    gluon_default_archs=("gfx1250",),
    backend_kwarg=False,
    bounds=(4, 8, 16, 31, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
)
def gemm_afp4wfp4_preshuffle(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
        gemm_afp4wfp4_preshuffle as gemm,
    )
    from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
        generate_gemm_afp4wfp4_inputs,
    )

    M, N, K = shape["M"], shape["N"], shape["K"]
    x, _, w, _, _, x_scales, w_scales, _, y = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        torch.bfloat16,
        output=True,
        shuffle_scales_fg=True,
        shuffle_weight_fg=True,
    )

    def call(config, x, w, x_scales, w_scales, y):
        return gemm(x, w, x_scales, w_scales, dtype=torch.bfloat16, y=y, config=config)

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

    return call, (x, w, x_scales, w_scales, y), should_skip


@kernel(
    "GEMM-AFP8WFP8_PRESHUFFLED",
    bits=(8, 8),
    gluon_archs=("gfx1250",),
    gluon_default_archs=("gfx1250",),
)
def gemm_afp8wfp8_preshuffle(shape, backend):
    import torch

    from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import (
        gemm_afp8wfp8_preshuffle as gemm,
    )
    from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
    from op_tests.triton_tests.gemm.basic.test_gemm_afp8wfp8 import generate_inputs

    M, N, K = shape["M"], shape["N"], shape["K"]
    x, _, w_kernel, _, x_scales_kernel, w_scales = generate_inputs(
        M, N, K, shuffle=True, x_scale_group_size=128
    )
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    def call(config, x, w_kernel, x_scales_kernel, w_scales, y):
        if config is not None and backend == "triton":
            compute_splitk_params(
                config, K
            )  # mutates: derives SPLITK_BLOCK_SIZE, may shrink BK
        return gemm(
            x,
            w_kernel,
            x_scales_kernel,
            w_scales,
            dtype=torch.bfloat16,
            y=y,
            config=config,
            x_scale_group_size=128,
            backend=backend,
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

    return call, (x, w_kernel, x_scales_kernel, w_scales, y), should_skip
