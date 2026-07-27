# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Unit test for fused AllReduce + RMSNorm + MXFP4 quantization.

Two coverage modes:

* CI smoke (default, ~2 min): runs ``CI_SHAPES`` (one shape per dispatch
  path -- 1-stage, 2-stage, fallback) at ``stage=auto`` /
  ``emit_bf16=False``, sweeping ``tp_size in [2, 4, 8]`` so every
  production TP is covered. Picked up automatically by
  ``.github/scripts/aiter_test.sh`` which executes every multi-GPU test
  with no flags. Pass ``-t N`` to restrict to a single TP for debugging.
* Local full sweep (~50 min across TP=2/4/8): ``--full`` runs
  ``FULL_SHAPES`` x stage=both x emit_bf16=[False, True], i.e. forces
  both 1-stage and 2-stage kernels through every shape they support
  in addition to the user-facing dispatcher path. ``--full`` also
  sweeps ``tp_size in [2, 4, 8]`` unless ``-t N`` is set.

Examples:
    # CI invocation (no flags), sweeps TP in [2, 4, 8]:
    python op_tests/multigpu_tests/test_fused_ar_rms_mxfp4_quant.py

    # CI smoke at a single TP (debug):
    python op_tests/multigpu_tests/test_fused_ar_rms_mxfp4_quant.py -t 8

    # Local dense sweep, all TPs:
    python op_tests/multigpu_tests/test_fused_ar_rms_mxfp4_quant.py --full

    # Single-shape single-mode debug:
    python op_tests/multigpu_tests/test_fused_ar_rms_mxfp4_quant.py \\
        -t 4 -s 36,7168 --stage 2stage --emit-bf16
"""

import argparse
import itertools
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from aiter import dtypes
from aiter.dist.device_communicators.custom_all_reduce import (
    _validate_mxfp4_hidden_dim,
)
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import checkAllclose
from aiter.utility.fp4_utils import mxfp4_to_f32

set_start_method("spawn", force=True)

logger = logging.getLogger("aiter")


def _shape_arg(value: str) -> tuple[int, int]:
    m, n = value.split(",")
    return int(m), int(n)


def _dequant_mxfp4(x_fp4: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x = mxfp4_to_f32(x_fp4).view(x_fp4.shape[0], -1)
    scale_u8 = scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)
    scale_f32 = torch.exp2(scale_u8.to(torch.float32) - 127).repeat_interleave(
        32, dim=-1
    )
    return x * scale_f32


def test_mxfp4_hidden_dim_validation_python_check():
    for n in (32, 512, 4096, 6144, 7168, 8192, 16384):
        _validate_mxfp4_hidden_dim(n, element_size=2)

    for bad_n in (0, 1, 31, 33, 4097):
        try:
            _validate_mxfp4_hidden_dim(bad_n, element_size=2)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for n={bad_n}")

    for bad_element_size in (0, -2, 3, 5):
        try:
            _validate_mxfp4_hidden_dim(4096, element_size=bad_element_size)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"expected ValueError for element_size={bad_element_size}"
            )


def _run_rank(
    tp_size: int,
    rank: int,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    distributed_init_method: str,
    emit_bf16: bool,
    stage_override: Optional[str] = None,
):
    if stage_override is not None:
        os.environ["AITER_AR_1STAGE"] = stage_override
    elif "AITER_AR_1STAGE" in os.environ:
        del os.environ["AITER_AR_1STAGE"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, 1)

    x = x.to(device)
    residual = residual.to(device)
    weight = weight.to(device)
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm_quant,
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = tensor_model_parallel_fused_allreduce_rmsnorm_quant(
        x, residual, weight, eps, quant_type="mxfp4", emit_bf16=emit_bf16
    )
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000.0
    if emit_bf16:
        out_fp4, res_out, scale, bf16_out = result
    else:
        out_fp4, res_out, scale = result
        bf16_out = None

    destroy_model_parallel()
    destroy_distributed_environment()
    torch.cuda.empty_cache()
    return (
        out_fp4.cpu(),
        scale.cpu(),
        res_out.cpu(),
        None if bf16_out is None else bf16_out.cpu(),
        us,
    )


def _expected_path(
    shape: tuple[int, int],
    tp_size: int,
    element_size: int,
    stage_override: Optional[str],
    emit_bf16: bool,
) -> str:
    """Mirror CudaCommunicator.fused_allreduce_rmsnorm_mxfp4_quant gating."""
    M, K = shape
    total_bytes = M * K * element_size
    pack_size = 16 // element_size if element_size > 0 else 0
    block_size = K // pack_size if pack_size > 0 else 0
    use_direct_mxfp4 = (
        M <= 4
        or (K <= 4096 and M <= 32)
        or (K <= 6144 and M <= 16)
        or (K == 8192 and M <= 8)
    )
    can_1stage = (
        stage_override != "0"
        and K % 32 == 0
        and K <= 16384
        and M <= 80
        and use_direct_mxfp4
    )
    can_2stage = (
        stage_override != "1"
        and K % 32 == 0
        and pack_size > 0
        and K <= 8192
        and block_size % tp_size == 0
        and (not emit_bf16 or block_size % 32 == 0)
        and total_bytes <= 512 * 1024
    )
    if stage_override is None:
        prefer_2stage = can_2stage and tp_size == 8 and M >= 16 and K <= 6144
        if prefer_2stage:
            can_1stage = False
    if can_1stage:
        return "direct_1stage"
    if can_2stage:
        return "direct_2stage"
    return "fallback"


def test_fused_ar_rmsnorm_mxfp4_quant(
    tp_size: int,
    shape: tuple[int, int],
    dtype: torch.dtype,
    emit_bf16: bool,
    distributed_init_method: Optional[str] = None,
    stage_override: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49383"
    if distributed_init_method is None:
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())

    eps = 1e-6
    x = torch.randn(shape, dtype=dtype)
    residual = torch.randn(shape, dtype=dtype)
    weight = torch.randn((shape[-1],), dtype=dtype)
    ref_residual = x * tp_size + residual
    ref = F.rms_norm(ref_residual, (shape[-1],), weight=weight, eps=eps)
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    ref_fp4, ref_scale = dynamic_mxfp4_quant(ref.cuda())
    ref_dequant = _dequant_mxfp4(ref_fp4, ref_scale).cpu()

    with Pool(processes=tp_size) as pool:
        futures = [
            pool.apply_async(
                _run_rank,
                args=(
                    tp_size,
                    rank,
                    x,
                    residual,
                    weight,
                    eps,
                    distributed_init_method,
                    emit_bf16,
                    stage_override,
                ),
            )
            for rank in range(tp_size)
        ]
        results = [future.get() for future in futures]

    max_dequant_err = 0.0
    max_bf16_err = 0.0
    max_residual_err = 0.0
    for rank, (out_fp4, scale, res_out, bf16_out, us) in enumerate(results):
        assert out_fp4.shape == ref_fp4.shape
        assert scale.shape == ref_scale.shape
        out_dequant = _dequant_mxfp4(out_fp4.cuda(), scale.cuda()).cpu()
        max_dequant_err = max(
            max_dequant_err,
            checkAllclose(
                ref_dequant,
                out_dequant,
                msg=f"mxfp4 dequant {shape=} {emit_bf16=} {stage_override=} "
                f"rank={rank} {us:.2f}us",
                atol=1.5,
                rtol=5e-1,
            ),
        )
        max_residual_err = max(
            max_residual_err,
            checkAllclose(
                ref_residual,
                res_out,
                msg=f"residual output {shape=} {emit_bf16=} {stage_override=} "
                f"rank={rank} {us:.2f}us",
                atol=1e-2,
                rtol=1e-2,
            ),
        )
        if emit_bf16:
            assert bf16_out is not None
            max_bf16_err = max(
                max_bf16_err,
                checkAllclose(
                    ref,
                    bf16_out,
                    msg=f"bf16 side output {shape=} {stage_override=} "
                    f"rank={rank} {us:.2f}us",
                    atol=1e-2,
                    rtol=1e-2,
                ),
            )
    element_size = torch.tensor([], dtype=dtype).element_size()
    expected_path = _expected_path(
        shape, tp_size, element_size, stage_override, emit_bf16
    )
    return {
        "shape": shape,
        "tp_size": tp_size,
        "dtype": str(dtype).replace("torch.", ""),
        "emit_bf16": emit_bf16,
        "stage_override": stage_override or "auto",
        "expected_path": expected_path,
        "min_us": min(us for *_, us in results),
        "max_us": max(us for *_, us in results),
        "mxfp4_dequant_err": max_dequant_err,
        "residual_err": max_residual_err,
        "bf16_err": max_bf16_err,
    }


# Mix of decode-sized (1-stage), prefill-sized within the 512 KiB 2-stage
# budget, and an oversized shape that should fall back to fused
# AR+RMSNorm + dynamic_mxfp4_quant.
CI_SHAPES = [
    # Covers all three dispatch paths at the default --tp-size 8:
    (1, 4096),  # direct_1stage (K<=4096 -> M<=32 at TP=8)
    (16, 7168),  # direct_2stage (16*7168*2 = 224 KiB <= 512 KiB; 7168%8 == 0)
    (128, 7168),  # fallback (1.75 MiB exceeds the 2-stage 512 KiB budget)
]

FULL_SHAPES = [
    (1, 4096),  # 1-stage
    (8, 7168),  # 1-stage
    (32, 7168),  # 2-stage at TP=8 (32*7168*2 = 448 KiB <= 512 KiB)
    (56, 7168),  # fallback at TP=8 (56*7168*2 = 784 KiB > 512 KiB)
    (16, 4096),  # 2-stage (block_size=512, 16*4096*2 = 128 KiB)
    (32, 8192),  # 2-stage at TP=8 (block_size=1024, 32*8192*2 = 512 KiB)
    (64, 7168),  # fallback (64*7168*2 = 896 KiB > 512 KiB)
    (128, 7168),  # fallback
]

DEFAULT_SHAPES = CI_SHAPES


def main():
    parser = argparse.ArgumentParser(
        description="Test fused AR+RMSNorm+MXFP4 quantization"
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=["fp16", "bf16"],
        default="bf16",
    )
    parser.add_argument(
        "-t",
        "--tp-size",
        type=int,
        default=None,
        help=(
            "Tensor-parallel world size. When unset, the test sweeps "
            "tp_size in [2, 4, 8] so the CI smoke covers every production "
            "TP. Set explicitly (e.g. -t 8) for single-TP debugging."
        ),
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=_shape_arg,
        nargs="*",
        default=None,
        help="shape(s), e.g. -s 1,4096 8,7168 32,7168",
    )
    parser.add_argument(
        "--emit-bf16",
        action="store_true",
        help=(
            "Also exercise the bf16 side-output path. Default (CI smoke) only "
            "runs emit_bf16=False; --full implies both."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["auto", "1stage", "2stage", "both"],
        default="auto",
        help=(
            "Which kernel variant(s) to exercise. 'auto' (default, CI smoke) "
            "runs only the user-facing dispatcher path. 'both' additionally "
            "forces 1-stage and 2-stage via AITER_AR_1STAGE; --full implies "
            "'both'."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run the dense local coverage matrix: FULL_SHAPES x stage=both x "
            "emit_bf16=[False, True]. Intended for local validation; the CI "
            "lane keeps the smoke defaults (CI_SHAPES x stage=auto x "
            "emit_bf16=False) to stay under ~1 minute."
        ),
    )
    parser.add_argument(
        "--skip-python-check",
        action="store_true",
        help="Skip the non-distributed hidden-dim validation test.",
    )
    args = parser.parse_args()

    if not args.skip_python_check:
        test_mxfp4_hidden_dim_validation_python_check()

    dtype = dtypes.d_dtypes[args.dtype]

    if args.full:
        shapes = args.shape if args.shape is not None else FULL_SHAPES
        emit_bf16_values = [False, True]
        stage_overrides = [None, "1", "0"]
    else:
        shapes = args.shape if args.shape is not None else DEFAULT_SHAPES
        emit_bf16_values = [False, True] if args.emit_bf16 else [False]
        stage_to_env = {"auto": None, "1stage": "1", "2stage": "0"}
        if args.stage == "both":
            stage_overrides = [None, "1", "0"]
        else:
            stage_overrides = [stage_to_env[args.stage]]

    tp_sizes = [args.tp_size] if args.tp_size is not None else [2, 4, 8]

    rows = []
    for tp_size, shape, emit_bf16, stage_override in itertools.product(
        tp_sizes, shapes, emit_bf16_values, stage_overrides
    ):
        element_size = torch.tensor([], dtype=dtype).element_size()
        expected = _expected_path(
            shape, tp_size, element_size, stage_override, emit_bf16
        )
        if expected == "fallback" and stage_override in ("1", "0"):
            # When forcing a specific kernel, only exercise shapes that the
            # kernel actually supports. Fallbacks under override would just
            # silently re-test the unfused reference path.
            continue
        rows.append(
            test_fused_ar_rmsnorm_mxfp4_quant(
                tp_size,
                shape,
                dtype,
                emit_bf16,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
                stage_override=stage_override,
            )
        )

    for row in rows:
        logger.info("fused AR+RMSNorm+MXFP4 row: %s", row)


if __name__ == "__main__":
    freeze_support()
    main()
