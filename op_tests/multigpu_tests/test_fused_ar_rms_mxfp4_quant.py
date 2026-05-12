# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

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
):
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
        tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant,
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant(
        x, residual, weight, eps, emit_bf16=emit_bf16
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


def test_fused_ar_rmsnorm_mxfp4_quant(
    tp_size: int,
    shape: tuple[int, int],
    dtype: torch.dtype,
    emit_bf16: bool,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49383"
    if distributed_init_method is None:
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port()
        )

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
                msg=f"mxfp4 dequant {shape=} {emit_bf16=} rank={rank} {us:.2f}us",
                atol=1.5,
                rtol=5e-1,
            ),
        )
        max_residual_err = max(
            max_residual_err,
            checkAllclose(
                ref_residual,
                res_out,
                msg=f"residual output {shape=} {emit_bf16=} rank={rank} {us:.2f}us",
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
                    msg=f"bf16 side output {shape=} rank={rank} {us:.2f}us",
                    atol=1e-2,
                    rtol=1e-2,
                ),
            )
    expected_path = "direct_1stage" if shape[0] <= 80 else "fallback"
    return {
        "shape": shape,
        "tp_size": tp_size,
        "dtype": str(dtype).replace("torch.", ""),
        "emit_bf16": emit_bf16,
        "expected_path": expected_path,
        "min_us": min(us for *_, us in results),
        "max_us": max(us for *_, us in results),
        "mxfp4_dequant_err": max_dequant_err,
        "residual_err": max_residual_err,
        "bf16_err": max_bf16_err,
    }


DEFAULT_SHAPES = [
    (1, 4096),
    (8, 7168),
    (32, 7168),
    (56, 7168),
    (128, 7168),
    (32, 8192),
]


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
    parser.add_argument("-t", "--tp-size", type=int, default=8)
    parser.add_argument(
        "-s",
        "--shape",
        type=_shape_arg,
        nargs="*",
        default=None,
        help="shape(s), e.g. -s 1,4096 8,7168 32,7168",
    )
    parser.add_argument("--emit-bf16", action="store_true")
    parser.add_argument(
        "--skip-python-check",
        action="store_true",
        help="Skip the non-distributed hidden-dim validation test.",
    )
    args = parser.parse_args()

    if not args.skip_python_check:
        test_mxfp4_hidden_dim_validation_python_check()

    dtype = dtypes.d_dtypes[args.dtype]
    shapes = args.shape if args.shape is not None else DEFAULT_SHAPES
    emit_bf16_values = [args.emit_bf16] if args.emit_bf16 else [False, True]

    rows = []
    for shape, emit_bf16 in itertools.product(shapes, emit_bf16_values):
        rows.append(
            test_fused_ar_rmsnorm_mxfp4_quant(
                args.tp_size,
                shape,
                dtype,
                emit_bf16,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
            )
        )

    for row in rows:
        logger.info("fused AR+RMSNorm+MXFP4 row: %s", row)


if __name__ == "__main__":
    freeze_support()
    main()
