# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native gfx1201 FlyDSL flash-attention wrapper op test."""

from __future__ import annotations

import argparse
import itertools

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_flash_attn_func, flydsl_fp8_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx1201"]
SEED = 0

# (label, batch, seq_q, seq_kv, heads, head_dim, causal). These are attention
# consumer paths: D64 uses the 256x64 tile, D128 covers the KV tail and the
# BLOCK_M=256/BLOCK_N=64 selector, Flux is the production self-attention path.
SHAPES = [
    ("d64_256", 1, 256, 256, 8, 64, False),
    ("d128_cross_tail", 1, 2048, 1400, 8, 128, False),
    ("causal_d128", 1, 512, 512, 8, 128, True),
    ("flux_d128", 1, 4096, 4096, 24, 128, False),
]


def _make_qkv(
    batch: int,
    seq_q: int,
    seq_kv: int,
    num_heads: int,
    head_dim: int,
    *,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(SEED)
    q = torch.randn(
        (batch, seq_q, num_heads, head_dim),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        (batch, seq_kv, num_heads, head_dim),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn(
        (batch, seq_kv, num_heads, head_dim),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    return q, k, v


def run_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """FP32 SDPA reference for BSHD tensors; it is never timed."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        is_causal=causal,
    )
    return out.transpose(1, 2).contiguous()


def _check_quality(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    head_dim: int,
    minimum: float,
    mean: float,
) -> tuple[float, float]:
    cosine = F.cosine_similarity(
        actual.float().reshape(-1, head_dim),
        expected.float().reshape(-1, head_dim),
        dim=1,
    )
    min_cos = cosine.min().item()
    mean_cos = cosine.mean().item()
    assert min_cos > minimum, f"min_cos={min_cos:.6f}"
    assert mean_cos > mean, f"mean_cos={mean_cos:.6f}"
    return min_cos, mean_cos


def _expect_value_error(fn, match: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert match in str(exc), f"expected {match!r}, got {exc!r}"
    else:
        raise AssertionError(f"expected ValueError containing {match!r}")


def _exercise_wrapper_contracts() -> None:
    """Check durable flash-attention wrapper boundaries outside timed rows."""
    # D512 exceeds 64 KiB even after the selector retries BLOCK_N=32.
    q, k, v = _make_qkv(1, 128, 128, 1, 512)
    _expect_value_error(
        lambda: flydsl_flash_attn_func(q, k, v),
        "65536-byte hardware limit",
    )

    # The public stream and preallocated-output path must compose.
    q, k, v = _make_qkv(1, 97, 97, 2, 128)
    launch_stream = torch.cuda.Stream(device=q.device)
    stream_out = torch.full_like(q, float("nan"))
    streamed = flydsl_flash_attn_func(q, k, v, stream=launch_stream, out=stream_out)
    launch_stream.synchronize()
    _check_quality(
        stream_out,
        run_torch(q, k, v, causal=False),
        head_dim=128,
        minimum=0.99,
        mean=0.999,
    )
    assert streamed.data_ptr() == stream_out.data_ptr()

    if torch.cuda.device_count() < 2:
        aiter.logger.warning("two-GPU FlyDSL FMHA check skipped: fewer than two GPUs")
        return

    # The FP8 consumer must select the input device, not the current device.
    device1 = torch.device("cuda", 1)
    with torch.cuda.device(device1):
        q, k, v = _make_qkv(1, 128, 128, 2, 128, device=device1)
        q8, k8, v8, sq, sk, sv = flydsl_fp8_quant(q, k, v)
        ref = run_torch(q, k, v, causal=False)
    torch.cuda.set_device(0)
    assert torch.cuda.current_device() == 0
    out = flydsl_flash_attn_func(q8, k8, v8, q_descale=sq, k_descale=sk, v_descale=sv)
    torch.cuda.synchronize(device1)
    assert out.device == device1
    _check_quality(out, ref, head_dim=128, minimum=0.97, mean=0.994)


@benchmark()
def run_flydsl_fmha(
    shape,
    batch,
    seq_q,
    seq_kv,
    num_heads,
    head_dim,
    dtype,
    causal,
):
    """Time BF16 and already-quantized FP8 attention consumers."""
    q, k, v = _make_qkv(batch, seq_q, seq_kv, num_heads, head_dim)
    ref = run_torch(q, k, v, causal=causal)
    out = torch.full_like(q, float("nan"))
    is_fp8 = dtype == "fp8"

    if is_fp8:
        # Quantization is intentionally outside timing: this test owns the
        # attention consumer contract; quantizer coverage is in its own op test.
        q8, k8, v8, sq, sk, sv = flydsl_fp8_quant(q, k, v)
        candidates = {
            "flydsl": lambda: flydsl_flash_attn_func(
                q8,
                k8,
                v8,
                causal=causal,
                q_descale=sq,
                k_descale=sk,
                v_descale=sv,
                out=out,
            )
        }
        minimum, mean = 0.97, 0.994
        element_bytes = 1
    else:
        candidates = {
            "flydsl": lambda: flydsl_flash_attn_func(q, k, v, causal=causal, out=out)
        }
        minimum, mean = 0.99, 0.999
        element_bytes = q.element_size()

    # QK^T + P@V costs 4*B*H*Sq*Skv*D FLOPs; a causal self-attention triangle
    # performs approximately half this work.  I/O is Q, K, V, and output.
    flops = 4.0 * batch * num_heads * seq_q * seq_kv * head_dim
    if causal:
        flops *= 0.5
    nbytes = (
        batch
        * num_heads
        * head_dim
        * ((seq_q + 2 * seq_kv) * element_bytes + seq_q * out.element_size())
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        result, us = run_perftest(fn)
        assert result.data_ptr() == out.data_ptr(), f"{name} did not use out buffer"
        err = checkAllclose(
            ref.to(dtypes.fp32),
            result.to(dtypes.fp32),
            rtol=2e-1 if is_fp8 else 2e-2,
            atol=2e-1 if is_fp8 else 2e-2,
            msg=f"{name}: {shape}",
        )
        _check_quality(result, ref, head_dim=head_dim, minimum=minimum, mean=mean)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main() -> None:
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "native FlyDSL FMHA unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Native gfx1201 FlyDSL FMHA config",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        choices=["bf16", "fp8"],
        nargs="*",
        default=["bf16", "fp8"],
        help="Attention paths to sweep.",
    )
    parser.add_argument(
        "-s",
        "--shape",
        choices=[shape[0] for shape in SHAPES],
        nargs="*",
        default=[shape[0] for shape in SHAPES],
        help="Attention shape groups to sweep.",
    )
    args = parser.parse_args()

    _exercise_wrapper_contracts()
    rows = []
    for dtype, config in itertools.product(args.dtype, SHAPES):
        shape, batch, seq_q, seq_kv, num_heads, head_dim, causal = config
        if shape not in args.shape:
            continue
        rows.append(
            run_flydsl_fmha(
                shape,
                batch,
                seq_q,
                seq_kv,
                num_heads,
                head_dim,
                dtype,
                causal,
            )
        )
    aiter.logger.info(
        "flydsl_fmha_gfx1201 summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
