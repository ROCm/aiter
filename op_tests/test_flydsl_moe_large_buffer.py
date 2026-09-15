# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""GEMM2 gather correctness across the 4 GiB activation-buffer boundary."""

import argparse
import functools
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
from aiter.ops.flydsl.fhmoe import flydsl_fhmoe_stage2
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
)
from aiter.ops.shuffle import (
    shuffle_scale,
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils


def run_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    active_tokens: torch.Tensor,
    token: int,
    topk: int,
) -> torch.Tensor:
    ref = torch.zeros((token, weight.shape[0]), device=x.device)
    ref[active_tokens] = (x @ weight.T).view(-1, topk, weight.shape[0]).mean(dim=1)
    return ref


@benchmark()
def test_large_buffer(
    token: int,
    a_dtype: str,
    mode: str = "atomic",
    storage: str = "own",
    family: str = "moe",
    b_dtype: str = "fp4",
    loads: tuple[str, ...] = ("sync", "async"),
    persist: bool = False,
    waves_per_eu: int | None = None,
    cu_num_mul: int = 1,
    inter_dim_pad: int = 0,
) -> dict[str, object]:
    # Match DSV4's top-k and intermediate width; a small output width keeps the
    # reference cheap. Only selected routes are active, as in expert parallelism.
    topk, inter_dim, model_dim, block_m = 6, 3072, 128, 64
    if a_dtype == "fp4":
        topk, inter_dim = 8, 2048
    assert 5 <= token < (1 << 24), "token must fit the packed routing format"
    torch.manual_seed(42)
    device = "cuda"
    pack = 1 if a_dtype == "fp8" else 2
    active_tokens = torch.tensor(
        [0, 1, token // 2, token - 2, token - 1], device=device
    )
    slots = torch.arange(topk, device=device)
    payload_rows = (active_tokens[:, None] * topk + slots).flatten()
    active_rows = payload_rows.numel()
    x = torch.randn((active_rows, inter_dim), device=device) / 4
    if inter_dim_pad:
        x[:, -inter_dim_pad:] = 0
    if a_dtype == "fp8":
        xq, xs = per_1x32_f8_scale_f8_quant(
            x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
        x_dequant = xq.float()
    else:
        xq, xs = per_1x32_f4_quant(x)
        x_dequant = fp4_utils.mxfp4_to_f32(xq)
    x_dequant *= fp4_utils.e8m0_to_f32(xs).repeat_interleave(32, -1)

    # Allocate the real large buffer, but initialize only rows reachable by this
    # dispatch. Row 0 is included, so invalid routes have a safe initialized row.
    payload = torch.empty(
        (token * topk, inter_dim // pack), dtype=torch.uint8, device=device
    )
    if storage == "offset_view":
        # The view's pointer includes a >4 GiB offset; its own range is small.
        offset = (1 << 32) + 256
        backing = torch.empty(
            offset + payload.numel(), dtype=torch.uint8, device=device
        )
        payload = backing[offset:].view_as(payload)
    payload.index_copy_(0, payload_rows, xq.view(torch.uint8))
    scales = torch.full(
        (token * topk, inter_dim // 32), 127, dtype=torch.uint8, device=device
    )
    scales.index_copy_(0, payload_rows, xs.view(torch.uint8))

    order = torch.randperm(active_rows, device=device)
    packed_ids = (active_tokens[:, None] | (slots << 24)).flatten().to(torch.int32)
    sorted_ids = torch.full((block_m,), token, dtype=torch.int32, device=device)
    sorted_ids[:active_rows] = packed_ids[order]
    sorted_weights = torch.zeros(block_m, device=device)
    sorted_weights[:active_rows] = 1 / topk
    expert_ids = torch.zeros(1, dtype=torch.int32, device=device)
    num_valid = torch.tensor([block_m, block_m], dtype=torch.int32, device=device)
    sorted_scales = mxfp4_moe_sort_fwd(
        scales,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid,
        token_num=token,
        cols=inter_dim,
    )
    w = torch.randn((model_dim, inter_dim), device=device) / 4
    if b_dtype == "fp4":
        wq, ws = per_1x32_f4_quant(w)
        w_dequant = fp4_utils.mxfp4_to_f32(wq)
        w_shuffled = shuffle_weight_a16w4(wq.view(1, model_dim, -1), 16, False)
        ws_shuffled = shuffle_scale_a16w4(ws, 1, False)
    else:
        wq, ws = per_1x32_f8_scale_f8_quant(
            w, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
        w_dequant = wq.float()
        w_shuffled = shuffle_weight(wq.view(1, model_dim, -1), layout=(16, 16))
        ws_shuffled = shuffle_scale(ws)
    w_dequant *= fp4_utils.e8m0_to_f32(ws).repeat_interleave(32, -1)
    ref = run_torch(x_dequant, w_dequant, active_tokens, token, topk)
    out_shape = (token, model_dim) if mode == "atomic" else (token, topk, model_dim)
    out = torch.empty(out_shape, dtype=torch.bfloat16, device=device)
    if mode == "reduce":
        ref = torch.zeros(out_shape, device=device)
        ref[active_tokens] = (x_dequant @ w_dequant.T).view(-1, topk, model_dim) / topk
    stage2 = flydsl_moe_stage2
    shared_kwargs = {}
    if family == "fhmoe":
        assert b_dtype == "fp4"
        stage2 = flydsl_fhmoe_stage2
        shared, shared_scale = per_1x32_f8_scale_f8_quant(
            w, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
        shared_kwargs = dict(
            shared_w2=shuffle_weight_a16w4(
                shared.view(1, model_dim, inter_dim), 16, False
            ),
            shared_w2_scale=shuffle_scale_a16w4(shared_scale, 1, False),
            shared_expert_id=1,
        )
        # Give each routed row exactly one expert, including the shared expert.
        w_shuffled = w_shuffled.expand(2, -1, -1).contiguous()
        ws_shuffled = ws_shuffled.flatten().repeat(2)
        expert_ids = torch.tensor([0, 1], dtype=torch.int32, device=device)
        routed = (sorted_ids >> 24) < topk // 2
        valid = (sorted_ids & 0xFFFFFF) < token
        routed_ids = torch.full_like(sorted_ids, token)
        shared_ids = torch.full_like(sorted_ids, token)
        routed_count = int((routed & valid).sum())
        shared_count = active_rows - routed_count
        routed_ids[:routed_count] = sorted_ids[routed & valid]
        shared_ids[:shared_count] = sorted_ids[~routed & valid]
        sorted_ids = torch.cat((routed_ids, shared_ids))
        sorted_weights = torch.zeros(2 * block_m, device=device)
        sorted_weights[:routed_count] = 1 / topk
        sorted_weights[block_m : block_m + shared_count] = 1 / topk
        num_valid = torch.tensor(
            [2 * block_m, 2 * block_m], dtype=torch.int32, device=device
        )
        sorted_scales = mxfp4_moe_sort_fwd(
            scales,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid,
            token_num=token,
            cols=inter_dim,
        )
        shared_dequant = shared.float() * fp4_utils.e8m0_to_f32(
            shared_scale
        ).repeat_interleave(32, -1)
        rows = (x_dequant @ w_dequant.T).view(-1, topk, model_dim)
        shared_rows = (x_dequant @ shared_dequant.T).view(-1, topk, model_dim)
        rows[:, topk // 2 :] = shared_rows[:, topk // 2 :]
        ref.zero_()
        ref[active_tokens] = rows.mean(dim=1) if mode == "atomic" else rows / topk

    def launch(async_copy: bool) -> torch.Tensor:
        out.zero_()
        return stage2(
            inter_states=payload.view(token, topk, -1),
            w2=w_shuffled,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=expert_ids,
            num_valid_ids=num_valid,
            out=out,
            topk=topk,
            tile_m=block_m,
            tile_n=model_dim,
            tile_k=256,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype="bf16",
            mode=mode,
            return_per_slot=(mode == "reduce"),
            w2_scale=ws_shuffled,
            a2_scale=sorted_scales,
            sorted_weights=sorted_weights,
            use_async_copy=async_copy,
            persist=persist,
            waves_per_eu=waves_per_eu,
            cu_num_mul=cu_num_mul,
            inter_dim_pad=inter_dim_pad,
            **shared_kwargs,
        )

    candidates = {
        "sync": functools.partial(launch, False),
        "async": functools.partial(launch, True),
    }
    ret: dict[str, object] = {"gfx": get_gfx(), "input_bytes": payload.numel()}
    flops = 2 * active_rows * model_dim * inter_dim
    # Logical traffic includes clearing the caller-owned output in launch().
    nbytes = (
        active_rows * inter_dim // pack + wq.numel() + out.numel() * out.element_size()
    )
    for name, fn in candidates.items():
        if name not in loads:
            continue
        actual, us = run_perftest(fn, num_iters=5, num_warmup=2)
        assert torch.isfinite(actual).all(), f"{name}: non-finite output"
        err = checkAllclose(ref, actual.float(), atol=0.02, rtol=0.02, msg=name)
        assert err == 0, f"{name}: incorrect gather from {payload.numel()} bytes"
        x_ref, y_actual = ref[active_tokens].double(), actual[active_tokens].double()
        logits_diff = float(
            1
            - 2 * (x_ref * y_actual).sum() / (x_ref.square() + y_actual.square()).sum()
        )
        assert logits_diff <= 0.01, logits_diff
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            fn()
        symbols = [event.name for event in prof.events() if "mfma_moe2_" in event.name]
        expected_variant = "_aglobal" if payload.numel() >= (1 << 32) else "_abuffer"
        assert symbols and all(expected_variant in symbol for symbol in symbols), (
            symbols
        )
        # Warmup is already complete. The graph includes the atomic target reset.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all(), f"{name}: non-finite replay output"
        assert (
            checkAllclose(ref, out.float(), atol=0.02, rtol=0.02, msg=f"{name} replay")
            == 0
        )
        # Change data and routing weights without changing shape or addresses.
        original_ids = sorted_ids.clone()
        for start in range(0, sorted_ids.numel(), block_m):
            ids = sorted_ids[start : start + block_m]
            count = int(((ids & 0xFFFFFF) < token).sum())
            ids[:count] = ids[:count].flip(0)
        sorted_scales.copy_(
            mxfp4_moe_sort_fwd(
                scales,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid,
                token_num=token,
                cols=inter_dim,
            )
        )
        sorted_weights.mul_(0.5)
        payload.index_copy_(
            0,
            payload_rows,
            xq.view(torch.uint8).bitwise_xor(0x80 if a_dtype == "fp8" else 0x88),
        )
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all(), f"{name}: non-finite updated replay output"
        assert (
            checkAllclose(
                ref * -0.5,
                out.float(),
                atol=0.02,
                rtol=0.02,
                msg=f"{name} updated replay",
            )
            == 0
        )
        sorted_weights.mul_(2)
        sorted_ids.copy_(original_ids)
        sorted_scales.copy_(
            mxfp4_moe_sort_fwd(
                scales,
                sorted_ids=sorted_ids,
                num_valid_ids=num_valid,
                token_num=token,
                cols=inter_dim,
            )
        )
        payload.index_copy_(0, payload_rows, xq.view(torch.uint8))
        ret.update(
            {
                f"{name} us": us,
                f"{name} TFLOPS": flops / us / 1e6,
                f"{name} TB/s": nbytes / us / 1e6,
                f"{name} err": err,
                f"{name} logits_diff": logits_diff,
            }
        )
    return ret


def main() -> None:
    if get_gfx() != "gfx950":
        aiter.logger.warning("MX MoE large-buffer regression requires gfx950")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[64, 233016, 233017, 262144]
    )
    parser.add_argument("--a-dtype", choices=["fp8", "fp4"], nargs="+", default=["fp8"])
    parser.add_argument(
        "--modes", choices=["atomic", "reduce"], nargs="+", default=["atomic"]
    )
    parser.add_argument(
        "--storage", choices=["own", "offset_view"], nargs="+", default=["own"]
    )
    parser.add_argument(
        "--families", choices=["moe", "fhmoe"], nargs="+", default=["moe"]
    )
    parser.add_argument("--b-dtype", choices=["fp4", "fp8"], nargs="+", default=["fp4"])
    parser.add_argument(
        "--loads", choices=["sync", "async"], nargs="+", default=["sync", "async"]
    )
    args = parser.parse_args()
    rows = [
        test_large_buffer(m, dtype, mode, storage, family, b_dtype, tuple(args.loads))
        for m, dtype, mode, storage, family, b_dtype in itertools.product(
            args.tokens,
            args.a_dtype,
            args.modes,
            args.storage,
            args.families,
            args.b_dtype,
        )
    ]
    aiter.logger.info(
        "GEMM2 large-buffer regression:\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
