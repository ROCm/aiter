import argparse
import functools
import itertools

import pandas as pd
import torch

import aiter.fused_moe as fused_moe_module
from aiter import logger
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.moe.small_m_sort import small_m_sort
from aiter.test_common import benchmark, checkAllclose, run_perftest


@benchmark()
def benchmark_small_m_sort(tokens, block_size):
    torch.manual_seed(337)
    ids = (
        torch.randn(tokens, 384, device="cuda", dtype=torch.float32)
        .topk(6, dim=-1)
        .indices.int()
    )
    weights = torch.rand(tokens, 6, device="cuda", dtype=torch.float32)
    output = torch.empty(tokens, 5120, device="cuda", dtype=torch.bfloat16)
    expected = moe_sorting(ids, weights, 384, 5120, output.dtype, block_size)
    extent = int(expected[3][0])
    live = expected[0][:extent] != ((6 << 24) | tokens)
    candidates = {
        "default": functools.partial(
            moe_sorting,
            ids,
            weights,
            384,
            5120,
            output.dtype,
            block_size,
            output=output,
        ),
        "small_m": functools.partial(
            small_m_sort,
            ids,
            weights,
            384,
            5120,
            output.dtype,
            block_size,
            output=output,
        ),
    }

    def run_flydsl():
        original = fused_moe_module._USE_FLYDSL_MOE_SORTING
        original_ck = fused_moe_module._USE_CK_MOE_SORTING
        try:
            fused_moe_module._USE_CK_MOE_SORTING = False
            fused_moe_module._USE_FLYDSL_MOE_SORTING = True
            return moe_sorting(
                ids, weights, 384, 5120, output.dtype, block_size, output=output
            )
        finally:
            fused_moe_module._USE_CK_MOE_SORTING = original_ck
            fused_moe_module._USE_FLYDSL_MOE_SORTING = original

    candidates["flydsl"] = run_flydsl
    result = {"gfx": get_gfx()}
    for name, operation in candidates.items():
        output.fill_(float("nan"))
        actual, elapsed = run_perftest(operation)
        assert actual is not None
        torch.testing.assert_close(expected[3], actual[3], atol=0, rtol=0)
        torch.testing.assert_close(
            expected[0][:extent], actual[0][:extent], atol=0, rtol=0
        )
        torch.testing.assert_close(
            expected[2][: extent // block_size],
            actual[2][: extent // block_size],
            atol=0,
            rtol=0,
        )
        errors = [
            checkAllclose(
                expected[1][:extent][live], actual[1][:extent][live], atol=0, rtol=0
            ),
        ]
        assert not any(errors)
        assert output.count_nonzero() == 0
        byte_count = (
            ids.nbytes
            + weights.nbytes
            + extent * (actual[0].element_size() + actual[1].element_size())
            + (extent // block_size) * actual[2].element_size()
            + actual[3].nbytes
            + output.nbytes
        )
        result[f"{name} us"] = elapsed
        result[f"{name} TFLOPS"] = 0.0
        result[f"{name} TB/s"] = byte_count / elapsed / 1e6
        result[f"{name} err"] = max(errors)
    return result


def main():
    if get_gfx() != "gfx942":
        logger.warning("small_m_sort is not supported on %s", get_gfx())
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", nargs="+", type=int, default=[1, 8, 32, 64])
    parser.add_argument("--block-size", nargs="+", type=int, default=[16, 32])
    options = parser.parse_args()
    rows = [
        benchmark_small_m_sort(tokens, block)
        for tokens, block in itertools.product(options.batch, options.block_size)
    ]
    logger.info(
        "Small-M sorting (I/O byte lower bound):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
