# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import (
    checkAllclose,
    benchmark,
    run_perftest,
    perftest,
)
from aiter import dtypes, get_gfx
import pandas as pd
import argparse

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


@benchmark()
def test_topk(
    batch_size,
    hiddensize,
    topk,
    largest,
    dtype,
):
    output = torch.randn((batch_size, hiddensize), dtype=dtype)
    device = output.device

    row = torch.arange(
        hiddensize, dtype=dtypes.i32, device=device
    )  # [0, 1, ..., length-1]
    topk_ids = torch.zeros((batch_size, topk), dtype=dtypes.i32, device=device)

    x = torch.arange(hiddensize, dtype=dtype).repeat(batch_size, 1)
    for b in range(batch_size):
        x[b] = x[b, torch.randperm(hiddensize)]

    (ref_value, ref_index), us_ref = run_perftest(
        torch.topk,
        x,
        topk,
        largest=largest,
        num_iters=1000,
        num_warmup=100,
    )

    id_ref, _ref = torch.sort(ref_index)

    _, us_aiter = run_perftest(
        aiter.topk_plain,
        x,
        topk_ids,
        topk,
        largest,
    )

    id_aiter, _aiter = torch.sort(topk_ids.to(torch.long))
    checkAllclose(
        id_ref,
        id_aiter,
        msg=(
            "topk_ids Performance Comparison:\n"
            "  {:<10} {:>12}\n"
            "  {:<10} {:>12}\n"
            "  {:<10} {:>12.2f}\n"
            "  {:<10} {:>12.2f}\n".format(
                "Method", "Time (μs)",
                "-"*10, "-"*12,
                "golden", us_ref,
                "aiter", us_aiter
            )
        ),
    )

    return {"err": 0, "us": us_aiter}


def run_test_suite():
    """Run comprehensive topk tests across different configurations"""

    # Test configurations
    BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    HIDDEN_SIZES = [1000, 1001, 100000, 100001]
    TOPK_VALUES = [1, 2, 8, 16, 32, 64]
    LARGEST_VALUES = [True]
    DTYPES = [dtypes.fp32]

    results = []

    for dtype in DTYPES:
        for largest in LARGEST_VALUES:
            for batch_size in BATCH_SIZES:
                for hiddensize in HIDDEN_SIZES:
                    for topk in TOPK_VALUES:
                        # Skip invalid configurations (topk > hiddensize)
                        if topk > hiddensize:
                            continue

                        print(
                            "\nTesting: batch_size={}, hiddensize={}, "
                            "topk={}, largest={}, dtype={}".format(
                                batch_size, hiddensize, topk, largest, dtype
                            )
                        )

                        try:
                            ret = test_topk(
                                batch_size,
                                hiddensize,
                                topk,
                                largest,
                                dtype,
                            )

                            results.append({
                                "batch_size": batch_size,
                                "hiddensize": hiddensize,
                                "topk": topk,
                                "largest": largest,
                                "dtype": str(dtype),
                                "error": ret["err"],
                                "time_us": ret["us"],
                                "status": "PASS"
                            })
                        except Exception as e:
                            aiter.logger.error("Test failed: {}".format(e))
                            results.append({
                                "batch_size": batch_size,
                                "hiddensize": hiddensize,
                                "topk": topk,
                                "largest": largest,
                                "dtype": str(dtype),
                                "error": None,
                                "time_us": None,
                                "status": "FAIL: {}".format(str(e))
                            })

    df = pd.DataFrame(results)

    # Display summary statistics
    aiter.logger.info("\n{}".format("="*80))
    aiter.logger.info("Test Summary")
    aiter.logger.info("{}".format("="*80))
    aiter.logger.info("Total tests run: {}".format(len(df)))
    aiter.logger.info("Passed: {}".format(len(df[df["status"] == "PASS"])))
    aiter.logger.info("Failed: {}".format(len(df[df["status"] != "PASS"])))
    aiter.logger.info("\n{}".format(df))

    # Save results to CSV
    output_file = "topk_test_results.csv"
    df.to_csv(output_file, index=False)
    aiter.logger.info("\nResults saved to: {}".format(output_file))

    # Display performance summary for passed tests
    if len(df[df["status"] == "PASS"]) > 0:
        passed_df = df[df["status"] == "PASS"].copy()
        aiter.logger.info("\n{}".format("="*80))
        aiter.logger.info("Performance Summary (Passed Tests)")
        aiter.logger.info("{}".format("="*80))
        aiter.logger.info("\nAverage time by batch size:")
        aiter.logger.info(passed_df.groupby("batch_size")["time_us"].mean())
        aiter.logger.info("\nAverage time by hiddensize:")
        aiter.logger.info(passed_df.groupby("hiddensize")["time_us"].mean())
        aiter.logger.info("\nAverage time by topk:")
        aiter.logger.info(passed_df.groupby("topk")["time_us"].mean())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test topk_plain with various configurations")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick test with minimal configurations"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single test case"
    )
    args = parser.parse_args()

    if args.single:
        # Single test case
        batch_size = 1000
        hiddensize = 100000
        topk = 64
        largest = True

        print("Running single test: batch_size={}, hiddensize={}, topk={}".format(
            batch_size, hiddensize, topk))
        ret = test_topk(
            batch_size,
            hiddensize,
            topk,
            largest,
            dtypes.fp32,
        )
        aiter.logger.info("Result: error={}, time={:.2f} μs".format(ret["err"], ret["us"]))
    else:
        # Run full test suite
        df = run_test_suite()
