#!/usr/bin/env python3
"""
Gemm Test Runner at Model Level
Manages test parameters and executes test_gemm.py/test_gemm_a4w4.py/test_gemm_a8w8.py with user-defined argument combinations
"""

import logging
import pandas as pd
from aiter import dtypes
from dataclasses import dataclass
import argparse

from op_tests.test_gemm_a8w8 import test_gemm as test_gemm_a8w8
from op_tests.test_gemm_a4w4 import test_gemm as test_gemm_a4w4
from op_tests.test_gemm import test_gemm as test_gemm
from aiter.jit.utils.chip_info import get_gfx

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

M = [1, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 8192, 16384, 32768]

@dataclass
class TestConfig:
    """
    Test configuration data class
    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., "Qwen3-32B", "Llama3-70B")
    attention_head : int
        Number of attention heads
    kv_head : int
        Number of the kv heads
    head_dim : int
        feature dimention per head
    intermediate_size : int
        feature dimention in MLP module
    is_moe : bool
        is the moe model or not
    """

    model_name: str
    attention_head: int
    kv_head: int
    head_dim: int
    intermediate_size: int
    is_moe : bool


TEST_CONFIGS = {
    # model,                  model_name,   attention_head,   kv_head,   head_dim,  intermediate_size    is_moe
    "Qwen3-32B":   TestConfig("Qwen3-32B",         64,           8,          80,         25600,          false),
    "Qwen3-32B":   TestConfig("Qwen3-30B",         16,           16,         128,        6144,           true),
    "Qwen3-235B":  TestConfig("Qwen3-235B",        32,           32,         128,        12288,          true),
    "Llama3-70B":  TestConfig("Llama3-70B",        64,           8,          128,        28672,          false),
    "Llama3-405B": TestConfig("Llama3-405B",       128,          8,          128,        53248,          false),
}


@dataclass
class Record:
    M: int
    N: int
    K: int
    TP: int
    quant_type: str = "torch.bfloat16"
    output_type: str = "torch.bfloat16"
    latency: float = 0.0
    bandwidth: float = 0.0
    throughput: float = 0.0


class GemmTestRunner:
    def __init__(self):
        return

    def get_model_in_single_card(self, config):
        # for Qwen3-32B or Qwen3-30B, the dim is not dividable by 32 with tp 8, we skip this case
        if (config.model_name == "Qwen3-32B" or config.model_name == "Qwen3-30B"):
            return true
        return false

    def get_gemm_shape(self, config: TestConfig, TP_list):
        test_name = str(config)
        logger.info(f"Running test: {test_name}")

        records = []
        for tp in TP_list:
            if self.get_model_in_single_card and (tp == 8 or tp == 4):
                continue

            hidden_size = config.attention_head * config.head_dim
            # attn qkv fused gemm
            QKV_K = hidden_size
            QKV_N = (
                config.attention_head // tp + 2 * ((config.kv_head + tp - 1) // tp)
            ) * config.head_dim
            # attn output gemm
            Out_K = (config.attention_head // tp) * config.head_dim
            Out_N = hidden_size

            if not config.is_moe:
                # mlp up-gate fused gemm
                Up_Gate_K = hidden_size
                Up_Gate_N = config.intermediate_size * 2 // tp
                # mlp up-gate non-fused gemm
                Up_N = config.intermediate_size // tp
                # mlp down gemm
                Down_K = config.intermediate_size // tp
                Down_N = hidden_size

            for m in M:
                records.append(Record(M=m, N=QKV_N, K=QKV_K, TP=tp))
                records.append(Record(M=m, N=Out_N, K=Out_K, TP=tp))
                if not config.is_moe:
                    records.append(Record(M=m, N=Up_Gate_N, K=Up_Gate_K, TP=tp))
                    records.append(Record(M=m, N=Up_N, K=Up_Gate_K, TP=tp))
                    records.append(Record(M=m, N=Down_N, K=Down_K, TP=tp))
        return records

    def calculate_throughput(self, m, n, k, latency):
        throughput = 2 * m * n * k / 1e6 / latency  # TFlops
        return throughput

    def calculate_bandwidth(self, m, n, k, latency, in_dtype, wei_dtype, out_dtype):
        typesize = {
            dtypes.d_dtypes["fp4x2"]: 0.5,
            dtypes.d_dtypes["fp8"]: 1,
            dtypes.d_dtypes["bf16"]: 2,
        }
        bandwidth = (
            (
                m * k * typesize[in_dtype]
                + k * n * typesize[wei_dtype]
                + m * n * typesize[out_dtype]
            )
            / latency
            / 1e6
        )  # TB/s
        return bandwidth

    def benchmark_gemm(self, l_dtype, l_quantDtype, records):
        df = []
        records_result = []
        for quantDtype in l_quantDtype:
            for dtype in l_dtype:
                for idx, record in enumerate(records):
                    latency = float("inf")
                    if quantDtype == dtypes.d_dtypes["fp8"]:
                        ret = test_gemm_a8w8(
                            dtype, record.M, record.N, record.K, quantDtype
                        )
                        ck_time = ret["ck us"]
                        ck_bpreshuffle_time = ret["ck bpreshuffle us"]
                        asm_time = ret["asm us"]

                        if ck_time is not None:
                            latency = min(latency, ck_time)
                        if ck_time is not None:
                            latency = min(latency, ck_bpreshuffle_time)
                        if asm_time is not None:
                            latency = min(latency, asm_time)
                    elif quantDtype == dtypes.d_dtypes["fp4x2"]:
                        ret = test_gemm_a4w4(dtype, record.M, record.N, record.K)
                        asm_no_splitK_time = ret["asm no splitK"]
                        asm_splitK_time = ret["asm splitK"]
                        ck_time = ret["ck"]
                        if ck_time is not None:
                            latency = min(latency, ck_time)
                        if asm_no_splitK_time is not None:
                            latency = min(latency, asm_no_splitK_time)
                        if asm_splitK_time is not None:
                            latency = min(latency, asm_splitK_time)
                    else:
                        ret = test_gemm(
                            dtypes.bf16, record.M, record.N, record.K, otype=dtypes.bf16
                        )
                        latency = ret["ck us"]
                    throughput = self.calculate_throughput(
                        record.M, record.N, record.K, latency
                    )
                    bandwidth = self.calculate_bandwidth(
                        record.M,
                        record.N,
                        record.K,
                        latency,
                        quantDtype,
                        quantDtype,
                        dtype,
                    )
                    records_result.append(
                        Record(
                            M=record.M,
                            N=record.N,
                            K=record.K,
                            TP=record.TP,
                            quant_type=quantDtype,
                            output_type=dtype,
                            latency=latency,
                            throughput=throughput,
                            bandwidth=bandwidth,
                        )
                    )

        df = pd.DataFrame(df)
        # aiter.logger.info(f"summary:\n{df}")
        return records_result

    def save_structs_to_csv(self, records, csv_file, fieldnames):
        import csv
        from dataclasses import asdict

        with open(csv_file, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow(asdict(rec))
        print(f"data write to {csv_file} success!!")

    def save_to_csv(self, records, csv_file_name):
        self.save_structs_to_csv(
            records,
            f"{csv_file_name}.csv",
            [
                "M",
                "N",
                "K",
                "TP",
                "quant_type",
                "output_type",
                "latency",
                "throughput",
                "bandwidth",
            ],
        )

    def extract_configuration(self, args):
        models = []
        if args.model is None:
            for key, value in TEST_CONFIGS.items():
                models.append(value)
        else:
            models.append(TEST_CONFIGS[args.model])

        if args.tensor_parallel is None:
            TP_list = [1, 4, 8]
        else:
            TP_list = [args.tensor_parallel]

        if args.dtype is None:
            output_types = [dtypes.d_dtypes["bf16"]]
        else:
            output_types = [dtypes.d_dtypes[args.dtype]]

        if args.quantDtype is None:
            quant_types = [dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]]
            if get_gfx() in ["gfx950"]:
                quant_types.append(dtypes.d_dtypes["fp4x2"])
        else:
            quant_types = [dtypes.d_dtypes[args.quantDtype]]

        return models, TP_list, output_types, quant_types

    def run_single_test(self, config, TP_list, output_types, quant_types):
        """Run a single test with given configuration"""
        records = self.get_gemm_shape(config, TP_list)
        records_result = self.benchmark_gemm(output_types, quant_types, records)
        return records_result

    def run_all_tests(self, args):
        """Run all defined tests"""
        logger.info("Starting all GEMM Benchmark from Models...")
        models, TP_list, output_types, quant_types = self.extract_configuration(args)
        logger.info(f"Total tests: {len(models)}")
        for model in models:
            records = self.run_single_test(model, TP_list, output_types, quant_types)
            self.save_to_csv(records, model.model_name)


def create_argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        nargs="?",
        const=None,
        default=None,
        help="""Data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-q",
        "--quantDtype",
        type=str,
        choices=["fp4x2", "fp8", "bf16"],
        nargs="?",
        const=None,
        default=None,
        help="""Date type of quantization.
        e.g.: -q fp8""",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["Qwen3-32B", "Qwen3-30B", "Qwen3-235B", "Llama3-70B", "Llama3-405B"],
        nargs="?",
        const=None,
        default=None,
        help="""Supported model.
        e.g. -m Llama3-70B""",
    )
    parser.add_argument(
        "-tp",
        "--tensor-parallel",
        type=int,
        choices=[1, 4, 8],
        nargs="?",
        const=None,
        default=None,
        help="""Tenosr Parallel size.
        e.g. -tp 8""",
    )
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    runner = GemmTestRunner()
    results = runner.run_all_tests(args)
