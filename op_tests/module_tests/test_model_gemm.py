#!/usr/bin/env python3
"""
Gemm Test Runner at Model Level
Manages test parameters and executes test_gemm.py/test_gemm_a4w4.py/test_gemm_a8w8.py with user-defined argument combinations
"""
import logging
from aiter import dtypes
from dataclasses import dataclass
import argparse


from aiter.jit.utils.chip_info import get_gfx
from utils.gemm_utils import save_gemm_benchmark_result, save_untuned_gemm_csv
from op_tests.module_tests.utils.triton_bench_utils import run_triton_a4w4

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    hidden_size: int
        feature dimention in attention
    intermediate_size : int
        feature dimention in MLP module
    is_moe : bool
        is the moe model or not
    """

    model_name: str
    attention_head: int
    kv_head: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    is_moe: bool


TEST_CONFIGS = {
    # model,                  model_name,   attention_head,   kv_head,   head_dim,  hidden_size, intermediate_size    is_moe
    "Qwen3-32B": TestConfig("Qwen3-32B", 64, 8, 80, 5120, 25600, False),
    "Qwen3-30B": TestConfig("Qwen3-30B", 16, 16, 128, 2048, 6144, True),
    "Qwen3-235B": TestConfig("Qwen3-235B", 32, 32, 128, 4096, 12288, True),
    "Llama3-70B": TestConfig("Llama3-70B", 64, 8, 128, 8192, 28672, False),
    "Llama3-405B": TestConfig("Llama3-405B", 128, 8, 128, 16384, 53248, False),
    "gpt-oss-120B": TestConfig("gpt-oss-120B", 64, 8, 64, 2880, 2880, True),
}


@dataclass
class Record:
    M: int
    N: int
    K: int
    TP: int
    output_type: str = "torch.bfloat16"
    quant_type: str = "torch.bfloat16"
    quant_method: str = "default"
    latency: float = 0.0
    bandwidth: float = 0.0
    throughput: float = 0.0
    latency_asm: float = 0.0
    bandwidth_asm: float = 0.0
    throughput_asm: float = 0.0
    latency_triton: float = 0.0
    bandwidth_triton: float = 0.0
    throughput_triton: float = 0.0


def to_record(
    M,
    N,
    K,
    TP,
    output_type,
    quant_type,
    quant_method,
    latency,
    bandwidth,
    throughput,
    latency_asm,
    bandwidth_asm,
    throughput_asm,
    latency_triton,
    bandwidth_triton,
    throughput_triton,
):
    return Record(
        M,
        N,
        K,
        TP,
        output_type,
        quant_type,
        quant_method,
        latency,
        bandwidth,
        throughput,
        latency_asm,
        bandwidth_asm,
        throughput_asm,
        latency_triton,
        bandwidth_triton,
        throughput_triton,
    )


##### Wrapper and unified the test_gemm API for CK/ASM/Triton Kernels #####
def run_a16w16_gemm(dtype, record, run_triton=False):
    from op_tests.test_gemm import test_gemm

    ret = test_gemm(dtype, record.M, record.N, record.K, otype=dtype)
    latency = ret["ck us"]

    latency_asm = 0.0
    latency_triton = 0.0
    if run_triton:
        from op_tests.module_tests.utils.triton_bench_utils import run_triton_a16w16

        latency_triton = run_triton_a16w16(M, N, K)
    return latency, latency_asm, latency_triton


def run_a8w8_gemm(dtype, record, fp8_quant_method, run_triton=False):
    latency_asm = 0.0
    if fp8_quant_method == "per_tensor":
        # from op_tests.test_gemm_a8w8 import test_skinny_gemm as test_gemm
        # print("----run per_tensor gemm")
        # cu_count = torch.cuda.get_device_properties(device="cuda").multi_processor_count
        # ret = test_gemm(
        #    dtype,
        #    record.M,
        #    record.N,
        #    record.K,
        #    quantDtype=dtypes.fp8,
        #    cu_count=cu_count,
        # )
        # latency = ret["us"]
        latency = 0.0
    elif fp8_quant_method == "per_token":
        from op_tests.test_gemm_a8w8 import test_gemm

        ret = test_gemm(dtype, record.M, record.N, record.K, dtypes.fp8)
        ck_time = ret["ck us"]
        ck_bpreshuffle_time = ret["ck bpreshuffle us"]
        asm_time = ret["asm us"]
        latency = float("inf")
        if ck_time is not None:
            latency = min(latency, ck_time)
        if ck_bpreshuffle_time is not None:
            latency = min(latency, ck_bpreshuffle_time)
        if asm_time is not None:
            latency_asm = asm_time
    elif fp8_quant_method == "per_block":
        from op_tests.test_gemm_a8w8_blockscale import test_gemm

        ret = test_gemm(dtype, record.M, record.N, record.K)
        latency = ret["us"]
    else:
        raise ValueError(f"Unsupported quantization method '{fp8_quant_method}'!")

    latency_triton = 0.0
    if run_triton:
        if fp8_quant_method == "per_tensor":
            from op_tests.module_tests.utils.triton_bench_utils import (
                run_triton_a8w8_per_tensor,
            )

            latency_triton = run_triton_a8w8_per_tensor(record.M, record.N, record.K)
        elif fp8_quant_method == "per_token":
            from op_tests.module_tests.utils.triton_bench_utils import (
                run_triton_a8w8_per_token,
            )

            latency_triton = run_triton_a8w8_per_token(record.M, record.N, record.K)
        elif fp8_quant_method == "per_block":
            from op_tests.module_tests.utils.triton_bench_utils import (
                run_triton_a8w8_blockscale,
            )

            latency_triton = run_triton_a8w8_blockscale(record.M, record.N, record.K)
        else:
            raise ValueError(f"Unsupported quantization method '{fp8_quant_method}'!")

    return latency, latency_asm, latency_triton


def run_a4w4_gemm(dtype, record, run_triton=False):
    from op_tests.test_gemm_a4w4 import test_gemm

    ret = test_gemm(dtype, record.M, record.N, record.K)
    ck_time = ret["ck"]
    latency = float("inf")
    if ck_time is not None:
        latency = min(latency, ck_time)

    latency_asm = float("inf")
    asm_no_splitK_time = ret["asm no splitK"]
    asm_splitK_time = ret["asm splitK"]
    if asm_no_splitK_time is not None:
        latency_asm = min(latency_asm, asm_no_splitK_time)
    if asm_splitK_time is not None:
        latency_asm = min(latency_asm, asm_splitK_time)

    latency_triton = 0.0
    if run_triton:
        latency_triton = run_triton_a4w4(record.M, record.N, record.K)

    return latency, latency_asm, latency_triton


class GemmTestRunner:
    def __init__(self):
        return

    def get_model_in_single_card(self, config):
        # for Qwen3-32B or Qwen3-30B, the dim is not dividable by 32 with tp 8, we skip this case
        if config.model_name == "Qwen3-32B" or config.model_name == "Qwen3-30B":
            return True
        return False

    def get_gemm_shape(self, config: TestConfig, M, TP_list):
        test_name = str(config)
        logger.info(f"Running test: {test_name}")

        records = []
        for tp in TP_list:
            if self.get_model_in_single_card and (tp == 8 or tp == 4):
                continue

            # attn qkv fused gemm
            QKV_K = config.hidden_size
            QKV_N = (
                config.attention_head // tp + 2 * ((config.kv_head + tp - 1) // tp)
            ) * config.head_dim
            # attn output gemm
            Out_K = (config.attention_head // tp) * config.head_dim
            Out_N = config.hidden_size

            if not config.is_moe:
                # mlp up-gate fused gemm
                Up_Gate_K = config.hidden_size
                Up_Gate_N = config.intermediate_size * 2 // tp
                # mlp up-gate non-fused gemm
                Up_N = config.intermediate_size // tp
                # mlp down gemm
                Down_K = config.intermediate_size // tp
                Down_N = config.hidden_size

            for m in M:
                records.append(Record(M=m, N=QKV_N, K=QKV_K, TP=tp))
                records.append(Record(M=m, N=Out_N, K=Out_K, TP=tp))
                if not config.is_moe:
                    records.append(Record(M=m, N=Up_Gate_N, K=Up_Gate_K, TP=tp))
                    records.append(Record(M=m, N=Up_N, K=Up_Gate_K, TP=tp))
                    records.append(Record(M=m, N=Down_N, K=Down_K, TP=tp))
        return records

    def calculate_throughput(self, m, n, k, latency):
        if latency == 0.0:
            return 0.0
        throughput = 2 * m * n * k / 1e6 / latency  # TFlops
        return throughput

    def calculate_bandwidth(self, m, n, k, latency, in_dtype, wei_dtype, out_dtype):
        if latency == 0.0:
            return 0.0
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

    def get_metrics(
        self, latency, latency_asm, latency_triton, record, dtype, quant_dtype
    ):
        throughput = self.calculate_throughput(record.M, record.N, record.K, latency)
        bandwidth = self.calculate_bandwidth(
            record.M,
            record.N,
            record.K,
            latency,
            quant_dtype,
            quant_dtype,
            dtype,
        )
        throughput_asm = self.calculate_throughput(
            record.M, record.N, record.K, latency_asm
        )
        bandwidth_asm = self.calculate_bandwidth(
            record.M,
            record.N,
            record.K,
            latency_asm,
            quant_dtype,
            quant_dtype,
            dtype,
        )
        throughput_triton = self.calculate_throughput(
            record.M, record.N, record.K, latency_triton
        )
        bandwidth_triton = self.calculate_bandwidth(
            record.M,
            record.N,
            record.K,
            latency_triton,
            quant_dtype,
            quant_dtype,
            dtype,
        )
        return (
            throughput,
            bandwidth,
            throughput_asm,
            bandwidth_asm,
            throughput_triton,
            bandwidth_triton,
        )

    def run_gemm(self, record, quant_dtype, fp8_quant_method, dtype, run_triton):
        if quant_dtype == dtypes.d_dtypes["fp8"]:
            latency, latency_asm, latency_triton = run_a8w8_gemm(
                dtype, record, fp8_quant_method, run_triton
            )
        elif quant_dtype == dtypes.d_dtypes["fp4x2"]:
            latency, latency_asm, latency_triton = run_a4w4_gemm(
                dtype, record, run_triton
            )
        else:
            latency, latency_asm, latency_triton = run_a16w16_gemm(
                dtype, record, run_triton
            )
        (
            throughput,
            bandwidth,
            throughput_asm,
            bandwidth_asm,
            throughput_triton,
            bandwidth_triton,
        ) = self.get_metrics(
            latency, latency_asm, latency_triton, record, dtype, quant_dtype
        )

        return (
            latency,
            throughput,
            bandwidth,
            latency_asm,
            throughput_asm,
            bandwidth_asm,
            latency_triton,
            throughput_triton,
            bandwidth_triton,
        )

    def extract_configuration(self, args):
        models = []
        if args.model is None:
            for key, value in TEST_CONFIGS.items():
                models.append(value)
        else:
            models.append(TEST_CONFIGS[args.model])
        if args.m is not None:
            M = list(args.m)
        else:
            M = [
                1,
                4,
                8,
                16,
                32,
                64,
                128,
                160,
                192,
                224,
                256,
                288,
                320,
                352,
                384,
                416,
                448,
                480,
                512,
                1024,
                2048,
                4096,
                8192,
                16384,
                32768,
            ]

        if args.tensor_parallel is None:
            TP_list = [1, 4, 8]
        else:
            TP_list = [args.tensor_parallel]

        if args.dtype is None:
            output_types = [dtypes.d_dtypes["bf16"]]
        else:
            output_types = [dtypes.d_dtypes[args.dtype]]

        if args.quant_dtype is None:
            quant_types = [dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]]
            if get_gfx() in ["gfx950"]:
                quant_types.append(dtypes.d_dtypes["fp4x2"])
        else:
            quant_types = [dtypes.d_dtypes[args.quant_dtype]]

        if args.fp8_quant_method is None:
            fp8_quant_methods = ["per_tensor", "per_token", "per_block"]
        else:
            fp8_quant_methods = [args.fp8_quant_method]

        return models, M, TP_list, output_types, quant_types, fp8_quant_methods

    def run_single_test(
        self,
        config,
        M,
        TP_list,
        fp8_quant_methods,
        output_types,
        quant_types,
        bench_triton,
    ):
        """Run a single test with given configuration"""
        records = self.get_gemm_shape(config, M, TP_list)

        records_result = []
        for quant_dtype in quant_types:
            if quant_dtype == dtypes.d_dtypes["fp8"]:
                quant_methods = fp8_quant_methods
            else:
                quant_methods = ["default"]

            for quant_method in quant_methods:
                for dtype in output_types:
                    for idx, record in enumerate(records):
                        (
                            latency,
                            throughput,
                            bandwidth,
                            latency_asm,
                            throughput_asm,
                            bandwidth_asm,
                            latency_triton,
                            throughput_triton,
                            bandwidth_triton,
                        ) = self.run_gemm(
                            record, quant_dtype, quant_method, dtype, bench_triton
                        )
                        records_result.append(
                            to_record(
                                M=record.M,
                                N=record.N,
                                K=record.K,
                                TP=record.TP,
                                quant_method=quant_method,
                                quant_type=quant_dtype,
                                output_type=dtype,
                                latency=latency,
                                throughput=throughput,
                                bandwidth=bandwidth,
                                latency_asm=latency_asm,
                                throughput_asm=throughput_asm,
                                bandwidth_asm=bandwidth_asm,
                                latency_triton=latency_triton,
                                throughput_triton=throughput_triton,
                                bandwidth_triton=bandwidth_triton,
                            )
                        )
        return records_result

    def run_all_tests(self, args):
        """Run all defined tests"""
        logger.info("Starting all GEMM Benchmark from Models...")
        models, M, TP_list, output_types, quant_types, fp8_quant_methods = (
            self.extract_configuration(args)
        )
        logger.info(f"Total tests: {len(models)}")
        for model in models:
            records = self.run_single_test(
                model,
                M,
                TP_list,
                fp8_quant_methods,
                output_types,
                quant_types,
                args.triton,
            )
            save_gemm_benchmark_result(records, model.model_name)
            if args.save_untuned_gemm:
                save_untuned_gemm_csv(
                    f"{model.model_name}.csv", f"{model.model_name}_untuned_gemm"
                )


def create_argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16"],
        nargs="?",
        const=None,
        default=None,
        help="""Data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "--quant_dtype",
        type=str,
        choices=["fp4x2", "fp8", "bf16"],
        nargs="?",
        const=None,
        default=None,
        help="""Date type of quantization.
        e.g.: -q fp8""",
    )
    parser.add_argument(
        "--fp8_quant_method",
        type=str,
        choices=["per_tensor", "per_token", "per_block"],
        nargs="?",
        const=None,
        default=None,
        help="""fp8 quantization method.
        e.g.: -qm per_token""",
    )
    parser.add_argument(
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
        "--m",
        type=dtypes.str2tuple,
        nargs="?",
        const=None,
        default=None,
        help="""M dimention of GEMM.
        e.g.: -m 1024,2048""",
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
    parser.add_argument(
        "--triton",
        action="store_true",
        help="benchmark triton kernel",
    )
    parser.add_argument(
        "--save_untuned_gemm",
        action="store_true",
        help="save the untuned_gemm and untun_gemm_bf16 csv files at model level",
    )
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    runner = GemmTestRunner()
    results = runner.run_all_tests(args)
