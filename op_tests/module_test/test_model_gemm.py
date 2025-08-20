#!/usr/bin/env python3
"""
MoE 2-Stage Test Runner
Manages test parameters and executes test_moe_2stage.py with user-defined argument combinations
"""

import subprocess
import os
import sys
import time
import logging
import random
import torch
import aiter
import pandas as pd
from aiter import dtypes
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """
    Test configuration data class

    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., "DeepSeek-R1", "Qwen3-235B-A22B")

    tp_size : int
        Tensor Parallel size (e.g., 8 for TP8, 4 for TP4)

    dim : str
        Model dimensions in format "hidden_dim,moe_intermediate_dim"
        Examples: "6144,4096", "8192,4096", "4096,2048"

    expert : int
        Number of experts in the MoE layer

    topk : int
        Number of top experts to use for each token
    """
    model_name: str
    tp_size: int
    attention_head: int
    kv_head: int
    head_dim: int
    intermediate_size: int

    def __str__(self):
        return f"{self.model_name}.tp{self.tp_size}.head_dim{self.head_dim}.intermediate_size{self.intermediate_size}"

TEST_CONFIGS = [
    # #       model_name,        tp_size,   attention_head,   kv_head,   head_dim,  intermediate_size
    # # Qwen3-32B -- TP8
    # TestConfig("Qwen3-32B",       8,         64,          8,         80,         25600),
    # # Qwen3-32B -- TP4
    # TestConfig("Qwen3-32B",       4,         64,          8,         80,         25600),
    # # Qwen3-32B -- TP1
    # TestConfig("Qwen3-32B",       1,         64,          8,         80,         25600),

    # Llama3-70B -- TP8
    TestConfig("Llama3-70B",        8,         64,          8,         128,         28672),
    # # Llama3-70B -- TP4
    # TestConfig("Llama3-70B",      4,         64,          8,         128,         28672),
    # # Llama3-70B -- TP1
    # TestConfig("Llama3-70B",      1,         64,          8,         128,         28672),

    # # Llama3-405B -- TP8
    # TestConfig("Llama3-405B",     8,         128,         8,         128,         53248),
    # # Llama3-405B -- TP4
    # TestConfig("Llama3-405B",     4,         128,         8,         128,         53248),
    # # Llama3-405B -- TP1
    # TestConfig("Llama3-405B",     1,         128,         8,         128,         53248),
]
import sys
from op_tests.test_gemm_a8w8 import test_gemm


@dataclass
class Record():
    M: int
    N: int
    K: int
    TP: int
    quant_type: str = "torch.bfloat16" 
    output_type: str = "torch.bfloat16"
    latency: float = 0.0
    bandwidth: float = 0.0
    throughput: float = 0.0

class FusedmoeTestRunner:
    def __init__(self):
        return
    
    def get_gemm_shape(self, config: TestConfig):
        
        test_name = str(config)
        logger.info(f"Running test: {test_name}")
            
        M = [1, 4, 8, 16, 32, 64, 128, 256]
        # M = [1]
        hidden_size = config.attention_head * config.head_dim
            
        QKV_K = hidden_size
        QKV_N = (config.attention_head // config.tp_size + 2 * ((config.kv_head + config.tp_size - 1) // config.tp_size)) * config.head_dim

        Out_K = (config.attention_head // config.tp_size) * config.head_dim
        Out_N = hidden_size

        Up_Gate_K = hidden_size
        Up_Gate_N = config.intermediate_size * 2 // config.tp_size

        Down_K = config.intermediate_size // config.tp_size
        Down_N = hidden_size

        records = []
        for m in M:
            records.append(Record(M=m, N=QKV_N, K=QKV_K, TP=config.tp_size))
            records.append(Record(M=m, N=Out_N, K=Out_K, TP=config.tp_size))
            records.append(Record(M=m, N=Up_Gate_N, K=Up_Gate_K, TP=config.tp_size))
            records.append(Record(M=m, N=Down_N, K=Down_K, TP=config.tp_size))
        return records

    def calculate_throughput(self, m, n, k, latency):
        throughput = 2 * m * n * k / 1e6 / latency # TFlops
        return throughput

    def calculate_bandwidth(self, m, n, k, latency, in_dtype, wei_dtype, out_dtype):
        bandwidth = (m * k * sys.getsizeof(in_dtype) + k * n * sys.getsizeof(wei_dtype) + m * n * sys.getsizeof(out_dtype)) / latency / 1e6
        return bandwidth

    def test_gemm_a8w8_pertoken_quant(self, l_dtype, l_quantDtype, records):
        df = []
        metrics = []
        for dtype in l_dtype:
            for quantDtype in l_quantDtype:
                for idx, record in enumerate(records):
                    print("dtype={}".format(dtype))
                    print("quantDtype={}".format(quantDtype))
                    ret = test_gemm(dtype, record.M, record.N, record.K, quantDtype)
                    print("--ret={}".format(ret))
                    ck_time=ret['ck us']
                    ck_bpreshuffle_time=ret['ck bpreshuffle us']
                    asm_time=ret['asm us']
                    latency = min(ck_time, ck_bpreshuffle_time)
                    if asm_time is not None:
                        latency = min(latency, asm_time)
                    print("---latency={}".format(latency))
                    records[idx].latency = latency
                    records[idx].quant_type = quantDtype
                    records[idx].output_type = dtype
                    records[idx].throughput = self.calculate_throughput(record.M, record.N, record.K, latency)
                    records[idx].bandwidth = self.calculate_bandwidth(record.M, record.N, record.K, latency, quantDtype, quantDtype, dtype)
            
        # df = pd.DataFrame(df)
        # print("df={}".format(df))
        # aiter.logger.info(f"summary:\n{df}")


    def run_single_test(self, config: TestConfig, timeout: int = 1000) -> Dict[str, Any]:
        """Run a single test with given configuration"""        
        records = self.get_gemm_shape(config)
        self.test_gemm_a8w8_pertoken_quant([dtypes.d_dtypes["bf16"]], [dtypes.d_dtypes["fp8"]], records)
        print("--records={}".format(records))
        return records

    def save_structs_to_csv(self, records, csv_file, fieldnames):
        import csv
        from dataclasses import dataclass, asdict
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
            ["M", "N", "K", "TP", "quant_type", "output_type", "latency", "throughput", "bandwidth"],
        )

    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all defined tests"""
        logger.info("Starting all MoE 2-stage tests...")
        logger.info(f"Total tests: {len(TEST_CONFIGS)}")
        records = self.run_single_test(TEST_CONFIGS[0])
        self.save_to_csv(records, TEST_CONFIGS[0].model_name)


def main():
    """Main function - runs all tests directly"""
    # Initialize and run tests
    runner = FusedmoeTestRunner()
    results = runner.run_all_tests()
    # runner.print_summary(results)

    # Exit with appropriate code
    #failed_count = sum(1 for r in results if r['status'] in ['FAIL', 'ERROR'])
    #sys.exit(1 if failed_count > 0 else 0)

if __name__ == "__main__":
    main()