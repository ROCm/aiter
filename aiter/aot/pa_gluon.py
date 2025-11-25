import os
import concurrent.futures
from csrc.cpp_itfs.utils import compile_hsaco_from_triton
from aiter.ops.triton.gluon.pa_decode_gluon import (
    paged_attention_decode_v2_gluon_fp8,
    paged_attention_decode_v2_reduce_kernel,
)
import torch
import triton.language as tl


def process_config(config):
    # print(config)
    (
        COMPUTE_TYPE,
        QUERY_GROUP_SIZE_POW2,
        HEAD_SIZE_POW2,
        KV_BLOCK_SIZE,
        CONTEXT_PARTITION_SIZE,
        KV_COMPUTE_BLOCK_SIZE,
        QUERY_QUANT_MODE,
        KV_QUANT_MODE,
        FP8_MAX_VALUE,
        VALUE_TRANSPOSED,
        IS_CAUSAL,
        waves_per_eu,
    ) = config
    compile_hsaco_from_triton(
        paged_attention_decode_v2_gluon_fp8.triton_kernel,
        torch.float32,
        torch.float32,
        torch.bfloat16 if COMPUTE_TYPE == tl.float8e4b8 else COMPUTE_TYPE,
        torch.float8_e4m3fnuz,
        torch.float8_e4m3fnuz,
        torch.float8_e4m3fnuz,
        torch.int32,
        torch.int32,
        0.125,
        torch.float32,
        torch.float32,
        torch.float32,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        COMPUTE_TYPE,
        QUERY_GROUP_SIZE_POW2,
        HEAD_SIZE_POW2,
        KV_BLOCK_SIZE,
        CONTEXT_PARTITION_SIZE,
        KV_COMPUTE_BLOCK_SIZE,
        QUERY_QUANT_MODE,
        KV_QUANT_MODE,
        FP8_MAX_VALUE,
        VALUE_TRANSPOSED,
        IS_CAUSAL,
        grid=(1, 1, 1),
        waves_per_eu=waves_per_eu,
        num_stages=1,
    )
    compile_hsaco_from_triton(
        paged_attention_decode_v2_reduce_kernel.triton_kernel,
        torch.bfloat16 if COMPUTE_TYPE == tl.float8e4b8 else COMPUTE_TYPE,
        torch.float32,
        torch.float32,
        torch.float32,
        torch.int32,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        QUERY_GROUP_SIZE_POW2,
        HEAD_SIZE_POW2,
        CONTEXT_PARTITION_SIZE,
        grid=(1, 1, 1),
        waves_per_eu=waves_per_eu,
        num_stages=1,
    )


def main():
    configs = []
    for COMPUTE_TYPE in [tl.float8e4b8]:
        for QUERY_GROUP_SIZE_POW2 in [16, 64]:
            for HEAD_SIZE_POW2 in [64, 128]:
                for CONTEXT_PARTITION_SIZE in [256]:
                    for KV_BLOCK_SIZE in [16, 64]:
                        for KV_COMPUTE_BLOCK_SIZE in [256]:
                            for QUERY_QUANT_MODE, KV_QUANT_MODE in [(1, 1), (0, 0)]:
                                for FP8_MAX_VALUE in [240.0]:
                                    for VALUE_TRANSPOSED in [0, 1]:
                                        for IS_CAUSAL in [0, 1]:
                                            if QUERY_GROUP_SIZE_POW2 == 64:
                                                waves_per_eu = 3
                                            else:
                                                waves_per_eu = 4
                                            configs.append(
                                                [
                                                    COMPUTE_TYPE,
                                                    QUERY_GROUP_SIZE_POW2,
                                                    HEAD_SIZE_POW2,
                                                    KV_BLOCK_SIZE,
                                                    CONTEXT_PARTITION_SIZE,
                                                    KV_COMPUTE_BLOCK_SIZE,
                                                    QUERY_QUANT_MODE,
                                                    KV_QUANT_MODE,
                                                    FP8_MAX_VALUE,
                                                    VALUE_TRANSPOSED,
                                                    IS_CAUSAL,
                                                    waves_per_eu,
                                                ]
                                            )

    with concurrent.futures.ProcessPoolExecutor(
        os.environ.get("MAX_JOBS", os.cpu_count())
    ) as executor:
        executor.map(process_config, configs)


if __name__ == "__main__":
    main()
