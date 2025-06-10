from collections import namedtuple
import os
import concurrent.futures
from csrc.cpp_itfs.pa.pa_ragged import compile

PAConfig = namedtuple(
    "PAConfig",
    [
        "gqa_ratio",
        "head_size",
        "npar_loops",
        "block_size",
        "dtype",
        "kv_dtype",
        "fp8_kv_dtype",
        "out_dtype",
        "alibi_enabled",
    ],
)


def process_config(config):
    return compile(
        config.gqa_ratio,
        config.head_size,
        config.npar_loops,
        config.dtype,
        config.kv_dtype,
        config.fp8_kv_dtype,
        config.out_dtype,
        config.block_size,
        config.alibi_enabled,
    )


def main():
    configs = []
    for gqa_ratio in range(1, 17):
        for alibi_enabled in ["false", "true"]:
            for block_size in [1, 16, 32]:
                for npar_loops in range(1, 9):
                    for head_size in [64, 128]:
                        configs.append(
                            PAConfig(
                                gqa_ratio=gqa_ratio,
                                head_size=head_size,
                                npar_loops=npar_loops,
                                dtype="_Float16",
                                kv_dtype="_Float16",
                                fp8_kv_dtype="auto",
                                out_dtype="_Float16",
                                block_size=block_size,
                                alibi_enabled=alibi_enabled,
                            )
                        )
                        configs.append(
                            PAConfig(
                                gqa_ratio=gqa_ratio,
                                head_size=head_size,
                                npar_loops=npar_loops,
                                dtype="__hip_bfloat16",
                                kv_dtype="__hip_bfloat16",
                                fp8_kv_dtype="auto",
                                out_dtype="__hip_bfloat16",
                                block_size=block_size,
                                alibi_enabled=alibi_enabled,
                            )
                        )
                        configs.append(
                            PAConfig(
                                gqa_ratio=gqa_ratio,
                                head_size=head_size,
                                npar_loops=npar_loops,
                                dtype="_Float16",
                                kv_dtype="uint8_t",
                                fp8_kv_dtype="fp8",
                                out_dtype="_Float16",
                                block_size=block_size,
                                alibi_enabled=alibi_enabled,
                            )
                        )
                        configs.append(
                            PAConfig(
                                gqa_ratio=gqa_ratio,
                                head_size=head_size,
                                npar_loops=npar_loops,
                                dtype="__hip_bfloat16",
                                kv_dtype="uint8_t",
                                fp8_kv_dtype="fp8",
                                out_dtype="__hip_bfloat16",
                                block_size=block_size,
                                alibi_enabled=alibi_enabled,
                            )
                        )

    with concurrent.futures.ProcessPoolExecutor(
        os.environ.get("MAX_JOBS", 16)
    ) as executor:
        executor.map(process_config, configs)


if __name__ == "__main__":
    main()
