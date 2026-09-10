import concurrent.futures
from collections import namedtuple

from aiter_worker_limits import (
    adopt_legacy_max_jobs,
    configure_worker_subprocesses,
    get_worker_count_for,
)
from csrc.cpp_itfs.pa.pa_v1 import compile

PAConfig = namedtuple(
    "PAConfig",
    [
        "gqa_ratio",
        "head_size",
        "npar_loops",
        "dtype",
        "kv_dtype",
        "fp8_kv_dtype",
        "out_dtype",
        "block_size",
        "alibi_enabled",
        "logits_soft_cap_enabled",
    ],
)


def process_config(config) -> None:
    # The compiled ctypes function is process-local; only success/failure
    # should cross the ProcessPoolExecutor boundary.
    compile(
        config.gqa_ratio,
        config.head_size,
        config.npar_loops,
        config.dtype,
        config.kv_dtype,
        config.fp8_kv_dtype,
        config.out_dtype,
        config.block_size,
        config.alibi_enabled,
        config.logits_soft_cap_enabled,
    )


def main():
    configs = []
    for gqa_ratio in range(1, 17):
        for alibi_enabled in [False, True]:
            for logits_soft_cap_enabled in [False, True]:
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
                                    logits_soft_cap_enabled=logits_soft_cap_enabled,
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
                                    logits_soft_cap_enabled=logits_soft_cap_enabled,
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
                                    logits_soft_cap_enabled=logits_soft_cap_enabled,
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
                                    logits_soft_cap_enabled=logits_soft_cap_enabled,
                                )
                            )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=get_worker_count_for(len(configs)),
        initializer=configure_worker_subprocesses,
    ) as executor:
        list(executor.map(process_config, configs))


if __name__ == "__main__":
    adopt_legacy_max_jobs()
    main()
