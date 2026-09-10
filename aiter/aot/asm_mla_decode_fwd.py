import concurrent.futures
from collections import namedtuple

from aiter_worker_limits import (
    adopt_legacy_max_jobs,
    configure_worker_subprocesses,
    get_worker_count_for,
)
from csrc.cpp_itfs.mla.asm_mla_decode_fwd import compile

MLAConfig = namedtuple(
    "MLAConfig",
    [
        "gqa_ratio",
        "page_size",
        "q_dtype",
        "kv_dtype",
        "num_kv_splits",
        "v_head_dim",
    ],
)


def process_config(config) -> None:
    # The compiled ctypes function is process-local; only success/failure
    # should cross the ProcessPoolExecutor boundary.
    compile(
        config.gqa_ratio,
        config.page_size,
        config.q_dtype,
        config.kv_dtype,
        config.num_kv_splits,
        config.v_head_dim,
    )


def main():
    configs = []
    for num_kv_splits in range(1, 17):
        configs.append(
            MLAConfig(
                gqa_ratio=16,
                page_size=1,
                q_dtype="__hip_bfloat16",
                kv_dtype="__hip_bfloat16",
                num_kv_splits=num_kv_splits,
                v_head_dim=512,
            )
        )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=get_worker_count_for(len(configs)),
        initializer=configure_worker_subprocesses,
    ) as executor:
        # Consume the iterator so worker compilation errors reach the caller.
        list(executor.map(process_config, configs))


if __name__ == "__main__":
    adopt_legacy_max_jobs()
    main()
