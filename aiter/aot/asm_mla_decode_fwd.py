from collections import namedtuple
import os
import concurrent.futures
from csrc.cpp_itfs.mla.asm_mla_decode_fwd import compile
from csrc.cpp_itfs.utils import AITER_CORE_DIR

MLAConfig = namedtuple(
    "MLAConfig",
    [
        "hsaco_path",
        "page_size",
        "q_itemsize",
        "kv_itemsize",
        "num_kv_splits",
        "v_head_dim",
    ],
)


def process_config(config):
    return compile(
        config.hsaco_path,
        config.page_size,
        config.q_itemsize,
        config.kv_itemsize,
        config.num_kv_splits,
        config.v_head_dim,
    )


def main():
    configs = []
    for num_kv_splits in range(1, 17):
        configs.append(
            MLAConfig(
                hsaco_path=f"{AITER_CORE_DIR}/hsa/mla_stage1_a16w16_bf16.co",
                page_size=1,
                q_itemsize=2,
                kv_itemsize=2,
                num_kv_splits=num_kv_splits,
                v_head_dim=512,
            )
        )

    with concurrent.futures.ProcessPoolExecutor(
        os.environ.get("MAX_JOBS", 16)
    ) as executor:
        executor.map(process_config, configs)


if __name__ == "__main__":
    main()
