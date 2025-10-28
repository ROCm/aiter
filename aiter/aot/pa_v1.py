from collections import OrderedDict
from jinja2 import Template

from csrc.cpp_itfs.pa.pa_v1 import MD_NAME
from csrc.cpp_itfs.utils import compile_template_ops, AITER_CORE_DIR

with open(f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_v1.cpp.jinja", "r") as f:
    src_template = Template(f.read())

def compile_aot(
    configs: list[OrderedDict],
    folder: str,
):
    return compile_template_ops(
        src_template,
        MD_NAME,
        configs,
        folder,
        includes=[
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/utils.h",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_kernels.cuh",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_v1.cuh",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_common.cuh",
            f"{AITER_CORE_DIR}/csrc/include",
            f"{AITER_CORE_DIR}/csrc/include/ck_tile/",
        ],
    )



def main():
    configs = []
    for gqa_ratio in [1, 5, 8, 10, 16]:
        for alibi_enabled in [False]:
            for logits_soft_cap_enabled in [False]:
                for block_size in [1, 16, 32]:
                    for npar_loops in range(1, 9):
                        for head_size in [64, 128]:
                            configs.append(
                                OrderedDict(
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
                                    partition_size=256,
                                    mtp=1,
                                )
                            )
                            configs.append(
                                OrderedDict(
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
                                    partition_size=256,
                                    mtp=1,
                                )
                            )
                            configs.append(
                                OrderedDict(
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
                                    partition_size=256,
                                    mtp=1,
                                )
                            )
                            configs.append(
                                OrderedDict(
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
                                    partition_size=256,
                                    mtp=1,
                                )
                            )

    compile_aot(
        configs=configs,
        folder="pa_v1_aot",
    )


if __name__ == "__main__":
    main()
