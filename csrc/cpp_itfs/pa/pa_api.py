import torch
import ctypes
from jinja2 import Template
from functools import lru_cache

from csrc.cpp_itfs.utils import AITER_CORE_DIR, compile_template_op

MD_NAME = "pa"

with open(f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa.cpp.jinja", "r") as f:
    src_template = Template(f.read())

dtype_map = {
    torch.bfloat16: "__hip_bfloat16",
    torch.float16: "_Float16",
    torch.float8_e4m3fnuz: "uint8_t",
    torch.float8_e4m3fn: "uint8_t",
}

@lru_cache(maxsize=None)
def load_all_libs(
    gqa_ratio: int,
    head_size: int,
    npar_loops: int,
    dtype: str,
    kv_dtype: str,
    fp8_kv_dtype: str,
    out_dtype: str,
    block_size: int,
    alibi_enabled: str,
    mtp: int = 1,
    quant_method: str = "vllm::Fp8QuantMethod::kPerTensor",
    v_shuffle: bool = False,
    folder: str = None
):
    combined_func = compile_template_op(
        src_template,
        MD_NAME,
        [
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/utils.h",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa.cuh",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_common.cuh",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_kernels.cuh",
            f"{AITER_CORE_DIR}/csrc/include",
            f"{AITER_CORE_DIR}/csrc/include/ck_tile",
        ],
        gqa_ratio=gqa_ratio,
        head_size=head_size,
        npar_loops=npar_loops,
        dtype=dtype,
        kv_dtype=kv_dtype,
        fp8_kv_dtype=fp8_kv_dtype,
        out_dtype=out_dtype,
        block_size=block_size,
        alibi_enabled=alibi_enabled,
        mtp=mtp,
        quant_method=quant_method,
        v_shuffle=v_shuffle,
        folder=folder,
    )

    return ctypes.cast(combined_func, ctypes.c_void_p).value


if __name__ == "__main__":
    load_all_libs(4, 128, 4, "__hip_bfloat16", "__hip_bfloat16", "auto", "__hip_bfloat16", 16, "false", 1, "vllm::Fp8QuantMethod::kPerTensor", True)
