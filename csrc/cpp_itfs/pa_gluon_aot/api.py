import ctypes
from functools import lru_cache

import aiter
import torch

from csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot import compile

dtype_map = {
    'float32': torch.float32,
    'float': torch.float32,
    'float64': torch.float64,
    'double': torch.float64,
    'float16': torch.float16,
    'half': torch.float16,
    'bfloat16': torch.bfloat16,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int': torch.int32,
    'int64': torch.int64,
    'long': torch.int64,
    'uint8': torch.uint8,
    'bool': torch.bool,
    'complex64': torch.complex64,
    'complex128': torch.complex128,
    'fp8': aiter.dtypes.fp8,
}


@lru_cache(maxsize=None)
def load_all_libs(
    compute_type: str,
    query_length: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    kv_block_size: int,
    context_partition_size: int,
    query_quant_mode: int,  # -1: no quant, 0: per-tensor, 1: per-token
    kv_quant_mode: int,  # -1: no quant, 0: per-tensor, 1: per-token
    kv_cache_dtype: str,
    value_transposed: bool,
    use_sinks: int,
    cdna_version: int,
):
    query_group_size = num_query_heads // num_kv_heads
    compute_type_torch = dtype_map[compute_type]

    fp8_max_value = 1.0
    if kv_quant_mode >= 0:
        fp8_max_value = torch.finfo(aiter.dtypes.fp8).max

    pa_decode_func = compile(
        compute_type=compute_type_torch,
        query_seq_len=query_length,
        one_query_group_size=query_group_size,
        head_size=head_size,
        kv_block_size=kv_block_size,
        context_partition_size=context_partition_size,
        query_quant_mode=query_quant_mode,
        kv_quant_mode=kv_quant_mode,
        fp8_max_value=fp8_max_value,
        value_transposed=int(value_transposed),
        is_causal=int(query_length > 1),
        use_sinks=use_sinks,
        cdna_version=cdna_version,
    )

    return ctypes.cast(pa_decode_func, ctypes.c_void_p).value


if __name__ == "__main__":
    load_all_libs(
        compute_type="bfloat16",
        query_length=4,
        num_query_heads=128,
        num_kv_heads=16,
        head_size=128,
        kv_block_size=16,
        context_partition_size=256,
        query_quant_mode=-1,
        kv_quant_mode=1,
        kv_cache_dtype="fp8",
        value_transposed=False,
        use_sinks=0,
        cdna_version=3,
    )
