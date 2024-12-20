import torch
from torch import Tensor
from typing import List, Optional
import functools
import pandas as pd
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, ATER_ROOT_DIR

MD_NAME = "module_gemm_a8w8"


@compile_ops(
    srcs=[f"{ATER_CSRC_DIR}/pybind/gemm_a8w8_pybind.cu",
        f"{ATER_CSRC_DIR}/ck_gemm_a8w8/gemm_a8w8.cu",
        f"{ATER_CSRC_DIR}/ck_gemm_a8w8/include",],
    blob_gen_cmd =  f"{ATER_CSRC_DIR}/ck_gemm_a8w8/gen_instances.py --working_path {{}} --tune_file ater/configs/a8w8_tuned_gemm.csv",
    md_name=MD_NAME,
    fc_name="gemm_a8w8",
)
def gemm_a8w8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
): ...


@compile_ops(
    srcs=[f"{ATER_CSRC_DIR}/pybind/gemm_a8w8_asm_pybind.cu",
        f"{ATER_CSRC_DIR}/py_itfs_cu/asm_gemm_a8w8.cpp"],
    md_name="module_gemm_a8w8_asm",
    flags_extra_hip=[f'-DATER_ASM_DIR=\\"{ATER_ROOT_DIR}/hsa/\\"'],
    fc_name="gemm_a8w8_asm",
)
def gemm_a8w8_asm(
    XQ: Tensor,             # A:[M, K] i8
    WQ: Tensor,             # B:[N, K] i8 -> shuffle layout(32,16)
    x_scale: Tensor,        # A_scale:[M, 1] f32
    w_scale: Tensor,        # B_scale:[1, N] f32
    out: Tensor,            # Out:[M, N] bf16
    bias: Tensor,           # bias:[1, N] f32
    sub_m: Optional[Tensor] = 128,
    sub_n: Optional[Tensor] = 128,
    pad_a: Optional[Tensor] = 0,
    pad_b: Optional[Tensor] = 0,
    pad_c: Optional[Tensor] = 0,
    splitK: Optional[Tensor] = 0,
): ...

@functools.lru_cache(maxsize=1024)
def get_ASMGEMM_config(
    M: int,
    N: int,
    K: int,
    bias: bool,
    dtype: torch.dtype
):
    if not hasattr(get_ASMGEMM_config, "asmgemm_dict"):
        asmGemmDictDf = pd.read_csv(f"{ATER_ROOT_DIR}/ater/configs/asm_a8w8_gemm.csv").drop_duplicates()
        asmGemmDictDf.bias = asmGemmDictDf.bias.apply(lambda s: True if s in ['True',1,'true'] else False)
        get_ASMGEMM_config.asmgemm_dict = asmGemmDictDf.set_index(['M','N','K','bias','outdtype']).to_dict('index')
    return get_ASMGEMM_config.asmgemm_dict.get((M,N,K,bias,str(dtype)), None)

def gemm_a8w8_ASM(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Tensor,
    dtype=torch.bfloat16,
    check = False
):
    """
    Notes for use gemm_a8w8_ASM:
    1. WQ(weight) must be shuffle, you can use \
        'weightshuffle = shuffle_weight(weight,layout=(32,16))'
    2. Use asm gemm must give bias, if not have bias, please give  \
        'bias=torch.zeros(n,dtype=torch.float32,device='cuda')'
    """
    if check:
        assert dtype in [torch.bfloat16,], \
            f"Output {dtype=} is currently not supported in gemm_a8w8_ASM"
        assert x_scale.dtype == torch.float32 and w_scale.dtype == torch.float32, \
            f"{x_scale.dtype=} or {w_scale.dtype=} must be torch.float32"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[-1]
    if x_scale.dtype == torch.float32 and w_scale.dtype == torch.float32 and \
        (asm_config := get_ASMGEMM_config(m,n,k,bias!=None,dtype)) != None:
        assert bias != None, "Use asm gemm must give bias, please give a \
            bias=torch.zeros(n,dtype=torch.float32,device='cuda')"
        splitK = asm_config['splitK']
        Y = torch.empty(m, n, dtype=dtype, device="cuda")
        return gemm_a8w8_asm(XQ, WQ, x_scale, w_scale, Y, bias, splitK=splitK)
    return None

def gemm_a8w8_CK(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    bias: Optional[Tensor] = None,
    dtype=torch.bfloat16,
):
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    Y = torch.empty(m, n, dtype=dtype, device="cuda")
    return gemm_a8w8(XQ, WQ, x_scale, w_scale, Y, bias)

@compile_ops(
    srcs=[f"{ATER_CSRC_DIR}/pybind/gemm_a8w8_tune_pybind.cu",
          f"{ATER_CSRC_DIR}/ck_gemm_a8w8/gemm_a8w8_tune.cu",
          f"{ATER_CSRC_DIR}/ck_gemm_a8w8/include",],
    blob_gen_cmd =  f"{ATER_CSRC_DIR}/ck_gemm_a8w8/gen_instances.py --working_path {{}} --tune",
    md_name="module_gemm_a8w8_tune",
    fc_name="gemm_a8w8_tune",
)
def gemm_a8w8_tune(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    kernelId: int,
): ...