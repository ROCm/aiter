import torch
from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR

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


def gemm_a8w8_bias(
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
    gemm_a8w8(XQ, WQ, x_scale, w_scale, Y, bias)
    return Y


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