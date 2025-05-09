import functools
import json
from pathlib import Path
from typing import Any, Dict, Optional
from warnings import warn

import torch

from op_tests.triton_tests.utils.moe_align_block_size_ref import \
    torch_moe_align_block_size
from ..common import str_to_torch_dtype
from ..quantization import quantize_fp8, quantize_int4, quantize_int8

M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024


################################################################################
############################### Config Retrival ################################
################################################################################
def get_config_file_name(dtype: Optional[str]) -> str:
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    return f"device_name={device_name}{dtype_selector}.json"


@functools.lru_cache
def get_moe_configs(dtype: Optional[str]) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(dtype)
    config_dir_path = Path(__file__).resolve().parent / "configs"
    config_file_path = config_dir_path / json_file_name

    # If a configuration has been found, return it
    if config_file_path.exists():
        with open(config_file_path) as f:
            return {key: val for key, val in json.load(f).items()}
    # If no config file is found, we will use the default configuration
    default_config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1
    }
    default_configs = {"small_M": default_config,
                       "medium_M": default_config,
                       "large_M": default_config}
    warn(f"Intended MoE config file {config_file_path} not found; use default config {default_config=}")
    return default_configs


def get_optimal_moe_config(input_dtype: str, aux_dtype: str, M: int):
    check_moe_dtype(input_dtype, aux_dtype)
    configs = get_moe_configs(input_dtype)

    if M < M_THRESHOLD_SMALL:
        config = configs["small_M"]
    elif M < M_THRESHOLD_MEDIUM:
        config = configs["medium_M"]
    else:
        config = configs["large_M"]

    return config


# TODO: add "int8_w8a8" type
def check_moe_dtype(input_dtype: Optional[str], aux_dtype: str):
    """Check the compliance on MoE inputs datatype

    Args:
        input_dtype (Optional[str]): Describes the compound type of inputs. One
            of the [None, "fp8_w8a8", "int8_w8a16", "int4_w4a16"]
        aux_dtype (str): Describes the exact type in input_type. E.g. int8_w8a16
            has weights in int8 but a16 can be fp16 or bf16. This variable
            clarifies the exact type of a16.
    """
    assert input_dtype in [None, "fp8_w8a8", "int8_w8a16", "int4_w4a16"], \
        "input_dtype can only be None or one of (fp8_w8a8, int8_w8a16, int4_w4a16)"

    match input_dtype:
        case "fp8_w8a8":
            assert aux_dtype in ["fp8_e4m3", "fp8_e5m2"], \
                "When input_dtype is fp8_w8a8, activation can only be fp8_e4m3 or fp8_e5m2"
        case ["int8_w8a16", "int4_w4a16"]:
            assert aux_dtype in ["bf16", "fp16"], \
                f"When input_dtype is {input_dtype}, activation can only be fp16 or bf16"
        case _:
            assert aux_dtype in ["bf16", "fp16"], \
                "When input dtype is unspecified, it can only be fp16 or bf16"


################################################################################
############################### Input Generation ###############################
################################################################################
def generate_moe_logits(M, E, top_k):
    # simulate the random logits
    values = torch.randn(M, E, dtype=torch.float16, device='cuda')
    softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)
    return topk_weights, topk_ids


def generate_moe_alignment(M, E, top_k, BLOCK_M):
    """Generate simulated MoE alignment from tokens to experts
    """
    topk_weights, topk_ids = generate_moe_logits(M, E, top_k)
    sorted_token_ids, expert_ids, num_tokens_post_padded = \
        torch_moe_align_block_size(topk_ids, block_size=BLOCK_M, num_experts=E)

    return topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded


def quantize_b_int4(b, group_size, has_zp):
    """Quantize operand B in MoE (expert tensor) to int4 format
    """
    E, N, K = b.shape
    b_q = torch.empty((E, N, K // 2), dtype=torch.uint8, device='cuda')
    # TODO: this assumes b_scale is torch.float32 because it is used as fp32 in triton kernel
    b_scale = torch.empty((E, N, K // group_size), dtype=torch.float32, device='cuda')
    if has_zp:
        b_zp = torch.empty((E, N // 2, K // group_size), dtype=torch.uint8, device='cuda')
    else:
        b_zp = None

    for e in range(E):
        q, scale, zp = quantize_int4(b[e].T, group_size=group_size, has_zp=has_zp)
        q = q.T
        q = q[:, 1::2] * 16 + q[:, ::2] #Note, 2<<4=16. For bf16, etc, torch doesn't have shift.
        b_q[e] = q
        b_scale[e] = scale.T
        if has_zp:
            zp = zp.T.contiguous().to(torch.uint8)
            zp = zp[1::2, :] << 4 | zp[::2, :] #Note, 2<<4=16. For bf16, etc, torch doesn't have shift.
            b_zp[e] = zp
    b = b_q

    return b, b_scale, b_zp


def generate_moe_input(M: int, N: int, K: int, top_k: int, E: int,
                 group_size, has_zp,
                 input_dtype: Optional[str] = None,
                 aux_dtype: Optional[str] = None):
    """Generate MoE test inputs based on the input and output datatypes

    Input datatypes can only be one of the four:
    * fp8_w8a8: Activation and weight are both fp8
    * int8_w8a16: Activation is 16-bit and weight is int8
    * int4_w4a16: Activation is 16-bit and weight is int4
    * None: default to have everything

    aux_dtype specifies the ambiguous datatype of inputs and output.
    For w8a8 it speicifies the dtype of fp8; for w8a16/w4a16 it specifies
    the dtype of 16-bit (fp16 or bf16)
    """

    check_moe_dtype(input_dtype, aux_dtype)
    aux_dtype = str_to_torch_dtype[aux_dtype]

    a = torch.randn((M, K) , dtype=torch.float16, device='cuda')
    b = torch.randn((E, N, K), dtype=torch.float16, device='cuda')
    c = torch.zeros((M, top_k, N), dtype=torch.float16, device='cuda')
    c_silu = torch.zeros((M * top_k, N // 2), dtype=torch.float16, device='cuda')

    # default to fp16
    match input_dtype:
        case "fp8_w8a8":
            b_zp = None
            a, _, a_scale = quantize_fp8(a, dim=(0,))
            b, _, b_scale = quantize_fp8(b, dim=(0,))
        case "int8_w8a16":
            a_scale = b_zp = None
            a = a.to(aux_dtype)
            b, _, b_scale = quantize_int8(b, dim=(0,))
        case "int4_w4a16":
            a_scale = None
            a = a.to(aux_dtype)
            b, b_scale, b_zp = quantize_b_int4(b, group_size, has_zp)
        case _:
            a_scale = b_scale = b_zp = None
            a = a.to(aux_dtype)
            b = b.to(aux_dtype)

    c = c.to(aux_dtype)
    c_silu = c_silu.to(aux_dtype)

    return a, b, c, c_silu, b_zp, a_scale, b_scale