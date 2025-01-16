from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, ATER_ROOT_DIR
import torch.nn.functional as F

MD_NAME = "module_moe"


@compile_ops("module_moe")
def topk_softmax(
    topk_weights: Tensor,
    topk_indices: Tensor,
    token_expert_indices: Tensor,
    gating_output: Tensor,
    need_renorm: bool,
): ...


@compile_ops("module_moe")
def moe_sum(input: Tensor, output: Tensor): ...


@compile_ops("module_moe")
def moe_align_block_size(
    topk_ids: Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: Tensor,
    experts_ids: Tensor,
    token_nums: Tensor,
    num_tokens_post_pad: Tensor,
): ...


@compile_ops("module_moe")
def fmoe(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weight_buf: Tensor,
    sorted_expert_ids: Tensor,
    num_tokens_post_padded: Tensor,
    topk: int,
): ...


@compile_ops("module_moe")
def fmoe_int8_g1u0(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weight_buf: Tensor,
    sorted_expert_ids: Tensor,
    num_tokens_post_padded: Tensor,
    topk: int,
    input_scale: Tensor,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc2_smooth_scale: Tensor,
): ...


@compile_ops("module_moe")
def fmoe_int8_g1u1(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weight_buf: Tensor,
    sorted_expert_ids: Tensor,
    num_tokens_post_padded: Tensor,
    topk: int,
    input_scale: Tensor,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc2_smooth_scale: Tensor,
): ...


@compile_ops("module_moe")
def fmoe_int8_g1u0_a16(
    out: Tensor,
    input: Tensor,  # bf16
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weight_buf: Tensor,
    sorted_expert_ids: Tensor,
    num_tokens_post_padded: Tensor,
    topk: int,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc1_smooth_scale: Tensor,
    fc2_smooth_scale: Tensor,
): ...
