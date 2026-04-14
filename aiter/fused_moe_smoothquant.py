import torch

from .fused_moe import moe_sorting

from csrc.cpp_itfs.hsaco_tools import get_kernel
from csrc.cpp_itfs.utils import AITER_CORE_DIR


def div_up(x, y):
    return (x + y - 1) // y


def fused_moe_gelu_sqi8(
    hidden_states,  # [num_tokens, model_dim] torch.bfloat16
    w1,  # [expert, inter_dim, model_dim] torch.int8
    w2,  # [expert, model_dim, inter_dim] torch.int8
    topk_weight,  # [num_tokens, topk] torch.float32
    topk_ids,  # [num_tokens, topk] torch.int32
    w1_scale,  # [expert, inter_dim, 1] torch.float32
    w2_scale,  # [expert, model_dim, 1] torch.float32
    a1_smooth_scale,  # [expert, 1, model_dim] torch.float32
    a2_smooth_scale,  # [expert, 1, inter_dim] torch.float32):
):
    device = hidden_states.device
    dtype = hidden_states.dtype

    def get_inter_dim(w1_shape, w2_shape):
        E, _, model_dim = w1_shape
        E, model_dim, inter_dim = w2_shape

        int4_war = model_dim // w1_shape[-1]
        inter_dim *= int4_war
        return E, model_dim, inter_dim

    M, topk = topk_ids.shape
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)
    w1_is_shuffled = getattr(w1, "is_shuffled", False)
    w2_is_shuffled = getattr(w2, "is_shuffled", False)

    assert not w1_is_shuffled
    assert not w2_is_shuffled

    block_size_M = 256

    expert_mask = None
    num_local_tokens = None
    moe_sorting_dispatch_policy = 0

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_out = moe_sorting(
        topk_ids,
        topk_weight,
        E,
        model_dim,
        dtype,
        block_size_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
    )

    def quant_act(
        x,
        topk,
        M,
        model_dim,
        smooth_scale,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        is_gemm1=True,
        topk_ids=None,
    ):
        kwargs = {
            "BLOCK_SIZE_M": 256,
            "TOPK": topk,
            "ROW_PER_BLOCK": 4,
            "ROW_PER_BLOCK2": 4,
            "ROW_PER_BLOCK1": 1,
            "BLOCK_M2": 8,
            "QUANT1_K": model_dim,
            "QUANT2_K": model_dim,
        }
        compile_time_args = tuple(kwargs.items())
        device = x.device
        x_quant = torch.empty((M, topk, model_dim), dtype=torch.int8, device=device)
        x_quant_scale = torch.empty(
            [sorted_ids.shape[0]], dtype=torch.float32, device=device
        )
        if is_gemm1:
            get_kernel(
                f"{AITER_CORE_DIR}/hsa/gfx950/fmoe_smoothquant/quant-i8:quant1",
                compile_time_args,
            )(
                [2 * div_up(M, kwargs["ROW_PER_BLOCK1"])],
                [64],
                x,
                smooth_scale,
                x_quant,
                x_quant_scale,
                topk_ids,
                M,
            )
            x_quant_scale.is_sorted = False
        else:
            get_kernel(
                f"{AITER_CORE_DIR}/hsa/gfx950/fmoe_smoothquant/quant-i8:quant2",
                compile_time_args,
            )(
                [
                    sorted_expert_ids.shape[0],
                    kwargs["BLOCK_SIZE_M"] // kwargs["BLOCK_M2"],
                ],
                [256],
                x,
                smooth_scale,
                x_quant,
                x_quant_scale,
                sorted_ids,
                sorted_expert_ids,
                num_valid_ids,
                M,
            )
        return x_quant, x_quant_scale

    # quantize hidden_states in sorted_ids [tok_topk]
    # print(sorted_ids.dtype, sorted_ids.shape)
    def moe_gemm(output, input, pt_scale, weight, pc_scale, stage_index):
        AB_dtype = "s8"
        is_input_over_4GB = input.numel() * input.element_size() > (1 << 32)
        is_pt_scale_sorted = getattr(pt_scale, "is_sorted", True)
        wg_M, wg_N = 256, 256
        is_gate_up = stage_index == 1
        bpreshuffle = False
        num_tokens = input.shape[0]
        num_oc_blocks = output.shape[-1] // wg_N
        num_e_blocks = sorted_expert_ids.shape[0]
        kwargs = {
            "is_input_over_4GB": is_input_over_4GB,
            "is_pt_scale_sorted": is_pt_scale_sorted,
            "AB_dtype": AB_dtype,
            "wg_M": wg_M,
            "wg_N": wg_N,
            "NUM_EXPERTS": E,
            "OC": output.shape[-1],
            "IC": input.shape[-1],
            "gate_up": is_gate_up,
            "bpreshuffle": bpreshuffle,
            "TOPK": topk,
        }
        get_kernel(
            f"{AITER_CORE_DIR}/hsa/gfx950/fmoe_smoothquant/moe_gemm_8wave_gelu",
            tuple(kwargs.items()),
        )(
            [num_oc_blocks * num_e_blocks],
            [8 * 64],
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            weight,
            pc_scale,
            input,
            pt_scale,
            output,
            num_tokens,
            num_oc_blocks * num_e_blocks,
        )

    num_tokens, _ = hidden_states.shape
    a1, a1_scale = quant_act(
        hidden_states,
        topk,
        hidden_states.shape[0],
        hidden_states.shape[1],
        a1_smooth_scale,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        True,
        topk_ids,
    )
    a2_bf16 = torch.empty(
        (num_tokens, topk, inter_dim),
        dtype=torch.bfloat16,
        device=device,
    )

    moe_gemm(a2_bf16, a1, a1_scale, w1, w1_scale, 1)

    a2, a2_scale = quant_act(
        a2_bf16,
        topk,
        a2_bf16.shape[0],
        a2_bf16.shape[2],
        a2_smooth_scale,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        False,
        topk_ids,
    )

    stage2_out = torch.empty(
        (num_tokens, topk, model_dim), dtype=torch.bfloat16, device=device
    )

    moe_gemm(stage2_out, a2, a2_scale, w2, w2_scale, 2)

    moe_out = stage2_out.sum(dim=1)

    return moe_out
