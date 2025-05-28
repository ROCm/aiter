from jinja2 import Template
from csrc.cpp_itfs.utils import (
    compile_template_op,
    transfer_hsaco,
    AITER_CORE_DIR,
    get_default_func_name,
    not_built,
    run_lib,
    CK_DIR,
)
import ctypes

MD_NAME = "asm_moe"
with open(f"{AITER_CORE_DIR}/csrc/cpp_itfs/moe/asm_moe.cpp.jinja", "r") as f:
    src_template = Template(f.read())


def get_heuristic_tile(
    inter_dim: int, max_num_m_blocks: int, available_tiles: list, num_cu: int
):
    empty_cu = num_cu
    tg_num = 0
    round = 0xFFFFFFFF
    selected_tile = 0

    for tile in available_tiles:
        if inter_dim % tile == 0:
            tg_num = inter_dim / tile * max_num_m_blocks
            local_round = (tg_num + num_cu - 1) / num_cu
            if local_round < round:
                round = local_round
                selected_tile = tile
                empty_cu = local_round * num_cu - tg_num
            elif local_round == round:
                if empty_cu > (local_round * num_cu - tg_num):
                    round = local_round
                    selected_tile = tile
                    empty_cu = local_round * num_cu - tg_num
    return selected_tile


def select_tile(
    input_dtype: str,
    gate_fusion: str,
    gate_dtype: str,
    inter_dim: int,
    max_num_m_blocks: int,
    num_cu: int,
    a16: bool = False,
    fc_scale_blkn: int = None,
    fc_scale_blkk: int = None,
):
    if a16:
        if gate_fusion == "g1u1":
            if gate_dtype == "uint8_t":
                if inter_dim % 320 == 0:
                    return 320
                else:
                    raise ValueError(
                        f"Unsupported inter_dim {inter_dim}, which should be divisible by 320"
                    )
            elif gate_dtype == "__hip_fp8_e4m3_fnuz":
                selected_tile = get_heuristic_tile(
                    inter_dim, max_num_m_blocks, [512, 320], num_cu
                )
                if selected_tile == 0:
                    raise ValueError(
                        f"Unsupported inter_dim {inter_dim}, which should be divisible by 320 or 512"
                    )
                return selected_tile
    else:
        if gate_fusion == "g1u0":
            tiles = [512, 448, 384, 320, 256, 192, 128]
            selected_tile = 0
            for tile in tiles:
                if inter_dim % tile == 0:
                    selected_tile = tile
                    break
            if selected_tile == 0:
                raise ValueError(
                    f"Unsupported inter_dim {inter_dim}, which should be divisible by 128, 192, 256, 320, 384, 448 or 512"
                )
            return selected_tile
        elif gate_fusion == "g1u1":
            if gate_dtype in ["uint32_t", "int32_t"]:
                selected_tile = get_heuristic_tile(
                    inter_dim, max_num_m_blocks, [512, 256, 128], num_cu
                )
                if selected_tile == 0:
                    raise ValueError(
                        f"Unsupported inter_dim {inter_dim}, which should be divisible by 128, 256 or 512"
                    )
                return selected_tile
            elif input_dtype == "uint8_t":
                selected_tile = get_heuristic_tile(
                    inter_dim,
                    max_num_m_blocks,
                    [512, 448, 384, 320, 256, 192, 128],
                    num_cu,
                )
                if selected_tile == 0:
                    raise ValueError(
                        f"Unsupported inter_dim {inter_dim}, which should be divisible by 128, 192, 256, 320, 384, 448 or 512"
                    )
                return selected_tile
            elif input_dtype == "__hip_fp8_e4m3_fnuz":
                selected_tile = get_heuristic_tile(
                    inter_dim,
                    max_num_m_blocks,
                    [512, 448, 384, 320, 256, 192, 128],
                    num_cu,
                )
                if selected_tile == 0:
                    raise ValueError(
                        f"Unsupported inter_dim {inter_dim}, which should be divisible by 128, 192, 256, 320, 384, 448 or 512"
                    )
                return selected_tile
            elif input_dtype == "__hip_bfloat16":
                if (
                    inter_dim % 256 == 0
                    and fc_scale_blkn == 128
                    and fc_scale_blkk == 128
                ):
                    return 256
                else:
                    raise ValueError(
                        f"Unsupported inter_dim {inter_dim}, which should be divisible by 256 and fc_scale_blkn and fc_scale_blkk should be 128"
                    )
    return 512


def select_hsaco(
    input_dtype: str,
    gate_fusion: str,
    gate_dtype: str,
    activation: str,
    selected_tile: int,
    is_smooth_scale=False,
    a16=False,
    enable_vskip=False,
):
    if a16:
        if gate_fusion == "g1u1":
            if gate_dtype == "uint8_t":
                return (
                    f"fmoe_int8_g1u1_smf_subGU_{selected_tile}",
                    f"fmoe_int8_g1u1_smf_subGU_{selected_tile}.co",
                    "uint8_t",
                    "uint16_t",
                    "true",
                )
            elif gate_dtype == "__hip_fp8_e4m3_fnuz":
                return (
                    f"fmoe_fp8_g1u1_smf_subGU_{selected_tile}",
                    f"fmoe_fp8_g1u1_smf_subGU_{selected_tile}.co",
                    "uint8_t",
                    "uint16_t",
                    "true",
                )
        elif gate_fusion == "g1u0":
            return (
                "fmoe_kernel_func",
                "fmoe_int8_g1u0_smf.co",
                "uint8_t",
                "uint16_t",
                "true",
            )
    else:
        if gate_fusion == "g1u0":
            if input_dtype == "__half":
                return (
                    "fmoe_kernel_func",
                    "fmoe_f16.co",
                    "uint16_t",
                    "uint16_t",
                    "false",
                )
            elif input_dtype == "__hip_bfloat16":
                return (
                    "fmoe_kernel_func",
                    "fmoe_b16.co",
                    "uint16_t",
                    "uint16_t",
                    "false",
                )
            elif input_dtype == "uint8_t":
                if activation == "gelu":
                    return (
                        f"fmoe_int8_g1u0_subGU_{selected_tile}_gelu",
                        f"moe/gelu/fmoe_int8_g1u0_subGU_{selected_tile}_gelu.co",
                        "uint8_t",
                        "uint16_t",
                        "false",
                    )
                elif activation == "silu":
                    return (
                        f"fmoe_int8_g1u0_subGU_{selected_tile}",
                        f"fmoe/silu/fmoe_int8_g1u0_subGU_{selected_tile}.co",
                        "uint8_t",
                        "uint16_t",
                        "false",
                    )

        elif gate_fusion == "g1u1":
            if gate_dtype in ["uint32_t", "int32_t"]:
                return (
                    f"fmoe_int4fp8_g1u1_subGU_{selected_tile}",
                    f"fmoe_int4fp8_g1u1_subGU_{selected_tile}_gelu.co",
                    "uint8_t",
                    "uint16_t",
                    "false",
                )
            elif input_dtype == "uint8_t":
                if is_smooth_scale:
                    return (
                        f"fmoe_int8_g1u1_multix_subGU_{selected_tile}",
                        f"fmoe_int8_g1u1_multix_subGU_{selected_tile}.co",
                        "uint8_t",
                        "uint16_t",
                        "false",
                    )
                elif activation == "gelu":
                    return (
                        f"fmoe_int8_g1u1_subGU_{selected_tile}_gelu",
                        f"fmoe/gelu/fmoe_int8_g1u1_subGU_{selected_tile}_gelu.co",
                        "uint8_t",
                        "uint16_t",
                        "false",
                    )
                elif activation == "silu":
                    return (
                        f"fmoe_int8_g1u1_subGU_{selected_tile}",
                        f"fmoe/silu/fmoe_int8_g1u1_subGU_{selected_tile}.co",
                        "uint8_t",
                        "uint16_t",
                        "false",
                    )
            elif input_dtype == "__hip_fp8_e4m3_fnuz":
                if is_smooth_scale:
                    return (
                        f"fmoe_fp8_g1u1_multix_subGU_{selected_tile}",
                        f"fmoe_fp8_g1u1_multix_subGU_{selected_tile}.co",
                        "uint8_t",
                        "uint16_t",
                        "false",
                    )
                elif activation == "gelu":
                    return (
                        f"fmoe_fp8_g1u1_subGU_{selected_tile}_gelu",
                        f"fmoe/gelu/fmoe_fp8_g1u1_subGU_{selected_tile}_gelu.co",
                        "uint8_t",
                        "uint16_t",
                        "false",
                    )
                elif activation == "silu":
                    return (
                        f"fmoe_fp8_g1u1_subGU_{selected_tile}",
                        f"fmoe/silu/fmoe_fp8_g1u1_subGU_{selected_tile}.co",
                        "uint8_t",
                        "uint16_t",
                        "false",
                    )
            elif input_dtype == "__hip_bfloat16":
                if enable_vskip:
                    return (
                        f"fmoe_fp8_blockscale_g1u1_subGU_{selected_tile}",
                        f"fmoe_fp8_blockscale_g1u1_subGU_{selected_tile}.co",
                        "uint8_t",
                        "uint16_t",
                        "false",
                    )
                else:
                    return (
                        f"fmoe_fp8_blockscale_g1u1_novs_subGU_{selected_tile}",
                        f"fmoe_fp8_blockscale_g1u1_novs_subGU_{selected_tile}.co",
                        "uint8_t",
                        "uint16_t",
                        "false",
                    )

    raise ValueError("Unsupported condition")


def compile(
    input_dtype: str,
    gate_fusion: str,
    gate_dtype: str,
    activation: str,
    selected_tile: int,
    is_smooth_scale=False,
    a16=False,
    enable_vskip=False,
    block_size=32,
    func_name=None,
    folder=None,
):
    if func_name is None:
        func_name = get_default_func_name(
            MD_NAME,
            (
                input_dtype,
                gate_fusion,
                gate_dtype,
                activation,
                selected_tile,
                is_smooth_scale,
                a16,
                enable_vskip,
                block_size,
            ),
        )

    if folder is None:
        folder = func_name

    if not_built(folder):
        kernel_name, co_name, input_dtype, output_dtype, switch_gxy = select_hsaco(
            input_dtype,
            gate_fusion,
            gate_dtype,
            activation,
            selected_tile,
            is_smooth_scale,
            a16,
            enable_vskip,
        )
        bin_size, bin_data = transfer_hsaco(f"{AITER_CORE_DIR}/hsa/{co_name}")
        return compile_template_op(
            src_template,
            MD_NAME,
            [
                f"{AITER_CORE_DIR}/csrc/cpp_itfs/utils.h",
                f"{AITER_CORE_DIR}/csrc/include",
                f"{CK_DIR}/include",
                f"{CK_DIR}/example/ck_tile/13_moe_sorting/moe_sorting_api.hpp",
            ],
            [f"{CK_DIR}/example/ck_tile/13_moe_sorting/moe_sorting_api.cpp"],
            bin_size=bin_size,
            bin_data=bin_data,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            kernel_name=kernel_name,
            selected_tile=selected_tile,
            switch_gxy=switch_gxy,
            block_size=block_size,
            func_name=func_name,
            folder=folder,
        )
    else:
        return run_lib(func_name, folder)


def asm_moe(
    hidden_states,  # [num_tokens, dim] M,K
    w1,  # [expert(local_expert:EP), inter_dim*2, dim] N,K
    w2,  # [expert(local_expert:EP), dim, inter_dim]
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert(local_expert:EP), inter_dim, 1]
    fc2_scale=None,  # [expert(local_expert:EP), model_dim, 1]
    fc1_smooth_scale=None,  # [expert(local_expert:EP), 1, model_dim]
    fc2_smooth_scale=None,  # [expert(local_expert:EP), 1, inter_dim]
    a16=False,
    per_tensor_quant_scale=None,
    expert_mask=None,
    activation="silu",
    output=None,  # [num_tokens, dim]
    sorted_token_ids=None,
    sorted_weights=None,
    sorted_expert_ids=None,
    num_valid_ids=None,
    a8=None,
    a8_scale=None,
    workspace=None,
):
    import torch
    from csrc.cpp_itfs.torch_utils import torch_to_hip_types, torch_to_c_types

    E, dim, inter_dim = w2.shape
    global_E = E
    if expert_mask is not None:
        global_E = expert_mask.numel()
    num_tokens, topk = topk_ids.shape
    device = hidden_states.device
    lastdim_mul = 8 if w1.dtype in {torch.int32, torch.uint32} else 1
    if fc1_smooth_scale is None:
        gate_fusion = "g1u0"
    elif a16:
        if (
            w1.dtype in [torch.float8_e4m3_fnuz, torch.int8]
            and inter_dim * 2 == w1.shape[1]
        ):
            gate_fusion = "g1u1"
        elif w1.dtype == torch.int8 and inter_dim == w1.shape[1]:
            gate_fusion = "g1u0"
        else:
            raise ValueError(
                f"Unsupported w1.dtype {w1.dtype}, which should be {torch.float8_e4m3_fnuz} or {torch.int8} and w1.shape[1] should be {inter_dim} or {inter_dim*2}"
            )
    else:
        if fc1_smooth_scale is not None:
            if expert_mask is not None:
                local_expert_hash = expert_mask.cumsum(0, dtype=torch.int32)
                local_expert_hash[local_expert_hash > 0] -= 1

        if inter_dim * lastdim_mul == w1.shape[1]:
            gate_fusion = "g1u0"
        elif inter_dim * 2 * lastdim_mul == w1.shape[1]:
            gate_fusion = "g1u1"
        else:
            raise ValueError(
                f"Invalid MoE weight: {w1.shape=} {w2.shape=} {lastdim_mul}"
            )

    input_dtype, gate_dtype = torch_to_hip_types(hidden_states.dtype, w1.dtype)

    block_size = 32
    if sorted_token_ids is None:
        max_num_tokens_padded = topk_ids.numel() + global_E * block_size - topk
        sorted_token_ids = torch.empty(
            (max_num_tokens_padded,), dtype=torch.int32, device=device
        )
    else:
        max_num_tokens_padded = sorted_token_ids.shape[0]

    if sorted_weights is None:
        sorted_weights = torch.empty(
            (max_num_tokens_padded,), dtype=torch.float, device=device
        )
    if sorted_expert_ids is None:
        max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
        sorted_expert_ids = torch.empty(
            (max_num_m_blocks,), dtype=torch.int32, device=device
        )
    else:
        max_num_m_blocks = sorted_expert_ids.shape[0]

    if num_valid_ids is None:
        num_valid_ids = torch.empty((1,), dtype=torch.int32, device=device)

    if output is None:
        output = torch.empty(
            (num_tokens, dim), dtype=hidden_states.dtype, device=device
        )

    num_cu = torch.cuda.get_device_properties().multi_processor_count
    selected_tile = select_tile(
        input_dtype, gate_fusion, gate_dtype, inter_dim, max_num_m_blocks, num_cu, a16
    )
    func = compile(
        input_dtype,
        gate_fusion,
        gate_dtype,
        activation,
        selected_tile,
        bool(fc2_smooth_scale),
        a16,
        block_size=block_size,
    )
    workspace_size_func = run_lib("moe_sorting_get_workspace_size", func.__name__)
    workspace_size = ctypes.c_int(0)
    workspace_size_func(
        *torch_to_c_types(num_tokens, global_E), ctypes.pointer(workspace_size)
    )
    workspace_size = workspace_size.value
    if workspace_size > 0 and workspace is None:
        workspace = torch.zeros(workspace_size, dtype=topk_ids.dtype, device=device)
    if fc2_smooth_scale is None:
        func(
            *torch_to_c_types(
                output,
                hidden_states,
                w1,
                w2,
                topk_weight,
                topk_ids,
                sorted_token_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                num_tokens,
                dim,
                inter_dim,
                topk,
                global_E,
                max_num_m_blocks,
                output.nbytes,
                hidden_states.stride(0),
                expert_mask,
                workspace,
                torch.cuda.current_stream(),
            )
        )
    else:
        func(
            *torch_to_c_types(
                output,
                hidden_states,
                w1,
                w2,
                topk_weight,
                topk_ids,
                fc1_scale,
                fc2_scale,
                fc1_smooth_scale,
                fc2_smooth_scale,
                a16,
                per_tensor_quant_scale,
                expert_mask,
                activation,
            )
        )
    return output


if __name__ == "__main__":
    pass
