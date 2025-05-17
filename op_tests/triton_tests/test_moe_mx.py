import pytest
import torch
import triton

from aiter.ops.triton.moe_op_mxfp4 import fused_moe_mxfp4
from op_tests.op_benchmarks.triton.utils.common import (
    str_to_torch_dtype,
    torch_to_tl_dtype,
)
from op_tests.op_benchmarks.triton.utils.moe import generate_moe_alignment

from .utils.fused_moe_ref import torch_moe
from .utils.quant_ref import torch_dynamic_mxfp4_quant

DEBUG_MODE = False


def alloc_rand(shape, device, dtype, requires_grad=True):
    if dtype.itemsize == 1:
        tmp = 2 ** -(torch.randint(4, 8, shape, device=device, dtype=torch.float16))
        return tmp.to(dtype).requires_grad_(requires_grad)
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)


@pytest.mark.parametrize(
    "M, N, K, E, top_k",
    [
        # # fp8 x mxfp4
        (16, 256, 256, 128, 4),
        (1000, 704, 800, 3, 1),
        (1000, 704, 800, 8, 2),
        (64, 14336, 4096, 8, 2),
        (16, 14336, 128, 8, 2),  # not working either
        (16, 14336, 4096, 4, 1),
        (1, 14336, 128, 4, 2),
        (3, 14336, 128, 4, 2),
        (16, 14336, 128, 1, 1),
        (64, 7186, 128, 8, 2),
        (64, 3584, 128, 8, 2),
        (64, 1792, 128, 8, 2),
        (64, 64, 128, 8, 2),
        (1, 1024, 16384, 2, 1),
    ],
)
@pytest.mark.parametrize(
    "a_dtype_str, b_dtype_str",
    [
        # Hardware native OCP
        ("fp8_e5m2", "mxfp4_e2m1"),
        ("mxfp4_e2m1", "mxfp4_e2m1"),
        # Software emulation that upcasts mxfp4 to fp16
        ("fp16", "mxfp4_e2m1"),
        ("bf16", "mxfp4_e2m1"),
    ],
)
@pytest.mark.parametrize("routed_weight", [False, True])
@pytest.mark.parametrize("swizzle_mx_scale", [False, True])
def test_fused_moe(
    M: int,
    N: int,
    K: int,
    top_k: int,
    E: int,
    a_dtype_str: str,
    b_dtype_str: str,
    routed_weight: bool,
    swizzle_mx_scale: bool,
):
    global _SKIP
    if triton.runtime.driver.active.get_current_target().arch not in ("gfx950"):
        pytest.skip("MXFP4 not supported on this architecture")

    is_a_mixed_input = a_dtype_str.startswith("mx")
    is_b_mixed_input = b_dtype_str.startswith("mx")
    a_dtype = str_to_torch_dtype[a_dtype_str]
    c_dtype = torch.bfloat16 if is_a_mixed_input else a_dtype
    fp16_dtype = torch.float16 if a_dtype_str == "fp16" else torch.bfloat16
    a_tri = alloc_rand((M, K), dtype=fp16_dtype, device="cuda", requires_grad=False)
    b_tri = alloc_rand((E, N, K), dtype=fp16_dtype, device="cuda", requires_grad=False)
    c_tri = torch.zeros(
        (M, top_k, N), dtype=c_dtype, device="cuda", requires_grad=False
    )
    a_scale = torch.tensor([1.00], dtype=torch.float32, device="cuda")
    b_scale = torch.tensor([1.00] * E, dtype=torch.float32, device="cuda")
    # Reference inputs
    a_ref, b_ref, c_ref = a_tri.clone(), b_tri.clone(), c_tri.clone()
    # Try fixed config for now
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }
    # Simulated moe_align_block_size()
    topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded = (
        generate_moe_alignment(M, E, top_k, config["BLOCK_SIZE_M"])
    )
    # Downcast a tensor to mxfp4 and upcast back for reference
    if is_a_mixed_input:
        # swizzle_axis = 0 if swizzle_mx_scale else None  # TODO Add Swizzle support
        a_tri, a_mx_scales, _ = torch_dynamic_mxfp4_quant(a_tri)

        # TODO Add Upcast support
        a_ref = a_tri
        # a_ref = torch_upcast_from_mxfp(
        #    a_tri, a_mx_scales, fp16_dtype, axis=1, swizzle_axis=swizzle_axisv
        # )
    else:
        a_ref = a_ref.to(fp16_dtype)
        a_mx_scales = None
    # Downcast b tensor to mxfp4 and upcast back for reference
    if is_b_mixed_input:
        # swizzle_axis = 1 if swizzle_mx_scale else None  # TODO Add Swizzle support
        b_tri, b_mx_scales = torch_dynamic_mxfp4_quant(b_tri)

        # TODO Add Upcast support
        b_ref = b_tri
        # b_ref = torch_upcast_from_mxfp(
        #    b_tri, b_mx_scales, fp16_dtype, axis=2, swizzle_axis=swizzle_axis
        # )
    # Triton
    fused_moe_mxfp4(
        a_tri,
        b_tri,
        c_tri,
        a_scale,
        b_scale,
        a_mx_scales,
        b_mx_scales,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        routed_weight,
        top_k,
        swizzle_mx_scale,
        swizzle_mx_scale,
        config,
        torch_to_tl_dtype[c_tri.dtype],
    )
    # Torch
    b_zp = None
    group_size = 0
    # a_scale and b_scale not used actually
    c_ref = torch_moe(
        a_ref,
        b_ref,
        c_ref,
        a_scale,
        b_scale,
        b_zp,
        group_size,
        topk_ids,
        topk_weights,
        routed_weight,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        dtype=torch.float16,
        fp8_w8a8=False,
        int8_w8a16=False,
        int4_w4a16=False,
        gelu=False,
    )

    torch.testing.assert_close(c_tri.to(fp16_dtype), c_ref.to(fp16_dtype))
