
import torch
import torch.nn.functional as F
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
    # torch_fp4_moe_2stage_with_smooth,
)
from aiter.fused_moe_bf16_asm import asm_moe
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.quant import get_torch_quant
from aiter.ops.shuffle import shuffle_weight, shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility.fp4_utils import (
    e8m0_shuffle,
)

from aiter.jit.utils.chip_info import get_gfx
# from bytenn_amd_ops.ops.triton.per_token_quant import pertoken_quant
from aiter import pertoken_quant

def _check_result(ref_out, test_out, label, atol=1.0, rtol=0.05, pass_pct=95.0):
    """Compare outputs and print result. Returns (passed, max_delta, pct_close)."""
    print(f"{label}: ref_out: {ref_out.shape}, test_out: {test_out.shape}")
    max_delta = (ref_out.float() - test_out.float()).abs().max().item()
    close_mask = torch.isclose(ref_out.float(), test_out.float(), atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100
    passed = pct_close > pass_pct

    print(
        f"  max_delta={max_delta:.4f}, {pct_close:.1f}% close (atol={atol}, rtol={rtol})"
    )
    print(f"  ref  sample: {ref_out.reshape(-1)[:8]}")
    print(f"  test sample: {test_out.reshape(-1)[:8]}")
    print(f"  --> {'PASS' if passed else 'FAIL'}")
    return passed, max_delta, pct_close


def timer(func):
    def func_wrapper(*args, **kwargs):
        import time, torch
        torch.cuda.synchronize()
        execution_start_time = time.perf_counter()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        execution_time = time.perf_counter() - execution_start_time
        print("========== {} {:.3f} ms ==========".format(func.__name__, execution_time * 1000))
        return result
    return func_wrapper


class Act(torch.nn.Module):
    def __init__(self, activation: str = "gelu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = activation

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        # 没有 up_proj (is_gated=False)，直接应用激活函数，不需要切分维度
        act_fn = F.silu if self.activation == "silu" else F.gelu
        return act_fn(x)


def native_w8a8_per_token_matmul(A, B, As=None, Bs=None, output_dtype=torch.float16):
    """Matrix multiplication function that supports per-token input quantization and per-column weight quantization."""
    A_calc = A.to(torch.float32)
    B_calc = B.to(torch.float32)

    assert A_calc.shape[-1] == B_calc.shape[-1], "Dimension mismatch"
    assert B_calc.ndim == 2 and B_calc.is_contiguous(), "B must be a 2D contiguous tensor"

    # Reshape input
    M = A_calc.numel() // A_calc.shape[-1]
    B_calc = B_calc.t()  # Transpose weight matrix
    N, K = B_calc.shape
    origin_C_shape = A_calc.shape[:-1] + (K,)
    A_calc = A_calc.reshape(M, N)

    # Base matmul
    C = torch.matmul(A_calc, B_calc)  # [M, K]
    
    # 兼容传入 None 的情况，进行纯浮点数计算
    if As is not None and Bs is not None:
        C = As * C * Bs.view(1, -1)  # Broadcast per-column scale

    return C.reshape(origin_C_shape).to(output_dtype)


def torch_w8a8_per_column_moe(a, w1, w2, w1_s=None, w2_s=None, topk_weight=None, topk_ids=None, activation="gelu", use_int8=True):
    """This function performs fused moe with per-column int8 quantization or pure bf16 using native torch."""
    B, D = a.shape
    topk = topk_weight.shape[-1]
    
    if use_int8:
        # Perform per-token quantization
        a_q, a_s = pertoken_quant(a, quant_dtype=dtypes.i8)
        a_s = a_s.view(B, -1, 1).repeat(1, topk, 1).reshape(-1, 1)  # [B*topk, 1]
    else:
        # 纯 bf16 模式：直接沿用输入激活值
        a_q = a
    
    # Repeat tokens to match topk
    a_q = a_q.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)

    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    # 直接展开外部传入的权重和索引，不再内部计算
    topk_weight_flat = topk_weight.view(-1)
    topk_ids_flat = topk_ids.view(-1)
    
    # Process each expert
    for i in range(w1.shape[0]):
        mask = topk_ids_flat == i
        if mask.sum():
            if use_int8:
                # ====== INT8 模式 ======
                # First MLP layer
                inter_out = native_w8a8_per_token_matmul(
                    a_q[mask], w1[i], a_s[mask], w1_s[i].view(-1), output_dtype=a.dtype
                )
                # Activation function
                act_out = Act(activation).forward_native(inter_out)
                # Quantize activation output with per-token
                act_out_q, act_out_s = pertoken_quant(act_out, quant_dtype=dtypes.i8)

                # Second MLP layer
                out[mask] = native_w8a8_per_token_matmul(
                    act_out_q, w2[i], act_out_s, w2_s[i].view(-1), output_dtype=a.dtype
                )
            else:
                # ====== 纯 BF16 模式（带权重反量化） ======
                # 1. 对 w1 进行反量化
                w1_dq = w1[i].to(a.dtype) * w1_s[i].view(-1, 1).to(a.dtype)
                inter_out = F.linear(a_q[mask], w1_dq)
                
                # Activation function
                act_out = Act(activation).forward_native(inter_out)
                
                # 2. 对 w2 进行反量化
                w2_dq = w2[i].to(a.dtype) * w2_s[i].view(-1, 1).to(a.dtype)
                out[mask] = F.linear(act_out, w2_dq)
            
    # Apply routing weights and sum
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight_flat.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


device = "cuda"
seq_len = 20480
num_experts = 400
model_dim = 4096
inter_dim = 1536
topk = 20
dtype = torch.bfloat16

torch.manual_seed(0)

x0 = torch.randn(seq_len, model_dim, dtype=dtype, device=device)

w1_bf16 = torch.randn(num_experts, inter_dim, model_dim, dtype=dtype, device=device) / 10
w2_bf16 = torch.randn(num_experts, model_dim, inter_dim, dtype=dtype, device=device) / 10

fp4_quant = get_torch_quant(QuantType.per_1x32)
w1_fp4, w1_scale = fp4_quant(w1_bf16)
w1_fp4 = w1_fp4.view(num_experts, inter_dim, model_dim // 2)
w2_fp4, w2_scale = fp4_quant(w2_bf16)
w2_fp4 = w2_fp4.view(num_experts, model_dim, inter_dim // 2)

# shuffle weights for w4a4
w1_qt_shuf = shuffle_weight(w1_fp4, (16, 16))
w2_qt_shuf = shuffle_weight_a16w4(w2_fp4, 16, False)
w1_scale_shuf = e8m0_shuffle(w1_scale)
w2_scale_shuf = shuffle_scale_a16w4(w2_scale, num_experts, False)


w1_i8, fc1_scale = pertoken_quant(w1_bf16.contiguous(), quant_dtype=dtypes.i8)
w2_i8, fc2_scale = pertoken_quant(w2_bf16.contiguous(), quant_dtype=dtypes.i8)
w1_i8_shuffle = shuffle_weight(w1_i8)
w2_i8_shuffle = shuffle_weight(w2_i8)
fc1_smooth_scale = torch.ones([num_experts, model_dim], dtype=torch.float32, device=device)
fc2_smooth_scale = torch.ones([num_experts, inter_dim], dtype=torch.float32, device=device)

score = torch.randn(seq_len, num_experts, dtype=dtype, device=device)
topk_weight, topk_ids = fused_topk(x0, score, topk, True)
topk_weight = topk_weight.to(torch.float32)
topk_ids = topk_ids.to(torch.int32)

def run_fp4_moe():
    return fused_moe(
        x0,
        w1_qt_shuf, #w1_fp4,
        w2_qt_shuf, #w2_fp4,
        topk_weight,
        topk_ids,
        None,
        ActivationType.Gelu,
        QuantType.per_1x32,
        False,
        w1_scale_shuf, #w1_scale,
        w2_scale_shuf, #w2_scale,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        0,
        dtype,
        0,
        0,
        None,
        None,
        0,
    )


def run_torch_fp4_2stage():
    # Stage1 input quant (fp4 + e8m0 scale, per_1x32)
    a1_qt, a1_scale = fp4_quant(x0, quant_dtype=dtypes.fp4x2)

    # Stage1 torch reference (g1u0 because w1 shape is [E, inter_dim, model_dim//2])
    stage1_out = torch_moe_stage1(
        a1_qt,
        w1_fp4,
        w2_fp4,
        topk_weight,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Gelu,
        quant_type=QuantType.per_1x32,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        doweight=False,
    )

    # Quantize stage1 output for stage2
    a2_qt, a2_scale = fp4_quant(
        stage1_out.view(stage1_out.shape[0] * stage1_out.shape[1], -1),
        quant_dtype=dtypes.fp4x2,
    )
    a2_qt = a2_qt.view(seq_len, topk, -1)

    # Stage2 torch reference
    return torch_moe_stage2(
        a2_qt,
        w1_fp4,
        w2_fp4,
        topk_weight,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        doweight=True,
    )


# def run_torch_fp4_2stage_with_smooth():
#     # Uses non-shuffled fp4 weights/scales for torch reference path.
#     return torch_fp4_moe_2stage_with_smooth(
#         x0,
#         w1_fp4,
#         w2_fp4,
#         topk_weight,
#         topk_ids,
#         dtype=dtype,
#         activation=ActivationType.Gelu,
#         quant_type=QuantType.per_1x32,
#         w1_scale=w1_scale,
#         w2_scale=w2_scale,
#         fc1_smooth_scale=fc1_smooth_scale,
#         fc2_smooth_scale=fc2_smooth_scale,
#         doweight_stage1=False,
#     )


def run_torch_int8_2stage():
    # Stage1 input quant (int8 + per-token scale)
    a1_qt, a1_scale = pertoken_quant(x0, quant_dtype=dtypes.i8)

    # Stage1 torch reference (g1u0: w1 is [E, inter_dim, model_dim])
    stage1_out = torch_moe_stage1(
        a1_qt,
        w1_i8,
        w2_i8,
        topk_weight,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Gelu,
        quant_type=QuantType.per_Token,
        a1_scale=a1_scale,
        w1_scale=fc1_scale,
        doweight=False,
    )

    # Quantize stage1 output for stage2 input
    a2_qt, a2_scale = pertoken_quant(
        stage1_out.view(stage1_out.shape[0] * stage1_out.shape[1], -1),
        quant_dtype=dtypes.i8,
    )
    a2_qt = a2_qt.view(seq_len, topk, -1)
    a2_scale = a2_scale.view(seq_len, topk)

    # Stage2 torch reference
    return torch_moe_stage2(
        a2_qt,
        w1_i8,
        w2_i8,
        topk_weight,
        topk_ids,
        dtype=dtype,
        quant_type=QuantType.per_Token,
        w2_scale=fc2_scale,
        a2_scale=a2_scale,
        doweight=True,
    )


def run_asm_moe():
    return asm_moe(
        x0,
        w1_i8_shuffle,
        w2_i8_shuffle,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
        get_gfx() != "gfx942",
        None,
        None,
        None,
        ActivationType.Gelu,
    )


torch_out = torch_w8a8_per_column_moe(x0, w1_i8, w2_i8, fc1_scale, fc2_scale, topk_weight, topk_ids, activation="gelu", use_int8=True)
fp4_out = run_fp4_moe()
torch_fp4_2stage_out = run_torch_fp4_2stage()
# torch_fp4_2stage_smooth_out = run_torch_fp4_2stage_with_smooth()
torch_int8_2stage_out = run_torch_int8_2stage()
asm_out = run_asm_moe()

fp4_cos_sim = F.cosine_similarity(torch_out.reshape(1, -1).float(), fp4_out.reshape(1, -1).float()).item()
asm_cos_sim = F.cosine_similarity(torch_out.reshape(1, -1).float(), asm_out.reshape(1, -1).float()).item()
fp4_torch_cosine_sim = F.cosine_similarity(fp4_out.reshape(1, -1).float(), torch_fp4_2stage_out.reshape(1, -1).float()).item()
print(f"fp4_torch_cosine_sim: {fp4_torch_cosine_sim}")
# fp4_torch_smooth_cosine_sim = F.cosine_similarity(
#     fp4_out.reshape(1, -1).float(), torch_fp4_2stage_smooth_out.reshape(1, -1).float()
# ).item()
# print(f"fp4_torch_smooth_cosine_sim: {fp4_torch_smooth_cosine_sim}")
fp4_torch2stage_cos_sim = F.cosine_similarity(
    torch_fp4_2stage_out.reshape(1, -1).float(), fp4_out.reshape(1, -1).float()
).item()
int8_torch2stage_cos_sim = F.cosine_similarity(
    torch_int8_2stage_out.reshape(1, -1).float(), torch_out.reshape(1, -1).float()
).item()
print(
    f"fp4_cos_sim: {fp4_cos_sim}, asm_cos_sim: {asm_cos_sim}, "
    f"fp4_torch2stage_cos_sim: {fp4_torch2stage_cos_sim}, "
    f"int8_torch2stage_cos_sim: {int8_torch2stage_cos_sim}"
)
_check_result(torch_out, fp4_out, "fp4_out")
_check_result(torch_out, asm_out, "asm_out")
_check_result(torch_fp4_2stage_out, fp4_out, "fp4_vs_torch_2stage")
# _check_result(torch_fp4_2stage_smooth_out, fp4_out, "fp4_vs_torch_2stage_with_smooth")
# _check_result(
#     torch_fp4_2stage_out,
#     torch_fp4_2stage_smooth_out,
#     "torch_2stage_vs_torch_2stage_with_smooth",
# )
# _check_result(torch_out, torch_fp4_2stage_out, "torch_out_vs_fp4_torch_2stage")
# _check_result(
#     torch_out, torch_fp4_2stage_smooth_out, "torch_out_vs_fp4_torch_2stage_with_smooth"
# )
_check_result(torch_out, torch_int8_2stage_out, "torch_out_vs_int8_torch_2stage")
exit()

@timer
def func1():
    return run_fp4_moe()
for _ in range(10):
    func1()


@timer
def func2():
    return run_asm_moe()
for _ in range(10):
    func2()
