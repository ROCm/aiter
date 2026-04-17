import torch
from aiter import ActivationType
from aiter.fused_moe_smoothquant import fused_moe_gelu_sqi8
from aiter.fused_moe_bf16_asm import torch_moe
from aiter.test_common import (
    checkAllclose,
    run_perftest,
)
from aiter import get_gfx


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def smooth_quant_w(
    weight,  # [num_experts, OC, IC]
    smooth_scale,  # [num_experts, IC]
):
    scaled_weight = weight / smooth_scale
    per_oc_scale = (
        scaled_weight.abs().max(dim=2, keepdim=True)[0]
    ) / 128  # [num_experts, OC, 1]
    quanted_weight = (
        (scaled_weight / per_oc_scale).round().clamp(-128, 127).to(torch.int8)
    )
    return quanted_weight, per_oc_scale  # [num_experts, OC, IC], [num_experts, OC, 1]


def test_fmoe_sqi8(
    num_tokens,
    model_dim,
    inter_dim,
    num_experts,
    topk,
    use_smoothquant,
    shared_smooth_up,
):
    device = "cuda"

    if get_gfx() != "gfx950":
        print("skip tests for unsupported platform")
        return

    x0 = torch.randn(num_tokens, model_dim, dtype=torch.bfloat16, device=device) * 0.001

    if use_smoothquant:
        # shared_smoothquant_up : share smooth scales between all experts
        fc1_smooth_scale = 1.0 / torch.randint(
            low=1,
            high=5,
            size=[1 if shared_smooth_up else num_experts, 1, model_dim],
            dtype=torch.float32,
            device=device,
        )
        fc2_smooth_scale = 1.0 / torch.randint(
            low=1,
            high=5,
            size=[num_experts, 1, inter_dim],
            dtype=torch.float32,
            device=device,
        )
    else:
        # just to get correct references
        fc1_smooth_scale = torch.ones(
            [num_experts, 1, model_dim], dtype=torch.float32, device=device
        )
        fc2_smooth_scale = torch.ones(
            [num_experts, 1, inter_dim], dtype=torch.float32, device=device
        )

    w1_f32 = torch.randn(num_experts, inter_dim, model_dim, dtype=torch.float32) * 0.001
    w2_f32 = torch.randn(num_experts, model_dim, inter_dim, dtype=torch.float32) * 0.001

    w1, fc1_scale = smooth_quant_w(w1_f32, fc1_smooth_scale)
    w2, fc2_scale = smooth_quant_w(w2_f32, fc2_smooth_scale)

    router_weights = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    ret_topk = torch.topk(router_weights, topk)
    x1 = ret_topk.values.to(torch.float32)
    x2 = ret_topk.indices.to(torch.int32)

    ref0 = torch_moe(
        x0,
        w1_f32,
        w2_f32,
        x1,
        x2,
        None,
        None,
        None,
        None,
        activation=ActivationType.Gelu,
    )
    ref1 = torch_moe(
        x0,
        w1,
        w2,
        x1,
        x2,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale.expand(num_experts, -1, -1),
        fc2_smooth_scale.expand(num_experts, -1, -1),
        activation=ActivationType.Gelu,
    )

    ret, dt = run_perftest(
        fused_moe_gelu_sqi8,
        x0,
        w1,
        w2,
        x1,
        x2,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale if use_smoothquant else None,
        fc2_smooth_scale if use_smoothquant else None,
    )

    err0 = checkAllclose(ref0, ret, rtol=1e-3, atol=1e-3, msg="check with ref0")
    err1 = checkAllclose(ref1, ret, rtol=1e-3, atol=1e-3, msg="check with ref1")

    logits_diff0 = calc_diff(ref0, ret)
    logits_diff1 = calc_diff(ref1, ret)
    print(
        f"{num_tokens=} {model_dim=} {inter_dim=} {num_experts=} {topk=} {use_smoothquant=} {shared_smooth_up=} {logits_diff0=:.6f}, {logits_diff1=:.6f}, {err0=:.6f}, {err1=:.6f}, {dt:.0f} us"
    )
    assert logits_diff0 < 0.01
    assert logits_diff1 < 0.01

    if use_smoothquant:
        smooth_info = "shared-up" if shared_smooth_up else "per-expert"
    else:
        smooth_info = "unused"
    ret = {
        "num_tokens":num_tokens,
        "model_dim":model_dim,
        "inter_dim":inter_dim,
        "num_experts":num_experts,
        "topk":topk,
        "smooth": smooth_info,
        "diff": logits_diff1,
        "time(us)":f"{dt:.0f}"
    }
    return ret

if __name__ == "__main__":
    torch.set_default_device("cuda")
    torch.set_printoptions(
        linewidth=3000,
        sci_mode=False,
        edgeitems=8,
    )
    # torch.cuda.set_device(0)
    torch.manual_seed(0)

    summary = []
    for num_experts, topk in [(800, 25), (400, 20)]:
        for num_tokens in [40960, 20480, 19147]:
            for use_smoothquant, shared_smooth_up in [(1, 0), (1, 1), (0, 0)]:
                ret = test_fmoe_sqi8(
                    num_tokens=num_tokens,
                    model_dim=4096,
                    inter_dim=1536,
                    num_experts=num_experts,
                    topk=topk,
                    use_smoothquant=use_smoothquant,
                    shared_smooth_up=shared_smooth_up,
                )
                summary.append(ret)
    
    # show summary in table
    try:
        import pandas as pd
        print(pd.DataFrame(summary).to_markdown(index=False))
    except Exception:
        pass