# SPDX-License-Identifier: MIT
"""Single-GPU AITER expert handoff with synthetic MXFP4 weights."""

import torch
from aiter.fused_moe import moe_sorting

from minimax_m3_gluon.front import front
from minimax_m3_gluon.host import allocate_workspace, expert_weights, run_prepared_experts


def main():
    torch.cuda.set_device(0)
    device = "cuda"
    torch.manual_seed(413)
    # Physical byte views match the preshuffled A16W4 M3 layout contract.
    w13 = torch.randint(0, 256, (129, 1536, 3072), device=device, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (129, 6144, 384), device=device, dtype=torch.uint8)
    s13 = torch.full((129, 1536, 192), 127, device=device, dtype=torch.uint8)
    s2 = torch.full((129, 6144, 24), 127, device=device, dtype=torch.uint8)
    weights = expert_weights(w13, s13, w2, s2)
    for batch in (1, 4, 8, 16):
        attn = torch.randn((batch, 6144), device=device, dtype=torch.bfloat16)
        residual = torch.randn_like(attn)
        gain = torch.zeros((6144,), device=device, dtype=torch.bfloat16)
        gate = torch.randn((128, 6144), device=device, dtype=torch.bfloat16) * 0.01
        bias = torch.randn((128,), device=device, dtype=torch.float32) * 0.05
        state = allocate_workspace(batch, device)
        front[(256,)](
            attn, residual, gain, gate, bias,
            state["HMid"], state["XNorm"], state["Scores"], state["Sigmoids"],
            state["RouteIDs"], state["RouteWeights"], state["SortIDs"],
            state["SortWeights"], state["SortExperts"], state["SortCounts"],
            state["SortM"], state["SortReverse"], state["SortOut"],
            state["Ready"], state["FrontEpochs"], state["FrontErrors"],
            B=batch, SHARED_WEIGHT=1.0, num_warps=8, enable_fp_fusion=False,
        )
        result = run_prepared_experts(state["XNorm"], weights, state)
        reference = dict(state)
        (
            reference["SortIDs"], reference["SortWeights"],
            reference["SortExperts"], reference["SortCounts"], reference["SortOut"],
        ) = moe_sorting(
            state["RouteIDs"], state["RouteWeights"], 129, 6144,
            torch.bfloat16, block_size=16, accumulate=True,
        )
        expected = run_prepared_experts(state["XNorm"], weights, reference)
        torch.cuda.synchronize()
        assert torch.isfinite(result).all().item()
        error = (result.float() - expected.float()).abs().max().item()
        assert torch.allclose(result.float(), expected.float(), atol=128, rtol=0.06), error
        print(f"B={batch}: AITER expert handoff vs reference sorting OK; max error={error:.3f}", flush=True)


if __name__ == "__main__":
    main()
