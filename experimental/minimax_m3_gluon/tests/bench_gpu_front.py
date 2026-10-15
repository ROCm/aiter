# SPDX-License-Identifier: MIT
"""Graph latency comparison for the fused M3 front and an unfused proxy."""

import torch
import triton.testing
from aiter.fused_moe import moe_sorting

from minimax_m3_gluon.front import front
from minimax_m3_gluon.host import allocate_workspace


def main():
    torch.cuda.set_device(0)
    for batch in (1, 2, 4, 8, 16):
        torch.manual_seed(500 + batch)
        attn = torch.randn((batch, 6144), device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(attn)
        gain = torch.randn(6144, device="cuda", dtype=torch.bfloat16) * 0.05
        gate = torch.randn((128, 6144), device="cuda", dtype=torch.bfloat16) * 0.01
        bias = torch.randn(128, device="cuda", dtype=torch.float32) * 0.05
        state = allocate_workspace(batch, "cuda")

        def fused():
            front[(256,)](
                attn, residual, gain, gate, bias,
                state["HMid"], state["XNorm"], state["Scores"], state["Sigmoids"],
                state["RouteIDs"], state["RouteWeights"], state["SortIDs"],
                state["SortWeights"], state["SortExperts"], state["SortCounts"],
                state["SortM"], state["SortReverse"], state["SortOut"],
                state["Ready"], state["FrontEpochs"], state["FrontErrors"],
                B=batch, SHARED_WEIGHT=1.0, num_warps=8, enable_fp_fusion=False,
            )

        def unfused_proxy():
            acc = attn.float() + residual.float()
            h = acc.bfloat16()
            x = (acc * torch.rsqrt(acc.square().mean(-1, keepdim=True) + 1e-6) *
                 (gain.float() + 1.0)).bfloat16()
            sig = torch.sigmoid((x @ gate.T).float())
            ids = torch.topk(sig + bias, 4, dim=-1).indices.to(torch.int32)
            weights = sig.gather(1, ids.long())
            weights = weights * (2.0 / weights.sum(-1, keepdim=True))
            ids = torch.cat((ids, torch.full((batch, 1), 128, device="cuda", dtype=torch.int32)), dim=-1)
            weights = torch.cat((weights, torch.ones((batch, 1), device="cuda")), dim=-1)
            _ = moe_sorting(ids, weights, 129, 6144, torch.bfloat16, block_size=16)
            return h, x

        fused()
        unfused_proxy()
        torch.cuda.synchronize()
        fused_us = triton.testing.do_bench_cudagraph(fused, rep=30) * 1000
        proxy_us = triton.testing.do_bench_cudagraph(unfused_proxy, rep=30) * 1000
        print(f"B={batch}: fused_front={fused_us:.2f}us, unfused_proxy={proxy_us:.2f}us", flush=True)


if __name__ == "__main__":
    main()
