# SPDX-License-Identifier: MIT
"""Check fused-front mailbox epochs and changed routes across graph replays."""

import torch

from minimax_m3_gluon.front import front
from minimax_m3_gluon.host import allocate_workspace


def main():
    torch.cuda.set_device(0)
    batch = 8
    attn = torch.randn((batch, 6144), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(attn)
    gain = torch.zeros(6144, device="cuda", dtype=torch.bfloat16)
    gate = torch.zeros((128, 6144), device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(128, device="cuda", dtype=torch.float32)
    state = allocate_workspace(batch, "cuda")

    def run():
        front[(256,)](
            attn, residual, gain, gate, bias,
            state["HMid"], state["XNorm"], state["Scores"], state["Sigmoids"],
            state["RouteIDs"], state["RouteWeights"], state["SortIDs"],
            state["SortWeights"], state["SortExperts"], state["SortCounts"],
            state["SortM"], state["SortReverse"], state["SortOut"],
            state["Ready"], state["FrontEpochs"], state["FrontErrors"],
            B=batch, SHARED_WEIGHT=1.0, num_warps=8, enable_fp_fusion=False,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for expert in (10, 20, 30, 40):
        bias.fill_(-10)
        bias[expert] = 10
        graph.replay()
        torch.cuda.synchronize()
        assert state["FrontErrors"].item() == 0
        assert torch.all(state["RouteIDs"][:, 0] == expert).item()
        print(f"graph replay routed to expert {expert}", flush=True)


if __name__ == "__main__":
    main()
