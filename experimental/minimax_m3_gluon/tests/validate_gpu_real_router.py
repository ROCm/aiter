# SPDX-License-Identifier: MIT
"""Check Gluon routing with gate/bias/norm tensors extracted from an M3 layer."""

import argparse
from pathlib import Path

import torch

from minimax_m3_gluon.front import front
from minimax_m3_gluon.host import allocate_workspace


def read_tensor(directory, name, dtype, shape):
    size = 1
    for extent in shape:
        size *= extent
    return torch.from_file(str(directory / f"{name}.bin"), dtype=dtype, size=size).reshape(shape).to("cuda")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_dir", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    # ATOM's loaded model view is BF16 although the checkpoint tensor is FP32.
    gate = read_tensor(args.weights_dir, "gate", torch.float32, (128, 6144)).bfloat16()
    bias = read_tensor(args.weights_dir, "bias", torch.float32, (128,))
    gain = read_tensor(args.weights_dir, "gain", torch.bfloat16, (6144,))
    for batch in (1, 8, 16):
        torch.manual_seed(833 + batch)
        attn = torch.randn((batch, 6144), dtype=torch.bfloat16, device="cuda")
        residual = torch.randn_like(attn)
        state = allocate_workspace(batch, "cuda")
        front[(256,)](
            attn, residual, gain, gate, bias,
            state["HMid"], state["XNorm"], state["Scores"], state["Sigmoids"],
            state["RouteIDs"], state["RouteWeights"], state["SortIDs"],
            state["SortWeights"], state["SortExperts"], state["SortCounts"],
            state["SortM"], state["SortReverse"], state["SortOut"],
            state["Ready"], state["FrontEpochs"], state["FrontErrors"],
            B=batch, SHARED_WEIGHT=1.0, num_warps=8, enable_fp_fusion=False,
        )
        torch.cuda.synchronize()
        acc = attn.float() + residual.float()
        norm = (acc * torch.rsqrt(acc.square().mean(-1, keepdim=True) + 1e-6) *
                (gain.float() + 1)).bfloat16()
        logits = (norm.float() @ gate.float().T).bfloat16().float()
        biased = torch.sigmoid(logits) + bias
        bits = biased.view(torch.int32)
        ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
        keys = (ordered & -128) | (127 - torch.arange(128, device="cuda", dtype=torch.int32))
        expected = torch.topk(keys, 4, dim=-1).indices
        got = state["RouteIDs"][:, :4]
        if not torch.equal(got, expected):
            print("max norm delta", (state["XNorm"].float() - norm.float()).abs().max().item())
            scores = torch.sigmoid(logits) + bias
            for token, slot in (got != expected).nonzero().tolist():
                mine, ref = int(got[token, slot]), int(expected[token, slot])
                print(
                    f"B={batch} token={token} slot={slot}: got={mine} ref={ref}; "
                    f"ref scores={scores[token, mine].item():.9f}/{scores[token, ref].item():.9f}; "
                    f"front sigmoid={state['Sigmoids'][token, mine].item():.9f}/"
                    f"{state['Sigmoids'][token, ref].item():.9f}; "
                    f"bias={bias[mine].item():.9f}/{bias[ref].item():.9f}",
                    flush=True,
                )
            raise AssertionError("route mismatch")
        print(f"B={batch}: real checkpoint BF16 gate routing OK", flush=True)


if __name__ == "__main__":
    main()
