# SPDX-License-Identifier: MIT
"""Single-GPU smoke test for the M3 fused front and AITER routing metadata."""

import ctypes as C
from pathlib import Path

import torch

from minimax_m3_gluon.front import front
from minimax_m3_gluon.host import allocate_workspace


def main():
    torch.cuda.set_device(0)
    device = "cuda"
    for batch in (1, 2, 4, 8, 16):
        torch.manual_seed(100 + batch)
        attn = torch.randn((batch, 6144), device=device, dtype=torch.bfloat16)
        residual = torch.randn_like(attn)
        gain = torch.randn((6144,), device=device, dtype=torch.bfloat16) * 0.05
        gate = torch.randn((128, 6144), device=device, dtype=torch.bfloat16) * 0.01
        bias = torch.randn((128,), device=device, dtype=torch.float32) * 0.05
        state = allocate_workspace(batch, device)
        kernel = front[(256,)](
            attn, residual, gain, gate, bias,
            state["HMid"], state["XNorm"], state["Scores"], state["Sigmoids"],
            state["RouteIDs"], state["RouteWeights"], state["SortIDs"],
            state["SortWeights"], state["SortExperts"], state["SortCounts"],
            state["SortM"], state["SortReverse"], state["SortOut"],
            state["Ready"], state["FrontEpochs"], state["FrontErrors"],
            B=batch, SHARED_WEIGHT=1.0, num_warps=8, enable_fp_fusion=False,
        )
        torch.cuda.synchronize()
        hip_path = next(
            line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libamdhip64.so" in line
        )
        hip = C.CDLL(hip_path)
        occupancy = hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
        occupancy.argtypes = [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t]
        occupancy.restype = C.c_int
        resident = C.c_int()
        rc = occupancy(C.byref(resident), C.c_void_p(kernel.function), 512, kernel.metadata.shared)
        assert rc == 0 and resident.value >= 1
        assert kernel.n_spills == 0, f"front spilled at B={batch}"
        if batch == 1:
            isa = kernel.asm["amdgcn"]
            assert "global_load_dwordx2" in isa and "sc1" in isa
            print(
                "front coherent ISA:",
                [line.strip() for line in isa.splitlines() if "sc1" in line][:4],
                flush=True,
            )
        assert state["FrontErrors"].item() == 0, "front mailbox timed out"
        midway = (attn.float() + residual.float()).bfloat16()
        assert torch.equal(state["HMid"], midway), "post-attention residual differs"
        norm = (attn.float() + residual.float())
        norm = norm * torch.rsqrt(norm.square().mean(-1, keepdim=True) + 1e-6)
        norm = (norm * (gain.float() + 1)).bfloat16()
        assert torch.allclose(state["XNorm"].float(), norm.float(), atol=0.04), "RMSNorm differs"
        logits = (norm.float() @ gate.float().T).bfloat16().float()
        sigmoid = torch.sigmoid(logits)
        selected = torch.topk(sigmoid + bias, 4, dim=-1).indices
        got = state["RouteIDs"][:, :4]
        assert torch.equal(got, selected), f"routing differs: {got} != {selected}"
        assert torch.all(state["RouteIDs"][:, 4] == 128)
        weights = sigmoid.gather(1, selected)
        weights = weights / weights.sum(-1, keepdim=True) * 2
        assert torch.allclose(state["RouteWeights"][:, :4], weights, atol=0.005)
        count = state["SortCounts"][0].item()
        assert 16 <= count <= batch * 5 * 16
        ids = state["SortIDs"][:count].cpu()
        experts = state["SortExperts"][:count // 16].cpu()
        expected = state["RouteIDs"].cpu()
        for block in range(count // 16):
            for packed in ids[block * 16 : (block + 1) * 16]:
                token = int(packed) & 0xFFFFFF
                if token == batch:
                    continue
                slot = int(packed) >> 24
                assert int(expected[token, slot]) == int(experts[block])
        print(
            f"B={batch}: route+metadata OK, blocks={count // 16}, "
            f"VGPR={kernel.n_regs}, resident/CU={resident.value}", flush=True
        )


if __name__ == "__main__":
    main()
