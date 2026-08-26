# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP GEMM1 loader equivalence. Single process, no torchrun:

    python op_tests/multigpu_tests/test_tp_gemm1.py --case equiv

The proof is physical, not tolerance-based. The same kernel source is compiled
twice and differs only in which A-operand loaders it instantiates. The reference
permutes A and its scale on the host and reads them with the ORIGINAL contiguous
loaders; the kernel under test leaves them dense and gathers by token id. Same
do_tile, same MFMA order, same epilogue, so the outputs must be bit-identical
and any difference is a loader bug.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# FlyDSL's disk-cache key walks *functions* reachable from the launcher; it never
# reads the source of a class, so editing ATileLoader / TPATileLoader / any other
# loader leaves the key unchanged and the stale binary is reused. That would make
# the negative controls below pass while testing nothing. Fingerprinting the whole
# mega_moe package restores the dependency. Must precede the flydsl import.
_MEGA_MOE_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "aiter", "ops", "flydsl", "kernels", "mega_moe")
)
_extra = os.environ.get("FLYDSL_EXTRA_SOURCE_DIRS", "")
os.environ["FLYDSL_EXTRA_SOURCE_DIRS"] = (
    f"{_extra}:{_MEGA_MOE_DIR}" if _extra else _MEGA_MOE_DIR
)

import torch  # noqa: E402

from aiter.fused_moe import moe_sorting  # noqa: E402
from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant  # noqa: E402
from aiter.ops.flydsl.kernels.mega_moe.tp_gemm1 import run_tp_gemm1  # noqa: E402
from tp_moe_stage1_ref import build_mxfp4_w1  # noqa: E402

MODEL_DIM, EXPERTS, TOPK, INTER = 7168, 384, 6, 384
SBM = 32


def _build(m_global, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(EXPERTS, generator=g)[:TOPK] for _ in range(m_global)]
    ).to(device=device, dtype=torch.int32)
    w = torch.rand((m_global, TOPK), generator=g).to(device=device, dtype=torch.float32)
    w = w / w.sum(-1, keepdim=True)
    sids, sw, seid, nv, _ = moe_sorting(
        ids, w, EXPERTS, MODEL_DIM, torch.bfloat16, block_size=SBM, accumulate=False
    )
    x = torch.randn((m_global, MODEL_DIM), generator=g).to(
        device=device, dtype=torch.bfloat16
    ) * (MODEL_DIM**-0.25)
    x_q, x_s = per_1x32_mx_quant(x, quant_mode="fp8")
    # One zeroed padding row at the end; gathered padding slots clamp to it.
    a = torch.cat(
        [x_q.view(torch.uint8), torch.zeros_like(x_q[:1].view(torch.uint8))], 0
    )
    s = torch.cat([x_s, torch.zeros_like(x_s[:1])], 0)
    return sids, sw, seid, nv, a.contiguous(), s.contiguous()


def case_equiv():
    device = torch.device("cuda", 0)
    _, _, w1_shuf, w1_scale_shuf = build_mxfp4_w1(
        EXPERTS, INTER, MODEL_DIM, device, seed=99
    )

    for m_global in (8, 64, 512, 1024):
        sids, _, seid, nv, a_dense, s_dense = _build(m_global, 1000 + m_global, device)
        nvalid = int(nv[0].item())
        n_tiles_m = nvalid // SBM
        max_sorted = ((sids.shape[0] + SBM - 1) // SBM) * SBM
        total_rows = a_dense.shape[0]
        pad_row = total_rows - 1

        trb = torch.arange(n_tiles_m + 64, dtype=torch.int32, device=device) * SBM

        # Reference: permute on the host, feed the ORIGINAL contiguous loaders.
        tok = (sids[:nvalid] & 0x00FFFFFF).clamp(max=pad_row).long()
        a_perm = a_dense[tok].contiguous()
        s_perm = s_dense[tok].contiguous()
        pad = (-a_perm.shape[0]) % SBM
        if pad:
            a_perm = torch.cat([a_perm, a_perm[:1].expand(pad, -1)], 0).contiguous()
            s_perm = torch.cat([s_perm, s_perm[:1].expand(pad, -1)], 0).contiguous()
        ref_ids = torch.arange(a_perm.shape[0], dtype=torch.int32, device=device)
        out_ref, os_ref = run_tp_gemm1(
            x=a_perm,
            scale_x=s_perm,
            w=w1_shuf,
            scale_w=w1_scale_shuf,
            tile_row_base=trb,
            expert_ids=seid,
            sorted_token_ids=ref_ids,
            num_valid_ids=nv,
            max_sorted=max_sorted,
            model_dim=MODEL_DIM,
            inter_dim=INTER,
            experts=EXPERTS,
            total_rows=a_perm.shape[0],
            gather=False,
            sort_block_m=SBM,
        )

        # Under test: dense A, gather by token id.
        out_got, os_got = run_tp_gemm1(
            x=a_dense,
            scale_x=s_dense,
            w=w1_shuf,
            scale_w=w1_scale_shuf,
            tile_row_base=trb,
            expert_ids=seid,
            sorted_token_ids=sids,
            num_valid_ids=nv,
            max_sorted=max_sorted,
            model_dim=MODEL_DIM,
            inter_dim=INTER,
            experts=EXPERTS,
            total_rows=total_rows,
            gather=True,
            sort_block_m=SBM,
        )
        torch.cuda.synchronize()

        pa = out_ref.view(torch.uint8)[:nvalid]
        pb = out_got.view(torch.uint8)[:nvalid]
        assert torch.equal(pa, pb), (
            f"m_global={m_global}: payload differs on "
            f"{int((pa != pb).sum())} of {pa.numel()} bytes"
        )
        rows = ((nvalid + 255) // 256) * 256
        cols = (((INTER // 32) + 7) // 8) * 8
        sa = os_ref[: rows * cols]
        sb = os_got[: rows * cols]
        assert torch.equal(
            sa, sb
        ), f"m_global={m_global}: mx scale differs on {int((sa != sb).sum())} bytes"
        # Only rows [0, nvalid) are written; the tail is uninitialised empty()
        # memory that no kernel reads, so checking it would read garbage.
        assert torch.isfinite(
            out_ref[:nvalid].float()
        ).all(), f"m_global={m_global}: ref non-finite"
        print(f"  m_global={m_global} nvalid={nvalid} tiles={n_tiles_m} bit-identical")

    print("case_equiv OK")


CASES = {"equiv": case_equiv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="equiv")
    args = ap.parse_args()
    torch.cuda.set_device(0)
    CASES[args.case]()


if __name__ == "__main__":
    main()
