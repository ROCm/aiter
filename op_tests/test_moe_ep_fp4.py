# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Regression test for GitHub issue #2343:
# GPU memory access fault with AITER CK MoE FP4 kernel + Expert Parallelism
# on MI355X (Kimi-K2.5-MXFP4).
#
# The root cause is uninitialized intermediate buffers (a2, moe_buf,
# sorted_expert_ids) allocated with torch.empty in the EP code path,
# causing the CK FP4 preshuffle kernel to read garbage expert IDs and
# access weight memory out of bounds.

import torch
import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.test_common import checkAllclose, run_perftest, perftest
from aiter.fused_moe import fused_topk, fused_moe, torch_moe
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils
from aiter.jit.utils.chip_info import get_gfx
import argparse

BLOCK_SIZE_M = 32
MAX_TOKENS = 4096 * 4


@perftest(num_warmup=1, num_iters=2)
def torch_moe_ref(
    hidden_states, w1, w2, topk_weight, topk_ids, expert_mask=None
):
    return torch_moe(
        hidden_states, w1, w2, topk_weight, topk_ids, expert_mask=expert_mask
    )


def test_fmoe_ep_fp4(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    shared_E=2,
    ep=4,
):
    """Test FP4 (MXFP4, per_1x32) fused_moe with Expert Parallelism.

    This specifically targets the configuration from issue #2343:
    large local expert counts (e.g. 96 out of 384) that previously caused
    GPU memory access faults due to uninitialized padding buffers.
    """
    if get_gfx() not in ["gfx950"]:
        print(f"  Skipping FP4 EP test on {get_gfx()} (requires gfx950/MI355X)")
        return

    ep_id = ep - 1
    total_experts = E + shared_E + 1
    expert_mask = torch.zeros(total_experts, dtype=dtypes.i32, device="cuda")
    expert_mask[ep_id * (E // ep) : (ep_id + 1) * E // ep] = 1
    local_E = torch.sum(expert_mask).item()
    fake_expertid = total_experts - 1
    expert_mask[-1] = 0
    expert_mask[E:-1] = 1

    qType = QuantType.per_1x32
    torch_quant = aiter.get_torch_quant(qType)

    input_data = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn(
        (local_E + shared_E, inter_dim * 2, model_dim), dtype=dtype, device="cuda"
    ) / 10
    w2 = torch.randn(
        (local_E + shared_E, model_dim, inter_dim), dtype=dtype, device="cuda"
    ) / 10
    score = torch.randn((token, E), device="cuda", dtype=dtype)

    total_topk_ids = torch.empty(
        (MAX_TOKENS, topk + shared_E + 1), dtype=dtypes.i32, device="cuda"
    )
    ns_topk_ids, s_topk_ids = total_topk_ids.split([topk, shared_E + 1], dim=1)
    shared_expert_ids = [E + i for i in range(shared_E + 1)]
    s_topk_ids_list = [[fake_expertid] * (shared_E + 1)] * MAX_TOKENS
    for i in range(ep_id, MAX_TOKENS, ep):
        s_topk_ids_list[i] = shared_expert_ids
    s_topk_ids[:] = torch.tensor(
        s_topk_ids_list, dtype=dtypes.i32, device="cuda"
    )

    total_topk_weights = torch.empty(
        (MAX_TOKENS, topk + shared_E + 1), dtype=dtypes.fp32, device="cuda"
    )
    ns_topk_weights, s_topk_weights = total_topk_weights.split(
        [topk, shared_E + 1], dim=1
    )
    s_topk_weights[:] = 0.1

    fused_topk(input_data, score, topk, True, ns_topk_ids, ns_topk_weights)
    topk_ids = total_topk_ids[:token]
    topk_weights = total_topk_weights[:token]

    w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)

    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    w1_qt_aiter = shuffle_weight(w1_qt, layout=(16, 16))
    w2_qt_aiter = shuffle_weight(w2_qt, layout=(16, 16))
    w1_scale_aiter = fp4_utils.e8m0_shuffle(w1_scale)
    w2_scale_aiter = fp4_utils.e8m0_shuffle(w2_scale)

    out_ck, avg_ck = run_perftest(
        fused_moe,
        input_data,
        w1_qt_aiter,
        w2_qt_aiter,
        topk_weights,
        topk_ids,
        expert_mask,
        w1_scale=w1_scale_aiter,
        w2_scale=w2_scale_aiter,
        quant_type=qType,
        activation=ActivationType.Silu,
        doweight_stage1=False,
        num_iters=3,
        num_warmup=1,
    )

    print(
        f"  [PASS] token={token}, model_dim={model_dim}, inter_dim={inter_dim}, "
        f"E={E}, topk={topk}, ep={ep}, local_E={local_E}, "
        f"avg={avg_ck:.2f} us -- no memory fault"
    )


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="FP4 MoE + Expert Parallelism regression test (issue #2343)",
)
parser.add_argument(
    "-m", "--token", type=int, nargs="*", default=[1, 32, 128],
    help="Token counts to test",
)
parser.add_argument(
    "-e", "--expert", type=int, nargs="*", default=[128, 384],
    help="Total expert counts (global)",
)
parser.add_argument(
    "-ep", "--expert_parallelism", type=int, nargs="*", default=[4, 8],
    help="EP degree",
)
parser.add_argument(
    "-k", "--topk", type=int, default=8, help="Top-k value",
)

args = parser.parse_args()

print("=" * 70)
print("FP4 MoE + Expert Parallelism regression test (issue #2343)")
print("=" * 70)

for E in args.expert:
    for ep in args.expert_parallelism:
        if E % ep != 0:
            continue
        local_E = E // ep
        for token in args.token:
            print(
                f"\nTest: E={E}, ep={ep}, local_E={local_E}, "
                f"token={token}, topk={args.topk}"
            )
            try:
                test_fmoe_ep_fp4(
                    dtypes.bf16,
                    token,
                    model_dim=1024,
                    inter_dim=512,
                    E=E,
                    topk=args.topk,
                    shared_E=2,
                    ep=ep,
                )
            except Exception as e:
                print(f"  [FAIL] {e}")

print("\n" + "=" * 70)
print("All FP4 EP tests completed.")
print("=" * 70)
