# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Test multi-B fused_moe: split weight tensors into multiple buffers and verify
that results match the single-tensor (list-of-1) FlyDSL reference.

Supports both even splits (E=256, DWDP=1,2,4,8) and uneven splits
(E=257 with fused shared expert, DWDP=4 [64,64,64,65], DWDP=8 [32x7+33]).

Mimics sglang DWDP production usage:
- Weights are 3D [E_i, N, K] per partition (separate allocations via .clone())
- Scales are 3D [E_i, N, K//32] per partition (matching sglang Quark format)
- Both are already pre-shuffled (shuffle_weight + e8m0_shuffle)
- is_shuffled attribute is set on each partition
"""

import torch
import aiter
from aiter import dtypes
from aiter.fused_moe import fused_topk, fused_moe
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils
from aiter.jit.utils.chip_info import get_gfx
import argparse

torch.set_default_device("cuda")


def test_multi_b_fused_moe(
    token=32,
    model_dim=4096,
    inter_dim=2048,
    E=256,
    topk=8,
    splits=None,
    dtype=dtypes.bf16,
):
    """Test multi-B fused_moe with arbitrary (even or uneven) splits.

    Args:
        splits: list of expert counts per partition, e.g. [64,64,64,64] or [64,64,64,65].
                If None, defaults to a single partition [E] (reference).
    """
    if splits is None:
        splits = [E]
    if get_gfx() not in ["gfx950"]:
        print("Skipping: multi-B test requires gfx950")
        return True

    assert sum(splits) == E, f"sum(splits)={sum(splits)} != E={E}"

    is_even = len(set(splits)) == 1
    split_desc = f"{len(splits)}-way even" if is_even else f"{len(splits)}-way uneven {splits}"

    print(f"\n{'='*60}")
    print(f"Testing multi-B: token={token}, E={E}, topk={topk}, {split_desc}")
    print(f"  model_dim={model_dim}, inter_dim={inter_dim}")
    print(f"{'='*60}")

    # === Create weights in sglang format ===
    w1_fp = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype)
    w2_fp = torch.randn((E, model_dim, inter_dim), dtype=dtype)

    torch_quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1_qt, w1_scale = torch_quant(w1_fp, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2_fp, quant_dtype=dtypes.fp4x2)

    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)

    # Pre-shuffle weights
    w1_qt_s = shuffle_weight(w1_qt, layout=(16, 16))
    w2_qt_s = shuffle_weight(w2_qt, layout=(16, 16))
    w1_qt_s.is_shuffled = True
    w2_qt_s.is_shuffled = True

    # Pre-shuffle scales (e8m0_shuffle, reshape to 3D)
    w1_scale_2d = fp4_utils.e8m0_shuffle(w1_scale)
    w2_scale_2d = fp4_utils.e8m0_shuffle(w2_scale)
    w1_scale_3d = w1_scale_2d.view(E, inter_dim * 2, -1)
    w2_scale_3d = w2_scale_2d.view(E, model_dim, -1)

    # Create input and routing
    hidden_states = torch.randn((token, model_dim), dtype=dtype)
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(hidden_states, score, topk, True)

    # === Reference: list-of-1 (FlyDSL path, no split) ===
    print("  Running reference (single tensor)...")
    out_ref = fused_moe(
        hidden_states, [w1_qt_s], [w2_qt_s], topk_weights, topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.per_1x32,
        w1_scale=[w1_scale_3d], w2_scale=[w2_scale_3d],
    )

    # === Multi-B: split into partitions ===
    print(f"  Running multi-B ({split_desc})...")
    w1_list, w2_list, w1s_list, w2s_list = [], [], [], []
    offset = 0
    for s in splits:
        w1_list.append(w1_qt_s[offset:offset+s].clone())
        w2_list.append(w2_qt_s[offset:offset+s].clone())
        w1s_list.append(w1_scale_3d[offset:offset+s].clone())
        w2s_list.append(w2_scale_3d[offset:offset+s].clone())
        offset += s

    for t in w1_list + w2_list:
        t.is_shuffled = True

    out_multi_b = fused_moe(
        hidden_states, w1_list, w2_list, topk_weights, topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.per_1x32,
        w1_scale=w1s_list, w2_scale=w2s_list,
    )

    # Compare via cosine similarity
    x, y = out_ref.double(), out_multi_b.double()
    denom = (x*x + y*y).sum()
    diff = (1 - 2*(x*y).sum()/denom).item() if denom > 0 else 0
    print(f"  Cosine diff: {diff:.6e}")

    if diff < 1e-4:
        print("  PASS")
        return True
    else:
        max_abs = (out_ref.float() - out_multi_b.float()).abs().max().item()
        print(f"  FAIL: cosine diff {diff:.6e} > 1e-4, max_abs={max_abs:.4f}")
        return False


# ---------------------------------------------------------------------------
# Predefined test configurations
# ---------------------------------------------------------------------------

# Even splits (E=256, DWDP=1,2,4,8)
EVEN_CONFIGS = [
    {"E": 256, "topk": 8, "splits": [256]},
    {"E": 256, "topk": 8, "splits": [128, 128]},
    {"E": 256, "topk": 8, "splits": [64, 64, 64, 64]},
    {"E": 256, "topk": 8, "splits": [32] * 8},
]

# Uneven splits (E=257, fused shared expert appended to last partition)
UNEVEN_CONFIGS = [
    {"E": 257, "topk": 9, "splits": [64, 64, 64, 65]},
    {"E": 257, "topk": 9, "splits": [32, 32, 32, 32, 32, 32, 32, 33]},
    {"E": 9, "topk": 4, "splits": [2, 2, 2, 3]},
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test multi-B fused MOE (even + uneven splits)")
    parser.add_argument("-t", "--token", type=int, nargs="*",
                        default=[1, 4, 8, 32, 128])
    parser.add_argument("--config", choices=["even", "uneven", "all"], default="all",
                        help="Test configuration set")
    parser.add_argument("--model_dim", type=int, default=4096)
    parser.add_argument("--inter_dim", type=int, default=2048)
    args = parser.parse_args()

    if args.config == "even":
        configs = EVEN_CONFIGS
    elif args.config == "uneven":
        configs = UNEVEN_CONFIGS
    else:
        configs = EVEN_CONFIGS + UNEVEN_CONFIGS

    passed = 0
    failed = 0
    for cfg in configs:
        for token in args.token:
            ok = test_multi_b_fused_moe(
                token=token,
                model_dim=args.model_dim,
                inter_dim=args.inter_dim,
                E=cfg["E"],
                topk=cfg["topk"],
                splits=cfg["splits"],
            )
            if ok:
                passed += 1
            else:
                failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    if failed > 0:
        exit(1)
