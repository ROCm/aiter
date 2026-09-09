# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`fused_compress_attn`'s paged e8m0 scale must match what each reader expects.

The FP4 indexer cache is written here and read by a mqa-logits kernel, and the
two families disagree on the permutation: FlyDSL's readers want MFMA-16x16, the
OPUS ones MFMA-32x32. Both are the same size per block, so writing the wrong one
is silent. `quant_mode` selects it (`flydsl16_fp4` / `opus32_fp4`).

Run the same compress twice, changing only that. Invert the fly16 permutation to
recover natural order, push it through `scale_to_opus`, and require the
opus32 run to match byte for byte.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from aiter.ops.flydsl.kernels.fused_compress_attn import (
    flydsl_fused_compress_attn,
)

_HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("test_flydsl_compress_attn")  # compress harness
T = _load("test_pa_mqa_logits_opus")  # scale_to_opus reference

# The CSA Indexer FP4 shape: D=128, RD=64, ratio=4, overlap, fp4 + ue8m0 + preshuffle.
SHAPE = ("indexer_fp4", 128, 64, 4, True, "fp4", True, True)
FP4_MODES = ("flydsl16_fp4", "opus32_fp4")


def _fly16_to_natural(scale_fly, k_per_block, groups_per_row):
    """Undo `cs_off = k_tile*(4*KVBS) + group4*KVBS + (slot%16)*4 + slot//16`.

    Returned as [num_blocks * k_per_block, groups_per_row], the order the
    reference permutations take as input.
    """
    nb = scale_fly.shape[0]
    flat = scale_fly.reshape(nb, -1)
    k_tiles = groups_per_row // 4
    out = torch.empty(
        nb, k_per_block, groups_per_row, dtype=torch.uint8, device=scale_fly.device
    )
    for g in range(groups_per_row):
        k_tile, group4 = divmod(g, 4)
        for slot in range(k_per_block):
            sflat = (slot % 16) * 4 + slot // 16
            out[:, slot, g] = flat[
                :, k_tile * (4 * k_per_block) + group4 * k_per_block + sflat
            ]
    assert k_tiles >= 1
    return out.reshape(nb * k_per_block, groups_per_row)


def _run(mode):
    inp = C._build_inputs(SHAPE, bs=2, mtp=0, mode="prefill")
    if inp["quant_mode"] not in ("fp4", *FP4_MODES):
        pytest.skip("harness did not produce the fp4 indexer shape")
    inp["cache_scale"].zero_()
    flydsl_fused_compress_attn(
        kv_in=inp["kv_in"], score_in=inp["score_in"], kv_state=inp["kv_state"],
        score_state=inp["score_state"], state_slot_mapping=inp["state_slot_mapping"],
        plan_gpu=inp["plan_gpu"], ape=inp["ape"], rms_weight=inp["rms_weight"],
        rms_eps=inp["rms_eps"], cos_cache=inp["cos_cache"], sin_cache=inp["sin_cache"],
        kv_cache=inp["kv_cache"], block_tables=inp["block_tables"],
        k_per_block=inp["k_per_block"], ratio=inp["ratio"], head_dim=inp["head_dim"],
        rope_head_dim=inp["rope_head_dim"], overlap=inp["overlap"], quant=inp["quant"],
        quant_mode=mode, cache_scale=inp["cache_scale"],
        use_ue8m0=inp["use_ue8m0"], preshuffle=inp["preshuffle"],
    )  # fmt: skip
    return inp["kv_cache"].clone(), inp["cache_scale"].clone(), inp


def test_opus32_kv_scale_matches_reference_permutation():
    kv_fly, s_fly, inp = _run("flydsl16_fp4")
    kv_opus, s_opus, _ = _run("opus32_fp4")
    # The DATA region is shared by both readers; only the scale permutation moves.
    assert torch.equal(kv_fly, kv_opus), "the scale layout must not move fp4 data"

    kpb = inp["k_per_block"]
    gpr = inp["head_dim"] // 32
    nat = _fly16_to_natural(s_fly, kpb, gpr)
    want = T.scale_to_opus(nat, kpb)
    assert torch.equal(s_opus.reshape(want.shape), want)


def test_layouts_are_permutations_of_one_another():
    """The failure mode this guards: same bytes, same multiset, different order."""
    _, s_fly, _ = _run("flydsl16_fp4")
    _, s_opus, _ = _run("opus32_fp4")
    assert s_fly.numel() == s_opus.numel()
    nz = s_fly != 0
    assert nz.any(), "compress wrote no scales -- the test would prove nothing"
    assert torch.equal(s_fly.flatten().sort().values, s_opus.flatten().sort().values)
    assert not torch.equal(s_fly, s_opus)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
