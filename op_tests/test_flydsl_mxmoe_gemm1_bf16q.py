# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXMOE GEMM1 ``_bf16q``: GEMM2 gets the same MXFP4 input as on the two-stage path.

The MXMOE GEMM1 quantizes its SwiGLU output to MXFP4 in its epilogue. With the
``_bf16q`` kernel-name token it first rounds the activation to bf16 and scales
the gate by alpha before exp2, as the two-stage path does (FlyDSL GEMM1 with a
bf16 output, then aiter's MXFP4 quant). The codes and E8M0 scales handed to
GEMM2 must then be equal on every route. The reference is the two-stage path
itself, not an fp32 torch model, because the claim is bit equality with it.
Shape: MiniMax-M3.1 at TP1 (128 experts, top-4, hidden 6144, inter 3072).

Run:
    pytest op_tests/test_flydsl_mxmoe_gemm1_bf16q.py -q
    python op_tests/test_flydsl_mxmoe_gemm1_bf16q.py   # timing
"""

import csv
import tempfile

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.mxfp4_kname import _parse_mxfp4_g1_kname
from aiter.test_common import checkAllclose, run_perftest

E, HIDDEN, INTER, TOPK = 128, 6144, 3072, 4
G1 = "flydsl_mxmoe_g1_a4w4_128x256x256_swiglu"
G1_BF16Q = G1 + "_bf16q"
G2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x256_reduce_persist_sbm128"
TWO_STAGE = (
    "flydsl_moe1_afp4_wfp4_bf16_t128x128x256_w2_bnt0",
    "flydsl_moe2_afp4_wfp4_bf16_t128x128x256_atomic",
)

_GFX950 = pytest.mark.skipif(
    not (torch.cuda.is_available() and get_gfx_runtime() == "gfx950"),
    reason="gfx950 + FlyDSL required",
)


def _unshuffle(scale):
    """Row-major E8M0 scales from the MoE GEMM layout (mx_scale_shuffle_idx)."""
    r, c = scale.shape
    return (
        scale.view(r // 32, c // 8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).reshape(r, c)
    )


def _case(tokens, seed):
    from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

    g = torch.Generator(device="cuda").manual_seed(seed)

    def mx(rows, cols, gate_up):
        # Random FP4 codes; E8M0 scales 2^-9 .. 2^-5 keep the GEMM outputs finite.
        q = torch.randint(0, 256, (E, rows, cols // 2), generator=g, device="cuda")
        s = torch.randint(118, 123, (E * rows, cols // 32), generator=g, device="cuda")
        w = moe_shuffle_weight(
            q.to(torch.uint8).view(dtypes.fp4x2), experts_cnt=E, gate_up=gate_up
        )
        w.is_shuffled = True
        return w, moe_shuffle_scale(s.to(torch.uint8), E, gate_up=gate_up)

    w1, w1_scale = mx(2 * INTER, HIDDEN, True)
    w2, w2_scale = mx(HIDDEN, INTER, False)
    x = torch.randn(tokens, HIDDEN, generator=g, device="cuda").to(dtypes.bf16)
    logits = torch.randn(tokens, E, generator=g, device="cuda")
    topk_weights, topk_ids = torch.softmax(logits, -1).topk(TOPK, -1)
    return {
        "hidden_states": x,
        "w1": w1,
        "w2": w2,
        "topk_weight": topk_weights,
        "topk_ids": topk_ids.to(torch.int32),
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
    }


def _select_row(mp, csv_dir, tokens, kn1, kn2):
    """Make (kn1, kn2) the only tuned fused_moe row for this shape."""
    import aiter.fused_moe as fm

    row = {
        "gfx": get_gfx_runtime(),
        "cu_num": get_cu_num(),
        "token": tokens,
        "model_dim": HIDDEN,
        "inter_dim": INTER,
        "expert": E,
        "topk": TOPK,
        "act_type": ActivationType.Swiglu,
        "dtype": dtypes.bf16,
        "q_dtype_a": dtypes.fp4x2,
        "q_dtype_w": dtypes.fp4x2,
        "q_type": QuantType.per_1x32,
        "use_g1u1": 1,
        "doweight_stage1": 0,
        "block_m": 128,
        "ksplit": 0,
        "kernelName1": kn1,
        "kernelName2": kn2,
        "us": 1.0,
        "run_1stage": 0,
    }
    path = f"{csv_dir}/{kn1}_{tokens}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    mp.setenv("AITER_CONFIG_FMOE", path)
    mp.setattr(fm, "cfg_2stages", None)
    fm.AITER_CONFIGS.get_config_file.cache_clear()
    fm.get_2stage_cfgs.cache_clear()


def _prepare(case):
    from aiter.fused_moe import fused_moe_prepare_stage2

    return fused_moe_prepare_stage2(
        **case,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        swiglu_limit=7.0,
    )


def _gemm2_input(case):
    """GEMM2's MXFP4 codes and E8M0 scales, one row per route (token * topk + slot)."""
    inputs = _prepare(case)
    tokens = case["hidden_states"].shape[0]
    # _args = (w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids, moe_out, topk)
    sorted_ids, num_valid = inputs._args[2], inputs._args[4]
    ids = sorted_ids[: int(num_valid[0])].long()
    token, slot = ids & 0xFFFFFF, ids >> 24
    rows = torch.nonzero((token < tokens) & (slot < TOPK)).view(-1)
    route = token[rows] * TOPK + slot[rows]
    assert torch.equal(torch.sort(route).values, torch.arange(tokens * TOPK).cuda())
    rows = rows[torch.argsort(route)]
    codes = inputs.a2.view(torch.uint8)
    # The two-stage path keeps a2 token-major; MXMOE writes one row per sorted slot.
    codes = codes.view(tokens * TOPK, -1) if codes.dim() == 3 else codes[rows]
    scales = _unshuffle(inputs.a2_scale.view(torch.uint8))[rows, : INTER // 32]
    return codes, scales


def test_bf16q_token_sets_only_bf16_act_quant():
    plain, bf16q = _parse_mxfp4_g1_kname(G1), _parse_mxfp4_g1_kname(G1_BF16Q)
    assert bf16q.pop("bf16_act_quant") and not plain.pop("bf16_act_quant")
    assert bf16q == plain


def test_aot_stage1_jobs_carry_bf16_act_quant(monkeypatch):
    import aiter.ops.flydsl.mxfp4_gemm1_kernels as g1
    from aiter.aot.flydsl import mxfp4_moe

    def stage1_jobs(kn1):
        row = {"token": 8192, "topk": TOPK, "model_dim": HIDDEN, "expert": E}
        row.update(inter_dim=INTER, kernelName1=kn1, kernelName2=G2)
        return [j for j in mxfp4_moe.jobs_from_rows([row]) if j["stage"] == 1]

    plain, bf16q = stage1_jobs(G1), stage1_jobs(G1_BF16Q)
    assert {j["bf16_act_quant"] for j in plain} == {False}
    assert {j["bf16_act_quant"] for j in bf16q} == {True}
    keys = [{mxfp4_moe._job_key(j)[:-1] for j in jobs} for jobs in (plain, bf16q)]
    assert keys[0] == keys[1]
    seen = []
    monkeypatch.setattr(g1, "flydsl_mxfp4_gemm1", lambda **kw: seen.append(kw))
    mxfp4_moe._compile_stage1(bf16q[0])
    assert seen[0]["bf16_act_quant"] is True


def test_unshuffle_inverts_shuffle_scale():
    from aiter.ops.shuffle import shuffle_scale

    scale = torch.randint(0, 256, (256, INTER // 32), dtype=torch.uint8)
    assert torch.equal(_unshuffle(shuffle_scale(scale)), scale)


@_GFX950
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("tokens", [4096, 8192, 16384])
def test_bf16q_gemm2_input_equals_two_stage(monkeypatch, tmp_path, tokens, seed):
    import aiter.fused_moe as fm

    case = _case(tokens, seed)
    got = {}
    try:
        for name, kn in (
            ("ref", TWO_STAGE),
            ("bf16q", (G1_BF16Q, G2)),
            ("plain", (G1, G2)),
        ):
            _select_row(monkeypatch, tmp_path, tokens, *kn)
            got[name] = _gemm2_input(case)
    finally:
        fm.AITER_CONFIGS.get_config_file.cache_clear()
        fm.get_2stage_cfgs.cache_clear()
    for i, what in enumerate(("MXFP4 codes", "E8M0 scales")):
        # fp32 holds every byte value exactly, so a mismatch log shows real deltas.
        ref, out = got["ref"][i].float(), got["bf16q"][i].float()
        err = checkAllclose(ref, out, rtol=0, atol=0, msg=f"{what}: ")
        assert err == 0, f"{what}: {err:.4%} of the bytes differ"
    # Without the token the fused epilogue keeps its own rounding.
    assert not torch.equal(got["plain"][0], got["ref"][0])


if __name__ == "__main__":
    print("us per call of fused_moe_prepare_stage2 (sort + GEMM1 + quant)")
    rows = (
        ("two-stage", TWO_STAGE),
        ("mxmoe", (G1, G2)),
        ("mxmoe_bf16q", (G1_BF16Q, G2)),
    )
    with pytest.MonkeyPatch.context() as mp, tempfile.TemporaryDirectory() as d:
        for tokens in (4096, 8192, 16384):
            case = _case(tokens, 0)
            line = [f"tokens {tokens:6d}"]
            for name, kn in rows:
                _select_row(mp, d, tokens, *kn)
                _, us = run_perftest(
                    _prepare, case, num_iters=20, num_warmup=3, num_rotate_args=1
                )
                line.append(f"{name} {us:8.1f}")
            print("  ".join(line))
