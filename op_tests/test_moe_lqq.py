# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from aiter.ops.enum import QuantType
import torch
import aiter
import os
from aiter.test_common import (
    checkAllclose,
)
from aiter.fused_moe import (
    fused_topk,
    fused_moe,
    torch_moe,
)
from aiter.ops.shuffle import shuffle_scale_zero_lqq_a8w4, shuffle_weight_lqq_a8w4
from aiter import ActivationType
from aiter import lqq_1x64_quant
from aiter import dtypes
import argparse


def moe_lqq_dequant(
    in_buffer: torch.Tensor,
    qscale_buf: torch.Tensor,
    qzero_buf: torch.Tensor,
    group_in_k_lqq: int = 64,
    output_dtype: torch.dtype = torch.int8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    eprt, M, N = in_buffer.shape
    numGroups = N // group_in_k_lqq

    assert qscale_buf.shape == (
        eprt,
        M,
        numGroups,
    ), f"qscale_buf shape mismatch: {qscale_buf.shape} != ({eprt}, {M}, {numGroups})"
    assert qzero_buf.shape == (
        eprt,
        M,
        numGroups,
    ), f"qzero_buf shape mismatch: {qzero_buf.shape} != ({eprt}, {M}, {numGroups})"
    assert (
        N % group_in_k_lqq == 0
    ), f"N={N} must be divisible by group_in_k_lqq={group_in_k_lqq}"

    scale_expanded = qscale_buf.unsqueeze(-1).expand(-1, -1, -1, group_in_k_lqq)
    zero_expanded = qzero_buf.unsqueeze(-1).expand(-1, -1, -1, group_in_k_lqq)

    in_reshaped = in_buffer.view(eprt, M, numGroups, group_in_k_lqq)

    out_reshaped = torch.addcmul(
        zero_expanded.to(torch.float32),
        in_reshaped.to(torch.float32),
        scale_expanded.to(torch.float32),
    )

    out = out_reshaped.view(eprt, M, N).to(output_dtype)

    return out.to(device)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!! do NOT overwrite this value !!!!!!!!!!!!!!!!!!!!!!!!!!!!
LQQ_I8_MAX = 119
#!!!!!!!!!!!!!!!!!!!!!!!!!!!! do NOT overwrite this value !!!!!!!!!!!!!!!!!!!!!!!!!!!!


def _get_local_expert_hash(expert_mask: torch.Tensor) -> torch.Tensor:
    """Cache global->local expert id map for EP."""
    local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32)
    local_expert_hash[local_expert_hash > 0] -= 1
    local_expert_hash[expert_mask == 0] = -1
    return local_expert_hash


def test_fmoe_lqq(
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    shared_E=2,
    ep=8,
    block_size_M=80,
):
    dtype = dtypes.bf16
    use_g1u1 = True
    quant_type = QuantType.lqq_1x64
    act_type = ActivationType.Silu
    group_in_k_lqq = 64

    # This gpu id in EP, this example use the last id
    ep_id = ep - 1
    # total_expert = unshared_expert + shared_expert + fake_expert(only use this fake expert id to mask)
    # expert_mask = torch.randint(
    #     0, 2, (E + shared_E + 1,), dtype=dtypes.i32, device="cuda"
    # )
    expert_mask = torch.zeros((E + shared_E + 1,), dtype=dtypes.i32, device="cuda")
    expert_mask[ep_id * (E // ep) : (ep_id + 1) * E // ep] = 1
    # # Get local expert Number in this gpu
    local_E = torch.sum(expert_mask).item()
    # The last expert
    fake_expertid = expert_mask.numel() - 1
    # Ensure fake expert to be masked
    expert_mask[-1] = 0
    # Ensure shared expert not to be maskedc
    expert_mask[E:-1] = 1

    local_expert_hash = _get_local_expert_hash(expert_mask)

    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    score = torch.randn((token, E), device="cuda", dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    if shared_E > 0:
        shared_E_score = 0.5
        s_topk_weights = torch.tensor(
            [
                [shared_E_score, shared_E_score],
            ]
            * token,
            dtype=dtypes.fp32,
            device=input.device,
        )
        topk_weights = torch.cat((topk_weights, s_topk_weights), dim=1)
        s_topk_ids = torch.tensor(
            [
                [E, E + 1],
            ]
            * token,
            dtype=dtypes.i32,
            device=input.device,
        )
        topk_ids = torch.cat((topk_ids, s_topk_ids), dim=1)

    eprt = local_E + shared_E

    ######################################################################################
    print("[test] batch: ", token)
    print("[test] expr: ", eprt, "topk: ", topk)
    print("[test] model_dim: ", model_dim, " inter_dim: ", inter_dim)

    a1_qt, a1_scale = aiter.pertoken_quant(input, quant_dtype=dtypes.i8)
    if shared_E > 0:
        fc1_smooth_scale = torch.randn(
            (eprt, model_dim), dtype=dtypes.fp32, device="cuda"
        )
        fc2_smooth_scale = torch.randn(
            (eprt, inter_dim), dtype=dtypes.fp32, device="cuda"
        )
    else:
        fc1_smooth_scale = None
        fc2_smooth_scale = None
    w1 = torch.randn((eprt, inter_dim * 2, model_dim), dtype=dtype, device="cuda")
    w2 = torch.randn((eprt, model_dim, inter_dim), dtype=dtype, device="cuda")
    exp_bias1 = torch.clamp(
        torch.randn((eprt, inter_dim * 2), dtype=dtype, device="cuda"),
        -1.0,
        1.0,
    )
    exp_bias2 = torch.clamp(
        torch.randn((E, model_dim), dtype=dtype, device="cuda"), -1.0, 1.0
    )

    w1_qt, w1_scale = aiter.pertoken_quant(
        w1, quant_dtype=torch.int8, dtypeMax=LQQ_I8_MAX
    )
    w2_qt, w2_scale = aiter.pertoken_quant(
        w2, quant_dtype=torch.int8, dtypeMax=LQQ_I8_MAX
    )

    w1_lqq_uint4, w1_lqq_scale, w1_lqq_zero_uint8, w1_lqq_zero = lqq_1x64_quant(
        w1_qt, group_in_k_lqq
    )
    w1_qt_dqt = moe_lqq_dequant(w1_lqq_uint4, w1_lqq_scale, w1_lqq_zero)
    w1_qt = w1_qt_dqt

    w2_lqq_uint4, w2_lqq_scale, w2_lqq_zero_uint8, w2_lqq_zero = lqq_1x64_quant(
        w2_qt, group_in_k_lqq
    )
    w2_qt_dqt = moe_lqq_dequant(w2_lqq_uint4, w2_lqq_scale, w2_lqq_zero)
    w2_qt = w2_qt_dqt

    w1_lqq = shuffle_weight_lqq_a8w4(w1_lqq_uint4).view(dtypes.i4x2)
    w2_lqq = shuffle_weight_lqq_a8w4(w2_lqq_uint4).view(dtypes.i4x2)
    w1_lqq_scale_shf = shuffle_scale_zero_lqq_a8w4(w1_lqq_scale, (4, 16))
    w2_lqq_scale_shf = shuffle_scale_zero_lqq_a8w4(w2_lqq_scale, (4, 16))
    w1_lqq_zero_uint8_shf = shuffle_scale_zero_lqq_a8w4(w1_lqq_zero_uint8, (4, 16))
    w2_lqq_zero_uint8_shf = shuffle_scale_zero_lqq_a8w4(w2_lqq_zero_uint8, (4, 16))

    ######################################################################################
    out_ref = torch_moe(
        input,
        w1_qt.to(dtype),
        w2_qt.to(dtype),
        topk_weights,
        topk_ids,
        fc1_scale=w1_scale,
        fc2_scale=w2_scale,
        fc1_smooth_scale=fc1_smooth_scale,
        fc2_smooth_scale=fc2_smooth_scale,
        expert_mask=expert_mask,
        activation=act_type,
    )

    ######################################################################################
    # 1. Run ASM
    old_flydsl = os.environ.get("AITER_USE_FLYDSL", "0")
    os.environ["AITER_USE_FLYDSL"] = "0"
    from aiter.fused_moe import get_2stage_cfgs

    get_2stage_cfgs.cache_clear()

    out_asm = fused_moe(
        input,
        w1_lqq,
        w2_lqq,
        topk_weights,
        topk_ids,
        expert_mask=expert_mask,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        w1_lqq_scale=w1_lqq_scale_shf,
        w1_lqq_zero=w1_lqq_zero_uint8_shf,
        w2_lqq_scale=w2_lqq_scale_shf,
        w2_lqq_zero=w2_lqq_zero_uint8_shf,
        fc1_smooth_scale=fc1_smooth_scale,
        fc2_smooth_scale=fc2_smooth_scale,
        quant_type=quant_type,
        activation=act_type,
        doweight_stage1=False,
        dtype=dtype,
        block_size_M=block_size_M,
    )

    # 2. Run FlyDSL if FLIR_PATH is set
    out_flydsl = None
    if os.getenv("FLIR_PATH"):
        print("[test] Running FlyDSL backend...")
        os.environ["AITER_USE_FLYDSL"] = "1"
        get_2stage_cfgs.cache_clear()
        aiter.logger.info(f"w1_lqq.shape: {w1_lqq.shape}")
        aiter.logger.info(f"w2_lqq.shape: {w2_lqq.shape}")
        out_flydsl = fused_moe(
            input,
            w1_lqq,
            w2_lqq,
            topk_weights,
            topk_ids,
            expert_mask=expert_mask,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            w1_lqq_scale=w1_lqq_scale_shf,
            w1_lqq_zero=w1_lqq_zero_uint8_shf,
            w2_lqq_scale=w2_lqq_scale_shf,
            w2_lqq_zero=w2_lqq_zero_uint8_shf,
            fc1_smooth_scale=fc1_smooth_scale,
            fc2_smooth_scale=fc2_smooth_scale,
            quant_type=quant_type,
            activation=act_type,
            doweight_stage1=False,
            dtype=dtype,
            block_size_M=32,
        )
    os.environ["AITER_USE_FLYDSL"] = old_flydsl

    def calc_diff(x: torch.Tensor, y: torch.Tensor):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        sim = torch.cosine_similarity(x.flatten(), y.flatten(), dim=0)
        return 1 - sim

    def checkLogitsDiff(x: torch.Tensor, y: torch.Tensor, tol=1e-3):
        cos_diff = calc_diff(x, y)
        aiter.logger.info(
            f"cosine_similarity: {1 - cos_diff:.6f}, diff: {cos_diff:.2e}"
        )
        return cos_diff < tol

    quant_rtol, quant_atol, quant_tol_err = 0.25, 1.0, 0.15
    aiter.logger.info("[test] Comparing ASM vs Ref...")
    checkAllclose(
        out_asm,
        out_ref,
        rtol=quant_rtol,
        atol=quant_atol,
        tol_err_ratio=quant_tol_err,
        msg="ASM vs Ref",
    )
    assert checkLogitsDiff(out_asm, out_ref), "ASM vs Ref logits diff is too large"

    if out_flydsl is not None:
        aiter.logger.info("[test] Comparing FlyDSL vs Ref...")
        checkAllclose(
            out_flydsl,
            out_ref,
            rtol=quant_rtol,
            atol=quant_atol,
            tol_err_ratio=quant_tol_err,
            msg="FlyDSL vs Ref",
        )
        assert checkLogitsDiff(
            out_flydsl, out_ref
        ), "FlyDSL vs Ref logits diff is too large"

        aiter.logger.info("[test] Comparing FlyDSL vs ASM...")
        checkAllclose(out_flydsl, out_asm, msg="FlyDSL vs ASM")
        assert checkLogitsDiff(
            out_flydsl, out_asm
        ), "FlyDSL vs ASM logits diff is too large"


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="select test",
)
parser.add_argument(
    "-t",
    "--token",
    type=int,
    nargs="*",
    default=None,
    help="""Token Num.
    e.g.: -t 128""",
)
parser.add_argument(
    "-md",
    "--model_dim",
    type=int,
    nargs="*",
    default=None,
    help="""Model dim.
    e.g.: -md 4096""",
)
parser.add_argument(
    "-id",
    "--inter_dim",
    type=int,
    nargs="*",
    default=None,
    help="""Intermediate dim.
    e.g.: -id 1024""",
)
parser.add_argument(
    "-e",
    "--expert",
    type=int,
    nargs="*",
    default=None,
    help="""Number of experts.
    e.g.: -e 32""",
)
parser.add_argument(
    "-k",
    "--topk",
    type=int,
    nargs="*",
    default=None,
    help="""Top-k value.
    e.g.: -k 5""",
)
parser.add_argument(
    "-ep",
    "--expert_parallelism",
    type=int,
    nargs="?",
    default=None,
    help="""Expert Parallelism.
    e.g.: -ep 8""",
)
parser.add_argument(
    "-se",
    "--shared_expert",
    type=int,
    nargs="?",
    default=None,
    help="""Shared experts.
    e.g.: -se 0""",
)
parser.add_argument(
    "-x",
    "--subx",
    type=int,
    nargs="*",
    default=None,
    help="""block_size_M value.
    e.g.: -x 80""",
)
args = parser.parse_args()

print("\nRunning moe_lqq test...")
expert = 128 if args.expert is None else args.expert[0]
topk = 6 if args.topk is None else args.topk[0]
tokens = 208 if args.token is None else args.token[0]
mdim = 5120 if args.model_dim is None else args.model_dim[0]
idim = 1536 if args.inter_dim is None else args.inter_dim[0]
ep = 1 if args.expert_parallelism is None else args.expert_parallelism
shared_E = 2 if args.shared_expert is None else args.shared_expert

for subX in [80] if args.subx is None else args.subx:
    test_fmoe_lqq(
        token=tokens,
        model_dim=mdim,
        inter_dim=idim,
        E=expert,
        topk=topk,
        shared_E=shared_E,
        ep=ep,
        block_size_M=subX,
    )
