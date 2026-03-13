# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from aiter.ops.enum import QuantType
import torch
import aiter
import os
from aiter.test_common import (
    checkAllclose,
    perftest,
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

MAX_TOKENS = 4096 * 9


def build_local_expert_hash(expert_mask: torch.Tensor) -> torch.Tensor:
    """Cache global->local expert id map for EP."""
    local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32)
    local_expert_hash[local_expert_hash > 0] -= 1
    local_expert_hash[expert_mask == 0] = -1
    return local_expert_hash


def make_fused_moe_perftest(*, num_iters: int, num_warmup: int):

    @perftest(num_iters=num_iters, num_warmup=num_warmup)
    def _run(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        expert_mask,
        local_expert_hash,
        w1_scale,
        w2_scale,
        a1_scale,
        w1_lqq_scale,
        w1_lqq_zero,
        w2_lqq_scale,
        w2_lqq_zero,
        fc1_smooth_scale,
        fc2_smooth_scale,
        quant_type,
        activation,
        dtype,
        block_size_M,
    ):
        return fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            expert_mask=expert_mask,
            local_expert_hash=local_expert_hash,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            w1_lqq_scale=w1_lqq_scale,
            w1_lqq_zero=w1_lqq_zero,
            w2_lqq_scale=w2_lqq_scale,
            w2_lqq_zero=w2_lqq_zero,
            fc1_smooth_scale=fc1_smooth_scale,
            fc2_smooth_scale=fc2_smooth_scale,
            quant_type=quant_type,
            activation=activation,
            doweight_stage1=False,
            dtype=dtype,
            block_size_M=block_size_M,
        )

    return _run


def test_fmoe_lqq(
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    shared_E=2,
    ep=8,
    block_size_M=32,
    evenly=False,
    flydsl_only=False,
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

    local_expert_hash = build_local_expert_hash(expert_mask)

    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    score = torch.randn((token, E), device="cuda", dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    print(f"after fused_topk topk_ids: {topk_ids}")

    if shared_E > 0:
        shared_E_score = 0.1
        # init total_topk_ids, inference time you just need to fill ns_topk_ids in total_topk_ids
        total_topk_ids = torch.empty(
            (MAX_TOKENS, topk + shared_E + 1), dtype=dtypes.i32, device=input.device
        )
        ns_topk_ids, s_topk_ids = total_topk_ids.split([topk, shared_E + 1], dim=1)
        shared_expert_ids = [E + i for i in range(shared_E + 1)]
        s_topk_ids_list = [[fake_expertid] * (shared_E + 1)] * MAX_TOKENS
        for i in range(ep_id, MAX_TOKENS, ep):
            s_topk_ids_list[i] = shared_expert_ids
        s_topk_ids[:] = torch.tensor(
            s_topk_ids_list, dtype=dtypes.i32, device=input.device
        )

        # init total_topk_weights, inference time you just need to fill ns_topk_weights in total_topk_weights
        total_topk_weights = torch.empty(
            (MAX_TOKENS, topk + shared_E + 1), dtype=dtypes.fp32, device=input.device
        )
        ns_topk_weights, s_topk_weights = total_topk_weights.split(
            [topk, shared_E + 1], dim=1
        )
        s_topk_weights[:] = shared_E_score

        # inference time, use fused_topk to fill ns_topk_ids and ns_topk_weights
        fused_topk(input, score, topk, True, ns_topk_ids, ns_topk_weights)
        # inference time, topk_ids simply slices total_topk_ids into the number of input tokens, same for topk_weights
        topk_ids = total_topk_ids[:token]
        topk_weights = total_topk_weights[:token]
    else:
        topk_weights, topk_ids = fused_topk(input, score, topk, True)

    eprt = local_E + shared_E

    if evenly:
        topk_ids_list2 = [
            [((i * topk) + j) % E for j in range(topk)] for i in range(token)
        ]
        topk_ids_absolute_average = torch.tensor(
            topk_ids_list2, device=topk_ids.device, dtype=topk_ids.dtype
        )
        topk_ids[:, :topk] = topk_ids_absolute_average

    ######################################################################################
    print("[test] batch: ", token)
    print("[test] expr: ", eprt, "topk: ", topk)
    print("[test] model_dim: ", model_dim, " inter_dim: ", inter_dim)

    a1_qt, a1_scale = aiter.pertoken_quant(input, quant_dtype=dtypes.i8)
    fc1_smooth_scale = torch.randn((eprt, model_dim), dtype=dtypes.fp32, device="cuda")
    fc2_smooth_scale = torch.randn((eprt, inter_dim), dtype=dtypes.fp32, device="cuda")
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
    from aiter.fused_moe import get_2stage_cfgs

    get_2stage_cfgs.cache_clear()

    run_fused_moe = make_fused_moe_perftest(num_iters=128, num_warmup=5)
    out_asm = None
    us_asm = None
    if not flydsl_only:
        os.environ["AITER_USE_FLYDSL"] = "0"
        try:
            out_asm, us_asm = run_fused_moe(
                input,
                w1_lqq,
                w2_lqq,
                topk_weights,
                topk_ids,
                expert_mask,
                local_expert_hash,
                w1_scale,
                w2_scale,
                a1_scale,
                w1_lqq_scale_shf,
                w1_lqq_zero_uint8_shf,
                w2_lqq_scale_shf,
                w2_lqq_zero_uint8_shf,
                fc1_smooth_scale,
                fc2_smooth_scale,
                quant_type,
                act_type,
                dtype,
                block_size_M,
            )
            aiter.logger.info(f"[bench] ASM: {us_asm:>8.2f} us")
        except Exception as err:
            print(f"[warning] ASM backend failed or is unsupported: {err}")

    # 2. Run FlyDSL if FLIR_PATH is set
    out_flydsl = None
    us_flydsl = None
    try:
        if os.getenv("FLIR_PATH"):
            print("[test] Running FlyDSL backend...")
            os.environ["AITER_USE_FLYDSL"] = "1"
            get_2stage_cfgs.cache_clear()
            out_flydsl, us_flydsl = run_fused_moe(
                input,
                w1_lqq,
                w2_lqq,
                topk_weights,
                topk_ids,
                expert_mask,
                local_expert_hash,
                w1_scale,
                w2_scale,
                a1_scale,
                w1_lqq_scale_shf,
                w1_lqq_zero_uint8_shf,
                w2_lqq_scale_shf,
                w2_lqq_zero_uint8_shf,
                fc1_smooth_scale,
                fc2_smooth_scale,
                quant_type,
                act_type,
                dtype,
                block_size_M,
            )
            aiter.logger.info(f"[bench] FlyDSL: {us_flydsl:>8.2f} us")
    except Exception as err:
        print(f"[warning] FlyDSL backend failed or is unsupported: {err}")
    finally:
        os.environ["AITER_USE_FLYDSL"] = old_flydsl

    def calc_diff(x: torch.Tensor, y: torch.Tensor):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        sim = torch.cosine_similarity(x.flatten(), y.flatten(), dim=0)
        return 1 - sim

    def logn(tag: str, x: torch.Tensor, y: torch.Tensor, n: int = 10):
        a = x.reshape(-1)[:n]
        b = y.reshape(-1)[:n]
        diff = a - b
        print(f"[a vs b] {tag} (first {n} elements)")
        print(f"    a    : {a.shape}")
        print(f"           {a}")
        print(f"    b    : {b.shape}")
        print(f"           {b}")
        print("    diff :")
        print(f"           {diff}")

    def log_top_abs_diff(tag: str, x: torch.Tensor, y: torch.Tensor, n: int = 10):
        a = x.reshape(-1)
        b = y.reshape(-1)
        abs_diff = (a - b).abs()
        k = min(n, abs_diff.numel())
        if k == 0:
            return
        top_vals, top_idx = torch.topk(abs_diff, k=k)
        print(f"[a vs b] {tag} (top {k} abs diff)")
        print(f"    idx  : {top_idx}")
        print(f"    a    : {a[top_idx]}")
        print(f"    b    : {b[top_idx]}")
        print("    |diff|:")
        print(f"           {top_vals}")

    def log_logits_diff(tag: str, x: torch.Tensor, y: torch.Tensor):
        cos_diff = calc_diff(x, y)
        aiter.logger.info(f"[logits_diff] {tag} {cos_diff:.2e}")
        return cos_diff

    quant_rtol, quant_atol, quant_tol_err = 0.25, 1.0, 0.15
    asm_vs_ref_diff = None
    flydsl_vs_ref_diff = None
    flydsl_vs_asm_diff = None
    if out_asm is not None:
        aiter.logger.info("[test] Comparing ASM vs Ref...")
        asm_vs_ref_diff = checkAllclose(
            out_asm,
            out_ref,
            rtol=quant_rtol,
            atol=quant_atol,
            tol_err_ratio=quant_tol_err,
            msg="ASM vs Ref",
        )
        log_logits_diff("ASM vs Ref", out_asm, out_ref)
        log_top_abs_diff("ASM vs Ref", out_asm, out_ref, n=10)
        logn("ASM vs Ref", out_asm, out_ref, n=10)

    if out_flydsl is not None:
        aiter.logger.info("[test] Comparing FlyDSL vs Ref...")
        flydsl_vs_ref_diff = checkAllclose(
            out_flydsl,
            out_ref,
            rtol=quant_rtol,
            atol=quant_atol,
            tol_err_ratio=quant_tol_err,
            msg="FlyDSL vs Ref",
        )
        log_logits_diff("FlyDSL vs Ref", out_flydsl, out_ref)
        log_top_abs_diff("FlyDSL vs Ref", out_flydsl, out_ref, n=10)
        logn("FlyDSL vs Ref", out_flydsl, out_ref, n=10)

        if out_asm is not None:
            aiter.logger.info("[test] Comparing FlyDSL vs ASM...")
            flydsl_vs_asm_diff = checkAllclose(out_flydsl, out_asm, msg="FlyDSL vs ASM")
            log_logits_diff("FlyDSL vs ASM", out_flydsl, out_asm)
            log_top_abs_diff("FlyDSL vs ASM", out_flydsl, out_asm, n=10)
            logn("FlyDSL vs ASM", out_flydsl, out_asm, n=10)

    result = {
        "token": token,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "E": E,
        "shared_E": shared_E,
        "topk": topk,
        "token_per_routed_expert": token * topk / ep / (E // ep),
        "ep": ep,
        "block_size_M": block_size_M,
    }
    if us_asm is not None:
        result["us_asm"] = f"{us_asm:.2f}"
    if us_flydsl is not None:
        result["us_flydsl"] = f"{us_flydsl:.2f}"
    if asm_vs_ref_diff is not None:
        result["asm_vs_ref_diff"] = f"{asm_vs_ref_diff:.6f}"
    if flydsl_vs_ref_diff is not None:
        result["flydsl_vs_ref_diff"] = f"{flydsl_vs_ref_diff:.6f}"
    if flydsl_vs_asm_diff is not None:
        result["flydsl_vs_asm_diff"] = f"{flydsl_vs_asm_diff:.6f}"
    return result


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
    "--block_m",
    type=int,
    nargs="*",
    default=[32],
    help="""block_size_M value. Default: 32.
    e.g.: -x 32""",
)
parser.add_argument(
    "-evenly",
    "--evenly",
    action="store_true",
    default=False,
    help="Use evenly distributed expert ids.",
)
parser.add_argument(
    "-flydsl_only",
    "--flydsl_only",
    action="store_true",
    default=False,
    help="Skip ASM and run only the FlyDSL backend.",
)
args = parser.parse_args()


expert = 128 if args.expert is None else args.expert[0]
topk = 6 if args.topk is None else args.topk[0]
mdim = 5120 if args.model_dim is None else args.model_dim[0]
idim = 1536 if args.inter_dim is None else args.inter_dim[0]
ep = 1 if args.expert_parallelism is None else args.expert_parallelism
shared_E = 2 if args.shared_expert is None else args.shared_expert
token_list = [208] if args.token is None else args.token

print("\nRunning moe_lqq test + benchmark...")
df = []
for m in token_list:
    for block_m in args.block_m:
        ret = test_fmoe_lqq(
            token=m,
            model_dim=mdim,
            inter_dim=idim,
            E=expert,
            topk=topk,
            shared_E=shared_E,
            ep=ep,
            block_size_M=block_m,
            evenly=args.evenly,
            flydsl_only=args.flydsl_only,
        )
        if ret is not None:
            df.append(ret)
if df:
    import pandas as pd

    df = pd.DataFrame(df)
    aiter.logger.info("moe_lqq summary:\n%s", df.to_markdown(index=False))
    df.to_csv("moe_lqq_summary.csv", index=False)
