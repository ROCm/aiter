# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import (
    checkAllclose,
    run_perftest,
    perftest,
)
from aiter.fused_moe import (
    fused_topk,
    fused_moe,
    torch_moe,
)

from aiter.fused_moe_bf16_asm import asm_moe
from aiter.ops.shuffle import (
    shuffle_weight,
    shuffle_weight_a16w4,
    shuffle_scale_a16w4,
    moe_shuffle_scale,
    moe_shuffle_weight,
)
from aiter import ActivationType
from aiter import QuantType
from aiter.ops.flydsl.moe_common import GateMode
from aiter import pertoken_quant
from aiter import dtypes
from aiter import get_gfx
from aiter.utility import fp4_utils
import argparse
import os

BLOCK_SIZE_M = 32
MAX_TOKENS = 4096 * 4


@perftest(num_warmup=1, num_iters=2)
def torch_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
    expert_mask=None,
    activation=ActivationType.Silu,
):
    return torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
        expert_mask,
        activation=activation,
    )


@perftest()
def asm_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
    a16=False,
    expert_mask=None,
    local_expert_hash=None,
):

    return asm_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
        a16,
        expert_mask=expert_mask,
        local_expert_hash=local_expert_hash,
    )


quant_algo = [
    "No",  # g1u0/ck(g1ux) support
    "int8quant",  # g1u1 support
    "fp8quant",  # g1u1 support
    "int8smoothquant",  # g1u1/g1u0 support
    "fp8smoothquant",  # g1u1 support
]


def test_fmoe_ep(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    quant="No",
    use_g1u1=False,
    shared_E=2,
    ep=8,
):
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
    # Ensure shared expert not to be masked
    expert_mask[E:-1] = 1

    quantAlgoId = quant_algo.index(quant)
    if quantAlgoId not in [0, 3] and not use_g1u1:
        print("g1u0 only could test no quant and int8smoothquant")
        return

    quantstr = quant_algo[quantAlgoId]
    quant_dtype = dtypes.i8 if quantstr.startswith("int8") else dtypes.fp8
    use_smooth = "smooth" in quantstr

    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    if use_g1u1:
        w1 = (
            torch.randn(
                (local_E + shared_E, inter_dim * 2, model_dim),
                dtype=dtype,
                device="cuda",
            )
            / 10
        )
    else:
        w1 = (
            torch.randn(
                (local_E + shared_E, inter_dim, model_dim), dtype=dtype, device="cuda"
            )
            / 10
        )
    w2 = (
        torch.randn(
            (local_E + shared_E, model_dim, inter_dim), dtype=dtype, device="cuda"
        )
        / 10
    )
    score = torch.randn((token, E), device="cuda", dtype=dtype)

    # if shared_E > 0:
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
    s_topk_ids[:] = torch.tensor(s_topk_ids_list, dtype=dtypes.i32, device=input.device)

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

    # else:
    #     topk_ids, topk_weights = fused_topk(input, score, topk, True)

    if quantAlgoId == 0:
        # ref2 implement
        ref2, avg_c = torch_moe_test(
            input, w1, w2, topk_weights, topk_ids, expert_mask=expert_mask
        )

        # b implement
        torch_quant = aiter.get_torch_quant(aiter.QuantType.No)
        w1_qt, w1_scale = torch_quant(w1, quant_dtype=None)
        w2_qt, w2_scale = torch_quant(w2, quant_dtype=None)
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape)
        w1_qt_aiter = shuffle_weight(w1_qt_aiter, layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2_qt_aiter, layout=(16, 16))

        # if use_g1u1:
        #     out_b = ref2
        #     avg_b = 9999
        #     print("asm g1u1 only support quant/smoothquant Now")
        # else:
        #     out_b, avg_b = asm_moe_test(
        #         input,
        #         w1_qt_aiter,
        #         w2_qt_aiter,
        #         topk_weights,
        #         topk_ids,
        #         expert_mask=expert_mask,
        #     )

        # test ck moe
        out_ck, avg_ck = run_perftest(
            fused_moe,
            input,
            w1_qt_aiter,
            w2_qt_aiter,
            topk_weights,
            topk_ids,
            expert_mask,
            w1_scale=None,
            w2_scale=None,
            quant_type=aiter.QuantType.No,
            activation=ActivationType.Silu,
            doweight_stage1=False,
        )

        # msg = f"[perf] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, asm_avg: {avg_b:>8.2f} us, ck_avg: {avg_ck:>8.2f} us, uplift: {avg_c/avg_b-1:.1%}"
        # checkAllclose(ref2, out_b, rtol=0.01, atol=10, msg=msg)
        checkAllclose(ref2, out_ck, rtol=0.01, atol=10, msg="ck check")

    else:
        w1, fc1_scale = pertoken_quant(w1, quant_dtype=quant_dtype)
        w2, fc2_scale = pertoken_quant(w2, quant_dtype=quant_dtype)

        sp1 = (local_E + shared_E, inter_dim)
        sp2 = (local_E + shared_E, model_dim)

        if not use_smooth:
            fc1_smooth_scale = None
            fc2_smooth_scale = None
        else:
            # [expert, 1, model_dim]
            fc1_smooth_scale = torch.randn(sp2, dtype=dtypes.fp32, device="cuda")
            # [expert, 1, inter_dim]
            fc2_smooth_scale = torch.randn(sp1, dtype=dtypes.fp32, device="cuda")

        # ref2 implement
        ref2, avg_c = torch_moe_test(
            input,
            w1,
            w2,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            expert_mask,
        )

        # b implement
        w1b = shuffle_weight(w1)
        w2b = shuffle_weight(w2)
        local_expert_hash = None
        if expert_mask is not None and use_smooth:
            local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32)
            local_expert_hash[local_expert_hash > 0] -= 1
            local_expert_hash[expert_mask == 0] = -1
        out_b, avg_b = asm_moe_test(
            input,
            w1b,
            w2b,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            expert_mask=expert_mask,
            local_expert_hash=local_expert_hash,
        )

        def calculateTensorsSize(*args):
            num_btype = 0
            for el in args:
                if isinstance(el, torch.Tensor):
                    num_btype += el.element_size() * el.numel()
            return num_btype

        num_tb = calculateTensorsSize(
            input,
            input,
            w1b,
            w2b,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
        ) / (1024 * 1024 * 1024 * 1024.0)
        bw = num_tb * 1e6 / avg_b
        print(
            f"[BW  ] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, dtype: {dtype}, asm_bandwidth: {bw:>8.2f}TB/s"
        )

        if use_smooth and (
            (
                (inter_dim % 512 == 0 or inter_dim % 320 == 0)
                and (w1b.dtype == dtypes.fp8 and inter_dim * 2 == w1b.shape[1])
            )
            or (
                (inter_dim % 256 == 0 or inter_dim % 320 == 0 or inter_dim % 384 == 0)
                and (w1b.dtype == dtypes.i8 and inter_dim * 2 == w1b.shape[1])
            )
            or (
                (inter_dim % 512 == 0)
                and (w1b.dtype == dtypes.i8 and inter_dim == w1b.shape[1])
            )
        ):
            out_b2, avg_b2 = asm_moe_test(
                input,
                w1b,
                w2b,
                topk_weights,
                topk_ids,
                fc1_scale,
                fc2_scale,
                fc1_smooth_scale,
                fc2_smooth_scale,
                a16=True,
                expert_mask=expert_mask,
            )
            msg = f"[perf] a8w8 asm: {avg_b:>8.2f} vs a16w8 asm: {avg_b2:>8.2f} ......"
            checkAllclose(out_b, out_b2, atol=10, msg=msg)

        msg = f"[perf] {use_g1u1=} {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, asm_avg: {avg_b:>8.2f} us ...... uplift: {avg_c/avg_b-1:.1%}"
        checkAllclose(ref2, out_b, rtol=0.01, atol=10, msg=msg)
        # checkAllclose(ref2, avg_ck, rtol=0.01, atol=10)


# ---------------------------------------------------------------------------
# EP end-to-end with per_1x32 mxfp4 (a8w4 / a4w4) via fused_moe
# ---------------------------------------------------------------------------


def _per_1x32_mxfp4_quant(w):
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w_qt, w_scale = torch_quant(w, quant_dtype=dtypes.fp4x2)
    w_qt = w_qt.view(w.shape[0], w.shape[1], w.shape[2] // 2)
    return w_qt, w_scale


def _randn_or_const(shape, *, const_init, scale=1.0, dtype=dtypes.bf16, device="cuda"):
    """randn (scaled) by default; a constant VALUE tensor when const_init is set.

    Mirrors test_flydsl_grouped_gemm_gfx1250.py's --const-init: the const path
    fills with VALUE exactly (the ``scale`` only applies to the random path)."""
    if const_init is not None:
        return torch.full(shape, float(const_init), dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device) * scale


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """1 - cosine-similarity in double; matches test_moe_2stage.calc_diff."""
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    # Under EP a single token can route entirely to non-local experts, so both
    # ref and out are all-zero and denom==0. That is a correct (empty) result,
    # not a mismatch — guard the division so it doesn't report a spurious NaN.
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


summary_table = []


def test_fmoe_ep_mxfp4(
    quant_label,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    shared_E=2,
    ep=8,
    ep_mode="real",
    fake_ep_rank=0,
    const_init=None,
):
    """End-to-end EP fused_moe with per_1x32 mxfp4 weights.
    quant_label ∈ {"a8w4_mxfp4", "a4w4_mxfp4"}.

    ep_mode selects how tokens/routing reach fused_moe:
      * "real" (default): simulate MORI dispatch. `token` is the GLOBAL token count;
        a source token is received iff it owns >=1 local expert (deduplicated), so
        total_recv <= token and non-local routes are masked. Buffer is trimmed to
        trim_M = token*topk (ATOM CUDAGraph bound) with a padded tail.
      * "fake": mirror ATOM fake-EP (ATOM_FAKE_EP + --fake-eplb,
        atom/model_ops/moe.py:121-136). `token` is the PER-RANK batch M; every
        token's topk picks are redirected onto this rank's local expert block via a
        rolling window, so all M*topk routes are local (no mask, no dedup) and the
        grouped GEMM runs the full M*topk load. total_recv == trim_M == M.
        fake_ep_rank selects the local block [rank*L,(rank+1)*L) (default 0, matches
        ATOM_FAKE_EP_RANK); ignored in real mode (which uses ep-1).

    `token` is the **global** token count (total across all EP ranks).
    The test mirrors ATOM's real MoriV2ModularKernel shapes after dispatch+trim:

      graph_bs    = token // ep          (per-rank tokens before dispatch)
      n_src       = token                (global source tokens across all ranks)
      total_recv  = #tokens routed to THIS rank (MORI's total_recv_t, <= n_src)
      trim_M      = graph_bs * topk * ep (= token * topk, ATOM's CUDAGraph trim bound)

    Unlike a balanced "every token arrives" assumption, `total_recv` is derived
    from the actual routing exactly like MORI dispatch (and run_ref() in
    test_mori_all2all.py): a source token is received iff it owns >=1 *local*
    expert (routed or active shared), deduplicated to one buffer row per token.
    MORI returns that count as the device scalar `total_recv_t`, which ATOM
    forwards to fused_moe as `num_local_tokens` (mirrors test_mega_moe.py's
    DeviceMoEPipeline._layer_step, where total_recv_t comes straight from
    op.dispatch and feeds moe_forward's num_local_tokens).

    The dispatch buffer has `trim_M` rows with the full `topk` routing dimension.
    Only the first `total_recv` rows carry valid data; the `[total_recv, trim_M)`
    tail is dead padding (fused_moe uses `num_local_tokens` to skip it).
    expert_mask filters non-local experts so fused_moe only computes on this rank's
    experts."""
    _gfx = get_gfx()
    if _gfx not in ["gfx950", "gfx1250"]:
        print(f"skip {quant_label}: mxfp4 requires gfx950/gfx1250, got {_gfx}")
        return
    if _gfx == "gfx1250" and quant_label != "a8w4_mxfp4":
        print(f"skip {quant_label} on gfx1250: only a8w4_mxfp4 supported")
        return

    # ---------- ATOM shape model (MoriV2ModularKernel) ----------
    # Before MORI dispatch: each of `ep` ranks holds `graph_bs` tokens, each
    #   with `topk` *global* expert ids (spanning all E experts).
    # MORI dispatch: sends each token to every rank that owns at least one of
    #   its topk experts. The token (hidden_states row) is sent once per unique
    #   dest_pe (deduplicated); all topk idx/weight travel with it.
    # After dispatch: recv buffer shape (mr, hidden_dim) with (mr, topk) ids.
    #   total_recv = number of unique tokens that landed on this rank.
    # After trim: buffer is sliced to graph_bs * topk * dp_size (CUDAGraph bound).
    #
    # Realistic simulation:
    #   experts_per_rank = E // ep. total_recv is computed from the actual routing
    #   (tokens owning >=1 local expert, deduplicated) -- MORI's total_recv_t --
    #   rather than assuming every token arrives. total_recv <= n_src = token.
    experts_per_rank = E // ep

    # Which EP rank this process simulates. real mode keeps the historical last
    # rank (ep-1); fake mode uses fake_ep_rank to match ATOM_FAKE_EP_RANK (default
    # 0) so the local expert block lines up block-for-block with the fake-EP
    # server (init_balance_router_logits' fake_ep_rank, moe.py:134).
    ep_id = fake_ep_rank if ep_mode == "fake" else ep - 1
    assert 0 <= ep_id < ep, f"ep_id={ep_id} out of range [0,{ep})"
    local_expert_start = ep_id * experts_per_rank
    local_expert_end = (ep_id + 1) * experts_per_rank
    expert_mask = torch.zeros((E + shared_E + 1,), dtype=dtypes.i32, device="cuda")
    expert_mask[local_expert_start:local_expert_end] = 1
    local_E = int(torch.sum(expert_mask).item())
    fake_expertid = expert_mask.numel() - 1
    expert_mask[-1] = 0
    expert_mask[E:-1] = 1

    dtype = dtypes.bf16

    if ep_mode == "fake":
        # ---------- ATOM fake-EP (ATOM_FAKE_EP + --fake-eplb) shape model ----------
        # Mirrors init_balance_router_logits()'s fake_ep_rank branch,
        # atom/model_ops/moe.py:121-136. Under ATOM_FAKE_EP the parallel config is
        # forced to dp_size=1 (FusedMoEParallelConfig.make, moe.py:191-202), so
        # use_all2all_kernels is False (moe.py:171-173) and there is NO MORI
        # dispatch/combine: the full per-rank batch flows straight into fused_moe
        # (forward_impl, moe.py:3654-3687). --fake-eplb then rewrites the router so
        # every token's topk picks land on THIS rank's local block via a rolling
        # window (moe.py:131-135). Consequences vs the real path:
        #   * `token` is the PER-RANK forward batch M (server-side M), not a global
        #     count -- nothing shrinks it.
        #   * all M*topk routes are local => no expert_mask drop, no dedup.
        #   * total_recv == M and trim_M == M (no CUDAGraph pad tail); the grouped
        #     GEMM computes the full M*topk load (a fully-loaded EP rank).
        M = token
        L = experts_per_rank
        graph_bs = M
        n_src = M
        total_recv = M
        trim_M = M

        input_ = _randn_or_const((M, model_dim), const_init=const_init, dtype=dtype)

        # Rolling-window local routing == moe.py:131-135:
        #   local_slot = (t*topk + k) % L ; expert_ids = ep_id*L + local_slot
        # topk <= L keeps the topk picks distinct within a token, and the window
        # advancing by topk each row keeps the L local experts balanced.
        assert topk <= L, f"fake-eplb requires topk({topk}) <= experts_per_rank({L})"
        t = torch.arange(M, device="cuda").unsqueeze(1)
        k = torch.arange(topk, device="cuda").unsqueeze(0)
        local_slot = (t * topk + k) % L
        expert_ids = ep_id * L + local_slot  # (M, topk) int64, all local

        # Faithful replay of moe.py:130-135 followed by the REAL routing primitive:
        # scatter 1.0 into synthetic router_logits exactly like the fake-eplb branch
        # (moe.py builds a full (max_num_tokens, E) table then slices [:M]; because
        # the formula is per-row, building M rows here is identical), then run
        # fused_topk with renormalize=True -- the same topk the real branch uses --
        # so topk_ids AND topk_weights come from the actual kernel, not hardcoded.
        # Equal selected logits => softmax+renormalize gives uniform weights, but
        # now via the real path. (routed_scaling_factor is a later model-forward
        # multiply, not part of routing, and the real branch doesn't model it
        # either.)
        router_logits = torch.zeros((M, E), dtype=dtype, device="cuda")
        router_logits.scatter_(1, expert_ids, 1.0)
        # No shared/fake column: moe_sorting reads topk from topk_ids.shape[1]
        # (aiter/fused_moe.py:396,527,569), so (M, topk) expands to exactly M*topk
        # local routes -- no masked padding column.
        topk_ids = torch.empty((M, topk), dtype=dtypes.i32, device="cuda")
        topk_weights = torch.empty((M, topk), dtype=dtypes.fp32, device="cuda")
        fused_topk(input_, router_logits, topk, True, topk_ids, topk_weights)

        # Every row is valid; no padded tail to skip.
        num_local_tokens = torch.tensor(
            [total_recv], dtype=dtypes.i32, device="cuda"
        )

        # Reference sees exactly the same batch + routing.
        ref_input = input_
        ref_topk_ids = topk_ids
        ref_topk_weights = topk_weights

        print(
            f"  [EP sim/fake] per_rank_M={M} topk={topk} ep={ep} L={L} "
            f"local_experts=[{local_expert_start},{local_expert_end}) "
            f"routes={M * topk} (all local, no mask/dedup) trim_M={trim_M}"
        )
    else:
        graph_bs = token // ep
        n_src = graph_bs * ep  # global source tokens (all EP ranks' pre-dispatch)
        trim_M = graph_bs * topk * ep  # ATOM's CUDAGraph trim bound (buffer rows)

        # ---------- Simulate MORI dispatch output ----------
        # Step 1: generate the `n_src` *source* tokens (what all EP ranks hold
        #   before dispatch). Each token picks topk experts from ALL E experts via
        #   fused_topk (the router runs *before* dispatch on the source rank).
        src_input = _randn_or_const((n_src, model_dim), const_init=const_init, dtype=dtype)
        src_score = torch.randn((n_src, E), dtype=dtype, device="cuda")
        src_topk_ids = torch.empty(
            (n_src, topk + shared_E + 1), dtype=dtypes.i32, device="cuda"
        )
        ns_ids, s_ids = src_topk_ids.split([topk, shared_E + 1], dim=1)
        shared_expert_ids = [E + i for i in range(shared_E + 1)]
        s_ids_list = [[fake_expertid] * (shared_E + 1)] * n_src
        for i in range(ep_id, n_src, ep):
            s_ids_list[i] = shared_expert_ids
        s_ids[:] = torch.tensor(s_ids_list, dtype=dtypes.i32, device="cuda")
        src_topk_weights = torch.empty(
            (n_src, topk + shared_E + 1), dtype=dtypes.fp32, device="cuda"
        )
        ns_wts, s_wts = src_topk_weights.split([topk, shared_E + 1], dim=1)
        s_wts[:] = 0.1
        fused_topk(src_input, src_score, topk, True, ns_ids, ns_wts)

        # Step 2: MORI dispatch keeps only the tokens that own >=1 *local* expert
        #   on this rank (routed OR active shared), deduplicated to one row per
        #   token -- exactly run_ref()'s `mask.any(1)` selection in
        #   test_mori_all2all.py. The received count is the device scalar MORI
        #   returns as total_recv_t.
        recv_mask = expert_mask[src_topk_ids.long()].bool().any(dim=1)
        recv_input = src_input[recv_mask].contiguous()
        recv_topk_ids = src_topk_ids[recv_mask].contiguous()
        recv_topk_weights = src_topk_weights[recv_mask].contiguous()
        total_recv = int(recv_input.shape[0])
        assert total_recv <= trim_M, f"total_recv={total_recv} exceeds trim_M={trim_M}"
        print(
            f"  [EP sim] global_token={token} topk={topk} ep={ep} graph_bs={graph_bs} "
            f"n_src={n_src} total_recv={total_recv} trim_M={trim_M}"
        )

        # Step 3: build the dispatch buffer at trim_M (ATOM's CUDAGraph bound).
        #   First total_recv rows = valid received tokens (from step 2).
        #   Rows [total_recv, trim_M) = dead padding (MORI arena tail).
        input_ = _randn_or_const((trim_M, model_dim), const_init=const_init, dtype=dtype)
        input_[:total_recv] = recv_input

        total_topk_ids = torch.zeros(
            (trim_M, topk + shared_E + 1), dtype=dtypes.i32, device="cuda"
        )
        total_topk_ids[:total_recv] = recv_topk_ids

        total_topk_weights = torch.zeros(
            (trim_M, topk + shared_E + 1), dtype=dtypes.fp32, device="cuda"
        )
        total_topk_weights[:total_recv] = recv_topk_weights

        topk_ids = total_topk_ids
        topk_weights = total_topk_weights

        # total_recv_t: device scalar matching MORI's dispatch return; fused_moe
        # gets it as num_local_tokens and processes only the first total_recv rows,
        # skipping the padded tail (mirrors DeviceMoEPipeline._layer_step in
        # test_mega_moe.py, where total_recv_t from op.dispatch feeds moe_forward's
        # num_local_tokens).
        total_recv_t = torch.tensor([total_recv], dtype=dtypes.i32, device="cuda")
        num_local_tokens = total_recv_t

        # Reference: same received tokens, same routing, same expert_mask.
        ref_input = recv_input
        ref_topk_ids = recv_topk_ids
        ref_topk_weights = recv_topk_weights

    total_local = local_E + shared_E
    w1 = _randn_or_const(
        (total_local, inter_dim * 2, model_dim),
        const_init=const_init,
        scale=0.1,
        dtype=dtype,
    )
    w2 = _randn_or_const(
        (total_local, model_dim, inter_dim),
        const_init=const_init,
        scale=0.1,
        dtype=dtype,
    )

    w1_qt, w1_scale = _per_1x32_mxfp4_quant(w1)
    w2_qt, w2_scale = _per_1x32_mxfp4_quant(w2)

    # Reference uses the dequantized mxfp4 weights so the comparison is
    # apples-to-apples (kernel sees the same quantized values). Mirrors the
    # mxfp4_to_f32 + e8m0_to_f32 dequant in aiter/fused_moe.py:2018-2021 and
    # the quantized-weight reference path in test_moe_2stage.py:333-345.
    def _dequant(w_qt, w_scale, orig_shape):
        wf = fp4_utils.mxfp4_to_f32(w_qt).view(*orig_shape)
        sf = fp4_utils.e8m0_to_f32(w_scale).view(orig_shape[0], orig_shape[1], -1)
        sf = sf.unsqueeze(-1).expand(-1, -1, -1, 32).reshape(*orig_shape)
        return (wf * sf).to(dtype)

    w1_deq = _dequant(w1_qt, w1_scale, w1.shape)
    w2_deq = _dequant(w2_qt, w2_scale, w2.shape)
    ref, _ = torch_moe_test(
        ref_input,
        w1_deq,
        w2_deq,
        ref_topk_weights,
        ref_topk_ids,
        expert_mask=expert_mask,
    )

    if _gfx == "gfx1250":
        # gfx1250 grouped GEMM path: FlyDSL grouped layout. Weights stay uint8
        # (a8w4 -> q_dtype_a=fp8), per_1x32 mxfp4 weights.
        w1_u8 = w1_qt.view(torch.uint8)
        w2_u8 = w2_qt.view(torch.uint8)
        _tdm_gugu = os.environ.get("AITER_EP_TDM_GUGU", "0") in ("1", "true", "True", "yes")
        if _tdm_gugu:
            # gugu (INTERLEAVE) stage1 layout so the EP path is routed through the
            # felix TDM batched GEMM (_grouped_a8w4_tdm_moe, gugu-only). gate/up are
            # row-interleaved ([g0,u0,g1,u1,...]) inside moe_shuffle_weight/scale,
            # matching test_flydsl_grouped_gemm_gfx1250.py --layout gugu. Also needs
            # AITER_GROUPED_A8W4_TDM_EP=1 (default) for the EP TDM dispatch.
            w1_a = moe_shuffle_weight(
                w1_u8, experts_cnt=total_local, is_guinterleave=True, gate_up=True
            )
            w2_a = shuffle_weight(w2_u8, layout=(16, 16))
            w1_s = moe_shuffle_scale(
                w1_scale.contiguous(),
                experts_cnt=total_local,
                is_guinterleave=True,
                gate_up=True,
            )
            w2_s = moe_shuffle_scale(w2_scale.contiguous(), experts_cnt=total_local)
            gate_mode = GateMode.INTERLEAVE.value
        else:
            # Default: SEPARATED (gguu) -> grouped mxscale masked GEMM path.
            w1_a = shuffle_weight(w1_u8, layout=(16, 16))
            w2_a = shuffle_weight(w2_u8, layout=(16, 16))
            w1_s = moe_shuffle_scale(w1_scale.contiguous(), experts_cnt=total_local)
            w2_s = moe_shuffle_scale(w2_scale.contiguous(), experts_cnt=total_local)
            gate_mode = GateMode.SEPARATED.value
        act = ActivationType.Silu
        os.environ["AITER_FORCE_A8W4"] = "1"
        os.environ.setdefault("AITER_USE_GROUPED_GEMM", "1")
    elif quant_label == "a8w4_mxfp4":
        # gfx950 a8w4 (fp8 activations, mxfp4 weights): use the CK a16w4 layout
        # — weights interleaved on N (gate/up) — paired with gate_mode=INTERLEAVE
        # at the call site. The FlyDSL fp4 (16,16) shuffle is separated and
        # would mis-route gate/up here.
        w1_a = shuffle_weight_a16w4(w1_qt, 16, True)
        w1_s = shuffle_scale_a16w4(w1_scale, total_local, True)
        w2_a = shuffle_weight_a16w4(w2_qt, 16, False)
        w2_s = shuffle_scale_a16w4(w2_scale, total_local, False)
        act = ActivationType.Silu
        gate_mode = GateMode.INTERLEAVE.value
        # Force the fp8 (a8w4) kernel regardless of token count. Below the
        # default AITER_BF16_FP8_MOE_BOUND (256) the picker selects bf16/a16w4,
        # which for Silu at ksplit<=1 has no kernel and dispatch-crashes.
        os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
    elif quant_label == "a4w4_mxfp4":
        # gfx950 a4w4 (fp4 activations, mxfp4 weights): FlyDSL fp4/fp4 layout —
        # shuffle (16,16) + e8m0 scale shuffle, gate/up separated. Matches
        # test_moe_2stage.py:251-255 and pairs with gate_mode=SEPARATED.
        w1_a = shuffle_weight(w1_qt, layout=(16, 16))
        w2_a = shuffle_weight(w2_qt, layout=(16, 16))
        w1_s = fp4_utils.e8m0_shuffle(w1_scale)
        w2_s = fp4_utils.e8m0_shuffle(w2_scale)
        w1_a.is_shuffled = True
        w2_a.is_shuffled = True
        act = ActivationType.Silu
        gate_mode = GateMode.SEPARATED.value
    else:
        raise ValueError(f"unknown quant_label: {quant_label}")
    # Optional per-kernel (gemm1/gemm2) timing, gated by AITER_EP_KERNEL_BENCH=1.
    # Mirrors the TP test's --scenario kernel: one eager fused_moe call with the
    # grouped kernel_bench hook set captures the gemm1/gemm2 launch callables,
    # then each is looped in isolation. Default behavior (env unset) is unchanged.
    _ep_kernel_bench = os.environ.get("AITER_EP_KERNEL_BENCH", "0") in (
        "1",
        "true",
        "True",
        "yes",
    )
    gemm1_us = None
    gemm2_us = None
    if _ep_kernel_bench and _gfx == "gfx1250":
        from aiter.ops.flydsl import grouped_moe_gfx1250 as _grouped

        _cap: list = []
        _grouped.kernel_bench_callable = _cap
        try:
            out = fused_moe(
                input_,
                w1_a,
                w2_a,
                topk_weights,
                topk_ids,
                expert_mask=expert_mask,
                activation=act,
                gate_mode=gate_mode,
                quant_type=QuantType.per_1x32,
                w1_scale=w1_s,
                w2_scale=w2_s,
                num_local_tokens=num_local_tokens,
            )
        finally:
            _grouped.kernel_bench_callable = None
        _ku = {}
        for _name, _callable in _cap:
            _, _u = run_perftest(
                _callable, num_warmup=5, num_iters=101, testGraph=False
            )
            _ku[_name] = _u
        gemm1_us = _ku.get("gemm1")
        gemm2_us = _ku.get("gemm2")
        us = (gemm1_us or 0.0) + (gemm2_us or 0.0)
        _g1s = "n/a" if gemm1_us is None else f"{gemm1_us:.2f}"
        _g2s = "n/a" if gemm2_us is None else f"{gemm2_us:.2f}"
        print(
            f"[ep-kernel-bench] {quant_label} token={token}(local={total_recv}) "
            f"gemm1 us = {_g1s} gemm2 us = {_g2s}"
        )
    else:
        out, us = run_perftest(
            fused_moe,
            input_,
            w1_a,
            w2_a,
            topk_weights,
            topk_ids,
            expert_mask=expert_mask,
            activation=act,
            gate_mode=gate_mode,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_s,
            w2_scale=w2_s,
            num_local_tokens=num_local_tokens,
            num_warmup=3,
            num_iters=16,
        )

    # Trim to valid prefix (total_recv rows); the [total_recv, trim_M) tail is padding.
    out = out[:total_recv]
    diff = (ref - out).float()
    abs_err = diff.abs()
    abs_mean = abs_err.mean().item()
    abs_max = abs_err.max().item()
    rel_mean = (abs_err / (ref.float().abs() + 1e-6)).mean().item()
    logits_diff = _calc_diff(ref, out)

    _msg = (
        f"{quant_label} ep={ep} token={token}(recv={total_recv}, trim_M={trim_M}) "
        f"model_dim={model_dim} inter_dim={inter_dim} E={E} topk={topk}"
    )
    if _gfx == "gfx1250":
        # The grouped a8w4 path quantizes activations to fp8, so the reference
        # (bf16 activations) differs elementwise by more than atol/rtol=5e-2
        # even without EP. Use the grouped tests' cosine criterion instead
        # (test_flydsl_grouped_gemm_gfx1250.py: logits_diff < 0.01).
        _logits_diff_tol = 0.01
        err = logits_diff
        _verdict = "PASSED" if logits_diff < _logits_diff_tol else "FAILED"
        print(
            f"[aiter] {_msg} logits_diff={logits_diff:.6f} "
            f"(tol {_logits_diff_tol}) {_verdict}"
        )
    else:
        err = checkAllclose(ref, out, atol=5e-2, rtol=5e-2, msg=_msg)

    summary_table.append(
        {
            "quant": quant_label,
            "ep_mode": ep_mode,
            "global_token": token,
            "total_recv": total_recv,
            "trim_M": trim_M,
            "model_dim": model_dim,
            "inter_dim": inter_dim,
            "E": E,
            "topk": topk,
            "ep": ep,
            "us": round(us, 2),
            "gemm1_us": None if gemm1_us is None else round(gemm1_us, 2),
            "gemm2_us": None if gemm2_us is None else round(gemm2_us, 2),
            "logits_diff": round(logits_diff, 6),
            "abs_mean": round(abs_mean, 4),
            "abs_max": round(abs_max, 4),
            "rel_mean": round(rel_mean, 4),
            "checkAllclose_err": err,
        }
    )


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="select test",
)
parser.add_argument(
    "-t",
    "--test",
    type=str,
    nargs="*",
    default=[
        "test_fmoe_16_bit",
        "g1u1_no_quant",
        "g1u1_int8quant",
        "g1u1_fp8quant",
        "g1u0_int8smoothquant",
        "g1u1_int8smoothquant",
        "g1u1_fp8smoothquant",
        "g1u1_a8w4_mxfp4",
        "g1u1_a4w4_mxfp4",
    ],
    help="""Select test to run.
    e.g.: -t g1u1_int8quant
          or -t test_fmoe_16_bit
          or -t g1u1_no_quant
          or -t g1u1_int8quant
          or -t g1u1_fp8quant
          or -t g1u0_int8smoothquant (only runs on gfx942)
          or -t g1u1_int8smoothquant
          or -t g1u1_fp8smoothquant
          or -t g1u1_a8w4_mxfp4          (EP, per_1x32 mxfp4 a8w4, gfx950)
          or -t g1u1_a4w4_mxfp4          (EP, per_1x32 mxfp4 a4w4, gfx950)""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    "--token",
    type=int,
    nargs="*",
    default=[128],
    help="""Global token count. For EP mxfp4 tests the dispatch buffer is
    token*topk rows (ATOM trim bound) with token valid rows.
    e.g.: -m 128""",
)
parser.add_argument(
    "-hd",
    "--hidden_dim",
    type=int,
    nargs="*",
    default=[4096],
    help="""Hidden states dim.
    e.g.: -hd 4096""",
)
parser.add_argument(
    "-id",
    "--inter_dim",
    type=int,
    nargs="*",
    default=[1024],
    help="""Intermediate dim.
    e.g.: -id 1024""",
)
parser.add_argument(
    "-e",
    "--expert",
    type=int,
    nargs="?",
    default=32,
    help="""Number of experts.
    e.g.: -e 32""",
)
parser.add_argument(
    "-k",
    "--topk",
    type=int,
    nargs="?",
    default=5,
    help="""Top-k value.
    e.g.: -k 5""",
)
parser.add_argument(
    "-ep",
    "--expert_parallelism",
    type=int,
    nargs="*",
    default=[8],
    help="""Expert Parallelism.
    e.g.: -ep 8""",
)
parser.add_argument(
    "--ep-mode",
    type=str,
    choices=["real", "fake"],
    default="real",
    help="""Routing model for the mxfp4 EP tests (g1u1_a8w4_mxfp4/g1u1_a4w4_mxfp4):
    real = simulate MORI dispatch: -m is GLOBAL tokens, dedup + mask, only
           ~1/ep of routes land locally (total_recv <= token).
    fake = ATOM fake-EP (ATOM_FAKE_EP + --fake-eplb, moe.py:121-136): -m is the
           PER-RANK batch M, every token's topk redirected onto local experts
           (full M*topk local routes, no mask/dedup). Aligns with
           serve_dsv4_fake_ep_trace.sh.""",
)
parser.add_argument(
    "--fake-ep-rank",
    type=int,
    default=0,
    help="""EP rank to simulate in --ep-mode fake (matches ATOM_FAKE_EP_RANK,
    default 0). Selects the local expert block [rank*L, (rank+1)*L).
    Ignored in --ep-mode real (which always uses ep-1).""",
)
parser.add_argument(
    "--const-init",
    type=float,
    nargs="?",
    const=0.0,
    default=None,
    metavar="VALUE",
    help="""initialize activations (input) and weights (w1/w2) to the constant
    VALUE instead of random values (mxfp4 EP tests only). Bare --const-init uses
    0.0 (zero-init). Routing scores stay random so expert selection is unchanged.
    Mirrors test_flydsl_grouped_gemm_gfx1250.py --const-init.""",
)

args = parser.parse_args()
gpu_arch = get_gfx()

for test in args.test:
    print(f"\nRunning test: {test}")
    if test == "test_fmoe_16_bit":
        print("test test_fmoe 16 bit")
        # print("\ng1u0 no quant")
        # for dtype in [dtypes.fp16, dtypes.bf16]:
        #     for m in [7, 128, 256]:
        #         for dim in [4096, 8192]:
        #             for hdim in [1024, 1280]:
        #                 for ep in [4, 8]:
        #                     test_fmoe_ep(
        #                         dtype, m, dim, hdim, 128, 6, quant="No", shared_E=2, ep=ep
        #                     )

    elif test == "g1u1_no_quant":
        for dtype in args.dtype:
            for m in args.token:
                for hdim in args.hidden_dim:
                    for idim in args.inter_dim:
                        for ep in args.expert_parallelism:
                            expert = args.expert
                            topk = args.topk
                            test_fmoe_ep(
                                dtype,
                                m,
                                hdim,
                                idim,
                                expert,
                                topk,
                                quant="No",
                                use_g1u1=True,
                                shared_E=2,
                                ep=ep,
                            )
    elif test == "g1u1_int8quant":
        for dtype in args.dtype:
            for m in args.token:
                for hdim in args.hidden_dim:
                    for idim in args.inter_dim:
                        expert = args.expert
                        topk = args.topk
                        for ep in args.expert_parallelism:
                            test_fmoe_ep(
                                dtype,
                                m,
                                hdim,
                                idim,
                                expert,
                                topk,
                                quant="int8quant",
                                use_g1u1=True,
                                shared_E=2,
                                ep=ep,
                            )
    elif test == "g1u1_fp8quant":
        for dtype in args.dtype:
            for m in args.token:
                for hdim in args.hidden_dim:
                    for idim in args.inter_dim:
                        expert = args.expert
                        topk = args.topk
                        for ep in args.expert_parallelism:
                            test_fmoe_ep(
                                dtype,
                                m,
                                hdim,
                                idim,
                                expert,
                                topk,
                                quant="fp8quant",
                                use_g1u1=True,
                                shared_E=2,
                                ep=ep,
                            )
    elif test == "g1u0_int8smoothquant":
        if gpu_arch != "gfx942":
            print(f"skip {test} on {gpu_arch}: only runs on gfx942")
            continue
        for dtype in args.dtype:
            for m in args.token:
                for hdim in args.hidden_dim:
                    for idim in args.inter_dim:
                        expert = args.expert
                        topk = args.topk
                        for ep in args.expert_parallelism:
                            test_fmoe_ep(
                                dtype,
                                m,
                                hdim,
                                idim,
                                expert,
                                topk,
                                quant="int8smoothquant",
                                use_g1u1=False,
                                shared_E=2,
                                ep=ep,
                            )
    elif test == "g1u1_int8smoothquant":
        for dtype in args.dtype:
            for m in args.token:
                for hdim in args.hidden_dim:
                    for idim in args.inter_dim:
                        expert = args.expert
                        topk = args.topk
                        for ep in args.expert_parallelism:
                            test_fmoe_ep(
                                dtype,
                                m,
                                hdim,
                                idim,
                                expert,
                                topk,
                                quant="int8smoothquant",
                                use_g1u1=True,
                                shared_E=0,
                                ep=ep,
                            )
    elif test in ("g1u1_a8w4_mxfp4", "g1u1_a4w4_mxfp4"):
        label = "a8w4_mxfp4" if test == "g1u1_a8w4_mxfp4" else "a4w4_mxfp4"
        for m in args.token:
            for hdim in args.hidden_dim:
                for idim in args.inter_dim:
                    for ep in args.expert_parallelism:
                        test_fmoe_ep_mxfp4(
                            label,
                            m,
                            hdim,
                            idim,
                            args.expert,
                            args.topk,
                            shared_E=0,
                            ep=ep,
                            ep_mode=args.ep_mode,
                            fake_ep_rank=args.fake_ep_rank,
                            const_init=args.const_init,
                        )
    elif test == "g1u1_fp8smoothquant":
        for dtype in args.dtype:
            for m in args.token:
                for hdim in args.hidden_dim:
                    for idim in args.inter_dim:
                        expert = args.expert
                        topk = args.topk
                        for ep in args.expert_parallelism:
                            test_fmoe_ep(
                                dtype,
                                m,
                                hdim,
                                idim,
                                expert,
                                topk,
                                quant="fp8smoothquant",
                                use_g1u1=True,
                                shared_E=2,
                                ep=ep,
                            )
    else:
        raise ValueError(f"Unknown test: {test}")

if summary_table:
    try:
        import pandas as pd

        _df = pd.DataFrame(summary_table)
        print("\nmoe_ep_mxfp4 summary (markdown):")
        print(_df.to_markdown(index=False))
    except Exception as _e:
        print(f"[summary] pandas unavailable ({_e}); raw rows:")
        for _row in summary_table:
            print(_row)
