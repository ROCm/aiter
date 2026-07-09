"""Per-M gemm1/gemm2 launch comparison for dsv4 a8w4 (fp8 a / fp4 w):

  baseline = current vendored FlyDSL mixed_moe stage1 (aiter.ops.flydsl.flydsl_moe_stage1)
  v2       = FlyDSL#753 "mxmoe v2" gemm1 (aiter.ops.flydsl.kernels.mxmoe_dispatcher.mxfp4_moe_gemm1)

Both kernels compute the same a8w4 up/gate MoE gemm but through different input
pipelines (baseline: aiter sort + unsorted a-scale, gather+fused-quant-out; v2:
opus sort + sorted/shuffled mxfp8 a-scale). We time ONLY each kernel's launch in
isolation (identical run_perftest settings), so the differing input prep does not
affect the comparison -- mirroring test_moe_2stage.py --kernel.

Baseline config per M comes from the tuned CSV's kernelName1 (parsed via
get_flydsl_kernel_params); v2 config per M comes from the #753 dispatcher's own
select_pipe_config / gemm1_use_nt (its production choice).

Usage:
  /opt/venv/bin/python bench_gemm1_v2_vs_baseline.py \
      --csv aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv \
      --model-dim 7168 --inter-dim 512 -E 384 -k 6

  /opt/venv/bin/python bench_gemm1_v2_vs_baseline.py --stage gemm2 \
      --csv aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv \
      --model-dim 7168 --inter-dim 512 -E 384 -k 6
"""

import argparse
import csv as _csv
import os

import torch

import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2
from aiter.ops.shuffle import shuffle_weight, shuffle_weight_a16w4, shuffle_scale_a16w4
from aiter.ops.quant import per_1x32_f8_scale_f8_quant, per_1x32_f4_quant
from aiter.utility import fp4_utils
from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort
from aiter.test_common import run_perftest
from aiter.ops.flydsl.moe_kernels import (
    flydsl_moe_stage1,
    flydsl_moe_stage2,
    get_flydsl_kernel_params,
    flydsl_kernel_name,
)
from aiter.fused_moe import is_v2_gemm2_kernel, parse_v2_gemm2_kernel
from aiter.ops.flydsl.kernels.moe_sorting_kernel import moe_sorting_flydsl
from aiter.ops.flydsl.kernels.mxmoe_dispatcher import (
    mxfp4_moe_gemm1,
    mxfp4_moe_gemm2,
    select_pipe_config,
    gemm1_use_nt,
    gemm2_use_nt,
)
from aiter.ops.opus import moe_stage2_a8w4_fused_adapter as _opus_a8w4

torch.set_default_device("cuda")

WARMUP, ITERS = 10, 50


# --- balanced routing / quant (mirror bench_stage1_a8w4.py) ------------------
def balanced_score(token, E, topk, dtype):
    score = torch.zeros((token, E), dtype=dtype)
    start = 0
    for t in range(token):
        idx = (torch.arange(start, start + topk)) % E
        score[t, idx] = 1.0
        start = (start + topk) % E
    return score


def quant_a_fp8(x):
    return per_1x32_f8_scale_f8_quant(
        x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )


def quant_a_fp4(x):
    return per_1x32_f4_quant(x, quant_dtype=dtypes.fp4x2, shuffle=False)


def quant_a(x, adtype):
    """Quantize stage1 activation per --adtype: fp8 (a8w4) or fp4 (a4w4)."""
    return quant_a_fp8(x) if adtype == "fp8" else quant_a_fp4(x)


def quant_w_fp4(w):
    return per_1x32_f4_quant(w, quant_dtype=dtypes.fp4x2, shuffle=False)


def _a_deq(a1_qt, a1_scale, token, model_dim, adtype):
    """Dequant the SAME quantized activation the kernels read (fp8 raw / fp4 e2m1)."""
    if adtype == "fp8":
        a_vals = a1_qt.float().view(token, model_dim // 32, 32)
    else:
        a_vals = fp4_utils.mxfp4_to_f32(a1_qt).view(token, model_dim // 32, 32)
    return (
        a_vals * fp4_utils.e8m0_to_f32(a1_scale).view(token, model_dim // 32, 1)
    ).view(token, model_dim).to(dtypes.bf16)


def _stage2_quant_sort(ref1, sorted_ids, num_valid_ids, token, topk, block_m,
                       sorted_weights, adtype):
    """Quantize/sort the stage2 activation in the same dtype the stage2 kernel reads."""
    quant_sort = (
        aiter.fused_dynamic_mxfp8_quant_moe_sort
        if adtype == "fp8" else
        aiter.fused_dynamic_mxfp4_quant_moe_sort
    )
    return quant_sort(
        ref1.contiguous().view(token * topk, ref1.shape[-1]),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        topk=topk,
        block_size=block_m,
        sorted_weights=sorted_weights,
    )


def _dequant_inter_sorted_quant(row, inter_dim, adtype):
    """Convert a v2 sorted intermediate row to fp32 for debug/cosine checks."""
    if adtype == "fp8":
        return row.view(torch.float8_e4m3fn).float()
    return fp4_utils.mxfp4_to_f32(row.view(dtypes.fp4x2)).view(inter_dim)


def _baseline_w1_shuffle(w1_qt, w1_scale, E, adtype):
    """Per-adtype baseline stage1 w1/w1_scale preshuffle.

    a8w4 (fp8 activation, gate/up interleave kernel): a16w4 gate-up-interleave
      shuffle -- shuffle_weight_a16w4 / shuffle_scale_a16w4 (matches
      test_moe_2stage.py -q7 and the flydsl_moe_stage1 docstring).
    a4w4 (fp4 activation): CK (16,16) weight shuffle + e8m0 scale shuffle
      (matches test_flydsl_moe_a4w4.py / test_moe_2stage.py preshuffle path).
    """
    if adtype == "fp8":
        return shuffle_weight_a16w4(w1_qt, 16, True), shuffle_scale_a16w4(w1_scale, E, True)
    return shuffle_weight(w1_qt, (16, 16)), e8m0_shuffle(w1_scale)


# --- v2 (#753) CK a16w4 layout helpers (ported verbatim from the PR test) ----
def _mxfp4_shuffle_weight_a16w4(x, gate_up, NLane=16, KPack=16):
    """CK a16w4 weight preshuffle (is_guinterleave path)."""
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)
    E, N, K_pk = x.shape
    if gate_up:
        N = N // 2
    KLane = 64 // NLane
    N0 = N // NLane
    K0 = K_pk // (KLane * KPack)
    if gate_up:
        x_ = x.view(E, 2, N0, NLane, K0, KLane, KPack).permute(0, 2, 1, 4, 5, 3, 6)
    else:
        x_ = x.view(E, N0, NLane, K0, KLane, KPack).permute(0, 1, 3, 4, 2, 5)
    return x_.contiguous().view(*x.shape).contiguous().view(x_type)


def _mxfp4_shuffle_scale_a16w4(src, E, gate_up):
    """CK a16w4 e8m0 scale preshuffle (is_guinterleave path)."""
    n_experts, k_ = src.shape
    n_ = n_experts // E
    K_Pack, N_Pack, N_Lane = 2, 2, 16
    K_Lane = 64 // N_Lane
    K1 = k_ // K_Pack // K_Lane
    N1 = n_ // N_Lane // N_Pack
    if gate_up:
        s = src.view(E, N_Pack, N1, N_Lane, K1, K_Pack, K_Lane).permute(0, 2, 4, 6, 3, 5, 1)
    else:
        s = src.view(E, N1, N_Pack, N_Lane, K1, K_Pack, K_Lane).permute(0, 1, 4, 6, 3, 5, 2)
    return s.contiguous().view(*src.shape).contiguous()


def _mxfp4_a_scale_sorted_shuffled(asc, sti, cumsum, max_sorted, H, BM=32, BK=256):
    """Torch reconstruction of moe_sort_scales: sort + CK-shuffle the e8m0 A-scale
    by sorted row, exactly as gemm1 consumes it (opus gather: sti & 0xFFFFFF)."""
    device = asc.device
    is_bm16 = BM < 32
    CHUNK_ROWS = 32 if is_bm16 else BM
    MN_PACK = 2
    K_PACK = BK // 128
    C_M1 = CHUNK_ROWS // (16 * MN_PACK)
    C_K1 = (H // 32) // (4 * K_PACK)
    K_LANE, N_LANE = 4, 16
    DWORDS_PER_CHUNK = C_M1 * C_K1 * K_LANE * N_LANE
    block_rows = 16 if is_bm16 else BM
    n_chunks = max_sorted // block_rows
    actual_sorted = int(cumsum[0].item())
    actual_n_chunks = (actual_sorted + block_rows - 1) // block_rows
    total_work = n_chunks * DWORDS_PER_CHUNK
    sti_c = sti & 0x00FFFFFF
    out = torch.zeros((total_work, 4), dtype=torch.uint8, device=device)
    wid = torch.arange(total_work, device=device)
    r = wid.clone()
    n_lane = r % N_LANE
    r //= N_LANE
    k_lane = r % K_LANE
    r //= K_LANE
    ku = r % C_K1
    r //= C_K1
    mi = r % C_M1
    r //= C_M1
    chunk = r
    valid_chunk = chunk < actual_n_chunks
    M = asc.shape[0]
    for ikxdl in range(K_PACK):
        for im_a in range(MN_PACK):
            if is_bm16 and im_a == 1:
                continue
            sorted_row = chunk * block_rows + (mi * MN_PACK + im_a) * 16 + n_lane
            rowok = (sorted_row < actual_sorted) & valid_chunk
            srow = torch.clamp(sorted_row, max=max_sorted - 1)
            stiv = sti_c[srow]
            tid = torch.where((stiv < M) & rowok, stiv, torch.zeros_like(stiv))
            k_idx = ku * K_PACK * 4 + ikxdl * 4 + k_lane
            byte = asc[tid.long(), k_idx.long()]
            out[:, ikxdl * MN_PACK + im_a] = torch.where(rowok, byte, torch.zeros_like(byte))
    return out.reshape(-1).contiguous()


def _u8v(t):
    return t.view(torch.uint8) if (t is not None and t.element_size() == 1 and t.dtype != torch.uint8) else t


# --- shared input build ------------------------------------------------------
def gen(token, model_dim, inter_dim, E, topk, block_m, adtype="fp8"):
    torch.manual_seed(0)
    inp = torch.randn((token, model_dim), dtype=dtypes.bf16) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtypes.bf16) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtypes.bf16) / 10
    score = balanced_score(token, E, topk, dtypes.bf16)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    w1_qt, w1_scale = quant_w_fp4(w1)      # fp4x2 weights + e8m0 scale
    w2_qt, w2_scale = quant_w_fp4(w2)      # fp4x2 weights + e8m0 scale
    a1_qt, a1_scale = quant_a(inp, adtype)  # fp8 (a8w4) or fp4 (a4w4) activations

    # torch reference stage1 (dequant the SAME quantized operands the kernels read)
    a_deq = _a_deq(a1_qt, a1_scale, token, model_dim, adtype)
    w1_deq = (
        fp4_utils.mxfp4_to_f32(w1_qt).view(E, inter_dim * 2, model_dim // 32, 32)
        * fp4_utils.e8m0_to_f32(w1_scale).view(E, inter_dim * 2, model_dim // 32, 1)
    ).view(E, inter_dim * 2, model_dim).to(dtypes.bf16)
    w2_deq = (
        fp4_utils.mxfp4_to_f32(w2_qt).view(E, model_dim, inter_dim // 32, 32)
        * fp4_utils.e8m0_to_f32(w2_scale).view(E, model_dim, inter_dim // 32, 1)
    ).view(E, model_dim, inter_dim).to(dtypes.bf16)
    ref1 = torch_moe_stage1(
        a_deq, w1_deq, w2_deq, topk_weights, topk_ids, dtype=dtypes.bf16,
        activation=ActivationType.Silu, quant_type=QuantType.No, doweight=False,
    )  # [token, topk, inter]
    ref2 = torch_moe_stage2(
        ref1, w1_deq, w2_deq, topk_weights, topk_ids, dtype=dtypes.bf16,
        quant_type=QuantType.No, doweight=True,
    )  # [token, hidden]

    # ---- baseline (aiter mixed_moe stage1) prep ----
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtypes.bf16, block_m
    )
    # w1/w1_scale preshuffle depends on the activation dtype (a8w4 vs a4w4);
    # see _baseline_w1_shuffle. w2 always uses the a16w4 down-proj layout.
    w1_qt_shuf, w1_scale_shuf = _baseline_w1_shuffle(w1_qt, w1_scale, E, adtype)
    base = dict(
        a1_qt=a1_qt,
        adtype=adtype,
        w1_qt_shuf=w1_qt_shuf,
        w1_scale_shuf=w1_scale_shuf,
        w2_qt_shuf=_mxfp4_shuffle_weight_a16w4(w2_qt, gate_up=False),
        w2_scale_shuf=_mxfp4_shuffle_scale_a16w4(w2_scale, E, gate_up=False),
        a1_scale_sort=moe_mxfp4_sort(
            a1_scale[:token, :].view(token, 1, -1), sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids, token_num=token, block_size=block_m,
        ),
        a2_qt=None,
        a2_scale=None,
        sorted_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        sorted_weights=sorted_weights, num_valid_ids=num_valid_ids,
    )
    a2_qt, a2_scale = _stage2_quant_sort(
        ref1, sorted_ids, num_valid_ids, token, topk, block_m, sorted_weights,
        adtype,
    )
    a2_cols = inter_dim if adtype == "fp8" else inter_dim // 2
    base["a2_qt"] = a2_qt.view(token, topk, a2_cols)
    base["a2_scale"] = a2_scale
    base["a2_dtype"] = adtype

    # ---- v2 (#753) prep: opus sort + CK a16w4 shuffles + sorted/shuffled a-scale ----
    H, INTER, NE = model_dim, inter_dim, E
    SBM = block_m  # sort unit == stage1 tile (BM_S1 chosen per M at bench time; SBM re-derived there)
    return dict(
        ref1=ref1, ref2=ref2, topk_ids=topk_ids, topk_weights=topk_weights,
        w1_qt=w1_qt, w1_scale=w1_scale, w2_qt=w2_qt, w2_scale=w2_scale,
        a1_qt=a1_qt, a1_scale=a1_scale,
        inp=inp, base=base, adtype=adtype, stage2_adtype=adtype,
    )


def build_v2_inputs(d, token, model_dim, inter_dim, E, topk, BM_S1):
    """Build v2 gemm1 inputs for the chosen stage1 tile BM_S1 (SBM==BM_S1)."""
    device = "cuda"
    H, INTER, NE, TOPK = model_dim, inter_dim, E, topk
    SBM = BM_S1
    w1u8 = _u8v(_mxfp4_shuffle_weight_a16w4(d["w1_qt"], gate_up=True))
    w1sc = _u8v(_mxfp4_shuffle_scale_a16w4(d["w1_scale"], NE, gate_up=True))
    w2u8 = _u8v(_mxfp4_shuffle_weight_a16w4(d["w2_qt"], gate_up=False))
    w2sc = _u8v(_mxfp4_shuffle_scale_a16w4(d["w2_scale"], NE, gate_up=False))

    topk_ids_i32 = d["topk_ids"].to(torch.int32)
    topk_w_f32 = d["topk_weights"].to(torch.float32)
    max_padded = token * TOPK + NE * SBM - TOPK
    max_sorted = ((max_padded + SBM - 1) // SBM) * SBM
    sti = torch.empty(max_sorted, dtype=torch.int32, device=device)
    swt = torch.empty(max_sorted, dtype=torch.float32, device=device)
    sei = torch.empty(max_sorted // SBM, dtype=torch.int32, device=device)
    nv = torch.empty(2, dtype=torch.int32, device=device)
    moe_buf = torch.empty((token, H), dtype=torch.bfloat16, device=device)
    moe_sorting_flydsl(topk_ids_i32, topk_w_f32, sti, swt, sei, nv, moe_buf, NE, SBM)
    torch.cuda.synchronize()
    cumsum = nv
    n = int(cumsum[0].item())

    # fp8 activation is 1 byte/elem; fp4x2 packs 2 elems/byte -> half-width payload.
    adtype = d.get("adtype", "fp8")
    a_row_bytes = H if adtype == "fp8" else H // 2
    aq = d["a1_qt"].view(torch.uint8).view(token, a_row_bytes).contiguous()
    asc = d["a1_scale"].view(torch.uint8).view(token, H // 32).contiguous()
    assh = _mxfp4_a_scale_sorted_shuffled(asc, sti, cumsum, max_sorted, H, BM=BM_S1)

    stage2_adtype = d.get("stage2_adtype", d.get("adtype", "fp8"))
    inter_cols = INTER if stage2_adtype == "fp8" else INTER // 2
    isq = torch.zeros((max_sorted, inter_cols), device=device, dtype=torch.uint8)
    isc_cols = INTER // 32
    isr = (((max_sorted * ((2 * INTER) // 64) * 4) + isc_cols - 1) // isc_cols + 31) // 32 * 32
    iss = torch.zeros((isr, isc_cols), device=device, dtype=torch.uint8)
    return dict(
        aq=aq, assh=assh, w1u8=w1u8, w1sc=w1sc, w2u8=w2u8, w2sc=w2sc,
        sei=sei, cumsum=cumsum, sti=sti, swt=swt,
        isq=isq, iss=iss, hidden=d["inp"], n=n, max_sorted=max_sorted,
    )


def time_baseline(d, token, topk, params):
    b = d["base"]
    adtype = b.get("adtype", "fp8")
    default_gate = "interleave" if adtype == "fp8" else "separated"
    def fn():
        return flydsl_moe_stage1(
            a=b["a1_qt"], w1=b["w1_qt_shuf"],
            sorted_token_ids=b["sorted_ids"],
            sorted_expert_ids=b["sorted_expert_ids"],
            num_valid_ids=b["num_valid_ids"], topk=topk,
            tile_m=params["tile_m"], tile_n=params["tile_n"], tile_k=params["tile_k"],
            a_dtype=adtype, b_dtype="fp4", out_dtype="bf16",
            w1_scale=b["w1_scale_shuf"], a1_scale=b["a1_scale_sort"],
            sorted_weights=None,
            k_batch=params.get("k_batch", 1),
            waves_per_eu=params.get("waves_per_eu", 3),
            gate_mode=params.get("gate_mode", default_gate),
            b_nt=params.get("b_nt", 2),
            k_wave=params.get("k_wave", 1),
        )
    out = fn()
    if isinstance(out, tuple):
        out = out[0]
    torch.cuda.synchronize()
    ref = d["ref1"]
    o = out.float().reshape(ref.shape)
    ok = torch.isclose(ref.float(), o, atol=1.0, rtol=0.05).float().mean().item() * 100
    _, us = run_perftest(fn, num_warmup=WARMUP, num_iters=ITERS)
    return us, ok


def _stage2_out_for_check(out, mode, token, topk, model_dim):
    if mode == "reduce":
        return out.view(token, topk, model_dim).sum(dim=1)
    return out


def _print_tensor(name, tensor):
    torch.cuda.synchronize()
    print(f"\n{name}:")
    print(tensor.detach().cpu())


def _print_close_stats(name, ref, got, atol=1.0, rtol=0.05):
    ref_f = ref.float()
    got_f = got.float()
    diff = (ref_f - got_f).abs()
    close = torch.isclose(ref_f, got_f, atol=atol, rtol=rtol).float().mean().item() * 100
    print(
        f"\n{name} diff stats: "
        f"close={close:.2f}% atol={atol} rtol={rtol} "
        f"max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g}"
    )


def time_baseline_gemm2(d, token, model_dim, topk, params, row=None, print_output=False):
    b = d["base"]
    row = row or {}
    kn2 = row.get("kernelName2", "")
    is_opus2 = _opus_a8w4.is_opus_a8w4_stage2_kernel(kn2)
    mode = params.get("mode", "atomic")
    stage2_adtype = b.get("a2_dtype", params.get("a_dtype", b.get("adtype", "fp8")))
    out_shape = (
        (token, model_dim)
        if is_opus2 else
        ((token, topk, model_dim) if mode == "reduce" else (token, model_dim))
    )
    out = torch.empty(out_shape, dtype=dtypes.bf16, device="cuda")

    if is_opus2:
        opus_values = _opus_a8w4.stage2_cfg_values(row, row.get("block_m", params["tile_m"]))

        def fn():
            return _opus_a8w4.opus_a8w4_stage2_wrapper(
                inter_states=b["a2_qt"],
                w1=None,
                w2=b["w2_qt_shuf"],
                sorted_token_ids=b["sorted_ids"],
                sorted_expert_ids=b["sorted_expert_ids"],
                num_valid_ids=b["num_valid_ids"],
                out=out,
                topk=topk,
                kernelName=kn2,
                w2_scale=b["w2_scale_shuf"].view(dtypes.fp8_e8m0),
                a2_scale=b["a2_scale"],
                sorted_weights=b["sorted_weights"],
                block_m=int(row.get("block_m", params["tile_m"])),
                **opus_values,
            )

        out.zero_()
        fn()
        torch.cuda.synchronize()
        ref = d["ref2"].float()
        got = out.float()
        if print_output:
            _print_close_stats("gemm2 baseline vs torch ref", ref, got)
            _print_tensor("torch ref gemm2 output", ref)
            _print_tensor("baseline gemm2 output", got)
        ok = torch.isclose(ref, got, atol=1.0, rtol=0.05).float().mean().item() * 100
        _, us = run_perftest(fn, num_warmup=WARMUP, num_iters=ITERS)
        return us, ok

    def fn():
        return flydsl_moe_stage2(
            inter_states=b["a2_qt"],
            w2=b["w2_qt_shuf"],
            sorted_token_ids=b["sorted_ids"],
            sorted_expert_ids=b["sorted_expert_ids"],
            num_valid_ids=b["num_valid_ids"],
            out=out,
            topk=topk,
            tile_m=params["tile_m"],
            tile_n=params["tile_n"],
            tile_k=params["tile_k"],
            a_dtype=stage2_adtype,
            b_dtype=params.get("b_dtype", "fp4"),
            out_dtype=params.get("out_dtype", "bf16"),
            mode=mode,
            w2_scale=b["w2_scale_shuf"].view(dtypes.fp8_e8m0),
            a2_scale=b["a2_scale"],
            sorted_weights=b["sorted_weights"],
            sort_block_m=params.get("sort_block_m", 0),
            persist=params.get("persist", None),
            waves_per_eu=params.get("waves_per_eu", None),
            b_nt=params.get("b_nt", 0),
            xcd_swizzle=params.get("xcd_swizzle", 0),
            return_per_slot=(mode == "reduce"),
        )

    if mode != "reduce":
        out.zero_()
    fn()
    torch.cuda.synchronize()
    ref = d["ref2"].float()
    got = _stage2_out_for_check(out, mode, token, topk, model_dim).float()
    if print_output:
        _print_close_stats("gemm2 baseline vs torch ref", ref, got)
        _print_tensor("torch ref gemm2 output", ref)
        _print_tensor("baseline gemm2 output", got)
    ok = torch.isclose(ref, got, atol=1.0, rtol=0.05).float().mean().item() * 100
    _, us = run_perftest(fn, num_warmup=WARMUP, num_iters=ITERS)
    return us, ok


def _v2_group_cosine(d, v, token, inter_dim, E, BM_S1, sample=64):
    """Per-32-group-normalized cosine of v2's dequant intermediate vs ref1.
    Group-normalizing removes the per-32 e8m0 scale, so we validate the gemm
    math + fp8 values without de-shuffling the (kernel-written) e8m0 scale."""
    isq, sti, sei = v["isq"], v["sti"], v["sei"]
    ref1 = d["ref1"]  # [token, topk, inter]
    topk_ids = d["topk_ids"]
    n = v["n"]
    stage2_adtype = d.get("stage2_adtype", d.get("adtype", "fp8"))
    tok_of = (sti[:n] & 0x00FFFFFF)
    real = (tok_of < token).nonzero(as_tuple=True)[0]
    if real.numel() == 0:
        return float("nan")
    idx = real[torch.linspace(0, real.numel() - 1, min(sample, real.numel())).long()]
    cos_all = []
    for r in idx.tolist():
        t = int(tok_of[r].item())
        e = int(sei[r // BM_S1].item())
        slot = (topk_ids[t] == e).nonzero(as_tuple=True)[0]
        if slot.numel() == 0:
            continue
        s = int(slot[0].item())
        vval = _dequant_inter_sorted_quant(isq[r], inter_dim, stage2_adtype)
        ref = ref1[t, s].float()                                  # [inter]
        vg = vval.view(inter_dim // 32, 32)
        rg = ref.view(inter_dim // 32, 32)
        vn = vg / (vg.norm(dim=1, keepdim=True) + 1e-8)
        rn = rg / (rg.norm(dim=1, keepdim=True) + 1e-8)
        cos_all.append((vn * rn).sum(dim=1).mean().item())
    return sum(cos_all) / len(cos_all) * 100 if cos_all else float("nan")


def time_v2(d, v, token, model_dim, inter_dim, E, topk, BM_S1, use_nt, BN, k_wave):
    adtype = d.get("adtype", "fp8")
    stage2_adtype = d.get("stage2_adtype", adtype)
    def fn():
        return mxfp4_moe_gemm1(
            a_quant=v["aq"], a_scale_sorted_shuffled=v["assh"],
            w1_u8=v["w1u8"], w1_scale_u8=v["w1sc"],
            sorted_expert_ids=v["sei"], cumsum_tensor=v["cumsum"],
            sorted_token_ids=v["sti"], inter_sorted_quant=v["isq"],
            inter_sorted_shuffled_scale=v["iss"], hidden_states=v["hidden"],
            n_tokens=token, NE=E, D_HIDDEN=model_dim, D_INTER=inter_dim, topk=topk,
            BM=BM_S1, use_nt=use_nt, interleave=True,
            a_dtype=adtype, out_dtype=stage2_adtype, act="silu", swiglu_limit=0.0,
            SBM=BM_S1, k_wave=k_wave, BN=BN, n_sorted_padded=v["n"],
            model_dim_pad=0, inter_dim_pad=0,
        )
    v["isq"].zero_()
    fn()
    torch.cuda.synchronize()
    ok = _v2_group_cosine(d, v, token, inter_dim, E, BM_S1)
    _, us = run_perftest(fn, num_warmup=WARMUP, num_iters=ITERS)
    return us, ok


def populate_baseline_v2_intermediate(d, v, token, topk, params, BM_S1):
    """Run baseline gemm1 (AITER_FMOE_V2 layout) into v2's sorted-row fp8 buffers.

    With AITER_FMOE_V2=1 the baseline flydsl gemm1 writes its fused-quant payload
    by sorted row and its e8m0 scale in the v2 layout, so the outputs can drive
    the v2 gemm2 directly -- letting --stage gemm2 compare v2 gemm2 on a BASELINE
    gemm1 producer instead of the v2 gemm1 producer.
    """
    a1_scale_sort = moe_mxfp4_sort(
        d["a1_scale"][:token, :].view(token, 1, -1),
        sorted_ids=v["sti"],
        num_valid_ids=v["cumsum"],
        token_num=token,
        block_size=BM_S1,
    )
    adtype = d["base"].get("adtype", "fp8")
    stage2_adtype = d["base"].get("a2_dtype", adtype)
    default_gate = "interleave" if adtype == "fp8" else "separated"
    gate_mode = params.get("gate_mode", default_gate)
    if os.environ.get("AITER_LOG_MORE", "0") == "1":
        _kn1 = flydsl_kernel_name(
            1, adtype, "fp4", stage2_adtype,
            params["tile_m"], params["tile_n"], params["tile_k"],
        )
        _kw = params.get("k_wave", 1)
        _bnt = params.get("b_nt", 2)
        print(
            f"[gemm1 producer] baseline flydsl_moe_stage1 (v2 layout)  "
            f"kernel={_kn1}  a_dtype={adtype} b_dtype=fp4 out={stage2_adtype}  "
            f"gate={gate_mode} k_wave={_kw} b_nt={_bnt}  "
            f"tile=({params['tile_m']}x{params['tile_n']}x{params['tile_k']})"
        )
    out, scale = flydsl_moe_stage1(
        a=d["a1_qt"],
        w1=d["base"]["w1_qt_shuf"],
        out=v["isq"],
        sorted_token_ids=v["sti"],
        sorted_expert_ids=v["sei"],
        num_valid_ids=v["cumsum"],
        topk=topk,
        tile_m=params["tile_m"],
        tile_n=params["tile_n"],
        tile_k=params["tile_k"],
        a_dtype=adtype,
        b_dtype="fp4",
        out_dtype=stage2_adtype,
        w1_scale=d["base"]["w1_scale_shuf"],
        a1_scale=a1_scale_sort,
        sorted_weights=None,
        k_batch=params.get("k_batch", 1),
        waves_per_eu=params.get("waves_per_eu", 3),
        gate_mode=gate_mode,
        b_nt=params.get("b_nt", 2),
        k_wave=params.get("k_wave", 1),
    )
    v["isq"] = out.view(torch.uint8).view_as(v["isq"])
    v["iss"] = scale.view(torch.uint8)
    torch.cuda.synchronize()


def print_gemm1_v2_layout_compare(d, v, token, model_dim, inter_dim, E, topk,
                                  BM_S1, use_nt, BN, k_wave, base_gemm1_params):
    if os.environ.get("AITER_FMOE_V2", "0") != "1":
        print("\nwarning: gemm1 v2-layout compare expects AITER_FMOE_V2=1")

    populate_baseline_v2_intermediate(d, v, token, topk, base_gemm1_params, BM_S1)
    baseline_isq = v["isq"].clone()

    v["isq"].zero_()
    v["iss"].zero_()
    mxfp4_moe_gemm1(
        a_quant=v["aq"], a_scale_sorted_shuffled=v["assh"],
        w1_u8=v["w1u8"], w1_scale_u8=v["w1sc"],
        sorted_expert_ids=v["sei"], cumsum_tensor=v["cumsum"],
        sorted_token_ids=v["sti"], inter_sorted_quant=v["isq"],
        inter_sorted_shuffled_scale=v["iss"], hidden_states=v["hidden"],
        n_tokens=token, NE=E, D_HIDDEN=model_dim, D_INTER=inter_dim, topk=topk,
        BM=BM_S1, use_nt=use_nt, interleave=True,
        a_dtype=d.get("adtype", "fp8"), out_dtype=d.get("stage2_adtype", d.get("adtype", "fp8")),
        act="silu", swiglu_limit=0.0,
        SBM=BM_S1, k_wave=k_wave, BN=BN, n_sorted_padded=v["n"],
        model_dim_pad=0, inter_dim_pad=0,
    )
    torch.cuda.synchronize()

    stage2_adtype = d.get("stage2_adtype", d.get("adtype", "fp8"))
    baseline = torch.stack(
        [_dequant_inter_sorted_quant(row, inter_dim, stage2_adtype) for row in baseline_isq]
    )
    v2 = torch.stack(
        [_dequant_inter_sorted_quant(row, inter_dim, stage2_adtype) for row in v["isq"]]
    )
    n_valid = v["n"]
    _print_close_stats(
        "gemm1 baseline-v2-layout vs v2 full isq",
        baseline, v2, atol=0.0, rtol=0.0,
    )
    _print_close_stats(
        "gemm1 baseline-v2-layout vs v2 valid isq",
        baseline[:n_valid], v2[:n_valid], atol=0.0, rtol=0.0,
    )
    _print_tensor(f"baseline gemm1 v2-layout {stage2_adtype} output", baseline)
    _print_tensor(f"v2 gemm1 sorted {stage2_adtype} output", v2)


def time_v2_gemm2(d, v, token, model_dim, inter_dim, E, topk, BM_S1, BM_S2, use_nt,
                  epilog, persist, BN, k_wave, base_gemm1_params=None,
                  print_output=False):
    stage2_adtype = d.get("stage2_adtype", d.get("adtype", "fp8"))
    if os.environ.get("AITER_FMOE_V2", "0") == "1":
        if base_gemm1_params is None:
            raise ValueError("base_gemm1_params is required when AITER_FMOE_V2=1")
        populate_baseline_v2_intermediate(d, v, token, topk, base_gemm1_params, BM_S1)
    else:
        # Populate the sorted fp8 intermediate exactly as v2 production gemm2 consumes it.
        mxfp4_moe_gemm1(
            a_quant=v["aq"], a_scale_sorted_shuffled=v["assh"],
            w1_u8=v["w1u8"], w1_scale_u8=v["w1sc"],
            sorted_expert_ids=v["sei"], cumsum_tensor=v["cumsum"],
            sorted_token_ids=v["sti"], inter_sorted_quant=v["isq"],
            inter_sorted_shuffled_scale=v["iss"], hidden_states=v["hidden"],
            n_tokens=token, NE=E, D_HIDDEN=model_dim, D_INTER=inter_dim, topk=topk,
            BM=BM_S1, use_nt=gemm1_use_nt(E, topk, token, BM_S1), interleave=True,
            a_dtype=d.get("adtype", "fp8"), out_dtype=stage2_adtype, act="silu", swiglu_limit=0.0,
            SBM=BM_S1, k_wave=k_wave, BN=BN, n_sorted_padded=v["n"],
            model_dim_pad=0, inter_dim_pad=0,
        )
        torch.cuda.synchronize()

    out_shape = (token, topk, model_dim) if epilog == "reduce" else (token, model_dim)
    out = torch.empty(out_shape, dtype=dtypes.bf16, device="cuda")

    def fn():
        return mxfp4_moe_gemm2(
            inter_sorted_quant=v["isq"],
            inter_sorted_shuffled_scale=v["iss"],
            w2_u8=v["w2u8"],
            w2_scale_u8=v["w2sc"],
            sorted_expert_ids=v["sei"],
            cumsum_tensor=v["cumsum"],
            sorted_token_ids=v["sti"],
            sorted_weights=v["swt"],
            out=out,
            M_logical=token,
            max_sorted=v["max_sorted"],
            NE=E,
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            topk=topk,
            BM=BM_S2,
            use_nt=use_nt,
            a_dtype=stage2_adtype,
            epilog=epilog,
            SBM=BM_S1,
            persist=persist,
            n_sorted_padded=v["n"],
            model_dim_pad=0,
            inter_dim_pad=0,
        )

    if epilog != "reduce":
        out.zero_()
    fn()
    torch.cuda.synchronize()
    ref = d["ref2"].float()
    got = _stage2_out_for_check(out, epilog, token, topk, model_dim).float()
    if print_output:
        _print_close_stats("gemm2 v2 vs torch ref", ref, got)
        _print_tensor("v2 gemm2 output", got)
    ok = torch.isclose(ref, got, atol=1.0, rtol=0.05).float().mean().item() * 100
    _, us = run_perftest(fn, num_warmup=WARMUP, num_iters=ITERS)
    return us, ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("gemm1", "gemm2"), default="gemm1")
    p.add_argument("--adtype", choices=("fp8", "fp4"), default="fp8",
                   help="stage1 activation dtype: fp8 (a8w4, default) or fp4 (a4w4). "
                        "Selects the matching a-quant + w1 preshuffle + kernel a_dtype.")
    p.add_argument("--csv", default="aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv")
    p.add_argument("--model-dim", type=int, default=7168)
    p.add_argument("--inter-dim", type=int, default=512)
    p.add_argument("-E", "--experts", type=int, default=384)
    p.add_argument("-k", "--topk", type=int, default=6)
    p.add_argument("--tokens", type=int, nargs="+", default=None,
                   help="override; default = all tuned M for the shape in the CSV")
    p.add_argument("--same-tile", action="store_true",
                   help="force v2 onto the baseline's tile config (BM=tile_m, k_wave, "
                        "use_nt=b_nt==2, BN=64 if k_wave>1 else 256) instead of v2's "
                        "own select_pipe_config -- isolates kernel-vs-kernel.")
    p.add_argument("--print-output", action="store_true",
                   help="print output tensors for the selected stage")
    p.add_argument("--print-baseline-output", action="store_true",
                   help="print the baseline gemm2 output tensor")
    args = p.parse_args()

    # gather tuned (token, block_m, kernelName1) for the requested shape
    rows = []
    with open(args.csv, newline="") as f:
        for r in _csv.DictReader(f):
            if (int(r["model_dim"]) == args.model_dim and int(r["inter_dim"]) == args.inter_dim
                    and int(r["expert"]) == args.experts and int(r["topk"]) == args.topk):
                rows.append(r)
    # dedup by token (keep first occurrence per tuned M)
    _seen = set()
    _uniq = []
    for r in sorted(rows, key=lambda r: int(r["token"])):
        t = int(r["token"])
        if t in _seen:
            continue
        _seen.add(t)
        _uniq.append(r)
    rows = _uniq
    if args.tokens is not None:
        want = set(args.tokens)
        rows = [r for r in rows if int(r["token"]) in want]
    if not rows:
        print("no matching CSV rows for shape")
        return

    _qtag = "a8w4" if args.adtype == "fp8" else "a4w4"
    print(f"dsv4 {_qtag} {args.stage}  md={args.model_dim} id={args.inter_dim} "
          f"E={args.experts} topk={args.topk}  (BALANCED, launch-only)")
    if args.stage == "gemm1":
        print("baseline = flydsl mixed_moe stage1 (CSV kernelName1) | "
              "v2 = FlyDSL#753 mxfp4_moe_gemm1\n")
    else:
        print("baseline = CSV kernelName2 stage2 (flydsl/opus) | "
              "v2 = FlyDSL#753 mxfp4_moe_gemm2\n")
    hdr = f"{'M':>7} {'blk':>4} | {'base us':>9} {'ok%':>5} | {'v2 us':>9} {'cos%':>5} {'v2 cfg':>18} | {'delta%':>7}"
    print(hdr)
    print("-" * len(hdr))

    results = []
    for r in rows:
        token = int(r["token"])
        bm_csv = int(r["block_m"])
        kn1 = r["kernelName1"]
        params1 = get_flydsl_kernel_params(kn1) or {}
        params1.setdefault("tile_m", bm_csv)
        params1.setdefault("tile_n", 64)
        params1.setdefault("tile_k", 256)
        sort_bm = params1["tile_m"]

        kn2 = r.get("kernelName2", "")
        # A gemm2 row can pin itself to the v2 kernel via a mxfp4_moe2_* name in
        # kernelName2. When it does (and we're benching gemm2), the v2 config is
        # read from the name and the baseline flydsl/opus side is skipped for
        # that row -- the CSV row IS the "use v2 for moe2" decision.
        v2_g2 = parse_v2_gemm2_kernel(kn2) if args.stage == "gemm2" else None
        is_opus2 = _opus_a8w4.is_opus_a8w4_stage2_kernel(kn2)
        params2 = get_flydsl_kernel_params(kn2) or {}
        params2.setdefault("tile_m", bm_csv)
        params2.setdefault("tile_n", 256)
        params2.setdefault("tile_k", 256)
        params2.setdefault("mode", "atomic")
        if is_opus2:
            opus_values = _opus_a8w4.stage2_cfg_values(r, bm_csv)
            params2["tile_m"] = int(opus_values["stage2_block_m"])
            params2["mode"] = "opus-route" if bool(opus_values["route_out"]) else "opus-atomic"

        d = gen(token, args.model_dim, args.inter_dim, args.experts, args.topk, sort_bm,
                adtype=args.adtype)
        if v2_g2 is not None:
            # v2-pinned gemm2 row: the CSV's chosen gemm2 IS the v2 kernel, so
            # the baseline side is that same v2 kernel. Time it below (after the
            # v2 config + inputs are built) so both columns measure it and match.
            base_us, base_ok = float("nan"), -1
        else:
            try:
                if args.stage == "gemm1":
                    base_us, base_ok = time_baseline(d, token, args.topk, params1)
                else:
                    base_us, base_ok = time_baseline_gemm2(
                        d, token, args.model_dim, args.topk, params2, row=r,
                        print_output=args.print_output or args.print_baseline_output
                    )
            except Exception as e:
                base_us, base_ok = float("nan"), -1
                print(f"{token:>7} {bm_csv:>4} | baseline FAIL: {str(e)[:70]}")

        if args.same_tile:
            # match the baseline tuned tile: BM, k_wave, nt; map baseline tile_n -> v2 BN.
            if args.stage == "gemm1":
                BM_S1 = params1["tile_m"]
                BM_v2 = BM_S1
                epilog = "atomic"
                persist = False
                KW_v2 = params1.get("k_wave", 1)
                use_nt = params1.get("b_nt", 2) == 2
                tn = params1.get("tile_n", 256)
            else:
                BM_S1 = sort_bm
                BM_v2 = min(params2["tile_m"], 64)
                epilog = params2.get("mode", "atomic")
                persist = bool(params2.get("persist", False))
                KW_v2 = params1.get("k_wave", 1)
                use_nt = params2.get("b_nt", 0) == 2
                tn = params1.get("tile_n", 256)
            # v2 BN in {64,256}: tile_n<=64 -> BN64, tile_n>=128 -> BN256 (v2 has no BN128,
            # so tile_n=128 is only approximated). BN64 structurally needs k_wave>=2.
            if tn <= 64:
                BN_v2 = 64
            elif tn <= 128:
                BN_v2 = 128
            else:
                BN_v2 = 256
            if BN_v2 == 64 and KW_v2 < 2:
                KW_v2 = 2
        else:
            # v2 dispatcher's own config for this M
            BM_v2, epilog, BM_S1, persist, BN_v2, KW_v2 = select_pipe_config(
                args.model_dim, args.inter_dim, args.experts, args.topk, token
            )
            if args.stage == "gemm2" and os.environ.get("AITER_FMOE_V2", "0") == "1":
                # The producer is baseline gemm1, so use its sort padding unit.
                BM_S1 = sort_bm
            if args.stage == "gemm2" and BM_S1 % BM_v2 != 0:
                # gemm2 consumes an SBM-strided sorted stream and requires SBM
                # to be a multiple of its BM. Tiny-M gemm1 may choose BM16, so
                # promote the standalone gemm2 bench input stream to BM32.
                BM_S1 = BM_v2
            if args.stage == "gemm1":
                use_nt = gemm1_use_nt(args.experts, args.topk, token, BM_S1)
            else:
                use_nt = gemm2_use_nt(args.experts, args.topk, token, BM_v2)

        if v2_g2 is not None:
            # Override the v2 gemm2 config with the CSV-pinned kernel name.
            # BN_v2/KW_v2 stay from select_pipe_config -- they only shape the
            # gemm1 producer that fills the intermediate, not the timed gemm2.
            BM_v2 = v2_g2["tile_m"]
            epilog = v2_g2["epilog"]
            persist = v2_g2["persist"]
            use_nt = v2_g2["use_nt"]
            BM_S1 = v2_g2["sort_block_m"] or BM_S1
            if os.environ.get("AITER_FMOE_V2", "0") == "1":
                # Producer is baseline gemm1; align sort unit to its tile.
                BM_S1 = sort_bm
            if BM_S1 % BM_v2 != 0:
                BM_S1 = BM_v2
        try:
            v = build_v2_inputs(d, token, args.model_dim, args.inter_dim,
                                args.experts, args.topk, BM_S1)
            if args.stage == "gemm1":
                if args.print_output:
                    print_gemm1_v2_layout_compare(
                        d, v, token, args.model_dim, args.inter_dim,
                        args.experts, args.topk, BM_S1, use_nt, BN_v2, KW_v2,
                        params1,
                    )
                v2_us, v2_nz = time_v2(
                    d, v, token, args.model_dim, args.inter_dim,
                    args.experts, args.topk, BM_S1, use_nt, BN_v2, KW_v2
                )
            else:
                v2_us, v2_nz = time_v2_gemm2(
                    d, v, token, args.model_dim, args.inter_dim,
                    args.experts, args.topk, BM_S1, BM_v2, use_nt, epilog, persist,
                    BN_v2, KW_v2, base_gemm1_params=params1,
                    print_output=args.print_output
                )
                if v2_g2 is not None:
                    # The CSV's chosen gemm2 is this same v2 kernel, so the
                    # baseline column re-times it independently -- both sides
                    # measure the identical kernel and should match.
                    base_us, base_ok = time_v2_gemm2(
                        d, v, token, args.model_dim, args.inter_dim,
                        args.experts, args.topk, BM_S1, BM_v2, use_nt, epilog,
                        persist, BN_v2, KW_v2, base_gemm1_params=params1,
                    )
        except Exception as e:
            v2_us, v2_nz = float("nan"), -1
            print(f"{token:>7} {bm_csv:>4} | v2 FAIL: {str(e)[:80]}")

        delta = (v2_us - base_us) / base_us * 100 if base_us == base_us and v2_us == v2_us else float("nan")
        if args.stage == "gemm1":
            cfg = f"bm{BM_S1}bn{BN_v2}kw{KW_v2}{'nt' if use_nt else ''}"
        else:
            cfg = f"bm{BM_v2}{epilog[0]}sbm{BM_S1}{'p' if persist else ''}{'nt' if use_nt else ''}"
        print(f"{token:>7} {bm_csv:>4} | {base_us:9.3f} {base_ok:5.1f} | "
              f"{v2_us:9.3f} {v2_nz:5.1f} {cfg:>18} | {delta:+7.1f}")
        results.append((token, base_us, v2_us, delta))

    print(f"\nsummary {args.stage} (v2 delta% vs baseline; negative = v2 faster):")
    for token, b, vv, dl in results:
        print(f"  M={token:>7}: base {b:8.3f}us  v2 {vv:8.3f}us  {dl:+6.1f}%")


if __name__ == "__main__":
    main()
