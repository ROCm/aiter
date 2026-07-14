"""Shared v2 gemm1/gemm2 data-prep helpers for dsv4 a8w4 / a4w4 (fp8/fp4 a, fp4 w).

Extracted verbatim from bench_gemm12_v2_vs_baseline.py so both the launch-comparison
bench and the fmoe tuner (csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py) can build
the same v2 gemm inputs:

  baseline = current vendored FlyDSL mixed_moe stage1 (aiter.ops.flydsl.flydsl_moe_stage1)
  v2       = FlyDSL#753 "mxmoe v2" gemm1 (aiter.ops.flydsl.kernels.mxmoe_dispatcher.mxfp4_moe_gemm1)

These functions cover: balanced routing, per-adtype a/w quant, torch reference
stage1/stage2, baseline preshuffle, v2 CK a16w4 weight/scale shuffles, sorted/shuffled
A-scale reconstruction, the shared `gen` input builder, `build_v2_inputs`, and the
baseline-into-v2-layout intermediate producer.
"""

import os

import torch

import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2
from aiter.ops.shuffle import shuffle_weight, shuffle_weight_a16w4, shuffle_scale_a16w4
from aiter.ops.quant import per_1x32_f8_scale_f8_quant, per_1x32_f4_quant
from aiter.utility import fp4_utils
from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort
from aiter.ops.flydsl.moe_kernels import (
    flydsl_moe_stage1,
    flydsl_kernel_name,
)
from aiter.ops.flydsl.kernels.moe_sorting_kernel import moe_sorting_flydsl


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


def populate_baseline_v2_intermediate(d, v, token, topk, params, BM_S1):
    """Run baseline gemm1 into v2's sorted-row fp8/fp4 buffers.

    flydslv2_* GEMM2 consumes the v2 sorted-row payload. This path lets
    --stage gemm2 compare v2 GEMM2 on a baseline GEMM1 producer.
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
        v2_output_layout=True,
    )
    v["isq"] = out.view(torch.uint8).view_as(v["isq"])
    v["iss"] = scale.view(torch.uint8)
    torch.cuda.synchronize()


def v2_stage1_sorted_ref(
    ref1, topk_ids, sti, sei, n, *, token, inter_dim, bm_s1, max_sorted
):
    """把 torch ref1 [token,topk,inter] gather 成 v2 sorted-stream 顺序 [max_sorted,inter]
    bf16 参考; 无效/pad 行为 0。gather 规则同 _v2_group_cosine:
    tok = sti & 0xFFFFFF, e = sei[row//bm_s1], slot = where(topk_ids[tok]==e)."""
    out = torch.zeros((max_sorted, inter_dim), dtype=torch.bfloat16,
                      device=ref1.device)
    tok_of = (sti[:n] & 0x00FFFFFF)
    rows = (tok_of < token).nonzero(as_tuple=True)[0]
    for r in rows.tolist():
        t = int(tok_of[r].item())
        e = int(sei[r // bm_s1].item())
        slot = (topk_ids[t] == e).nonzero(as_tuple=True)[0]
        if slot.numel() == 0:
            continue
        out[r] = ref1[t, int(slot[0].item())].to(torch.bfloat16)
    return out


def v2_stage1_dequant_cosine_err(
    ref, res, msg="", printLog=True, *, inter_dim, adtype
):
    """compare_fn: res=原始 isq [max_sorted,inter_cols] uint8; ref=v2_stage1_sorted_ref
    产出的 [max_sorted,inter] bf16。对每有效行 (ref 非全 0) 做 per-32-group cosine,
    返回 err_ratio = 1 - mean_cos (与 cosine_diff_compare 语义一致)。"""
    valid = (ref.abs().sum(dim=1) > 0).nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        return 1.0
    cos_all = []
    for r in valid.tolist():
        vval = _dequant_inter_sorted_quant(res[r], inter_dim, adtype)
        rg = ref[r].float().view(inter_dim // 32, 32)
        vg = vval.view(inter_dim // 32, 32)
        rn = rg / (rg.norm(dim=1, keepdim=True) + 1e-8)
        vn = vg / (vg.norm(dim=1, keepdim=True) + 1e-8)
        cos_all.append((vn * rn).sum(dim=1).mean().item())
    cos = sum(cos_all) / len(cos_all)
    err = 1.0 - cos
    if printLog:
        print(f"{msg}[v2_stage1 cos={cos:.4f} err={err:.4f}]")
    return max(err, 0.0)
