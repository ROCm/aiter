# SPDX-License-Identifier: MIT
"""Replay real ATOM decode-MLA dumps and compare num_kv_splits=1 vs =2.

Consumes the .pt files written by atom.utils.mla_decode_dump.dump_decode_mla
(kv_layout="seg"), and for each (layer, step):
  * runs aiter.mla.mla_decode_fwd twice on the SAME inputs, once with
    num_kv_splits=1 and once with num_kv_splits=2;
  * builds an fp32 ground-truth MLA attention from the same (dequantized) inputs;
  * reports cosine / max-rel-error of each split count vs the fp32 GT, and of
    split=2 vs split=1.

If split=2 is consistently worse than split=1 vs the fp32 GT, the regression is
in the aiter kernel / triton stage2 reduce on real-data distributions. If both
are equally close to GT (and split2~split1), the kernel is fine and the
end-to-end gsm8k drop comes from elsewhere in ATOM.

Usage:
    python op_tests/replay_mla_split_compare.py ~/mla_decode_dump
    python op_tests/replay_mla_split_compare.py ~/mla_decode_dump/mla_decode_layer000_step0_rank0.pt
"""

import os
import sys
import glob
import torch
import aiter


def _str2dtype(s):
    return {
        "torch.float8_e4m3fn": torch.float8_e4m3fn,
        "torch.float8_e4m3fnuz": getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.uint8": torch.uint8,
    }.get(s, torch.float8_e4m3fn)


def _unpack_seg_pages(seg_kv_compact, page_size, kv_lora, pe_dim):
    """seg_kv_compact: [U, page_size, 1, kv_lora+pe] seg-packed as
    [page_size*kv_lora nope][page_size*pe pe] per page. Returns fp32 nope/pe."""
    U = seg_kv_compact.shape[0]
    flat = seg_kv_compact.reshape(U, -1).to(torch.float32)
    nope = flat[:, : page_size * kv_lora].reshape(U, page_size, kv_lora)
    pe = flat[:, page_size * kv_lora :].reshape(U, page_size, pe_dim)
    return nope, pe


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.reshape(-1).float(), b.reshape(-1).float(), dim=0
    ).item()


def _max_rel(a, b):
    a = a.float()
    b = b.float()
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)


def _run_split(b, dev, num_kv_splits, force=True):
    """Run mla_decode_fwd. If force=True, pass an explicit uniform
    num_kv_splits_indptr so we bypass get_meta_param's fp8 clamp and actually
    exercise the requested split count (otherwise short ctx collapses to 1)."""
    q = b["q"].to(dev)
    seg_kv = b["kv_compact"].to(dev).contiguous()
    o = torch.zeros_like(b["o_server"]).to(dev)
    qo_indptr = b["qo_indptr"].to(dev).to(torch.int32)
    kv_indptr = b["kv_indptr"].to(dev).to(torch.int32)
    kv_indices = b["kv_indices"].to(dev).to(torch.int32)
    kv_last = b["kv_last_page_lens"].to(dev).to(torch.int32)
    q_scale_t = b["q_scale"].to(dev) if b["q_scale"] is not None else None
    kv_scale_t = b["kv_scale"].to(dev) if b["kv_scale"] is not None else None
    bs = kv_indptr.numel() - 1
    indptr = None
    if force:
        indptr = (
            torch.arange(0, (bs + 1) * num_kv_splits, num_kv_splits, dtype=torch.int32)
            .to(dev)
        )
    aiter.mla.mla_decode_fwd(
        q,
        seg_kv,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last,
        b["max_q_len"],
        page_size=b["page_size"],
        nhead_kv=1,
        sm_scale=b["sm_scale"],
        num_kv_splits=num_kv_splits,
        num_kv_splits_indptr=indptr,
        q_scale=q_scale_t,
        kv_scale=kv_scale_t,
    )
    return o


def _stage1_partials(b, dev, num_kv_splits):
    """Run ONLY stage1 (asm) and return per-split (logits, attn_lse) so we can
    check whether NaN already exists in the partials BEFORE the triton stage2
    reduce. Mirrors the host-side buffer allocation in mla.mla_decode_fwd.

    The buffers are pre-seeded with a finite *empty-split* sentinel
    (data=0, lse=-1e20); any non-finite value afterwards was WRITTEN by stage1,
    which decisively places the NaN in the asm kernel (not in stage2 / not an
    uninitialized read). Returns (logits, attn_lse) or (None, None) on failure.
    """
    try:
        q = b["q"].to(dev)
        seg_kv = b["kv_compact"].to(dev).contiguous()
        o = torch.zeros_like(b["o_server"]).to(dev)
        qo_indptr = b["qo_indptr"].to(dev).to(torch.int32)
        kv_indptr = b["kv_indptr"].to(dev).to(torch.int32)
        kv_indices = b["kv_indices"].to(dev).to(torch.int32)
        kv_last = b["kv_last_page_lens"].to(dev).to(torch.int32)
        q_scale_t = b["q_scale"].to(dev) if b["q_scale"] is not None else None
        kv_scale_t = b["kv_scale"].to(dev) if b["kv_scale"] is not None else None
        bs = kv_indptr.numel() - 1
        indptr = torch.arange(
            0, (bs + 1) * num_kv_splits, num_kv_splits, dtype=torch.int32
        ).to(dev)
        total_s, nhead, v_head_dim = o.shape
        logits = torch.empty(
            (total_s, num_kv_splits, nhead, v_head_dim),
            dtype=torch.float32,
            device=dev,
        )
        attn_lse = torch.empty(
            (total_s, num_kv_splits, nhead, 1), dtype=torch.float32, device=dev
        )
        # finite empty-split sentinel: any NaN/Inf left after stage1 was written
        # by the asm kernel itself.
        logits.fill_(0.0)
        attn_lse.fill_(-1.0e20)
        aiter.mla_decode_stage1_asm_fwd(
            q,
            seg_kv,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last,
            indptr,
            None,
            None,
            None,
            b["max_q_len"],
            b["page_size"],
            1,  # nhead_kv
            b["sm_scale"],
            logits,
            attn_lse,
            o,
            None,  # final_lse
            q_scale_t,
            kv_scale_t,
        )
        return logits, attn_lse
    except Exception as e:  # noqa: BLE001 - diagnostic, never fatal
        print(f"    [stage1-probe] failed: {type(e).__name__}: {e}")
        return None, None


def _kernel_geom(b):
    """Reconstruct the per-batch page geometry exactly as the asm kernel sees it
    (SUB_KV = page_size pages, strided over `passes` splits), so NaN batches can
    be correlated with full_pages / tail / tail_owner / per-split loop_cnt.

    Mirrors sp3:
      full_pages = n_pages - (tail_len>0 ? 1 : 0)         (sp3 6280-6288)
      tail_len   = last_page_len   (0 if the last page is exactly full)
      tail_owner = full_pages % passes                     (sp3 6303-6313)
      loop_cnt(z)= 0 if full_pages<=z else (full_pages-1-z)//passes + 1  (sp3 6294-6301)
    """
    page_size = int(b["page_size"])
    passes = 2
    kvi = b["kv_indptr"].to(torch.int64)
    klp = b["kv_last_page_lens"].to(torch.int64)
    bs = kvi.numel() - 1
    rows = []
    for i in range(bs):
        n_pages = int((kvi[i + 1] - kvi[i]).item())
        last = int(klp[i].item()) if i < klp.numel() else 0
        tail_len = last if (0 < last < page_size) else 0
        full_pages = n_pages - (1 if tail_len > 0 else 0)
        tail_owner = (full_pages % passes) if full_pages > 0 else -1
        loop = [
            (0 if full_pages <= z else (full_pages - 1 - z) // passes + 1)
            for z in range(passes)
        ]
        rows.append(
            {
                "i": i,
                "n_pages": n_pages,
                "last": last,
                "tail_len": tail_len,
                "full_pages": full_pages,
                "tail_owner": tail_owner,
                "loop0": loop[0],
                "loop1": loop[1],
            }
        )
    return rows


def _stage1_per_batch_diag(b, logits, attn_lse):
    """Per-(batch) finiteness of stage1 partials, correlated with kernel page
    geometry. Returns (nan_rows, ok_rows, summary_str)."""
    qoi = b["qo_indptr"].to(torch.int64)
    bs = qoi.numel() - 1
    geom = _kernel_geom(b)
    lg_fin = torch.isfinite(logits).reshape(logits.shape[0], -1)
    lse_fin = torch.isfinite(attn_lse).reshape(attn_lse.shape[0], -1)
    nan_rows, ok_rows = [], []
    for i in range(bs):
        r0, r1 = int(qoi[i]), int(qoi[i + 1])
        if r1 <= r0:
            continue
        fin = bool(lg_fin[r0:r1].all()) and bool(lse_fin[r0:r1].all())
        (ok_rows if fin else nan_rows).append(geom[i])

    def _set(rows, key):
        return sorted(set(r[key] for r in rows))

    summary = ""
    if nan_rows:
        summary = (
            f"NaN batches={len(nan_rows)}/{bs} | "
            f"full_pages(nan)={_set(nan_rows,'full_pages')} "
            f"tail_len(nan)={_set(nan_rows,'tail_len')} "
            f"tail_owner(nan)={_set(nan_rows,'tail_owner')} "
            f"loop0(nan)={_set(nan_rows,'loop0')} loop1(nan)={_set(nan_rows,'loop1')}\n"
            f"      OK : full_pages={_set(ok_rows,'full_pages')} "
            f"tail_len={_set(ok_rows,'tail_len')} "
            f"tail_owner={_set(ok_rows,'tail_owner')} "
            f"loop0={_set(ok_rows,'loop0')} loop1={_set(ok_rows,'loop1')}"
        )
    return nan_rows, ok_rows, summary


def _fp32_ref(b, dev):
    """fp32 ground-truth MLA decode attention (decode_qlen=1)."""
    page_size = b["page_size"]
    kv_lora = b["kv_lora_rank"]
    pe_dim = b["qk_rope_head_dim"]
    sm_scale = b["sm_scale"]
    q_scale = b["q_scale"].item() if b["q_scale"] is not None else 1.0
    kv_scale = b["kv_scale"].item() if b["kv_scale"] is not None else 1.0
    q_f32 = b["q"].to(torch.float32)  # [B, nhead, kv_lora+pe]
    nope, pe = _unpack_seg_pages(b["kv_compact"], page_size, kv_lora, pe_dim)
    kv_indptr = b["kv_indptr"].to(torch.int64)
    kv_indices = b["kv_indices"].to(torch.int64)
    kv_last = b["kv_last_page_lens"].to(torch.int64)
    bs = kv_indptr.numel() - 1
    outs = []
    for i in range(bs):
        p0 = int(kv_indptr[i].item())
        p1 = int(kv_indptr[i + 1].item())
        n_pages = p1 - p0
        if n_pages == 0:
            outs.append(torch.zeros_like(q_f32[i][:, :kv_lora]))
            continue
        last = int(kv_last[i].item())
        seq_len = (n_pages - 1) * page_size + last
        pages = kv_indices[p0:p1]
        k_nope = torch.cat([nope[pages[j]] for j in range(n_pages)], dim=0)[:seq_len]
        k_pe = torch.cat([pe[pages[j]] for j in range(n_pages)], dim=0)[:seq_len]
        k_nope = k_nope * kv_scale
        k_pe = k_pe * kv_scale
        kf = torch.cat([k_nope, k_pe], dim=-1)  # [seq, kv_lora+pe]
        qi = q_f32[i] * q_scale  # [nhead, kv_lora+pe]
        sc = (qi @ kf.T) * sm_scale  # [nhead, seq]
        sc = sc - sc.max(dim=-1, keepdim=True).values
        w = torch.softmax(sc, dim=-1)
        outs.append(w @ k_nope)  # [nhead, kv_lora]
    return torch.stack(outs, dim=0)  # [B, nhead, kv_lora]


def _normalize(b):
    """Accept both dump schemas:
      * dump_decode_mla:   kv_compact / kv_indices / o_server / kv_layout='seg'
      * _dump_seg_decode_failure: seg_kv_compact / kv_indices_remapped / o (no kv_layout)
    Both are the seg layout; map the latter onto the former's key names in place."""
    if "kv_compact" not in b and "seg_kv_compact" in b:
        b["kv_compact"] = b["seg_kv_compact"]
    if "kv_indices" not in b and "kv_indices_remapped" in b:
        b["kv_indices"] = b["kv_indices_remapped"]
    if "o_server" not in b and "o" in b:
        b["o_server"] = b["o"]
    # seg_decode dump always stores last_page_lens; both schemas already have
    # kv_indptr / qo_indptr / q / page_size / sm_scale / kv_lora_rank.
    b.setdefault("kv_layout", "seg")
    return b


def replay_one(path, dev="cuda"):
    b = _normalize(torch.load(path, map_location="cpu"))
    if b.get("kv_layout") != "seg":
        print(f"[skip] {os.path.basename(path)}: kv_layout={b.get('kv_layout')} (need 'seg')")
        return None
    kv_lora = b["kv_lora_rank"]
    o1 = _run_split(b, dev, 1)
    o2 = _run_split(b, dev, 2)
    ref = _fp32_ref(b, dev).to(dev)
    # kernel out is [B, nhead, v_head_dim(=kv_lora)]
    o1c = o1[..., :kv_lora]
    o2c = o2[..., :kv_lora]
    f1 = torch.isfinite(o1c).all().item()
    f2 = torch.isfinite(o2c).all().item()
    kvi = b["kv_indptr"].to(torch.int64)
    per_batch_pages = (kvi[1:] - kvi[:-1])
    max_pages = int(per_batch_pages.max().item())

    # --- Probe stage1 partials for split=2 BEFORE the stage2 reduce. ---
    # Decisive attribution: if NaN already exists in (logits|attn_lse) here, the
    # asm stage1 kernel produced it (stage2 / seg-layout-of-stage2 exonerated).
    s1_logits_fin = s1_lse_fin = None
    s1_split_diag = ""
    s1_geom_diag = ""
    s1_nan_geom = []
    s1_ok_geom = []
    lg2, lse2 = _stage1_partials(b, dev, 2)
    if lg2 is not None:
        s1_logits_fin = bool(torch.isfinite(lg2).all().item())
        s1_lse_fin = bool(torch.isfinite(lse2).all().item())
        if not (s1_logits_fin and s1_lse_fin):
            # which split index carries the NaN? reduce over (s, head, dv).
            lg_fin_per_split = torch.isfinite(lg2).all(dim=3).all(dim=2).all(dim=0).cpu()
            lse_fin_per_split = torch.isfinite(lse2).all(dim=3).all(dim=2).all(dim=0).cpu()
            bad_logit_splits = (~lg_fin_per_split).nonzero().reshape(-1).tolist()
            bad_lse_splits = (~lse_fin_per_split).nonzero().reshape(-1).tolist()
            # classify the partials. The LSE distinguishes the two mechanisms:
            #   lse all -inf  => L (softmax denom) underflowed to EXACTLY 0
            #                    => R*(1/0)=0*inf=NaN  (a ==0 guard fixes it)
            #   lse has NaN   => L became NaN (e.g. a garbage lane = +inf gives
            #                    exp(inf-inf)=NaN in the sum AND R)  => a ==0
            #                    guard MISSES it; need a non-finite guard + R reset
            n_posinf = int((lg2 == float("inf")).sum().item())
            n_neginf = int((lg2 == float("-inf")).sum().item())
            n_nan = int(torch.isnan(lg2).sum().item())
            lse_posinf = int((lse2 == float("inf")).sum().item())
            lse_neginf = int((lse2 == float("-inf")).sum().item())
            lse_nan = int(torch.isnan(lse2).sum().item())
            if lse_nan > 0:
                kind = "L==NaN (lse has NaN -> +inf lane: exp(inf-inf); R also NaN)"
            elif lse_neginf > 0 and n_posinf == 0:
                kind = "L==0 exactly (lse=-inf -> 0/0; ==0 guard applies)"
            elif n_posinf or lse_posinf:
                kind = "OVERFLOW(+inf -> bad running-max)"
            else:
                kind = "other"
            s1_split_diag = (
                f"stage1 NaN: logits_bad_splits={bad_logit_splits} "
                f"lse_bad_splits={bad_lse_splits} "
                f"logits[+inf={n_posinf} -inf={n_neginf} nan={n_nan}] "
                f"lse[+inf={lse_posinf} -inf={lse_neginf} nan={lse_nan}] kind={kind} "
                f"(=> asm stage1 wrote NaN; stage2 exonerated)"
            )
            # correlate NaN batches with kernel page geometry
            s1_nan_geom, s1_ok_geom, s1_geom_diag = _stage1_per_batch_diag(
                b, lg2, lse2
            )
    # Per-row (per-token) finiteness of forced split=2, mapped back to its batch
    # so we can correlate NaN rows with that batch's KV length (pages). If NaNs
    # cluster on SHORT batches, the empty per-batch split is the culprit.
    nan_diag = ""
    if not f2:
        qoi = b["qo_indptr"].to(torch.int64)
        klp = b["kv_last_page_lens"].to(torch.int64)
        page_size = int(b["page_size"])
        bs = qoi.numel() - 1
        row_finite = torch.isfinite(o2c.reshape(o2c.shape[0], -1)).all(dim=1).cpu()
        nan_tok, ok_tok, nan_last, ok_last = [], [], [], []
        for i in range(bs):
            r0, r1 = int(qoi[i]), int(qoi[i + 1])
            pg = int(per_batch_pages[i].item()) if i < per_batch_pages.numel() else 0
            last = int(klp[i].item()) if i < klp.numel() else 0
            tok = (pg - 1) * page_size + last if pg > 0 else 0
            rows_ok = bool(row_finite[r0:r1].all()) if r1 > r0 else True
            if rows_ok:
                ok_tok.append(tok); ok_last.append(last)
            else:
                nan_tok.append(tok); nan_last.append(last)

        def _fmt(xs):
            return f"{min(xs)}..{max(xs)}" if xs else "-"

        nan_diag = (
            f"nan_batches={len(nan_tok)}/{bs} "
            f"nan_tok={_fmt(nan_tok)} nan_last={sorted(set(nan_last))} | "
            f"ok_tok={_fmt(ok_tok)} ok_last={sorted(set(ok_last))}"
        )
    res = {
        "name": os.path.basename(path),
        "layer": b["layer_num"],
        "bs": b["kv_indptr"].numel() - 1,
        "pages": max_pages,
        "kv_indptr": b["kv_indptr"].tolist(),
        "finite1": f1,
        "finite2": f2,
        "stage1_logits_finite": s1_logits_fin,
        "stage1_lse_finite": s1_lse_fin,
        "stage1_diag": s1_split_diag,
        "stage1_geom_diag": s1_geom_diag,
        "stage1_nan_geom": s1_nan_geom,
        "stage1_ok_geom": s1_ok_geom,
        "nan_diag": nan_diag,
        "cos1_ref": _cos(o1c, ref) if f1 else float("nan"),
        "cos2_ref": _cos(o2c, ref) if f2 else float("nan"),
        "cos2_1": _cos(o2c, o1c) if (f1 and f2) else float("nan"),
        "relerr2_1": _max_rel(o2c, o1c) if (f1 and f2) else float("nan"),
    }
    return res


def main(arg):
    if os.path.isdir(arg):
        paths = sorted(
            glob.glob(os.path.join(arg, "mla_decode_*.pt"))
            + glob.glob(os.path.join(arg, "seg_decode_*.pt"))
        )
    else:
        paths = [arg]
    if not paths:
        print(f"No dumps found under {arg}")
        return
    print(f"Found {len(paths)} dump(s)\n")
    header = (
        f"{'name':42s} {'bs':>3s} {'pg':>4s} {'fin1':>4s} {'fin2':>4s} "
        f"{'cos1_ref':>10s} {'cos2_ref':>10s} {'cos2_1':>10s} {'relerr2_1':>10s}"
    )
    print(header)
    print("-" * len(header))
    worst = []
    diag_printed = 0
    for p in paths:
        r = replay_one(p)
        if r is None:
            continue
        print(
            f"{r['name']:42s} {r['bs']:>3d} {r['pages']:>4d} {str(r['finite1']):>4s} {str(r['finite2']):>4s} "
            f"{r['cos1_ref']:>10.6f} {r['cos2_ref']:>10.6f} {r['cos2_1']:>10.6f} {r['relerr2_1']:>10.4f}"
        )
        if r.get("stage1_diag"):
            print(f"    -> {r['stage1_diag']}")
        if r.get("stage1_geom_diag") and diag_printed < 5:
            print(f"    -> {r['stage1_geom_diag']}")
        if r.get("nan_diag") and diag_printed < 5:
            print(f"    -> {r['nan_diag']}")
            diag_printed += 1
        worst.append(r)
    if worst:
        import statistics

        c1 = [r["cos1_ref"] for r in worst if r["cos1_ref"] == r["cos1_ref"]]
        c2 = [r["cos2_ref"] for r in worst if r["cos2_ref"] == r["cos2_ref"]]
        c21 = [r["cos2_1"] for r in worst if r["cos2_1"] == r["cos2_1"]]
        re21 = [r["relerr2_1"] for r in worst if r["relerr2_1"] == r["relerr2_1"]]
        print("\n=== summary (forced 1-split vs forced 2-split) ===")
        if c21:
            print(
                f"cos(forced_split2, forced_split1): mean = {statistics.mean(c21):.6f}  "
                f"min = {min(c21):.6f}"
            )
        if re21:
            print(
                f"max-rel-err(forced_split2 vs split1): mean = {statistics.mean(re21):.6f}  "
                f"max = {max(re21):.6f}"
            )
        if c1:
            print(f"\n(ref is a rough fp32 baseline, may be misaligned)")
            print(f"mean cos(split1, fp32_ref) = {statistics.mean(c1):.6f}  min = {min(c1):.6f}")
        if c2:
            print(f"mean cos(split2, fp32_ref) = {statistics.mean(c2):.6f}  min = {min(c2):.6f}")
        n_nonfinite2 = sum(1 for r in worst if not r["finite2"])
        print(f"non-finite split2 outputs: {n_nonfinite2}/{len(worst)}")

        # Stage1-partial attribution summary: of the dumps whose split=2 OUTPUT
        # is non-finite, how many already had NaN in the stage1 PARTIALS?
        probed = [r for r in worst if r.get("stage1_logits_finite") is not None]
        if probed:
            s1_nan = sum(
                1
                for r in probed
                if not (r["stage1_logits_finite"] and r["stage1_lse_finite"])
            )
            print(
                f"stage1-partial probe: {s1_nan}/{len(probed)} dumps have NaN in "
                f"stage1 (logits|attn_lse) BEFORE stage2."
            )
            out_nan_s1_nan = sum(
                1
                for r in probed
                if (not r["finite2"])
                and not (r["stage1_logits_finite"] and r["stage1_lse_finite"])
            )
            out_nan = sum(1 for r in probed if not r["finite2"])
            if out_nan:
                print(
                    f"  of {out_nan} dumps with non-finite split2 OUTPUT, "
                    f"{out_nan_s1_nan} already had NaN in stage1 partials "
                    f"(=> asm stage1 root cause; stage2 reduce exonerated)."
                )

        # Cross-dump geometry correlation: pool every NaN batch's page geometry
        # to find the discriminating feature (the asm trigger condition).
        all_nan_geom = []
        for r in worst:
            all_nan_geom.extend(r.get("stage1_nan_geom") or [])
        if all_nan_geom:
            def _dist(key):
                from collections import Counter
                c = Counter(g[key] for g in all_nan_geom)
                return ", ".join(f"{k}:{v}" for k, v in sorted(c.items()))

            print("\n=== stage1 NaN batch geometry (pooled over all dumps) ===")
            print(f"  total NaN batches: {len(all_nan_geom)}")
            print(f"  full_pages -> count : {_dist('full_pages')}")
            print(f"  tail_len   -> count : {_dist('tail_len')}")
            print(f"  tail_owner -> count : {_dist('tail_owner')}")
            print(f"  loop0      -> count : {_dist('loop0')}")
            print(f"  loop1      -> count : {_dist('loop1')}")
            fp_par = {}
            for g in all_nan_geom:
                fp_par[g["full_pages"] % 2] = fp_par.get(g["full_pages"] % 2, 0) + 1
            print(f"  full_pages%2 -> count : {dict(sorted(fp_par.items()))}")

            # DECISIVE: does the SAME (full_pages, tail_len) geometry appear in
            # BOTH NaN and OK batches? Geometry fully determines kernel control
            # flow, so a collision => NaN is DATA-dependent (numerical overflow
            # in the multi-pass softmax/accumulator), NOT a control-flow/masking
            # bug. No collision => geometry alone triggers it (control-flow bug).
            all_ok_geom = []
            for r in worst:
                all_ok_geom.extend(r.get("stage1_ok_geom") or [])
            nan_keys = set((g["full_pages"], g["tail_len"]) for g in all_nan_geom)
            ok_keys = set((g["full_pages"], g["tail_len"]) for g in all_ok_geom)
            collide = sorted(nan_keys & ok_keys)
            print(
                f"\n  (full_pages,tail_len) keys: nan-only={len(nan_keys - ok_keys)} "
                f"ok-only={len(ok_keys - nan_keys)} BOTH={len(collide)}"
            )
            if collide:
                print(
                    "  >>> COLLISION: same geometry is BOTH NaN and OK -> NaN is "
                    "DATA-dependent (numerical), not control-flow."
                )
                print(f"      sample colliding (full_pages,tail_len): {collide[:12]}")
            else:
                print(
                    "  >>> NO collision: geometry alone determines NaN -> "
                    "control-flow/masking bug."
                )

        topn = sorted(worst, key=lambda r: -(r["relerr2_1"] if r["relerr2_1"] == r["relerr2_1"] else -1))[:10]
        print("\n=== worst 10 by max-rel-err(split2 vs split1) ===")
        for r in topn:
            print(f"  {r['name']:42s} cos2_1={r['cos2_1']:.6f}  relerr2_1={r['relerr2_1']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/mla_decode_dump"))
