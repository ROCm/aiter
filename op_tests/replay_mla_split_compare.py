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
    max_pages = int((kvi[1:] - kvi[:-1]).max().item())
    res = {
        "name": os.path.basename(path),
        "layer": b["layer_num"],
        "bs": b["kv_indptr"].numel() - 1,
        "pages": max_pages,
        "kv_indptr": b["kv_indptr"].tolist(),
        "finite1": f1,
        "finite2": f2,
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
    for p in paths:
        r = replay_one(p)
        if r is None:
            continue
        print(
            f"{r['name']:42s} {r['bs']:>3d} {r['pages']:>4d} {str(r['finite1']):>4s} {str(r['finite2']):>4s} "
            f"{r['cos1_ref']:>10.6f} {r['cos2_ref']:>10.6f} {r['cos2_1']:>10.6f} {r['relerr2_1']:>10.4f}"
        )
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

        topn = sorted(worst, key=lambda r: -(r["relerr2_1"] if r["relerr2_1"] == r["relerr2_1"] else -1))[:10]
        print("\n=== worst 10 by max-rel-err(split2 vs split1) ===")
        for r in topn:
            print(f"  {r['name']:42s} cos2_1={r['cos2_1']:.6f}  relerr2_1={r['relerr2_1']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/mla_decode_dump"))
