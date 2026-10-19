"""Routing-level check of the ATOM FlyDSL HERD path for DSV4-Flash.

ATOM path : topk_gating(K+1, no renorm) -> herd_route_psum_remap (fused drop)
aiter ref : herd_topk_gating (fused sqrtsoftplus HERD selector)
torch ref : popularity / min-score drop over the same K+1 candidates
"""

import torch

from aiter import topk_gating
from aiter.ops.flydsl.grouped_moe_gfx1250 import herd_route_psum_remap
from aiter.ops.flydsl.herd_topk import herd_topk_gating

E, K, SCALE, TILE_M = 256, 6, 1.5, 32


def torch_drop(cand_ids, cand_scores):
    tokens, kp1 = cand_ids.shape
    pop = torch.bincount(cand_ids.reshape(-1).long(), minlength=E)
    order = torch.argsort(cand_ids, dim=1)
    ids, scores = cand_ids.gather(1, order), cand_scores.gather(1, order)
    cp = pop[ids.long()]
    least = cp == cp.min(1, keepdim=True).values
    s = torch.where(least, scores, torch.full_like(scores, float("inf")))
    least &= s == s.min(1, keepdim=True).values
    cols = torch.arange(kp1, device=ids.device).expand(tokens, kp1)
    drop = torch.where(least, cols, torch.full_like(cols, kp1)).min(1).values
    keep = cols != drop[:, None]
    kept_ids = ids[keep].reshape(tokens, kp1 - 1)
    kept = scores[keep].reshape(tokens, kp1 - 1)
    return kept_ids, kept / kept.sum(1, keepdim=True).clamp_min(1e-20) * SCALE


def kept_from_rows(rows, masked_m, psum):
    starts = (psum - masked_m).long()
    r = rows.long()
    in_band = (r[..., None] >= starts) & (r[..., None] < psum.long())
    assert bool((in_band.sum(-1) == 1).all()), "row not in exactly one expert band"
    return in_band.int().argmax(-1).to(torch.int32)


def sort_rows(ids, w):
    order = torch.argsort(ids, dim=1)
    return ids.gather(1, order), w.gather(1, order)


torch.manual_seed(0)
dev = "cuda"
for m in (16, 32, 48, 64, 128):
    for trial in range(5):
        logits = torch.randn(m, E, device=dev, dtype=torch.float32).to(torch.bfloat16)
        bias = (torch.randn(E, device=dev) * 0.05).float()

        cand_ids = torch.empty(m, K + 1, dtype=torch.int32, device=dev)
        cand_w = torch.empty(m, K + 1, dtype=torch.float32, device=dev)
        topk_gating(cand_w, cand_ids, logits, bias, False, 1.0, score_func="sqrtsoftplus")
        masked_m, rows, psum, gather_w = herd_route_psum_remap(
            cand_ids, cand_w, E, 4096, TILE_M, SCALE
        )
        atom_ids = kept_from_rows(rows, masked_m, psum)
        atom_ids, atom_w = sort_rows(atom_ids, gather_w)

        t_ids, t_w = sort_rows(*torch_drop(cand_ids, cand_w))

        ref_ids = torch.empty(m, K, dtype=torch.int32, device=dev)
        ref_w = torch.empty(m, K, dtype=torch.float32, device=dev)
        herd_topk_gating(ref_w, ref_ids, logits, bias, True, SCALE, "sqrtsoftplus")
        ref_ids, ref_w = sort_rows(ref_ids, ref_w)

        std_ids = torch.empty(m, K, dtype=torch.int32, device=dev)
        std_w = torch.empty(m, K, dtype=torch.float32, device=dev)
        topk_gating(std_w, std_ids, logits, bias, True, SCALE, score_func="sqrtsoftplus")

        tok_vs_torch = (atom_ids == t_ids).all(1).float().mean().item()
        tok_vs_aiter = (atom_ids == ref_ids).all(1).float().mean().item()
        same = (atom_ids == ref_ids).all(1)
        w_err = (atom_w - ref_w)[same].abs().max().item() if same.any() else float("nan")
        w_err_t = (atom_w - t_w).abs().max().item()
        print(
            f"M={m:4d} trial={trial} ids==torch {tok_vs_torch:.3f} "
            f"ids==herd_topk_gating {tok_vs_aiter:.3f} "
            f"|w-torch|max {w_err_t:.2e} |w-aiter|max {w_err:.2e} "
            f"masked_m.sum {int(masked_m.sum())}/{m * K} "
            f"experts {int((masked_m > 0).sum())} vs top-{K} "
            f"{int(torch.unique(std_ids).numel())}"
        )
