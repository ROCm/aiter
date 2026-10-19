"""Hash layers on ATOM's Triton decode path: which experts does it pick?

    python check_hash_routing.py [dump_dir]

dump_dir holds H<layer>_S<step>.pt files from
`DUMP_PATCH=atom_hash_compare.patch bash dump_router.sh triton_herd`: the router
logits of DSV4's hash-routed layers, the experts the model's own hash routing
(custom_routing_function, tid2eid[input_ids]) picks, and the experts ATOM's
Triton decode path actually used. This replays aiter's Triton routing() on the
same logits with HERD off and on -- exactly the call ATOM makes -- and reports
how many of each token's 6 experts agree with the hash routing.
"""

import collections
import glob
import os
import statistics
import sys

import torch

import aiter.ops.triton.moe.moe_routing.routing as routing_mod
from aiter.ops.triton.moe.moe_routing.routing import routing

TOPK = 6


def overlap(a, b):
    return torch.tensor([len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a, b)], dtype=torch.float)


def main():
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "hash_compare"
    files = sorted(glob.glob(os.path.join(dump_dir, "H*_S*.pt")))
    if not files:
        raise SystemExit(f"no H*_S*.pt under {dump_dir}")
    routing_mod._HERD_MIN_M, routing_mod._HERD_MAX_M = 16, 128
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in files:
        layer = os.path.basename(path)[:3]
        d = torch.load(path)
        logits = d["logits"].cuda()
        ids_hash = d["ids_hash"].long()
        res[(layer, "hash")]["unique"].append(ids_hash.unique().numel())
        for herd in (False, True):
            routing_mod._USE_HERD = herd
            rd, _, _ = routing(
                logits, TOPK, score_mode="sqrtsoftplus", bias=None, renorm=True,
                routed_scaling_factor=d["routed_scaling_factor"],
            )
            ids = rd.topk_ids.long().cpu()
            ov = overlap(ids_hash, ids)
            r = res[(layer, herd)]
            r["shared"].append(ov.mean().item())
            r["identical"].append((ov == TOPK).float().mean().item())
            r["disjoint"].append((ov == 0).float().mean().item())
            r["unique"].append(ids.unique().numel())
            if herd:
                same = torch.equal(torch.sort(ids, 1).values, torch.sort(d["ids_triton"].long(), 1).values)
                r["matches_atom"].append(same)

    mean = statistics.mean
    print("| layer | Triton routing | experts shared with hash routing (of 6) | identical tokens | "
          "tokens with none shared | unique experts: Triton / hash |")
    print("|---|---|---|---|---|---|")
    for layer in sorted({k[0] for k in res}):
        for herd in (False, True):
            r = res[(layer, herd)]
            note = ""
            if herd:
                note = " (replay == ids ATOM used)" if all(r["matches_atom"]) else " (replay != ids ATOM used)"
            print(f"| {layer} | {'HERD' if herd else 'no HERD'}{note} | {mean(r['shared']):.2f} | "
                  f"{mean(r['identical']):.1%} | {mean(r['disjoint']):.1%} | "
                  f"{mean(r['unique']):.1f} / {mean(res[(layer, 'hash')]['unique']):.1f} |")


if __name__ == "__main__":
    main()
