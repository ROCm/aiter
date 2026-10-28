"""Speculative-verify rows (q = 4 a request) vs the same rows as independent q = 1 requests
(block table row per token, context pos + 1) re-run after: their in-step keys then come
from the cache. torchrun --nproc_per_node 4 tests/probes/spec_check.py [rows] [layers]"""
import os, sys
import torch
import torch.distributed as dist
from types import SimpleNamespace
sys.path.insert(0, "/home/vigokhal/m3_monokernel/bench")
from m3_inputs import make_layers, BLOCK, BT_WIDTH
from vllm.models.minimax_m3.amd.mono.gluon.runner import GluonStandalone

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 48
L = int(sys.argv[2]) if len(sys.argv) > 2 else 1
Q, ctx = 4, int(os.environ.get("CTX", 1536))
rank = int(os.environ["RANK"]); torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
dist.init_process_group("gloo"); group = dist.new_group(backend="gloo")
reqs = rows // Q
need = (ctx + BLOCK - 1) // BLOCK
nb = reqs * need + 8
layers = make_layers(L, nb, rank, dev)
g = torch.Generator(device=dev).manual_seed(4242)
perm = torch.randperm(nb, generator=g, device=dev).to(torch.int32)
bt4 = torch.zeros(64, BT_WIDTH, dtype=torch.int32, device=dev)
for r in range(reqs):
    bt4[r, :need] = perm[r * need:(r + 1) * need]
j = torch.arange(rows, device=dev)
positions = (ctx - Q + j % Q).long()
req = j // Q
slot = (bt4[req, positions // BLOCK].long() * BLOCK + positions % BLOCK).contiguous()
md4 = SimpleNamespace(decode=SimpleNamespace(block_table=bt4[:reqs], seq_lens=torch.full((reqs,), ctx, dtype=torch.int32,
                      device=dev), decode_query_len=Q), slot_mapping=slot, num_prefills=0)
bt1 = torch.zeros(64, BT_WIDTH, dtype=torch.int32, device=dev)
bt1[:rows] = bt4[req]
md1 = SimpleNamespace(decode=SimpleNamespace(block_table=bt1[:rows], seq_lens=(positions + 1).int(), decode_query_len=1),
                      slot_mapping=slot, num_prefills=0)
hidden = torch.randn(rows, 6144, generator=torch.Generator(device=dev).manual_seed(5), device=dev).bfloat16()
resid = (torch.randn(rows, 6144, generator=torch.Generator(device=dev).manual_seed(6), device=dev) * 4).bfloat16()
outs = {}
for name, q, md in (("q4", Q, md4), ("q1", 1, md1)):
    r = GluonStandalone(layers, rank, group, dev)
    r.ql = q
    r.md = md
    assert r.prepare(rows)
    h, rr = r.forward(positions, hidden, resid)
    torch.cuda.synchronize()
    r.raise_on_errors()
    outs[name] = (h.float().clone(), rr.float().clone())
    dist.barrier(group=group)
if rank == 0:
    for i, nm in enumerate(("ar", "hm")):
        a, b = outs["q4"][i], outs["q1"][i]
        d = (a - b).norm(dim=1) / b.norm(dim=1)
        print(f"rows {rows} L={L} {nm}: exact {torch.equal(a, b)} nRMSE {((a - b).norm() / b.norm()).item():.2e} "
              f"worst row {d.max().item():.2e} (row {d.argmax().item()}); by jq: "
              + " ".join(f"{(d[j % Q == k]).max().item():.1e}" for k in range(Q)), flush=True)
dist.barrier(group=group)
os._exit(0)
