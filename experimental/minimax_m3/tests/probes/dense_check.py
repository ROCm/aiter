"""Dense all-reduce hand-offs (VLLM_M3_GLUON_DENSE_AR=1), hybrid, 2 layers, layer 0 an EAGLE3 aux
layer, two steps (different inputs). Checks, after each step:
- every dense slot is back to DSENT (0xFFFFFFFF) on every rank: each was read and reset once;
- ar_out[0] (the last layer's FFN all-reduce, ffn_tail) == bf16 of the rank-ordered f32 sum of the
  4 ranks' AITER partials (moe_out, still the last layer's);
- ar_out[1] (layer 0's, written by layer 1's norm tasks from what they read) is consistent with
  their h: bf16(ar_out[1] + hm of layer 0) == layer 1's "h" mailbox.
torchrun --nproc_per_node 4 tests/probes/dense_check.py [rows]"""
import os
import sys
import torch
import torch.distributed as dist
sys.path.insert(0, "/home/vigokhal/m3_monokernel/bench")
from m3_inputs import make_layers, make_step
from vllm.models.minimax_m3.amd.mono.gluon.runner import GluonStandalone
from vllm.models.minimax_m3.amd.mono.gluon import layout as LY

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 48
rank = int(os.environ["RANK"]); torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
dist.init_process_group("gloo"); group = dist.new_group(backend="gloo")
ctx = 1536
nb = rows * ((ctx + 127) // 128) + 8
layers = make_layers(2, nb, rank, dev)
r = GluonStandalone(layers, rank, group, dev)
r.aux_sparse = (0,)
md, positions = make_step(rows, ctx, nb, dev)
r.md = md
assert r.prepare(rows)
half = LY.StepLayout().half()
HP = LY.HIDDEN // 2
g = torch.Generator(device=dev)
ok_all = True
for step in range(2):
    hidden0 = torch.randn(rows, 6144, generator=g.manual_seed(5 + 10 * step), device=dev).bfloat16()
    resid0 = (torch.randn(rows, 6144, generator=g.manual_seed(6 + 10 * step), device=dev) * 4).bfloat16()
    r.forward(positions, hidden0, resid0)
    torch.cuda.synchronize()
    e = r.errors[:LY.BLOCKS].cpu()
    if e.any():
        nz = e.nonzero().flatten().tolist()
        print(f"rank {rank} step {step}: layer {r.errors[LY.BLOCKS].item()} code4 CTAs "
              f"{[c for c in nz if e[c] == 4]} c14 {[c for c in nz if e[c] == 14]} c15 {[c for c in nz if e[c] == 15]} c16 {[c for c in nz if e[c] == 16]} code10 {[c for c in nz if e[c] == 10]} "
              f"code1 {[c for c in nz if e[c] == 1]} others {sorted(set(e[nz].tolist()))}", flush=True)
    if e.any():
        pr = LY.peer_regions()
        pb = r.peers.bytes
        for par in range(2):
            for nm, nx in (("attnd", "ffnd"), ("aagd", "fagd")):
                seg = pb[par * LY.peer_half() + pr[nm]:par * LY.peer_half() + pr[nx]].view(torch.int32)
                nzw = (seg != -1).nonzero().flatten()
                if nm == "aagd":
                    tok = (nzw // 16) % LY.MAX_S
                    src = nzw * 0
                    tsk = nzw // (16 * LY.MAX_S)
                else:
                    tok = (nzw // HP) % LY.MAX_S
                    src = nzw // (LY.MAX_S * HP)
                    tsk = (nzw % HP) // 16
                print(f"rank {rank} par {par} {nm}: non-DSENT words {nzw.numel()} src {sorted(set(src.tolist()))} "
                      f"tokens {sorted(set(tok.tolist()))[:20]} tasks {sorted(set(tsk.tolist()))}", flush=True)
        ag = pb[pr["aagd"]:pr["fagd"]].view(torch.int32).view(LY.N_O, LY.MAX_S, 16)  # task-major
        st = [c for c in e.nonzero().flatten().tolist() if e[c] in (4, 14, 15, 16)]
        for t in st[:4]:
            w = ag[t, :rows]
            hw = r.scratch.view(torch.int64)[LY.region("h") // 8:][: rows * HP].view(rows, HP)[:, 16 * t:16 * t + 16]
            ht = (hw >> 32)
            print(f"rank {rank} task {t}: h tags per group {[sorted(set(ht[16 * q:16 * q + 16].flatten().tolist())) for q in range(3)]}", flush=True)
            print(f"rank {rank} task {t}: aagd slots DSENT {(w == -1).sum().item()}/{w.numel()} "
                  f"per group {[(w[16 * q:16 * q + 16] == -1).sum().item() for q in range(3)]} per token {(w == -1).sum(1).tolist()}", flush=True)
    r.raise_on_errors()
    out = []
    # 1. dense regions all DSENT
    pb = r.peers.bytes
    bad = 0
    for par in range(2):
        for off, n in LY.dense_spans():
            seg = pb[par * LY.peer_half() + off:][:n]
            bad += int((seg != 255).sum().item())
    out.append(f"non-DSENT bytes {bad}")
    ok = bad == 0
    # 2. last layer's all-reduce vs the partials
    parts = [torch.zeros(rows, 6144, dtype=torch.bfloat16) for _ in range(4)]
    dist.all_gather(parts, r.moe_out[:rows].cpu().contiguous(), group=group)
    ref = parts[0].float()
    for p in parts[1:]:
        ref = ref + p.float()
    ref = ref.bfloat16()
    got = r.ar_out[0, :rows].cpu()
    eq = torch.equal(got.view(torch.int16), ref.view(torch.int16))
    out.append(f"ar_out[0] exact {eq} (mismatch {(got != ref).sum().item()})")
    ok &= eq
    # 3. aux: the norm tasks' ar_out[1] vs their h
    words = r.scratch.view(torch.int64)
    h1 = (words[(1 * half + LY.region("h")) // 8:][: rows * HP].view(rows, HP) & 0xFFFFFFFF).to(torch.int32)
    hm0 = r.scratch[0 * half + LY.region("hm"):].view(torch.int32)[: rows * HP].view(rows, HP)
    h1 = h1.contiguous().view(torch.bfloat16).view(rows, 6144).float()
    hm0 = hm0.contiguous().view(torch.bfloat16).view(rows, 6144).float()
    a1 = r.ar_out[1, :rows].float()
    eq2 = torch.equal((a1 + hm0).bfloat16(), h1.bfloat16())
    out.append(f"aux consistent {eq2} (ar_out[1] rms {a1.pow(2).mean().sqrt().item():.3e})")
    ok &= eq2
    oks = [None] * 4
    dist.all_gather_object(oks, (ok, "; ".join(out)), group=group)
    if rank == 0:
        for i, (o, s) in enumerate(oks):
            print(f"step {step} rank {i}: {'OK ' if o else 'BAD'} {s}", flush=True)
            ok_all &= o
if rank == 0:
    print("DENSE CHECK", "PASS" if ok_all else "FAIL", flush=True)
r.close()
dist.destroy_process_group()
