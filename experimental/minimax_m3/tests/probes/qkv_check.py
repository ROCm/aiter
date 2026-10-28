"""qkv GEMV output vs an fp32 reference, hybrid, 2 layers (layer 0: the norm tasks' input,
layer 1: the FFN owners' when XMX): reference = (h rstd (1 + g_in)) @ W^T s, h = the kernel's
own "h" mailbox (bf16(ar + res)) of that layer.
torchrun --nproc_per_node 4 tests/probes/qkv_check.py [rows]"""
import os, sys
import torch
import torch.distributed as dist
sys.path.insert(0, "/home/vigokhal/m3_monokernel/bench")
from m3_inputs import make_layers, make_step
from vllm.models.minimax_m3.amd.mono.gluon.runner import GluonStandalone
from vllm.models.minimax_m3.amd.mono.gluon import layout as LY
from vllm.models.minimax_m3.amd.mono.gluon import kernel as K

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 48
rank = int(os.environ["RANK"]); torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
dist.init_process_group("gloo"); group = dist.new_group(backend="gloo")
ctx = 1536
nb = rows * ((ctx + 127) // 128) + 8
layers = make_layers(int(os.environ.get("M3_NL", "2")), nb, rank, dev)
r = GluonStandalone(layers, rank, group, dev)
md, positions = make_step(rows, ctx, nb, dev)
r.md = md
assert r.prepare(rows)
g = torch.Generator(device=dev)
hidden0 = torch.randn(rows, 6144, generator=g.manual_seed(5), device=dev).bfloat16()
resid0 = (torch.randn(rows, 6144, generator=g.manual_seed(6), device=dev) * 4).bfloat16()
r.forward(positions, hidden0, resid0)
torch.cuda.synchronize()
r.raise_on_errors()
half = LY.StepLayout().half()
words = r.scratch.view(torch.int64)
HP, QP = 6144 // 2, LY.QKV_ROWS // 2


def pairs(par, region, n, width):
    w = words[(par * half + LY.region(region)) // 8:][: n * width].view(n, width)
    v = (w & 0xFFFFFFFF).to(torch.int32)
    lo = (v << 16).view(torch.float32)
    hi = (v & -65536).view(torch.float32)
    return torch.stack([lo, hi], -1).view(n, 2 * width)


out = []
for li in range(len(layers)):
    lw = layers[li]
    h = pairs(li % 2, "h", rows, HP)
    q = pairs(li % 2, "qkv", rows, QP)
    nq = int(K.ROWS16.value if lw.index_topk else K.NQ_REUSE.value) * 16
    xn = h * torch.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * (1 + lw.g_in.float())
    # w_qkv is aiter shuffle_weight(layout=(16, 16)) FP8: [N/16, K/32, 2, 16 n, 16 k] -> [N, K]
    N, Kd = lw.w_qkv.shape
    w = lw.w_qkv.view(torch.uint8).view(N // 16, Kd // 32, 2, 16, 16).permute(0, 3, 1, 2, 4).reshape(N, Kd)
    ref = (xn @ w.view(lw.w_qkv.dtype).float().T * lw.s_qkv.float())[:, :nq]
    got = q[:, :nq]
    e = ((got - ref).norm() / ref.norm()).item()
    worst = ((got - ref).norm(dim=1) / ref.norm(dim=1)).max().item()
    seg = " ".join(f"{n}:{((got[:, a:b] - ref[:, a:b]).norm() / ref[:, a:b].norm()).item():.2e}"
                   for n, a, b in (("q", 0, 2048), ("k", 2048, 2176), ("v", 2176, 2304), ("iq", 2304, 2432),
                                   ("ik", 2432, 2560)) if b <= nq)
    hn = (h.norm() / (h.numel() ** 0.5)).item()
    xw = words[(li % 2 * half + LY.region("x8s")) // 8:][:rows]
    xs = (xw & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    xt = (xw >> 32).to(torch.int32)
    xn_ = h * torch.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * (1 + lw.g_in.float())
    xs_ref = xn_.abs().amax(-1) / 448
    if os.environ.get("VLLM_M3_GLUON_PLAIN_HO", "1") == "1":  # plain i32 words
        wv = r.scratch[li % 2 * half + LY.region("x8"):].view(torch.int32)[: rows * 1536].view(rows, 1536)
        wt = torch.zeros(1, dtype=torch.int32)
    else:
        ww = words[(li % 2 * half + LY.region("x8")) // 8:][: rows * 1536].view(rows, 1536)
        wt = (ww >> 32).to(torch.int32)
        wv = (ww & 0xFFFFFFFF).to(torch.int32)
    by = torch.stack([(wv >> (8 * i)) & 255 for i in range(4)], -1).view(rows, 6144).to(torch.uint8)
    deq = by.view(torch.float8_e4m3fn).float() * xs[:, None]
    e8 = ((deq - xn_).norm() / xn_.norm()).item()
    bad = ((deq - xn_).abs() > 0.1 * xn_.abs().amax(-1, keepdim=True)).nonzero()
    q8 = (deq @ w.view(lw.w_qkv.dtype).float().T * lw.s_qkv.float())[:, :nq]
    e9 = ((got - q8).norm() / q8.norm()).item()
    rowe = ((got - q8).norm(dim=1) / q8.norm(dim=1))
    qw = words[(li % 2 * half + LY.region("qkv")) // 8:][: rows * QP].view(rows, QP)
    out.append(f"qkv tags {(qw >> 32).unique().tolist()[:4]} got rms {got.pow(2).mean().sqrt().item():.3e} ref rms {q8.pow(2).mean().sqrt().item():.3e} corr {torch.nn.functional.cosine_similarity(got.flatten(), q8.flatten(), 0).item():.3f}")
    out.append(f"qkv vs its own x8: nRMSE {e9:.3e} rows bad {(rowe > 0.05).nonzero().flatten().tolist()[:12]}")
    out.append(f"x8 L{li}: tags {wt.unique().tolist()[:4]} deq nRMSE {e8:.3e} bad {bad.shape[0]} first {bad[:3].tolist()}")
    out.append(f"x8s L{li}: tags {xt.unique().tolist()[:4]} scale/ref min {(xs / xs_ref).min().item():.4f} max {(xs / xs_ref).max().item():.4f}")
    out.append(f"L{li} ({'select' if lw.index_topk else 'reuse'}, {nq} rows) nRMSE {e:.3e} worst row {worst:.3e} [{seg}] h rms {hn:.2f}")
if rank == 0 and os.environ.get("M3_SAVE_ROUTE"):
    torch.save({"ids": r.moe_ids[:rows].cpu(), "w": r.moe_w[:rows].cpu(), "x": r.moe_x[:rows].cpu()},
               os.environ["M3_SAVE_ROUTE"])
if rank == 0:
    print(f"rows {rows} XMX={os.environ.get('VLLM_M3_GLUON_XMX', '1')}: " + "; ".join(out), flush=True)
r.close()
dist.destroy_process_group()
