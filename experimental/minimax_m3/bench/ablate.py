"""Per-region ablation of the FlyDSL and Gluon M3 layer kernels (TP4, torchrun).

Layers 0..L run once normally: every mailbox then holds layer L's tagged data. Layer L
is then relaunched with one region enabled (FlyDSL: VLLM_M3_FLY_STAGES; Gluon: STAGES):
every wait in it succeeds on its first load, so the replay times the region's own
compute and memory traffic, alone on the GPU. Region time = replay - empty-kernel
replay, CUDA-graph timed, max over ranks.

  torchrun ... bench/ablate.py --rows 1 [--layer 1] [--regions qkv,split,...]
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
from m3_inputs import StandaloneRunner, make_layers, make_spec_step, make_step  # noqa: E402
from bench_layer_tp import FLY_POINTS, timeline_summary  # noqa: E402

# region -> (FlyDSL stage bits, Gluon STAGES bits)
REGIONS = {
    "empty": (0, 0),
    "qkv": (1, 1 | 2),            # input norm, FP8, qkv GEMV, head norm / RoPE, KV insert (+ index k)
    "split": (2 | 4, 4),          # block select + split attention
    "merge+o": (8 | 16, 8),       # partitions merged, o GEMV, attention all-reduce
    "router": (32, 16),           # logits, scores, xn publish
    "moe": (64, 32 | 64 | 128 | 256),  # routing, shared + routed up / gate, down, FFN all-reduce
    "all": (127, 511),
}
# hybrid (VLLM_M3_GLUON_HYBRID=1): the previous layer's FFN all-reduce runs first in every
# launch unless bit 1024; the MoE is AITER's (not in the kernel)
HYB_REGIONS = {
    "empty": 1024, "far": 0, "qkv": 1 | 2 | 1024, "norm": 2048 | 1024, "pnorm": 4096 | 1024, "split": 4 | 1024, "merge+o": 8 | 1024, "router": 16 | 1024,
    "all": 511,
    # one stage alone (kernel.py AB_* bits)
    "qkv_only": (1 << 23) | 1024, "ik": 2 | 1024, "own": (1 << 24) | 1024, "split_only": (1 << 25) | 1024,
    "merge": (1 << 26) | 1024, "o": (1 << 27) | 1024, "router_only": (1 << 28) | 1024, "topk": (1 << 29) | 1024, "o_gemm": (1 << 30) | 1024,
}
if os.environ.get("VLLM_M3_GLUON_HYBRID") == "1":
    REGIONS = {k: (REGIONS.get(k, (0, 0))[0], v) for k, v in HYB_REGIONS.items()}


def time_graph(fn, group, n_inner=20, reps=15):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for _ in range(n_inner):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / n_inner)
    ts.sort()
    t = torch.tensor([ts[len(ts) // 2]])
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    del g
    return t.item()


class Fly:
    def __init__(self, layers, rank, group, dev, rows, md, positions, hidden, resid, L, timeline=False):
        self.r = StandaloneRunner(layers, rank, group, dev, timeline=timeline)
        self.r.md = md
        assert self.r.prepare(rows)
        self.full = self.r.k4
        self.rows, self.positions, self.hidden, self.resid, self.L = rows, positions, hidden, resid, L
        self.it = layers[L].index_topk
        self.cache = {}

    def fill(self):
        r = self.r
        r.k4 = self.full
        r.begin(self.positions, self.rows)
        r.ars[0][: self.rows].copy_(self.hidden)
        res = self.resid
        for i in range(self.L):
            res = r.layer(i, res)
        self.res_in = res
        r.layer(self.L, res)
        torch.cuda.synchronize()

    def variant(self, bits):
        if bits not in self.cache:
            os.environ["VLLM_M3_FLY_STAGES"] = str(bits)
            self.cache[bits] = {self.it: self.r._build_k4(self.rows, index_topk=self.it)}
            os.environ.pop("VLLM_M3_FLY_STAGES")
        k = {self.rows: self.cache[bits]}
        res = self.res_in

        def fn():
            self.r.k4 = k
            self.r.layer(self.L, res)
        return fn


class Glu:
    def __init__(self, layers, rank, group, dev, rows, md, positions, hidden, resid, L, timeline=False):
        from vllm.models.minimax_m3.amd.mono.gluon.runner import GluonStandalone

        self.r = GluonStandalone(layers, rank, group, dev, timeline=timeline)
        self.r.ql = int(os.environ.get("M3_ABL_QL", "1"))
        self.r.md = md
        assert self.r.prepare(rows)
        self.rows, self.positions, self.hidden, self.resid, self.L = rows, positions, hidden, resid, L
        d = md.decode
        self.args = lambda li: self.r._args(positions, md.slot_mapping, d.block_table, d.seq_lens, hidden, resid, li)

    def _launch(self, li, stages):
        from vllm.models.minimax_m3.amd.mono.gluon.kernel import m3_step_kernel

        consts = self.r._consts(self.rows)
        consts["STAGES"] = stages
        consts["LOOP"] = False
        m3_step_kernel[(256,)](*self.args(li), **consts)

    def fill(self):
        for li in range(self.L + 1):
            self._launch(li, 511)
        torch.cuda.synchronize()

    def variant(self, bits):
        return lambda: self._launch(self.L, bits)

    def dump(self, bits, path):
        from vllm.models.minimax_m3.amd.mono.gluon.kernel import m3_step_kernel

        consts = self.r._consts(self.rows)
        consts["STAGES"] = bits
        consts["LOOP"] = False
        cc = m3_step_kernel.warmup(*self.args(self.L), grid=(256,), **consts)
        open(path, "w").write(cc.asm["amdgcn"])

    def check(self, what):
        torch.cuda.synchronize()
        try:
            self.r.raise_on_errors()
        except RuntimeError as e:
            print(f"rank {self.r.rank} gluon {what}: {e}", flush=True)
            self.r.errors.zero_()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=1536)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--ql", type=int, default=1, help="rows a request (speculative verify): rows // ql requests")
    ap.add_argument("--backends", default="flydsl,gluon")
    ap.add_argument("--regions", default=",".join(REGIONS))
    ap.add_argument("--timeline", action="store_true", help="also each region's stamps (one replay, rank 0)")
    ap.add_argument("--dump", default="", help="directory: each Gluon region build's ISA (FlyDSL: FLYDSL_DUMP_IR)")
    a = ap.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    rows, L = a.rows, a.layer
    os.environ["M3_ABL_QL"] = str(a.ql)
    reqs = rows // a.ql
    nb = reqs * ((a.ctx + a.ql + 127) // 128) + 8
    layers = make_layers(L + 2, nb, rank, dev, max_pos=a.ctx + a.ql + 16)  # layer L is never the step's last
    if a.ql > 1:
        md, positions = make_spec_step(reqs, a.ql, a.ctx, nb, dev)
    else:
        md, positions = make_step(rows, a.ctx, nb, dev)
    hidden = torch.randn(rows, 6144, generator=torch.Generator(device=dev).manual_seed(5), device=dev).bfloat16()
    resid = (torch.randn(rows, 6144, generator=torch.Generator(device=dev).manual_seed(6), device=dev) * 4).bfloat16()
    backs = {}
    for b in a.backends.split(","):
        cls = Fly if b == "flydsl" else Glu
        backs[b] = cls(layers, rank, group, dev, rows, md, positions, hidden, resid, L, a.timeline)
    out = {b: {} for b in backs}
    for b, x in backs.items():
        x.fill()
        dist.barrier(group=group)
        for reg in a.regions.split(","):
            bits = REGIONS[reg][0 if b == "flydsl" else 1]
            fn = x.variant(bits)
            out[b][reg] = time_graph(fn, group)
            if a.timeline and reg not in ("empty",):
                tl = x.r.tl_buf
                tl[L].zero_()
                torch.cuda.synchronize()
                dist.barrier(group=group)
                fn()
                torch.cuda.synchronize()
                if rank == 0 and os.environ.get("M3_TL_SAVE"):
                    torch.save(tl[L].cpu(), f"{os.environ['M3_TL_SAVE']}_{b}_{reg}.pt")
                if rank == 0:
                    pts = FLY_POINTS if b == "flydsl" else getattr(x.r, "TL_NAMES", {})
                    print(f"  {b} {reg} stamps (us after the first CTA start):\n" + timeline_summary(tl.cpu(), L, pts),
                          flush=True)
            if b == "gluon":
                x.check(reg)
                if a.dump and rank == 0:
                    os.makedirs(a.dump, exist_ok=True)
                    x.dump(bits, f"{a.dump}/gluon_{reg}_rows{rows}.s")
    if rank == 0:
        print(f"rows={rows} ql={a.ql} ctx={a.ctx} layer={L} ({'selects' if layers[L].index_topk else 'reuses'}): "
              f"us per launch, region = replay - empty", flush=True)
        names = list(backs)
        print(f"  {'region':10s}" + "".join(f" {n:>9s}" for n in names) + ("   gap  ratio" if len(names) == 2 else ""))
        for reg in a.regions.split(","):
            vals = [out[n][reg] - (out[n]["empty"] if reg != "empty" else 0) for n in names]
            line = f"  {reg:10s}" + "".join(f" {v:9.2f}" for v in vals)
            if len(names) == 2 and reg != "empty":
                line += f" {vals[1] - vals[0]:+6.2f} {vals[1] / max(vals[0], 1e-3):5.2f}x"
            print(line, flush=True)
    dist.barrier(group=group)
    for x in backs.values():
        x.r.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
