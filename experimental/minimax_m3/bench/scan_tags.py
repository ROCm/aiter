"""Run a few Gluon steps, then list scratch / peer regions holding tags above the step's
last tag (a word the lean wait would accept early).   torchrun ... bench/scan_tags.py ROWS"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
from m3_inputs import make_layers, make_step  # noqa: E402


def main():
    rows = int(sys.argv[1])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    from vllm.models.minimax_m3.amd.mono.gluon.runner import GluonStandalone
    from vllm.models.minimax_m3.amd.mono.gluon import layout as LY
    L = 16
    nb = rows * 12 + 8
    layers = make_layers(L, nb, rank, dev)
    md, positions = make_step(rows, 1536, nb, dev)
    r = GluonStandalone(layers, rank, group, dev)
    r.md = md
    assert r.prepare(rows)
    h = torch.randn(rows, 6144, device=dev).bfloat16()
    res = torch.randn(rows, 6144, device=dev).bfloat16()
    for _ in range(3):
        r.forward(positions, h, res)
    torch.cuda.synchronize()
    ep = int(r.epoch[0].item())
    last = (ep - 1) * 64 + L  # the finished step's last tag
    half = LY.StepLayout().half()
    regs = LY.StepLayout().regions()
    sc = r.scratch.view(torch.int64)
    tags = (sc >> 32).to(torch.int64)
    if rank == 0:
        print(f"epoch {ep} last tag {last}")
        for par in range(2):
            for name, (off, n) in regs.items():
                t = tags[(par * half + off) // 8:(par * half + off + n) // 8]
                bad = (t > last).nonzero().flatten()
                if len(bad):
                    print(f"  parity {par} {name}: {len(bad)}/{len(t)} words with tag > {last}; first idx "
                          f"{bad[:4].tolist()} tags {t[bad[:4]].tolist()}")
        pt = (r.peers.bytes.view(torch.int64) >> 32)
        pb = (pt > last).nonzero().flatten()
        print(f"  peers: {len(pb)} words with tag > {last}" + (f" first {pb[:4].tolist()} {pt[pb[:4]].tolist()}" if len(pb) else ""))
    dist.barrier(group=group)
    r.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
