"""
test_triton_ar.py -- TritonAllReduce vs the fp32 sum (within 1 ulp),
bf16 and fp16, sizes across one-shot / two-shot and uneven parts, calls that
alternate modes back to back, and a HIP graph replayed with new data. Checks
that all ranks get the same bits.

    torchrun --nproc_per_node=4 test_triton_ar.py
"""
import os

import torch
import torch.distributed as dist

from aiter.ops.triton.triton_ar import TritonAllReduce

SIZES = [int(v) for v in os.environ.get("SIZES", "8,1024,4096,32768,32776,65536,65544,"
         "262144,1048576,1048584,8,4194304,1024").split(",")]  # elements; one-shot <= 32768


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo")
    kw = {}
    if os.environ.get("ONE_SHOT_BYTES"):
        kw["one_shot_bytes"] = int(os.environ["ONE_SHOT_BYTES"])
    ar = TritonAllReduce(rank, world, max_bytes=max(SIZES) * 2, **kw)
    ar.register(dist.group.WORLD)

    def inputs(n, seed, dtype):
        g = torch.Generator(device=dev).manual_seed(seed)
        xs = [torch.randn(n, generator=g, dtype=torch.float32, device=dev).to(dtype)
              for _ in range(world)]
        ref = xs[0].float()
        for t in xs[1:]:
            ref = ref + t.float()
        return xs[rank], ref.to(dtype)

    failures = []

    def check(tag, n, y, ref):
        # tl.sum adds the ranks as a tree, not in rank order: with fp16 inputs
        # the fp32 sum is not always exact, so allow 1 ulp; ranks must agree
        d = (y.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
        if (d > 1).any():
            bad = int((d > 1).sum())
            idx = (d > 1).nonzero().flatten()[:8].tolist()
            failures.append(f"{tag} n={n}: {bad} values off by > 1 ulp from the fp32 sum "
                            f"(first at {idx})")
        mine = y.view(torch.int16).to(torch.int32).cpu()  # gloo has no int16
        parts = [torch.empty_like(mine) for _ in range(world)]
        dist.all_gather(parts, mine)  # gloo: CPU tensors
        if not all(torch.equal(parts[0], p) for p in parts):
            failures.append(f"{tag} n={n}: ranks differ")

    for dtype in (torch.bfloat16, torch.float16):
        tag = str(dtype).split(".")[-1]
        cases = [inputs(n, i, dtype) for i, n in enumerate(SIZES)]
        outs = [ar(x) for x, _ in cases]
        torch.cuda.synchronize()
        for n, y, (_, ref) in zip(SIZES, outs, cases):
            check(f"eager {tag}", n, y, ref)
        xs = [c[0].clone() for c in cases]
        for x in xs:
            ar(x)
        torch.cuda.synchronize()
        dist.barrier()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            gouts = [ar(x) for x in xs]
        for seed in (100, 200):
            new = [inputs(n, seed + i, dtype) for i, n in enumerate(SIZES)]
            for x, nw_ in zip(xs, new):
                x.copy_(nw_[0])
            torch.cuda.synchronize()
            dist.barrier()
            gr.replay()
            torch.cuda.synchronize()
            for n, y, nw_ in zip(SIZES, gouts, new):
                check(f"graph {tag} seed {seed}", n, y, nw_[1])
        del gr
    t = torch.tensor([len(failures)])
    dist.all_reduce(t)
    if failures:
        print(f"rank {rank}: {failures[:6]}", flush=True)
    if rank == 0:
        print("PASS" if t.item() == 0 else f"FAIL ({t.item()} cases)", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
