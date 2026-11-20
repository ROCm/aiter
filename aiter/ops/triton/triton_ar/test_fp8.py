"""test_fp8.py -- TritonAllReduce(fp8=True) vs the fp32 sum quantised the
same way (per 128 values, amax/448); ranks must agree; eager + graph."""
import os
import torch
import torch.distributed as dist
from aiter.ops.triton.triton_ar import TritonAllReduce

SIZES = [int(v) for v in os.environ.get("SIZES", "65544,262144,1048576,1048584,4194304,4194312,262152").split(",")]


def qref(s, dtype):
    v = s.to(dtype).float()
    n = v.numel()
    pad = (-n) % 128
    vp = torch.nn.functional.pad(v, (0, pad)).view(-1, 128)
    sc = vp.abs().amax(1, keepdim=True) / 448.0
    sc = torch.where(sc > 0, sc, torch.ones_like(sc))
    q = (vp / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc
    return q.flatten()[:n].to(dtype)


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
    dist.init_process_group("gloo")
    ar = TritonAllReduce(rank, world, max_bytes=max(SIZES) * 2, fp8=True)
    ar.register(dist.group.WORLD)
    fails = []
    for dtype in (torch.bfloat16, torch.float16):
        def inputs(n, seed):
            g = torch.Generator(device=dev).manual_seed(seed)
            xs = [torch.randn(n, generator=g, device=dev).to(dtype) for _ in range(world)]
            return xs[rank], sum(t.float() for t in xs)

        def check(tag, n, y, s):
            ref = qref(s, dtype)
            # the 128-groups follow the parts (64-word tiles from each part start)
            # one fp8 step (12.5% of the value, or the subnormal step) of
            # slack: values at a rounding tie may round either way
            d = (y.float() - ref.float()).abs()
            pad = (-n) % 128
            sc = torch.nn.functional.pad(s.abs(), (0, pad)).view(-1, 128).amax(1, keepdim=True) / 448
            step = sc.expand(-1, 128).flatten()[:n] * 2.0 ** -8
            tol = 0.126 * ref.float().abs() + step
            err = ((y.float() - s).abs().max() / s.abs().max()).item()
            bad = int((d > tol).sum()) + int(y.isnan().sum())
            mine = y.view(torch.int16).int().cpu(); parts = [torch.empty_like(mine) for _ in range(world)]
            dist.all_gather(parts, mine)
            if not all(torch.equal(parts[0], p) for p in parts):
                fails.append(f"{tag} n={n}: ranks differ")
            if bad:
                fails.append(f"{tag} n={n}: {bad} off the fp8 reference, idx {((d > tol) | y.isnan()).nonzero().flatten()[:4].tolist()}")
            return err
        cases = [inputs(n, i) for i, n in enumerate(SIZES)]
        outs = [ar(x, fp8=True) for x, _ in cases]
        torch.cuda.synchronize()
        errs = [check(f"eager {dtype}", n, y, s) for n, y, (_, s) in zip(SIZES, outs, cases)]
        if rank == 0:
            print(f"{dtype}: max |err|/max|sum| per size {[f'{e:.4f}' for e in errs]}", flush=True)
        xs = [c[0].clone() for c in cases]
        for x in xs:
            ar(x, fp8=True)
        torch.cuda.synchronize(); dist.barrier()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            gouts = [ar(x, fp8=True) for x in xs]
        for seed in (100, 200):
            new = [inputs(n, seed + i) for i, n in enumerate(SIZES)]
            for x, nn in zip(xs, new):
                x.copy_(nn[0])
            torch.cuda.synchronize(); dist.barrier()
            gr.replay(); torch.cuda.synchronize()
            for n, y, nn in zip(SIZES, gouts, new):
                check(f"graph {dtype} {seed}", n, y, nn[1])
        del gr
    t = torch.tensor([len(fails)]); dist.all_reduce(t)
    if fails:
        print(f"rank {rank}: {fails[:6]}", flush=True)
    if rank == 0:
        print("PASS" if t.item() == 0 else f"FAIL ({t.item()})", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
