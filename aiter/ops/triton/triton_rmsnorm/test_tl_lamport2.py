"""
test_tl_lamport2.py -- TlLamport2RMSNorm correctness: one-shot and two-shot
sizes (also uneven parts), calls that alternate the 2 modes back to back,
graph replays with new data; bf16 and static-scale fp8 output.

    torchrun --nproc_per_node=4 test_tl_lamport2.py
"""
import os

import torch
import torch.distributed as dist

H, EPS, SCALE = 8192, 1e-6, 0.05
ROWS = [int(r) for r in os.environ.get("ROWS", "1,3,4,5,16,64,100,1,128,4,256,1027,2").split(",")]


def _trace(msg):
    if os.environ.get("TRACE"):
        print(f"[trace rank {os.environ['RANK']}] {msg}", flush=True)


def main():
    if os.environ.get("DUMP_AFTER"):
        import faulthandler
        faulthandler.dump_traceback_later(int(os.environ["DUMP_AFTER"]), exit=True)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo")
    from aiter.ops.triton.triton_rmsnorm import TlLamport2RMSNorm
    _trace("init")
    op = TlLamport2RMSNorm(rank, world, H, max(ROWS),
                           blk_words=int(os.environ.get("BLK", "256")),
                           num_warps=int(os.environ.get("WARPS", "1")))
    _trace("alloc")
    op.register(dist.group.WORLD)
    _trace(f"registered bases {[hex(b) for b in op._bases[:world]]}")
    if os.environ.get("PRECOMPILE"):
        op.precompile()
        torch.cuda.synchronize()
        dist.barrier()
        _trace("precompiled")
    w = (torch.rand(H, generator=torch.Generator(device=dev).manual_seed(5),
                    dtype=torch.float32, device=dev) + 0.5).to(torch.bfloat16)
    sc = torch.tensor([SCALE], dtype=torch.float32, device=dev)

    def inputs(rows, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        xs = [torch.randn(rows, H, generator=g, dtype=torch.bfloat16, device=dev)
              for _ in range(world)]
        res = torch.randn(rows, H, generator=g, dtype=torch.bfloat16, device=dev)
        s = xs[0].float()
        mag = xs[0].float().abs()
        for t in xs[1:]:
            s = s + t.float()
            mag = mag + t.float().abs()
        v = (s + res.float()).to(torch.bfloat16)
        mag = mag + res.float().abs()
        return xs[rank], res, (v, mag), None

    def bad(out, res_out, ref, _unused, quant):
        # res_out: the kernel's tl.sum adds the [NGPU, BLK] rows as a tree
        # ((x0+x2)+(x1+x3) at 4 GPUs), not in rank order. Where the values
        # cancel, fp32 rounding in either order is amplified (1027 rows:
        # 0.53125 - 0.531252 -> 8 bf16 ulp at 1.8e-6, 2026-11-16). Allow
        # 1 ulp, or the fp32 bound 4 * 2^-24 * (sum |x_i| + |res|);
        # ranks_differ() checks that all ranks agree bit for bit.
        v, mag = ref
        dr = (res_out.view(torch.int16).int() - v.view(torch.int16).int()).abs()
        tol = 4 * 2.0 ** -24 * mag
        n = int(((dr > 1) & ((res_out.float() - v.float()).abs() > tol)).sum())
        # out: the norm of the kernel's own res_out, so the norm check does
        # not inherit the all-reduce's order of summation
        vf = res_out.float()
        y = vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
        if quant:
            q = (y / SCALE).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            a, b = out.view(torch.uint8).int(), q.view(torch.uint8).int()
            same = (a >> 7) == (b >> 7)
            steps = torch.where(same, ((a & 0x7F) - (b & 0x7F)).abs(), (a & 0x7F) + (b & 0x7F))
            return n + int((steps > 1).sum())
        d = (out.view(torch.int16).int() - y.to(torch.bfloat16).view(torch.int16).int()).abs()
        return n + int((d > 1).sum())

    failures = []

    def ranks_differ(out, res_out):
        mine = torch.cat([out.view(torch.uint8).flatten().int(),
                          res_out.view(torch.uint8).flatten().int()]).cpu()
        parts = [torch.empty_like(mine) for _ in range(world)]
        dist.all_gather(parts, mine)  # gloo: CPU tensors
        return not all(torch.equal(parts[0], q) for q in parts)
    for quant in (False, True):
        s_arg = sc if quant else None
        tag = "fp8" if quant else "bf16"
        cases = [inputs(r, i) for i, r in enumerate(ROWS)]
        _trace(f"{tag} eager launch")
        outs = [tuple(t.clone() for t in op(x, r, w, EPS, s_arg)) for x, r, _, _ in cases]
        torch.cuda.synchronize()
        _trace(f"{tag} eager done")
        if op.maxit:
            d = op.dbg.view(-1, 8)
            print(f"[dbg rank {rank}] missing per (program, peer): "
                  f"{[(p, g, int(d[p, g])) for p in range(d.shape[0]) for g in range(world) if d[p, g]]}",
                  flush=True)
            # host view of one-shot buffer 0 of my inbox: sentinel halves per source
            buf = op._buf.local_view()[: world * op.slot * 4].view(torch.int16).view(world, -1)
            host = buf.cpu()
            print(f"[dbg rank {rank}] sentinel halves in memory per source (buffer 0): "
                  f"{[int((host[g] == -1).sum()) for g in range(world)]} of {host.shape[1]}",
                  flush=True)
        for rows, o, (_, _, v, y) in zip(ROWS, outs, cases):
            if bad(*o, v, y, quant):
                failures.append(f"eager {tag} rows {rows} mode {op.mode(rows)}")
            if ranks_differ(*o):
                failures.append(f"eager {tag} rows {rows} mode {op.mode(rows)}: ranks differ")
        xs = [c[0].clone() for c in cases]
        rs = [c[1].clone() for c in cases]
        for x, r in zip(xs, rs):
            op(x, r, w, EPS, s_arg)
        torch.cuda.synchronize()
        dist.barrier()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            gouts = [op(x, r, w, EPS, s_arg) for x, r in zip(xs, rs)]
        for seed in (100, 200):
            new = [inputs(r, seed + i) for i, r in enumerate(ROWS)]
            for x, r, n in zip(xs, rs, new):
                x.copy_(n[0])
                r.copy_(n[1])
            torch.cuda.synchronize()
            dist.barrier()
            gr.replay()
            torch.cuda.synchronize()
            for rows, o, n in zip(ROWS, gouts, new):
                if bad(*o, n[2], n[3], quant):
                    failures.append(f"graph {tag} seed {seed} rows {rows} mode {op.mode(rows)}")
                if ranks_differ(*o):
                    failures.append(f"graph {tag} seed {seed} rows {rows} mode {op.mode(rows)}: ranks differ")
        del gr
    t = torch.tensor([len(failures)])
    dist.all_reduce(t)
    if failures:
        print(f"rank {rank}: {failures[:8]}", flush=True)
    if rank == 0:
        print("PASS" if t.item() == 0 else f"FAIL ({t.item()} cases)", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
