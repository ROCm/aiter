"""Experiment driver for the gfx1250 gluon batched_gemm_a16w8 kernel.

Run ONLY through gpurun.sh (one process per GPU). Prints one JSON line per run.
"""

import argparse
import copy
import gc
import importlib.util
import json
import sys

import torch
import triton
import triton.experimental.gluon.language as gl

from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
    _pad_interval,
    create_shared_layouts,
    create_wmma_layouts,
)
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.types import get_fp8_dtypes

KERNEL_NAME = "_batched_gemm_a16w8_gfx1250_persistent_kernel"


def load_module(path):
    if not path:
        from aiter.ops.triton._gluon_kernels.gfx1250.gemm.batched import (
            batched_gemm_a16w8 as m,
        )

        return m
    spec = importlib.util.spec_from_file_location(f"kvariant_{abs(hash(path))}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_inputs(B, M, N, K, strided_x, transpose_bm, seed=0):
    _, e4m3 = get_fp8_dtypes()
    g = torch.Generator(device="cuda").manual_seed(seed)
    if strided_x:
        buf = torch.randn(
            (M, B, K + 64), dtype=torch.bfloat16, device="cuda", generator=g
        )
        x = buf[..., :K].transpose(0, 1)
    else:
        x = torch.randn((B, M, K), dtype=torch.bfloat16, device="cuda", generator=g)
    w = (torch.randn((B, N, K), device="cuda", generator=g) * 0.1).to(e4m3)
    s = torch.tensor(0.02, dtype=torch.float32, device="cuda")
    y = torch.empty(
        (M, B, N) if transpose_bm else (B, M, N), dtype=torch.bfloat16, device="cuda"
    )
    return x, w, s, y


def launch(mod, x, w, s, yq, transpose_bm, config):
    """Mirror of aiter.ops.triton.gemm.batched.batched_gemm_a16w8 launch logic.

    Extra config keys starting with 'X_' are passed to the kernel as constexprs
    (name without the prefix), for experimental kernel variants.
    """
    B, M, K = x.shape
    N = w.shape[1]
    y = yq.transpose(0, 1) if transpose_bm else yq
    BM, BN, BK = config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"], config["BLOCK_SIZE_K"]
    store_mode = config.get("STORE_MODE", 1)
    if store_mode == 2 and _pad_interval(BN, y.element_size() * 8) != BN:
        store_mode = 1
    num_warps = config["num_warps"]
    wmma_layout, op_a, op_b = create_wmma_layouts(num_warps)
    sh_a, sh_b = create_shared_layouts(
        BM, BN, BK, "TN", x.element_size() * 8, elem_bits_b=w.element_size() * 8
    )
    sh_c = gl.PaddedSharedLayout.with_identity_for(
        [[_pad_interval(BN, y.element_size() * 8), config.get("C_PAD", 16)]],
        [BM, BN],
        [1, 0],
    )
    tpn = min(32, BN // 8)
    store_layout = gl.BlockedLayout([1, 8], [32 // tpn, tpn], [num_warps, 1], [1, 0])
    num_tiles = B * triton.cdiv(M, BM) * triton.cdiv(N, BN)
    num_wgs = min(num_tiles, get_num_sms() * config.get("WG_PER_CU", 1))
    extra = {k[2:]: v for k, v in config.items() if k.startswith("X_") and k != "X_TAG"}
    kern = getattr(mod, config.get("KERNEL", KERNEL_NAME))
    kern[(num_wgs,)](
        x, w, y, s, B, M, N, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), y.stride(0), y.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        NUM_BUFFERS=config.get("NUM_BUFFERS", 2), NUM_WGS=num_wgs, STORE_MODE=store_mode,
        SHARED_LAYOUT_A=sh_a, SHARED_LAYOUT_B=sh_b, SHARED_LAYOUT_C=sh_c,
        STORE_LAYOUT=store_layout, WMMA_LAYOUT=wmma_layout,
        OPERAND_LAYOUT_A=op_a, OPERAND_LAYOUT_B=op_b,
        num_warps=num_warps, waves_per_eu=config.get("waves_per_eu", 1),
        **extra,
    )  # fmt: skip
    return yq


def check(yq, x, w, s, transpose_bm):
    B = x.shape[0]
    heads = sorted({0, 1, B // 3, B // 2, B - 2, B - 1} & set(range(B)))
    got = (yq.transpose(0, 1) if transpose_bm else yq)[heads].float().cpu()
    xs = x[heads].float().cpu()
    ws = (w[heads].float() * s.float()).cpu()
    ref = torch.bmm(xs, ws.transpose(1, 2))
    err = (got - ref).abs()
    tol = 1e-3 + 1e-2 * ref.abs()
    bad = int((err > tol).sum())
    return {"max_abs_err": float(err.max()), "n_bad": bad, "ok": bad == 0}


def time_graph(fn_list, calls, replays):
    """Per-launch median/min us from CUDA-graph replay of `calls` launches cycling fn_list."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn_list[0]()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        gc.disable()
        try:
            with torch.cuda.graph(graph, stream=stream):
                for i in range(calls):
                    fn_list[i % len(fn_list)]()
        finally:
            gc.enable()
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(replays):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / calls)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def kernel_meta(mod, name):
    kern = getattr(mod, name)
    out = []
    try:
        for cache in kern.device_caches.values():
            for ck in cache[0].values():
                md = ck.metadata
                out.append(
                    {
                        "n_regs": getattr(ck, "n_regs", None),
                        "n_spills": getattr(ck, "n_spills", None),
                        "shared": getattr(md, "shared", None),
                        "num_warps": getattr(md, "num_warps", None),
                    }
                )
    except Exception as e:  # noqa: BLE001
        out.append({"meta_error": repr(e)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--b", type=int, default=128)
    p.add_argument("--m", type=int, default=1536)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--strided-x", action="store_true")
    p.add_argument("--transpose-bm", action="store_true")
    p.add_argument(
        "--config", type=str, default=None, help="JSON dict (merged over tuned)"
    )
    p.add_argument(
        "--configs-file", type=str, default=None, help="JSONL of configs to sweep"
    )
    p.add_argument("--module", type=str, default=None, help="kernel variant .py")
    p.add_argument("--label", type=str, default="")
    p.add_argument("--check", action="store_true")
    p.add_argument(
        "--hot", action="store_true", help="reuse one input set (L2/MALL warm)"
    )
    p.add_argument("--calls", type=int, default=24)
    p.add_argument("--replays", type=int, default=25)
    p.add_argument(
        "--dobench",
        action="store_true",
        help="also time the public wrapper with do_bench",
    )
    p.add_argument(
        "--eager", type=int, default=0, help="just launch N times (for rocprofv3)"
    )
    p.add_argument("--dump-asm", type=str, default=None)
    p.add_argument(
        "--fullcheck",
        type=int,
        default=0,
        help="launch variant N times, compare full output bitwise vs repo kernel",
    )
    a = p.parse_args()

    B, M, N, K = a.b, a.m, a.n, a.k
    mod = load_module(a.module)
    tuned, is_tuned = get_gemm_config("BATCHED_GEMM-A16W8", M, N, K, backend="gluon")
    configs = []
    if a.configs_file:
        with open(a.configs_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    c = copy.deepcopy(tuned)
                    c.update(json.loads(line))
                    configs.append(c)
    else:
        c = copy.deepcopy(tuned)
        if a.config:
            c.update(json.loads(a.config))
        configs.append(c)

    nbytes = B * M * K * 2 + B * N * K * 1 + B * M * N * 2
    flops = 2.0 * B * M * N * K

    base = make_inputs(B, M, N, K, a.strided_x, a.transpose_bm)
    per_copy = sum(t.numel() * t.element_size() for t in base if t.dim() > 0)
    if a.hot:
        copies = [base]
    else:
        budget = min(1024 << 20, torch.cuda.mem_get_info()[0] // 4)
        n = max(2, min(8, budget // per_copy + 1))
        copies = [base] + [
            make_inputs(B, M, N, K, a.strided_x, a.transpose_bm, seed=i + 1)
            for i in range(n - 1)
        ]

    for cfg in configs:
        rec = {"label": a.label, "B": B, "M": M, "N": N, "K": K,
               "strided_x": a.strided_x, "transpose_bm": a.transpose_bm,
               "hot": a.hot, "module": a.module or "repo", "is_tuned_shape": is_tuned,
               "config": {k: v for k, v in cfg.items()}}  # fmt: skip
        try:
            fns = [
                (
                    lambda c=c_, cfg=cfg: launch(
                        mod, *c[:4], a.transpose_bm, copy.deepcopy(cfg)
                    )
                )
                for c_ in copies
            ]
            fns[0]()
            torch.cuda.synchronize()
            if a.check:
                rec["check"] = check(copies[0][3], *copies[0][:3], a.transpose_bm)
            if a.fullcheck:
                repo = load_module(None)
                base_cfg = {k: v for k, v in cfg.items() if not k.startswith("X_")}
                x0, w0, s0, y0 = copies[0]
                y_ref = torch.empty_like(y0)
                launch(repo, x0, w0, s0, y_ref, a.transpose_bm, copy.deepcopy(base_cfg))
                torch.cuda.synchronize()
                bad = 0
                for _ in range(a.fullcheck):
                    y0.fill_(float("nan"))
                    launch(mod, x0, w0, s0, y0, a.transpose_bm, copy.deepcopy(cfg))
                    torch.cuda.synchronize()
                    bad += int(not torch.equal(y0, y_ref))
                rec["fullcheck"] = {"runs": a.fullcheck, "mismatching_runs": bad}
            if a.eager:
                for i in range(a.eager):
                    fns[i % len(fns)]()
                torch.cuda.synchronize()
            else:
                med, lo = time_graph(fns, max(a.calls, len(fns)), a.replays)
                rec.update(us=round(med, 2), us_min=round(lo, 2),
                           eff_TBps=round(nbytes / (med * 1e-6) / 1e12, 2),
                           TFLOPs=round(flops / (med * 1e-6) / 1e12, 1))  # fmt: skip
            if a.dobench:
                from aiter.ops.triton.gemm.batched.batched_gemm_a16w8 import (
                    batched_gemm_a16w8,
                )

                x, w, s, y = copies[0]
                ms = triton.testing.do_bench(
                    lambda x=x, w=w, s=s, y=y: batched_gemm_a16w8(
                        x, w, s, YQ=y, transpose_bm=a.transpose_bm
                    ),
                    warmup=25, rep=100,
                )  # fmt: skip
                rec["dobench_us"] = round(ms * 1e3, 2)
            rec["meta"] = kernel_meta(mod, cfg.get("KERNEL", KERNEL_NAME))
            if a.dump_asm:
                kern = getattr(mod, cfg.get("KERNEL", KERNEL_NAME))
                for cache in kern.device_caches.values():
                    for i, ck in enumerate(cache[0].values()):
                        with open(f"{a.dump_asm}.{i}.amdgcn", "w") as f:
                            f.write(ck.asm.get("amdgcn", ""))
                        with open(f"{a.dump_asm}.{i}.ttgir", "w") as f:
                            f.write(str(ck.asm.get("ttgir", "")))
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)[:500]
        print("RESULT " + json.dumps(rec), flush=True)
        torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(main())
