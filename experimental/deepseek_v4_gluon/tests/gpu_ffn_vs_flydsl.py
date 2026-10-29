# SPDX-License-Identifier: MIT
"""GPU: the Gluon V4.1 FFN launch against vLLM's FlyDSL one, same runner.

torchrun --nproc-per-node=TP (patched-Triton image, /data/DeepSeek-V4.1-Flash
mounted). Built as upstream's ``tests/models/test_dsv41_mono_numerics.py``
builds it: ``DeepseekV4MoE`` from the real config, random weights in the
loaded layout through vLLM's own post-load processing, upstream's
``DSV41MonoLayer`` runner (scratch, peer buffers, epoch). Each width: FlyDSL's
FFN launch, then the Gluon one on the same buffers and inputs (the next epoch);
the outputs and every mailbox region's value words compared:

- bitwise: residual / post / comb / pre out, LIN, PMIX, normed, X8R / X8 and
  codes, LOGIT, ROUTE, SMID, SMQ / SMC (the same instructions in the same order)
- tolerance: MID (ug's K split differs: DESIGN.md), the reduced output
"""

import os
import sys

import torch
import torch.distributed as dist

MODEL = os.environ.get("DSV41_MODEL", "/data/DeepSeek-V4.1-Flash")
HIDDEN, HC = 5120, 4


def _random_moe(moe, gd, gr, inter):
    """Upstream ``_random_moe``: this rank's shards of the MXFP4 routed experts
    and MXFP8 shared expert in the loaded layout; the router alike on ranks."""
    u8 = torch.uint8
    for lin in (moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj):
        n = lin.weight.shape[0]
        w = torch.randn(lin.weight.shape, generator=gd, device=gd.device)
        lin.weight.data.copy_(w.clamp(-6, 6).to(lin.weight.dtype))
        sc = lin.weight_scale.data.view(u8)
        rows = -(-n // 32) if sc.shape[0] == n else sc.shape[0]
        blocks = torch.randint(118, 123, (rows, sc.shape[1]), generator=gd, device=gd.device,
                               dtype=u8)  # fmt: skip
        sc.copy_(blocks.repeat_interleave(32, 0)[:n] if sc.shape[0] == n else blocks)
    e = moe.experts.routed_experts
    for name, lo, hi in (("weight", 0, 256), ("weight_scale", 118, 122)):
        for w in ("w13", "w2"):
            t = getattr(e, f"{w}_{name}").data.view(u8)
            t.copy_(torch.randint(lo, hi, t.shape, generator=gd, device=gd.device, dtype=u8))
    padded = e.w13_weight.shape[1] // 2
    if padded > inter:
        w13, w2 = e.w13_weight.data.view(u8), e.w2_weight.data.view(u8)
        for half in (0, padded):
            w13[:, half + inter : half + padded] = 0
        w2[..., inter // 2 :] = 0
    for t in (moe.gate.weight, moe.gate.e_score_correction_bias):
        t.data.copy_(torch.randn(t.shape, generator=gr, device=gr.device) * 0.02)


def poison(sc, lay, regions):
    """Plain-data regions := 0xFF (NaN as bf16 / e4m3) before the Gluon launch:
    a region it fails to write cannot pass as FlyDSL's leftover. (A tagged
    region left unwritten fails its consumer's poll: the health word.)"""
    for k in regions:
        o, n = lay[k]
        sc[o : o + n].fill_(0xFF)
    torch.accelerator.synchronize()


def values(region: torch.Tensor) -> torch.Tensor:
    """A tagged-pair region's value words (the tags differ by one epoch)."""
    return region.view(torch.int32).view(-1, 2)[:, 0]


def bench(M, rank, run_fly, run_gl, n=20, reps=15):
    """Each launch's time, us: ``n`` back to back in a CUDA graph, the ranks
    replaying together (the launches wait on their peers), median of ``reps``.
    Same inputs every launch: the touched experts' weights stay warm in the
    MALL, for both alike."""
    us = {}
    for name, fn in (("flydsl", run_fly), ("gluon", run_gl)):
        for _ in range(3):
            fn()
        torch.accelerator.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                fn()
        ts = []
        for _ in range(reps):
            dist.barrier()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.accelerator.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000 / n)
        us[name] = sorted(ts)[reps // 2]
        del g
    if rank == 0:
        print(f"BENCH S={M}: flydsl {us['flydsl']:.2f} us  gluon {us['gluon']:.2f} us  "
              f"gluon/flydsl {us['gluon'] / us['flydsl']:.3f}", flush=True)


def main():
    # the checkpoint's MXFP4 experts run on AITER's MoE (its flags read at import)
    os.environ["VLLM_ROCM_USE_AITER"] = "1"
    os.environ["VLLM_ROCM_USE_AITER_MOE"] = "1"
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.deepseek_v41.amd.model import DeepseekV4MoE
    from vllm.models.deepseek_v41.amd.mono import layer as L
    from vllm.models.deepseek_v41.amd.mono.runner import DSV41MonoLayer, MonoLayerWeights
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.worker.workspace import init_workspace_manager

    from deepseek_v4_gluon.v41.ffn import build_constexprs, kffn, ptr

    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(device)
    rocm_aiter_ops.refresh_env_variables()
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(tp, rank, "env://", rank, "nccl")
        ensure_model_parallel_initialized(tp, 1)
    init_workspace_manager(device)
    vc = EngineArgs(model=MODEL, tensor_parallel_size=tp, moe_backend="aiter", load_format="dummy",
                    max_model_len=65536, language_model_only=True).create_engine_config()  # fmt: skip
    g = torch.Generator().manual_seed(1000)
    with set_current_vllm_config(vc), set_default_torch_dtype(torch.bfloat16), device:
        moe = DeepseekV4MoE(vc, prefix="model.layers.21.ffn")
    _random_moe(moe, torch.Generator(device=device).manual_seed(2001 + rank),
                torch.Generator(device=device).manual_seed(2000), inter=2304 // tp)  # fmt: skip
    process_weights_after_loading(moe, vc.model_config, device)
    f32 = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).to(device)  # noqa: E731
    mix = (2 + HC) * HC
    e, sh = moe.experts.routed_experts, moe.shared_experts
    u8 = lambda t: t.view(torch.uint8)  # noqa: E731
    mw = MonoLayerWeights(
        attn=None, hc_attn_fn=None, hc_attn_scale=None, hc_attn_base=None, attn_norm=None,
        hc_ffn_fn=f32(mix, HC * HIDDEN, sc=0.01), hc_ffn_scale=f32(3, sc=0.5).abs() + 0.5,
        hc_ffn_base=f32(mix, sc=0.1), ffn_norm=(1 + f32(HIDDEN, sc=0.1)).bfloat16(),
        gate_w=moe.gate.weight, bias=moe.gate.e_score_correction_bias, w13=e.w13_weight,
        w13_s=e.w13_weight_scale, w2=e.w2_weight, w2_s=e.w2_weight_scale,
        sgu=sh.gate_up_proj.weight, sgu_s=u8(sh.gate_up_proj.weight_scale),
        sw2=sh.down_proj.weight, sw2_s=u8(sh.down_proj.weight_scale),
    )  # fmt: skip
    runner = DSV41MonoLayer(tp, rank, get_tp_group().cpu_group, device)
    health = torch.zeros(4, dtype=torch.int32, device=device)
    widths = [int(a) for a in sys.argv[1:]] or [1, 4, 8, 16, 17, 24, 32, 48]
    bad = 0
    for M in widths:
        gi = torch.Generator().manual_seed(77 + M)
        part = (torch.randn(M, HIDDEN, generator=gi) * 0.5).bfloat16().to(device)
        res = (torch.randn(M, HC, HIDDEN, generator=gi) * 2).bfloat16().to(device)
        post = (torch.rand(M, HC, 1, generator=gi) * 2).to(device)
        comb = torch.softmax(torch.randn(M, HC, HC, generator=gi), -1).to(device)
        pre = (torch.rand(M, HC, generator=gi) + 0.5).to(device)
        lay = L.scratch_layout(M, tp)
        sc = runner.scratch(M)

        def snap():
            return {k: sc[o : o + n].clone() for k, (o, n) in lay.items()}

        torch.accelerator.synchronize()
        fly = runner.ffn(mw, part, res, post, comb, pre)
        torch.accelerator.synchronize()
        s_fly = snap()
        poison(sc, lay, ("normed", "x8r", "x8rs", "x8", "x8s"))
        outs = runner._outs(M, res)
        out, res_out, post_out, comb_out, pre_out = outs
        c = build_constexprs(M, tp)
        dist.barrier()
        # FlyDSL's raw-address ABI, as pointer arguments
        P = lambda a: ptr(a if isinstance(a, int) else a.data_ptr())  # noqa: E731
        kffn[(256,)](
            *map(P, (part, res, post, comb, pre, mw.hc_ffn_fn, mw.hc_ffn_scale, mw.hc_ffn_base,
                     mw.ffn_norm, res_out, post_out, comb_out, pre_out, mw.gate_w, mw.bias,
                     mw.w13, mw.w13_s, mw.w2, mw.w2_s, mw.sgu, mw.sgu_s, mw.sw2, mw.sw2_s, out,
                     sc, runner.peer.local, runner.peer.addresses)),
            rank, P(runner.epoch), 0, P(health),
            **c, num_warps=8, enable_fp_fusion=False,
        )  # fmt: skip
        torch.accelerator.synchronize()
        s_gl = snap()
        h = health.tolist()
        rep = [f"rank {rank} S={M}: health {h}"]
        for name, a, b in (("res_out", fly[1], res_out), ("post", fly[2], post_out),
                           ("comb", fly[3], comb_out), ("pre", fly[4], pre_out)):  # fmt: skip
            ok = torch.equal(a, b)
            bad += not ok
            rep.append(f"  {name:8s} bitwise {'OK' if ok else 'DIFF'}")
        plain = {"normed", "x8r", "x8rs", "x8", "x8s"}
        for k in ("lin", "pmix", "normed", "x8r", "x8rs", "x8", "x8s", "logit", "route", "smid",
                  "smq", "smc"):  # fmt: skip
            a, b = s_fly[k], s_gl[k]
            if k not in plain:
                a, b = values(a), values(b)
            ok = torch.equal(a, b)
            bad += not ok
            nd = int((a != b).sum()) if not ok else 0
            rep.append(f"  {k:8s} bitwise {'OK' if ok else f'DIFF ({nd} words)'}")
        cos = torch.nn.functional.cosine_similarity(fly[0].float().flatten(),
                                                    out.float().flatten(), dim=0).item()  # fmt: skip
        rel = ((fly[0].float() - out.float()).norm() / fly[0].float().norm()).item()
        ok = cos > 0.9999 and rel < 1e-2
        bad += not ok
        rep.append(f"  out      cos {cos:.6f} rel {rel:.2e} {'OK' if ok else 'BAD'}")
        print("\n".join(rep), flush=True)
        if os.environ.get("BENCH"):
            bench(M, rank, lambda: runner.ffn(mw, part, res, post, comb, pre), lambda: kffn[(256,)](
                *map(P, (part, res, post, comb, pre, mw.hc_ffn_fn, mw.hc_ffn_scale, mw.hc_ffn_base,
                         mw.ffn_norm, res_out, post_out, comb_out, pre_out, mw.gate_w, mw.bias,
                         mw.w13, mw.w13_s, mw.w2, mw.w2_s, mw.sgu, mw.sgu_s, mw.sw2, mw.sw2_s,
                         out, sc, runner.peer.local, runner.peer.addresses)),
                rank, P(runner.epoch), 0, P(health), **c, num_warps=8, enable_fp_fusion=False))
    dist.barrier()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
