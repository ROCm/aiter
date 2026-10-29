# SPDX-License-Identifier: MIT
"""GPU: the Gluon V4.1 K2 launch (attention back + FFN half) against vLLM's
FlyDSL one, same runner, at K2's ABI.

torchrun --nproc-per-node=TP (``gpu_run.sh``). The FFN half's weights as in
``gpu_ffn_vs_flydsl`` (``DeepseekV4MoE``, random, vLLM's post-load processing);
the attention back's inputs synthetic: a rotated q, fp8_ds_mla window and
compressed caches (pages of 576 B rows then 8 B codes), K1's key table (top-k
compressed rows, then window rows, -1 past a token's keys), cos / sin, sink,
MXFP8 wo_a / wo_b. Each width: FlyDSL's K2, then the Gluon K2 on the same
buffers (the next epoch); compared bitwise: the back's regions (partials,
chunk stats, XO, X8B and codes), the FFN regions and outputs as in the FFN
test. ``BENCH=1``: CUDA-graph timing of both."""

import os
import sys

import torch
import torch.distributed as dist

from deepseek_v4_gluon.tests.gpu_ffn_vs_flydsl import (
    HC,
    HIDDEN,
    MODEL,
    _random_moe,
    bench,
    poison,
    values,
)

NOPE, ROPE, DATA, KEYS, HEAD_DIM = 448, 64, 576, 640, 512


def fp8_bytes(shape, g, device):
    """Random E4M3 bytes, no NaN codes (0x7F / 0xFF)."""
    b = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)
    b[(b & 0x7F) == 0x7F] = 0
    return b.to(device)


def kv_cache(pages, block, g, device):
    """[pages, block * 584] fp8_ds_mla pages: rows of 448 e4m3 + 64 bf16 rope, then
    a row's 7 E8M0 codes + a pad byte."""
    data = fp8_bytes((pages, block, DATA), g, "cpu")
    rope = (torch.randn(pages, block, ROPE, generator=g) * 0.5).bfloat16()
    data[:, :, NOPE:] = rope.view(torch.uint8).view(pages, block, 2 * ROPE)
    codes = torch.randint(118, 127, (pages, block, 8), generator=g, dtype=torch.uint8)
    codes[:, :, 7] = 0
    return torch.cat([data.view(pages, -1), codes.view(pages, -1)], 1).contiguous().to(device)


def main():
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

    from deepseek_v4_gluon.v41.ffn import ptr
    from deepseek_v4_gluon.v41.k2 import build_constexprs, k2

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
    # the attention back's weights and tables (a rank its own)
    ga = torch.Generator().manual_seed(3000 + rank)
    heads, groups = 64 // tp, 8 // tp
    o_rows, group_k = groups * 1024, heads // groups * HEAD_DIM
    woa = fp8_bytes((o_rows, group_k), ga, device)
    woa_s = torch.randint(118, 123, (o_rows // 32, group_k // 32), generator=ga, dtype=torch.uint8).to(device)
    wob = fp8_bytes((HIDDEN, o_rows), ga, device)
    wob_s = torch.randint(118, 123, (HIDDEN // 32, o_rows // 32), generator=ga, dtype=torch.uint8).to(device)
    block, pages = 64, 24
    swa = kv_cache(pages, block, ga, device)
    comp = kv_cache(pages, block, ga, device)
    max_pos = 4096
    ang = torch.rand(max_pos, ROPE // 2, generator=ga) * 6.28
    cos_sin = torch.cat([ang.cos(), ang.sin()], 1).contiguous().to(device)
    sink = (torch.randn(heads, generator=ga)).to(device)
    zrec = torch.zeros(1024, dtype=torch.uint8, device=device)
    dummy = torch.zeros(4, dtype=torch.int32, device=device)
    qk_scale = torch.tensor([HEAD_DIM**-0.5 * 1.4426950408889634]).view(torch.int32).item()
    ratio = int(os.environ.get("RATIO", "1"))
    widths = [int(a) for a in sys.argv[1:]] or [1, 4, 16, 17, 48]
    bad = 0
    for M in widths:
        gi = torch.Generator().manual_seed(77 + M)
        res = (torch.randn(M, HC, HIDDEN, generator=gi) * 2).bfloat16().to(device)
        post = (torch.rand(M, HC, generator=gi) * 2).to(device)
        comb = torch.softmax(torch.randn(M, HC, HC, generator=gi), -1).to(device)
        pre = (torch.rand(M, HC, generator=gi) + 0.5).to(device)
        q = (torch.randn(M, heads, HEAD_DIM, generator=ga) * 0.5).bfloat16().to(device)
        # key table: token t's kv_len keys, top-k compressed rows first (ratio > 0)
        kt = torch.full((M, KEYS), -1, dtype=torch.int32)
        klen = torch.zeros(M, dtype=torch.int32)
        for t in range(M):
            n = int(torch.randint(1, KEYS + 1, (1,), generator=gi)) if t % 3 else KEYS
            nc = min(n, int(torch.randint(0, 513, (1,), generator=gi))) if ratio else 0
            cs = torch.randint(0, pages * block, (nc,), generator=gi, dtype=torch.int32)
            kt[t, :nc] = cs | (-(2**31))
            kt[t, nc:n] = torch.randint(0, pages * block, (n - nc,), generator=gi, dtype=torch.int32)
            klen[t] = n
        kt, klen = kt.to(device), klen.to(device)
        pos = torch.randint(0, max_pos, (M,), generator=gi, dtype=torch.int64).to(device)
        lay = L.scratch_layout(M, tp)
        sc = runner.scratch(M)

        def snap():
            return {k: sc[o : o + n].clone() for k, (o, n) in lay.items()}

        def attn_args():
            return (q.data_ptr(), swa.data_ptr(), swa.stride(0), block, comp.data_ptr(),
                    comp.stride(0), block, kt.data_ptr(), klen.data_ptr(), dummy.data_ptr(),
                    dummy.data_ptr(), dummy.data_ptr(), dummy.data_ptr(), dummy.data_ptr(),
                    dummy.data_ptr(), 0, pos.data_ptr(), sink.data_ptr(), qk_scale,
                    cos_sin.data_ptr(), woa.data_ptr(), woa_s.data_ptr(), wob.data_ptr(),
                    wob_s.data_ptr(), zrec.data_ptr())  # fmt: skip

        def ffn_args(outs):
            out, res_out, post_out, comb_out, pre_out = outs
            return (res.data_ptr(), post.data_ptr(), comb.data_ptr(), pre.data_ptr(),
                    mw.hc_ffn_fn.data_ptr(), mw.hc_ffn_scale.data_ptr(), mw.hc_ffn_base.data_ptr(),
                    mw.ffn_norm.data_ptr(), res_out.data_ptr(), post_out.data_ptr(),
                    comb_out.data_ptr(), pre_out.data_ptr(), mw.gate_w.data_ptr(),
                    mw.bias.data_ptr(), mw.w13.data_ptr(), mw.w13_s.data_ptr(), mw.w2.data_ptr(),
                    mw.w2_s.data_ptr(), mw.sgu.data_ptr(), mw.sgu_s.data_ptr(),
                    mw.sw2.data_ptr(), mw.sw2_s.data_ptr(), out.data_ptr(), sc.data_ptr(),
                    runner.peer.local, runner.peer.addresses.data_ptr(), rank,
                    runner.epoch.data_ptr(), 0)  # fmt: skip

        k2_fly = runner.kernels(M, ratio)[1]
        fly = runner._outs(M, res)
        torch.accelerator.synchronize()
        dist.barrier()
        k2_fly(*attn_args(), *ffn_args(fly), stream=torch.cuda.current_stream())
        torch.accelerator.synchronize()
        s_fly = snap()
        poison(sc, lay, ("part", "pstat", "xo", "xos", "x8b", "x8bs", "normed", "x8r", "x8rs",
                         "x8", "x8s"))  # fmt: skip
        mine = runner._outs(M, res)
        c = build_constexprs(M, tp, ratio)
        INTS = {2, 3, 5, 6, 15, 18}  # strides, blocks, bt_stride, qk_scale

        def run_gl(outs):
            aa = [a if i in INTS else ptr(a) for i, a in enumerate(attn_args())]
            fa = list(ffn_args(outs))
            fa = [ptr(a) for a in fa[:26]] + [fa[26], ptr(fa[27]), fa[28]]
            k2[(256,)](*aa, *fa, ptr(health.data_ptr()), **c, num_warps=8, enable_fp_fusion=False)

        dist.barrier()
        run_gl(mine)
        torch.accelerator.synchronize()
        s_gl = snap()
        rep = [f"rank {rank} S={M} ratio={ratio}: health {health.tolist()}"]
        for name, a, b in (("res_out", fly[1], mine[1]), ("post", fly[2], mine[2]),
                           ("comb", fly[3], mine[3]), ("pre", fly[4], mine[4])):  # fmt: skip
            ok = torch.equal(a, b)
            bad += not ok
            rep.append(f"  {name:8s} bitwise {'OK' if ok else 'DIFF'}")
        # the live chunks' partials and stats (a dead chunk is never written)
        ng = 2 if heads // 16 == 2 and M * 10 <= 256 else 1
        ck = 16 * (8 // ng)
        nch = -(-KEYS // ck)
        live = (torch.arange(nch, device=device)[None, :] * ck < klen[:, None])
        plain = {"part", "pstat", "xo", "xos", "x8b", "x8bs", "normed", "x8r", "x8rs", "x8", "x8s"}
        for k in ("part", "pstat", "xo", "xos", "x8b", "x8bs", "lin", "pmix", "normed", "x8r",
                  "x8rs", "x8", "x8s", "logit", "route", "smid", "smq", "smc"):  # fmt: skip
            a, b = s_fly[k], s_gl[k]
            if k == "part":
                a, b = (x.view(torch.int16)[: M * nch * heads * HEAD_DIM].view(M, nch, -1)[live]
                        for x in (a, b))  # fmt: skip
            elif k == "pstat":
                a, b = (x.view(torch.int32)[: M * nch * heads * 2].view(M, nch, -1)[live]
                        for x in (a, b))  # fmt: skip
            elif k not in plain:
                a, b = values(a), values(b)
            ok = torch.equal(a, b)
            bad += not ok
            nd = int((a != b).sum()) if not ok else 0
            rep.append(f"  {k:8s} bitwise {'OK' if ok else f'DIFF ({nd} of {a.numel()})'}")
        cos = torch.nn.functional.cosine_similarity(fly[0].float().flatten(),
                                                    mine[0].float().flatten(), dim=0).item()  # fmt: skip
        rel = ((fly[0].float() - mine[0].float()).norm() / fly[0].float().norm()).item()
        ok = cos > 0.9999 and rel < 1e-2
        bad += not ok
        rep.append(f"  out      cos {cos:.6f} rel {rel:.2e} {'OK' if ok else 'BAD'}")
        print("\n".join(rep), flush=True)
        if os.environ.get("DUMP") and rank == 0:
            torch.save({"fly": {k: s_fly[k].cpu() for k in ("part", "pstat", "xo", "xos")},
                        "gl": {k: s_gl[k].cpu() for k in ("part", "pstat", "xo", "xos")},
                        "kt": kt.cpu(), "klen": klen.cpu(), "heads": heads},
                       f"/cache/k2dump_S{M}.pt")  # fmt: skip
        if os.environ.get("BENCH"):
            # FlyDSL launches on the stream it is given (the capturing one)
            bench(M, rank, lambda: k2_fly(*attn_args(), *ffn_args(fly),
                                         stream=torch.cuda.current_stream()),
                  lambda: run_gl(mine))  # fmt: skip
    dist.barrier()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
