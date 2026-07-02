#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dump gfx1250 (MI450) grouped-MoE a8w4/fp4 GEMM ISA on any box (no launch).

Cross-compiles stage1 (fused gate/up) and stage2 (down) for a given MoE problem
size and reports per-kernel resource usage + a first-order occupancy estimate.

Mechanism (works on a non-gfx1250 box, e.g. gfx942):
  ARCH=gfx1250      -> FlyDSL/LLVM backend cross-emits gfx1250
  COMPILE_ONLY=1    -> each @flyc.jit compiles then returns (no dispatch/launch)
  FLYDSL_DUMP_IR=1  -> dump MLIR stages + 21_final_isa.s into FLYDSL_DUMP_DIR
  + patch arch detection so the aiter gfx1250 gates/asserts pass.

Defaults reproduce this command's exact kernel config (cfg_row=None -> defaults):

  test_flydsl_grouped_gemm_gfx1250.py --scenario kernel --data-format a8w4 \
    --layout gugu --experts 256 --tokens 64 --topk 6 --model-dim 4096 \
    --inter-dim 2048 --act swiglu --real-gemm --wst

i.e. a8w4 / gugu / swiglu / tile 64x256x256 / num_buffers=2 / contiguous-M
(token>16) / wave-specialized-TDM (--wst). Override any of these via flags to
sweep tiles or force the persist scheduler.

Examples:
  python dump_isa.py                       # the command above
  python dump_isa.py --persist             # persist scheduler (dense masked)
  python dump_isa.py --tile-m 16 --no-wst  # sweep a smaller M tile
"""

from __future__ import annotations

import argparse
import math
import os
import re


# gfx1250 = MI450 (WGP arch, wave32). LDS from flydsl SMEM_CAPACITY_MAP;
# VGPR file per SIMD32 confirmed at 1024; a WGP has 4x SIMD32 and one unified
# 320KB LDS. Occupancy is counted per WGP.
MI450_LDS_PER_WGP = 327680      # 320 KB unified WGP$
MI450_VGPR_PER_SIMD32 = 1024
MI450_SIMD32_PER_WGP = 4


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-format", choices=("a8w4", "fp4"), default="a8w4")
    p.add_argument("--layout", choices=("gguu", "gugu"), default="gugu",
                   help="stage1 weight layout (gugu=INTERLEAVE, gguu=SEPARATED)")
    p.add_argument("--act", choices=("swiglu", "silu"), default="swiglu")
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--model-dim", type=int, default=4096)
    p.add_argument("--inter-dim", type=int, default=2048)
    p.add_argument("--max-m", type=int, default=0,
                   help="per-expert max_m; 0 = derive (contiguous: tokens*topk).")
    p.add_argument("--tile-m", type=int, default=64)
    p.add_argument("--tile-n", type=int, default=256)
    p.add_argument("--tile-k", type=int, default=256)
    p.add_argument("--m-warp", type=int, default=1)
    p.add_argument("--n-warp", type=int, default=4)
    p.add_argument("--num-buffers", type=int, default=2)
    p.add_argument("--out-dtype", choices=("bf16", "f16"), default="bf16")
    # Scheduler: default mirrors the command (contiguous-M, non-persist).
    p.add_argument("--persist", action="store_true",
                   help="grouped_persistent_m=True (dense masked; disables contiguous)")
    p.add_argument("--persist-mn", action="store_true",
                   help="fold M and N into one persist stream (implies --persist; "
                        "same as env AITER_GROUPED_PERSIST_MN=1)")
    p.add_argument("--contiguous", dest="contiguous", action="store_true", default=None,
                   help="force DeepGEMM contiguous-M (default: on when tokens>16)")
    p.add_argument("--no-contiguous", dest="contiguous", action="store_false")
    p.add_argument("--wst", dest="wst", action="store_true", default=True,
                   help="wave-specialized TDM (default on, matches --wst)")
    p.add_argument("--no-wst", dest="wst", action="store_false")
    p.add_argument("--as-prologue", action="store_true", help="tdm_as_in_prologue")
    p.add_argument("--stage", choices=("both", "1", "2"), default="both")
    p.add_argument("--dump-dir", default="/tmp/dump_isa")
    p.add_argument("--wgp-num", type=int, default=0,
                   help="MI450 WGP count, for a persistent_workers recommendation "
                        "(persistent_workers = occ x WGP).")
    return p.parse_args()


def _setup_env(dump_dir: str) -> None:
    os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")
    os.environ["ARCH"] = "gfx1250"
    os.environ["COMPILE_ONLY"] = "1"
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = dump_dir


def _patch_arch() -> None:
    import flydsl.runtime.device as _dev

    _dev.get_rocm_arch = lambda *a, **k: "gfx1250"
    import aiter.ops.flydsl.kernels.gemm_mxscale_gfx1250 as _gm

    _gm.get_hip_arch = lambda *a, **k: "gfx1250"


def _parse_isa(path: str) -> dict:
    keys = {
        "vgpr": r"\.vgpr_count:\s*(\d+)",
        "vgpr_spill": r"\.vgpr_spill_count:\s*(\d+)",
        "sgpr": r"\.sgpr_count:\s*(\d+)",
        "sgpr_spill": r"\.sgpr_spill_count:\s*(\d+)",
        "lds": r"\.group_segment_fixed_size:\s*(\d+)",
        "wg": r"\.max_flat_workgroup_size:\s*(\d+)",
    }
    with open(path) as f:
        text = f.read()
    out = {}
    for name, pat in keys.items():
        m = re.search(pat, text)
        out[name] = int(m.group(1)) if m else None
    return out


def _occupancy(res: dict) -> dict:
    """First-order MI450 occupancy (workgroups per WGP), min over VGPR and LDS.

    wave32; a workgroup of wg_waves spreads across the WGP's 4 SIMD32. Each
    SIMD32 hosts ceil(wg_waves / 4) waves of the workgroup, so the VGPR budget
    it consumes there is vgpr * waves_per_simd. LDS is a single 320KB pool per
    WGP shared by all resident workgroups."""
    vgpr, lds, wg = res["vgpr"], res["lds"], res["wg"]
    if not (vgpr and lds and wg):
        return {}
    wg_waves = max(1, wg // 32)
    waves_per_simd = math.ceil(wg_waves / MI450_SIMD32_PER_WGP)
    occ_vgpr = MI450_VGPR_PER_SIMD32 // (vgpr * waves_per_simd)
    occ_lds = MI450_LDS_PER_WGP // lds
    return {"occ_vgpr": occ_vgpr, "occ_lds": occ_lds,
            "occ": max(0, min(occ_vgpr, occ_lds))}


def main() -> None:
    args = _parse_args()
    _setup_env(args.dump_dir)

    import torch
    _patch_arch()
    import aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250 as mg

    E = args.experts
    K, INTER = args.model_dim, args.inter_dim
    tile_m = args.tile_m

    if args.persist_mn:
        args.persist = True  # MN-persist is a variant of the persist scheduler
    contiguous = args.contiguous
    if contiguous is None:
        contiguous = args.tokens > 16
    if args.persist:
        contiguous = False  # persist and contiguous are mutually exclusive

    if args.max_m > 0:
        raw_max_m = args.max_m
    elif contiguous:
        raw_max_m = args.tokens * args.topk
    else:
        raw_max_m = args.tokens
    warp_tile_m = tile_m // args.m_warp
    MAXM = max(warp_tile_m, math.ceil(raw_max_m / warp_tile_m) * warp_tile_m)

    print(f"[cfg] data_format={args.data_format} layout={args.layout} act={args.act}")
    print(f"[cfg] E={E} tokens={args.tokens} topk={args.topk} "
          f"model_dim={K} inter_dim={INTER} max_m={MAXM}")
    print(f"[cfg] tile={tile_m}x{args.tile_n}x{args.tile_k} "
          f"m_warp={args.m_warp} n_warp={args.n_warp} num_buffers={args.num_buffers}")
    print(f"[cfg] persist={args.persist} persist_mn={args.persist_mn} "
          f"contiguous={contiguous} wst={args.wst} as_prologue={args.as_prologue}")

    dev = "cuda"
    i32, u8 = torch.int32, torch.uint8
    outdt = torch.bfloat16 if args.out_dtype == "bf16" else torch.float16
    pack = 2 if args.data_format == "fp4" else 1

    def _cfg(act, layout):
        return mg._GroupedA8W4Config(
            model_dim=K, inter_dim=INTER, experts=E, max_m=MAXM,
            tile_m=tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
            m_warp=args.m_warp, n_warp=args.n_warp, num_buffers=args.num_buffers,
            waves_per_eu=None, out_dtype=args.out_dtype,
            use_tdm_store=True, inst_prefetch=False,
            wave_specialized_tdm=args.wst, tdm_as_in_prologue=args.as_prologue,
            split_k=1, cluster_m=1, cluster_n=1, use_scale_opsel=False,
            expert_sched_mode=False, grouped_persistent_m=args.persist,
            grouped_persistent_mn=args.persist_mn,
            grouped_contiguous_m=contiguous, persistent_workers=None,
            data_format=args.data_format, act=act, stage1_weight_layout=layout)

    def _trigger(launch, N, two_n, Kk):
        """Call the mode-appropriate launcher with example tensors to force a
        trace+compile (COMPILE_ONLY returns before any real dispatch)."""
        ntiles = E * math.ceil(MAXM / tile_m)
        c = torch.zeros((E, MAXM, N), dtype=outdt, device=dev)
        a = torch.zeros((E, MAXM, Kk // pack), dtype=u8, device=dev)
        b = torch.zeros((E, two_n, Kk // 2), dtype=u8, device=dev)
        a_scale = torch.zeros((E, MAXM, Kk // 32), dtype=u8, device=dev)
        b_scale = torch.zeros((E, two_n // 32, (Kk // 32) * 32), dtype=u8, device=dev)
        masked_m = torch.zeros((E,), dtype=i32, device=dev)
        m_tile_prefix = torch.zeros((E + 1,), dtype=i32, device=dev)
        m_tile_map = torch.zeros((ntiles,), dtype=i32, device=dev)
        stream = torch.cuda.current_stream()
        common = (c, a, b, a_scale, b_scale, masked_m, m_tile_prefix, m_tile_map)
        if args.persist:
            # launch_mxscale_gemm_masked_persistent(..., m, n, swiglu, stream)
            launch(*common, MAXM, N, 0.0, stream=stream)
        else:
            # launch_mxscale_gemm_masked(..., m_tile_bound, m, n, swiglu, stream)
            launch(*common, ntiles, MAXM, N, 0.0, stream=stream)

    jobs = []
    if args.stage in ("both", "1"):
        two_n = 2 * INTER if args.layout == "gugu" else INTER
        stage1_n = INTER  # fused gate/up output width
        jobs.append(("stage1", args.act, args.layout, K, stage1_n, two_n,
                     f"dumpisa_gemm1_{args.data_format}_{args.layout}_{args.act}"
                     f"_{'persistMN' if args.persist_mn else ('persist' if args.persist else ('contig' if contiguous else 'dense'))}"
                     f"{'_wst' if args.wst else ''}_t{tile_m}x{args.tile_n}x{args.tile_k}"))
    if args.stage in ("both", "2"):
        jobs.append(("stage2", "silu", "gguu", INTER, K, K,
                     f"dumpisa_gemm2_{args.data_format}"
                     f"_{'persistMN' if args.persist_mn else ('persist' if args.persist else ('contig' if contiguous else 'dense'))}"
                     f"{'_wst' if args.wst else ''}_t{tile_m}x{args.tile_n}x{args.tile_k}"))

    compiled_tags = []
    for name, act, layout, Kk, N, two_n, tag in jobs:
        stage1_act = act if name == "stage1" else None
        print(f"\n[dump] compiling {name}: K={Kk} N={N} tag={tag}", flush=True)
        cfg = _cfg(act, layout)
        launch = mg._compile_base_a8w4_gemm(
            K=Kk, N=N, cfg=cfg, stage1_act=stage1_act,
            stage1_weight_layout=layout, kernel_tag=tag)
        try:
            _trigger(launch, N=N, two_n=two_n, Kk=Kk)
            print(f"[dump] {name} compiled", flush=True)
        except Exception as exc:  # noqa: BLE001
            # A trace-time error can still leave a valid final_isa.s behind.
            print(f"[dump] {name} trigger raised (ISA may still be dumped): {exc!r}",
                  flush=True)
        compiled_tags.append(tag)

    print("\n" + "=" * 78)
    print(f"{'kernel':<52} {'VGPR':>5} {'SGPR':>5} {'LDS(KB)':>8} {'occ/WGP':>7}")
    print("-" * 78)
    for root, _dirs, files in sorted(os.walk(args.dump_dir)):
        if "21_final_isa.s" not in files:
            continue
        tagname = os.path.basename(root)
        if not any(t in tagname for t in compiled_tags):
            continue
        path = os.path.join(root, "21_final_isa.s")
        r = _parse_isa(path)
        occ = _occupancy(r)
        vs = f"{r['vgpr']}" + (f"+{r['vgpr_spill']}sp" if r.get("vgpr_spill") else "")
        ss = f"{r['sgpr']}" + (f"+{r['sgpr_spill']}sp" if r.get("sgpr_spill") else "")
        lds_kb = f"{r['lds'] / 1024:.0f}" if r["lds"] else "?"
        occ_s = (f"{occ['occ']} (v{occ['occ_vgpr']}/l{occ['occ_lds']})"
                 if occ else "?")
        print(f"{tagname[:52]:<52} {vs:>5} {ss:>5} {lds_kb:>8} {occ_s:>7}")
        if args.wgp_num and occ:
            print(f"{'':<52} -> persistent_workers = {occ['occ'] * args.wgp_num} "
                  f"(= occ {occ['occ']} x {args.wgp_num} WGP)")
    print("=" * 78)
    print(f"ISA + MLIR stages under: {args.dump_dir}")
    print(f"MI450 limits: LDS={MI450_LDS_PER_WGP}B/WGP  "
          f"VGPR={MI450_VGPR_PER_SIMD32}/SIMD32  {MI450_SIMD32_PER_WGP} SIMD32/WGP")


if __name__ == "__main__":
    main()
