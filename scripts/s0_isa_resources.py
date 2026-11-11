#!/usr/bin/env python3
"""S0: compile-only ISA dump + resources for incumbent packed-BN64, base_pr and champion_exact.

Orchestrator:  python scripts/s0_isa_resources.py --out bench_out/s0
Worker (internal): --one <kind> <G> <causal> <dtype> <shuffled> <thr> <outdir>
kind: inc | base | champ | rtbm<N> (router, non-packed, block_m N) | dense.
No GPU needed (CPU tensors, COMPILE_ONLY, fresh FlyDSL cache per job).

Dense/non-bn64 ISA gate (replaces fmha_isa_gate.py): every kernel outside packed BN64 must stay identical.
  python scripts/s0_isa_resources.py --dense --out bench_out/s0/dense_base           # snapshot
  python scripts/s0_isa_resources.py --dense --out bench_out/s0/dense_new --compare bench_out/s0/dense_base
"""

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

W = Path(__file__).resolve().parent.parent
BASE = "/home/jograner/.claude/jobs/178b5e41/tmp/author-results/silotiger-877-flydsl-prefill-attention/base_pr.py"
sys.path.insert(0, str(W / "scripts"))


def worker(kind, G, causal, dtype, shuf, thr, outdir):
    import torch

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = outdir / "cache"
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(cache)
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"  # needed to read the artifact back
    os.environ["COMPILE_ONLY"] = "1"
    os.environ["FLYDSL_GPU_ARCH"] = "gfx950"
    sys.path.insert(0, str(W))
    import fmha_isa_gate as gate

    gate.LLVM_BIN_CANDIDATES = (
        "/usr/local/lib/python3.14/dist-packages/_rocm_sdk_devel/lib/llvm/bin",
    )
    fp8 = torch.float8_e4m3fn
    if kind == "inc" or kind.startswith("rtbm"):
        from aiter.aot.flydsl import unified_attention as ua

        packed = kind == "inc"
        job = {
            "num_heads": 4 * G,
            "num_kv_heads": 4,
            "causal": causal,
            "out_dtype_str": dtype,
            "shuffled_kv_cache": shuf,
            "rescale_threshold": thr,
            "block_m": 128 if packed else int(kind[4:]),
            "packed_bn64": packed,
        }
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            ua._compile_prefill(job)
    elif kind == "dense":
        # flydsl_flash_attn_fp8_func's dense build; G = num_heads / num_kv_heads, Sq = Skv = 512.
        from torch._subclasses.fake_tensor import FakeTensorMode

        from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import _build_fp8

        H, B, S = 8 * G, 2, 512
        exe = _build_fp8(
            num_heads=H,
            num_kv_heads=8,
            causal=causal,
            rescale_threshold=thr,
            daz=True,
            lazy_rescale=True,
            setprio=True,
            enable_stagger=True,
            block_m=256 if causal else 128,
            out_dtype=dtype,
        )
        with FakeTensorMode():
            q = torch.empty((B * S * H * 128,), dtype=fp8)
            kv = torch.empty((B * S * 8 * 128,), dtype=fp8)
            o = torch.empty(
                (B * S * H * 128,),
                dtype=torch.bfloat16 if dtype == "bf16" else torch.float16,
            )
            d = torch.ones((1,), dtype=torch.float32)
            exe(
                q.view(torch.int8),
                kv.view(torch.int8),
                kv.view(torch.int8),
                o,
                B,
                S,
                softmax_scale=128**-0.5,
                q_descale=d,
                k_descale=d,
                v_descale=d,
            )
    else:
        if kind == "base":
            spec = importlib.util.spec_from_file_location("base_pr", BASE)
        else:
            spec = importlib.util.spec_from_file_location(
                "champion_exact", str(W / "champion_exact.py")
            )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        ns, mq, nb, mb = 4, 4, 8, 4
        q = torch.empty((ns * mq, 64, 128), dtype=fp8)
        out = torch.empty((ns * mq, 64, 128), dtype=torch.bfloat16)
        if shuf:
            k = torch.empty((nb, 4, 8, 64, 16), dtype=fp8)
            v = torch.empty((nb, 4, 4, 128, 16), dtype=fp8)
        else:
            k = torch.empty((nb, 64, 4, 128), dtype=fp8)
            v = torch.empty((nb, 64, 4, 128), dtype=fp8)
        cuq = torch.arange(0, (ns + 1) * mq, mq, dtype=torch.int32)
        skv = torch.full((ns,), mb * 64, dtype=torch.int32)
        bt = torch.zeros((ns, mb), dtype=torch.int32)
        d = torch.ones((1,), dtype=torch.float32)
        # threshold: 6.0 if max_seqlen_k <= 4096 else 4.0
        msk = 256 if thr == 6.0 else 8192
        m.attention(q, k, v, out, cuq, skv, bt, d, d, d, mq, msk, 128**-0.5)
    tools = (gate.llvm_tool("llvm-objdump"), gate.llvm_tool("llvm-readelf"))
    blobs = gate.code_objects(cache)
    for i, b in enumerate(blobs):
        gate.dump_one(b, outdir / f"k__{i}", tools)
    shutil.rmtree(cache, ignore_errors=True)
    print("blobs", len(blobs))
    return 0 if blobs else 1


def note_val(txt, key):
    mm = re.search(rf"\.{key}:\s+(\S+)", txt)
    return mm.group(1) if mm else None


def resources(outdir):
    n = (Path(outdir) / "k__0.notes").read_text()
    ins = [
        l
        for l in (Path(outdir) / "k__0.s").read_text().splitlines()
        if l.startswith("\t") and not l.strip().startswith(("s_endpgm", ".", ";"))
    ]
    g = lambda k: note_val(n, k)
    return {
        "vgpr": g("vgpr_count"),
        "agpr": g("agpr_count"),
        "sgpr": g("sgpr_count"),
        "lds": g("group_segment_fixed_size"),
        "scratch": g("private_segment_fixed_size"),
        "spill_v": g("vgpr_spill_count"),
        "spill_s": g("sgpr_spill_count"),
        "ninsn": len(ins),
        "wavefront": g("wavefront_size"),
    }


def norm(outdir):
    s = (Path(outdir) / "k__0.s").read_text()
    s = re.sub(r"flash_attn\w*|_Z\w+|flydsl_unified\w*", "KNAME", s)
    return s.splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(W / "bench_out/s0"))
    ap.add_argument("--one", nargs=7)
    ap.add_argument(
        "--dense", action="store_true", help="non-bn64 set (router groups 1/2, dense)"
    )
    ap.add_argument(
        "--packed", action="store_true", help="incumbent packed-bn64 set only"
    )
    ap.add_argument("--compare", help="dir of a previous --dense run to diff against")
    a = ap.parse_args()
    if a.one:
        kind, G, c, dt, sh, thr, od = a.one
        sys.exit(worker(kind, int(G), c == "1", dt, sh == "1", float(thr), od))
    out = Path(a.out).resolve() / ("" if a.dense else "s0_isa")
    jobs = []
    if a.dense:
        for G in (1, 2):
            for c in (1, 0):
                for sh in (0, 1):
                    for bm in (128, 256):
                        jobs.append((f"rtbm{bm}", G, c, "bf16", sh, 6.0))
            jobs.append(("rtbm128", G, 1, "f16", 0, 4.0))
        for c in (1, 0):
            jobs.append(("dense", 1, c, "bf16", 0, 6.0))
            jobs.append(("dense", 2, c, "bf16", 0, 6.0))
        jobs.append(("dense", 1, 1, "f16", 0, 4.0))
    else:
        for G in (4, 8, 16):
            for c in (1, 0):
                for sh in (0, 1):
                    jobs.append(("inc", G, c, "bf16", sh, 6.0))
        for c in (1, 0):
            for sh in (0, 1):
                jobs.append(("inc", 16, c, "f16", sh, 6.0))
        for sh in (0, 1):
            for thr in (6.0, 4.0):
                jobs.append(("inc", 16, 1, "bf16", sh, thr) if thr == 4.0 else None)
                if not a.packed:
                    jobs.append(("base", 16, 1, "bf16", sh, thr))
                    jobs.append(("champ", 16, 1, "bf16", sh, thr))
    jobs = [j for j in jobs if j]
    env = dict(
        os.environ, ENABLE_CK="0", FLYDSL_RUNTIME_ENABLE_CACHE="0", PYTHONPATH=str(W)
    )

    def run(j):
        kind, G, c, dt, sh, thr = j
        name = f"{kind}_G{G}_c{c}_{dt}_sh{sh}_t{thr:g}"
        od = out / name
        shutil.rmtree(od, ignore_errors=True)
        r = subprocess.run(
            [
                sys.executable,
                __file__,
                "--one",
                kind,
                str(G),
                str(c),
                dt,
                str(sh),
                str(thr),
                str(od),
            ],
            env=env,
            cwd="/tmp",
            capture_output=True,
            check=False,
            text=True,
        )
        (out / f"{name}.log").write_text(r.stdout + r.stderr)
        return name, r.returncode

    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(run, jobs))
    print("| config | rc | vgpr | agpr | sgpr | lds | scratch | spill v/s | ninsn |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, rc in sorted(res):
        if rc:
            print(f"| {name} | FAIL {rc} |")
            continue
        r = resources(out / name)
        print(
            f"| {name} | 0 | {r['vgpr']} | {r['agpr']} | {r['sgpr']} | {r['lds']} | {r['scratch']} | {r['spill_v']}/{r['spill_s']} | {r['ninsn']} |"
        )
    print()
    if a.dense:
        if a.compare:
            ref_root = Path(a.compare).resolve()
            bad = 0
            for name, rc in sorted(res):
                fa, fb = ref_root / name / "k__0.s", out / name / "k__0.s"
                same = fa.exists() and fb.exists() and fa.read_text() == fb.read_text()
                bad += not same
                print(f"{name}: {'IDENTICAL' if same else 'DIFFER'}")
            print("DENSE GATE", "PASS" if bad == 0 else f"FAIL ({bad})")
        return
    if a.packed:
        return
    for sh in (0, 1):
        for thr in (6.0, 4.0):
            ref = f"base_G16_c1_bf16_sh{sh}_t{thr:g}"
            for other in (
                f"inc_G16_c1_bf16_sh{sh}_t{thr:g}",
                f"champ_G16_c1_bf16_sh{sh}_t{thr:g}",
            ):
                if (
                    not (out / ref / "k__0.s").exists()
                    or not (out / other / "k__0.s").exists()
                ):
                    print(other, "vs", ref, "MISSING")
                    continue
                A, B = norm(out / ref), norm(out / other)
                import difflib

                if A == B:
                    print(f"{other} vs {ref}: IDENTICAL ({len(A)} lines)")
                else:
                    d = [
                        l
                        for l in difflib.unified_diff(A, B, lineterm="", n=0)
                        if not l.startswith(("---", "+++"))
                    ]
                    first = next((l for l in d if l.startswith("@@")), "")
                    print(
                        f"{other} vs {ref}: DIFFER lines {len(A)} -> {len(B)} (delta {len(B)-len(A)}), diff lines {len(d)}, first {first}"
                    )
                    (out / f"{other}__vs__{ref}.diff").write_text("\n".join(d))


main()
