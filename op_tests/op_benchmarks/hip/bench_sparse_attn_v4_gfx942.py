#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Verify and benchmark the gfx942 HIP sparse_attn_v4_paged_2src kernel (Paths A & C).

Checks accuracy against ATOM's Triton `sparse_attn_v4_paged_2src` (+ optional
fused `inverse_rope`) across decode (CSA+SWA, SWA-only, varlen, -1 sentinels,
all-invalid, H=8/16/32) and prefill (M=1808, 8192), then benchmarks both.

    python op_tests/op_benchmarks/hip/bench_sparse_attn_v4_gfx942.py
"""

import argparse
import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, "/app/ATOM")

import torch
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.model_ops.v4_kernels import (
    sparse_attn_v4_paged_2src,
    sparse_attn_v4_paged_2src_reference,
)

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
SOURCE = os.path.join(REPO, "csrc", "kernels", "sparse_attn_v4_gfx942_kernels.cu")
D = 512
ROPE_DIM = 64
SCALE = D ** -0.5


def build_so(source=SOURCE):
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    with open(source, "rb") as f:
        digest = hashlib.sha1(f.read() + arch.encode()).hexdigest()[:12]
    out = os.path.join(tempfile.gettempdir(), f"sparse_attn_v4_{arch}_{digest}.so")
    if not os.path.exists(out):
        tmp = f"{out}.{os.getpid()}"
        subprocess.run(
            [
                "hipcc", "-O3", f"--offload-arch={arch}", "-std=c++17",
                "-shared", "-fPIC", "-Wno-hip-deprecated-builtins",
                "-o", tmp, source,
            ],
            check=True,
        )
        os.replace(tmp, out)
    return out


class HipSparseAttnV4:
    """ctypes wrapper for sparse_attn_v4_gfx942_kernels.cu with cached workspace."""

    def __init__(self, so_path=None):
        path = build_so() if so_path is None else so_path
        lib = ctypes.CDLL(path)
        self.splits_for = lib.sparse_attn_v4_gfx942_splits
        self.splits_for.restype = ctypes.c_int
        self.splits_for.argtypes = [ctypes.c_int] * 4

        self.ws_bytes = lib.sparse_attn_v4_gfx942_workspace_bytes
        self.ws_bytes.restype = ctypes.c_longlong
        self.ws_bytes.argtypes = [ctypes.c_int] * 3

        self.launch = lib.sparse_attn_v4_gfx942_launch
        self.launch.restype = ctypes.c_int
        self.launch.argtypes = (
            [ctypes.c_void_p] * 9
            + [ctypes.c_longlong] * 6
            + [ctypes.c_int] * 3
            + [ctypes.c_float]
            + [ctypes.c_void_p] * 4
            + [ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p]
        )
        self.num_cus = torch.cuda.get_device_properties(0).multi_processor_count
        self._ws = {}

    def _workspace(self, num_tokens, num_heads, num_splits):
        if num_splits <= 1:
            return 0
        key = (num_tokens, num_heads, num_splits)
        if key not in self._ws:
            nbytes = self.ws_bytes(num_tokens, num_heads, num_splits)
            self._ws[key] = torch.zeros(max(nbytes, 1), dtype=torch.uint8, device="cuda")
        return self._ws[key].data_ptr()

    def __call__(
        self,
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out=None,
        num_splits=0,
        cos_cache=None,
        sin_cache=None,
        positions=None,
        rope_dim=0,
    ):
        T, H, D_in = q.shape
        assert D_in == D
        if out is None:
            out = torch.empty_like(q)
        if num_splits <= 0:
            avg_k = (kv_indices_prefix.numel() + kv_indices_extend.numel()) // max(T, 1)
            num_splits = self.splits_for(T, H, avg_k, self.num_cus)
        ws_ptr = self._workspace(T, H, num_splits)
        rc = self.launch(
            q.data_ptr(),
            unified_kv.data_ptr(),
            kv_indices_prefix.data_ptr(),
            kv_indptr_prefix.data_ptr(),
            kv.data_ptr(),
            kv_indices_extend.data_ptr(),
            kv_indptr_extend.data_ptr(),
            attn_sink.data_ptr(),
            out.data_ptr(),
            q.stride(0),
            q.stride(1),
            unified_kv.stride(0),
            kv.stride(0),
            out.stride(0),
            out.stride(1),
            T,
            H,
            num_splits,
            float(softmax_scale),
            ws_ptr,
            0 if cos_cache is None else cos_cache.data_ptr(),
            0 if sin_cache is None else sin_cache.data_ptr(),
            0 if positions is None else positions.data_ptr(),
            0 if cos_cache is None else cos_cache.stride(0),
            int(rope_dim),
            torch.cuda.current_stream().cuda_stream,
        )
        if rc != 0:
            raise RuntimeError(f"sparse_attn_v4_gfx942_launch returned {rc}")
        return out


def make_case(T, H, csa_lens, swa_lens, seed=0, inject_sentinels=False, pool_cap=65536):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5
    sink = torch.randn(H, device="cuda", dtype=torch.float32, generator=g) * 0.25

    def build_src(lens, cap):
        tot = sum(lens)
        if tot == 0:
            return (
                torch.empty(1, D, device="cuda", dtype=torch.bfloat16),
                torch.empty(0, device="cuda", dtype=torch.int32),
                torch.zeros(T + 1, device="cuda", dtype=torch.int32),
            )
        rows = min(cap, max(tot, 64))
        pool = torch.randn(rows, D, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5
        idx = torch.randint(0, rows, (tot,), device="cuda", dtype=torch.int32, generator=g)
        if inject_sentinels and tot > 4:
            idx[::7] = -1
        ptr = torch.tensor([0] + lens, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
        return pool, idx, ptr

    cpool, cidx, cptr = build_src(csa_lens, pool_cap)
    spool, sidx, sptr = build_src(swa_lens, pool_cap)
    return q, cpool, cidx, cptr, spool, sidx, sptr, sink


def bench_fn(fn, graph=True, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if graph:
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            fn()
        torch.cuda.synchronize()
        fn = cg.replay
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def verify(hip_op):
    rope = RotaryEmbedding(ROPE_DIM, 102400, base=160000).to("cuda")
    cases = [
        ("decode_c1_csa_swa", 1, 16, [512], [128], 0, False),
        ("decode_c4_csa_swa", 4, 16, [512] * 4, [128] * 4, 0, False),
        ("decode_c8_csa_swa", 8, 16, [512] * 8, [128] * 8, 0, False),
        ("decode_c1_swa_only", 1, 16, [0], [128], 0, False),
        ("decode_c8_swa_only", 8, 16, [0] * 8, [128] * 8, 0, False),
        ("decode_c1_csa_only", 1, 16, [512], [0], 0, False),
        ("decode_varlen_ragged", 6, 16, [0, 17, 512, 333, 64, 511], [128, 3, 128, 19, 0, 127], 0, False),
        ("decode_sentinels", 4, 16, [200, 512, 16, 45], [128, 64, 16, 12], 0, True),
        ("decode_all_empty", 2, 16, [0, 0], [0, 0], 4, False),
        ("decode_h8_tp8", 4, 8, [512] * 4, [128] * 4, 0, False),
        ("decode_h32_tp2", 4, 32, [512] * 4, [128] * 4, 0, False),
        ("decode_unsplit_s1", 4, 16, [512] * 4, [128] * 4, 1, False),
        ("decode_split_s10", 8, 16, [512] * 8, [128] * 8, 10, False),
        ("decode_split_s20", 1, 16, [512], [128], 20, False),
        ("prefill_256_csa_swa", 256, 16, [512] * 256, [128] * 256, 1, False),
        ("prefill_256_swa", 256, 16, [0] * 256, [128] * 256, 1, False),
    ]
    print("=== Accuracy Verification (HIP vs Triton sparse_attn_v4_paged_2src + iRoPE) ===")
    for idx_c, (name, T, H, cl, sl, ns, sent) in enumerate(cases):
        q, cp, ci, cptr, sp, si, sptr, sink = make_case(T, H, cl, sl, seed=100 + idx_c, inject_sentinels=sent)
        ref = (
            sparse_attn_v4_paged_2src_reference(q, cp, ci, cptr, sp, si, sptr, sink, SCALE)
            if sent
            else sparse_attn_v4_paged_2src(q, cp, ci, cptr, sp, si, sptr, sink, SCALE)
        )
        got = hip_op(q, cp, ci, cptr, sp, si, sptr, sink, SCALE, num_splits=ns)
        got2 = hip_op(q, cp, ci, cptr, sp, si, sptr, sink, SCALE, num_splits=ns)
        err = (got.float() - ref.float()).abs().max().item()
        err2 = (got2.float() - ref.float()).abs().max().item()

        pos = torch.randint(0, 100000, (T,), device="cuda", dtype=torch.int64)
        ref_ir = rope(ref.unsqueeze(0).clone(), pos, inverse=True).squeeze(0)
        got_ir = hip_op(
            q, cp, ci, cptr, sp, si, sptr, sink, SCALE, num_splits=ns,
            cos_cache=rope.cos_cache, sin_cache=rope.sin_cache, positions=pos, rope_dim=ROPE_DIM,
        )
        err_ir = (got_ir.float() - ref_ir.float()).abs().max().item()
        ok = err <= 2e-2 and err2 <= 2e-2 and err_ir <= 2e-2
        print(f"  [{('PASS' if ok else 'FAIL')}] {name:<22} max_abs={err:.4e} (2nd={err2:.4e}) fused_iRoPE={err_ir:.4e}")
        if not ok:
            raise AssertionError(f"Accuracy check failed on {name}: err={err}, err2={err2}, err_ir={err_ir}")
    print("All accuracy checks PASSED!\n")


def benchmark(hip_op):
    rope = RotaryEmbedding(ROPE_DIM, 102400, base=160000).to("cuda")
    print("=== Path A: Decode CUDAGraph Benchmark (H=16, D=512, MI308X) ===")
    print(f"{'case':<12} {'M':>3} {'Tri attn':>10} {'HIP attn':>10} {'spd':>6} | {'Tri+iR':>10} {'HIP+iR':>10} {'spd':>6} {'xL save':>10}")
    print("-" * 92)
    for kind, csa, swa, layers in [("csa+swa", 512, 128, 38), ("swa-only", 0, 128, 2)]:
        for M in (1, 4, 8, 16, 24, 48, 64):
            q, cp, ci, cptr, sp, si, sptr, sink = make_case(M, 16, [csa] * M, [swa] * M, seed=M)
            out_t = torch.empty_like(q)
            out_h = torch.empty_like(q)
            pos = torch.full((M,), 99999, device="cuda", dtype=torch.int64)

            t_tri = bench_fn(lambda: sparse_attn_v4_paged_2src(q, cp, ci, cptr, sp, si, sptr, sink, SCALE, out=out_t), True, 5, 40)
            t_hip = bench_fn(lambda: hip_op(q, cp, ci, cptr, sp, si, sptr, sink, SCALE, out=out_h), True, 5, 40)

            def tri_ir():
                sparse_attn_v4_paged_2src(q, cp, ci, cptr, sp, si, sptr, sink, SCALE, out=out_t)
                rope(out_t.unsqueeze(0), pos, inverse=True)

            def hip_ir():
                hip_op(
                    q, cp, ci, cptr, sp, si, sptr, sink, SCALE, out=out_h,
                    cos_cache=rope.cos_cache, sin_cache=rope.sin_cache, positions=pos, rope_dim=ROPE_DIM,
                )

            t_tri_ir = bench_fn(tri_ir, True, 5, 40)
            t_hip_ir = bench_fn(hip_ir, True, 5, 40)
            save_us = (t_tri_ir - t_hip_ir) * layers
            print(
                f"{kind:<12} {M:3d} {t_tri:8.1f}us {t_hip:8.1f}us {t_tri/t_hip:5.2f}x | "
                f"{t_tri_ir:8.1f}us {t_hip_ir:8.1f}us {t_tri_ir/t_hip_ir:5.2f}x {save_us:8.1f}us",
                flush=True,
            )

    print("\n=== Path C: Prefill Eager Benchmark (H=16, D=512, MI308X) ===")
    print(f"{'case':<12} {'M':>5} {'Tri ms':>9} {'HIP ms':>9} {'spd':>6} | {'Tri+iR':>9} {'HIP+iR':>9} {'x18 save':>10} {'x38 save':>10}")
    print("-" * 92)
    for kind, csa, swa in [("csa+swa", 512, 128), ("swa-only", 0, 128)]:
        for M in (1808, 8192):
            q, cp, ci, cptr, sp, si, sptr, sink = make_case(M, 16, [csa] * M, [swa] * M, seed=M)
            out_t = torch.empty_like(q)
            out_h = torch.empty_like(q)
            pos = torch.arange(M, device="cuda", dtype=torch.int64) + 90000

            t_tri = bench_fn(lambda: sparse_attn_v4_paged_2src(q, cp, ci, cptr, sp, si, sptr, sink, SCALE, out=out_t), False, 2, 5) / 1000.0
            t_hip = bench_fn(lambda: hip_op(q, cp, ci, cptr, sp, si, sptr, sink, SCALE, out=out_h, num_splits=1), False, 2, 5) / 1000.0

            def tri_ir():
                sparse_attn_v4_paged_2src(q, cp, ci, cptr, sp, si, sptr, sink, SCALE, out=out_t)
                rope(out_t.unsqueeze(0), pos, inverse=True)

            def hip_ir():
                hip_op(
                    q, cp, ci, cptr, sp, si, sptr, sink, SCALE, out=out_h, num_splits=1,
                    cos_cache=rope.cos_cache, sin_cache=rope.sin_cache, positions=pos, rope_dim=ROPE_DIM,
                )

            t_tri_ir = bench_fn(tri_ir, False, 2, 5) / 1000.0
            t_hip_ir = bench_fn(hip_ir, False, 2, 5) / 1000.0
            save18 = (t_tri_ir - t_hip_ir) * (18 if csa else 2)
            save38 = (t_tri_ir - t_hip_ir) * (38 if csa else 2)
            print(
                f"{kind:<12} {M:5d} {t_tri:7.3f}ms {t_hip:7.3f}ms {t_tri/t_hip:5.2f}x | "
                f"{t_tri_ir:7.3f}ms {t_hip_ir:7.3f}ms {save18:8.1f}ms {save38:8.1f}ms",
                flush=True,
            )

    # Also benchmark realistic prefill locality (contiguous sliding-window SWA + block-local CSA).
    for M in (1808, 8192):
        g = torch.Generator(device="cuda").manual_seed(M + 77)
        q = torch.randn(M, 16, D, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5
        sink = torch.zeros(16, device="cuda", dtype=torch.float32)
        cp = torch.randn(65536, D, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5
        blk = (torch.arange(M, device="cuda") // 256)[:, None] * 1024
        ci = ((blk + torch.randint(0, 4096, (M, 512), device="cuda", generator=g)) % 65536).to(torch.int32).flatten()
        cptr = torch.arange(M + 1, device="cuda", dtype=torch.int32) * 512
        sp = torch.randn(M + 128, D, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5
        si = (torch.arange(M, device="cuda", dtype=torch.int32)[:, None] + torch.arange(128, device="cuda", dtype=torch.int32)[None, :]).flatten()
        sptr = torch.arange(M + 1, device="cuda", dtype=torch.int32) * 128
        out_t = torch.empty_like(q)
        out_h = torch.empty_like(q)
        pos = torch.arange(M, device="cuda", dtype=torch.int64) + 90000

        t_tri = bench_fn(lambda: sparse_attn_v4_paged_2src(q, cp, ci, cptr, sp, si, sptr, sink, SCALE, out=out_t), False, 2, 5) / 1000.0
        t_hip = bench_fn(lambda: hip_op(q, cp, ci, cptr, sp, si, sptr, sink, SCALE, out=out_h, num_splits=1), False, 2, 5) / 1000.0

        def tri_ir():
            sparse_attn_v4_paged_2src(q, cp, ci, cptr, sp, si, sptr, sink, SCALE, out=out_t)
            rope(out_t.unsqueeze(0), pos, inverse=True)

        def hip_ir():
            hip_op(
                q, cp, ci, cptr, sp, si, sptr, sink, SCALE, out=out_h, num_splits=1,
                cos_cache=rope.cos_cache, sin_cache=rope.sin_cache, positions=pos, rope_dim=ROPE_DIM,
            )

        t_tri_ir = bench_fn(tri_ir, False, 2, 5) / 1000.0
        t_hip_ir = bench_fn(hip_ir, False, 2, 5) / 1000.0
        print(
            f"{'csa+swa(loc)':<12} {M:5d} {t_tri:7.3f}ms {t_hip:7.3f}ms {t_tri/t_hip:5.2f}x | "
            f"{t_tri_ir:7.3f}ms {t_hip_ir:7.3f}ms {(t_tri_ir-t_hip_ir)*18:8.1f}ms {(t_tri_ir-t_hip_ir)*38:8.1f}ms",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()
    hip_op = HipSparseAttnV4()
    if not args.skip_verify:
        verify(hip_op)
    if not args.skip_bench:
        benchmark(hip_op)


if __name__ == "__main__":
    main()
