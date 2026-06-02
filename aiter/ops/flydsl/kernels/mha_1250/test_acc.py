"""Accuracy test: compare FMHA kernel output against PyTorch reference.

BSHD layout: q(B,SQ,H,D_qk), k(B,SK,H,D_qk), v(B,SK,H,D_v), o(B,SQ,H,D_v).
Strides for Q/K/V in bytes, strides for O in elements.

Usage:
    bash run_test.sh
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "FlyDSL")
_BUILD_PKGS = os.path.join(_REPO, "build-fly", "python_packages")
for p in [_BUILD_PKGS, os.path.join(_REPO, "python"), _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["ARCH"] = "gfx1250"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext

import ctypes
from flydsl.expr.numeric import Float32 as _Float32, Float64 as _Float64
if not hasattr(_Float32, "_reusable_slot_spec"):
    @classmethod
    def _f32_slot_spec(cls, arg):
        return ctypes.c_float, lambda a: a.value if hasattr(a, "value") else a
    _Float32._reusable_slot_spec = _f32_slot_spec
    _Float32._reusable_ctype = ctypes.c_float
if not hasattr(_Float64, "_reusable_slot_spec"):
    @classmethod
    def _f64_slot_spec(cls, arg):
        return ctypes.c_double, lambda a: a.value if hasattr(a, "value") else a
    _Float64._reusable_slot_spec = _f64_slot_spec
    _Float64._reusable_ctype = ctypes.c_double

from fmha_kernel_gfx1250 import (
    fmha_fwd_kernel,
    BLOCK_SIZE,
    _lds_alloc_k_a,
    _lds_alloc_k_b,
    _lds_alloc_v_a,
    _lds_alloc_v_b,
)

HEAD_QK = 192
HEAD_V = 128
BLOCK_M = 128
BPP = 2


@flyc.jit
def launch_fmha(
    ptr_O: fx.Tensor, ptr_Q: fx.Tensor, ptr_K: fx.Tensor, ptr_V: fx.Tensor,
    ptr_LSE: fx.Tensor,
    ptr_cu_seqlens_q: fx.Tensor, ptr_cu_seqlens_k: fx.Tensor,
    scalar_f: fx.Float32, kv_seq_len: fx.Int32,
    stride_q_seq: fx.Int32, stride_q_tg: fx.Int32,
    stride_q_head: fx.Int32, stride_q_batch: fx.Int32,
    gqa: fx.Int32,
    stride_k_seq: fx.Int32, stride_k_head: fx.Int32, stride_k_batch: fx.Int32,
    stride_v_seq: fx.Int32, stride_v_head: fx.Int32, stride_v_batch: fx.Int32,
    stride_o_seq: fx.Int32, stride_o_head: fx.Int32, stride_o_batch: fx.Int32,
    q_seq_len: fx.Int32, num_heads: fx.Int32, batch_size: fx.Int32,
):
    _lds_alloc_k_a.finalized = False
    _lds_alloc_k_b.finalized = False
    _lds_alloc_v_a.finalized = False
    _lds_alloc_v_b.finalized = False
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        _lds_alloc_k_a.finalize()
        _lds_alloc_k_b.finalize()
        _lds_alloc_v_a.finalize()
        _lds_alloc_v_b.finalize()

    from flydsl.expr.arith import _to_raw
    num_tg = arith.index_cast(T.index, arith.ceildivui(
        _to_raw(q_seq_len), arith.constant(BLOCK_M, type=T.i32)))
    grid_x = arith.index_cast(T.index, batch_size)
    grid_z = arith.index_cast(T.index, num_heads)

    launcher = fmha_fwd_kernel(
        ptr_O, ptr_Q, ptr_K, ptr_V, ptr_LSE,
        ptr_cu_seqlens_q, ptr_cu_seqlens_k,
        scalar_f, kv_seq_len,
        stride_q_seq, stride_q_tg, stride_q_head, stride_q_batch,
        gqa,
        stride_k_seq, stride_k_head, stride_k_batch,
        stride_v_seq, stride_v_head, stride_v_batch,
        stride_o_seq, stride_o_head, stride_o_batch,
        q_seq_len,
    )
    launcher.launch(grid=(grid_x, num_tg, grid_z), block=(BLOCK_SIZE, 1, 1))

launch_fmha.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}


def _ref_mha(q, k, v, sm_scale, causal=True):
    qk = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    if causal:
        S_q, S_kv = qk.shape[-2], qk.shape[-1]
        mask = torch.triu(torch.ones(S_q, S_kv, device=qk.device, dtype=torch.bool), diagonal=1)
        qk = qk.masked_fill(mask, float('-inf'))
    p = torch.softmax(qk, dim=-1)
    return torch.matmul(p, v.float())


def _checkAllclose(a, b, rtol=1e-2, atol=1e-2, msg=""):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    if isClose.all():
        print(f"{msg}\033[32mpassed\033[0m")
        return 0.0
    mask = ~isClose
    num = mask.sum()
    pct = (num / a.numel()).item()
    delta = (a[mask] - b[mask]).abs()
    color = "\033[31m" if pct > 0.05 else "\033[33m"
    print(f"{msg}{color}{'FAIL' if pct > 0.05 else 'warning'}\033[0m  "
          f"max_err={delta.max():.6f}  {pct:.1%} ({num}/{a.numel()})")
    return pct


def run_test(B, H, SQ, SK):
    scale = 1.0 / math.sqrt(HEAD_QK)
    torch.manual_seed(42)

    # BSHD layout: (B, S, H, D)
    q = torch.randn(B, SQ, H, HEAD_QK, dtype=torch.bfloat16).cuda()
    k = torch.randn(B, SK, H, HEAD_QK, dtype=torch.bfloat16).cuda()
    v = torch.randn(B, SK, H, HEAD_V, dtype=torch.bfloat16).cuda()
    o = torch.zeros(B, SQ, H, HEAD_V, dtype=torch.bfloat16).cuda()
    lse = torch.zeros(B, H, SQ, dtype=torch.float32).cuda()

    # Dummy cu_seqlens for BSHD (uniform per-batch seq_len)
    cu_q = torch.arange(0, (B + 1) * SQ, SQ, dtype=torch.int32, device='cuda')
    cu_k = torch.arange(0, (B + 1) * SK, SK, dtype=torch.int32, device='cuda')

    # BSHD strides
    stride_q_seq_b = H * HEAD_QK * BPP
    stride_q_tg_b = BLOCK_M * H * HEAD_QK * BPP
    stride_q_head_b = HEAD_QK * BPP
    stride_q_batch_b = SQ * H * HEAD_QK * BPP
    stride_k_seq_b = H * HEAD_QK * BPP
    stride_k_head_b = HEAD_QK * BPP
    stride_k_batch_b = SK * H * HEAD_QK * BPP
    stride_v_seq_b = H * HEAD_V * BPP
    stride_v_head_b = HEAD_V * BPP
    stride_v_batch_b = SK * H * HEAD_V * BPP
    stride_o_seq_e = H * HEAD_V
    stride_o_head_e = HEAD_V
    stride_o_batch_e = SQ * H * HEAD_V

    args = (
        o, q, k, v, lse,
        cu_q, cu_k,
        scale, SK,
        stride_q_seq_b, stride_q_tg_b, stride_q_head_b, stride_q_batch_b,
        1,  # gqa
        stride_k_seq_b, stride_k_head_b, stride_k_batch_b,
        stride_v_seq_b, stride_v_head_b, stride_v_batch_b,
        stride_o_seq_e, stride_o_head_e, stride_o_batch_e,
        SQ, H, B,
    )

    launch_fmha(*args)
    torch.cuda.synchronize()

    # ref expects BHSD: transpose BSHD -> BHSD for the reference and back for comparison
    ref_bhsd = _ref_mha(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        scale, causal=True)
    ref = ref_bhsd.transpose(1, 2).contiguous()  # back to BSHD
    o_cpu = o.cpu().float()
    ref_cpu = ref.cpu().float()
    tag = f"B={B} H={H} SQ={SQ} SK={SK}"
    err = _checkAllclose(o_cpu, ref_cpu, rtol=1e-2, atol=1e-2, msg=f"  [{tag}] ")
    if err >= 0.05:
        # per-(batch, head) breakdown
        for b in range(B):
            for h in range(H):
                close_bh = torch.isclose(o_cpu[b, :, h, :], ref_cpu[b, :, h, :], rtol=1e-2, atol=1e-2)
                fail_bh = (~close_bh).sum().item() / close_bh.numel()
                max_bh = (o_cpu[b, :, h, :] - ref_cpu[b, :, h, :]).abs().max().item()
                print(f"    batch {b} head {h}: max_err={max_bh:.6f} fail={fail_bh:.1%}")
        # token-level pass map for h=0 to spot a pattern
        if H >= 2:
            pat = []
            for t in range(min(SQ, 128)):
                ok = torch.allclose(o_cpu[0, t, 0, :], ref_cpu[0, t, 0, :], rtol=1e-2, atol=1e-2)
                pat.append('.' if ok else 'X')
            print(f"    tok-pass-pattern h=0: {''.join(pat)}")
    return err < 0.05


def run_test_thd(seqs_q, seqs_k, H):
    """THD varlen test: q/k/v are packed (total_tokens, H, D)."""
    B = len(seqs_q)
    assert B == len(seqs_k)
    scale = 1.0 / math.sqrt(HEAD_QK)
    torch.manual_seed(42)

    total_q = sum(seqs_q)
    total_k = sum(seqs_k)

    # THD packed tensors
    q = torch.randn(total_q, H, HEAD_QK, dtype=torch.bfloat16).cuda()
    k = torch.randn(total_k, H, HEAD_QK, dtype=torch.bfloat16).cuda()
    v = torch.randn(total_k, H, HEAD_V, dtype=torch.bfloat16).cuda()
    o = torch.zeros(total_q, H, HEAD_V, dtype=torch.bfloat16).cuda()

    cu_q = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
    cu_k = torch.zeros(B + 1, dtype=torch.int32, device='cuda')
    for i in range(B):
        cu_q[i + 1] = cu_q[i] + seqs_q[i]
        cu_k[i + 1] = cu_k[i] + seqs_k[i]

    max_sq = max(seqs_q)
    max_sk = max(seqs_k)
    # LSE is per-batch but kernel writes ignore it
    lse = torch.zeros(B, H, max_sq, dtype=torch.float32).cuda()

    # THD strides: per-token byte stride = H*D*BPP
    stride_q_seq_b = H * HEAD_QK * BPP
    stride_q_tg_b = BLOCK_M * H * HEAD_QK * BPP
    stride_q_head_b = HEAD_QK * BPP
    stride_q_batch_b = 0  # unused in THD
    stride_k_seq_b = H * HEAD_QK * BPP
    stride_k_head_b = HEAD_QK * BPP
    stride_k_batch_b = 0  # unused
    stride_v_seq_b = H * HEAD_V * BPP
    stride_v_head_b = HEAD_V * BPP
    stride_v_batch_b = 0  # unused
    stride_o_seq_e = H * HEAD_V
    stride_o_head_e = HEAD_V
    stride_o_batch_e = 0  # unused

    args = (
        o, q, k, v, lse,
        cu_q, cu_k,
        scale, max_sk,
        stride_q_seq_b, stride_q_tg_b, stride_q_head_b, stride_q_batch_b,
        1,  # gqa
        stride_k_seq_b, stride_k_head_b, stride_k_batch_b,
        stride_v_seq_b, stride_v_head_b, stride_v_batch_b,
        stride_o_seq_e, stride_o_head_e, stride_o_batch_e,
        max_sq, H, B,
    )

    launch_fmha(*args)
    torch.cuda.synchronize()

    # Reference: per-batch (variable seq_len)
    err_max = 0.0
    fail_total = 0
    n_total = 0
    o_cpu = o.cpu().float()
    for i in range(B):
        sq, sk = seqs_q[i], seqs_k[i]
        qb = q[cu_q[i]:cu_q[i + 1]].transpose(0, 1).contiguous()  # (H, sq, D)
        kb = k[cu_k[i]:cu_k[i + 1]].transpose(0, 1).contiguous()
        vb = v[cu_k[i]:cu_k[i + 1]].transpose(0, 1).contiguous()
        ref_bh = _ref_mha(qb, kb, vb, scale, causal=True)  # (H, sq, D)
        ref_b = ref_bh.transpose(0, 1).contiguous()  # (sq, H, D)
        o_b = o_cpu[cu_q[i]:cu_q[i + 1]]
        ref_b_cpu = ref_b.cpu().float()
        close = torch.isclose(o_b, ref_b_cpu, rtol=1e-2, atol=1e-2)
        fail_b = (~close).sum().item()
        err_b = (o_b - ref_b_cpu).abs().max().item()
        err_max = max(err_max, err_b)
        fail_total += fail_b
        n_total += close.numel()
        if fail_b > 0:
            print(f"      batch {i} sq={sq} sk={sk}: max_err={err_b:.4f} fail={fail_b/close.numel():.1%}")
            # which tokens fail
            tok_fail = []
            for t in range(sq):
                if not torch.allclose(o_b[t], ref_b_cpu[t], rtol=1e-2, atol=1e-2):
                    tok_fail.append(t)
            print(f"      bad tokens: {tok_fail[:30]}{'...' if len(tok_fail)>30 else ''} (total {len(tok_fail)})")
    pct = fail_total / n_total
    tag = f"THD seqs_q={seqs_q} seqs_k={seqs_k} H={H}"
    color = "\033[32m" if pct < 0.05 else "\033[31m"
    label = "passed" if pct < 0.05 else "FAIL"
    print(f"  [{tag}] {color}{label}\033[0m  max_err={err_max:.6f}  {pct:.1%}")
    return pct < 0.05


if __name__ == "__main__":
    print("=" * 60)
    print("FMHA Accuracy Tests (BSHD layout, D_qk=192 D_v=128, causal)")
    print("=" * 60)

    tests = [
        # small + smoke
        (1, 1, 128, 128),
        (1, 1, 256, 256),
        (1, 1, 300, 300),
        (1, 1, 384, 384),
        # H > 1
        (1, 2, 128, 128),
        (1, 2, 256, 256),
        (1, 4, 128, 128),
        (1, 8, 256, 256),
        # B > 1
        (2, 1, 128, 128),
        (4, 1, 256, 256),
        # B > 1 and H > 1
        (2, 2, 128, 128),
        (2, 2, 256, 256),
        (2, 4, 256, 256),
        (4, 4, 128, 128),
        # larger SQ / SK
        (1, 1, 512, 512),
        (1, 2, 512, 512),
        (1, 1, 1024, 1024),
        (1, 4, 1024, 1024),
        # non-multiple-of-128 SQ/SK
        (1, 2, 300, 300),
        (2, 4, 481, 481),
        (1, 2, 1000, 1000),
        # uneven SQ vs SK (with causal)
        (1, 2, 128, 256),
        (1, 2, 256, 384),
        (2, 4, 384, 512),
    ]

    n_pass = 0
    for B, H, SQ, SK in tests:
        try:
            ok = run_test(B, H, SQ, SK)
            if ok:
                n_pass += 1
        except Exception as e:
            print(f"  [B={B} H={H} SQ={SQ} SK={SK}] ERROR: {e}")
            import traceback; traceback.print_exc()

    print("=" * 60)
    print("THD varlen tests (different seq_len per batch)")
    print("=" * 60)
    thd_tests = [
        # uniform seq_lens (should match BSHD result)
        ([128], [128], 1),
        ([128, 128], [128, 128], 1),
        ([128, 128], [128, 128], 2),
        # variable seq_len per batch
        ([128, 256], [128, 256], 1),
        ([128, 256], [128, 256], 2),
        ([256, 128, 384], [256, 128, 384], 1),
        ([256, 128, 384], [256, 128, 384], 2),
        ([300, 481, 200], [300, 481, 200], 2),
        # uneven q vs k
        ([128, 128], [256, 256], 1),
        ([128, 256], [256, 384], 2),
        ([200, 300], [400, 500], 4),
        # bigger
        ([512, 1024], [512, 1024], 2),
        ([100, 200, 300, 400], [100, 200, 300, 400], 2),
    ]
    n_thd = 0
    for seqs_q, seqs_k, H in thd_tests:
        try:
            ok = run_test_thd(seqs_q, seqs_k, H)
            if ok:
                n_thd += 1
        except Exception as e:
            print(f"  [THD seqs_q={seqs_q} H={H}] ERROR: {e}")
            import traceback; traceback.print_exc()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"\nTHD: {n_thd}/{len(thd_tests)} passed")

    print(f"\n{'='*60}")
    print(f"{n_pass}/{len(tests)} passed")
    print(f"{'='*60}")
