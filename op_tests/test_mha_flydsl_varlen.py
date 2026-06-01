"""Unit test for FlyDSL MHA varlen kernel on gfx1250.

Tests with THD packed layout and variable-length sequences via cu_seqlens.
Supports both causal and non-causal modes.

Usage:
    bash run_mha_flydsl_varlen.sh
"""

import os
import sys
import math
import torch

_KERNEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "aiter", "ops", "flydsl", "kernels", "mha_1250")
_KERNEL_DIR = os.path.abspath(_KERNEL_DIR)

_REPO = os.path.join(_KERNEL_DIR, "FlyDSL")
_BUILD_PKGS = os.path.join(_REPO, "build-fly", "python_packages")
for p in [_BUILD_PKGS, os.path.join(_REPO, "python"), _KERNEL_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["ARCH"] = "gfx1250"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

import flydsl  # noqa: E402
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext

from fmha_kernel_gfx1250 import (
    compile_fmha_fwd,
    BLOCK_SIZE,
    _lds_alloc_k_a,
    _lds_alloc_k_b,
    _lds_alloc_v_a,
    _lds_alloc_v_b,
)

HEAD_DIM_QK = 192
HEAD_DIM_V = 128
BPP = 2


def _make_launcher(is_causal: bool):
    kernel = compile_fmha_fwd(is_causal=is_causal)

    @flyc.jit
    def launch(
        ptr_O: fx.Tensor, ptr_Q: fx.Tensor, ptr_K: fx.Tensor, ptr_V: fx.Tensor,
        ptr_LSE: fx.Tensor, ptr_cu_q: fx.Tensor, ptr_cu_k: fx.Tensor,
        scalar_f: fx.Float32,
        stride_q: fx.Int32, stride_k: fx.Int32, stride_v: fx.Int32, stride_o: fx.Int32,
        gqa: fx.Int32, max_sq: fx.Int32, max_sk: fx.Int32,
        nheads: fx.Int32, batch: fx.Int32,
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
            _to_raw(max_sq), arith.constant(128, type=T.i32)))
        launcher = kernel(
            ptr_O, ptr_Q, ptr_K, ptr_V, ptr_LSE, ptr_cu_q, ptr_cu_k,
            scalar_f, stride_q, stride_k, stride_v, stride_o,
            gqa, max_sq, max_sk,
        )
        launcher.launch(
            grid=(arith.index_cast(T.index, batch), num_tg,
                  arith.index_cast(T.index, nheads)),
            block=(BLOCK_SIZE, 1, 1),
        )
    return launch


_launchers = {}
def get_launcher(is_causal):
    if is_causal not in _launchers:
        _launchers[is_causal] = _make_launcher(is_causal)
    return _launchers[is_causal]


def _ref_mha_varlen(q, k, v, cu_q, cu_k, scale, causal=False):
    """PyTorch reference for varlen THD layout, per-batch."""
    B = len(cu_q) - 1
    H = q.shape[1]
    outs = []
    for b in range(B):
        sq = cu_q[b+1] - cu_q[b]
        sk = cu_k[b+1] - cu_k[b]
        qb = q[cu_q[b]:cu_q[b+1]].float()
        kb = k[cu_k[b]:cu_k[b+1]].float()
        vb = v[cu_k[b]:cu_k[b+1]].float()
        qk = torch.bmm(qb.permute(1,0,2), kb.permute(1,2,0)) * scale
        if causal:
            mask = torch.triu(torch.ones(sq, sk, device=qk.device, dtype=torch.bool), diagonal=1)
            qk = qk.masked_fill(mask.unsqueeze(0), float('-inf'))
        p = torch.softmax(qk, dim=-1)
        ob = torch.bmm(p, vb.permute(1,0,2))
        outs.append(ob.permute(1,0,2))
    return torch.cat(outs, dim=0)


def _checkAllclose(a, b, rtol=1e-2, atol=1e-2, msg=""):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    if isClose.all():
        print(f"{msg}\033[32mpassed~\033[0m")
        return 0.0
    mask = ~isClose
    num = mask.sum()
    pct = (num / a.numel()).item()
    delta = (a[mask] - b[mask]).abs()
    color = "\033[31m" if pct > 0.05 else "\033[33m"
    print(f"{msg}{color}{'failed' if pct > 0.05 else 'warning'}!\033[0m  "
          f"max={delta.max():.6f}  {pct:.1%} ({num}/{a.numel()})")
    return pct


def run_varlen_test(cu_q_list, cu_k_list, H=1, causal=False):
    device = torch.device("cuda")
    torch.manual_seed(42)

    cu_q, cu_k = cu_q_list, cu_k_list
    B = len(cu_q) - 1
    total_q, total_k = cu_q[-1], cu_k[-1]
    max_sq = max(cu_q[i+1] - cu_q[i] for i in range(B))
    max_sk = max(cu_k[i+1] - cu_k[i] for i in range(B))

    q = torch.randn(total_q, H, HEAD_DIM_QK, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, H, HEAD_DIM_QK, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_k, H, HEAD_DIM_V, dtype=torch.bfloat16, device=device)
    o = torch.zeros(total_q, H, HEAD_DIM_V, dtype=torch.bfloat16, device=device)
    lse = torch.zeros(B, H, max_sq, dtype=torch.float32, device=device)
    cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor(cu_k, dtype=torch.int32, device=device)

    stride_q = H * HEAD_DIM_QK * BPP
    stride_k = H * HEAD_DIM_QK * BPP
    stride_v = H * HEAD_DIM_V * BPP
    stride_o = H * HEAD_DIM_V

    launcher = get_launcher(causal)
    launcher(o, q, k, v, lse, cu_seqlens_q, cu_seqlens_k,
             1.0 / math.sqrt(HEAD_DIM_QK),
             stride_q, stride_k, stride_v, stride_o,
             1, max_sq, max_sk, H, B)
    torch.cuda.synchronize()

    ref = _ref_mha_varlen(q, k, v, cu_q, cu_k,
                          1.0 / math.sqrt(HEAD_DIM_QK), causal=causal)

    seqs = [cu_q[i+1] - cu_q[i] for i in range(B)]
    tag = f"B={B} H={H} seqs={seqs} causal={causal}"
    err = _checkAllclose(o.cpu().float(), ref.cpu().float(),
                         rtol=1e-2, atol=1e-2, msg=f"  [{tag}] ")

    # import pdb;pdb.set_trace()

    return err < 0.05


if __name__ == "__main__":
    print("=" * 60)
    print("FlyDSL MHA Varlen Unit Tests")
    print("=" * 60)

    tests = [
        ([0, 128], [0, 128], 1, False),
        ([0, 128], [0, 128], 1, True),
        ([0, 184], [0, 184], 1, False),
        ([0, 184], [0, 184], 1, True),
        ([0, 341], [0, 341], 1, False),
        ([0, 341], [0, 341], 1, True),
        ([0, 481, 581, 982], [0, 481, 581, 982], 1, True),
    ]

    n_pass = 0
    for cu_q, cu_k, H, causal in tests:
        seqs = [cu_q[i+1] - cu_q[i] for i in range(len(cu_q)-1)]
        try:
            ok = run_varlen_test(cu_q, cu_k, H=H, causal=causal)
            if ok:
                n_pass += 1
        except Exception as e:
            print(f"  [seqs={seqs} causal={causal}] ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"{n_pass}/{len(tests)} passed")
    print(f"{'='*60}")
