"""Accuracy test: compare FMHA kernel output against PyTorch reference.

Runs the compiled kernel on GPU, reads output O, and compares with:
    ref_O = softmax(Q @ K^T * scale) @ V

Usage:
    python test_accuracy.py
"""

import math
import os
import sys

_REPO = os.path.join(os.path.dirname(__file__), "FlyDSL")
_BUILD_PKGS = os.path.join(_REPO, "build-fly", "python_packages")
for p in [_BUILD_PKGS, os.path.join(_REPO, "python"), os.path.dirname(__file__)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["ARCH"] = "gfx1250"
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

import flydsl  # noqa: E402
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T

from fmha_kernel_gfx1250 import (
    fmha_fwd_kernel,
    BLOCK_SIZE,
    _lds_alloc_k_a,
    _lds_alloc_k_b,
    _lds_alloc_v_a,
    _lds_alloc_v_b,
)
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir


@flyc.jit
def launch_fmha(
    ptr_O: fx.Tensor,
    ptr_Q: fx.Tensor,
    ptr_K: fx.Tensor,
    ptr_V: fx.Tensor,
    ptr_LSE: fx.Tensor,
    scalar_f: fx.Float32,
    kv_seq_len: fx.Int32,
    stride_q_seq: fx.Int32,
    stride_q_tg: fx.Int32,
    stride_q_head: fx.Int32,
    stride_q_batch: fx.Int32,
    gqa: fx.Int32,
    stride_k_seq: fx.Int32,
    stride_k_head: fx.Int32,
    stride_k_batch: fx.Int32,
    stride_v_seq: fx.Int32,
    stride_v_head: fx.Int32,
    stride_v_batch: fx.Int32,
    stride_o_seq: fx.Int32,
    stride_o_head: fx.Int32,
    stride_o_batch: fx.Int32,
    q_seq_len: fx.Int32,
    num_heads: fx.Int32,
    batch_size: fx.Int32,
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

    q_seq_raw = _to_raw(q_seq_len)
    num_tg = arith.index_cast(T.index, arith.divui(q_seq_raw, arith.constant(128, type=T.i32)))
    # grid_x = arith.index_cast(T.index, num_heads)   # by = head index
    # grid_z = arith.index_cast(T.index, batch_size)  # bz = batch index

    grid_x = arith.index_cast(T.index, batch_size)  # bz = batch  (grid.x)
    grid_z = arith.index_cast(T.index, num_heads)  # by = head   (grid.z)

    launcher = fmha_fwd_kernel(
        ptr_O,
        ptr_Q,
        ptr_K,
        ptr_V,
        ptr_LSE,
        scalar_f,
        kv_seq_len,
        stride_q_seq,
        stride_q_tg,
        stride_q_head,
        stride_q_batch,
        gqa,
        stride_k_seq,
        stride_k_head,
        stride_k_batch,
        stride_v_seq,
        stride_v_head,
        stride_v_batch,
        stride_o_seq,
        stride_o_head,
        stride_o_batch,
        q_seq_len,
    )
    launcher.launch(
        grid=(grid_x, num_tg, grid_z),
        block=(BLOCK_SIZE, 1, 1),
        # stream=stream,
    )


def _ref_mha(q, k, v, sm_scale):
    """PyTorch reference: standard multi-head attention."""
    qk = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    p = torch.softmax(qk, dim=-1)
    return torch.matmul(p, v.float())


def _ref_raw_o(q, k, v, sm_scale):
    """Reference: raw O = P_unnorm @ V (before dividing by row_sum)."""
    qk = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    row_max = qk.max(dim=-1, keepdim=True).values
    p_unnorm = torch.exp(qk - row_max)
    return torch.matmul(p_unnorm, v.float())


def _bench_fn(fn, args, warmup, rep):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def run_test(B, H, S_q, S_kv, D, use_ones=False, D_qk=None):
    # D_qk: QK head dim (Q and K); D: V head dim (V and O)
    if D_qk is None:
        D_qk = D
    scale = 1.0 / (D_qk**0.5)

    torch.manual_seed(42)
    if use_ones:
        q = torch.ones(B, H, S_q, D_qk, dtype=torch.bfloat16).cuda()
        k = torch.ones(B + 1, H, S_kv, D_qk, dtype=torch.bfloat16).cuda()
        v = torch.ones(B, H, S_kv, D, dtype=torch.bfloat16).cuda()
    else:
        q = torch.randn(B, H, S_q, D_qk, dtype=torch.bfloat16).cuda()
        k = torch.randn(B + 1, H, S_kv, D_qk, dtype=torch.bfloat16).cuda()
        v = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16).cuda()
    # q = torch.ones(B, H, S_q, D_qk, dtype=torch.bfloat16).cuda()
    # k = torch.ones(B, H, S_kv, D_qk, dtype=torch.bfloat16).cuda()
    # v = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16).cuda()
    o = torch.zeros(B, H, S_q, D, dtype=torch.bfloat16).cuda()
    lse = torch.zeros(B, H, S_q, dtype=torch.float32).cuda()

    BPP = 2  # bytes per element (bf16)

    # Q/K strides use D_qk; V/O strides use D
    stride_q_seq = D_qk * BPP
    stride_q_tg = 128 * D_qk * BPP  # one query tile = 4 waves × 32 rows = 128 rows
    stride_q_head = S_q * D_qk * BPP
    stride_q_batch = H * S_q * D_qk * BPP

    stride_k_seq = D_qk * BPP
    stride_k_head = S_kv * D_qk * BPP
    stride_k_batch = H * S_kv * D_qk * BPP

    stride_v_seq = D * BPP
    stride_v_head = S_kv * D * BPP
    stride_v_batch = H * S_kv * D * BPP

    # O strides in ELEMENTS (epilogue multiplies by 2 internally)
    stride_o_seq = D
    stride_o_head = S_q * D
    stride_o_batch = H * S_q * D

    gqa = 1
    q_seq_len = S_q

    ref = _ref_mha(q, k[:B], v, scale)
    ref_raw = _ref_raw_o(q, k[:B], v, scale)

    # Reference row_sums for _DEBUG_ROWSUM comparison
    qk_ref = torch.matmul(q.float(), k[:B].float().transpose(-2, -1)) * scale
    row_max_ref = qk_ref.max(dim=-1, keepdim=True).values
    p_unnorm_ref = torch.exp(qk_ref - row_max_ref)
    row_sums_ref = p_unnorm_ref.sum(dim=-1)  # [B, H, S_q]

    _per_tile_ref = None
    launch_fmha(
        o,
        q,
        k,
        v,
        lse,
        scale,
        S_kv,
        stride_q_seq,
        stride_q_tg,
        stride_q_head,
        stride_q_batch,
        gqa,
        stride_k_seq,
        stride_k_head,
        stride_k_batch,
        stride_v_seq,
        stride_v_head,
        stride_v_batch,
        stride_o_seq,
        stride_o_head,
        stride_o_batch,
        q_seq_len,
        H,
        B,
    )
    torch.cuda.synchronize()

    out_f32 = o.cpu().float()
    ref_f32 = ref.cpu().float()

    ref_raw_f32 = ref_raw.cpu().float()
    # import pdb;pdb.set_trace()

    # # Print raw O comparison at specific elements
    # print(f"  Raw O comparison (row 0, first 16 cols):")
    # print(f"  {'col':>5s} {'kernel':>10s} {'ref_raw':>10s} {'err':>10s}")
    # for c in range(16):
    #     k_val = out_f32[0, 0, 0, c].item()
    #     r_val = ref_raw_f32[0, 0, 0, c].item()
    #     print(f"    {c:3d}  {k_val:10.4f} {r_val:10.4f} {abs(k_val-r_val):10.4f}")
    # print(f"  Raw O comparison (row 0, cols 64-79):")
    # for c in range(64, 80):
    #     k_val = out_f32[0, 0, 0, c].item()
    #     r_val = ref_raw_f32[0, 0, 0, c].item()
    #     print(f"    {c:3d}  {k_val:10.4f} {r_val:10.4f} {abs(k_val-r_val):10.4f}")

    # n_nan = torch.isnan(out_f32).sum().item()
    # n_inf = torch.isinf(out_f32).sum().item()
    # n_zero = (out_f32 == 0).sum().item()
    # n_total = out_f32.numel()
    # print(f"  Output stats: NaN={n_nan}/{n_total} Inf={n_inf} Zero={n_zero}")
    # if n_nan < n_total:
    #     finite = out_f32[~torch.isnan(out_f32)]
    #     print(f"  Finite range: [{finite.min().item():.6f}, {finite.max().item():.6f}]")
    # print(f"  Ref (normalized) range: [{ref_f32.min().item():.6f}, {ref_f32.max().item():.6f}]")
    # print(f"  Ref (raw P@V) range:    [{ref_raw_f32.min().item():.6f}, {ref_raw_f32.max().item():.6f}]")

    # abs_err = (out_f32 - ref_f32).abs()
    # raw_abs_err = (out_f32 - ref_raw_f32).abs()
    # max_err = abs_err.max().item()
    # mean_err = abs_err.mean().item()
    # rel_err = (abs_err / (ref_f32.abs() + 1e-6)).mean().item()
    # raw_max_err = raw_abs_err.max().item()
    # raw_mean_err = raw_abs_err.mean().item()

    # print(f"  Shape: B={B} H={H} S_q={S_q} S_kv={S_kv} D={D}")
    # print(f"  vs normalized: max={max_err:.6f} mean={mean_err:.6f}")
    # print(f"  vs raw P@V:    max={raw_max_err:.6f} mean={raw_mean_err:.6f}")
    # if use_ones:
    #     print(f"  Ratio check: out[0,0,0,0]={out_f32[0,0,0,0].item():.6f} ref={ref_f32[0,0,0,0].item():.6f} raw={ref_raw_f32[0,0,0,0].item():.6f}")
    # print(f"  Max abs error:  {max_err:.6f}")
    # print(f"  Mean abs error: {mean_err:.6f}")
    # print(f"  Mean rel error: {rel_err:.6f}")
    # # Print error distribution for debugging large errors
    # if max_err > 0.04 and S_q > 128:
    #     err_2d = abs_err[0, 0]  # [S_q, D]
    #     row_max = err_2d.max(dim=-1).values
    #     col_max = err_2d.max(dim=0).values
    #     top_rows = row_max.topk(10).indices.tolist()
    #     print(f"  Top error rows: {top_rows[:5]}")
    #     print(f"  WG0 rows (0-127) max error: {row_max[:128].max():.4f}")
    #     print(f"  WG1 rows (128-255) max error: {row_max[128:].max():.4f}")
    #     print(f"  WG1 row 128 col 0: kernel={out_f32[0,0,128,0]:.4f} ref={ref_f32[0,0,128,0]:.4f}")
    #     print(f"  WG1 row 131 col 0: kernel={out_f32[0,0,131,0]:.4f} ref={ref_f32[0,0,131,0]:.4f}")
    #     print(f"  WG1 all-row mean err: {err_2d[128:].mean():.4f}")
    #     print(f"  WG0 all-row mean err: {err_2d[:128].mean():.4f}")

    # passed = True #max_err < 0.04
    abs_err = (out_f32 - ref_f32).abs()
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()
    passed = max_err < 0.04 and not math.isnan(max_err)
    print(f"  Shape: B={B} H={H} S_q={S_q} S_kv={S_kv} D={D}")
    print(f"  Max abs error: {max_err:.6f}  Mean: {mean_err:.6f}")

    if max_err > 0.04 or math.isnan(max_err):
        err_2d = abs_err[0, 0]  # [S_q, D]
        row_max = err_2d.max(dim=-1).values
        bad_rows = (row_max > 0.04).nonzero(as_tuple=True)[0].tolist()
        print(f"  Bad Q rows (err>0.04): {bad_rows[:20]} / {len(bad_rows)} total")
        for br in bad_rows[:5]:
            cols_k = out_f32[0, 0, br].tolist()
            cols_r = ref_f32[0, 0, br].tolist()
            print(
                f"  Row {br}: kernel[:4]={[round(x, 2) for x in cols_k[:4]]} ref[:4]={[round(x, 2) for x in cols_r[:4]]} max_err={abs_err[0, 0, br].max().item():.2f}"
            )
        good_rows = [r for r in range(min(S_q, 128)) if r not in bad_rows]
        if good_rows:
            gr = good_rows[0]
            cols_k = out_f32[0, 0, gr].tolist()
            cols_r = ref_f32[0, 0, gr].tolist()
            print(
                f"  Good row {gr}: kernel[:4]={[round(x, 4) for x in cols_k[:4]]} ref[:4]={[round(x, 4) for x in cols_r[:4]]}"
            )
    #     # print(f"  Bad Q rows (err>0.04): {bad_rows[:10]} / {len(bad_rows)} total")
    #     # if bad_rows:
    #     #     for br in bad_rows[:3]:
    #     #         cols = out_f32[0,0,br].tolist()
    #     #         big_cols = [c for c in range(128) if abs(cols[c]) > 1e6]
    #     #         print(f"  Row {br}: first4={[round(x,2) for x in cols[:4]]} max_err={abs_err[0,0,br].max().item():.2f} big_cols({len(big_cols)})={big_cols[:8]}")
    #     if 0 not in bad_rows:
    #         print("pass!!!!!!!!!!!!!!!!!!")
    #         passed = True
    #     else:
    #         print(out_f32[0][0][0])
    #         print(ref_f32[0][0][0])
    #         import pdb; pdb.set_trace()
    # else:
    #     passed = True
    # # passed = max_err < 0.04
    # print(passed)
    print(f"  Result: {'PASS' if passed else 'FAIL'} (threshold=0.04)")

    # # --- benchmark ---
    def _bench_call():
        launch_fmha(
            o,
            q,
            k,
            v,
            lse,
            scale,
            S_kv,
            stride_q_seq,
            stride_q_tg,
            stride_q_head,
            stride_q_batch,
            gqa,
            stride_k_seq,
            stride_k_head,
            stride_k_batch,
            stride_v_seq,
            stride_v_head,
            stride_v_batch,
            stride_o_seq,
            stride_o_head,
            stride_o_batch,
            q_seq_len,
            H,
            B,
        )

    for _ in range(25):
        _bench_call()
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(100):
        _bench_call()
    end_evt.record()
    torch.cuda.synchronize()
    ms = start_evt.elapsed_time(end_evt) / 100
    flops = 2.0 * B * H * S_q * S_kv * D_qk + 2.0 * B * H * S_q * S_kv * D
    tflops = flops / (ms * 1e-3) / 1e12
    bytes_accessed = (
        B * H * S_q * D_qk * BPP + B * H * S_kv * D_qk * BPP + B * H * S_kv * D * BPP + B * H * S_q * D * BPP
    )
    bw = bytes_accessed / (ms * 1e-3) / 1e9
    print(f"  Perf: {ms:.4f} ms | {tflops:.2f} TFLOPS | {bw:.1f} GB/s")

    return passed


if __name__ == "__main__":
    print("=" * 60)
    print("FMHA Accuracy Test")
    print("=" * 60)

    # Kernel is now QK_HDIM=192, V_HDIM=128
    configs = []  # no square-dim tests (kernel is 192-QK)
    configs_192 = [
        {"B": 1, "H": 64, "S_q": 8192, "S_kv": 8192, "D": 128, "D_qk": 192},
        # {'B':1,'H':1,'S_q':128,'S_kv':128,'D':128,'D_qk':192},
        # {'B':1,'H':1,'S_q':256,'S_kv':256,'D':128,'D_qk':192},
        # {'B':1,'H':1,'S_q':512,'S_kv':512,'D':128,'D_qk':192},
        # {'B':1,'H':1,'S_q':128,'S_kv':128,'D':128,'D_qk':192},
        # {'B':1,'H':256,'S_q':128,'S_kv':128,'D':128,'D_qk':192},
        # {'B':2,'H':2,'S_q':256,'S_kv':256,'D':128,'D_qk':192},
    ]

    all_passed = True

    # # All-ones test first (sanity check)
    # print("\nTest: ALL-ONES (B=1 H=1 S_q=128 S_kv=128 D=128)")
    # try:
    #     ok = run_test(1, 1, 128, 128, 128, use_ones=True, D_qk=192)
    #     if not ok:
    #         all_passed = False
    # except Exception as e:
    #     print(f"  ERROR: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     all_passed = False

    # # Identity-V test: V = eye(128,128) → raw O should be identity of P
    # print("\nTest: IDENTITY-V (V=I, Q=K=ones)")
    # try:
    #     B, H, S_q, S_kv, D = 1, 1, 128, 128, 128
    #     D_qk = 192
    #     scale = 1.0 / (D_qk ** 0.5)
    #     q = torch.ones(B, H, S_q, D_qk, dtype=torch.bfloat16).cuda()
    #     k = torch.ones(B, H, S_kv, D_qk, dtype=torch.bfloat16).cuda()
    #     v = torch.eye(S_kv, D, dtype=torch.bfloat16).view(1,1,S_kv,D).cuda()
    #     o = torch.zeros(B, H, S_q, D, dtype=torch.bfloat16).cuda()
    #     lse = torch.zeros(B, H, S_q, dtype=torch.float32).cuda()
    #     BPP = 2
    #     stride_q_seq = D_qk*BPP; stride_q_tg = S_q*D_qk*BPP; stride_q_head = S_q*D_qk*BPP; stride_q_batch = H*S_q*D_qk*BPP
    #     stride_k_seq = D_qk*BPP; stride_k_head = S_kv*D_qk*BPP; stride_k_batch = H*S_kv*D_qk*BPP
    #     stride_v_seq = D*BPP; stride_v_head = S_kv*D*BPP; stride_v_batch = H*S_kv*D*BPP
    #     stride_o_seq = D; stride_o_head = S_q*D; stride_o_batch = H*S_q*D
    #     launch_fmha(
    #         o, q, k, v, lse, scale, S_kv,
    #         stride_q_seq, stride_q_tg, stride_q_head, stride_q_batch, 1,
    #         stride_k_seq, stride_k_head, stride_k_batch,
    #         stride_v_seq, stride_v_head, stride_v_batch,
    #         stride_o_seq, stride_o_head, stride_o_batch, S_q, 0, 1, 1,
    #     )
    #     torch.cuda.synchronize()
    #     out_f32 = o.cpu().float()
    #     # Raw O should be identity: O[q,d] = sum_kv P_unnorm[q,kv] * V[kv,d] = P_unnorm[q,d]
    #     # P_unnorm = exp(0) = 1, so O[q,d] = V[d,d] = 1 if d<128 else 0 → O = I(128)
    #     # But all rows same since uniform Q/K → O[any_row] = (1,1,...,1) since V is 128x128 identity
    #     print(f"  Row 0 first 8: {[f'{out_f32[0,0,0,c].item():.2f}' for c in range(8)]}")
    #     print(f"  Expected:      {['1.00']*8}")
    #     print(f"  Row 0 cols 30-35: {[f'{out_f32[0,0,0,c].item():.2f}' for c in range(30,36)]}")
    #     print(f"  Row 0 cols 62-69: {[f'{out_f32[0,0,0,c].item():.2f}' for c in range(62,70)]}")
    #     print(f"  Row 0 cols 126-127: {[f'{out_f32[0,0,0,c].item():.2f}' for c in range(126,128)]}")
    #     # Check: which columns have value > 0.5?
    #     nonzero_mask = out_f32[0,0,0,:].abs() > 0.5
    #     print(f"  Nonzero cols count: {nonzero_mask.sum().item()} (expected 128)")
    #     print(f"  Sum: {out_f32[0,0,0,:].sum().item():.1f} (expected 128.0)")
    # except Exception as e:
    #     print(f"  ERROR: {e}")
    #     import traceback; traceback.print_exc()

    # # Single-col test: V[kv,d]=1 for specific d, else 0
    # print("\nTest: SINGLE-COL (V[:,d]=1 for d=0,32,64,96)")
    # try:
    #     B, H, S_q, S_kv, D = 1, 1, 128, 128, 128
    #     D_qk = 192
    #     scale = 1.0 / (D_qk ** 0.5)
    #     q = torch.ones(B, H, S_q, D_qk, dtype=torch.bfloat16).cuda()
    #     k = torch.ones(B, H, S_kv, D_qk, dtype=torch.bfloat16).cuda()
    #     for test_col in [0, 1, 16, 32, 48, 64, 96]:
    #         v = torch.zeros(B, H, S_kv, D, dtype=torch.bfloat16).cuda()
    #         v[:, :, :, test_col] = 1.0
    #         o = torch.zeros(B, H, S_q, D, dtype=torch.bfloat16).cuda()
    #         lse = torch.zeros(B, H, S_q, dtype=torch.float32).cuda()
    #         BPP = 2
    #         stride_q_seq = D_qk*BPP; stride_q_tg = S_q*D_qk*BPP; stride_q_head = S_q*D_qk*BPP; stride_q_batch = H*S_q*D_qk*BPP
    #         stride_k_seq = D_qk*BPP; stride_k_head = S_kv*D_qk*BPP; stride_k_batch = H*S_kv*D_qk*BPP
    #         stride_v_seq = D*BPP; stride_v_head = S_kv*D*BPP; stride_v_batch = H*S_kv*D*BPP
    #         stride_o_seq = D; stride_o_head = S_q*D; stride_o_batch = H*S_q*D
    #         launch_fmha(
    #             o, q, k, v, lse, scale, S_kv,
    #             stride_q_seq, stride_q_tg, stride_q_head, stride_q_batch, 1,
    #             stride_k_seq, stride_k_head, stride_k_batch,
    #             stride_v_seq, stride_v_head, stride_v_batch,
    #             stride_o_seq, stride_o_head, stride_o_batch, S_q, 0, 1, 1,
    #         )
    #         torch.cuda.synchronize()
    #         out_f32 = o.cpu().float()
    #         nonzero_cols = (out_f32[0,0,0,:].abs() > 0.5).nonzero().squeeze(-1).tolist()
    #         val_at_col = out_f32[0,0,0,test_col].item()
    #         print(f"  V[:,{test_col}]=1: O at col {test_col}={val_at_col:.1f}, nonzero cols={nonzero_cols}")
    # except Exception as e:
    #     print(f"  ERROR: {e}")
    #     import traceback; traceback.print_exc()

    # # Column-ID test: V[kv,d]=d+1, Q=K=ones → raw O should be 128*(d+1)
    # print("\nTest: COLUMN-ID (V[kv,d]=d+1, Q=K=ones)")
    # try:
    #     B, H, S_q, S_kv, D = 1, 1, 128, 128, 128
    #     D_qk = 192
    #     scale = 1.0 / (D_qk ** 0.5)
    #     q = torch.ones(B, H, S_q, D_qk, dtype=torch.bfloat16).cuda()
    #     k = torch.ones(B, H, S_kv, D_qk, dtype=torch.bfloat16).cuda()
    #     v = torch.arange(1, D+1, dtype=torch.bfloat16).view(1,1,1,D).expand(B,H,S_kv,D).contiguous().cuda()
    #     o = torch.zeros(B, H, S_q, D, dtype=torch.bfloat16).cuda()
    #     lse = torch.zeros(B, H, S_q, dtype=torch.float32).cuda()
    #     BPP = 2
    #     stride_q_seq = D_qk*BPP; stride_q_tg = S_q*D_qk*BPP; stride_q_head = S_q*D_qk*BPP; stride_q_batch = H*S_q*D_qk*BPP
    #     stride_k_seq = D_qk*BPP; stride_k_head = S_kv*D_qk*BPP; stride_k_batch = H*S_kv*D_qk*BPP
    #     stride_v_seq = D*BPP; stride_v_head = S_kv*D*BPP; stride_v_batch = H*S_kv*D*BPP
    #     stride_o_seq = D; stride_o_head = S_q*D; stride_o_batch = H*S_q*D
    #     launch_fmha(
    #         o, q, k, v, lse, scale, S_kv,
    #         stride_q_seq, stride_q_tg, stride_q_head, stride_q_batch, 1,
    #         stride_k_seq, stride_k_head, stride_k_batch,
    #         stride_v_seq, stride_v_head, stride_v_batch,
    #         stride_o_seq, stride_o_head, stride_o_batch, S_q, 0, 1, 1,
    #     )
    #     torch.cuda.synchronize()
    #     out_f32 = o.cpu().float()
    #     ref_vals = [128.0 * (d+1) for d in range(D)]
    #     print(f"  Col 0: kernel={out_f32[0,0,0,0].item():.1f} expected=128.0")
    #     print(f"  Col 1: kernel={out_f32[0,0,0,1].item():.1f} expected=256.0")
    #     print(f"  Col 63: kernel={out_f32[0,0,0,63].item():.1f} expected={128*64:.1f}")
    #     print(f"  Col 64: kernel={out_f32[0,0,0,64].item():.1f} expected={128*65:.1f}")
    #     print(f"  Col 127: kernel={out_f32[0,0,0,127].item():.1f} expected={128*128:.1f}")
    #     print(f"  Cols 0-15: {[f'{out_f32[0,0,0,c].item():.0f}' for c in range(16)]}")
    #     print(f"  Exp  0-15: {[f'{128*(c+1):.0f}' for c in range(16)]}")
    #     print(f"  Cols 16-31: {[f'{out_f32[0,0,0,c].item():.0f}' for c in range(16,32)]}")
    #     print(f"  Exp  16-31: {[f'{128*(c+1):.0f}' for c in range(16,32)]}")
    #     print(f"  Cols 32-47: {[f'{out_f32[0,0,0,c].item():.0f}' for c in range(32,48)]}")
    #     print(f"  Exp  32-47: {[f'{128*(c+1):.0f}' for c in range(32,48)]}")
    #     print(f"  Cols 48-63: {[f'{out_f32[0,0,0,c].item():.0f}' for c in range(48,64)]}")
    #     print(f"  Exp  48-63: {[f'{128*(c+1):.0f}' for c in range(48,64)]}")
    #     print(f"  Cols 64-79: {[f'{out_f32[0,0,0,c].item():.0f}' for c in range(64,80)]}")
    #     print(f"  Exp  64-79: {[f'{128*(c+1):.0f}' for c in range(64,80)]}")
    #     print(f"  Cols 80-95: {[f'{out_f32[0,0,0,c].item():.0f}' for c in range(80,96)]}")
    #     print(f"  Exp  80-95: {[f'{128*(c+1):.0f}' for c in range(80,96)]}")
    #     print(f"  Cols 96-111: {[f'{out_f32[0,0,0,c].item():.0f}' for c in range(96,112)]}")
    #     print(f"  Exp  96-111: {[f'{128*(c+1):.0f}' for c in range(96,112)]}")
    #     print(f"  Cols 112-127: {[f'{out_f32[0,0,0,c].item():.0f}' for c in range(112,128)]}")
    #     print(f"  Exp  112-127: {[f'{128*(c+1):.0f}' for c in range(112,128)]}")
    # except Exception as e:
    #     print(f"  ERROR: {e}")
    #     import traceback; traceback.print_exc()

    # # Uniform-softmax test: K=Q=ones, V=random → isolates PV dataflow
    # print("\nTest: UNIFORM-SOFTMAX (K=Q=ones, V=random)")
    # try:
    #     B, H, S_q, S_kv, D = 1, 1, 128, 128, 128
    #     D_qk = 192
    #     scale = 1.0 / (D_qk ** 0.5)
    #     torch.manual_seed(42)
    #     q = torch.ones(B, H, S_q, D_qk, dtype=torch.bfloat16).cuda()
    #     k = torch.ones(B, H, S_kv, D_qk, dtype=torch.bfloat16).cuda()
    #     v = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16).cuda()
    #     o = torch.zeros(B, H, S_q, D, dtype=torch.bfloat16).cuda()
    #     lse = torch.zeros(B, H, S_q, dtype=torch.float32).cuda()
    #     BPP = 2
    #     stride_q_seq = D_qk * BPP; stride_q_tg = S_q * D_qk * BPP
    #     stride_q_head = S_q * D_qk * BPP; stride_q_batch = H * S_q * D_qk * BPP
    #     stride_k_seq = D_qk * BPP; stride_k_head = S_kv * D_qk * BPP
    #     stride_k_batch = H * S_kv * D_qk * BPP
    #     stride_v_seq = D * BPP; stride_v_head = S_kv * D * BPP
    #     stride_v_batch = H * S_kv * D * BPP
    #     stride_o_seq = D; stride_o_head = S_q * D; stride_o_batch = H * S_q * D
    #     ref_raw = torch.matmul(
    #         torch.ones(B, H, S_q, S_kv, dtype=torch.float32).cuda() * (1.0/S_kv),
    #         v.float()
    #     ) * S_kv  # raw = P_unnorm @ V, P_unnorm=exp(0)=1 for uniform case
    #     # Actually for uniform Q=K=ones: QK[i,j] = sum_d(q*k)*scale = D*scale = 1.0
    #     # So softmax(1,1,...,1) = 1/128 each, raw_O = exp(1-1)*V.sum(0) = V.sum(0)
    #     ref_raw_correct = v.float().mean(dim=-2, keepdim=True).expand_as(v)  # normalized: each row = V.mean(dim=kv)
    #     launch_fmha(
    #         o, q, k, v, lse, scale, S_kv,
    #         stride_q_seq, stride_q_tg, stride_q_head, stride_q_batch, 1,
    #         stride_k_seq, stride_k_head, stride_k_batch,
    #         stride_v_seq, stride_v_head, stride_v_batch,
    #         stride_o_seq, stride_o_head, stride_o_batch, S_q, 0, 1, 1,
    #     )
    #     torch.cuda.synchronize()
    #     out_f32 = o.cpu().float()
    #     ref_f32 = ref_raw_correct.cpu().float()
    #     abs_err = (out_f32 - ref_f32).abs()
    #     max_err = abs_err.max().item()
    #     mean_err = abs_err.mean().item()
    #     print(f"  Row 0, cols 0-7  kernel: {[f'{out_f32[0,0,0,c].item():.4f}' for c in range(8)]}")
    #     print(f"  Row 0, cols 0-7  ref:    {[f'{ref_f32[0,0,0,c].item():.4f}' for c in range(8)]}")
    #     print(f"  Row 16, cols 0-7 kernel: {[f'{out_f32[0,0,16,c].item():.4f}' for c in range(8)]}")
    #     print(f"  Row 0, cols 64-71 kernel: {[f'{out_f32[0,0,0,c].item():.4f}' for c in range(64,72)]}")
    #     print(f"  Row 0, cols 64-71 ref:    {[f'{ref_f32[0,0,0,c].item():.4f}' for c in range(64,72)]}")
    #     print(f"  Row 16, cols 64-71 kernel: {[f'{out_f32[0,0,16,c].item():.4f}' for c in range(64,72)]}")
    #     # Check row consistency (all rows should be same for uniform softmax)
    #     row_var = out_f32[0,0,:,:].var(dim=0).mean().item()
    #     print(f"  Row variance (should be ~0): {row_var:.6f}")
    #     # Check partial sums to diagnose which KV positions are summed
    #     v_f32 = v.cpu().float()
    #     for start in [0, 32, 64, 96]:
    #         psum = v_f32[0,0,start:start+32,0].sum().item()
    #         print(f"  V[:,0] sum rows {start}-{start+31}: {psum:.4f}")
    #     print(f"  V[:,0] full sum: {v_f32[0,0,:,0].sum().item():.4f}")
    #     print(f"  Kernel[0,0,0,0] = {out_f32[0,0,0,0].item():.4f}")
    #     # Check col 64
    #     for start in [0, 32, 64, 96]:
    #         psum = v_f32[0,0,start:start+32,64].sum().item()
    #         print(f"  V[:,64] sum rows {start}-{start+31}: {psum:.4f}")
    #     print(f"  V[:,64] full sum: {v_f32[0,0,:,64].sum().item():.4f}")
    #     print(f"  Kernel[0,0,0,64] = {out_f32[0,0,0,64].item():.4f}")
    #     print(f"  Max abs error: {max_err:.6f}  Mean: {mean_err:.6f}")
    #     ok = max_err < 1.0
    #     print(f"  Result: {'PASS' if ok else 'FAIL'}")
    #     if not ok:
    #         all_passed = False
    # except Exception as e:
    #     print(f"  ERROR: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     all_passed = False

    for cfg in configs:
        print(f"\nTest: RANDOM B={cfg[0]} H={cfg[1]} S_q={cfg[2]} S_kv={cfg[3]} D={cfg[4]}")
        try:
            ok = run_test(*cfg)
            if not ok:
                all_passed = False
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

    for cfg in configs_192:
        print(f"\nTest: RANDOM D_qk=192 B={cfg['B']} H={cfg['H']} S_q={cfg['S_q']} S_kv={cfg['S_kv']} D_v={cfg['D']}")
        try:
            ok = run_test(cfg["B"], cfg["H"], cfg["S_q"], cfg["S_kv"], cfg["D"], D_qk=cfg["D_qk"])
            if not ok:
                all_passed = False
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)
