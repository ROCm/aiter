#!/usr/bin/env python3
"""
Workload that exercises each attention backend one at a time, separated by
roctx markers so the wrapper script can attribute kernel dispatches to the
correct backend.

Usage (standalone):  python profile_attention_grids.py
Usage (profiled):    ./profile_attention_grids.sh   (recommended)
"""
import os
import sys
import math
import torch

try:
    from roctx import roctx_range  # type: ignore[import]
except ImportError:
    roctx_range = None

try:
    from rocm_utils import roctx_range_push, roctx_range_pop  # type: ignore[import]
except ImportError:
    roctx_range_push = roctx_range_pop = None

try:
    import ctypes
    _libroctx = ctypes.CDLL("libroctx64.so")
    _roctxRangePush = _libroctx.roctxRangePushA
    _roctxRangePush.argtypes = [ctypes.c_char_p]
    _roctxRangePush.restype = ctypes.c_int
    _roctxRangePop = _libroctx.roctxRangePop
    _roctxRangePop.restype = ctypes.c_int
    HAS_ROCTX = True
except OSError:
    HAS_ROCTX = False


def marker_push(name: str):
    if HAS_ROCTX:
        _roctxRangePush(name.encode())
    torch.cuda.synchronize()


def marker_pop():
    torch.cuda.synchronize()
    if HAS_ROCTX:
        _roctxRangePop()

# ---------------------------------------------------------------------------
# Configurable shape — override via env vars
# ---------------------------------------------------------------------------
SEQS   = int(os.environ.get("PROF_SEQS",  "248"))
NQH    = int(os.environ.get("PROF_NQH",   "8"))
NKH    = int(os.environ.get("PROF_NKH",   "1"))
HDIM   = int(os.environ.get("PROF_HDIM",  "64"))
MAXK   = int(os.environ.get("PROF_MAXK",  "4096"))
BLK    = int(os.environ.get("PROF_BLK",   "64"))
ITERS  = int(os.environ.get("PROF_ITERS", "3"))

device = "cuda"
dtype  = torch.bfloat16
scale  = 1.0 / math.sqrt(HDIM)

print(f"Shape: seqs={SEQS} hq={NQH} hk={NKH} hdim={HDIM} maxk={MAXK} blk={BLK} iters={ITERS}")
print(f"roctx markers: {'available' if HAS_ROCTX else 'NOT available (kernel names still work)'}")
print()

# ---------------------------------------------------------------------------
# Allocate tensors
# ---------------------------------------------------------------------------
torch.manual_seed(42)
needed   = (MAXK + BLK - 1) // BLK
num_phys = needed * SEQS
q  = torch.randn(SEQS, NQH, HDIM, dtype=dtype, device=device)
k  = torch.randn(num_phys, BLK, NKH, HDIM, dtype=dtype, device=device)
v  = torch.randn_like(k)
cu = torch.arange(SEQS + 1, dtype=torch.int32, device=device)
sl = torch.full((SEQS,), MAXK, dtype=torch.int32, device=device)
bt = torch.arange(num_phys, dtype=torch.int32, device=device).reshape(SEQS, needed)
out = torch.zeros_like(q)

# ---------------------------------------------------------------------------
# Imports — guarded so we can still show the script even if not installed
# ---------------------------------------------------------------------------
try:
    from aiter.ops.unified_attention import unified_attention_fwd
    HAS_CK_UA = True
except ImportError:
    HAS_CK_UA = False

try:
    import aiter.ops.triton.attention.unified_attention as ua_mod
    from aiter.ops.triton.attention.unified_attention import unified_attention as triton_ua
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    from aiter.ops.mha import mha_varlen_fwd
    HAS_CK_SK = True
except ImportError:
    HAS_CK_SK = False


triton_kw = dict(
    q=q, k=k, v=v, out=out, cu_seqlens_q=cu,
    max_seqlen_q=1, seqused_k=sl, max_seqlen_k=MAXK,
    softmax_scale=scale, causal=True, window_size=(-1, -1),
    block_table=bt, softcap=0.0,
    q_descale=None, k_descale=None, v_descale=None,
    alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None,
)

def run_backend(name, fn, iters=ITERS):
    print(f"  Running {name} ({iters} iters)... ", end="", flush=True)
    torch.cuda.synchronize()
    # warmup
    fn()
    torch.cuda.synchronize()

    marker_push(name)
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    marker_pop()
    print("done")


# ---------------------------------------------------------------------------
# Run each backend
# ---------------------------------------------------------------------------
print("=" * 60)
print("Profiling attention backends")
print("=" * 60)

# 1. CK-UA
if HAS_CK_UA:
    run_backend("CK-UA", lambda: unified_attention_fwd(
        out, q, k, v, bt, sl, cu,
        mask_type=2, scale_s=scale,
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
    ))
else:
    print("  CK-UA: not available (skipped)")

# 2. Triton 2D
if HAS_TRITON:
    saved_ck = ua_mod._try_ck_unified_attention
    ua_mod._try_ck_unified_attention = lambda *a, **kw: False
    saved_2d = ua_mod.use_2d_kernel

    ua_mod.use_2d_kernel = lambda *a, **kw: True
    run_backend("Triton-2D", lambda: ua_mod.unified_attention(**triton_kw))

    # 3. Triton 3D
    ua_mod.use_2d_kernel = lambda *a, **kw: False
    run_backend("Triton-3D", lambda: ua_mod.unified_attention(**triton_kw))

    ua_mod.use_2d_kernel = saved_2d
    ua_mod._try_ck_unified_attention = saved_ck
else:
    print("  Triton 2D/3D: not available (skipped)")

# 4. CK-SK (split-KV via mha_varlen_fwd)
if HAS_CK_SK:
    cu_k = torch.zeros(SEQS + 1, dtype=torch.int32, device=device)
    cu_k[1:] = torch.cumsum(sl, dim=0)
    try:
        run_backend("CK-SK", lambda: mha_varlen_fwd(
            q.reshape(-1, NQH, HDIM),
            k.reshape(-1, NKH, HDIM),
            v.reshape(-1, NKH, HDIM),
            cu_seqlens_q=cu,
            cu_seqlens_k=cu_k,
            max_seqlen_q=1,
            max_seqlen_k=MAXK,
            min_seqlen_q=1,
            dropout_p=0.0,
            softmax_scale=scale,
            logits_soft_cap=0.0,
            zero_tensors=False,
            is_causal=True,
            window_size_left=-1,
            window_size_right=-1,
            sink_size=0,
            return_softmax_lse=False,
            return_dropout_randval=False,
            block_table=bt,
        ))
    except Exception as e:
        print(f"  CK-SK: error ({e})")

print("\nDone. If run under rocprofv3, check the CSV output.")
