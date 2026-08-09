#!/usr/bin/env python3
"""Device-free compile gate for FlyDSL kernels.

Compiles kernels with COMPILE_ONLY=1 and dumps final ISA, so a migration can be
checked by diffing ISA before/after with no GPU. Pointer args are faked via
from_c_void_p, which needs no device allocation.

Usage:
    PYTHONPATH=/tmp/envstub FLYDSL_GPU_ARCH=gfx950 COMPILE_ONLY=1 \
    FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir> \
    python compile_gate.py [name ...]
"""
import hashlib
import os
import sys
import traceback

from flydsl.compiler.jit_argument import from_c_void_p
from flydsl.expr.typing import Int32

ADDR = 0x100000


def P(elem=Int32):
    return from_c_void_p(elem, ADDR)


def _route_maps():
    from aiter.ops.flydsl.kernels.moe_route_maps import build_moe_route_maps_module

    build_moe_route_maps_module()(P(), P(), P(), P(), 16, 2, 8, 1)


def _topids_to_rows():
    from aiter.ops.flydsl.kernels.moe_route_maps import (
        build_moe_topids_to_rows_module,
    )

    build_moe_topids_to_rows_module()(P(), P(), P(), 16, 8, 1)


# --- buffer_ops descriptor port (commit 54db2345e) ---------------------------
# moe_gemm_2stage: exercises create_buffer_resource_from_addr (2 sites) plus
# buffer_load (38) / buffer_store (4) -> the _create_i32_constant / _mask_offset
# helpers that the port rewrote. compile_moe_gemm2 returns a @flyc.jit launcher
# and compiles LAZILY, so the launcher must be CALLED with fake args (9 pointers
# + 4 i32 + stream) to force one compile.
#
# NOTE: compile_moe_gemm1 (all in_dtype) and compile_moe_gemm2 fp8/int8/bf16 are
# uncompilable in the current working tree (pre-existing 'a0' UnboundLocalError /
# rocdl.exp2 errors, IDENTICAL at 54db2345e~1 -> unrelated to buffer_ops), so
# only the a16wi4 (int4_bf16) gemm2 path is gated here.
def _g2_args():
    return (P(), P(), P(), P(), P(), P(), P(), P(), P(),
            64, 4096, 1024, 64, __import__("flydsl.expr", fromlist=["Stream"]).Stream(None))


def _g2(tile_m, tile_n, **kw):
    from aiter.ops.flydsl.kernels.moe_gemm_2stage import compile_moe_gemm2

    launcher = compile_moe_gemm2(
        model_dim=4096, inter_dim=1024, experts=8, topk=2,
        tile_m=tile_m, tile_n=tile_n, tile_k=128, in_dtype="int4_bf16",
        group_size=32, scale_is_bf16=True, out_dtype="bf16", **kw)
    launcher(*_g2_args())


def _moe_g2_i4_tn256():
    _g2(16, 256, doweight_stage2=True)


def _moe_g2_i4_tn128():
    _g2(16, 128, doweight_stage2=True)


def _moe_g2_i4_noacc():
    _g2(16, 256, doweight_stage2=True, accumulate=False)


def _moe_g2_i4_tm32():
    _g2(32, 256, doweight_stage2=True)


def _moe_g2_i4_nodw():
    _g2(16, 256, doweight_stage2=False)


# preshuffle_gemm: exercises create_buffer_resource (3) + buffer_load (3).
# The launcher takes fx.Tensor args and compiles lazily, so it must be CALLED.
# Tensor args are faked with torch under FakeTensorMode (device-free), mirroring
# aiter/aot/flydsl/gemm.py::_compile_preshuffle_to_cache.
def _ps(in_dtype, tile_k, **kw):
    import torch
    import flydsl.expr as fx
    from torch._subclasses.fake_tensor import FakeTensorMode

    from aiter.ops.flydsl.kernels.preshuffle_gemm import compile_preshuffle_gemm

    m = n = k = 4096
    atype = {"fp8": torch.float8_e4m3fn, "int8": torch.int8,
             "bf16": torch.bfloat16, "fp16": torch.float16}[in_dtype]
    with FakeTensorMode():
        a = torch.empty((m * k,), dtype=atype)
        b = torch.empty((n * k,), dtype=atype)
        out = torch.empty((m * n,), dtype=torch.bfloat16)
        sa = torch.empty((m,), dtype=torch.float32)
        sb = torch.empty((n,), dtype=torch.float32)
        bias = torch.empty(0, dtype=torch.bfloat16)
        exe = compile_preshuffle_gemm(N=n, K=k, tile_m=128, tile_n=128,
                                      tile_k=tile_k, in_dtype=in_dtype,
                                      out_dtype="bf16", **kw)
        exe(out, a, b, sa, sb, bias, m, n, fx.Stream(0))


def _ps_fp8():
    _ps("fp8", 128)


def _ps_int8():
    _ps("int8", 128)


def _ps_bf16():
    _ps("bf16", 64)


def _ps_fp8_bias():
    _ps("fp8", 128, epilogue="bias_silu")


# qk_norm_rope_quant_gfx1250 (RDNA / gfx1250): the ONLY case that exercises the
# is_rdna_arch flags branch of the descriptor (bit24 + OOB_SELECT=2). Uses
# create_buffer_resource (3) + create_buffer_resource_from_addr (2) +
# buffer_load/store. Requires FLYDSL_GPU_ARCH=gfx1250.
#
# The launcher mixes fx.Pointer and fx.Tensor args and compiles lazily. Tensor
# args use _cached_from_dlpack (real CPU torch tensors, device-free); pointer
# args use from_c_void_p. Config is DeepSeek-V4 (H=16, D=512, RD=64); D=128
# hits an unrelated fp8-pack assertion so is avoided.
def _qk(quant, scale_dtype, q_weighted, group_size=128):
    import torch
    import flydsl.expr as fx
    from flydsl.expr.typing import Uint8

    from aiter.ops.flydsl.kernels.qk_norm_rope_quant_gfx1250 import (
        _cached_from_dlpack,
        compile_flydsl_qk_norm_rope_quant_gfx1250 as C,
    )

    H, D, RD = 16, 512, 64
    pu = lambda: from_c_void_p(Uint8, ADDR)  # noqa: E731
    launcher = C(num_q_heads=H, head_dim=D, rope_head_dim=RD, quant=quant,
                 group_size=group_size, scale_dtype=scale_dtype,
                 q_weighted=q_weighted)
    qw = _cached_from_dlpack(torch.zeros((D,), dtype=torch.bfloat16))
    kw = _cached_from_dlpack(torch.zeros((D,), dtype=torch.bfloat16))
    cos = _cached_from_dlpack(torch.zeros((16, RD // 2), dtype=torch.float32))
    sin = _cached_from_dlpack(torch.zeros((16, RD // 2), dtype=torch.float32))
    launcher(pu(), pu(), qw, kw, cos, sin, pu(), pu(), pu(), pu(), pu(),
             D, pu(), pu(), pu(), 0, 0, 1, 64, fx.Stream(0))


def _qk_quant_e8m0():
    _qk(True, "e8m0", False)


def _qk_quant_fp32():
    _qk(True, "fp32", False)


def _qk_noquant():
    _qk(False, "fp32", False)


def _qk_quant_qweighted():
    _qk(True, "e8m0", True)


def _qk_quant_g64():
    _qk(True, "e8m0", False, group_size=64)


# --- moe_sorting migration (commit 323058f94) --------------------------------
# 82 sites moved to the layout API. Both entry points return lazily-compiled
# launchers taking fx.Tensor args (faked with torch under FakeTensorMode).
# The multiphase p0v2_p23 launcher fuses all its kernels into ONE module, so
# one ISA hash covers the whole path. Covers oneshot + multiphase and both
# has_mask values (plus has_local_tokens).
def _ms_oneshot(E, topk, mt, has_mask, has_local):
    import torch
    import flydsl.expr as fx
    from torch._subclasses.fake_tensor import FakeTensorMode

    from aiter.ops.flydsl.kernels.moe_sorting_kernel import (
        _compile_moe_sorting_oneshot,
    )

    L = _compile_moe_sorting_oneshot(
        num_experts=E, topk=topk, max_tokens=mt,
        has_mask=has_mask, has_local_tokens=has_local)
    i32, f32 = torch.int32, torch.float32
    with FakeTensorMode():
        L(torch.empty((mt, topk), dtype=i32), torch.empty((mt, topk), dtype=f32),
          torch.empty((mt * topk + E * mt,), dtype=i32),
          torch.empty((mt * topk + E * mt,), dtype=f32),
          torch.empty((mt * topk + E,), dtype=i32), torch.empty((2,), dtype=i32),
          torch.empty((0, 0), dtype=i32),
          torch.ones((E if has_mask else 1,), dtype=i32),
          torch.zeros((1,), dtype=i32), mt, 0, 8, fx.Stream(None))


def _ms_multiphase(E, topk, has_mask, has_local):
    import torch
    import flydsl.expr as fx
    from torch._subclasses.fake_tensor import FakeTensorMode

    from aiter.ops.flydsl.kernels.moe_sorting_kernel import (
        _compile_moe_sorting_multiphase,
    )

    launch_p0v2_p23 = _compile_moe_sorting_multiphase(
        num_experts=E, topk=topk, unit_size=32,
        has_mask=has_mask, has_local_tokens=has_local, k4_block=256)[5]
    i32, f32 = torch.int32, torch.float32
    T = 256
    with FakeTensorMode():
        args = [torch.empty((T, topk), dtype=i32),
                torch.empty((1 << 20,), dtype=i32),
                torch.empty((T, topk), dtype=f32),
                torch.empty((T * topk + E * 128,), dtype=i32),
                torch.empty((T * topk + E * 128,), dtype=f32),
                torch.empty((T * topk + E,), dtype=i32),
                torch.empty((2,), dtype=i32), torch.empty((0, 0), dtype=i32),
                torch.ones((E if has_mask else 1,), dtype=i32),
                torch.zeros((1,), dtype=i32)]
        launch_p0v2_p23(*args, T, 32, 1024, 0, 8, fx.Stream(None))


def _ms_oneshot_mask0():
    _ms_oneshot(8, 2, 128, False, False)


def _ms_oneshot_mask1():
    _ms_oneshot(8, 2, 128, True, False)


def _ms_oneshot_loc1():
    _ms_oneshot(8, 2, 128, False, True)


def _ms_oneshot_e256():
    _ms_oneshot(256, 8, 128, False, False)


def _ms_mp_mask0():
    _ms_multiphase(8, 2, False, False)


def _ms_mp_mask1():
    _ms_multiphase(8, 2, True, False)


def _ms_mp_loc1():
    _ms_multiphase(8, 2, False, True)


def _ms_mp_e256():
    _ms_multiphase(256, 8, False, False)


# --- causal_conv1d migration (commit cfbeaaa8b) ------------------------------
# 21 sites moved off the buffer_ops shim. Builder returns a lazily-compiled
# launcher (12 fx.Tensor + 19 Int32 + stream); Tensor args faked with torch
# under FakeTensorMode, mirroring the real caller's stride-arg construction.
def _cc(width, has_bias, silu, dtype_str="bf16"):
    import torch
    import flydsl.expr as fx
    from torch._subclasses.fake_tensor import FakeTensorMode

    from aiter.ops.flydsl.causal_conv1d_flydsl import (
        build_causal_conv1d_flydsl_module as B,
    )

    L = B(width, has_bias, silu, tm=64, tn=64, block_threads=256,
          dtype_str=dtype_str)
    dt = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    i32, i8 = torch.int32, torch.int8
    dim = kd = vd = 128
    cu = 64
    with FakeTensorMode():
        x = torch.empty((cu, dim), dtype=dt)
        w = torch.empty((dim, width), dtype=dt)
        bias = torch.empty((dim,), dtype=dt)
        cs = torch.empty((4, dim, width), dtype=dt)
        cidx = torch.empty((4,), dtype=i32)
        hinit = torch.empty((4,), dtype=i8)
        qsl = torch.empty((5,), dtype=i32)
        batch = torch.empty((5,), dtype=i32)
        coff = torch.empty((5,), dtype=i32)
        q = torch.empty((cu, kd), dtype=dt)
        k = torch.empty((cu, kd), dtype=dt)
        v = torch.empty((cu, vd), dtype=dt)
        L(x, w, bias, cs, cidx, hinit, qsl, batch, coff, q, k, v,
          dim, kd, vd, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
          cs.stride(0), cs.stride(1), cs.stride(2), cidx.stride(0),
          q.stride(0), q.stride(1), k.stride(0), k.stride(1),
          v.stride(0), v.stride(1), 4, 2, fx.Stream(None))


def _cc_w4_bias_silu():
    _cc(4, True, True)


def _cc_w2_plain():
    _cc(2, False, False)


def _cc_w3_bias():
    _cc(3, True, False)


def _cc_w4_silu():
    _cc(4, False, True)


def _cc_w3_fp16():
    _cc(3, False, False, "fp16")


def _cc_w2_bias_silu():
    _cc(2, True, True)


# name -> zero-arg callable that triggers one compile
CASES = {
    "route_maps": _route_maps,
    "topids_to_rows": _topids_to_rows,
    # buffer_ops target (1)
    "moe_g2_i4_tn256": _moe_g2_i4_tn256,
    "moe_g2_i4_tn128": _moe_g2_i4_tn128,
    "moe_g2_i4_noacc": _moe_g2_i4_noacc,
    "moe_g2_i4_tm32": _moe_g2_i4_tm32,
    "moe_g2_i4_nodw": _moe_g2_i4_nodw,
    "ps_fp8": _ps_fp8,
    "ps_int8": _ps_int8,
    "ps_bf16": _ps_bf16,
    "ps_fp8_bias": _ps_fp8_bias,
    "qk_quant_e8m0": _qk_quant_e8m0,
    "qk_quant_fp32": _qk_quant_fp32,
    "qk_noquant": _qk_noquant,
    "qk_quant_qweighted": _qk_quant_qweighted,
    "qk_quant_g64": _qk_quant_g64,
    # moe_sorting target (2)
    "ms_oneshot_mask0": _ms_oneshot_mask0,
    "ms_oneshot_mask1": _ms_oneshot_mask1,
    "ms_oneshot_loc1": _ms_oneshot_loc1,
    "ms_oneshot_e256": _ms_oneshot_e256,
    "ms_mp_mask0": _ms_mp_mask0,
    "ms_mp_mask1": _ms_mp_mask1,
    "ms_mp_loc1": _ms_mp_loc1,
    "ms_mp_e256": _ms_mp_e256,
    # causal_conv1d target (3)
    "cc_w4_bias_silu": _cc_w4_bias_silu,
    "cc_w2_plain": _cc_w2_plain,
    "cc_w3_bias": _cc_w3_bias,
    "cc_w4_silu": _cc_w4_silu,
    "cc_w3_fp16": _cc_w3_fp16,
    "cc_w2_bias_silu": _cc_w2_bias_silu,
}


def main():
    wanted = sys.argv[1:] or sorted(CASES)
    rc = 0
    for name in wanted:
        try:
            CASES[name]()
            print(f"[ok]   {name}")
        except Exception as e:  # noqa: BLE001 - report and keep going
            rc = 1
            print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}")
            if os.environ.get("GATE_TRACEBACK"):
                traceback.print_exc()

    dump = os.environ.get("FLYDSL_DUMP_DIR")
    if dump and os.path.isdir(dump):
        print("--- ISA ---")
        for root, _, files in sorted(os.walk(dump)):
            for f in sorted(files):
                if f.endswith("_final_isa.s"):
                    p = os.path.join(root, f)
                    with open(p, "rb") as fh:
                        h = hashlib.sha256(fh.read()).hexdigest()[:16]
                    print(f"{h}  {os.path.basename(root)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
