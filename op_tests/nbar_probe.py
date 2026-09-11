# SPDX-License-Identifier: MIT
# Throwaway probe: does FlyDSL emit + allocate gfx1250 named barriers?
# Declares @__nbar[4] (target-ext), one wave inits(mc=2)+barrier, every wave
# joins/signal_var/wait on bar[warp&3]. Compile-check the ISA/descriptor, then
# a GPU run to confirm the 2-wave rendezvous actually synchronizes.
import os

os.environ["AITER_ENABLE_EXPERIMENTAL"] = "1"

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as L
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import gpu, rocdl

WAVE = 32
NW = 8
NBAR = 4
MC = 2  # member count = 2 waves (the SIMD pair)

_PTR3 = "!llvm.ptr<3>"
_ARR = '!llvm.array<4 x target<"amdgcn.named.barrier", 0>>'


def _inject_global():
    gbody = CompilationContext.get_current().gpu_module_body
    # avoid double-declare across retraces
    for op in gbody.operations:
        if op.operation.name == "llvm.mlir.global":
            return
    with ir.InsertionPoint.at_block_begin(gbody):
        g = ir.Operation.parse(
            f"llvm.mlir.global internal @__nbar() {{addr_space = 3 : i32}} : {_ARR}"
        )
        gbody.append(g)


def _bar_ptr(grp_i32_val):
    ptr3 = ir.Type.parse(_PTR3)
    arr = ir.Type.parse(_ARR)
    i32 = ir.IntegerType.get_signless(32)
    nwf = ir.Attribute.parse("#llvm.gep_no_wrap_flags<none>")
    base = L.AddressOfOp(ptr3, "__nbar").result
    c0 = L.ConstantOp(i32, ir.IntegerAttr.get(i32, 0)).result
    return L.getelementptr(
        ptr3, base, [c0, grp_i32_val], [-2147483648, -2147483648], arr, nwf
    )


@flyc.kernel
def nbar_probe(Out: fx.Tensor):
    _inject_global()
    tid = fx.Int32(gpu.thread_id("x"))
    wave = tid // fx.Int32(WAVE)
    grp = wave % fx.Int32(NBAR)  # (warp & 3), non-negative

    # Each wave inits its own group barrier to member count 2 (idempotent across the
    # pair), publish via workgroup barrier, then join + signal + wait.
    bp = _bar_ptr(grp.ir_value())
    rocdl.s_barrier_init(bp, MC)
    gpu.barrier()
    rocdl.s_barrier_join(bp)
    rocdl.s_barrier_signal_var(bp, MC)
    rocdl.s_barrier_wait(1)

    # trivial write so the launch is observable
    buf = fx.rocdl.make_buffer_tensor(Out)
    tO = fx.logical_divide(buf, fx.make_layout(1, 1))
    copy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
    rO = fx.make_rmem_tensor(1, fx.Int32)
    fx.memref_store_vec(fx.Vector.from_elements([grp], fx.Int32), rO)
    fx.copy_atom_call(copy, rO, fx.slice(tO, (None, tid)))


@flyc.jit
def launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    nbar_probe(Out).launch(grid=(1,), block=(NW * WAVE,), stream=stream)


if __name__ == "__main__":
    out = torch.full((NW * WAVE,), -1, dtype=torch.int32, device="cuda")
    launch(out)
    torch.cuda.synchronize()
    print("out (per-thread grp = warp&3):")
    print(out.view(NW, WAVE)[:, 0].tolist())  # one per wave
    print("OK: no hang" if True else "")
