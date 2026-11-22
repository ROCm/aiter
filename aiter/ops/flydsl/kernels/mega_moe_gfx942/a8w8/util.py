# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Helpers of the a8w8 EP8 mega MoE kernels (from pyhip contrib/flydsl helpers.py and
gemm_a8w8_blockscale_gfx942_common.py, branch luwei/942-fp8-megamoe)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, rocdl
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import is_target_address_space


def run_compiled(launch, *args):
    """Runs a @flyc.jit launcher through its CompiledFunction for the current device.

    Calling the JitFunction costs 60-90 us of host time per launch, the CompiledFunction a few us. The first call
    on a device compiles (or loads from the FlyDSL cache) and runs it.
    """
    by_device = getattr(launch, "_compiled_by_device", None)
    if by_device is None:
        by_device = launch._compiled_by_device = {}
    device = torch.cuda.current_device()
    compiled = by_device.get(device)
    if compiled is None:
        by_device[device] = flyc.compile(launch, *args)
    else:
        compiled(*args)


def global_ptr(addr):
    """!llvm.ptr<1> at the i64 byte address `addr`."""
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), fx.Int64(addr).ir_value())


def atomic_add_i32(addr, value, scope="agent-one-as"):
    """Monotonic i32 fetch-add at an i64 address ("agent-one-as": device, "one-as": system); returns the old value."""
    op = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        global_ptr(addr),
        fx.Int32(value).ir_value(),
        llvm.AtomicOrdering.monotonic,
        syncscope=scope,
        alignment=4,
    )
    return fx.Int32(op.result)


def load_i32_atomic(addr, scope="agent-one-as"):
    """Monotonic i32 load at an i64 address, coherent at `scope` (for polling a flag)."""
    i32 = ir.IntegerType.get_signless(32)
    return fx.Int32(
        llvm.LoadOp(
            i32,
            global_ptr(addr),
            alignment=4,
            ordering=llvm.AtomicOrdering.monotonic,
            syncscope=scope,
        ).result
    )


def fence(ordering, scope="agent-one-as"):
    """LLVM fence: "release" writes back dirty L2 lines and waits for prior memory ops, "acquire" invalidates caches."""
    llvm.FenceOp(getattr(llvm.AtomicOrdering, ordering), syncscope=scope)


@flyc.jit
def spin_wait_ge(addr, target, scope, l1_only=False):
    """Poll the i32 at `addr` until it is >= target, then acquire at `scope`.

    l1_only: invalidate only the L1 instead (`buffer_inv sc0`). On gfx942 the scoped acquire also walks the L2 for
    non-local (MTYPE NC) lines; local memory lines are kept coherent by probes, so data in local memory that was
    written through (or written back) before the flag needs only the L1 invalidated.
    """
    seen = load_i32_atomic(addr, scope)
    while seen < target:
        fx.rocdl.s_sleep(2)
        seen = load_i32_atomic(addr, scope)
    if l1_only:
        llvm.inline_asm(
            ir.Type.parse("!llvm.void"), [], "buffer_inv sc0", "", has_side_effects=True
        )
    else:
        fence("acquire", scope)


def buffer_rsrc_at(addr, num_bytes):
    """Buffer resource for `num_bytes` at an i64 address (e.g. a peer's buffer from a pointer table)."""
    ptr = fx.inttoptr(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16), fx.Int64(addr)
    )
    return fx.rocdl.get_buffer_rsrc(
        fx.rocdl.make_buffer_ptr(ptr, num_records_bytes=num_bytes)
    )


def _offset_i32(value):
    """只提取已经准备好的i32 SSA；动态trunc/单位换算必须在调用之前完成。"""
    if isinstance(value, (fx.Int32, fx.Uint32)):
        return value.ir_value()
    if (
        isinstance(value, ir.Value)
        and isinstance(value.type, ir.IntegerType)
        and value.type.width == 32
    ):
        return value
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and -(1 << 31) <= value < (1 << 32)
    ):
        return fx.Uint32(value & 0xFFFFFFFF).ir_value()
    raise TypeError("offset需要预计算的32位SSA或32位编译期整数")


class BufferTensor(fx.Tensor):
    """buffer Tensor的单条原生访存；不缓存额外动态Python字段。"""

    @dsl_loc_tracing
    def __getitem__(self, coord):
        result = super().__getitem__(coord)
        return type(self)(result) if isinstance(result, fx.Tensor) else result

    def _packet_bits(self):
        if not is_target_address_space(
            self.address_space, TargetAddressSpace.BufferDesc
        ):
            raise TypeError("显式buffer访存需要BufferDesc Tensor")
        if (
            not self.layout.is_static
            or fx.rank(self.layout) != 1
            or self.stride.to_py_value() not in (1, (1,))
        ):
            raise ValueError("显式buffer访存只接受静态连续一维packet")
        bits = fx.size(self).to_py_value() * self.dtype.width
        if bits not in (32, 64, 128):
            raise ValueError("packet只支持32/64/128bit；16B+8B必须显式发出两条请求")
        return bits

    @dsl_loc_tracing
    def load(self, *, voffset_bytes=None, soffset_bytes=0, aux=0):
        if voffset_bytes is None:
            if not isinstance(soffset_bytes, int) or soffset_bytes != 0 or aux != 0:
                raise ValueError("显式soffset/aux需要同时提供完整voffset_bytes")
            return super().load()
        bits = self._packet_bits()
        if not isinstance(aux, int):
            raise TypeError("aux必须是编译期整数")
        resource = fx.rocdl.get_buffer_rsrc(fx.get_iter(self))
        result = rocdl.RawPtrBufferLoadOp(
            ir.VectorType.get([bits // 32], fx.Uint32.ir_type),
            resource,
            _offset_i32(voffset_bytes),
            _offset_i32(soffset_bytes),
            aux=ir.IntegerAttr.get(fx.Int32.ir_type, aux),
        ).result
        return Vec(result).bitcast(self.dtype)


def pack_bf16x2_rne(values):
    """Four FP32 -> two BF16x2 dwords, rounded to nearest even (pack_bf16x2 truncates normal values)."""
    bits = [fx.Float32(values[index]).bitcast(fx.Int32) for index in range(4)]
    rounded = [bit + (0x7FFF + (bit >> 16 & 1)) for bit in bits]
    return [
        rounded[pair * 2] >> 16 & 0xFFFF | rounded[pair * 2 + 1] & -65536
        for pair in range(2)
    ]
