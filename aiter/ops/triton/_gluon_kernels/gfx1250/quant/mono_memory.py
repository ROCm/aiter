# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna5 import CachePolicy, buffer


@gluon.jit
def load_dev(base, offsets, mask, USE_BUFFER: gl.constexpr = True):
    """Read BF16 activation values at device scope, with zero for masked lanes."""
    if USE_BUFFER:
        desc = buffer.make_buffer_descriptor(base, 0xFFFFFFFF)
        return buffer.load(desc, offsets.to(gl.int32), mask, other=0, scope="gpu")
    else:
        # Keep 64-bit addressing when a legal column stride exceeds buffer bounds.
        bits, _ = gl.inline_asm_elementwise(
            "v_mov_b32 $0, 0; v_cmp_ne_u32 vcc_lo, 0, $3; "
            "s_and_saveexec_b32 $1, vcc_lo; "
            "global_load_u16 $0, $2, off scope:SCOPE_DEV; "
            "s_wait_loadcnt 0; s_mov_b32 exec_lo, $1;",
            constraints="=&v,=&s,v,v,~{vcc},~{memory}",
            args=[base + offsets, mask.to(gl.int32)],
            dtype=(gl.uint32, gl.uint32),
            is_pure=False,
            pack=1,
        )
        return bits.to(gl.uint16).to(gl.bfloat16, bitcast=True)


@gluon.jit
def store_dev_ht(base, offsets, value, mask, USE_BUFFER: gl.constexpr = True):
    """Store an eight- or sixteen-bit handoff value with device scope and HT."""
    if USE_BUFFER:
        desc = buffer.make_buffer_descriptor(base, 0xFFFFFFFF)
        buffer.store(
            value,
            desc,
            offsets.to(gl.int32),
            mask,
            scope="gpu",
            cache_policy=CachePolicy(temporal="high"),
        )
    else:
        if base.dtype.element_ty.primitive_bitwidth == 16:
            bits = value.to(gl.uint16, bitcast=True).to(gl.uint32)
            instruction: gl.constexpr = "global_store_b16"
        else:
            bits = value.to(gl.uint8, bitcast=True).to(gl.uint32)
            instruction: gl.constexpr = "global_store_b8"
        gl.inline_asm_elementwise(
            "v_cmp_ne_u32 vcc_lo, 0, $3; s_and_saveexec_b32 $0, vcc_lo; "
            + instruction
            + " $1, $2, off th:TH_STORE_HT scope:SCOPE_DEV; "
            "s_mov_b32 exec_lo, $0;",
            constraints="=&s,v,v,v,~{vcc},~{memory}",
            args=[base + offsets, bits, mask.to(gl.int32)],
            dtype=gl.uint32,
            is_pure=False,
            pack=1,
        )
