# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""mHC residual layout conversions (FlyDSL).

The mHC gemm reads and writes the residual in a shuffled layout::

    resS[kb][head][row][kk]  <-  res[row][head][k],   k = kb * KS + kk

``KS`` is the kernel-side tile width, so one ``kk`` run is a contiguous
``KS * itemsize`` byte block in *both* layouts. That makes all three
conversions pure block permutations -- nothing is read, combined or computed,
blocks only move -- so each one is a copy whose only interesting part is the
index math::

    repeat     dst shuffled, src a plain [M, hidden] row broadcast over head
    shuffle    dst shuffled, src res[row][head]
    unshuffle  dst res[row][head], src shuffled

Both run indices are affine in (row, head, kb), and a module is built per
(hidden_size, hc_mult, KS, itemsize), so every divisor is a compile-time power
of two and the addressing is shifts and adds.

Whichever side is shuffled takes the scattered accesses, and the two sides are
not interchangeable: a run is only 64 bytes and consecutive kb sit ``hc_mult *
M`` runs apart in the shuffled layout, so scattering the stores costs several
times what scattering the loads does. Each direction therefore walks its own
destination in address order:

    to shuffled (repeat, shuffle)   block = (rows tile, head, kb)
    from shuffled (unshuffle)       block = (row, head), threads walk kb

Block : (BLOCK_THREADS, 1, 1)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32

from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_copy_atom,
    ptr_buf_tensor,
)

BLOCK_THREADS = 256

# One buffer_{load,store}_dwordx4 per unit.
UNIT_BYTES = 16
UNIT_I32 = UNIT_BYTES // 4

KINDS = ("repeat", "shuffle", "unshuffle")


def units_per_run(ks: int, itemsize: int) -> int:
    """Number of ``UNIT_BYTES`` accesses that tile one KS run."""
    run_bytes = ks * itemsize
    if run_bytes % UNIT_BYTES:
        raise ValueError(
            f"a KS run is {run_bytes} bytes, which {UNIT_BYTES}-byte units cannot tile"
        )
    units = run_bytes // UNIT_BYTES
    if units & (units - 1):
        raise ValueError(f"units per run must be a power of two, got {units}")
    return units


def to_shuffled_row_blocks(m: int, ks: int, itemsize: int) -> int:
    """Grid x extent for the repeat/shuffle direction."""
    total = m * units_per_run(ks, itemsize)
    return (total + BLOCK_THREADS - 1) // BLOCK_THREADS


def build_mhc_res_layout_module(
    kind: str,
    hidden_size: int,
    hc_mult: int,
    ks: int,
    itemsize: int,
):
    """Return a JIT launcher for one mHC residual layout conversion.

    Launcher signature: ``(src, dst, M, row_blocks, stream)``. ``src`` and
    ``dst`` are contiguous tensors of the element type ``itemsize`` came from;
    ``row_blocks`` is :func:`to_shuffled_row_blocks` and is ignored by the
    unshuffle direction.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if hidden_size % ks:
        raise ValueError(f"hidden_size {hidden_size} is not a multiple of KS {ks}")

    run_units = units_per_run(ks, itemsize)
    unit_shift = run_units.bit_length() - 1
    unit_mask = run_units - 1
    num_kb = hidden_size // ks
    # A (row, head) slab is the whole hidden dim: num_kb runs.
    units_per_slab = num_kb * run_units
    slab_iters = (units_per_slab + BLOCK_THREADS - 1) // BLOCK_THREADS

    # repeat reads one plain row per (row, head): its source ignores head.
    per_head_src = kind != "repeat"
    module_name = f"mhc_res_{kind}_h{hidden_size}_hc{hc_mult}_ks{ks}_b{itemsize}"

    if kind == "unshuffle":

        @flyc.kernel(name=module_name)
        def layout_kernel(src: fx.Pointer, dst: fx.Pointer, M: Int32):
            row = fx.Uint32(fx.block_idx.x)
            head = fx.Uint32(fx.block_idx.y)
            tid = fx.Uint32(fx.thread_idx.x)
            rows = fx.Uint32(M)

            src_t = ptr_buf_tensor(src, fx.Int32, unit_elems=UNIT_I32)
            dst_t = ptr_buf_tensor(dst, fx.Int32, unit_elems=UNIT_I32)
            atom = buf_copy_atom(UNIT_BYTES, fx.Int32)
            # Hoisted: a per-iteration fragment costs more than the copy.
            frag = fx.make_fragment_like(fx.slice(src_t, (0, None)))

            # res[row][head] is one contiguous hidden-sized slab, so the flat
            # unit index is the store offset and the block writes one span.
            slab_base = (row * hc_mult + head) * units_per_slab
            for it in range_constexpr(slab_iters):
                uidx = tid + it * BLOCK_THREADS
                if uidx < fx.Uint32(units_per_slab):
                    kb = uidx >> fx.Uint32(unit_shift)
                    sub = uidx & fx.Uint32(unit_mask)
                    shuffled_run = (kb * hc_mult + head) * rows + row
                    s_unit = (shuffled_run << fx.Uint32(unit_shift)) + sub
                    fx.copy(atom, fx.slice(src_t, (s_unit, None)), frag)
                    fx.copy(atom, frag, fx.slice(dst_t, (slab_base + uidx, None)))

        @flyc.jit
        def launch_mhc_res_layout(
            src: fx.Pointer,
            dst: fx.Pointer,
            M: fx.Int32,
            row_blocks: fx.Int32,
            stream: fx.Stream,
        ):
            launcher = layout_kernel(src, dst, M)
            launcher.launch(
                grid=(fx.Int64(M), fx.Int64(hc_mult), 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    else:

        @flyc.kernel(name=module_name)
        def layout_kernel(src: fx.Pointer, dst: fx.Pointer, M: Int32):
            head = fx.Uint32(fx.block_idx.y)
            kb = fx.Uint32(fx.block_idx.z)
            tid = fx.Uint32(fx.thread_idx.x)
            rows = fx.Uint32(M)

            src_t = ptr_buf_tensor(src, fx.Int32, unit_elems=UNIT_I32)
            dst_t = ptr_buf_tensor(dst, fx.Int32, unit_elems=UNIT_I32)
            atom = buf_copy_atom(UNIT_BYTES, fx.Int32)
            frag = fx.make_fragment_like(fx.slice(src_t, (0, None)))

            # The (kb, head) slab is M runs back to back, so the flat unit
            # index is the store offset and the block writes one span.
            uidx = fx.Uint32(fx.block_idx.x) * BLOCK_THREADS + tid
            if uidx < rows * run_units:
                row = uidx >> fx.Uint32(unit_shift)
                sub = uidx & fx.Uint32(unit_mask)
                if const_expr(per_head_src):
                    plain_row = row * hc_mult + head
                else:
                    plain_row = row
                s_unit = ((plain_row * num_kb + kb) << fx.Uint32(unit_shift)) + sub
                slab_base = ((kb * hc_mult + head) * rows) << fx.Uint32(unit_shift)
                fx.copy(atom, fx.slice(src_t, (s_unit, None)), frag)
                fx.copy(atom, frag, fx.slice(dst_t, (slab_base + uidx, None)))

        @flyc.jit
        def launch_mhc_res_layout(
            src: fx.Pointer,
            dst: fx.Pointer,
            M: fx.Int32,
            row_blocks: fx.Int32,
            stream: fx.Stream,
        ):
            launcher = layout_kernel(src, dst, M)
            launcher.launch(
                grid=(fx.Int64(row_blocks), fx.Int64(hc_mult), fx.Int64(num_kb)),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    launch_mhc_res_layout.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_mhc_res_layout
