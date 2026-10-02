# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""EP global->local expert LUT build (FlyDSL), single-block parallel scan.

Collapses the host ``ne + cumsum + sub + where`` chain (6 elementwise/scan
launches) into one kernel: read the (E_global,) 0/1 ``expert_mask``, scan first
within each wave and then across wave totals, and write
``g2l_lut[i] = mask[i] ? prefix_incl[i]-1 : E`` (sentinel ``E`` = dropped route).

Mirrors ``moe_contiguous_psum`` (same single-block scan idiom) so the whole
gfx1250 grouped path stays on one compiler/runtime instead of pulling Triton
into the decode hot path. E_global fits in a single workgroup for supported
models; larger masks fall back to torch (see grouped_moe_gfx1250).

Also zero-inits the ``(E,)`` per-bucket route counter as a side output, folding
the separate host ``torch.zeros(E)`` launch (that ``moe_route_g2l`` atomically
increments) into this same pre-route kernel.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu

from aiter.ops.flydsl.kernels.kernels_common import get_warp_size
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_buf_tensor,
)
from aiter.ops.flydsl.kernels.topk.topk_per_row_decode import (
    _warp_inclusive_prefix_i32,
)

MAX_G2L_EXPERTS = 1024


def build_moe_g2l_lut_module(
    clear_counter: bool = True,
    max_experts: int = 512,
):
    """JIT launcher: single-block build of the EP global->local expert LUT."""
    wave_size = get_warp_size()
    if (
        max_experts <= 0
        or max_experts > MAX_G2L_EXPERTS
        or max_experts & (max_experts - 1)
        or max_experts > wave_size * wave_size
    ):
        raise ValueError(
            f"max_experts must be a power of two <= "
            f"min({MAX_G2L_EXPERTS}, wave_size**2), got {max_experts}"
        )
    num_waves = max_experts // wave_size

    @fx.struct
    class SharedStorage:
        wave_prefix: fx.Array[fx.Int32, num_waves, 16]

    @flyc.kernel(
        name=f"moe_g2l_lut_n{max_experts}_w{wave_size}",
        known_block_size=[max_experts, 1, 1],
    )
    def g2l_kernel(
        mask: fx.Pointer,  # (n,) int32 0/1 expert mask
        lut: fx.Pointer,  # (n,) int32 out: global->local, sentinel E
        counter: fx.Pointer,  # (E,) int32 out: per-bucket route counter, zeroed
        nvt: fx.Pointer,  # (1,) int32 in: num_local_tokens (= total_recv)
        nvr_out: fx.Pointer,  # (1,) int32 out: num_valid_routes = nvt * topk
        n: fx.Int32,
        E: fx.Int32,
        topk: fx.Int32,
    ):
        c0 = fx.Int32(0)
        c1 = fx.Int32(1)
        tid = gpu.thread_idx.x
        lane = tid % fx.Int32(wave_size)
        wave = tid // fx.Int32(wave_size)

        mask_p = ptr_buf_tensor(mask)
        lut_p = ptr_buf_tensor(lut)

        # num_valid_routes = num_local_tokens * topk (the EP dead-tail bound),
        # folded in here to drop a standalone torch elementwise launch at decode.
        if tid == c0:
            ptr_buf_tensor(nvr_out)[0] = ptr_buf_tensor(nvt)[0] * topk

        # Generic grouped-MoE owns this reset. MegaMoE's TDM dispatch can
        # instead fold it into its existing tail and compile these stores
        # away while keeping this kernel available to other callers.
        if const_expr(clear_counter) and tid < E:
            ptr_buf_tensor(counter)[tid] = c0

        in_range = tid < n
        enabled = c0
        if in_range:
            enabled = (mask_p[tid] != c0).select(c1, c0)
        inclusive = _warp_inclusive_prefix_i32(enabled, lane, wave_size)

        storage = fx.SharedAllocator().allocate(SharedStorage)
        wave_prefix = storage.wave_prefix.peek().view(fx.make_layout(num_waves, 1))
        if lane == fx.Int32(wave_size - 1):
            wave_prefix[wave] = inclusive
        gpu.barrier()

        if wave == c0:
            active = lane < fx.Int32(num_waves)
            safe_lane = active.select(lane, c0)
            wave_total = active.select(wave_prefix[safe_lane], c0)
            prefix = (
                _warp_inclusive_prefix_i32(wave_total, lane, wave_size) - wave_total
            )
            if active:
                wave_prefix[lane] = prefix
        gpu.barrier()

        if in_range:
            local = wave_prefix[wave] + inclusive - c1
            lut_p[tid] = (enabled != c0).select(local, E)

    @flyc.jit
    def launch_g2l(
        mask: fx.Pointer,
        lut: fx.Pointer,
        counter: fx.Pointer,
        nvt: fx.Pointer,
        nvr_out: fx.Pointer,
        n: fx.Int32,
        E: fx.Int32,
        topk: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        g2l_kernel(mask, lut, counter, nvt, nvr_out, n, E, topk).launch(
            grid=(1, 1, 1),
            block=(max_experts, 1, 1),
            stream=stream,
        )

    launch_g2l.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_g2l
