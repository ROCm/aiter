# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Hyper-connection mix kernels for gfx1250 (Qwen3.8-Flash-Next).

``compile_hc_up_mix`` fuses the hyper-connection up projection with the gated
stream mean, and optionally the scaled SiLU that produces its input:

    gate  = bf16(x * sigmoid(x)), x = bf16(d / HC)   # silu_scale=HC; else gate=d
    up    = gate @ w_up.T                           # [M, HC*H], rounded to BF16
    mixed = mean_s(sigmoid(up[:, s*H + h]) * normed[:, s*H + h])

The up GEMM is the gfx1250 TDM/WMMA a16w16 main loop. ``w_up`` must be passed
with its rows reordered stream-minor (row ``h*HC + s`` holds original row
``s*H + h``, see :func:`hc_interleave_up_weight`), so the eight consecutive
output columns a lane holds after the WMMA are two hidden columns times the
four streams, and the epilogue reduces the streams without leaving the lane.
Rounding matches the unfused path: the up output is rounded to BF16 before the
sigmoid, the stream sum is FP32 in stream order, and the mean is rounded once.

With ``silu_scale`` the whole ``[tile_m, R]`` A operand is loaded into LDS up
front and the workgroup rewrites it once, eight elements per thread, before the
first WMMA. Applying it per fragment instead repeats it in every wave on the
K loop's critical path and costs more than the separate kernel it replaces.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.protocol import dsl_size_of
from flydsl.expr import const_expr, gpu, idx2crd, range_constexpr, rocdl, tdm_ops
from flydsl.expr.typing import T
from flydsl.runtime.device import check_smem_capacity
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from aiter.ops.flydsl.kernels.tensor_shim import ptr_buf_tensor

WMMA_M, WMMA_N, WMMA_K = 16, 16, 32
WAVE_SIZE, LDS_PAD_A, LDS_PAD_B = 32, 8, 8
_TDMS_PER_TILE = 2  # A + B
KERNARG_PRELOAD_COUNT = 8
# Each lane owns 8 consecutive output columns after the WMMA.
_LANE_COLS = 8


def compile_hc_up_mix(
    *,
    H: int,
    HC: int,
    R: int,
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 64,
    m_warp: int = 1,
    n_warp: int = 4,
    num_buffers: int = 2,
    waves_per_eu: int | None = None,
    silu_scale: int = 0,
):
    """Build the fused ``[scaled SiLU +] up GEMM + gated stream mean`` launcher.

    Launcher arguments: ``(mixed, gate, w_up_interleaved, normed, M,
    ld_mixed, ld_gate, ld_normed, stream)``. ``mixed`` is ``[M, H]``,
    ``gate`` ``[M, R]`` (the pre-SiLU down output when ``silu_scale`` is set; its
    row stride may exceed ``R``), ``w_up_interleaved`` ``[HC*H, R]`` and
    ``normed`` ``[M, HC*H]``, all BF16 with unit inner stride. ``silu_scale``
    overrides ``num_buffers``: every K tile is resident.
    """

    def _req(cond, msg):
        if not cond:
            raise ValueError(msg)

    N, K = HC * H, R
    if silu_scale:
        # The SiLU pass rewrites the whole A operand in LDS once, so every K
        # tile is resident before the first WMMA.
        num_buffers = -(-K // tile_k) + 1
    _req(HC == 4, f"the lane-local stream reduction needs HC == 4, got {HC}")
    _req(_LANE_COLS % HC == 0, "a lane's columns must hold whole stream groups")
    _req(2 <= num_buffers <= 8, f"num_buffers must be in [2, 8], got {num_buffers}")
    num_warps, block_threads = m_warp * n_warp, m_warp * n_warp * WAVE_SIZE
    warp_tile_m, warp_tile_n = tile_m // m_warp, tile_n // n_warp
    num_k_tiles = (K + tile_k - 1) // tile_k
    _req(not silu_scale or K % tile_k == 0, "the SiLU pass needs whole K tiles")
    for _nm, _v, _d in (
        ("tile_k", tile_k, WMMA_K),
        ("tile_m", tile_m, WMMA_M),
        ("tile_n", tile_n, WMMA_N),
        ("warp_tile_m", warp_tile_m, WMMA_M),
        ("warp_tile_n", warp_tile_n, WMMA_N),
        ("N", N, tile_n),
    ):
        _req(_v % _d == 0, f"{_nm} ({_v}) must be a multiple of {_d}")
    _req(tile_k & (tile_k - 1) == 0, f"tile_k must be a power of 2, got {tile_k}")
    _req(
        num_k_tiles >= num_buffers - 1,
        f"num_k_tiles {num_k_tiles} < {num_buffers - 1} (K={K}, tile_k={tile_k})",
    )
    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    elem_bytes = 2
    k_wmma_steps = tile_k // WMMA_K
    _fx_elem = fx.BFloat16
    wmma_m_rep, wmma_n_rep = warp_tile_m // WMMA_M, warp_tile_n // WMMA_N
    N_ACCS = wmma_m_rep * wmma_n_rep

    a_tile_shape = (tile_m, tile_k)
    lds_a_stride = tile_k + LDS_PAD_A
    lds_a_elems = tile_m * lds_a_stride + LDS_PAD_A
    b_tile_shape = (tile_n, tile_k)
    lds_b_stride = tile_k + LDS_PAD_B
    lds_b_elems = tile_n * lds_b_stride + LDS_PAD_B

    @fx.struct
    class SharedStorage:
        a: fx.Array[_fx_elem, num_buffers * lds_a_elems, 16]
        b: fx.Array[_fx_elem, num_buffers * lds_b_elems, 16]

    check_smem_capacity(dsl_size_of(SharedStorage), gpu_arch)

    @flyc.kernel(
        name=f"hc_up_mix_h{H}_r{R}_s{silu_scale}_t{tile_m}x{tile_n}x{tile_k}_b{num_buffers}"
    )
    def hc_up_mix_kernel(
        arg_mixed: fx.Pointer,
        arg_gate: fx.Pointer,
        arg_w: fx.Pointer,
        arg_normed: fx.Pointer,
        i32_m: fx.Int32,
        i32_ld_mixed: fx.Int32,
        i32_lda: fx.Int32,
        i32_ld_normed: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()
        elem_ty = _fx_elem.ir_type

        tx, bx, by = gpu.thread_id("x"), gpu.block_id("x"), gpu.block_id("y")
        blk_m, blk_n = fx.Uint64(bx) * tile_m, fx.Uint64(by) * tile_n

        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16), (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1)
        )
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx, lane_kgrp, lane16 = (
            fx.Uint64(fx.get(thr_coord, 0)),
            fx.Uint64(fx.get(thr_coord, 1)),
            fx.Uint64(fx.get(thr_coord, 2)),
            fx.Uint64(fx.get(thr_coord, 3)),
        )
        warp_m_base, warp_n_base = wave_m_idx * warp_tile_m, wave_n_idx * warp_tile_n
        m_idx = fx.Uint64(i32_m)

        lds = fx.SharedAllocator(static=True).allocate(SharedStorage).peek()
        big_a_mem, big_b_mem = lds.a.ptr, lds.b.ptr
        blk_m64, blk_n64 = fx.Int64(blk_m), fx.Int64(blk_n)

        def _mk_side(ptr, base_off, tile, outer_stride, lds_stride, oob):
            shape = [tile[0], K]
            if oob is not None:
                shape[0] = oob
            gt = fx.Tensor(
                fx.make_view(
                    fx.add_offset(ptr, base_off),
                    fx.make_layout(tuple(shape), (outer_stride, 1)),
                )
            )
            atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
                fx.rocdl.TensorLoad(),
                gt,
                fx.make_layout(tile, (lds_stride, 1)),
                tile,
                num_warps=num_warps,
            )
            return atom, fx.zipped_divide(coord, tile)[None, (0, None)]

        m_oob = fx.Int32(m_idx - blk_m)
        lda64 = fx.Int64(fx.Uint64(i32_lda))
        atom_a, blk_a = _mk_side(
            arg_gate, blk_m64 * lda64, a_tile_shape, i32_lda, lds_a_stride, m_oob
        )
        atom_b, blk_b = _mk_side(
            arg_w, blk_n64 * K, b_tile_shape, fx.Int32(K), lds_b_stride, None
        )

        def _slot_off(buf_idx, slot_elems):
            return fx.Uint64(buf_idx * slot_elems)

        def _lane_bases(warp_base, reps, lds_stride):
            row_off = (warp_base + lane16) * lds_stride
            k_off = lane_kgrp * 8
            return [
                row_off + rep * WMMA_M * lds_stride + k_off
                for rep in range_constexpr(reps)
            ]

        MEM, BASES, ELEMS, REPS, ATOM, BLK, SHAPE, STRIDE = range(8)
        A_SIDE = (
            big_a_mem,
            _lane_bases(warp_m_base, wmma_m_rep, lds_a_stride),
            lds_a_elems,
            wmma_m_rep,
            atom_a,
            blk_a,
            a_tile_shape,
            lds_a_stride,
        )
        B_SIDE = (
            big_b_mem,
            _lane_bases(warp_n_base, wmma_n_rep, lds_b_stride),
            lds_b_elems,
            wmma_n_rep,
            atom_b,
            blk_b,
            b_tile_shape,
            lds_b_stride,
        )

        wave = fx.rocdl.wave_id()

        def issue_tdm_loads(buf_idx, k_tile):
            for sd in (A_SIDE, B_SIDE):
                lds_box = fx.Tensor(
                    fx.make_view(
                        fx.add_offset(sd[MEM], _slot_off(buf_idx, sd[ELEMS])),
                        fx.make_layout(
                            (sd[SHAPE], 1),
                            ((sd[STRIDE], 1), sd[SHAPE][0] * sd[STRIDE]),
                        ),
                    )
                )
                ts, tg = fx.rocdl.cdna5.tdm_partition(
                    sd[ATOM], wave, fx.make_layout(num_warps, 1), lds_box, sd[BLK]
                )
                fx.copy(sd[ATOM], tg[None, fx.Int32(k_tile)], ts[None, 0])

        def _silu(v):
            # scaled_silu: x = bf16(d / scale); bf16(x * sigmoid(x)).
            out = []
            for e in range_constexpr(8):
                x = (v[e].to(fx.Float32) / silu_scale).to(fx.BFloat16).to(fx.Float32)
                sig = fx.Float32(1.0) / (fx.Float32(1.0) + fx.exp(fx.Float32(0.0) - x))
                out.append((x * sig).to(fx.BFloat16))
            return fx.Vector.from_elements(out, fx.BFloat16)

        def _frag(sd, slot_off, ks, rep):
            vec8_ty = T.vec(8, elem_ty)
            halves = []
            for k_half in range_constexpr(2):
                off = sd[BASES][rep] + slot_off + (ks * WMMA_K + k_half * 16)
                v = fx.Vector(
                    fx.ptr_load(fx.add_offset(sd[MEM], off), result_type=vec8_ty)
                )
                halves.append(v)
            return halves[0].shuffle(halves[1], list(range(16)))

        def load_frags(sd, buf_idx):
            off = _slot_off(buf_idx, sd[ELEMS])
            return [
                _frag(sd, off, ks, rep)
                for ks in range_constexpr(k_wmma_steps)
                for rep in range_constexpr(sd[REPS])
            ]

        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, _fx_elem, fx.Float32)
        )

        def _rmem_vec(v, n, dtype):
            t = fx.make_rmem_tensor(n, dtype)
            t.store(v if isinstance(v, fx.Vector) else fx.Vector(v))
            return t

        def wmma_tile(accs_in, a_frags, b_frags):
            a_ts = [_rmem_vec(f, 16, _fx_elem) for f in a_frags]
            b_ts = [_rmem_vec(f, 16, _fx_elem) for f in b_frags]
            acc_ts = [_rmem_vec(a, 8, fx.Float32) for a in accs_in]
            for ks in range_constexpr(k_wmma_steps):
                for wm in range_constexpr(wmma_m_rep):
                    a_f = a_ts[ks * wmma_m_rep + wm]
                    for wn in range_constexpr(wmma_n_rep):
                        idx = wm * wmma_n_rep + wn
                        fx.gemm(
                            wmma_atom,
                            acc_ts[idx],
                            b_ts[ks * wmma_n_rep + wn],
                            a_f,
                            acc_ts[idx],
                        )
            return [t.load() for t in acc_ts]

        ld_normed = fx.Uint32(i32_ld_normed)
        ld_mixed = fx.Uint32(i32_ld_mixed)
        # Rows past M fall outside these records: their loads return 0 and their
        # stores are dropped, so the epilogue needs no row guard.
        normed_t = ptr_buf_tensor(
            arg_normed,
            fx.BFloat16,
            num_records_bytes=fx.Uint32(i32_m) * ld_normed * elem_bytes,
        )
        mixed_t = ptr_buf_tensor(
            arg_mixed,
            fx.BFloat16,
            num_records_bytes=fx.Uint32(i32_m) * ld_mixed * elem_bytes,
        )
        one = fx.Float32(1.0)
        inv_hc = fx.Float32(1.0 / HC)

        def epilogue_mix(final_accs):
            for wm in range_constexpr(wmma_m_rep):
                row32 = fx.Uint32(blk_m + warp_m_base + wm * WMMA_M + lane16)
                for wn in range_constexpr(wmma_n_rep):
                    col_base = blk_n + warp_n_base + wn * WMMA_N + lane_kgrp * 8
                    h0 = fx.Uint32(col_base) // HC
                    acc = fx.Vector(final_accs[wm * wmma_n_rep + wn])
                    for j in range_constexpr(_LANE_COLS // HC):
                        h = h0 + j
                        total = fx.Float32(0.0)
                        for s in range_constexpr(HC):
                            u = acc[j * HC + s].to(fx.BFloat16).to(fx.Float32)
                            x = normed_t[row32 * ld_normed + s * H + h].to(fx.Float32)
                            total = total + x / (one + fx.exp(fx.Float32(0.0) - u))
                        mixed_t[row32 * ld_mixed + h] = (total * inv_hc).to(fx.BFloat16)

        acc_zero = fx.full(8, 0.0, fx.Float32)
        accs = [acc_zero] * N_ACCS

        for i in range_constexpr(num_buffers - 1):
            issue_tdm_loads(i, i)
        tdm_ops.tensor_wait(0 if silu_scale else (num_buffers - 2) * _TDMS_PER_TILE)
        gpu.barrier()
        if const_expr(silu_scale > 0):
            vec8_ty = T.vec(8, elem_ty)
            vecs_per_slot = tile_m * tile_k // 8
            total_vecs = num_k_tiles * vecs_per_slot
            for j in range_constexpr((total_vecs + block_threads - 1) // block_threads):
                v_idx = fx.Uint32(tx) + j * block_threads
                slot = v_idx // vecs_per_slot
                within = v_idx - slot * vecs_per_slot
                r = within // (tile_k // 8)
                c8 = (within - r * (tile_k // 8)) * 8
                off = fx.Uint64(slot * lds_a_elems + r * lds_a_stride + c8)
                # Lanes past the last vector re-transform nothing: clamp them
                # onto a padding element row they own alone.
                ok = v_idx < total_vecs
                off = ok.select(off, fx.Uint64(num_buffers * lds_a_elems - 8))
                ptr = fx.add_offset(big_a_mem, off)
                v = fx.Vector(fx.ptr_load(ptr, result_type=vec8_ty))
                fx.ptr_store(_silu(v), ptr)
            gpu.barrier()
        cur_a, cur_b = load_frags(A_SIDE, 0), load_frags(B_SIDE, 0)

        # K is short (R = 320), so the whole K loop is unrolled at trace time.
        for t in range_constexpr(num_k_tiles):
            nxt = t + 1
            if const_expr(nxt + num_buffers - 2 < num_k_tiles):
                issue_tdm_loads(
                    (nxt + num_buffers - 2) % num_buffers, nxt + num_buffers - 2
                )
            if const_expr(nxt < num_k_tiles):
                outstanding = min(num_buffers - 2, num_k_tiles - 1 - nxt)
                tdm_ops.tensor_wait(max(0, outstanding) * _TDMS_PER_TILE)
                gpu.barrier()
                na = load_frags(A_SIDE, nxt % num_buffers)
                nb = load_frags(B_SIDE, nxt % num_buffers)
                accs = wmma_tile(accs, cur_a, cur_b)
                cur_a, cur_b = na, nb
            else:
                accs = wmma_tile(accs, cur_a, cur_b)

        epilogue_mix(accs)

    @flyc.jit
    def launch_hc_up_mix(
        arg_mixed: fx.Pointer,
        arg_gate: fx.Pointer,
        arg_w: fx.Pointer,
        arg_normed: fx.Pointer,
        i32_m: fx.Int32,
        i32_ld_mixed: fx.Int32,
        i32_lda: fx.Int32,
        i32_ld_normed: fx.Int32,
        stream: fx.Stream,
    ):
        gx = (fx.Uint64(i32_m) + (tile_m - 1)) // tile_m
        gy = N // tile_n
        wpe = int(waves_per_eu) if waves_per_eu is not None else 0
        launcher = hc_up_mix_kernel(
            arg_mixed,
            arg_gate,
            arg_w,
            arg_normed,
            i32_m,
            i32_ld_mixed,
            i32_lda,
            i32_ld_normed,
            value_attrs={
                "rocdl.flat_work_group_size": f"{block_threads},{block_threads}",
                "rocdl.waves_per_eu": wpe if wpe >= 1 else None,
            },
        )
        launcher.launch(grid=(gx, gy, 1), block=(block_threads, 1, 1), stream=stream)

    launch_hc_up_mix.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": True,
        "unroll-threshold": 0,
        "amdgpu-kernarg-preload-count": KERNARG_PRELOAD_COUNT,
    }
    return launch_hc_up_mix


def hc_interleave_up_weight(w_up, hc: int):
    """``[HC*H, R]`` up weight -> stream-minor rows (row ``h*HC + s``)."""
    n, r = w_up.shape
    return w_up.view(hc, n // hc, r).transpose(0, 1).reshape(n, r).contiguous()


__all__ = ["compile_hc_up_mix", "hc_interleave_up_weight"]
