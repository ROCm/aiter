# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# Do NOT add `from __future__ import annotations`: PEP 563 stringifies the
# annotations and defeats flydsl's runtime-arg detection in the JIT cache key.

"""Fused up-projection + gated-mean for the Gated-Residual mix.

This is the memory-heavy tail of ``combine_and_mix``. The shipped path runs the
up-GEMM and the gated mean as two launches that round-trip the full
``hc_count*stream_dim`` gate tensor through HBM. Here the gate is produced by an
MFMA up-GEMM and consumed immediately in registers, so it never touches memory:

    gate[m, s*stream_dim + c] = lora[m, :] . w_up[s*stream_dim + c, :]
    block_input[m, c]         = mean_s sigmoid(gate[...]) * xn[m, s*stream_dim + c]

One workgroup owns a ``BLOCK_M x BLOCK_N`` tile of ``block_input`` (a token block
and a stream-channel block ``c``). For each of the ``hc_count`` streams it runs a
skinny ``[BLOCK_M, lowrank] x [lowrank, BLOCK_N]`` matrix-core GEMM, applies the
sigmoid gate against the matching ``xn`` slice, and accumulates the stream mean.
The MFMA shape is parameterized (``mma_m/n/k``) so the same body targets gfx950
and gfx942.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.act import _sigmoid_f32
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, _run_compiled
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.common import ab_k_perm, arch_name, mfma_bf16
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.tuned import up_gate_mix_config


@lru_cache(maxsize=16)
def _build_up_gate_mix_norm(
    hc_count: int,
    stream_dim: int,
    lowrank: int,
    w_len: int,
    block_m: int,
    block_n: int,
    m_waves: int,
    n_waves: int,
    mma_m: int,
    mma_n: int,
    mma_k: int,
    _skip_norm: bool = False,
    _skip_gemm: bool = False,
):
    """K2 (SILOTIGER-1042 milestone 3): up-GEMM + gated mean that re-forms ``xn``
    on-the-fly from ``r2`` + ``rrms`` instead of reading a materialized ``xn``.

    Identical to :func:`_build_up_gate_mix` except the gated-mean multiply builds
    ``xn = r2 * rrms[m, s] * (1 + w[c])`` in registers (matching the reference's
    normalize-then-bf16-round order) right where the base kernel loaded ``xn``.
    This lets K2 consume K1's stored ``r2`` (+ the tiny per-stream ``rrms``)
    directly -- the ``xn`` HBM round-trip and the interim norm-rebuild launch of
    the milestone-2 tail both disappear.
    """
    hidden = hc_count * stream_dim
    inv_hc = 1.0 / hc_count
    block_threads = m_waves * n_waves * 64
    assert block_m % (m_waves * mma_m) == 0
    assert block_n % (n_waves * mma_n) == 0
    assert lowrank % mma_k == 0
    assert stream_dim % block_n == 0
    n_cblocks = stream_dim // block_n
    k_group = mma_k // 4
    VEC = 8
    assert block_n % VEC == 0
    assert (block_m * (block_n // VEC)) % block_threads == 0

    @flyc.kernel(
        name=f"gr_k2_up_gate_mix_norm_hc{hc_count}_hs{stream_dim}_r{lowrank}"
        f"_bm{block_m}_bn{block_n}",
        known_block_size=[block_threads, 1, 1],
    )
    def kernel(
        lora: fx.Tensor,  # [M, lowrank] bf16
        r2: fx.Tensor,  # [M, hidden] bf16 (un-normalized combined residual)
        rrms: fx.Tensor,  # [M, hc] f32
        w: fx.Tensor,  # [w_len] f32 (grouped-RMSNorm weight, widened)
        w_up: fx.Tensor,  # [hidden, lowrank] bf16
        block_input: fx.Tensor,  # [M, stream_dim] bf16
    ):
        tid = fx.thread_idx.x
        bid_m = fx.block_idx.x
        bid_c = fx.block_idx.y

        lora_buf = fx.rocdl.make_buffer_tensor(lora, max_size=True)
        wup_buf = fx.rocdl.make_buffer_tensor(w_up, max_size=True)
        out_buf = fx.rocdl.make_buffer_tensor(block_input, max_size=True)
        r2_g = GTensor(r2, T.bf16, (1, hidden))
        rrms_g = GTensor(rrms, fx.Float32, (1, hc_count))
        w_g = GTensor(w, fx.Float32, (1, w_len))

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(mma_m, mma_n, mma_k, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((m_waves, n_waves, 1), (n_waves, 1, 0)),
            fx.make_tile(None, None, ab_k_perm(mma_k)),
        )
        thr_mma = tiled_mma.thr_slice(tid)

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        thr_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(tid)

        out_g = GTensor(block_input, T.bf16, (1, stream_dim))

        @fx.struct
        class SharedStorage:
            gate: fx.Array[fx.Float32, block_m * block_n, 16]

        storage = fx.SharedAllocator().allocate(SharedStorage)
        smem = storage.gate.peek().ptr

        bA = fx.slice(fx.zipped_divide(lora_buf, (block_m, lowrank)), (None, bid_m))
        frag_A = thr_mma.make_fragment_A(bA)
        copy_frag_A = thr_copy_A.retile(frag_A)
        fx.copy(copy_atom, thr_copy_A.partition_S(bA), copy_frag_A)

        row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
        col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
        cRow = thr_mma.partition_C(row_coords)
        cCol = thr_mma.partition_C(col_coords)

        bB_all = fx.zipped_divide(wup_buf, (block_n, lowrank))
        gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_c]
        frag_gate = thr_mma.make_fragment_C(gC)
        lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))

        vecs_per_row = block_n // VEC
        iters = (block_m * vecs_per_row) // block_threads
        acc = [[fx.Float32(0.0)] * VEC for _ in range_constexpr(iters)]
        c_base = bid_c * block_n

        gate_n = fx.size(frag_gate.shape).unpack()
        sGate = fx.make_view(smem, lds_layout)
        for s in range_constexpr(hc_count):
            n_tile = s * n_cblocks + bid_c
            bB = fx.slice(bB_all, (None, n_tile))
            frag_B = thr_mma.make_fragment_B(bB)
            copy_frag_B = thr_copy_B.retile(frag_B)
            fx.copy(copy_atom, thr_copy_B.partition_S(bB), copy_frag_B)

            frag_gate.fill(0.0)
            if not _skip_gemm:
                fx.gemm(
                    tiled_mma, frag_gate, frag_A, frag_B, frag_gate,
                    traversal_order=fx.GemmTraversalOrder.KNM,
                )
            fx.gpu.barrier()
            for i in range_constexpr(gate_n):
                sGate[fx.get_scalar(cRow[i]), fx.get_scalar(cCol[i])] = frag_gate[i]
            fx.gpu.barrier()

            for it in range_constexpr(iters):
                v = it * block_threads + tid
                local_m = v // vecs_per_row
                c0 = (v % vecs_per_row) * VEC
                g_m = bid_m * block_m + local_m
                g_c = c_base + c0
                col = s * stream_dim + g_c
                gate_vec = fx.ptr_load(
                    smem + local_m * block_n + c0,
                    result_type=fx.Vector.make_type(VEC, fx.Float32),
                )
                # Re-form xn on the fly: widen r2, * rrms * (1+w), re-round to bf16
                # (reference order), then back to f32 for the gated-mean multiply.
                r2_vec = fx.Vector(
                    r2_g.load(g_m * hidden + col, vec_size=VEC)
                ).to(fx.Float32)
                # Measurement-only: skip the rrms*(1+w) reform + bf16 round to
                # isolate the K2 normalize tax (see §6.9's K1 _skip_norm).
                xn_vec = r2_vec
                if not _skip_norm:
                    rr = fx.Float32(rrms_g.load(g_m * hc_count + s, vec_size=1))
                    # f32 buffer loads are 128-bit max (v8f32 won't isel), so read
                    # the VEC-wide (1+w) slice in dwordx4 chunks.
                    wv = []
                    for c4 in range_constexpr(VEC // 4):
                        w4 = fx.Vector(
                            w_g.load((col + c4 * 4) % w_len, vec_size=4)
                        ).to(fx.Float32)
                        for e4 in range_constexpr(4):
                            wv.append(w4[e4])
                    xn_e = [
                        (r2_vec[e] * rr * (fx.Float32(1.0) + wv[e]))
                        for e in range_constexpr(VEC)
                    ]
                    xn_vec = fx.Vector.from_elements(xn_e, dtype=fx.Float32).to(
                        fx.BFloat16
                    ).to(fx.Float32)
                for e in range_constexpr(VEC):
                    acc[it][e] = acc[it][e] + _sigmoid_f32(gate_vec[e]) * xn_vec[e]

        for it in range_constexpr(iters):
            v = it * block_threads + tid
            local_m = v // vecs_per_row
            c0 = (v % vecs_per_row) * VEC
            g_m = bid_m * block_m + local_m
            g_c = c_base + c0
            out_vec = [acc[it][e] * fx.Float32(inv_hc) for e in range_constexpr(VEC)]
            out_g.store(
                g_m * stream_dim + g_c,
                fx.Vector.from_elements(out_vec, dtype=fx.Float32).to(fx.BFloat16),
                vec_size=VEC,
            )

    @flyc.jit
    def launch(
        lora: fx.Tensor,
        r2: fx.Tensor,
        rrms: fx.Tensor,
        w: fx.Tensor,
        w_up: fx.Tensor,
        block_input: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        tokens = fx.Int32(fx.get_scalar(lora.shape[0]))
        grid_m = (tokens + block_m - 1) // block_m
        kernel(lora, r2, rrms, w, w_up, block_input).launch(
            grid=(fx.Int64(grid_m), n_cblocks, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


def flydsl_up_gate_mix_norm(
    lora: torch.Tensor,
    r2: torch.Tensor,
    rrms: torch.Tensor,
    norm_weight: torch.Tensor,
    w_up: torch.Tensor,
    hc_count: int,
    block_input: torch.Tensor | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    m_waves: int | None = None,
    n_waves: int | None = None,
    stream: torch.cuda.Stream | None = None,
    _skip_norm: bool = False,
    _skip_gemm: bool = False,
) -> torch.Tensor:
    """K2: fused up-GEMM + gated mean that re-forms ``xn`` from ``r2`` + ``rrms``.

    Drop-in for :func:`flydsl_up_gate_mix` where the caller has ``r2`` (K1's
    stored combined residual) and ``rrms`` (K1's per-stream 1/rms) instead of a
    materialized ``xn`` -- removing the ``xn`` HBM round-trip and the norm-rebuild
    launch. Returns ``block_input`` [M, stream_dim] bf16.
    """
    assert lora.dtype == torch.bfloat16 and lora.is_contiguous()
    assert r2.dtype == torch.bfloat16 and r2.is_contiguous()
    assert rrms.dtype == torch.float32 and rrms.is_contiguous()
    tokens, lowrank = lora.shape
    hidden = r2.shape[1]
    stream_dim = hidden // hc_count
    assert w_up.shape == (hidden, lowrank)
    assert rrms.shape == (tokens, hc_count)
    w = norm_weight.reshape(-1).float().contiguous()
    w_len = w.numel()
    assert w_len in (stream_dim, hidden)

    arch = arch_name(lora.device)
    tuned = up_gate_mix_config(arch, tokens)
    block_m, block_n, m_waves, n_waves = _resolve_up_gate_cfg(
        tuned, tokens, block_m, block_n, m_waves, n_waves,
    )
    assert tokens % block_m == 0
    if block_input is None:
        block_input = torch.empty(tokens, stream_dim, dtype=torch.bfloat16, device=lora.device)
    if stream is None:
        stream = torch.cuda.current_stream()

    mma = mfma_bf16(arch)
    launch = _build_up_gate_mix_norm(
        hc_count, stream_dim, lowrank, w_len, block_m, block_n, m_waves, n_waves,
        mma.mma_m, mma.mma_n, mma.mma_k, _skip_norm, _skip_gemm,
    )
    _run_compiled(launch, lora, r2, rrms, w, w_up, block_input, fx.Stream(stream))
    return block_input


_UP_GATE_DEFAULTS = dict(block_m=64, block_n=64, m_waves=1, n_waves=2)


def _resolve_up_gate_cfg(tuned, tokens, block_m, block_n, m_waves, n_waves):
    """Merge explicit args over the tuned config over the hard-coded defaults.

    A tuned ``block_m`` that does not divide ``tokens`` is dropped back to the
    default so the tail assertion never fires on a table-selected block size.
    """
    base = dict(_UP_GATE_DEFAULTS)
    if tuned:
        base.update(tuned)
        if tokens % base["block_m"]:
            base = dict(_UP_GATE_DEFAULTS)
    override = dict(block_m=block_m, block_n=block_n, m_waves=m_waves, n_waves=n_waves)
    for k, v in override.items():
        if v is not None:
            base[k] = v
    return base["block_m"], base["block_n"], base["m_waves"], base["n_waves"]