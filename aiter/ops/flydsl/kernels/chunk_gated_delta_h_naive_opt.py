# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Gated Delta Net K5 hidden-state recurrence kernel -- NAIVE-OPT fork.

This is the un-pipelined NAIVE baseline fork (see
``chunk_gated_delta_h_naive.py``) with the OPT-DGL w-load ENABLED by
default. Instead of staging w through VGPRs (``buffer_load`` ->
``ds_write_b128``), w is streamed directly HBM->LDS with
``buffer_load_lds`` (gfx950 direct-global-to-LDS), a full K row per DMA,
and a TWO-SIDED XOR swizzle (swizzle the DMA source column + re-apply the
same swizzle on the GEMM1 LDS read) so the contiguous DGL layout still
scatters across LDS banks. This removes the w ds_write and its VGPR
staging while keeping IDENTICAL numerics.

It produces the SAME public VK h-layout ``(B, NT, H, V, K)`` at ANY BV as
the naive/baseline forks. All other scheduling (no prefetch / no software
pipeline) is unchanged from the naive fork -- this fork isolates the DGL
w-load win.

Measured ~-18% kernel cycles vs the naive fork on varlen-32k-aws
SeqLen1000 (gfx950).
"""

from .chunk_gated_delta_h_naive import compile_chunk_gated_delta_h_naive


def compile_chunk_gated_delta_h_naive_opt(**kwargs):
    """Compile the NAIVE-OPT (DGL w-load) GDN K5 baseline-fork kernel.

    Thin wrapper over :func:`compile_chunk_gated_delta_h_naive` that forces
    the OPT-DGL w-load path on (``W_DGL=True``), regardless of the
    ``FLYDSL_K5_W_DGL`` env flag. Same signature / outputs otherwise.
    """
    kwargs.pop("W_DGL", None)
    return compile_chunk_gated_delta_h_naive(W_DGL=True, **kwargs)


__all__ = [
    "compile_chunk_gated_delta_h_naive_opt",
]
