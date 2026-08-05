# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GDN K5 inter-chunk state scan — gfx942 (CDNA3 / MI300X) FlyDSL kernel.

For each chunk t (serial over NT chunks):
  1. Store h snapshot for downstream K6
  2. v_new = u - w @ h   (delta correction via MFMA)
  3. Gated decay + state update:
       v_new *= exp(g_last - g_cumsum)
       h = h * exp(g_last) + k^T @ v_new
"""

from __future__ import annotations


def compile_chunk_gated_delta_h_gfx942(
    *,
    K: int,
    V: int,
    BT: int = 64,
    BV: int = 32,
    H: int,
    Hg: int,
    USE_G: bool = True,
    USE_GK: bool = False,
    USE_INITIAL_STATE: bool = True,
    STORE_FINAL_STATE: bool = True,
    SAVE_NEW_VALUE: bool = True,
    IS_VARLEN: bool = True,
    WU_CONTIGUOUS: bool = True,
    STATE_DTYPE_BF16: bool = False,
    G_IS_LOG2_SCALED: bool = False,
):
    """Build the gfx942 FlyDSL launcher for one compile-time configuration.

    Signature matches ``compile_chunk_gated_delta_h`` so
    ``_get_or_compile`` in ``linear_attention_prefill_kernels`` can call either
    without modification.

    Returns a @flyc.jit function:
            launch_fn(k, v, w, v_new, g, gk, h, h0, ht,
                      cu_seqlens, chunk_offsets,
                      T_val, T_flat, N_val, stream)

    """
    raise NotImplementedError(
        "FlyDSL GDN K5 on gfx942 is not yet implemented. "
        "The blockers are: (1) v_mfma_f32_16x16x32_bf16 → v_mfma_f32_16x16x16bf16_1k, "
        "(2) ds_read_b64_tr_b16 → transposed LDS store, "
        "(3) BV ≤ 32 LDS cap. "
        "Use use_chunk_hip=True or the Triton default path on gfx942."
    )
