# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc.

"""MFMA ownership views for the paged K stripes and shuffled V vectors."""

import flydsl.expr as fx


def wide_fp8_mma(shuffled=False):
    atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(32, 32, 64, fx.Float8E4M3FN))
    permutation = None
    if shuffled:
        # The cache stores intact 16-token vectors; exchange the two outer K modes.
        permutation = fx.make_tile(None, None, fx.make_layout((16, 2, 2), (1, 32, 16)))
    return fx.make_tiled_mma(atom, fx.make_layout((1, 1, 1), (0, 0, 0)), permutation)


def shuffled_v_copy(lane):
    atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float8E4M3FN)
    return atom, fx.make_tiled_copy_A(atom, wide_fp8_mma(shuffled=True)).get_slice(
        fx.Int32(lane)
    )


def shuffled_v_layout(head_dim):
    return fx.make_layout((head_dim, (16, 4)), (16, (1, head_dim * 16)))


def shuffled_v_tokens():
    mma = wide_fp8_mma(shuffled=True)
    # Broadcast the D coordinate to derive only the token start of each dword.
    tokens = fx.make_layout((32, 64), (0, 1))
    return fx.composition(tokens, mma.tv_layout_A_tiled)


def k_stripe_layout(head_dim, vec_kv):
    mma = wide_fp8_mma()
    rows = fx.select(mma.tile_size_mnk, [0]).to_py_value() * 2
    reduction = head_dim
    stripes = reduction // vec_kv
    return fx.make_layout(
        ((reduction // vec_kv, rows // stripes), stripes),
        ((rows * vec_kv, stripes), 1),
    )


def probability_word_layout():
    # P starts in QK's C ownership, with the row/column roles exchanged for PV.
    mma = wide_fp8_mma()
    c_tokens = fx.composition(fx.make_layout((32, 32), (1, 0)), mma.tv_layout_C_tiled)
    # Four adjacent fp8 values form one word after the existing packed conversion.
    words = fx.coalesce(fx.slice(fx.recast_layout(c_tokens, 8, 32), (0, None)))
    tile_words = fx.select(mma.tile_size_mnk, [0]).to_py_value() // 4
    return fx.make_layout(
        (fx.size(words).to_py_value(), 2),
        (fx.get_stride(words).to_py_value(), tile_words),
    )


def probability_swap_pairs():
    ownership = probability_word_layout()
    compact = fx.make_layout((4, 2), (1, 4))
    tokens = shuffled_v_tokens()
    pairs = []
    for half in range(2):
        for pair in range(2):
            indices = []
            for part in range(2):
                token = fx.crd2idx((0, half * 16 + pair * 8), tokens).to_py_value()
                token += part * 16
                coord = fx.idx2crd(token // 4, ownership)
                indices.append(fx.crd2idx(coord, compact).to_py_value())
            pairs.append(tuple(indices))
    return pairs
