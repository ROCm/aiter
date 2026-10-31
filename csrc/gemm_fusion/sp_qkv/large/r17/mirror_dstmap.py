#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pure-python (no torch, no FlyDSL) reference of the R18 r13 destination map and of the kernel's select-chain index arithmetic.

The golden map (large/r13 device/layout/quad_fp8_fieldmap.hpp, select_qkv_from_peers) per destination slab d (packed columns, in order):
    Q  raw [d*2048, +2048)  -> packed [0, 2048)
    K  raw [8192 + d*128, +128) -> packed [2048, 2176)
    V  raw [8704 + d*128, +128) -> packed [2176, 2304)
    (indexer) IQ raw [9216 + d*128, +128) -> packed [2304, 2432);  IK raw [9728, +128) REPLICATED to every d -> packed [2432, 2560)
S layout: S[dst][row][F] bf16, slab stride = M rows of the launch, F = 2304 (dense) / 2560 (indexer).
"""


def f_of(indexer):
    return 2560 if indexer else 2304


def n_of(indexer):
    return 9856 if indexer else 9216


def raw_to_dst_pk(n, indexer):
    """Independent (readable) map: returns [(dst, packed_col)] (one entry, or four for the replicated IK columns)."""
    if n < 8192:
        return [(n // 2048, n % 2048)]
    if n < 8704:
        return [((n - 8192) // 128, 2048 + (n - 8192) % 128)]
    if n < 9216:
        return [((n - 8704) // 128, 2176 + (n - 8704) % 128)]
    if not indexer:
        raise ValueError("raw column beyond dense N")
    if n < 9728:
        return [((n - 9216) // 128, 2304 + (n - 9216) % 128)]
    if n < 9856:
        return [(d, 2432 + (n - 9728)) for d in range(4)]
    raise ValueError("raw column beyond N")


def kernel_dst_pk(nb, indexer):
    """Mirror of _dst_pack_dense / _dst_pack_indexer (nested select chains, floor division / modulo on possibly negative unselected branches)."""
    k_r, v_r, iq_r = nb - 8192, nb - 8704, nb - 9216
    in_q, in_k, in_v, in_iq = nb < 8192, nb < 8704, nb < 9216, nb < 9728
    if not indexer:
        dst = nb // 2048 if in_q else (k_r // 128 if in_k else v_r // 128)
        pk = nb % 2048 if in_q else (2048 + k_r % 128 if in_k else 2176 + v_r % 128)
        return dst, pk
    dst = (
        nb // 2048
        if in_q
        else (
            k_r // 128
            if in_k
            else (v_r // 128 if in_v else (iq_r // 128 if in_iq else 0))
        )
    )
    pk = (
        nb % 2048
        if in_q
        else (
            2048 + k_r % 128
            if in_k
            else (
                2176 + v_r % 128
                if in_v
                else (2304 + iq_r % 128 if in_iq else 2432 + (nb - 9728))
            )
        )
    )
    return dst, pk


def s_offset(dst, row, pk, m_rows, indexer):
    return (dst * m_rows + row) * f_of(indexer) + pk


def column_tables(indexer):
    """(dst list, pk list, raw n list) over every raw column, IK replicated entries included (used by the microbench reference)."""
    out = []
    for n in range(n_of(indexer)):
        for d, pk in raw_to_dst_pk(n, indexer):
            out.append((n, d, pk))
    return out
