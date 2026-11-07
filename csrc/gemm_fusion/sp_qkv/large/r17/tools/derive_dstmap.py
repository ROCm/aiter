#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Derive flydsl_dstmap_gemm.py (R18 r13 / large/r17) from aiter/ops/flydsl/kernels/preshuffle_gemm.py.

usage: derive_dstmap.py <aiter checkout root> <output file>
Every edit asserts its anchor count so an upstream change fails loudly instead of silently producing a different kernel.
"""

import hashlib
import os
import sys

root, out = sys.argv[1], sys.argv[2]
src_path = os.path.join(root, "aiter", "ops", "flydsl", "kernels", "preshuffle_gemm.py")
with open(src_path, encoding="utf-8", newline="") as _f:
    t = _f.read().replace("\r\n", "\n")
sha = hashlib.sha256(t.encode("utf-8")).hexdigest()


def rep(old, new, cnt=1):
    global t
    assert t.count(old) == cnt, f"anchor count {t.count(old)} != {cnt}: {old[:90]!r}"
    t = t.replace(old, new)


# ---- 1. imports (the vendored copy lives outside the aiter package) ----
rep(
    "from .mfma_preshuffle_pipeline import xcd_remap_bx_by",
    "from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import xcd_remap_bx_by",
)
rep(
    "from .splitk_epilogue import CPOL_COHERENT, splitk_reduce_epilogue",
    "from aiter.ops.flydsl.kernels.splitk_epilogue import CPOL_COHERENT, splitk_reduce_epilogue",
)

# ---- 2. new compile parameter dst_map ----
rep(
    '    split_k: int = 1,\n):\n    """Compile preshuffle GEMM (fp8/int8/fp16/bf16).',
    '    split_k: int = 1,\n    dst_map: str = "none",  # R18 r13: "none" | "dense" (F=2304, N=9216) | "indexer" (F=2560, N=9856, IK replicated)\n    sys_release: bool = True,  # R18 r13 BG-1: dst_map S stores are sc0|sc1 write-through + vmcnt(0) at the end (system-scope release before the SDMA put reads S); False = plain stores (no release)\n):\n    """Compile preshuffle GEMM (fp8/int8/fp16/bf16).',
)
rep(
    '    if lds_stage not in (1, 2):\n        raise ValueError(f"lds_stage must be 1 or 2, got {lds_stage}")\n',
    '    if lds_stage not in (1, 2):\n        raise ValueError(f"lds_stage must be 1 or 2, got {lds_stage}")\n'
    '    if dst_map not in ("none", "dense", "indexer"):\n        raise ValueError(f"dst_map must be none/dense/indexer, got {dst_map!r}")\n'
    '    if dst_map != "none" and not (in_dtype == "fp8" and out_dtype == "bf16" and split_k == 1 and epilogue == "none" and N == (9216 if dst_map == "dense" else 9856)):\n'
    '        raise ValueError("dst_map needs fp8 in / bf16 out, split_k 1, no epilogue and N = 9216 (dense) / 9856 (indexer)")\n'
    '    dst_f = 2304 if dst_map == "dense" else 2560  # slab columns per destination (Q 2048 + K 128 + V 128 [+ IQ 128 + IK 128])\n',
)

# ---- module-level destination map helpers (single exit; pure functions of the Int32 raw column base nb, a multiple of 16) ----
mod_helpers = '''

def _dst_pack_dense(nb):
    """raw column base -> (destination, packed column base); dense: Q 0..8191, K 8192..8703, V 8704..9215."""
    in_q = nb < fx.Int32(8192)
    in_k = nb < fx.Int32(8704)
    k_r = nb - fx.Int32(8192)
    v_r = nb - fx.Int32(8704)
    dst_v = in_q.select(nb // fx.Int32(2048), in_k.select(k_r // fx.Int32(128), v_r // fx.Int32(128)))
    pk_v = in_q.select(nb % fx.Int32(2048), in_k.select(fx.Int32(2048) + k_r % fx.Int32(128), fx.Int32(2176) + v_r % fx.Int32(128)))
    return dst_v, pk_v


def _dst_pack_indexer(nb):
    """indexer: + IQ 9216..9727 (per destination) and IK 9728..9855 (destination 0 here; replicated to 1..3 by the caller)."""
    in_q = nb < fx.Int32(8192)
    in_k = nb < fx.Int32(8704)
    in_v = nb < fx.Int32(9216)
    in_iq = nb < fx.Int32(9728)
    k_r = nb - fx.Int32(8192)
    v_r = nb - fx.Int32(8704)
    iq_r = nb - fx.Int32(9216)
    dst_v = in_q.select(nb // fx.Int32(2048), in_k.select(k_r // fx.Int32(128), in_v.select(v_r // fx.Int32(128), in_iq.select(iq_r // fx.Int32(128), fx.Int32(0)))))
    pk_v = in_q.select(
        nb % fx.Int32(2048),
        in_k.select(
            fx.Int32(2048) + k_r % fx.Int32(128),
            in_v.select(fx.Int32(2176) + v_r % fx.Int32(128), in_iq.select(fx.Int32(2304) + iq_r % fx.Int32(128), fx.Int32(2432) + (nb - fx.Int32(9728)))),
        ),
    )
    return dst_v, pk_v

'''
rep(
    "@functools.lru_cache(maxsize=1024)\ndef compile_preshuffle_gemm(",
    mod_helpers + "\n@functools.lru_cache(maxsize=1024)\ndef compile_preshuffle_gemm(",
)
rep(
    '    dst_f = 2304 if dst_map == "dense" else 2560',
    '    _dst_pack = _dst_pack_dense if dst_map == "dense" else _dst_pack_indexer\n    dst_f = 2304 if dst_map == "dense" else 2560\n    dst_cpol = CPOL_COHERENT if (dst_map != "none" and sys_release) else 0  # BG-1: write-through (sc0|sc1) S stores; the SDMA engine does not snoop the XCD L2s',
)
# ---- 3. ptpc order: bf16_rne(fp32(fp32(acc * Ws) * As)) (the r13 / PR695 ptpc_cast order) when dst_map is on ----
rep(
    "                    val_s = (val * s_a_vals[mi][ii]) * s_b_vals[ni]\n",
    '                    if const_expr(dst_map != "none"):\n'
    "                        val_s = (val * s_b_vals[ni]) * s_a_vals[mi][ii]  # r13 ptpc_cast order: (acc * Ws) * As\n"
    "                    else:\n"
    "                        val_s = (val * s_a_vals[mi][ii]) * s_b_vals[ni]\n",
)

# ---- 4. destination-major store ----
helper_and_store = """        if const_expr(dst_map != "none"):
            # R18 r13: S[dst][row][F] bf16 (slab stride = M rows of this launch), raw column n -> (dst, packed column) by the golden select_qkv_from_peers map:
            #   Q  n in [0, 8192)       : dst = n // 2048,          packed = n % 2048
            #   K  n in [8192, 8704)    : dst = (n - 8192) // 128,  packed = 2048 + (n - 8192) % 128
            #   V  n in [8704, 9216)    : dst = (n - 8704) // 128,  packed = 2176 + (n - 8704) % 128
            #   IQ n in [9216, 9728)    : dst = (n - 9216) // 128,  packed = 2304 + (n - 9216) % 128      (indexer only)
            #   IK n in [9728, 9856)    : REPLICATED to all 4 dst,  packed = 2432 + (n - 9728)             (indexer only)
            # A 16-column MFMA block never straddles a 128-column segment, so (dst, packed base) is computed once per (ni) block (_dst_pack_dense / _dst_pack_indexer, module level, single exit).
            s_tiles = _epi_tiles(
                arg_out,
                1,
                max_size=False,
                num_records_bytes=fx.Int64(i32_m) * fx.Int64(4 * dst_f * out_elem_bytes),
            )
            scalar_out = fx.make_rmem_tensor(fx.make_layout(1, 1), out_elem_cls)
            scalar_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(dst_cpol), out_elem_cls)
            final_values = Vec(frag_C_out.load())
            oob = fx.Int32(i32_m * (4 * dst_f))
            for ni in range_constexpr(num_acc_n):
                nb = by_n + (ni * num_waves + wave_id) * 16
                dst, pk = _dst_pack(nb)
                col = nb + lane_mod_16
                for mi in range_constexpr(m_repeat):
                    for ii in range_constexpr(4):
                        p = ni * (m_repeat * 4) + mi * 4 + ii
                        row = bx_m + mi * 16 + lane_div_16 * 4 + ii
                        in_bounds = (row < i32_m) & (col < i32_n)
                        s_index = fx.Int32((dst * i32_m + row) * dst_f + pk + lane_mod_16)
                        safe_index = in_bounds.select(s_index, oob)
                        fx.memref_store_vec(Vec.filled(1, final_values[p], out_elem_cls), scalar_out)
                        fx.copy(scalar_atom, scalar_out, fx.slice(s_tiles, (None, safe_index)))
                if const_expr(dst_map == "indexer"):
                    if nb >= fx.Int32(9728):  # IK block: replicate to destinations 1..3 (destination 0 was stored above)
                        for dd3 in range_constexpr(3):
                            for mi in range_constexpr(m_repeat):
                                for ii in range_constexpr(4):
                                    p = ni * (m_repeat * 4) + mi * 4 + ii
                                    row = bx_m + mi * 16 + lane_div_16 * 4 + ii
                                    in_bounds = (row < i32_m) & (col < i32_n)
                                    s_index = fx.Int32((fx.Int32(dd3 + 1) * i32_m + row) * dst_f + pk + lane_mod_16)
                                    safe_index = in_bounds.select(s_index, oob)
                                    fx.memref_store_vec(Vec.filled(1, final_values[p], out_elem_cls), scalar_out)
                                    fx.copy(scalar_atom, scalar_out, fx.slice(s_tiles, (None, safe_index)))
            if const_expr(sys_release):
                # BG-1: per-wave system-scope release = the write-through stores above completed (vmcnt 0), as in the split-K epilogue (no agent fence / L2 flush needed)
                rocdl.s_waitcnt(0)
        elif const_expr(n_ragged):
"""
rep(
    "        if const_expr(n_ragged):\n            # pC_g uses a padded-N tile layout",
    helper_and_store + "            # pC_g uses a padded-N tile layout",
)

header = (
    "# SPDX-License-Identifier: MIT\n"
    "# R18 r13 / large/r17: FlyDSL preshuffle GEMM with a DESTINATION-MAJOR epilogue (dst_map = dense | indexer). GENERATED by tools/derive_dstmap.py from\n"
    f"# aiter/ops/flydsl/kernels/preshuffle_gemm.py (sha256 {sha}); do not edit by hand. UNCOMPILED / UNTESTED source (no GPU, no FlyDSL here).\n"
)
with open(out, "w", encoding="utf-8", newline="") as _f:
    _f.write(header + t)
print("derived from", sha)
