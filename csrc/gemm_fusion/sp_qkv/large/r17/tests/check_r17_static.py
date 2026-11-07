#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""large/r17 static checks (no GPU, no FlyDSL, no torch): the destination map is a bijection onto S (IK replicated x4), the kernel's select-chain arithmetic equals the
independent map for every 16-column block, the generated kernel file carries the expected anchors, and it re-derives byte-identically when the aiter source is present.
"""

import ast
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
import mirror_dstmap as mm

fails = []


def need(c, m):
    print(("PASS " if c else "FAIL ") + m)
    if not c:
        fails.append(m)


for indexer in (False, True):
    F, N = mm.f_of(indexer), mm.n_of(indexer)
    cover = {}
    for n, d, pk in mm.column_tables(indexer):
        cover[(d, pk)] = cover.get((d, pk), 0) + 1
    need(
        len(cover) == 4 * F and all(v == 1 for v in cover.values()),
        f"{'indexer' if indexer else 'dense'}: every (dst, packed) of S[4][*][{F}] is written exactly once ({4 * F} cells; raw N = {N})",
    )
    nrep = sum(1 for n in range(N) if len(mm.raw_to_dst_pk(n, indexer)) == 4)
    need(
        nrep == (128 if indexer else 0),
        f"{'indexer' if indexer else 'dense'}: replicated raw columns = {nrep} (IK 128 for the indexer, none dense)",
    )
    ok = True
    for nb in range(0, N, 16):
        ref = mm.raw_to_dst_pk(nb, indexer)
        d, pk = mm.kernel_dst_pk(nb, indexer)
        if (d, pk) != ref[0] or any(
            mm.raw_to_dst_pk(nb + j, indexer)[0] != (d, pk + j) for j in range(16)
        ):
            ok = False
    need(
        ok,
        f"{'indexer' if indexer else 'dense'}: kernel select chains == independent map for all {N // 16} 16-column blocks (and the 16 lanes are contiguous in packed order)",
    )
# golden spot checks (large/r13 quad_fp8_fieldmap.hpp: ht_raw(0)=0, ht_raw(16)=8192, ht_raw(17)=8704, ht_raw(34)=8320, ht_raw(35)=8832, ht_raw(54)=6144, ht_raw(70)=8576)
need(
    mm.raw_to_dst_pk(0, False) == [(0, 0)]
    and mm.raw_to_dst_pk(8192, False) == [(0, 2048)]
    and mm.raw_to_dst_pk(8704, False) == [(0, 2176)]
    and mm.raw_to_dst_pk(8320, False) == [(1, 2048)]
    and mm.raw_to_dst_pk(6144, False) == [(3, 0)]
    and mm.raw_to_dst_pk(8576, False) == [(3, 2048)],
    "golden spot checks of the r13 field map",
)
need(
    mm.raw_to_dst_pk(9216 + 128, True) == [(1, 2304)]
    and mm.raw_to_dst_pk(9728 + 5, True) == [(d, 2437) for d in range(4)],
    "indexer extension: IQ per destination at packed 2304.., IK replicated at 2432..",
)

with open(os.path.join(ROOT, "flydsl_dstmap_gemm.py"), encoding="utf-8") as _f:
    src = _f.read()
ast.parse(src)
need(
    "def _dst_pack_dense(nb):" in src
    and "def _dst_pack_indexer(nb):" in src
    and 'dst_map: str = "none"' in src,
    "kernel module parses; dst_map option and the module-level pack helpers are present",
)
need(
    "val_s = (val * s_b_vals[ni]) * s_a_vals[mi][ii]" in src,
    "ptpc order (acc * Ws) * As (r13 ptpc_cast) in the dst_map epilogue",
)
need(
    "s_index = fx.Int32((dst * i32_m + row) * dst_f + pk + lane_mod_16)" in src
    and "safe_index = in_bounds.select(s_index, oob)" in src
    and "num_records_bytes=fx.Int64(i32_m) * fx.Int64(4 * dst_f * out_elem_bytes)"
    in src,
    "S index = ((dst*M + row)*F + packed + lane) with OOB rows/columns redirected to the one-past-end element",
)
need(
    "for dd3 in range_constexpr(3):" in src and "if nb >= fx.Int32(9728):" in src,
    "IK replication: destinations 1..3 stored under a wave-uniform runtime condition",
)
need(
    "from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import xcd_remap_bx_by"
    in src
    and "from .mfma" not in src,
    "imports are absolute (the vendored module lives outside the aiter package)",
)
aiter_root = os.path.abspath(os.path.join(ROOT, "..", "..", "..", "..", ".."))
src_path = os.path.join(
    aiter_root, "aiter", "ops", "flydsl", "kernels", "preshuffle_gemm.py"
)
if os.path.exists(src_path):
    out = os.path.join(HERE, "_rederived.py")
    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "tools", "derive_dstmap.py"),
            aiter_root,
            out,
        ],
        check=True,
        capture_output=True,
    )
    with open(out, encoding="utf-8") as _f:
        same = _f.read() == src
    os.remove(out)
    need(
        same,
        "re-deriving from the current aiter preshuffle_gemm.py reproduces flydsl_dstmap_gemm.py byte-identically",
    )
else:
    print("SKIP re-derivation (aiter checkout not found)")
need(
    "sys_release: bool = True" in src
    and 'dst_cpol = CPOL_COHERENT if (dst_map != "none" and sys_release) else 0' in src
    and "fx.rocdl.BufferCopy16b(dst_cpol)" in src
    and "if const_expr(sys_release):" in src
    and src.count("rocdl.s_waitcnt(0)") >= 2,
    "BG-1: sys_release (default True): the dst_map S stores are sc0|sc1 write-through (CPOL_COHERENT) and each wave ends with vmcnt(0) (system-scope release for the SDMA put)",
)
print("ALL PASS" if not fails else f"FAILED: {len(fails):d}")
sys.exit(1 if fails else 0)
