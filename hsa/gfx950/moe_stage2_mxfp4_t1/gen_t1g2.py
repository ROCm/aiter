# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Offline generator of the gfx950 MXFP4 MoE stage-2 kernels in this directory.

They serve aiter/ops/asm/moe_stage2_mxfp4_t1.py: the routed-MoE down projection (stage 2) of one shape,
MiniMax-M3.1-Flash at TP2 (E 128, top-4, model_dim H 6144, inter_dim I 1536 per rank), with inputs exactly as
AITER's MXFP4 a4w4 stage 1 and weight preshuffle leave them:
  w2         [E, H, I/2] fp4x2, preshuffled 16x16: tile (n0, k0) = 16 rows x 128 K = 1 KB at e W2_E + (12 n0 + k0) KB;
             lane i = r + 16 g of a wave reads tile + 16 i = row r, K [32 g, 32 g + 32) of the K-step (MFMA A operand).
  w2_scale   e8m0, e8m0_shuffle per 32 rows: chunk (1536 B) [c8 6][a 4][b 16][c 2][d 2], row 16 d + b, col 8 c8 + 4 c + a;
             lane (r, g) reads the dword at chunk + 256 c8 + 4 lane; K-step kk takes byte 2 (kk % 2) + d.
  x          inter_states [max_sorted, I/2] fp4x2 per sorted row (B operand: lane (n, g) row n, 16 B at 64 kk + 16 g).
  xs         a2_scale, the same e8m0_shuffle over sorted rows.
  routing    sorted_expert_ids per block of BM rows (ascending, an expert's padding rows at the end of its range),
             num_valid_ids[0] = valid sorted rows (a multiple of BM), sorted_token_ids (padding = id & 0xffffff >= M).
MFMA: v_mfma_scale_f32_16x16x128_f8f6f4, cbsz:4 blgp:4 (fp4 x fp4), op_sel / op_sel_hi pick the scale byte.
The structure of each variant is in v2_source's docstring; the assembly structure (prologue, buffer resources,
MFMA / epilogue idioms, XCD-dealt work order) follows Teemu Virolainen's MoE down-projection generator.
Served variants (the launcher picks one by M = moe_out.shape[0]):
  v2n   v2 with nontemporal output stores: unweighted bf16 rows into flat [max_sorted, H], then
        aiter.mxfp4_moe_scatter_reduce;
  v4g8  dense groups of up to 8 blocks of 128 rows in LPT order, the W2 slice reused over the group.
Not built (SERVED below): v2ao, v2 with a bf16 atomic epilogue into moe_out, walking AITER's gemm2 tile order. It
measured no faster than AITER's own atomic gemm2 at the shapes it could serve (M 1024 to 2285), so no launcher picks it.

Usage (CPU only; clang and ld.lld from ROCm's LLVM: --llvm-bin, else the _rocm_sdk_devel wheel, $ROCM_PATH/llvm/bin
or /opt/rocm/llvm/bin):
    python3 hsa/gfx950/moe_stage2_mxfp4_t1/gen_t1g2.py            # rebuild the .co files, the csv, build_info.json
    python3 hsa/gfx950/moe_stage2_mxfp4_t1/gen_t1g2.py --check DIR    # rebuild in DIR, compare with build_info.json
    python3 hsa/gfx950/moe_stage2_mxfp4_t1/gen_t1g2.py --asm-dir DIR    # also write the generated assembly
The .co file name is the kernel symbol. build_info.json records the toolchain, the commands and each kernel's source
sha1 and code-object sha256.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


RSRC3 = 0x00020000  # buffer resource dword 3: gfx9 raw dword format


class KArgs:
    """Ordered kernargs: ("name", "ptr") 8 B or ("name", "u32") 4 B, natural alignment, declaration order."""

    def __init__(self, *args):
        self.args, self.off, o = list(args), {}, 0
        for name, kind in args:
            size = 8 if kind == "ptr" else 4
            o = (o + size - 1) // size * size
            self.off[name] = o
            o += size
        self.size = (o + 7) // 8 * 8

    def metadata(self):
        out = []
        for name, kind in self.args:
            if kind == "ptr":
                out.append(
                    f"      - {{ .name: {name}, .size: 8, .offset: {self.off[name]}, "
                    ".value_kind: global_buffer, .address_space: global }"
                )
            else:
                out.append(
                    f"      - {{ .name: {name}, .size: 4, .offset: {self.off[name]}, .value_kind: by_value }}"
                )
        return "\n".join(out)


def kernel_source(name, body, ka, *, lds, vgprs, agprs, sgprs, block):
    acc_off = (vgprs + 3) // 4 * 4
    total = acc_off + agprs
    assert vgprs <= 256 and agprs <= 256 and total <= 512, (vgprs, agprs)
    assert sgprs <= 102, sgprs
    return f""".amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl {name}
.p2align 8
.type {name},@function
{name}:
{chr(10).join(body)}
.Lfunc_end:
.size {name}, .Lfunc_end-{name}

.rodata
.p2align 6
.amdhsa_kernel {name}
  .amdhsa_group_segment_fixed_size {lds}
  .amdhsa_kernarg_size {ka.size}
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_next_free_vgpr {total}
  .amdhsa_accum_offset {acc_off}
  .amdhsa_next_free_sgpr {sgprs}
  .amdhsa_reserve_vcc 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 2 ]
amdhsa.kernels:
  - .name: {name}
    .symbol: {name}.kd
    .kernarg_segment_size: {ka.size}
    .kernarg_segment_align: 8
    .group_segment_fixed_size: {lds}
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .sgpr_count: {sgprs}
    .vgpr_count: {total}
    .agpr_count: {agprs}
    .max_flat_workgroup_size: {block}
    .args:
{ka.metadata()}
...
.end_amdgpu_metadata
"""


def rsrc(sbase, ptr_sgpr, nbytes_sgpr=None, nbytes=None):
    """Buffer resource in s[sbase:sbase+3] from a 64-bit pointer in s[ptr_sgpr:+1] and a byte size (SGPR or const)."""
    n = f"s{nbytes_sgpr}" if nbytes_sgpr is not None else str(nbytes)
    return [
        f"  s_mov_b32 s{sbase}, s{ptr_sgpr}",
        f"  s_and_b32 s{sbase + 1}, s{ptr_sgpr + 1}, 0xffff",
        f"  s_mov_b32 s{sbase + 2}, {n}",
        f"  s_mov_b32 s{sbase + 3}, {RSRC3:#x}",
    ]


def _tool(name):
    dirs = []
    spec = importlib.util.find_spec("_rocm_sdk_devel")
    if spec is not None and spec.submodule_search_locations:
        dirs += [
            os.path.join(p, "lib/llvm/bin") for p in spec.submodule_search_locations
        ]
    if os.environ.get("ROCM_PATH"):
        dirs.append(os.path.join(os.environ["ROCM_PATH"], "llvm/bin"))
    dirs.append("/opt/rocm/llvm/bin")
    for d in dirs:
        p = os.path.join(d, name)
        if os.access(p, os.X_OK):
            return p
    return shutil.which(name)


H, I, E, TOPK = 6144, 1536, 128, 4
KT = I // 128  # 12 K-steps of 128
W2_E = H * I // 2  # fp4 bytes per expert
S2_E = H * I // 32  # e8m0 bytes per expert
XROW = I // 2  # 768 B per activation row
OROW = 2 * H  # 12288 B per bf16 output row
CHUNK = 1536  # e8m0_shuffle bytes per 32 rows (I/32 = 48 columns)


def magic_div(d, nmax):
    """(m, s) with floor(w / d) == (w * m) >> (32 + s) for 0 <= w < nmax (s_mul_hi_u32 then s_lshr)."""
    for s in range(16):
        m = -(-(1 << (32 + s)) // d)
        if (
            m < (1 << 32)
            and all(
                (w * m) >> (32 + s) == w // d
                for w in (nmax - 1, nmax - 2, d - 1, d, 2 * d - 1)
            )
            and all(
                (w * m) >> (32 + s) == w // d
                for w in range(0, nmax, max(1, nmax // 4096))
            )
        ):
            return m, s
    raise ValueError(d)


def spart_tile(bid, M0, N0, gnum=4, m01=2):
    """AITER's gemm2 tile order (aiter/ops/flydsl/kernels/mxmoe_dispatcher.py _spart_output_tile_index = ck_tile
    GemmSpatiallyLocalTilePartitioner::GetOutputTileIndex; MXFP4_G2_SPART 402 = GroupNum 4, M01 2): 1D block id ->
    (m_block, n_block). Host model of the v2ao mapping, checked against the kernel's work dump.
    """
    mn = M0 * N0
    gs = -(-mn // gnum)
    big = gnum - (gs * gnum - mn)
    gy, gx = bid // gnum, bid % gnum
    remap = gx * gs + gy if gx <= big else gx * gs + big - gx + gy
    im0, in0 = remap // N0, remap % N0
    m0_mod = M0 % m01
    m01a = m01 if im0 < M0 - m0_mod else m0_mod
    im00, im01 = im0 // m01, im0 % m01
    loc = in0 + im01 * N0
    return loc % m01a + im00 * m01, loc // m01a


KA2 = KArgs(
    ("w2", "ptr"),
    ("s2", "ptr"),
    ("x", "ptr"),
    ("xs", "ptr"),
    ("eids", "ptr"),
    ("sids", "ptr"),
    ("nvalid", "ptr"),
    ("out", "ptr"),
    ("x_bytes", "u32"),
    ("xs_bytes", "u32"),
    ("out_bytes", "u32"),
    ("eids_bytes", "u32"),
    ("sids_bytes", "u32"),
    ("M", "u32"),
)
KA2A = KArgs(
    *KA2.args, ("sw", "ptr")
)  # atomic epilogue: + sorted_weights (out = moe_out, out_bytes = M 12288)


class _Lgkm:
    """Outstanding LDS ops of one wave, in issue order (no SMEM in flight where it is used): lgkmcnt for a given op."""

    def __init__(self):
        self.q = []

    def issue(self, tag):
        self.q.append(tag)
        assert len(self.q) <= 15, self.q

    def wait(self, tag):
        if tag not in self.q:
            return None
        i = self.q.index(tag)
        n = len(self.q) - i - 1
        self.q = self.q[i + 1 :]
        return f"  s_waitcnt lgkmcnt({n})"

    def clear(self):
        self.q = []


SBUF = (
    26624  # LDS bytes per activation stage buffer: 24 fragments x 1 KB + 2 KB a2 scales
)
TBUF = 2 * SBUF  # per-wave output transpose buffers (16 rows x 144 B)
TPITCH = 144


def v2_source(
    BM=128,
    NSL=24,
    store_nt=False,
    w2_nt=False,
    tag="v2",
    dbg=None,
    parts=("w2", "dma", "st"),
    scan_clamp=True,
    G=None,
    dense=False,
    atomic=False,
    atomic_nt=False,
    order="xcd",
):
    """v2: CTA = (expert e, 256-feature slice s, sorted block b) of 4 waves, 64 VGPR + 192 AGPR so 2 CTAs share
    a CU (one CTA's loads overlap the other's MFMAs). Work index w (XCD-dealt over the real work, P read on the device)
    -> b' = w / NSL; the run of blocks of e = eids[b'] is found by a 64-wide ballot scan; inside the run slice-outer,
    block-inner, so concurrent CTAs of one expert slice share its W2 in L2. W2 slice in AGPRs, loaded once per CTA.
    Activations + a2 scales by LDS DMA (buffer_load_dwordx4 ... lds) in 32-row stages, double-buffered; MFMA B operands
    via ds_read_b128 (3-slot ring). Rows up to the last valid one of the block (sorted_token_ids & 0xffffff < M) are
    computed, all-padding 16-row groups skipped. Output: bf16 through a per-wave LDS transpose, 128 B line stores.

    Debug knobs: dbg="dump" writes the per-CTA work scalars (16 dwords at out + 64 CTA) and exits before
    any W2/activation access; parts drops VMEM groups ("w2": W2 tiles + scales, "dma": activation DMA, "st": stores);
    scan_clamp=False restores the unclamped expert-run scan addresses (lane b' - l < 0 -> voffset 4 (b' - l) wraps).
    dbg="timeline" runs the full kernel and has wave 0 of each working CTA write 16 dwords at
    out_bytes - 64 (CTA + 1) (past the last valid flat row): s_memrealtime at start / end, s_memtime at start, after the
    work mapping, after the W2 slice + stage-0 activations arrived (.Lpro barrier) and at the end, HW_ID, XCC_ID,
    e, s, b, ng.

    v3 (G set): multi-block CTAs. Of the run's NSL n work items (slice s = q / n, r = q % n) only r < ng_e work,
    ng_e = ceil(n / Gb) balanced groups (Gb = G 128 / BM blocks): group r = blocks [r n / ng_e, (r + 1) n / ng_e) of
    the run, one contiguous row range [R0, R0 + nb BM). The CTA loads its W2 slice once and streams the range through
    the same double-buffered 32-row stages in a runtime loop (2 unrolled stage bodies, s83 = 32 st); s79 = RV = rows up
    to the last valid one ((nb - 1) BM + nv_end of the group's last block, the only one that can hold padding: AITER
    pads an expert's range at its end). The dump record has nb in place of nwork.

    v4 (G set, dense=True): the launch holds working CTAs only, so the in-order SE round-robin dispatch never
    stalls behind a pattern of idle CTAs (as v3 did). Groups are aligned to the expert run (group k of expert e = blocks
    [RS_e + k Gb, + Gb), the run's last group may be partial) and ordered size-descending: every full group in expert
    order, then the partial groups of size Gb - 1, ..., 1, each class in expert order. CTA i -> group
    g = i % 8 + 8 (i / 8 / NSL), slice s = i / 8 % NSL: a group's NSL slices are consecutive on one XCD, groups go
    round-robin over the XCDs, CTAs with g >= Ng (only at each XCD's tail) exit. Mapping on the device: the 256
    threads write each expert's run start / end into LDS (RS / RE at TBUF, before any transpose use) from the run
    ends of sorted_expert_ids, then every wave scans the 128 full-group counts (DPP prefix) and ranks the partial
    groups by ballot + mbcnt. Dump record: mark, CTA, nblk, nb, Nfull, g, size, e, 0, 0, RS_e, n_e, k, s, bA, row0.

    atomic=True (v2 only): out is moe_out itself, zeroed before the call (AITER's sort does so for atomic
    kernelName2), and the epilogue adds w[row] x tile into moe_out[token] with buffer_atomic_pk_add_bf16 (fp32 x
    sorted_weights, RNE to bf16, as AITER's atomic epilogue). Two full 128 B row segments per atomic (lanes 0-31 / 32-63),
    8 per 16-row group, through the same LDS transpose tile. Per-row token byte offsets min(id, M) 12288 and weights
    sit in a 1 KB LDS table (TTAB / WTAB) written by every wave in the prologue; padding rows get voffset >= M 12288 =
    out_bytes, which the buffer range check drops (soffset 512 s + 128 w is outside the check).

    order="spart" (v2 only): AITER's gemm2 tile order instead of the XCD-dealt expert runs: CTA i < M0 NSL
    (M0 = nblk) -> (block m, slice n) = spart_tile(i, M0, NSL) (ck_tile spatially local partitioner, GroupNum 4, M01 2,
    as AITER's flydsl_moe2_layout gemm2 with MXFP4_G2_SPART 402), so a token's atomic adds land in about AITER's
    order. No expert-run scan. Dump record: mark, CTA, nblk, mn, gs, gy, gx, e, big, remap, im0, in0, loc, s, b, row0.
    """
    assert H == NSL * 256 and BM in (32, 64, 128)
    assert not (
        atomic and (G is not None or dbg)
    ), "atomic epilogue: v2 work mapping only"
    name = f"t1g2_{tag}_bm{BM}"
    lbm = BM.bit_length() - 1
    NSTMAX = BM // 32
    v3 = G is not None
    if v3:
        Gb = G * 128 // BM  # blocks per group
        lgb = Gb.bit_length() - 1
        assert Gb >= 1 and 1 << lgb == Gb
    assert v3 or not dense
    assert order in ("xcd", "spart") and (order == "xcd" or not v3), order
    mg, ms = magic_div(NSL, 1 << 22)
    snt = " nt" if store_nt else ""
    wnt = " nt" if w2_nt else ""
    ant = " nt" if atomic_nt else ""
    TTAB = (
        TBUF + 4 * 16 * TPITCH
    )  # atomic: 128 token byte offsets, then 128 weights (WTAB)
    WTAB = TTAB + 512
    lds = TTAB + (1024 if atomic else 0)
    nst = (
        (16 if atomic else 4) if "st" in parts else 0
    )  # VMEM ops of one stage's epilogues (2 groups)
    tl = dbg == "timeline"
    vRING, vACC, vE, vSA, vSB = 28, 40, 56, 10, 22
    c = []
    A = c.append
    lg = _Lgkm()

    def dma(st, buf=None):
        """Stage st's 32 rows: wave w copies fragments 6w..6w+5 (f = 12 gi + kk, lane-linear MFMA B order) and bytes
        [512 w, +512) of the a2-scale chunk, into stage buffer st % 2 (or buf). 7 VMEM ops per wave.
        """
        if "dma" not in parts:
            return
        sb = ((st if buf is None else buf) % 2) * SBUF
        for jj in range(6):
            A(f"  s_add_u32 m0, s81, {sb + 1024 * jj}")
            A(f"  s_add_u32 s78, s75, {24576 * st + 64 * jj}")
            A("  buffer_load_dwordx4 v3, s[36:39], s78 offen lds")
        A(f"  s_add_u32 m0, s82, {sb}")
        A(f"  s_add_u32 s78, s76, {CHUNK * st}")
        A("  s_mov_b32 exec_hi, 0")
        A("  buffer_load_dwordx4 v1, s[40:43], s78 offen lds")
        A("  s_mov_b32 exec_hi, -1")

    def w2_tiles(q0, q1):
        for q in range(q0, q1):  # 4 KB windows of the wave's contiguous 48 KB
            for u in range(4):
                ti = 4 * q + u  # = 12 t + kk
                if "w2" in parts:
                    A(
                        f"  buffer_load_dwordx4 a[{4 * ti}:{4 * ti + 3}], v1, s[28:31], s73 offen offset:{1024 * u}{wnt}"
                    )
            A("  s_add_u32 s73, s73, 4096")

    def mfma(t, kk, gi, slot):
        d = vACC + 4 * t
        a = 4 * (KT * t + kk)
        sa, sb = vSA + 6 * (t // 2) + kk // 2, vSB + kk // 2
        sel_a, sel_b = 2 * (kk % 2) + t % 2, 2 * (kk % 2) + gi
        b = vRING + 4 * slot
        cin = "0" if kk == 0 else f"v[{d}:{d + 3}]"
        A(
            f"  v_mfma_scale_f32_16x16x128_f8f6f4 v[{d}:{d + 3}], a[{a}:{a + 3}], v[{b}:{b + 3}], {cin}, v{sa}, v{sb} "
            f"op_sel:[{sel_a & 1},{sel_b & 1},0] op_sel_hi:[{sel_a >> 1},{sel_b >> 1},0] cbsz:4 blgp:4"
        )

    def epilogue(st, gi):
        if atomic:
            r0 = 32 * st + 16 * gi  # block rows r0 .. r0 + 15
            A(
                f"  ds_read_b32 v{vE}, v9 offset:{WTAB + 4 * r0}"
            )  # w[row n] (lane (n, g))
        A("  s_nop 7\n  s_nop 7\n  s_nop 7")
        if atomic:
            A("  s_waitcnt lgkmcnt(0)")
            for i in range(
                16
            ):  # fp32 acc x sorted_weights, then RNE to bf16 (AITER's order)
                A(f"  v_mul_f32 v{vACC + i}, v{vACC + i}, v{vE}")
        for t in range(4):
            d = vACC + 4 * t
            A(f"  v_cvt_pk_bf16_f32 v{vE + 2 * t}, v{d}, v{d + 1}")
            A(f"  v_cvt_pk_bf16_f32 v{vE + 2 * t + 1}, v{d + 2}, v{d + 3}")
        for t in range(4):  # lane (n, g): row n, features 16 t + 4 g .. + 3
            A(f"  ds_write_b64 v5, v[{vE + 2 * t}:{vE + 2 * t + 1}] offset:{32 * t}")
        if (
            atomic
        ):  # lane L: dword L & 31 of row 2 j + (L >> 5) -> moe_out[token] + 512 s + 128 w + 4 (L & 31)
            for j in range(8):  # token byte offsets (padding: M 12288, out of range)
                A(f"  ds_read_b32 v{vACC + j}, v7 offset:{TTAB + 4 * (r0 + 2 * j)}")
            A("  s_waitcnt lgkmcnt(7)")  # <= 15 LDS ops in flight (4-bit lgkmcnt)
            for j in range(8):
                A(f"  ds_read_b32 v{vE + j}, v6 offset:{2 * TPITCH * j}")
            for j in range(
                8
            ):  # LDS ops complete in order: data read j has 7 - j behind it
                A(f"  s_waitcnt lgkmcnt({7 - j})")
                A(f"  v_add_u32 v{vACC + j}, v{vACC + j}, v8")
                if "st" in parts:
                    A(
                        f"  buffer_atomic_pk_add_bf16 v{vE + j}, v{vACC + j}, s[52:55], s77 offen{ant}"
                    )
            lg.clear()
            return
        A(
            f"  ds_read_b128 v[{vE}:{vE + 3}], v6"
        )  # lane L: row L >> 3, 16 B at 16 (L & 7)
        A(f"  ds_read_b128 v[{vE + 4}:{vE + 7}], v6 offset:{8 * TPITCH}")  # rows 8..15
        A("  s_waitcnt lgkmcnt(0)")
        lg.clear()
        A(
            f"  s_add_u32 s78, s77, {(32 * (st or 0) + 16 * gi) * OROW}"
        )  # v3: s77 runs with the stage (st None)
        if "st" in parts:
            A(f"  buffer_store_dwordx4 v[{vE}:{vE + 3}], v7, s[52:55], s78 offen{snt}")
            A(
                f"  buffer_store_dwordx4 v[{vE + 4}:{vE + 7}], v8, s[52:55], s78 offen{snt}"
            )

    def map_v4():
        """s57 = P on entry; leaves s57 nblk, s58 g, s70 s, s60 e, s71 bA, s62 nb (dump: s59 s61 s67 s68 s69)."""
        A(f"  s_add_u32 s57, s57, {BM - 1}")
        A(f"  s_lshr_b32 s57, s57, {lbm}")  # nblk
        A("  s_lshr_b32 s58, s2, 3")  # j: position on XCD i % 8
        A(f"  s_mul_hi_u32 s59, s58, {mg:#x}")
        if ms:
            A(f"  s_lshr_b32 s59, s59, {ms}")  # j / NSL
        A(f"  s_mul_i32 s3, s59, {NSL}")
        A("  s_sub_u32 s70, s58, s3")  # slice s = j % NSL
        A("  s_lshl_b32 s59, s59, 3")
        A("  s_and_b32 s3, s2, 7")
        A("  s_add_u32 s58, s59, s3")  # group g = i % 8 + 8 (j / NSL)
        A(f"  s_min_u32 s3, s57, {E}")
        if Gb > 1:
            A(f"  s_mul_i32 s3, s3, {Gb - 1}")
        A("  s_add_u32 s3, s3, s57")
        if lgb:
            A(
                f"  s_lshr_b32 s3, s3, {lgb}"
            )  # (nblk + min(nblk, E) (Gb - 1)) / Gb >= Ng
        A("  s_cmp_ge_u32 s58, s3")
        A("  s_cbranch_scc1 .Lend")  # uniform over the CTA's waves (before any barrier)
        # RS[e] / RE[e] (first block / one past the last block of e's run) in LDS at TBUF / TBUF + 512, zero for
        # experts without blocks; block j where a run ends writes RE[e_j] = j + 1 and RS[e_j+1] = j + 1 (RS of the
        # first run stays 0); blocks of experts >= E are skipped
        A("  v_mov_b32 v31, 0")
        A("  v_lshlrev_b32 v32, 2, v0")
        A(f"  ds_write_b32 v32, v31 offset:{TBUF}")  # 256 threads zero RS and RE
        for u in range(4):
            A(f"  v_add_u32 v{10 + u}, {256 * u}, v0")
        A("  s_sub_u32 s61, s57, 1")  # nblk - 1
        A("  s_mov_b32 s59, 0")  # first block of the trip (1024 blocks per trip)
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_barrier")
        A(".Lrs:")
        for u in range(4):
            A(f"  v_add_u32 v{40 + u}, s59, v{10 + u}")  # j
            A(f"  v_lshlrev_b32 v{44 + u}, 2, v{40 + u}")
            A(f"  buffer_load_dword v{48 + u}, v{44 + u}, s[44:47], 0 offen")
            A(f"  buffer_load_dword v{52 + u}, v{44 + u}, s[44:47], 0 offen offset:4")
            A(f"  v_add_u32 v{56 + u}, 1, v{40 + u}")  # j + 1
        A("  s_waitcnt vmcnt(0)")
        for u in range(4):
            j, ce, de, j1 = 40 + u, 48 + u, 52 + u, 56 + u
            A(f"  v_cmp_ne_u32 s[62:63], v{ce}, v{de}")
            A(f"  v_cmp_eq_u32 vcc, s61, v{j}")
            A("  s_or_b64 s[62:63], s[62:63], vcc")  # a run ends at j ...
            A(f"  v_cmp_gt_u32 vcc, s57, v{j}")
            A("  s_and_b64 s[62:63], s[62:63], vcc")  # ... and j < nblk
            A(f"  v_cmp_gt_u32 vcc, s61, v{j}")
            A(
                "  s_and_b64 s[64:65], s[62:63], vcc"
            )  # block j + 1 exists: it starts a run
            A(f"  v_cmp_gt_u32 vcc, {E}, v{de}")
            A("  s_and_b64 s[64:65], s[64:65], vcc")
            A(f"  v_cmp_gt_u32 vcc, {E}, v{ce}")
            A("  s_and_b64 s[62:63], s[62:63], vcc")
            A(f"  v_lshlrev_b32 v{ce}, 2, v{ce}")
            A(f"  v_lshlrev_b32 v{de}, 2, v{de}")
            A("  s_mov_b64 exec, s[62:63]")
            A(f"  ds_write_b32 v{ce}, v{j1} offset:{TBUF + 512}")  # RE[e_j] = j + 1
            A("  s_mov_b64 exec, s[64:65]")
            A(f"  ds_write_b32 v{de}, v{j1} offset:{TBUF}")  # RS[e_j+1] = j + 1
            A("  s_mov_b64 exec, -1")
        A("  s_add_u32 s59, s59, 1024")
        A("  s_cmp_lt_u32 s59, s57")
        A("  s_cbranch_scc1 .Lrs")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_barrier")
        # every wave: lane l holds experts l and l + 64
        A(f"  v_add_u32 v31, {TBUF}, v2")
        A("  ds_read2_b32 v[32:33], v31 offset1:64")  # RS
        A("  ds_read2_b32 v[34:35], v31 offset0:128 offset1:192")  # RE
        A("  s_waitcnt lgkmcnt(0)")
        A("  v_sub_u32 v34, v34, v32")  # n_e
        A("  v_sub_u32 v35, v35, v33")
        A(f"  v_lshrrev_b32 v36, {lgb}, v34")  # f_e = full groups of e
        A(f"  v_lshrrev_b32 v39, {lgb}, v35")
        A("  s_nop 1")
        for src, dst in (
            (36, 41),
            (39, 42),
        ):  # inclusive prefix of f over the 64 lanes (DPP, 2 wait states each)
            A(f"  v_add_u32_dpp v{dst}, v{src}, v{src} row_shr:1 bound_ctrl:0")
            A("  s_nop 1")
            A(f"  v_add_u32_dpp v{dst}, v{src}, v{dst} row_shr:2 bound_ctrl:0")
            A("  s_nop 1")
            A(f"  v_add_u32_dpp v{dst}, v{src}, v{dst} row_shr:3 bound_ctrl:0")
            A("  s_nop 1")
            A(f"  v_add_u32_dpp v{dst}, v{dst}, v{dst} row_shr:4 bank_mask:0xe")
            A("  s_nop 1")
            A(f"  v_add_u32_dpp v{dst}, v{dst}, v{dst} row_shr:8 bank_mask:0xc")
            A("  s_nop 1")
            A(f"  v_add_u32_dpp v{dst}, v{dst}, v{dst} row_bcast:15 row_mask:0xa")
            A("  s_nop 1")
            A(f"  v_add_u32_dpp v{dst}, v{dst}, v{dst} row_bcast:31 row_mask:0xc")
        A("  s_nop 4")
        A("  v_readlane_b32 s73, v41, 63")  # full groups of experts 0..63
        A("  v_readlane_b32 s74, v42, 63")  # ... of experts 64..127
        A("  v_add_u32 v42, s73, v42")  # F_incl(e) for e = l + 64
        A("  s_add_u32 s61, s73, s74")  # Nfull
        A("  s_cmp_lt_u32 s58, s61")
        A("  s_cbranch_scc0 .Lpart")
        A("  v_cmp_ge_u32 vcc, s58, v41")  # full group g: e = #{e' : F_incl(e') <= g}
        A("  s_bcnt1_i32_b64 s62, vcc")
        A("  v_cmp_ge_u32 vcc, s58, v42")
        A("  s_bcnt1_i32_b64 s63, vcc")
        A("  s_add_u32 s60, s62, s63")  # e
        A("  s_and_b32 s3, s60, 63")
        A("  s_nop 4")
        A("  s_cmp_lt_u32 s60, 64")
        A("  s_cbranch_scc0 .Lfh1")
        for k, (sd, vv) in enumerate(
            (("s64", 41), ("s65", 36), ("s67", 32), ("s68", 34))
        ):
            A(f"  v_readlane_b32 {sd}, v{vv}, s3")  # F_incl(e), f_e, RS_e, n_e
        A("  s_branch .Lfh2")
        A(".Lfh1:")
        for k, (sd, vv) in enumerate(
            (("s64", 42), ("s65", 39), ("s67", 33), ("s68", 35))
        ):
            A(f"  v_readlane_b32 {sd}, v{vv}, s3")
        A(".Lfh2:")
        A("  s_sub_u32 s64, s64, s65")  # F_excl(e)
        A("  s_sub_u32 s69, s58, s64")  # k: the group's index among e's full groups
        A(f"  s_mov_b32 s59, {Gb}")
        A("  s_branch .Lmapped")
        A(".Lpart:")  # partial groups, classes of size Gb - 1 .. 1 (<= 1 per expert)
        A("  s_sub_u32 s69, s58, s61")
        A(f"  v_and_b32 v36, {Gb - 1}, v34")
        A(f"  v_and_b32 v39, {Gb - 1}, v35")
        for sz in range(Gb - 1, 0, -1):
            A(f"  v_cmp_eq_u32 s[62:63], {sz}, v36")
            A(f"  v_cmp_eq_u32 s[64:65], {sz}, v39")
            A("  s_bcnt1_i32_b64 s3, s[62:63]")
            A("  s_bcnt1_i32_b64 s26, s[64:65]")
            A("  s_add_u32 s27, s3, s26")
            A(f"  s_mov_b32 s59, {sz}")
            A("  s_cmp_lt_u32 s69, s27")
            A("  s_cbranch_scc1 .Lsel")
            A("  s_sub_u32 s69, s69, s27")
        A("  s_branch .Lend")  # g >= Ng: idle
        A(".Lsel:")  # the s69-th expert (in order) of the class mask
        A("  s_mov_b32 s26, 0")
        A("  s_cmp_lt_u32 s69, s3")
        A("  s_cbranch_scc1 .Lsel1")
        A("  s_sub_u32 s69, s69, s3")
        A("  s_mov_b64 s[62:63], s[64:65]")
        A("  s_mov_b32 s26, 64")
        A(".Lsel1:")
        A("  v_mbcnt_lo_u32_b32 v43, s62, 0")
        A("  v_mbcnt_hi_u32_b32 v43, s63, v43")  # mask bits below lane l
        A("  v_cmp_eq_u32 vcc, s69, v43")
        A("  s_and_b64 vcc, vcc, s[62:63]")
        A("  s_ff1_i32_b64 s3, vcc")
        A("  s_add_u32 s60, s26, s3")  # e
        A("  s_nop 4")
        A("  s_cmp_eq_u32 s26, 0")
        A("  s_cbranch_scc0 .Lph1")
        A("  v_readlane_b32 s67, v32, s3")  # RS_e
        A("  v_readlane_b32 s68, v34, s3")  # n_e
        A("  s_branch .Lph2")
        A(".Lph1:")
        A("  v_readlane_b32 s67, v33, s3")
        A("  v_readlane_b32 s68, v35, s3")
        A(".Lph2:")
        A(
            f"  s_lshr_b32 s69, s68, {lgb}"
        )  # k = f_e: the partial group follows e's full groups
        A(".Lmapped:")
        A("  s_mov_b32 s62, s59")  # nb
        A(f"  s_lshl_b32 s3, s69, {lgb}")
        A("  s_add_u32 s71, s67, s3")  # bA = RS_e + k Gb; s70 = slice s

    # ---- kernargs, lane and wave constants
    if tl:
        A("  s_memrealtime s[88:89]")
        A("  s_memtime s[90:91]")
    A("  s_load_dwordx16 s[4:19], s[0:1], 0x0")  # w2 s2 x xs eids sids nvalid out
    A(
        "  s_load_dwordx4 s[20:23], s[0:1], 0x40"
    )  # x_bytes xs_bytes out_bytes eids_bytes
    A("  s_load_dwordx2 s[24:25], s[0:1], 0x50")  # sids_bytes M
    A("  v_readfirstlane_b32 s56, v0")
    A("  s_lshr_b32 s56, s56, 6")  # wave w
    A("  v_and_b32 v4, 63, v0")  # lane l
    A("  v_lshlrev_b32 v1, 4, v4")  # 16 l
    A("  v_lshlrev_b32 v2, 2, v4")  # 4 l
    A("  v_and_b32 v9, 15, v4")  # n
    A("  v_lshrrev_b32 v28, 4, v4")  # g
    A("  v_lshrrev_b32 v29, 3, v4")  # l >> 3
    A("  v_and_b32 v30, 7, v4")  # l & 7
    A(
        "  s_lshr_b32 s3, s56, 1"
    )  # DMA fragments of wave w: gi = w / 2, kk = 6 (w % 2) + jj
    A(f"  s_mul_i32 s3, s3, {16 * XROW}")
    A("  s_and_b32 s26, s56, 1")
    A("  s_mul_i32 s26, s26, 384")
    A("  s_add_u32 s3, s3, s26")
    A(f"  v_mul_u32_u24 v3, {XROW}, v9")
    A("  v_lshl_add_u32 v3, v28, 4, v3")
    A("  v_add_u32 v3, s3, v3")  # vDMA = (16 gi + n) 768 + 16 g + 384 (w % 2)
    A(f"  s_mul_i32 s26, s56, {16 * TPITCH}")
    A(f"  s_add_u32 s26, s26, {TBUF}")  # this wave's transpose buffer
    A(f"  v_mul_u32_u24 v5, {TPITCH}, v9")
    A("  v_lshl_add_u32 v5, v28, 3, v5")
    A("  v_add_u32 v5, s26, v5")  # vTW = TB + 144 n + 8 g
    A(f"  v_mul_u32_u24 v6, {TPITCH}, v29")
    A("  v_lshl_add_u32 v6, v30, 4, v6")
    A("  v_add_u32 v6, s26, v6")  # vTR = TB + 144 (l >> 3) + 16 (l & 7)
    A(f"  v_mul_u32_u24 v7, {OROW}, v29")
    A("  v_lshl_add_u32 v7, v30, 4, v7")  # vST0 = (l >> 3) 12288 + 16 (l & 7)
    A(f"  v_add_u32 v8, {8 * OROW:#x}, v7")  # vST1 = vST0 + 8 rows
    if atomic:  # atomic epilogue addressing replaces v6 / v7 / v8; v9 -> 4 n
        A("  v_lshrrev_b32 v6, 5, v4")
        A(f"  v_mul_u32_u24 v6, {TPITCH}, v6")
        A("  v_and_b32 v8, 31, v4")
        A("  v_lshlrev_b32 v8, 2, v8")  # 4 (l & 31): dword of the 128 B row segment
        A("  v_add3_u32 v6, s26, v6, v8")  # vTR = TB + 144 (l >> 5) + 4 (l & 31)
        A("  v_lshrrev_b32 v7, 5, v4")
        A("  v_lshlrev_b32 v7, 2, v7")  # 4 (l >> 5): TTAB row of the lane
        A("  v_lshlrev_b32 v9, 2, v9")  # 4 n: WTAB row of the MFMA output lane
    A("  s_mul_i32 s81, s56, 6144")  # act DMA M0 base (+ SBUF for buffer 1)
    A("  s_mul_i32 s82, s56, 512")
    A("  s_add_u32 s82, s82, 24576")  # a2-scale DMA M0 base
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_load_dword s57, s[16:17], 0x0")  # P = num_valid_ids[0]
    c += rsrc(28, 4, nbytes=-1)  # w2
    c += rsrc(32, 6, nbytes=-1)  # w2_scale
    c += rsrc(36, 8, nbytes_sgpr=20)  # x
    c += rsrc(40, 10, nbytes_sgpr=21)  # a2_scale
    c += rsrc(44, 12, nbytes_sgpr=23)  # sorted_expert_ids
    c += rsrc(48, 14, nbytes_sgpr=24)  # sorted_token_ids
    c += rsrc(52, 18, nbytes_sgpr=22)  # out
    A("  s_waitcnt lgkmcnt(0)")
    if dense:
        map_v4()
    elif order == "spart":
        # ---- work index: AITER's gemm2 tile order, spart_tile(i, M0, NSL) (GroupNum 4, M01 2), M0 = nblk
        A(f"  s_add_u32 s57, s57, {BM - 1}")
        A(f"  s_lshr_b32 s57, s57, {lbm}")  # M0 = nblk
        A(f"  s_mul_i32 s26, s57, {NSL}")  # mn
        A("  s_cmp_ge_u32 s2, s26")
        A("  s_cbranch_scc1 .Lend")
        A("  s_add_u32 s27, s26, 3")
        A("  s_lshr_b32 s27, s27, 2")  # gs = ceil(mn / 4)
        A("  s_lshl_b32 s3, s27, 2")
        A("  s_sub_u32 s3, s3, s26")
        A("  s_sub_u32 s61, 4, s3")  # big = 4 - (4 gs - mn)
        A("  s_lshr_b32 s58, s2, 2")  # gy
        A("  s_and_b32 s59, s2, 3")  # gx
        A("  s_mul_i32 s66, s59, s27")
        A("  s_add_u32 s66, s66, s58")  # gx gs + gy
        A("  s_add_u32 s3, s66, s61")
        A("  s_sub_u32 s3, s3, s59")  # gx gs + big - gx + gy
        A("  s_cmp_le_u32 s59, s61")
        A("  s_cselect_b32 s66, s66, s3")  # remap
        A(f"  s_mul_hi_u32 s67, s66, {mg:#x}")
        if ms:
            A(f"  s_lshr_b32 s67, s67, {ms}")  # im0 = remap / NSL
        A(f"  s_mul_i32 s3, s67, {NSL}")
        A("  s_sub_u32 s68, s66, s3")  # in0 = remap % NSL
        A("  s_and_b32 s3, s67, 1")
        A(f"  s_mul_i32 s3, s3, {NSL}")
        A("  s_add_u32 s69, s68, s3")  # loc = in0 + (im0 % 2) NSL
        A("  s_and_b32 s3, s57, 1")
        A("  s_sub_u32 s3, s57, s3")  # M0 - M0 % 2
        A("  s_lshr_b32 s70, s69, 1")
        A("  s_and_b32 s64, s69, 1")
        A("  s_cmp_lt_u32 s67, s3")  # in a full band M01 = 2, in the last (odd) band 1
        A("  s_cselect_b32 s70, s70, s69")  # s = loc / M01
        A("  s_cselect_b32 s64, s64, 0")  # loc % M01
        A("  s_and_b32 s71, s67, -2")
        A("  s_add_u32 s71, s71, s64")  # b = loc % M01 + 2 (im0 / 2)
        A("  s_lshl_b32 s3, s71, 2")
        A("  s_buffer_load_dword s60, s[44:47], s3")  # e
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  s_cmp_ge_u32 s60, {E}")
        A("  s_cbranch_scc1 .Lend")
    else:
        # ---- work index: XCD-dealt over nwork = nblk * NSL (CTA i on XCD i % 8): w = (i % 8) per + i / 8
        A(f"  s_add_u32 s57, s57, {BM - 1}")
        A(f"  s_lshr_b32 s57, s57, {lbm}")  # nblk
        A(f"  s_mul_i32 s26, s57, {NSL}")  # nwork
        A("  s_add_u32 s27, s26, 7")
        A("  s_lshr_b32 s27, s27, 3")  # per
        A("  s_lshr_b32 s58, s2, 3")
        A("  s_cmp_ge_u32 s58, s27")
        A("  s_cbranch_scc1 .Lend")
        A("  s_and_b32 s3, s2, 7")
        A("  s_mul_i32 s3, s3, s27")
        A("  s_add_u32 s58, s58, s3")  # w
        A("  s_cmp_ge_u32 s58, s26")
        A("  s_cbranch_scc1 .Lend")
        A(f"  s_mul_hi_u32 s59, s58, {mg:#x}")
        if ms:
            A(f"  s_lshr_b32 s59, s59, {ms}")  # b' = w / NSL
        # ---- the run of blocks of e = eids[b']: lanes look at b' - l and b' + l
        A("  s_lshl_b32 s3, s59, 2")
        A("  s_buffer_load_dword s60, s[44:47], s3")  # e

        def lo_addr():  # lanes below block 0 read eids[0] (masked off by v31 >= 0)
            if scan_clamp:
                A("  v_max_i32 v32, 0, v31")
                A("  v_lshlrev_b32 v32, 2, v32")
            else:
                A("  v_lshlrev_b32 v32, 2, v31")

        def hi_addr():  # lanes past the last block read eids[nblk - 1] (masked off)
            if scan_clamp:
                A("  s_sub_u32 s3, s57, 1")
                A("  v_min_u32 v35, s3, v34")
                A("  v_lshlrev_b32 v35, 2, v35")
            else:
                A("  v_lshlrev_b32 v35, 2, v34")

        A("  v_sub_u32 v31, s59, v4")
        lo_addr()
        A("  buffer_load_dword v33, v32, s[44:47], 0 offen")
        A("  v_add_u32 v34, s59, v4")
        hi_addr()
        A("  buffer_load_dword v36, v35, s[44:47], 0 offen")
        A("  s_waitcnt vmcnt(0) lgkmcnt(0)")
        A(f"  s_cmp_ge_u32 s60, {E}")
        A("  s_cbranch_scc1 .Lend")
        A("  s_mov_b32 s61, 0")
        A("  s_mov_b32 s65, s59")
        A(".Llo:")
        A("  v_cmp_eq_u32 vcc, s60, v33")
        A("  v_cmp_le_i32_e64 s[62:63], 0, v31")  # block index >= 0
        A("  s_and_b64 s[62:63], s[62:63], vcc")
        A("  s_not_b64 s[62:63], s[62:63]")
        A("  s_ff1_i32_b64 s64, s[62:63]")  # first lane off the run, -1 if all 64 on it
        A("  s_cmp_lg_i32 s64, -1")
        A("  s_cbranch_scc1 .Llo_done")
        A("  s_add_u32 s61, s61, 64")
        A("  s_sub_u32 s65, s65, 64")
        A("  v_sub_u32 v31, s65, v4")
        lo_addr()
        A("  buffer_load_dword v33, v32, s[44:47], 0 offen")
        A("  s_waitcnt vmcnt(0)")
        A("  s_branch .Llo")
        A(".Llo_done:")
        A("  s_add_u32 s61, s61, s64")  # c_lo: blocks of the run at or below b'
        A("  s_mov_b32 s66, 0")
        A("  s_mov_b32 s65, s59")
        A(".Lhi:")
        A("  v_cmp_eq_u32 vcc, s60, v36")
        A("  v_cmp_gt_u32_e64 s[62:63], s57, v34")  # block index < nblk
        A("  s_and_b64 s[62:63], s[62:63], vcc")
        A("  s_not_b64 s[62:63], s[62:63]")
        A("  s_ff1_i32_b64 s64, s[62:63]")
        A("  s_cmp_lg_i32 s64, -1")
        A("  s_cbranch_scc1 .Lhi_done")
        A("  s_add_u32 s66, s66, 64")
        A("  s_add_u32 s65, s65, 64")
        A("  v_add_u32 v34, s65, v4")
        hi_addr()
        A("  buffer_load_dword v36, v35, s[44:47], 0 offen")
        A("  s_waitcnt vmcnt(0)")
        A("  s_branch .Lhi")
        A(".Lhi_done:")
        A("  s_add_u32 s66, s66, s64")  # c_hi: blocks of the run at or above b'
        A("  s_sub_u32 s67, s59, s61")
        A("  s_add_u32 s67, s67, 1")  # b0 = first block of the run
        A("  s_add_u32 s68, s61, s66")
        A("  s_sub_u32 s68, s68, 1")  # n = blocks in the run
        A(f"  s_mul_i32 s3, s67, {NSL}")
        A(
            "  s_sub_u32 s69, s58, s3"
        )  # q = w - NSL b0 in [0, NSL n): slice q / n, block b0 + q % n
        A("  v_cvt_f32_u32 v31, s68")
        A("  v_rcp_iflag_f32 v31, v31")
        A("  v_cvt_f32_u32 v32, s69")
        A("  v_mul_f32 v32, v32, v31")
        A("  v_cvt_u32_f32 v32, v32")
        A("  s_nop 4")  # VALU write -> v_readfirstlane read: without wait states the
        A(
            "  v_readfirstlane_b32 s70, v32"
        )  # read saw the v_mul result (a work dump showed s = float bits of q / n)
        A("  s_mul_i32 s3, s70, s68")
        A("  s_sub_u32 s71, s69, s3")
        A("  s_cmp_lt_i32 s71, 0")
        A("  s_cbranch_scc0 .Lq1")
        A("  s_sub_u32 s70, s70, 1")
        A("  s_add_u32 s71, s71, s68")
        A(".Lq1:")
        A("  s_cmp_lt_i32 s71, s68")
        A("  s_cbranch_scc1 .Lq2")
        A("  s_add_u32 s70, s70, 1")
        A("  s_sub_u32 s71, s71, s68")
        A(".Lq2:")
        if v3:  # s71 = r: group r of the run's ng_e, or idle

            def sdiv(
                q, num, den, lab
            ):  # q = num / den (small unsigned), float rcp + one correction each way
                A(f"  v_cvt_f32_u32 v31, {den}")
                A("  v_rcp_iflag_f32 v31, v31")
                A(f"  v_cvt_f32_u32 v32, {num}")
                A("  v_mul_f32 v32, v32, v31")
                A("  v_cvt_u32_f32 v32, v32")
                A("  s_nop 4")
                A(f"  v_readfirstlane_b32 {q}, v32")
                A(f"  s_mul_i32 s3, {q}, {den}")
                A(f"  s_sub_u32 s3, {num}, s3")
                A("  s_cmp_lt_i32 s3, 0")
                A(f"  s_cbranch_scc0 {lab}a")
                A(f"  s_sub_u32 {q}, {q}, 1")
                A(f"  s_add_u32 s3, s3, {den}")
                A(f"{lab}a:")
                A(f"  s_cmp_lt_i32 s3, {den}")
                A(f"  s_cbranch_scc1 {lab}b")
                A(f"  s_add_u32 {q}, {q}, 1")
                A(f"{lab}b:")

            A(f"  s_add_u32 s64, s68, {Gb - 1}")
            if lgb:
                A(f"  s_lshr_b32 s64, s64, {lgb}")  # ng_e = ceil(n / Gb)
            A("  s_cmp_ge_u32 s71, s64")
            A(
                "  s_cbranch_scc1 .Lend"
            )  # r >= ng_e: idle (another CTA of this slice covers the blocks)
            A("  s_mul_i32 s65, s71, s68")  # r n
            sdiv("s63", "s65", "s64", ".Lg0")  # start = r n / ng_e
            A("  s_add_u32 s65, s65, s68")
            sdiv("s62", "s65", "s64", ".Lg1")  # end = (r + 1) n / ng_e
            A("  s_sub_u32 s62, s62, s63")  # nb >= 1 (ng_e <= n)
            A("  s_add_u32 s71, s63, s67")  # bA; s70 = slice s
        else:
            A("  s_add_u32 s71, s71, s67")  # b; s70 = slice s
    # ---- block rows, bases
    A(f"  s_lshl_b32 s72, s71, {lbm}")  # row0 = b BM
    if dbg == "dump":  # lane 0 writes 16 dwords at out + 64 CTA, then exits
        A("  s_mov_b64 exec, 1")
        A("  s_lshl_b32 s3, s2, 6")
        regs = [
            "0x7e57ab1e",
            "s2",
            "s57",
            "s62" if v3 else "s26",
            "s27",
            "s58",
            "s59",
            "s60",
            "s61",
            "s66",
            "s67",
            "s68",
            "s69",
            "s70",
            "s71",
            "s72",
        ]
        if dense:
            regs = [
                "0x7e57ab1e",
                "s2",
                "s57",
                "s62",
                "s61",
                "s58",
                "s59",
                "s60",
                "0",
                "0",
                "s67",
                "s68",
                "s69",
                "s70",
                "s71",
                "s72",
            ]
        for k, r in enumerate(regs):
            A(f"  v_mov_b32 v{40 + k}, {r}")
        for k in range(len(regs)):
            A(f"  buffer_store_dword v{40 + k}, off, s[52:55], s3 offset:{4 * k}")
        A("  s_waitcnt vmcnt(0)")
        A("  s_endpgm")
    if tl:
        A("  s_memtime s[92:93]")
    if v3:  # sorted_token_ids of the group's last block bA + nb - 1
        A("  s_sub_u32 s3, s62, 1")
        A(f"  s_lshl_b32 s3, s3, {lbm}")
        A("  s_add_u32 s3, s3, s72")
        A("  s_lshl_b32 s3, s3, 2")
    else:
        A("  s_lshl_b32 s3, s72, 2")
    A("  buffer_load_dword v37, v2, s[48:51], s3 offen")
    nsid = 1
    if BM == 128:
        A("  buffer_load_dword v38, v2, s[48:51], s3 offen offset:256")
        nsid = 2
    A(f"  s_mul_i32 s75, s72, {XROW}")  # activation bytes of the block
    A("  s_lshr_b32 s76, s72, 5")
    A(f"  s_mul_i32 s76, s76, {CHUNK}")
    A("  s_mul_i32 s3, s56, 512")
    A("  s_add_u32 s76, s76, s3")  # a2-scale chunk of the block + 512 w
    if atomic:
        A(
            "  s_mul_i32 s77, s70, 512"
        )  # moe_out: the token row comes per lane from TTAB
    else:
        A(f"  s_mul_i32 s77, s72, {OROW}")
        A("  s_mul_i32 s3, s70, 512")
        A("  s_add_u32 s77, s77, s3")
    A("  s_lshl_b32 s3, s56, 7")
    A(
        "  s_add_u32 s77, s77, s3"
    )  # out: row0 12288 + 512 s + 128 w (atomic: 512 s + 128 w)
    dma(0)
    A(f"  s_mul_i32 s74, s60, {S2_E}")  # W2 scales: e S2_E + (8 s + 2 w) 1536
    A(f"  s_mul_i32 s3, s70, {8 * CHUNK}")
    A("  s_add_u32 s74, s74, s3")
    A(f"  s_mul_i32 s3, s56, {2 * CHUNK}")
    A("  s_add_u32 s74, s74, s3")
    for ch in range(2):
        for c8 in range(KT // 2):
            if "w2" in parts:
                A(
                    f"  buffer_load_dword v{vSA + 6 * ch + c8}, v2, s[32:35], s74 offen offset:{ch * CHUNK + 256 * c8}"
                )
    A(
        f"  s_mul_i32 s73, s60, {W2_E}"
    )  # W2 tiles: e W2_E + (16 s + 4 w) 12 KB, 48 contiguous KB
    A(f"  s_mul_i32 s3, s70, {16 * KT * 1024}")
    A("  s_add_u32 s73, s73, s3")
    A(f"  s_mul_i32 s3, s56, {4 * KT * 1024}")
    A("  s_add_u32 s73, s73, s3")
    w2_tiles(0, 10)
    n_dma = (
        7 if "dma" in parts else 0
    )  # VMEM ops per dma(); with groups dropped (probes) the waits shrink
    n_w2 = 1 if "w2" in parts else 0  # so each still waits for the same data
    n1 = nsid + n_dma + n_w2 * (12 + 40)  # <= 63 outstanding (vmcnt is 6 bits)
    A(f"  s_waitcnt vmcnt({n1 - nsid})")  # sorted_token_ids of the block
    A("  v_and_b32 v37, 0xffffff, v37")
    A("  v_cmp_gt_u32 vcc, s25, v37")
    A("  s_mov_b64 s[84:85], vcc")
    if BM == 32:
        A("  s_mov_b32 s85, 0")
    if BM == 128:
        A("  v_and_b32 v38, 0xffffff, v38")
        A("  v_cmp_gt_u32 vcc, s25, v38")
        A("  s_mov_b64 s[86:87], vcc")
        A("  s_cmp_lg_u64 s[86:87], 0")
        A("  s_cbranch_scc0 .Lnvlo")
        A("  s_flbit_i32_b64 s3, s[86:87]")
        A("  s_sub_u32 s79, 0x80, s3")
        A("  s_branch .Lnvok")
        A(".Lnvlo:")
    if v3:
        A("  s_mov_b32 s79, 0")
    A("  s_cmp_lg_u64 s[84:85], 0")
    A(f"  s_cbranch_scc0 {'.Lnvok' if v3 else '.Lempty'}")
    A("  s_flbit_i32_b64 s3, s[84:85]")
    A("  s_sub_u32 s79, 64, s3")
    A(".Lnvok:")  # s79 = nv_end: 1 + last valid row of the block
    if v3:  # RV = (nb - 1) BM + nv_end of the last block
        A("  s_sub_u32 s3, s62, 1")
        A(f"  s_lshl_b32 s3, s3, {lbm}")
        A("  s_add_u32 s79, s79, s3")
        A("  s_cmp_eq_u32 s79, 0")
        A("  s_cbranch_scc1 .Lempty")
    A("  s_add_u32 s80, s79, 15")
    A("  s_lshr_b32 s80, s80, 4")  # ng: 16-row groups to compute
    if (
        atomic
    ):  # sorted_weights pointer (s[44:47], the eids descriptor, is dead after the mapping); token offsets
        A(f"  s_load_dwordx2 s[44:45], s[0:1], {KA2A.off['sw']:#x}")
        A("  v_min_u32 v43, s25, v37")
        A(
            f"  v_mul_u32_u24 v43, {OROW:#x}, v43"
        )  # row l: min(token, M) 12288 (padding -> out of range)
        if BM == 128:
            A("  v_min_u32 v44, s25, v38")
            A(f"  v_mul_u32_u24 v44, {OROW:#x}, v44")
    A(
        f"  s_waitcnt vmcnt({48 - (nsid if atomic else 0)})"
    )  # room for 8 tiles (+ weights) + DMA(1)
    w2_tiles(10, 12)
    if atomic:
        A("  s_waitcnt lgkmcnt(0)")
        c.extend(
            rsrc(44, 44, nbytes_sgpr=24)
        )  # sorted_weights: as long as sorted_token_ids
        A("  s_lshl_b32 s3, s72, 2")
        A("  buffer_load_dword v41, v2, s[44:47], s3 offen")
        if BM == 128:
            A("  buffer_load_dword v42, v2, s[44:47], s3 offen offset:256")
    if NSTMAX > 1 or v3:
        A("  s_cmp_le_u32 s79, 32")
        A("  s_cbranch_scc1 .Lnod0")
        dma(1)
        A(f"  s_waitcnt vmcnt({n_dma})")
        A("  s_branch .Lpro")
        A(".Lnod0:")
    A("  s_waitcnt vmcnt(0)")
    A(".Lpro:")
    A("  s_barrier")
    if (
        atomic
    ):  # every wave writes the whole (identical) tables and reads only after its own writes: no barrier
        A(f"  ds_write_b32 v2, v43 offset:{TTAB}")
        A(f"  ds_write_b32 v2, v41 offset:{WTAB}")
        if BM == 128:
            A(f"  ds_write_b32 v2, v44 offset:{TTAB + 256}")
            A(f"  ds_write_b32 v2, v42 offset:{WTAB + 256}")
        A("  s_waitcnt lgkmcnt(0)")
    if tl:
        A("  s_memtime s[94:95]")
        A("  s_waitcnt lgkmcnt(0)")

    # ---- stages of 32 rows (groups gi of 16)
    def body(st, sb):
        """One stage's MFMAs + epilogues from buffer sb; st static (v2) or None (v3: rows s83 .. s83 + 31)."""
        for c8 in range(KT // 2):
            A(f"  ds_read_b32 v{vSB + c8}, v2 offset:{sb + 24576 + 256 * c8}")
            lg.issue(f"s{c8}")
        for i in range(3):
            A(
                f"  ds_read_b128 v[{vRING + 4 * i}:{vRING + 4 * i + 3}], v1 offset:{sb + 1024 * i}"
            )
            lg.issue(f"f{i}")
        for gi in range(2):
            if gi == 1:
                if st is None:  # only the CTA's last stage can have 1 group
                    A("  s_add_u32 s3, s83, 16")
                    A("  s_cmp_le_u32 s79, s3")
                else:
                    A(f"  s_cmp_le_u32 s80, {2 * st + 1}")
                A("  s_cbranch_scc1 .Ldone")
            for kk in range(KT):
                i = KT * gi + kk
                wt = lg.wait(f"f{i}")
                if wt:
                    A(wt)
                for t in range(4):
                    mfma(t, kk, gi, i % 3)
                j = i + 3
                if j < 2 * KT:
                    A(
                        f"  ds_read_b128 v[{vRING + 4 * (j % 3)}:{vRING + 4 * (j % 3) + 3}], v1 offset:{sb + 1024 * j}"
                    )
                    lg.issue(f"f{j}")
            epilogue(st, gi)

    if v3:  # runtime stage loop, entered at stage 0 (DMA 0 / 1 in flight)
        A("  s_mov_b32 s83, 0")
        A("  s_branch .Lb0")
        for h in (0, 1):
            A(f".Ltop{h}:")
            A("  s_cmp_le_u32 s79, s83")
            A("  s_cbranch_scc1 .Ldone")
            A(
                f"  s_waitcnt vmcnt({nst})"
            )  # my DMA(st): only stage st-1's stores (4; atomic 16) came after
            A("  s_barrier")
            A("  s_add_u32 s3, s83, 32")
            A("  s_cmp_le_u32 s79, s3")
            A(f"  s_cbranch_scc1 .Lnodr{h}")
            dma(1, buf=1 - h)  # stage st + 1 (s75 / s76 point at stage st)
            A(f".Lnodr{h}:")
            if h == 0:
                A(".Lb0:")
            body(None, h * SBUF)
            A("  s_add_u32 s75, s75, 24576")
            A(f"  s_add_u32 s76, s76, {CHUNK}")
            A(f"  s_add_u32 s77, s77, {32 * OROW}")
            A("  s_add_u32 s83, s83, 32")
        A("  s_branch .Ltop0")
    else:
        for st in range(NSTMAX):
            sb = (st % 2) * SBUF
            if st > 0:
                A(f"  s_cmp_le_u32 s80, {2 * st}")
                A("  s_cbranch_scc1 .Ldone")
                A(
                    f"  s_waitcnt vmcnt({nst})"
                )  # my DMA(st): only stage st-1's stores (4; atomic 16) came after
                A("  s_barrier")
                if st + 1 < NSTMAX:
                    A(f"  s_cmp_le_u32 s79, {32 * (st + 1)}")
                    A(f"  s_cbranch_scc1 .Lnod{st}")
                    dma(st + 1)
                    A(f".Lnod{st}:")
            body(st, sb)
    A(".Ldone:")
    if tl:
        A("  s_memtime s[96:97]")
        A("  s_memrealtime s[98:99]")
        A("  s_getreg_b32 s3, hwreg(HW_REG_HW_ID)")
        A("  s_getreg_b32 s26, hwreg(HW_REG_XCC_ID)")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_cmp_lg_u32 s56, 0")
        A("  s_cbranch_scc1 .Ltl_end")
        A("  s_mov_b64 exec, 1")
        regs = [
            "0x7e57ab1e",
            "s2",
            "s88",
            "s89",
            "s98",
            "s99",
            "s90",
            "s92",
            "s94",
            "s96",
            "s3",
            "s26",
            "s60",
            "s70",
            "s71",
            "s80",
        ]
        for k, r in enumerate(regs):
            A(f"  v_mov_b32 v{40 + k}, {r}")
        A("  s_lshl_b32 s27, s2, 6")
        A("  s_sub_u32 s27, s22, s27")
        A("  s_sub_u32 s27, s27, 64")
        for k in range(len(regs)):
            A(f"  buffer_store_dword v{40 + k}, off, s[52:55], s27 offset:{4 * k}")
        A(".Ltl_end:")
    A("  s_endpgm")
    A(".Lempty:")
    A("  s_waitcnt vmcnt(0)")
    A(".Lend:")
    A("  s_endpgm")
    ka = KA2A if atomic else KA2
    src = kernel_source(
        name,
        c,
        ka,
        lds=lds,
        vgprs=64,
        agprs=4 * 4 * KT,
        sgprs=100 if tl else 88,
        block=256,
    )
    if (
        dense
    ):  # >= 8 NSL ceil(Ng / 8) for any routing: Ng = sum_e ceil(n_e / Gb) <= (nblk + min(nblk, E) (Gb - 1)) / Gb

        def grid(neid):
            return 8 * NSL * (-(-((neid + min(neid, E) * (Gb - 1)) // Gb) // 8))

    else:

        def grid(neid):
            return 8 * (-(-neid * NSL // 8))

    return {
        "name": name,
        "src": src,
        "ka": ka,
        "NSL": NSL,
        "BM": BM,
        "vgprs": 64,
        "agprs": 4 * 4 * KT,
        "lds": lds,
        "sids": True,
        "Gb": Gb if v3 else None,
        "dense": dense,
        "grid": grid,
        "atomic": atomic,
        "order": order,
    }


# Served variants; aiter/ops/asm/moe_stage2_mxfp4_t1.py picks one by token count.
SOURCES = {
    "v2n": lambda BM=128: v2_source(BM, store_nt=True, tag="v2n"),
    "v2ao": lambda BM=128: v2_source(BM, tag="v2ao", atomic=True, order="spart"),
    "v4g8": lambda BM=128: v2_source(BM, tag="v4g8", G=8, dense=True),
}
# The (variant, block_m) pairs the launcher can pick: v2n and v4g8 at block_m 128,
# the only block_m of AITER's FlyDSL scatter epilogue.
SERVED = (
    ("v2n", 128),
    ("v4g8", 128),
)
CSV_NAME = "moe_stage2_mxfp4_t1.csv"
CSV_HEADER = """\
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 MXFP4 MoE stage-2 kernels for one shape. Generated by gen_t1g2.py, read
// by aiter/ops/asm/moe_stage2_mxfp4_t1.py.
//
// Columns:
//   knl_name     : kernel symbol in the .co
//   co_name      : .co file name, relative to this directory
//   BM           : block_m of the sort the kernel reads
//   atomic       : 1 = bf16 atomic epilogue into moe_out (v2ao);
//                  0 = bf16 rows into a flat buffer, then scatter-reduce
//   dense        : 1 = dense groups of up to Gb blocks of one expert (v4g8)
//   Gb           : blocks per group (dense kernels; else 0)
//   NSL          : 256-feature slices of D_HIDDEN per block or group
//   NE, TOPK, D_HIDDEN, D_INTER : the shape the kernel serves
//   kernarg_size : bytes of the packed kernarg
//
"""
CSV_COLS = (
    "knl_name",
    "co_name",
    "BM",
    "atomic",
    "dense",
    "Gb",
    "NSL",
    "NE",
    "TOPK",
    "D_HIDDEN",
    "D_INTER",
    "kernarg_size",
)
COMMANDS = (
    "clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c {s} -o {o}",
    "ld.lld -shared {o} -o {co}",
)


def _tool_at(name, llvm_bin):
    p = os.path.join(llvm_bin, name) if llvm_bin else _tool(name)
    if p is None or not os.access(p, os.X_OK):
        raise FileNotFoundError(f"{name} not found; pass --llvm-bin")
    return p


def assemble(src, name, out_dir, clang, lld):
    """gfx950 assembly -> out_dir/<name>.co (COMMANDS); returns the .s path."""
    s, o, co = (os.path.join(out_dir, f"{name}.{ext}") for ext in ("s", "o", "co"))
    with open(s, "w") as f:
        f.write(src)
    for tool, cmd in zip((clang, lld), COMMANDS):
        argv = [tool] + [w.format(s=s, o=o, co=co) for w in cmd.split()[1:]]
        r = subprocess.run(argv, capture_output=True, text=True, check=False)
        if r.returncode:
            raise RuntimeError(f"{name}: {argv[0]} failed:\n{r.stderr[-2000:]}")
    os.remove(o)
    return s


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def generate(out_dir, llvm_bin=None, asm_dir=None):
    """Build the SERVED kernels, the registry csv and build_info.json in out_dir."""
    clang, lld = _tool_at("clang", llvm_bin), _tool_at("ld.lld", llvm_bin)
    version = subprocess.run(
        [clang, "--version"], capture_output=True, text=True, check=False
    ).stdout.splitlines()[0]
    os.makedirs(out_dir, exist_ok=True)
    rows, kernels = [], {}
    for tag, bm in SERVED:
        d = SOURCES[tag](BM=bm)
        name = d["name"]
        s = assemble(d["src"], name, out_dir, clang, lld)
        if asm_dir:
            os.makedirs(asm_dir, exist_ok=True)
            shutil.move(s, os.path.join(asm_dir, f"{name}.s"))
        else:
            os.remove(s)
        kernels[name] = {
            "source_sha1": hashlib.sha1(d["src"].encode()).hexdigest(),
            "co_sha256": sha256(os.path.join(out_dir, f"{name}.co")),
        }
        rows.append(
            (name, f"{name}.co", bm, int(d["atomic"]), int(d["dense"]))
            + (d["Gb"] or 0, d["NSL"], E, TOPK, H, I, d["ka"].size)
        )
    with open(os.path.join(out_dir, CSV_NAME), "w") as f:
        f.write(CSV_HEADER + ",".join(CSV_COLS) + "\n")
        f.writelines(",".join(str(v) for v in r) + "\n" for r in rows)
    info = {
        "generator": "gen_t1g2.py",
        "clang": version,
        "commands": [
            c.format(s="<name>.s", o="<name>.o", co="<name>.co") for c in COMMANDS
        ],
        "kernels": kernels,
    }
    with open(os.path.join(out_dir, "build_info.json"), "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    return info


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=HERE, help="output directory (default: here)")
    ap.add_argument("--llvm-bin", help="directory that holds clang and ld.lld")
    ap.add_argument("--asm-dir", help="also write the generated assembly here")
    ap.add_argument(
        "--check",
        metavar="SCRATCH",
        help="build into SCRATCH and compare with this directory's build_info.json",
    )
    a = ap.parse_args()
    if not a.check:
        info = generate(a.out, a.llvm_bin, a.asm_dir)
        print(f"{len(info['kernels'])} kernels built with {info['clang']}")
        return 0
    with open(os.path.join(HERE, "build_info.json")) as f:
        want = json.load(f)["kernels"]
    got = generate(a.check, a.llvm_bin, a.asm_dir)["kernels"]
    bad = sorted(n for n in want.keys() | got.keys() if want.get(n) != got.get(n))
    for n in bad:
        print(f"MISMATCH {n}: shipped {want.get(n)}, rebuilt {got.get(n)}")
    print(f"{len(got) - len(bad)} of {len(want)} kernels match build_info.json")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
