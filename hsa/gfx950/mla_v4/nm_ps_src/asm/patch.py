# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""rt.s (disassembly of the AITER #5195 v4 nm ASM, reassembles to the same .text / kernel descriptor) -> the persistent
HCA decode kernel. Script-driven rewrites only; the output is never hand-edited. Library: patch(src_text, ...) -> text.
The kernel symbol is left as in rt.s; gen.py renames it afterwards.

Launch: grid (2P,1,1), 256 threads. WG w -> (p, hg) (xcd = w&7, loc = w>>3, p = xcd*(P/8) + loc/2, hg = loc&1 when
P%8 == 0, else p = w/2, hg = w&1). s2 := hg for the whole kernel. Kernarg: see README.md (21 x 16 B slots).
Entry (splice/pre.hip, the in-kernel split planner, spliced into the prologue): every WG plans its own partition p
(split and no-split plans from one search; the no-split plan is chosen when its max partition load stays within
NS_WANY of the split plan's), publishes desc[p] (hg == 0 WG) and leaves (begin_row, end_row, begin_tile, end_tile,
flags) in a243..a247, per / Pa / org in a248..a250. s0,s1,s2 are saved in a240..a242 around the splice.
Walk (per WG): rows r = begin_row .. end_row; piece (r, tb, te):
  tb = r==begin_row ? begin_tile : 0,  te = r==end_row ? end_tile : 2^20 (clamped by K),
  split = (r==begin_row && flags&1) || (r==end_row && flags&2),  slot = 2p + (r != begin_row).
  Rows with K = 0 inside the walk are skipped without entering the ASM (they are left unwritten).
  Each piece re-enters the ASM's own prologue with s3=r, s4=0, v0=thread id; at the ASM's s_endpgm the split piece
  runs the fused combine (splice/fin.hip: last-arriver merge through agent-scope atomic counters at kernarg 0x98),
  then all 4 waves barrier, r++, loop. Empty partition: begin_row=N > end_row=-1 -> exit.
Patches of the ASM body:
  P1 chunk ([tb,te) contiguous tiles): s68 += 32tb, s71 = min(K-32tb, 32(te-tb)), s100=0 (no round-robin offset),
     s65=s67, s94=4*s67 (index stride of consecutive tiles).
  P2 s82 := split ? 2 : 1 (selects partial fp32 path vs bf16 path in the epilogue).
  P3 sink: the ASM's "split 0 owns the sink" tests (s_cmp_eq_u32 s4, 0 in the epilogue) become "s82 == 1": sink
     applied exactly once, in the unsplit epilogue; split pieces carry none (the fused combine adds it).
  P4 partial LSE in the log2 domain: split pieces overwrite lse = m*s5 + log2(ln).
  P5 partial store retargeted to the slot layout: at label_8068 o_acc base += slot*0x40000, lse_acc base += slot*512,
     s89 (row) := 0 -> o_acc[slot][hg*64+h][:], lse_acc[slot][hg*64+h] (the ASM's own lane addressing).
  P6 bf16 path (label_86D0): base = out (kernarg 0x08), row = s89 = r; plus (if lse_out != 0)
     lse_out[r][hg*64+16w+ln%16] = lse (natural log, sink included).
  P7 reloc: epilogue O-transpose LDS scratch 0x0..0x87ff -> 0x1f000..0x277ff.
  sc1: the partial path's stores (label_8068 .. label_86CC) get sc1 (agent-scope write-through, read by fin.hip).
  nowait / nowait2: epilogue store waits relaxed to lgkmcnt-only (stores need no completion wait before the next
     piece; the fused combine waits vmcnt(0) itself before its counter atomic).
  kfast: the entry touches the kernarg lines the ASM prologue reads (0x00, 0xc0, 0x100, 0x140) so its s_loads hit the
     K$; the prologue's dependent indptr round trip becomes moves (kv_indptr[r], [r+1] from the wrapper's load in
     s84/s85; split_indptr / qo_indptr = arange contract -> r, r+1).
  kv0early: Q staging moved out of the KV ring (nope 0x2000.. -> 0x1f000.., rope 0x12800.. -> 0x17000..) so the KV
     tile-0 loads (label_0A6C block) move up to right after the page indices land.
  align: two never-executed s_nop pads keep the hot code at its original byte offset mod 256 (asm/align.py).
Registers used by the wrapper (never touched by rt.s): s30 (r), s31 (slot or -1), s38/s39 (tb/te), s42 s43 s84 s85
(scratch), s101 (p), v248 (thread id), v249 (lse address), v250..v254 (readfirstlane temps), a230..a250.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "splice"))
import align as AL  # noqa: E402
import splice as SP  # noqa: E402

K = "_ZN5aiter36mla_a8w8_qh64_qseqlen1_gqaratio64_nmE"  # kernel symbol of rt.s
KF_MAP = {
    "s_load_dword s68, s[28:29], 0x0": "\ts_mov_b32 s68, s84",
    "s_load_dword s69, s[28:29], 0x4": "\ts_mov_b32 s69, s85",
    "s_load_dword s93, s[40:41], 0x0": "\ts_mov_b32 s93, s3",
    "s_load_dword s92, s[40:41], 0x4": "\ts_add_u32 s92, s3, 1",
    "s_load_dword s89, s[36:37], 0x0": "\ts_mov_b32 s89, s3",
    "s_load_dword s90, s[36:37], 0x4": "\ts_add_u32 s90, s3, 1",
}


def patch(src_text, clang, readelf, hipcc, workdir):
    """src_text: rt.s. Returns (patched text, marks, pads)."""
    _pre, _kd = SP.body(
        hipcc,
        os.path.join(SP.HERE, "pre.hip"),
        "t8pre",
        "T8PRE",
        workdir,
        vmax=240,
        smax=100,
        allow_lds=True,
    )  # uses <= 4.1 KB of LDS; ends with a barrier before the ASM
    assert _kd["group_segment_fixed_size"] <= 8192, _kd["group_segment_fixed_size"]
    _fin, _kf = SP.body(
        hipcc,
        os.path.join(SP.HERE, "fin.hip"),
        "t8fin",
        "T8FIN",
        workdir,
        vmax=240,
        smax=100,
        allow_lds=True,
    )
    assert _kf["group_segment_fixed_size"] <= 0x1F000

    ENTRY = [
        "\tv_mov_b32_e32 v248, v0",
        "\tv_accvgpr_write_b32 a240, s0",
        "\tv_accvgpr_write_b32 a241, s1",
        "\tv_accvgpr_write_b32 a242, s2",
        _pre,
        "T8PRE_RET:",
        "\ts_waitcnt vmcnt(0) lgkmcnt(0)",
        "\ts_nop 4",
        "\tv_accvgpr_read_b32 v250, a240",
        "\tv_accvgpr_read_b32 v251, a241",
        "\tv_accvgpr_read_b32 v252, a242",
        "\ts_nop 4",
        "\tv_readfirstlane_b32 s0, v250",
        "\tv_readfirstlane_b32 s1, v251",
        "\tv_readfirstlane_b32 s2, v252",
        "\ts_load_dword s42, s[0:1], 0xa0",  # P
        "\ts_load_dword s43, s[0:1], 0x0",
        "\ts_load_dword s84, s[0:1], 0xc0",
        "\ts_load_dword s85, s[0:1], 0x100",
        "\ts_load_dword s101, s[0:1], 0x140",  # kfast: warm the K$ lines of the ASM prologue
        "\ts_waitcnt lgkmcnt(0)",
        "\ts_and_b32 s43, s42, 7",
        "\ts_cmp_eq_u32 s43, 0",
        "\ts_cbranch_scc0 T8_MAP_PLAIN",
        "\ts_and_b32 s84, s2, 7",
        "\ts_lshr_b32 s85, s2, 3",
        "\ts_lshr_b32 s43, s42, 3",
        "\ts_mul_i32 s84, s84, s43",
        "\ts_lshr_b32 s43, s85, 1",
        "\ts_add_u32 s101, s84, s43",
        "\ts_and_b32 s2, s85, 1",
        "\ts_branch T8_MAPPED",
        "T8_MAP_PLAIN:",
        "\ts_lshr_b32 s101, s2, 1",
        "\ts_and_b32 s2, s2, 1",
        "T8_MAPPED:",
        "\tv_accvgpr_read_b32 v250, a243",
        "\ts_nop 4",
        "\tv_readfirstlane_b32 s30, v250",  # r = begin_row
    ]
    PIECE = [
        "T8_PIECE_START:",
        "\ts_and_b32 s1, s1, 0xffff",
        "\ts_load_dwordx2 s[42:43], s[0:1], 0x60",
        "\ts_load_dwordx2 s[76:77], s[0:1], 0x40",
        "\ts_load_dwordx2 s[74:75], s[0:1], 0xa0",
        "\ts_waitcnt lgkmcnt(0)",
        "\tv_accvgpr_read_b32 v250, a243",
        "\tv_accvgpr_read_b32 v251, a244",
        "\tv_accvgpr_read_b32 v252, a245",
        "\tv_accvgpr_read_b32 v253, a246",
        "\tv_accvgpr_read_b32 v254, a247",
        "\ts_nop 4",
        "\tv_readfirstlane_b32 s64, v250",
        "\tv_readfirstlane_b32 s65, v251",
        "\tv_readfirstlane_b32 s66, v252",
        "\tv_readfirstlane_b32 s67, v253",
        "\tv_readfirstlane_b32 s68, v254",
        "\ts_add_u32 s85, s75, -1",
        "\ts_min_i32 s85, s30, s85",
        "\ts_max_i32 s85, s85, 0",
        "\ts_lshl_b32 s85, s85, 2",
        "\ts_load_dwordx2 s[78:79], s[76:77], s85",  # kv_indptr[min(r, N-1)], kv_indptr[..+1]
        "\ts_waitcnt lgkmcnt(0)",
        # s64 begin_row s65 end_row s66 begin_tile s67 end_tile s68 flags
        "\ts_cmp_gt_i32 s30, s65",
        "\ts_cbranch_scc1 T8_DONE",
        # K = 0 row inside the walk (e.g. bucket padding): skip without entering the ASM (no LDS use, no barrier)
        "\ts_cmp_eq_u32 s78, s79",
        "\ts_cbranch_scc0 T8_KNZ",
        "\ts_add_u32 s30, s30, 1",
        "\ts_branch T8_PIECE_START",
        "T8_KNZ:",
        "\tv_accvgpr_write_b32 a236, s78",
        "\tv_accvgpr_write_b32 a237, s79",
        "\ts_and_b32 s69, s68, 1",
        "\ts_bfe_u32 s70, s68, 0x10001",
        "\ts_cmp_eq_u32 s30, s64",
        "\ts_cselect_b32 s38, s66, 0",
        "\ts_cselect_b32 s71, s69, 0",
        "\ts_cselect_b32 s72, 0, 1",
        "\ts_cmp_eq_u32 s30, s65",
        "\ts_cselect_b32 s39, s67, 0x100000",
        "\ts_cselect_b32 s73, s70, 0",
        "\ts_or_b32 s71, s71, s73",
        "\ts_lshl_b32 s73, s101, 1",
        "\ts_add_u32 s72, s73, s72",
        "\ts_cmp_eq_u32 s71, 0",
        "\ts_cselect_b32 s31, -1, s72",
        "\ts_mov_b32 s3, s30",
        "\ts_mov_b32 s4, 0",
        "\ts_mov_b32 s84, s78",
        "\ts_mov_b32 s85, s79",
        "\ts_mov_b32 s88, 0",
        "\ts_mov_b32 s70, 0",
        "\tv_mov_b32_e32 v0, v248",
        "\ts_branch T8_PADA_END",
        "@@PADA@@",
        "T8_PADA_END:",  # alignment pad (skipped), see align.py
    ]
    EPI_PART = [  # label_8068: o_acc / lse_acc slot bases
        "\ts_mul_i32 s75, s31, 0x40000",
        "\ts_add_u32 s8, s8, s75",
        "\ts_addc_u32 s9, s9, 0",
        "\ts_lshl_b32 s75, s31, 9",
        "\ts_add_u32 s12, s12, s75",
        "\ts_addc_u32 s13, s13, 0",
        "\ts_mov_b32 s89, 0",
    ]
    EPI_BF16 = [  # label_86D0: out base + optional natural-log LSE of the unsplit row
        "\ts_load_dwordx2 s[8:9], s[0:1], 0x8",
        "\ts_load_dwordx2 s[42:43], s[0:1], 0x18",
        "\ts_waitcnt lgkmcnt(0)",
        "\ts_and_b32 s9, s9, 0xffff",
        "\ts_or_b32 s9, s9, 0x40000",
        "\ts_cmp_eq_u64 s[42:43], 0",
        "\ts_cbranch_scc1 T8_NOLSE",
        "\ts_lshl_b32 s84, s89, 9",
        "\ts_lshl_b32 s85, s2, 8",
        "\ts_add_u32 s84, s84, s85",
        "\ts_lshl_b32 s85, s7, 6",
        "\ts_add_u32 s84, s84, s85",
        "\ts_add_u32 s42, s42, s84",
        "\ts_addc_u32 s43, s43, 0",
        "\tv_and_b32_e32 v249, 15, v0",
        "\tv_lshlrev_b32_e32 v249, 2, v249",
        "\tglobal_store_dword v249, v33, s[42:43]",
        "T8_NOLSE:",
    ]
    END = [
        "\ts_mov_b64 exec, -1",
        # fused combine (split pieces only)
        "\ts_cmp_eq_u32 s31, -1",
        "\ts_cbranch_scc1 T8FIN_SKIP",
        "\tv_accvgpr_write_b32 a230, s30",
        "\tv_accvgpr_write_b32 a231, s31",
        "\tv_accvgpr_write_b32 a232, s101",
        "\tv_accvgpr_write_b32 a233, s2",
        "\tv_accvgpr_write_b32 a234, s38",
        "\tv_accvgpr_write_b32 a235, s39",
        "\tv_mov_b32_e32 v0, v248",
        "\ts_nop 4",
        _fin,
        "T8FIN_RET:",
        "\ts_waitcnt vmcnt(0) lgkmcnt(0)",
        "\ts_mov_b64 exec, -1",
        "\ts_nop 4",
        "\tv_accvgpr_read_b32 v250, a240",
        "\tv_accvgpr_read_b32 v251, a241",
        "\tv_accvgpr_read_b32 v252, a233",
        "\tv_accvgpr_read_b32 v253, a230",
        "\tv_accvgpr_read_b32 v254, a232",
        "\ts_nop 4",
        "\tv_readfirstlane_b32 s0, v250",
        "\tv_readfirstlane_b32 s1, v251",
        "\tv_readfirstlane_b32 s2, v252",
        "\tv_readfirstlane_b32 s30, v253",
        "\tv_readfirstlane_b32 s101, v254",
        "T8FIN_SKIP:",
        "\ts_barrier",
        "\ts_add_u32 s30, s30, 1",
        "\ts_branch T8_PIECE_START",
        "@@PADB@@",
        "T8_DONE:",
        "\ts_endpgm",
    ]

    out, done = [], {}

    def mark(k):
        done[k] = done.get(k, 0) + 1

    lines = src_text.split("\n")
    _sa, _sb = lines.index("label_8068:"), lines.index("label_86CC:")
    # kv0early: Q-wait position and the label_0A6C KV tile-0 block
    _qa = next(i for i, ln in enumerate(lines) if ln.strip() == "s_waitcnt vmcnt(2)")
    _k0 = lines.index("label_0A6C:")
    _k1 = next(
        i
        for i in range(_k0, _k0 + 80)
        if lines[i].strip() == "s_waitcnt lgkmcnt(0)"
        and lines[i + 1].strip() == "s_barrier"
    )
    KV0BLK = lines[_k0 + 1 : _k1]
    assert (
        lines[_qa + 1].strip() == "s_barrier"
        and any("global_load_lds_dwordx4" in x for x in KV0BLK)
        and len(KV0BLK) == 50
    ), len(KV0BLK)
    for i, ln in enumerate(lines):
        t = ln.strip()
        if _sa < i < _sb and t.startswith("buffer_store"):
            ln = ln + " sc1"
            t = ln.strip()
            mark("sc1")
        if ln == f"{K}:":
            out += [ln] + ENTRY + PIECE
            mark("entry")
            continue
        # P1 chunk
        if t == "s_sub_u32 s71, s69, s68":
            out += [
                ln,
                "\ts_lshl_b32 s85, s38, 5",
                "\ts_add_u32 s68, s68, s85",
                "\ts_sub_u32 s71, s71, s85",
                "\ts_sub_u32 s84, s39, s38",
                "\ts_lshl_b32 s84, s84, 5",
                "\ts_min_u32 s71, s71, s84",
            ]
            mark("P1a")
            continue
        if t == "s_mul_i32 s100, s4, s67":
            out += [ln, "\ts_mov_b32 s100, 0"]
            mark("P1b")
            continue
        if t == "s_mul_i32 s65, s82, s67":
            out += ["\ts_mov_b32 s65, s67"]
            mark("P1c")
            continue
        if t == "s_mul_i32 s94, s94, s82":
            out += [ln, "\ts_mul_i32 s94, s67, 4"]
            mark("P1d")
            continue
        # P2
        if t == "s_mov_b32 s82, s98":
            out += [ln, "\ts_cmp_eq_u32 s31, -1", "\ts_cselect_b32 s82, 1, 2"]
            mark("P2")
            continue
        # P3 (both "s_cmp_eq_u32 s4, 0" inside the epilogue region)
        if t == "s_cmp_eq_u32 s4, 0" and i > 4000:
            out += ["\ts_cmp_eq_u32 s82, 1"]
            mark("P3")
            continue
        # P4
        if t == "v_fma_f32 v33, v27, s6, v26":
            out += [
                ln,
                "\ts_cmp_eq_u32 s82, 1",
                "\ts_cbranch_scc1 T8_LSE_NAT",
                "\tv_fma_f32 v33, v45, s5, v27",
                "T8_LSE_NAT:",
            ]
            mark("P4")
            continue
        # P5 / P6
        if ln == "label_8068:":
            out += [ln] + EPI_PART
            mark("P5")
            continue
        if ln == "label_86D0:":
            out += [ln] + EPI_BF16
            mark("P6")
            continue
        # P7 reloc
        if (
            t in ("v_lshlrev_b32_e32 v16, 2, v16", "v_lshlrev_b32_e32 v10, 2, v10")
            and i > 4000
        ):
            r = t.split()[1].rstrip(",")
            out += [ln, f"\tv_add_u32_e32 {r}, 0x1f000, {r}"]
            mark("P7")
            continue
        # kv0early
        if i < 520:
            if t == "s_add_u32 s56, 0x2000, s75" and i < 300:
                out += ["\ts_add_u32 s56, 0x1f000, s75"]
                mark("kv0e")
                continue
            if t == "v_add_u32_e32 v16, 0x12800, v16" and i < 330:
                out += ["\tv_add_u32_e32 v16, 0x17000, v16"]
                mark("kv0e")
                continue
            if t == "s_add_u32 m0, 0x12800, s75" and i < 330:
                out += ["\ts_add_u32 m0, 0x17000, s75"]
                mark("kv0e")
                continue
            if i == _qa:
                out += ["\ts_waitcnt vmcnt(10)"] + KV0BLK + ["\ts_waitcnt vmcnt(8)"]
                mark("kv0e")
                continue
            if i == _qa + 1:
                out += [
                    ln,
                    "\tv_add_u32_e32 v10, 0x1d000, v10",
                    "\tv_add_u32_e32 v34, 0x1d000, v34",
                ]
                mark("kv0e")
                continue
            if t == "s_waitcnt vmcnt(0)" and lines[i - 1] == "label_09EC:":
                out += ["\ts_waitcnt vmcnt(6)"]
                mark("kv0e")
                continue
            if _k0 < i < _k1:
                mark("kv0drop")
                continue
        # kfast
        if i < 80 and t in KF_MAP:
            out += [KF_MAP[t]]
            mark("kfast")
            continue
        # nowait2 / nowait
        if (
            t == "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"
            and lines[i - 1] == "label_8EF4:"
        ):
            out += ["\ts_waitcnt lgkmcnt(0)"]
            mark("nowait2")
            continue
        if t == "s_waitcnt vmcnt(0) lgkmcnt(0)" and 4641 < i < 5431:
            out += ["\ts_waitcnt lgkmcnt(0)"]
            mark("nowait")
            continue
        if t == "s_endpgm":
            out += END
            mark("end")
            continue
        out.append(ln)
    exp = dict(
        entry=1,
        P1a=1,
        P1b=1,
        P1c=1,
        P1d=1,
        P2=1,
        P3=2,
        P4=1,
        P5=1,
        P6=1,
        end=1,
        sc1=33,
        nowait=2,
        nowait2=1,
        kfast=6,
        kv0e=6,
        kv0drop=50,
    )
    exp["P7"] = sum(
        1
        for i, ln in enumerate(lines)
        if ln.strip()
        in ("v_lshlrev_b32_e32 v16, 2, v16", "v_lshlrev_b32_e32 v10, 2, v10")
        and i > 4000
    )
    assert done == exp, (done, exp)
    text, pads = AL.align(src_text, "\n".join(out), clang, readelf)
    return text, done, pads
