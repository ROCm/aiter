# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Spec builders and the validated cell table for MiniMax-M3 SP4 QKV (pure Python, no torch).

Field map authority: golden case_config.py field_spec (sp4-attention-fusion-cases-20261009):
  q (0,8192,4) k (8192,512,4) v (8704,512,4) [iq (9216,512,4) ik (9728,128,1)]  -> dense N=9216 F=2304; indexer N=9856 F=2560.
Reuse uses the indexer GEMM fields but drops iq/ik in the exchange (compact F=2304, approved).
"""

from __future__ import annotations

from aiter.gemm_fusion.spec import (
    FusionSpec,
    GemmSpec,
    HeadNormRopeCache,
    SPExchange,
    qkv_fields,
)

MODES = ("dense", "indexer", "reuse")
WORLD = 4
HIDDEN_K = 6144
# Golden case_config.cache_page_size = 16 if has_indexer else 128: dense -> page128; indexer AND reuse (N=9856) -> page16.
KV_PAGE = {"dense": 128, "indexer": 16, "reuse": 16}


def kv_cache_str(mode: str) -> str:
    return f"shuffle_page{KV_PAGE[mode]}_fp8_ptok"


def sp_qkv_spec(mode: str, world: int = WORLD, eps: float = 1e-6) -> FusionSpec:
    if mode not in MODES:
        raise ValueError(mode)
    if mode == "dense":
        fields = qkv_fields(q=8192, kv=512, world=world, kv_shards=world)
        n, drop, idx, num = 9216, (), None, "triton_norm_rope_cache_v1"
    else:
        fields = qkv_fields(
            q=8192, kv=512, world=world, iq=512, ik=128, kv_shards=world, ik_shards=1
        )
        n = 9856
        if mode == "indexer":
            drop, idx, num = (), "fp8_unit_page128", "aiter_fused_qknorm_idxrqknorm_v1"
        else:
            drop, idx, num = ("iq", "ik"), None, "triton_norm_rope_cache_v1"
    return FusionSpec(
        GemmSpec(n=n, k=HIDDEN_K),
        (
            SPExchange(world=world, fields=fields, drop=drop),
            HeadNormRopeCache(
                head_dim=128,
                q_heads=16,
                kv_heads=1,
                rotary_dim=64,
                eps=eps,
                numerics=num,
                kv_cache=kv_cache_str(mode),
                index=idx,
            ),
        ),
    )


def mode_of(spec: FusionSpec) -> str:
    """Recover the mode from a spec (used by the variant to select the compiled geometry)."""
    ex = spec.stages[0]
    names = {f.name for f in ex.fields}
    if "iq" not in names:
        return "dense"
    return "reuse" if "iq" in ex.drop else "indexer"


# (mode, tokens T) -> (vid, params). local M = T / 4. Ledger r165 (small) / r152 (large, filled in S2-2b-iii).
SMALL_CELLS = {
    ("dense", 64): (
        "sp_qkv.small.v1",
        {"src": "r17", "sk": 3, "nh": 1, "no_credit": 1},
    ),
    ("dense", 128): (
        "sp_qkv.small.v1",
        {
            "src": "r18",
            "sk": 3,
            "nh": 1,
            "no_credit": 1,
            "b_rowsplit": 1,
            "rx_bypass": 1,
        },
    ),
    ("indexer", 64): (
        "sp_qkv.small.v1",
        {"src": "r17", "sk": 3, "nh": 1, "no_credit": 1, "push8": 1},
    ),
    ("indexer", 128): (
        "sp_qkv.small.v1",
        {"src": "r17", "sk": 3, "nh": 1, "no_credit": 1},
    ),
    ("reuse", 64): (
        "sp_qkv.small.v1",
        {"src": "r18", "sk": 3, "nh": 1, "no_credit": 1, "reuse_no_idx": 1},
    ),
    ("reuse", 128): (
        "sp_qkv.small.v1",
        {"src": "r17", "sk": 3, "nh": 1, "no_credit": 1, "push_halves": 1},
    ),
}

# The kernel source the shipped PROFILE runs for every small cell: r19 (R12: per-launch kernargs, per-comm ERR/STEP page, call_id tags; derived from r18 so it carries
# every r17/r18 shape switch). SMALL_CELLS keeps the per-cell r17/r18 sources of the pre-R12 build: they stay selectable per row (set "src" back) for the A/B.
SMALL_PROFILE_SRC = "r19"


def small_cell_params(mode: str, T: int) -> dict:
    """The profile params of a small cell: the tuned plan params with the kernel source the profile selects."""
    return {**SMALL_CELLS[(mode, T)][1], "src": SMALL_PROFILE_SRC}


# ---- small-M coverage beyond the two tuned cells (T64 = m16, T128 = m32) -----------------------------------------------------------------------------------------------
# R13 P2: the decode buckets below T=64 (ATOM's default capture list pads to T = 4, 8, 16, 32, 48 -> local m = 1, 2, 4, 8, 12) run on r19 with a COPY of their mode's m=16
# params (same build, same owner / windows / context: no new unit, no new window). R13 MP64 (r20, SM_MP 64) serves local m 33..64: an exact row at m=64 (T=256, the default
# captured bucket) and a range rule over 33..64 (any padded T = 4 m in 132..256, e.g. a custom capture list or an eager step). r20 has ONE push shape (the r15-r18 shape switches
# do not exist), requires rx_bypass=1 in every mode and B_HEADS=1; reuse keeps reuse_no_idx=0 (untuned). The MP128 slot serves local m 65..128 (an exact row at m=128 = T=512, the
# default captured bucket, and a range rule over 65..128): r22 (tuned MP128, SM_MP 128, 3-slot Rx in ONE 4 MiB view) {nh 1, b_quad 1, gemm_bn128 0, rx3 1} (SMALL_R22_IN_PROFILE),
# which replaced r21 (SM_MP 128, nh=2, its Rx needs an 8 MiB arena view; still selectable, SMALL_R21_IN_PROFILE). The MP256 slot serves local m 129..255 (T 516..1020) with ONE range
# rule per mode and no exact row (the default capture list pads to T = 256, 512 at most): r23 (SM_MP 256, 3-slot Rx in an 8 MiB view) {nh 1, b_quad 1, rx3 1, gemm_bn128 0, pfk 4,
# p_nt 2, push_passtrip 1} (SMALL_R23_IN_PROFILE). So every local m 1..255 fuses; local m 256..1023 has no fused path (deliberately native: no row, no rule) and the large path
# starts at m=1024.
SMALL_P2_M = (1, 2, 4, 8, 12)
# R16 pad-M: the smallest real local m the LARGE plans are padded into a bucket for (the profile field m_pad_min_large, read by match.bucket_for_m); below it only exact rows. From the
# Evidence m0 study of large/r13g at the 1024 bucket (same-window ratios vs native): native wins below m_real ~ 896 for dense (~768 for indexer / reuse), the pad path wins from there up
# (B/C 1.07-1.23 at every m_real >= 1024), so m_real 896..1023 is served by the 1024 bucket and local m 257..895 stays native (outside the small envelope, below the large band).
M_PAD_MIN_LARGE = 896
SMALL_R20_ROW_M = (64,)
SMALL_R20_RULE_M = (33, 64)  # (m_min, m_max), every m (m_multiple 1)
SMALL_R20_SRC = "r20"
SMALL_R21_ROW_M = (128,)
SMALL_R21_RULE_M = (65, 128)  # (m_min, m_max), every m (m_multiple 1)
SMALL_R21_SRC = "r21"
# MP128 slot (exact rows at m=128 + the rule over 65..128), history and numbers:
#   r21 nh=2 (216 / 240 B WGs, grid 336 / 368) beat native in the fresh GPU58 timing (tick-free, X1 on, two windows): T=260 (m=65) 1.35-1.42x, T=384 (m=96) 1.25-1.30x, T=512 (m=128)
#   1.10-1.16x, all 3 modes; r21 nh=1: 0.95-1.18x (an earlier formal timing had put nh=1 at 0.90-0.97x at T=512, which is why r21 was removed once: nh=1 is not worth selecting).
#   r22 {nh 1, b_quad 1, gemm_bn128 0, rx3 1} (B-quad, BM128 x BN64 GEMM, 3-slot Rx: ONE 4 MiB view, grid 228 / 248 <= 256 resident) is the SHIPPED arm: it passed all 8 GPU gates
#   (outputs byte-identical to r21 nh=2 at 924 / 924, the controls bite, G10 quiet / loud, hang, dead-pass, G8) and measured T=512 1.19-1.23x of native vs r21 nh=2 at 1.09-1.16x
#   (BN128 is a net loss: GEMM 200 -> 360 VGPR + 128 AGPR). The shipped arm also sets p_nt=2 (P stored sc0 sc1 nt): 3.7-3.9 % faster at T512 (58.2 / 59.0 / 58.1 us, B/C 1.23-1.28x),
#   3.7-4.3 % at T384, never slower at T260; outputs byte-exact vs golden in all arms, RAN .text identical. It also sets push_passtrip=1 (the push live-set option; all gates passed:
#   sweep 63 / 63, byte-identical to the previous build, controls, G10 quiet / loud, G8, hang, dead-pass): another 4.3-4.5 % at T512 (55.7 / 56.5 / 55.6 us, B/C 1.28-1.33x) and 1.3-3.4 % at
#   T260 / T384. r21 stays in the code, selectable (src r21 in a row / rule, or SMALL_R21_IN_PROFILE with SMALL_R22_IN_PROFILE off).
# SMALL_R21_IN_PROFILE / SMALL_R22_IN_PROFILE gate the MP128 rows / rule (gen_profile reads them at CALL time; r22 takes the slot when both are on; both off = the r21-off layout,
# m 65..255 native, which tests switch on to keep it exercised).
SMALL_R21_IN_PROFILE = False
SMALL_R21_PROFILE_NH = 2  # B_HEADS of r21's rows / rule WHEN SMALL_R21_IN_PROFILE (small_r21_params(mode, nh) accepts 1 or 2)
SMALL_R22_SRC = "r22"
SMALL_R22_IN_PROFILE = True
SMALL_R22_PROFILE_NH = 1  # 1 = the B-quad design (grid 228 / 248 <= 256 resident); 2 = B-quad with 2 heads per B WG (grid 174 / 188). A row / rule can also select r22 directly + the knobs
# r23 (MP256 v1, R14 card: BM256 GEMM, 2 iterations per RAN WG, 3-slot Rx 7,077,888 / 7,864,320 B = one 8 MiB view, P 28.3 / 30.3 MB) IS in the shipped profile (SMALL_R23_IN_PROFILE
# = True): ONE range rule per mode covers local m 129..255 (T 516..1020; no exact row: the default capture list pads to T = 256, 512 at most) with small_r23_params(mode,
# SMALL_R23_PROFILE_NH): the arm {nh 1, b_quad 1, rx3 1, gemm_bn128 0, pfk 4, p_nt 2, push_passtrip 1} for every mode. GPU58 (the all-passtrip build): every gate passed (ISA 0 scratch, sweep
# 45 / 45, hang, A-8, controls (iii) / (iv), G10 quiet / loud, G8; N-5: caches byte-exact, only 1-ULP q differences in one cell (m=192, pad 1), deterministic); T=1020 C 92.1 / 92.8 /
# 92.3 us = 1.24-1.30x of native, T=516 1.34-1.39x. r23 requires rx3 + b_quad (the header #errors otherwise); gemm_bn128 is optional (default 0: BM256 x BN64 PFK4, the GPU-measured better arm).
SMALL_R23_SRC = "r23"
SMALL_R23_IN_PROFILE = True
SMALL_R23_RULE_M = (
    129,
    255,
)  # (m_min, m_max), every m (m_multiple 1); the plan itself takes local M 1..256
SMALL_R23_PROFILE_NH = 1
# R16 pad-M: ONE exact r23 row per mode at the top of its accepted range (SM_MP 256, local m 1..256, T=1024): the BUCKET that real local m 129..256 are padded into (bucket_for_m).
# The rule over 129..255 keeps serving exact m; the row adds no owner / unit / window (same params as the rule).
SMALL_R23_ROW_M = (256,)
_NATIVE_NOTE_R23_ONLY = "no fused path (T 260..512): native by design in this layout (r23 MP256 serves T 516..1020; no MP128 source is selected)"
_NATIVE_NOTE_R21_ON = "no fused path (T 516..1020): native by design; the small kernels end at m=128 (r21 MP128, nh=2: 1.10-1.42x of native over T 260..512, GPU58 tick-free X1 timing), the large path starts at m=1024"
_NATIVE_NOTE_R22_ON = "no fused path (T 516..1020): native by design; the small kernels end at m=128 (r22 MP128 {nh 1, BN64, p_nt 2, push_passtrip 1}: T512 B/C 1.28-1.33x of native (55.7 / 56.5 / 55.6 us; 1.23-1.28x before passtrip, 1.19-1.23x before p_nt) vs r21 nh=2 at 1.09-1.16x, GPU58 timing, all gates passed), the large path starts at m=1024"
_NATIVE_NOTE_R21_OFF = "no fused path (T 260..1020, incl. the default capture bucket T=512): native by design; the small kernels end at m=64, the large path starts at m=1024 (r21 MP128 is not selected in this layout)"
NATIVE_M_RANGES = ()  # the SHIPPED layout (r22 + r23 on): every local m 1..255 fuses (local m 257..895 is outside the small envelope and below the large band: native; m 896..1023 pads into the 1024 bucket, the large rows start at 1024)


def native_m_ranges() -> tuple[dict, ...]:
    """The documented native ranges for the current profile definition (inside the small envelope, local m 1..255). Shipped (r22 and r23 on): none, every local m fuses.
    r23 on without an MP128 source: m 65..128. An MP128 source (r21 or r22) without r23: m 129..255. Neither: m 65..255.
    """
    mp128 = SMALL_R21_IN_PROFILE or SMALL_R22_IN_PROFILE
    if SMALL_R23_IN_PROFILE:
        return (
            ()
            if mp128
            else ({"m_min": 65, "m_max": 128, "note": _NATIVE_NOTE_R23_ONLY},)
        )
    if SMALL_R22_IN_PROFILE:
        return ({"m_min": 129, "m_max": 255, "note": _NATIVE_NOTE_R22_ON},)
    if mp128:  # r21 only
        return ({"m_min": 129, "m_max": 255, "note": _NATIVE_NOTE_R21_ON},)
    return ({"m_min": 65, "m_max": 255, "note": _NATIVE_NOTE_R21_OFF},)


def small_r20_params(mode: str) -> dict:
    """Params of the MP64 build of a mode (same for every m of the r20 rows and rule: one owner per mode)."""
    if mode not in MODES:
        raise ValueError(mode)
    return {
        "src": SMALL_R20_SRC,
        "sk": 3,
        "nh": 1,
        "no_credit": 1,
        "rx_bypass": 1,
    }


def small_r21_params(mode: str, nh: int = 1) -> dict:
    """Params of the MP128 build of a mode (same for every m of the r21 rows and rule: one owner per mode). nh = B_HEADS: 1 (default, 432 / 480 B WGs) or 2 (216 / 240,
    the A/B for the occupancy caveat); a profile row / rule selects it through its params.
    """
    if mode not in MODES:
        raise ValueError(mode)
    if nh not in (1, 2):
        raise ValueError(f"r21 supports nh in (1, 2), got {nh}")
    return {
        "src": SMALL_R21_SRC,
        "sk": 3,
        "nh": nh,
        "no_credit": 1,
        "rx_bypass": 1,
    }


def small_r22_params(mode: str, nh: int = 1) -> dict:
    """Params of the r22 (tuned MP128) build of a mode: {src r22, sk 3, nh, no_credit 1, rx_bypass 1} plus the option switches EXPLICITLY at the PROFILE values b_quad 1,
    gemm_bn128 0, rx3 1, p_nt 2, push_passtrip 1: the GPU result puts r22 with BN64 (gemm_bn128=0) as the best MP128 arm (T512 1.19-1.23x vs r21 nh=2 at 1.09-1.16x; BN128 is a net loss), and
    p_nt=2 (the GEMM epilogue stores P as sc0 sc1 nt, values and RAN .text unchanged) is 3.7-3.9 % faster at T512 (58.2 / 59.0 / 58.1 us, B/C 1.23-1.28x), 3.7-4.3 % at T384 and
    never slower at T260, byte-exact vs golden in all arms, and push_passtrip=1 (two one-pass push sub-trips, pk 72 floats) on top is another 4.3-4.5 % at T512 (55.7 / 56.5 / 55.6 us,
    B/C 1.28-1.33x) and 1.3-3.4 % at T260 / T384 (sweep 63 / 63, byte-identical to the previous build, controls, G10 quiet / loud, G8, hang, dead-pass); so the profile builds that arm.
    (The source's own defaults are gemm_bn128 1, p_nt 0 and push_passtrip 0, small_defs' defaults for an r22 row that does not set them: a row / rule overrides through its params.) nh = B_HEADS 1 (default: B-quad, 228 / 248 WGs) or 2.
    """
    if mode not in MODES:
        raise ValueError(mode)
    if nh not in (1, 2):
        raise ValueError(f"r22 supports nh in (1, 2), got {nh}")
    return {
        "src": SMALL_R22_SRC,
        "sk": 3,
        "nh": nh,
        "no_credit": 1,
        "rx_bypass": 1,
        "b_quad": 1,
        "gemm_bn128": 0,
        "rx3": 1,
        "p_nt": 2,
        "push_passtrip": 1,
    }


def small_r23_params(mode: str, nh: int = 1) -> dict:
    """Params of the r23 (MP256) build of a mode, at the PROFILE values (SMALL_R23_IN_PROFILE is True: all GPU gates passed on the all-passtrip build):
    {src r23, sk 3, nh, no_credit 1, rx_bypass 1} plus the options explicit: b_quad 1 and rx3 1 (REQUIRED by the source), gemm_bn128 0 (BM256 x BN64), pfk 4
    (BN64's default: 4 x (32 + 8) KiB = 160 KiB), p_nt 2 (the P store flavour that won on r22) and push_passtrip 1 (the push live-set option), the same for every mode. nh = B_HEADS 1
    (default) or 2. The remaining options (ik_reload_p, ik_seq, push_ws_late, sk 2, ...) are NOT part of the dict: they stay at the source defaults and a row / rule can set them.
    """
    if mode not in MODES:
        raise ValueError(mode)
    if nh not in (1, 2):
        raise ValueError(f"r23 supports nh in (1, 2), got {nh}")
    return {
        "src": SMALL_R23_SRC,
        "sk": 3,
        "nh": nh,
        "no_credit": 1,
        "rx_bypass": 1,
        "b_quad": 1,
        "rx3": 1,
        "gemm_bn128": 0,
        "pfk": 4,
        "p_nt": 2,
        "push_passtrip": 1,
    }


def _mp128_params(mode: str) -> dict | None:
    """The params of the MP128 slot (exact rows at m=128 + the rule over 65..128) of the current profile definition: r22 when SMALL_R22_IN_PROFILE, else r21 when
    SMALL_R21_IN_PROFILE, else None (no MP128 rows / rule: m 65..255 native)."""
    if SMALL_R22_IN_PROFILE:
        return small_r22_params(mode, SMALL_R22_PROFILE_NH)
    if SMALL_R21_IN_PROFILE:
        return small_r21_params(mode, SMALL_R21_PROFILE_NH)
    return None


def small_exact_cells() -> list[tuple[str, int, str, dict]]:
    """Every EXACT small profile row as (mode, local m, vid, params), ordered by mode then m: the two tuned cells (m16, m32), the P2 copies of the m16 params
    (m = 1, 2, 4, 8, 12), the r20 row(s) at m=64 and the r21 row(s) at m=128."""
    out = []
    for mode in MODES:
        vid16, p16 = SMALL_CELLS[(mode, 64)]
        cells = {
            T // 4: (vid, small_cell_params(mode, T))
            for (md, T), (vid, _p) in SMALL_CELLS.items()
            if md == mode
        }
        for m in SMALL_P2_M:
            cells[m] = (vid16, {**p16, "src": SMALL_PROFILE_SRC})
        for m in SMALL_R20_ROW_M:
            cells[m] = (vid16, small_r20_params(mode))
        mp128 = _mp128_params(mode)
        for m in SMALL_R21_ROW_M if mp128 else ():
            cells[m] = (vid16, mp128)
        for m in SMALL_R23_ROW_M if SMALL_R23_IN_PROFILE else ():
            cells[m] = (vid16, small_r23_params(mode, SMALL_R23_PROFILE_NH))
        for m in sorted(cells):
            out.append((mode, m, cells[m][0], cells[m][1]))
    return out


def small_range_rules() -> list[dict]:
    """The small range rules as dicts {mode, m_min, m_max, m_multiple, vid, params, name, note}, per mode (m_multiple 1, disjoint ranges; exact rows win):
    local m 1..16 on r19 with a COPY of the mode's m=16 params, 17..32 on r19 with a COPY of the m=32 params, 33..64 on r20 (MP64), and (only when SMALL_R21_IN_PROFILE / SMALL_R22_IN_PROFILE) 65..128
    on r21 (MP128). So eager decode and custom capture lists at ANY local m <= 64 fuse (<= 128 with r21). The shape switches follow the copy: push8 (an M <= 16 shape) only below 17, b_rowsplit /
    push_halves (M > 16 shapes) only above 16; the generator validates every covered m against the real plan constructor.
    """
    out = []
    for mode in MODES:
        vid, _p = SMALL_CELLS[(mode, 64)]
        ranges = [
            (
                1,
                16,
                small_cell_params(mode, 64),
                "r19_m1-16",
                "copy of the mode's m=16 params (r19); exact rows win at m 1, 2, 4, 8, 12, 16",
            ),
            (
                17,
                32,
                small_cell_params(mode, 128),
                "r19_m17-32",
                "copy of the mode's m=32 params (r19); exact row wins at m=32",
            ),
            (
                SMALL_R20_RULE_M[0],
                SMALL_R20_RULE_M[1],
                small_r20_params(mode),
                f"mp64_m{SMALL_R20_RULE_M[0]}-{SMALL_R20_RULE_M[1]}",
                "R13 MP64 (src r20), any padded T = 4 m in 132..256; m=64 (T=256, the default capture bucket) is an exact row",
            ),
        ]
        mp128 = _mp128_params(mode)
        if mp128:
            ranges.append(
                (
                    SMALL_R21_RULE_M[0],
                    SMALL_R21_RULE_M[1],
                    mp128,
                    f"mp128_m{SMALL_R21_RULE_M[0]}-{SMALL_R21_RULE_M[1]}",
                    (
                        f"R14 MP128 tuned (src r22, nh={mp128['nh']}, 4 MiB Rx view), any padded T = 4 m in 260..512; m=128 (T=512, the default capture bucket) is an exact row"
                        if mp128["src"] == SMALL_R22_SRC
                        else f"R13 MP128 (src r21, nh={mp128['nh']}, 8 MiB Rx view), any padded T = 4 m in 260..512; m=128 (T=512, the default capture bucket) is an exact row"
                    ),
                )
            )
        if SMALL_R23_IN_PROFILE:
            ranges.append(
                (
                    SMALL_R23_RULE_M[0],
                    SMALL_R23_RULE_M[1],
                    small_r23_params(mode, SMALL_R23_PROFILE_NH),
                    f"mp256_m{SMALL_R23_RULE_M[0]}-{SMALL_R23_RULE_M[1]}",
                    f"R14 MP256 (src r23, nh={SMALL_R23_PROFILE_NH}, 8 MiB Rx view, 3 slots), any padded T = 4 m in 516..1020",
                )
            )
        for lo, hi, params, label, note in ranges:
            out.append(
                {
                    "mode": mode,
                    "m_min": lo,
                    "m_max": hi,
                    "m_multiple": 1,
                    "vid": vid,
                    "params": params,
                    "name": f"{mode}_{label}",
                    "note": note,
                }
            )
    return out


# Large cells: (mode, tokens T) -> (vid, params). local M = T/4 in {2048, 4096, 8192}. Ledger r152 + r154 (gcnasm 822c1d1, r11 recipe).
# DEVIATION (recorded): all nine cells use the single LG-QSDMA-r11 code base (NO_GRIDREAD=1 default) with the per-cell plan below,
# instead of the individual r6b/r6x4/r7/r8g candidates of ledger r152 (r11 plans reproduce them; gcnasm 822c1d1 README: all 2-12 us faster, same numerics).
#   plan: n = SDMA puts per peer; half_first 0 = r6x plan, 1 = half-N first group (r7), 2 = quarter-first n3q (r9, dense M=2048 only, n=3).
LARGE_CELLS = {
    ("dense", 8192): (
        "sp_qkv.large.v1",
        {"n": 3, "half_first": 2, "no_gridread": 1},
    ),  # r11 n3q
    ("dense", 16384): (
        "sp_qkv.large.v1",
        {"n": 2, "half_first": 0, "no_gridread": 1},
    ),  # r6b n2
    ("dense", 32768): (
        "sp_qkv.large.v1",
        {"n": 4, "half_first": 0, "no_gridread": 1},
    ),  # r6b n4
    ("indexer", 8192): (
        "sp_qkv.large.v1",
        {"n": 2, "half_first": 1, "no_gridread": 1},
    ),  # r7 n2h
    ("indexer", 16384): (
        "sp_qkv.large.v1",
        {"n": 4, "half_first": 0, "no_gridread": 1},
    ),  # r6x4 n4
    ("indexer", 32768): (
        "sp_qkv.large.v1",
        {"n": 4, "half_first": 0, "no_gridread": 1},
    ),  # r6x4 n4
    ("reuse", 8192): (
        "sp_qkv.large.v1",
        {"n": 2, "half_first": 1, "no_gridread": 1},
    ),  # r8g F0 n2h (dense build, compact)
    ("reuse", 16384): (
        "sp_qkv.large.v1",
        {"n": 4, "half_first": 0, "no_gridread": 1},
    ),  # r6x4 n4
    ("reuse", 32768): (
        "sp_qkv.large.v1",
        {"n": 4, "half_first": 0, "no_gridread": 1},
    ),  # r6x4 n4
}

LARGE_PROFILE_SRC = "r12"  # the large rows/rules of the shipped profile run the R12 kernels; "r11" stays selectable per row/rule for the A/B
# R16 pad-M: the large rows / rules run src r13 (lg_launch_v3: the kernel gets the valid m_real, the bucket M only selects the plan). Evidence on r13g (Triton 3.8 env, same-window
# ratios): ISA clean (0 spills); G1 byte-identical to r12 at m == M; odd m_real 1025..4351 byte-exact vs native; the m_real disagreement fault is detected as 0x31 with abort / reset /
# recovery working; timing r13g = r12 within +-1 % at the anchors and the pad path beats native at every m_real >= 1024 (B/C 1.07-1.23). False = the r12 layout (A/B, rollback);
# gen_profile reads it at CALL time.
LARGE_R13_IN_PROFILE = True
LARGE_R13_SRC = "r13"


def large_profile_src() -> str:
    """The kernel source of the large rows / rules of the profile being generated: r13 when LARGE_R13_IN_PROFILE, else LARGE_PROFILE_SRC (r12)."""
    return LARGE_R13_SRC if LARGE_R13_IN_PROFILE else LARGE_PROFILE_SRC


def large_cell_params(mode: str, T: int) -> dict:
    """The profile params of a large cell: the tuned plan params plus the kernel source the profile selects."""
    return {**LARGE_CELLS[(mode, T)][1], "src": large_profile_src()}
