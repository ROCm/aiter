"""
Per-WMMA fine-grained schedule table for d128 FMHA on gfx1250.

160-row table (96 GEMM1 + 64 GEMM2 WMMAs) where each row is a flat list of
instruction tokens emitted in order between consecutive WMMAs.

TOKEN TYPES
-----------
  0-3   P0_Mx  : PART0 per MSB 0-3 (max3/permlane/mul/fma ops)
  4     P1     : PART1 cross-MSB merge
  5-8   P2_Mx  : PART2 per MSB 0-3 (exp/pkfma/pkadd/cvt/sum-tree)
  9-12  K_Mx   : ds_load_b128 for K tile MSB 0-3 (1 token = 1 load)
 13-16  V_Mx   : ds_load_tr16_b128 for V tile MSB 0-3 (1 token = 1 load)
 17     O_RESC0: O-rescale closure (1 arith.mulf v8f32 = 4 v_pk_mul_f32 per token)
 18     TDM    : tensor_load_to_lds (1 token = 1 TDM load)

ROW SEMANTICS
-------------
Row G1w19 = [P2_M0, P2_M1, O_RESC0]
  → between WMMA 19 and WMMA 20: emit 1 PART2 exp from MSB0,
    1 PART2 exp from MSB1, then 1 O-rescale closure (4 pk_mul).

G0w00 = [TDM, TDM]
  → between WMMA 0 and WMMA 1: emit 2 tensor_load_to_lds
    (no K loads here because can_issue=False when has_tdm+gemm_idx=0).

This matches the actual ISA pattern (L1279-1285 area):
  WMMA0 → tensor_load×2 → WMMA1 → ds_load,exp,ds_load,ds_load → WMMA2
"""

from typing import List

# ---------------------------------------------------------------------------
# Token constants
# ---------------------------------------------------------------------------
P0_M0, P0_M1, P0_M2, P0_M3 = 0, 1, 2, 3  # PART0 per MSB (max3/mul/fma, 1cy)
P1 = 4  # PART1 cross-MSB merge (1cy)
# PART2 cheap ops: setup(7) + pkfma(16) = ops[0..22], 1cy each
P2_M0, P2_M1, P2_M2, P2_M3 = 5, 6, 7, 8  # PART2 pkfma/setup per MSB (1cy)
# PART2 expensive ops: pair_exp = ops[23..30], 3cy each
EXP_M0, EXP_M1, EXP_M2, EXP_M3 = 19, 20, 21, 22  # pair_exp per MSB (3cy)

K_M0, K_M1, K_M2, K_M3 = 9, 10, 11, 12  # K tile ds_load_b128 by MSB
V_M0, V_M1, V_M2, V_M3 = 13, 14, 15, 16  # V tile ds_load_tr16_b128 by MSB

O_RESC0 = 17  # O-rescale v2f32 sub-op (1 v_pk_mul_f32 per token; 16 tokens per stage)
TDM = 18  # tensor_load_to_lds

# PART2 boundary: ops[0..PART2_EXP_START-1] = setup+pkfma
# (cheap), ops[PART2_EXP_START..] = exp
PART2_EXP_START = 24  # = PART2_SETUP_A(8) + pkfma(16)

_P2_BASE = 5
_EXP_BASE = 19  # EXP_Mx = _EXP_BASE + msb
_K_BASE = 9
_V_BASE = 13

# Shorthand maps for readability in the table
_K = [K_M0, K_M1, K_M2, K_M3]
_V = [V_M0, V_M1, V_M2, V_M3]
_P2 = [P2_M0, P2_M1, P2_M2, P2_M3]  # pkfma/setup (cheap, 1cy)
_EXP = [EXP_M0, EXP_M1, EXP_M2, EXP_M3]  # pair_exp (expensive, 3cy)

# ---------------------------------------------------------------------------
# Constants (must match fmha_core_loop.py)
# ---------------------------------------------------------------------------
_NUM_MSB = 4
_GEMM_INST_COUNT = 24  # WMMAs per GEMM1 stage
_PV_GEMM_INST_COUNT = 16  # WMMAs per GEMM2 stage
_N_LDS_PER_MSB = 6  # K tile loads per MSB per stage (QK_HDIM=192)
_N_LDS_V_PER_MSB = 4  # V tile loads per MSB per stage (V_HDIM=128)
_TDM_PER_STAGE = 2  # tensor_load_to_lds per stage
# O-rescale tile groups per GEMM1 stage
# (16 tokens = 4 MSBs x 4 sub-ops)
_N_PV_WMMA_N = 4
_NUM_G1_STAGES = 4
_NUM_G2_STAGES = 4

# ---------------------------------------------------------------------------
# Cycle cost table (for balance analysis)
# ---------------------------------------------------------------------------
CY = {
    TDM: 4,  # tensor_load_to_lds
    O_RESC0: 1,  # 1 × v_pk_mul_f32
}
# P2 in exp phase (stages 0-1) = 3 cy; in cheap phase (stages 2-3) = 1 cy.
# For builder purposes we pass a per-token cy_p2 argument (3 or 1).
_CY_K = 1  # ds_load_b128
_CY_V = 1  # ds_load_tr16_b128
_CY_P0 = 1  # max3 / permlane / mul
_CY_P1 = 1  # permlane / merge


def row_cycles(row: List[int], cy_p2: int = 1) -> int:
    """Compute cycle estimate for a schedule row.
    Token costs: TDM=4, O_RESC0=1, EXP_Mx=3, P2_Mx=1(pkfma), P0/P1/K/V=1.
    """
    total = 0
    for t in row:
        if t == TDM:
            total += 4
        elif t == O_RESC0:
            total += 1
        elif 19 <= t <= 22:
            total += 3  # EXP_Mx: pair_exp (3cy)
        elif 5 <= t <= 8:
            total += 1  # P2_Mx: pkfma/setup (1cy)
        elif 9 <= t <= 16:
            total += 1  # K or V load (1cy)
        elif 0 <= t <= 4:
            total += 1  # P0, P1 (1cy)
        else:
            total += 1
    return total


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _interleave(a_tokens: List[int], b_tokens: List[int]) -> List[int]:
    """Interleave two token lists: [a0, b0, a1, b1, ...]."""
    result = []
    for i in range(max(len(a_tokens), len(b_tokens))):
        if i < len(a_tokens):
            result.append(a_tokens[i])
        if i < len(b_tokens):
            result.append(b_tokens[i])
    return result


def _distribute(tokens: List[int], n_wmma: int) -> List[List[int]]:
    """Distribute tokens evenly across n_wmma rows."""
    rows = [[] for _ in range(n_wmma)]
    if not tokens:
        return rows
    per_wmma = -(-len(tokens) // n_wmma)  # ceil
    ti = 0
    for w in range(n_wmma):
        batch = min(per_wmma, len(tokens) - ti)
        for _ in range(batch):
            rows[w].append(tokens[ti])
            ti += 1
        if ti >= len(tokens):
            break
    return rows


def _k_tokens_for_stage(has_tdm_wmma0: bool) -> List[List[int]]:
    """Build per-WMMA K tile load token lists for one GEMM1 stage.

    K schedule: each MSB gets N_LDS_PER_MSB=6 loads.
    Phase-1 slot = 1 load, phase-2 slot = 2 loads → 3 per WMMA.
    WMMA 0 has no K loads when has_tdm (can_issue=False).
    Layout:
      WMMA 0 : 0 K loads (TDM only) if has_tdm, else 3 K loads for MSB0
      WMMAs 1-2 : 3 K loads for MSB0 (or 0-1 if WMMA 0 had some)
      WMMAs 3-4 : MSB1, WMMAs 5-6: MSB2, WMMAs 7-8: MSB3
    """
    rows_k = [[] for _ in range(_GEMM_INST_COUNT)]
    load_per_wmma = 3  # 1 (phase1) + 2 (phase2)

    if has_tdm_wmma0:
        # WMMA 0: no K loads; WMMAs 1..2*N_MSB (8 WMMAs) carry loads
        w_start = 1
    else:
        # WMMA 0 carries first 3 loads for MSB0
        w_start = 0

    w = w_start
    for msb in range(_NUM_MSB):
        remaining = _N_LDS_PER_MSB
        while remaining > 0 and w < _GEMM_INST_COUNT:
            n = min(load_per_wmma, remaining)
            rows_k[w].extend([_K[msb]] * n)
            remaining -= n
            w += 1

    return rows_k


def _v_tokens_for_stage() -> List[List[int]]:
    """Build per-WMMA V tile load token lists (always starts at WMMA 0)."""
    rows_v = [[] for _ in range(_GEMM_INST_COUNT)]
    load_per_wmma = 3
    w = 0
    for msb in range(_NUM_MSB):
        remaining = _N_LDS_PER_MSB
        while remaining > 0 and w < _GEMM_INST_COUNT:
            n = min(load_per_wmma, remaining)
            rows_v[w].extend([_V[msb]] * n)
            remaining -= n
            w += 1
    return rows_v


def _exp_tokens(exp_per_msb: int, n_wmma_active: int) -> List[int]:
    """Round-robin exp tokens: [P2_M0, P2_M1, P2_M2, P2_M3, P2_M0, ...]."""
    tokens = []
    counts = [0] * _NUM_MSB
    msb = 0
    total = exp_per_msb * _NUM_MSB
    while sum(counts) < total:
        for _ in range(_NUM_MSB):
            if counts[msb] < exp_per_msb:
                break
            msb = (msb + 1) % _NUM_MSB
        if counts[msb] >= exp_per_msb:
            break
        tokens.append(_P2[msb])
        counts[msb] += 1
        msb = (msb + 1) % _NUM_MSB
    return tokens


def _cheap_tokens(cheap_per_msb: int) -> List[int]:
    """Round-robin cheap PART2 tokens (pkadd/cvt/sum)."""
    tokens = []
    counts = [0] * _NUM_MSB
    msb = 0
    total = cheap_per_msb * _NUM_MSB
    while sum(counts) < total:
        for _ in range(_NUM_MSB):
            if counts[msb] < cheap_per_msb:
                break
            msb = (msb + 1) % _NUM_MSB
        if counts[msb] >= cheap_per_msb:
            break
        tokens.append(_P2[msb])
        counts[msb] += 1
        msb = (msb + 1) % _NUM_MSB
    return tokens


# ---------------------------------------------------------------------------
# GEMM1 stage builder
# ---------------------------------------------------------------------------


def _build_gemm1_stage(
    stage_idx: int,
    exp_per_msb: int,
    cheap_per_msb: int,
    lds_type: str = "K",
    has_tdm: bool = False,
) -> List[List[int]]:
    """Build balanced 24-row schedule for one GEMM1 stage.

    Design principles:
    1. SAME-MSB GROUPING: within each row, tokens from the same MSB bank are
       placed consecutively to minimize s_set_vgpr_msb switches.
       K-load WMMAs: [K_mX, K_mX, K_mX, P2mX, P2mX] — all MSB-X ops together.
       Exp-only WMMAs: [P2m0, P2m0, P2m1, P2m1] — consecutive same-MSB pairs.

    2. CYCLE BALANCE: ~7 cy/WMMA target.
       K-load WMMAs (1-8): 3K(3cy) + 2exp(6cy) = 9 cy
       Exp-only WMMAs (9-18): 2exp = 6 cy  (O_RESC0 moved to rows 14-17)
       O_RESC0 WMMAs (14-17): O_RESC0(4cy) + 1exp(3cy) = 7 cy
       Trailing WMMAs (18-22): 2exp = 6 cy
       → eliminates empty WMMAs, range 6-9 cy (vs old 0-12 cy)

    3. WMMA 0 with TDM: [TDM, TDM] only — no K or exp (can_issue=False).
    """
    n = _GEMM_INST_COUNT  # 24
    rows = [[] for _ in range(n)]
    K_TOK = _K if lds_type == "K" else _V

    # Mutable counters per MSB
    _n_lds = _N_LDS_V_PER_MSB if lds_type == "V" else _N_LDS_PER_MSB
    k_rem = [_n_lds] * _NUM_MSB  # K/V loads remaining
    exp_rem = [exp_per_msb] * _NUM_MSB  # exp ops remaining
    ch_rem = [cheap_per_msb] * _NUM_MSB  # cheap ops remaining

    # ---- WMMA 0: TDM only when has_tdm ----
    if has_tdm:
        rows[0] = [TDM] * _TDM_PER_STAGE
        k_start = 1
    else:
        k_start = 0

    # ---- K-load WMMAs: 2 WMMAs per MSB, all same-MSB grouped ----
    # Pattern per WMMA: [K_mX×3, P2mX×2] — K loads then exp, same bank X.
    w = k_start
    for msb in range(_NUM_MSB):
        loads_left = k_rem[msb]  # = _N_LDS_PER_MSB (6)
        while loads_left > 0 and w < n - _N_PV_WMMA_N:
            row: List[int] = []
            # K loads for this MSB (up to 3 per WMMA, same bank)
            n_k = min(3, loads_left)
            row.extend([K_TOK[msb]] * n_k)
            loads_left -= n_k
            k_rem[msb] -= n_k
            # Same-MSB exp directly after K loads (2 per WMMA)
            for _ in range(2):
                if exp_rem[msb] > 0:
                    row.append(_P2[msb])
                    exp_rem[msb] -= 1
            rows[w] = row
            w += 1

    # ---- O_RESC0: placed earlier (rows 14-17), 1 token per row ----
    # O_RESC0 in row r = emitted AFTER WMMA r = BEFORE WMMA r+1.
    # Placing at rows 14-17 (before WMMAs 15-18) is earlier than the old
    # rows 19-22, but still valid since O-rescale just needs to run before
    # the next GEMM2 PV stage, not before a specific GEMM1 WMMA.
    o_start = 14  # was: n - _N_PV_WMMA_N - 1 = 19

    # ---- Build flat P2 token list: same-MSB consecutive for bank grouping ----
    p2_flat: List[int] = []
    for msb in range(_NUM_MSB):
        p2_flat.extend([_P2[msb]] * exp_rem[msb])
    for msb in range(_NUM_MSB):
        p2_flat.extend([_P2[msb]] * ch_rem[msb])

    # ---- Fill K/V-load WMMAs with extra P2 tokens only when needed ----
    # For exp stages (exp_per_msb > 0): K-load WMMAs already have 2 exp tokens
    #   from the K-load phase → adding more would over-densify (skip).
    # For cheap-only stages (exp_per_msb == 0, e.g. stage 3): V-load WMMAs only
    #   have 3cy of loads → add 4 cheap P2 tokens to reach 7cy.
    ti = 0
    k_load_end = w  # first WMMA after K/V loads
    if exp_per_msb == 0 and cheap_per_msb > 0:
        for lw in range(k_start, k_load_end):
            for _ in range(4):  # 4 cheap tokens: 3cy loads + 4cy = 7cy
                if ti < len(p2_flat):
                    rows[lw].append(p2_flat[ti])
                    ti += 1

    # ---- Distribute remaining tokens into post-load WMMAs ----
    # Compute per-WMMA count dynamically so ALL tokens are scheduled (no post-tail).
    remaining = len(p2_flat) - ti
    post_wmma_count = n - k_load_end
    o_rows = set(range(o_start, o_start + _N_PV_WMMA_N))
    # O_RESC0 rows each get 1 P2 token + 4 O_RESC0; non-O_RESC0 rows get per_wmma tokens
    n_orsc_rows = sum(1 for rw in range(k_load_end, n) if rw in o_rows)
    n_regular_rows = post_wmma_count - n_orsc_rows
    tokens_for_regular = max(0, remaining - n_orsc_rows)
    per_wmma = -(-tokens_for_regular // max(1, n_regular_rows))  # ceil

    for row_w in range(k_load_end, n):
        row = []
        if row_w in o_rows:
            # O_RESC0 WMMA: 1 P2 + 4 O_RESC0
            if ti < len(p2_flat):
                row.append(p2_flat[ti])
                ti += 1
            row.extend([O_RESC0] * 4)
        else:
            # Regular WMMA: per_wmma P2 tokens (auto-sized to fit all remaining)
            for _ in range(per_wmma):
                if ti < len(p2_flat):
                    row.append(p2_flat[ti])
                    ti += 1
        rows[row_w] = row

    return rows


# ---------------------------------------------------------------------------
# GEMM2 stage builder
# ---------------------------------------------------------------------------


def _build_gemm2_stage(
    stage_idx: int,
    p0_per_msb: int,
    p1_total: int,
    p2_per_msb: int,
    exp_per_msb_list: "List[int] | int" = 0,
    lds_type: str = "V",
    has_tdm: bool = False,
    lds_per_msb: int = None,
    loads_per_wmma: int = None,
    row_cap: int = 6,
) -> List[List[int]]:
    """Build 16-row GEMM2 stage schedule.

    Key design rules (matching the reference schedule):
    1. V loads: 2 per WMMA (lds_per_msb=4, loads_per_wmma=2 → [V×2, companion×4])
       K loads: 3 per WMMA (lds_per_msb=6, loads_per_wmma=3 → [K×3, companion×3])
       → None auto-derives from lds_type.
    2. EXP capped at MAX 2 per WMMA row (= 6cy max from EXP). Do NOT pack 3-5.
    3. EXP interleaved with P0/P1/P2 in post-load WMMAs: when P0/P1/P2 overflow
       is present, add 1 EXP per WMMA alongside; remaining WMMAs get 2 EXP.
    4. exp_per_msb_list: list of 4 EXP counts per MSB (e.g. [8,0,0,0] = only MSB0).
       Each stage handles ONE MSB's EXP so EXP is spread evenly across 4 stages.
    """
    if lds_per_msb is None:
        lds_per_msb = 4 if lds_type == "V" else _N_LDS_PER_MSB  # V=4, K=6
    if loads_per_wmma is None:
        loads_per_wmma = 2 if lds_type == "V" else 3  # V=2, K=3

    if isinstance(exp_per_msb_list, int):
        exp_per_msb_list = [exp_per_msb_list] * _NUM_MSB

    n = _PV_GEMM_INST_COUNT
    LD_TOK = _K if lds_type == "K" else _V
    rows = [[] for _ in range(n)]
    p0_rem = [p0_per_msb] * _NUM_MSB
    p2_rem = [p2_per_msb] * _NUM_MSB
    exp_rem = list(exp_per_msb_list)  # per-MSB EXP remaining

    # Max companion tokens per load WMMA = 6 - loads_per_wmma
    _max_companion = 6 - loads_per_wmma  # 4 for V, 3 for K

    if has_tdm:
        rows[0] = [TDM] * _TDM_PER_STAGE
        w_start = 1
    else:
        w_start = 0

    # ---- Load WMMAs: same-MSB grouped, companion priority P0 > P2 > EXP ----
    w = w_start
    for msb in range(_NUM_MSB):
        ld_rem = lds_per_msb
        while ld_rem > 0 and w < n:
            row: List[int] = []
            n_ld = min(loads_per_wmma, ld_rem)
            row.extend([LD_TOK[msb]] * n_ld)
            ld_rem -= n_ld
            if p0_rem[msb] > 0:
                n_c = min(_max_companion, p0_rem[msb])
                row.extend([msb] * n_c)
                p0_rem[msb] -= n_c
            elif p2_rem[msb] > 0:
                n_c = min(_max_companion, p2_rem[msb])
                row.extend([_P2[msb]] * n_c)
                p2_rem[msb] -= n_c
            elif exp_rem[msb] > 0:
                n_c = min(2, exp_rem[msb])  # EXP capped at 2 always
                row.extend([_EXP[msb]] * n_c)
                exp_rem[msb] -= n_c
            rows[w] = row
            w += 1
    k_load_end = w

    # ---- Build post-load token lists ----
    # P0 overflow: interleaved (op-major) across MSBs instead of MSB-major.
    # With 4 MSBs, same-op firing together gives 3 other-MSB VALUs between
    # each consecutive dep pair (merge1→merge2, add→perm, perm→max3), giving
    # VALU_DEP_4 (0 stall on gfx1250, just encoding hint).
    # Constraint: ds_load companions in load WMMAs remain same-MSB (unchanged
    # above). Only this post-load section is interleaved.
    p01_flat: List[int] = []
    max_p0_rem = max(p0_rem) if p0_rem else 0
    for step in range(max_p0_rem):
        for msb in range(_NUM_MSB):
            if step < p0_rem[msb]:
                p01_flat.append(msb)
    p01_flat.extend([P1] * p1_total)
    for msb in range(_NUM_MSB):
        p01_flat.extend([_P2[msb]] * p2_rem[msb])

    # exp_flat: remaining EXP per MSB (same-MSB consecutive)
    exp_flat: List[int] = []
    for msb in range(_NUM_MSB):
        exp_flat.extend([_EXP[msb]] * exp_rem[msb])

    post_count = n - k_load_end
    if (len(p01_flat) + len(exp_flat)) == 0 or post_count == 0:
        return rows

    # ---- Fill post-load WMMAs ----
    # Goal: distribute p01 and exp tokens so that:
    #   - EXP never exceeds 2 per WMMA (avoids 15cy from packing 5)
    #   - p01 tokens evenly share remaining slots with EXP
    # Strategy: compute per-WMMA p01 quota = ceil(|p01|/post_count), then
    # each WMMA gets quota p01 tokens + up to 2 EXP.  Once p01 is done,
    # remaining WMMAs get up to 2 EXP.
    _ROW_CAP = row_cap
    _MAX_EXP = 2
    p01_per_wmma = -(-len(p01_flat) // post_count) if post_count else 0  # ceil div

    p01_i = 0
    exp_i = 0

    for rw in range(k_load_end, n):
        row = rows[rw]
        slots = _ROW_CAP - len(row)
        if slots <= 0:
            continue

        # Determine how many p01 tokens fit this WMMA
        p01_want = min(p01_per_wmma, len(p01_flat) - p01_i, slots)
        added = 0
        while added < p01_want and p01_i < len(p01_flat):
            row.append(p01_flat[p01_i])
            p01_i += 1
            added += 1

        # Remaining slots → EXP (up to 2)
        exp_slots = min(_MAX_EXP, slots - added, len(exp_flat) - exp_i)
        for _ in range(exp_slots):
            row.append(exp_flat[exp_i])
            exp_i += 1

    return rows


# ---------------------------------------------------------------------------
# GEMM1 schedule: 96 rows
# ---------------------------------------------------------------------------
# Budget derivation (ALU_PER_STAGE = [40,52,56,168, 120,120,132,132]):
#   GEMM1_EXP_OPS=24, cheap=33 per MSB
#   Stage 0 (softmax_stage 4, budget=120): 10 exp/MSB, 0 cheap
#   Stage 1 (softmax_stage 5, budget=120): 10 exp/MSB, 0 cheap
#   Stage 2 (softmax_stage 6, budget=132):  4 exp/MSB, 21 cheap
#   Stage 3 (softmax_stage 7, budget=132):  0 exp/MSB, 33 cheap
# Stages 0,1 have TDM K loads (main loop). Stage 2 has K loads, no TDM.
# Stage 3 has V tile loads (lds_type='V'), no TDM.


def build_gemm1_schedule() -> List[List[int]]:
    # All 4 stages as direct flat table — edit rows to adjust load distribution.
    # Budgets: P2(exp)=10+10+4+0/MSB, P2(cheap)=0+0+21+33/MSB, O_RESC0=4/stage
    # Token order per row: ds_loads first (K_Mx/V_Mx), then P2/O_RESC0.
    # Each O_RESC0 = arith.mulf(v8f32) = 4 v_pk_mul_f32 = 4 cy.
    # 4 independently-placeable tokens per stage (1 per MSB closure).
    sched = [
        # ---- Stage 0 (G0): K+TDM, exp=10/MSB ----
        [TDM, TDM, P2_M0],  # G0w00
        [K_M0, K_M0, K_M0, O_RESC0],  # G0w01
        [K_M0, K_M0, K_M0, P2_M0, P2_M0],  # G0w02
        [K_M1, K_M1, K_M1, P2_M1, P2_M1],  # G0w03
        [K_M1, K_M1, K_M1, P2_M1, P2_M1],  # G0w04
        [K_M2, K_M2, K_M2, P2_M2, P2_M2],  # G0w05
        [K_M2, K_M2, K_M2, P2_M2, P2_M0],  # G0w06
        [K_M3, K_M3, K_M3, P2_M3, P2_M3],  # G0w07
        [K_M3, K_M3, K_M3, P2_M3],  # G0w08
        [P2_M0, P2_M0],  # G0w09
        [P2_M0, P2_M0],  # G0w10
        [P2_M1, P2_M1],  # G0w11
        [P2_M1, P2_M1],  # G0w12
        [P2_M1, O_RESC0],  # G0w13
        [P2_M1, P2_M0],  # G0w14
        [P2_M2, P2_M3],  # G0w15
        [P2_M2, O_RESC0],  # G0w16
        [P2_M2, O_RESC0],  # G0w17
        [P2_M0, P2_M0],  # G0w18
        [P2_M0, P2_M0],  # G0w19
        [P2_M0, P2_M0],  # G0w20
        [P2_M0, P2_M0],  # G0w21
        [P2_M0, P2_M0],  # G0w22
        [P2_M0, P2_M2],  # G0w23
        # ---- Stage 1 (G1): K+TDM, exp=10/MSB ----
        [TDM, TDM, P2_M2],  # G1w00
        [K_M0, K_M0, K_M0, O_RESC0],  # G1w01
        [K_M0, K_M0, K_M0, P2_M0],  # G1w02
        [K_M1, K_M1, K_M1, P2_M1, P2_M1],  # G1w03
        [K_M1, K_M1, K_M1, P2_M1, P2_M1],  # G1w04
        [K_M2, K_M2, K_M2, P2_M2, P2_M2],  # G1w05
        [K_M2, K_M2, K_M2, P2_M2, P2_M2],  # G1w06
        [K_M3, K_M3, K_M3, P2_M3],  # G1w07
        [K_M3, K_M3, K_M3, P2_M3],  # G1w08
        [P2_M3, P2_M3],  # G1w09
        [P2_M3, P2_M3],  # G1w10
        [P2_M3, P2_M3],  # G1w11
        [P2_M1, P2_M1],  # G1w12
        [P2_M1, P2_M1],  # G1w13
        [P2_M1, P2_M2],  # G1w14
        [P2_M1, O_RESC0],  # G1w15
        [P2_M2, O_RESC0],  # G1w16
        [P2_M2, O_RESC0],  # G1w17
        [P2_M2, P2_M2],  # G1w18
        [P2_M2, P2_M2],  # G1w19
        [P2_M3, P2_M3],  # G1w20
        [P2_M3, P2_M3],  # G1w21
        [P2_M3, P2_M3],  # G1w22
        [P2_M3, P2_M0],  # G1w23
        # ---- Stage 2 (G2): K, exp=4/MSB + cheap=21/MSB ----
        [K_M0, K_M0, K_M0, P2_M0, P2_M3],  # G2w00
        [K_M0, K_M0, K_M0, P2_M0, P2_M2],  # G2w01
        [K_M1, K_M1, K_M1, P2_M1, P2_M1],  # G2w02
        [K_M1, K_M1, K_M1, P2_M1, P2_M1],  # G2w03
        [K_M2, K_M2, K_M2, P2_M2, P2_M2],  # G2w04
        [K_M2, K_M2, K_M2, P2_M2, P2_M2],  # G2w05
        [K_M3, K_M3, K_M3, P2_M3, P2_M3],  # G2w06
        [K_M3, K_M3, K_M3, P2_M3, P2_M3],  # G2w07
        [P2_M0, P2_M0, P2_M0, P2_M0, P2_M0, P2_M0],  # G2w08
        [P2_M0, P2_M0, P2_M0, P2_M0, P2_M0, P2_M0, P2_M0],  # G2w09
        [P2_M0, P2_M0, P2_M0, P2_M0, P2_M0, P2_M0, P2_M0],  # G2w10
        [P2_M1, P2_M1, P2_M1, P2_M1, P2_M1, P2_M1, P2_M1],  # G2w11
        [P2_M1, P2_M1, P2_M1, P2_M1, P2_M1, P2_M1, P2_M1],  # G2w12
        [P2_M1, P2_M1, P2_M1, P2_M1, P2_M1, P2_M1, P2_M1],  # G2w13
        [P2_M2, P2_M2, O_RESC0],  # G2w14
        [P2_M2, P2_M2, O_RESC0],  # G2w15
        [P2_M2, P2_M2, O_RESC0],  # G2w16
        [P2_M3, P2_M3, O_RESC0],  # G2w17
        [P2_M2, P2_M2, P2_M2, P2_M2, P2_M2, P2_M2],  # G2w18
        [P2_M2, P2_M2, P2_M2, P2_M2, P2_M2, P2_M2],  # G2w19
        [P2_M2, P2_M2, P2_M2, P2_M3, P2_M3, P2_M3],  # G2w20
        [P2_M3, P2_M3, P2_M3, P2_M3, P2_M3, P2_M3],  # G2w21
        [P2_M3, P2_M3, P2_M3, P2_M3, P2_M3, P2_M3],  # G2w22
        [P2_M0, P2_M0, P2_M0, P2_M0, P2_M3, P2_M3],  # G2w23
        # ---- Stage 3 (G3): V, cheap=33/MSB ----
        [V_M0, V_M0, P2_M0, P2_M0],  # G3w00
        [V_M0, V_M0, P2_M0, P2_M0],  # G3w01
        [V_M1, V_M1, P2_M1, P2_M1],  # G3w02
        [V_M1, V_M1, P2_M1, P2_M1, P2_M2],  # G3w03
        [V_M2, V_M2, P2_M2, P2_M2],  # G3w04
        [V_M2, V_M2, P2_M2, P2_M2],  # G3w05
        [V_M3, V_M3, P2_M3, P2_M3],  # G3w06
        [V_M3, V_M3, P2_M3, P2_M1],  # G3w07
        [P2_M1, P2_M1, P2_M3],  # G3w08
        [P2_M1, P2_M0],  # G3w09
        [P2_M1, P2_M3],  # G3w10
        [P2_M1, P2_M0, P2_M3],  # G3w11
        [P2_M2, P2_M3],  # G3w12
        [P2_M2, P2_M2],  # G3w13
        [P2_M3, P2_M3],  # G3w14
        [P2_M3, P2_M3, P2_M1],  # G3w15
        [O_RESC0, P2_M2],  # G3w16
        [O_RESC0, P2_M2],  # G3w17
        [O_RESC0, P2_M2, P2_M0],  # G3w18
        [O_RESC0, P2_M2, P2_M3],  # G3w19
        [P2_M0, P2_M3, P2_M1],  # G3w20
        [P2_M0],  # G3w21
        [P2_M3],  # G3w22
        [],  # G3w23
    ]
    assert len(sched) == 96
    return sched


# ---------------------------------------------------------------------------
# GEMM2 schedule: 64 rows
# ---------------------------------------------------------------------------
# Strict ordering: P0 (max) → P1 (cross-MSB delta) → P2 (pkfma) → EXP (pair_exp)
#
# Key dependency: PART2 setup op[2] reads delta[b] = PART1 output.
# → P1 must finish BEFORE any P2 token fires.
# → P1 placed in stage 1 post-load (after all P0 overflow), ensuring
#   PART1 completes before stage 2 where P2 companions start. ✓
#
# Stage 0 (V+TDM): P0=10/MSB
# Stage 1 (V+TDM): P0=12/MSB + P1=8 (P1 after all P0 overflow in post-load)
# Stage 2 (V):     ALL P2=24/MSB (load-WMMAs give 8/MSB; post-load 64 tokens, row_cap=8)
# Stage 3 (K):     ALL EXP=8/MSB at 2/WMMA
# Totals: P0=22/MSB ✓  P1=8 ✓  P2=24/MSB ✓  EXP=8/MSB=32 total ✓


def build_gemm2_schedule() -> List[List[int]]:
    sched = []
    # All 4 stages as direct flat table — edit rows to adjust load distribution.
    # Budgets: P0=10+12/MSB, P1=8, P2=24/MSB, EXP=8/MSB
    # Token order per row: ds_loads first (V_Mx/K_Mx), then ALU (P0/P1/P2/EXP).
    sched.extend(
        [
            # ---- Stage 0 (G0): V+TDM, P0=10/MSB ----
            [TDM, TDM, P0_M0, P0_M0, P0_M0],  # G0w00
            [V_M0, V_M0, P0_M0, P0_M0, P0_M0],  # G0w01
            [V_M0, V_M0, P0_M0, P0_M0, P0_M0],  # G0w02
            [V_M1, V_M1, P0_M1, P0_M1, P0_M1, P0_M1],  # G0w03
            [V_M1, V_M1, P0_M1, P0_M1, P0_M1, P0_M1],  # G0w04
            [V_M2, V_M2, P0_M2, P0_M2, P0_M2, P0_M2],  # G0w05
            [V_M2, V_M2, P0_M2, P0_M2, P0_M2, P0_M2],  # G0w06
            [V_M3, V_M3, P0_M3, P0_M3, P0_M3, P0_M3],  # G0w07
            [V_M3, V_M3, P0_M3, P0_M3, P0_M3, P0_M3],  # G0w08
            [P0_M0, P0_M2, P0_M2, P0_M2, P0_M3],  # G0w09
            [P0_M0, P0_M1, P0_M1, P0_M3],  # G0w10
            [P0_M0, P0_M0, P0_M0, P0_M0, P0_M1],  # G0w11
            [P0_M0, P0_M1, P0_M1, P0_M2, P0_M2],  # G0w12
            [P0_M1, P0_M2, P0_M2, P0_M3, P0_M3],  # G0w13
            [P0_M2, P0_M3, P0_M3, P0_M1, P0_M1],  # G0w14
            [P0_M3, P0_M3],  # G0w15
            # ---- Stage 1 (G1): V+TDM, P0=12/MSB + P1=8 ----
            [TDM, TDM, P0_M0],  # G1w00
            [V_M0, V_M0, P0_M0, P0_M1, P0_M2, P0_M3],  # G1w01
            [V_M0, V_M0, P0_M0, P0_M1, P0_M2, P0_M3],  # G1w02
            [V_M1, V_M1, P0_M1, P0_M0, P0_M2, P0_M3],  # G1w03
            [V_M1, V_M1, P0_M1, P0_M0],  # G1w04
            [V_M2, V_M2, P0_M2, P0_M2, P0_M1],  # G1w05
            [V_M2, V_M2, P0_M2, P0_M0, P0_M1],  # G1w06
            [V_M3, V_M3, P0_M3],  # G1w07
            [V_M3, V_M3, P0_M3, P0_M3],  # G1w08
            [P1, P1, P1, P1],  # G1w09
            [P1, P1, P1, P1],  # G1w10
            [P2_M1, P2_M1, P2_M1, P2_M1, P2_M2, P2_M2, P2_M2, P2_M2],  # G1w11
            [P2_M0, P2_M0, P2_M0, P2_M0, P2_M3, P2_M3, P2_M3, P2_M3],  # G1w12
            [P2_M0, P2_M1, P2_M2, P2_M3],  # G1w13
            [P2_M0, P2_M1, P2_M2, P2_M3, P2_M0, P2_M1, P2_M2, P2_M3],  # G1w14
            [P2_M0, P2_M1, P2_M2, P2_M3],  # G1w15
            # ---- Stage 2 (G2): V, P2=24/MSB ----
            [V_M0, V_M0, P2_M0, P2_M0, P2_M0],  # G2w00
            [V_M0, V_M0, P2_M0, P2_M0, P2_M0],  # G2w01
            [V_M1, V_M1, P2_M1, P2_M2, P2_M2],  # G2w02
            [V_M1, V_M1, P2_M1, P2_M2, P2_M2],  # G2w03
            [V_M2, V_M2, P2_M2, P2_M1, P2_M1],  # G2w04
            [V_M2, V_M2, P2_M2, P2_M1, P2_M1],  # G2w05
            [V_M3, V_M3, P2_M3, P2_M3, P2_M3, P2_M3],  # G2w06
            [V_M3, V_M3, P2_M3, P2_M3, P2_M3],  # G2w07
            [P2_M0, P2_M0, P2_M0, P2_M0, P2_M0, P2_M0],  # G2w08
            [P2_M0, P2_M0, P2_M0, P2_M1],  # G2w09
            [P2_M1, P2_M1, P2_M1, P2_M1, P2_M1, P2_M1],  # G2w10
            [P2_M1, P2_M1, P2_M1, P2_M2, P2_M2],  # G2w11
            [P2_M2, P2_M2, P2_M2, P2_M2, P2_M2, P2_M2],  # G2w12
            [P2_M3, P2_M3, P2_M3],  # G2w13
            [P2_M3, P2_M3, P2_M3, P2_M3],  # G2w14
            [P2_M0, P2_M0, P2_M1, P2_M1],  # G2w15
            # ---- Stage 3 (G3): K, EXP=8/MSB ----
            [K_M0, K_M0, K_M0, EXP_M0, EXP_M0],  # G3w00
            [K_M0, K_M0, K_M0, EXP_M0],  # G3w01
            [K_M1, K_M1, K_M1, EXP_M1, EXP_M1],  # G3w02
            [K_M1, K_M1, K_M1, EXP_M1, EXP_M1],  # G3w03
            [K_M2, K_M2, K_M2, P2_M2, EXP_M2],  # G3w04
            [K_M2, K_M2, K_M2, P2_M2, EXP_M2],  # G3w05
            [K_M3, K_M3, K_M3, P2_M3, EXP_M3],  # G3w06
            [K_M3, K_M3, K_M3, P2_M3, EXP_M3],  # G3w07
            [EXP_M0, EXP_M0, EXP_M0],  # G3w08
            [EXP_M0, EXP_M0, EXP_M1],  # G3w09
            [EXP_M1, EXP_M1, EXP_M1],  # G3w10
            [EXP_M3, EXP_M3, EXP_M3],  # G3w11
            [EXP_M2, EXP_M2, EXP_M2],  # G3w12
            [EXP_M2, EXP_M2, EXP_M2],  # G3w13
            [EXP_M3, EXP_M3, EXP_M3],  # G3w14
            [],  # G3w15
        ]
    )
    assert len(sched) == 64
    return sched


# ---------------------------------------------------------------------------
# Pre-built tables
# ---------------------------------------------------------------------------
GEMM1_SCHEDULE: List[List[int]] = build_gemm1_schedule()
GEMM2_SCHEDULE: List[List[int]] = build_gemm2_schedule()


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------
def g1_row_idx(stage: int, wmma: int) -> int:
    return stage * _GEMM_INST_COUNT + wmma


def g2_row_idx(stage: int, wmma: int) -> int:
    return stage * _PV_GEMM_INST_COUNT + wmma


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------
_TOKEN_NAMES = {
    0: "P0m0",
    1: "P0m1",
    2: "P0m2",
    3: "P0m3",
    4: "P1",
    5: "P2m0",
    6: "P2m1",
    7: "P2m2",
    8: "P2m3",  # pkfma/setup (1cy)
    9: "K_m0",
    10: "K_m1",
    11: "K_m2",
    12: "K_m3",
    13: "V_m0",
    14: "V_m1",
    15: "V_m2",
    16: "V_m3",
    17: "ORS0",
    18: "TDM",
    19: "EXPm0",
    20: "EXPm1",
    21: "EXPm2",
    22: "EXPm3",  # pair_exp (3cy!)
}


def print_schedule(schedule, name, wmma_per_stage):
    print(f"\n{'='*70}")
    print(f"  {name}  ({len(schedule)} rows)")
    print(f"{'='*70}")
    for i, row in enumerate(schedule):
        stage = i // wmma_per_stage
        wmma = i % wmma_per_stage
        counts = {}
        for t in row:
            counts[t] = counts.get(t, 0) + 1
        summary = " ".join(
            f"{_TOKEN_NAMES.get(t,str(t))}:{n}" for t, n in sorted(counts.items())
        )
        names = [_TOKEN_NAMES.get(t, str(t)) for t in row]
        print(f"  G{stage}w{wmma:02d} [{i:3d}]: {names!s:<52} {summary}")


def validate_schedule(schedule, name):
    for i, row in enumerate(schedule):
        for t in row:
            assert 0 <= t <= 22, f"{name}[{i}] invalid token {t}"  # 0-18 + EXP 19-22


# ---------------------------------------------------------------------------
# CSV schedule tool
# ---------------------------------------------------------------------------
# Workflow:
#   1. Export current schedule:
#      save_schedule_to_csv(
#          GEMM1_SCHEDULE, GEMM2_SCHEDULE, 'sched.csv')
#   2. Edit sched.csv manually (token names or integers, space-separated per row)
#   3. Reload: g1, g2 = parse_schedule_from_csv('sched.csv')
#   4. Validate: validate_schedule(g1, 'GEMM1'); validate_schedule(g2, 'GEMM2')
#   5. Use g1/g2 instead of the module-level GEMM1_SCHEDULE/GEMM2_SCHEDULE
#
# CSV format (no header required, comments with #):
#   gemm,stage,wmma,tokens
#   1,0,0,TDM TDM
#   1,0,1,K_m0 K_m0 K_m0 P2m0 P2m0
#   2,0,0,TDM TDM
#   2,0,1,V_m0 V_m0 V_m0 P2m0 P2m0 P2m0
#
# Token names: P0m0..P0m3, P1, P2m0..P2m3, K_m0..K_m3, V_m0..V_m3,
#              ORSC, TDM, EXPm0..EXPm3   (or raw integers 0-22)

_TOKEN_NAME_TO_ID: dict = {v: k for k, v in _TOKEN_NAMES.items()}


def save_schedule_to_csv(
    g1_schedule: List[List[int]],
    g2_schedule: List[List[int]],
    filename: str,
) -> None:
    """Export GEMM1+GEMM2 schedules to a CSV file for manual editing."""
    with open(filename, "w", newline="") as f:
        f.write(
            "# FMHA schedule -- edit tokens column,"
            " then reload with"
            " parse_schedule_from_csv\n"
        )
        f.write(
            "# Token names: "
            + "  ".join(f"{v}={k}" for k, v in sorted(_TOKEN_NAMES.items()))
            + "\n"
        )
        f.write("gemm,stage,wmma,tokens\n")
        for i, row in enumerate(g1_schedule):
            stage = i // _GEMM_INST_COUNT
            wmma = i % _GEMM_INST_COUNT
            toks = " ".join(_TOKEN_NAMES.get(t, str(t)) for t in row)
            f.write(f"1,{stage},{wmma},{toks}\n")
        for i, row in enumerate(g2_schedule):
            stage = i // _PV_GEMM_INST_COUNT
            wmma = i % _PV_GEMM_INST_COUNT
            toks = " ".join(_TOKEN_NAMES.get(t, str(t)) for t in row)
            f.write(f"2,{stage},{wmma},{toks}\n")


def parse_schedule_from_csv(filename: str):
    """Parse a manually edited schedule CSV.

    Returns (g1_schedule, g2_schedule) as List[List[int]] (96 and 64 rows).
    Rows not present in the CSV are left empty ([]).
    Use validate_schedule() afterwards to catch token-range errors.
    """
    g1: List[List[int]] = [[] for _ in range(_GEMM_INST_COUNT * 4)]
    g2: List[List[int]] = [[] for _ in range(_PV_GEMM_INST_COUNT * 4)]

    with open(filename, newline="") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("gemm"):
                continue
            parts = line.split(",", 3)
            if len(parts) < 3:
                raise ValueError(
                    f"{filename}:{lineno}: expected "
                    f"gemm,stage,wmma[,tokens], "
                    f"got {line!r}"
                )
            gemm = int(parts[0])
            stage = int(parts[1])
            wmma = int(parts[2])
            toks_str = parts[3].strip() if len(parts) > 3 else ""

            tokens: List[int] = []
            for tok in toks_str.split():
                if not tok:
                    continue
                if tok.lstrip("-").isdigit():
                    tokens.append(int(tok))
                elif tok in _TOKEN_NAME_TO_ID:
                    tokens.append(_TOKEN_NAME_TO_ID[tok])
                else:
                    raise ValueError(
                        f"{filename}:{lineno}: unknown token {tok!r} "
                        f"(known: {sorted(_TOKEN_NAME_TO_ID)})"
                    )

            if gemm == 1:
                idx = stage * _GEMM_INST_COUNT + wmma
                if not (0 <= idx < len(g1)):
                    raise ValueError(
                        f"{filename}:{lineno}: GEMM1 "
                        f"stage={stage} wmma={wmma} "
                        f"out of range"
                    )
                g1[idx] = tokens
            elif gemm == 2:
                idx = stage * _PV_GEMM_INST_COUNT + wmma
                if not (0 <= idx < len(g2)):
                    raise ValueError(
                        f"{filename}:{lineno}: GEMM2 "
                        f"stage={stage} wmma={wmma} "
                        f"out of range"
                    )
                g2[idx] = tokens
            else:
                raise ValueError(
                    f"{filename}:{lineno}: " f"gemm must be 1 or 2, got {gemm}"
                )

    return g1, g2


if __name__ == "__main__":
    validate_schedule(GEMM1_SCHEDULE, "GEMM1")
    validate_schedule(GEMM2_SCHEDULE, "GEMM2")
    print_schedule(GEMM1_SCHEDULE, "GEMM1_SCHEDULE", _GEMM_INST_COUNT)
    print_schedule(GEMM2_SCHEDULE, "GEMM2_SCHEDULE", _PV_GEMM_INST_COUNT)
    print(f"\nTotal rows: {len(GEMM1_SCHEDULE)+len(GEMM2_SCHEDULE)}")
