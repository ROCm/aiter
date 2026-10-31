import re
import sys

r = sys.argv[1]


def rd(p):
    with open(p, encoding="utf-8", newline="") as _f:
        t = _f.read()
    nl = "\r\n" if "\r\n" in t else "\n"
    return t.replace("\r\n", "\n"), nl


def wr(p, t, nl):
    with open(p, "w", encoding="utf-8", newline="") as _f:
        _f.write(t.replace("\n", nl))


def rep(t, old, new, cnt=1):
    assert t.count(old) == cnt, (f"count {t.count(old):d} != {cnt:d}: ") + old[:100]
    return t.replace(old, new)


# ---- ABI: SM_PUSH_PAR is not built in r22 (reviewer A-9: 96 push WGs + B-quad blow residency; aggregation A-1..A-8 are for a later round) ----
p = r + "/interface/sm_abi_v3.h"
t, nl = rd(p)
t = rep(
    t,
    "#ifndef SM_PUSH_PAR\n#define SM_PUSH_PAR 0\n#endif",
    '#ifndef SM_PUSH_PAR\n#define SM_PUSH_PAR 0\n#endif\n#if SM_PUSH_PAR\n#error "SM_PUSH_PAR (96 push WGs + release aggregation) is NOT built in r22 (R14 review A-9: it blows residency with B-quad; A-1..A-8 apply when it is implemented)"\n#endif',
)
t = rep(
    t,
    "//  SM_PUSH_PAR (0, behind the D-9 recount): 96 push WGs x 256 threads (the 2 halves become separate WGs) with per-(XCC, destination) release AGGREGATION (agent-scope counter; the last arriver does the single system release): <= 3 releases per XCC, K = 8 + IK_WGS.",
    "//  SM_PUSH_PAR (0; NOT built in r22, #error if set): P-load parallelism via 96 push WGs + per-(XCC, destination) release aggregation, behind the reviewer's D-9 recount (R14 review A-1..A-9). The t1_roles.hpp table is already generic in PAR/AGG_N.",
)
t = t.replace(
    "#define SM_C_AGG(wu) (SM_C_FLAGS + 384u + 8u*(wu))   // r22 SM_PUSH_PAR: u64 release-aggregation counter per unit WG (24 words = bytes 640-831), zeroed with the control window by reset()\n",
    "",
)
wr(p, t, nl)
# ---- kernel ----
p = r + "/device/ran_kernel.cu"
t, nl = rd(p)
# LDS-free requirement of the quad + Rx ordinal helper
m = re.search(r"#define SM_HEAD_TAIL_SYNC[^\n]*\n", t)
assert m
t = (
    t[: m.end()]
    + """// r22 B-quad (R14 review B-3): the two 256-thread halves of a B WG run head_row CONCURRENTLY on different passes: head_row must not touch the per-WG LDS staging (s_row/s_part) and must contain no barrier.
// That holds exactly for SM_RMS_ORDER 3/4 with SM_AMAX_REG=1 (SM_LDS_LEGACY = 0, SM_HEAD_TAIL_SYNC = 0); tests/check_lds_protection.py (r11) proves the hazard model per configuration.
#if SM_B_QUAD && (SM_LDS_LEGACY || SM_HEAD_TAIL_SYNC)
#error "SM_B_QUAD needs a barrier-free LDS-free head_row (SM_RMS_ORDER 3/4, SM_AMAX_REG=1)"
#endif
"""
    + t[m.end() :]
)
rep_anchor = "SM_INL uint32_t rowc(uint32_t m, uint32_t Mm1)"
t = rep(
    t,
    rep_anchor,
    """// r22 R14 F1 (4) / review C-1: the ONE helper for the Rx source slot. A window of destination d holds a slot per REMOTE source: ordinal(s, d) = s < d ? s : s - 1 (3 slots), or the rank s (4 slots, SM_RX3 = 0).
// Writers (push, IK owner) use rx_ord(rank, d); the reader (B WG of destination `rank`) uses rx_ord(s, rank). Control-window indices (flags, tags) stay RANK-indexed (review C-3).
SM_INL int rx_ord(uint32_t s, uint32_t d) { return static_cast<int>(SM_RX3 ? (s < d ? s : s - 1u) : s); }
"""
    + rep_anchor,
)
# Rx call sites
n1 = t.count("sml::rx_view()(static_cast<int>(par), static_cast<int>(rank), ")
n2 = t.count("sml::rx_view()(static_cast<int>(par), static_cast<int>(s), ")
assert (n1, n2) == (2, 5), (n1, n2)
t = t.replace(
    "sml::rx_view()(static_cast<int>(par), static_cast<int>(rank), ",
    "sml::rx_view()(static_cast<int>(par), rx_ord(rank, d), ",
)
t = t.replace(
    "sml::rx_view()(static_cast<int>(par), static_cast<int>(s), ",
    "sml::rx_view()(static_cast<int>(par), rx_ord(s, rank), ",
)
# shared flag for B-4
t = rep(
    t,
    "    __shared__ uint32_t s_ok;\n",
    "    __shared__ uint32_t s_ok;\n    __shared__ uint32_t s_tagbad;                               // r22 B-quad (review B-4): ONE tag decision gates both 256-thread halves\n",
)
# retire guard (B-2)
a = "if (wgid >= SM_T1_PUSH_WGS && __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(tid) >> 6) >= 4u) return;"
assert t.count(a) == 1
t = t.replace(
    a,
    "if (wgid >= SM_T1_PUSH_WGS && (!SM_B_QUAD || wgid < SM_T1_B0) && __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(tid) >> 6) >= 4u) return;   // r22 B-2 (MANDATORY): the B-quad WGs (wgid >= B0) use ALL 8 waves (half 1 = passes 4qd+2..3) and must NOT retire waves 4-7; IK/self WGs still do",
)
t = rep(
    t,
    "    const int lane16 = tid & 15, row16 = (tid & 255) >> 4;",
    "#if SM_B_QUAD\n    if (tid == 0) s_tagbad = 0u;\n#endif\n    const int lane16 = tid & 15, row16 = (tid & 255) >> 4;",
)
# B header
a = t.index("        const uint32_t h0 = (pair / SM_B_SPLIT) * NH;")
b = t.index("#define SM_BPASS_ON(pp) ((pbase + (pp)) == rsp)\n") + len(
    "#define SM_BPASS_ON(pp) ((pbase + (pp)) == rsp)\n"
)
old = t[a:b]
new = (
    """#if SM_B_QUAD
        const uint32_t h0 = (pair / SM_B_SPLIT) * NH;                                // r22 B-quad: pair = head-group * 2 + quad
        const uint32_t qd = pair % SM_B_SPLIT;                                       // this WG's 4-pass quad {4qd .. 4qd+3}, WG-uniform
        const uint32_t pbase = 4u * qd + 2u * (static_cast<uint32_t>(tid) >> 8);     // half hf (tid >> 8) owns passes {4qd + 2hf, 4qd + 2hf + 1}
        const bool live_pass = (4u * qd) < passes;                                   // B-5: a DEAD quad (no live row) still performs its flag ACQUIRE but no Rx load, no tag check, no stores
#define SM_BPASS_ON(pp) true
#else
"""
    + old
    + """#endif
"""
)
t = t[:a] + new + t[b:]
# B-4: one tag decision for both halves
old_tag = """            if (__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(tg != E)) != 0u) {
                ok = false;
                if ((tid & 63) == 0) set_err(a, ka.err, rank, SM_ERRR_TAG_MISMATCH, E, tg, 0ull);
            }
"""
assert t.count(old_tag) == 2
new_tag = (
    """#if SM_B_QUAD
            if (__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(tg != E)) != 0u) {
                if ((tid & 63) == 0) { s_tagbad = 1u; set_err(a, ka.err, rank, SM_ERRR_TAG_MISMATCH, E, tg, 0ull); }
            }
            mori::cco::impl::syncThreads();                                          // B-4: ONE decision gates BOTH halves (all 8 waves reach this barrier: ok and live_pass are WG-uniform)
            if (s_tagbad != 0u) ok = false;
#else
"""
    + old_tag
    + """#endif
"""
)
t = t.replace(old_tag, new_tag)
wr(p, t, nl)
# ---- layouts: 12-pair bijection (review C-1) ----
p = r + "/device/layout/ran_layouts.hpp"
t, nl = rd(p)
t = t.rstrip("\n")
assert t.endswith("}")
t = (
    t[:-1]
    + """// review C-1: the ordinal map must be a bijection {s != d} -> {0, 1, 2} for EVERY destination d (12 ordered pairs); mirrors rx_ord() in ran_kernel.cu (RX3) and the identity for 4 slots.
constexpr unsigned rxord(unsigned s, unsigned d) { return SM_RX3 ? (s < d ? s : s - 1u) : s; }
constexpr bool rxord_bijective() {
    for (unsigned d = 0; d < 4u; ++d) { unsigned seen = 0u; for (unsigned s = 0; s < 4u; ++s) { if (s == d) continue; const unsigned o = rxord(s, d); if (o >= SM_RX_SLOTS || (seen & (1u << o))) return false; seen |= 1u << o; } if (SM_RX3 && seen != 7u) return false; }
    return true;
}
static_assert(rxord_bijective(), "C-1: rx_ord(s, d) is injective into the Rx slots for all 12 (s, d) pairs (RX3: a bijection onto {0,1,2})");
}
"""
)
wr(p, t, nl)
print("kernel ok")
