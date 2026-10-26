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


# ---------- traits ----------
p = r + "/device/layout/sk_traits.hpp"
t, nl = rd(p)
t = rep(
    t,
    "constexpr int THREADS = 256, PRODUCER_WAVES = 2, CONSUMER_WAVES = 2;",
    "constexpr int THREADS = static_cast<int>(SM_GEMM_THREADS), PRODUCER_WAVES = 2, CONSUMER_WAVES = (SM_MP > 128u) ? 4 : 2;   // r23 BM256: 2 producer + 4 consumer waves (wave cw: half hi = cw >> 1 owns rows 128*hi.., cw & 1 selects the 16-row group inside each 32-row tile)",
)
t = rep(
    t,
    "constexpr int PFK = SM_GEMM_BN128 ? 4 : 3;",
    "constexpr int PFK = (SM_MP > 128u) ? 3 : (SM_GEMM_BN128 ? 4 : 3);",
)
t = rep(
    t,
    'static_assert(PFK >= 3 && PFK - 2 <= 4, "wait_newer handles newer in [0,4]");',
    'static_assert(PFK >= 3 && PFK - 2 <= 4, "wait_newer handles newer in [0,4]");\nstatic_assert(THREADS == 64 * (PRODUCER_WAVES + CONSUMER_WAVES) && LDS_BYTES == PFK * STAGE_BYTES, "BM256: 6 waves, PFK3 stage 48 KiB = 144 KiB LDS at 1 WG/CU");',
)
wr(p, t, nl)
# ---------- GEMM ----------
p = r + "/device/sk_gemm_fp8.cu"
t, nl = rd(p)
for k in (1, 2, 3, 4):
    t = rep(
        t,
        f"s_waitcnt_vmcnt(number<{k:d} * MB>{{}})",
        f"wn<MB, {k:d}>()",
    )
t = rep(
    t,
    "template<int MB> OPUS_D void wait_newer(int newer) {",
    """// vmcnt immediate (6 bits, max 63): only newer in [0, PFK-2] is ever executed; a constant above 63 is clamped so the dead switch arms still compile (the static_assert in wait_newer proves the live ones fit).
template<int MB, int K> OPUS_D void wn() { if constexpr (K * MB <= 63) s_waitcnt_vmcnt(number<K * MB>{}); else s_waitcnt_vmcnt(number<63>{}); }
template<int MB> OPUS_D void wait_newer(int newer) {
    static_assert((sks::PFK - 2) * MB <= 63, "live vmcnt immediates must fit 6 bits: (PFK-2) * (BQ + AQ) <= 63");""",
)
t = rep(
    t,
    "    constexpr int MT = (AQ + 1) / 2;",
    "    constexpr int MT = (AQ > 8) ? 4 : (AQ + 1) / 2;   // r23: 32-row accumulator tiles per consumer wave (<= 4); rows of wave cw: 128*(cw>>1) + 32*t + 16*(cw&1) + 0..15",
)
t = rep(
    t,
    "        const int cw = wave - sks::PRODUCER_WAVES;",
    "        const int cw_all = wave - sks::PRODUCER_WAVES;                           // 0..CONSUMER_WAVES-1\n        const int hi = cw_all >> 1, cw = cw_all & 1;                              // r23: hi selects the 128-row half (0 at MP <= 128), cw the 16-row group of each 32-row tile",
)
t = rep(
    t,
    "            const int a_row = t * 32 + cw * 16 + l16;",
    "            const int a_row = hi * 128 + t * 32 + cw * 16 + l16;",
)
t = rep(t, "            if (cw < AQ) {", "            if (hi * 8 + cw < AQ) {")
t = rep(
    t,
    "                    if (2 * t + cw < AQ) {",
    "                    if (hi * 8 + 2 * t + cw < AQ) {",
)
t = rep(
    t,
    "                    if (2 * t + cw < AQ) accum[t] =",
    "                    if (hi * 8 + 2 * t + cw < AQ) accum[t] =",
)
t = rep(
    t,
    "                const int m = t * 32 + u_cm(y), n = col_base + u_cn(y);",
    "                const int m = hi * 128 + t * 32 + u_cm(y), n = col_base + u_cn(y);",
)
t = rep(
    t,
    'extern "C" __global__ __launch_bounds__(256)',
    'extern "C" __global__ __launch_bounds__(SM_GEMM_THREADS)',
)
i = t.index("    switch ((args.M + 15u) / 16u) {")
j = t.index("    }", i) + len("    }")
cases = "".join(
    f"        case {k:d}u: smg::gemm_body<{k:d}>(args, stage); break;\n"
    for k in range(1, 16)
)
t = (
    t[:i]
    + "    switch ((args.M + 15u) / 16u) {                                                       // WG-uniform: AQ = ceil(M/16) in 1..16 (M <= SM_MP = 256)\n"
    + cases
    + "        default: smg::gemm_body<16>(args, stage); break;\n    }"
    + t[j:]
)
wr(p, t, nl)
print("gemm ok")
