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


# ---------- ABI ----------
p = r + "/interface/sm_abi_v3.h"
t, nl = rd(p)
t = re.sub(
    r"#define SM_MP        128u[^\n]*",
    "#define SM_MP        256u           // r23 (R14 card MP256 v1): padded rows per source (live rows M <= 256: local m 129..255, decode T = 516..1020); P / Rx / stamps are sized by it",
    t,
    count=1,
)
t = re.sub(
    r"#define SM_NPASS     \(SM_MP / 16u\)[^\n]*",
    """#define SM_ITERS     (SM_MP / 128u) // r23: every RAN role walks the rows in SM_ITERS chunks of 128 rows; role counts below are those of one 128-row chunk (r22): global pass = chunk-pass + 8 * iteration
#define SM_NPASS     (128u / 16u)   // 16-row passes per 128-row chunk (8); push halves / IK owners / self WGs own 2 per iteration, B WGs (octo) own a 4-pass quad per iteration; the push role has NPASS/4 row groups of 24 unit WGs
#define SM_NPASS_TOT (SM_MP / 16u)  // 16-row passes per source in total (16 at MP256)""",
    t,
    count=1,
)
t = rep(
    t,
    "#define SM_GEMM_BN          (SM_GEMM_BN128 ? 128u : 64u)",
    '#define SM_GEMM_BN          (SM_GEMM_BN128 ? 128u : 64u)\n#define SM_GEMM_THREADS     (SM_MP > 128u ? 384u : 256u)   // 2 producer waves + 2 (MP <= 128) / 4 (MP256, BM256: 4 consumer waves x MT4) consumer waves\n#if SM_MP > 128u && !(SM_RX3 && SM_B_QUAD && SM_GEMM_BN128)\n#error "MP256 needs SM_RX3=1 (4 slots would need 9.4 MB > 8 MiB), SM_B_QUAD=1 and SM_GEMM_BN128=1"\n#endif',
)
wr(p, t, nl)
# ---------- roles ----------
p = r + "/device/layout/t1_roles.hpp"
t, nl = rd(p)
t = rep(
    t,
    'static_assert(SM_NPASS == 8u && SM_MP == 128u, "r22 is the MP128 build: 8 row passes of 16 rows");',
    """static_assert(SM_NPASS == 8u && SM_MP == 128u * SM_ITERS && SM_ITERS >= 1u, "role counts are those of one 128-row chunk (8 passes); SM_ITERS = MP/128 chunks are walked by each WG");
static_assert(SM_NPASS_TOT == SM_NPASS * SM_ITERS, "total passes = chunk passes x iterations");
// r23 (card D2): MP256 keeps the r22 role counts (grid 228 / 248); every WG does SM_ITERS = 2 iterations: push 2 passes x 2 halves x 2 iterations = 8 passes per row group; IK owner 2 x 2; self 2 x 2; B octo = quad x 2 iterations.
// The push / IK release happens ONCE after the last iteration, so K = releasers + IK_WGS is unchanged (16 / 20).
static_assert(SM_ITERS != 2u || (SM_MP == 256u && SM_B_QUAD && !SM_PUSH_PAR), "MP256 v1: B octo (B_QUAD), 48 push WGs, 2 iterations");""",
)
wr(p, t, nl)
# ---------- layouts ----------
p = r + "/device/layout/ran_layouts.hpp"
t, nl = rd(p)
t = rep(
    t,
    'static_assert(SM_MP == 128u, "r21 layouts are written for MP == 128");',
    'static_assert(SM_MP == 256u, "r23 layouts are written for MP == 256");',
)
a = t.index("static_assert(SM_RX_BYTES <= 8ull * 1024ull * 1024ull")
b = t.index("\n", t.index('"X-5 recomputed for the Rx slot count");'))
t = (
    t[:a]
    + """static_assert(SM_RX3 && SM_RX_BYTES > 4ull * 1024ull * 1024ull && SM_RX_BYTES <= 8ull * 1024ull * 1024ull, "R14 card MP256: 3 slots = 7,077,888 B (dense) / 7,864,320 B (indexer, reuse): needs the 8 MiB view and fits it (4 slots would be 9.4 / 10.5 MB)");
static_assert(SM_DENSE ? (SM_RX_BYTES == 7077888ull) : (SM_RX_BYTES == 7864320ull), "X-5 recomputed for MP256 x 3 slots");"""
    + t[b:]
)
wr(p, t, nl)
# ---------- driver ----------
p = r + "/host/driver_sm.cpp"
t, nl = rd(p)
t = rep(
    t,
    "hipModuleLaunchKernel(c->fg, SM_GEMM_NTILES, SM_SK, 1, 256, 1, 1, 0, s, ga, nullptr)",
    "hipModuleLaunchKernel(c->fg, SM_GEMM_NTILES, SM_SK, 1, SM_GEMM_THREADS, 1, 1, 0, s, ga, nullptr)",
)
wr(p, t, nl)
print("abi/roles/layouts/driver ok")
