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


# ================= traits =================
p = r + "/device/layout/sk_traits.hpp"
t, nl = rd(p)
t = re.sub(
    r"constexpr int BM = static_cast<int>\(SM_MP\), BN = 64, BK = 128;[^\n]*",
    "constexpr int BM = static_cast<int>(SM_MP), BN = static_cast<int>(SM_GEMM_BN), BK = 128;      // r22: BN = 128 (SM_GEMM_BN128, default) or 64 (r21); BM == SM_MP; MT = BM/32 accumulator tiles per consumer wave",
    t,
    count=1,
)
t = re.sub(
    r"constexpr int SK = static_cast<int>\(SM_SK\), WG_PER_CU = 2;[^\n]*",
    "constexpr int SK = static_cast<int>(SM_SK), WG_PER_CU = SM_GEMM_BN128 ? 1 : 2;          // BN128: stage 32 KiB x PFK4 = 128 KiB LDS => 1 WG/CU (216 / 231 WGs <= 256 CUs, one round); BN64: 2 WG/CU",
    t,
    count=1,
)
t = re.sub(
    r"constexpr int B_STAGE_BYTES = BN \* BK;[^\n]*",
    "constexpr int B_STAGE_BYTES = BN * BK;               // BN64: 8192 (4 bands x 2048 B); BN128: 16384 (8 bands), physical shuffle order",
    t,
    count=1,
)
t = re.sub(
    r"constexpr int STAGE_BYTES = A_STAGE_BYTES \+ B_STAGE_BYTES;[^\n]*",
    "constexpr int STAGE_BYTES = A_STAGE_BYTES + B_STAGE_BYTES;   // BN128: 32768; BN64: 24576",
    t,
    count=1,
)
t = re.sub(
    r"constexpr int PFK = 3;[^\n]*",
    "constexpr int PFK = SM_GEMM_BN128 ? 4 : 3;           // r22 ring depth: BN128 PFK4 (128 KiB); r21 BN64 PFK3 (72 KiB); PFK-1 blocks in flight per WG",
    t,
    count=1,
)
t = re.sub(
    r"constexpr int LDS_BYTES = PFK \* STAGE_BYTES;[^\n]*",
    "constexpr int LDS_BYTES = PFK * STAGE_BYTES;         // BN128: 131072 (x1 WG/CU <= 163840); BN64: 73728 (x2)",
    t,
    count=1,
)
t = rep(
    t,
    "static_assert(EM == 1 && EN == 4 && EK == 1 && BM % 32 == 0 && BM >= 32);",
    "static_assert(EM == 1 && EN == BN / 16 && (EN == 4 || EN == 8) && EK == 1 && BM % 32 == 0 && BM >= 32);",
)
t = re.sub(
    r"constexpr int B_INSTR_PER_WAVE = 4;[^\n]*",
    "constexpr int B_INSTR_PER_WAVE = BN / 16;            // 1 KiB instructions per producer wave per K block: (BN/16 bands x 2 halves) / 2 producer waves = 4 (BN64) / 8 (BN128)",
    t,
    count=1,
)
t = rep(
    t,
    "opus::make_tuple(4, 2, 2, 2, 16), opus::make_tuple(2048, 1024, 512, 256, 16)",
    "opus::make_tuple(BN / 16, 2, 2, 2, 16), opus::make_tuple(2048, 1024, 512, 256, 16)",
)
wr(p, t, nl)
# ================= GEMM =================
p = r + "/device/sk_gemm_fp8.cu"
t, nl = rd(p)
t = rep(
    t,
    "    constexpr int MB = 4 + AQ;",
    "    constexpr int BQ = sks::B_INSTR_PER_WAVE;          // B vmem instructions per producer wave per K block: 4 (BN64) / 8 (BN128)\n    constexpr int MB = BQ + AQ;",
)
t = rep(t, "    int b_src[4], b_dst[4];", "    int b_src[BQ], b_dst[BQ];")
t = rep(
    t,
    "        for (int q = 0; q < 4; ++q) {\n            const int f = wave * 4 + q, nb = f >> 1, half = f & 1;",
    "        for (int q = 0; q < BQ; ++q) {\n            const int f = wave * BQ + q, nb = f >> 1, half = f & 1;",
)
t = rep(
    t,
    "            for (int q = 0; q < 4; ++q)\n",
    "            for (int q = 0; q < BQ; ++q)\n",
)
t = rep(
    t,
    "        int b_off0[4], b_off1[4];",
    "        int b_off0[sks::EN], b_off1[sks::EN];",
)
t = rep(
    t, "for (int nb = 0; nb < 4; ++nb) {", "for (int nb = 0; nb < sks::EN; ++nb) {", 2
)
t = rep(t, "                    U32x8 tt[4];", "                    U32x8 tt[sks::EN];")
t = rep(
    t,
    "__builtin_memcpy(&vb, &tt[0], 128);",
    "__builtin_memcpy(&vb, &tt[0], sizeof(vb));",
)
wr(p, t, nl)
# ================= driver =================
p = r + "/host/driver_sm.cpp"
t, nl = rd(p)
t = rep(
    t,
    "constexpr uint64_t RX_BYTES = 2ull * 4ull * SM_MP * SM_F * 2ull;        // [par][src][Mp][F] bf16",
    "constexpr uint64_t RX_BYTES = SM_RX_BYTES;                                  // [par][slot (3 remote ordinals / 4 ranks)][Mp][F] bf16 (r22: SM_RX3)",
)
wr(p, t, nl)
print("traits+gemm+driver ok")
