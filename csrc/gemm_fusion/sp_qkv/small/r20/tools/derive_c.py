import re
import sys

r = sys.argv[1]
p = r + "/device/ran_kernel.cu"
with open(p, encoding="utf-8", newline="") as _f:
    t = _f.read()
nl = "\r\n" if "\r\n" in t else "\n"
t = t.replace("\r\n", "\n")


def rep(old, new, cnt=1):
    global t
    assert t.count(old) == cnt, (f"count {t.count(old):d} != {cnt:d}: ") + old[:110]
    t = t.replace(old, new)


body0 = t.index("__device__ __forceinline__ void ran_body(")
# ---- decide(): r18/r15 shape guards gone (M <= 16 row-split refusal generalised: dead passes allowed, M >= 1) ----
rep(
    "#if SM_B_ROWSPLIT\n    if (M <= 16u) return 0x40u;                                                      // r18 R4: the row-split B WGs (row pass 1 = rows 16..31) need M > 16 (T128 builds); device-side guard besides the host refusal\n#endif\n",
    "",
)
rep(
    "#if SM_PUSH8\n    if (M > 16u) return 0x40u;                                                       // G-1: SM_PUSH8 handles one row pass per half (M <= 16 only); device-side guard in addition to the host refusal\n#endif\n",
    "",
)
# ---- row16 is the row inside a 256-thread half (the push WG has two halves) ----
rep(
    "    const int lane16 = tid & 15, row16 = tid >> 4;",
    "    const int lane16 = tid & 15, row16 = (tid & 255) >> 4;      // r20: row within the 256-thread half (push WGs have two halves = passes {0,1} / {2,3})",
)
# ---- IK owners: TWO WGs, owner o owns passes {2o, 2o+1} ----
rep(
    "    if (wgid == SM_T1_PUSH_WGS) {\n        // ======== IK OWNER WG (exactly ONE per source rank)",
    "    if (wgid >= SM_T1_PUSH_WGS && wgid < SM_T1_S0) {\n        // ======== IK OWNER WG (r20: TWO per source rank, owner o = wgid - SM_T1_PUSH_WGS owns row passes {2o, 2o+1}; each pushes ITS rows to the 3 destinations and releases once per destination => K = 8 + 2)",
)
rep(
    "        const uint32_t n0 = head_n0(0u, 19u);",
    "        const uint32_t pbase = 2u * (wgid - SM_T1_PUSH_WGS);                        // first global row pass of this IK owner\n        const uint32_t n0 = head_n0(0u, 19u);",
)
# ---- push: keep only the generalised-HALVES shape of the legacy body (24 x 512; half hf owns passes {2hf, 2hf+1}) ----
i = t.index(
    "#if SM_PUSH8 || SM_PUSH_HALVES\n        // ======== r15 PUSH WG, 512 threads"
)
j = t.index("#else\n\n        if (tid == 0) SM_STAMP(rec, 33);", i)
t = t[:i] + t[j + len("#else\n") :]
k = t.index("    } else if (wgid < SM_T1_B0) {")
assert t[:k].endswith("#endif\n")
t = t[: k - len("#endif\n")] + t[k:]
rep(
    "        const uint32_t g0 = t1r::g0(wgid), g1 = t1r::g0(wgid + 1u), nu = g1 - g0;",
    "        const uint32_t pbase = 2u * (static_cast<uint32_t>(tid) >> 8);                // r20: half hf owns passes {2hf, 2hf+1} (rows 32hf .. 32hf+31)\n        const uint32_t g0 = t1r::g0(wgid), g1 = t1r::g0(wgid + 1u), nu = g1 - g0;",
)
# ---- self: one head x two passes (2 row groups) ----
rep(
    "        const uint32_t h = wgid - SM_T1_S0;                                  // self heads 0..UPD-1 (IK self is fused by the IK owner WG)",
    "        const uint32_t sk_ = wgid - SM_T1_S0;                                // r20: self WG = (head h, row group sp): k = 2h + sp\n        const uint32_t h = sk_ / SM_SELF_SPLIT;                              // self heads 0..UPD-1 (IK self is fused by the IK owner WGs)\n        const uint32_t pbase = 2u * (sk_ % SM_SELF_SPLIT);                   // first global row pass of this self WG",
)
# ---- B: one head-group x ONE row pass; dead passes acquire but load/check/store nothing ----
a = t.index("#if SM_B_ROWSPLIT\n        const uint32_t h0 = (pair / SM_B_SPLIT) * NH;")
b = t.index("#define SM_BPASS_ON(pp) true\n#endif\n", a) + len(
    "#define SM_BPASS_ON(pp) true\n#endif\n"
)
t = (
    t[:a]
    + """        const uint32_t h0 = (pair / SM_B_SPLIT) * NH;                                // r20: pair = head-group * 4 + row pass
        const uint32_t rsp = pair % SM_B_SPLIT;                                      // this WG's row pass (0..3), WG-uniform: only rows 16*rsp .. 16*rsp+15
        const uint32_t pbase = 2u * (rsp >> 1);                                      // local pass dimension = {pbase, pbase+1}; exactly one of them is this WG's pass
        const bool live_pass = rsp < passes;                                         // R2.5: a dead pass (m <= 16*rsp) still performs its flag ACQUIRE but no Rx load, no tag check, no stores
#define SM_BPASS_ON(pp) ((pbase + (pp)) == rsp)
"""
    + t[b:]
)
old = '            if (ok) {\n            _Pragma("unroll") for (uint32_t kk = 0; kk < NH; ++kk) _Pragma("unroll") for (uint32_t pass = 0; pass < 2u; ++pass) {\n                const uint32_t rc = rowc('
rep(old, old.replace("if (ok) {", "if (ok && live_pass) {"), 2)
# ---- every row/pass expression inside the kernel body: global pass = pbase + local pass ----
head, tail = t[:body0], t[body0:]
n1 = len(re.findall(r"16u \* pass", tail))
tail = tail.replace("16u * pass", "16u * (pbase + pass)")
n2 = tail.count("if (pass < passes && SM_BPASS_ON(pass)) head_row(")
tail = tail.replace(
    "if (pass < passes && SM_BPASS_ON(pass)) head_row(",
    "if ((pbase + pass) < passes && SM_BPASS_ON(pass)) head_row(",
)
n3 = tail.count("if (pass < passes) head_row(")
tail = tail.replace(
    "if (pass < passes) head_row(", "if ((pbase + pass) < passes) head_row("
)
t = head + tail
print("row exprs", n1, "B head_row", n2, "other head_row", n3)
with open(p, "w", encoding="utf-8", newline="") as _f:
    _f.write(t.replace("\n", nl))
print("ran stage ok")
