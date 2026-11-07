import sys

r20, r21 = sys.argv[1], sys.argv[2]
# fix labels in the roles model
p = r21 + "/tests/check_r21_roles.py"
with open(p, encoding="utf-8") as _f:
    t = _f.read()
for a, b in (
    ("grid %d = push 24 + IK %d", "grid %d = push 48 + IK %d"),
    ("m in {33,40,48,49,63,64}", "m in {65,80,96,97,113,127,128}"),
    (
        "pushed exactly once by the two halves",
        "pushed exactly once (2 row groups of 24 unit WGs x 2 halves x 2 passes)",
    ),
):
    assert t.count(a) == 1, a
    t = t.replace(a, b)
with open(p, "w", encoding="utf-8", newline="") as _f:
    _f.write(t)
# static checker
with open(r20 + "/tests/check_r20_static.py", encoding="utf-8") as _f:
    t = _f.read()


def rep(old, new, cnt=1):
    global t
    assert t.count(old) == cnt, (t.count(old), old[:80])
    t = t.replace(old, new)


rep('"""r20 static source checks', '"""r21 (MP128) static source checks')
rep(
    "3 remaining release sites (IK owner x destinations, push, test-only ADD_BEFORE_RETIRE)",
    "3 remaining release sites (IK owners x destinations, push, test-only ADD_BEFORE_RETIRE)",
)
rep(
    "need('(static_cast<uint32_t>(tid) >> 8)' in body and 'row16 = (tid & 255) >> 4' in body, 'push halves: hf = tid >> 8, row inside the half')",
    "need('4u * (wgid / t1r::UNIT_WGS) + 2u * (static_cast<uint32_t>(tid) >> 8)' in body and 'row16 = (tid & 255) >> 4' in body and 'const uint32_t wu_ = wgid % t1r::UNIT_WGS;' in body, 'push: row group = wgid / 24 owns passes {4g..4g+3}, half hf = tid >> 8 owns {4g+2hf, 4g+2hf+1}, unit WG = wgid % 24')",
)
rep(
    "'#error \"r20 requires SM_RX_BYPASS=1' in abi",
    "'#error \"r20 requires SM_RX_BYPASS=1' in abi",
)
rep(
    "need('#define SM_MP        64u' in abi and '#define SM_DBG_RAN_MAXWG   512u' in abi and '#define SM_T1_IK_WGS        2u' in abi, 'SM_MP 64, MAXWG 512, IK_WGS 2')",
    "need('#define SM_MP        128u' in abi and '#define SM_DBG_RAN_MAXWG   1024u' in abi and '#define SM_T1_IK_WGS        (SM_NPASS / 2u)' in abi and '#define SM_T1_PUSH_WGS      (24u * (SM_NPASS / 4u))' in abi, 'SM_MP 128, MAXWG 1024, IK_WGS NPASS/2, PUSH_WGS 24 x NPASS/4')",
)
rep(
    "need('kcount(0) == 10u' in rl and 'IK_WGS * 2u == SM_NPASS' in rl, 'IK K = 8 + IK_WGS static_assert (indexer 10, dense/reuse 8)')",
    "need('kcount(0) == 20u' in rl and 'kcount(0) == 16u' in rl and 'IK_WGS * 2u == SM_NPASS' in rl and '8u * ROWGROUPS + SM_T1_IK_WGS' in rl, 'K = 8*ROWGROUPS + IK_WGS static_assert (indexer 20, dense/reuse 16)')",
)
rep(
    "need('static_assert(SM_RX_BYTES <= 4ull * 1024ull * 1024ull' in lay and 'SM_RX_BYTES == 2359296ull' in lay and",
    "need('SM_RX_BYTES > 4ull * 1024ull * 1024ull && SM_RX_BYTES <= 8ull * 1024ull * 1024ull' in lay and 'SM_RX_BYTES == 4718592ull' in lay and 'SM_RX_BYTES == 5242880ull' in lay and",
)
rep(
    "'4 MiB arena view + X-5 asserts + MP-strided rx_view')",
    "'8 MiB arena view (Rx > 4 MiB) + X-5 asserts + MP-strided rx_view')",
)
rep(
    "need('constexpr int PFK = 4;' in tr and", "need('constexpr int PFK = 3;' in tr and"
)
rep("'GEMM BM64 / PFK4 / 64 KiB LDS')", "'GEMM BM128 / PFK3 / 72 KiB LDS')")
rep(
    "'case 3u: smg::gemm_body<3>' in g",
    "'case 7u: smg::gemm_body<7>' in g and 'default: smg::gemm_body<8>' in g",
)
rep("AQ 1..4 dispatch", "AQ 1..8 dispatch")
with open(r21 + "/tests/check_r21_static.py", "w", encoding="utf-8", newline="") as _f:
    _f.write(t)
print("ok")
