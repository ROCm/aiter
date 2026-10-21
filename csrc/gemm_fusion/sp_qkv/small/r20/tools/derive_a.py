import sys,re
r=sys.argv[1]
def rd(p):
    t=open(p,encoding='utf-8',newline='').read(); nl='\r\n' if '\r\n' in t else '\n'; return t.replace('\r\n','\n'),nl
def wr(p,t,nl): open(p,'w',encoding='utf-8',newline='').write(t.replace('\n',nl))
def rep(t,old,new,cnt=1):
    assert t.count(old)==cnt,('count %d != %d: '%(t.count(old),cnt))+old[:110]
    return t.replace(old,new)
# ---------------- ABI ----------------
p=r+'/interface/sm_abi_v3.h'; t,nl=rd(p)
t=rep(t,'#define SM_MP        32u            // padded rows per source (live rows M<=16 for T=64)','''#define SM_MP        64u            // r20 (R13 card A, MP64): padded rows per source (live rows M <= 64: local m 33..64, decode T = 132..256); P / Rx / stamps are sized by it
#define SM_NPASS     (SM_MP / 16u)  // 16-row passes per source (4); every RAN role owns 2 of them (push halves, IK owners, self row-split) or 1 (B row-split)''')
t=rep(t,'#define SM_DBG_RAN_MAXWG   128u','#define SM_DBG_RAN_MAXWG   512u    // r20: grid <= 304 WGs (R2.5: MAXWG >= 512 BEFORE any stamps run)')
i=t.index('// r15 (D10): push-role shape switches'); j=t.index('#ifndef SM_R5X_INDEXER')
t=t[:i]+'''// r20 role shape (fixed, no switches): 24 push WGs (3 per XCC) at 512 threads; the two 256-thread halves own row passes {0,1} and {2,3} (generalised HALVES, one round trip, one wait, K = 8 unit pushers per (src,dst)); every non-push role uses waves 0-3 only
// (waves 4-7 retire at kernel start, I-1 pattern). The r15-r18 shape switches are removed: setting them is an error.
#ifndef SM_PUSH8
#define SM_PUSH8 0
#endif
#ifndef SM_PUSH_HALVES
#define SM_PUSH_HALVES 0
#endif
#ifndef SM_NT_STORES
#define SM_NT_STORES 0
#endif
#if SM_PUSH8 || SM_PUSH_HALVES
#error "r20 has ONE push shape (24 x 512, halves = 2 passes each); SM_PUSH8 / SM_PUSH_HALVES do not exist"
#endif
#define SM_RAN_THREADS      512u
#define SM_T1_PUSH_WGS      24u
'''+t[j:]
t=rep(t,'#ifndef SM_B_ROWSPLIT\n#define SM_B_ROWSPLIT 0\n#endif\n#ifndef SM_RX_BYPASS\n#define SM_RX_BYPASS 0\n#endif','''#ifndef SM_B_ROWSPLIT
#define SM_B_ROWSPLIT 0
#endif
#if SM_B_ROWSPLIT
#error "r20: the B row split is intrinsic (SM_B_SPLIT = SM_NPASS, one 16-row pass per B WG); SM_B_ROWSPLIT does not exist"
#endif
// r20 D-10 (R13 card R2.4): SM_RX_BYPASS = 1 is REQUIRED in every MP64 mode (27 acquires per XCC without an INVL2 queue). gfx950-ONLY hardware dependency (see above) for ALL small modes at MP64.
#ifndef SM_RX_BYPASS
#define SM_RX_BYPASS 1
#endif
#if !SM_RX_BYPASS
#error "r20 requires SM_RX_BYPASS=1 in every mode (R13 card R2.4 D-10)"
#endif''')
t=rep(t,'#if SM_B_ROWSPLIT\n#define SM_B_SPLIT 2u\n#else\n#define SM_B_SPLIT 1u\n#endif','#define SM_B_SPLIT SM_NPASS            // r20: one B WG = one head-group x ONE 16-row pass (dead passes still acquire)\n#define SM_SELF_SPLIT 2u               // r20: a self WG = one head x TWO 16-row passes (row-split in 2 groups)')
t=rep(t,'#define SM_T1_IK_WGS        1u    // ONE IK owner WG per source rank','#define SM_T1_IK_WGS        2u    // r20: TWO IK owner WGs per source rank (row-split: owner o owns passes {2o, 2o+1}); every (src,dst) flag counts 8 unit pushers + 2')
t=rep(t,'#define SM_T1_SELF_WGS      SM_T1_UPD','#define SM_T1_SELF_WGS      (SM_T1_UPD * SM_SELF_SPLIT)')
t=rep(t,'#define SM_T1_B_HEADS       2u    // U3 A/B flag: heads per B WG (1 -> 54 B WGs, 2 -> 27 B WGs pipelined)','#define SM_T1_B_HEADS       1u    // r20 default NH1: B_PER_PAIR = 18 x 4 row passes = 72 (dense/reuse) / 80 (indexer); grid 276 / 304')
t=rep(t,'#define SM_T1_GRID          (SM_T1_B0 + SM_T1_B_WGS)   // indexer: 74 / 104; reuse: 69 / 96','#define SM_T1_GRID          (SM_T1_B0 + SM_T1_B_WGS)   // r20 NH1: dense/reuse 24 + 36 + 216 = 276; indexer 24 + 2 + 38 + 240 = 304')
wr(p,t,nl)
# ---------------- roles ----------------
p=r+'/device/layout/t1_roles.hpp'; t,nl=rd(p)
t=rep(t,'uint32_t seen[24][2] = {};','uint32_t seen[24][SM_B_SPLIT] = {};')
t=rep(t,'static_assert(SM_T1_GRID == SM_T1_PUSH_WGS + SM_T1_IK_WGS + SM_T1_UPD + 3u * B_PER_PAIR, "grid = push + IK + self + 3*B_PER_PAIR");','''static_assert(SM_T1_GRID == SM_T1_PUSH_WGS + SM_T1_IK_WGS + SM_T1_SELF_WGS + 3u * B_PER_PAIR, "grid = push + IK + self + 3*B_PER_PAIR");
// ---- r20 (R13 card A, R2.3 D-9, R2.5): MP64 role table checks ----
static_assert(SM_NPASS == 4u && SM_MP == 64u, "r20 is the MP64 build: 4 row passes of 16 rows");
static_assert(SM_B_SPLIT == SM_NPASS, "B WGs: one row pass each (dead passes still acquire flag[src], R2.5)");
static_assert(SM_SELF_SPLIT * 2u == SM_NPASS, "self WGs: 2 passes each, 2 row groups");
static_assert(IK_WGS == 0u || IK_WGS * 2u == SM_NPASS, "IK owners (indexer): 2 passes each => IK_WGS == 2; dense/reuse have none");
static_assert(SM_R5X_INDEXER ? (IK_WGS == 2u && kcount(0) == 10u && kcount(1) == 10u && kcount(2) == 10u) : (IK_WGS == 0u && kcount(0) == 8u && kcount(1) == 8u && kcount(2) == 8u), "R2.3 D-9: K = 8 unit pushers + IK_WGS (indexer 10, dense/reuse 8) for EVERY (src, dst) pair");
static_assert(SM_T1_GRID <= SM_DBG_RAN_MAXWG && SM_T1_GRID <= 512u, "grid fits the stamp records and the 512 slots (R2.5 MAXWG >= 512)");
static_assert(SM_T1_GRID == (SM_R5X_INDEXER ? 304u : 276u) || SM_T1_B_HEADS != 1u, "NH1 grids: dense/reuse 276, indexer 304");''')
wr(p,t,nl)
# ---------------- layouts ----------------
p=r+'/device/layout/ran_layouts.hpp'; t,nl=rd(p)
t=rep(t,'constexpr int kF = static_cast<int>(SM_F), kN = static_cast<int>(SM_N);','constexpr int kF = static_cast<int>(SM_F), kN = static_cast<int>(SM_N), kMP = static_cast<int>(SM_MP);')
t=rep(t,'// Rx window: [par=2][src=4][Mp=32][F=2560] bf16 (element offsets)','// Rx window: [par=2][src=4][Mp=SM_MP=64][F] bf16 (element offsets)')
t=rep(t,'opus::make_tuple(2, 4, 32, kF), opus::make_tuple(4 * 32 * kF, 32 * kF, kF, 1)','opus::make_tuple(2, 4, kMP, kF), opus::make_tuple(4 * kMP * kF, kMP * kF, kF, 1)')
t=rep(t,'// Partials: [SK][Mp=32][N] fp32 (SK = SM_SK, 2 or 3)','// Partials: [SK][Mp=SM_MP=64][N] fp32 (SK = SM_SK, 2 or 3)')
t=rep(t,'opus::make_tuple(2, 32, kN), opus::make_tuple(32 * kN, kN, 1)','opus::make_tuple(2, kMP, kN), opus::make_tuple(kMP * kN, kN, 1)')
t=rep(t,'// rx_view strides: par * (4*32*F) + src * (32*F) + row * F + col','// rx_view strides: par * (4*MP*F) + src * (MP*F) + row * F + col')
t=rep(t,'1ull * (4ull * 32ull * SM_F) + 3ull * (32ull * SM_F)','1ull * (4ull * SM_MP * SM_F) + 3ull * (1ull * SM_MP * SM_F)')
t=rep(t,'static_assert(SM_MP == 32u, "rx_view is written for MP == 32");','''static_assert(SM_MP == 64u, "r20 layouts are written for MP == 64");
static_assert(SM_RX_BYTES == 2ull * 4ull * SM_MP * SM_F * 2ull, "window bytes = [par 2][src 4][MP][F] bf16");
static_assert(SM_RX_BYTES <= 4ull * 1024ull * 1024ull, "R13: the Rx window must fit the 4 MiB arena view (dense 2,359,296 B, indexer 2,621,440 B at MP = 64)");
static_assert(SM_DENSE ? (SM_RX_BYTES == 2359296ull) : (SM_RX_BYTES == 2621440ull), "R2.4 X-5 recomputed for MP = 64");''')
wr(p,t,nl)
# ---------------- build script default NH1 ----------------
p=r+'/build_device.sh'; t,nl=rd(p)
t=rep(t,'B_HEADS="${B_HEADS:-2}"','B_HEADS="${B_HEADS:-1}"'); wr(p,t,nl)
print('abi/roles/layouts/build ok')