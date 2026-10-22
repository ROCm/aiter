import sys

root = sys.argv[1]


def rd(p):
    with open(p, encoding="utf-8", newline="") as _f:
        t = _f.read()
    nl = "\r\n" if "\r\n" in t else "\n"
    return t.replace("\r\n", "\n"), nl


def wr(p, t, nl):
    with open(p, "w", encoding="utf-8", newline="") as _f:
        _f.write(t.replace("\n", nl))


def rep(t, old, new, cnt=1):
    assert t.count(old) == cnt, (f"count {t.count(old):d} != {cnt:d}: ") + old[:110]
    return t.replace(old, new)


p = root + "/small/r19/host/driver_sm.cpp"
t, nl = rd(p)
t = rep(
    t,
    "h.Ascale = nullptr; h.cwin_loc            // r19: Ascale/positions/slot_mapping/index_slot_mapping are per-launch kernargs, never read from the block\n    ;h.cwin_loc = static_cast<uint8_t*>(c->clocal);",
    "h.Ascale = nullptr;          // r19: Ascale/positions/slot_mapping/index_slot_mapping are per-launch kernargs, never read from the block\n    h.cwin_loc = static_cast<uint8_t*>(c->clocal);",
)
wr(p, t, nl)
# ===================== shared host/sm_fusion_driver.cpp: ABI-3 entry points (r17/r18 builds, SM_ABI_VERSION 2, keep the v1 set) =====================
p = root + "/host/sm_fusion_driver.cpp"
t, nl = rd(p)
t = rep(
    t,
    'extern "C" {\nint32_t sm_attach_v1(',
    '#if SM_ABI_VERSION < 3\nextern "C" {\nint32_t sm_attach_v1(',
)
# close the v1-only section before sm_bind_new_v1? keep bind/free/info shared: split at the v1 attach end and at launch/poison
a = t.index("// Create a binding (device-resident SmDev block).")
t = t[:a] + '}\n#endif\nextern "C" {\n' + t[a:]
b = t.index("// Enqueue sm_gemm then RAN on `stream` for this binding")
c = t.index("int32_t sm_blk_free_v1(void* blk_)")
v1launch = t[b:c]
t = t[:b] + "#if SM_ABI_VERSION < 3\n" + v1launch + "#endif\n" + t[c:]
# v3 additions: attach + launch + free of dgpeers + (no poison_fast)
d = t.index("// Poison word (u32 at SM_C_POISON of the control window)")
poison_part = t[d:]
assert poison_part.rstrip().endswith("}")
v3 = """#if SM_ABI_VERSION >= 3
// ---- R12 (card r2 D2.3/D2.6): ABI-3 entry points. Signatures match the aiter stub (small_rt.py load_driver(abi=3)).
// attach(rank, comm, rx_win, rx_local, ctl_win, ctl_local, gpage_local, gpeers, gemm_hsaco, ran_hsaco, &ctx): like sm_attach_v1 plus the runtime-owned per-comm ERR/STEP/DIAG page.
//   gpage_local = local address of the page in a registered window/arena (u32 ERR at +0, u64 STEP at +8, DIAG u64[8] at +64); gpeers == NULL: the driver resolves the 3 peer ERR words with ccoGetPeerPtr(comm, gpage_local, p)
//   and uploads a device-resident SmGfPeers; non-NULL: a device-resident SmGfPeers* supplied by the runtime. There is NO per-plan poison read in ABI 3 (sm_poison_* are gone; the runtime reads ERR from the page).
int32_t sm_attach_v3(uint32_t rank, void* comm, void* rwin, void* rlocal, void* cwin, void* clocal, void* gpage_local, void* gpeers,
                     const char* gemm_hsaco, const char* ran_hsaco, void** out_ctx) {
    if (!out_ctx) return E_ARG;
    *out_ctx = nullptr;
    if (!comm || !rwin || !rlocal || !cwin || !clocal || !gpage_local || !gemm_hsaco || !ran_hsaco || rank > 3u) return E_ARG;
    if ((reinterpret_cast<uintptr_t>(gpage_local) & 63u) != 0u) { sm_log("attach", "rc=FAIL gpage_local not 64 B aligned"); return E_ARG; }
    auto* c = new Ctx(); *out_ctx = c; c->rank = static_cast<int>(rank);
    c->comm = static_cast<mori::cco::ccoComm*>(comm);
    gf::DeviceScope scope(dev_of(c));
    if (!scope.ok()) { sm_log("device", "hipSetDevice(%d) rc=FAIL", dev_of(c)); return E_HIP; }
    c->rwin = reinterpret_cast<mori::cco::ccoWindow_t>(rwin); c->rlocal = rlocal;
    c->cwin = reinterpret_cast<mori::cco::ccoWindow_t>(cwin); c->clocal = clocal;
    c->gpage = static_cast<uint8_t*>(gpage_local); c->gpage_owned = false;
    for (int p = 0; p < 4; ++p) {
        c->cpeer[p] = static_cast<uint8_t*>(mori::cco::ccoGetPeerPtr(c->comm, c->clocal, p));
        c->rpeer[p] = static_cast<uint8_t*>(mori::cco::ccoGetPeerPtr(c->comm, c->rlocal, p));
        if (!c->cpeer[p] || !c->rpeer[p]) { sm_log("attach", "rc=FAIL null peer pointer p=%d", p); return E_CCO_WIN_C; }
        if (p == c->rank && (c->cpeer[p] != static_cast<uint8_t*>(c->clocal) || c->rpeer[p] != static_cast<uint8_t*>(c->rlocal))) {
            sm_log("attach", "rc=FAIL self peer pointer != local window base"); return E_CCO_WIN_C; }
    }
    if (gpeers) {
        c->dgpeers = static_cast<SmGfPeers*>(gpeers);                  // runtime-owned device-resident table: not freed by sm_free_v1
    } else {
        SmGfPeers h{};
        for (int p = 0; p < 4; ++p) {
            void* pp = (p == c->rank) ? nullptr : mori::cco::ccoGetPeerPtr(c->comm, c->gpage, p);
            h.err[p] = static_cast<uint32_t*>(pp);
            if (p != c->rank && !pp) { sm_log("attach", "rc=FAIL null peer ERR pointer p=%d (gpage_local must lie in a registered window); pass gpeers or fix the page", p); return E_CCO_WIN_C; }
        }
        SmGfPeers* d = nullptr;
        if (hipMalloc(reinterpret_cast<void**>(&d), sizeof(h)) != hipSuccess || hipMemcpy(d, &h, sizeof(h), hipMemcpyHostToDevice) != hipSuccess) { sm_log("attach", "gpeers upload rc=FAIL"); return E_HIP; }
        c->dgpeers = d; c->dgpeers_owned = true;
    }
    if (hipMalloc(reinterpret_cast<void**>(&c->P), sizeof(float) * SM_SK * SM_MP * SM_N) != hipSuccess) { sm_log("P", "hipMalloc rc=FAIL"); return E_HIP; }
    if (!sm_image_matches(gemm_hsaco, SM_EXPECTED_GEMM_SHA256) || !sm_image_matches(ran_hsaco, SM_EXPECTED_RAN_SHA256)) { sm_log("bundle", "rc=FAIL image hash mismatch"); return E_HASH; }
    if (hipModuleLoad(&c->mg, gemm_hsaco) != hipSuccess || hipModuleLoad(&c->mr, ran_hsaco) != hipSuccess) { sm_log("module", "rc=FAIL"); return E_HIP; }
    if (hipModuleGetFunction(&c->fg, c->mg, "sm_gemm_v1") != hipSuccess || hipModuleGetFunction(&c->fr, c->mr, "sm_ran_v3") != hipSuccess) { sm_log("resolve", "rc=FAIL"); return E_HIP; }
    gf::flush_sticky();
    sm_log("attach", "rc=OK abi=3");
    return OK;
}
// launch(ctx, blk, A, W, strideA, Ascale, positions, slot_mapping, t_valid, call_id, stream): per-launch tensors are kernel arguments (staging removed); rows t >= t_valid are padded in-kernel (pos 0, slot -1).
// positions / slot_mapping must hold at least t_valid int64 elements, Ascale at least M floats (the kernel never reads beyond them).
int32_t sm_launch_blk_v3(void* ctx, void* blk_, const void* A, const void* W, uint32_t strideA, const float* Ascale, const int64_t* positions, const int64_t* slot_mapping,
                         uint32_t t_valid, uint32_t call_id, void* stream) {
    auto* c = static_cast<Ctx*>(ctx); auto* blk = static_cast<Blk*>(blk_); if (!c || !blk || !blk->dblk) return E_STATE;
    if (!own_ctx(c)) return E_STATE;
    if (!A || !W || !Ascale || !positions || !slot_mapping || t_valid > blk->b.T) return E_ARG;
    gf::DeviceScope scope(dev_of(c));
    std::lock_guard<std::mutex> swap_lk(g_swap_mu);
    SmBindingV1 saved_b = c->b; SmDev* saved_dblk = c->dblk;
    c->b = blk->b; c->b.A = A; c->b.W = W; c->b.strideA = strideA; c->dblk = blk->dblk;
    const hipError_t e = enqueue_v3(c, static_cast<hipStream_t>(stream), Ascale, positions, slot_mapping, t_valid, call_id);
    c->b = saved_b; c->dblk = saved_dblk;
    if (e != hipSuccess) { c->last_hip = static_cast<int>(e); std::fprintf(stderr, "SM_LAUNCH stage=enqueue err=%d (%s)\\n", (int)e, hipGetErrorString(e)); return E_HIP; }
    return OK;
}
#endif
"""
# insert v3 block after the v1 launch block (before sm_blk_free_v1) and make poison_fast v1-only
c2 = t.index("int32_t sm_blk_free_v1(void* blk_)")
t = t[:c2] + v3 + t[c2:]
d = t.index("// Poison word (u32 at SM_C_POISON of the control window)")
tail = t[d:]
t = t[:d] + "#if SM_ABI_VERSION < 3\n" + tail.rstrip()[:-1].rstrip() + "\n#endif\n}\n"
# sm_free_v1: free the owned peer table and the page only if we own it
t = rep(
    t,
    "    if (c->dblk) hipFree(c->dblk);\n    if (c->dbg) hipFree(c->dbg);",
    """    if (c->dblk) hipFree(c->dblk);
#if SM_ABI_VERSION >= 3
    if (c->dgpeers && c->dgpeers_owned) hipFree(c->dgpeers);
    if (c->gpage && c->gpage_owned) hipFree(c->gpage);
#endif
    if (c->dbg) hipFree(c->dbg);""",
)
wr(p, t, nl)
# Ctx needs dgpeers_owned
p = root + "/small/r19/host/driver_sm.cpp"
t, nl = rd(p)
t = rep(
    t,
    "bool gpage_owned = false; SmGfPeers* dgpeers = nullptr;",
    "bool gpage_owned = false, dgpeers_owned = false; SmGfPeers* dgpeers = nullptr;",
)
wr(p, t, nl)
print("fusion driver ok")
