import sys,re
root=sys.argv[1]
def rd(p):
    t=open(p,encoding='utf-8',newline='').read(); nl='\r\n' if '\r\n' in t else '\n'; return t.replace('\r\n','\n'),nl
def wr(p,t,nl): open(p,'w',encoding='utf-8',newline='').write(t.replace('\n',nl))
def rep(t,old,new,cnt=1):
    assert t.count(old)==cnt,('count %d != %d: '%(t.count(old),cnt))+old[:110]
    return t.replace(old,new)
# ===================== small/r19/host/driver_sm.cpp =====================
p=root+'/small/r19/host/driver_sm.cpp'; t,nl=rd(p)
t=rep(t,'    SmBindingV1 b{}; bool bound = false; SmDev* dblk = nullptr;','''    SmBindingV1 b{}; bool bound = false; SmDev* dblk = nullptr;
    uint8_t* gpage = nullptr;            // r19: per-comm ERR/STEP/DIAG page (u32 ERR at +0, u64 STEP at +8, DIAG u64[8] at +64); runtime-owned in the fusion path, driver-private in the harness path
    bool gpage_owned = false; SmGfPeers* dgpeers = nullptr;   // device-resident peer ERR table (null = no propagation)''')
t=rep(t,'''    if (hipMalloc(reinterpret_cast<void**>(&c->P), sizeof(float) * SM_SK * SM_MP * SM_N) != hipSuccess) { sm_log("P", "hipMalloc rc=FAIL"); return E_HIP; }   // >=256B aligned''','''    if (hipMalloc(reinterpret_cast<void**>(&c->P), sizeof(float) * SM_SK * SM_MP * SM_N) != hipSuccess) { sm_log("P", "hipMalloc rc=FAIL"); return E_HIP; }   // >=256B aligned
    // r19 harness path: a driver-private ERR/STEP page (zeroed; STEP 0, no peer propagation). The fusion path (sm_attach_v3) uses the runtime's per-comm page instead.
    if (hipMalloc(reinterpret_cast<void**>(&c->gpage), 4096) != hipSuccess || hipMemset(c->gpage, 0, 4096) != hipSuccess) { sm_log("gpage", "rc=FAIL"); return E_HIP; }
    c->gpage_owned = true;''')
t=rep(t,'hipModuleGetFunction(&c->fr, c->mr, "sm_ran_v1")','hipModuleGetFunction(&c->fr, c->mr, "sm_ran_v3")')
t=rep(t,'    if (hipMemset(c->clocal, 0, SM_C_BYTES) != hipSuccess || hipMemset(c->rlocal, 0, RX_BYTES) != hipSuccess) return E_HIP;','''    if (hipMemset(c->clocal, 0, SM_C_BYTES) != hipSuccess || hipMemset(c->rlocal, 0, RX_BYTES) != hipSuccess) return E_HIP;   // control window incl. the r19 tag words (bytes 320-383)
    if (c->gpage && c->gpage_owned && hipMemset(c->gpage, 0, 4096) != hipSuccess) return E_HIP;                       // harness page: ERR/STEP/DIAG''')
# device block: per-launch inputs are NOT in the block any more
t=rep(t,'h.P = c->P; h.Wscale = c->b.Wscale; h.Ascale = c->b.Ascale; h.cwin_loc','h.P = c->P; h.Wscale = c->b.Wscale; h.Ascale = nullptr; h.cwin_loc            // r19: Ascale/positions/slot_mapping/index_slot_mapping are per-launch kernargs, never read from the block\n    ;h.cwin_loc')
t=rep(t,'    h.positions = c->b.positions; h.slot_mapping = c->b.slot_mapping; h.q_out','    h.positions = nullptr; h.slot_mapping = nullptr; h.q_out')
t=rep(t,'index_cache = c->b.index_cache; h.index_slot_mapping = c->b.index_slot_mapping;','index_cache = c->b.index_cache; h.index_slot_mapping = nullptr; h.gpeers = c->dgpeers;')
# enqueue -> v3
i=t.index('hipError_t enqueue(Ctx* c, hipStream_t s) {'); j=t.index('}\n}\n\nextern "C" {\nint32_t sm_launch_eager_v1',i)
new='''// r19: RAN kernarg v3 = ONE by-value 64 B struct (SmRanArgsV3), passed through ONE kernelParams pointer exactly like the GEMM. err/step point into the per-comm page.
hipError_t enqueue_v3(Ctx* c, hipStream_t s, const float* Ascale, const int64_t* positions, const int64_t* slot_mapping, uint32_t t_valid, uint32_t call_id) {
    SmGemmArgsV1 g{}; g.A = c->b.A; g.W = c->b.W; g.P = c->P; g.M = c->b.M; g.N = SM_N; g.K = 6144u; g.strideA = c->b.strideA; g.sk = SM_SK; g.Cwin = c->cwin; g.dbg = c->dbg;
    SmRanArgsV3 ka{};
    ka.blk = c->dblk; ka.Ascale = Ascale; ka.positions = positions; ka.slot_mapping = slot_mapping;
    ka.err = reinterpret_cast<const uint32_t*>(c->gpage); ka.step = reinterpret_cast<const uint64_t*>(c->gpage + 8);
    ka.M = c->b.M; ka.rank = static_cast<uint32_t>(c->rank); ka.t_valid = t_valid; ka.call_id = call_id;
    static_assert(sizeof(SmRanArgsV3) == 64, "RAN kernarg v3 = 64 B");
    void* ga[] = {&g};
    void* ra[] = {&ka};
    hipError_t e = hipModuleLaunchKernel(c->fg, SM_GEMM_NTILES, SM_SK, 1, 256, 1, 1, 0, s, ga, nullptr);
    if (e != hipSuccess) return e;
    return hipModuleLaunchKernel(c->fr, RAN_GRID, 1, 1, SM_RAN_THREADS, 1, 1, 0, s, ra, nullptr);          // r15: 256 (default) or 512 (PUSH8 / PUSH_HALVES)
}
// Harness (Phase-1 style) launches use the bound tensors; t_valid = T (no pad), call_id = 0.
hipError_t enqueue(Ctx* c, hipStream_t s) { return enqueue_v3(c, s, c->b.Ascale, c->b.positions, c->b.slot_mapping, c->b.T, 0u); }
'''
t=t[:i]+new+t[j:]
# poison (retired) -> ERR word
t=rep(t,'''int32_t sm_poison_v1(void* ctx, uint32_t* out) {
    auto* c = static_cast<Ctx*>(ctx); if (!c || !out) return E_ARG;
    return hipMemcpy(out, static_cast<uint8_t*>(c->clocal) + SM_C_POISON, 4, hipMemcpyDeviceToHost) == hipSuccess ? OK : E_HIP;''','''int32_t sm_poison_v1(void* ctx, uint32_t* out) {            // r19: returns the per-comm ERR word (SM_C_POISON is retired)
    auto* c = static_cast<Ctx*>(ctx); if (!c || !out || !c->gpage) return E_ARG;
    return hipMemcpy(out, c->gpage, 4, hipMemcpyDeviceToHost) == hipSuccess ? OK : E_HIP;''')
wr(p,t,nl)
print('driver_sm ok')