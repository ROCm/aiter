"""A/B: gluon decode with KV/block-table strides as runtime args (A) vs constexpr
(B). Same kernel otherwise; B lets the compiler bake the fixed vLLM-cache strides
in. Isolates the attention kernel (reduce is identical). triton 3.8 only.
Version A is a scratchpad copy of the current kernel with the stride annotations
reverted to gl.int32 (so nothing in the tree is touched)."""
import sys, math, json, importlib.util
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import triton
import bench_ua as B

DEV, DT, HS, RCP = B.DEV, B.DT, B.HEAD_SIZE, B.RCP_LN2
TILE = 64
torch.manual_seed(0)

glu_B = B.glu_2d  # current module = constexpr strides
_p = "/tmp/claude-0/-app-aiter-aiter-ops-triton--gluon-kernels-gfx950-attention/ebb894d0-4157-466e-9eba-e0bbf66f1c29/scratchpad/ua_rtstride.py"
_spec = importlib.util.spec_from_file_location("ua_rtstride", _p)
_ua = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ua)
glu_A = _ua.kernel_unified_attention_2d  # runtime strides


def cache_keys(kern):
    ks = set()
    for dc in kern.device_caches.values():
        cache = dc[0] if isinstance(dc, tuple) else dc
        try:
            ks |= set(cache.keys())
        except AttributeError:
            pass
    return ks


def stat_new(kern, before):
    for dc in kern.device_caches.values():
        cache = dc[0] if isinstance(dc, tuple) else dc
        try:
            items = list(cache.items())
        except AttributeError:
            continue
        for kkey, ck in items:
            if kkey not in before:
                md = getattr(ck, "metadata", None)
                return (getattr(md, "shared", None), getattr(ck, "n_regs", None), getattr(ck, "n_spills", None))
    return (None, None, None)


SHAPES = [(C, ctx, Hq, Hkv) for ctx in (1024, 8192) for (Hq, Hkv) in ((64, 8), (8, 1))
          for C in (16, 32, 64, 128)]
records = []


def measure(C, ctx, Hq, Hkv):
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS)
    q = torch.randn(C, Hq, HS, dtype=DT, device=DEV)
    k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
    seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    _, byts = B.decode_flops_bytes(C, ctx, Hq, Hkv)
    a, r = B.select_3d_config(HS, TILE, ctx, B.TARGET_PRGMS, C * Hkv, DT, DT, False, 1, 0)
    S = a["NUM_SEGMENTS_PER_SEQ"]; wpe = a["waves_per_eu"]; rnw = r["num_warps"]
    seg = B.alloc_segm(C, Hq, S); ref = torch.empty_like(q)
    B.launch_tri_3d(q, k, v, cu, seqk, bt, scale, 16, 16 // nqpk, TILE, a["num_warps"], S, wpe, a["num_stages"], seg)
    B.launch_reduce(ref, cu, seqk, bt, TILE, S, rnw, 16 // nqpk, seg)
    torch.cuda.synchronize()
    Sg = B.select_gluon_num_splits(C, Hkv, ctx // TILE)
    res = {}
    for tag, kern in (("A_runtime", glu_A), ("B_constexpr", glu_B)):
        B.glu_2d = kern  # launch_glu_2d closes over the module global
        og = torch.empty_like(q)
        if Sg == 1:
            def gf():
                B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                                NUM_SPLITS=1, ALL_DECODE=True, MFMA_DIM=16, NUM_BUFFERS=1)
            before = cache_keys(kern); gf(); torch.cuda.synchronize()
            lds, regs, spills = stat_new(kern, before)
            xc = (og.float() - ref.float()).abs().max().item()
            red = 0.0
        else:
            so, sm, se = B.alloc_segm(C, Hq, Sg)
            def gf():
                B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                                NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=1)
                B.launch_reduce(og, cu, seqk, bt, TILE, Sg, rnw, 16 // nqpk, (so, sm, se))
            before = cache_keys(kern)
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                            NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=1)
            lds, regs, spills = stat_new(kern, before)
            B.launch_reduce(og, cu, seqk, bt, TILE, Sg, rnw, 16 // nqpk, (so, sm, se))
            torch.cuda.synchronize()
            xc = (og.float() - ref.float()).abs().max().item()
            red = B.pick(B.profile_kernels(gf), "reduce_segments")
        attn = B.pick(B.profile_kernels(gf), "unified_attention_2d")
        res[tag] = dict(attn=attn, tot=attn + red, lds=lds, regs=regs, spills=spills, xc=xc)
    B.glu_2d = glu_B
    a_, b_ = res["A_runtime"], res["B_constexpr"]
    records.append(dict(C=C, ctx=ctx, Hq=Hq, Hkv=Hkv, split=Sg,
                        A_attn=a_["attn"], B_attn=b_["attn"], A_tot=a_["tot"], B_tot=b_["tot"],
                        A_gbps=B.gbps(byts, a_["tot"]), B_gbps=B.gbps(byts, b_["tot"]),
                        A_regs=a_["regs"], B_regs=b_["regs"], A_spills=a_["spills"], B_spills=b_["spills"],
                        A_lds=a_["lds"], B_lds=b_["lds"], xc=max(a_["xc"], b_["xc"])))
    print(f"C{C:>3} ctx{ctx} {Hq}/{Hkv} S{Sg}: attn A {a_['attn']:.1f}us B {b_['attn']:.1f}us "
          f"(B/A {a_['attn']/b_['attn']:.3f}x) | GB/s A {B.gbps(byts,a_['tot']):.0f} B {B.gbps(byts,b_['tot']):.0f} "
          f"| regs A{a_['regs']} B{b_['regs']} | xc{max(a_['xc'],b_['xc']):.0e}")


print(f"=== stride constexpr A/B (triton {triton.__version__}), single-buffer decode ===")
for sh in SHAPES:
    measure(*sh)
json.dump(records, open("/app/aiter/bench_gluon_ua/ab_stride_constexpr.json", "w"), indent=0)
print("wrote ab_stride_constexpr.json")
