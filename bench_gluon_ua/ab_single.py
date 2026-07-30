"""Benchmark the CURRENTLY-COMPILED gluon decode kernel (bench_ua.glu_2d) across
the decode grid; label the output by argv[1]. Run once per stride variant (the
kernel file is edited between runs so each process loads exactly one version):
    python ab_single.py B_constexpr   # strides gl.constexpr (current)
    python ab_single.py A_runtime     # strides gl.int32
Records attention-kernel time (the isolated signal; reduce is identical), total
time, GB/s, VGPRs/spills/LDS, xcheck vs triton. Single-buffer decode (nb=1)."""
import sys, math, json
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import triton
import bench_ua as B

DEV, DT, HS, RCP = B.DEV, B.DT, B.HEAD_SIZE, B.RCP_LN2
TILE = 64
torch.manual_seed(0)
LABEL = sys.argv[1] if len(sys.argv) > 1 else "unknown"
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 1
SHAPES = [(C, ctx, Hq, Hkv) for ctx in (1024, 8192) for (Hq, Hkv) in ((64, 8), (8, 1))
          for C in (16, 32, 64, 128)]
records = []


def ck_keys():
    ks = set()
    for dc in B.glu_2d.device_caches.values():
        c = dc[0] if isinstance(dc, tuple) else dc
        try:
            ks |= set(c.keys())
        except AttributeError:
            pass
    return ks


def stat_new(before):
    for dc in B.glu_2d.device_caches.values():
        c = dc[0] if isinstance(dc, tuple) else dc
        try:
            items = list(c.items())
        except AttributeError:
            continue
        for k, ck in items:
            if k not in before:
                md = getattr(ck, "metadata", None)
                return (getattr(md, "shared", None), getattr(ck, "n_regs", None), getattr(ck, "n_spills", None))
    return (None, None, None)


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
    og = torch.empty_like(q)
    if Sg == 1:
        def gf():
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                            NUM_SPLITS=1, ALL_DECODE=True, MFMA_DIM=16, NUM_BUFFERS=NB)
        before = ck_keys(); gf(); torch.cuda.synchronize()
        lds, regs, spills = stat_new(before)
        xc = (og.float() - ref.float()).abs().max().item(); red = 0.0
    else:
        so, sm, se = B.alloc_segm(C, Hq, Sg)
        def gf():
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                            NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=NB)
            B.launch_reduce(og, cu, seqk, bt, TILE, Sg, rnw, 16 // nqpk, (so, sm, se))
        before = ck_keys()
        B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                        NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=NB)
        lds, regs, spills = stat_new(before)
        B.launch_reduce(og, cu, seqk, bt, TILE, Sg, rnw, 16 // nqpk, (so, sm, se))
        torch.cuda.synchronize()
        xc = (og.float() - ref.float()).abs().max().item()
        red = B.pick(B.profile_kernels(gf), "reduce_segments")
    attn = B.pick(B.profile_kernels(gf), "unified_attention_2d")
    records.append(dict(label=LABEL, C=C, ctx=ctx, Hq=Hq, Hkv=Hkv, split=Sg,
                        attn_us=attn, tot_us=attn + red, gbps=B.gbps(byts, attn + red),
                        regs=regs, spills=spills, lds=lds, xc=xc))
    print(f"[{LABEL}] C{C:>3} ctx{ctx} {Hq}/{Hkv} S{Sg}: attn {attn:6.1f}us tot {attn+red:6.1f}us "
          f"{B.gbps(byts, attn+red):5.0f}GB/s regs{regs} sp{spills} lds{lds} xc{xc:.0e}")


print(f"=== [{LABEL}] triton {triton.__version__}, single-buffer decode ===")
for sh in SHAPES:
    measure(*sh)
json.dump(records, open(f"/app/aiter/bench_gluon_ua/ab_{LABEL}.json", "w"), indent=0)
print(f"wrote ab_{LABEL}.json")
