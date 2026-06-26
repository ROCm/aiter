import torch, itertools
from aiter import per_1x32_f4_quant_hip
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.flydsl.kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

def bench(fn, it, wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(it): fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / it

def prep(S):
    M=N=K=S
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)/10
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)/10
    xq, xs = per_1x32_f4_quant_hip(x, shuffle=True)
    wq0, ws0 = dynamic_mxfp4_quant(w)
    wq = shuffle_weight(wq0, layout=(16,16)); ws = e8m0_shuffle(ws0)
    return (xq.view(torch.uint8), wq.view(torch.uint8), xs.view(torch.uint8), ws.view(torch.uint8))

def try_cfg(S, args, it, **kw):
    M=N=K=S
    try:
        launch = compile_preshuffle_gemm_a8(N=N, K=K, in_dtype="fp4", out_dtype="bf16", **kw)
        c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        s = torch.cuda.current_stream()
        def fly(): launch(c, *args, M, N, s); return c
        fly(); torch.cuda.synchronize()
        return 2*M*N*K/(bench(fly, it)*1e-3)/1e12
    except Exception as e:
        return None

def run(S):
    it = 200 if S<=2048 else (50 if S==4096 else (30 if S==8192 else 20))
    args = prep(S)
    base = dict(lds_stage=2, use_async_copy=True)
    # Stage A: tiles
    tilesA = [(64,64,128),(128,64,128),(256,64,128),(128,128,128),(128,256,128),
              (256,128,128),(256,192,128),(256,256,128),(192,128,128),(128,128,256),(256,128,256)]
    A=[]
    for tm,tn,tk in tilesA:
        tf = try_cfg(S, args, it, tile_m=tm, tile_n=tn, tile_k=tk, **base)
        if tf: A.append((tf,(tm,tn,tk)))
    A.sort(reverse=True)
    print(f"[S={S}] StageA top3: {[ (round(t),c) for t,c in A[:3] ]}", flush=True)
    top = [c for _,c in A[:3]]
    # Stage B: waves_per_eu x cshuffle on top tiles
    B=[]
    for (tm,tn,tk) in top:
        for wpe in (None,1,2,3,4):
            for csh in (False, True):
                tf = try_cfg(S, args, it, tile_m=tm, tile_n=tn, tile_k=tk,
                             waves_per_eu=wpe, use_cshuffle_epilog=csh, **base)
                if tf: B.append((tf,(tm,tn,tk,wpe,csh)))
    B.sort(reverse=True)
    print(f"[S={S}] StageB top3: {[ (round(t),c) for t,c in B[:3] ]}", flush=True)
    tm,tn,tk,wpe,csh = B[0][1]
    # Stage C: preload depths
    C=[(B[0][0],(B[0][1],(-1,-1)))]
    for dsrd,dvmem in [(0,0),(1,1),(2,2),(3,3),(1,2),(2,1),(2,3),(3,2)]:
        tf = try_cfg(S, args, it, tile_m=tm, tile_n=tn, tile_k=tk, waves_per_eu=wpe,
                     use_cshuffle_epilog=csh, dsrd_preload=dsrd, dvmem_preload=dvmem, **base)
        if tf: C.append((tf,((tm,tn,tk,wpe,csh),(dsrd,dvmem))))
    C.sort(reverse=True)
    best = C[0]
    print(f"RESULT S={S} | best {best[0]:.0f} TFLOP/s | cfg tile={best[1][0]} preload={best[1][1]}", flush=True)

if __name__ == "__main__":
    for S in (1024, 2048, 4096, 8192, 16384):
        run(S)
    print("DONE", flush=True)
