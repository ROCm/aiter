from qkv_wmma import *

@g.jit
def linear(X,W,S,Y,N:gl.constexpr,K:gl.constexpr,FORMAT:gl.constexpr,GRID:gl.constexpr,GROUPED:gl.constexpr=False,FLOAT_OUT:gl.constexpr=False,Gamma=None,BN:gl.constexpr=64):
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,32])
    AL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    BL:gl.constexpr=gl.BlockedLayout([4,1],[32,1],[1,4],[0,1])
    DA:gl.constexpr=gl.DotOperandLayout(0,ML,8)
    DB:gl.constexpr=gl.DotOperandLayout(1,ML,8)
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL))
    ka=gl.arange(0,128,layout=gl.SliceLayout(0,AL))
    kb=gl.arange(0,128,layout=gl.SliceLayout(1,BL))
    rn=gl.arange(0,BN,layout=gl.SliceLayout(0,BL))
    for tile in range(gl.program_id(0),gl.cdiv(N,BN),GRID):
        ns=tile*BN+rn
        acc=gl.full((16,BN),0,gl.float32,ML)
        for block in range(K//128):
            if GROUPED:
                xbase=tile*BN//1024*K
            else:
                xbase=0
            av=dev_load(X,xbase+block*128+ka[None,:]+rm[:,None]*0,rm[:,None]==0,0.0,cache='.cg').to(gl.bfloat16)
            k=block*128+kb
            if FORMAT == 4:
                raw=dev_load(W,ns[None,:]*(K//2)+k[:,None]//2,cache='.cg').to(gl.uint8)
                v=(raw>>((k[:,None]%2)*4))&15
                a=v&7
                w=gl.where(a<2,a.to(gl.float32)*.5,gl.where(a<4,(a.to(gl.float32)-2)*.5+1,gl.where(a<6,a.to(gl.float32)-2,(a.to(gl.float32)-6)*2+4)))
                w=gl.where(v>=8,-w,w)
                scale=dev_load(S,ns[None,:]*(K//32)+k[:,None]//32,cache='.cg').to(gl.uint32)
                w=w*(scale<<23).to(gl.float32,bitcast=True)
            elif FORMAT == 8:
                w=dev_load(W,ns[None,:]*K+k[:,None],cache='.cg').to(gl.float32)
                scale=dev_load(S,(ns[None,:]//128)*(K//128)+k[:,None]//128,cache='.cg').to(gl.uint32)
                w=w*(scale<<23).to(gl.float32,bitcast=True)
            else:
                w=dev_load(W,ns[None,:]*K+k[:,None],cache='.cg').to(gl.float32)
            hi=w.to(gl.bfloat16)
            aa=gl.convert_layout(av,DA)
            bb=gl.convert_layout(hi,DB)
            acc=gl.amd.gfx1250.wmma(aa,bb,acc)
            if FORMAT == 16:
                lo=(w-hi.to(gl.float32)).to(gl.bfloat16)
                acc=gl.amd.gfx1250.wmma(aa,gl.convert_layout(lo,DB),acc)
        out=gl.convert_layout(acc,AL)
        if Gamma is not None:
            out=out*dev_load(Gamma,0,cache='.cg')
        if not FLOAT_OUT:
            out=out.to(gl.bfloat16)
        on=tile*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,AL))
        dev_store(out,Y,on[None,:]+rm[:,None]*0,rm[:,None]==0,cache='.cg')

@g.jit
def mhc_wmma(R,W,S,B,M,Temp,Counter,Health,GRID:gl.constexpr):
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,32])
    AL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    BL:gl.constexpr=gl.BlockedLayout([4,1],[32,1],[1,4],[0,1])
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL))
    ka=gl.arange(0,128,layout=gl.SliceLayout(0,AL))
    kb=gl.arange(0,128,layout=gl.SliceLayout(1,BL))
    rn=gl.arange(0,16,layout=gl.SliceLayout(0,BL))
    for task in range(gl.program_id(0),256,GRID):
        nt=task//128;ks=task%128
        n=nt*16+rn
        av=dev_load(R,ks*128+ka[None,:]+rm[:,None]*0,rm[:,None]==0,0.0,cache='.cg')
        w=dev_load(W,n[None,:]*16384+ks*128+kb[:,None],n[None,:]<24,0.0,cache='.cg').to(gl.float32)
        hi=w.to(gl.bfloat16);lo=(w-hi.to(gl.float32)).to(gl.bfloat16)
        aa=gl.convert_layout(av,gl.DotOperandLayout(0,ML,8))
        acc=gl.amd.gfx1250.wmma(aa,gl.convert_layout(hi,gl.DotOperandLayout(1,ML,8)),gl.full((16,16),0,gl.float32,ML))
        acc=gl.amd.gfx1250.wmma(aa,gl.convert_layout(lo,gl.DotOperandLayout(1,ML,8)),acc)
        out=gl.convert_layout(acc,AL)
        on=nt*16+gl.arange(0,16,layout=gl.SliceLayout(0,AL))
        dev_store(out,Temp,ks*24+on[None,:]+rm[:,None]*0,(rm[:,None]==0)&(on[None,:]<24),cache='.cg')
    sync(Counter,Health,GRID)
    L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
    i=gl.arange(0,1024,layout=L)
    si=gl.arange(0,128,layout=L)
    for row in range(gl.program_id(0),24,GRID):
        dot=gl.sum(dev_load(Temp,si*24+row,cache='.cg'),0)
        sq=0.0
        for b in range(16):
            x=dev_load(R,b*1024+i,cache='.cg').to(gl.float32)
            sq+=gl.sum(x*x,0)
        z=dot*gl.rsqrt(sq/16384+1e-6)
        idx=gl.where(row<4,0,gl.where(row<8,1,2))
        dev_store(z*dev_load(S,idx,cache='.cg')+dev_load(B,row,cache='.cg'),M,row,cache='.cg')
    sync(Counter,Health,GRID)

@g.jit
def attention_wmma(Q,KV,Cache,Sink,O,GRID:gl.constexpr):
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,32])
    AL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    BL:gl.constexpr=gl.BlockedLayout([4,1],[32,1],[1,4],[0,1])
    DA:gl.constexpr=gl.DotOperandLayout(0,ML,8)
    DB:gl.constexpr=gl.DotOperandLayout(1,ML,8)
    m=gl.arange(0,16,layout=gl.SliceLayout(1,AL))
    ak=gl.arange(0,512,layout=gl.SliceLayout(0,AL))
    bk=gl.arange(0,512,layout=gl.SliceLayout(1,BL))
    n=gl.arange(0,128,layout=gl.SliceLayout(0,BL))
    for head in range(gl.program_id(0),64,GRID):
        q=dev_load(Q,head*512+ak[None,:]+m[:,None]*0,m[:,None]==0,0.0,cache='.cg')
        k=dev_load(Cache,n[None,:]*512+bk[:,None],cache='.cg')
        scores=gl.amd.gfx1250.wmma(gl.convert_layout(q,DA),gl.convert_layout(k,DB),gl.full((16,128),0,gl.float32,ML))
        scores=gl.convert_layout(scores,AL)*0.04419417382415922
        sink=dev_load(Sink,head,cache='.cg').to(gl.float32)
        maximum=gl.maximum(gl.max(scores,1),sink)
        probs=gl.exp(scores-maximum[:,None])
        denom=gl.sum(probs,1)+gl.exp(sink-maximum)
        probs=probs/denom[:,None]
        hi=probs.to(gl.bfloat16)
        lo=(probs-hi.to(gl.float32)).to(gl.bfloat16)
        vk=gl.arange(0,128,layout=gl.SliceLayout(1,BL))
        vn=gl.arange(0,512,layout=gl.SliceLayout(0,BL))
        v=dev_load(Cache,vk[:,None]*512+vn[None,:],cache='.cg')
        vb=gl.convert_layout(v,DB)
        acc=gl.amd.gfx1250.wmma(gl.convert_layout(hi,DA),vb,gl.full((16,512),0,gl.float32,ML))
        acc=gl.amd.gfx1250.wmma(gl.convert_layout(lo,DA),vb,acc)
        out=gl.convert_layout(acc,AL).to(gl.bfloat16)
        on=gl.arange(0,512,layout=gl.SliceLayout(0,AL))
        dev_store(out,O,head*512+on[None,:]+m[:,None]*0,m[:,None]==0,cache='.cg')
