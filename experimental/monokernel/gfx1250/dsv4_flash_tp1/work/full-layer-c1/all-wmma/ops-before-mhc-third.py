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
    ak=gl.arange(0,128,layout=gl.SliceLayout(0,AL))
    bk=gl.arange(0,128,layout=gl.SliceLayout(1,BL))
    n=gl.arange(0,128,layout=gl.SliceLayout(0,BL))
    for task in range(gl.program_id(0),256,GRID):
        head=task//4;ot=task%4
        scores=gl.full((16,128),0,gl.float32,ML)
        for kt in range(4):
            q=dev_load(Q,head*512+kt*128+ak[None,:]+m[:,None]*0,m[:,None]==0,0.0,cache='.cg')
            k=dev_load(Cache,n[None,:]*512+kt*128+bk[:,None],cache='.cg')
            scores=gl.amd.gfx1250.wmma(gl.convert_layout(q,DA),gl.convert_layout(k,DB),scores)
        sc=gl.convert_layout(scores,AL)*0.04419417382415922
        score=gl.sum(gl.where(m[:,None]==0,sc,0.0),0)
        maximum=dev_load(Sink,head,cache='.cg').to(gl.float32)
        denom=1.0
        probs=gl.full((128,),0,gl.float32,gl.SliceLayout(0,AL))
        alphas=gl.full((128,),0,gl.float32,gl.SliceLayout(0,AL))
        for chunk in range(8):
            valid=ak//16==chunk
            new_m=gl.maximum(gl.max(gl.where(valid,score,float('-inf')),0),maximum)
            alpha=gl.exp(maximum-new_m)
            pc=gl.where(valid,gl.exp(score-new_m),0.)
            denom=denom*alpha+gl.sum(pc,0)
            maximum=new_m
            probs=gl.where(valid,pc,probs)
            alphas=gl.where(valid,alpha,alphas)
        pm=gl.where(m[:,None]==ak[None,:]//16,probs[None,:],0.)
        hi=pm.to(gl.bfloat16);residual=pm-hi.to(gl.float32)
        lo=residual.to(gl.bfloat16);tiny=(residual-lo.to(gl.float32)).to(gl.bfloat16)
        v=dev_load(Cache,bk[:,None]*512+ot*128+n[None,:],cache='.cg')
        vb=gl.convert_layout(v,DB)
        term=gl.amd.gfx1250.wmma(gl.convert_layout(hi,DA),vb,gl.full((16,128),0,gl.float32,ML))
        term=gl.amd.gfx1250.wmma(gl.convert_layout(lo,DA),vb,term)
        term=gl.amd.gfx1250.wmma(gl.convert_layout(tiny,DA),vb,term)
        term=gl.convert_layout(term,AL)
        out=gl.full((128,),0,gl.float32,gl.SliceLayout(0,AL))
        for chunk in range(8):
            alpha=gl.sum(gl.where(ak==chunk*16,alphas,0.),0)
            row=gl.sum(gl.where(m[:,None]==chunk,term,0.),0)
            out=out*alpha+row
        dev_store((out/denom).to(gl.bfloat16),O,head*512+ot*128+ak,cache='.cg')

@g.jit
def routed_linear(X,W,S,Y,N:gl.constexpr,K:gl.constexpr,FORMAT:gl.constexpr,GRID:gl.constexpr,GROUPED:gl.constexpr=False,FLOAT_OUT:gl.constexpr=False,Gamma=None,BN:gl.constexpr=32,Ids=None,PER_EXPERT_INPUT:gl.constexpr=False):
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,32])
    AL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    BL:gl.constexpr=gl.BlockedLayout([4,1],[32,1],[1,4],[0,1])
    DA:gl.constexpr=gl.DotOperandLayout(0,ML,8)
    DB:gl.constexpr=gl.DotOperandLayout(1,ML,8)
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL))
    ka=gl.arange(0,128,layout=gl.SliceLayout(0,AL))
    kb=gl.arange(0,128,layout=gl.SliceLayout(1,BL))
    rn=gl.arange(0,BN,layout=gl.SliceLayout(0,BL))
    for tile in range(gl.program_id(0),6*gl.cdiv(N,BN),GRID):
        slot=tile//gl.cdiv(N,BN)
        ntile=tile%gl.cdiv(N,BN)
        expert=dev_load(Ids,slot,cache='.cg').to(gl.int64)
        ns=ntile*BN+rn
        acc=gl.full((16,BN),0,gl.float32,ML)
        for block in range(K//128):
            if GROUPED:
                xbase=tile*BN//1024*K
            else:
                xbase=slot*K if PER_EXPERT_INPUT else 0
            av=dev_load(X,xbase+block*128+ka[None,:]+rm[:,None]*0,rm[:,None]==0,0.0,cache='.cg').to(gl.bfloat16)
            k=block*128+kb
            if FORMAT == 4:
                raw=dev_load(W,expert*N*(K//2)+ns[None,:]*(K//2)+k[:,None]//2,cache='.cg').to(gl.uint8)
                v=(raw>>((k[:,None]%2)*4))&15
                a=v&7
                w=gl.where(a<2,a.to(gl.float32)*.5,gl.where(a<4,(a.to(gl.float32)-2)*.5+1,gl.where(a<6,a.to(gl.float32)-2,(a.to(gl.float32)-6)*2+4)))
                w=gl.where(v>=8,-w,w)
                scale=dev_load(S,expert*N*(K//32)+ns[None,:]*(K//32)+k[:,None]//32,cache='.cg').to(gl.uint32)
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
            out=out*dev_load(Gamma,slot,cache='.cg')
        if not FLOAT_OUT:
            out=out.to(gl.bfloat16)
        on=slot*N+ntile*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,AL))
        dev_store(out,Y,on[None,:]+rm[:,None]*0,rm[:,None]==0,cache='.cg')

