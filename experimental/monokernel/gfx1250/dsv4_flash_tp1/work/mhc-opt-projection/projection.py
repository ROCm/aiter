from mhc import *
@g.jit
def project(R,W,Temp,BK:gl.constexpr,BN:gl.constexpr,SQ:gl.constexpr=False):
    pid=gl.program_id(0);nt=pid//(16384//BK);ks=pid%(16384//BK)
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,32])
    AL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    BL:gl.constexpr=gl.BlockedLayout([4,1],[32,1],[1,4],[0,1])
    SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[16,BK],[1,0])
    xs=gl.allocate_shared_memory(R.dtype.element_ty,[16,BK],SL)
    WL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[BN,BK],[1,0])
    ws=gl.allocate_shared_memory(W.dtype.element_ty,[BN,BK],WL)
    xd=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=R,shape=[1,16384],strides=[16384,1],block_shape=[16,BK],layout=SL)
    wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=W,shape=[24,16384],strides=[16384,1],block_shape=[BN,BK],layout=WL)
    gl.amd.gfx1250.tdm.async_load(xd,[0,ks*BK],xs,cache_modifier='.cg')
    gl.amd.gfx1250.tdm.async_load(wd,[nt*BN,ks*BK],ws,cache_modifier='.cg')
    gl.amd.gfx1250.tdm.async_wait(0)
    av=xs.load(AL);w=ws.permute([1,0]).load(BL)
    if SQ:
        rm0=gl.arange(0,16,layout=gl.SliceLayout(1,AL))
        sq=gl.sum(gl.sum(gl.where(rm0[:,None]==0,av.to(gl.float32)*av.to(gl.float32),0.),1),0)
        if nt==0:dev_store(sq,Temp,128*24+ks)
    hi=w.to(gl.bfloat16);lo=(w-hi.to(gl.float32)).to(gl.bfloat16)
    aa=gl.convert_layout(av,gl.DotOperandLayout(0,ML,8))
    acc=gl.amd.gfx1250.wmma(aa,gl.convert_layout(hi,gl.DotOperandLayout(1,ML,8)),gl.full((16,BN),0,gl.float32,ML))
    acc=gl.amd.gfx1250.wmma(aa,gl.convert_layout(lo,gl.DotOperandLayout(1,ML,8)),acc)
    out=gl.convert_layout(acc,AL)
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL));n=nt*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,AL))
    dev_store(out,Temp,ks*24+n[None,:]+rm[:,None]*0,(rm[:,None]==0)&(n[None,:]<24))


@g.jit
def region(R,W,S,B,AN,M,Temp,Raw,X,Counter,Health,BK:gl.constexpr,BN:gl.constexpr,GRID:gl.constexpr):
    if gl.program_id(0)<gl.cdiv(24,BN)*(16384//BK):project(R,W,Temp,BK,BN)
    sync(Counter,Health,GRID,3)
    mhc_reduce(R,S,B,M,Temp)
    sync(Counter,Health,GRID,3)
    sinkhorn(M)
    sync(Counter,Health,GRID,3)
    mhc_pre(R,M,Raw,GRID)
    sync(Counter,Health,GRID,3)
    norm(Raw,AN,X,4096,1,True,GRID)

@g.jit
def reduce_sq(S,B,M,Temp,SPLIT:gl.constexpr):
    row=gl.program_id(0)
    if row<24:
        L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
        si=gl.arange(0,SPLIT,layout=L)
        dot=gl.sum(dev_load(Temp,si*24+row),0)
        sq=gl.sum(dev_load(Temp,128*24+si),0)
        z=dot*gl.rsqrt(sq/16384+1.e-6);idx=gl.where(row<4,0,gl.where(row<8,1,2))
        dev_store(z*dev_load(S,idx)+dev_load(B,row),M,row)

@g.jit
def region_sq(R,W,S,B,AN,M,Temp,Raw,X,Counter,Health,BK:gl.constexpr,BN:gl.constexpr,GRID:gl.constexpr):
    if gl.program_id(0)<gl.cdiv(24,BN)*(16384//BK):project(R,W,Temp,BK,BN,True)
    sync(Counter,Health,GRID,3)
    reduce_sq(S,B,M,Temp,16384//BK)
    sync(Counter,Health,GRID,3)
    sinkhorn(M)
    sync(Counter,Health,GRID,3)
    mhc_pre(R,M,Raw,GRID)
    sync(Counter,Health,GRID,3)
    norm(Raw,AN,X,4096,1,True,GRID)

from tail_snapshot import tail_fused
@g.jit
def region_tail(R,W,S,B,AN,M,Temp,Raw,X,Counter,Health,BK:gl.constexpr,BN:gl.constexpr,GRID:gl.constexpr):
    if gl.program_id(0)<gl.cdiv(24,BN)*(16384//BK):project(R,W,Temp,BK,BN)
    sync(Counter,Health,GRID,3)
    tail_fused(R,S,B,AN,M,Temp,Raw,X,1)
@g.jit
def project_packed(R,W,Temp,BK:gl.constexpr,BN:gl.constexpr,SQ:gl.constexpr=False):
    pid=gl.program_id(0);nt=pid//(16384//BK);ks=pid%(16384//BK)
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,32])
    AL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    BL:gl.constexpr=gl.BlockedLayout([4,1],[32,1],[1,4],[0,1])
    SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[16,BK],[1,0])
    xs=gl.allocate_shared_memory(R.dtype.element_ty,[16,BK],SL)
    WL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,8]],[BN,BK],[1,0])
    ws=gl.allocate_shared_memory(W.dtype.element_ty,[BN,BK],WL)
    xd=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=R,shape=[1,16384],strides=[16384,1],block_shape=[16,BK],layout=SL)
    wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=W,shape=[48,16384],strides=[16384,1],block_shape=[BN,BK],layout=WL)
    wsl=gl.allocate_shared_memory(W.dtype.element_ty,[BN,BK],WL)
    gl.amd.gfx1250.tdm.async_load(xd,[0,ks*BK],xs,cache_modifier='.cg')
    gl.amd.gfx1250.tdm.async_load(wd,[nt*BN,ks*BK],ws,cache_modifier='.cg')
    gl.amd.gfx1250.tdm.async_load(wd,[24+nt*BN,ks*BK],wsl,cache_modifier='.cg')
    gl.amd.gfx1250.tdm.async_wait(0)
    av=xs.load(AL);w=ws.permute([1,0]).load(BL)
    if SQ:
        rm0=gl.arange(0,16,layout=gl.SliceLayout(1,AL))
        sq=gl.sum(gl.sum(gl.where(rm0[:,None]==0,av.to(gl.float32)*av.to(gl.float32),0.),1),0)
        if nt==0:dev_store(sq,Temp,128*24+ks)
    hi=w;lo=wsl.permute([1,0]).load(BL)
    aa=gl.convert_layout(av,gl.DotOperandLayout(0,ML,8))
    acc=gl.amd.gfx1250.wmma(aa,gl.convert_layout(hi,gl.DotOperandLayout(1,ML,8)),gl.full((16,BN),0,gl.float32,ML))
    acc=gl.amd.gfx1250.wmma(aa,gl.convert_layout(lo,gl.DotOperandLayout(1,ML,8)),acc)
    out=gl.convert_layout(acc,AL)
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL));n=nt*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,AL))
    dev_store(out,Temp,ks*24+n[None,:]+rm[:,None]*0,(rm[:,None]==0)&(n[None,:]<24))

@g.jit
def region_packed(R,W,S,B,AN,M,Temp,Raw,X,Counter,Health,BK:gl.constexpr,BN:gl.constexpr,GRID:gl.constexpr):
    if gl.program_id(0)<gl.cdiv(24,BN)*(16384//BK):project_packed(R,W,Temp,BK,BN)
    sync(Counter,Health,GRID,3)
    tail_fused(R,S,B,AN,M,Temp,Raw,X,1)

from rcp_snapshot import tail_fused as tail_rcp
@g.jit
def hint(W,PACKED:gl.constexpr,LOCAL:gl.constexpr):
    p=gl.program_id(0);t=gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
    if LOCAL:
        n=(p//128)*16+t//2
        off=n*16384+(p%128)*128+(t%2)*64
        addr=(W+gl.where((t<32)&(n<24),off,0)).to(gl.uint64)
    else:
        line=p*128+t
        addr=(W+gl.minimum(line,6143)*(128 if PACKED else 64)).to(gl.uint64)
    gl.inline_asm_elementwise('global_prefetch_b8 $1, off scope:SCOPE_DEV th:TH_LOAD_NT\n v_mov_b32 $0, 0',constraints='=v,v',args=[addr],dtype=gl.int32,is_pure=False,pack=1)

@g.jit
def final_region(R,W,S,B,AN,M,Temp,Raw,X,Counter,Health,PACKED:gl.constexpr,PREFETCH:gl.constexpr):
    if PREFETCH==1:
        if gl.program_id(0)<48:hint(W,PACKED,False)
    elif PREFETCH==2:hint(W,PACKED,True)
    if PACKED:project_packed(R,W,Temp,128,16)
    else:project(R,W,Temp,128,16)
    sync(Counter,Health,256,3)
    tail_rcp(R,S,B,AN,M,Temp,Raw,X,1)
