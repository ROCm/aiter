from stage import stage_qb
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_load,dev_store
from tail import quant_scale

@g.jit
def project_fused(X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,INLINE:gl.constexpr,TAIL:gl.constexpr,MODE:gl.constexpr,BN:gl.constexpr,SPLIT:gl.constexpr,DIRECT:gl.constexpr,WAVES:gl.constexpr,BK:gl.constexpr,UNROLL:gl.constexpr,UPSCALE:gl.constexpr=False,REPLICATE:gl.constexpr=False,CACHE:gl.constexpr=".cg",SCALAR:gl.constexpr=False,PACKED:gl.constexpr=False,OUTLAY:gl.constexpr=0,NVEC:gl.constexpr=1,COUNTSTRIDE:gl.constexpr=1,NATIVE:gl.constexpr=False,NORMCACHE:gl.constexpr=".cg",READY:gl.constexpr=False,Ready=None,QBW=None,QBDesc=None,QBShared=None,PREFETCH:gl.constexpr=False,TDM:gl.constexpr=False,ALLTDM:gl.constexpr=False):
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2],[0,4]] if WAVES==8 else [[0,1],[0,2]] if WAVES==4 else [[0,1]] if WAVES==2 else [],instr_shape=[16,16,128] if NATIVE else [16,16,32])
    DA:gl.constexpr=gl.DotOperandLayout(0,ML,16 if NATIVE else 8)
    DB:gl.constexpr=gl.DotOperandLayout(1,ML,16 if NATIVE else 8)
    AL:gl.constexpr=gl.to_linear_layout(DA,[16,BK]) if DIRECT else gl.BlockedLayout([1,4],[1,32],[WAVES,1],[1,0])
    BL:gl.constexpr=gl.to_linear_layout(DB,[BK,BN]) if DIRECT else gl.BlockedLayout([4,1],[32,1],[1,WAVES],[0,1])
    if NATIVE==2:
        ASL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DA,[16,BK//32],scale_factor=32)
        BSL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DB,[BN,BK//32],scale_factor=32)
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL))
    ka=gl.arange(0,BK,layout=gl.SliceLayout(0,AL))
    kb=gl.arange(0,BK,layout=gl.SliceLayout(1,BL))
    rn=gl.arange(0,BN,layout=gl.SliceLayout(0,BL))
    tile=gl.program_id(0)//SPLIT;ks=gl.program_id(0)%SPLIT
    ns=tile*BN+rn
    if ALLTDM:
        QSL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[BK,16]],[BN,BK],[1,0])
        qsm=gl.allocate_shared_memory(W.dtype.element_ty,[2,BN,BK],QSL)
        qdesc=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=W+tile*BN*4096+ks*256,shape=[BN,256],strides=[4096,1],block_shape=[BN,BK],layout=QSL)
        for j in gl.static_range(2):
            gl.amd.gfx1250.tdm.async_load(qdesc,[0,j*BK],qsm.index(j))
        gl.amd.gfx1250.tdm.async_wait(0)
    acc=gl.full((16,BN),0,gl.float32,ML)
    for chunk in range(ks*(4096//SPLIT//BK//UNROLL),(ks+1)*(4096//SPLIT//BK//UNROLL)):
        for u in gl.static_range(UNROLL):
            block=chunk*UNROLL+u
            if INLINE:
                iq=gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[WAVES],[0]))
                xv=dev_load(X,block*128+iq,cache=CACHE).to(gl.float32)
                e,scale=quant_scale(xv)
                qv=(xv/scale).to(gl.float8e4nv)
                qv=gl.convert_layout(qv,gl.SliceLayout(0,AL))
                av=(qv.to(gl.uint8,bitcast=True)[None,:]+gl.full((16,128),0,gl.uint8,AL)).to(gl.float8e4nv,bitcast=True)
            elif REPLICATE:av=dev_load(X,block*BK+ka[None,:]+rm[:,None]*0,cache=CACHE)
            else:av=dev_load(X,block*BK+ka[None,:]+rm[:,None]*0,rm[:,None]==0,0.,cache=CACHE)
            if PACKED:
                kk=block*BK+kb[:,None]
                wi=((ns[None,:]//16)*128+kk//32)*512+(ns[None,:]%16)*32+kk%32
            else:wi=ns[None,:]*4096+block*BK+kb[:,None]
            if ALLTDM:
                bv=gl.amd.cdna4.async_copy.load_shared_relaxed(qsm.index(block-ks*2).permute([1,0]),BL)
            else:
                bv=dev_load(W,wi,cache=CACHE)
            if NATIVE:
                sa=(gl.load(XS+block).to(gl.uint32)<<23).to(gl.float32,bitcast=True)
                sb=(gl.load(WS+(tile*BN//128)*32+block).to(gl.uint32)<<23).to(gl.float32,bitcast=True)
                aa=gl.convert_layout(av,DA);bb=gl.convert_layout(bv,DB)
                if NATIVE==2:
                    asc=gl.load(XS+block)+gl.full((16,BK//32),0,gl.uint8,ASL)
                    bsc=gl.load(WS+(tile*BN//128)*32+block)+gl.full((BN,BK//32),0,gl.uint8,BSL)
                    acc=gl.amd.gfx1250.wmma_scaled(aa,asc,'e4m3',bb,bsc,'e4m3',acc)
                else:
                    temp=gl.amd.gfx1250.wmma_scaled(aa,None,'e4m3',bb,None,'e4m3',gl.full((16,BN),0,gl.float32,ML))
                    acc+=temp*(sa*sb)
            else:
                if SCALAR and BK==128:
                    ax=(e if INLINE else gl.load(XS+block))+gl.full((16,BK),0,gl.uint8,AL)
                    bx=gl.load(WS+(tile*BN//128)*32+block)+gl.full((BK,BN),0,gl.uint8,BL)
                else:
                    ax=dev_load(XS,(block*BK+ka[None,:])//128,cache=CACHE)
                    bx=dev_load(WS,(ns[None,:]//128)*32+(block*BK+kb[:,None])//128,cache=CACHE)
                if UPSCALE:
                    a16=gl.amd.gfx1250.scaled_upcast(av,ax+gl.full((16,BK),0,gl.uint8,AL),gl.bfloat16)
                    if W.dtype.element_ty==gl.bfloat16:b16=bv
                    else:b16=gl.amd.gfx1250.scaled_upcast(bv,bx,gl.bfloat16)
                else:
                    ax=(ax.to(gl.uint32)<<23).to(gl.float32,bitcast=True)
                    bx=(bx.to(gl.uint32)<<23).to(gl.float32,bitcast=True)
                    a16=(av.to(gl.float32)*ax).to(gl.bfloat16)
                    b16=(bv.to(gl.float32)*bx).to(gl.bfloat16)
                aa=gl.convert_layout(a16,DA)
                bb=gl.convert_layout(b16,DB)
                acc=gl.amd.gfx1250.wmma(aa,bb,acc)
    OL:gl.constexpr=gl.to_linear_layout(ML,[16,BN]) if OUTLAY==0 else gl.BlockedLayout([1,1],[2,16],[1,WAVES],[1,0]) if OUTLAY==1 else gl.BlockedLayout([1,1],[1,32],[WAVES,1],[1,0]) if OUTLAY>=3 else gl.BlockedLayout([1,1],[1,32],[1,WAVES],[1,0])
    out=gl.convert_layout(acc,OL)
    om=gl.arange(0,16,layout=gl.SliceLayout(1,OL))
    on=tile*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,OL))
    if MODE!=1:dev_store(out,P,ks*1536+on[None,:]+om[:,None]*0,om[:,None]==0)
    else:gl.atomic_add(Acc+on[None,:]+om[:,None]*0,out,sem='relaxed',scope='gpu',mask=om[:,None]==0)
    gl.inline_asm_elementwise('s_wait_loadcnt 0\n s_wait_storecnt 0\n s_mov_b32 $0, 0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
    if WAVES>1 and OUTLAY!=3:gl.barrier()
    group=gl.where(tile*BN<1024,0,1)
    ticket=gl.atomic_add(Count+group*COUNTSTRIDE,1,sem='relaxed',scope='gpu')
    if PREFETCH or TDM:
        stage_qb(QBW,gl.program_id(0),QBDesc,QBShared,PREFETCH,TDM)
    if ticket==gl.where(group==0,1024//BN*SPLIT-1,512//BN*SPLIT-1):
        if group==0:
            finish_fused(P,Acc,NW,Count,Y,LP,LS,Cos,Sin,KVOut,TAIL,0,1024,SPLIT,WAVES,MODE,NVEC,COUNTSTRIDE,NORMCACHE)
            if READY:
                gl.inline_asm_elementwise('s_wait_loadcnt 0\n s_wait_storecnt 0\n s_mov_b32 $0, 0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
                gl.barrier()
                gl.atomic_add(Ready,1,sem='relaxed',scope='gpu')
        else:finish_fused(P,Acc,NW,Count,Y,LP,LS,Cos,Sin,KVOut,TAIL,1024,512,SPLIT,WAVES,MODE,NVEC,COUNTSTRIDE,NORMCACHE)

@g.jit
def finish_fused(P,Acc,NW,Count,Y,LP,LS,Cos,Sin,KVOut,TAIL:gl.constexpr,BASE:gl.constexpr,N:gl.constexpr,SPLIT:gl.constexpr,WAVES:gl.constexpr,MODE:gl.constexpr,NVEC:gl.constexpr,COUNTSTRIDE:gl.constexpr,NORMCACHE:gl.constexpr):
    L:gl.constexpr=gl.BlockedLayout([NVEC],[32],[WAVES],[0])
    i=gl.arange(0,N,layout=L)
    if MODE!=1:
        x=gl.full((N,),0.,gl.float64 if MODE==2 else gl.float32,L)
        for s in gl.static_range(SPLIT):x+=dev_load(P,s*1536+BASE+i)
    else:x=dev_load(Acc,BASE+i)
    x=x.to(gl.bfloat16).to(gl.float32)
    y=x*gl.rsqrt(gl.sum(x*x,0)/N+1.e-6)*dev_load(NW,BASE+i,cache=NORMCACHE)
    dev_store(y,Y,BASE+i)
    if TAIL:
        if BASE==0:
            QL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[0,1])
            yy=gl.convert_layout(y,gl.SliceLayout(1,gl.BlockedLayout([1,1],[32,1],[4,1],[0,1]))) if False else y
            yy=gl.convert_layout(gl.reshape(yy,(8,128)),QL)
            amax=gl.maximum(gl.max(gl.abs(yy),1),1.e-10)/448.
            e=gl.minimum(((amax.to(gl.uint32,bitcast=True)&2147483647)+8388607)>>23,254)
            scale=(e<<23).to(gl.float32,bitcast=True)
            q=(yy/scale[:,None]).to(gl.float8e4nv)
            rr=gl.arange(0,8,layout=gl.SliceLayout(1,QL))
            cc=gl.arange(0,128,layout=gl.SliceLayout(0,QL))
            dev_store(q,LP,rr[:,None]*128+cc[None,:])
            dev_store(e.to(gl.uint8),LS,rr)
        else:
            other=gl.gather(y,i^1,0)
            c=dev_load(Cos,(i-448)//2,i>=448,1.,cache='.ca')
            sn=dev_load(Sin,(i-448)//2,i>=448,0.,cache='.ca')
            dev_store(y*c+gl.where(i%2==0,-1.,1.)*other*sn,KVOut,i)
    if MODE==1:dev_store(0.,Acc,BASE+i)
    gl.atomic_xchg(Count+(BASE//1024)*COUNTSTRIDE,0,sem='relaxed',scope='gpu')
