import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_load,dev_store
from tail import norm_rope_body

@g.jit
def project_qb(X,XS,W,WS,P,Count,Cos,Sin,O,TILE,TAIL:gl.constexpr,BN:gl.constexpr,SPLIT:gl.constexpr,DIRECT:gl.constexpr,WAVES:gl.constexpr,BK:gl.constexpr,UNROLL:gl.constexpr,UPSCALE:gl.constexpr=False,REPLICATE:gl.constexpr=False,CACHE:gl.constexpr=".cg",SCALAR:gl.constexpr=False,PACKED:gl.constexpr=False,TDM:gl.constexpr=False,FirstTile=None):
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]] if WAVES==4 else [[0,1]] if WAVES==2 else [],instr_shape=[16,16,32])
    DA:gl.constexpr=gl.DotOperandLayout(0,ML,8)
    DB:gl.constexpr=gl.DotOperandLayout(1,ML,8)
    AL:gl.constexpr=gl.to_linear_layout(DA,[16,BK]) if DIRECT else gl.BlockedLayout([1,4],[1,32],[WAVES,1],[1,0])
    BL:gl.constexpr=gl.to_linear_layout(DB,[BK,BN]) if DIRECT else gl.BlockedLayout([4,1],[32,1],[1,WAVES],[0,1])
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL))
    ka=gl.arange(0,BK,layout=gl.SliceLayout(0,AL))
    kb=gl.arange(0,BK,layout=gl.SliceLayout(1,BL))
    rn=gl.arange(0,BN,layout=gl.SliceLayout(0,BL))
    tile=TILE;ks=0
    ns=tile*BN+rn
    acc=gl.full((16,BN),0,gl.float32,ML)
    if TDM:
        gl.amd.gfx1250.tdm.async_wait(0)
    for chunk in range(1024//BK//UNROLL):
        for u in gl.static_range(UNROLL):
            block=chunk*UNROLL+u
            if REPLICATE:av=dev_load(X,block*BK+ka[None,:]+rm[:,None]*0,cache=CACHE)
            else:av=dev_load(X,block*BK+ka[None,:]+rm[:,None]*0,rm[:,None]==0,0.,cache=CACHE)
            if PACKED:
                kk=block*BK+kb[:,None]
                wi=((ns[None,:]//16)*128+kk//32)*512+(ns[None,:]%16)*32+kk%32
            else:wi=ns[None,:]*1024+block*BK+kb[:,None]
            if TDM:
                if block==0:
                    bv=gl.amd.cdna4.async_copy.load_shared_relaxed(FirstTile.permute([1,0]),BL)
                else:
                    bv=dev_load(W,wi,cache=CACHE)
            else:
                bv=dev_load(W,wi,cache=CACHE)
            if SCALAR and BK==128 and BN<=128:
                ax=gl.load(XS+block)+gl.full((16,BK),0,gl.uint8,AL)
                bx=gl.load(WS+(tile*BN//128)*8+block)+gl.full((BK,BN),0,gl.uint8,BL)
            else:
                ax=dev_load(XS,(block*BK+ka[None,:])//128,cache=CACHE)
                bx=dev_load(WS,(ns[None,:]//128)*8+(block*BK+kb[:,None])//128,cache=CACHE)
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
    OL:gl.constexpr=gl.to_linear_layout(ML,[16,BN])
    out=gl.convert_layout(acc,OL)
    om=gl.arange(0,16,layout=gl.SliceLayout(1,OL))
    on=tile*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,OL))
    dev_store(out.to(gl.bfloat16).to(gl.float32),P,on[None,:]+om[:,None]*0,om[:,None]==0)
    if TAIL:
        gl.inline_asm_elementwise('s_wait_loadcnt 0\n s_wait_storecnt 0\n s_mov_b32 $0, 0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
        gl.barrier()
        head=tile//(512//BN)
        ticket=gl.atomic_add(Count+head,1,sem='relaxed',scope='gpu')
        if ticket==512//BN-1:
            norm_rope_body(P,Cos,Sin,O,head,True,WAVES)
            gl.atomic_xchg(Count+head,0,sem='relaxed',scope='gpu')


@g.jit
def qb_kernel(X,XS,W,WS,P,Count,Cos,Sin,O,TAIL:gl.constexpr,BN:gl.constexpr=128,WAVES:gl.constexpr=4,U:gl.constexpr=1):
    project_qb(X,XS,W,WS,P,Count,Cos,Sin,O,gl.program_id(0),TAIL,BN,1,True,WAVES,128,U,True,True,'.ca',True,False)
