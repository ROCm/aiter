import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_load,dev_store
from tail import norm_rope_body

@g.jit
def project_qb(X,XS,W,WS,P,Count,Cos,Sin,O,TILE,TAIL:gl.constexpr,BN:gl.constexpr,SPLIT:gl.constexpr,DIRECT:gl.constexpr,WAVES:gl.constexpr,BK:gl.constexpr,UNROLL:gl.constexpr,UPSCALE:gl.constexpr=False,REPLICATE:gl.constexpr=False,CACHE:gl.constexpr=".cg",SCALAR:gl.constexpr=False,PACKED:gl.constexpr=False,TDM:gl.constexpr=False,FirstTile=None,Desc=None):
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]] if WAVES==4 else [[0,1]] if WAVES==2 else [],instr_shape=[16,16,128])
    DA:gl.constexpr=gl.DotOperandLayout(0,ML,16)
    DB:gl.constexpr=gl.DotOperandLayout(1,ML,16)
    ASL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DA,[16,BK//32],scale_factor=32)
    BSL:gl.constexpr=gl.amd.gfx1250.get_wmma_scale_layout(DB,[BN,BK//32],scale_factor=32)
    AL:gl.constexpr=gl.to_linear_layout(DA,[16,BK]) if DIRECT else gl.BlockedLayout([1,4],[1,32],[WAVES,1],[1,0])
    BL:gl.constexpr=gl.to_linear_layout(DB,[BK,BN]) if DIRECT else gl.BlockedLayout([4,1],[32,1],[1,WAVES],[0,1])
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL))
    ka=gl.arange(0,BK,layout=gl.SliceLayout(0,AL))
    kb=gl.arange(0,BK,layout=gl.SliceLayout(1,BL))
    rn=gl.arange(0,BN,layout=gl.SliceLayout(0,BL))
    tile=TILE;ks=0
    ns=tile*BN+rn
    acc=gl.full((16,BN),0,gl.float32,ML)
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
                if block<7:
                    gl.amd.gfx1250.tdm.async_wait(1)
                else:
                    gl.amd.gfx1250.tdm.async_wait(0)
                bv=gl.amd.cdna4.async_copy.load_shared_relaxed(FirstTile.index(block%2).permute([1,0]),BL)
                # All readers must finish before this slot is recycled.
                gl.barrier()
                if block<6:
                    gl.amd.gfx1250.tdm.async_load(Desc,[0,(block+2)*128],FirstTile.index(block%2))
            else:
                bv=dev_load(W,wi,cache=CACHE)
            asc=gl.load(XS+block)+gl.full((16,BK//32),0,gl.uint8,ASL)
            bsc=gl.load(WS+(tile*BN//128)*8+block)+gl.full((BN,BK//32),0,gl.uint8,BSL)
            acc=gl.amd.gfx1250.wmma_scaled(gl.convert_layout(av,DA),asc,'e4m3',gl.convert_layout(bv,DB),bsc,'e4m3',acc)
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
