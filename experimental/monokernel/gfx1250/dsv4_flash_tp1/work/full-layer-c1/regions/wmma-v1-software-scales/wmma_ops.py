from kernel import *

@g.jit
def wmma_matvec(X,W,S,Y,N:gl.constexpr,K:gl.constexpr,GRID:gl.constexpr,BN:gl.constexpr=32):
    ML:gl.constexpr=gl.amd.AMDWMMALayout(version=3,transposed=True,warp_bases=[[0,1],[0,2]],instr_shape=[16,16,64])
    AL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    BL:gl.constexpr=gl.BlockedLayout([4,1],[32,1],[1,4],[0,1])
    CL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[1,0])
    rm=gl.arange(0,16,layout=gl.SliceLayout(1,AL))
    ka=gl.arange(0,128,layout=gl.SliceLayout(0,AL))
    kb=gl.arange(0,128,layout=gl.SliceLayout(1,BL))
    rn=gl.arange(0,BN,layout=gl.SliceLayout(0,BL))
    for tile in range(gl.program_id(0),gl.cdiv(N,BN),GRID):
        ns=tile*BN+rn
        acc=gl.full((16,BN),0,gl.float32,ML)
        for kblock in range(K//128):
            x=dev_load(X,kblock*128+ka,cache='.cg')
            amax=gl.maximum(gl.max(gl.abs(x),0),1e-10)
            bits=(amax/448.).to(gl.uint32,bitcast=True)
            exp=gl.minimum(((bits&2147483647)+8388607)>>23,254)
            xs=(exp<<23).to(gl.float32,bitcast=True)
            av=gl.where(rm[:,None]==0,(x/xs)[None,:],0).to(gl.float8e4nv)
            bv=dev_load(W,ns[None,:]*K+kblock*128+kb[:,None],cache='.cg')
            aa=gl.convert_layout(av,gl.DotOperandLayout(0,ML,16))
            bb=gl.convert_layout(bv,gl.DotOperandLayout(1,ML,16))
            part=gl.amd.gfx1250.wmma(aa,bb,gl.full((16,BN),0,gl.float32,ML))
            ws=dev_load(S,(ns//128)*(K//128)+kblock,cache='.cg').to(gl.uint32)
            ws=(ws<<23).to(gl.float32,bitcast=True)
            scale=gl.convert_layout(ws[None,:]+gl.full((16,BN),0,gl.float32,BL),ML)
            acc=acc+part*(scale*xs)
        out=gl.convert_layout(acc,CL)
        om=gl.arange(0,16,layout=gl.SliceLayout(1,CL))
        on=tile*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,CL))
        dev_store(out.to(gl.bfloat16),Y,on[None,:]+om[:,None]*0,om[:,None]==0,cache='.cg')
