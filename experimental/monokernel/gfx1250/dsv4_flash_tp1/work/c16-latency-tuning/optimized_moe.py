from triton.experimental.gluon import language as gl
import os
MOE_LOOKAHEAD = gl.constexpr(int(os.getenv("MOE_LOOKAHEAD","0")))
MOE_N = gl.constexpr(int(os.getenv("MOE_N","256")))
MOE_BUFFERS = gl.constexpr(int(os.getenv("MOE_BUFFERS","2")))
from edge_sync import publish,consume
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from optimized_moe_stage import _moe_gemm_a8w4_decode as stage

from combined_scaled import sync
@g.jit
def phase(Count,Health,GRID:gl.constexpr):
 gl.amd.gfx1250.tdm.async_wait(0)
 return sync(Count,Health,GRID,3)

@g.jit
def fused(X,XS,W1,S1,W2,S2,I,IS,Y,Gather,Hist,Offs,Pad,Map,Gamma,Count,Health,Z,Scatter,
          W1S:gl.constexpr,S1S:gl.constexpr,W2S:gl.constexpr,S2S:gl.constexpr,
          LIMIT:gl.constexpr,M:gl.constexpr,GRID:gl.constexpr,BUFFERS:gl.constexpr=4,Tiles=None):
    gl.static_assert(Tiles is None or MOE_N==256)
    pid=gl.program_id(0)
    for task in range(pid,gl.load(Pad)*(4096//MOE_N),GRID):
        if MOE_LOOKAHEAD:prefetch_next(W1,S1,W1S,S1S,Map,Pad,task,4096)
        stage(I,2048,1,X,4096,1,XS,128,1,
              W1,W1S[0],W1S[1],W1S[2],S1,S1S[0],S1S[1],S1S[2],
              None,None,None,None,None,M,4096,4096,Gather,Hist,Offs,Pad,Map,M*6,4096//MOE_N,
              True,1.0,LIMIT,2,False,6,16,MOE_N,512,1,MOE_BUFFERS,
              'GFX1250_SCALE',True,True,False,4,
              YMxScale=IS,stride_y_mx_m=64,stride_y_mx_n=1,HAS_MX_OUT=True,TaskID=task)
        if Tiles is not None:publish(Tiles+(256+task//16)*64,True)
    if Tiles is None:ok=phase(Count,Health,GRID)
    else:ok=True
    if ok:
        for task in range(pid,gl.load(Pad)*(4096//MOE_N),GRID):
            if Tiles is not None:consume(Tiles+(256+task//16)*64,Health,16,True,1)
            if MOE_LOOKAHEAD:prefetch_next(W2,S2,W2S,S2S,Map,Pad,task,2048)
            stage(Y,4096,1,I,2048,1,IS,64,1,
                  W2,W2S[0],W2S[1],W2S[2],S2,S2S[0],S2S[1],S2S[2],
                  None,None,None,None,Gamma,M*6,4096,2048,None,Hist,Offs,Pad,Map,M*6,4096//MOE_N,
                  False,1.0,LIMIT,1,False,6,16,MOE_N,512,1,MOE_BUFFERS,
                  'GFX1250_SCALE',True,True,False,4,TaskID=task)
            if Tiles is not None:
                expert=gl.load(Map+task//16)&65535
                start=gl.load(Offs+expert);nr=gl.load(Hist+expert)
                for rr in range(nr):publish(Tiles+(320+(start+rr)*16+task%16)*64,True)
    # Every CTA enters phase2 even after timeout, avoiding a second deadlock.
    if Tiles is None:ok2=phase(Count+1,Health,GRID)
    else:ok2=True
    for task in range(pid,M*32,GRID):
        token=task//32;tile=task%32
        cols=tile*128+gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
        acc=gl.full((128,),0,gl.float32,layout=gl.BlockedLayout([1],[32],[4],[0]))
        if ok and ok2:
            for j in gl.static_range(6):
                row=gl.load(Scatter+token*6+j).to(gl.int32)
                if Tiles is not None:consume(Tiles+(320+row*16+tile//2)*64,Health,1,True,1)
                vals=gl.amd.gfx1250.buffer_load(Y,row*4096+cols,cache='.cg')
                acc+=vals.to(gl.float32)
        else:
            acc=gl.full((128,),float('nan'),gl.float32,layout=gl.BlockedLayout([1],[32],[4],[0]))
        gl.store(Z+token*4096+cols,acc)

@g.jit
def prefetch_next(W,S,WS:gl.constexpr,SS:gl.constexpr,Map,Pad,Task,K:gl.constexpr):
 ntiles:gl.constexpr=4096//MOE_N
 next_task=Task+256
 active=next_task<gl.load(Pad)*ntiles
 packed=gl.load(Map+next_task//ntiles,active,0)
 expert=packed&65535
 tile=next_task%ntiles
 SL:gl.constexpr=gl.SwizzledSharedLayout(1,1,1,[1,0])
 wd=gl.amd.gfx1250.tdm.make_tensor_descriptor(W+expert*WS[0],[4096//16,K//2*16],[WS[1],WS[2]],[MOE_N//16,512//2*16],SL)
 sd=gl.amd.gfx1250.tdm.make_tensor_descriptor(S+expert*SS[0],[4096//32,K],[SS[1],SS[2]],[MOE_N//32,512],SL)
 gl.amd.gfx1250.tdm.prefetch(wd,[tile*(MOE_N//16),0],pred=active,speculative=False)
 gl.amd.gfx1250.tdm.prefetch(sd,[tile*(MOE_N//32),0],pred=active,speculative=False)
