"""Experimental C1 routed FP4 MoE, reusing exact AITER stage math."""
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from moe_aiter_scoped import _moe_gemm_a8w4_decode as stage

@g.jit
def fused(X,XS,W1,S1,W2,S2,I,IS,Y,Gather,Hist,Offs,Pad,Map,Gamma,Count,Health,Z,Scatter,Ready,
          W1S:gl.constexpr,S1S:gl.constexpr,W2S:gl.constexpr,S2S:gl.constexpr,
          LIMIT:gl.constexpr,BUFFERS:gl.constexpr=2,FUSE:gl.constexpr=True):
    stage(I,2048,1,X,4096,1,XS,128,1,
          W1,W1S[0],W1S[1],W1S[2],S1,S1S[0],S1S[1],S1S[2],
          None,None,None,None,None,1,4096,4096,Gather,Hist,Offs,Pad,Map,6,32,
          True,1.0,LIMIT,2,False,6,16,128,256,1,BUFFERS,
          'GFX1250_SCALE',True,True,False,4,
          YMxScale=IS,stride_y_mx_m=64,stride_y_mx_n=1,HAS_MX_OUT=True)
    if FUSE:
        expert_slot=gl.program_id(0)//32
        gl.amd.gfx1250.tdm.async_wait(0)
        gl.inline_asm_elementwise("s_wait_storecnt 0\n s_mov_b32 $0, 0",constraints="=s,s,~{memory}",args=[expert_slot],dtype=gl.int32,is_pure=False,pack=1)
        gl.barrier()
        ticket=gl.atomic_add(Count+expert_slot,1,sem='relaxed',scope='gpu')
        target=(ticket//32+1)*32
        seen=gl.atomic_add(Count+expert_slot,0,sem='relaxed',scope='gpu')
        spins=0
        while (seen!=target)&(spins<100000):
            seen=gl.atomic_add(Count+expert_slot,0,sem='relaxed',scope='gpu')
            spins+=1
        if seen!=target:
            gl.atomic_or(Health,1,sem='relaxed',scope='gpu')
        else:
            stage(Y,4096,1,I,2048,1,IS,64,1,
                  W2,W2S[0],W2S[1],W2S[2],S2,S2S[0],S2S[1],S2S[2],
                  None,None,None,None,Gamma,6,4096,2048,None,Hist,Offs,Pad,Map,6,32,
                  False,1.0,LIMIT,1,False,6,16,128,256,1,BUFFERS,
                  'GFX1250_SCALE',True,True,False,4)

            gl.amd.gfx1250.tdm.async_wait(0)
            gl.inline_asm_elementwise("s_wait_storecnt 0\n s_mov_b32 $0, 0",constraints="=s,s,~{memory}",args=[expert_slot],dtype=gl.int32,is_pure=False,pack=1)
            gl.barrier()
            tile=gl.program_id(0)%32
            old=gl.atomic_or(Ready+tile,1<<expert_slot,sem='relaxed',scope='gpu')
            if (old|(1<<expert_slot))==63:
                cols=tile*128+gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
                acc=gl.full((128,),0,gl.float32,layout=gl.BlockedLayout([1],[32],[4],[0]))
                for j in gl.static_range(6):
                    row=gl.load(Scatter+j)
                    vals=gl.amd.gfx1250.buffer_load(Y,row*4096+cols,cache='.cg')
                    acc+=vals.to(gl.float32)
                gl.store(Z+cols,acc)
                gl.atomic_xchg(Ready+tile,0,sem='relaxed',scope='gpu')
