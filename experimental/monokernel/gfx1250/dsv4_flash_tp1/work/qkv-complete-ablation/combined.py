from qa import project_fused
from qb import project_qb
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def clock():
    return gl.inline_asm_elementwise('s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\n s_wait_kmcnt 0', constraints='=s', args=[], dtype=gl.uint64, is_pure=False, pack=1)

@g.jit
def sync(Counter, Health, GRID: gl.constexpr):
    gl.inline_asm_elementwise('s_wait_storecnt 0\n s_mov_b32 $0, 0', constraints='=s,~{memory}', args=[], dtype=gl.int32, is_pure=False, pack=1)
    gl.barrier()
    ticket = gl.atomic_add(Counter, 1, sem='relaxed', scope='gpu').to(gl.uint32)
    target = (ticket // GRID + 1) * GRID
    seen = gl.atomic_add(Counter, 0, sem='relaxed', scope='gpu')
    start = clock()
    spins = 0
    while ((seen - target).to(gl.int32) < 0) & (spins < 1000000) & (clock() - start < 1000000000):
        seen = gl.atomic_add(Counter, 0, sem='relaxed', scope='gpu')
        spins += 1
    if (seen - target).to(gl.int32) < 0:
        gl.atomic_or(Health, 2, sem='relaxed', scope='gpu')
    gl.barrier()


@g.jit
def complete(X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,QBW,QBS,QP,QC,QOut,Counter,Health,BN:gl.constexpr=128,GRID:gl.constexpr=192):
    if gl.program_id(0)<192:
        project_fused(X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,True,True,1,128,16,True,4,128,1,True,True,'.ca',True,False,1,1,128,0,'.ca')
    sync(Counter,Health,GRID)
    for tile in range(gl.program_id(0),32768//BN,GRID):
        project_qb(LP,LS,QBW,QBS,QP,QC,Cos,Sin,QOut,tile,True,BN,1,True,4,128,1,True,True,'.ca',True,False)
