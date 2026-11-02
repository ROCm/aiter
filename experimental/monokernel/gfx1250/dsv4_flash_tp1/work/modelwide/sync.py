from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from device_scope_helpers import dev_load,dev_store
@g.jit
def clock():
    return gl.inline_asm_elementwise('s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\n s_wait_kmcnt 0',constraints='=s',args=[],dtype=gl.uint64,is_pure=False,pack=1)

@g.jit
def poll_read(P,STYLE:gl.constexpr):
    if STYLE==0:
        v=gl.atomic_add(P,0,sem='relaxed',scope='gpu')
    elif STYLE==1:
        v=gl.inline_asm_elementwise('s_load_b32 $0, $1, 0 scope:SCOPE_DEV th:TH_LOAD_RT\n s_wait_kmcnt 0',constraints='=s,s,~{memory}',args=[P.to(gl.uint64)],dtype=gl.uint32,is_pure=False,pack=1)
    elif STYLE==2:
        v=gl.inline_asm_elementwise('s_load_b32 $0, $1, 0 scope:SCOPE_DEV th:TH_LOAD_HT\n s_wait_kmcnt 0',constraints='=s,s,~{memory}',args=[P.to(gl.uint64)],dtype=gl.uint32,is_pure=False,pack=1)
    else:
        v=dev_load(P,0)
    return v.to(gl.uint32)

@g.jit
def wait_target(P,Health,target,STYLE:gl.constexpr,Debug=None):
    start=clock();spins=0
    seen=poll_read(P,STYLE)
    while ((seen-target).to(gl.int32)<0)&(spins<10000000)&(clock()-start<1000000000):
        seen=poll_read(P,STYLE);spins+=1
    ok=(seen-target).to(gl.int32)>=0
    if not ok:
        gl.atomic_or(Health,2,sem='relaxed',scope='gpu')
        if Debug is not None:
            dev_store(target,Debug,gl.program_id(0)*4)
            dev_store(seen,Debug,gl.program_id(0)*4+1)
            dev_store(spins,Debug,gl.program_id(0)*4+2)
            dev_store(clock()-start,Debug,gl.program_id(0)*4+3)
    gl.barrier()
    return ok

@g.jit
def sync(Counter,Health,GRID:gl.constexpr,STYLE:gl.constexpr):
    gl.inline_asm_elementwise('s_wait_loadcnt 0\n s_wait_storecnt 0\n s_mov_b32 $0, 0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
    gl.barrier()
    ticket=gl.atomic_add(Counter,1,sem='relaxed',scope='gpu').to(gl.uint32)
    target=(ticket//GRID+1)*GRID
    return wait_target(Counter,Health,target,STYLE)

