from stage import stage_qb
from qa import project_fused
from qb import project_qb
from device_scope_helpers import dev_load,dev_store
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

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

@g.jit
def complete(X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,QBW,QBS,QP,QC,QOut,Counter,Health,BN:gl.constexpr=128,GRID:gl.constexpr=256,STYLE:gl.constexpr=1,U:gl.constexpr=4,SEM:gl.constexpr=False,Ready=None,Epoch=None,QAMODE:gl.constexpr=1,PREFETCH:gl.constexpr=False,TDM:gl.constexpr=False):
    pid=gl.program_id(0)
    if TDM:
        SL:gl.constexpr=gl.PaddedSharedLayout.with_identity_for([[128,16]],[128,128],[1,0])
        qbshared=gl.allocate_shared_memory(QBW.dtype.element_ty,[128,128],SL)
        qbdesc=gl.amd.gfx1250.tdm.make_tensor_descriptor(base=QBW+pid*131072,shape=[128,1024],strides=[1024,1],block_shape=[128,128],layout=SL)
    else:
        qbshared=None
        qbdesc=None
    if SEM:
        target=poll_read(Epoch+pid,1)+1
    if pid<192:
        project_fused(X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,True,True,QAMODE,128,16,True,4,128,1,True,True,'.ca',True,False,1,1,128,0,'.ca',READY=SEM,Ready=Ready,QBW=QBW,QBDesc=qbdesc,QBShared=qbshared,PREFETCH=PREFETCH,TDM=TDM)
    else:
        stage_qb(QBW,pid,qbdesc,qbshared,PREFETCH,TDM)
    if SEM:
        ok=wait_target(Ready,Health,target,STYLE)
        dev_store(target,Epoch,pid)
    else:
        ok=sync(Counter,Health,GRID,STYLE)
    # Timeout is diagnostic: all waves continue uniformly; health!=0 means
    # invalid output. Avoid a per-wave success branch around CTA barriers.
    for tile in range(pid,32768//BN,GRID):
        project_qb(LP,LS,QBW,QBS,QP,QC,Cos,Sin,QOut,tile,True,BN,1,True,4,128,U,True,True,'.ca',True,False,TDM=TDM,FirstTile=qbshared)
