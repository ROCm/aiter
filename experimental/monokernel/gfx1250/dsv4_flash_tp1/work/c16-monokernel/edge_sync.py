"""Producer publication and device-scope consumer polling.

Full-layer counters reset at the existing entry barrier, then use target=N.
No edge may be reused within an invocation. The kernel must run stream-ordered
with exclusive ownership of its state. Standalone callers may instead supply a
monotonic invocation epoch if each counter has the same producers every call.
"""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from combined_scaled import wait_target,clock,poll_read
from device_scope_helpers import dev_load,dev_store
from masked_prefetch import masked_line_prefetch

@g.jit
def poll_edge(E,STYLE:gl.constexpr):
 if STYLE==4:
  return gl.inline_asm_elementwise('global_load_b32 $0, $1, off scope:SCOPE_DEV th:TH_LOAD_NT\n s_wait_loadcnt 0',constraints='=v,v,~{memory}',args=[E.to(gl.uint64)],dtype=gl.uint32,is_pure=False,pack=1)
 return poll_read(E,STYLE)

@g.jit
def publish(E,IS_PRODUCER):
 if IS_PRODUCER:
  gl.inline_asm_elementwise('s_wait_storecnt 0\n s_mov_b32 $0, 0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
  gl.barrier()
  ticket=gl.atomic_add(E,1,sem='release',scope='gpu')
  gl.inline_asm_elementwise('v_mov_b32 $0, $1',constraints='=v,v,~{memory}',args=[ticket],dtype=gl.int32,is_pure=False,pack=1)

@g.jit
def consume(E,H,N:gl.constexpr,IS_CONSUMER,Epoch,Debug=None,POLL_STYLE:gl.constexpr=4):
 if IS_CONSUMER:
  p=gl.program_id(0)
  target=gl.cast(Epoch,gl.uint32)*N
  start=clock();spins=0;seen=poll_edge(E,POLL_STYLE)
  while ((seen-target).to(gl.int32)<0)&(spins<500000)&(clock()-start<100000000):
   gl.inline_asm_elementwise('s_sleep 1\n s_mov_b32 $0, 0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)
   seen=poll_edge(E,POLL_STYLE);spins+=1
  if (seen-target).to(gl.int32)<0:
   gl.atomic_or(H,2|(gl.cast(1,gl.int32)<<gl.cast((E.to(gl.uint64)//1280)%20+4,gl.int32)),sem='relaxed',scope='gpu')
   if Debug is not None:dev_store(seen+1,Debug,p)
  gl.barrier()
  # Order subsequent reads after the device-coherent completion load.
  gl.inline_asm_elementwise('s_wait_loadcnt 0\n global_inv scope:SCOPE_DEV\n s_wait_loadcnt 0\n s_mov_b32 $0, 0',constraints='=s,~{memory}',args=[],dtype=gl.int32,is_pure=False,pack=1)


@g.jit
def edge(E,H,N:gl.constexpr,IS_PRODUCER,IS_CONSUMER,Epoch):
 publish(E,IS_PRODUCER)
 consume(E,H,N,IS_CONSUMER,Epoch)

@g.jit
def prefetch_bytes(P,BYTES:gl.constexpr):
 # Masked non-speculative L2 hint; actual tile transfers remain TDM.
 p=gl.program_id(0)
 L:gl.constexpr=gl.BlockedLayout([1],[32],[4],[0])
 i=gl.arange(0,128,layout=L)
 for chunk in gl.static_range(gl.cdiv(BYTES,256*128*256)):
  off=((chunk*256+p)*128+i)*256
  masked_line_prefetch(P,off,BYTES)
