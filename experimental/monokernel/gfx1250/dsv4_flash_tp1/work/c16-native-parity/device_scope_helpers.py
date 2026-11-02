import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

@g.jit
def dev_load(ptr,offset,mask=None,other=None,cache:gl.constexpr='.cg'):
 off=gl.cast(offset,gl.int32)
 if len(off.shape)==0:
  one=gl.arange(0,1,layout=gl.BlockedLayout([1],[32],[gl.num_warps()],[0]))
  m=one==0
  if mask is not None:m=m&mask
  v=gl.amd.gfx1250.buffer_load(ptr,off+one,mask=m,other=other,cache=cache)
  return gl.sum(v,0).to(v.dtype)
 else:
  return gl.amd.gfx1250.buffer_load(ptr,off,mask=mask,other=other,cache=cache)

@g.jit
def dev_store(value,ptr,offset,mask=None,cache:gl.constexpr='.cg'):
 value=gl.cast(value,ptr.dtype.element_ty)
 off=gl.cast(offset,gl.int32)
 if len(off.shape)==0:
  one=gl.arange(0,1,layout=gl.BlockedLayout([1],[32],[gl.num_warps()],[0]))
  m=one==0
  if mask is not None:m=m&mask
  gl.amd.gfx1250.buffer_store(value+gl.full((1,),0,value.dtype,gl.BlockedLayout([1],[32],[gl.num_warps()],[0])),ptr,off+one,mask=m,cache=cache)
 else:
  gl.amd.gfx1250.buffer_store(value,ptr,off,mask=mask,cache=cache)
