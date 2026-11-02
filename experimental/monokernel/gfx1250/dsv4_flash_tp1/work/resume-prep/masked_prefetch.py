from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
@g.jit
def masked_line_prefetch(Base,ByteOffset,NBytes):
 address=(Base.to(gl.uint64)+ByteOffset.to(gl.uint64))
 valid=(ByteOffset>=0)&(ByteOffset<NBytes)
 gl.inline_asm_elementwise(
  'v_cmp_ne_u32_e64 $0, $3, 0\n'
  's_and_saveexec_b32 $0, $0\n'
  'global_prefetch_b8 $2, off scope:SCOPE_DEV th:TH_LOAD_NT\n'
  's_mov_b32 exec_lo, $0\n'
  'v_mov_b32 $1, 0',
  constraints='=&s,=v,v,v,~{memory}',args=[address,valid.to(gl.uint32)],
  dtype=(gl.uint32,gl.int32),is_pure=False,pack=1)
@g.jit
def test_prefetch(Base,NBytes:gl.constexpr):
 t=gl.program_id(0)*128+gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[4],[0]))
 masked_line_prefetch(Base,t*256,NBytes)
