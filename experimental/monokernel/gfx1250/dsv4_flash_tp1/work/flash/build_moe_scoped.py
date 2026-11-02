import ast
from pathlib import Path
r=Path(__file__).parent
class Scope(ast.NodeTransformer):
 def visit_Call(self,node):
  self.generic_visit(node)
  name=ast.unparse(node.func)
  if name in ('gl.amd.gfx1250.tdm.async_load','gl.amd.gfx1250.tdm.async_store'):
   node.keywords.append(ast.keyword(arg='cache_modifier',value=ast.Constant('.cg')))
  if name in ('gl.amd.gfx1250.buffer_load','gl.amd.gfx1250.buffer_store'):
   node.keywords.append(ast.keyword(arg='cache',value=ast.Constant('.cg')))
  return node
s=ast.unparse(Scope().visit(ast.parse((r/'moe_aiter_original.py').read_text())))
s=s.replace('num_tokens.to(gl.uint16)','gl.cast(num_tokens, gl.uint16)')
(r/'moe_aiter_scoped.py').write_text(s+'\n')
s=(r/'moe_stage_prototype.py').read_text().replace('from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a8w4 import','from moe_aiter_scoped import')
s=s.replace("'GFX1250_SCALE',False,True,False,4", "'GFX1250_SCALE',True,True,False,4")
s=s.replace("sem='acq_rel'","sem='relaxed'").replace("sem='acquire'","sem='relaxed'")
s=s.replace('        gl.barrier()', '''        gl.inline_asm_elementwise("s_wait_storecnt 0\\n s_mov_b32 $0, 0",constraints="=s,s,~{memory}",args=[expert_slot],dtype=gl.int32,is_pure=False,pack=1)
        gl.barrier()''')
(r/'moe_stage_scoped.py').write_text(s)
s=(r/'test_moe_stage.py').read_text().replace('from moe_stage_prototype import','from moe_stage_scoped import').replace("'moe_stage_results.json'","'moe_stage_scoped_results.json'").replace("f'moe-stage-b","f'moe-stage-scoped-b")
(r/'test_moe_scoped.py').write_text(s)
