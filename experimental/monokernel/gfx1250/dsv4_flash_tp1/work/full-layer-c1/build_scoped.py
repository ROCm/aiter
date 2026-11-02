"""Device-scope coherent-L2 variant; explicit store drain + relaxed semaphore."""
import ast,shutil
from pathlib import Path
r=Path(__file__).parent;out=r/'scoped';out.mkdir(exist_ok=True)
class DeviceScope(ast.NodeTransformer):
 def visit_Call(self,node):
  self.generic_visit(node)
  name=ast.unparse(node.func)
  if name in ('gl.load','gl.store'):
   node.keywords.append(ast.keyword(arg='cache_modifier',value=ast.Constant('.cg')))
  return node
s=(r/'kernel.py').read_text().replace("sem='acq_rel'","sem='relaxed'").replace("sem='acquire'","sem='relaxed'")
s=s.replace('    gl.barrier()\n    ticket =', '''    gl.inline_asm_elementwise("s_wait_storecnt 0\\n s_mov_b32 $0, 0", constraints="=s,~{memory}", args=[], dtype=gl.int32, is_pure=False, pack=1)
    gl.barrier()
    ticket =''')
s=ast.unparse(DeviceScope().visit(ast.parse(s)))
(out/'kernel.py').write_text(s+'\n')
for name in ['run.py','reference.py','build_split.py']:shutil.copy2(r/name,out/name)
(out/'test_sync.py').write_text(ast.unparse(DeviceScope().visit(ast.parse((r/'test_sync.py').read_text())))+'\n')
