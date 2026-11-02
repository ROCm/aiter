"""Use gfx1250 buffer operations: generic global .cg is only a traffic hint."""
import ast,shutil
from pathlib import Path
r=Path(__file__).parent;out=r/'buffer-scoped';out.mkdir(exist_ok=True)
def parts(expr):
 if isinstance(expr,ast.Name):return expr,ast.Constant(value=0)
 if isinstance(expr,ast.BinOp) and isinstance(expr.op,ast.Add):
  base,offset=parts(expr.left)
  if isinstance(offset,ast.Constant) and offset.value==0:return base,expr.right
  return base,ast.BinOp(left=offset,op=ast.Add(),right=expr.right)
 raise ValueError(ast.unparse(expr))
class DeviceScope(ast.NodeTransformer):
 def visit_Call(self,node):
  self.generic_visit(node);name=ast.unparse(node.func)
  if name in ('gl.load','gl.store'):
   base,offset=parts(node.args[0])
   if name=='gl.load':node.args=[base,offset,*node.args[1:]];target='dev_load'
   else:node.args=[node.args[1],base,offset,*node.args[2:]];target='dev_store'
   node.func=ast.parse(target,mode='eval').body
   node.keywords.append(ast.keyword(arg='cache',value=ast.Constant('.cg')))
  return node
s=(r/'kernel.py').read_text().replace("sem='acq_rel'","sem='relaxed'").replace("sem='acquire'","sem='relaxed'")
s=s.replace('    gl.barrier()\n    ticket =', '''    gl.inline_asm_elementwise("s_wait_storecnt 0\\n s_mov_b32 $0, 0", constraints="=s,~{memory}", args=[], dtype=gl.int32, is_pure=False, pack=1)
    gl.barrier()
    ticket =''')
(out/'kernel.py').write_text('from device_scope_helpers import dev_load, dev_store\n'+ast.unparse(DeviceScope().visit(ast.parse(s)))+'\n')
for name in ['run.py','reference.py','build_split.py','device_scope_helpers.py']:shutil.copy2(r/name,out/name)
(out/'test_sync.py').write_text('from device_scope_helpers import dev_load, dev_store\n'+ast.unparse(DeviceScope().visit(ast.parse((r/'test_sync.py').read_text())))+'\n')
