import ast,shutil
from pathlib import Path
r=Path(__file__).parent;src=r/'buffer-scoped';out=r/'packed';out.mkdir(exist_ok=True)
for name in ['kernel.py','device_scope_helpers.py','reference.py','split_control.py','test_sync.py']:shutil.copy2(src/name,out/name)
module=ast.parse((src/'kernel.py').read_text());fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='full_layer')
keys=[n.arg for n in fn.args.args if n.arg!='GRID']
class PointerTable(ast.NodeTransformer):
 def visit_Name(self,node):
  if isinstance(node.ctx,ast.Load) and node.id in keys:
   i=keys.index(node.id)
   return ast.parse(f'get_pointer(Args, {i}, Types[{i}])',mode='eval').body
  return node
fn.body=[PointerTable().visit(n) for n in fn.body];fn.name='packed_full_layer'
fn.args.args=[ast.arg(arg='Args'),ast.arg(arg='Types',annotation=ast.parse('gl.constexpr',mode='eval').body),fn.args.args[-1]]
fn.args.defaults=[ast.Constant(value=256)]
header='''from kernel import *

@g.jit
def get_pointer(Args, index:gl.constexpr, dtype:gl.constexpr):
    address=gl.load(Args+index,volatile=True)
    return address.to(gl.pointer_type(dtype))

'''
(out/'packed_entry.py').write_text(header+ast.unparse(ast.fix_missing_locations(fn))+'''

class PackedAdapter:
    def __init__(self, original):
        self.fn=original.fn
        self.table=None
    def warmup(self,*values,**kwargs):
        import torch
        if self.table is None:
            self.table=torch.tensor([x.data_ptr() for x in values],device='cuda',dtype=torch.uint64)
            mapping={torch.float32:gl.float32,torch.bfloat16:gl.bfloat16,torch.float8_e4m3fn:gl.float8e4nv,torch.int32:gl.int32,torch.int64:gl.int64,torch.uint8:gl.uint8,torch.int8:gl.int8}
            self.types=tuple(mapping[x.dtype] for x in values)
        return packed_full_layer.warmup(self.table,self.types,**kwargs)
    def __getitem__(self,grid):
        def launch(*values,**kwargs):
            return packed_full_layer[grid](self.table,self.types,**kwargs)
        return launch
''')
s=(src/'run.py').read_text().replace('from kernel import full_layer','from kernel import full_layer\nfrom packed_entry import PackedAdapter\nfull_layer=PackedAdapter(full_layer)')
(out/'run.py').write_text(s)
