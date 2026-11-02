import ast,copy
from pathlib import Path
p=Path(__file__).parent;r=p.parent
src=ast.parse((r/'regions/region_kernel.py').read_text())
fn=next(n for n in src.body if isinstance(n,ast.FunctionDef))
qsrc=ast.parse((r/'regions/wmma/region_kernel.py').read_text());qfn=next(n for n in qsrc.body if isinstance(n,ast.FunctionDef))
fn.args=copy.deepcopy(qfn.args)
fn.args.args.append(ast.arg(arg='MHC_TEMP'));fn.args.defaults.append(ast.Constant(None))
fn.body[1].body=copy.deepcopy(qfn.body[1].body)
class Rewrite(ast.NodeTransformer):
 def visit_Call(self,n):
  n=self.generic_visit(n)
  if isinstance(n.func,ast.Name):
   if n.func.id=='matvec':n.func.id='linear'
   elif n.func.id=='mhc_mix':
    n.func.id='mhc_wmma';n.args=n.args[:-1]+[ast.Name(id=x,ctx=ast.Load()) for x in ['MHC_TEMP','Counter','Health','GRID']]
   elif n.func.id=='attention':n.func.id='attention_wmma'
  return n
fn=Rewrite().visit(fn)
(p/'region_kernel.py').write_text('from ops import *\n\n'+ast.unparse(ast.fix_missing_locations(fn))+'\n')
full=copy.deepcopy(fn);full.name='all_wmma_layer';body=[]
for i,region in enumerate(full.body):
 body.extend(region.body)
 if i<6:body.extend(ast.parse('sync(Counter,Health,GRID)').body)
full.body=body
(p/'full_kernel.py').write_text('from ops import *\n\n'+ast.unparse(ast.fix_missing_locations(full))+'\n')
