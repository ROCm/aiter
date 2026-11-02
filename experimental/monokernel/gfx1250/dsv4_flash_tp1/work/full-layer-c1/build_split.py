"""Generate a same-arithmetic split-launch control for the full-layer kernel."""
import ast
from pathlib import Path
p=Path(__file__).parent
src=ast.parse((p/'kernel.py').read_text())
fn=next(n for n in src.body if isinstance(n,ast.FunctionDef) and n.name=='full_layer')
stages=[]; current=[]
for stmt in fn.body:
 if isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Call) and isinstance(stmt.value.func,ast.Name) and stmt.value.func.id=='sync':
  stages.append(current);current=[]
 else:current.append(stmt)
if current:stages.append(current)
fn.name='split_stage';fn.args.args.append(ast.arg(arg='STAGE',annotation=ast.Attribute(value=ast.Name(id='gl',ctx=ast.Load()),attr='constexpr',ctx=ast.Load())));fn.args.defaults.append(ast.Constant(value=0))
fn.body=[ast.If(test=ast.Compare(left=ast.Name(id='STAGE',ctx=ast.Load()),ops=[ast.Eq()],comparators=[ast.Constant(value=i)]),body=body,orelse=[]) for i,body in enumerate(stages)]
mod=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
(p/'split_control.py').write_text('from kernel import *\n\n'+ast.unparse(mod)+'\n\nNUM_STAGES='+str(len(stages))+'\n')
print(len(stages),'split stages')
