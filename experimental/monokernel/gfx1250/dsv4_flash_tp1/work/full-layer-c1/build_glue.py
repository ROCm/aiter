import ast,shutil
from pathlib import Path
# Importing the generator rebuilds the known buffer-scoped source, without GPU use.
from build_buffer_scoped import DeviceScope
r=Path(__file__).parent;src=r/'buffer-scoped';out=r/'local-glue';out.mkdir(exist_ok=True)
for name in ['device_scope_helpers.py','reference.py','run.py','test_sync.py','split_control.py']:shutil.copy2(src/name,out/name)
module=ast.parse((src/'kernel.py').read_text());fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='full_layer')
stages=[];cur=[]
for statement in fn.body:
 if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Name) and statement.value.func.id=='sync':stages.append(cur);cur=[]
 else:cur.append(statement)
if cur:stages.append(cur)
for i,st in enumerate(stages):print(i,ast.unparse(ast.Module(body=st,type_ignores=[])).splitlines()[0])
# The split control deliberately retains the original 33 stages and arithmetic.
(out/'control_kernel.py').write_text((src/'kernel.py').read_text())
p=out/'split_control.py';p.write_text(p.read_text().replace('from kernel import *','from control_kernel import *'))
replacements={2:(3,'mhc_local(R,MixA,AN,X,XQ)'),21:(3,'mhc_local(R2,MixF,FN,FF,FFQ)')}
# Resolve positions by function names, rather than relying on stale stage numbers.
new=[];i=0
while i<len(stages):
 text=ast.unparse(ast.Module(body=stages[i],type_ignores=[]))
 if text.startswith('mhc_pre(R,'):
  new.extend(ast.parse('mhc_local(R,MixA,AN,X,XQ)').body);i+=3
 elif text.startswith('mhc_pre(R2,'):
  # FFQ originally shares a stage with FFMX/router. Preserve those after local FFQ.
  new.extend(ast.parse('mhc_local(R2,MixF,FN,FF,FFQ)').body);i+=2
  new.extend(ast.parse('sync(Counter,Health,GRID)').body)
  remaining=[n for n in stages[i] if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='quant' and isinstance(n.value.args[1],ast.Name) and n.value.args[1].id=='FFQ')]
  new.extend(remaining);i+=1
 elif text.startswith('activation(SG,'):
  new.extend(ast.parse('act_quant_local(SG,SU,SA,SAQ,1,128,True)').body);i+=2
 elif text.startswith('activation(EG,'):
  new.extend(ast.parse('act_quant_local(EG,EU,EA,EAQ,6,32,False)').body);i+=2
 elif text.startswith('i = gl.arange'):
  new.extend(ast.parse('combine_post(EO,SO,R2,MixF,Sum,Out)').body);i=len(stages)
 else:new.extend(stages[i]);i+=1
 if i<len(stages):new.extend(ast.parse('sync(Counter,Health,GRID)').body)
fn.body=new
helpers=DeviceScope().visit(ast.parse((r/'fused_glue.py').read_text()))
module.body=[*helpers.body,*module.body]
(out/'kernel.py').write_text(ast.unparse(ast.fix_missing_locations(module))+'\n')
