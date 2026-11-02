import ast,copy,json
from pathlib import Path
p=Path(__file__).resolve().parent
base=p.parent/'buffer-scoped'
for name in ['kernel.py','device_scope_helpers.py','split_control.py']:
 (p/name).write_text((base/name).read_text())
src=ast.parse((base/'kernel.py').read_text())
fn=next(n for n in src.body if isinstance(n,ast.FunctionDef) and n.name=='full_layer')
stages=[];current=[];barrier=None
for stmt in fn.body:
 if isinstance(stmt,ast.Expr) and isinstance(stmt.value,ast.Call) and getattr(stmt.value.func,'id',None)=='sync':
  stages.append(current);current=[];barrier=stmt
 else: current.append(stmt)
if current:stages.append(current)
regions=[('attention_input_mhc',0,4),('qkv_projections_norm_rope',5,10),('cache_and_attention',11,12),('attention_output_projections',13,17),('attention_post_ffn_pre',18,22),('router_shared_routed_moe',23,30),('expert_sum_final_mhc',31,32)]
new=copy.deepcopy(fn);new.name='region_kernel'
new.args.args.append(ast.arg(arg='REGION',annotation=ast.parse('gl.constexpr',mode='eval').body));new.args.defaults.append(ast.Constant(0));new.body=[]
for rid,(name,start,end) in enumerate(regions):
 body=[]
 for s in range(start,end+1):
  body.extend(copy.deepcopy(stages[s]))
  if s<end:body.append(copy.deepcopy(barrier))
 new.body.append(ast.If(test=ast.parse(f'REGION == {rid}',mode='eval').body,body=body,orelse=[]))
(p/'region_kernel.py').write_text('from kernel import *\n\n'+ast.unparse(ast.fix_missing_locations(new))+'\n')
(p/'mapping.json').write_text(json.dumps([dict(name=n,start=s,end=e,operations=[ast.unparse(ast.Module(body=stages[i],type_ignores=[])) for i in range(s,e+1)]) for n,s,e in regions],indent=2))
prefix=(base/'run.py').read_text().split("print('Compiling full layer'")[0]
(p/'bench.py').write_text(prefix+(p/'bench_tail.py').read_text())
