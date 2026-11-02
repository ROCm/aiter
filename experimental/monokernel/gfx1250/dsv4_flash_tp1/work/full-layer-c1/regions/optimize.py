import ast,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;q=p/'expert-tiles';q.mkdir(exist_ok=True)
for name in ['kernel.py','device_scope_helpers.py','split_control.py','bench.py','mapping.json']:
 shutil.copyfile(p/name,q/name)
s=(p/'kernel.py').read_text();a=s.index('@g.jit\ndef matvec');b=s.index('@g.jit\ndef norm',a)
f=s[a:b].replace('def matvec(', 'def routed_matvec(').replace('Gamma=None):','Gamma=None, Ids=None):')
f=f.replace('gl.cdiv(N, 8)','gl.cdiv(N * 6, 8)')
f=f.replace('n = tile * 8 + rn','outn = tile * 8 + rn\n        expert = dev_load(Ids, outn // N, cache=\'.cg\').to(gl.int64)\n        n = outn % N')
f=f.replace("W, n[:, None] * (K // 2)","W, expert[:, None] * N * (K // 2) + n[:, None] * (K // 2)")
f=f.replace("S, n[:, None] * (K // 32)","S, expert[:, None] * N * (K // 32) + n[:, None] * (K // 32)")
f=f.replace("dev_load(Gamma, 0,", "dev_load(Gamma, outn // N,")
f=f.replace('Y, n,','Y, outn,')
(q/'optimized_kernel.py').write_text('from kernel import *\n\n'+f)
r=ast.parse((p/'region_kernel.py').read_text())
class Rewrite(ast.NodeTransformer):
 def visit_For(self,node):
  if isinstance(node.target,ast.Name) and node.target.id=='slot':
   text=ast.unparse(node)
   if 'EW1' in text:
    return ast.parse('routed_matvec(FFMX, EW1, ES1, EG, 2048, 4096, 4, GRID, FLOAT_OUT=True, Ids=Ids)\nrouted_matvec(FFMX, EW3, ES3, EU, 2048, 4096, 4, GRID, FLOAT_OUT=True, Ids=Ids)').body
   if 'EW2' in text:
    return ast.parse('routed_matvec(EAQ.reshape if False else EAQ, EW2, ES2, EO, 4096, 2048, 4, GRID, Gamma=Gamma, Ids=Ids)').body
  return self.generic_visit(node)
r=Rewrite().visit(r)
# Down projection reads a different activation vector for each selected expert.
f=(q/'optimized_kernel.py').read_text().replace("x = dev_load(X, k, cache='.cg')\n                x = x[None, :]", "if Gamma is not None:\n                    x = dev_load(X, (outn // N)[:, None] * K + k[None, :], cache='.cg')\n                else:\n                    x = dev_load(X, k, cache='.cg')\n                    x = x[None, :]")
(q/'optimized_kernel.py').write_text(f)
(q/'region_kernel.py').write_text(ast.unparse(ast.fix_missing_locations(r)).replace('from kernel import *','from optimized_kernel import *').replace('EAQ.reshape if False else EAQ','EAQ')+'\n')
