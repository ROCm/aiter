from pathlib import Path
import re,json,hashlib
r=Path('/work/prefill-dispatch-fix');out={}
for tag,file in [('native','native-server.log'),('fixed','fixed-server.log'),('wrapped','wrapped-server.log')]:
 s=(r/file).read_text();hits=re.findall(r'Using cache directory: (\S+) for',s)
 if not hits:continue
 p=Path(hits[-1])/'computation_graph.py';g=p.read_text()
 out[tag]=dict(cache_dir=hits[-1],sha256=hashlib.sha256(g.encode()).hexdigest(),custom_op_calls=sum('torch.ops.atom.dsv4_modelwide_state_c1.default(' in l for l in g.splitlines()),clone_lines=sum('.clone(' in l or 'aten.clone' in l for l in g.splitlines()))
 (r/f'graph-{tag}.py').write_text(g)
assert out['native']['sha256']==out['fixed']['sha256']
assert out['fixed']['custom_op_calls']==0
assert out['wrapped']['custom_op_calls']==43
(r/'graph-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
