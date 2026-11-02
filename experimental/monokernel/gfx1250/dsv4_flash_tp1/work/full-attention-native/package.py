from pathlib import Path
import tarfile,hashlib,json
work=Path('/work');r=work/'full-attention-native';paths=[]
for name in ['full-attention-native','mhc-native-rms-fix','atom-actual-microbench','mhc-optimized']:
 for p in sorted((work/name).rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or 'inductor' in p.parts or p.name=='ARTIFACT_SHA256.json' or p.suffix=='.tgz':continue
  paths.append(p)
manifest={str(p.relative_to(work)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(r/'ARTIFACT_SHA256.json').write_text(json.dumps(manifest,indent=2))
a=work/'full-attention-gfx1250-20261020.tgz'
with tarfile.open(a,'w:gz') as t:
 for p in paths:t.add(p,arcname='dsv4-monokernel/'+str(p.relative_to(work)),recursive=False)
 t.add(r/'ARTIFACT_SHA256.json',arcname='dsv4-monokernel/full-attention-native/ARTIFACT_SHA256.json')
h=hashlib.sha256(a.read_bytes()).hexdigest();a.with_suffix('.tgz.sha256').write_text(h+'  '+a.name+'\n')
with tarfile.open(a) as t:
 for m in t.getmembers():
  key=m.name.removeprefix('dsv4-monokernel/')
  if m.isfile() and key in manifest:assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==manifest[key],key
print(json.dumps({'archive':str(a),'sha256':h,'bytes':a.stat().st_size,'files':len(paths)}))
