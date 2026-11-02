from pathlib import Path
import json,tarfile,hashlib
root=Path('/work');r=root/'full-layer-native'
names=set(json.loads((r/'dependency-files.json').read_text()))
for directory in ['full-layer-native','ffn-c1']:
 for p in (root/directory).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_MANIFEST.json':names.add(str(p.relative_to(root)))
names.update(['atom-actual-microbench/attention/graphbatch.py','flash/candidate/dsv4_hybrid.py'])
manifest={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(names)}
(r/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2));names.add('full-layer-native/PACKAGE_MANIFEST.json')
out=root/'full-layer-gfx1250-20261020.tgz'
with tarfile.open(out,'w:gz',compresslevel=1) as t:
 for name in sorted(names):t.add(root/name,arcname=name,recursive=False)
with tarfile.open(out) as t:
 for name,h in manifest.items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==h,name
print(json.dumps({'files':len(names),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}),flush=True)
