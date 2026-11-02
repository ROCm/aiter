from pathlib import Path
import tarfile,hashlib,json
work=Path('/work');root=work/'atom-actual-microbench'
paths=[]
for base in [root,work/'mhc-optimized']:
 for p in sorted(base.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or 'inductor' in p.parts:continue
  if p.suffix=='.tgz' or p.name=='ARTIFACT_SHA256.json':continue
  paths.append(p)
manifest={str(p.relative_to(work)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(root/'ARTIFACT_SHA256.json').write_text(json.dumps(manifest,indent=2))
archive=work/'atom-actual-microbench-20261020.tgz'
with tarfile.open(archive,'w:gz') as t:
 for p in paths:t.add(p,arcname='dsv4-monokernel/'+str(p.relative_to(work)),recursive=False)
 t.add(root/'ARTIFACT_SHA256.json',arcname='dsv4-monokernel/atom-actual-microbench/ARTIFACT_SHA256.json')
h=hashlib.sha256(archive.read_bytes()).hexdigest();archive.with_suffix('.tgz.sha256').write_text(h+'  '+archive.name+'\n')
with tarfile.open(archive) as t:
 for m in t.getmembers():
  key=m.name.removeprefix('dsv4-monokernel/')
  if key in manifest and m.isfile():assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==manifest[key],key
print(json.dumps({'archive':str(archive),'sha256':h,'files':len(paths),'bytes':archive.stat().st_size}))
