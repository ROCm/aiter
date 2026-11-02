from pathlib import Path
import hashlib,json,tarfile
p=Path('/work/mhc-optimized')
(p/'container.txt').write_text('Container: vigokhal_dsv4_tp4\nImage: rocm/fw-bringup-mlperf:gfx1250-atom-20261015\nImage ID: sha256:d1434db7030a8355a332b2a9237598fadb07a8bf7bb6dbbd081567a13d54686a\nHost /home/vigokhal/dsv4-monokernel mounted at /work\nSelected experiments use HIP_VISIBLE_DEVICES=0 (independent experiments used1,2,3).\n')
files=[];links={}
for f in sorted(p.rglob('*')):
 if '__pycache__' in f.parts or f.name=='ARTIFACT_SHA256.json':continue
 if f.is_symlink():links[str(f.relative_to(p))]=str(f.readlink())
 elif f.is_file():files.append(f)
m={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
(p/'ARTIFACT_SHA256.json').write_text(json.dumps(dict(files=m,symlinks=links),indent=2))
a=Path('/work/mhc-optimized-20261020.tgz')
with tarfile.open(a,'w:gz') as tar:
 for f in files+[p/'ARTIFACT_SHA256.json']+[p/s for s in links]:tar.add(f,arcname='mhc-optimized/'+str(f.relative_to(p)))
checked=0;checked_links=0
with tarfile.open(a,'r|gz') as tar:
 for member in tar:
  rel=member.name.split('/',1)[1]
  if rel in links:
   assert member.issym() and member.linkname==links[rel];checked_links+=1
  elif rel in m:
   assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==m[rel];checked+=1
assert checked==len(m) and checked_links==len(links)
sha=hashlib.sha256(a.read_bytes()).hexdigest();Path(str(a)+'.sha256').write_text(sha+'  '+a.name+'\n')
print(json.dumps(dict(archive=str(a),bytes=a.stat().st_size,verified_files=checked,verified_links=checked_links,sha256=sha),indent=2))
