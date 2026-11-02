from pathlib import Path
import hashlib,json,tarfile
base=Path('/home/vigokhal/dsv4-monokernel');exp=base/'rollout-c1248'
dirs=['integrated-c8','rollout-c1248','latency-overlap','modelwide','batch-full','concurrency-rollout','full-attention-native','ffn-c1/moe','c8-tuning','c8-ablation','atom-actual-microbench/qkv']
files=[]
for name in dirs:
 for p in (base/name).rglob('*'):
  if p.name=='ARTIFACT_MANIFEST.json' or '__pycache__' in p.parts or not p.is_file():continue
  if p.suffix not in ['.py','.sh','.json','.jsonl','.log','.patch','.txt','.tsv','.md','.amdgcn','.done','.ready']:continue
  if name!='rollout-c1248' and p.stat().st_size>=20_000_000:continue
  files.append(p)
for name in ['mhc-optimized/fixture.pt','mhc-optimized/mhc-fixture.pt']:
 p=base/name
 if p.exists():files.append(p)
files=sorted(set(files))
manifest={str(p.relative_to(base)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
(exp/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2));files.append(exp/'ARTIFACT_MANIFEST.json')
dest=Path('/home/vigokhal/monokernel-c1248-rollout-20261021.tgz');tarfile.pwd=None;tarfile.grp=None
with tarfile.open(dest,'w:gz',compresslevel=1) as tar:
 for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
with tarfile.open(dest) as tar:
 for name,meta in manifest.items():
  f=tar.extractfile(name);assert f is not None
  assert hashlib.sha256(f.read()).hexdigest()==meta['sha256'],name
meta=dict(path=str(dest),bytes=dest.stat().st_size,files=len(files),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),verified_entries=len(manifest))
dest.with_suffix('.tgz.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
