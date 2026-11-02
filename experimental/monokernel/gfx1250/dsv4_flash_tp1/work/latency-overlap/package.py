from pathlib import Path
import hashlib,json,tarfile
base=Path('/home/vigokhal/dsv4-monokernel');exp=base/'latency-overlap'
dirs=['integrated-c8','latency-overlap','modelwide','batch-full','concurrency-rollout','full-attention-native','ffn-c1/moe','c8-tuning','c8-ablation','atom-actual-microbench/qkv']
files=[]
for name in dirs:
 for p in (base/name).rglob('*'):
  if p.name!='ARTIFACT_MANIFEST.json' and p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.sh','.json','.jsonl','.log','.patch','.txt','.tsv','.md','.amdgcn'] and p.stat().st_size<20_000_000:files.append(p)
for name in ['mhc-optimized/fixture.pt','mhc-optimized/mhc-fixture.pt']:
 p=base/name
 if p.exists():files.append(p)
files=sorted(set(files))
manifest={str(p.relative_to(base)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
(exp/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
files.append(exp/'ARTIFACT_MANIFEST.json')
dest=Path('/home/vigokhal/monokernel-whole-overlap-20261021.tgz')
tarfile.pwd=None;tarfile.grp=None
with tarfile.open(dest,'w:gz',compresslevel=1) as tar:
 for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
meta=dict(path=str(dest),bytes=dest.stat().st_size,files=len(files),sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
dest.with_suffix('.tgz.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
