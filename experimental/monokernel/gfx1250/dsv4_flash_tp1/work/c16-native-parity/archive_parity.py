from pathlib import Path
import tarfile,json,hashlib,datetime
from zoneinfo import ZoneInfo
home=Path('/home/vigokhal');src=home/'dsv4-monokernel/c16-native-parity';out=home/'monokernel-c16-native-parity-20261022.tgz'
resume=home/'MONOKERNEL_RESUME.txt'
header=f'''C16 native-design parity investigation {datetime.datetime.now(ZoneInfo('America/Chicago')).isoformat(timespec='seconds')} Central: all workers finished.
Report/source: {src}/REPORT.md
Complete bundle: {out} (includes previous complete native-projection archive and predecessor archives).
Reproduce with /work/c16-native-parity/run_native_parity.sh parts|elements|progressive|advantage|stress; full arguments in REPORT.md. Four independent GPUs, second timing round.
Standalone native/Gluon us: QA3.601/2.717; QB2.998/2.908; OA4.580/3.736; OB6.149/5.893; inputQuant1.827/1.541; qaNormQuant1.776/1.939; QKRoPE1.797/1.924; inverseRoPEQuant1.671/1.601. Native geometry, K-major scales, per-wave TDM, direct WMMA scale LDS mapping, vector coefficient loads.
Generated native ISA revealed exact reduction/FMA ordering: ROPE_ORDERED=2 ROPE_FMA=7 now matches every QKV intermediate, Q/KV/cache exactly over64 changing inputs. Output chain also exact64inputs. Masked L2-prefetch inline-asm SCC clobber fixed locally and backported to /work/concurrency-rollout/masked_prefetch.py; prior archives unchanged. POLL=3 buffer loads fail some replays; use validated scalar load POLL=2 here. Failed experiments retained but excluded.
Single-launch composition still regresses; no promotion. Final advantage trial native/separateGluon/singlelaunch us: QKV13.209/12.310/24.410; output15.521/14.405/20.786 (OB6buffers). Full progressive table and exact source/ISA included. Keep prior faster selected full-layer implementation. No C16 serving, E2E, GSM8K, Git push or Docker image push this pass. C16 serving remains disabled.

'''
if not resume.read_text().startswith('C16 native-design parity investigation '):resume.write_text(header+resume.read_text())
# Avoid expensive username/group lookup while archiving.
tarfile.pwd=None;tarfile.grp=None
members={}
def addfile(tf,path,name):
 info=tf.gettarinfo(str(path),arcname=name)
 if path.is_file():
  h=hashlib.sha256()
  with path.open('rb') as f:
   while block:=f.read(8<<20):h.update(block)
  members[name]=dict(bytes=path.stat().st_size,sha256=h.hexdigest())
  with path.open('rb') as f:tf.addfile(info,f)
 else:tf.addfile(info)
with tarfile.open(out,'w:gz',compresslevel=1) as tf:
 for f in sorted(src.rglob('*')):
  if '__pycache__' not in f.parts:addfile(tf,f,'c16-native-parity/'+str(f.relative_to(src)))
 addfile(tf,resume,'MONOKERNEL_RESUME.txt')
 addfile(tf,home/'dsv4-monokernel/concurrency-rollout/masked_prefetch.py','backports/concurrency-rollout/masked_prefetch.py')
 for suffix in ['', '.json']:
  f=home/('monokernel-c16-native-projections-20261022.tgz'+suffix)
  if f.exists():addfile(tf,f,'previous/'+f.name)
# Streaming verification also checks decompression and file bytes.
seen=0
with tarfile.open(out,'r|gz') as tf:
 for info in tf:
  if not info.isfile():continue
  h=hashlib.sha256();f=tf.extractfile(info)
  while block:=f.read(8<<20):h.update(block)
  assert h.hexdigest()==members[info.name]['sha256'],info.name
  seen+=1
h=hashlib.sha256()
with out.open('rb') as f:
 while block:=f.read(8<<20):h.update(block)
meta=dict(archive=str(out),bytes=out.stat().st_size,sha256=h.hexdigest(),verified_files=seen,members=members)
Path(str(out)+'.json').write_text(json.dumps(meta,indent=2))
print(json.dumps({k:v for k,v in meta.items() if k!='members'}),flush=True)
