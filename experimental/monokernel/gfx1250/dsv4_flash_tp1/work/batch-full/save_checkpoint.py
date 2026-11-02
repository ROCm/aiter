from pathlib import Path
import json,datetime,tarfile,hashlib,subprocess
r=Path(__file__).resolve().parent
subprocess.run(['python3',str(r/'collect_results.py')],check=True,stdout=subprocess.DEVNULL)
results=json.loads((r/'RESULTS.json').read_text())
if results['status']!='complete':raise RuntimeError('Final benchmark results not complete')
p=r/'CHECKPOINT.json';s=json.loads(p.read_text());s.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pending=[],status='Batched full layer implemented; padding bug fixed; matched-concurrency subset accuracy and second-client benchmarks completed',results=str(r/'RESULTS.json'),next_work=['Optimize C8 full-layer performance; it currently trails native','Broaden accuracy validation beyond32 GSM8K examples; C4 equal scores have different failed examples','Consider isolated routed-MoE scheduling/occupancy ablations within the full kernel'],accuracy_fixed={'C2':31,'C4':30,'C8':30,'denominator':32,'C8_extra_failure':'16.00 vs16; numerically correct'},running_servers={'GPU0':'corrected full HTTP11700 idle after C8','GPU1':'native HTTP11500 idle','GPU2':'corrected full HTTP11800 idle after C4','GPU3':'corrected full HTTP11600 idle after C2'});p.write_text(json.dumps(s,indent=2))
resume=Path('/home/vigokhal/MONOKERNEL_RESUME.txt')
old=resume.read_text() if resume.exists() else ''
header='LATEST: 2026-10-21 batched Flash full-layer monokernel\nRead /home/vigokhal/dsv4-monokernel/batch-full/CHECKPOINT.json and RESULTS.json.\nFull attention + routed/shared MoE, all43 layers, real multirow WMMA C1/C2/C4/C8.\nFixed graph-padding state-slot0 corruption; explicit BatchIds masks required.\nUse bounded TDM prefetch and early TDM. Inline prefetch controls faulted/stalled in larger fused kernels.\nResults use second client only. C2 faster, C4 approximately flat, C8 slower; see RESULTS.json.\nSmall GSM8K subset: C2 31/32, C4 30/32, C8 30/32 (16.00 vs16 formatting accounts for extra C8 failure).\nBundle /home/vigokhal/monokernel-batch-full-20261021.tgz contains sources, scripts, logs, results, and code dependencies; base image/model remain separate.\nNo new image or Git branch push performed for this batched extension.\n\nEarlier records follow:\n'
if not old.startswith(header):resume.write_text(header+old)
# Include reproducible source dependencies and compact records, not model weights,
# binary caches, or the separately preserved multi-GB profiling traces.
base=r.parent;files=[]
for name in ['batch-full','modelwide','concurrency-rollout','full-attention-native','ffn-c1/moe']:
 for f in (base/name).rglob('*'):
  if f.is_file() and '__pycache__' not in f.parts and f.suffix in ['.py','.sh','.json','.jsonl','.log','.patch'] and f.stat().st_size<20_000_000:files.append(f)
dest=Path('/home/vigokhal/monokernel-batch-full-20261021.tgz');tarfile.pwd=None;tarfile.grp=None
with tarfile.open(dest,'w:gz',compresslevel=1) as tar:
 for f in sorted(set(files)):tar.add(f,arcname=str(f.relative_to(base)),recursive=False)
meta=dict(path=str(dest),bytes=dest.stat().st_size,files=len(set(files)),sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
dest.with_suffix(dest.suffix+'.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
