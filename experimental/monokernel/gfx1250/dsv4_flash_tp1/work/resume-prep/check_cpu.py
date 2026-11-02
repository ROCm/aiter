"""CPU-only handoff verification. Does not import Torch/Triton or start containers."""
import ast,hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent.parent
home=root.parent
expected=json.loads((home/'dsv4-image-20261021/PUSH_RESULT.json').read_text())
result={'gpu_work_started':False,'image':expected['image'],'expected_digest':expected['digest']}
result['local_image_id']=subprocess.check_output(['docker','image','inspect',expected['image'],'--format','{{.Id}}'],text=True).strip()
assert result['local_image_id']==expected['digest']
result['container_status']=subprocess.check_output(['docker','inspect','vigokhal_dsv4_tp4','--format','{{.State.Status}}'],text=True).strip()
model=Path('/data/DeepSeek-V4-Flash')
index=json.loads((model/'model.safetensors.index.json').read_text())
shards=set(index['weight_map'].values())
missing=[s for s in sorted(shards) if not (model/s).is_file()]
assert not missing,missing
result['checkpoint_shards_present']=len(shards)
py=['concurrency-rollout/qa_batch.py','concurrency-rollout/qkv_batch.py','concurrency-rollout/qkv_integration.py','modelwide/runtime.py','modelwide/model_layer_precise.py','modelwide/install.py']
sh=['concurrency-rollout/server.sh','concurrency-rollout/eager-server.sh','concurrency-rollout/clients.sh','concurrency-rollout/run_flash_experiment.sh']
result['source_sha256']={}
for name in py+sh:
 p=root/name;data=p.read_bytes();result['source_sha256'][name]=hashlib.sha256(data).hexdigest()
 if name in py:ast.parse(data,filename=name)
 else:subprocess.run(['bash','-n',str(p)],check=True)
archive=home/'dsv4-monokernel-resume-20261021.tgz'
want=Path(str(archive)+'.sha256').read_text().split()[0]
with archive.open('rb') as f:got=hashlib.file_digest(f,'sha256').hexdigest()
assert got==want,(got,want)
result['archive_sha256']=got
result['checks_passed']=True
(root/'resume-prep/CPU_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
