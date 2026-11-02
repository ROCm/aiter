"""Generate commands for later use; never execute them."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
container='vigokhal_dsv4_tp4'
reset=['ATOM_DSV4_BATCH_QKV_NATIVE_CONTROL','ATOM_DSV4_BATCH_QKV_NATIVE_TAIL','ATOM_DSV4_BATCH_QA','ATOM_DSV4_BATCH_SERIAL','ATOM_DSV4_DUAL_REGION','ATOM_DSV4_MODELWIDE_VALIDATE','ATOM_DSV4_NATIVE_STATE','AMD_SERIALIZE_KERNEL','HIP_LAUNCH_BLOCKING','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES']
configs=[('native-control',0,11000,False,{'ATOM_DSV4_BATCH_QKV_NATIVE_CONTROL':'1'}),('fused-graph',1,11100,False,{}),('native-tail',2,11200,False,{'ATOM_DSV4_BATCH_QKV_NATIVE_TAIL':'1'}),('fused-eager',3,11300,True,{})]
runs=[]
for name,gpu,port,eager,extra in configs:
 tag='resumed-'+name
 env={'HIP_VISIBLE_DEVICES':str(gpu),'ATOM_DSV4_BATCH_QKV':'1','ATOM_DSV4_FULL_LAYER_EXPERIMENTAL':'0','ATOM_DSV4_MODELWIDE':'0','CONCURRENCY':'8','CAPTURE_SIZES':'[1,2,4,8]','HTTP_PORT':str(port),'PORT':str(port+1),'SERVER_LOG':f'/work/concurrency-rollout/{tag}-server.log',**extra}
 server=['docker','exec',container,'env']
 for key in reset:server+=['-u',key]
 server += [f'{k}={v}' for k,v in env.items()]+['bash',f'/work/concurrency-rollout/{"eager-server" if eager else "server"}.sh']
 short=['docker','exec',container,'env','PYTHONPATH=/app/aiter:/app/ATOM','python','-m','atom.benchmarks.benchmark_serving','--model=/data/DeepSeek-V4-Flash','--backend=vllm',f'--base-url=http://localhost:{port}','--dataset-name=random','--random-input-len=1024','--random-output-len=64','--random-range-ratio=1','--num-prompts=16','--max-concurrency=8','--request-rate=inf','--ignore-eos','--seed=0','--save-result','--result-dir=/work/concurrency-rollout',f'--result-filename={tag}-short-diagnostic.json']
 long=[x.replace('--random-output-len=64','--random-output-len=1024').replace('-short-diagnostic.json','-long-diagnostic.json') for x in short]
 perf=['docker','exec',container,'bash','/work/concurrency-rollout/clients.sh',tag,str(port),'8']
 runs.append({'name':name,'gpu':gpu,'server_argv':server,'short_diagnostic_client_argv':short,'long_diagnostic_client_argv':long,'two_client_performance_argv_after_correctness':None if eager or extra else perf})
(root/'commands.json').write_text(json.dumps({'executed':False,'note':'User has not authorized GPU work to resume. Commands only. Start/restore container separately after authorization. Check GPUs and ports are free; use new result tags if files already exist.','runs':runs},indent=2)+'\n')
print('Prepared four diagnostic configurations; executed none.')
