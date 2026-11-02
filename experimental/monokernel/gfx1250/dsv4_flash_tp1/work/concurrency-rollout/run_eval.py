import sys,time,urllib.request,subprocess
from pathlib import Path
port,tag,con,count=sys.argv[1:]
for _ in range(240):
 try:urllib.request.urlopen(f'http://localhost:{port}/health',timeout=2);break
 except Exception:time.sleep(2)
else:raise RuntimeError('server not healthy')
cmd=['docker','exec','-e','PYTHONPATH=/app/aiter:/app/ATOM','-e','HF_HUB_OFFLINE=1','-e','HF_DATASETS_OFFLINE=1','vigokhal_dsv4_tp4','lm_eval','--model','local-completions','--model_args',f'model=/data/DeepSeek-V4-Flash,base_url=http://localhost:{port}/v1/completions,num_concurrent={con},max_retries=1,tokenized_requests=False,trust_remote_code=True','--tasks','gsm8k','--num_fewshot','3','--limit',count,'--gen_kwargs','max_gen_toks=512,temperature=0','--log_samples','--output_path',f'/work/concurrency-rollout/gsm8k-{tag}']
with (Path(__file__).parent/f'gsm8k-{tag}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
