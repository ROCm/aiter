import json,subprocess,time,urllib.request,sys
from pathlib import Path
port,tag=sys.argv[1:3];r=Path(__file__).resolve().parent.parent/'concurrency-rollout'
base=json.loads((r.parent/'resume-prep/commands.json').read_text())['runs'][1]
for _ in range(300):
 try:urllib.request.urlopen(f'http://localhost:{port}/health',timeout=2);break
 except Exception:time.sleep(2)
else:raise TimeoutError('server startup')
for stage in ['short','long']:
 args=[x.replace('11100',port).replace('resumed-fused-graph',tag) for x in base[f'{stage}_diagnostic_client_argv']]
 with (r/f'{tag}-{stage}-diagnostic.log').open('w') as f:
  p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT);deadline=time.monotonic()+600
  while p.poll() is None:
   if time.monotonic()>deadline:p.terminate();raise TimeoutError(stage)
   time.sleep(2)
 assert p.returncode==0,p.returncode
 d=json.loads((r/f'{tag}-{stage}-diagnostic.json').read_text());assert d['completed']==16,d['completed']
 print(tag,stage,'completed16',flush=True)
