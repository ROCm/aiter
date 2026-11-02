import json,subprocess,time,urllib.request,concurrent.futures
from pathlib import Path
r=Path(__file__).resolve().parent;out=r.parent/'concurrency-rollout'
cfg=json.loads((r/'commands.json').read_text());status={}
def run(c):
 name=c['name'];f=(out/f'resumed-{name}-launch.log').open('x');srv=subprocess.Popen(c['server_argv'],stdout=f,stderr=subprocess.STDOUT)
 status[name]={'server_pid':srv.pid,'stage':'startup'};save()
 port=next(x.split('=',1)[1] for x in c['server_argv'] if x.startswith('HTTP_PORT='))
 try:
  for _ in range(450):
   if srv.poll() is not None:raise RuntimeError('server exited during startup')
   try:urllib.request.urlopen(f'http://localhost:{port}/health',timeout=2);break
   except Exception:time.sleep(2)
  else:raise TimeoutError('startup')
  for stage in ('short','long'):
   status[name]['stage']=stage;save()
   with (out/f'resumed-{name}-{stage}-diagnostic.log').open('x') as log:
    p=subprocess.run(c[f'{stage}_diagnostic_client_argv'],stdout=log,stderr=subprocess.STDOUT,timeout=900 if name=='fused-eager' else 300)
   if p.returncode:raise RuntimeError(f'{stage} client exit{p.returncode}')
   d=json.loads((out/f'resumed-{name}-{stage}-diagnostic.json').read_text())
   if d['completed']!=16:raise RuntimeError(f'{stage} incomplete {d["completed"]}/16')
  status[name]['stage']='completed'
 except Exception as e:status[name]['error']=str(e)
 save();return status[name]
def save():
 # Each diagnostic result is written separately to avoid concurrent file writes.
 for name,s in list(status.items()):(r/f'matrix-{name}.json').write_text(json.dumps(s,indent=2))
with concurrent.futures.ThreadPoolExecutor(4) as ex:
 for res in ex.map(run,cfg['runs']):print(json.dumps(res),flush=True)
