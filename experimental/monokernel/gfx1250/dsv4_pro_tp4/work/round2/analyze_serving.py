from pathlib import Path
import subprocess,sys,concurrent.futures
r=Path('/work');mode=sys.argv[1];tasks=[]
for rank in range(4):
 traces=sorted((r/f'traces/{mode}/rank_{rank}').glob('*.pt.trace.json.gz'))
 assert len(traces)==2,(rank,traces)
 for c,trace in zip((64,128),traces):tasks.append((rank,c,trace))
def run(task):
 rank,c,trace=task;out=r/f'analysis/{mode}_c{c}_rank{rank}'
 for script in ('analyze_decode.py','map_decode.py'):
  with (r/f'round2/{mode}_c{c}_rank{rank}_{script}.log').open('w') as f:
   subprocess.run([sys.executable,str(r/script),str(trace),'--out',str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
 return str(out)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 for out in ex.map(run,tasks):print(out,flush=True)
