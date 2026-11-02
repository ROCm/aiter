from pathlib import Path
import subprocess,sys
root=Path('/work/flash');mode=sys.argv[1] if len(sys.argv)>1 else 'baseline'
trace_root=root/'traces' if mode=='baseline' else root/'traces'/mode
traces=sorted((trace_root/'rank_0').glob('*.pt.trace.json.gz'))
assert len(traces)>=3,(len(traces),traces)
for c,trace in zip((1,16,128),traces[:3]):
 out=root/'analysis'/f'{mode}_c{c}'
 for script in ('analyze_decode.py','map_decode.py'):
  with (root/f'{mode}_c{c}_{script}.log').open('w') as f:
   subprocess.run([sys.executable,'/work/'+script,str(trace),'--out',str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print(str(out),flush=True)
