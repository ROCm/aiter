#!/usr/bin/env python3
import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
mode=sys.argv[1];folder=sys.argv[2] if len(sys.argv)>2 else mode
for rank in range(4):
 traces=sorted((root/f'traces/{folder}/rank_{rank}').glob('*.pt.trace.json.gz'))
 assert len(traces)==2,(rank,traces)
 for c,trace in zip((16,64),traces):
  out=root/f'analysis/{mode}_c{c}_rank{rank}'
  for script in ('analyze_decode.py','map_decode.py'):
   with (root/f'{mode}_c{c}_rank{rank}_{script}.log').open('w') as f:
    subprocess.run([sys.executable,str(root/script),str(trace),'--out',str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
print(f'{mode}: both profiles analyzed for all4ranks')
