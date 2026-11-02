from pathlib import Path
import subprocess,sys
r=Path('/work')
for rank in range(4):
 ts=list((r/f'traces/round2_v1/rank_{rank}').glob('*.pt.trace.json.gz'));assert len(ts)==1
 for script in ('analyze_decode.py','map_decode.py'):
  with (r/f'round2/profile128_rank{rank}_{script}.log').open('w') as f:
   subprocess.run([sys.executable,str(r/script),str(ts[0]),'--out',str(r/f'analysis/round2_v1_c128_rank{rank}')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('128 profiles analyzed across4ranks')
