import subprocess,json,sys
from pathlib import Path
kind,port=sys.argv[1:3]
r=Path(__file__).resolve().parent.parent/'concurrency-rollout'
for c in [8,2,4]:
 for rnd in [1,2]:assert not (r/f'{kind}-c{c}-round{rnd}.json').exists(),'Refusing result overwrite'
 subprocess.run(['docker','exec','vigokhal_dsv4_tp4','bash','/work/concurrency-rollout/clients.sh',kind,port,str(c)],check=True,timeout=600)
 for rnd in [1,2]:
  d=json.loads((r/f'{kind}-c{c}-round{rnd}.json').read_text());assert d['completed']==c*8 and d['total_output_tokens']==c*8*1024,d
 print(json.dumps({'kind':kind,'concurrency':c,'second':{k:d[k] for k in ['completed','median_ttft_ms','median_tpot_ms','total_token_throughput']}}),flush=True)
