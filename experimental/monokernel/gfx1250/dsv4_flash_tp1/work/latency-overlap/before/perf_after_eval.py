from pathlib import Path
import json,time,subprocess,sys
port=int(sys.argv[1]);con=int(sys.argv[2]);root=Path(f'/work/batch-full/gsm8k-fixed-c{con}')
for _ in range(900):
 files=list(root.rglob('results*.json')) if root.exists() else []
 if files:
  try:data=json.loads(files[-1].read_text());break
  except json.JSONDecodeError:pass
 time.sleep(1)
else:raise RuntimeError('Accuracy results not available')
score=data['results']['gsm8k']['exact_match,strict-match'];threshold={2:31/32,4:30/32,8:31/32}[con]
print(dict(concurrency=con,score=score,baseline=threshold),flush=True)
if score<threshold:raise RuntimeError('Accuracy below matched native subset; hold performance rerun for diagnosis')
subprocess.run(['bash','/work/concurrency-rollout/clients.sh','batch-full-fixed',str(port),str(con)],check=True)
