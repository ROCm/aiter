import time,json,subprocess
from pathlib import Path
p=Path('/work/concurrency-rollout/integrated-optimized-c8-round2.json')
for _ in range(1800):
 if p.exists():
  try:
   d=json.loads(p.read_text())
   if d['completed']==64:break
  except (ValueError,KeyError):pass
 time.sleep(1)
else:raise RuntimeError('No completed second performance client')
subprocess.run(['bash','/work/batch-full/eval.sh','12800','integrated-optimized-c8'],check=True)
