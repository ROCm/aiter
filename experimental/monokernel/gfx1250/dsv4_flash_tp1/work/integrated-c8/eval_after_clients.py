from pathlib import Path
import time,subprocess
p=Path('/work/concurrency-rollout/batch-full-fused-c8-round2.json')
for _ in range(1800):
 if p.exists():break
 time.sleep(1)
else:raise RuntimeError('Performance clients did not finish')
subprocess.run(['bash','/work/batch-full/eval.sh','11600','full'],check=True)
