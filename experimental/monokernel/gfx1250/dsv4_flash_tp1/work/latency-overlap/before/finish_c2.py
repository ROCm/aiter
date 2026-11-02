from pathlib import Path
import time,os,signal,subprocess
p=Path('/work/concurrency-rollout/batch-full-fused-c2-round2.json')
while not p.exists():time.sleep(1)
# Stop only the superseded serial driver and its client descendants. Its C2
# pair is already complete; C4/C8 now have dedicated servers and result tags.
procs={}
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  cmd=d.joinpath('cmdline').read_bytes().split(b'\0')
  fields=d.joinpath('stat').read_text().split(') ',1)[1].split()
  procs[int(d.name)]=(int(fields[1]),cmd)
 except (OSError,ValueError):pass
roots={pid for pid,(_,cmd) in procs.items() if cmd[:2]==[b'bash',b'/work/batch-full/fused_clients.sh'] or cmd[:2]==[b'python',b'/work/batch-full/eval_after_clients.py']}
selected=set(roots)
while True:
 more={pid for pid,(ppid,_) in procs.items() if ppid in selected}-selected
 if not more:break
 selected|=more
for pid in sorted(selected,reverse=True):
 try:os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:pass
print('Stopped superseded drivers',sorted(selected),flush=True)
time.sleep(3)
subprocess.run(['bash','/work/batch-full/eval.sh','11600','full'],check=True)
