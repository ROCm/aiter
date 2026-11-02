"""Restore saved ATOM integration files with timestamped backups.
Run inside the documented image with this project mounted at /work.
"""
from pathlib import Path
import datetime,json,shutil,subprocess
root=Path('/work');app=Path('/app/ATOM')
base='f19778df4054e94d64d6f6bf6dd68f5681bd5945'
if (app/'.git').exists():
 subprocess.run(['git','-C',str(app),'merge-base','--is-ancestor',base,'HEAD'],check=True)
backup=root/'modelwide'/('install-backup-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
names=json.loads((root/'modelwide/atom-source/FILES.json').read_text())
for name in names:
 src=root/'modelwide/atom-source'/name;dst=app/name
 if dst.exists() and dst.read_bytes()!=src.read_bytes():
  saved=backup/name;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,saved)
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
script=Path('/app/scripts/run_dsv4_flash_experiment.sh')
shutil.copy2(root/'concurrency-rollout/run_flash_experiment.sh',script);script.chmod(0o755)
print('Restored saved ATOM files; any previous differing files backed up to',backup)
print('Use /app/scripts/run_dsv4_flash_experiment.sh. Read /work/concurrency-rollout/REPORT.md first.')
