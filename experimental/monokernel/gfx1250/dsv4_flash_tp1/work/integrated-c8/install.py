from pathlib import Path
p=Path('/app/ATOM/atom/models/deepseek_v4.py')
s=p.read_text();old='sys.path.insert(0, "/work/batch-full")';new='sys.path.insert(0, os.environ.get("ATOM_DSV4_BATCH_FULL_DIR", "/work/batch-full"))'
if old in s:
 Path('/work/integrated-c8/deepseek_v4.before.py').write_text(s)
 p.write_text(s.replace(old,new,1))
elif new not in s:raise RuntimeError('Expected full-batch integration point missing')
Path('/work/integrated-c8/deepseek_v4.after.py').write_text(p.read_text())
print('Installed configurable batch-full module directory; default remains previous path')
