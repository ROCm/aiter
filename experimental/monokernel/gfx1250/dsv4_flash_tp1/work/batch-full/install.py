from pathlib import Path
p=Path('/app/ATOM/atom/models/deepseek_v4.py')
s=p.read_text()
old='''                from runtime import ServingLayer
                self._experimental_full_layer = ServingLayer()'''
new='''                if os.environ.get("ATOM_DSV4_BATCH_FULL", "0") == "1":
                    sys.path.insert(0, "/work/batch-full")
                    from runtime_batch import ServingLayer
                else:
                    from runtime import ServingLayer
                self._experimental_full_layer = ServingLayer()'''
if new not in s:
 if old not in s:raise RuntimeError('Expected integration point missing')
 Path('/work/batch-full/deepseek_v4.before_batch_full.py').write_text(s)
 p.write_text(s.replace(old,new,1))
Path('/work/batch-full/deepseek_v4.with_batch_full.py').write_text(p.read_text())
print('Installed opt-in ATOM_DSV4_BATCH_FULL adapter')
