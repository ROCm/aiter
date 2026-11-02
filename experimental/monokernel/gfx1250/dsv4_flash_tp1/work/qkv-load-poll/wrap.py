from harness import *
from variants import run,State
r=reference(X);rows=[]
for name in ['grid_scratch_fp32','grid_scratch_rt_fp32','qready_scratch_fp32']:
 for initial in [-512,-256,2147483392]:
  s=State()
  if name.startswith('qready'):s.ready.fill_(initial);s.epoch.fill_(initial)
  else:s.bar.fill_(initial)
  run(s,name);torch.cuda.synchronize();cg=torch.cuda.CUDAGraph()
  with torch.cuda.graph(cg):run(s,name)
  for _ in range(512):cg.replay()
  torch.cuda.synchronize()
  row=dict(name=name,initial=initial,q=metric(s.q,r['q']),kv=metric(s.kv,r['kv']),state=s.check_state())
  assert row['q']['relative_rms']<.001 and row['kv']['relative_rms']<.001 and all(v==0 if k=='health' else v for k,v in row['state'].items())
  rows.append(row)
(root/'wrap.json').write_text(json.dumps(rows,indent=2));print('WRAP PASS',len(rows),flush=True)
