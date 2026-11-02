from common import *
kind=sys.argv[1]
if kind=='q':
 exec(Path('native_qkv_cache.py').read_text().split('rr=native();')[0])
 value=x; baseline=value.clone(); health_tensor=health
 ref=native; fused_fn=fused
else:
 exec(Path('native_output.py').read_text().split('rr=n(o);')[0])
 value=o; baseline=value.clone(); health_tensor=a[-1]
 ref=lambda:n(o); fused_fn=fo
records=[]
for j in range(64):
 torch.manual_seed(4210+j)
 value.copy_(baseline*(.5+(j%7)*.25)+torch.randn_like(baseline)*(.01 if kind=='q' else .02))
 expected=ref().clone()
 prior=fused_fn(previous_op).clone() if kind=="q" else None
 actual=fused_fn();first=actual.clone();fused_fn()
 torch.cuda.synchronize()
 v=err(actual,expected);v.update(iteration=j,replay_exact=torch.equal(actual,first),health=int(health_tensor[0]))
 assert v['finite'] and not v['health'] and v['replay_exact'],v
 if kind=='o':assert torch.equal(actual,expected),v
 else:
  v['previous_error']=err(prior,expected)
  v['exact_previous']=torch.equal(actual,prior)
  assert v['exact_previous'],v
 records.append(v)
print('STRESS_PASS',kind,len(records),'max_rrms',max(z['relative_rms'] for z in records),flush=True)
(ROOT/('stress_'+kind+'.json')).write_text(json.dumps(records,indent=2))
