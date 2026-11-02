from common import *
kind=sys.argv[1];stage=int(sys.argv[2])
src=Path('faithful_progressive.py').read_text().split('nr();uf();torch.cuda.synchronize();')[0]
exec(compile(src,'faithful_progressive.py','exec'))
seed=9781;torch.manual_seed(seed);data=[]
for i in range(64):
 if kind=='q':
  x.copy_(f['live'].cuda().reshape(1,4096).repeat(16,1)+torch.randn_like(x)*(.005+.0005*i))
  for ten in [y,qbr,qo,kv]:ten.fill_(float('nan'))
  for ten in [xq,xs,q,s]:ten.view(torch.uint8).fill_(255)
 else:
  o.copy_(torch.randn_like(o)*(.05+.005*i))
  for ten in [a[3],a[7]]:ten.fill_(float('nan'))
  for ten in [a[0],a[1],a[4],a[5]]:ten.view(torch.uint8).fill_(255)
 nr();ff();torch.cuda.synchronize();v=check();data.append(dict(input=i,validation=v))
 if int(health[0]):raise RuntimeError(('health',i,int(health[0])))
 if i%8==0:print('INPUT',i,v,flush=True)
maxima={}
for d in data:
 for k,v in d['validation'].items():
  if not isinstance(v,dict):continue
  maxima.setdefault(k,dict(relative_rms=0.,max_abs=0.,finite=True))
  maxima[k]['finite'] &= v['finite']
  for e in ['relative_rms','max_abs']:maxima[k][e]=max(maxima[k][e],v[e])
result=dict(seed=seed,inputs=64,kind=kind,stage=stage,maxima=maxima,data=data,health=int(health[0]))
(ROOT/f'stress_{kind}{stage}.json').write_text(json.dumps(result,indent=2));print('FINAL',maxima,'HEALTH',int(health[0]),flush=True)
assert all(v['finite'] for v in maxima.values())
