"""Optional checkpoint-wide numerical check; saves all tested region weight fixtures."""
from common import *
from accepted import QAKVNorm
import os
out=root/'layer-validation';out.mkdir(exist_ok=True)
fixtures=root/'layer-fixtures';fixtures.mkdir(exist_ok=True)
use_saved=bool(list(fixtures.glob('layer_*.pt'))) and os.environ.get('USE_CHECKPOINT')!='1'
if use_saved:
 layers=[int(p.stem.split('_')[1]) for p in sorted(fixtures.glob('layer_*.pt'))]
else:
 model=Path(os.environ.get('MODEL_ROOT','/data/DeepSeek-V4-Flash'));index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
 layers=sorted({int(m.group(1)) for name in index if (m:=re.fullmatch(r'layers\.(\d+)\.attn\.wq_a\.weight',name))})
def get(name):
 with safe_open(str(model/index[name]),framework='pt',device='cpu') as f:return f.get_tensor(name)
# Original nine inputs, plus 16 random vectors, with independently quantized inputs.
for seed in range(16):
 x=torch.randn(4096,device='cuda',generator=torch.Generator(device='cuda').manual_seed(20000+seed))*(2.**(seed-8))
 z=x.bfloat16().float().reshape(-1,128);a=z.abs().amax(-1).clamp_min(1.e-10)/448.;e=(((a.view(torch.int32)&0x7fffffff)+0x7fffff)>>23).clamp_max(254);sc=(e<<23).view(torch.float32)
 bank.append(((z/sc[:,None]).to(torch.float8_e4m3fn).flatten(),e.to(torch.uint8)))
xf=torch.stack([x.double()*torch.pow(2.,s.double()-127).repeat_interleave(128) for x,s in bank])
states=[QAKVNorm(split=16),QAKVNorm(split=32)];records=[]
for layer in layers:
 if use_saved:
  saved=torch.load(fixtures/f'layer_{layer:02d}.pt',map_location='cpu',weights_only=True)
  wc,sc,nc=saved['W_bytes'],saved['WS'],saved['NW']
 else:
  p=f'layers.{layer}.attn.'
  qa=get(p+'wq_a.weight');kv=get(p+'wkv.weight')
  if list(qa.shape)!=[1024,4096] or list(kv.shape)!=[512,4096]:
   records.append(dict(layer=layer,skipped=True,qa_shape=list(qa.shape),kv_shape=list(kv.shape)));continue
  wc=torch.cat([qa.view(torch.uint8),kv.view(torch.uint8)])
  sc=torch.cat([get(p+'wq_a.scale').view(torch.uint8),get(p+'wkv.scale').view(torch.uint8)])
  nc=torch.cat([get(p+'q_norm.weight').float(),get(p+'kv_norm.weight').float()])
  torch.save(dict(W_bytes=wc,WS=sc,NW=nc,layer=layer),fixtures/f'layer_{layer:02d}.pt')
 w=wc.cuda().view(torch.float8_e4m3fn);ws=sc.cuda();nw=nc.cuda()
 wr=w.double()*torch.pow(2.,ws.double()-127).repeat_interleave(128,0).repeat_interleave(128,1)
 z=(xf@wr.t()).bfloat16().float();ref=torch.cat([z[:,:1024]*torch.rsqrt(z[:,:1024].square().mean(-1,keepdim=True)+1.e-6)*nw[:1024],z[:,1024:]*torch.rsqrt(z[:,1024:].square().mean(-1,keepdim=True)+1.e-6)*nw[1024:]],dim=-1)
 for state in states:
  ys=[]
  for repeat in range(2):
   for x,xs in bank:ys.append(state(x,xs,w,ws,nw).clone())
  actual=torch.stack(ys);expected=ref.repeat(2,1);d=actual-expected
  rms=(d.square().mean(-1)/expected.square().mean(-1).clamp_min(1e-30)).sqrt()
  q=(d[:,:1024].square().mean(-1)/expected[:,:1024].square().mean(-1).clamp_min(1e-30)).sqrt()
  k=(d[:,1024:].square().mean(-1)/expected[:,1024:].square().mean(-1).clamp_min(1e-30)).sqrt()
  row=dict(layer=layer,split=state.split,cases=len(bank),checked_outputs=len(ys),finite=torch.isfinite(actual).all().item(),max_relative_rms=rms.max().item(),max_q_relative_rms=q.max().item(),max_kv_relative_rms=k.max().item(),max_abs=d.abs().max().item(),counter_zero=torch.count_nonzero(state.count).item()==0,acc_zero=torch.count_nonzero(state.acc).item()==0)
  row['passed']=row['finite'] and max(row['max_relative_rms'],row['max_q_relative_rms'],row['max_kv_relative_rms'])<.001 and row['counter_zero'] and row['acc_zero']
  records.append(row);print(row,flush=True)
 (out/'results.json').write_text(json.dumps(records,indent=2))
print('DONE',flush=True)
