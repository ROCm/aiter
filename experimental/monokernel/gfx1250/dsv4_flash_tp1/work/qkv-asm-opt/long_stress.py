from common import *
from accepted import QAKVNorm
out=root/'long-stress';out.mkdir(exist_ok=True)
extra=[torch.zeros(4096,device='cuda'),torch.ones(4096,device='cuda'),-torch.ones(4096,device='cuda')]
for seed in range(256):
 gen=torch.Generator(device='cuda').manual_seed(10000+seed)
 x=torch.randn(4096,device='cuda',generator=gen)
 if seed%4==0:x=live.float()+x*.01
 else:x=x*(2.**((seed%17)-8))
 extra.append(x.bfloat16())
for x in extra:
 z=x.float().reshape(-1,128);amax=z.abs().amax(-1).clamp_min(1.e-10)/448.
 e=(((amax.view(torch.int32)&0x7fffffff)+0x7fffff)>>23).clamp_max(254)
 scale=(e<<23).view(torch.float32)
 bank.append(((z/scale[:,None]).to(torch.float8_e4m3fn).flatten(),e.to(torch.uint8)))

xf=torch.stack([x.double()*torch.pow(2.,s.double()-127).repeat_interleave(128) for x,s in bank])
layers=[];references=[]
for p in sorted((root/'layer-fixtures').glob('layer_*.pt')):
 d=torch.load(p,map_location='cpu',weights_only=True);w=d['W_bytes'].cuda().view(torch.float8_e4m3fn);ws=d['WS'].cuda();nw=d['NW'].cuda();layers.append((w,ws,nw))
 wr=w.double()*torch.pow(2.,ws.double()-127).repeat_interleave(128,0).repeat_interleave(128,1)
 z=(xf@wr.t()).bfloat16().float();references.append(torch.cat([z[:,:1024]*torch.rsqrt(z[:,:1024].square().mean(-1,keepdim=True)+1.e-6)*nw[:1024],z[:,1024:]*torch.rsqrt(z[:,1024:].square().mean(-1,keepdim=True)+1.e-6)*nw[1024:]],dim=-1))
state=QAKVNorm(split=16);count=1000
outputs=torch.empty((count,1536),device='cuda');expected=torch.stack([references[i%43][(i*37)%len(bank)] for i in range(count)])
def run(i):
 w,ws,nw=layers[i%43];x,xs=bank[(i*37)%len(bank)];state(x,xs,w,ws,nw,out=outputs[i])
run(0);cg=torch.cuda.CUDAGraph()
with torch.cuda.graph(cg):
 for i in range(count):run(i)
records=[]
for replay in range(100):
 cg.replay()
 d=outputs-expected
 all_rms=(d.square().mean(-1)/expected.square().mean(-1).clamp_min(1e-30)).sqrt()
 q=(d[:,:1024].square().mean(-1)/expected[:,:1024].square().mean(-1).clamp_min(1e-30)).sqrt()
 k=(d[:,1024:].square().mean(-1)/expected[:,1024:].square().mean(-1).clamp_min(1e-30)).sqrt()
 row=dict(replay=replay,checked_outputs=count,finite=torch.isfinite(outputs).all().item(),max_relative_rms=all_rms.max().item(),max_q_relative_rms=q.max().item(),max_kv_relative_rms=k.max().item(),counter_zero=torch.count_nonzero(state.count).item()==0,acc_zero=torch.count_nonzero(state.acc).item()==0)
 row['passed']=row['finite'] and max(row['max_relative_rms'],row['max_q_relative_rms'],row['max_kv_relative_rms'])<.001 and row['counter_zero'] and row['acc_zero']
 records.append(row)
 if replay%10==0 or not row['passed']:print(row,flush=True)
 if not row['passed']:break
(out/'results.json').write_text(json.dumps(dict(layers=43,available_inputs=len(bank),unique_pairs=count,checked_outputs=sum(r['checked_outputs'] for r in records),all_passed=all(r['passed'] for r in records),records=records),indent=2))
print('DONE',flush=True)
