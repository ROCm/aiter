from common import *
from accepted import QAKVNorm
from initall import initialize_all
out=root/'layer-sequence-bench';out.mkdir(exist_ok=True)
Count=torch.empty(1536,device='cuda',dtype=torch.int32);P=torch.empty((32,1536),device='cuda')
layers=[]
for p in sorted((root/'layer-fixtures').glob('layer_*.pt')):
 d=torch.load(p,map_location='cpu',weights_only=True);layers.append((d['W_bytes'].cuda().view(torch.float8_e4m3fn),d['WS'].cuda(),d['NW'].cuda()))
assert len(layers)==43
s16=QAKVNorm(split=16);s32=QAKVNorm(split=32)
def run(mode,idx):
 w,ws,nw=layers[idx];x,xs=bank[idx%len(bank)]
 if mode==0:
  project[(384,)](x,xs,w,ws,nw,P,Acc,SS,Count,Y,MODE=0,**kw);reduce_norm[(2,)](P,Acc,nw,Y,ATOMIC=False,**kw)
 elif mode==1:
  project_opt[(384,)](x,xs,w,ws,P,BN=32,SPLIT=8,DIRECT=True,WAVES=2,BK=128,UNROLL=1,UPSCALE=True,REPLICATE=True,CACHE='.ca',SCALAR=True,PACKED=False,num_warps=2,enable_fp_fusion=False);reduce_opt[(2,)](P,nw,Y,SPLIT=8,**kw)
 elif mode==2:s16(x,xs,w,ws,nw,out=Y)
 else:s32(x,xs,w,ws,nw,out=Y)
names=['original_2launch','optimized_2launch','accepted_split16','accepted_split32'];graphs=[];repeat=129
for mode in range(4):
 run(mode,0);cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):
  for i in range(repeat):run(mode,i%43)
 for _ in range(10):cg.replay()
 graphs.append(cg)
torch.cuda.synchronize();rounds=[]
for rnd in [1,2]:
 samples={n:[] for n in names}
 for sample in range(40):
  for mode in (range(4) if sample%2==0 else reversed(range(4))):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
   a.record();graphs[mode].replay();b.record();b.synchronize();samples[names[mode]].append(a.elapsed_time(b)*1000/repeat)
 rows={n:dict(median_us=statistics.median(v),min_us=min(v),max_us=max(v),samples_us=v) for n,v in samples.items()}
 rounds.append(dict(round=rnd,weight_layers=43,invocations_per_graph=repeat,results=rows));print(rnd,{n:r['median_us'] for n,r in rows.items()},flush=True)
(out/'timings.json').write_text(json.dumps(rounds,indent=2));print('DONE',flush=True)
