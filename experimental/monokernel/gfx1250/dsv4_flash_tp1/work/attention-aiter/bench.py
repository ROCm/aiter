from common import *
configs=[('vector',True,1,16),('aiter_s1',False,1,16),('aiter_s2',False,2,16),('aiter_s4',False,4,16)]
cache=makecache();idx=torch.arange(128,device='cuda',dtype=torch.int32)
entries=[];resources={}
for name,vector,s,bh in configs:
 obj=AttentionRegion(cache.clone(),idx,Sink,splits=s,block_h=bh,vector=vector);obj(Q,KV);torch.cuda.synchronize()
 k=obj.obj;resources[name]=dict(vgpr=k.n_regs,lds=k.metadata.shared,spills=k.n_spills)
 (root/f'{name}.amdgcn').write_text(k.asm['amdgcn']);(root/f'{name}.llir').write_text(k.asm['llir'])
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):
  for _ in range(128):obj(Q,KV)
 for _ in range(5):cg.replay()
 entries.append((name,obj,cg))
for fused,s in [(False,1),(True,1),(False,4),(True,4)]:
 name=('qkv_attention_fused' if fused else 'qkv_attention_two')+f'_s{s}'
 obj=QKVAttentionRegion(W,WS,NW,BW,BS,cache.clone(),idx,Sink,fused=fused,splits=s);obj(X,Cos,Sin);torch.cuda.synchronize()
 if fused:
  k=obj.obj;resources[name]=dict(vgpr=k.n_regs,lds=k.metadata.shared,spills=k.n_spills)
  (root/f'{name}.amdgcn').write_text(k.asm['amdgcn']);(root/f'{name}.llir').write_text(k.asm['llir'])
 cg=torch.cuda.CUDAGraph()
 with torch.cuda.graph(cg):
  for _ in range(128):obj(X,Cos,Sin)
 for _ in range(5):cg.replay()
 entries.append((name,obj,cg))
# Matched QKV + previous vector attention chain.
obj=AttentionRegion(cache.clone(),idx,Sink,vector=True,splits=1);obj(Q,KV);torch.cuda.synchronize()
cg=torch.cuda.CUDAGraph()
with torch.cuda.graph(cg):
 for _ in range(128):
  q,kv=qregion(X,Cos,Sin);obj(q,kv)
for _ in range(5):cg.replay()
entries.append(('qkv_vector_two',obj,cg))
(root/'resources.json').write_text(json.dumps(resources,indent=2));results=[]
for rnd in [1,2]:
 samples={name:[] for name,_,_ in entries}
 for i in range(40):
  for name,obj,cg in entries if i%2==0 else reversed(entries):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
   a.record();cg.replay();b.record();b.synchronize();samples[name].append(a.elapsed_time(b)*1000/128)
 row=dict(round=rnd,results={name:dict(median_us=statistics.median(samples[name]),samples_us=samples[name],health=obj.check()) for name,obj,_ in entries})
 results.append(row);(root/'timings.json').write_text(json.dumps(results,indent=2));print(rnd,{n:v['median_us'] for n,v in row['results'].items()},flush=True)
(root/'environment.json').write_text(json.dumps(dict(gpu=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,triton=__import__('triton').__version__),indent=2))
