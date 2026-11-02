from common import *
from faithful_gemm import entry as ge
from faithful_ops import quant_entry as qe,norm_entry as ne,rope_entry as re,inverse_entry as ie
from faithful_composed import qkv as fused_q,output as fused_o
from atom.model_ops.linear import per_1x128_e8m0_quant
obnb=int(os.getenv("FAITHFUL_OB_BUFFERS","4"))
kind=sys.argv[1];stage=int(sys.argv[2]);fused_only=os.getenv('FUSED_ONLY')=='1'
if os.getenv('COMPACT_O')=='1':from faithful_composed import compact_output as fused_o
if os.getenv("COOP")=="1":kw["launch_cooperative_grid"]=True
kw2=dict(num_warps=2,enable_fp_fusion=False);kw1=dict(num_warps=1,enable_fp_fusion=False)
epochs=torch.zeros(256,device='cuda',dtype=torch.int32)
if kind=='q':
 exec(Path('native_qkv_cache.py').read_text().split('rr=native();')[0])
 qbr=torch.empty((16,32768),device='cuda',dtype=torch.bfloat16)
 ref={}
 def nr():
  xx,ss=per_1x128_e8m0_quant(x,torch.float8_e4m3fn,True);ref.update(xq=xx,xs=ss)
  if stage>=2:aa=n.qa(xx,x_scale=ss);ref['raw']=aa
  if stage>=3:qq,qs=n.norm(aa[:,:1024]);ref.update(q=qq,qs=qs)
  if stage>=4:rr=n.qb(qq,x_scale=qs);ref['qb']=rr
  if stage>=5:out=qk_norm_rope_maybe_quant(rr,aa[:,1024:],n.nw,n.cos,n.sin,pos,64,512,64,1e-6,swa_kv=cache,swa_dest_rows=dest,batch_id_per_q_token=bid);ref.update(qo=out.q_sa,kv=out.kv)
 def uf():
  qe[(32,)](x,xq,xs,16,4096,**kw2)
  if stage>=2:ge[(48,)](xq,xs,wp,ws,y,16,1536,4096,16,32,1024,4,**kw2)
  if stage>=3:ne[(16,)](y,n.norm.weight,q,s,16,**kw)
  if stage>=4:ge[(256,)](q,s,qbp,qbs,qbr,16,32768,1024,16,128,512,2,**kw)
  if stage>=5:re[(260,)](y,qbr,n.nw,cos,sin,qo,kv,cache2,dest,bid,16,16*128,**kw)
 def ff():return fused_q[(256,)](x,xq,xs,wp,ws,y,n.norm.weight,q,s,qbp,qbs,qbr,n.nw,cos,sin,qo,kv,cache2,dest,bid,counter,health,epochs,16,stage,**(kw2 if stage<=2 else kw))
 def check():
  vs={'xq':err(xq,ref['xq']),'xs':err(xs.flatten(),ref['xs'].view(torch.uint8).flatten())}
  for ss,aa,bb in [(2,y,'raw'),(3,q,'q'),(4,qbr,'qb'),(5,qo,'qo'),(5,kv,'kv')]:
   if stage>=ss:vs[bb]=err(aa,ref[bb])
  if stage>=3:vs['qs']=err(s.flatten(),ref['qs'].view(torch.uint8).flatten())
  if stage>=5:vs['cache']=err(cache2,cache)
  vs['health']=int(health[0]);return vs
else:
 exec(Path('native_output.py').read_text().split('rr=n(o);')[0])
 import atom.models.deepseek_v4 as dv
 orig=dv.inverse_rope_group_quant;cap={}
 def hook(*args,**kwargs):cap['args']=args;cap['kwargs']=kwargs;return orig(*args,**kwargs)
 dv.inverse_rope_group_quant=hook;n.oa(o);dv.inverse_rope_group_quant=orig
 ref={};counter=a[-2];health=a[-1]
 def nr():
  xx,ss=orig(*cap['args'],**cap['kwargs']);ref.update(xq=xx,xs=ss)
  if stage>=2:aa=dv.batched_gemm_a8w8_mxscale_bpreshuffle(xx,n._wo_a_w_fp8,ss,n._wo_a_w_scale,dtype=torch.bfloat16).reshape(16,8192);ref['oa']=aa
  if stage>=3:qq,qs=per_1x128_e8m0_quant(aa,torch.float8_e4m3fn,True);ref.update(aq=qq,as_=qs)
  if stage>=4:rr=n.wo_b(qq,x_scale=qs);ref['ob']=rr
 def uf():
  ie[(512,)](o,c,sn,a[0],a[1],16,**kw1)
  if stage>=2:ge[(256,)](a[0],a[1],wap,sa,a[3],16,8192,4096,32,32,1024,4,True,**kw)
  if stage>=3:qe[(64,)](a[3],a[4],a[5],16,8192,**kw2)
  if stage>=4:ge[(64,)](a[4],a[5],wbp,sb,a[7],16,4096,8192,16,64,512,obnb,**kw)
 def ff():return fused_o[(256,)](o,c,sn,a[0],a[1],wap,sa,a[3],a[4],a[5],wbp,sb,a[7],counter,health,epochs,16,stage,**kw)
 def check():
  vs={'xq':err(a[0],ref['xq']),'xs':err(a[1],ref['xs'].view(torch.uint8))}
  for ss,aa,bb in [(2,a[3],'oa'),(3,a[4],'aq'),(4,a[7],'ob')]:
   if stage>=ss:vs[bb]=err(aa,ref[bb])
  if stage>=3:vs['as']=err(a[5].flatten(),ref['as_'].view(torch.uint8).flatten())
  vs['health']=int(health[0]);return vs
nr();uf();torch.cuda.synchronize();val={'unfused':check()};print('UNFUSED_CHECK',val,flush=True)

if kind=='q':
 for ten in [y,qbr,qo,kv]:ten.fill_(float('nan'))
 xq.view(torch.uint8).fill_(255);xs.fill_(255);q.view(torch.uint8).fill_(255);s.fill_(255)
else:
 for ten in [a[3],a[7]]:ten.fill_(float('nan'))
 for ten in [a[0],a[1],a[4],a[5]]:ten.view(torch.uint8).fill_(255)
obj=ff();torch.cuda.synchronize();val['fused']=check();print('FUSED_CHECK',val,flush=True)
import ctypes
hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')
active=ctypes.c_int()
status=hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(active),ctypes.c_void_p(obj.function),ctypes.c_int(obj.metadata.num_warps*32),ctypes.c_size_t(obj.metadata.shared))
print('OCCUPANCY',status,active.value,'MPs',torch.cuda.get_device_properties(0).multi_processor_count,flush=True)
(ROOT/f'{kind}{stage}.amdgcn').write_text(obj.asm['amdgcn'])
(ROOT/f'{kind}{stage}_resources.json').write_text(json.dumps(dict(regs=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared,metadata=str(obj.metadata))))
print('STATE',counter.data_ptr(),counter[::64][:70].tolist(),health.tolist(),epochs.tolist(),flush=True) if int(health[0]) else None
assert int(health[0])==0,val
assert all(v.get('relative_rms',0)<1e-4 for v in val['fused'].values() if isinstance(v,dict)),val
for _ in range(20):
 ff();torch.cuda.synchronize()
 if int(health[0]):
  print('REPLAY_STATE',counter.data_ptr(),counter[::64][:70].tolist(),health.tolist(),epochs.tolist(),flush=True)
  raise RuntimeError('counter health failure')
bench(f'{kind}{stage}',{'native':nr,'unfused_gluon':uf,'composed':ff},val)
assert int(health[0])==0,int(health[0])
