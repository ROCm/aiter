"""Isolated four-wave SWA arithmetic and oracle fixture. GPU0 only."""
import os,sys,json,importlib.util
from pathlib import Path
os.environ.update(HIP_VISIBLE_DEVICES='0',NATIVE_ATTENTION_WAVES='1' if os.getenv('TASK_NATIVE','1')=='1' or os.getenv('PACK_ATTENTION','0')=='1' else '4',NATIVE_MOE_PACK='1',NATIVE_ROUTING_REFERENCE='serving')
for k in ('NATIVE_INDEPENDENT_BUNDLES','NATIVE_LOCAL_REDUCE_ADD','NATIVE_SKIP_FINAL_EDGE','NATIVE_WAVE_EDGE','NATIVE_C8_SYNC'):
 os.environ.pop(k,None)
ROOT=Path('/work/swa-fourwave');REF=Path('/work/c16-native-reference')
sys.path[:0]=[str(ROOT),str(REF),str(REF/'ffn'),str(REF/'output')]
import torch
import full_swa
from bridge.stage import build
from ffn.capture import Capture
import utilities as u

def adapt(e,name):
 factory=build
 if os.getenv('NATIVE_PARITY','1')=='1' and e['function'].__name__ in ('_gemm_a16w16_bandwidth_bound_kernel','_gemm_a16w16_gfx1250_bandwidth_bound_kernel','_keepk_sort0'):
  from native_stage import build as factory
 mod,args,m=factory(e,ROOT/'generated',name,physical_warps=4)
 assert e['compiled'].metadata.num_warps<=4,(name,e['compiled'].metadata)
 # Check physical waves too: warp specialization can add worker waves.
 asm=e['compiled'].asm['amdgcn']
 (ROOT/'generated'/f'{name}.s').write_text(asm)
 return dict(name=name,body=mod.body,values=args,types=m['types'][:-4],threads=m['num_warps']*32,total=m['total'],lds=m['lds'],module=mod,entry=e)

def capture(fn,grid,args=(),kw=None,name='custom'):
 with Capture(ROOT/'capture'/name) as c:fn[grid](*args,**(kw or {}))
 return adapt(c.entries[-1],name)

def recompile(e,name,overrides=None,fn=None):
 args=list(e['args']);kw=dict(e['kwargs']);kw.update(num_warps=4)
 orig=e['function'];fn=fn or orig
 for k,v in (overrides or {}).items():
  idx=orig.arg_names.index(k) if k in orig.arg_names else -1
  if 0<=idx<len(args):args[idx]=v
  else:kw[k]=v
 grid=kw.pop('grid');kw.pop('warmup',None)
 return capture(fn,grid,args,kw,name)

def prepare():
 print('Preparing checkpoint/native fixtures',flush=True)
 tagged=os.getenv('TAGGED_MHC','0')=='1'
 if os.getenv('MHC_ROLES','1')=='1':assert not tagged and os.getenv('NATIVE_TASK_DESIGN','1')=='1', 'parallel mHC roles require the native task schedule'
 if os.getenv('PACK_ATTENTION','0')=='1':assert os.getenv('ATTN_SPLITS','8')=='16', 'packing and attention retiling are separate experiments'
 if tagged:assert os.getenv('NATIVE_TASK_DESIGN','1')=='1'
 p=full_swa.prepare();st=p['full']['stages'];ke=p['keep'];original=list(st)
 print('Fixtures ready',flush=True)
 # Local copy: make fused RMSNorm run within the four compute waves. Preserve
 # all numerical operations; only disable the extra Sinkhorn worker partition.
 src=(ROOT/'source/mhc_native.py').read_text().replace('WS: gl.constexpr = FUSE_RMSNORM  # whole rows per CTA, one K-block','WS: gl.constexpr = False  # four-wave schedule: no extra worker waves')
 if os.getenv('NATIVE_TASK_DESIGN','1')=='1':
  src=src.replace('if not WS:', 'if not WS and SINKHORN_REPEAT > 0:').replace('if not WS and pid_k == 0:', 'if not WS and SINKHORN_REPEAT > 0 and pid_k == 0:')
 if tagged:
  from tagged_source import transform
  src=transform(src)
 path=ROOT/'mhc_four.py';path.write_text(src)
 spec=importlib.util.spec_from_file_location('mhc_four',path);mh=importlib.util.module_from_spec(spec);spec.loader.exec_module(mh)
 direct=os.getenv('DIRECT_RESIDUAL','1')=='1'
 if not direct:st[0]=capture(u.unshuffle,(256,),(ke['residual'],p['first']['fixture'].keep[1]),dict(num_warps=4),'residual_unshuffle4')
 # The native HIP first-pre GEMM already uses four waves and its exact128 splits.
 first=p['first'];p0,c0,x0=first['outputs'];ma=first['fixture'];o,sq=ma.keep[2:4];plain=ma.keep[1]
 if direct:
  import shuffled_pre
  st[0]=dict(st[0],name='unshuffle_removed',total=0,threads=0,lds=0)
  st[1]=dict(st[1],body=shuffled_pre.gemm,name='mhc_pre_shuffled_tdm',values=[t.data_ptr() for t in (o,sq,ke['residual'],ma.fn)])
  plain=ke['residual']
 chunk,ap,r4,r16,sk,sm,sr=mh.create_reduce_apply_layouts(4,1,4,128)
 args=(p0,c0,x0,o,sq,ma.scale,ma.base,plain,ma.norm,16,32,1e-6,1e-6,1e-6,1e-6,2.)
 kw=dict(C=4096,HC=4,S=128,ROWS=1,C_CTA=4096,CHUNK=chunk,N_CHUNKS=4096//chunk,SINKHORN_REPEAT=0 if os.getenv('NATIVE_TASK_DESIGN','1')=='1' else 20,RES_SHUFFLED=direct,RES_KS=64 if direct else 32,FUSE_RMSNORM=True,APPLY=ap,RED4=r4,RED16=r16,SK=sk,SMEM_MIX=sm,SMEM_RES=sr,num_warps=4,enable_fp_fusion=False)
 if tagged:
  mail=torch.zeros((16,2048),device='cuda',dtype=torch.int64);tag_health=torch.zeros(1,device='cuda',dtype=torch.int32)
  p['tagged']={'mail':mail,'health':tag_health}
  kw.update(TAGGED=True,tagged_ptr=mail,tag_epoch=17)
 st[2]=capture(mh._mhc_pre_reduce_apply_gfx1250_kernel,(16,1),args,kw,'first_reduce4')
 if tagged:
  k=st[2]['entry']['compiled']
  runtime_names=[n for i,(n,t) in enumerate(k.src.signature.items()) if t!='constexpr' and (i,) not in k.src.constants]
  st[2]['epoch_arg']=runtime_names.index('tag_epoch')
 from first_post import wrap
 st[2]=wrap(st[2],[t.data_ptr() for t in (p0,c0,o,sq,ma.scale,ma.base)],tagged=tagged,parallel=os.getenv("MHC_ROLES","1")=="1")
 if os.getenv('QB_FOUR_BARRIER','1')=='1':
  import qkv_four
  st[6]=dict(st[6],body=qkv_four.qb_body,name='qb_four_wave_barrier')
 print('First reduction prepared',flush=True)
 # Re-export native attention with physical four-wave barriers.
 for i,e in enumerate(p['attn']['keep'][0].entries):st[8+i]=adapt(e,'attention4_'+str(i))
 if os.getenv('TASK_NATIVE','1')=='1' or os.getenv('PACK_ATTENTION','0')=='1':
  from attention_pack import build as pack_attention
  st[8]=pack_attention(p['attn']['keep'][0].entries[0])
 if os.getenv('TASK_NATIVE','1')!='1' and os.getenv('ATTN_SPLITS','8')!='16':
  splits=int(os.getenv('ATTN_SPLITS','8'));assert ke['idx'].numel()==16*128, 'retiled attention is validated for 128 keys per token' ;block=int(os.getenv('ATTN_BLOCK_K','16'))
  for j,e in enumerate(p['attn']['keep'][0].entries):
   args=list(e['args']);kw=dict(e['kwargs']);kw.pop('warmup',None);grid=kw.pop('grid')
   for key,value in dict(KV_SPLITS=splits,**({'BLOCK_K':block} if j==0 else {})).items():
    idx=e['function'].arg_names.index(key)
    if idx<len(args):args[idx]=value
    else:kw[key]=value
   if j==0:grid=(16,4,splits)
   st[8+j]=capture(e['function'],grid,args,kw,'attention_retile_'+str(j))
 # Mid GEMM native4, reducer native source with the extra partition removed.
 entries=ke['mf'].keep[0]
 st[14]=adapt(entries[0],'mid_projection4')
 st[15]=recompile(entries[1],'mid_reduce4',fn=mh._mhc_pre_reduce_apply_gfx1250_kernel,overrides={'tagged_ptr':None,'tag_epoch':17,'TAGGED':False} if tagged else None)
 if os.getenv('MHC_ROLES','1')=='1':
  from mhc_roles import build as build_roles
  st[15]=build_roles(entries[1],capture)
 shared=p['shared'];sgu=shared['entries'][0][0][4];activated=shared['acts'][0][0][0]
 st[18]=capture(u.act,(16,),(sgu,activated),dict(num_warps=4,enable_fp_fusion=False),'shared_act4')
 for i,s in enumerate(st):
  if s.get('name')=='learned_scratch_initialize':
   keep=next(e for e in p['routed']['capture'].entries if e['function'].__name__=='_keepk_sort0');bound=dict(zip(keep['function'].arg_names,keep['args']));bound.update(keep['kwargs'])
   st[i]=capture(u.clear,(1,),tuple(bound[k] for k in ('Pop','Hist','Part')),dict(num_warps=4),'routing_clear4')
  elif 'entry' in s:
   e=s['entry']
   if e['compiled'].metadata.num_warps>4:
    assert i in (23,26,29), ('Unaudited native wave-count exception',i,e['function'].__name__)
    overrides={k:4 for k in ('NUM_WARPS','num_warps') if k in e['function'].arg_names}
    st[i]=recompile(e,'stage4_'+str(i),overrides)
   elif s not in st[:21]:
    st[i]=adapt(e,'stage4_'+str(i))
 if tagged:
  import tagged_quant
  st[3]=dict(st[3],name='tagged_input_quant',body=tagged_quant.body,types=['uint64']*5+['uint32'],values=st[3]['values']+[mail.data_ptr(),tag_health.data_ptr(),17],epoch_arg=5)
 st[-1]=capture(u.add_row,(16,),(p['shared']['output'],p['routed']['output'],p['add']['output']),dict(num_warps=4),'output_add4')
 if os.getenv('NATIVE_PARITY','1')=='1':
  from native_compiled_gemms import apply
  apply(st)
  import native_hip_four
  if direct:st[1]=dict(st[1],body=native_hip_four.pre,name='native_pre_full_cta_barrier')
  st[5]=dict(st[5],body=native_hip_four.norm,name='native_qnorm_full_cta_barrier')
  from native_policy import check
  check(st)
 if os.getenv('TASK_NATIVE','1')=='1':
  from native_task_bodies import apply
  apply(st,p)
  st[30]=capture(u.add_native_vector_rows,(16,),(p['shared']['output'],p['routed']['output'],p['add']['output']),dict(num_warps=4),'native_vector_output_add')
  if os.getenv('NATIVE_TOPK','1')=='1' and os.getenv('FOLD_REDUCE','1')=='1' and os.getenv('FOLD_ACT','1')=='1':
   from native_policy import check_tasks
   check_tasks(st)
 for i,s in enumerate(st):
  assert s['threads']<=128,(i,s.get('name'),s['threads'])
  print(i,s.get('name'),s['threads'],s['total'],s['lds'],flush=True)
 (ROOT/'STAGES.json').write_text(json.dumps([dict(i=i,name=s.get('name',str(i)),threads=s['threads'],tasks=s['total'],lds=s['lds']) for i,s in enumerate(st)],indent=2))
 a,b=[st[j]['entry']['bound'] for j in (27,28)]
 assert a['grid_n']==b['grid_n']==32 and a['ExptData'].data_ptr()==b['ExptData'].data_ptr()
 p['stages4']=st;p['original_stages']=original
 return p

if __name__=='__main__':
 p=prepare();print('PREPARED',flush=True)
