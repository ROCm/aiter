from common import *
import numpy as np
from safetensors import safe_open
from aiter.tuned_gemm import tgemm
from atom.model_ops.v4_kernels.fused_compress import fused_compress_attn
from atom.model_ops.v4_kernels.state_writes import update_compressor_states
from atom.model_ops.v4_kernels.compress_plan import CompressPlan
from compress_defs import batch
case=int(sys.argv[1]);r=128 if case==1 else 4;d=128 if case==2 else 512;quant=case==2;df=d*(2 if r==4 else 1);state=r*(2 if r==4 else 1);n=df*2;m=8
layer=[2,3,2][case];prefix=f'layers.{layer}.attn.'+('indexer.' if quant else '')+'compressor.'
index=json.load(open('/data/DeepSeek-V4-Flash/model.safetensors.index.json'))['weight_map']
def load(s):
 k=prefix+s
 with safe_open('/data/DeepSeek-V4-Flash/'+index[k],framework='pt') as fh:return fh.get_tensor(k).cuda().contiguous()
w=torch.cat([load('wkv.weight'),load('wgate.weight')]);ape=load('ape');norm=load('norm.weight')
torch.manual_seed(713+case);rand=lambda *s:torch.randn(*s,device='cuda');iz=lambda *s:torch.zeros(*s,device='cuda',dtype=torch.int32)
x=rand(m,4096).bfloat16()*.05;y=torch.empty((m,n),device='cuda',dtype=torch.bfloat16);cos=rand(2304,32).bfloat16();sin=rand(2304,32).bfloat16()
ks=rand(m,state,df);ss=rand(m,state,df);ks2=ks.clone();ss2=ss.clone();cache=torch.zeros((m*16,256//r,d),device='cuda',dtype=torch.float8_e4m3fn if quant else torch.bfloat16);scale=torch.zeros((m*16,256//r),device='cuda');cache2=cache.clone();scale2=scale.clone()
slots=torch.arange(m,device='cuda',dtype=torch.int32);blocks=torch.arange(m*16,device='cuda',dtype=torch.int32).reshape(m,16)
cp=iz(m,4);wp=iz(m,4);ncp=iz(m,4);nwp=iz(m,4);ncp[:,0]=slots;ncp[:,1]=slots;nwp[:,0]=slots;nwp[:,1]=slots
plan=CompressPlan(ncp,nwp,m,m,np.arange(m+1,dtype=np.int32))
c=(ape,norm,cos,sin,cp,wp,slots,slots,ks2,ss2,cache2,scale2,blocks)
meta=(d,r,r==4,state,ks2.stride(0),ks2.stride(1),ss2.stride(0),ss2.stride(1),cache2.stride(0),cache2.stride(1),scale2.stride(0),quant)
ct=iz(1);h=iz(1)
def native():
 out=tgemm.mm(x,w,None,otype=torch.bfloat16);kv=out[:,:df];score=out[:,df:]
 fused_compress_attn(kv_in=kv,score_in=score,kv_state=ks,score_state=ss,plan=plan,state_slot_mapping=slots,ape=ape,rms_weight=norm,rms_eps=1e-6,cos_cache=cos,sin_cache=sin,kv_cache=cache,block_tables=blocks,k_per_block=256//r,overlap=r==4,ratio=r,head_dim=d,rope_head_dim=64,quant=quant,cache_scale=scale,use_ue8m0=True,preshuffle=True,fp8_max=448.)
 update_compressor_states(kv,score,ape,ks,ss,write_plan=nwp,state_slot_mapping=slots,ratio=r,overlap=r==4)
 return out
def fused():
 batch[(256,)](x,w,y,c,ct,h,meta,n,m,**kw)
 return y
for pos in [1024,1535]:
 cp[:,2]=pos if (pos+1)%r==0 else -1;cp[:,3]=state-1;wp[:,2]=pos
 ncp[:,2:]=cp[:,2:];nwp[:,2:]=wp[:,2:]
 ks2.copy_(ks);ss2.copy_(ss);cache2.copy_(cache);scale2.copy_(scale)
 rr=native();vv=fused();torch.cuda.synchronize();check=dict(projection=err(vv,rr),state=err(ks2,ks),cache=err(cache2,cache),health=int(h))
 bench('compress_case'+str(case)+'_pos'+str(pos),{'native':native,'fused':fused},check)
