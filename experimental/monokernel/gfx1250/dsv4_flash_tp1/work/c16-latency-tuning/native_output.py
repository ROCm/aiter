from common import *
import importlib.util
spec=importlib.util.spec_from_file_location('native_output_reference','/work/full-attention-native/output-native/native_output.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
n=mod.NativeOutput();n.positions=torch.full((M,),1023,device='cuda',dtype=torch.int64)
f=n.fixture;wa=f['OA'].cuda().contiguous();sa=f['OAS'].cuda().view(torch.uint8).contiguous();wb=f['OB'].cuda().contiguous();sb=f['OBS'].cuda().view(torch.uint8).contiguous()
if os.getenv('OUTPUT_PACKED','0')=='1':
 def pack(w):
  n,k=w.shape;return w.reshape(n//16,16,k//16,16).permute(0,2,1,3).contiguous().reshape(n,k)
 wa=pack(wa);wb=pack(wb)
o=torch.randn((M,64,512),device='cuda',dtype=torch.bfloat16)*.1
sys.path.insert(0,'/work/atom-actual-microbench/qkv')
from native import NativeQKV
nq=NativeQKV();c=nq.cos[1023].reshape(1,32).repeat(M,1);sn=nq.sin[1023].reshape(1,32).repeat(M,1)
e=lambda shape,dt=torch.bfloat16:torch.empty(shape,device='cuda',dtype=dt)
a=[e((M,32768),torch.float8_e4m3fn),e((M,256),torch.uint8),e((4,M,8192),torch.float32),e((M,8192)),e((M,8192),torch.float8_e4m3fn),e((M,64),torch.uint8),e((8,M,4096),torch.float32),e((M,4096)),torch.zeros(1024*64,device='cuda',dtype=torch.int32),torch.zeros(1,device='cuda',dtype=torch.int32)]
from isolated import output
from frozen_before.isolated import output as previous_op
def fo(op=output):
 op[(256,)](o,c,sn,wa,sa,wb,sb,*a,M,**kw)
 return a[7]
rr=n(o);fo();torch.cuda.synchronize();assert int(a[-1])==0
bench('output_projection',{'native':lambda:n(o),'previous':lambda:fo(previous_op),'fused':fo},err(a[7],rr))
