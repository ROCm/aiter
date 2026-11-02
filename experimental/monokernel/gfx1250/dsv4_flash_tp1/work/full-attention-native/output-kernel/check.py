import sys,json
from pathlib import Path
sys.path.insert(0,'/work/mhc-optimized')
import torch
from output_region import output_kernel,output_stage
root=Path(__file__).resolve().parent
f=torch.load('/work/full-attention-native/output-native/fixture.pt',weights_only=True)
print({k:(v.shape,v.dtype) for k,v in f.items() if isinstance(v,torch.Tensor)},flush=True)
a={k:v.cuda().contiguous() for k,v in f.items() if isinstance(v,torch.Tensor)}
a['OAS']=a['OAS'].view(torch.uint8);a['OBS']=a['OBS'].view(torch.uint8)
q=torch.load('/work/mhc-optimized/fixture.pt',weights_only=True)
cos=q['cos'].cuda();sin=q['sin'].cuda()
o=torch.randn(64,512,device='cuda').bfloat16()
z=lambda shape,dtype=torch.float32:torch.zeros(shape,device='cuda',dtype=dtype)
xq=z(32768,torch.float8_e4m3fn);xs=z(256,torch.uint8);ap=z((4,8192));ay=z(8192,torch.bfloat16);aq=z(8192,torch.float8_e4m3fn);asc=z(64,torch.uint8);bp=z((8,4096));y=z(4096,torch.bfloat16);counter=z(1,torch.int32);health=z(1,torch.int32)
args=[o,cos,sin,a['OA'],a['OAS'],a['OB'],a['OBS'],xq,xs,ap,ay,aq,asc,bp,y]
kw=dict(num_warps=4,enable_fp_fusion=False)
for i in range(5):
 output_stage[(256,)](*args,STAGE=i,**kw);torch.cuda.synchronize();print('stage',i,'ok',flush=True)
ref=y.clone();obj=output_kernel[(256,)](*args,counter,health,**kw);torch.cuda.synchronize();print('fused',torch.equal(y,ref),health,obj.n_regs,obj.metadata,flush=True)
(root/'output.amdgcn').write_text(obj.asm['amdgcn'])
