import torch,numpy as np
from post import post
z=torch.load('diff.pt',weights_only=False);r,x,m,ref,out=[z[k] for k in ['r','x','m','ref','out']]
obj=post[(256,)](r,x,m,out,FMA=2,num_warps=4,enable_fp_fusion=False)
open('post.amdgcn','w').write(obj.asm['amdgcn'])
for _,j,k in (out!=ref).nonzero().tolist():
 vals=[float(r[0,i,k]) for i in range(4)];cs=[float(m[8+i*4+j]) for i in range(4)];xx=float(x[0,k]);ps=float(m[4+j]);a=np.float32(vals[0]*cs[0]);a=np.float32(xx*ps+float(a))
 for i in range(1,4):a=np.float32(vals[i]*cs[i]+float(a))
 print(vals,cs,xx,ps,'f32chain',a,'bf16',torch.tensor(float(a)).bfloat16(),'out',out[0,j,k],'ref',ref[0,j,k])
