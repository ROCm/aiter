import sys,json
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'prefix'));sys.path.insert(0,'/app/aiter')
import torch,aiter
from layout import unshuffle_kernel
r=torch.randn((1,4,4096),device='cuda').bfloat16();rs=aiter.mhc_res_shuffle(r);out=torch.empty_like(r)
k=unshuffle_kernel[(256,)](rs,out,num_warps=4,enable_fp_fusion=False);torch.cuda.synchronize()
assert torch.equal(out,r)
(root/'layout-check.json').write_text(json.dumps({'bitwise_equal':True,'lds':k.metadata.shared,'vgpr':k.n_regs,'spills':k.n_spills}))
print('layout pass')
