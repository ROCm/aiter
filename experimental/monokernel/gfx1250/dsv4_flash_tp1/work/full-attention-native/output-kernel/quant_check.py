exec(open('/work/full-attention-native/output-kernel/native_check.py').read().split('rows=[]')[0])
from aiter.ops.inverse_rope_group_quant import inverse_rope_group_quant
for case in range(8):
 o.copy_(torch.randn(o.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(991000+case)).bfloat16())
 nx,ns=inverse_rope_group_quant(o.view(1,64,512),n.positions,c,si,8)
 output_stage[(256,)](*args,STAGE=0,**kw);torch.cuda.synchronize()
 print(case,'qdiff',int((nx.flatten().view(torch.uint8)!=xq.view(torch.uint8)).sum()),'sdiff',int((ns.flatten().view(torch.uint8)!=xs).sum()),flush=True)
