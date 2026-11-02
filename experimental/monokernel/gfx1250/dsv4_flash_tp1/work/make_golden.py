import json
from pathlib import Path
import torch, aiter
from aiter.ops.quant import per_group_quant_hip
root=Path('/work/golden');root.mkdir(exist_ok=True)
summary=[]
for m,c,shuffled in [(m,7168,s) for m in (1,2,7,8,16,32,64,128) for s in (False,True)]+[(16,4096,True)]:
    torch.manual_seed(1234+m)
    x=torch.randn(m,c,device='cuda',dtype=torch.bfloat16)
    res=torch.randn(m,4,c,device='cuda',dtype=torch.bfloat16)
    if m==2:x.zero_();res.zero_()
    post=torch.sigmoid(torch.randn(m,4,device='cuda'))*2
    comb=torch.softmax(torch.randn(m,4,4,device='cuda'),-1)
    fn=aiter.mhc_shuffle_fn(torch.randn(24,4*c,device='cuda')*0.01)
    scale=torch.tensor([0.25,0.25,0.25],device='cuda')
    base=torch.randn(24,device='cuda')*0.2
    norm=(torch.randn(c,device='cuda')*0.05+1).bfloat16()
    if shuffled:res=aiter.mhc_res_shuffle(res)
    args=(x,res,post,comb,fn,scale,base)
    kwargs=dict(norm_weight=norm,res_preshuffle=shuffled,w_preshuffle_bf16=True,
                hc_post_mult_value=2.0,backend='gluon')
    expected=aiter.mhc_fused_post_pre(*args,**kwargs)
    q,qs=per_group_quant_hip(expected[2],quant_dtype=torch.float8_e4m3fn,
                           group_size=128,transpose_scale=True,scale_type=torch.float8_e8m0fnu)
    name=f'm{m}_c{c}_shuffle{int(shuffled)}'
    torch.save(dict(args=tuple(t.cpu() for t in args),norm=norm.cpu(),shuffled=shuffled,
                    expected=tuple(t.cpu() for t in (*expected,q,qs))),root/f'{name}.pt')
    summary.append(name)
(root/'manifest.json').write_text(json.dumps(summary,indent=2))
