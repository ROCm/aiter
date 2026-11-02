"""Retest Flash Gluon projection candidates with rotated weights and exact checks."""
import copy,json,statistics
from pathlib import Path
import torch
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8_preshuffle
from triton.testing import do_bench_cudagraph
r=Path(__file__).parent;prior=json.loads((r/'attention_screen.json').read_text())
if (r/'attention_tune.json').exists():prior+=json.loads((r/'attention_tune.json').read_text())
keys=sorted({(z['m'],z['n'],z['k']) for z in prior});rows=[];selected={}
torch.manual_seed(1250)
for m,n,k in keys:
 valid=sorted([z for z in prior if (z['m'],z['n'],z['k'])==(m,n,k) and 'us' in z],key=lambda z:z['us'])
 if not valid:continue
 variants=[];seen=set()
 for z in valid:
  key=json.dumps(z['config'],sort_keys=True)
  if key not in seen:variants.append(z);seen.add(key)
  if len(variants)==2:break
 wf=torch.randn(n,k,device='cuda')*.1
 wb=wf.reshape(n//128,128,k//128,128).permute(0,2,1,3)
 ws=torch.pow(2.,torch.ceil(torch.log2(wb.abs().amax((2,3)).clamp(min=1e-10)/448))).to(torch.float8_e8m0fnu)
 wq=shuffle_weight((wf/ws.float().repeat_interleave(128,0).repeat_interleave(128,1)).to(torch.float8_e4m3fn),layout=(16,16))
 weights=[wq.clone() for _ in range(32)]
 x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
 xq,xs=per_group_quant_hip(x,quant_dtype=torch.float8_e4m3fn,group_size=128,transpose_scale=True,scale_type=torch.float8_e8m0fnu)
 ref=gemm_a8w8_blockscale_bpreshuffle(xq,wq,xs,ws)
 def base():
  for w in weights:gemm_a8w8_blockscale_bpreshuffle(xq,w,xs,ws)
 for z in variants:
  cfg=z['config'];row=dict(m=m,n=n,k=k,config=cfg,name=z['name'])
  try:
   got=gemm_afp8wfp8_preshuffle(xq,wq,xs.view(torch.uint8),ws.view(torch.uint8),backend='gluon',is_x_scale_transposed=True,config=copy.deepcopy(cfg))
   torch.testing.assert_close(got,ref,rtol=0,atol=0)
   def cand():
    for w in weights:gemm_afp8wfp8_preshuffle(xq,w,xs.view(torch.uint8),ws.view(torch.uint8),backend='gluon',is_x_scale_transposed=True,config=copy.deepcopy(cfg))
   bs=[];cs=[]
   for rep in range(5):
    for mode in ((False,True) if rep%2==0 else (True,False)):
     (cs if mode else bs).append(do_bench_cudagraph(cand if mode else base,rep=100)*1000/32)
   bu=statistics.median(bs);cu=statistics.median(cs)
   row.update(baseline_us=bu,candidate_us=cu,baseline_samples=bs,candidate_samples=cs,weight_rotation=32,exact=True)
   key=f'{m},{n},{k}'
   if cu<bu*.97 and bu-cu>.2:
    old=selected.get(key)
    if old is None or cu<old['candidate_us']:selected[key]=row
  except Exception as e:row['error']=str(e)
  rows.append(row);print({kk:vv for kk,vv in row.items() if kk!='config'},flush=True)
  (r/'attention_finalists.json').write_text(json.dumps(rows,indent=2))
 del weights,wq,wf,wb
(r/'selected_attention.json').write_text(json.dumps(selected,indent=2))
(r/'candidate/dsv4_flash_attention_configs.json').write_text(json.dumps({k:v['config'] for k,v in selected.items()},indent=2))
