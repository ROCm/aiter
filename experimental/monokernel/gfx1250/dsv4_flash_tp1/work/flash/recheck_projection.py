import copy,json,itertools
from pathlib import Path
import torch
from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8_preshuffle
from triton.testing import do_bench_cudagraph
r=Path(__file__).parent
d=torch.load(r/'failing-inputs/8_4096_8192.pt',weights_only=False)
d={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in d.items()}
rows=[]
for split,bk,bn in itertools.product([1,2,4,8],[128,256],[32,64,128]):
 c=copy.deepcopy(d['config']);c.update(NUM_KSPLIT=split,BLOCK_SIZE_K=bk,BLOCK_SIZE_N=bn)
 row={'config':c}
 try:
  def run():return gemm_afp8wfp8_preshuffle(d['x'],d['weight'],d['x_scale'].view(torch.uint8),d['weight_scale'].view(torch.uint8),backend='gluon',is_x_scale_transposed=True,config=copy.deepcopy(c))
  y=run();row['mismatches']=(y!=d['reference']).sum().item();row['max_abs']=(y.float()-d['reference'].float()).abs().max().item();row['us']=do_bench_cudagraph(run,rep=50)*1000
 except Exception as e:row['error']=str(e)
 rows.append(row);print(row,flush=True);(r/'real_projection_recheck.json').write_text(json.dumps(rows,indent=2))
