"""ATOM layer0 native mHC pre + RMSNorm, packed default policy.
Input residual MUST be preshuffled with aiter.mhc_res_shuffle.
Returns (normalized_x [M,D], post [M,4], comb [M,4,4]).
Packing is one-time setup and excluded from timing, as in ATOM weight loading.
"""
import aiter
class NativeMHC:
 def __init__(self,fn,scale,base,norm_weight):
  self.fn=aiter.mhc_shuffle_fn(fn.contiguous().float())
  self.scale=scale;self.base=base;self.norm_weight=norm_weight
 def __call__(self,residual):
  post,comb,x=aiter.mhc_pre(residual.view(-1,4,self.norm_weight.numel()),self.fn,self.scale,self.base,1e-6,1e-6,1e-6,2.,20,self.norm_weight,1e-6,w_preshuffle_bf16=1,res_preshuffle=True)
  return x,post.squeeze(-1),comb
 def post_pre(self,x,residual,post,comb):
  return aiter.mhc_fused_post_pre(x,residual.view(-1,4,self.norm_weight.numel()),post.view(-1,4,1),comb,self.fn,self.scale,self.base,1e-6,1e-6,1e-6,2.,20,self.norm_weight,1e-6,w_preshuffle_bf16=True,res_preshuffle=True)
