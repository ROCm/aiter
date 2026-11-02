from ffn_setup import *
f=FFNState();checks=[]
def err(a,b):return float(((a.float()-b.float()).square().mean()/b.float().square().mean().clamp_min(1e-30)).sqrt())
for seed in range(32):
 torch.manual_seed(seed);f.r.normal_();ref=f.native(f.r).clone();f.fused();torch.cuda.synchronize()
 checks.append({'seed':seed,'final':err(f.out,ref),'norm':err(f.x,f.native_norm),'logits':err(f.logits,f.native_logits),'shared':err(f.so,f.native_shared),'routed':err(f.mo['z'],f.native_routed),'ids':f.map.tolist(),'native_ids':f.rd.topk_ids.tolist(),'gamma':f.gamma.tolist(),'native_gamma':f.rd.gate_scal.tolist(),'health':f.health.item()})
(root/'ffn-validation.json').write_text(json.dumps(checks,indent=2));print(checks,flush=True)
