from layer_setup import *
f=LayerState();data=json.loads((root/'validation.json').read_text());worst=sorted(data,key=lambda x:x['native_rms'],reverse=True)[:5]
inputs=list(fa.ps.f2['inputs']);gen=torch.Generator().manual_seed(20261020);inputs += [torch.randn(4,4096,generator=gen).bfloat16()*.05 for _ in range(8)]
def err(a,b):return float(((a.float()-b.float()).square().mean()/b.float().square().mean().clamp_min(1e-30)).sqrt())
rows=[]
for c in worst:
 f.a.setpos(c['position']);f.r.copy_(aiter.mhc_res_shuffle(inputs[c['case']].cuda().view(1,4,4096)))
 ref=f.native().clone();native_ids=f.f.rd.topk_ids.tolist();native_norm=f.f.native_norm.clone();f.fused();our=f.f.out.clone();ar=f.a.rout.clone();ourids=f.f.map.tolist()
 same_input=f.f.native(ar).clone();torch.cuda.synchronize()
 rows.append({**c,'native_ids':native_ids,'our_ids':ourids,'native_ffn_on_fused_attention_ids':f.f.rd.topk_ids.tolist(),'same_input_ffn_rms':err(our,same_input),'ffn_norm_rms':err(f.f.x,f.f.native_norm)})
(root/'diagnosis.json').write_text(json.dumps(rows,indent=2));print(rows,flush=True)
