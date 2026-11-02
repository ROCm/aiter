from wmma_ops import projection_probe
# Check each raw WMMA projection against independent FP32 torch matmul on the
# exact packed operands used by that projection (separate from propagation).
projection_checks={}
for name,x,xs,w,ws,n,k in [('QA',extra['XP'],extra['XS'],d['QA'],d['QAS'],1024,4096),('KV',extra['XP'],extra['XS'],d['KVW'],d['KVS'],512,4096),('QB',extra['LP'],extra['LS'],d['QB'],d['QBS'],32768,1024)]:
 out=torch.empty(n,device='cuda',dtype=torch.bfloat16)
 projection_probe[(256,)](x,xs,w,ws,out,n,k,256,num_warps=4,enable_fp_fusion=False)
 wf=w.float()*torch.pow(2.,ws.float()-127).repeat_interleave(128,0).repeat_interleave(128,1)
 xf=x.float()*torch.pow(2.,xs.float()-127).repeat_interleave(128)
 ref=torch.mv(wf,xf).bfloat16()
 delta=out.float()-ref.float()
 rr=(delta.square().mean()/ref.float().square().mean()).sqrt().item()
 projection_checks[name]=dict(relative_rms=rr,max_abs=delta.abs().max().item(),exact=torch.equal(out,ref))
 assert rr<.005,projection_checks
 del wf,xf,ref,out
(root/'projection-reference.json').write_text(json.dumps(projection_checks,indent=2))
print('independent projections',projection_checks,flush=True)
# New residual inputs derived from actual embedding rows, not just one fixture.
saved_r=d['R'].clone();cases=[]
for tok in [0,1,7,43,100,1024,8192,32000]:
 with safe_open(str(p/index['embed.weight']),framework='pt',device='cpu') as f:r=f.get_slice('embed.weight')[tok:tok+1].cuda()
 d['R'].copy_(r.repeat(4,1))
 for st in range(5):split_stage[(args.grid,)](*values,STAGE=st,**kw)
 split();expected_case={k:d[k].clone() for k in expected}
 wmma();torch.cuda.synchronize()
 metrics={k:((d[k].float()-v.float()).square().mean()/v.float().square().mean()).sqrt().item() for k,v in expected_case.items()}
 assert max(metrics.values())<.005,(tok,metrics)
 assert d['Health'].item()==0
 cases.append(dict(token=tok,relative_rms=metrics))
d['R'].copy_(saved_r)
for st in range(5):split_stage[(args.grid,)](*values,STAGE=st,**kw)
# Grid counter wrap and changing-input graph replay validation.
for initial in [2147483648-256,-256]:
 d['Counter'].fill_(initial);wmma();torch.cuda.synchronize();assert d['Health'].item()==0
cg=torch.cuda.CUDAGraph()
with torch.cuda.graph(cg):
 d['XQ'].neg_();wmma()
for _ in range(1000):cg.replay()
torch.cuda.synchronize();assert d['Health'].item()==0
actual={k:d[k].clone() for k in expected};split();torch.cuda.synchronize()
metrics={k:((actual[k].float()-d[k].float()).square().mean()/d[k].float().square().mean()).sqrt().item() for k in expected}
assert max(metrics.values())<.005,metrics
(root/'extended-validation.json').write_text(json.dumps(dict(embedding_cases=cases,graph_replays=1000,graph_relative_rms=metrics,counter_wrap=True,health=d['Health'].item()),indent=2))
print('EXTENDED VALIDATION PASSED',flush=True)
