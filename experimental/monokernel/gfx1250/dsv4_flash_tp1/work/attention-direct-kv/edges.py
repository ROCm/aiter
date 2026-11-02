from common import *
rows=[]
for pages,t in [(1,128),(128,127),(128,128),(1024,1024)]:
 for slot in sorted(set([0,pages//2,pages-1])):
  for s in [1,2,4]:
   for mode in ['normal','duplicate','absent','invalid']:
    c=makecache(pages);idx=(torch.arange(t,device='cuda')%pages).int()
    if mode=='duplicate':idx[::3]=slot
    if mode=='absent':idx[idx==slot]=-1
    if mode=='invalid':idx.fill_(-1)
    refcache=c.clone();refcache[slot]=KV;ref=reference(Q,refcache,idx,Sink)
    c[slot].fill_(123)
    obj=AttentionRegion(c,idx,Sink,splits=s,write_slot=slot,direct=True)
    out=obj(Q,KV);torch.cuda.synchronize();e=error(out,ref)
    assert e<.001 and obj.check() and torch.equal(c,refcache),(pages,slot,s,mode,e)
    rows.append(dict(pages=pages,t=t,slot=slot,split=s,mode=mode,error=e))
(root/'edges.json').write_text(json.dumps(rows,indent=2));print('passed',len(rows),flush=True)
