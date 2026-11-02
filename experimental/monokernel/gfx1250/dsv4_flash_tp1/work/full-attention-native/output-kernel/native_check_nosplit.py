exec(open('/work/full-attention-native/output-kernel/check.py').read().split('for i in range(5):')[0])
sys.path.insert(0,'/work/full-attention-native/output-native')
from native_output import NativeOutput
n=NativeOutput(position=1024)
c,si=n.rotary_emb.cos_sin_2d();print('cosshapes',c.shape,si.shape,flush=True)
cos=c[1024].contiguous();sin=si[1024].contiguous();args[1]=cos;args[2]=sin
err=lambda x,y:float(((x.float()-y.float()).square().mean()/y.float().square().mean()).sqrt())
kw.update(SA=1,SB=1)
rows=[]
for case in range(64):
 o.copy_(torch.randn(o.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(991000+case)).bfloat16())
 ya=n.oa(o.view(1,64,512));yb=n.wo_b(ya)
 output_kernel[(256,)](*args,counter,health,**kw);torch.cuda.synchronize()
 row={'case':case,'oa':err(ay,ya.flatten()),'ob':err(y,yb.flatten()),'health':int(health.item())};rows.append(row)
 if case<4:print(row,flush=True)
print('max',{k:max(r[k] for r in rows) for k in ['oa','ob','health']},flush=True)
(root/'native-validation-nosplit.json').write_text(json.dumps(rows,indent=2))
