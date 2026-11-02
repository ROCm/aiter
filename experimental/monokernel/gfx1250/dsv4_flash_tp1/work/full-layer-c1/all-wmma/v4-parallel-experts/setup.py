"""Isolated checkpoint-backed full-layer test, not an ATOM serving benchmark."""
import argparse, inspect, json, time
from pathlib import Path
import torch
from safetensors import safe_open
from kernel import full_layer

root=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--compile-only',action='store_true');ap.add_argument('--grid',type=int,default=64);args=ap.parse_args()
torch.manual_seed(0)
assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx1250')
assert args.grid<=torch.cuda.get_device_properties(0).multi_processor_count
p=Path('/data/DeepSeek-V4-Flash');index=json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
handles={}
def tensor(key):
 name=index[key]
 if name not in handles:handles[name]=safe_open(str(p/name),framework='pt',device='cpu')
 return handles[name].get_tensor(key)
def load(key):return tensor('layers.0.'+key).cuda()
def scale(key):return load(key).view(torch.uint8)
d={}
for short,key in [('HAF','hc_attn_fn'),('HAS','hc_attn_scale'),('HAB','hc_attn_base'),('HFN','hc_ffn_fn'),('HFS','hc_ffn_scale'),('HFB','hc_ffn_base'),('AN','attn_norm.weight'),('FN','ffn_norm.weight'),('QN','attn.q_norm.weight'),('KN','attn.kv_norm.weight'),('Sink','attn.attn_sink'),('Gate','ffn.gate.weight'),('Tid','ffn.gate.tid2eid')]:d[short]=load(key)
for w,s,key in [('QA','QAS','attn.wq_a'),('QB','QBS','attn.wq_b'),('KVW','KVS','attn.wkv'),('OA','OAS','attn.wo_a'),('OB','OBS','attn.wo_b'),('SW1','SS1','ffn.shared_experts.w1'),('SW3','SS3','ffn.shared_experts.w3'),('SW2','SS2','ffn.shared_experts.w2')]:
 d[w]=load(key+'.weight');d[s]=scale(key+'.scale')
print('Dense weights loaded',flush=True)
for w,s,suffix in [('EW1','ES1','w1'),('EW3','ES3','w3'),('EW2','ES2','w2')]:
 ws=load('ffn.experts.0.'+suffix+'.weight');ss=scale('ffn.experts.0.'+suffix+'.scale')
 d[w]=torch.empty((256,*ws.shape),dtype=ws.dtype,device='cuda');d[s]=torch.empty((256,*ss.shape),dtype=ss.dtype,device='cuda')
 for e in range(256):
  d[w][e].copy_(tensor(f'layers.0.ffn.experts.{e}.{suffix}.weight'))
  d[s][e].copy_(tensor(f'layers.0.ffn.experts.{e}.{suffix}.scale').view(torch.uint8))
 print(w,'loaded',flush=True)
token=42
with safe_open(str(p/index['embed.weight']),framework='pt',device='cpu') as f:embed=f.get_slice('embed.weight')[token:token+1].cuda()
d['R']=embed.repeat(4,1).contiguous();d['Out']=torch.empty_like(d['R']);d['Token']=torch.tensor([token],device='cuda',dtype=torch.int64)
d['Cache']=torch.randn((128,512),device='cuda',dtype=torch.bfloat16)*.2
# Controlled rotary angle; uses the same exact input table in both paths. Live
# ATOM replay must supply its YaRN table instead of this unit-test fixture.
angles=1024*10000.**(-torch.arange(32,device='cuda',dtype=torch.float32)/32)
d['Cos']=angles.cos();d['Sin']=angles.sin()
sizes={'MixA':24,'MixF':24,'R2':16384,'X':4096,'XQ':4096,'QL':1024,'QLQ':1024,'Q':32768,'KV':512,'QRot':32768,'KVRot':512,'AO':32768,'AR':32768,'AQ':32768,'OL':8192,'OLQ':8192,'Attn':4096,'FF':4096,'FFQ':4096,'FFMX':4096,'SG':2048,'SU':2048,'SA':2048,'SAQ':2048,'SO':4096,'Logits':256,'Ids':6,'Gamma':6,'EG':12288,'EU':12288,'EA':12288,'EAQ':12288,'EO':24576,'Sum':4096}
fp32={'QL','Q','KV','EG','EU','EA','MixA','MixF','XQ','QLQ','AQ','OLQ','FFQ','FFMX','SAQ','EAQ','Gamma'}
for k,n in sizes.items():d[k]=torch.empty(n,device='cuda',dtype=torch.int32 if k=='Ids' else torch.float32 if k in fp32 else torch.bfloat16)
d['Counter']=torch.zeros(1,device='cuda',dtype=torch.int32);d['Health']=torch.zeros_like(d['Counter'])
keys=list(inspect.signature(full_layer.fn).parameters)[:-1]; values=[d[k] for k in keys]
