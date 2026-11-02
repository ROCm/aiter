from pathlib import Path
import json,torch
from safetensors import safe_open
root=Path(__file__).resolve().parent
src=root.parent/'qkv-asm-opt/fixture.pt'
f=torch.load(src,weights_only=True)
p=Path('/data/DeepSeek-V4-Flash');idx=json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
for short,key in [('QB_bytes','layers.0.attn.wq_b.weight'),('QBS','layers.0.attn.wq_b.scale')]:
 with safe_open(str(p/idx[key]),framework='pt') as h:f[short]=h.get_tensor(key).view(torch.uint8)
# Preserve original live BF16 input; other previous bank entries reconstructed exactly.
f['inputs']=[f['live'].reshape(-1).bfloat16()]+[(x.view(torch.float8_e4m3fn).float()*torch.pow(2.,s.float()-127).repeat_interleave(128)).bfloat16() for x,s in f['bank'][1:]]
a=1024*10000.**(-torch.arange(32,dtype=torch.float32)/32)
f['cos']=a.cos();f['sin']=a.sin()
f['rotary_provenance']='Controlled position1024 base10000 interleaved rotary fixture; not a live YaRN table.'
torch.save(f,root/'fixture.pt')
print('fixture saved',flush=True)
