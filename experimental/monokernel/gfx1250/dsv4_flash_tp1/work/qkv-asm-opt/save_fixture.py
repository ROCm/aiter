from common import *
fixture=dict(W_bytes=W.view(torch.uint8).cpu(),WS=WS.cpu(),NW=NW.cpu(),bank=[(x.view(torch.uint8).cpu(),s.cpu()) for x,s in bank],live=live.cpu(),provenance=dict(model='/data/DeepSeek-V4-Flash',layer=0,projection_input='live-input.pt',weight_names=['layers.0.attn.wq_a.weight','layers.0.attn.wq_a.scale','layers.0.attn.wkv.weight','layers.0.attn.wkv.scale','layers.0.attn.q_norm.weight','layers.0.attn.kv_norm.weight']))
torch.save(fixture,root/'fixture.pt');print('Saved',root/'fixture.pt')
