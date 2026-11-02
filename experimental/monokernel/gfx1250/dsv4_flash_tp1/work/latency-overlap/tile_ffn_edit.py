from pathlib import Path
r=Path('/home/vigokhal/dsv4-monokernel/integrated-c8')
p=r/'optimized_moe.py';s=p.read_text();s='from edge_sync import publish,consume\n'+s
s=s.replace('BUFFERS:gl.constexpr=4):','BUFFERS:gl.constexpr=4,Tiles=None):')
s=s.replace('    ok=phase(Count,Health,GRID)', '''        if Tiles is not None:publish(Tiles+(256+task//16)*64,True)
    if Tiles is None:ok=phase(Count,Health,GRID)
    else:ok=True''')
s=s.replace('        for task in range(pid,gl.load(Pad)*16,GRID):\n            stage(Y', '        for task in range(pid,gl.load(Pad)*16,GRID):\n            if Tiles is not None:consume(Tiles+(256+task//16)*64,Health,16,True,1)\n            stage(Y')
s=s.replace('    # Every CTA enters phase2', '''            if Tiles is not None:
                expert=gl.load(Map+task//16)&65535
                start=gl.load(Offs+expert);nr=gl.load(Hist+expert)
                for rr in range(nr):publish(Tiles+(320+(start+rr)*16+task%16)*64,True)
    # Every CTA enters phase2''')
s=s.replace('    ok2=phase(Count+1,Health,GRID)','    if Tiles is None:ok2=phase(Count+1,Health,GRID)\n    else:ok2=True')
s=s.replace('                vals=gl.amd.gfx1250.buffer_load', '                if Tiles is not None:consume(Tiles+(320+row*16+tile//2)*64,Health,1,True,1)\n                vals=gl.amd.gfx1250.buffer_load')
p.write_text(s)
p=r/'ffn_batch.py';s=p.read_text().replace('Epoch=0):','Epoch=0,Tiles=None,TILE_MASK:gl.constexpr=0):')
s=s.replace('  publish(Edges+15*320,p<64)', '  if Tiles is not None and TILE_MASK&8:publish(Tiles+(200+(p%32)//2)*64,p<64)\n  else:publish(Edges+15*320,p<64)')
s=s.replace('  consume(Edges+15*320,Health,64,p<M*16,Epoch=Epoch)', '  if Tiles is not None and TILE_MASK&8:consume(Tiles+(200+p%16)*64,Health,4,p<M*16,1)\n  else:consume(Edges+15*320,Health,64,p<M*16,Epoch=Epoch)')
s=s.replace('  shared_body(AQ,AS,SW2,SS2,SO,2048,32,M)\n  consume', '  shared_body(AQ,AS,SW2,SS2,SO,2048,32,M)\n  if Tiles is not None and TILE_MASK&8:publish(Tiles+(216+p//4)*64,p<128)\n  consume')
s=s.replace('W1S,S1S,W2S,S2S,10.,M,256,3)', 'W1S,S1S,W2S,S2S,10.,M,256,3,Tiles if TILE_MASK&4 else None)')
s=s.replace(' sync(Counter,Health,256,3)\n if gl.program_id(0)<M*32:', ' if Tiles is None or not TILE_MASK&8:sync(Counter,Health,256,3)\n if gl.program_id(0)<M*32:\n  if Tiles is not None and TILE_MASK&8:consume(Tiles+(216+gl.program_id(0)%32)*64,Health,4,True,1)')
s=s.replace(' sync(Counter,Health,256,3)\n post_body', ' if Tiles is None or not TILE_MASK&8:sync(Counter,Health,256,3)\n post_body')
# shared-expert tile handoffs need the producer-only FFN schedule.
s=s.replace(' if DO_PRE:\n', ' if Tiles is not None and TILE_MASK&8:gl.static_assert(Edges is not None)\n if DO_PRE:\n',1)
p.write_text(s)
p=r/'model_layer_batch.py';s=p.read_text().replace('OVERLAP=OVERLAP,Epoch=Epoch)\n sync(F[-2]', 'OVERLAP=OVERLAP,Epoch=Epoch,Tiles=Tiles,TILE_MASK=TILE_MASK)\n sync(F[-2]');p.write_text(s)
