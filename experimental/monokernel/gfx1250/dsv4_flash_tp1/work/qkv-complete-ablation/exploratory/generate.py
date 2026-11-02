from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'base/fused.py').read_text()
s=s.replace('def project_fused(X,XS,W,WS,P,NW,Acc,Count,Y,MODE:', 'def project_fused(X,XS,W,WS,P,NW,Acc,Count,Y,LP,LS,Cos,Sin,KVOut,INLINE:gl.constexpr,TAIL:gl.constexpr,MODE:')
a='            if REPLICATE:av='
start=s.index(a);end=s.index('            if PACKED:',start)
s=s[:start]+'''            if INLINE:
                iq=gl.arange(0,128,layout=gl.BlockedLayout([1],[32],[WAVES],[0]))
                xv=dev_load(X,block*128+iq,cache=CACHE).to(gl.float32)
                e,scale=quant_scale(xv)
                qv=(xv/scale).to(gl.float8e4nv)
                qv=gl.convert_layout(qv,gl.SliceLayout(0,AL))
                av=(qv.to(gl.uint8,bitcast=True)[None,:]+gl.full((16,128),0,gl.uint8,AL)).to(gl.float8e4nv,bitcast=True)
            elif REPLICATE:av=dev_load(X,block*BK+ka[None,:]+rm[:,None]*0,cache=CACHE)
            else:av=dev_load(X,block*BK+ka[None,:]+rm[:,None]*0,rm[:,None]==0,0.,cache=CACHE)
'''+s[end:]
s=s.replace('ax=gl.load(XS+block)+gl.full((16,BK),0,gl.uint8,AL)','ax=(e if INLINE else gl.load(XS+block))+gl.full((16,BK),0,gl.uint8,AL)')
s=s.replace('finish_fused(P,Acc,NW,Count,Y,','finish_fused(P,Acc,NW,Count,Y,LP,LS,Cos,Sin,KVOut,TAIL,')
s=s.replace('def finish_fused(P,Acc,NW,Count,Y,LP,LS,Cos,Sin,KVOut,TAIL,','def finish_fused(P,Acc,NW,Count,Y,LP,LS,Cos,Sin,KVOut,TAIL:gl.constexpr,')
s=s.replace('    dev_store(y,Y,BASE+i)','''    dev_store(y,Y,BASE+i)
    if TAIL:
        if BASE==0:
            QL:gl.constexpr=gl.BlockedLayout([1,4],[1,32],[4,1],[0,1])
            yy=gl.convert_layout(y,gl.SliceLayout(1,gl.BlockedLayout([1,1],[32,1],[4,1],[0,1]))) if False else y
            yy=gl.convert_layout(gl.reshape(yy,(8,128)),QL)
            amax=gl.maximum(gl.max(gl.abs(yy),1),1.e-10)/448.
            e=gl.minimum(((amax.to(gl.uint32,bitcast=True)&2147483647)+8388607)>>23,254)
            scale=(e<<23).to(gl.float32,bitcast=True)
            q=(yy/scale[:,None]).to(gl.float8e4nv)
            rr=gl.arange(0,8,layout=gl.SliceLayout(1,QL))
            cc=gl.arange(0,128,layout=gl.SliceLayout(0,QL))
            dev_store(q,LP,rr[:,None]*128+cc[None,:])
            dev_store(e.to(gl.uint8),LS,rr)
        else:
            other=gl.gather(y,i^1,0)
            c=dev_load(Cos,(i-448)//2,i>=448,1.,cache='.ca')
            sn=dev_load(Sin,(i-448)//2,i>=448,0.,cache='.ca')
            dev_store(y*c+gl.where(i%2==0,-1.,1.)*other*sn,KVOut,i)''')
s=s.replace('from device_scope_helpers import dev_load,dev_store','from device_scope_helpers import dev_load,dev_store\nfrom tail import quant_scale')
(p/'qa.py').write_text(s)
