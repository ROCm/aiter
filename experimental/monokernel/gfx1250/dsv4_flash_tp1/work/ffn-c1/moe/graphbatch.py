import ctypes,torch
hip=ctypes.CDLL('libamdhip64.so');P=ctypes.c_void_p;I=ctypes.c_int;SZ=ctypes.c_size_t
for name,args in {'hipGraphCreate':[ctypes.POINTER(P),ctypes.c_uint],'hipGraphAddChildGraphNode':[ctypes.POINTER(P),P,ctypes.POINTER(P),SZ,P],'hipGraphInstantiate':[ctypes.POINTER(P),P,ctypes.POINTER(P),ctypes.c_char_p,SZ],'hipGraphLaunch':[P,P]}.items():
 getattr(hip,name).argtypes=args;getattr(hip,name).restype=I
def ok(x):assert x==0,x
class GraphBatch:
 def __init__(self,fn,n=128):
  self.inner=torch.cuda.CUDAGraph(keep_graph=True)
  with torch.cuda.graph(self.inner):self.out=fn()
  self.g=P();ok(hip.hipGraphCreate(ctypes.byref(self.g),0));prev=P()
  for i in range(n):
   node=P();deps=(P*1)(prev) if i else None
   ok(hip.hipGraphAddChildGraphNode(ctypes.byref(node),self.g,deps,1 if i else 0,self.inner.raw_cuda_graph()));prev=node
  self.exec=P();ok(hip.hipGraphInstantiate(ctypes.byref(self.exec),self.g,None,None,0))
 def replay(self):ok(hip.hipGraphLaunch(self.exec,torch.cuda.current_stream().cuda_stream))
for name,args in {'hipGraphGetNodes':[P,ctypes.POINTER(P),ctypes.POINTER(SZ)],'hipGraphGetEdges':[P,ctypes.POINTER(P),ctypes.POINTER(P),ctypes.POINTER(SZ)],'hipGraphNodeGetType':[P,ctypes.POINTER(I)],'hipGraphKernelNodeGetParams':[P,P],'hipGraphAddKernelNode':[ctypes.POINTER(P),P,ctypes.POINTER(P),SZ,P]}.items():
 getattr(hip,name).argtypes=args;getattr(hip,name).restype=I
class FlatGraphBatch:
 def __init__(self,fn,n=128):
  self.inner=torch.cuda.CUDAGraph(keep_graph=True)
  with torch.cuda.graph(self.inner):self.out=fn()
  raw=self.inner.raw_cuda_graph();sz=SZ();ok(hip.hipGraphGetNodes(raw,None,ctypes.byref(sz)));nodes=(P*sz.value)();ok(hip.hipGraphGetNodes(raw,nodes,ctypes.byref(sz)))
  ns=list(nodes);ec=SZ();ok(hip.hipGraphGetEdges(raw,None,None,ctypes.byref(ec)));fr=(P*ec.value)();to=(P*ec.value)();ok(hip.hipGraphGetEdges(raw,fr,to,ctypes.byref(ec)))
  deps={a:set() for a in ns}
  for a,b in zip(fr,to):deps[b].add(a)
  order=[]
  while deps:
   ready=[a for a,v in deps.items() if not v];assert ready,'cycle'
   for a in ready:
    order.append(a);del deps[a]
    for v in deps.values():v.discard(a)
  self.params=[];self.node_types=[]
  for node in order:
   typ=I();ok(hip.hipGraphNodeGetType(node,ctypes.byref(typ)));self.node_types.append(typ.value)
   if typ.value==5:continue
   assert typ.value==0,('unsupportednode',typ.value)
   params=(ctypes.c_longdouble*256)();ok(hip.hipGraphKernelNodeGetParams(node,params));self.params.append(params)
  self.g=P();ok(hip.hipGraphCreate(ctypes.byref(self.g),0));prev=P()
  for _ in range(n):
   for params in self.params:
    node=P();d=(P*1)(prev) if prev.value else None
    ok(hip.hipGraphAddKernelNode(ctypes.byref(node),self.g,d,1 if prev.value else 0,params));prev=node
  self.exec=P();ok(hip.hipGraphInstantiate(ctypes.byref(self.exec),self.g,None,None,0));self.kernel_count=n*len(self.params)
 def replay(self):ok(hip.hipGraphLaunch(self.exec,torch.cuda.current_stream().cuda_stream))
