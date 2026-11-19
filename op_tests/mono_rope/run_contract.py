"""Hardware boundary probe for exact r2 A05/A07 bodies, with negative controls."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import struct
from aiter.ops.mono_standard.build_callable import build

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--fixture-id',required=True)
p.add_argument('--fixture-root',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--cache-root',type=Path,required=True)
p.add_argument('--capture-lib',type=Path,required=True)
p.add_argument('--suite',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--ranks',type=int,nargs='+',default=[0,1,2,3])
p.add_argument('--a07-cos-bf16',action='store_true')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
# Bind the probe evidence to the exact scripts and public candidates used.
import shutil
source_dir=Path(__file__).resolve().parent
source_hashes={}
sources = {'run_contract.py': Path(__file__), **{name: Path(__file__).resolve().parents[2]/'aiter/ops/mono_standard'/name for name in ('build_callable.py','codegen.py','loader.py')}}
for name,source in sources.items():
    source_hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    (a.output/'probe-sources').mkdir(exist_ok=True)
    shutil.copy2(source,a.output/'probe-sources'/name)
(a.output/'probe-sources.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
os.environ['MONO_EVIDENCE_DIR']=str(a.output/'artifacts')
from dsv41_single_layer.runtime import require_lock,configure
require_lock()
configure(a.source,a.cache_root)
import torch
from dsv41_single_layer.archive import TensorReader,verify,write_json
from dsv41_single_layer.capture import observed
from dsv41_single_layer.replay import compare
from triton.backends.amd.driver import HIPLauncher, _get_path_to_hip_runtime_dylib
from triton.backends.driver import get_kernel_argument_layout
captured_a05={}
original_init,original_call=HIPLauncher.__init__,HIPLauncher.__call__
def init(self,src,metadata):
    original_init(self,src,metadata)
    self._sentinel_signature=src.signature
    self._sentinel_name=metadata.name

def call(self,*args,**kwargs):
    result=original_call(self,*args,**kwargs)
    if self._sentinel_name.startswith('_rope_window_fp8_kernel'):
        if self.global_scratch_size or self.profile_scratch_size:
            raise ValueError('probe does not support scratch allocation')
        values=[]
        for path,ty in get_kernel_argument_layout(self._sentinel_signature.values()):
            value=args[9:]
            for index in path: value=value[index]
            if ty.startswith('*'):
                values.append(('Q',0 if value is None else value.data_ptr() if hasattr(value,'data_ptr') else int(value)))
            elif ty in ('i32','u32'):
                values.append(('I',int(value)&0xffffffff))
            elif ty in ('i64','u64'):
                values.append(('Q',int(value)&0xffffffffffffffff))
            else: raise ValueError(f'unsupported ABI argument {ty}')
        values += [('Q',0),('Q',0)]
        captured_a05.update(values=values,grid=list(args[:3]))
    return result
HIPLauncher.__init__,HIPLauncher.__call__=init,call
from op_tests.mono_rope import replay_candidate as mono_candidates
hip=C.CDLL(_get_path_to_hip_runtime_dylib())
hip.hipModuleLoad.argtypes=[C.POINTER(C.c_void_p),C.c_char_p]
hip.hipModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p]
hip.hipModuleLaunchKernel.argtypes=[C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.c_void_p,C.c_void_p]
bridge=C.CDLL(str(a.capture_lib))
bridge.sentinel_a07_args.argtypes=[C.c_void_p,C.c_void_p]

def checked(status):
    if status: raise RuntimeError(f'HIP error {status}')


def load(report):
    module,fun=C.c_void_p(),C.c_void_p()
    checked(hip.hipModuleLoad(C.byref(module),report['harness'].encode()))
    checked(hip.hipModuleGetFunction(C.byref(fun),module,b'mono_harness'))
    return module,fun


def pack(values,metadata):
    if len(values)!=len(metadata['.args']): raise ValueError('argument count mismatch')
    data=bytearray(120)
    for (fmt,value),spec in zip(values,metadata['.args']):
        if struct.calcsize(fmt)!=spec['.size']: raise ValueError('argument size mismatch')
        struct.pack_into('<'+fmt,data,spec['.offset'],value)
    return bytes(data)

root=a.fixture_root/a.fixture_id
verified=verify(root);rows=[];builds={};handles=[]

def epoch_for(gen):
    if not isinstance(gen,int) or gen<1 or gen*9+8>0xffffffff:raise ValueError('MONO epoch overflow or invalid generation')
    return gen*9
for invalid in [0,-1,((0xffffffff-8)//9)+1]:
    try:epoch_for(invalid)
    except ValueError:pass
    else:raise AssertionError('invalid generation accepted')
patterns=[0x2468ace0,0x55555555,0x10203040]
for rank in a.ranks:
    records=json.loads((root/f'rank{rank}/capture.json').read_text())['stages']
    for stage in (('A07',) if a.a07_cos_bf16 else ('A05','A07')):
        reader=TensorReader(root,device='cuda:0');record=records[stage]
        expected=reader.restore(record['outputs']);saved=reader.restore(record['inputs']);args,kwargs=saved['args'],dict(saved['kwargs'])
        if stage=='A05':initial_ring=kwargs['ring'][0].clone()
        else:
            template=reader.restore(record['outputs'])['return'];kwargs.update(x_fp8=template[0],x_scale=template[1])
        if a.a07_cos_bf16:
            from dsv41_single_layer.replay import native_callable
            args=list(args);args[2]=args[2].to(torch.bfloat16);args[3]=args[3].to(torch.bfloat16)
            native=native_callable(record['callable'])(*args,**kwargs)
            expected=observed(stage,args,kwargs,tuple(t.clone() for t in native))
        returned=mono_candidates.KERNELS[stage](*args,**kwargs);torch.cuda.synchronize()
        actual=observed(stage,args,kwargs,returned);initial_checks=compare(actual,expected)
        write_json(a.output/f'{stage}-r{rank}-initial-checks.json',initial_checks)
        assert initial_checks and all(c['passed'] for c in initial_checks)
        audit=json.loads(Path(str(a.suite)+'-audit-'+stage.lower()).joinpath('audit.json').read_text());item=audit['objects'][0]
        if stage=='A05':
            symbol=item['kernels'][0]['.name'];packed=pack(captured_a05['values'],item['kernels'][0]);assert captured_a05['grid']==[16,16,1]
        else:
            buffer,name=C.create_string_buffer(120),C.create_string_buffer(1024);assert bridge.sentinel_a07_args(buffer,name)==1
            packed,symbol=buffer.raw,name.value.decode();assert symbol in [k['.name'] for k in item['kernels']]
        regions=[args[0],*kwargs['query_fp8'],kwargs['ring'][0]] if stage=='A05' else list(returned)
        assert all(t.is_contiguous() and t.numel()*t.element_size()%4==0 for t in regions)
        shadows=[torch.empty_like(t) for t in regions]
        descriptors=b''.join(struct.pack('<QQIIII',t.data_ptr(),shadow.data_ptr(),t.numel()*t.element_size(),0,0,0) for t,shadow in zip(regions,shadows))
        region_table=torch.tensor(list(descriptors),device='cuda',dtype=torch.uint8)
        for corrupt in [False,True]:
            key=stage+'-'+symbol+'-'+str(corrupt)
            if key not in builds:
                builds[key]=build(item['binary'],symbol,a.output/(stage+('-negative' if corrupt else '-callable')),corrupt)
                builds[key]['handle']=load(builds[key]);handles.append(builds[key]['handle'])
            fun=builds[key]['handle'][1]
            for pattern_index,pattern in enumerate(patterns[:1] if corrupt else patterns):
                flags=torch.zeros(256,device='cuda',dtype=torch.int32)
                for gen in ([1] if corrupt else [1,2,3,(0xffffffff-8)//9]):
                    epoch=epoch_for(gen);monitor=torch.full((1024,8),-1,device='cuda',dtype=torch.int32)
                    if stage=='A05':kwargs['ring'][0].copy_(initial_ring)
                    for t in (kwargs['query_fp8'] if stage=='A05' else returned):t.view(torch.uint8).fill_(0x7f)
                    for t in shadows:t.view(torch.uint8).fill_(0x5a)
                    delay=256 if gen==2 else 0
                    data=C.create_string_buffer(packed+struct.pack('<QQIIQII',flags.data_ptr(),monitor.data_ptr(),epoch,pattern,region_table.data_ptr(),len(regions),delay))
                    size=C.c_size_t(160);extra=(C.c_void_p*5)(1,C.addressof(data),2,C.addressof(size),3)
                    def launch():checked(hip.hipModuleLaunchKernel(fun,16,16,1,128,1,1,8192 if stage=='A07' else 0,torch.cuda.current_stream().cuda_stream,None,extra))
                    mode='graph' if gen in [2,3] else 'eager'
                    if mode=='graph':
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):launch()
                        graph.replay()
                    else:launch()
                    torch.cuda.synchronize()
                    values=monitor.cpu().to(torch.int64).bitwise_and(0xffffffff)
                    wanted=torch.tensor([epoch^(1 if corrupt else 0),pattern^(2 if corrupt else 0),0 if corrupt else 0xffffffff,0xcafe,epoch+8,0,1,0],dtype=torch.int64)
                    monitor_ok=bool(torch.all(values==wanted));flags_ok=bool(torch.all(flags.to(torch.int64).bitwise_and(0xffffffff)==epoch+8))
                    checks=compare(actual,expected);copies=[torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(regions,shadows)]
                    passed=monitor_ok and flags_ok and all(c['passed'] for c in checks) and all(copies)
                    row=dict(rank=rank,kernel=stage,cos_bf16=a.a07_cos_bf16,pattern=pattern,gen=gen,epoch=epoch,mode=mode,delayed_TG=255 if delay else None,negative_control=corrupt,monitor_ok=monitor_ok,flags_ok=flags_ok,output_checks=checks,all_outputs_copied_with_DEV=copies,passed=passed)
                    rows.append(row);write_json(a.output/'result.json',dict(complete=False,runs=rows));print(json.dumps({k:v for k,v in row.items() if k!='output_checks'}),flush=True)
                    write_json(a.output/f'{stage}-r{rank}-p{pattern_index}-g{gen}-control{int(corrupt)}-monitor.json',values.tolist())
                    if not passed:raise RuntimeError('MONO callable/join/output contract failed')
write_json(a.output/'result.json',dict(complete=True,passed=True,physical_gpu=os.environ['GPU_LOCK_GPUS'],runs=rows,invalid_generations_rejected=[0,-1,((0xffffffff-8)//9)+1],builds={k:{n:v for n,v in b.items() if n!='handle'} for k,b in builds.items()},scope='Exact op callable + nine bounded all-256-TG joins + DEV copies of all outputs. Sequential generations reuse flags without resetting; near-u32 boundary and slow TG exercised. This is not the complete A01-A10 attention pipeline.'))
