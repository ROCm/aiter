# SPDX-License-Identifier: MIT
"""ROUND-PAD2 offline final-ISA identity + gate manifest; no torch/FlyDSL import."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

p = argparse.ArgumentParser()
p.add_argument('--stage', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--final-phase', default='final3')
p.add_argument('--cpu-phase', default='cpu3')
p.add_argument('--compact-output', type=Path)
a = p.parse_args()
root = a.stage
base = root/'base/results/unset'
final = root/a.final_phase/'results'
paths = sorted(base.glob('*/dispatch.json'))
assert len(paths) == 12, len(paths)
records = []

def real_isa(rec, result_root):
    # Compiler container /results is the phase's result root.
    return result_root / Path(rec['isa']['path']).relative_to('/results')

for path in paths:
    before = json.loads(path.read_text())
    case = before['case']
    old_isa = real_isa(before, root/'base/results').read_bytes()
    for pad in ('unset','0','1'):
        current = json.loads((final/pad/case/'dispatch.json').read_text())
        new_isa = real_isa(current, final).read_bytes()
        exact = '-exact-' in case
        required = pad != '1' or exact
        identical = new_isa == old_isa
        if required:
            assert identical, ('ISA changed', pad, case)
            for key in ('kernel_name','grid','launch_dimensions','tensor_shapes'):
                assert current[key] == before[key], (key,pad,case,current[key],before[key])
            assert all(current['pointer_reuse'].values()), (pad,case,current['pointer_reuse'])
        records.append({'case':case,'pad_env':pad,'exact':exact,'required_identity':required,
            'isa_byte_identical_to_base':identical,'sha256':hashlib.sha256(new_isa).hexdigest(),
            'kernel_name':current['kernel_name'],'grid':current['grid'],
            'launch_dimensions':current['launch_dimensions'],'resources':current['resources'],
            'pointer_reuse':current['pointer_reuse'],'isa':str(real_isa(current,final))})

native_root = root/'native/results'
native_paths = sorted((native_root/'base').glob('*/dispatch.json'))
assert len(native_paths) == 20, len(native_paths)
native_records = []
for path in native_paths:
    before = json.loads(path.read_text())
    current = json.loads((native_root/'1'/before['case']/'dispatch.json').read_text())
    old_isa = real_isa(before,native_root).read_bytes()
    new_isa = real_isa(current,native_root).read_bytes()
    assert old_isa == new_isa, ('native ISA changed',before['case'])
    for key in ('kernel_name','grid','launch_dimensions','tensor_shapes'):
        assert current[key] == before[key], ('native identity',key,before['case'])
    assert before['launch_dimensions'] == before['true_dimensions'], before['case']
    assert current['launch_dimensions'] == current['true_dimensions'], current['case']
    for rec in (before,current):
        assert all(rec['pointer_reuse'].values()), (rec['case'],rec['pointer_reuse'])
        if rec['dtype'] == 'fp8':
            assert rec['pointer_reuse'].get('compressed_k_scale') is True, rec['case']
    native_records.append({'case':current['case'],'pad_env':'1','exact':True,
        'required_identity':True,'isa_byte_identical_to_base':True,
        'sha256':hashlib.sha256(new_isa).hexdigest(),'kernel_name':current['kernel_name'],
        'grid':current['grid'],'launch_dimensions':current['launch_dimensions'],
        'resources':current['resources'],'pointer_reuse':current['pointer_reuse'],
        'isa':str(real_isa(current,native_root))})

rel = Path('aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py')
old_source = (root/'base/source'/rel).read_text()
new_source = (root/a.final_phase/'source'/rel).read_text()
old_ast, new_ast = ast.parse(old_source), ast.parse(new_source)
old_functions = {n.name:n for n in old_ast.body if isinstance(n,ast.FunctionDef)}
new_functions = {n.name:n for n in new_ast.body if isinstance(n,ast.FunctionDef)}
builders = {}
for name in sorted(set(old_functions) - {'qsa_paged_mqa_logits_page4'}):
    old = ast.get_source_segment(old_source,old_functions[name])
    new = ast.get_source_segment(new_source,new_functions[name])
    assert old == new, ('kernel builder source changed',name)
    builders[name] = hashlib.sha256(new.encode()).hexdigest()

launch_branches = {}
old_wrapper = old_functions['qsa_paged_mqa_logits_page4']
new_wrapper = new_functions['qsa_paged_mqa_logits_page4']
for tag, condition in (('bf16', "not is_fp8 and _BF16_PATH == 'wmma'"),
                       ('fp8', "is_fp8 and _FP8_PATH == 'wmma'")):
    old_node = next(n for n in old_wrapper.body if isinstance(n,ast.If) and ast.unparse(n.test) == condition)
    new_node = next(n for n in new_wrapper.body if isinstance(n,ast.If) and ast.unparse(n.test) == condition)
    old = ast.get_source_segment(old_source,old_node)
    new = ast.get_source_segment(new_source,new_node)
    assert old == new, ('original launch branch changed',tag)
    launch_branches[tag] = hashlib.sha256(new.encode()).hexdigest()
def scalar_tail(source, wrapper):
    start = next(i for i,n in enumerate(wrapper.body) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id == 'threads' for t in n.targets))
    return ''.join(source.splitlines(keepends=True)[wrapper.body[start].lineno-1:wrapper.body[-1].end_lineno])
old_scalar, new_scalar = scalar_tail(old_source,old_wrapper), scalar_tail(new_source,new_wrapper)
assert old_scalar == new_scalar, 'original scalar launch tail changed'
launch_branches['scalar'] = hashlib.sha256(new_scalar.encode()).hexdigest()

# Reuse existing strict no-device ISA checks on actual selected PAD=1 kernels.
sys.path.insert(0,str(root/a.final_phase/'source/aiter/ops/flydsl/kernels/mqa_logits/_g2'))
from check_tdm_isa_cpu import check as check_bf16
from check_fp8_tdm_isa_cpu import check as check_fp8
from summarize_persist_cpu import check_persist
strict = []
for item in records + native_records:
    if item['pad_env'] != '1':
        continue
    isa = Path(item['isa'])
    text = isa.read_text()
    assert not re.search(r'waterfall|\bv_readfirstlane_b32\b',text,re.I), item['case']
    name = item['kernel_name']
    if name.startswith('qsa_paged_mqa_logits_page4_gfx1250_fp8'):
        gate = check_fp8(isa, '_tdm' in name)
    elif '_persist' in name:
        gate = check_persist(isa,int(name.rsplit('persist',1)[1]))
    elif '_tdm' in name:
        br,bc = map(int,re.search(r'_r(\d+)_c(\d+)_',name).groups())
        gate = check_bf16(isa,br,bc)
    else:
        assert item['resources']['group_segment_fixed_size'] == 0
        gate = {'status':'PASS','lds_bytes':0}
    strict.append({'case':item['case'],'status':gate['status'],'lds_bytes':gate['lds_bytes']})

cpu = root/a.cpu_phase/'results'
status = (cpu/'gate-status.txt').read_text().splitlines()
gates = []
for line in status:
    match = re.fullmatch(r'(.+) exit=(\d+)',line)
    assert match, line
    name,code = match[1],int(match[2])
    log = (cpu/(name+'.log')).read_text()
    count = re.search(r'Ran (\d+) tests?',log)
    gates.append({'gate':name,'exit_code':code,'unittest_count':int(count[1]) if count else None,
        'pass_lines':len(re.findall(r'^\s*\[PASS\]',log,re.M)),
        'log':str(cpu/(name+'.log')),'log_sha256':hashlib.sha256(log.encode()).hexdigest()})
assert len(gates) == 13, gates
assert all(g['exit_code'] == 0 for g in gates), gates
golden_manifest = (cpu/'golden.sha256').read_text()
image_id = json.loads((cpu/'image-inspect.json').read_text())[0]['Id']
for phase in ('base',a.final_phase,a.cpu_phase,'native'):
    result_root = root/phase/'results'
    assert (result_root/'golden.sha256').read_text() == golden_manifest, ('frozen golden hash mismatch',phase)
    assert json.loads((result_root/'image-inspect.json').read_text())[0]['Id'] == image_id, ('image mismatch',phase)
    versions = json.loads((result_root/'versions.json').read_text())
    assert versions['no_kfd'] and versions['no_dri'] and versions['COMPILE_ONLY'] == '1', phase
out = {'status':'PASS','final_phase':a.final_phase,'cpu_phase':a.cpu_phase,'base_commit':'f7c91d5dedf206e41ec13f0daa3fa3e1c82029a8',
 'source_before_sha256':hashlib.sha256((root/'base/source'/rel).read_bytes()).hexdigest(),
 'source_after_sha256':hashlib.sha256((root/a.final_phase/'source'/rel).read_bytes()).hexdigest(),
 'base_dispatch_compiles':12,'final_dispatch_compiles':36,
 'native_benchmark_dispatch_compiles':40,'verified_total_dispatch_compiles':88,
 'native_benchmark_identity_cases':20,'native_benchmark_isa':native_records,
 'total_required_isa_identity_comparisons':52,
 'required_isa_identity_comparisons':sum(r['required_identity'] for r in records),
 'exact_bucket_pad_on_identity_cases':8,'strict_selected_isa_checks':strict,
 'unchanged_original_functions_sha256':builders,'unchanged_original_function_count':len(builders),
 'unchanged_original_launch_branches_sha256':launch_branches,
 'versions':json.loads((cpu/'versions.json').read_text()),
 'test_pad_bucket_sha256':hashlib.sha256((root/a.final_phase/'source/aiter/ops/flydsl/kernels/mqa_logits/_g2/test_pad_bucket_cpu.py').read_bytes()).hexdigest(),
 'test_pad_bucket_trace_sha256':hashlib.sha256((root/a.final_phase/'source/aiter/ops/flydsl/kernels/mqa_logits/_g2/test_pad_bucket_trace_cpu.py').read_bytes()).hexdigest(),
 'pad_bucket_document_sha256':hashlib.sha256((root/a.final_phase/'source/aiter/ops/flydsl/kernels/mqa_logits/_g2/PAD_BUCKET.md').read_bytes()).hexdigest(),
 'image_id':json.loads((cpu/'image-inspect.json').read_text())[0]['Id'],
 'verification_scripts_sha256':{name:hashlib.sha256((root/'tools'/name).read_bytes()).hexdigest() for name in
    ('compile_pad2_dispatch_cpu.py','host_pad2_cpu_isa.sh','container_pad2_cpu_isa.sh','verify_pad2_dispatch_isa_cpu.py')},
 'dispatch_isa':records,'cpu_gates':gates,
 'unittest_total':sum(g['unittest_count'] or 0 for g in gates),
 'golden_manifest':(cpu/'golden.sha256').read_text(),
 'scope':'CPU-only no-device compiler and real production wrapper identity vs ROUND-PAD base. Frozen public golden is correctness reference, not optimized-kernel ISA. No GPU runtime, shader dispatch, GPU arithmetic or performance claim.'}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(out,indent=2)+'\n')
if a.compact_output:
    compact = {key:value for key,value in out.items() if key not in
               ('dispatch_isa','native_benchmark_isa','strict_selected_isa_checks')}
    compact['complete_evidence_path'] = str(a.output)
    compact['complete_evidence_sha256'] = hashlib.sha256(a.output.read_bytes()).hexdigest()
    compact['strict_selected_isa_checks'] = len(strict)
    compact['all_strict_selected_isa_checks_passed'] = all(item['status'] == 'PASS' for item in strict)
    unique = {}
    for item in records + native_records:
        unique[item['kernel_name']] = {'sha256':item['sha256'],'resources':item['resources']}
    compact['isa_kernels'] = unique
    compact['representative_identity_cases'] = [
        {'case':item['case'],'pad_env':item['pad_env'],'sha256':item['sha256']}
        for item in records if item['required_identity']]
    compact['native_benchmark_identity'] = [
        {key:item[key] for key in ('case','kernel_name','sha256','grid','launch_dimensions')}
        | {'all_input_output_pointers_reused':all(item['pointer_reuse'].values()),
           'fp8_scale_pointer_reused':item['pointer_reuse'].get('compressed_k_scale')}
        for item in native_records]
    a.compact_output.parent.mkdir(parents=True,exist_ok=True)
    a.compact_output.write_text(json.dumps(compact,indent=2)+'\n')
print('PAD2_PASS compiles=88 required_isa_identity=52 representative_exact_on=8 benchmark_native_on=20 strict_isa=32 cpu_gates=13 unittest_total='+str(out['unittest_total']))
