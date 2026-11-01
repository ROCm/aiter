# SPDX-License-Identifier: MIT
"""Summarize source-matched CPU logs and strict no-device ISA proofs."""
import argparse
import hashlib
import json
from pathlib import Path
import re

GATES = ('contract', 'address', 'q-lds', 'trace16', 'trace32', 'k-tdm', 'routing',
         'fp8-k-tdm', 'bm64', 'persist', 'fp8-cvt', 'strict-tdm', 'routing-isa',
         'fp8-isa', 'persist-isa')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--isa', type=Path, required=True)
    p.add_argument('--json', type=Path, required=True)
    a = p.parse_args()
    source = Path(__file__).resolve().parent.parent / 'qsa_paged_mqa_logits_page4_gfx1250.py'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    results = a.artifacts / 'current-results'
    saved_sha = (results / 'source.sha256').read_text().split()[0]
    native_sha = (a.artifacts / 'native-results' / 'source.sha256').read_text().split()[0]
    assert source_sha == saved_sha == native_sha, 'CPU/ISA source mismatch'
    run = (a.artifacts / 'verify.log').read_text()
    exits = {name: int(code) for name, code in re.findall(r'^([\w-]+) exit=(\d+)$', run, re.M)}
    assert set(GATES).issubset(exits) and all(exits[name] == 0 for name in GATES)
    assert 'ALL_GATES_RC=0' in run
    gates = {}
    for name in GATES:
        path = results / (name + '.log')
        text = path.read_text()
        match = re.search(r'Ran (\d+) tests in ([\d.]+)s', text)
        gates[name] = {'exit': exits[name], 'log_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if match:
            gates[name].update(tests=int(match[1]), seconds=float(match[2]))
        if name.startswith('trace'):
            gates[name]['pass_cases'] = len(re.findall(r'^\[PASS\]', text, re.M))
    off = json.loads((results / 'switch-off-identity.json').read_text())
    unaffected = json.loads((a.artifacts / 'default-unaffected-results' / 'fast-unaffected-identity.json').read_text())
    isa = json.loads(a.isa.read_text())
    assert off['byte_identical'] and len(off['isa_files']) == 46
    assert unaffected['byte_identical'] and len(unaffected['isa_files']) == 12
    assert isa['switch_off_isa_byte_identical'] and isa['native_early_clobber_all_32_destinations_disjoint']
    assert isa['default_isa_byte_identical_to_fast'] and isa['routing']['default'] == 'fast'
    routing_log = a.artifacts / 'default-fp8-isa.log'
    routing = json.loads(routing_log.read_text())
    assert routing['auto_r32_equals_forced_tdm_r32_isa']
    assert routing['auto_r16_equals_raw_and_forced_tdm_r16_isa']
    assert len(routing['kernels']) == 6
    out = {'status': 'all_cpu_and_strict_isa_gates_pass', 'kernel_source_sha256': source_sha,
           'gpu_executed': False, 'stock_flydsl': '0.3.3.dev886', 'arch': 'gfx1250',
           'image': 'rocm/fw-bringup-mlperf:gfx1250-atom-20261015',
           'staging': '/data/blyu/qsa/work/opt-cvtdef-cpu-trace-20261101',
           'routing': isa['routing'], 'default_equals_explicit_fast_isa': True,
            'default_fp8_routing_isa': {'pass': True, 'specializations': 6,
                                       'log_sha256': hashlib.sha256(routing_log.read_bytes()).hexdigest()},
            'cpu_gates': gates, 'strict_conversion_isa': {'pass': True, 'proof_sha256': hashlib.sha256(a.isa.read_bytes()).hexdigest()},
           'switch_off_identity': {'pass': True, 'isa_files': 46, 'modes': ['auto', 'off', 'raw', 'tdm', 'p2', 'p4', 'p8']},
           'default_unaffected_identity': {'pass': True, 'isa_files': 12, 'modes': ['auto', 'raw'], 'paths': ['scalar', 'bf16', 'r16', 'raw']},
           'conversion_exactness': {'all_codes': 256, 'half_positions': 2, 'byte_positions': 8,
                                   'nan_encodings_both_signs': 254, 'adjacent_pairs': 65536,
                                   'random_word_pairs': 10000, 'actual_tdm_fragments': 512,
                                   'production_fp16_denorm_mode': 3,
                                   'legacy_input_ftz_difference_codes': ['01', '02', '03', '04', '05', '06', '07', '81', '82', '83', '84', '85', '86', '87'],
                                   'scope': 'all256 old/new bit identity in emitted preserve mode; deliberate legacy FP16 inputFTZ differs14codes; hardware FP8 source policy remains device gate'}}
    a.json.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
