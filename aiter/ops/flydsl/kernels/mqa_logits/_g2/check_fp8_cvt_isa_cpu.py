# SPDX-License-Identifier: MIT
"""Strict CPU-only real-ISA conversion evidence (no FlyDSL/device import)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def summary(path):
    text = path.read_text()
    # This label is the common end of the full-r32 uniform-reuse branch; its
    # incoming vcc branch selects the champion raw mixed/tail fallback.
    match = re.search(r's_cbranch_vccnz (\.LBB\d+_\d+)', text)
    assert match, 'full-r32 guard branch missing'
    label = match[1]
    full = text[match.end():text.index(label + ':', match.end())]
    def instructions(s):
        return Counter(re.findall(r'^\s+([vs]_[a-zA-Z0-9_]+)\s', s, re.M))
    counts = instructions(full)
    native_ranges = re.findall(r'v_cvt_scale_pk8_bf16_fp8 v\[(\d+):(\d+)\], v\[(\d+):(\d+)\]', full)
    for dlo, dhi, slo, shi in native_ranges:
        destination = set(range(int(dlo), int(dhi) + 1))
        source = set(range(int(slo), int(shi) + 1))
        assert len(destination) == 4 and len(source) == 2
        assert destination.isdisjoint(source), 'pk8 early-clobber source/destination overlap'
    def n(prefix):
        return sum(v for k, v in counts.items() if k.startswith(prefix))
    result = {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'dispatch': re.search(r'^\s*\.globl\s+(\S+)', text, re.M)[1],
        'resources': {key: int(re.search(r'\.amdhsa_' + key + r'\s+(\d+)', text)[1])
                      for key in ('next_free_vgpr', 'next_free_sgpr', 'group_segment_fixed_size',
                                  'private_segment_fixed_size', 'float_denorm_mode_16_64')},
        'waterfall_markers': len(re.findall('waterfall|readfirstlane', text, re.I)),
        'full_chunk_per_wave': {
            'native_pk8_bf16_fp8': n('v_cvt_scale_pk8_bf16_fp8'),
            'legacy_pk_mul_f16': n('v_pk_mul_f16'),
            'legacy_cvt_f32_f16': n('v_cvt_f32_f16'),
            'native_pk_min_u16': n('v_pk_min_u16'),
            'wmma': n('v_wmma_'), 'lds_b128_loads': n('ds_load_b128'),
            'opcodes': dict(sorted(counts.items())),
        },
        'whole_kernel': dict(sorted(instructions(text).items())),
    }
    # ds opcodes do not start v/s, count separately.
    result['full_chunk_per_wave']['lds_b128_loads'] = len(re.findall(r'^\s+ds_load_b128\s', full, re.M))
    result['whole_kernel']['ds_load_b128'] = len(re.findall(r'^\s+ds_load_b128\s', text, re.M))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--champion', type=Path, required=True)
    p.add_argument('--current', type=Path, required=True)
    p.add_argument('--fast', type=Path, required=True)
    p.add_argument('--default', type=Path, required=True)
    p.add_argument('--json', type=Path, required=True)
    p.add_argument('--micro-current', type=Path)
    p.add_argument('--micro-fast', type=Path)
    a = p.parse_args()
    assert a.champion.read_bytes() == a.current.read_bytes(), 'switch-off ISA not byte-identical'
    assert a.default.read_bytes() == a.fast.read_bytes(), 'default ISA differs from explicit fast'
    before, after = summary(a.current), summary(a.fast)
    b, f = before['full_chunk_per_wave'], after['full_chunk_per_wave']
    assert b['legacy_pk_mul_f16'] == 128 and b['legacy_cvt_f32_f16'] == 256, b
    assert f['native_pk8_bf16_fp8'] == 32 and f['native_pk_min_u16'] == 128, f
    assert f['legacy_pk_mul_f16'] == f['legacy_cvt_f32_f16'] == 0
    assert b['wmma'] == f['wmma'] == 32
    assert b['lds_b128_loads'] == f['lds_b128_loads'] == 32
    assert before['waterfall_markers'] == after['waterfall_markers'] == 0
    for item in (before, after):
        assert item['resources']['float_denorm_mode_16_64'] == 3
        assert item['resources']['group_segment_fixed_size'] == 9216
        assert item['resources']['private_segment_fixed_size'] == 0
    out = {'switch_off_isa_byte_identical': True, 'before': before, 'after': after,
           'scope': 'full r32 single-request 64-column chunk per wave; 4 N tiles x 4 K32 steps, 8192 lane-code conversions; raw fallback retained',
           'conversion_core_per_8_codes': {'before': {'pk_mul_f16': 4, 'cvt_f32_f16': 8},
                                           'after': {'input_sign_mask': 2, 'native_pk8': 1, 'packed_min': 4, 'sign_perm': 4, 'sign_and_or': 4}}}
    out['default_isa_byte_identical_to_fast'] = True
    out['default_sha256'] = after['sha256']
    out['routing'] = {'default': 'fast', 'current': 'legacy', 'fast': 'fast'}
    out['native_early_clobber_all_32_destinations_disjoint'] = True
    if a.micro_current or a.micro_fast:
        assert a.micro_current and a.micro_fast
        micro = {}
        for tag, path in (('before', a.micro_current), ('after', a.micro_fast)):
            text = path.read_text()
            body = text[text.index('buffer_load_b64'):text.index('buffer_store_b128')]
            lines = re.findall(r'^\s+(v_[^\n]+)', body, re.M)
            micro[tag] = {
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'conversion_valu_issued_instructions_per_8_codes': len(lines),
                'conversion_valu_scalar_operations_per_8_codes': sum(2 if ' :: ' in line else 1 for line in lines),
                'native_ops': len(re.findall(r'v_cvt_scale_pk8_bf16_fp8', body)),
                'legacy_widen_ops': len(re.findall(r'v_cvt_f32_f16', body)),
                'legacy_packed_mul_ops': len(re.findall(r'v_pk_mul_f16', body)),
                'scope': 'actual helper only: excludes load/store, address generation, scalar setup and delays; VOPD counted as one issued instruction/two operations',
            }
        assert micro['before']['conversion_valu_issued_instructions_per_8_codes'] == 32
        assert micro['before']['conversion_valu_scalar_operations_per_8_codes'] == 38
        assert micro['after']['conversion_valu_issued_instructions_per_8_codes'] == 15
        assert micro['after']['conversion_valu_scalar_operations_per_8_codes'] == 15
        out['helper_micro_isa'] = micro
        out['complete_conversion_scalar_operations_per_chunk_per_wave'] = {'before': 1216, 'after': 480}
    a.json.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('before', 'after')}, indent=2))
    print('resources before', before['resources'], 'after', after['resources'])


if __name__ == '__main__':
    main()
