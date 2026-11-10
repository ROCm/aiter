# SPDX-License-Identifier: MIT
"""Test the incremental cost of DST-only MSB changes after gfx1250 WMMA.

Run under gpu-lock; the script checks GPU idleness before initializing HIP:
  gpu-lock python3 op_tests/flydsl_tests/test_gfx1250_msb_latency.py --run

This is an explicit hardware performance test, not an ordinary pytest test.
Handwritten ISA preserves all source fields and bypasses compiler scheduling.
Pass --expert-scheduling to match FlyDSL's expert scheduling mode 2. It is
enabled after initialization drains and restored after the timed work drains.
Use --accumulator-bank 3 --variants f540 to test F5 -> 40 -> F5 with
the accumulator registers physically relocated to bank 3.
The assertion is equivalence within one shader-counter cycle per round, using
a paired 95% bootstrap interval. It is not a proof of universal zero latency.
"""

import argparse
import ctypes
import functools
import hashlib
import json
import pathlib
import random
import re
import statistics
import struct
import subprocess
import sys
import unittest


TOOLS = pathlib.Path('/opt/rocm/llvm/bin')
UNROLL = 32
WMMA_MODE = 0xA5  # SRC0=1, SRC1=1, SRC2=2, DST=2.
MODES = {
    'batched_no_msb': WMMA_MODE,
    'ds_only': WMMA_MODE,
    'wmma_only': WMMA_MODE,
    'no_msb': WMMA_MODE,
    'nop_control': WMMA_MODE,
    'same_msb': WMMA_MODE,
    'delay_control': 0x25,
    'dst_only': 0x25,
    'dst_only_bank1': 0x65,
    'src0_only': 0xA4,
    'src1_only': 0xA1,
    'src2_only': 0x85,
    'src0_dst': 0x24,
    'src1_dst': 0x21,
    'src2_dst': 0x05,
    'src12_dst': 0x01,
    'src012_dst': 0x00,
    'f540': 0x40,
}


def bank_modes(variant: str, accumulator_bank: int) -> tuple[int, int]:
    """Return WMMA/DS states while relocating both accumulator operands."""
    if accumulator_bank not in (2, 3):
        raise ValueError('Accumulator bank must be 2 or 3')
    if variant == 'f540' and accumulator_bank != 3:
        raise ValueError('The f540 case requires accumulator bank 3')
    wmma_mode = 0x05 | (accumulator_bank << 4) | (accumulator_bank << 6)
    mode = MODES[variant]
    # Fields unchanged by the original case follow the relocated accumulator.
    for shift in (4, 6):
        if (mode >> shift) & 3 == 2:
            mode = (mode & ~(3 << shift)) | (accumulator_bank << shift)
    return wmma_mode, mode


def set_msb(old: int, new: int) -> str:
    """Encode old and new region fields explicitly."""
    return f'    s_set_vgpr_msb 0x{old:02x}{new:02x}'


def kernel_source(name: str, variant: str, group: int, gap: int,
                  ds_reads: int = 1, ds_bits: int = 128,
                  unroll: int | None = None, block_wait: bool = False,
                  middle_op: str = 'ds', lane_stride: int = 16,
                  ds_base_vgpr: int = 96, wmma_a_vgpr: int = 0,
                  expert_scheduling: bool = False,
                  accumulator_bank: int = 2) -> str:
    """Repeat WMMA/MSB/DS/MSB with independent accumulators within a group."""
    if group not in (1, 2, 4) or gap != 0 or ds_reads not in (1, 2, 3, 4, 5):
        raise ValueError('Use 1/2/4 WMMAs, 1..5 DS reads, and no gap')
    if ds_bits not in (32, 128):
        raise ValueError('DS reads must be 32 or 128 bits')
    if middle_op not in ('ds', 'vadd') or (middle_op == 'vadd' and ds_reads != 1):
        raise ValueError('VADD replacement requires one middle instruction')
    if lane_stride not in (4, 16) or (lane_stride == 4 and ds_bits != 32):
        raise ValueError('Use a 4-byte stride for b32 or a 16-byte stride')
    if unroll is None:
        unroll = UNROLL // ds_reads
    if not 1 <= unroll <= UNROLL // ds_reads:
        raise ValueError('Unrolled block must fit 32 distinct DS sinks')
    wmma_mode, mode = bank_modes(variant, accumulator_bank)
    accumulator_init = accumulator_bank << 6
    if ds_base_vgpr not in (0, 96) or wmma_a_vgpr not in (0, 32):
        raise ValueError('Unsupported DS address or WMMA A register')
    if ds_base_vgpr == 0 and (wmma_a_vgpr != 32 or mode & 3 != 1
                             or middle_op != 'ds'):
        raise ValueError('DS v0 requires A at v32, SRC0 bank 1, and DS mode')
    enter = ('    s_nop 0' if variant == 'nop_control' else
             set_msb(wmma_mode, mode))
    if variant == 'delay_control':
        enter = '    s_nop 63\n' + enter
    leave = ('    s_nop 0' if variant == 'nop_control' else
             set_msb(mode, wmma_mode))
    if variant in ('no_msb', 'wmma_only', 'ds_only', 'batched_no_msb'):
        enter = leave = ''
    # Duplicate the LDS address so SRC0 changes read identical data.
    initialize = [
        f'    v_lshlrev_b32 v96, {lane_stride.bit_length()-1}, v0',
        '    v_mov_b32 v80, 0x7f7f7f7f',
        '    v_mov_b32 v81, 0x22222222',
        '    ds_store_b32 v96, v81',
    ]
    if lane_stride == 16:
        initialize += [f'    ds_store_b32 v96, v81 offset:{offset}'
                       for offset in (4, 8, 12)]
    initialize += [set_msb(0, 0x40),
                   '    v_mov_b32 v96, v96']
    initialize += [f'    v_mov_b32 v{i}, 0x22222222'
                   for i in range(wmma_a_vgpr, wmma_a_vgpr+8)]
    if wmma_a_vgpr == 32:
        # Mode 0x40 copies the low-bank LDS address into physical v256.
        # Also initialize it in the matching v96 control for equal setup.
        initialize += ['    v_mov_b32 v0, v96']
    initialize += [f'    v_mov_b32 v{i}, 0x22222222'
                   for i in range(64, 80)]
    initialize += [set_msb(0x40, accumulator_init)]
    initialize += [f'    v_mov_b32 v{i}, 0' for i in range(group*16)]
    initialize += ['    s_wait_dscnt 0',
                   '    s_wait_alu depctr_va_vdst(0)']
    wmma = '\n'.join(
        f'    v_wmma_scale_f32_32x16x128_f4 v[{i}:{i+15}], '
        f'v[64:79], v[{wmma_a_vgpr}:{wmma_a_vgpr+7}], '
        f'v[{i}:{i+15}], v80, v80'
        for i in range(0, group*16, 16))
    if variant == 'ds_only':
        wmma = ''
    padding = '\n'.join('    s_nop 0' for _ in range(gap))
    # Four-register sink spacing is retained for both DS widths. Destinations
    # are distinct within a block and reused across blocks without consumers.
    loads = '\n'.join(
        f'    ds_load_b{ds_bits} v[.Lds_dst_{name}+{i*4}'
        + (f':.Lds_dst_{name}+{i*4+3}' if ds_bits == 128 else '')
        + f'], v{ds_base_vgpr}' for i in range(ds_reads))
    if middle_op == 'vadd':
        # Both source regions contain the same lane-dependent value.
        loads = f'    v_add_nc_u32_e32 v[.Lds_dst_{name}], v96, v96'
    if variant == 'wmma_only':
        loads = ''
    final_ds_sink = 128 + (unroll*ds_reads-1)*4
    ds_result = ('    v_mov_b32 v5, 0' if variant == 'wmma_only' else
                 f'    v_mov_b32 v5, v{final_ds_sink}')
    body = f"""    .set .Lds_dst_{name}, 128
    .rept {unroll}
{wmma}
{padding}
{enter}
{loads}
{leave}
    .set .Lds_dst_{name}, .Lds_dst_{name}+{4*ds_reads}
    .endr"""
    if variant == 'batched_no_msb':
        # Keep instruction counts while reducing WMMA/DS alternations.
        body = f"""    .rept {unroll}
{wmma}
    .endr
    .set .Lds_dst_{name}, 128
    .rept {unroll}
{loads}
    .set .Lds_dst_{name}, .Lds_dst_{name}+{4*ds_reads}
    .endr"""
    # Validate every accumulator separately, outside the timed region.
    extra_accumulators = '\n'.join(
        f'    v_mov_b32 v{5+i}, v{16*i}' for i in range(1, group))
    extra_stores = '\n'.join(
        f'    global_store_b32 v96, v{5+i}, s[4:5] offset:{512*i}'
        for i in range(1, group))
    # Reassert extended VGPR mode after writing SCHED_MODE: the disassembler
    # otherwise loses its physical-register annotations at the setreg. All
    # mode setup is outside timing and is identical except for the mode value.
    scheduling = ('    s_setreg_imm32_b32 '
                  'hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), '
                  f'{2 if expert_scheduling else 0}')
    return f'''
.text
.p2align 8
.globl {name}
.type {name},@function
{name}:
    s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
    s_nop 0
    s_load_b128 s[4:7], s[0:1], 0
    s_wait_kmcnt 0
{chr(10).join(initialize)}
{scheduling}
    s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
    s_nop 0
{set_msb(accumulator_init, wmma_mode)}
    s_mov_b32 s12, s6
    s_get_shader_cycles_u64 s[8:9]
.Lloop_{name}:
{'    s_wait_dscnt 0' if block_wait else ''}
{body}
    s_sub_u32 s12, s12, 1
    s_cmp_lg_u32 s12, 0
    s_cbranch_scc1 .Lloop_{name}
    s_wait_dscnt 0
    s_wait_alu depctr_va_vdst(0)
    s_get_shader_cycles_u64 s[10:11]
    s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 0
    s_sub_u32 s14, s10, s8
    s_subb_u32 s15, s11, s9
{set_msb(wmma_mode, accumulator_bank)}
    v_mov_b32 v4, v0
{extra_accumulators}
{set_msb(accumulator_bank, (mode >> 6) & 3)}
{ds_result}
{set_msb((mode >> 6) & 3, 0)}
    v_mov_b32 v2, s14
    v_mov_b32 v3, s15
    v_lshlrev_b32 v96, 4, v0
    s_wait_alu depctr_va_vdst(0)
    global_store_b128 v96, v[2:5], s[4:5]
{extra_stores}
    s_wait_storecnt 0
    s_endpgm
.Lend_{name}:
.size {name}, .Lend_{name}-{name}
.rodata
.p2align 6
.amdhsa_kernel {name}
    .amdhsa_group_segment_fixed_size 512
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 1024
    .amdhsa_next_free_sgpr 16
    .amdhsa_reserve_vcc 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
'''


def build(output: pathlib.Path, groups: list[int], gaps: list[int],
          variants: list[str], ds_reads: int = 1, ds_bits: int = 128,
          unroll: int | None = None, block_wait: bool = False,
          middle_op: str = 'ds', lane_stride: int = 16,
          ds_base_vgpr: int = 96, wmma_a_vgpr: int = 0,
          expert_scheduling: bool = False,
          accumulator_bank: int = 2) -> dict:
    """Assemble and disassemble the exact code object used in the test."""
    output.mkdir(parents=True, exist_ok=True)
    if unroll is None:
        unroll = UNROLL // ds_reads
    configurations = {}
    bodies = ['.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"\n'
              '.amdhsa_code_object_version 6']
    metadata = []
    for group in groups:
        for gap in gaps:
            for variant in variants:
                name = f'msb_{variant}_g{group}_gap{gap}'
                configurations[name] = (variant, group, gap, ds_reads,
                                        ds_bits, unroll, block_wait, middle_op,
                                        lane_stride, ds_base_vgpr, wmma_a_vgpr,
                                        expert_scheduling, accumulator_bank)
                bodies.append(kernel_source(name, variant, group, gap,
                                            ds_reads, ds_bits, unroll,
                                            block_wait, middle_op, lane_stride,
                                            ds_base_vgpr, wmma_a_vgpr,
                                            expert_scheduling,
                                            accumulator_bank))
                metadata.append(f'''
  - .name: {name}
    .symbol: {name}.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 512
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 16
    .vgpr_count: 1024
    .max_flat_workgroup_size: 32
    .args:
      - .name: output
        .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
      - .name: iterations
        .size: 4
        .offset: 8
        .value_kind: by_value
      - .name: padding
        .size: 4
        .offset: 12
        .value_kind: by_value
''')
    bodies.append('.amdgpu_metadata\n---\namdhsa.version: [1, 2]\n'
                  'amdhsa.kernels:\n' + ''.join(metadata) +
                  '...\n.end_amdgpu_metadata\n')
    source = output/'msb_latency.s'
    source.write_text('\n'.join(bodies))
    commands = [
        [str(TOOLS/'llvm-mc'), '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx1250',
         '-filetype=obj', str(source), '-o', str(output/'msb_latency.o')],
        [str(TOOLS/'ld.lld'), '-shared', str(output/'msb_latency.o'),
         '-o', str(output/'msb_latency.hsaco')],
        [str(TOOLS/'llvm-objdump'), '-d', '--mcpu=gfx1250',
         str(output/'msb_latency.hsaco')],
    ]
    (output/'build_commands.json').write_text(json.dumps(commands, indent=2))
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True,
                                check=True)
    (output/'msb_latency.disasm').write_text(result.stdout)
    audit_assembly(result.stdout, configurations, output)
    return configurations


def audit_assembly(disassembly: str, configurations: dict,
                   output: pathlib.Path):
    """Check the actual code object's timed sequences and physical operands."""
    symbols = list(re.finditer(r'^\w+ <(msb_\w+)>:', disassembly, re.M))
    results = {}
    for index, symbol in enumerate(symbols):
        name = symbol[1]
        (variant, group, gap, ds_reads, ds_bits, unroll,
         block_wait, middle_op, lane_stride, ds_base_vgpr,
         wmma_a_vgpr, expert_scheduling, accumulator_bank) = configurations[name]
        wmma_mode, mode = bank_modes(variant, accumulator_bank)
        end = symbols[index+1].start() if index+1 < len(symbols) else None
        text = disassembly[symbol.end():end]
        instructions = [line.split('//')[0].strip()
                        for line in text.splitlines() if '// ' in line]
        clocks = [i for i, line in enumerate(instructions)
                  if line.startswith('s_get_shader_cycles_u64')]
        assert len(clocks) == 2
        scheduling = [line for line in instructions
                      if 'hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2)' in line]
        prefix = 's_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), '
        assert scheduling == [prefix + ('2' if expert_scheduling else '0'),
                              prefix + '0'], scheduling
        assert scheduling[0] in instructions[:clocks[0]]
        assert scheduling[-1] in instructions[clocks[1]+1:]
        assert (f'v_lshlrev_b32_e32 v96, {lane_stride.bit_length()-1}, v0'
                in instructions[:clocks[0]])
        timed = instructions[clocks[0]+1:clocks[1]]
        wmmas = [line for line in timed if line.startswith('v_wmma_')]
        actual_wmmas = 0 if variant == 'ds_only' else group
        assert len(wmmas) == actual_wmmas*unroll
        for fragment, line in enumerate(wmmas):
            start = 256*accumulator_bank + (fragment % group)*16
            assert f'/*v[{start}:{start+15}]*/' in line
            a_start = 256 + wmma_a_vgpr
            assert (f'v[64:79] /*v[320:335]*/, '
                    f'v[{wmma_a_vgpr}:{wmma_a_vgpr+7}] '
                    f'/*v[{a_start}:{a_start+7}]*/') in line
        sets = [int(line.split()[1], 0) for line in timed
                if line.startswith('s_set_vgpr_msb')]
        expected = [] if variant in (
            'no_msb', 'nop_control', 'wmma_only', 'ds_only',
            'batched_no_msb') else [
            wmma_mode << 8 | mode,
            mode << 8 | wmma_mode]*unroll
        assert sets == expected, (name, sets, expected)
        changed = wmma_mode ^ mode
        changed_fields = [field for index, field in enumerate(
            ('src0', 'src1', 'src2', 'dst')) if changed & (3 << (index*2))]
        if variant in ('src0_only', 'src1_only', 'src2_only'):
            assert changed_fields == [variant.split('_')[0]]
        loads = [line for line in timed if line.startswith('ds_load_')]
        actual_ds_reads = (ds_reads if variant != 'wmma_only'
                           and middle_op == 'ds' else 0)
        assert len(loads) == actual_ds_reads*unroll
        for index, load in enumerate(loads):
            destination = 128 + index*4 + ((mode >> 6) & 3)*256
            if destination >= 256:
                annotation = (f'/*v[{destination}:{destination+3}]*/'
                              if ds_bits == 128 else f'/*v{destination}*/')
                assert annotation in load
            else:
                assert load.startswith(
                    f'ds_load_b{ds_bits} v[{destination}:{destination+3}],'
                    if ds_bits == 128 else
                    f'ds_load_b{ds_bits} v{destination},')
            base = f'v{ds_base_vgpr}'
            region = mode & 3
            if region:
                base += f' /*v{ds_base_vgpr+256*region}*/'
            assert load.endswith(', ' + base), load
        adds = [line for line in timed if line.startswith('v_add_nc_u32_e32')]
        actual_adds = int(middle_op == 'vadd' and variant != 'wmma_only')
        assert len(adds) == actual_adds*unroll
        for index, add in enumerate(adds):
            destination = 128 + index*4 + ((mode >> 6) & 3)*256
            expected_destination = f'v{128+index*4}'
            if destination >= 256:
                expected_destination += f' /*v{destination}*/'
            sources = []
            for shift in (0, 2):
                region = (mode >> shift) & 3
                sources.append('v96' + (f' /*v{96+256*region}*/'
                                        if region else ''))
            assert add.split(None, 1)[1] == ', '.join(
                [expected_destination] + sources), add
        one_round = ['v_wmma_scale_f32_32x16x128_f4']*actual_wmmas
        one_round += ['s_nop']*gap
        if variant == 'delay_control':
            one_round += ['s_nop']
        if variant not in ('no_msb', 'wmma_only', 'ds_only', 'batched_no_msb'):
            one_round += [
                's_nop' if variant == 'nop_control' else 's_set_vgpr_msb']
        one_round += [f'ds_load_b{ds_bits}']*actual_ds_reads
        one_round += ['v_add_nc_u32_e32']*actual_adds
        if variant not in ('no_msb', 'wmma_only', 'ds_only', 'batched_no_msb'):
            one_round += [
                's_nop' if variant == 'nop_control' else 's_set_vgpr_msb']
        expected_opcodes = (['s_wait_dscnt'] if block_wait else [])
        if variant == 'batched_no_msb':
            expected_opcodes += (
                ['v_wmma_scale_f32_32x16x128_f4']*actual_wmmas*unroll
                + [f'ds_load_b{ds_bits}']*actual_ds_reads*unroll
                + ['v_add_nc_u32_e32']*actual_adds*unroll)
        else:
            expected_opcodes += one_round*unroll
        expected_opcodes += [
            's_sub_co_u32', 's_cmp_lg_u32', 's_cbranch_scc1',
            's_wait_dscnt', 's_wait_alu']
        assert [line.split()[0] for line in timed] == expected_opcodes
        results[name] = {'timed_instructions': timed,
                         'accumulator_bank': accumulator_bank,
                         'wmma_mode': wmma_mode,
                         'ds_mode': mode,
                         'expert_scheduling_mode':
                             2 if expert_scheduling else 0,
                         'physical_pins_checked': True,
                         'forward_changed_fields': changed_fields,
                         'no_wait_between_wmma_and_ds': True,
                         'unroll': unroll,
                         'block_wait': block_wait,
                         'wmma_per_round': actual_wmmas,
                         'ds_per_round': actual_ds_reads,
                         'vadd_per_round': actual_adds,
                         'msb_per_round': len(sets) // unroll,
                         'ds_bits': ds_bits}
        results[name]['lane_stride_bytes'] = lane_stride
        results[name]['ds_base_physical_vgpr'] = (
            ds_base_vgpr + 256*(mode & 3))
        results[name]['wmma_a_physical_vgpr'] = 256 + wmma_a_vgpr
        results[name]['lds_bank_per_lane'] = [
            (lane*lane_stride//4) % 64 for lane in range(32)]
    assert len(results) == len(configurations)
    (output/'assembly_audit.json').write_text(json.dumps(results, indent=2))


def difference_interval(candidate: list[float], baseline: list[float]) -> dict:
    """Estimate a paired mean difference and its 95% bootstrap interval."""
    differences = [a-b for a, b in zip(candidate, baseline)]
    randomizer = random.Random(42)
    means = sorted(statistics.fmean(randomizer.choices(
        differences, k=len(differences))) for _ in range(4000))
    return {'mean_difference_cycles': statistics.fmean(differences),
            'ci95_cycles': [means[100], means[3899]],
            'paired_differences_cycles': differences}


def assert_equivalent(comparison: dict):
    """Reject differences outside the predefined one-cycle equivalence band."""
    low, high = comparison['ci95_cycles']
    assert -1.0 <= low <= high <= 1.0, comparison


def assert_sensitive(comparison: dict):
    """Require detection of the intentionally inserted 64-cycle NOP."""
    assert comparison['ci95_cycles'][0] >= 16.0, comparison


def measurement_tests(measurements: dict, groups: list[int], gaps: list[int],
                      output: pathlib.Path) -> unittest.TestSuite:
    """Check DST equivalence and sensitivity without assuming SRC cost."""
    suite = unittest.TestSuite()
    results = {}
    for group in groups:
        for gap in gaps:
            suffix = f'g{group}_gap{gap}'
            for candidate, baseline in (
                ('same_msb', 'nop_control'),
                ('dst_only', 'nop_control'),
                ('dst_only', 'same_msb'),
                ('dst_only_bank1', 'same_msb'),
                ('delay_control', 'dst_only'),
            ):
                if any(f'msb_{variant}_{suffix}' not in measurements
                       for variant in (candidate, baseline)):
                    continue
                name = f'{candidate}_vs_{baseline}_{suffix}'
                comparison = difference_interval(
                    measurements[f'msb_{candidate}_{suffix}'],
                    measurements[f'msb_{baseline}_{suffix}'])
                results[name] = comparison
                check = (assert_sensitive if candidate == 'delay_control'
                         else assert_equivalent)
                test = functools.update_wrapper(
                    functools.partial(check, comparison), check)
                suite.addTest(unittest.FunctionTestCase(test, description=name))
    (output/'comparisons.json').write_text(json.dumps(results, indent=2))
    source_results = {}
    for group in groups:
        for gap in gaps:
            suffix = f'g{group}_gap{gap}'
            for candidate, baseline in (
                ('src0_only', 'same_msb'),
                ('src0_only', 'dst_only'),
                ('src0_dst', 'dst_only'),
                ('src012_dst', 'dst_only'),
                ('same_msb', 'wmma_only'),
                ('no_msb', 'wmma_only'),
                ('same_msb', 'no_msb'),
                ('dst_only', 'no_msb'),
                ('batched_no_msb', 'no_msb'),
                ('batched_no_msb', 'wmma_only'),
                ('f540', 'no_msb'),
                ('f540', 'same_msb'),
                ('f540', 'dst_only_bank1'),
            ):
                if any(f'msb_{variant}_{suffix}' not in measurements
                       for variant in (candidate, baseline)):
                    continue
                name = f'{candidate}_vs_{baseline}_{suffix}'
                comparison = difference_interval(
                    measurements[f'msb_{candidate}_{suffix}'],
                    measurements[f'msb_{baseline}_{suffix}'])
                comparison['assertion'] = 'diagnostic_no_assumed_latency'
                source_results[name] = comparison
                print(f"{name}: delta="
                      f"{comparison['mean_difference_cycles']:+.6f} "
                      f"cycles/round, CI95={comparison['ci95_cycles']}",
                      flush=True)
    (output/'source_comparisons.json').write_text(
        json.dumps(source_results, indent=2))
    return suite


class Hip:
    """Small HIP module launcher for the preassembled kernels."""

    def __init__(self, code: pathlib.Path):
        self.library = ctypes.CDLL('libamdhip64.so.7')
        pointer = ctypes.c_void_p
        unsigned = ctypes.c_uint
        signatures = {
            'hipInit': [unsigned],
            'hipSetDevice': [ctypes.c_int],
            'hipMalloc': [ctypes.POINTER(pointer), ctypes.c_size_t],
            'hipFree': [pointer],
            'hipModuleLoad': [ctypes.POINTER(pointer), ctypes.c_char_p],
            'hipModuleGetFunction': [ctypes.POINTER(pointer), pointer,
                                     ctypes.c_char_p],
            'hipModuleLaunchKernel': [pointer] + [unsigned]*7 + [pointer]*3,
            'hipDeviceSynchronize': [],
            'hipMemcpy': [pointer, pointer, ctypes.c_size_t, ctypes.c_int],
            'hipModuleUnload': [pointer],
        }
        for name, signature in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = signature
            function.restype = ctypes.c_int
        self.call('hipInit', 0)
        self.call('hipSetDevice', 0)
        self.module = pointer()
        self.device = pointer()
        self.call('hipModuleLoad', ctypes.byref(self.module),
                  str(code).encode())
        self.call('hipMalloc', ctypes.byref(self.device), 2048)
        self.functions = {}

    def call(self, name: str, *args):
        """Raise on every failed HIP call."""
        error = getattr(self.library, name)(*args)
        if error:
            raise RuntimeError(f'{name} failed with HIP error {error}')

    def launch(self, name: str, iterations: int, group: int,
               unroll: int = UNROLL, middle_op: str = 'ds',
               lane_stride: int = 16) -> int:
        """Return cycles and validate LDS data and every lane's accumulator."""
        if name not in self.functions:
            function = ctypes.c_void_p()
            self.call('hipModuleGetFunction', ctypes.byref(function),
                      self.module, name.encode())
            self.functions[name] = function
        count = ctypes.c_uint(iterations)
        padding = ctypes.c_uint(0)
        arguments = (ctypes.c_void_p*3)(
            ctypes.addressof(self.device), ctypes.addressof(count),
            ctypes.addressof(padding))
        self.call('hipModuleLaunchKernel', self.functions[name], 1, 1, 1,
                  32, 1, 1, 0, None, arguments, None)
        self.call('hipDeviceSynchronize')
        data = (ctypes.c_uint32*512)()
        self.call('hipMemcpy', data, self.device, 2048, 2)
        expected_float = iterations*unroll*128.0
        if name.startswith('msb_ds_only_'):
            expected_float = 0.0
        expected = struct.unpack('I', struct.pack('f', expected_float))[0]
        cycles = data[0] | data[1] << 32
        for lane in range(32):
            values = tuple(data[lane*4:lane*4+4])
            assert values[:2] == (data[0], data[1]), values
            assert values[2] == expected, (name, lane, values, expected)
            expected_ds = 0 if name.startswith('msb_wmma_only_') else 0x22222222
            if middle_op == 'vadd' and not name.startswith('msb_wmma_only_'):
                expected_ds = lane*lane_stride*2
            assert values[3] == expected_ds, (name, lane, values)
            for accumulator in range(1, group):
                value = data[128*accumulator+lane*4]
                assert value == expected, (
                    name, lane, accumulator, value, expected)
        return cycles

    def close(self):
        """Release HIP resources after all launches complete."""
        self.call('hipFree', self.device)
        self.call('hipModuleUnload', self.module)


def main():
    """Build, measure interleaved variants, and test the latency hypothesis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=pathlib.Path,
                        default=pathlib.Path('flydsl_pin_vgpr/msb_rept'))
    parser.add_argument('--samples', type=int, default=51)
    parser.add_argument('--iterations', type=int, default=512)
    parser.add_argument('--groups', type=int, nargs='+', choices=[1, 2, 4],
                        default=[1], help='WMMAs per round, separate accumulators')
    parser.add_argument('--variants', nargs='+', choices=list(MODES),
                        default=[v for v in MODES if v != 'f540'],
                        help='MSB variants to compare')
    parser.add_argument('--accumulator-bank', type=int, choices=[2, 3],
                        default=2, help='Physical bank for WMMA SRC2 and DST')
    parser.add_argument('--ds-reads', type=int, choices=[1, 2, 3, 4, 5], default=1,
                        help='DS reads per round')
    parser.add_argument('--ds-bits', type=int, choices=[32, 128], default=128,
                        help='Bit width of each DS read')
    parser.add_argument('--middle-op', choices=['ds', 'vadd'], default='ds',
                        help='Replace the DS read with a vector integer add')
    parser.add_argument('--lane-stride', type=int, choices=[4, 16], default=16,
                        help='Byte stride between lanes in the LDS address')
    parser.add_argument('--ds-base-vgpr', type=int, choices=[0, 96], default=96,
                        help='Encoded register holding the DS address')
    parser.add_argument('--wmma-a-vgpr', type=int, choices=[0, 32], default=0,
                        help='Encoded first register of the WMMA A operand')
    parser.add_argument('--expert-scheduling', action='store_true',
                        help='Enable HW_REG_WAVE_SCHED_MODE=2 for the timed loop')
    parser.add_argument('--unroll', type=int,
                        help='Rounds per block; defaults to 32 / ds-reads')
    parser.add_argument('--block-wait', action='store_true',
                        help='Restore the older DS wait at every block start')
    parser.add_argument('--gaps', type=int, nargs='+', choices=[0],
                        default=[0], help='No padding in the repeated sequence')
    args = parser.parse_args()
    if args.samples < 15 or not 1 <= args.iterations <= 65536:
        parser.error('Use at least 15 samples and 1..65536 iterations')
    unroll = args.unroll if args.unroll is not None else UNROLL // args.ds_reads
    if not 1 <= unroll <= UNROLL // args.ds_reads:
        parser.error('Unrolled block must fit 32 distinct DS sinks')
    if args.middle_op == 'vadd' and args.ds_reads != 1:
        parser.error('VADD replacement requires --ds-reads 1')
    if args.lane_stride == 4 and args.ds_bits != 32:
        parser.error('--lane-stride 4 requires --ds-bits 32')
    if 'f540' in args.variants and args.accumulator_bank != 3:
        parser.error('--variants f540 requires --accumulator-bank 3')
    configurations = build(args.output, args.groups, args.gaps, args.variants,
                           args.ds_reads, args.ds_bits, unroll, args.block_wait,
                           args.middle_op, args.lane_stride,
                           args.ds_base_vgpr, args.wmma_a_vgpr,
                           args.expert_scheduling, args.accumulator_bank)
    if not args.run:
        print('Assembly built; pass --run under gpu-lock to measure.')
        return
    check = subprocess.run(['rocm-smi', '--showpids'], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.output/'idle.log').write_text(check.stdout)
    print(check.stdout, flush=True)
    if check.returncode or 'No KFD PIDs currently running' not in check.stdout:
        raise SystemExit('GPU is busy or idleness is unknown; deferred.')
    (args.output/'command.json').write_text(json.dumps({
        'argv': sys.argv, 'cwd': str(pathlib.Path.cwd()),
        'source_sha256': hashlib.sha256(
            pathlib.Path(__file__).read_bytes()).hexdigest(),
        'hsaco_sha256': hashlib.sha256(
            (args.output/'msb_latency.hsaco').read_bytes()).hexdigest(),
        'timer': 's_get_shader_cycles_u64',
        'equivalence_margin_cycles': 1.0,
        'sequence': f'rept {unroll} of (wmma x group, msb, '
                    f'{args.middle_op} x{args.ds_reads}, msb)',
        'wmma_per_round': args.groups, 'msb_per_round': 2,
        'ds_per_round': args.ds_reads if args.middle_op == 'ds' else 0,
        'middle_op': args.middle_op,
        'lane_stride_bytes': args.lane_stride,
        'ds_base_vgpr': args.ds_base_vgpr,
        'wmma_a_vgpr': args.wmma_a_vgpr,
        'expert_scheduling_mode': 2 if args.expert_scheduling else 0,
        'accumulator_bank': args.accumulator_bank,
        'ds_bits': args.ds_bits,
        'block_wait': args.block_wait,
        'per_variant_counts': {
            variant: {'ds': args.ds_reads if variant != 'wmma_only'
                      and args.middle_op == 'ds' else 0,
                      'vadd': int(variant != 'wmma_only'
                                  and args.middle_op == 'vadd'),
                      'msb': 0 if variant in (
                          'wmma_only', 'no_msb', 'nop_control', 'ds_only',
                          'batched_no_msb') else 2,
                      'wmma': 0 if variant == 'ds_only' else args.groups}
            for variant in args.variants},
    }, indent=2))
    hip = Hip(args.output.resolve()/'msb_latency.hsaco')
    measurements = {name: [] for name in configurations}
    raw_cycles = {name: [] for name in configurations}
    try:
        for _ in range(5):
            for name in configurations:
                hip.launch(name, args.iterations, configurations[name][1],
                           unroll, args.middle_op, args.lane_stride)
        randomizer = random.Random(0)
        for sample in range(args.samples):
            names = list(configurations)
            randomizer.shuffle(names)
            for name in names:
                cycles = hip.launch(
                    name, args.iterations, configurations[name][1], unroll,
                    args.middle_op, args.lane_stride)
                raw_cycles[name].append(cycles)
                measurements[name].append(cycles/(args.iterations*unroll))
            if sample % 5 == 0:
                print(f'Completed sample {sample+1}/{args.samples}', flush=True)
    finally:
        hip.close()
    (args.output/'measurements.json').write_text(json.dumps({
        'iterations': args.iterations, 'unroll': unroll,
        'ds_per_round': args.ds_reads if args.middle_op == 'ds' else 0,
        'middle_op': args.middle_op, 'wmma_per_round': args.groups,
        'lane_stride_bytes': args.lane_stride,
        'ds_base_vgpr': args.ds_base_vgpr,
        'wmma_a_vgpr': args.wmma_a_vgpr,
        'expert_scheduling_mode': 2 if args.expert_scheduling else 0,
        'accumulator_bank': args.accumulator_bank,
        'ds_bits': args.ds_bits,
        'block_wait': args.block_wait,
        'samples': args.samples,
        'clock_instruction': 's_get_shader_cycles_u64',
        'raw_total_cycles': raw_cycles,
        'cycles_per_round': measurements}, indent=2))
    for name, values in measurements.items():
        print(f'{name}: median={statistics.median(values):.3f} '
              f'range={min(values):.3f}..{max(values):.3f} cycles/round')
    suite = measurement_tests(measurements, args.groups, args.gaps, args.output)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (args.output/'test_result.json').write_text(json.dumps({
        'tests_run': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'passed': result.wasSuccessful(),
    }, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
