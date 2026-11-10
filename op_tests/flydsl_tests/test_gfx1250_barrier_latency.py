"""Measure adjacent barrier signal/wait on gfx1250 using shader-clock cycles.

Build only by default. Run with --run under gpu-lock. Each launch contains one
workgroup of 4 or 8 wave32 waves. No tensor, LDS, or WMMA work is issued. This
measures an otherwise idle rendezvous, not waits caused by unequal producer work.
"""

import argparse
import ctypes
import json
import pathlib
import re
import statistics
import subprocess

TOOLS = pathlib.Path('/opt/rocm/llvm/bin')
UNROLL = 32
VARIANTS = ('empty', 'nop_pair', 'signal_wait')


def kernel_source(name: str, mode: int) -> str:
    """Generate a kernel with exact instruction order and per-thread outputs."""
    body = {'empty': '', 'nop_pair': 's_nop 0\n    s_nop 0',
            'signal_wait': 's_barrier_signal -1\n    s_barrier_wait -1'}[name]
    return f'''
.text
.p2align 8
.globl {name}
.type {name},@function
{name}:
    s_load_b128 s[4:7], s[0:1], 0
    s_wait_kmcnt 0
    s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), {mode}
    s_mov_b32 s12, s6
    // Align wave arrival before starting the clocks in every variant.
    s_barrier_signal -1
    s_barrier_wait -1
    s_get_shader_cycles_u64 s[8:9]
.Lloop_{name}:
    .rept {UNROLL}
    {body}
    .endr
    s_sub_u32 s12, s12, 1
    s_cmp_lg_u32 s12, 0
    s_cbranch_scc1 .Lloop_{name}
    s_get_shader_cycles_u64 s[10:11]
    s_sub_u32 s14, s10, s8
    s_subb_u32 s15, s11, s9
    s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 0
    v_mov_b32 v2, s14
    v_mov_b32 v3, s15
    v_lshlrev_b32 v1, 3, v0
    s_wait_alu depctr_va_vdst(0)
    global_store_b64 v1, v[2:3], s[4:5]
    s_wait_storecnt 0
    s_endpgm
.Lend_{name}:
.size {name}, .Lend_{name}-{name}
.rodata
.p2align 6
.amdhsa_kernel {name}
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_wavefront_size32 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 4
    .amdhsa_next_free_sgpr 16
    .amdhsa_reserve_vcc 0
.end_amdhsa_kernel
'''


def build(output: pathlib.Path, mode: int) -> pathlib.Path:
    """Assemble, disassemble, and verify the timed instructions."""
    output.mkdir(parents=True, exist_ok=True)
    source = ['.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"',
              '.amdhsa_code_object_version 6']
    metadata = []
    for name in VARIANTS:
        source.append(kernel_source(name, mode))
        metadata.append(f'''
  - .name: {name}
    .symbol: {name}.kd
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .sgpr_count: 16
    .vgpr_count: 4
    .max_flat_workgroup_size: 256
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
    source.append('.amdgpu_metadata\n---\namdhsa.version: [1, 2]\n'
                  'amdhsa.kernels:\n' + ''.join(metadata)
                  + '...\n.end_amdgpu_metadata\n')
    path = output / 'barrier_latency.s'
    path.write_text('\n'.join(source))
    code = output / 'barrier_latency.hsaco'
    commands = [
        [str(TOOLS / 'llvm-mc'), '-triple=amdgcn-amd-amdhsa',
         '-mcpu=gfx1250', '-filetype=obj', str(path),
         '-o', str(output / 'barrier_latency.o')],
        [str(TOOLS / 'ld.lld'), '-shared', str(output / 'barrier_latency.o'),
         '-o', str(code)],
        [str(TOOLS / 'llvm-objdump'), '-d', '--mcpu=gfx1250', str(code)],
    ]
    (output / 'build_commands.json').write_text(json.dumps(commands, indent=2))
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True,
                                check=False)
        if result.returncode:
            raise RuntimeError(result.stderr)
    disassembly = result.stdout
    (output / 'barrier_latency.disasm').write_text(disassembly)
    symbols = list(re.finditer(r'^\w+ <(\w+)>:', disassembly, re.MULTILINE))
    audit = {}
    for index, symbol in enumerate(symbols):
        name = symbol[1]
        end = symbols[index + 1].start() if index + 1 < len(symbols) else None
        instructions = [line.split('//')[0].strip() for line in
                        disassembly[symbol.end():end].splitlines()
                        if '// ' in line]
        clocks = [i for i, line in enumerate(instructions)
                  if line.startswith('s_get_shader_cycles_u64')]
        assert len(clocks) == 2, (name, clocks)
        timed = instructions[clocks[0] + 1:clocks[1]]
        pair = {'empty': [], 'nop_pair': ['s_nop 0'] * 2,
                'signal_wait': ['s_barrier_signal -1',
                                's_barrier_wait 0xffff']}[name]
        assert timed[:-3] == pair * UNROLL, (name, timed)
        assert [line.split()[0] for line in timed[-3:]] == [
            's_sub_co_u32', 's_cmp_lg_u32', 's_cbranch_scc1']
        audit[name] = timed
    assert set(audit) == set(VARIANTS)
    (output / 'assembly_audit.json').write_text(json.dumps(audit, indent=2))
    return code


class Hip:
    """Load the assembly and read back each wave's shader-clock duration."""

    def __init__(self, code: pathlib.Path):
        self.library = ctypes.CDLL('libamdhip64.so.7')
        pointer = ctypes.c_void_p
        unsigned = ctypes.c_uint
        signatures = {
            'hipInit': [unsigned], 'hipSetDevice': [ctypes.c_int],
            'hipMalloc': [ctypes.POINTER(pointer), ctypes.c_size_t],
            'hipFree': [pointer],
            'hipModuleLoad': [ctypes.POINTER(pointer), ctypes.c_char_p],
            'hipModuleGetFunction': [ctypes.POINTER(pointer), pointer,
                                     ctypes.c_char_p],
            'hipModuleLaunchKernel': [pointer] + [unsigned] * 7 + [pointer] * 3,
            'hipDeviceSynchronize': [],
            'hipMemcpy': [pointer, pointer, ctypes.c_size_t, ctypes.c_int],
            'hipMemset': [pointer, ctypes.c_int, ctypes.c_size_t],
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
        self.call('hipModuleLoad', ctypes.byref(self.module), str(code).encode())
        self.call('hipMalloc', ctypes.byref(self.device), 2048)
        self.functions = {}
        for name in VARIANTS:
            function = pointer()
            self.call('hipModuleGetFunction', ctypes.byref(function),
                      self.module, name.encode())
            self.functions[name] = function

    def call(self, name: str, *args):
        """Check every HIP API result."""
        error = getattr(self.library, name)(*args)
        if error:
            raise RuntimeError(f'{name} failed with HIP error {error}')

    def launch(self, name: str, waves: int, iterations: int) -> list[int]:
        """Launch one workgroup; validate every lane and return per-wave cycles."""
        count, padding = ctypes.c_uint(iterations), ctypes.c_uint(0)
        arguments = (ctypes.c_void_p * 3)(
            ctypes.addressof(self.device), ctypes.addressof(count),
            ctypes.addressof(padding))
        self.call('hipMemset', self.device, 0, 2048)
        self.call('hipModuleLaunchKernel', self.functions[name], 1, 1, 1,
                  waves * 32, 1, 1, 0, None, arguments, None)
        self.call('hipDeviceSynchronize')
        data = (ctypes.c_uint64 * 256)()
        self.call('hipMemcpy', data, self.device, 2048, 2)
        durations = []
        for wave in range(waves):
            values = list(data[wave * 32:(wave + 1) * 32])
            assert values[0] > 0 and len(set(values)) == 1, (wave, values)
            durations.append(values[0])
        assert all(x == 0 for x in data[waves * 32:])
        return durations

    def close(self):
        """Release resources after all launches finish."""
        self.call('hipFree', self.device)
        self.call('hipModuleUnload', self.module)


def main():
    """Build and optionally measure both block sizes in interleaved order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--iterations', type=int, default=128)
    parser.add_argument('--samples', type=int, default=31)
    parser.add_argument('--sched-mode', type=int, choices=(0, 2), default=2)
    args = parser.parse_args()
    if args.iterations < 1 or args.samples < 1:
        parser.error('iterations and samples must be positive')
    code = build(args.output, args.sched_mode)
    if not args.run:
        return
    idle = subprocess.run(['rocm-smi', '--showpids'], capture_output=True,
                          text=True, check=True)
    print(idle.stdout, flush=True)
    if 'No KFD PIDs currently running' not in idle.stdout:
        raise SystemExit('GPU is busy; deferred.')
    hip = Hip(code)
    records = []
    rounds = args.iterations * UNROLL
    try:
        for waves in (4, 8):
            for name in VARIANTS:
                for _ in range(5):
                    hip.launch(name, waves, args.iterations)
            for sample in range(args.samples):
                order = VARIANTS if sample % 2 == 0 else VARIANTS[::-1]
                for name in order:
                    durations = hip.launch(name, waves, args.iterations)
                    records.append({'waves': waves, 'variant': name,
                                    'sample': sample, 'wave_cycles': durations,
                                    'cycles_per_round': max(durations) / rounds})
    finally:
        hip.close()
    summary = []
    for waves in (4, 8):
        values = {name: [r['cycles_per_round'] for r in records
                         if r['waves'] == waves and r['variant'] == name]
                  for name in VARIANTS}
        row = {'waves': waves, 'threads': waves * 32,
               'median_cycles_per_round': {
                   name: statistics.median(v) for name, v in values.items()},
               'min_max_cycles_per_round': {
                   name: [min(v), max(v)] for name, v in values.items()},
               'paired_net_cycles_per_round': statistics.median([
                   a - b for a, b in zip(values['signal_wait'],
                                        values['empty'])])}
        summary.append(row)
        print(json.dumps(row), flush=True)
    report = {'grid': [1, 1, 1], 'wave_size': 32,
              'sched_mode': args.sched_mode, 'rounds': rounds,
              'samples': args.samples,
              'aggregation': 'Median over launches of max wave cycles / rounds',
              'summary': summary, 'records': records}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
