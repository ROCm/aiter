#!/usr/bin/env python3
"""
Profile kernel resource usage (LDS, VGPR, occupancy) using rocprofv3.
"""
import subprocess
import re
import sys
import os

def run_benchmark_with_profiling(num_stages, use_pingpong=True):
    """Run benchmark with rocprofv3 to capture kernel stats."""

    # Set up environment
    env = os.environ.copy()
    env['HIP_VISIBLE_DEVICES'] = '0'

    if not use_pingpong:
        env['TRITON_HIP_USE_BLOCK_PINGPONG'] = '0'

    # rocprofv3 command to capture kernel stats
    # --stats: basic kernel statistics including LDS, VGPR, occupancy
    cmd = [
        'rocprofv3',
        '--stats',
        'python', 'op_tests/op_benchmarks/triton/bench_fav3_sage.py',
        '-b', '1',
        '-sq', '75600',
        '-hq', '8',
        '-d', '128'
    ]

    print(f"\n{'='*80}")
    print(f"Profiling: num_stages={num_stages}, use_pingpong={use_pingpong}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300
        )

        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print("ERROR: Profiling timed out")
        return None, None
    except Exception as e:
        print(f"ERROR: {e}")
        return None, None

def parse_rocprof_stats(stdout, stderr):
    """Parse rocprofv3 output to extract resource usage."""

    # Look for kernel statistics in output
    # rocprofv3 --stats typically outputs:
    # - Kernel Name
    # - Grid Size
    # - Workgroup Size
    # - LDS Usage (bytes)
    # - VGPR Usage (registers per workitem)
    # - SGPR Usage
    # - Occupancy (waves per SIMD)

    combined = stdout + stderr

    stats = {
        'kernel_name': None,
        'lds_bytes': None,
        'vgpr_count': None,
        'sgpr_count': None,
        'occupancy': None,
        'grid_size': None,
        'workgroup_size': None,
    }

    # Parse for sage_fwd kernel
    lines = combined.split('\n')

    for i, line in enumerate(lines):
        if 'sage_fwd' in line.lower():
            stats['kernel_name'] = 'sage_fwd'

            # Look in surrounding lines for stats
            context = lines[max(0, i-5):min(len(lines), i+20)]

            for ctx_line in context:
                # LDS usage pattern
                if 'lds' in ctx_line.lower() or 'shared' in ctx_line.lower():
                    match = re.search(r'(\d+)\s*(?:bytes?|B)', ctx_line, re.IGNORECASE)
                    if match:
                        stats['lds_bytes'] = int(match.group(1))

                # VGPR pattern
                if 'vgpr' in ctx_line.lower() or 'vector.*reg' in ctx_line.lower():
                    match = re.search(r'(\d+)', ctx_line)
                    if match:
                        stats['vgpr_count'] = int(match.group(1))

                # SGPR pattern
                if 'sgpr' in ctx_line.lower() or 'scalar.*reg' in ctx_line.lower():
                    match = re.search(r'(\d+)', ctx_line)
                    if match:
                        stats['sgpr_count'] = int(match.group(1))

                # Occupancy pattern
                if 'occupancy' in ctx_line.lower() or 'waves.*simd' in ctx_line.lower():
                    match = re.search(r'(\d+\.?\d*)', ctx_line)
                    if match:
                        stats['occupancy'] = float(match.group(1))

    return stats

def analyze_resources(stats, num_stages):
    """Analyze resource usage and predict occupancy limits."""

    print(f"\nKernel: {stats.get('kernel_name', 'Unknown')}")
    print(f"Configuration: num_stages={num_stages}")
    print(f"\n{'Resource Usage:':^40}")
    print(f"  LDS per workgroup:  {stats.get('lds_bytes', 'N/A')} bytes")
    print(f"  VGPR per workitem:  {stats.get('vgpr_count', 'N/A')}")
    print(f"  SGPR per workgroup: {stats.get('sgpr_count', 'N/A')}")
    print(f"  Occupancy:          {stats.get('occupancy', 'N/A')} waves/SIMD")

    # Calculate theoretical occupancy limits
    if stats.get('lds_bytes'):
        lds_bytes = stats['lds_bytes']
        lds_per_cu = 65536  # gfx950
        max_wg_per_cu_lds = lds_per_cu // lds_bytes if lds_bytes > 0 else 'inf'

        print(f"\n{'Occupancy Analysis:':^40}")
        print(f"  LDS per CU: {lds_per_cu} bytes")
        print(f"  Max workgroups/CU (LDS limit): {max_wg_per_cu_lds}")

        if isinstance(max_wg_per_cu_lds, int) and max_wg_per_cu_lds == 1:
            print(f"  ⚠️  WARNING: Only 1 workgroup fits per CU due to LDS!")
            print(f"  ⚠️  This limits occupancy and hurts latency hiding")

    if stats.get('vgpr_count'):
        vgpr_per_workitem = stats['vgpr_count']
        vgpr_per_cu = 65536  # gfx950
        workgroup_size = 512  # 8 warps * 64 threads (typical for num_warps=8)

        vgpr_per_wg = vgpr_per_workitem * workgroup_size
        max_wg_per_cu_vgpr = vgpr_per_cu // vgpr_per_wg if vgpr_per_wg > 0 else 'inf'

        print(f"\n  VGPR per CU: {vgpr_per_cu}")
        print(f"  VGPR per workgroup: {vgpr_per_wg} ({vgpr_per_workitem} * {workgroup_size})")
        print(f"  Max workgroups/CU (VGPR limit): {max_wg_per_cu_vgpr}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python profile_kernel_resources.py <num_stages> [use_pingpong]")
        print("Example: python profile_kernel_resources.py 3")
        print("Example: python profile_kernel_resources.py 4 1")
        sys.exit(1)

    num_stages = int(sys.argv[1])
    use_pingpong = sys.argv[2] != '0' if len(sys.argv) > 2 else True

    # Update num_stages in config file
    import fileinput

    config_file = 'aiter/ops/triton/attention/fav3_sage.py'

    # Read current value
    with open(config_file, 'r') as f:
        content = f.read()

    # Replace num_stages value
    import re
    pattern = r'(if arch == "gfx950":.*?"num_stages":\s*)(\d+)'
    new_content = re.sub(
        pattern,
        rf'\g<1>{num_stages}',
        content,
        count=1,
        flags=re.DOTALL
    )

    with open(config_file, 'w') as f:
        f.write(new_content)

    print(f"Set num_stages={num_stages} in {config_file}")

    # Run profiling
    stdout, stderr = run_benchmark_with_profiling(num_stages, use_pingpong)

    if stdout is None:
        print("Profiling failed")
        return

    # Parse stats
    stats = parse_rocprof_stats(stdout, stderr)

    # Analyze
    analyze_resources(stats, num_stages)

    # Save raw output for manual inspection
    output_file = f'rocprof_output_stages{num_stages}_pingpong{int(use_pingpong)}.txt'
    with open(output_file, 'w') as f:
        f.write("=== STDOUT ===\n")
        f.write(stdout)
        f.write("\n\n=== STDERR ===\n")
        f.write(stderr)

    print(f"\nRaw output saved to: {output_file}")

if __name__ == '__main__':
    main()
