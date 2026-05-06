#!/usr/bin/env python3
"""
Extract LDS and VGPR usage from compiled Triton kernels.
Parses the HSACO files in .triton/cache to get resource usage.
"""
import subprocess
import re
import os
import glob
import json

def find_latest_kernel_hsaco():
    """Find the most recently compiled sage_fwd kernel HSACO."""
    cache_dir = os.path.expanduser("~/.triton/cache")

    # Find all HSACO files
    hsaco_files = glob.glob(f"{cache_dir}/**/*.hsaco", recursive=True)

    if not hsaco_files:
        print("No HSACO files found in .triton/cache")
        return None

    # Filter for sage_fwd kernels
    sage_files = [f for f in hsaco_files if 'sage_fwd' in os.path.basename(os.path.dirname(f))]

    if not sage_files:
        print("No sage_fwd kernel found, using most recent HSACO")
        sage_files = hsaco_files

    # Sort by modification time, newest first
    sage_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    return sage_files[0]

def extract_kernel_metadata(hsaco_path):
    """Extract kernel metadata using roc-obj-ls."""

    try:
        result = subprocess.run(
            ['roc-obj-ls', '-v', hsaco_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        return result.stdout
    except Exception as e:
        print(f"Error running roc-obj-ls: {e}")
        return None

def parse_kernel_stats(metadata):
    """Parse LDS, VGPR, SGPR from roc-obj-ls output."""

    stats = {
        'kernel_name': None,
        'lds_bytes': None,
        'vgpr_count': None,
        'sgpr_count': None,
        'wave_size': None,
        'workgroup_size': None,
    }

    lines = metadata.split('\n')

    for i, line in enumerate(lines):
        # Kernel name
        if 'Name:' in line or 'Kernel' in line:
            match = re.search(r'(?:Name:|Kernel)\s*[:=]?\s*(\S+)', line)
            if match:
                stats['kernel_name'] = match.group(1)

        # LDS size (workgroup group segment size)
        if 'group_segment_fixed_size' in line.lower() or 'lds' in line.lower():
            match = re.search(r'(\d+)', line)
            if match:
                stats['lds_bytes'] = int(match.group(1))

        # VGPR count (per work-item)
        if 'vgpr' in line.lower() or 'granulated_wavefront_vgpr_count' in line.lower():
            match = re.search(r'(\d+)', line)
            if match:
                # ROCm reports granulated count, actual = (gran + 1) * granule_size
                # For gfx9+: granule_size = 8 for VGPRs
                gran_count = int(match.group(1))
                if 'granulated' in line.lower():
                    stats['vgpr_count'] = (gran_count + 1) * 8
                else:
                    stats['vgpr_count'] = gran_count

        # SGPR count (per wavefront)
        if 'sgpr' in line.lower() or 'granulated_workitem_vgpr_count' in line.lower():
            match = re.search(r'(\d+)', line)
            if match:
                # For gfx9+: granule_size = 16 for SGPRs
                gran_count = int(match.group(1))
                if 'granulated' in line.lower():
                    stats['sgpr_count'] = (gran_count + 1) * 16
                else:
                    stats['sgpr_count'] = gran_count

        # Workgroup size
        if 'workgroup_size' in line.lower() or 'block_size' in line.lower():
            match = re.search(r'(\d+)', line)
            if match:
                stats['workgroup_size'] = int(match.group(1))

    return stats

def analyze_occupancy(stats):
    """Calculate theoretical occupancy limits."""

    print(f"\n{'='*60}")
    print(f"Kernel: {stats.get('kernel_name', 'Unknown')}")
    print(f"{'='*60}")

    lds_bytes = stats.get('lds_bytes')
    vgpr_count = stats.get('vgpr_count')
    sgpr_count = stats.get('sgpr_count')
    workgroup_size = stats.get('workgroup_size', 512)  # Default 8 warps * 64

    print(f"\nResource Usage per Workgroup:")
    print(f"  LDS (shared memory): {lds_bytes} bytes")
    print(f"  VGPR per work-item:  {vgpr_count}")
    print(f"  SGPR per wavefront:  {sgpr_count}")
    print(f"  Workgroup size:      {workgroup_size} work-items")

    # GFX950 limits
    lds_per_cu = 65536  # bytes
    vgpr_per_cu = 65536  # total VGPRs
    waves_per_cu = 64    # max waves

    wave_size = 64  # AMD wavefront size
    waves_per_wg = workgroup_size // wave_size

    print(f"\n{'Occupancy Analysis (GFX950)':^60}")
    print(f"  CU limits: {lds_per_cu}B LDS, {vgpr_per_cu} VGPRs, {waves_per_cu} waves")

    # LDS limit
    if lds_bytes:
        max_wg_lds = lds_per_cu // lds_bytes if lds_bytes > 0 else float('inf')
        print(f"\n  LDS limit:")
        print(f"    {lds_bytes}B per workgroup")
        print(f"    Max {max_wg_lds} workgroups/CU")

        if max_wg_lds == 1:
            print(f"    ⚠️  BOTTLENECK: Only 1 workgroup fits!")
            print(f"    ⚠️  Occupancy limited to 1/{waves_per_wg} waves = low latency hiding")
        elif max_wg_lds == 2:
            print(f"    ✓ 2 workgroups fit (good occupancy)")

    # VGPR limit
    if vgpr_count:
        vgpr_per_wg = vgpr_count * workgroup_size
        max_wg_vgpr = vgpr_per_cu // vgpr_per_wg if vgpr_per_wg > 0 else float('inf')

        print(f"\n  VGPR limit:")
        print(f"    {vgpr_count} VGPRs/work-item × {workgroup_size} work-items = {vgpr_per_wg} VGPRs/workgroup")
        print(f"    Max {max_wg_vgpr} workgroups/CU")

        if vgpr_per_wg > vgpr_per_cu // 2:
            print(f"    ⚠️  High VGPR usage - may limit occupancy")

    # Wave limit
    max_wg_waves = waves_per_cu // waves_per_wg if waves_per_wg > 0 else float('inf')
    print(f"\n  Wave limit:")
    print(f"    {waves_per_wg} waves/workgroup")
    print(f"    Max {max_wg_waves} workgroups/CU")

    # Effective occupancy
    if lds_bytes and vgpr_count:
        effective_wg = min(max_wg_lds, max_wg_vgpr, max_wg_waves)
        effective_waves = effective_wg * waves_per_wg
        occupancy_pct = (effective_waves / waves_per_cu) * 100

        print(f"\n  Effective occupancy:")
        print(f"    {effective_wg} workgroups/CU (limited by {'LDS' if effective_wg == max_wg_lds else 'VGPR' if effective_wg == max_wg_vgpr else 'waves'})")
        print(f"    {effective_waves}/{waves_per_cu} waves = {occupancy_pct:.1f}% theoretical occupancy")

def main():
    import sys

    # Clear cache and run benchmark
    if len(sys.argv) > 1:
        num_stages = int(sys.argv[1])

        # Update num_stages
        config_file = 'aiter/ops/triton/attention/fav3_sage.py'
        with open(config_file, 'r') as f:
            content = f.read()

        pattern = r'(if arch == "gfx950":.*?"num_stages":\s*)(\d+)'
        new_content = re.sub(pattern, rf'\g<1>{num_stages}', content, count=1, flags=re.DOTALL)

        with open(config_file, 'w') as f:
            f.write(new_content)

        print(f"Set num_stages={num_stages}")

        # Clear Triton cache
        cache_dir = os.path.expanduser("~/.triton/cache")
        os.system(f"rm -rf {cache_dir}/*")
        print(f"Cleared Triton cache")

        # Run benchmark once to compile kernel
        print(f"\nRunning benchmark to compile kernel...")
        os.system("HIP_VISIBLE_DEVICES=0 python op_tests/op_benchmarks/triton/bench_fav3_sage.py -b 1 -sq 75600 -hq 8 -d 128 > /dev/null 2>&1")
        print("Done")

    # Find HSACO
    hsaco_path = find_latest_kernel_hsaco()
    if not hsaco_path:
        print("ERROR: No HSACO file found")
        return

    print(f"\nAnalyzing: {hsaco_path}")

    # Extract metadata
    metadata = extract_kernel_metadata(hsaco_path)
    if not metadata:
        print("ERROR: Failed to extract metadata")
        return

    # Save raw metadata
    metadata_file = f"kernel_metadata_stages{sys.argv[1] if len(sys.argv) > 1 else 'unknown'}.txt"
    with open(metadata_file, 'w') as f:
        f.write(metadata)
    print(f"Raw metadata saved to: {metadata_file}")

    # Parse stats
    stats = parse_kernel_stats(metadata)

    # Analyze
    analyze_occupancy(stats)

if __name__ == '__main__':
    main()
