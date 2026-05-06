#!/usr/bin/env python3
"""Extract VGPR/SGPR spills from HSACO metadata."""

import subprocess
import struct
import sys
import glob
import os

def find_latest_hsaco():
    """Find most recently compiled HSACO."""
    cache_dir = os.path.expanduser("~/.triton/cache")
    hsaco_files = glob.glob(f"{cache_dir}/**/*.hsaco", recursive=True)

    if not hsaco_files:
        print("No HSACO files found")
        return None

    # Most recent
    hsaco_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return hsaco_files[0]

def extract_metadata(hsaco_path):
    """Extract kernel metadata from HSACO."""
    result = subprocess.run(
        ['readelf', '-n', hsaco_path],
        capture_output=True,
        text=True
    )

    # Find the hex dump
    lines = result.stdout.split('\n')
    hex_data = []

    for line in lines:
        if 'description data:' in line:
            # Start collecting hex bytes
            idx = lines.index(line)
            for next_line in lines[idx:]:
                if next_line.strip().startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f')):
                    # Parse hex bytes
                    parts = next_line.strip().split()
                    for part in parts:
                        if len(part) == 2 and all(c in '0123456789abcdef' for c in part):
                            hex_data.append(int(part, 16))
                else:
                    break
            break

    # Convert to bytes and decode MessagePack
    import msgpack
    data = bytes(hex_data)
    metadata = msgpack.unpackb(data)

    return metadata

def main():
    hsaco_path = find_latest_hsaco()
    if not hsaco_path:
        sys.exit(1)

    print(f"Analyzing: {hsaco_path}\n")

    try:
        metadata = extract_metadata(hsaco_path)

        # Navigate to kernel info
        kernels = metadata.get('amdhsa.kernels', [])
        if not kernels:
            print("No kernel metadata found")
            return

        kernel = kernels[0]

        # Extract key metrics
        name = kernel.get('.name', 'unknown')
        vgpr_count = kernel.get('.vgpr_count', 0)
        vgpr_spill_count = kernel.get('.vgpr_spill_count', 0)
        sgpr_count = kernel.get('.sgpr_count', 0)
        sgpr_spill_count = kernel.get('.sgpr_spill_count', 0)
        lds_size = kernel.get('.group_segment_fixed_size', 0)
        private_size = kernel.get('.private_segment_fixed_size', 0)
        wavefront_size = kernel.get('.wavefront_size', 64)

        print(f"Kernel: {name}")
        print(f"{'='*60}\n")

        print(f"Register Usage:")
        print(f"  VGPRs:               {vgpr_count}")
        print(f"  VGPR spills:         {vgpr_spill_count}", end='')
        if vgpr_spill_count > 0:
            print(f"  ⚠️  SPILLING TO MEMORY!")
        else:
            print(f"  ✓")

        print(f"  SGPRs:               {sgpr_count}")
        print(f"  SGPR spills:         {sgpr_spill_count}", end='')
        if sgpr_spill_count > 0:
            print(f"  ⚠️  SPILLING TO MEMORY!")
        else:
            print(f"  ✓")

        print(f"\nMemory Usage:")
        print(f"  LDS (shared):        {lds_size} bytes ({lds_size/1024:.1f} KB)")
        print(f"  Private (stack):     {private_size} bytes ({private_size/1024:.1f} KB)")
        print(f"  Wavefront size:      {wavefront_size}")

        # Estimate spill overhead
        if vgpr_spill_count > 0:
            print(f"\n⚠️  VGPR Spill Impact:")
            print(f"  {vgpr_spill_count} VGPRs spilled to private memory (slow!)")
            print(f"  Each spill = ~400-600 cycle latency vs ~4 cycles for VGPR access")
            print(f"  This can severely degrade performance!")

        if sgpr_spill_count > 0:
            print(f"\n⚠️  SGPR Spill Impact:")
            print(f"  {sgpr_spill_count} SGPRs spilled to memory")
            print(f"  Less critical than VGPR spills but still impacts performance")

    except Exception as e:
        print(f"Error extracting metadata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
