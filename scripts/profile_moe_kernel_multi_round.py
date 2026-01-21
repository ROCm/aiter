#!/usr/bin/env python3
"""
Multi-round MOE kernel profiling with rocprofv3

Splits counters into multiple rounds to avoid rocprofv3 limitations.
Runs profiling multiple times and merges results.
"""

import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import torch

# Add paths
current_dir = Path(__file__).parent
aiter_root = current_dir.parent
sys.path.insert(0, str(aiter_root))
sys.path.insert(0, str(aiter_root / "csrc/ck_gemm_moe_2stages_codegen"))

import aiter
from aiter import QuantType, ActivationType
import gemm_moe_tune
from gemm_moe_tune import FmoeTuner

# Split counters into manageable groups (8-10 per round)
COUNTER_GROUPS = {
    'round1_cache_memory': [
        "TCC_HIT", "TCC_MISS", "TCC_TAG_STALL_sum",
        "TCC_EA0_RDREQ_sum", 
        "TCP_PENDING_STALL_CYCLES_sum",
    ],
    
    'round2_compute': [
        "LDSBankConflict", "OccupancyPercent", "MemUnitStalled", "MfmaUtil",
        "SQ_INSTS_MFMA", "SQ_VALU_MFMA_BUSY_CYCLES",
        "MfmaFlops", "MfmaFlopsF16",
    ],
    
    'round3_waves': [
        "SQ_WAVES_sum",  # This is a derived expression that exists
        "SQ_WAVE_CYCLES",  # Raw counter (no _sum)
        "SQ_BUSY_CU_CYCLES",  # Raw counter (no _sum)
        "SQ_WAIT_INST_ANY",  # Raw counter
        "SQ_WAIT_ANY",  # Raw counter
        "GRBM_GUI_ACTIVE",  # Raw counter (no _sum)
        "GRBM_COUNT",  # Raw counter (no _sum)
    ],
    
    'round4_instructions': [
        "SQ_INSTS_VALU",  # Raw counter (no _sum)
        "SQ_INSTS_VMEM",  # Raw counter (no _sum)
        "SQ_INSTS_LDS",  # Raw counter (no _sum)
        "SQ_INSTS_SALU",  # Raw counter (no _sum)
        "SQ_INSTS_SMEM",  # Raw counter (no _sum)
        "SQ_INSTS",  # Total instructions
    ],
    
    'round5_active_lds': [
        "SQ_ACTIVE_INST_VALU",  # Raw counter (no _sum)
        "SQ_ACTIVE_INST_VMEM",  # Raw counter (no _sum)
        "SQ_ACTIVE_INST_LDS",  # Raw counter (no _sum)
        "SQ_ACTIVE_INST_ANY",  # Raw counter (no _sum)
        "SQ_LDS_BANK_CONFLICT",  # Raw counter (no _sum)
        "SQ_LDS_IDX_ACTIVE",  # Raw counter (no _sum)
    ],
    
    'round6_mfma_hw': [
        "MfmaFlopsBF16",  # Derived expression
        "MfmaFlopsF32",  # Derived expression
        "SQ_VALU_MFMA_BUSY_CYCLES",  # Raw counter (no _sum)
        "CU_NUM",  # Derived expression
        "SIMD_NUM",  # Derived expression
        "max_bandwidth",  # Constant
    ],

    'round7_fetch_only': [
        "FETCH_SIZE",
    ],
    
    'round8_wrreq_only': [
        "TCC_EA0_WRREQ_sum",
    ],
    
    'round9_write_size_only': [
        "WRITE_SIZE",
    ]
}

# Import script generation functions from the main profiling script
from profile_moe_kernel import (
    generate_asm_1stage_script,
    generate_ck_stage1_script,
    generate_ck_stage2_script,
    generate_asm_stage1_script,
    get_gpu_memory_info,
    clear_gpu_memory,
)


def create_kernel_execution_script(row, script_path):
    """Create execution script (no stage1 kernel lookup needed for multi-round)."""
    config_idx = int(row['config_idx'])
    kernel_name = row['kernel_name']
    kernel_type = str(row['kernel_type'])
    stage = str(row['stage'])
    
    token = int(row['token'])
    model_dim = int(row['model_dim'])
    inter_dim = int(row['inter_dim'])
    expert = int(row['expert'])
    topk = int(row['topk'])
    block_m = int(row['block_m'])
    
    dtype_str = str(row['dtype'])
    q_dtype_a_str = str(row['q_dtype_a'])
    q_dtype_w_str = str(row['q_dtype_w'])
    q_type_str = str(row['q_type'])
    act_type_str = str(row['act_type'])
    use_g1u1 = bool(row['use_g1u1'])
    doweight_stage1 = bool(row['doweight_stage1'])
    
    if stage == "asm_1stage":
        code = generate_asm_1stage_script(config_idx, kernel_name, token, model_dim, inter_dim,
                                         expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                                         q_type_str, act_type_str, use_g1u1)
    elif kernel_type == "ck" and stage == "stage2":
        code = generate_ck_stage2_script(config_idx, kernel_name, token, model_dim, inter_dim,
                                         expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                                         q_type_str, act_type_str, use_g1u1, doweight_stage1)
    elif kernel_type == "ck" and stage == "stage1":
        code = generate_ck_stage1_script(config_idx, kernel_name, token, model_dim, inter_dim,
                                         expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                                         q_type_str, act_type_str, use_g1u1, doweight_stage1)
    else:
        code = generate_asm_stage1_script(config_idx, kernel_name, stage, token, model_dim, inter_dim,
                                         expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                                         q_type_str, act_type_str, use_g1u1, doweight_stage1)
    
    script_path.write_text(code)


def run_rocprofv3_round(row, output_dir, round_name, counters):
    """Run rocprofv3 for a specific counter group."""
    config_idx = int(row['config_idx'])
    kernel_type = str(row['kernel_type'])
    stage = str(row['stage'])
    block_m = int(row['block_m'])
    kernel_name = row['kernel_name'][:]
    
    base_name = f"cfg{config_idx}_{kernel_type}_{stage}_block{block_m}_{kernel_name}_{round_name}"
    script_name = f"{base_name}.py"
    script_path = output_dir / script_name
    
    create_kernel_execution_script(row, script_path)
    
    output_file = output_dir / base_name
    counter_str = " ".join(counters)
    
    cmd = [
        "rocprofv3",
        "--pmc", counter_str,
        "--output-format", "csv",
        "-d", str(output_dir.absolute()),
        "-o", str(output_file.absolute()),
        "--",
        sys.executable, str(script_path.absolute())
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            script_path.unlink()  # Clean up script
            return base_name
        else:
            print(f"    ✗ Failed (exit {result.returncode})")
            return None
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        return None


def parse_rocprofv3_csv(config_name, output_dir, counters):
    """Parse rocprofv3 counter collection CSV."""
    counter_file = output_dir / f"{config_name}_counter_collection.csv"
    
    if not counter_file.exists():
        return {}
    
    try:
        df = pd.read_csv(counter_file)
        if len(df) == 0:
            return {}
        
        # Get last dispatch
        last_dispatch_id = df['Dispatch_Id'].max()
        target_rows = df[df['Dispatch_Id'] == last_dispatch_id]
        
        counter_data = {}
        for _, row in target_rows.iterrows():
            if row['Counter_Name'] in counters:
                counter_data[row['Counter_Name']] = row['Counter_Value']
        
        # Clean up CSV files
        for csv_file in output_dir.glob(f"{config_name}*.csv"):
            csv_file.unlink()
        
        return counter_data
    except Exception as e:
        print(f"    Error parsing: {e}")
        return {}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-round MOE kernel profiling')
    parser.add_argument('-i', '--input', required=True, help='Input CSV with kernel configs')
    parser.add_argument('-o', '--output-dir', default='profiling_multiround', help='Output directory')
    
    args = parser.parse_args()
    
    kernels = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print("MULTI-ROUND PROFILING WITH EXTENDED COUNTERS")
    print(f"{'='*80}")
    print(f"Kernels to profile: {len(kernels)}")
    print(f"Profiling rounds: {len(COUNTER_GROUPS)}")
    print(f"Total counters: {sum(len(c) for c in COUNTER_GROUPS.values())}")
    print(f"Output: {output_dir.absolute()}\n")
    
    all_results = []
    
    for kernel_idx, (_, row) in enumerate(kernels.iterrows(), 1):
        if str(row.get('error', '')).strip() == 'failed':
            print(f"[{kernel_idx}/{len(kernels)}] Skipping failed kernel")
            continue
        
        print(f"[{kernel_idx}/{len(kernels)}] Config {row['config_idx']} {row['stage']}: {row['kernel_name'][:40]}")
        
        merged_counters = {}
        
        # Run each counter group
        for round_idx, (round_name, counters) in enumerate(COUNTER_GROUPS.items(), 1):
            print(f"  Round {round_idx}/{len(COUNTER_GROUPS)} ({round_name}): {len(counters)} counters...")
            
            clear_gpu_memory()
            
            config_name = run_rocprofv3_round(row, output_dir, round_name, counters)
            if config_name:
                round_data = parse_rocprofv3_csv(config_name, output_dir, counters)
                merged_counters.update(round_data)
                print(f"    ✓ Collected {len(round_data)} counters")
            else:
                print(f"    ✗ Failed to collect counters")
        
        # Merge with kernel info
        result_row = dict(row)
        result_row.update(merged_counters)
        all_results.append(result_row)
        
        # Save incrementally
        output_file = output_dir / "kernels_with_counters.csv"
        pd.DataFrame([result_row]).to_csv(
            output_file,
            mode='a',
            header=(kernel_idx == 1),
            index=False
        )
        
        print(f"  → Collected {len(merged_counters)} total counters\n")
    
    print(f"{'='*80}")
    print(f"PROFILING COMPLETE")
    print(f"{'='*80}")
    print(f"Output: {output_dir / 'kernels_with_counters.csv'}")
    print(f"Profiled: {len(all_results)} kernels")
    print(f"Total counter columns: {len(merged_counters)}")
    print(f"\nNext step:")
    print(f"  python comprehensive_moe_analysis.py -i {output_dir / 'kernels_with_counters.csv'} -o analysis")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
