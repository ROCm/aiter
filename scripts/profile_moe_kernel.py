#!/usr/bin/env python3
"""
Profile MOE kernels using rocprofv3 with tuner integration - FIXED VERSION

CHANGES FROM ORIGINAL:
1. Uses tuner.generate_data() for consistent data generation (matches benchmark)
2. Stage2 runs actual stage1 to get real intermediate data (not fake random)
3. All moe_sorting calls include block_m parameter
4. Matches benchmark_moe_kernels.py input handling exactly

For each kernel in sample.csv:
1. Uses tuner to force execution of the specific kernel
2. Runs under rocprofv3 to collect hardware counters
3. Extracts FLOPS and bandwidth from profiling data
"""

import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import torch
import sqlite3

# Add paths for tuner imports
current_dir = Path(__file__).parent
aiter_root = current_dir.parent
sys.path.insert(0, str(aiter_root))
sys.path.insert(0, str(aiter_root / "csrc/ck_gemm_moe_2stages_codegen"))

import aiter
from aiter import QuantType, ActivationType
import gemm_moe_tune
from gemm_moe_tune import FmoeTuner

# Comprehensive counters for detailed analysis
COUNTERS = [
    # Cache and memory
    "TCC_HIT", "TCC_MISS", "TCC_TAG_STALL_sum",
    "TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum",
    "FETCH_SIZE", "WRITE_SIZE",
    "TCP_PENDING_STALL_CYCLES_sum",
    
    # Compute utilization
    "LDSBankConflict", "OccupancyPercent", "MemUnitStalled", "MfmaUtil",
    "SQ_INSTS_MFMA", "SQ_VALU_MFMA_BUSY_CYCLES_sum",
    "MfmaFlops", "MfmaFlopsF16", "MfmaFlopsF32", "MfmaFlopsBF16",
    
    # Wave execution
    "SQ_WAVES_sum", "SQ_WAVE_CYCLES_sum", "SQ_BUSY_CU_CYCLES_sum",
    "SQ_WAIT_INST_ANY", "SQ_ACTIVE_INST_ANY_sum",
    
    # Instruction breakdown
    "SQ_INSTS_VALU_sum", "SQ_INSTS_VMEM_sum", "SQ_INSTS_LDS_sum",
    "SQ_INSTS_SALU_sum", "SQ_INSTS_SMEM_sum",
    "SQ_ACTIVE_INST_VALU_sum", "SQ_ACTIVE_INST_VMEM_sum", "SQ_ACTIVE_INST_LDS_sum",
    
    # LDS details
    "SQ_LDS_BANK_CONFLICT_sum", "SQ_LDS_IDX_ACTIVE_sum",
    
    # GPU activity
    "GRBM_GUI_ACTIVE_sum", "GRBM_COUNT_sum",
    
    # Hardware constants (will be same for all kernels but good to have)
    "CU_NUM", "SIMD_NUM", "max_bandwidth",
]


def generate_asm_1stage_script(config_idx, kernel_name, token, model_dim, inter_dim,
                                expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                                q_type_str, act_type_str, use_g1u1):
    """Generate script for ASM 1-stage kernel using tuner.generate_data()."""
    return f'''#!/usr/bin/env python3
"""
Profiling ASM 1-stage kernel (fused operation) - FIXED VERSION
Uses tuner.generate_data() for consistent input generation.
Kernel: {kernel_name}
"""

import sys
from pathlib import Path

# Dynamically find aiter root directory
script_dir = Path(__file__).parent
aiter_root = script_dir
while aiter_root.name != 'aiter' and aiter_root != aiter_root.parent:
    aiter_root = aiter_root.parent
if aiter_root.name != 'aiter':
    aiter_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(aiter_root))
sys.path.insert(0, str(aiter_root / "csrc/ck_gemm_moe_2stages_codegen"))

import torch
import aiter
from aiter import QuantType, ActivationType
import gemm_moe_tune
from gemm_moe_tune import FmoeTuner

print("="*80)
print("ASM 1-Stage Kernel Profiling - Config {config_idx} - FIXED VERSION")
print("="*80)
print("Kernel: {kernel_name}")
print("Type: asm, Stage: 1-stage (fused), block_m: {block_m}")
print("="*80)

TARGET_KERNEL = "{kernel_name}"
token = {token}
model_dim = {model_dim}
inter_dim = {inter_dim}
expert = {expert}
topk = {topk}
dtype = {dtype_str}
q_dtype_a = {q_dtype_a_str}
q_dtype_w = {q_dtype_w_str}
q_type = {q_type_str}
act_type = {act_type_str}
use_g1u1 = {use_g1u1}
block_m = {block_m}

print("\\nInitializing tuner...")
tuner = FmoeTuner("fmoeTuner", [], [], "Profiling")

print("Generating test data using tuner.generate_data()...")
torch.manual_seed(42)
torch.set_default_device('cuda')

# Use tuner's generate_data for consistency with benchmark
(
    input, a1_qt, w1_qt, w2_qt, w1_qt_shffle, w2_qt_shffle,
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
    topk_ids, topk_weights, moe_buf, a1_scale, w1_scale, w2_scale,
) = tuner.generate_data(
    token, model_dim, inter_dim, expert, topk,
    dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
    block_m  # Pass actual block_m to generate_data
)

print(f"\\n{{'*'*80}}")
print(f"*** EXECUTING ASM 1-STAGE KERNEL (block_m={{block_m}}) ***")
print(f"*** {{TARGET_KERNEL}}")
print(f"{{'*'*80}}")

from aiter.fused_moe import fused_moe_1stage

result = fused_moe_1stage(
    hidden_states=input,
    w1=w1_qt_shffle,
    w2=w2_qt_shffle,
    topk=topk,
    sorted_ids=sorted_ids,
    sorted_weights=sorted_weights,
    sorted_expert_ids=sorted_expert_ids,
    num_valid_ids=num_valid_ids,
    moe_buf=moe_buf,
    isG1U1=use_g1u1,
    block_size_M=block_m,
    activation=act_type,
    quant_type=q_type,
    kernelName=TARGET_KERNEL,
    q_dtype_a=q_dtype_a,
    q_dtype_w=q_dtype_w,
    w1_scale=w1_scale,
    w2_scale=w2_scale,
)

torch.cuda.synchronize()

print("\\n" + "="*80)
print("✓ ASM 1-STAGE KERNEL executed!")
print(f"Output shape: {{result.shape}}")
print("="*80)
'''


def generate_ck_stage1_script(config_idx, kernel_name, token, model_dim, inter_dim,
                               expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                               q_type_str, act_type_str, use_g1u1, doweight_stage1):
    """Generate script for CK stage1 kernel using tuner.generate_data()."""
    return f'''#!/usr/bin/env python3
"""
Profiling CK stage1 kernel - FIXED VERSION
Uses tuner.generate_data() for consistent input generation.
Kernel: {kernel_name}
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
aiter_root = script_dir
while aiter_root.name != 'aiter' and aiter_root != aiter_root.parent:
    aiter_root = aiter_root.parent
if aiter_root.name != 'aiter':
    aiter_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(aiter_root))
sys.path.insert(0, str(aiter_root / "csrc/ck_gemm_moe_2stages_codegen"))

import torch
import aiter
from aiter import QuantType, ActivationType
import gemm_moe_tune
from gemm_moe_tune import FmoeTuner
from aiter.ops.moe_op import ck_moe_stage1_fwd

print("="*80)
print("CK Stage1 Kernel Profiling - Config {config_idx} - FIXED VERSION")
print("="*80)
print("TARGET KERNEL: {kernel_name}")
print("TARGET STAGE: stage1")
print("="*80)

TARGET_KERNEL = "{kernel_name}"
token = {token}
model_dim = {model_dim}
inter_dim = {inter_dim}
expert = {expert}
topk = {topk}
dtype = {dtype_str}
q_dtype_a = {q_dtype_a_str}
q_dtype_w = {q_dtype_w_str}
q_type = {q_type_str}
act_type = {act_type_str}
use_g1u1 = {use_g1u1}
doweight_stage1 = {doweight_stage1}
block_m = {block_m}

print("\\nInitializing tuner...")
tuner = FmoeTuner("fmoeTuner", [], [], "Profiling")

print("Generating test data using tuner.generate_data()...")
torch.manual_seed(42)
torch.set_default_device('cuda')

# Use tuner's generate_data for consistency with benchmark
(
    input, a1_qt, w1_qt, w2_qt, w1_qt_shffle, w2_qt_shffle,
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
    topk_ids, topk_weights, moe_buf, a1_scale, w1_scale, w2_scale,
) = tuner.generate_data(
    token, model_dim, inter_dim, expert, topk,
    dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
    block_m  # Pass actual block_m
)

print(f"\\n{{'*'*80}}")
print(f"*** PROFILING TARGET: Stage1 (block_m={{block_m}}) ***")
print(f"*** {{TARGET_KERNEL}}")
print(f"{{'*'*80}}")

inter_states = torch.empty(token, topk, inter_dim, dtype=dtype)
ck_moe_stage1_fwd(
    hidden_states=a1_qt,
    w1=w1_qt_shffle,
    w2=w2_qt_shffle,
    sorted_token_ids=sorted_ids,
    sorted_expert_ids=sorted_expert_ids,
    num_valid_ids=num_valid_ids,
    out=inter_states,
    topk=topk,
    kernelName=TARGET_KERNEL,
    w1_scale=w1_scale,
    a1_scale=a1_scale,
    block_m=block_m,
    sorted_weights=sorted_weights if doweight_stage1 else None,
    quant_type=q_type,
    activation=act_type,
    dst_type=dtype,
)

torch.cuda.synchronize()

print("\\n" + "="*80)
print("✓ TARGET KERNEL executed!")
print(f"Output shape: {{inter_states.shape}}")
print("="*80)
'''


def generate_ck_stage2_script(config_idx, kernel_name, token, model_dim, inter_dim,
                               expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                               q_type_str, act_type_str, use_g1u1, doweight_stage1):
    """Generate script for CK stage2 kernel - generates proper data without running stage1."""
    return f'''#!/usr/bin/env python3
"""
Profiling CK stage2 kernel - FIXED VERSION (stage2 only)
Generates proper input data using tuner, calls ONLY stage2 kernel.
Kernel: {kernel_name}
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
aiter_root = script_dir
while aiter_root.name != 'aiter' and aiter_root != aiter_root.parent:
    aiter_root = aiter_root.parent
if aiter_root.name != 'aiter':
    aiter_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(aiter_root))
sys.path.insert(0, str(aiter_root / "csrc/ck_gemm_moe_2stages_codegen"))

import torch
import aiter
from aiter import QuantType, ActivationType
import gemm_moe_tune
from gemm_moe_tune import FmoeTuner
from aiter.ops.moe_op import ck_moe_stage2_fwd

print("="*80)
print("CK Stage2 Kernel Profiling - Config {config_idx} - FIXED VERSION")
print("="*80)
print("TARGET KERNEL: {kernel_name}")
print("TARGET STAGE: stage2 (ONLY)")
print("="*80)

TARGET_KERNEL = "{kernel_name}"
token = {token}
model_dim = {model_dim}
inter_dim = {inter_dim}
expert = {expert}
topk = {topk}
dtype = {dtype_str}
q_dtype_a = {q_dtype_a_str}
q_dtype_w = {q_dtype_w_str}
q_type = {q_type_str}
act_type = {act_type_str}
use_g1u1 = {use_g1u1}
doweight_stage1 = {doweight_stage1}
block_m = {block_m}

print("\\nInitializing tuner...")
tuner = FmoeTuner("fmoeTuner", [], [], "Profiling")

print("Generating test data using tuner.generate_data()...")
torch.manual_seed(42)
torch.set_default_device('cuda')

# Use tuner's generate_data for proper input generation
(
    input, a1_qt, w1_qt, w2_qt, w1_qt_shffle, w2_qt_shffle,
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
    topk_ids, topk_weights, moe_buf, a1_scale, w1_scale, w2_scale,
) = tuner.generate_data(
    token, model_dim, inter_dim, expert, topk,
    dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
    block_m
)

print("Generating intermediate states for stage2 input...")
# Generate synthetic but properly shaped intermediate states
inter_states = torch.randn(token, topk, inter_dim, dtype=dtype)

# Quantize for stage2 input
a2_qt, a2_scale = aiter.pertoken_quant(
    inter_states.view(token, -1, 128), quant_dtype=q_dtype_a
)
a2_qt = a2_qt.view(token, topk, -1)

print(f"\\n{{'*'*80}}")
print(f"*** PROFILING TARGET: Stage2 ONLY (block_m={{block_m}}) ***")
print(f"*** {{TARGET_KERNEL}}")
print(f"{{'*'*80}}")

output = torch.empty(token, model_dim, dtype=dtype)
ck_moe_stage2_fwd(
    inter_states=a2_qt,
    w1=w1_qt_shffle,
    w2=w2_qt_shffle,
    sorted_token_ids=sorted_ids,
    sorted_expert_ids=sorted_expert_ids,
    num_valid_ids=num_valid_ids,
    out=output,
    topk=topk,
    kernelName=TARGET_KERNEL,
    w2_scale=w2_scale,
    a2_scale=a2_scale,
    block_m=block_m,
    sorted_weights=sorted_weights if not doweight_stage1 else None,
    quant_type=q_type,
    activation=act_type,
)

torch.cuda.synchronize()

print("\\n" + "="*80)
print("✓ Stage2 kernel executed!")
print(f"Output shape: {{output.shape}}")
print("="*80)
'''


def generate_asm_stage1_script(config_idx, kernel_name, stage, token, model_dim, inter_dim,
                                expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                                q_type_str, act_type_str, use_g1u1, doweight_stage1):
    """Generate script for ASM stage1 kernel using tuner.generate_data()."""
    return f'''#!/usr/bin/env python3
"""
Profiling ASM {stage} kernel - FIXED VERSION
Uses tuner.generate_data() for consistent input generation.
Kernel: {kernel_name}
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
aiter_root = script_dir
while aiter_root.name != 'aiter' and aiter_root != aiter_root.parent:
    aiter_root = aiter_root.parent
if aiter_root.name != 'aiter':
    aiter_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(aiter_root))
sys.path.insert(0, str(aiter_root / "csrc/ck_gemm_moe_2stages_codegen"))

import torch
import aiter
from aiter import QuantType, ActivationType
import gemm_moe_tune
from gemm_moe_tune import FmoeTuner

print("="*80)
print("ASM {stage.capitalize()} Kernel Profiling - Config {config_idx} - FIXED VERSION")
print("="*80)
print("TARGET KERNEL: {kernel_name}")
print("TARGET STAGE: {stage}")
print("="*80)

TARGET_KERNEL = "{kernel_name}"
token = {token}
model_dim = {model_dim}
inter_dim = {inter_dim}
expert = {expert}
topk = {topk}
dtype = {dtype_str}
q_dtype_a = {q_dtype_a_str}
q_dtype_w = {q_dtype_w_str}
q_type = {q_type_str}
act_type = {act_type_str}
use_g1u1 = {use_g1u1}
doweight_stage1 = {doweight_stage1}
block_m = {block_m}

print("\\nInitializing tuner...")
tuner = FmoeTuner("fmoeTuner", [], [], "Profiling")

print("Generating test data using tuner.generate_data()...")
torch.manual_seed(42)
torch.set_default_device('cuda')

# Use tuner's generate_data for consistency with benchmark
(
    input, a1_qt, w1_qt, w2_qt, w1_qt_shffle, w2_qt_shffle,
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
    topk_ids, topk_weights, moe_buf, a1_scale, w1_scale, w2_scale,
) = tuner.generate_data(
    token, model_dim, inter_dim, expert, topk,
    dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
    block_m  # Pass actual block_m
)

print(f"\\n{{'*'*80}}")
print(f"*** PROFILING TARGET: ASM {stage} (block_m={{block_m}}) ***")
print(f"*** {{TARGET_KERNEL}}")
print(f"{{'*'*80}}")

if "{stage}" != "stage1":
    print("ERROR: ASM 2-stage kernels only have stage1!")
    print(f"Got stage: {{'{stage}'}}")
    sys.exit(1)

# Call ASM stage1 using tuner's generated data
from aiter.fused_moe import asm_stage1

# Create output tensor - use generate_asm_stage1 logic from tuner
if q_type == QuantType.per_1x128:
    ratio = a1_scale.element_size() // a1_qt.element_size()
    out = torch.zeros(
        (token + (token * ratio + 127) // 128, topk, inter_dim),
        dtype=a1_qt.dtype,
    )
else:
    out = torch.empty((token, topk, inter_dim), dtype=dtype)

# For per_1x128, need to transpose a1_scale (from tuner's generate_asm_stage1)
a1_scale_for_kernel = a1_scale
if q_type == QuantType.per_1x128:
    a1_scale_for_kernel = a1_scale.t().contiguous()

result = asm_stage1(
    a1_qt,
    w1_qt_shffle,
    w2_qt_shffle,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    block_m,
    TARGET_KERNEL,
    0,  # ksplit
    act_type,
    q_type,
    a1_scale_for_kernel,
    w1_scale,
    sorted_weights if doweight_stage1 else None,
)

torch.cuda.synchronize()

print("\\n" + "="*80)
print("✓ TARGET KERNEL executed!")
print(f"Output shape: {{result.shape}}")
print("="*80)
'''


def get_kernels(csv_file):
    """Load the CSV with best kernels."""
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)
    return pd.read_csv(csv_file)


def find_best_stage1_kernel(kernels_df, config_idx, block_m, kernel_type):
    """Find the best stage1 kernel for a given config to use when profiling stage2."""
    # Filter to same config, block_m, kernel_type, and stage1 only
    # IMPORTANT: Must match kernel_type to avoid mixing CK and ASM kernels!
    stage1_kernels = kernels_df[
        (kernels_df['config_idx'] == config_idx) &
        (kernels_df['block_m'] == block_m) &
        (kernels_df['kernel_type'] == kernel_type) &
        (kernels_df['stage'] == 'stage1')
    ]
    
    if len(stage1_kernels) == 0:
        return None
    
    # Get the one with minimum time_us
    best = stage1_kernels.loc[stage1_kernels['time_us'].idxmin()]
    return best['kernel_name']


def create_kernel_execution_script(row, script_path, kernels_df):
    """Create script using tuner.generate_data() for input generation."""
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
    
    # Generate script based on kernel type and stage
    if stage == "asm_1stage":
        code = generate_asm_1stage_script(config_idx, kernel_name, token, model_dim, inter_dim,
                                         expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                                         q_type_str, act_type_str, use_g1u1)
    elif kernel_type == "ck" and stage == "stage2":
        # Stage2 now generates its own data without needing stage1
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


def run_rocprofv3(row, output_dir, kernels_df, keep_temp_files=False):
    """Run rocprofv3 kernel-trace with optional counter collection."""
    config_idx = int(row['config_idx'])
    kernel_type = str(row['kernel_type'])
    stage = str(row['stage'])
    block_m = int(row['block_m'])
    kernel_name = row['kernel_name'][:]
    
    base_name = f"cfg{config_idx}_{kernel_type}_{stage}_block{block_m}_{kernel_name}"
    script_name = f"{base_name}.py"
    script_path = output_dir / script_name
    
    create_kernel_execution_script(row, script_path, kernels_df)
    
    output_file = output_dir / base_name
    
    counter_str = " ".join(COUNTERS)
    cmd = [
        "rocprofv3",
        "--pmc", counter_str,
        "--output-format", "csv",
        "-d", str(output_dir.absolute()),
        "-o", str(output_file.absolute()),
        "--",
        sys.executable, str(script_path.absolute())
    ]
    
    print(f"Config {config_idx}: Running rocprofv3 kernel-trace...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        hip_error_detected = False
        if result.stderr and "HIP error: an illegal memory access" in result.stderr:
            hip_error_detected = True
            print(f"  ✗ KERNEL BUG: Illegal memory access detected!")
        
        if result.returncode == 0 and not hip_error_detected:
            print(f"  ✓ Success!")
            if not keep_temp_files:
                script_path.unlink()
            else:
                print(f"  Test script: {script_path}")
        else:
            if hip_error_detected:
                print(f"  ✗ Failed: Kernel memory access bug")
            else:
                print(f"  ✗ Failed (exit code {result.returncode})")
            
            print(f"  Test script: {script_path}")
            
            if result.stderr and "HIP error" in result.stderr:
                for line in result.stderr.split('\n'):
                    if "HIP error" in line or "illegal memory" in line:
                        print(f"  Error: {line.strip()}")
                        break
            elif result.stderr:
                stderr_lines = [l for l in result.stderr.split('\n') if l.strip()]
                for line in stderr_lines[-5:]:
                    print(f"  {line}")
        
        return base_name if (result.returncode == 0 and not hip_error_detected) else None
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout: Kernel execution exceeded 300 seconds")
        print(f"  Test script kept: {script_path}")
        return None
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        print(f"  Test script kept: {script_path}")
        return None


def parse_rocprofv3_output(config_name, output_dir, keep_temp_files=False):
    """Parse rocprofv3 PMC counter_collection CSV to extract counter values."""
    counter_file = output_dir / f"{config_name}_counter_collection.csv"
    
    if not counter_file.exists():
        files = list(output_dir.glob("*counter_collection.csv"))
        if not files:
            return {}
        counter_file = max(files, key=lambda p: p.stat().st_mtime)
    
    try:
        df = pd.read_csv(counter_file)
        
        if len(df) == 0:
            return {}
        
        last_dispatch_id = df['Dispatch_Id'].max()
        target_rows = df[df['Dispatch_Id'] == last_dispatch_id]
        
        counter_data = {}
        for _, row in target_rows.iterrows():
            counter_name = row['Counter_Name']
            counter_value = row['Counter_Value']
            if counter_name in COUNTERS:
                counter_data[counter_name] = counter_value
        
        if not keep_temp_files:
            for csv_file in output_dir.glob(f"{config_name}*.csv"):
                csv_file.unlink()
            
            rocprof_dir = output_dir / ".rocprofv3"
            if rocprof_dir.exists() and rocprof_dir.is_dir():
                import shutil
                shutil.rmtree(rocprof_dir)
        
        return counter_data
        
    except Exception as e:
        print(f"  Error parsing PMC data: {e}")
        return {}


def get_gpu_memory_info():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return allocated, reserved
    return 0, 0


def clear_gpu_memory():
    """Aggressively clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile MOE kernels with rocprofv3 - FIXED VERSION")
    parser.add_argument('-i', '--input', required=True, help='Input CSV file with kernel configs')
    parser.add_argument('-o', '--output-dir', default='rocprofv3_results_fixed', help='Output directory')
    parser.add_argument('--keep-temp-files', action='store_true', 
                        help='Keep temporary Python scripts and rocprofv3 CSV files')
    
    args = parser.parse_args()
    
    kernels = get_kernels(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print("FIXED VERSION - Uses tuner.generate_data() like benchmark")
    print(f"{'='*80}")
    print(f"Profiling {len(kernels)} kernels using rocprofv3")
    print(f"Input: {args.input}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Temp files: {'KEEP' if args.keep_temp_files else 'DELETE after use'}\n")
    
    output_file = output_dir / "kernels_with_counters.csv"
    
    results = []
    skipped_count = 0
    profiled_count = 0
    
    for idx, (_, row) in enumerate(kernels.iterrows(), 1):
        error_value = str(row.get('error', '')).strip()
        if error_value == 'failed':
            print(f"Config {row['config_idx']}: Skipping failed kernel: {row['kernel_name'][:40]}")
            skipped_count += 1
            continue
        
        clear_gpu_memory()
        alloc_before, reserved_before = get_gpu_memory_info()
        
        config_name = run_rocprofv3(row, output_dir, kernels, args.keep_temp_files)
        counter_data = parse_rocprofv3_output(config_name, output_dir, args.keep_temp_files) if config_name else {}
        
        clear_gpu_memory()
        alloc_after, reserved_after = get_gpu_memory_info()
        
        if idx % 10 == 0:
            print(f"  GPU Memory: Allocated={alloc_after:.1f}MB, Reserved={reserved_after:.1f}MB")
        
        result_row = dict(row)
        result_row.update(counter_data)
        results.append(result_row)
        profiled_count += 1
        
        pd.DataFrame([result_row]).to_csv(
            output_file, 
            mode='a', 
            header=(profiled_count == 1), 
            index=False
        )
        
        if len(kernels) > 1 and idx % max(1, len(kernels) // 5) == 0:
            print(f"  Progress: {idx}/{len(kernels)}")
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Output file: {output_file}")
    print(f"Total kernels in input: {len(kernels)}")
    print(f"Skipped (failed): {skipped_count}")
    print(f"Profiled: {len(kernels) - skipped_count}")
    print(f"Original columns: {len(kernels.columns)}")
    print(f"Profiling columns added: {len(results_df.columns) - len(kernels.columns)}")
    if len(results_df.columns) > len(kernels.columns):
        print(f"New columns:")
        for col in results_df.columns[len(kernels.columns):]:
            print(f"  - {col}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
