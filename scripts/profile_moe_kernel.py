#!/usr/bin/env python3
"""
Profile MOE kernels using rocprofv3 with tuner integration - MULTI-ROUND VERSION

Multi-round profiling mode:
- Splits counters into groups to avoid rocprofv3 counter limits
- Runs profiling multiple times with different counter sets
- Merges results from all rounds

CHANGES FROM ORIGINAL:
1. Uses tuner.generate_data() for consistent data generation (matches benchmark)
2. Stage2 generates proper input data
3. All moe_sorting calls include block_m parameter
4. Matches benchmark_moe_kernels.py input handling exactly
5. Multi-round profiling to collect comprehensive counter data

For each kernel in sample.csv:
1. Uses tuner to force execution of the specific kernel
2. Runs under rocprofv3 multiple times with different counter groups
3. Merges counter data from all profiling rounds
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import torch

# Add paths for tuner imports
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
    'r1': [
        "FETCH_SIZE",  # Memory read size
    ],

    'r2': [
        "WRITE_SIZE",  # Memory write size
    ],

    'r3': [
        "TCC_HIT",  # L2 cache hits
        "TCC_MISS",  # L2 cache misses
        "TCC_TAG_STALL_sum",  # TCC tag stalls
        "TCP_PENDING_STALL_CYCLES_sum",  # TCP pending stalls
        "SQ_VALU_MFMA_BUSY_CYCLES",  # Raw counter (no _sum)
        "SQ_INSTS_VALU_MFMA_MOPS_F8",
        "SQ_INSTS_VALU_MFMA_MOPS_I8",
        "SQ_INSTS_VALU_MFMA_MOPS_BF16",
        "MfmaUtil",  # MFMA utilization percentage
        "SQ_BUSY_CU_CYCLES",  # Compute unit busy cycles
    ],

    'r4': [
        "SQ_WAIT_ANY",  # Total stall cycles
        "MemUnitStalled",  # Memory unit stalls
        "SQ_WAVES_sum",  # Total wavefronts
        "SQ_WAVE_CYCLES",  # Wavefront execution cycles
        "OccupancyPercent",  # Wave occupancy percentage
        "GRBM_GUI_ACTIVE",  # GPU active cycles
        "SQ_INSTS_VALU",  # VALU instructions executed
        "SQ_INSTS_MFMA",  # MFMA instructions executed
        "SQ_INSTS_LDS",  # LDS instructions
        "SQ_LDS_BANK_CONFLICT",  # LDS bank conflicts
        "LDSBankConflict",
        "SQ_LDS_IDX_ACTIVE",  # Active LDS indices
    ],

}


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


def create_kernel_execution_script(row, script_path):
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


def run_rocprofv3_round(row, script_path, output_dir, round_name, counters, keep_temp_files):
    """Run rocprofv3 for a specific counter group using existing script."""
    config_idx = int(row['config_idx'])
    kernel_type = str(row['kernel_type'])
    stage = str(row['stage'])
    block_m = int(row['block_m'])
    kernel_name = row['kernel_name']
    
    base_name = f"cfg{config_idx}_{kernel_type}_{stage}_block{block_m}_{kernel_name}_{round_name}"
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
        
        hip_error_detected = False
        if result.stderr and "HIP error: an illegal memory access" in result.stderr:
            hip_error_detected = True
            print(f"    ✗ KERNEL BUG: Illegal memory access detected!")
        
        if result.returncode == 0 and not hip_error_detected:
            return base_name
        else:
            if hip_error_detected:
                print(f"    ✗ Failed: Kernel memory access bug")
            else:
                print(f"    ✗ Failed (exit code {result.returncode})")
            
            if result.stderr and "HIP error" in result.stderr:
                for line in result.stderr.split('\n'):
                    if "HIP error" in line or "illegal memory" in line:
                        print(f"    Error: {line.strip()}")
                        break
            elif result.stderr:
                stderr_lines = [l for l in result.stderr.split('\n') if l.strip()]
                for line in stderr_lines[-5:]:
                    print(f"    {line}")
            
            return None
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout: Kernel execution exceeded 300 seconds")
        return None
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        return None


def parse_rocprofv3_output(config_name, output_dir, keep_temp_files, expected_counters):
    """Parse rocprofv3 PMC counter_collection CSV to extract counter values."""
    counter_file = output_dir / f"{config_name}_counter_collection.csv"
    
    if not counter_file.exists():
        return {}
    
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
            if counter_name in expected_counters:
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
        print(f"    Error parsing PMC data: {e}")
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


def run_multi_round_profiling(row, output_dir, keep_temp_files):
    """Run profiling in multiple rounds with different counter groups, reusing script."""
    config_idx = int(row['config_idx'])
    kernel_type = str(row['kernel_type'])
    stage = str(row['stage'])
    block_m = int(row['block_m'])
    kernel_name = row['kernel_name']
    
    print(f"Config {config_idx} {kernel_type} {stage}: {kernel_name[:40]}")
    
    # Create script once (will be reused across all rounds)
    script_name = f"cfg{config_idx}_{kernel_type}_{stage}_block{block_m}_{kernel_name}.py"
    script_path = output_dir / script_name
    
    print(f"  Creating execution script: {script_name}")
    create_kernel_execution_script(row, script_path)
    
    merged_counters = {}
    
    # Run each counter group, reusing the same script
    for round_idx, (round_name, counters) in enumerate(COUNTER_GROUPS.items(), 1):
        print(f"  Round {round_idx}/{len(COUNTER_GROUPS)} ({round_name}): {len(counters)} counters...")
        
        clear_gpu_memory()
        
        config_name = run_rocprofv3_round(row, script_path, output_dir, round_name, counters, keep_temp_files)
        
        if config_name:
            round_data = parse_rocprofv3_output(config_name, output_dir, keep_temp_files, counters)
            merged_counters.update(round_data)
            print(f"    ✓ Collected {len(round_data)} counters")
        else:
            print(f"    ✗ Failed to collect counters")
    
    # Clean up script after all rounds (unless keep_temp_files is True)
    if not keep_temp_files:
        script_path.unlink()
    else:
        print(f"  Kept execution script: {script_path}")
    
    # Calculate MFMA_FLOPS if we have the necessary counters
    mfma_f8 = merged_counters.get('SQ_INSTS_VALU_MFMA_MOPS_F8', 0)
    mfma_i8 = merged_counters.get('SQ_INSTS_VALU_MFMA_MOPS_I8', 0)
    mfma_bf16 = merged_counters.get('SQ_INSTS_VALU_MFMA_MOPS_BF16', 0)
    if mfma_f8 or mfma_i8 or mfma_bf16:
        merged_counters['MFMA_FLOPS'] = (mfma_f8 + mfma_i8 + mfma_bf16) * 512
    
    print(f"  → Collected {len(merged_counters)} total counters\n")
    
    return merged_counters


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile MOE kernels with rocprofv3 - Multi-round mode")
    parser.add_argument('-i', '--input', required=True, help='Input CSV file with kernel configs')
    parser.add_argument('-o', '--output-dir', default='profiling_multiround', help='Output directory')
    parser.add_argument('--keep-temp-files', action='store_true', 
                        help='Keep temporary Python scripts and rocprofv3 CSV files')
    
    args = parser.parse_args()
    
    kernels = get_kernels(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    num_kernels = len(kernels)
    num_counters = sum(len(c) for c in COUNTER_GROUPS.values())
    original_col_count = len(kernels.columns)
    
    print(f"\n{'='*80}")
    print("MULTI-ROUND PROFILING MODE")
    print(f"{'='*80}")
    print(f"Profiling rounds: {len(COUNTER_GROUPS)}")
    print(f"Total counters: {num_counters}")
    print(f"Kernels to profile: {num_kernels}")
    print(f"Input: {args.input}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Temp files: {'KEEP' if args.keep_temp_files else 'DELETE after use'}")
    print(f"{'='*80}\n")
    
    output_file = output_dir / "kernels_with_counters.csv"
    
    skipped_count = 0
    profiled_count = 0
    
    for idx, (_, row) in enumerate(kernels.iterrows(), 1):
        error_value = str(row.get('error', '')).strip()
        if error_value == 'failed':
            print(f"[{idx}/{num_kernels}] Skipping failed kernel: {row['kernel_name'][:40]}")
            skipped_count += 1
            continue
        
        clear_gpu_memory()
        
        print(f"[{idx}/{num_kernels}]", end=" ")
        counter_data = run_multi_round_profiling(row, output_dir, args.keep_temp_files)
        
        if idx % 10 == 0:
            clear_gpu_memory()
            alloc, reserved = get_gpu_memory_info()
            print(f"  GPU Memory: Allocated={alloc:.1f}MB, Reserved={reserved:.1f}MB")
        
        result_row = dict(row)
        result_row.update(counter_data)
        profiled_count += 1
        
        pd.DataFrame([result_row]).to_csv(
            output_file, 
            mode='a', 
            header=(profiled_count == 1), 
            index=False
        )
    
    # Read final results for summary
    if profiled_count > 0:
        results_df = pd.read_csv(output_file)
        new_col_count = len(results_df.columns)
    else:
        new_col_count = original_col_count
    
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Output file: {output_file}")
    print(f"Total kernels in input: {num_kernels}")
    print(f"Skipped (failed): {skipped_count}")
    print(f"Profiled: {profiled_count}")
    print(f"Original columns: {original_col_count}")
    print(f"Profiling columns added: {new_col_count - original_col_count}")
    if profiled_count > 0 and new_col_count > original_col_count:
        print(f"New columns:")
        for col in results_df.columns[original_col_count:]:
            print(f"  - {col}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
