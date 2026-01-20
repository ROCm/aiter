#!/usr/bin/env python3
"""
Profile MOE kernels using rocprofv3 with tuner integration.

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

# Note: With new ROCm image, rocprofv3 --pmc now works without crashes!

# Essential counters (active)
COUNTERS = [
    # "MemUnitStalled", "FetchSize", "MfmaFlops", "MfmaFlopsF16", "MfmaFlopsF32", "MfmaFlopsF64", 
    "TCC_HIT", "TCC_MISS", "LDSBankConflict", "OccupancyPercent", "MemUnitStalled", "MfmaUtil", "SQ_WAIT_INST_ANY", "TCC_TAG_STALL_sum",
    
]

# Full comprehensive counter list (commented for reference)
# ALL_COUNTERS = [
#     "SQ_INSTS_MFMA", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_SMEM",
#     "SQ_INSTS_LDS", "SQ_WAVE_CYCLES", "SQ_ACTIVE_CYCLES", "SQ_WAVES_EXE",
#     "TCC_HIT_sum", "TCC_MISS_sum", "TCP_TCC_READ_REQ_sum", "TCP_TCC_WRITE_REQ_sum",
#     "TCC_PENDING_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_WRREQ_sum", "TCC_MC_RDREQ_sum", "TCC_MC_WRREQ_sum",
#     "TCC_EA0_RDREQ_32B_sum", "TCC_EA0_RDREQ_64B_sum", "TCC_EA0_RDREQ_96B_sum",
#     "TCC_EA0_WRREQ_32B_sum", "TCC_EA0_WRREQ_64B_sum", "TCC_EA0_WRREQ_96B_sum",
#     "SQ_LDS_BANK_CONFLICT", "SQ_LDS_ADDR_CONFLICT", "SQ_LDS_READ", "SQ_LDS_WRITE",
#     "SQ_LDS_READ_REQ", "SQ_WAIT_LGKM_CYCLES", "SQ_WAIT_MEM_CYCLES",
#     "SQ_WAIT_ANY_CYCLES", "SQ_WAIT_ANY", "SQ_WAIT_INST_ANY_CYCLES", "SQ_INSTS_EXECUTED",
#     "GRBM_COUNT", "SQ_IFETCH", "SQ_INSTS_SMEM_NORM", "SQ_WAIT_DEPENDENCY",
#     "GPU_UTIL", "GRBM_GUI_ACTIVE", "SQ_UTILIZATION", "SQ_UTILIZATION_BUSY", "GRBM_SPI_BUSY", "SQ_SCALARS_WRITTEN",
#     "SQ_INSTS_FP32", "SQ_INSTS_FP16", "SQ_INSTS_FP64", "SQ_INSTS_INT32",
#     "MfmaFlops", "MfmaFlopsF16", "MfmaFlopsF32", "MfmaFlopsF64",
#     "TCP_TCC_ATOMIC_REQ_sum", "TCC_ATOMIC_WITH_RET_sum", "TCC_ATOMIC_WO_RET_sum",
#     "TCC_HIT", "TCC_MISS", "TCC_NC_READ_sum",
# ]

def generate_asm_1stage_script(config_idx, kernel_name, token, model_dim, inter_dim,
                                expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                                q_type_str, act_type_str, use_g1u1):
    """Generate script for ASM 1-stage kernel (matching profile_asm_1stage.py template)."""
    return f'''#!/usr/bin/env python3
"""
Profiling ASM 1-stage kernel (fused operation).
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
    # Fallback: assume script is in aiter/scripts or subdirectory
    aiter_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(aiter_root))

import torch
import aiter
from aiter import QuantType, ActivationType
from aiter.ops.shuffle import shuffle_weight

print("="*80)
print("ASM 1-Stage Kernel Profiling - Config {config_idx}")
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

print("\\nGenerating test data...")
torch.manual_seed(42)
torch.set_default_device('cuda')

hidden_states = torch.randn(token, model_dim, dtype=dtype)
topk_weights = torch.randn(token, topk, dtype=torch.float32)
topk_ids = torch.randint(0, expert, (token, topk), dtype=torch.int32)

w1 = torch.randn(expert, inter_dim, model_dim, dtype=dtype)
w2 = torch.randn(expert, model_dim, inter_dim, dtype=dtype)

print("Quantizing...")
a1_qt, a1_scale = aiter.get_torch_quant(q_type)(hidden_states, quant_dtype=q_dtype_a)
w1_qt, w1_scale = aiter.pertoken_quant(w1.view(expert, -1, 128), quant_dtype=q_dtype_w)
w2_qt, w2_scale = aiter.pertoken_quant(w2.view(expert, -1, 128), quant_dtype=q_dtype_w)

w1_qt = w1_qt.view(expert, inter_dim, model_dim)
w2_qt = w2_qt.view(expert, model_dim, inter_dim)

w1_qt_shffle = shuffle_weight(w1_qt, (16, 16))
w2_qt_shffle = shuffle_weight(w2_qt, (16, 16))

print("Sorting...")
from aiter.fused_moe import moe_sorting
sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
    topk_ids, topk_weights, expert, model_dim, dtype
)

print(f"\\n{{'*'*80}}")
print(f"*** EXECUTING ASM 1-STAGE KERNEL (block_m={{block_m}}) ***")
print(f"*** {{TARGET_KERNEL}}")
print(f"{{'*'*80}}")

from aiter.fused_moe import fused_moe_1stage

result = fused_moe_1stage(
    hidden_states=hidden_states,
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


def generate_ck_stage2_script(config_idx, kernel_name, token, model_dim, inter_dim,
                               expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                               q_type_str, act_type_str, use_g1u1, doweight_stage1):
    """Generate script for CK stage2 kernel (matching profile_ck_stage2.py template)."""
    return f'''#!/usr/bin/env python3
"""
Profiling CK stage2 kernel.
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
    # Fallback: assume script is in aiter/scripts or subdirectory
    aiter_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(aiter_root))

import torch
import aiter
from aiter import QuantType, ActivationType
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.moe_op import ck_moe_stage2_fwd

print("="*80)
print("CK Stage2 Kernel Profiling - Config {config_idx}")
print("="*80)
print("TARGET KERNEL: {kernel_name}")
print("TARGET STAGE: stage2")
print("Config: token={token}, model_dim={model_dim}, inter_dim={inter_dim}, expert={expert}, topk={topk}")
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

print("\\nGenerating test data...")
torch.manual_seed(42)
torch.set_default_device('cuda')

hidden_states = torch.randn(token, model_dim, dtype=dtype)
topk_weights = torch.randn(token, topk, dtype=torch.float32)
topk_ids = torch.randint(0, expert, (token, topk), dtype=torch.int32)

w1 = torch.randn(expert, inter_dim, model_dim, dtype=dtype)
w2 = torch.randn(expert, model_dim, inter_dim, dtype=dtype)

print("Quantizing...")
a1_qt, a1_scale = aiter.get_torch_quant(q_type)(hidden_states, quant_dtype=q_dtype_a)
w1_qt, w1_scale = aiter.pertoken_quant(w1.view(expert, -1, 128), quant_dtype=q_dtype_w)
w2_qt, w2_scale = aiter.pertoken_quant(w2.view(expert, -1, 128), quant_dtype=q_dtype_w)

w1_qt = w1_qt.view(expert, inter_dim, model_dim)
w2_qt = w2_qt.view(expert, model_dim, inter_dim)

w1_qt_shffle = shuffle_weight(w1_qt, (16, 16))
w2_qt_shffle = shuffle_weight(w2_qt, (16, 16))

print("Sorting...")
from aiter.fused_moe import moe_sorting
sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
    topk_ids, topk_weights, expert, model_dim, dtype
)

print("SKIPPING Stage1 - Generating fake intermediate data...")
fake_inter_states = torch.randn(token, topk, inter_dim, dtype=dtype)
a2_qt, a2_scale = aiter.pertoken_quant(fake_inter_states.view(token, -1, 128), quant_dtype=q_dtype_a)
a2_qt = a2_qt.view(token, topk, -1)

print(f"\\n{{'*'*80}}")
print(f"*** PROFILING TARGET: Stage2 (block_m={{block_m}}) ***")
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
print("✓ TARGET KERNEL executed!")
print(f"Output shape: {{output.shape}}")
print("="*80)
'''


def generate_ck_stage1_script(config_idx, kernel_name, token, model_dim, inter_dim,
                               expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                               q_type_str, act_type_str, use_g1u1, doweight_stage1):
    """Generate script for CK stage1 kernel."""
    return f'''#!/usr/bin/env python3
"""
Profiling CK stage1 kernel.
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
    # Fallback: assume script is in aiter/scripts or subdirectory
    aiter_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(aiter_root))

import torch
import aiter
from aiter import QuantType, ActivationType
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.moe_op import ck_moe_stage1_fwd

print("="*80)
print("CK Stage1 Kernel Profiling - Config {config_idx}")
print("="*80)
print("TARGET KERNEL: {kernel_name}")
print("TARGET STAGE: stage1")
print("Config: token={token}, model_dim={model_dim}, inter_dim={inter_dim}, expert={expert}, topk={topk}")
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

print("\\nGenerating test data...")
torch.manual_seed(42)
torch.set_default_device('cuda')

hidden_states = torch.randn(token, model_dim, dtype=dtype)
topk_weights = torch.randn(token, topk, dtype=torch.float32)
topk_ids = torch.randint(0, expert, (token, topk), dtype=torch.int32)

w1 = torch.randn(expert, inter_dim, model_dim, dtype=dtype)
w2 = torch.randn(expert, model_dim, inter_dim, dtype=dtype)

print("Quantizing...")
a1_qt, a1_scale = aiter.get_torch_quant(q_type)(hidden_states, quant_dtype=q_dtype_a)
w1_qt, w1_scale = aiter.pertoken_quant(w1.view(expert, -1, 128), quant_dtype=q_dtype_w)
w2_qt, w2_scale = aiter.pertoken_quant(w2.view(expert, -1, 128), quant_dtype=q_dtype_w)

w1_qt = w1_qt.view(expert, inter_dim, model_dim)
w2_qt = w2_qt.view(expert, model_dim, inter_dim)

w1_qt_shffle = shuffle_weight(w1_qt, (16, 16))
w2_qt_shffle = shuffle_weight(w2_qt, (16, 16))

print("Sorting...")
from aiter.fused_moe import moe_sorting
sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
    topk_ids, topk_weights, expert, model_dim, dtype
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


def generate_asm_stage1_script(config_idx, kernel_name, stage, token, model_dim, inter_dim,
                                expert, topk, block_m, dtype_str, q_dtype_a_str, q_dtype_w_str,
                                q_type_str, act_type_str, use_g1u1, doweight_stage1):
    """Generate script for ASM stage1 kernel."""
    return f'''#!/usr/bin/env python3
"""
Profiling ASM {stage} kernel.
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
    # Fallback: assume script is in aiter/scripts or subdirectory
    aiter_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(aiter_root))

import torch
import aiter
from aiter import QuantType, ActivationType
from aiter.ops.shuffle import shuffle_weight

print("="*80)
print("ASM {stage.capitalize()} Kernel Profiling - Config {config_idx}")
print("="*80)
print("TARGET KERNEL: {kernel_name}")
print("TARGET STAGE: {stage}")
print("Config: token={token}, model_dim={model_dim}, inter_dim={inter_dim}, expert={expert}, topk={topk}")
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

print("\\nGenerating test data...")
torch.manual_seed(42)
torch.set_default_device('cuda')

hidden_states = torch.randn(token, model_dim, dtype=dtype)
topk_weights = torch.randn(token, topk, dtype=torch.float32)
topk_ids = torch.randint(0, expert, (token, topk), dtype=torch.int32)

# For g1u1, w1 is inter_dim*2 (gate+up), else just inter_dim (from tune.py's generate_data)
if use_g1u1:
    w1 = torch.randn(expert, inter_dim * 2, model_dim, dtype=dtype)
else:
    w1 = torch.randn(expert, inter_dim, model_dim, dtype=dtype)
w2 = torch.randn(expert, model_dim, inter_dim, dtype=dtype)

print("Quantizing...")
# For per_1x128, need special handling (from tune.py's generate_data)
if q_type == QuantType.per_1x128:
    a1_qt, a1_scale = aiter.pertoken_quant(
        hidden_states.view(token, -1, 128), quant_dtype=q_dtype_a
    )
    a1_qt = a1_qt.view(token, model_dim)
    a1_scale = a1_scale.squeeze(-1)
else:
    a1_qt, a1_scale = aiter.get_torch_quant(q_type)(hidden_states, quant_dtype=q_dtype_a)

w1_qt, w1_scale = aiter.pertoken_quant(w1.view(expert, -1, 128), quant_dtype=q_dtype_w)
w2_qt, w2_scale = aiter.pertoken_quant(w2.view(expert, -1, 128), quant_dtype=q_dtype_w)

w1_qt = w1_qt.view(w1.shape)
w2_qt = w2_qt.view(w2.shape)

w1_qt_shffle = shuffle_weight(w1_qt, (16, 16))
w2_qt_shffle = shuffle_weight(w2_qt, (16, 16))

print("Sorting...")
from aiter.fused_moe import moe_sorting
sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
    topk_ids, topk_weights, expert, model_dim, dtype, block_m
)

print(f"\\n{{'*'*80}}")
print(f"*** PROFILING TARGET: ASM {stage} (block_m={{block_m}}) ***")
print(f"*** {{TARGET_KERNEL}}")
print(f"{{'*'*80}}")

if "{stage}" != "stage1":
    print("ERROR: ASM 2-stage kernels only have stage1!")
    print(f"Got stage: {{'{stage}'}}")
    sys.exit(1)

# Call ASM stage1 using positional arguments (matching tune.py's run_asm_stage1)
from aiter.fused_moe import asm_stage1

# Create output tensor with special handling for per_1x128 (from tune.py's generate_asm_stage1)
if q_type == QuantType.per_1x128:
    ratio = a1_scale.element_size() // a1_qt.element_size()
    out = torch.zeros(
        (token + (token * ratio + 127) // 128, topk, inter_dim),
        dtype=a1_qt.dtype,  # Use quantized dtype, not output dtype!
    )
else:
    out = torch.empty((token, topk, inter_dim), dtype=dtype)

# For per_1x128, a1_scale needs to be transposed (from tune.py's generate_asm_stage1)
a1_scale_for_kernel = a1_scale
if q_type == QuantType.per_1x128:
    a1_scale_for_kernel = a1_scale.t().contiguous()

result = asm_stage1(
    a1_qt,                                          # input
    w1_qt_shffle,                                   # w1
    w2_qt_shffle,                                   # w2
    sorted_ids,                                     # sorted_ids
    sorted_expert_ids,                              # sorted_expert_ids
    num_valid_ids,                                  # num_valid_ids
    out,                                            # out
    topk,                                           # topk
    block_m,                                        # block_m
    TARGET_KERNEL,                                  # kernelName
    0,                                              # ksplit
    act_type,                                       # activation
    q_type,                                         # quant_type
    a1_scale_for_kernel,                            # a1_scale (transposed for per_1x128)
    w1_scale,                                       # w1_scale
    sorted_weights if doweight_stage1 else None,   # sorted_weights
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
    return pd.read_csv(csv_path)

def create_kernel_execution_script(row, script_path):
    """
    Create script matching your template format - directly calls kernel functions.
    """
    # Extract all config info
    config_idx = int(row['config_idx'])
    kernel_name = row['kernel_name']
    kernel_type = str(row['kernel_type'])  # asm or ck
    stage = str(row['stage'])  # stage1, stage2, or asm_1stage
    
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

def run_rocprofv3(row, output_dir, keep_temp_files=False):
    """Run rocprofv3 kernel-trace with optional counter collection."""
    config_idx = int(row['config_idx'])
    kernel_type = str(row['kernel_type'])
    stage = str(row['stage'])
    block_m = int(row['block_m'])
    kernel_name = row['kernel_name'][:]  # Truncate for filename
    
    # Create descriptive script name with config_idx prefix for uniqueness
    base_name = f"cfg{config_idx}_{kernel_type}_{stage}_block{block_m}_{kernel_name}"
    script_name = f"{base_name}.py"
    script_path = output_dir / script_name
    
    create_kernel_execution_script(row, script_path)
    
    # Use same base name for rocprofv3 output files
    output_file = output_dir / base_name
    
    # rocprofv3 is now in PATH with new ROCm image
    # Add --pmc with counters to collect performance metrics
    counter_str = " ".join(COUNTERS)
    cmd = [
        "rocprofv3",
        "--pmc", counter_str,
        "--output-format", "csv",  # Save as CSV, not database
        "-d", str(output_dir.absolute()),
        "-o", str(output_file.absolute()),
        "--",
        sys.executable, str(script_path.absolute())
    ]
    print(cmd)
    
    print(f"Config {config_idx}: Running rocprofv3 kernel-trace...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"  ✓ Success!")
            if not keep_temp_files:
                # Delete temp script after successful execution
                script_path.unlink()
            else:
                print(f"  Test script: {script_path}")
        else:
            print(f"  ✗ Failed (exit code {result.returncode})")
            print(f"  Test script: {script_path}")
            
            # Show full stdout and stderr for debugging
            if result.stdout:
                print(f"\n  --- STDOUT ---")
                for line in result.stdout.split('\n')[:20]:  # First 20 lines
                    if line.strip():
                        print(f"  {line}")
            
            if result.stderr:
                print(f"\n  --- STDERR (last 20 lines) ---")
                stderr_lines = result.stderr.split('\n')
                for line in stderr_lines[-20:]:  # Last 20 lines
                    if line.strip():
                        print(f"  {line}")
        
        return base_name if result.returncode == 0 else None
    except Exception as e:
        print(f"  Exception: {e}")
        print(f"  Test script kept: {script_path}")
        return None

def parse_rocprofv3_output(config_name, output_dir, keep_temp_files=False):
    """Parse rocprofv3 PMC counter_collection CSV to extract counter values."""
    # rocprofv3 creates cfg_N_counter_collection.csv with counter data
    counter_file = output_dir / f"{config_name}_counter_collection.csv"
    
    if not counter_file.exists():
        # Try glob pattern
        files = list(output_dir.glob("*counter_collection.csv"))
        if not files:
            return {}
        counter_file = max(files, key=lambda p: p.stat().st_mtime)
    
    try:
        df = pd.read_csv(counter_file)
        
        if len(df) == 0:
            return {}
        
        # Get last dispatch (target kernel - others are PyTorch overhead)
        last_dispatch_id = df['Dispatch_Id'].max()
        target_rows = df[df['Dispatch_Id'] == last_dispatch_id]
        
        # Extract counter values from Counter_Name and Counter_Value columns
        counter_data = {}
        for _, row in target_rows.iterrows():
            counter_name = row['Counter_Name']
            counter_value = row['Counter_Value']
            if counter_name in COUNTERS:
                counter_data[counter_name] = counter_value
        
        # Clean up rocprofv3 CSV files after parsing (unless keep_temp_files is set)
        if not keep_temp_files:
            # Delete all rocprofv3 output CSV files for this config
            for csv_file in output_dir.glob(f"{config_name}*.csv"):
                csv_file.unlink()
            
            # Also clean up .rocprofv3 directory if it exists
            rocprof_dir = output_dir / ".rocprofv3"
            if rocprof_dir.exists() and rocprof_dir.is_dir():
                import shutil
                shutil.rmtree(rocprof_dir)
        
        return counter_data
        
    except Exception as e:
        print(f"  Error parsing PMC data: {e}")
        return {}

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile MOE kernels with rocprofv3")
    parser.add_argument('-i', '--input', required=True, help='Input CSV file with kernel configs')
    parser.add_argument('-o', '--output-dir', default='rocprofv3_results', help='Output directory')
    parser.add_argument('--keep-temp-files', action='store_true', 
                        help='Keep temporary Python scripts and rocprofv3 CSV files (default: delete after use)')
    
    args = parser.parse_args()
    
    kernels = get_kernels(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nProfiling {len(kernels)} kernels using rocprofv3")
    print(f"Input: {args.input}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Temp files: {'KEEP' if args.keep_temp_files else 'DELETE after use'}\n")
    
    # Prepare output file and write header
    output_file = output_dir / "kernels_with_counters.csv"
    
    results = []
    skipped_count = 0
    profiled_count = 0
    for idx, (_, row) in enumerate(kernels.iterrows(), 1):
        # Check if kernel failed during benchmarking
        error_value = str(row.get('error', '')).strip()
        if error_value == 'failed':
            print(f"Config {row['config_idx']}: Skipping failed kernel: {row['kernel_name'][:40]}")
            skipped_count += 1
            # Don't write failed kernels to output CSV
            continue
        
        config_name = run_rocprofv3(row, output_dir, args.keep_temp_files)
        counter_data = parse_rocprofv3_output(config_name, output_dir, args.keep_temp_files) if config_name else {}
        
        result_row = dict(row)
        result_row.update(counter_data)
        results.append(result_row)
        profiled_count += 1
        
        # Write to CSV immediately after processing each kernel
        # Write header only for the first successfully profiled kernel
        pd.DataFrame([result_row]).to_csv(
            output_file, 
            mode='a', 
            header=(profiled_count == 1), 
            index=False
        )
        
        if len(kernels) > 1 and idx % max(1, len(kernels) // 5) == 0:
            print(f"  Progress: {idx}/{len(kernels)}")
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
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
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
