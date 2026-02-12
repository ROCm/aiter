#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
MOE (Mixture of Experts) kernel script generator.

Generates profiling scripts for various MOE kernel types:
- ASM 1-stage (fused)
- ASM 2-stage (stage1)
- CK stage1
- CK stage2
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

from .base import ScriptGenerator, ScriptConfig


@dataclass(frozen=True)
class MoeKernelConfig(ScriptConfig):
    """
    Configuration for MOE kernel profiling.
    
    Immutable dataclass parsed from CSV row containing all parameters
    needed to generate a profiling script.
    """
    config_idx: int
    kernel_name: str
    kernel_type: str  # 'asm', 'ck'
    stage: str        # 'stage1', 'stage2', 'asm_1stage'
    
    # Problem dimensions
    token: int
    model_dim: int
    inter_dim: int
    expert: int
    topk: int
    block_m: int
    
    # Data types (as strings for code generation)
    dtype: str
    q_dtype_a: str
    q_dtype_w: str
    q_type: str
    act_type: str
    
    # Flags
    use_g1u1: bool
    doweight_stage1: bool
    
    # Benchmark results from input (optional)
    quant_time_us: float = 0.0
    error: str = ""
    
    @property
    def unique_id(self) -> str:
        """Unique identifier for this kernel configuration."""
        return f"cfg{self.config_idx}_{self.kernel_type}_{self.stage}_block{self.block_m}"
    
    @property
    def script_name(self) -> str:
        """Filename for generated script."""
        # Truncate kernel name if too long
        kname = self.kernel_name[:50] if len(self.kernel_name) > 50 else self.kernel_name
        return f"{self.unique_id}_{kname}.py"
    
    @property
    def combined_kernel_name(self) -> str:
        """Name used in combined result files."""
        return f"cfg{self.config_idx}_{self.kernel_name}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'config_idx': self.config_idx,
            'kernel_name': self.kernel_name,
            'kernel_type': self.kernel_type,
            'stage': self.stage,
            'token': self.token,
            'model_dim': self.model_dim,
            'inter_dim': self.inter_dim,
            'expert': self.expert,
            'topk': self.topk,
            'block_m': self.block_m,
            'dtype': self.dtype,
            'q_dtype_a': self.q_dtype_a,
            'q_dtype_w': self.q_dtype_w,
            'q_type': self.q_type,
            'act_type': self.act_type,
            'use_g1u1': self.use_g1u1,
            'doweight_stage1': self.doweight_stage1,
            'quant_time_us': self.quant_time_us,
            'error': self.error,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'MoeKernelConfig':
        """Create from dictionary."""
        return cls(**d)
    
    @classmethod
    def from_csv_row(cls, row: pd.Series) -> 'MoeKernelConfig':
        """Create from pandas Series (CSV row)."""
        return cls(
            config_idx=int(row['config_idx']),
            kernel_name=str(row['kernel_name']),
            kernel_type=str(row['kernel_type']),
            stage=str(row['stage']),
            token=int(row['token']),
            model_dim=int(row['model_dim']),
            inter_dim=int(row['inter_dim']),
            expert=int(row['expert']),
            topk=int(row['topk']),
            block_m=int(row['block_m']),
            dtype=str(row['dtype']),
            q_dtype_a=str(row['q_dtype_a']),
            q_dtype_w=str(row['q_dtype_w']),
            q_type=str(row['q_type']),
            act_type=str(row['act_type']),
            use_g1u1=bool(row['use_g1u1']),
            doweight_stage1=bool(row['doweight_stage1']),
            quant_time_us=float(row.get('quant_time_us', 0.0)),
            error=str(row.get('error', '')),
        )
    
    def __str__(self) -> str:
        return f"Config {self.config_idx} {self.kernel_type} {self.stage}: {self.kernel_name[:40]}"


class MoeScriptGenerator(ScriptGenerator):
    """
    Generator for MOE kernel profiling scripts.
    
    Creates Python scripts that set up test data using the tuner
    and execute specific MOE kernels.
    """
    
    def generate(self, config: MoeKernelConfig) -> str:
        """Generate script for the given MOE kernel configuration."""
        if config.kernel_type == "triton" and config.stage == "triton_1stage":
            return self._generate_triton_1stage(config)
        elif config.stage == "asm_1stage":
            return self._generate_asm_1stage(config)
        elif config.kernel_type == "ck" and config.stage == "stage2":
            return self._generate_ck_stage2(config)
        elif config.kernel_type == "ck" and config.stage == "stage1":
            return self._generate_ck_stage1(config)
        else:
            return self._generate_asm_stage1(config)
    
    def write_script(self, config: MoeKernelConfig, output_dir: Path) -> Path:
        """
        Generate and write script to file.
        
        Args:
            config: Kernel configuration
            output_dir: Directory to write the script
            
        Returns:
            Path to the written script file
        """
        content = self.generate(config)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        script_path = output_dir / config.script_name
        script_path.write_text(content)
        script_path.chmod(0o755)  # Make executable
        
        return script_path
    
    @classmethod
    def load_configs(
        cls,
        csv_file: Path,
        skip_failed: bool = True,
        resume_from: Optional[Path] = None,
    ) -> List[MoeKernelConfig]:
        """
        Load MOE kernel configurations from CSV file.
        
        Args:
            csv_file: Path to CSV file
            skip_failed: Skip rows where error column is 'failed'
            resume_from: If provided, skip already-profiled kernels
            
        Returns:
            List of MoeKernelConfig objects
        """
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Get already-profiled kernels for resume mode
        profiled_ids: Set[str] = set()
        if resume_from:
            resume_dir = Path(resume_from)
            # Check for existing output directories
            for subdir in resume_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith('cfg'):
                    counters = subdir / 'counters.csv'
                    trace = subdir / 'trace_kernel_trace.csv'
                    if counters.exists() and trace.exists():
                        profiled_ids.add(subdir.name)
        
        configs = []
        for _, row in df.iterrows():
            # Skip failed kernels
            if skip_failed:
                error_value = str(row.get('error', '')).strip()
                if error_value == 'failed':
                    continue
            
            config = MoeKernelConfig.from_csv_row(row)
            
            # Skip already-profiled in resume mode
            if resume_from and config.unique_id in profiled_ids:
                continue
            
            configs.append(config)
        
        return configs
    
    def _get_common_setup(self) -> str:
        """Return common setup code for all scripts."""
        return '''#!/usr/bin/env python3
"""
Auto-generated MOE kernel profiling script.
Uses tuner.generate_data() for consistent input generation.
"""

import sys
from pathlib import Path

# Dynamically find aiter root directory
script_dir = Path(__file__).parent
aiter_root = script_dir
while aiter_root.name != 'aiter' and aiter_root != aiter_root.parent:
    aiter_root = aiter_root.parent
if aiter_root.name != 'aiter':
    # Try going up from op_tests
    for parent in Path(__file__).resolve().parents:
        if parent.name == 'aiter' and (parent / 'aiter').exists():
            aiter_root = parent
            break
    else:
        aiter_root = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(aiter_root))
sys.path.insert(0, str(aiter_root / "csrc/ck_gemm_moe_2stages_codegen"))

import torch
import aiter
from aiter import QuantType, ActivationType
import gemm_moe_tune
from gemm_moe_tune import FmoeTuner
'''
    
    def _get_data_generation_code(self, config: MoeKernelConfig) -> str:
        """Return code for generating test data."""
        return f'''
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
    block_m
)
'''

    def _generate_asm_1stage(self, config: MoeKernelConfig) -> str:
        """Generate script for ASM 1-stage kernel (fused operation)."""
        return f'''{self._get_common_setup()}

print("="*80)
print("ASM 1-Stage Kernel Profiling - Config {config.config_idx}")
print("="*80)
print("Kernel: {config.kernel_name}")
print("Type: asm, Stage: 1-stage (fused), block_m: {config.block_m}")
print("="*80)

TARGET_KERNEL = "{config.kernel_name}"
token = {config.token}
model_dim = {config.model_dim}
inter_dim = {config.inter_dim}
expert = {config.expert}
topk = {config.topk}
dtype = {config.dtype}
q_dtype_a = {config.q_dtype_a}
q_dtype_w = {config.q_dtype_w}
q_type = {config.q_type}
act_type = {config.act_type}
use_g1u1 = {config.use_g1u1}
doweight_stage1 = {config.doweight_stage1}
block_m = {config.block_m}
{self._get_data_generation_code(config)}

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
    doweight_stage1=doweight_stage1,
    M=token,
    device=input.device,
)

torch.cuda.synchronize()

print("\\n" + "="*80)
print("ASM 1-STAGE KERNEL executed!")
print(f"Output shape: {{result.shape}}")
print("="*80)
'''

    def _generate_ck_stage1(self, config: MoeKernelConfig) -> str:
        """Generate script for CK stage1 kernel."""
        return f'''{self._get_common_setup()}
from aiter.ops.moe_op import ck_moe_stage1_fwd

print("="*80)
print("CK Stage1 Kernel Profiling - Config {config.config_idx}")
print("="*80)
print("TARGET KERNEL: {config.kernel_name}")
print("TARGET STAGE: stage1")
print("="*80)

TARGET_KERNEL = "{config.kernel_name}"
token = {config.token}
model_dim = {config.model_dim}
inter_dim = {config.inter_dim}
expert = {config.expert}
topk = {config.topk}
dtype = {config.dtype}
q_dtype_a = {config.q_dtype_a}
q_dtype_w = {config.q_dtype_w}
q_type = {config.q_type}
act_type = {config.act_type}
use_g1u1 = {config.use_g1u1}
doweight_stage1 = {config.doweight_stage1}
block_m = {config.block_m}
{self._get_data_generation_code(config)}

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
print("CK Stage1 kernel executed!")
print(f"Output shape: {{inter_states.shape}}")
print("="*80)
'''

    def _generate_ck_stage2(self, config: MoeKernelConfig) -> str:
        """Generate script for CK stage2 kernel."""
        return f'''{self._get_common_setup()}
from aiter.ops.moe_op import ck_moe_stage2_fwd

print("="*80)
print("CK Stage2 Kernel Profiling - Config {config.config_idx}")
print("="*80)
print("TARGET KERNEL: {config.kernel_name}")
print("TARGET STAGE: stage2 (ONLY)")
print("="*80)

TARGET_KERNEL = "{config.kernel_name}"
token = {config.token}
model_dim = {config.model_dim}
inter_dim = {config.inter_dim}
expert = {config.expert}
topk = {config.topk}
dtype = {config.dtype}
q_dtype_a = {config.q_dtype_a}
q_dtype_w = {config.q_dtype_w}
q_type = {config.q_type}
act_type = {config.act_type}
use_g1u1 = {config.use_g1u1}
doweight_stage1 = {config.doweight_stage1}
block_m = {config.block_m}
{self._get_data_generation_code(config)}

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
print("Stage2 kernel executed!")
print(f"Output shape: {{output.shape}}")
print("="*80)
'''

    def _generate_asm_stage1(self, config: MoeKernelConfig) -> str:
        """Generate script for ASM stage1 kernel."""
        return f'''{self._get_common_setup()}
from aiter.fused_moe import asm_stage1

print("="*80)
print("ASM {config.stage.capitalize()} Kernel Profiling - Config {config.config_idx}")
print("="*80)
print("TARGET KERNEL: {config.kernel_name}")
print("TARGET STAGE: {config.stage}")
print("="*80)

TARGET_KERNEL = "{config.kernel_name}"
token = {config.token}
model_dim = {config.model_dim}
inter_dim = {config.inter_dim}
expert = {config.expert}
topk = {config.topk}
dtype = {config.dtype}
q_dtype_a = {config.q_dtype_a}
q_dtype_w = {config.q_dtype_w}
q_type = {config.q_type}
act_type = {config.act_type}
use_g1u1 = {config.use_g1u1}
doweight_stage1 = {config.doweight_stage1}
block_m = {config.block_m}
{self._get_data_generation_code(config)}

print(f"\\n{{'*'*80}}")
print(f"*** PROFILING TARGET: ASM {config.stage} (block_m={{block_m}}) ***")
print(f"*** {{TARGET_KERNEL}}")
print(f"{{'*'*80}}")

if "{config.stage}" != "stage1":
    print("ERROR: ASM 2-stage kernels only have stage1!")
    print(f"Got stage: {{'{config.stage}'}}")
    sys.exit(1)

# Create output tensor
if q_type == QuantType.per_1x128:
    ratio = a1_scale.element_size() // a1_qt.element_size()
    out = torch.zeros(
        (token + (token * ratio + 127) // 128, topk, inter_dim),
        dtype=a1_qt.dtype,
    )
else:
    out = torch.empty((token, topk, inter_dim), dtype=dtype)

# For per_1x128, transpose a1_scale
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
print("ASM Stage1 kernel executed!")
print(f"Output shape: {{result.shape}}")
print("="*80)
'''

    def _generate_triton_1stage(self, config: MoeKernelConfig) -> str:
        """Generate script for Triton e2e kernel."""
        return f'''#!/usr/bin/env python3
"""
Auto-generated Triton e2e kernel profiling script.
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent
aiter_root = script_dir
while aiter_root.name != 'aiter' and aiter_root != aiter_root.parent:
    aiter_root = aiter_root.parent
if aiter_root.name != 'aiter':
    for parent in Path(__file__).resolve().parents:
        if parent.name == 'aiter' and (parent / 'aiter').exists():
            aiter_root = parent
            break
    else:
        aiter_root = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(aiter_root))
sys.path.insert(0, str(aiter_root / "csrc/ck_gemm_moe_2stages_codegen"))

import torch
from aiter import QuantType, ActivationType, dtypes
from gemm_moe_tune_triton import generate_data_triton_1stage, run_triton_1stage

print("="*80)
print("Triton E2E Kernel Profiling - Config {config.config_idx}")
print("="*80)
print("TARGET KERNEL: {config.kernel_name}")
print("="*80)

TARGET_KERNEL = "{config.kernel_name}"
token = {config.token}
model_dim = {config.model_dim}
inter_dim = {config.inter_dim}
expert = {config.expert}
topk = {config.topk}
dtype = {config.dtype}
block_m = {config.block_m}

print("\\nGenerating test data...")
torch.manual_seed(42)
torch.set_default_device('cuda')

persistent = "_persistent_" in TARGET_KERNEL and "non_persistent" not in TARGET_KERNEL

parts = TARGET_KERNEL.split('_')
config = {{}}

if persistent:
    for part in parts:
        if part.startswith('M') and '-' not in part:
            config['BLOCK_SIZE_M'] = int(part[1:])
        elif 'N1-' in part:
            config['BLOCK_SIZE_N1'] = int(part.split('-')[1])
        elif 'N2-' in part:
            config['BLOCK_SIZE_N2'] = int(part.split('-')[1])
        elif 'K' in part and '-' in part:
            k_val = int(part.split('-')[1])
            config['BLOCK_SIZE_K1'] = k_val
            config['BLOCK_SIZE_K2'] = k_val
else:
    for part in parts:
        if part.startswith('M') and '-' not in part:
            config['BLOCK_SIZE_M'] = int(part[1:])
        elif part.startswith('N') and part[1:].isdigit():
            config['BLOCK_SIZE_N'] = int(part[1:])
        elif 'K1-' in part:
            config['BLOCK_SIZE_K1'] = int(part.split('-')[1])
        elif 'K2-' in part:
            config['BLOCK_SIZE_K2'] = int(part.split('-')[1])
        elif part.startswith('GM'):
            config['GROUP_SIZE_M'] = int(part[2:])

(input, a1_qt, w1, w2, sorted_ids, topk_weights, expert_ids, num_post_pad, _, _, _, _, _, _, topk_ids) = generate_data_triton_1stage(token, model_dim, inter_dim, expert, topk, dtype, block_m)

print(f"\\n{{'*'*80}}")
print(f"*** EXECUTING TRITON E2E KERNEL (block_m={{block_m}}) ***")
print(f"*** {{TARGET_KERNEL}}")
print(f"{{'*'*80}}")

result = run_triton_1stage(input, a1_qt, w1, w2, sorted_ids, topk_weights, expert_ids, num_post_pad, None, None, None, topk_ids, topk, config, dtype, persistent)

torch.cuda.synchronize()

print("\\n" + "="*80)
print("TRITON E2E KERNEL executed!")
print(f"Output shape: {{result.shape}}")
print("="*80)
'''
