import logging
import os
from typing import Any
import torch
try:
    from diffusers import CogVideoXPipeline
    from diffusers.utils import export_to_video
except ImportError:
    raise ImportError(
        "diffusers library is not installed. Please install it using:\n"
        "pip install diffusers"
    )
from op_tests.sagev1_tests.core import sageattn
import torch.nn.functional as F
import argparse


class InputCaptureWrapper:
    """
    Wrapper that captures input tensors during inference for later benchmarking.
    Saves q, k, v tensors and kwargs to disk up to max_captures calls.
    Also tracks unique shapes to understand how inputs change across pipeline steps.
    """
    def __init__(self, fn, save_dir, name="attention", max_captures=None):
        self.fn = fn
        self.save_dir = save_dir
        self.name = name
        self.max_captures = max_captures  # None = unlimited
        self.captured_shapes = {}  # Maps shape_key -> list of call indices
        self.call_idx = 0
        self.saved_count = 0
        self.shape_history = []  # Track shape at each call for analysis
        os.makedirs(save_dir, exist_ok=True)
        
        if max_captures:
            logging.getLogger(__name__).info(f"Input capture limited to {max_captures} saves")
    
    def __call__(self, q, k, v, **kwargs):
        # Create a unique key based on shapes and dtype
        shape_key = (tuple(q.shape), tuple(k.shape), tuple(v.shape), str(q.dtype))
        
        # Track which calls have this shape (always track, even if not saving)
        if shape_key not in self.captured_shapes:
            self.captured_shapes[shape_key] = []
        self.captured_shapes[shape_key].append(self.call_idx)
        self.shape_history.append(shape_key)
        
        # Save tensors only if under the capture limit
        if self.max_captures is None or self.saved_count < self.max_captures:
            save_path = os.path.join(self.save_dir, f"{self.name}_input_{self.call_idx:06d}.pt")
            torch.save({
                'q': q.detach().cpu().clone(),
                'k': k.detach().cpu().clone(),
                'v': v.detach().cpu().clone(),
                'q_shape': list(q.shape),
                'k_shape': list(k.shape),
                'v_shape': list(v.shape),
                'dtype': str(q.dtype),
                'call_idx': self.call_idx,
                'kwargs': {key: val for key, val in kwargs.items() if not isinstance(val, torch.Tensor)},
            }, save_path)
            self.saved_count += 1
            
            if self.saved_count % 100 == 0:  # Log every 100 saves to avoid spam
                logging.getLogger(__name__).info(
                    f"Captured input {self.saved_count}/{self.max_captures or 'unlimited'}: "
                    f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
                )
            
            # Log when we hit the limit
            if self.max_captures and self.saved_count == self.max_captures:
                logging.getLogger(__name__).info(
                    f"Reached capture limit of {self.max_captures}. Continuing without saving."
                )
        
        self.call_idx += 1
        return self.fn(q, k, v, **kwargs)
    
    def summary(self):
        """Print summary of captured inputs."""
        logger = logging.getLogger(__name__)
        logger.info(f"=== Input Capture Summary for {self.name} ===")
        logger.info(f"Total attention calls: {self.call_idx}")
        logger.info(f"Inputs saved to disk: {self.saved_count}")
        if self.max_captures:
            logger.info(f"Capture limit: {self.max_captures}")
        logger.info(f"Unique shapes: {len(self.captured_shapes)}")
        logger.info(f"Saved to: {self.save_dir}")
        logger.info("")
        logger.info("Shape distribution:")
        for shape_key, call_indices in sorted(self.captured_shapes.items(), key=lambda x: -len(x[1])):
            q_shape, k_shape, v_shape, dtype = shape_key
            logger.info(f"  q={q_shape}, k={k_shape}, v={v_shape}, dtype={dtype}")
        logger.info("=" * 50)
        
        # Also save metadata summary for benchmark script
        metadata_path = os.path.join(self.save_dir, f"{self.name}_metadata.pt")
        torch.save({
            'total_calls': self.call_idx,
            'saved_count': self.saved_count,
            'max_captures': self.max_captures,
            'captured_shapes': {str(k): v for k, v in self.captured_shapes.items()},
            'shape_history': self.shape_history,
            'kernel_name': self.name,
        }, metadata_path)
        logger.info(f"Metadata saved to: {metadata_path}")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="THUDM/CogVideoX-2b", help='Model path')
parser.add_argument('--compile', action='store_true', help='Compile the model')
parser.add_argument('--attention_type', type=str, default='sdpa', choices=['sdpa', 'sagev1', 'fa2', 'fa3', 'fa3_fp8', 'sagev1_fa3'], help='Attention type')
parser.add_argument('--save_inputs', action='store_true', help='Save attention inputs for later benchmarking')
parser.add_argument('--input_dir', type=str, default='./captured_inputs', help='Directory to save captured inputs')
parser.add_argument('--max_captures', type=int, default=10, help='Maximum number of inputs to save (use 0 for unlimited)')
args = parser.parse_args()

# Global variable to hold the capture wrapper for summary reporting
_capture_wrapper = None

# Check if we're on AMD GPU - torch.compile with inductor backend has compatibility issues
# with AMD's triton-rocm due to missing NVIDIA-specific metadata (cluster_dims)
def _is_amd_gpu():
    """
    Detect if running on AMD GPU (ROCm).
    Most reliable check is torch.version.hip which is set on ROCm builds.
    """
    # Primary check: torch.version.hip is set on ROCm builds
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return True
    
    return False

# Log detection result for debugging
logger.debug(f"AMD GPU detected: {_is_amd_gpu()}")
logger.debug(f"torch.version.hip: {getattr(torch.version, 'hip', None)}")


if args.attention_type == 'sagev1':
    attn_fn = sageattn
    if args.save_inputs:
        # 0 means unlimited
        max_caps = args.max_captures if args.max_captures > 0 else None
        _capture_wrapper = InputCaptureWrapper(attn_fn, args.input_dir, name="sagev1", max_captures=max_caps)
        F.scaled_dot_product_attention = _capture_wrapper
    else:
        F.scaled_dot_product_attention = attn_fn
elif args.attention_type == 'sdpa':
    attn_fn = torch.nn.functional.scaled_dot_product_attention
    
    # Wrapper for PyTorch's native scaled_dot_product_attention
    # scaled_dot_product_attention: (batch, heads, seqlen, dim) - BHSD format
    # torch.nn.functional.scaled_dot_product_attention expects BHSD format
    def sdpa_wrapper(query, key, value, is_causal=False, softmax_scale=None, **kwargs: Any):
        # PyTorch SDPA uses 'scale' parameter instead of 'softmax_scale'
        return attn_fn(
            query, key, value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=softmax_scale
        )
    
    if args.save_inputs:
        # 0 means unlimited
        max_caps = args.max_captures if args.max_captures > 0 else None
        _capture_wrapper = InputCaptureWrapper(sdpa_wrapper, args.input_dir, name="sdpa", max_captures=max_caps)
        F.scaled_dot_product_attention = _capture_wrapper
    else:
        F.scaled_dot_product_attention = sdpa_wrapper
elif args.attention_type == 'fa2':
    from aiter.ops.triton.mha import flash_attn_func as _fa2

    # Wrapper to convert BHSD to BSHD
    # scaled_dot_product_attention: (batch, heads, seqlen, dim)
    # flash_attn_func: (batch, seqlen, heads, dim)
    def fa2_wrapper(query, key, value, is_causal=False, softmax_scale=None, **kwargs: Any):
        original_dtype = query.dtype
        
        # Transpose from BHSD to BSHD
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        
        # Call flash attention
        out = _fa2(q, k, v, dropout_p=0.0, softmax_scale=softmax_scale, causal=is_causal)
        
        # Transpose back from BSHD to BHSD
        return out.transpose(1, 2).to(original_dtype)
    
    if args.save_inputs:
        # 0 means unlimited
        max_caps = args.max_captures if args.max_captures > 0 else None
        _capture_wrapper = InputCaptureWrapper(fa2_wrapper, args.input_dir, name="fa2", max_captures=max_caps)
        F.scaled_dot_product_attention = _capture_wrapper
    else:
        F.scaled_dot_product_attention = fa2_wrapper
elif args.attention_type == 'fa3':
    from aiter.ops.triton.mha_v3 import flash_attn_func as _fa3

    # Wrapper to convert BHSD to BSHD
    # scaled_dot_product_attention: (batch, heads, seqlen, dim)
    # flash_attn_func (v3): (batch, seqlen, heads, dim)
    def fa3_wrapper(query, key, value, is_causal=False, softmax_scale=None, **kwargs: Any):
        original_dtype = query.dtype
        
        # Transpose from BHSD to BSHD
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        
        # Call flash attention v3
        out = _fa3(q, k, v, softmax_scale=softmax_scale, causal=is_causal)
        
        # Transpose back from BSHD to BHSD
        return out.transpose(1, 2).to(original_dtype)
    
    if args.save_inputs:
        # 0 means unlimited
        max_caps = args.max_captures if args.max_captures > 0 else None
        _capture_wrapper = InputCaptureWrapper(fa3_wrapper, args.input_dir, name="fa3", max_captures=max_caps)
        F.scaled_dot_product_attention = _capture_wrapper
    else:
        F.scaled_dot_product_attention = fa3_wrapper
elif args.attention_type == 'fa3_fp8':
    from aiter.ops.triton.mha_v3 import flash_attn_fp8_func as _fa3_fp8

    # Wrapper to convert BHSD to BSHD
    # scaled_dot_product_attention: (batch, heads, seqlen, dim)
    # flash_attn_fp8_func: (batch, seqlen, heads, dim)
    def fa3_fp8_wrapper(query, key, value, is_causal=False, softmax_scale=None, **kwargs: Any):
        # Store original dtype to restore after FP8 attention (which returns FP32)
        original_dtype = query.dtype
        
        # Transpose from BHSD to BSHD
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        
        # Call flash attention (ignores attn_mask and dropout_p for now)
        # Note: flash_attn_fp8_func returns FP32 output for numerical stability
        out = _fa3_fp8(q, k, v, softmax_scale=softmax_scale, causal=is_causal)
        
        # Transpose back from BSHD to BHSD and restore original dtype
        # This is necessary because FP8 attention returns FP32 but downstream
        # layers (like to_out linear) expect the original dtype (e.g., float16)
        return out.transpose(1, 2).to(original_dtype)
    
    if args.save_inputs:
        # 0 means unlimited
        max_caps = args.max_captures if args.max_captures > 0 else None
        _capture_wrapper = InputCaptureWrapper(fa3_fp8_wrapper, args.input_dir, name="fa3_fp8", max_captures=max_caps)
        F.scaled_dot_product_attention = _capture_wrapper
    else:
        F.scaled_dot_product_attention = fa3_fp8_wrapper
elif args.attention_type == 'sagev1_fa3':
    from aiter.ops.triton.mha_v3 import flash_attn_fp8_func as _fa3_fp8
    from aiter.ops.triton.sage_v1 import sage_attn_v1_wrapper_func as _sagev1_fa3

    # Wrapper to convert BHSD to BSHD
    # scaled_dot_product_attention: (batch, heads, seqlen, dim)
    # sage_attn_v1_wrapper_func: (batch, seqlen, heads, dim)
    def sagev1_fa3_wrapper(query, key, value, is_causal=False, softmax_scale=None, **kwargs: Any):
        original_dtype = query.dtype
        
        # Transpose from BHSD to BSHD
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        
        # Call sagev1 attention (ignores attn_mask and dropout_p for now)
        # Note: returns FP32 output for numerical stability
        out = _sagev1_fa3(q, k, v, softmax_scale=softmax_scale, causal=is_causal)
        
        # Transpose back from BSHD to BHSD and restore original dtype
        return out.transpose(1, 2).to(original_dtype)
    
    if args.save_inputs:
        # 0 means unlimited
        max_caps = args.max_captures if args.max_captures > 0 else None
        _capture_wrapper = InputCaptureWrapper(sagev1_fa3_wrapper, args.input_dir, name="_sagev1_fa3", max_captures=max_caps)
        F.scaled_dot_product_attention = _capture_wrapper
    else:
        F.scaled_dot_product_attention = sagev1_fa3_wrapper
else:
    raise ValueError(f"Attention type {args.attention_type} not supported")

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

pipe = CogVideoXPipeline.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16
).to("cuda")

if args.compile:
    if _is_amd_gpu():
        logger.warning("torch.compile with inductor backend is not fully compatible with AMD GPUs.")
        logger.warning("Disabling compilation to avoid 'cluster_dims' metadata error.")
        args.compile = False
    else:
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

with torch.no_grad():
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

output_file = f"cogvideox-2b_{args.attention_type}.mp4"
export_to_video(video, output_file, fps=8)
logger.info(f"Video exported to {output_file}")

# Print capture summary if inputs were saved
if args.save_inputs and _capture_wrapper is not None:
    _capture_wrapper.summary()
