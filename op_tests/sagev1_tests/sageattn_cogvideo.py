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

from utils import make_bshd_wrapper, InputCaptureWrapper




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
parser.add_argument('--attention_type', type=str, default='sdpa', choices=['sdpa', 'sagev1', 'fav2', 'fav3', 'fav3_fp8', 'fav3_sage'], help='Attention type')
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


# Define attention wrapper based on attention_type
kernel_name = args.attention_type
attn_fn = None

if args.attention_type == 'sagev1':
    attn_fn = sageattn

elif args.attention_type == 'sdpa':
    _sdpa_fn = torch.nn.functional.scaled_dot_product_attention
    
    # Wrapper for PyTorch's native scaled_dot_product_attention
    # scaled_dot_product_attention: (batch, heads, seqlen, dim) - BHSD format
    def sdpa_wrapper(query, key, value, is_causal=False, softmax_scale=None, **kwargs: Any):
        # PyTorch SDPA uses 'scale' parameter instead of 'softmax_scale'
        return _sdpa_fn(
            query, key, value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=softmax_scale
        )
    attn_fn = sdpa_wrapper

elif args.attention_type == 'fav2':
    from aiter.ops.triton.mha import flash_attn_func as _fav2
    attn_fn = make_bshd_wrapper(_fav2, dropout_p=0.0)

elif args.attention_type == 'fav3':
    from aiter.ops.triton.mha_v3 import flash_attn_func as _fav3
    attn_fn = make_bshd_wrapper(_fav3)

elif args.attention_type == 'fav3_fp8':
    from aiter.ops.triton.mha_v3 import flash_attn_fp8_func as _fav3_fp8
    attn_fn = make_bshd_wrapper(_fav3_fp8)

elif args.attention_type == 'fav3_sage':
    from aiter.ops.triton.fav3_sage import fav3_sage_wrapper_func as _fav3_sage
    kernel_name = "_fav3_sage"  # Override kernel name for this variant
    attn_fn = make_bshd_wrapper(_fav3_sage)

else:
    raise ValueError(f"Attention type {args.attention_type} not supported")

# Apply attention function with optional input capture wrapper
if args.save_inputs:
    max_caps = args.max_captures if args.max_captures > 0 else None
    _capture_wrapper = InputCaptureWrapper(attn_fn, args.input_dir, name=kernel_name, max_captures=max_caps)
    F.scaled_dot_product_attention = _capture_wrapper
else:
    F.scaled_dot_product_attention = attn_fn

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
