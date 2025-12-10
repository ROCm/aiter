import logging
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
parser.add_argument('--attention_type', type=str, default='sdpa', choices=['sdpa', 'sage', 'fa3', 'fa3_fp8'], help='Attention type')
args = parser.parse_args()

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


if args.attention_type == 'sage':
    F.scaled_dot_product_attention = sageattn
# TODO: add AMD fa2 and fa3 Triton kernels
elif args.attention_type == 'fa3':
    raise NotImplementedError("AMD fa3 Triton kernel is not yet supported")
#     from sageattention.fa3_wrapper import fa3
#     F.scaled_dot_product_attention = fa3
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
    
    F.scaled_dot_product_attention = fa3_fp8_wrapper

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
