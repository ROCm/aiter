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
            save_path = os.path.join(self.save_dir, f"{self.name}_input_{self.call_idx}.pt")
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


def make_bshd_wrapper(inner_fn, **fixed_kwargs):
    """
    Creates a BHSD->BSHD wrapper for attention functions that expect BSHD format.
    
    BHSD = (batch, heads, seqlen, dim) - PyTorch's scaled_dot_product_attention format
    BSHD = (batch, seqlen, heads, dim) - Flash attention format
    
    Args:
        inner_fn: Attention function expecting BSHD format with signature:
                  inner_fn(q, k, v, softmax_scale=..., causal=..., **fixed_kwargs)
        **fixed_kwargs: Additional fixed arguments to pass to inner_fn (e.g., dropout_p=0.0)
    """
    def wrapper(query, key, value, is_causal=False, softmax_scale=None, **kwargs: Any):
        original_dtype = query.dtype
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        out = inner_fn(q, k, v, softmax_scale=softmax_scale, causal=is_causal, **fixed_kwargs)
        return out.transpose(1, 2).to(original_dtype)
    return wrapper