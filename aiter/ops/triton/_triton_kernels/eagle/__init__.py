"""
Eagle Speculative Decoding for AMD GPU

This module provides a Triton-based implementation of Eagle speculative decoding,
ported from sglang for compatibility with AMD GPUs via ROCm.

Key Features:
- Tree-based draft token generation
- Parallel verification with target model  
- Greedy and probabilistic sampling support
- Full ROCm/HIP compatibility via Triton kernels

Usage Example:
    >>> from aiter.ops.triton._triton_kernels.eagle import (
    ...     build_tree_kernel_efficient,
    ...     verify_tree_greedy_func,
    ... )
    >>> 
    >>> # Build tree structure
    >>> tree_mask, positions, ... = build_tree_kernel_efficient(
    ...     verified_id, parent_list, top_scores_index, draft_tokens,
    ...     seq_lens, seq_lens_sum, topk, spec_steps, num_draft_tokens
    ... )
    >>> 
    >>> # Verify draft tokens
    >>> predicts, accept_index, accept_length = verify_tree_greedy_func(
    ...     predicts, accept_index, accept_length, candidates,
    ...     retrive_index, retrive_next_token, retrive_next_sibling,
    ...     target_predict
    ... )

Author: AIter Team
License: MIT
"""

from .tree_kernels import (
    build_tree_efficient_triton,
    verify_tree_greedy_triton,
    build_tree_kernel_triton,
    verify_tree_greedy_kernel,
    tree_speculative_sampling_kernel,
)

# Re-export with sglang-compatible names
build_tree_kernel_efficient = build_tree_efficient_triton
verify_tree_greedy_func = verify_tree_greedy_triton

__all__ = [
    # Triton kernel functions
    'build_tree_efficient_triton',
    'verify_tree_greedy_triton',
    'build_tree_kernel_triton',
    'verify_tree_greedy_kernel',
    'tree_speculative_sampling_kernel',
    
    # sglang-compatible aliases
    'build_tree_kernel_efficient',
    'verify_tree_greedy_func',
]

__version__ = '0.1.0'

