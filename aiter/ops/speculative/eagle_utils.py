"""
EAGLE Speculative Decoding Utilities

Extracted from sglang for lightweight use with aiter.

Author: AIter Team
"""

import torch
from typing import List, Tuple, Optional
from enum import IntEnum


class TreeMaskMode(IntEnum):
    """Tree mask generation modes."""
    FULL_MASK = 0           # Full attention mask
    QLEN_ONLY = 1           # Query length only
    QLEN_ONLY_BITPACKING = 2  # Bitpacked query length


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Organize draft model outputs into structured format.
    
    This function processes multi-step draft outputs and selects the top-k
    most promising candidates based on their scores.
    
    Args:
        score_list: List of score tensors from each draft step
                   Each tensor: [batch, num_candidates]
        token_list: List of token tensors from each draft step
                   Each tensor: [batch, num_candidates]
        parents_list: List of parent indices for tree structure
                     Each tensor: [batch, num_candidates]
        num_draft_tokens: Number of draft tokens to select
        
    Returns:
        parent_list: [batch, num_parents] parent indices
        top_scores_index: [batch, num_draft_tokens-1] indices of selected tokens
        draft_tokens: [batch, num_draft_tokens-1] selected draft tokens
        
    Example:
        >>> # 3 steps of draft generation with topk=4
        >>> scores = [torch.randn(2, 4) for _ in range(3)]  # batch=2
        >>> tokens = [torch.randint(0, 1000, (2, 4)) for _ in range(3)]
        >>> parents = [torch.zeros(2, 1, dtype=torch.long)]  # root
        >>> parents += [torch.arange(4).unsqueeze(0).expand(2, -1) for _ in range(2)]
        >>> parent_list, indices, draft = organize_draft_results(
        ...     scores, tokens, parents, num_draft_tokens=8
        ... )
    """
    # Concatenate all scores and flatten: [batch, total_candidates]
    score_list = torch.cat(score_list, dim=1).flatten(1)
    
    # Concatenate all tokens: [batch, total_candidates]
    ss_token_list = torch.cat(token_list, dim=1)
    
    # Select top-k candidates based on scores
    top_scores = torch.topk(score_list, num_draft_tokens - 1, dim=-1)
    top_scores_index = top_scores.indices
    
    # Sort indices to maintain generation order
    top_scores_index = torch.sort(top_scores_index).values
    
    # Gather selected draft tokens
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)
    
    # Build parent list (exclude last step as it has no children)
    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(
            batch_size, 0,
            dtype=torch.long,
            device=parents_list[0].device
        )
    
    return parent_list, top_scores_index, draft_tokens


def build_tree_structure(
    verified_id: torch.Tensor,
    parent_list: torch.Tensor,
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
    spec_steps: int,
    num_draft_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    use_triton: bool = False,  # Disabled by default due to compatibility issues
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build tree structure for EAGLE speculative decoding.
    
    Note: Current implementation uses a simplified CPU-based approach for maximum
    compatibility. The Triton kernel implementation is work-in-progress.
    
    Args:
        verified_id: [batch] last verified token ID
        parent_list: [batch, num_parents] parent indices
        top_scores_index: [batch, num_draft_tokens-1] selected token indices
        draft_tokens: [batch, num_draft_tokens-1] draft token IDs
        seq_lens: [batch] current sequence lengths
        topk: Number of candidates per step
        spec_steps: Number of speculative steps
        num_draft_tokens: Total number of draft tokens
        tree_mask_mode: How to generate attention mask
        use_triton: Whether to use Triton kernel (experimental, default: False)
        
    Returns:
        tree_mask: Attention mask for draft tokens
        positions: Position IDs for each draft token
        retrieve_index: Flattened indices for retrieving tokens
        retrieve_next_token: Tree navigation - next token
        retrieve_next_sibling: Tree navigation - next sibling
        draft_tokens_with_verified: All tokens including verified
    """
    batch_size = verified_id.shape[0]
    device = verified_id.device
    seq_lens_sum = seq_lens.sum().item()
    
    if use_triton:
        try:
            # Try using Triton kernel (experimental)
            from aiter.ops.triton._triton_kernels.eagle import build_tree_efficient_triton
            
            return build_tree_efficient_triton(
                verified_id,
                parent_list,
                top_scores_index,
                draft_tokens,
                seq_lens,
                seq_lens_sum,
                topk,
                spec_steps,
                num_draft_tokens,
                tree_mask_mode=int(tree_mask_mode),
            )
        except Exception as e:
            # Fall back to CPU implementation if Triton fails
            import warnings
            warnings.warn(f"Triton kernel failed ({e}), falling back to CPU implementation")
            use_triton = False
    
    # CPU fallback: simplified linear structure
    if tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask_size = seq_lens_sum * num_draft_tokens + num_draft_tokens * num_draft_tokens * batch_size
    else:
        tree_mask_size = num_draft_tokens * batch_size * num_draft_tokens
    
    tree_mask = torch.ones((tree_mask_size,), dtype=torch.bool, device=device)
    
    # Positions: simple sequential positions
    positions = torch.arange(num_draft_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
    positions = positions + seq_lens.unsqueeze(1)
    positions = positions.flatten()
    
    # Retrieve indices: simple linear mapping
    retrieve_index = torch.arange(
        batch_size * num_draft_tokens,
        device=device
    ).view(batch_size, num_draft_tokens)
    
    # Tree navigation: simple chain structure (0 -> 1 -> 2 -> ...)
    retrieve_next_token = torch.full(
        (batch_size, num_draft_tokens),
        -1,
        dtype=torch.long,
        device=device
    )
    for i in range(num_draft_tokens - 1):
        retrieve_next_token[:, i] = i + 1
    
    retrieve_next_sibling = torch.full(
        (batch_size, num_draft_tokens),
        -1,
        dtype=torch.long,
        device=device
    )
    
    # Combine verified ID with draft tokens
    draft_tokens_with_verified = torch.cat(
        [verified_id.unsqueeze(1), draft_tokens],
        dim=1
    ).flatten()
    
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens_with_verified,
    )


def verify_tree_greedy(
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    use_triton: bool = False,  # Disabled by default due to compatibility issues
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Verify draft tokens using greedy matching.
    
    Note: Current implementation uses a simple CPU-based greedy verification
    for maximum compatibility. The Triton kernel implementation is work-in-progress.
    
    Args:
        candidates: [batch, num_draft_tokens] candidate token IDs
        target_predict: [batch, num_draft_tokens] target model predictions
        retrieve_index: [batch, num_draft_tokens] retrieval indices
        retrieve_next_token: [batch, num_draft_tokens] next token pointers
        retrieve_next_sibling: [batch, num_draft_tokens] sibling pointers
        use_triton: Whether to use Triton kernel (experimental, default: False)
        
    Returns:
        predicts: [batch * num_draft_tokens] accepted predictions
        accept_index: [batch, max_accepts] indices of accepted tokens
        accept_length: [batch] number of accepted tokens per sequence
    """
    batch_size, num_draft_tokens = candidates.shape
    device = candidates.device
    
    # Prepare output tensors
    predicts = torch.zeros(
        (batch_size * num_draft_tokens,),
        dtype=torch.int32,
        device=device,
    )
    
    max_accepts = num_draft_tokens
    accept_index = torch.full(
        (batch_size, max_accepts),
        -1,
        dtype=torch.int32,
        device=device,
    )
    
    accept_length = torch.zeros(
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    
    if use_triton:
        try:
            # Try using Triton kernel (experimental)
            from aiter.ops.triton._triton_kernels.eagle import verify_tree_greedy_triton
            
            return verify_tree_greedy_triton(
                predicts,
                accept_index,
                accept_length,
                candidates,
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                target_predict,
            )
        except Exception as e:
            # Fall back to CPU implementation if Triton fails
            import warnings
            warnings.warn(f"Triton kernel failed ({e}), falling back to CPU implementation")
            use_triton = False
    
    # CPU fallback: simple greedy verification
    for b in range(batch_size):
        accepted = 0
        for i in range(num_draft_tokens):
            if candidates[b, i] == target_predict[b, i]:
                predicts[b * num_draft_tokens + i] = target_predict[b, i].item()
                accept_index[b, accepted] = i
                accepted += 1
            else:
                # Stop at first mismatch (greedy)
                break
        accept_length[b] = accepted
    
    return predicts, accept_index, accept_length


def compute_tree_statistics(
    accept_lengths: torch.Tensor,
    num_draft_tokens: int,
    num_steps: int,
) -> dict:
    """
    Compute statistics about tree acceptance.
    
    Args:
        accept_lengths: [batch] number of accepted tokens
        num_draft_tokens: Total draft tokens per sequence
        num_steps: Number of speculation steps
        
    Returns:
        stats: Dictionary with acceptance statistics
    """
    stats = {
        'mean_accept_length': accept_lengths.float().mean().item(),
        'max_accept_length': accept_lengths.max().item(),
        'min_accept_length': accept_lengths.min().item(),
        'acceptance_rate': accept_lengths.float().mean().item() / num_draft_tokens,
        'total_proposed': num_draft_tokens * accept_lengths.numel(),
        'total_accepted': accept_lengths.sum().item(),
        'batch_size': accept_lengths.numel(),
    }
    
    return stats

