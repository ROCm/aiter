"""
Speculative Decoding Utilities

Pure utility functions extracted from sglang for use with aiter.
These functions have no framework dependencies.

Author: AIter Team
"""

import torch
from typing import List, Tuple, Optional


def fast_topk_torch(
    scores: torch.Tensor,
    k: int,
    dim: int = -1,
    sorted: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast top-k selection using PyTorch.
    Falls back to torch.topk when CUDA kernels are unavailable.
    
    Args:
        scores: Input tensor [batch, vocab_size] or [batch, seq, vocab_size]
        k: Number of top elements to select
        dim: Dimension along which to select top-k
        sorted: Whether to sort the results
        
    Returns:
        values: Top-k values
        indices: Top-k indices
    """
    return torch.topk(scores, k, dim=dim, sorted=sorted)


def select_top_k_tokens(
    logits: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k tokens from logits.
    
    Args:
        logits: [batch, vocab_size] or [batch, num_candidates, vocab_size]
        k: Number of tokens to select
        
    Returns:
        scores: [batch, k] top-k scores (log probabilities)
        token_ids: [batch, k] top-k token IDs
    """
    if logits.dim() == 3:
        batch, num_candidates, vocab_size = logits.shape
        logits = logits.view(batch * num_candidates, vocab_size)
        scores, indices = fast_topk_torch(logits, k, dim=-1)
        scores = scores.view(batch, num_candidates, k)
        indices = indices.view(batch, num_candidates, k)
    else:
        scores, indices = fast_topk_torch(logits, k, dim=-1)
    
    return scores, indices


def generate_token_bitmask(
    token_ids: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """
    Generate a bitmask for token IDs.
    Used for efficient token filtering in speculative decoding.
    
    Args:
        token_ids: [batch, num_tokens] token IDs
        vocab_size: Size of vocabulary
        
    Returns:
        bitmask: [batch, vocab_size] boolean mask
    """
    batch_size, num_tokens = token_ids.shape
    device = token_ids.device
    
    # Create scatter mask
    bitmask = torch.zeros(
        (batch_size, vocab_size),
        dtype=torch.bool,
        device=device,
    )
    bitmask.scatter_(1, token_ids, True)
    
    return bitmask


def next_power_of_2(n: int) -> int:
    """Return the next power of 2 greater than or equal to n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def calculate_acceptance_rate(
    accept_lengths: torch.Tensor,
    num_draft_tokens: int,
) -> float:
    """
    Calculate the acceptance rate for speculative decoding.
    
    Args:
        accept_lengths: [batch] number of accepted tokens per batch
        num_draft_tokens: Total number of draft tokens proposed
        
    Returns:
        acceptance_rate: Ratio of accepted to proposed tokens
    """
    total_accepted = accept_lengths.sum().item()
    total_proposed = num_draft_tokens * accept_lengths.numel()
    
    if total_proposed == 0:
        return 0.0
    
    return total_accepted / total_proposed


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample tokens from logits with optional temperature, top-p, and top-k.
    
    Args:
        logits: [batch, vocab_size] logits
        temperature: Sampling temperature (> 0)
        top_p: Nucleus sampling threshold (0 < top_p <= 1)
        top_k: Top-k sampling threshold (> 0)
        
    Returns:
        token_ids: [batch] sampled token IDs
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    probs = torch.softmax(logits, dim=-1)
    
    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, probs.size(-1))
        indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
        probs[indices_to_remove] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # Apply top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        probs[indices_to_remove] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # Sample from the filtered distribution
    token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return token_ids


def pad_to_alignment(
    tensor: torch.Tensor,
    alignment: int,
    dim: int = 0,
    value: float = 0.0,
) -> torch.Tensor:
    """
    Pad tensor to alignment along specified dimension.
    Useful for efficient kernel execution.
    
    Args:
        tensor: Input tensor
        alignment: Target alignment (e.g., 8, 16, 32)
        dim: Dimension to pad
        value: Padding value
        
    Returns:
        padded_tensor: Padded tensor
    """
    size = tensor.size(dim)
    if size % alignment == 0:
        return tensor
    
    pad_size = (alignment - size % alignment)
    
    # Create padding specification
    pad_spec = [0, 0] * tensor.ndim
    pad_spec[-(dim * 2 + 1)] = pad_size
    
    return torch.nn.functional.pad(tensor, pad_spec, value=value)

