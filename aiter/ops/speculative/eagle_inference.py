"""
EAGLE Lightweight Inference

A simplified EAGLE speculative decoding implementation that doesn't require
a full inference framework. Works with standard PyTorch models.

Author: AIter Team

Example:
    >>> from aiter.ops.speculative import EAGLEInference, EAGLEConfig
    >>> 
    >>> # Initialize
    >>> config = EAGLEConfig(topk=4, num_steps=3, num_draft_tokens=8)
    >>> eagle = EAGLEInference(draft_model, target_model, config)
    >>> 
    >>> # Generate
    >>> input_ids = torch.tensor([[1, 2, 3, 4]])
    >>> output_ids, stats = eagle.generate(
    ...     input_ids,
    ...     max_new_tokens=100,
    ...     temperature=0.0,  # greedy
    ... )
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import logging

from .eagle_utils import (
    organize_draft_results,
    build_tree_structure,
    verify_tree_greedy,
    compute_tree_statistics,
    TreeMaskMode,
)
from .spec_utils import (
    select_top_k_tokens,
    sample_from_logits,
)

logger = logging.getLogger(__name__)


@dataclass
class EAGLEConfig:
    """Configuration for EAGLE speculative decoding."""
    
    # Tree structure
    topk: int = 4                      # Number of candidates per step
    num_steps: int = 3                 # Number of speculation steps
    num_draft_tokens: int = 8          # Total draft tokens (topk^num_steps)
    
    # Sampling parameters
    temperature: float = 0.0           # 0.0 for greedy
    top_p: Optional[float] = None      # Nucleus sampling
    top_k: Optional[int] = None        # Top-k sampling
    
    # Attention mask mode
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK
    
    # Device
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    
    # Statistics
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.topk > 0, "topk must be positive"
        assert self.num_steps > 0, "num_steps must be positive"
        assert self.temperature >= 0.0, "temperature must be non-negative"
        
        # Auto-calculate num_draft_tokens if not specified correctly
        expected_tokens = sum(self.topk ** i for i in range(1, self.num_steps + 1))
        if self.num_draft_tokens != expected_tokens:
            logger.warning(
                f"num_draft_tokens={self.num_draft_tokens} doesn't match "
                f"expected {expected_tokens} for topk={self.topk}, num_steps={self.num_steps}. "
                f"Using {expected_tokens}."
            )
            self.num_draft_tokens = expected_tokens


class EAGLEInference:
    """
    Lightweight EAGLE speculative decoding inference.
    
    This class provides speculative decoding without requiring a full
    inference framework like SGLang or vLLM. It works with standard
    PyTorch models.
    
    Attributes:
        draft_model: Small fast model for draft generation
        target_model: Large accurate model for verification
        config: EAGLE configuration
        
    Example:
        >>> # Load models (HuggingFace or custom)
        >>> draft_model = AutoModelForCausalLM.from_pretrained("draft_model")
        >>> target_model = AutoModelForCausalLM.from_pretrained("target_model")
        >>> 
        >>> # Create EAGLE inference
        >>> config = EAGLEConfig(topk=4, num_steps=3)
        >>> eagle = EAGLEInference(draft_model, target_model, config)
        >>> 
        >>> # Generate text
        >>> input_ids = tokenizer.encode("Hello", return_tensors="pt")
        >>> output_ids, stats = eagle.generate(input_ids, max_new_tokens=50)
    """
    
    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        config: EAGLEConfig,
    ):
        """
        Initialize EAGLE inference.
        
        Args:
            draft_model: Fast draft model (typically smaller)
            target_model: Accurate target model (typically larger)
            config: EAGLE configuration
        """
        self.draft_model = draft_model.to(config.device).eval()
        self.target_model = target_model.to(config.device).eval()
        self.config = config
        
        # Move models to specified dtype
        if config.dtype != torch.float32:
            self.draft_model = self.draft_model.to(config.dtype)
            self.target_model = self.target_model.to(config.dtype)
        
        # Statistics tracking
        self.reset_statistics()
        
        logger.info(f"Initialized EAGLE with config: {config}")
    
    def reset_statistics(self):
        """Reset generation statistics."""
        self.stats = {
            'total_draft_tokens': 0,
            'total_accepted_tokens': 0,
            'total_target_forward': 0,
            'total_draft_forward': 0,
            'acceptance_rates': [],
        }
    
    @torch.no_grad()
    def draft_step(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Single draft generation step.
        
        Args:
            input_ids: [batch, seq_len] input token IDs
            past_key_values: Cached key-values from previous steps
            
        Returns:
            scores: [batch, topk] log probabilities
            token_ids: [batch, topk] generated token IDs
            new_past_key_values: Updated KV cache
        """
        # Forward through draft model
        outputs = self.draft_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        logits = outputs.logits[:, -1, :]  # [batch, vocab]
        
        # Sample top-k tokens
        if self.config.temperature == 0.0:
            # Greedy: just take top-k
            scores, token_ids = select_top_k_tokens(logits, self.config.topk)
        else:
            # Sample with temperature
            log_probs = torch.log_softmax(logits / self.config.temperature, dim=-1)
            scores, token_ids = select_top_k_tokens(log_probs, self.config.topk)
        
        return scores, token_ids, outputs.past_key_values
    
    @torch.no_grad()
    def generate_draft_tree(
        self,
        input_ids: torch.Tensor,
        verified_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a tree of draft tokens.
        
        Args:
            input_ids: [batch, seq_len] current input sequence
            verified_id: [batch] last verified token
            
        Returns:
            draft_tokens: [batch, num_draft_tokens-1] generated drafts
            parent_list: [batch, num_parents] parent structure
            top_scores_index: [batch, num_draft_tokens-1] token selection indices
            tree_mask: Attention mask for tree
            positions: Position IDs
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        score_list = []
        token_list = []
        parents_list = [torch.zeros(batch_size, 1, dtype=torch.long, device=device)]
        
        current_input = verified_id.unsqueeze(1)  # [batch, 1]
        past_key_values = None
        
        # Multi-step draft generation
        for step in range(self.config.num_steps):
            if step == 0:
                # First step: generate from verified token
                scores, tokens, past_key_values = self.draft_step(
                    current_input,
                    past_key_values=None,
                )
            else:
                # Subsequent steps: generate from all candidates
                # Expand input for all candidates
                batch_size, num_candidates = tokens.shape
                expanded_input = tokens.reshape(-1, 1)  # [batch*candidates, 1]
                
                scores, tokens, past_key_values = self.draft_step(
                    expanded_input,
                    past_key_values=None,  # Simplified: no KV cache reuse
                )
                
                # Reshape back
                scores = scores.view(batch_size, -1)
                tokens = tokens.view(batch_size, -1)
                
                # Parent indices for this level
                parents = torch.arange(
                    num_candidates,
                    dtype=torch.long,
                    device=device
                ).unsqueeze(0).expand(batch_size, -1)
                parents_list.append(parents)
            
            score_list.append(scores)
            token_list.append(tokens)
            
            current_input = tokens
            self.stats['total_draft_forward'] += 1
        
        # Organize results
        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list,
            token_list,
            parents_list,
            self.config.num_draft_tokens,
        )
        
        self.stats['total_draft_tokens'] += draft_tokens.numel()
        
        return draft_tokens, parent_list, top_scores_index
    
    @torch.no_grad()
    def verify_drafts(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        verified_id: torch.Tensor,
        parent_list: torch.Tensor,
        top_scores_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Verify draft tokens with target model.
        
        Args:
            input_ids: [batch, seq_len] current sequence
            draft_tokens: [batch, num_draft_tokens-1] draft tokens
            verified_id: [batch] last verified token
            parent_list: [batch, num_parents] parent structure
            top_scores_index: [batch, num_draft_tokens-1] indices
            
        Returns:
            accepted_tokens: [batch, num_accepted] accepted tokens
            num_accepted: Number of accepted tokens
            stats: Acceptance statistics
        """
        batch_size = input_ids.shape[0]
        seq_lens = torch.tensor(
            [input_ids.shape[1]] * batch_size,
            dtype=torch.int32,
            device=input_ids.device,
        )
        
        # Build tree structure
        (
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens_with_verified,
        ) = build_tree_structure(
            verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            seq_lens,
            self.config.topk,
            self.config.num_steps,
            self.config.num_draft_tokens,
            self.config.tree_mask_mode,
        )
        
        # Prepare input for target model: [batch, seq_len + num_draft_tokens]
        candidates = draft_tokens_with_verified.view(
            batch_size, self.config.num_draft_tokens
        )
        
        # Run target model on full sequence + drafts
        # Note: This is simplified - in production you'd want to use
        # the tree_mask for efficient attention
        full_input = torch.cat([input_ids, candidates], dim=1)
        
        outputs = self.target_model(
            input_ids=full_input,
            use_cache=False,
        )
        
        # Get predictions for draft positions
        logits = outputs.logits[:, -self.config.num_draft_tokens:, :]
        target_predict = logits.argmax(dim=-1)  # [batch, num_draft_tokens]
        
        self.stats['total_target_forward'] += 1
        
        # Verify using greedy matching
        predicts, accept_index, accept_length = verify_tree_greedy(
            candidates,
            target_predict,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
        )
        
        # Extract accepted tokens
        max_accept = accept_length.max().item()
        if max_accept > 0:
            # Gather accepted tokens
            accepted_tokens = []
            for i in range(batch_size):
                n_accept = accept_length[i].item()
                if n_accept > 0:
                    indices = accept_index[i, :n_accept]
                    tokens = candidates[i, indices]
                    accepted_tokens.append(tokens)
                else:
                    accepted_tokens.append(torch.tensor([], device=input_ids.device))
            
            self.stats['total_accepted_tokens'] += accept_length.sum().item()
        else:
            accepted_tokens = [torch.tensor([], device=input_ids.device)] * batch_size
        
        # Compute statistics
        tree_stats = compute_tree_statistics(
            accept_length,
            self.config.num_draft_tokens,
            self.config.num_steps,
        )
        
        if tree_stats['acceptance_rate'] > 0:
            self.stats['acceptance_rates'].append(tree_stats['acceptance_rate'])
        
        return accepted_tokens, accept_length, tree_stats
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        return_stats: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Generate text using EAGLE speculative decoding.
        
        Args:
            input_ids: [batch, seq_len] input token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (overrides config)
            top_p: Nucleus sampling (overrides config)
            top_k: Top-k sampling (overrides config)
            return_stats: Whether to return statistics
            
        Returns:
            output_ids: [batch, seq_len + new_tokens] generated sequence
            stats: Generation statistics (if return_stats=True)
            
        Example:
            >>> input_ids = torch.tensor([[1, 2, 3, 4]])
            >>> output_ids, stats = eagle.generate(
            ...     input_ids,
            ...     max_new_tokens=50,
            ...     temperature=0.7,
            ... )
            >>> print(f"Generated {output_ids.shape[1] - input_ids.shape[1]} tokens")
            >>> print(f"Acceptance rate: {stats['mean_acceptance_rate']:.2%}")
        """
        # Override config if specified
        if temperature is not None:
            self.config.temperature = temperature
        if top_p is not None:
            self.config.top_p = top_p
        if top_k is not None:
            self.config.top_k = top_k
        
        self.reset_statistics()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        current_ids = input_ids.clone()
        tokens_generated = 0
        
        while tokens_generated < max_new_tokens:
            # Get last token as verified
            verified_id = current_ids[:, -1]
            
            # Generate draft tree
            draft_tokens, parent_list, top_scores_index = self.generate_draft_tree(
                current_ids,
                verified_id,
            )
            
            # Verify drafts
            accepted_tokens, accept_length, tree_stats = self.verify_drafts(
                current_ids,
                draft_tokens,
                verified_id,
                parent_list,
                top_scores_index,
            )
            
            # Append accepted tokens
            if accept_length.sum() > 0:
                # For simplicity, handle batch_size=1
                if batch_size == 1:
                    if len(accepted_tokens[0]) > 0:
                        current_ids = torch.cat(
                            [current_ids, accepted_tokens[0].unsqueeze(0)],
                            dim=1,
                        )
                        tokens_generated += len(accepted_tokens[0])
                    else:
                        # No acceptance: fall back to target model greedy
                        outputs = self.target_model(input_ids=current_ids, use_cache=False)
                        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        current_ids = torch.cat([current_ids, next_token], dim=1)
                        tokens_generated += 1
                else:
                    # Batch handling: more complex, simplified here
                    logger.warning("Batch generation with varying acceptance not fully implemented")
                    break
            else:
                # No tokens accepted: fall back
                outputs = self.target_model(input_ids=current_ids, use_cache=False)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                tokens_generated += 1
            
            if self.config.verbose:
                logger.info(
                    f"Generated {tokens_generated}/{max_new_tokens} tokens. "
                    f"Acceptance rate: {tree_stats['acceptance_rate']:.2%}"
                )
        
        # Compute final statistics
        final_stats = None
        if return_stats:
            final_stats = {
                'total_tokens_generated': tokens_generated,
                'total_draft_tokens': self.stats['total_draft_tokens'],
                'total_accepted_tokens': self.stats['total_accepted_tokens'],
                'total_target_forward': self.stats['total_target_forward'],
                'total_draft_forward': self.stats['total_draft_forward'],
                'mean_acceptance_rate': (
                    sum(self.stats['acceptance_rates']) / len(self.stats['acceptance_rates'])
                    if self.stats['acceptance_rates'] else 0.0
                ),
                'speedup_ratio': (
                    self.stats['total_accepted_tokens'] / self.stats['total_target_forward']
                    if self.stats['total_target_forward'] > 0 else 0.0
                ),
            }
        
        return current_ids, final_stats

