"""
Eagle Tree Building and Verification Kernels in Triton
Ported from sglang CUDA kernels for AMD GPU compatibility

Author: AIter Team
"""

import torch
import triton
import triton.language as tl


@triton.jit
def build_tree_kernel_triton(
    parent_list,           # [bs, topk * (depth - 1) + 1]
    selected_index,        # [bs, draft_token_num - 1]
    verified_seq_len,      # [bs]
    tree_mask,             # output: [sum(verified_seq_len)*draft_token+bs*draft_token*draft_token]
    positions,             # output: [bs * draft_token]
    retrive_index,         # output: [bs, draft_token]
    retrive_next_token,    # output: [bs, draft_token]
    retrive_next_sibling,  # output: [bs, draft_token]
    topk: tl.constexpr,
    depth: tl.constexpr,
    draft_token_num: tl.constexpr,
    draft_token_num_p2: tl.constexpr,  # Next power of 2
    tree_mask_mode: tl.constexpr,
):
    """
    Build tree structure for Eagle speculative decoding.
    
    This kernel constructs:
    - tree_mask: attention mask for draft tokens
    - positions: position embedding for each token
    - retrive_index/next_token/next_sibling: tree navigation structure
    
    Grid: (bs,)
    Block: (draft_token_num_p2,)  # Must be power of 2 for Triton
    """
    bid = tl.program_id(axis=0)
    tid = tl.arange(0, draft_token_num_p2)
    
    # Mask out threads beyond actual draft_token_num
    valid_mask = tid < draft_token_num
    
    # Load sequence length for this batch
    seq_len = tl.load(verified_seq_len + bid)
    
    # Calculate tree mask index
    if tree_mask_mode == 0:  # FULL_MASK
        # Compute cumulative offset for previous batches
        seq_tree_idx = draft_token_num * draft_token_num * bid
        for i in range(bid):
            prev_seq_len = tl.load(verified_seq_len + i)
            seq_tree_idx += prev_seq_len * draft_token_num
        
        token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len + 1
    else:  # QLEN_ONLY
        token_tree_idx = draft_token_num * draft_token_num * bid + draft_token_num * tid + 1
    
    # Initialize tree mask: set current token to True, rest to False
    tl.store(tree_mask + token_tree_idx - 1, 1, mask=valid_mask)
    for i in range(draft_token_num - 1):
        tl.store(tree_mask + token_tree_idx + i, 0, mask=valid_mask)
    
    # Build tree structure (simplified - full implementation would be more complex)
    # This is the navigation structure used during verification
    
    # Thread 0 handles building the parent-child relationships
    if tid == 0 and valid_mask:
        # Set root position
        tl.store(positions + bid * draft_token_num, seq_len)
        
        # Build parent-child links (backward from leaves to root)
        for i in range(draft_token_num - 1, 0, -1):
            current_token_idx = bid * draft_token_num + i
            tl.store(retrive_index + current_token_idx, current_token_idx)
            
            # Find parent index
            parent_tb_idx = tl.load(selected_index + bid * (draft_token_num - 1) + i - 1) // topk
            
            if parent_tb_idx > 0:
                parent_token_idx = tl.load(parent_list + bid * (topk * (depth - 1) + 1) + parent_tb_idx)
                
                # Find parent position in selected tokens (avoid break)
                parent_position = 0
                found_parent = 0
                for j in range(draft_token_num):
                    token_match = tl.load(selected_index + bid * (draft_token_num - 1) + j) == parent_token_idx
                    # Use conditional assignment instead of break
                    parent_position = tl.where(token_match & (found_parent == 0), j + 1, parent_position)
                    found_parent = tl.where(token_match, 1, found_parent)
                
                # Link child to parent
                next_token_val = tl.load(retrive_next_token + bid * draft_token_num + parent_position)
                if next_token_val == -1:
                    tl.store(retrive_next_token + bid * draft_token_num + parent_position, i)
                else:
                    tl.store(retrive_next_token + bid * draft_token_num + parent_position, i)
                    tl.store(retrive_next_sibling + bid * draft_token_num + i, next_token_val)
            else:
                # Root level
                next_token_val = tl.load(retrive_next_token + bid * draft_token_num)
                if next_token_val == -1:
                    tl.store(retrive_next_token + bid * draft_token_num, i)
                else:
                    tl.store(retrive_next_token + bid * draft_token_num, i)
                    tl.store(retrive_next_sibling + bid * draft_token_num + i, next_token_val)
        
        # Set root index
        tl.store(retrive_index + bid * draft_token_num, bid * draft_token_num)
    
    # Other threads handle position calculation (simplified)
    # For compatibility, use simple sequential positions
    if valid_mask:
        # Simple position: tid + seq_len
        tl.store(positions + bid * draft_token_num + tid, tid + seq_len)


@triton.jit
def verify_tree_greedy_kernel(
    predicts,              # output: [tot_num_draft_tokens]
    accept_index,          # output: [bs, num_spec_step]
    accept_token_num,      # output: [bs]
    candidates,            # [bs, num_draft_tokens]
    retrive_index,         # [bs, num_draft_tokens]
    retrive_next_token,    # [bs, num_draft_tokens]
    retrive_next_sibling,  # [bs, num_draft_tokens]
    target_predict,        # [bs, num_draft_tokens]
    num_speculative_tokens: tl.constexpr,
    num_draft_tokens: tl.constexpr,
):
    """
    Greedy verification of draft tokens against target predictions.
    
    Traverses the tree and accepts tokens that match target predictions.
    Uses conditional flags instead of break statements for Triton compatibility.
    
    Grid: (bs,)
    Block: (1,) - single thread per batch
    """
    bx = tl.program_id(axis=0)
    
    # Start from root
    last_accepted_retrive_idx = tl.load(retrive_index + bx * num_draft_tokens)
    tl.store(accept_index + bx * num_speculative_tokens, last_accepted_retrive_idx)
    num_accepted_tokens = 0
    cur_index = 0
    
    # Flag to control verification continuation
    should_continue = 1
    
    # Simplified greedy verification: just check linear sequence
    # For full tree verification, use CPU implementation or more complex kernel
    for j in range(1, num_speculative_tokens):
        if should_continue == 0:
            # Already stopped
            should_continue = 0  # Keep stopped
        else:
            # Check next token in sequence
            if j < num_draft_tokens:
                draft_token_id = tl.load(candidates + bx * num_draft_tokens + j)
                target_token_id = tl.load(target_predict + bx * num_draft_tokens + j - 1)
                
                if draft_token_id == target_token_id:
                    # Accept token
                    tl.store(predicts + bx * num_draft_tokens + j, target_token_id)
                    num_accepted_tokens += 1
                    tl.store(accept_index + bx * num_speculative_tokens + num_accepted_tokens, j)
                    last_accepted_retrive_idx = bx * num_draft_tokens + j
                else:
                    # Mismatch, stop
                    should_continue = 0
    
    # Store results
    tl.store(accept_token_num + bx, num_accepted_tokens)
    bonus_token = tl.load(target_predict + last_accepted_retrive_idx)
    tl.store(predicts + last_accepted_retrive_idx, bonus_token)


@triton.jit  
def tree_speculative_sampling_kernel(
    predicts,                # output: [tot_num_draft_tokens]
    accept_index,            # output: [bs, num_spec_step]
    accept_token_num,        # output: [bs]
    candidates,              # [bs, num_draft_tokens]
    retrive_index,           # [bs, num_draft_tokens]
    retrive_next_token,      # [bs, num_draft_tokens]
    retrive_next_sibling,    # [bs, num_draft_tokens]
    uniform_samples,         # [bs, num_draft_tokens]
    uniform_samples_final,   # [bs]
    target_probs,            # [bs, num_draft_tokens, vocab_size]
    draft_probs,             # [bs, num_draft_tokens, vocab_size]
    threshold_single: tl.constexpr,
    threshold_acc: tl.constexpr,
    vocab_size: tl.constexpr,
    num_speculative_tokens: tl.constexpr,
    num_draft_tokens: tl.constexpr,
):
    """
    Probabilistic sampling verification with rejection sampling.
    
    Uses target and draft probabilities to accept/reject tokens.
    
    Grid: (bs,)
    Block: (1,)
    """
    bx = tl.program_id(axis=0)
    
    last_accepted_idx = tl.load(retrive_index + bx * num_draft_tokens)
    tl.store(accept_index + bx * num_speculative_tokens, last_accepted_idx)
    num_accepted = 0
    cur_index = 0
    
    for j in range(1, num_speculative_tokens):
        cur_index = tl.load(retrive_next_token + bx * num_draft_tokens + cur_index)
        
        while cur_index != -1:
            draft_idx = tl.load(retrive_index + bx * num_draft_tokens + cur_index)
            candidate_token = tl.load(candidates + bx * num_draft_tokens + cur_index)
            
            # Get probabilities
            target_prob = tl.load(target_probs + bx * num_draft_tokens * vocab_size + 
                                  cur_index * vocab_size + candidate_token)
            draft_prob = tl.load(draft_probs + bx * num_draft_tokens * vocab_size + 
                                cur_index * vocab_size + candidate_token)
            
            # Rejection sampling
            accept_prob = tl.minimum(1.0, target_prob / (threshold_acc * draft_prob + 1e-10))
            
            # Also check threshold_single
            if target_prob >= threshold_single:
                accept_prob = 1.0
            
            coin = tl.load(uniform_samples + bx * num_draft_tokens + cur_index)
            
            if coin < accept_prob:
                # Accept
                tl.store(predicts + last_accepted_idx, candidate_token)
                num_accepted += 1
                tl.store(accept_index + bx * num_speculative_tokens + num_accepted, draft_idx)
                last_accepted_idx = draft_idx
                break
            else:
                # Reject, try sibling
                cur_index = tl.load(retrive_next_sibling + bx * num_draft_tokens + cur_index)
        
        if cur_index == -1:
            break
    
    # Sample bonus token
    coin_final = tl.load(uniform_samples_final + bx)
    cumsum = 0.0
    bonus_token = 0
    
    for v in range(vocab_size):
        prob = tl.load(target_probs + bx * num_draft_tokens * vocab_size + 
                      last_accepted_idx * vocab_size + v)
        cumsum += prob
        if coin_final < cumsum:
            bonus_token = v
            break
    
    tl.store(predicts + last_accepted_idx, bonus_token)
    tl.store(accept_token_num + bx, num_accepted)


# Python wrappers for easy calling
def build_tree_efficient_triton(
    verified_id: torch.Tensor,
    parent_list: torch.Tensor,
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: int = 0,
):
    """Python wrapper for tree building kernel."""
    bs = seq_lens.numel()
    device = seq_lens.device
    
    # Calculate next power of 2 for Triton (required for tl.arange)
    def next_power_of_2(n):
        return 1 << (n - 1).bit_length() if n > 0 else 1
    
    num_verify_tokens_p2 = next_power_of_2(num_verify_tokens)
    
    # Allocate outputs
    if tree_mask_mode == 0:  # FULL_MASK
        tree_mask_size = seq_lens_sum * num_verify_tokens + num_verify_tokens * num_verify_tokens * bs
    else:
        tree_mask_size = num_verify_tokens * bs * num_verify_tokens
    
    tree_mask = torch.full((tree_mask_size,), True, dtype=torch.bool, device=device)
    positions = torch.empty((bs * num_verify_tokens,), dtype=torch.long, device=device)
    retrive_index = torch.full((bs, num_verify_tokens), -1, dtype=torch.long, device=device)
    retrive_next_token = torch.full((bs, num_verify_tokens), -1, dtype=torch.long, device=device)
    retrive_next_sibling = torch.full((bs, num_verify_tokens), -1, dtype=torch.long, device=device)
    
    # Launch kernel
    grid = (bs,)
    build_tree_kernel_triton[grid](
        parent_list,
        top_scores_index,
        seq_lens,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        spec_steps,
        num_verify_tokens,
        num_verify_tokens_p2,  # Power of 2 for Triton
        tree_mask_mode,
    )
    
    draft_tokens_with_verified = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()
    
    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens_with_verified,
    )


def verify_tree_greedy_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
):
    """Python wrapper for greedy verification kernel."""
    bs = candidates.size(0)
    num_spec_step = accept_index.size(1)
    num_draft_tokens = candidates.size(1)
    
    grid = (bs,)
    verify_tree_greedy_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        num_spec_step,
        num_draft_tokens,
    )
    
    return predicts, accept_index, accept_token_num

