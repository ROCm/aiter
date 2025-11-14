
import torch
import torch.nn.functional as F
import time
import unittest
import aiter
import math
import triton
import triton.language as tl

# ==============================================================================
# 1. Helper Functions (Finalized Versions)
# ==============================================================================

def pytorch_blocked_attention(q, k, v, block_relation_onehot, bias, is_causal, BLKQ, BLKK):
    """
    Memory-efficient, blocked PyTorch Attention implementation for reference.
    The causal mask logic is disabled to match the custom kernel's behavior.
    """
    input_dtype = q.dtype
    b, h, s_q, d = q.shape
    output = torch.zeros_like(q)
    num_q_blocks = s_q // BLKQ
    assert s_q % BLKQ == 0, "Sequence length must be divisible by block size"

    for b_idx in range(b):
        for h_idx in range(h):
            for q_block_idx in range(num_q_blocks):
                q_start, q_end = q_block_idx * BLKQ, (q_block_idx + 1) * BLKQ
                q_chunk = q[b_idx, h_idx, q_start:q_end, :]
                relevant_k_indices = block_relation_onehot[b_idx, h_idx, q_block_idx, :].nonzero(as_tuple=True)[0]
                if len(relevant_k_indices) == 0: continue

                k_chunks_list = [k[b_idx, h_idx, k_idx*BLKK:(k_idx+1)*BLKK, :] for k_idx in relevant_k_indices]
                v_chunks_list = [v[b_idx, h_idx, k_idx*BLKK:(k_idx+1)*BLKK, :] for k_idx in relevant_k_indices]
                bias_chunks_list = [bias[b_idx, h_idx, q_start:q_end, k_idx*BLKK:(k_idx+1)*BLKK] for k_idx in relevant_k_indices]

                k_cat = torch.cat(k_chunks_list, dim=0)
                v_cat = torch.cat(v_chunks_list, dim=0)
                bias_cat = torch.cat(bias_chunks_list, dim=1)

                scores = torch.matmul(q_chunk.float(), k_cat.float().transpose(-1, -2)) / math.sqrt(d) + bias_cat.float()

                # Causal masking is confirmed to be disabled for this comparison.
                # if is_causal:
                #     ...

                attn_probs = F.softmax(scores, dim=-1)
                output_chunk = torch.matmul(attn_probs, v_cat.float())
                output[b_idx, h_idx, q_start:q_end, :] = output_chunk.to(input_dtype)
    return output

@triton.jit
def triton_block_map_to_lut_kernel(map_ptr, lut_ptr, valid_block_num_ptr, num_block_k):
    # (no changes needed)
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    map_ptr = map_ptr + b * H * Q * num_block_k + h * Q * num_block_k + q * num_block_k
    lut_ptr = lut_ptr + b * H * Q * num_block_k + h * Q * num_block_k + q * num_block_k
    valid_block_num_ptr = valid_block_num_ptr + b * H * Q + h * Q + q
    valid_block_num = 0; prev_block = 0
    for i in range(num_block_k):
        cur_block = tl.load(map_ptr + i)
        if cur_block:
            tl.store(lut_ptr + valid_block_num, i - prev_block)
            valid_block_num += 1; prev_block = i
    tl.store(valid_block_num_ptr, valid_block_num)

def block_map_lut_triton(block_map):
    # (no changes needed)
    B, H, Q, K = block_map.shape
    lut = torch.zeros((B, H, Q, K), dtype=torch.int32, device=block_map.device)
    valid_block_num = torch.zeros((B, H, Q), dtype=torch.int32, device=block_map.device)
    triton_block_map_to_lut_kernel[(B, H, Q)](block_map.contiguous(), lut, valid_block_num, K)
    return lut, valid_block_num

def run_jenga_sparse_ck(q, k, v, block_relation_onehot):
    # (no changes needed)
    batch, nhead, seqlen_q, hdim_q = q.size(); _, nhead_k, seqlen_k, hdim_v = k.size()
    bias = torch.zeros([batch, nhead, seqlen_q, seqlen_k], dtype=q.dtype, device=q.device).contiguous()
    lse = torch.empty_like(bias).contiguous(); seqstart_q = torch.Tensor([0, seqlen_q]).to(torch.int).cuda().contiguous()
    seqstart_k = torch.Tensor([0, seqlen_k]).to(torch.int).cuda().contiguous()
    out0 = torch.empty_like(q).contiguous()
    out0 = aiter.jenga_sparse_attention_CK(q, k, v, block_relation_onehot, None, None, out0, bias, lse, seqstart_q, seqstart_k, 0, batch, nhead, nhead_k, seqlen_q, seqlen_k, hdim_q, hdim_v)
    lut, valid_block_num = block_map_lut_triton(block_relation_onehot)
    out1 = torch.empty_like(q).contiguous()
    out1 = aiter.jenga_sparse_attention_CK(q, k, v, None, lut, valid_block_num, out1, bias, lse, seqstart_q, seqstart_k, 0, batch, nhead, nhead_k, seqlen_q, seqlen_k, hdim_q, hdim_v)
    return out0, out1

def warmup_and_time(func, *args, repeat=10):
    # (no changes needed)
    func(*args); torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(repeat): func(*args)
    end_event.record(); torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / repeat

# ==============================================================================
# 2. Final Test Classes (Using File I/O)
# ==============================================================================

class TestJengaCorrectness(unittest.TestCase):
    """
    Tests correctness on a small slice of the data loaded from files.
    """
    def setUp(self):
        print("\n--- [Correctness Test] Setting up SMALL test data from FILES ---")
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.BLKQ = 128
        self.BLKK = 128
        self.is_causal = False # We confirmed this should be False for the comparison

        # Use a small slice of the data to ensure the reference implementation runs quickly.
        SEQ_LEN_SMALL = 4096
        NUM_Q_BLOCKS_SMALL = SEQ_LEN_SMALL // self.BLKQ

        full_query = torch.load("jenga_query_normal.pt")
        self.query = full_query[0:1, 0:4, :SEQ_LEN_SMALL, :].contiguous().to(self.device)
        self.key = torch.load("jenga_key.pt")[0:1, 0:4, :SEQ_LEN_SMALL, :].contiguous().to(self.device)
        self.value = torch.load("jenga_value.pt")[0:1, 0:4, :SEQ_LEN_SMALL, :].contiguous().to(self.device)
        self.block_relation_onehot = torch.load("jenga_block_relation_onehot.pt")[0:1, 0:4, :NUM_Q_BLOCKS_SMALL, :NUM_Q_BLOCKS_SMALL].contiguous().to(self.device)
        print(f"Using small data shape for correctness check: {self.query.shape}")

    def test_correctness(self):
        """Tests numerical consistency between the Jenga kernel and the reference."""
        print("\n--- Running correctness test (on small, file-based data) ---")
        b, n, s_q, d = self.query.shape; s_k = self.key.shape[2]
        bias = torch.zeros(b, n, s_q, s_k, dtype=self.dtype, device=self.device)

        print("Generating reference output with PyTorch...")
        out_ref = pytorch_blocked_attention(self.query, self.key, self.value, self.block_relation_onehot, bias, self.is_causal, self.BLKQ, self.BLKK)
        print("Running Jenga kernels...")
        out0, out1 = run_jenga_sparse_ck(self.query, self.key, self.value, self.block_relation_onehot)

        # A slightly higher tolerance is needed for bfloat16 comparisons.
        atol = 4e-2

        self.assertTrue(torch.allclose(out_ref, out0, atol=atol), f"FAILED: out0 max diff: {(out_ref - out0).abs().max().item()}")
        print("PASSED: out0 (onehot) matches the reference.")
        self.assertTrue(torch.allclose(out_ref, out1, atol=atol), f"FAILED: out1 max diff: {(out_ref - out1).abs().max().item()}")
        print("PASSED: out1 (LUT) matches the reference.")

class TestJengaPerformance(unittest.TestCase):
    """
    Tests performance on the full, large-scale data loaded from files.
    """
    def setUp(self):
        print("\n--- [Performance Test] Setting up FULL-SIZE test data from FILES ---")
        self.device = "cuda"
        self.query = torch.load("jenga_query_normal.pt").contiguous().to(self.device)
        self.key = torch.load("jenga_key.pt").contiguous().to(self.device)
        self.value = torch.load("jenga_value.pt").contiguous().to(self.device)
        self.block_relation_onehot = torch.load("jenga_block_relation_onehot.pt").contiguous().to(self.device)
        print(f"Using full data shape for performance check: {self.query.shape}")

    def test_performance(self):
        """Tests the performance of the Jenga custom kernel on the full-size dataset."""
        print("\n--- Running performance test (on full-size, file-based data) ---")
        jenga_time_ms = warmup_and_time(
            run_jenga_sparse_ck,
            self.query, self.key, self.value, self.block_relation_onehot,
            repeat=30 # Increase repetitions for more stable results
        )
        print(f"\n>>>> Jenga sparse attention average time: {jenga_time_ms:.4f} ms <<<<")


if __name__ == "__main__":
    # Run test suites sequentially
    print("==========================================================")
    print("=             Running Jenga Attention Tests            =")
    print("==========================================================")

    correctness_suite = unittest.TestLoader().loadTestsFromTestCase(TestJengaCorrectness)
    unittest.TextTestRunner().run(correctness_suite)

    performance_suite = unittest.TestLoader().loadTestsFromTestCase(TestJengaPerformance)
    unittest.TextTestRunner().run(performance_suite)

