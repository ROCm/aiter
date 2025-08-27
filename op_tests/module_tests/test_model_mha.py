import torch
import logging
from aiter import dtypes
from dataclasses import dataclass
from op_tests.test_mha import run_ck, run_torch
from aiter.test_common import checkAllclose, perftest


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
TEST_NUM_ITERS = 100


@perftest(num_iters=TEST_NUM_ITERS)
def run_ck_(
    q,
    k,
    v,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    deterministic=False,
    return_lse=True,
    return_attn_probs=False):
    return run_ck(q, k, v, bias, alibi_slopes, dout, dropout_p, 
                  causal, window_size, deterministic, return_lse, return_attn_probs)


def evaluate_mha(
        batch_size,
        num_heads,
        num_kv_heads,
        seq_len_q,
        seq_len_kv,
        head_dim,
        causal,
        bias_type,
        deterministic,
        dtype):
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()
    assert num_heads % num_kv_heads == 0
    
    q = torch.randn(
        batch_size, 
        seq_len_q, 
        num_heads, 
        head_dim, 
        device="cuda", 
        dtype=dtype, 
        requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seq_len_kv,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seq_len_kv,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    attn_bias = None
    alibi_slopes = None
    if bias_type == "bias":
        attn_bias = torch.randn(
            seq_len_q, seq_len_kv, device="cuda", dtype=dtype, requires_grad=True
        )
    elif bias_type == "alibi":
        alibi_slopes = torch.rand(batch_size, num_heads, device="cuda", dtype=dtypes.fp32)
    
    dout = torch.randn(
        batch_size,
        seq_len_q,
        num_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    return_lse = True
    return_attn_probs = True
    dropout_p = 0.0
    window_size = (-1, -1)

    (out, dropout_mask, dq, dk, dv, dbias), avg_mha_ck = run_ck_(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        causal,
        window_size,
        deterministic,
        return_lse,
        return_attn_probs,
    )

    out_ref, dq_ref, dk_ref, dv_ref, dbias_ref = run_torch(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        dropout_mask,
        causal,
        window_size,
    )

    out_pt, dq_pt, dk_pt, dv_pt, dbias_pt = run_torch(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        dropout_mask,
        causal,
        window_size,
        upcast=False,
        reorder_ops=True,
    )

    # print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    # print(f"Output Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    out_tol = max(2 * (out_pt - out_ref).abs().max().item(), 0.01)
    assert (out - out_ref).abs().max().item() <= out_tol

    # print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    # print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    # print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    # print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    # print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    # print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")

    dq_tol = max(10 * (dq_pt - dq_ref).abs().max().item(), 0.01)
    dk_tol = max(10 * (dk_pt - dk_ref).abs().max().item(), 0.01)
    dv_tol = max(10 * (dv_pt - dv_ref).abs().max().item(), 0.01)

    assert (dq - dq_ref).abs().max().item() <= dq_tol
    assert (dk - dk_ref).abs().max().item() <= dk_tol
    assert (dv - dv_ref).abs().max().item() <= dv_tol

    if attn_bias is not None:
        # print(f"dBias max diff: {(dbias - dbias_ref).abs().max().item()}")
        # print(f"dBias Pytorch max diff: {(dbias_pt - dbias_ref).abs().max().item()}")
        dbias_tol = max(10 * (dbias_pt - dbias_ref).abs().max().item(), 0.01)
        assert (dbias - dbias_ref).abs().max().item() <= dbias_tol

    return avg_mha_ck


@dataclass
class TestConfig:
    """
    Test configuration data class
    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., "Qwen3-32B", "Llama3-70B")
    attention_head : int
        Number of attention heads
    kv_head : int
        Number of the kv heads
    head_dim : int
        feature dimention per head
    intermediate_size : int
        feature dimention in MLP module
    is_moe : bool
        is the moe model or not
    """

    model_name: str
    attention_head: int
    kv_head: int
    head_dim: int
    intermediate_size: int
    is_moe: bool


MHA_CONFIG_DICT = {
    # model,                  model_name,   attention_head,   kv_head,   head_dim,  intermediate_size    is_moe
    "Qwen3-32B": TestConfig("Qwen3-32B", 64, 8, 80, 25600, False),
    "Qwen3-30B": TestConfig("Qwen3-30B", 16, 16, 128, 6144, True),
    "Qwen3-235B": TestConfig("Qwen3-235B", 32, 32, 128, 12288, True),
    "Llama3-70B": TestConfig("Llama3-70B", 64, 8, 128, 28672, False),
    "Llama3-405B": TestConfig("Llama3-405B", 128, 8, 128, 53248, False),
}


@dataclass
class Record:
    batch_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    TP: int
    seq_len_q: int = 1024
    seq_len_kv: int = 1024
    quant_type: str = "torch.bfloat16"
    output_type: str = "torch.bfloat16"
    latency: float = 0.0
    bandwidth: float = 0.0
    throughput: float = 0.0


class MHABenchmark:
    def __init__(self, dtypes=[torch.float16, torch.bfloat16]):
        self.dtypes = dtypes
        self.config_dict = MHA_CONFIG_DICT
        self.records = None
    
    def get_model_in_single_card(self, config):
        # for Qwen3-32B or Qwen3-30B, the dim is not dividable by 32 with tp 8, we skip this case
        if config.model_name == "Qwen3-32B" or config.model_name == "Qwen3-30B":
            return True
        return False
    
    def get_mha_shapes(self, TP_list):
        test_name = str(self.config_dict)
        logger.info(f"Running test: {test_name}")
        records = []
        for model_name, config in self.config_dict.items():
            logger.info(f"Collecting: {model_name}")
            for tp in TP_list:
                if self.get_model_in_single_card(config) and (tp == 8 or tp == 4):
                    continue
                num_heads = config.attention_head // tp
                num_kv_heads =  (config.kv_head + tp - 1) // tp
                head_dim = config.head_dim
                records.append(Record(
                    batch_size=1,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    TP=tp))
        self.records = records

    def benchmark_mha(self):
        assert self.records is not None
        records_result = []
        for dtype in self.dtypes:
            for idx, record in enumerate(self.records):
                latency = float("inf")
                bias_type = 'no'
                deterministic = False
                logger.info(f"Processing: {record}, {dtype}")
                ret = evaluate_mha(
                    record.batch_size,
                    record.num_heads,
                    record.num_kv_heads,
                    record.seq_len_q,
                    record.seq_len_kv,
                    record.head_dim,
                    True,
                    bias_type,
                    deterministic,
                    dtype)
                records_result.append(ret)
        return records_result


if __name__ == '__main__':
    runner = MHABenchmark()
    runner.get_mha_shapes([1])
    rets = runner.benchmark_mha()
    print(rets)
