import torch
import aiter
import logging
import argparse
from aiter import dtypes
from aiter.test_common import perftest
from dataclasses import dataclass, asdict


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
TEST_NUM_ITERS = 100


@perftest(num_iters=TEST_NUM_ITERS)
def dryrun_ck_perf(
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
    return_attn_probs=False,
):
    out, _, S_dmask = aiter.flash_attn_func(
        q,
        k,
        v,
        dropout_p,
        causal=causal,
        window_size=window_size,
        bias=bias,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_lse=return_lse,
        return_attn_probs=return_attn_probs,
    )
    return out


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
    dtype,
):
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
        requires_grad=False,
    )
    k = torch.randn(
        batch_size,
        seq_len_kv,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=False,
    )
    v = torch.randn(
        batch_size,
        seq_len_kv,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=False,
    )

    attn_bias = None
    alibi_slopes = None
    if bias_type == "bias":
        attn_bias = torch.randn(
            seq_len_q, seq_len_kv, device="cuda", dtype=dtype, requires_grad=True
        )
    elif bias_type == "alibi":
        alibi_slopes = torch.rand(
            batch_size, num_heads, device="cuda", dtype=dtypes.fp32
        )

    return_lse = True
    return_attn_probs = True
    dropout_p = 0.0
    window_size = (-1, -1)

    (out), avg_mha_ck = dryrun_ck_perf(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        None,
        dropout_p,
        causal,
        window_size,
        deterministic,
        return_lse,
        return_attn_probs,
    )

    return avg_mha_ck


@dataclass
class LLMConfig:
    model_name: str
    num_heads_q: int
    num_heads_kv: int
    head_dim_qk: int
    head_dim_v: int


SDPA_CONFIG_DICT = {
    # model_name, num_heads_q, num_heads_kv, head_dim_qk, head_dim_v
    "Llama3.1-70B": LLMConfig("Llama3.1-70B", 64, 8, 128, 128),
    "Qwen3-235B": LLMConfig("Qwen3-235B", 64, 4, 128, 128),
}


@dataclass
class SDPARecord:
    batch_size: int
    num_heads_q: int
    num_heads_kv: int
    head_dim_qk: int
    head_dim_v: int
    seq_len_q: int
    seq_len_kv: int
    dtype: str
    is_causal: bool = True
    latency_us: float = 0.0
    gbps: float = 0.0
    tflops: float = 0.0


class SDPAAnalyzer:
    def __init__(self, record: SDPARecord):
        self.batch_size = record.batch_size
        self.num_heads_q = record.num_heads_q
        self.num_heads_kv = record.num_heads_kv
        self.head_dim_qk = record.head_dim_qk
        self.head_dim_v = record.head_dim_v
        self.seq_len_q = record.seq_len_q
        self.seq_len_kv = record.seq_len_kv
        self.dtype = record.dtype
        self.is_causal = record.is_causal
        if record.dtype == "torch.bfloat16" or record.dtype == "torch.float16":
            self.element_bytes = 2
        elif record.dtype == "torch.float":
            self.element_bytes = 4
        else:
            raise "Data type not supported"
        assert self.num_heads_q % self.num_heads_kv == 0

    def getFLOP(self):
        L2 = self.seq_len_q * self.seq_len_kv
        if self.is_causal:
            n = min(self.seq_len_q, self.seq_len_kv)
            L2eff = (self.seq_len_kv + self.seq_len_kv - n + 1) * n / 2.0
        else:
            L2eff = L2
        BH = self.batch_size * self.num_heads_q
        qk_FLOP = 2 * BH * L2eff * self.head_dim_qk
        qksv_FLOP = 2 * BH * L2 * self.head_dim_v
        return qk_FLOP + qksv_FLOP

    def getFlashAttnBytes(self):
        read_q = self.batch_size * self.num_heads_q * self.seq_len_q * self.head_dim_qk
        read_k = (
            self.batch_size * self.num_heads_kv * self.seq_len_kv * self.head_dim_qk
        )
        read_v = self.batch_size * self.num_heads_kv * self.seq_len_kv * self.head_dim_v
        write_o = self.batch_size * self.num_heads_q * self.seq_len_q * self.head_dim_v
        return (read_q + read_k + read_v + write_o) * self.element_bytes


class MHABenchmark:
    def __init__(
        self,
        dtypes=[torch.float16],
        batch_sizes=[1, 8, 16, 32, 64, 128, 256],
        seq_len_q_kvs=[
            [1024, 1024],
            [4096, 4096],
            [10240, 10240],
            [1, 1024],
            [1, 4096],
            [1, 10240],
        ],
    ):
        self.records = None
        self.config_dict = SDPA_CONFIG_DICT
        self.dtypes = dtypes
        self.batch_sizes = batch_sizes
        self.seq_len_q_kvs = seq_len_q_kvs

    def get_mha_shapes(self):
        records = []
        for model_name, config in self.config_dict.items():
            logger.info(f"Collecting: {model_name}")
            for dtype in self.dtypes:
                for seq_len_q_kv in self.seq_len_q_kvs:
                    for batch_size in self.batch_sizes:
                        records.append(
                            SDPARecord(
                                batch_size=batch_size,
                                num_heads_q=config.num_heads_q,
                                num_heads_kv=config.num_heads_kv,
                                head_dim_qk=config.head_dim_qk,
                                head_dim_v=config.head_dim_v,
                                seq_len_q=seq_len_q_kv[0],
                                seq_len_kv=seq_len_q_kv[1],
                                dtype=str(dtype),
                            )
                        )
        self.records = records

    def benchmark_mha(self):
        assert self.records is not None
        records_result = []
        for _, record in enumerate(self.records):
            logger.info(f"Processing: {record}")
            latency = float("inf")
            bias_type = "no"
            deterministic = False
            ret = evaluate_mha(
                record.batch_size,
                record.num_heads_q,
                record.num_heads_kv,
                record.seq_len_q,
                record.seq_len_kv,
                record.head_dim_v,
                record.is_causal,
                bias_type,
                deterministic,
                eval(record.dtype),
            )
            time_us = float(ret)
            analyzer = SDPAAnalyzer(record)
            total_flop = analyzer.getFLOP()
            total_bytes = analyzer.getFlashAttnBytes()
            tflops = total_flop * 1e-6 / time_us
            gbps = total_bytes * 1e-3 / time_us
            record.latency_us = time_us
            record.gbps = gbps
            record.tflops = tflops
            records_result.append(record)
        return records_result


def save_mha_benchmark_result(records, csv_file_name):
    import csv
    from dataclasses import asdict

    csv_file = f"{csv_file_name}.csv"
    fieldnames = [
        "batch_size",
        "num_heads_q",
        "num_heads_kv",
        "head_dim_qk",
        "head_dim_v",
        "seq_len_q",
        "seq_len_kv",
        "dtype",
        "is_causal",
        "latency_us",
        "gbps",
        "tflops",
    ]
    with open(csv_file, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))
    print(f"data write to {csv_file} success!!")


def create_argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MHA bench results",
    )
    parser.add_argument(
        "-o",
        "--output_name",
        type=str,
        nargs="?",
        const=None,
        default="test_model_mha_results",
    )
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    runner = MHABenchmark()
    runner.get_mha_shapes()
    rets = runner.benchmark_mha()
    print("\n===== Results =====\n")
    for ret in rets:
        print(asdict(ret))
    save_mha_benchmark_result(rets, args.output_name)
