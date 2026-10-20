"""Build an AITER-style model_configs.json for the SemiAnalysis AgentX (InferenceX agentic-coding) AMD matrix.

Inputs:
  - recipe_rows.json  (parsed from InferenceX/inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/*/*/mi3*/agentic.yaml)
  - ARCH below        (from each checkpoint's Hugging Face config.json, text_config where nested)
  - INFERENCEX_DIR    (clone of SemiAnalysis's InferenceX repo, default /tmp/InferenceX, used only for its commit hash)

Conventions follow ROCm/aiter op_tests/op_benchmarks/triton/utils/model_configs.json:
  - family keys and size keys contain no '-' or '_' (get_model_configs splits on them)
  - intermediate_size = fused gate+up N per rank = 2 * moe_intermediate_size / moe_tp
  - num_expert / top_k  = per-rank values under EP (E/ep, ceil(top_k/ep)); full values under TP
  - num_attention_heads / num_key_value_heads are full-model values (as in AITER); attn_tp gives the split
"""

import collections
import json
import math
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_CONFIGS_DIR = SCRIPT_DIR / "../../../utils/model_configs"
# Cloned outside the aiter tree (see gen_model_configs_agentx.sh); override with INFERENCEX_DIR.
INFERENCEX_DIR = Path(os.environ.get("INFERENCEX_DIR", "/tmp/InferenceX"))


def inferencex_commit_hash():
    return subprocess.check_output(
        ["git", "-C", str(INFERENCEX_DIR), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


ARCH = {
    # family-prefix: (hf checkpoint(s), arch dict)
    "qwen35": {
        "hf": ["Qwen/Qwen3.5-397B-A17B-FP8", "amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2"],
        "attention_type": "hybrid: 3x gated-deltanet linear attn + 1x full GQA (attn_output_gate)",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "num_hidden_layers": 60,
        "n_routed_experts": 512,
        "top_k": 10,
        "moe_intermediate_size": 1024,
        "n_shared_experts": 1,
        "shared_expert_intermediate_size": 1024,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
    },
    "kimik3": {
        "hf": ["moonshotai/Kimi-K3"],
        "attention_type": "hybrid: KDA linear attn (69 layers) + MLA full attn (24 layers)",
        "hidden_size": 7168,
        "num_attention_heads": 96,
        "num_key_value_heads": 96,
        "vocab_size": 163840,
        "num_hidden_layers": 93,
        "n_routed_experts": 896,
        "top_k": 16,
        "moe_intermediate_size": 3072,
        "n_shared_experts": 2,
        "dense_intermediate_size": 33792,
        "first_k_dense_replace": 1,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "linear_num_heads": 96,
        "linear_head_dim": 128,
    },
    "minimaxm3": {
        "hf": ["MiniMaxAI/MiniMax-M3-MXFP8", "amd/MiniMax-M3-MXFP4"],
        "attention_type": "GQA + block-sparse attention (layers 3+), qk-norm, partial rotary 64",
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 200064,
        "num_hidden_layers": 60,
        "n_routed_experts": 128,
        "top_k": 4,
        "moe_intermediate_size": 3072,
        "n_shared_experts": 1,
        "shared_expert_intermediate_size": 3072,
        "dense_intermediate_size": 12288,
        "first_k_dense_replace": 3,
        "sparse_num_index_heads": 4,
        "sparse_index_dim": 128,
        "sparse_block_size": 128,
        "sparse_topk_blocks": 16,
    },
    "dsv4pro": {
        "hf": ["deepseek-ai/DeepSeek-V4-Pro-0813"],
        "attention_type": "DSv4 MLA-style (single latent KV head, head_dim 512) + sparse indexer + SWA 128",
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "vocab_size": 129280,
        "num_hidden_layers": 61,
        "n_routed_experts": 384,
        "top_k": 6,
        "moe_intermediate_size": 3072,
        "n_shared_experts": 1,
        "kv_lora_rank": 1024,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 448,
        "qk_rope_head_dim": 64,
        "v_head_dim": 512,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 1024,
        "sliding_window": 128,
    },
    "dsv41flash": {
        "hf": ["deepseek-ai/DeepSeek-V4.1-Flash"],
        "attention_type": "DSv4 MLA-style (single latent KV head, head_dim 512) + sparse indexer + SWA 128",
        "hidden_size": 5120,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "vocab_size": 129280,
        "num_hidden_layers": 40,
        "n_routed_experts": 384,
        "top_k": 6,
        "moe_intermediate_size": 2304,
        "n_shared_experts": 1,
        "kv_lora_rank": 1024,
        "q_lora_rank": 1280,
        "qk_nope_head_dim": 448,
        "qk_rope_head_dim": 64,
        "v_head_dim": 512,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 512,
        "sliding_window": 128,
    },
    "glm52": {
        "hf": ["zai-org/GLM-5.2-FP8", "amd/GLM-5.2-MXFP4"],
        "attention_type": "MLA + DSA sparse indexer",
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 192,
        "vocab_size": 154880,
        "num_hidden_layers": 78,
        "n_routed_experts": 256,
        "top_k": 8,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 1,
        "dense_intermediate_size": 12288,
        "first_k_dense_replace": 3,
        "kv_lora_rank": 512,
        "q_lora_rank": 2048,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
    },
    "musespark": {
        "hf": ["amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4"],
        "attention_type": "GQA, 3x sliding-window (2048) + 1x full",
        "hidden_size": 8192,
        "num_attention_heads": 128,
        "num_key_value_heads": 16,
        "head_dim": 64,
        "vocab_size": 202048,
        "num_hidden_layers": 62,
        "n_routed_experts": 256,
        "top_k": 8,
        "moe_intermediate_size": 4096,
        "n_shared_experts": 0,
        "sliding_window": 2048,
    },
}

# recipe dir name -> arch key, recipe hw dir suffix -> precision tag
RECIPE_MODEL = {
    "qwen3.5": "qwen35",
    "kimik3": "kimik3",
    "minimaxm3": "minimaxm3",
    "dsv4": "dsv4pro",
    "dsv41flash": "dsv41flash",
    "glm5.2": "glm52",
    "musespark": "musespark",
}
SGLANG_DEFAULT_CHUNK = (
    16384  # used only where the recipe leaves chunked-prefill-size unset (Muse Spark)
)

KEEP_EXTRA = [
    "kv_lora_rank",
    "q_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "n_shared_experts",
    "shared_expert_intermediate_size",
    "dense_intermediate_size",
    "first_k_dense_replace",
    "sliding_window",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "full_attention_interval",
    "linear_num_heads",
    "linear_head_dim",
    "sparse_num_index_heads",
    "sparse_index_dim",
    "sparse_block_size",
    "sparse_topk_blocks",
]


def main(rows_path, out_csv):
    MODEL_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = (
        MODEL_CONFIGS_DIR / f"model_configs_semi_agentx_{inferencex_commit_hash()}.json"
    )
    with open(rows_path) as f:
        rows = json.load(f)
    groups = collections.OrderedDict()
    for r in rows:
        akey = RECIPE_MODEL[r["model"]]
        prec = r["hw"].split("-")[1]  # e.g. mi355x-fp4-mtp -> fp4
        prec = {"fp4": "mxfp4", "fp8": "fp8"}[prec]
        if akey in (
            "dsv4pro",
            "dsv41flash",
            "kimik3",
        ):  # these ship one checkpoint; fp4 experts
            prec = "fp4"
        W = r["gpus"]
        dp = max(1, r["dp"])
        if r["dpattn"] and r["engine"] == "atom":
            dp = W  # ATOM enable-dp-attention: one DP rank per GPU
        attn_tp = max(1, W // dp)
        ep = max(1, r["ep"])
        moe_tp = W // ep
        size = (
            f"tp{attn_tp}"
            + (f"dp{dp}" if dp > 1 else "")
            + (f"ep{ep}" if ep > 1 else "")
            + (f"dcp{r['dcp']}" if (r.get("dcp") or 1) > 1 else "")
        )
        fam = akey + prec
        g = groups.setdefault(
            (fam, size),
            {
                "akey": akey,
                "W": W,
                "dp": dp,
                "attn_tp": attn_tp,
                "ep": ep,
                "moe_tp": moe_tp,
                "pts": set(),
                "sources": set(),
            },
        )
        chunk = r["chunk"] or SGLANG_DEFAULT_CHUNK
        if r["engine"] == "sglang" and dp > 1:
            chunk //= dp  # sglang's chunked-prefill-size is global under DP attention; vLLM/ATOM budgets are per rank
        g["pts"].add((r["conc"], r["verify"], chunk))
        g["sources"].add(f"{r['hw'].split('-')[0]}/{r['engine']}")

    out = collections.OrderedDict()
    csv = [
        "family,size,hw_engine,gpus,attn_tp,attn_dp,ep,moe_tp,concurrency,verify_tokens,decode_M_per_dp_rank,prefill_chunk"
    ]
    for (fam, size), g in sorted(groups.items()):
        a = ARCH[g["akey"]]
        E, K = a["n_routed_experts"], a["top_k"]
        decode_M = sorted({math.ceil(c / g["dp"]) * v for c, v, _ in g["pts"]})
        prefill_M = sorted({ch for _, _, ch in g["pts"]})
        moe_decode_M = (
            sorted({c * v for c, v, _ in g["pts"]})
            if (g["dp"] > 1 and g["ep"] == 1)
            else None
        )
        e = collections.OrderedDict(
            hidden_size=a["hidden_size"],
            intermediate_size=2 * a["moe_intermediate_size"] // g["moe_tp"],
            num_attention_heads=a["num_attention_heads"],
            num_key_value_heads=a["num_key_value_heads"],
        )
        if "head_dim" in a:
            e["head_dim"] = a["head_dim"]
        e["vocab_size"] = a["vocab_size"]
        e["num_expert"] = E // g["ep"]
        e["top_k"] = math.ceil(K / g["ep"])
        e["n_routed_experts"] = E
        e["moe_intermediate_size"] = a["moe_intermediate_size"]
        for k in KEEP_EXTRA:
            if k in a:
                e[k] = a[k]
        e["attn_tp"] = g["attn_tp"]
        e["attn_dp"] = g["dp"]
        e["ep"] = g["ep"]
        e["moe_tp"] = g["moe_tp"]
        e["attention_type"] = a["attention_type"]
        e["decode_M"] = decode_M
        if moe_decode_M:
            e["moe_decode_M"] = (
                moe_decode_M  # DP attention + TP MoE: tokens are all-gathered before MoE
            )
        e["prefill_M"] = prefill_M
        e["silu_mul_benchmark_M"] = sorted(
            set(moe_decode_M or decode_M) | set(prefill_M)
        )
        e["concurrency"] = sorted({c for c, _, _ in g["pts"]})
        e["verify_tokens"] = sorted({v for _, v, _ in g["pts"]})
        e["hf_checkpoint"] = (
            a["hf"][0]
            if len(a["hf"]) == 1
            else next(
                h for h in a["hf"] if ("MXFP4" in h.upper()) == fam.endswith("mxfp4")
            )
        )
        e["agentx_sources"] = sorted(g["sources"])
        out.setdefault(fam, collections.OrderedDict())[size] = e
        for c, v, ch in sorted(g["pts"]):
            csv.append(
                f"{fam},{size},{'|'.join(sorted(g['sources']))},{g['W']},{g['attn_tp']},{g['dp']},{g['ep']},{g['moe_tp']},{c},{v},{math.ceil(c / g['dp']) * v},{ch}"
            )

    # compact JSON in the AITER style: one entry per line-ish
    with open(out_json, "w") as f:
        txt = json.dumps(out, indent=2)
        import re

        txt = re.sub(
            r"\[\s+([^\[\]{}]*?)\s+\]",
            lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]",
            txt,
        )
        f.write(txt + "\n")
    with open(out_csv, "w") as f:
        f.write("\n".join(csv) + "\n")
    for fam, sizes in out.items():
        print(fam, list(sizes))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
