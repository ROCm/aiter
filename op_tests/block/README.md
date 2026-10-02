# Attention block tests

End-to-end functional + perf tests for a whole model attention block (qkv_proj → attention → o_proj), wiring together the individual aiter kernels exactly as the serving stack (ATOM) dispatches them.

## test_attention_block.py — GPT-OSS

Runs OpenAI's GPT-OSS attention block end to end and checks it against a torch reference: `qkv_proj` → RoPE(YaRN) + paged KV write → attention (per-head sinks + alternating sliding window + GQA) → `o_proj`.

GPT-OSS's `head_dim=64` forces the triton/gluon path, so the kernels exercised are: `get_rope` (YaRN), `fused_qk_rope_reshape_and_cache` (RoPE + paged cache write), `flash_attn_varlen_func(sink_ptr=)` (prefill), and `pa_decode_gluon` (decode).

Two arch backends mirror ATOM's selector: `asm` (AiterBackend — `flash_attn_varlen_func` + `pa_decode_gluon`, gfx942/gfx950) and `triton` (TritonMHABackend / `use_flash_layout` — `unified_attention` for both phases, portable incl. gfx1250). `--backend auto` picks `triton` on gfx1250, else `asm`.

Validated on MI355X (gfx950), MI308X (gfx942) — both backends — and gfx1250 (triton backend). fp8 KV cache works on both backends (`dtypes.fp8` auto-selects e4m3fn / e4m3fnuz per arch): on `asm` it applies to decode (prefill reads K/V directly in bf16); on `triton` it applies to both phases (both read the paged cache), including gfx1250.

### Quick start

```bash
python3 op_tests/block/test_attention_block.py --model gpt-oss-120b --phase both
```

### Key flags

| Flag | Meaning |
|---|---|
| `--model` | `gpt-oss-120b` / `gpt-oss-20b` / `small` (same attention shape) |
| `--phase` | `prefill` / `decode` / `both` |
| `--batch` | sequence count = serving **concurrency** (list ok) |
| `--seqlen` | prefill prompt length (list ok) |
| `--ctx-len` | decode KV context length (list ok) |
| `--layer-parity` | `even` (sliding window) / `odd` (full causal) / `both` |
| `--kv-cache-dtype` | `bf16` / `fp8` (asm: decode only; triton: both phases) |
| `--backend` | `auto` / `asm` (gfx942/950) / `triton` (portable, incl. gfx1250) |
| `--hipgraph` | `auto` (off prefill, on decode) / `on` / `off` |

### Mapping a serving scenario to parameters

Concurrency = number of concurrently-running requests = the **decode batch size** (`--batch`); each decode step advances every running sequence by one token (`query_len=1`).

Prefill is per request (a prompt is prefilled when admitted, usually singly or chunked), so it is concurrency-independent: `--batch 1 --seqlen <input_len>`.

Decode context grows over generation from `input_len` to `input_len + output_len`; pass both ends to `--ctx-len` to bracket the cost.

GPT-OSS alternates sliding-window (even) and full-causal (odd) layers, so use `--layer-parity both`.

### Recipes: input/output length × concurrency

GPT-OSS-120B, concurrency sweep 1 / 16 / 64 / 128.

**8K / 1K** (input=8192, output=1024 → decode ctx 8192→9216):

```bash
# prefill: per-request cost (concurrency-independent)
python3 op_tests/block/test_attention_block.py --model gpt-oss-120b \
    --phase prefill --batch 1 --seqlen 8192 --layer-parity both

# decode: batch = concurrency; ctx at start (8192) and end (9216)
python3 op_tests/block/test_attention_block.py --model gpt-oss-120b \
    --phase decode --batch 1 16 64 128 --ctx-len 8192 9216 --layer-parity both
```

**1K / 1K** (input=1024, output=1024 → decode ctx 1024→2048):

```bash
python3 op_tests/block/test_attention_block.py --model gpt-oss-120b \
    --phase prefill --batch 1 --seqlen 1024 --layer-parity both

python3 op_tests/block/test_attention_block.py --model gpt-oss-120b \
    --phase decode --batch 1 16 64 128 --ctx-len 1024 2048 --layer-parity both
```

For fp8 KV (common for long context), add `--kv-cache-dtype fp8` to the decode commands.

| Scenario | Phase | length flag | `--batch` (concurrency) |
|---|---|---|---|
| 8K/1K | prefill | `--seqlen 8192` | `1` |
| 8K/1K | decode | `--ctx-len 8192 9216` | `1 16 64 128` |
| 1K/1K | prefill | `--seqlen 1024` | `1` |
| 1K/1K | decode | `--ctx-len 1024 2048` | `1 16 64 128` |

Note: `--batch N` on prefill models N prompts prefilled in one step (rare; schedulers usually chunk), and `batch=128 × seqlen=8192` is ~1M query tokens — heavy/possibly OOM. The decode test uses a uniform `--ctx-len` across the batch, so treat each value as one snapshot of continuous batching.
