# GLM-5.3 IQ2R MoE on gfx950

`aiter/iq2r_glm53.py` runs the GLM-5.3 routed experts (256 experts, top-8, hidden 6144) from 2-bit IQ2R-v2
weights on MI355X/MI350X (gfx950), at TP4 (`inter_dim` 512) and TP8 (`inter_dim` 256):

- the IQ2R-v2 format with a 512-entry (`glm53-packed-v2`) or 1,024-entry (`glm53-packed-v3`) codebook;
- the shared expert kept 8-bit: stored block FP8 and, for batches of up to 256 tokens, run inside the IQ2R
  launches as MXFP8;
- a recipe with LDLQ on gate/up, using layer Hessians from the calibration corpus and a decision corpus.

IQ2R was developed by the MK1 team at AMD. Its learned codebooks are seeded from the IQ2_XXS grid of
ggml/llama.cpp (MIT).

## Format

`aiter/ops/iq2r_v2.py`. Each group of 8 weights is a 9-bit index into a learned 512-entry E4M3 codebook (one
per expert for gate/up, learned on the interleaved gate0, up0, gate1, ... rows, and one for down) plus 8
explicit sign bits. The scales:

| | |
|---|---|
| per row and 256 inputs | E8M0 `d` |
| per 32 weights (4 bits) | level `l`: `m[l % 4] * 2^(l / 4)`, `m = 2^(j/4)` |
| codewords | `E4M3(codebook * m[c])`, four pre-scaled copies |
| per projection | FP32 output gain (gate and up separately) |
| bits per weight | ~2.28 |

The power-of-two parts (`d` and `2^(l / 4)`) form the scaled-MFMA E8M0
operand, the mantissa `m[c]` selects a codebook copy in LDS, and the gain is
applied once per output, so there is no per-weight arithmetic. The
copies' E4M3 rounding is part of the format: the encoder, the reference
decoder (`iq2r_v2_materialize`) and the kernels agree bit for bit. Gate and
up share one codebook; on six GLM-5.3 experts, separate gate and up
codebooks lowered the weighted weight error by only 0.09%, for twice the
codebook memory.

In a lane group `g`, the metadata high nibble holds the exponent of block
`g`, whose MFMA scale that lane supplies, and the class of block `s(g)`,
`s = (0, 2, 1, 3)`. The classes a lane needs for its own codewords are then
in its own nibble or its neighbour group's, one `v_permlane16_swap` away.

`iq2r_glm53_pack` turns the per-expert records into one packed stack per
layer (the four 9-bit indices of an atom pack as low32 + high4, the gate/up
record signs transpose into bit planes, and the gate/up records regroup into
quads of four 16-column blocks). The packed auxiliary holds the codebook copies, the `d` table laid out per lane
(`[N/64][K/256][16][4]` for gate/up, `[N/48][K/256][16][4]` for down) and
the two gains. Every transform is per record, so a packed checkpoint is cut
to any supported TP rank with contiguous slices (`iq2r_glm53_slice_gate`,
`iq2r_glm53_slice_gate_aux`, `iq2r_slice_input_data`,
`iq2r_glm53_slice_down_aux`); `iq2r_glm53_unpack` inverts the packing.

### 1,024-entry codebook (`glm53-packed-v3`)

Everything above holds; the codebook has 1,024 entries and indices 10 bits
(`codebook_entries=1024` in the writers, readers and `iq2r_v2_materialize`):

- Seed: the IQ2_XXS grid at `2^(j/4)`, j = 0..3 (256 entries each, scaled from
  the E4M3-rounded grid); the learner keeps entries in bytes 0x08..0x78.
- Format name `iq2r-1024-fullsign-c4e4q-e8m0d256-native-v2`, packed layout
  `glm53-packed-v3`. 512-entry checkpoints keep `glm53-packed-v2`, and every
  512-entry byte is unchanged.
- Records: each 8-weight index gains a tenth bit, four per atom and lane. Atoms
  0 and 1 of a triplet keep theirs in the atom-two record's fourth metadata
  byte (padding in the 512-entry layout); atom 2's go to one byte per lane after
  every two K tiles, so records are stored in slabs of two K tiles (256 inputs,
  the tensor-parallel slicing unit): 7,296 instead of 7,168 bytes per down slab.
- Packed gate/up quads: an atom's packed indices are
  `i0 | i1 << 10 | i2 << 20 | i3 << 30` (40 bits); each six-tile K slice adds a
  12-byte word per lane after its six tiles (+5.6%).
- About 2.41 bits per weight against 2.28 (+4.3% expert bytes).

The packers infer the codebook size from the auxiliary size; pack, unpack and
TP slicing round-trip bit for bit for both sizes.

## Building the checkpoint

Two tools turn the block-FP8 GLM-5.3 checkpoint into the packed checkpoint
ATOM serves. They need `safetensors` (`pip install safetensors`, or the `iq2r`
extra), and calibration needs Hugging Face Transformers with GLM MoE DSA
support. The `kernels` package provides the FP8 matmul
(`kernels-community/finegrained-fp8`); without it, each block-FP8 weight is
dequantized to BF16 for its matmul.

`calibrate.py` patches private Transformers internals (the BF16 fallback,
`--fp8-kernel-dir`, and a workaround for 5.16), so it is tested with one
version only: Transformers 5.16.1. Versions used for the documented 1,024-entry
LDLQ checkpoint:

| Step | Versions | FP8 matmul |
|---|---|---|
| corpus replies | ATOM serving the FP8 checkpoint behind its OpenAI-compatible endpoint; ATOM version not recorded | ATOM |
| importance statistics | an earlier version of `calibrate.py` (with a `--per-row` switch, now always on); package versions not recorded | Transformers' Triton FP8 path (from its log) |
| layer Hessians (`H_all`), `compile` | Transformers 5.16.1, PyTorch 2.11.0 on ROCm (HIP 7.15), no `kernels` package | dequantized to BF16 |
| decision Hessians | a capture script outside aiter (see the published recipe below); Transformers 5.16.1, PyTorch 2.11.0, HIP 7.15 | Transformers' Triton FP8 path (`FineGrainedFP8Config(dequantize=False)`), BF16 activations |
| Python standard-library prompts | Python 3.12.3 (Ubuntu's `/usr/lib/python3.12`) | - |

That earlier `calibrate.py` also stored shared-expert statistics. `compile`
reads but does not use them; they are part of the calibration hash the layer
files record. The current `calibrate.py` does not capture them, so rerunning
it gives a different calibration hash and slightly different importance (the
FP8 matmul path differs too).

    # router-weighted importance, plus layer Hessians for LDLQ
    python3 -m aiter.quantization.iq2r.glm5.calibrate --model-dir $FP8 \
        --text-file corpus.jsonl --output calibration.pt \
        --hessian-dir hessians --decision-text-file decisions.jsonl

    # one process per GPU, each on its own MoE layers (3..77) ...
    SLICES=(3-12 13-22 23-32 33-42 43-51 52-60 61-69 70-77)
    for i in 0 1 2 3 4 5 6 7; do
      HIP_VISIBLE_DEVICES=$i python3 -m aiter.quantization.iq2r.glm5.compile \
          --model-dir $FP8 --output-dir $OUT --calibration-cache calibration.pt \
          --codebook-entries 1024 --ldlq --hessian-dir hessians \
          --layers ${SLICES[$i]} &
    done; wait
    # ... then one run over all layers writes the shards, index and config
    python3 -m aiter.quantization.iq2r.glm5.compile --model-dir $FP8 \
        --output-dir $OUT --calibration-cache calibration.pt \
        --codebook-entries 1024 --ldlq --hessian-dir hessians

Without `--codebook-entries 1024` the compiler builds 512-entry
`glm53-packed-v2` checkpoints; without `--ldlq` it skips the LDLQ pass, and
`calibrate` then needs neither `--hessian-dir` nor `--decision-text-file`.

Calibration runs the FP8 model over the corpus (one conversation per
sequence, up to 16,384 tokens each and 1M in total) and takes about 25
minutes on 8 MI350X; the Hessian capture adds a pass over each corpus. An
expert that no corpus token reaches gets one forced pass over up to 512
(`FORCE_UNOBSERVED_TOKENS`) of its layer's input tokens, which the artifact
records apart from the routed counts. On a host without Hugging Face Hub
access, `--fp8-kernel-dir` loads a local build of
`kernels-community/finegrained-fp8`.

Encoding takes about 6 s per expert (gate/up and down) on one GPU with 512
entries, 9 s with 1,024, and about 30 minutes per MoE layer with LDLQ (encode,
LDLQ, checks). Layer files already in `$OUT` that were built from the same
calibration and recipe are kept, so the last run only checks them, copies the
remaining FP8 tensors (the shared expert and MTP layer 78 included) into new
shards and writes the index, config and tokenizer. The checkpoint directory is
240.8 GB (224.3 GiB) with 512 entries and 250.3 GB (233.1 GiB) with 1,024
entries, with or without LDLQ (GB = 10^9 bytes, GiB = 2^30 bytes; the FP8
source is 755.7 GB).

The commands above are the general use. The exact steps, settings and hashes
of the published 1,024-entry LDLQ checkpoint are under "Recipe of the published
checkpoint" below; its decision bucket was not computed by `calibrate`.

## Recipe

Steps 1-6 are the code's defaults. The 1,024-entry codebook
(`--codebook-entries 1024`; the default is 512) and LDLQ (step 7, `--ldlq`;
off by default) are opt-in. The documented checkpoint uses both; the commands
under "Building the checkpoint" run the same method, and "Recipe of the
published checkpoint" lists where its build differs. The method does not depend on GLM-5.3
and is not tuned to one benchmark; the compiler itself targets GLM-5.3's
shapes.

1. **Only the routed experts are 2-bit; the shared expert stays 8-bit.**
   - What: the 256 routed experts of each MoE layer are encoded. The shared
     expert keeps its block-FP8 weights in the checkpoint shards, and
     `iq2r_modules` lists only the routed experts. At serving time it runs
     inside the IQ2R launches as MXFP8 (see Kernels), or as an ordinary FP8
     MLP for larger batches.
   - Why: each token picks 8 of the 256 routed experts, so a routed expert
     sees about 1 token in 32, and its output is scaled by the routing
     weight. Every token passes through the shared expert, with weight 1, so
     the shared expert's quantization error lands on every token in every
     layer. It is also small: keeping it FP8 costs about 2 GB on GLM-5.3
     (75 layers x 37.7 M weights x ~0.7 byte that 2 bits would save).
   - Measured: with everything else the same, the shared expert at 2 bits
     scored 49 and 50 of 60 on AIME-60, against 54 and 54 in FP8.
   - Serving runs the MoE on the 256 routed experts with top-8 routing:
     `iq2r_glm53_moe_out` takes `[256, ...]` stacks with `[tokens, 8]` ids and
     weights (see Kernels).
2. **Calibration: router-weighted statistics, one conversation per sequence.**
   - What: `calibrate.py` runs the FP8 model over the calibration
     corpus below, each row (one whole conversation, rendered with the chat
     template) its own sequence of up to 16,384 tokens. For every routed
     expert it records `sum r^2 x^2 / sum r^2` per input channel: x is the
     MoE input for gate/up and `silu(gate) * up` for down, r the token's
     routing weight to that expert.
   - Why per row: as in serving, each token attends only to its own
     conversation, and long agent sessions keep their full context. Packing
     unrelated texts into short windows would give activations the model
     never produces in use.
   - Why router-weighted: an expert's output is multiplied by r, so a
     token's error in that expert reaches the layer output scaled by r, and
     its squared error by r^2. Tokens the router barely sends to the expert
     should barely shape its importance.
3. **Importance: normalized, with a floor and a fallback.**
   - What: that moment divided by its mean, floored at 1e-6. An expert that
     fewer than 256 corpus tokens reach (3 of 19,200 on GLM-5.3) uses its
     weight's column energy `sum_n W[n, k]^2` instead.
   - Why: dividing by the mean keeps the cost on the same scale for every
     matrix; the floor keeps every channel in the cost. With under 256
     tokens the moment is mostly noise from a few tokens, and column energy
     is a data-free stand-in.
4. **Codebook: learned per matrix, from scratch.**
   - What (`iq2r_v2_learn_codebook`): one codebook per expert for the
     interleaved gate/up matrix and one for down. A 512-entry codebook is
     seeded with the IQ2_XXS grid at 1x and sqrt(2)x; a 1,024-entry one with
     the E4M3-rounded grid at `2^(j/4)`, j = 0..3 (256 entries each). Then up
     to 4 rounds of importance-weighted k-means on 65,536 sampled 8-vectors
     (2,048 random 256-weight blocks), alternating with the encoder: each
     round encodes the sample, then runs 4 Lloyd steps at the sample's fixed
     scales. Codewords stay at or below the seed maximum. A round is kept only
     if the weighted cost of the whole matrix drops; the first round that does
     not ends the learning. The sample is drawn with a fixed per-matrix seed,
     `(0x10A0 + 1024 * layer + 2 * expert + is_down) * 2`.
   - Why: each matrix's weights have their own distribution, and a codebook
     fitted to it lowers the error a fixed grid leaves. Starting from the
     public IQ2_XXS grid keeps the recipe independent of any previous
     checkpoint. Judging each round on
     the whole matrix stops k-means from overfitting its sample. Gate and up
     share a codebook because separate ones lowered the weighted error by
     only 0.09% for twice the memory.
5. **Encoder: weighted error, exhaustive scale search.**
   - What (`iq2r_v2_encode`, after the IQ2_XXS quantizer of ggml/llama.cpp
     as adapted by antirez): each weight's squared error is weighted by
     `importance * sqrt(sigma^2 + w^2)` (sigma^2 the mean square of its
     256-weight block). A group's starting scale comes from 13 candidate
     least-squares fits; then `d` (three exponents) and every group's level
     (all 16 of c4e4q) are searched exhaustively, re-assigning codewords
     for each choice.
   - Why: importance puts the error where activations are large;
     `sqrt(sigma^2 + w^2)` protects the large weights of a block more than
     its small ones. The search space is small (3 x 16 choices per group),
     so searching it exactly costs little and finds the best scales the
     format can store.
6. **Gain: a slight output shrink.**
   - What: each matrix's output is scaled by one FP32 gain so that
     `<W, Wq> / <W, W> = 0.98` (separate gains for the gate and the up
     rows), applied once per output in the kernel.
   - Why: quantization leaves Wq slightly too large or too small as a
     whole, which one number per matrix corrects. Aiming a little below 1
     follows least squares: when the reconstruction carries noise, scaling
     it slightly down lowers the expected output error. 0.98 was chosen by
     KL divergence on GLM-5.3.

7. **LDLQ on gate/up (with `--ldlq`).**
   - What: after the normal encode, gate/up are re-encoded with LDLQ: the
     Cholesky factor of `H^-1` once per layer, error feedback over K in
     8-weight vectors (128-column lazy batches), each vector keeping its scale
     and codebook copy; gains refitted (0.98). `H = H_all/tr + H_decision/tr`
     (decision weight 1.0), plus `1.0 * mean(diag(H)) * I` damping; FP64
     Cholesky. Down is not re-encoded.
   - `H_all = mean x x^T` over the scored positions of the calibration corpus
     (positions that predict an assistant token, or the end-of-turn token
     right after assistant text). `H_decision` is the same over the positions
     of the decision corpus where the FP8 model's top-1 next token is end of
     turn (`<|endoftext|>`, `<|user|>`, `<|observation|>`), `</think>` or
     `<tool_call>`. Rows with a `kind` give one Hessian per kind; the decision
     bucket is their mean after each is divided by its trace, weighted by the
     kind's token count (tokens fed, after the 16,384-token cut). The
     published checkpoint used fixed weights and its own capture instead (see
     "Recipe of the published checkpoint").
   - Why: the diagonal importance of step 3 ignores correlations between
     input channels; LDLQ uses them, so the error of one vector is
     compensated in the next. The decision bucket puts extra weight on the
     positions where the model decides to stop, hand over or call a tool,
     where 2-bit errors showed up as longer thinking and missed task ends.
   - Measured: on 18 held-out gate/up matrices, LDLQ with the decision bucket
     lowered the relative output error to 0.63 of plain encoding over all
     evaluation positions and to 0.50 at decision positions.
   - It changes only codeword indices and signs: no bits, no kernel change.

The layer files record the calibration hash, the Hessians' digest and every
setting above, so a compile run only keeps existing layer files built the
same way.

### Calibration corpus

A general mix of 164 rows, 266,396 tokens: conversations, agent sessions and
plain prose. Every reply is generated by GLM-5.3 (FP8) and rendered with the
serving chat template, so the captured activations are the ones the model
produces in use. The mix covers agent work, chat, reasoning, code, other
languages, long context and plain prose, so no single use dominates the
importance.

| Share | `kind` | Rows | Tokens | Source |
|---:|---|---:|---:|---|
| 24.5% | `agent` | 10 | 65,251 | 5 mini-swe-agent trajectories on SWE-bench Verified instances not in SWE-bench Lite (first 8,192 tokens each); 5 Terminal-Bench 1.0 sessions (1,468-7,520 tokens, whole) |
| 19.7% | `chat` | 48 | 52,364 | 24 topics x 2 conversations (thinking on, thinking off), each with one follow-up turn |
| 18.5% | `reasoning` | 16 | 49,324 | thinking on: 8 MATH-500 level 4-5 problems, 8 SuperGPQA questions |
| 12.3% | `code` | 32 | 32,784 | explain, review, test or refactor Python standard-library functions; write programs |
| 9.6% | `multilingual` | 40 | 25,548 | 5 tasks in each of zh, es, fr, de, ja, ko, hi, ar |
| 9.4% | `long` | 2 | 25,037 | questions on a long WikiText-2 excerpt and on a full source file |
| 6.0% | `prose` | 16 | 16,088 | WikiText-2 train windows, no reply |
| | total | 164 | 266,396 | |

Tokens are counted with the GLM-5.3 tokenizer, no special tokens added. The
kinds were concatenated as they came out, with no reweighting: a kind's share
is its token count. The corpus has no AIME, GPQA or SWE-bench Lite text. Its 5
terminal sessions are Terminal-Bench 1.0 tasks (`terminal-bench-core` 0.1.1):
csv-to-parquet, fix-permissions, heterogeneous-dates,
organization-json-generator and sqlite-db-truncate. None is one of the three
Terminal-Bench 2.0 tasks evaluated below, but Terminal-Bench 2.0 also contains
sqlite-db-truncate, so a full Terminal-Bench 2.0 run overlaps the corpus on
that task.

The corpus is not published with aiter. It is one JSONL file: 164 rows, one
`{"kind", "id", "text"}` object per row, 1,083,120 bytes, sha256
`1aab8c266a7a062ece43b5ae2b2a511e9344459f93370faf53a0bcffeb891bc8`. Its
calibration artifact and Hessians record
`corpus_sha256 = 1a1bc46c4e157c8835001ab83636bf009c4d5bccaedb1a06b85f7fe48ad81f8a`
(the hash `calibrate.py` takes over the row texts). Any corpus with a similar
mix can stand in for it. The replies are sampled, so a rebuild with the steps
below gives a comparable corpus, not the same bytes.

#### How it was built

Rows are written in this order: the generated rows in job order (reasoning,
code, chat, multilingual, long), then prose, then the 5 SWE-bench rows and the
5 Terminal-Bench rows (each sorted by file path). Row ids:
`math:<MATH-500 unique_id>`, `supergpqa:<uuid>`, `code:<file>:<k>`,
`codew:<k>`, `chat:<topic>:<0|1>`, `ml:<language>:<k>`, `long:wiki`,
`long:argparse`, `wiki:<first row>`, `swe:<instance>`, `tbench:<task>`.

Prompts. One `random.Random(20261104)` draws everything, in this order:

1. Reasoning: `rng.sample` of 8 of the 262 MATH-500 test problems of level 4
   or 5, in dataset order; `rng.sample` of 8 of the `m-a-p/SuperGPQA` train
   rows with difficulty "hard" (revision `4430d445`), each written as the
   question, the options as `A. ...`, `B. ...`, and "Answer with the letter
   of the correct option."
2. Code: the top-level `*.py` files of the Python 3.12.3 standard library,
   sorted, then `rng.shuffle`d. In each file, one top-level function or class
   whose `end_lineno - lineno` is 15 to 60 is picked with `rng.choice`; the
   first 24 files that have one are used. Snippet k gets task k % 4: "Explain
   what this Python code does, step by step." / "Review this code: point out
   bugs, edge cases and possible improvements." / "Write pytest unit tests for
   this code." / "Refactor this code for readability and explain your
   changes.", followed by the snippet in a Python code fence. Then 8 fixed
   write-a-program tasks: an ISO-8601 duration parser with tests (Python), an
   LRU cache without functools (Python), the k most frequent words (Rust),
   customers whose spending grew three months in a row (SQL), a thread-safe
   bounded queue (C++), a log rotation script (bash), a JSON deep merge
   (TypeScript), Dijkstra's algorithm (Go). In both lists, even k is thinking
   on.
3. Chat: 24 fixed everyday prompts (for example "Plan a 5-day trip to Japan
   for a family with two kids on a moderate budget." and "Explain inflation
   and why central banks raise interest rates."). Topic k runs twice, v = 0
   with thinking on and v = 1 off, each with the follow-up `(k + v) % 2` of
   "Can you make that shorter and more concrete?" and "Thanks. What are the
   most common mistakes people make here?".
4. Multilingual: 5 fixed everyday tasks written in each language (zh, es, fr,
   de, ja, ko, hi, ar); tasks 0, 2 and 4 thinking on.
5. Long context, thinking off: from a start row `rng.randrange(0, rows - 2000)`
   of WikiText-2 train, 600 rows joined by blank lines, first 60,000
   characters, then "Based only on the text above: list the main topics it
   covers, then answer: which people or places are mentioned most often, and
   in what context?"; and the first 55,000 characters of Python 3.12.3's
   `argparse.py` in a code fence, then "Describe the architecture of this
   module: the main classes, how parsing flows, and how subparsers and help
   formatting work."
6. Prose, after generation: 16 times, a start row `rng.randrange(0, rows - 50)`,
   50 rows joined by blank lines, the first 1,024 tokens decoded back to text
   (one window has only 728 tokens).

WikiText-2 is `Salesforce/wikitext`, config `wikitext-2-raw-v1`, train split,
with empty rows dropped and the rest stripped: 23,767 rows. The full lists of
24 chat prompts and 40 multilingual tasks are not published.

Replies. Each request goes to `/v1/chat/completions` with model
`zai-org/GLM-5.3`, temperature 1.0 and top_p 0.95. Thinking and answer share
`max_tokens`:

| `kind` | Thinking | `max_tokens` per reply |
|---|---|---|
| reasoning | on | 6,000 |
| code | on for even k | 1,500 (4,500 when off) |
| chat | on / off | 700 (3,700 when off) per turn |
| multilingual | on for tasks 0, 2, 4 | 700 (3,700 when off) |
| long | off | 4,500 |

The GLM-5.3 template has no thinking switch, so the model always reasons, and
the server returns the reasoning as `reasoning_content`. A thinking-on reply
is stored as `<think>{reasoning}</think>{answer}`. A thinking-off reply keeps
only the answer (the template then writes `<think></think>`); its extra 3,000
tokens pay for the discarded reasoning. The finish reason was not checked:
replies cut by the budget were kept. 73 of the 148 rows with replies end in a
final turn with no answer: code 30 of 32, chat 21 of 48, multilingual 15 of
40, reasoning 5 of 16, long 2 of 2. A thinking-on reply cut while reasoning
keeps its partial reasoning, closed by the `</think>` the generator adds. A
thinking-off reply whose budget ran out while reasoning is empty: both long
replies, 14 code replies, 4 final and 2 first chat turns. The server's
settings (parallelism, KV cache type) were not recorded.

Rendering. `tok = AutoTokenizer.from_pretrained(<FP8 checkpoint>)`, with
`tok.chat_template` set to the published 2-bit checkpoint's
`chat_template.jinja`. That template differs from the FP8 checkpoint's only in
how it orders several tool results of one turn; both render these rows the
same. Each row is `tok.apply_chat_template(messages, tokenize=False)` with no
other arguments:

- every row starts with `[gMASK]<sop><|system|>Reasoning Effort: Max` (the
  template's default effort);
- earlier turns keep their reasoning (`clear_thinking` is false by default);
- no generation prompt and no end-of-turn token are added, so no row ends
  with one.

Agent sessions:

- SWE-bench: mini-swe-agent 2.4.5 with its SWE-bench runner and bash tool on
  `princeton-nlp/SWE-bench_Verified` (test): astropy__astropy-13398,
  django__django-16263, matplotlib__matplotlib-25775, pydata__xarray-7229 and
  pylint-dev__pylint-8898, none in SWE-bench Lite. How these 5 were chosen is
  not recorded. Settings: temperature 0.95, top_p 1.0, max_tokens 32,768 per
  call, step limit 150, 4 hours per instance; all 5 submitted a patch. A row
  is the trajectory's system, user, assistant and tool messages (with
  `reasoning_content`, `tool_calls` and `tool_call_id`; tool-call arguments
  decoded from JSON to objects), rendered with
  `tools=[bash(command: string)]` and cut to its first 8,192 tokens.
- Terminal-Bench: terminal-bench harness 0.2.18 with the terminus-2 agent at
  its defaults (temperature 0.7, no episode limit), `terminal-bench-core`
  0.1.1, one attempt per task. fix-git was also run, but its image failed to
  build. A row is the last episode's request messages plus that episode's
  response as the final assistant turn, rendered without tools. terminus-2
  replies are JSON keystrokes, and its log keeps no reasoning. The 8,192-token
  cap applied, but no session reached it.

#### Building a comparable corpus

File format. `--text-file` takes one row per nonempty line. A line starting
with `{` is JSON and its `text` (else `prompt`, else `content`) field is the
row; any other line is the row as is. Each row is tokenized exactly as
written, with no special tokens added, and fed as its own sequence of up to
16,384 tokens.

Render with the chat template first. `calibrate.py` does not apply a chat
template. For conversations, write each whole conversation already rendered
with the checkpoint's `chat_template.jinja` (role tags and `<think>` blocks
included), for example
`tokenizer.apply_chat_template(messages, tokenize=False)`. Raw prompt and
reply text gives different activations from the ones the model sees in
serving.

End each conversation with its end-of-turn token. With
`add_generation_prompt=False`, the GLM-5.3 template stops right after the
last reply's text (or its last `</tool_call>`) and writes no end-of-turn
token. GLM-5.3 ends a turn with `<|user|>`, `<|observation|>` or
`<|endoftext|>` (its `eos_token_id`). A position is scored only if the token
it predicts is in the row, so without that token the last stop decision of
every row is lost, both for `H_all` and for the decision bucket. Append
`<|user|>` after a finished last reply (the token that follows a reply in
serving), or `<|observation|>` if the last reply ends with a tool call. Append
nothing after a reply cut by a token budget. The decision corpus below appends
`<|user|>` to its generated rows; the calibration corpus above does not.

Check the finish reason. The calibration corpus kept cut replies, and half of
its rows end without an answer. That still gives the importance statistics
real activations, but those rows carry no stop decision. Give replies enough
budget to finish, as the decision corpus does.

Plain prose rows need no template. They count for the importance statistics,
but add nothing to `H_all`: it only uses positions after an `<|assistant|>`
tag.

Keep the rows away from any benchmark you will evaluate on; this corpus
leaves out SWE-bench Lite, AIME and GPQA.

Size. About 160 rows and 250-300K tokens leaves most of the 19,200 routed
experts with at least 256 routed tokens. This corpus reaches every routed
expert (at least 14 tokens), and 3 experts fall below 256 and use the
weight-energy fallback. A much smaller corpus moves more experts onto the
fallback. `calibrate.py` stops at 1M tokens.

### Decision corpus

The decision corpus (`--decision-text-file`) is a separate set of rendered
conversations, kept apart from the calibration corpus and from the
evaluation sets; only its decision positions are used. It has the same row
format as the calibration corpus, with a `kind` per row (`{"kind", "text"}`;
a missing `kind` counts as one kind). It is not published; `calibrate`
accepts any corpus in this format. The file has 268 rows
(`{"kind", "id", "text"}`), 5,171,701 bytes, sha256
`f1c60e0f156c34d38e5cc1ffb89c589c41bd421375c3613d4672e0b9ef21ecd7`
(`calibrate`'s text hash
`b859267a9717bd7fde56c0c0bd6a67ed3a82159a12b2ed05200e16f9acc33504`). Rows are
grouped by kind, in the order of the table below.

Decision positions are the scored positions (as for `H_all`) where the FP8
model's top-1 next token is end of turn (`<|endoftext|>`, `<|user|>`,
`<|observation|>`), `</think>` or `<tool_call>`. They are found by running the
FP8 model over the row, so no labels are needed.

Weighting. `calibrate` builds one decision Hessian per kind, divides each by
its trace and averages them. `--decision-weights kind=w,...` sets the weights
directly (one per kind, normalized). Without it, each kind is weighted by its
token count in the file (tokens fed, after the 16,384-token cut per row); the
number of decision positions does not count, so then size each kind to its
target share. The
documented build targets the calibration corpus's shares without prose (prose
has no replies, so no decisions), renormalized: agent 24.5 / 94.0 = 26.1%,
split evenly into SWE and terminal agents; chat 21.0%; reasoning 19.7%, split
evenly into math and science; code 13.1%; multilingual 10.2%; long context
10.0%. The published checkpoint did not take the weights from token counts:
it used the fixed weights in the table (their sum, 0.996, is normalized to 1),
which `--decision-weights` reproduces.

Long rows. A decision row longer than 16,384 tokens is cut. By default
(`--decision-keep end`) `calibrate` keeps its last 16,384 tokens, so the
row's last reply and its stop decision are measured; `--decision-keep start`
keeps the first 16,384 instead. The calibration corpus (`--text-file`) always
keeps the first 16,384.

| `kind` | Rows | Tokens in file | Tokens captured | Decision positions | Weight used | Source |
|---|---:|---:|---:|---:|---:|---|
| `swe_agent` | 21 | 172,032 | 172,032 | 1,360 | 0.130 | agent trajectories on SWE-bench Verified instances not in SWE-bench Lite |
| `terminal_agent` | 41 | 170,207 | 170,207 | 145 | 0.130 | terminus-2 sessions on LiteCoder-RL (18 rows), Aider-Polyglot (21) and SWE-bench Verified non-Lite (2) tasks |
| `math_reasoning` | 16 | 131,072 | 131,072 | 222 | 0.098 | FP8 reasoning traces on MATH-500 level 4-5 problems |
| `chat` | 66 | 272,008 | 259,158 | 196 | 0.208 | generated: UltraChat-200k first user turns, each with its second user turn (or a fixed follow-up) |
| `code` | 35 | 167,753 | 154,954 | 68 | 0.130 | generated: 9 Python standard-library functions or classes (explain / review / write tests / refactor), 26 Magicoder-OSS-Instruct problems |
| `multilingual` | 59 | 133,426 | 127,917 | 110 | 0.102 | generated: Aya dataset prompts in zh, es, fr, de, ja, ko, hi, ar, pt, ru |
| `long_context` | 7 | 128,488 | 107,828 | 13 | 0.100 | generated: questions on 4 WikiText-2 train excerpts and 3 Python standard-library modules (up to 45,000 characters each) |
| `science_reasoning` | 23 | 133,693 | 106,144 | 35 | 0.098 | generated: SuperGPQA (train, difficulty "hard") multiple-choice questions |
| total | 268 | 1,308,679 | 1,229,312 | 2,149 | 0.996 | |

"Tokens captured" is what the published capture fed (see below). Of the 2,149
positions, 749 predict end of turn, 1,068 `</think>` and 332 `<tool_call>`
(all from `swe_agent`). Decision positions are counted on the FP8 model. Long
context and science have few; more rows would firm them up.

Generated kinds (`chat`, `code`, `multilingual`, `long_context`,
`science_reasoning`). One `random.Random(20261119)` draws the prompts.
UltraChat, Magicoder and SuperGPQA are streamed in their stored order; the
first 4,000 rows that qualify are shuffled and the first n taken:

| `kind` | Requests | Prompt source (Hugging Face, revision) | Selection |
|---|---:|---|---|
| `chat` | 91 | `HuggingFaceH4/ultrachat_200k` `train_sft` (`8049631c`) | first user turn (first 4,000 characters), then the second user turn if it exists and is under 1,500 characters, else one of 4 fixed follow-ups |
| `code` | 67 | Python 3.12.3 standard library, as in the calibration corpus (33); `ise-uiuc/Magicoder-OSS-Instruct-75K` `train` (`5f839b1f`) (34) | stdlib snippet with task k % 4; Magicoder `problem` (first 6,000 characters) |
| `multilingual` | 70 | `CohereForAI/aya_dataset` `train` (`f9ea0458`) | 7 per language code zho, spa, fra, deu, jpn, kor, hin, arb, por, rus: that language's rows shuffled, `inputs` of 21-1,999 characters |
| `long_context` | 8 | WikiText-2 train as above; the 60 largest top-level standard-library modules | even k: 500 rows from a random start, first 45,000 characters; odd k: a random module, first 45,000 characters; each with one of two fixed questions |
| `science_reasoning` | 35 | `m-a-p/SuperGPQA` `train` (`4430d445`) | difficulty "hard", written as in the calibration corpus |

Replies came from the FP8 checkpoint served by ATOM: TP4, FP8 KV cache, up to
32 sequences, the published 2-bit checkpoint's chat template. Its
non-expert linear layers were requantized at load to per-token, per-channel
FP8; the routed and shared experts, the router, the embeddings and the LM head
kept their stored weights. Sampling: temperature 1.0, top_p 0.95,
`max_tokens` 12,000 per reply (16,000 for science), shared by thinking and
answer. A conversation with any reply that did not finish (finish reason not
`stop`) was dropped: 63 of 271. Of the rest (chat 75, code 35, multilingual
63, long 7, science 28), the first 66, 35, 59, 7 and 23 in request order were
used. The last reply keeps its reasoning as `<think>...</think>`; the first
chat reply keeps only its answer. Each row is
`apply_chat_template(conversation, tokenize=False) + "<|user|>"`.

Math. All 262 MATH-500 test problems of level 4-5, answered by the FP8
checkpoint at temperature 0, top_p 1.0, up to 16,384 tokens, with "Please
reason step by step, and put your final answer within \boxed{}." appended to
the request (not to the rendered rows). The server for these traces was not
recorded. The 249 finished traces were shuffled with `random.Random(0)`. Each
was rendered as one user turn and one reply (`<think>reasoning</think>answer`),
with its own `[gMASK]<sop>` header, and the renderings were concatenated. The
rows are the first 16 consecutive 8,192-token windows, so a trace can be cut
at a row boundary. No end-of-turn token is added. The next trace's header
tokens are not role tokens, so the position after each answer is still
scored, and its end-of-turn decision counts.

SWE agent. The trajectories come from mini-swe-agent 2.4.5 runs (bash tool,
step limit 150, temperature 0.95, top_p 1.0, up to 32,768 tokens per reply,
2 hours per instance) of MXFP4 and experimental 2-bit builds of GLM-5.3, not
of the FP8 model; the FP8 model only picks the decision positions. The runs
cover a 200-instance SWE-bench Verified subset: every instance rated an hour
or more, plus 155 rated "15 min - 1 hour", stratified by repository with seed
20261023. SWE-bench Lite instances and 8 instances held out for evaluation
are excluded. The rest are shuffled with `random.Random(12345)`. Per instance,
one trajectory is picked at random, rendered with the chat template and the
bash tool definition, kept if it has at least 2,048 tokens, and cut to its
last 8,192 tokens. The first 21 are used.

Terminal agent. Harbor 0.25.0 with its terminus-2 agent, driven by the FP8
checkpoint (the same ATOM setup as the generated kinds) at temperature 0.95,
8 trials at a time, one attempt per task. A proxy capped thinking at 16,384
tokens. When a reply hit that cap without acting, the proxy closed the
reasoning with `</think>` and continued for up to 4,096 answer tokens. No
step limit was set; each task's own time limit applies. Tasks: the first 40
of Harbor's `LiteCoder/LiteCoder-rl` (all ran), the first 40 of
`aider/aider-polyglot` (34 ran) and 40 SWE-bench Verified non-Lite instances
(4 ran; the other images could not install `tmux`). One row per
conversation: its longest logged request plus the reply, rendered with the
chat template, cut to its last 8,192 tokens. Conversations under 1,024 tokens
are skipped. Conversations are ordered by the SHA-1 of their first two
messages, and the first 41 are used. terminus-2 replies are JSON keystrokes,
so these rows have no `<tool_call>` decisions.

How the published bucket was computed. A capture script outside aiter ran
the FP8 model (Transformers 5.16.1, FP8 matmuls on Transformers' Triton path)
over each kind's rows. It used the same scored-position and decision rules as
`calibrate`. The `swe_agent`, `terminal_agent` and `math_reasoning` rows were
fed as written (8,192 tokens each at most). Each generated row was fed as its
last 8,192 tokens (`long_context`: last 16,384), cut from the front, so the
earlier turns of long generated rows were not seen. Each kind's mean `x x^T`
over its decision positions was saved in FP16. The bucket is
`sum_k w_k H_k / tr(H_k) / sum_k w_k`, with the fixed weights w_k of the
table. It replaced the decision bucket of each layer file of the
`calibrate --hessian-dir` capture.

`calibrate --decision-text-file decisions.jsonl --decision-weights <the
table's weights>` (step 4 of the recipe below) differs from that capture in
three ways:

- it feeds each row's last 16,384 tokens (`--decision-keep end`), so the
  generated rows are fed whole (all `long_context` rows keep their end; all
  other rows fit), and more decision positions may be found; that count was
  not measured. To feed exactly what the capture fed, cut each generated row
  to its last 8,192 tokens (`long_context`: 16,384) before writing the file;
- it keeps FP32 Hessians instead of FP16;
- without the `kernels` package it dequantizes the FP8 weights to BF16 for
  each matmul.

Without `--decision-weights` it would weight kinds by tokens fed: `chat`
0.211, `swe_agent` 0.134, `terminal_agent` 0.132, `code` 0.130,
`science_reasoning` 0.104, `multilingual` 0.104, `math_reasoning` 0.102,
`long_context` 0.084, against 0.209, 0.131, 0.131, 0.131, 0.098, 0.102,
0.098 and 0.100 (the published weights, normalized). The effect of these
differences on the checkpoint was not measured.

Separation from evaluation. No row comes from the evaluation sets below
(AIME, SWE-bench Lite, Terminal-Bench 2.0, GSM8K) or from GPQA: the agent
tasks are SWE-bench Verified instances outside SWE-bench Lite, LiteCoder-RL
and Aider-Polyglot, and no Terminal-Bench dataset was used.

### Recipe of the published checkpoint

The published checkpoint is the 1,024-entry LDLQ checkpoint evaluated under
"Quality against MXFP4". `$FP8` is the GLM-5.3 block-FP8 checkpoint
(`model_config_sha256`
`81d66e3cc5d9abd08d64c4bf236fb9acbbaa984915a06492dcbec904d01ab5c1` as
`calibrate` hashes its config; its Hub revision was not recorded). In order:

1. Build the calibration corpus (`corpus.jsonl`, sha256 `1aab8c26...891bc8`)
   and the decision corpus (`decisions.jsonl`, sha256 `f1c60e0f...21ecd7`)
   as described above.
2. Importance statistics, with the earlier `calibrate.py` (Versions above).
   The current equivalent:

       python3 -m aiter.quantization.iq2r.glm5.calibrate --model-dir $FP8 \
           --text-file corpus.jsonl --output calibration.pt

   Result: 266,396 tokens, every routed expert reached (no forced pass;
   fewest routed tokens 14), 3 experts below 256 tokens. The artifact file
   has sha256
   `503ec966955d00c87f80d2455055cfe06b4e6e3633ace4828bed05bc1b6b46d9`. Its
   importance hash, which the layer files record as `iq2r_calibration_sha256`,
   is `f41cf225f3ceaa2253378c84f44ba9b02c9a8071243f55ab2870e1a406e2f9b8`. It
   includes the shared-expert statistics.
3. Layer Hessians `H_all`, with the current `calibrate.py`:

       python3 -m aiter.quantization.iq2r.glm5.calibrate --model-dir $FP8 \
           --text-file corpus.jsonl --hessian-dir hessians

   Result: 160,731 scored positions per layer over 266,396 tokens. The run
   also passed an earlier decision corpus; step 4 replaced that bucket.
4. Decision bucket: the per-kind capture and fixed-weight merge described in
   "How the published bucket was computed" (2,149 positions). With the
   current `calibrate.py`, steps 3 and 4 are one run (it differs from the
   capture as listed there):

       python3 -m aiter.quantization.iq2r.glm5.calibrate --model-dir $FP8 \
           --text-file corpus.jsonl --hessian-dir hessians \
           --decision-text-file decisions.jsonl --decision-keep end \
           --decision-weights swe_agent=0.130,terminal_agent=0.130,math_reasoning=0.098,chat=0.208,code=0.130,multilingual=0.102,long_context=0.100,science_reasoning=0.098

   In the published build, the capture's bucket was written instead, as the
   `decision` tensor (FP32) and `n_decision` of every
   `hessian-layer-NNNN.pt`. The manifest's per-file digests were recomputed
   (sha256 over the `all`, then the `decision` tensor bytes). The digest
   `compile` records, sha256 of the manifest's `files` object as sorted JSON,
   is `03658f3387c4d4591f6e1b828b71856555ba53573f03bb11a9e7fb5f4f14bfdd`.
5. Compile on 8 GPUs, one worker per GPU. Each worker takes the next MoE layer
   (3..77) that no worker has started and runs

       HIP_VISIBLE_DEVICES=$GPU python3 -m aiter.quantization.iq2r.glm5.compile \
           --model-dir $FP8 --output-dir $OUT --calibration-cache calibration.pt \
           --codebook-entries 1024 --ldlq --hessian-dir hessians --layers $LAYER

   until no layer is left. One more run without `--layers` checks the
   layers, then writes the shards, index, config and tokenizer.

Everything else is fixed in the code; `compile` has no flags for it:

| Setting | Value |
|---|---|
| levels | c4e4q (16 per group of 32), E8M0 `d` per row and 256 inputs |
| codebook | 1,024 entries per expert and projection (one for interleaved gate/up, one for down), seeded from the IQ2_XXS grid |
| codebook learner | up to 4 rounds of 4 Lloyd steps; 65,536 sampled 8-vectors (2,048 blocks of 256); seed `(0x10A0 + 1024 * layer + 2 * expert + is_down) * 2` |
| importance fallback | column energy for experts with fewer than 256 routed tokens |
| encoder | 13 starting scales per group, exhaustive search of `d` (3 exponents) and the 16 levels |
| gain | `<W, Wq> / <W, W> = 0.98`, gate and up rows separately |
| LDLQ | gate/up only; `H = H_all/tr + 1.0 * H_decision/tr`, damping `1.0 * mean(diag)`; FP64 Cholesky of `H^-1`; 8-weight vectors, 128-column lazy batches; gains refitted |
| shared expert, MTP layer 78 | copied as block FP8 |

Every one of the 150 layer files (`iq2r-layer-NNNN-{gate-up,down}.safetensors`)
carries the same encoding metadata, apart from `iq2r_layer` and
`iq2r_projection`:

| Key | Value |
|---|---|
| `iq2r_format` | `iq2r-1024-fullsign-c4e4q-e8m0d256-native-v2` |
| `iq2r_layout` | `glm53-packed-v3` |
| `iq2r_codebook_entries` | `1024` |
| `iq2r_codebook_iterations` | `4` |
| `iq2r_codebook_sample_vectors` | `65536` |
| `iq2r_v2_levels` | `c4e4q` |
| `iq2r_v2_target_gain` | `0.98` |
| `iq2r_v2_min_routed_tokens` | `256` |
| `iq2r_v2_shared_expert` | `fp8` |
| `iq2r_quality` | `calibrated-o0` |
| `iq2r_activation_basis` | `native` |
| `iq2r_calibration_sha256` | `f41cf225f3ceaa2253378c84f44ba9b02c9a8071243f55ab2870e1a406e2f9b8` |
| `iq2r_ldlq` | `{"beta_decision": 1.0, "buckets": ["all", "decision"], "damp": 1.0, "hessians_sha256": "03658f33...4bfdd", "projections": ["gate_up"]}` |

The checkpoint directory is 250,257,799,841 bytes (250.3 GB, 233.1 GiB).

Reproducing it. Neither corpus, the calibration artifact nor the Hessians are
published, and the corpus replies are sampled. A rebuild from public data
with these steps therefore gives a comparable checkpoint, not the same bytes.
`compile` keeps a layer file only if its metadata matches the current run, so
a rebuild with a different calibration hash or Hessian digest re-encodes every
layer.

## Dispatch

`iq2r_glm53_moe_out` looks up the launch choice for the token count in
`aiter/configs/iq2r_glm53_tuned.csv` (keyed on gfx, cu_num, token, inter_dim,
codebook size and whether the shared expert runs in the launch; rows with the
shared expert are looked up first). Other token counts use the next tuned row up that can run
them, else `_default_config`.
`AITER_LOG_TUNED_CONFIG=1` logs the choice once per shape. Each row names:

- the gate kernel and its grid: `decode` (M2-1024; M1 always runs the M1
  gate kernel) or `prefill`;
- the down kernel and its grid:
  - `per_token` (M1, M2, M4): one workgroup per token runs all 8 routes plus
    the shared-expert slot and sums them in the kernel;
  - `m16`: one 16-token M atom per pass, TP4 and TP8;
  - `m16_2wg`: `m16` built for two workgroups per CU, TP4 only;
  - `m32`: two M atoms per pass, TP8 only;
  - `prefill`;
- the number of prefill down chunks.

`AITER_CONFIG_IQ2R_GLM53` points the lookup at a different CSV. To tune
new shapes, add them to `aiter/configs/iq2r_glm53_untuned.csv` and run
`python3 csrc/kernels/iq2r/iq2r_glm53_tune.py`, which adds them to
`iq2r_glm53_tuned.csv`. The tuner uses the common aiter tuner flags:
`--all` retunes every shape, `--run_config [CSV]` times the launch choices
a CSV makes, and `-o2 FILE` keeps the timing of every candidate.

The tuner checks each candidate against a dense reference, then times it
by CUDA graph replay, as serving runs it; eager timing adds host overhead
that hides the best small-M launches. Each call in the graph reads its own
copy of the layer weights and its own routing, with at least 1 GiB of
expert weights per replay, so the weights come from HBM as they do when a
serving step walks the MoE layers. The decode kernels accept up to 1024
tokens so the CSV can choose them for MTP verify batches, which are
(1 + speculative tokens) x concurrency.

`python3 op_tests/test_iq2r_glm53.py` times the tuned and default configs
over a token sweep (`-t`, `-i` select the token counts and widths), prints
a summary table, then runs the correctness tests.

## Kernels

`iq2r_glm53_moe_out` runs sort/quantize, gate+up GEMM with SwiGLU and MXFP8 requantization, down GEMM and the
weighted top-8 reduction.

**256 experts, top-8.** The checkpoint keeps the shared expert out of the 2-bit stacks, so the MoE gets
`[256, ...]` stacks with `[tokens, 8]` ids and weights; the shared expert joins as described below.

- Routes keep nine slots per token (`token * 9 + k`). With top-8, slot 8 is empty: the sort kernels
  (`glm53_route_id`) read `[tokens, 8]` ids and treat slot 8, negative ids and ids >= 256 as invalid.
- The sorts group routes by expert (0-255, then the invalid routes) and build the gate and down task lists from
  the groups. Invalid routes get no tasks and `scatter = -1`, so the reductions skip them.
- The reductions read `[tokens, 8]` weights; the `per_token` down runs 8 route waves (2 per SIMD).

| Stage | Kernel | Notes |
|---|---|---|
| M1 gate | `glm53_gate_m1_kernel` | Loads copy 0 of the book (4 KB) and forms copies 1-3 in LDS; non-temporal weight loads |
| Decode gate (`decode`) | `glm53_gate_decode_kernel` | Per-XCD item ranges; the book is formed after the item's loads are issued; buffer-descriptor K loop with scalar offsets; per-atom MFMA scale `op_sel`; non-temporal weight loads; 2 workgroups per CU |
| Prefill gate | `glm53_gate_prefill_kernel` | Book formed from copy 0; 2 waves per SIMD |
| M1-M4 down (`per_token`) | `glm53_down_per_token_kernel` (512 entries), `glm53_down_per_token_1024_kernel` | Each route wave forms its own book in LDS; activation row by DMA; 2 triplets per CTA at M4 |
| Decode down | `glm53_down_tp4_kernel` (`m16`, `m16_2wg`), `glm53_down_tp8_kernel` (`m16`, `m32`) | K tiles outermost; levels and scale bytes once per K tile; 16 KB of copies staged by DMA |
| Prefill down | `glm53_down_prefill_tp4/tp8_kernel` | Book formed from copy 0; levels and scale bytes once per K tile; TP4 items cover 768 columns |

**Copy formation.** The packed auxiliary stores the four 4 KB codebook copies `E4M3(codebook * 2^(c/4))`. Most
kernels read only copy 0 and form copies 1-3 in LDS with integer byte adds (`iq2r_v2_copy_words`): for a
normal E4M3 byte `b`, copy c's byte is `b + delta_c[b & 7]`. This is exact only for bytes 0x08..0x78 (2^-6 to
256): `iq2r_v2_learn_codebook` keeps its entries in that range, and `iq2r_glm53_pack` rejects any codebook
byte outside it (`iq2r_v2_check_codebook`). Every codebook byte of the GLM-5.3 IQ2R-v2 checkpoint is in
0x08..0x4f.

**1,024 entries.** Every gate and down kernel takes the codebook size as a template parameter; the host picks it
from the packed sizes, so 512-entry checkpoints run the same instantiations (their generated code is unchanged).

| Kernel | 1,024-entry change | LDS (bytes) | Workgroups per CU |
|---|---|---|---|
| M1 gate | 8 KB copy 0, copies formed in LDS; loads in consumption order, decode overlaps them | 65,536 | 2 |
| Decode gate | copies formed in LDS; one tenth-bit load per item and wave | 65,536 | 2 |
| Prefill gate | copies formed in LDS; one tenth-bit dword live | 49,920 | 2 |
| `per_token` (M1-4) | `glm53_down_per_token_1024_kernel`: copy 0 per route (8 KB, DMA); a gathered codeword becomes its class's copy in registers (`b + delta_c[b & 7]`) | 68,736-71,552 | 1 |
| Decode downs | copy 0 staged, copies 1-3 formed by each wave; TP8 activation scales in registers; `m16_2wg` at TP4 M8-16 | 36,864-41,216 | 3 (TP4), 4 (TP8) |
| Prefill downs | TP8 scales in registers; TP4 at two workgroups per CU with an unrolled K loop | 40,960 (TP8), 66,560 (TP4) | 4, 2 |

Eight routes' 32 KB copies do not fit in a CU's 160 KiB of LDS, hence the `per_token` kernel's in-register copies.

**Shared expert in the IQ2R launches.** For batches of up to `FUSED_SHARED_TOKENS` (256) tokens, the shared
expert runs as route slot 8 of every token (weight 1) inside the gate and down launches:

- Weights: MXFP8 (E4M3 bytes, one E8M0 scale per 32 inputs), repacked at load from the checkpoint's block FP8
  (`iq2r_glm53_pack_shared`) in MFMA operand order; about 0.7 GB per GPU at TP4. The FP8 copy stays loaded for
  larger batches.
- Gate: extra work items in the M1 and decode gate launches (16 x MA tokens x 16 gate/up rows over the whole K,
  one K768 slice per wave), writing SiLU(gate) * up in BF16 to the workspace (`shared_act`). With all of a
  launch's workgroups resident, the shared items run on their own workgroups.
- Down: `per_token` spreads the shared expert's column blocks x K tiles over its eight waves after their routes and
  adds the result in its top-8 sum; the decode downs run extra items writing the shared route rows, which the
  route reduction adds with weight 1. Both quantize the BF16 activation to MXFP8 with the routed path's rounding.
- No workgroup waits on another (a split-K version with partials and an arrival counter was 3-5x slower: an
  agent-scope fence compiles to an L2 writeback and invalidate on gfx950).
- Larger batches use the prefill kernels and the separate shared MLP.


## Correctness

- **Tests** (`op_tests/test_iq2r_glm53.py`, `op_tests/test_iq2r_glm53_unpack.py`):
  - the MoE against a dense reference, TP4 and TP8, M = 1..4096, with the tuned and with explicit launch
    choices;
  - ids -1 and 256 contribute nothing;
  - the copy-formation formula on every byte in 0x08..0x78 (and that 0x07 breaks it);
  - the learner's range on zero and tiny weights, and the packer's rejection of out-of-range books;
  - `iq2r_glm53_unpack` inverts `iq2r_glm53_pack`, also on TP slices.
- **1,024 entries and the fused shared expert:** every MoE test also runs on 1,024-entry layers;
  `test_glm53_down_decode_bit_exact` checks that every down kernel decodes bit-exactly for both sizes (one-hot
  activations, each output `BF16(gain * W[:, k])` against `iq2r_v2_materialize`);
  `test_glm53_moe_fused_shared_expert` (M = 1..1024, TP4/TP8, both sizes) matches the dense reference with the
  shared expert (relative RMSE <= 0.021, cosine >= 0.99978) and repeats bit-identically. `op_tests/test_iq2r_ldlq.py`
  covers the LDLQ pass; compile checks every LDLQ matrix bit-exact (records decode to the encoding, each packed
  layer round-trips).
- **On the GLM-5.3 checkpoint's own bytes:** `pack(unpack(x)) == x` on all 75 MoE layers. On layers 5, 20, 40, 60
  and 75 (TP8 rank 0 and TP4 rank 1 slices, captured layer inputs with their real top-8 routing, M = 1..4096),
  the MoE output matches the dense reference built from the unpacked bytes in all 90 cases: relative RMSE at
  most 0.025, cosine at least 0.9997. On a 1,024-entry checkpoint: `pack(unpack(x)) == x` on layer 3, and the MoE
  on its TP4 and TP8 rank-0 shards matches the dense reference from the unpacked bytes at M = 1..1024 (relative
  RMSE at most 0.019, cosine at least 0.99983).

## Measurements

One MI355X (gfx950, 256 CUs), ROCm 7.2.4, PyTorch 2.10. Whole MoE (`iq2r_glm53_moe_out`), the tuner's method:
CUDA-graph replay; every call reads its own copy of the layer, at least 1 GiB of expert weights per replay, so
experts come from HBM; synthetic weights.

Against the IQ2R (v1) kernels this format replaces (257 experts with the fused shared expert, top-9, their tuned
CSV); both on the same GPU, alternating, 6 rounds each; µs, best round (median in brackets):

| M | TP4 IQ2R | TP4 IQ2R-v2 | Δ | TP8 IQ2R | TP8 IQ2R-v2 | Δ |
|---|---|---|---|---|---|---|
| 1 | 22.69 (22.70) | 17.73 (17.77) | -21.9% | 19.07 (19.10) | 16.37 (16.41) | -14.2% |
| 2 | 25.78 (25.81) | 20.00 (20.02) | -22.4% | 19.95 (20.06) | 17.84 (17.93) | -10.6% |
| 4 | 36.65 (36.71) | 26.73 (26.84) | -27.1% | 23.17 (23.51) | 20.09 (20.37) | -13.3% |
| 8 | 52.33 (52.40) | 47.73 (47.92) | -8.8% | 35.30 (35.60) | 32.63 (32.84) | -7.6% |
| 16 | 76.07 (76.17) | 71.60 (72.11) | -5.9% | 51.13 (51.27) | 45.87 (46.32) | -10.3% |
| 32 | 109.09 (109.34) | 104.07 (104.38) | -4.6% | 68.21 (68.59) | 63.07 (63.27) | -7.5% |
| 64 | 146.15 (146.70) | 139.21 (139.50) | -4.7% | 84.93 (85.43) | 79.44 (79.80) | -6.5% |
| 128 | 169.05 (170.38) | 158.97 (159.37) | -6.0% | 101.06 (101.75) | 91.60 (91.87) | -9.4% |
| 256 | 188.75 (189.99) | 174.79 (175.14) | -7.4% | 110.63 (112.11) | 102.88 (103.41) | -7.0% |
| 512 | 257.62 (259.57) | 223.26 (223.55) | -13.3% | 148.23 (149.11) | 128.32 (130.03) | -13.4% |
| 1024 | 293.53 (295.45) | 261.55 (262.06) | -10.9% | 211.57 (212.79) | 166.76 (168.62) | -21.2% |
| 2048 | 448.61 (449.08) | 419.34 (419.44) | -6.5% | 304.70 (305.14) | 277.91 (278.84) | -8.8% |
| 4096 | 747.83 (748.96) | 715.96 (716.82) | -4.3% | 509.62 (510.52) | 462.57 (464.70) | -9.2% |

IQ2R-v2 is faster at every tuned token count; the smallest margin is -4.3% (TP4 M4096). The IQ2R column
includes the shared expert (fused as route 9); IQ2R-v2 runs it as a separate FP8 MLP, not in this table.

1,024 entries against 512 (same procedure on one MI350X, 16 rounds, every round timing both back to back; µs
median, ratio = median of the per-round ratios; routed experts only):

| M | TP4 512 | TP4 1,024 | ratio | TP8 512 | TP8 1,024 | ratio |
|---|---|---|---|---|---|---|
| 1 | 18.34 | 20.66 | 1.126 | 16.96 | 18.11 | 1.068 |
| 2 | 20.81 | 22.98 | 1.104 | 18.59 | 19.54 | 1.051 |
| 4 | 32.20 | 36.77 | 1.128 | 21.62 | 22.91 | 1.061 |
| 8 | 56.40 | 59.64 | 1.060 | 36.93 | 37.69 | 1.019 |
| 16 | 84.32 | 89.32 | 1.060 | 51.72 | 55.07 | 1.065 |
| 32 | 125.67 | 137.15 | 1.093 | 74.75 | 75.19 | 0.998 |
| 64 | 172.84 | 181.06 | 1.049 | 92.69 | 94.87 | 1.027 |
| 128 | 195.91 | 208.00 | 1.061 | 106.98 | 112.97 | 1.059 |
| 256 | 210.38 | 227.05 | 1.077 | 120.90 | 122.71 | 1.013 |
| 512 | 269.31 | 291.02 | 1.081 | 153.71 | 164.88 | 1.074 |
| 1024 | 318.67 | 338.49 | 1.063 | 197.10 | 210.30 | 1.065 |
| 2048 | 502.15 | 530.85 | 1.057 | 323.61 | 346.44 | 1.070 |
| 4096 | 849.79 | 897.17 | 1.056 | 545.31 | 572.44 | 1.051 |

The floor is the +4.3% weight bytes; M1-4 also pay the `per_token` kernel's in-register copies (TP4 +10-13%), the prefill gate
the 10-bit unpack. On real checkpoint layers the ratios are within a few points of these (up to 1.18 at TP4 M4).

The fused shared expert adds to the MoE call (TP8, 1,024 entries, tuned rows): M1 +2.4 µs, M2 +2.7, M4 +2.7, M8
+2.3, M16 +0.2, M32 +3.3; at TP4 M1 +2.3, M4 +4.2, M16 +2.3, M64 +15, M256 +23. A separate shared-expert MLP cost
about 27 µs per layer at concurrency 1 and 93 µs at 64.

## Serving: memory and throughput against MXFP4

ATOM `benchmark_serving`, random prompts, 1024 input and 1024 output tokens
(`--random-range-ratio 1`, `--ignore-eos`), 10 x concurrency prompts after
2 x concurrency warmups, one server at a time, FP8 KV cache,
`--max-model-len 262144`, `--max-num-seqs 256`, CUDA graphs for batch 1-256,
non-expert layers PTPC FP8 in both arms. Standard kernels only: ATOM's GLM
MonoKernel (a fused small-batch decode that supports only MXFP4 experts) is
off (`ATOM_MONO_ENABLE=0`). One host with 8 MI350X.

- IQ2R: 1,024-entry checkpoint, native kernels, shared expert in the IQ2R
  launches (tuned rows); "separate" keeps it a separate FP8 MLP.
- MXFP4: the GLM-5.3 FP8 checkpoint with routed and shared experts quantized to
  MXFP4 at load, A4W4.

TP4, output tokens/s (mean TPOT, mean TTFT):

| Concurrency | MXFP4 | IQ2R, fused shared | IQ2R, separate shared | fused / MXFP4 |
|---|---|---|---|---|
| 1 | 85 (11.7 ms, 0.10 s) | 91 (10.9 ms, 0.10 s) | 79 (12.6 ms, 0.11 s) | 1.07 |
| 2 | 180 (11.0 ms, 0.20 s) | 167 (11.2 ms, 0.78 s) | 149 (12.7 ms, 0.79 s) | 0.93 |
| 4 | 315 (11.8 ms, 0.88 s) | 313 (12.1 ms, 0.71 s) | 279 (13.7 ms, 0.74 s) | 0.99 |
| 8 | 524 (13.3 ms, 2.05 s) | 539 (13.8 ms, 1.08 s) | 491 (15.2 ms, 1.10 s) | 1.03 |
| 16 | 964 (15.8 ms, 0.78 s) | 961 (15.8 ms, 0.85 s) | 890 (17.2 ms, 0.85 s) | 1.00 |
| 32 | 1589 (19.4 ms, 0.74 s) | 1558 (19.7 ms, 0.91 s) | 1465 (21.0 ms, 0.91 s) | 0.98 |
| 64 | 2492 (24.8 ms, 0.94 s) | 2456 (24.8 ms, 1.26 s) | 2305 (26.6 ms, 1.17 s) | 0.99 |
| 128 | 3838 (31.8 ms, 1.64 s) | 3707 (32.5 ms, 2.08 s) | 3550 (34.0 ms, 2.08 s) | 0.97 |
| 256 | 5708 (41.3 ms, 3.63 s) | 5450 (43.3 ms, 3.80 s) | 5265 (44.9 ms, 3.80 s) | 0.95 |

TP8:

| Concurrency | MXFP4 | IQ2R, fused shared | IQ2R, separate shared | fused / MXFP4 |
|---|---|---|---|---|
| 1 | 87 (11.4 ms, 0.11 s) | 91 (10.8 ms, 0.11 s) | 79 (12.6 ms, 0.11 s) | 1.05 |
| 2 | 158 (11.6 ms, 1.10 s) | 168 (11.0 ms, 0.94 s) | 146 (12.8 ms, 0.95 s) | 1.06 |
| 4 | 331 (11.2 ms, 0.89 s) | 331 (11.3 ms, 0.81 s) | 289 (13.0 ms, 0.88 s) | 1.00 |
| 8 | 604 (12.2 ms, 1.13 s) | 591 (12.4 ms, 1.14 s) | 530 (14.0 ms, 1.16 s) | 0.98 |
| 16 | 1190 (13.2 ms, 0.30 s) | 1153 (13.5 ms, 0.43 s) | 1012 (15.2 ms, 0.65 s) | 0.97 |
| 32 | 2055 (15.2 ms, 0.44 s) | 2037 (15.2 ms, 0.50 s) | 1832 (17.0 ms, 0.49 s) | 0.99 |
| 64 | 3205 (19.3 ms, 0.66 s) | 3198 (19.1 ms, 0.96 s) | 2954 (20.9 ms, 0.75 s) | 1.00 |
| 128 | 5227 (23.4 ms, 1.08 s) | 5031 (24.2 ms, 1.30 s) | 4743 (25.7 ms, 1.31 s) | 0.96 |
| 256 | 7910 (30.4 ms, 2.00 s) | 7595 (31.3 ms, 2.44 s) | 7285 (32.8 ms, 2.41 s) | 0.96 |

- With the shared expert in the launches, IQ2R's time per output token is at or
  below MXFP4's up to concurrency 64 (within 0.3 ms at TP4 c=4 and TP8 c=4-16).
  At 128-256 it trails by 0.7-2.0 ms: decode batches then carry prefill chunks
  of more than 256 tokens, which use the separate shared MLP.
- TP4 c=2: the median first-token time equals MXFP4's (193 vs 180 ms); the mean
  carries one request that waits about 6 s the first time a 2,048-token prefill
  batch forms (intermittent, IQ2R only, not resolved).
- With ATOM's MonoKernel on (its default), MXFP4 is faster at very small
  batches; IQ2R has no MonoKernel path.

Memory per GPU, from ATOM's budget at 0.9 utilization (512-entry checkpoint,
shared expert as a separate FP8 MLP, one MI355X node): weights plus peak
activations 36.1 GB (MXFP4 57.1 GB) at TP8 and 62.8 GB (MXFP4 104.4 GB) at TP4,
leaving 11% (TP8) and 30% (TP4) more KV cache than MXFP4. The 1,024-entry
checkpoint adds about 4.3% expert bytes and the fused path a 0.7 GB MXFP8
copy of the shared expert per GPU at TP4 (not measured in the budget).

## Quality against MXFP4

**1,024 entries with LDLQ, full SWE-bench Lite.** All 300 tasks
(`princeton-nlp/SWE-bench_Lite`, test split), one attempt per task per arm,
both arms at the same time on one host with 8 MI350X (IQ2R on 4 GPUs, MXFP4
A4W4 on the other 4, TP4 each), the same aiter and ATOM code, FP8 KV cache,
standard kernels. ATOM's `run_swebench_lite.sh` settings (mini-swe-agent
2.4.5, SWE-bench harness 4.1.0, temperature 0, 150 steps, 32 agents per arm),
with three changes applied to both arms so that no setting other than the
262,144-token context decides an outcome: the task containers have no network
access (`--network none`; with network the agents fetched upstream fixes), the
task limit is 2 hours, and each request allows up to 65,536 new tokens (the
server default is 8,192):

| | IQ2R, 1,024 entries + LDLQ | MXFP4 A4W4 |
|---|---|---|
| Resolved | 194 / 300 (64.7%) | 197 / 300 (65.7%) |
| Resolved by both / only this arm | 182 / 12 | 182 / 15 |
| Harness errors | 0 | 0 |

- On par with MXFP4: the arms agree on 273 of 300 tasks and split the other
  27 by 12 to 15, within run-to-run noise (exact McNemar test p = 0.70).

**1,024 entries with LDLQ, full Terminal-Bench 2.0.** All 89 tasks, one
attempt per task per arm, both arms at the same time on one host with 8
MI350X (IQ2R on 4 GPUs, MXFP4 A4W4 on the other 4, TP4 each), the same aiter
and ATOM code, FP8 KV cache, `--max-num-seqs 16`, standard kernels. Harbor
0.25.0 with its terminus-2 agent (temperature 0.95, the benchmark's task time
limits), 8 tasks at a time per arm, thinking capped at 16,384 tokens plus
4,096 answer tokens by a proxy:

| | IQ2R, 1,024 entries + LDLQ | MXFP4 A4W4 |
|---|---|---|
| Solved | 60 / 89 (67.4%) | 62 / 89 (69.7%) |
| Solved by both / only this arm | 56 / 4 | 56 / 6 |
| Infrastructure errors | 0 | 0 |

- On par with MXFP4: the two arms disagree on 10 tasks, 4 to 6, which is
  within run-to-run noise (exact McNemar test p = 0.75). One attempt per task
  moves an 89-task score by a few tasks between runs.
- Only IQ2R: build-cython-ext, polyglot-rust-c, pytorch-model-cli,
  schemelike-metacircular-eval. Only MXFP4: build-pov-ray, largest-eigenval,
  mteb-leaderboard, path-tracing, regex-chess, train-fasttext.
- These are not leaderboard settings (one attempt, capped thinking); they
  compare the two arms under identical conditions.

**1,024 entries with LDLQ (decision corpus above), paired tasks.**
Native kernels, TP4, FP8 KV cache, max-num-seqs 16; base and MXFP4 side by side,
then the LDLQ arm, on one day:

| | 1,024 entries, no LDLQ | 1,024 entries, LDLQ + decision corpus | MXFP4 A4W4 |
|---|---|---|---|
| AIME-60 (T 1.0, top-p 0.95, 2 passes, 24K thinking + 8K answer) | 54, 54 | 55, 55 | 57, 57 |
| AIME median output tokens | 23.6K, 22.6K | 15.4K, 14.6K | 14.9K, 12.5K |
| AIME answers forced at the budget | 29, 28 | 20, 21 | 20, 21 |
| SWE-bench Lite, 15 split tasks (2 runs) | 3, 7 | 8, 7 | 8, 5 |
| Terminal-Bench 2.0, 3 hard tasks (5 attempts each) | 11/15 | 12/15 | 12/15 |

- The 15 split tasks are the SWE-bench Lite tasks that exactly one of the
  two models (512-entry IQ2R and MXFP4 A8W4) solved in the 150-task run
  below.
- The 3 Terminal-Bench 2.0 tasks are cobol-modernization, write-compressor
  and torch-tensor-parallelism, chosen because 2-bit builds before this recipe
  failed them.
- LDLQ with the decision bucket removes most of the 1,024-entry model's extra
  thinking (median output 15K vs 23K tokens, as MXFP4) and lifts the 15 SWE
  tasks from 10 to 15 of 30.
- These are small, hard-case subsets: 15-task SWE runs move by a few tasks, the
  Terminal-Bench tasks by about +-4 of 15. The full benchmarks above replace
  them as the comparison of record.
- The MXFP4 arm here is A4W4; the 512-entry comparison below used A8W4
  (9, 11, 12 on the same 15 tasks).
- No degenerate output in any arm (no `</think>` collapses, no runaway
  generations).

**512-entry recipe without LDLQ (`glm53-packed-v2`), shared expert separate.**
Against MXFP4 routed experts on the same FP8 checkpoint (shared expert block FP8
in both), ATOM on MI355X, FP8 KV cache:

| Benchmark | IQ2R, 512 entries | MXFP4 |
|---|---|---|
| GSM8K, flexible-extract / strict-match (TP8) | 0.9666 / 0.9674 (± 0.005) | 0.9697 / 0.9704 (± 0.005) |
| SWE-bench Lite, resolved (TP4) | 95 / 300 (31.7% ± 2.7) | 99 / 300 (33.0% ± 2.7) |

- GSM8K: ATOM's CI command for GLM-5.3 (lm_eval 0.4.13 `local-chat-completions`,
  chat template, 5-shot, all 1319 questions, temperature 0, up to 16,384
  generated tokens). Both pass ATOM's 0.96 gate (FP8: 0.9719 / 0.9727).
- SWE-bench Lite: `princeton-nlp/SWE-bench_Lite`, test split, all 300 tasks,
  with ATOM's `.github/scripts/atomesh/run_swebench_lite.sh` and its defaults
  (mini-swe-agent 2.4.5, SWE-bench harness 4.1.0, temperature 0, 150 steps,
  1-hour task limit, 32 agents). Both arms ran at the same time on one node.

### 512-entry recipe with other agent settings

This evaluation ran SWE-bench with settings other than ATOM's script:
mini-swe-agent 2.4.5, temperature 0.95, thinking capped at 16,384 tokens plus
4,096 answer tokens, 150 steps, 4 hours per task, no network. Its absolute
SWE-bench scores are higher than the ATOM-script results above. "Decoded"
is the same checkpoint decoded to FP8 at load instead of served by the native
kernels. TP8 unless noted, FP8 KV cache, 16 concurrent
sequences except two of the MXFP4 29-task runs at 64; one run per number
unless several are listed.

| Measure | IQ2R, 512 entries | MXFP4 |
|---|---|---|
| **SWE-bench Lite, 150 tasks** (the first 150 in dataset order, in three batches of 50; TP4, 24 agents) | **107** (38, 38, 31 per batch); ~107 with the 15 split tasks rerun twice | **110** A8W4 (39, 36, 35); ~111.7 with the 15 tasks rerun; first batch A4W4 38 |
| SWE-bench Lite, 29-task subset | native kernels 19, 20; decoded 19, 24 | 22, 22, 22 (A4W4); 21 (on the ATOM build with the native kernels) |
| AIME-60, sampled, 24K thinking budget | native kernels 55, 53; decoded 54, 53 | 55, 55, 55, 54 |
| KL vs FP8 (teacher-forced, 4 SWE agent + 1 math 8K windows) | 0.090 | 0.034 |

What the numbers say:
- **About 4-5 tasks behind MXFP4 on SWE-bench.** On the 150 tasks, 6 were
  solved only by IQ2R-v2 and 9 only by MXFP4. Those 15 tasks were run twice
  more on both: IQ2R-v2 7 and 5, MXFP4 11 and 12. Over the three runs MXFP4 did
  better on 11 of the 15, IQ2R-v2 on 1 (sign test p ~0.006). With the 135 tasks
  both agreed on held fixed, the 150-task estimate is about 107 for IQ2R-v2 and
  111.7 for MXFP4 A8W4 (~3 points of resolve rate). AIME is level.
- **The native kernels score like the decoded checkpoint.** The same checkpoint served
  by the native kernels and decoded to FP8 at load gives the same AIME and,
  within noise, the same SWE-bench subset scores.
- **29 tasks are too few to compare recipes:** the same checkpoint scored 19
  and 24 in two runs.
- **Behaviour:** the 2-bit model takes more agent steps than MXFP4 (median 88
  against 76 per task) and about 1.4x the output tokens; no collapses; about
  0.7% of replies switch to another script (none for MXFP4).
- **The shared expert must stay FP8:** at 2 bits it cost AIME 54, 54 -> 49, 50
  and SWE-bench 29-task 3.
- **Speed:** see Measurements above.

## Not measured

- GSM8K on the 1,024-entry LDLQ recipe; SWE-bench Lite and Terminal-Bench 2.0 with several attempts per task.
- Serving of the 1,024-entry checkpoint on MI355X (the serving tables are MI350X); its per-GPU memory budget.
- Longer inputs and outputs than 1024/1024; MTP.
- Variation across GPUs and nodes; token counts between tuned rows; M > 4096; power.
