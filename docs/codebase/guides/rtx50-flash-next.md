[Codebase docs](../README.md) › [Guides](README.md) › Qwen3.8-Flash-Next on an RTX 50 card

# Qwen3.8-Flash-Next on an RTX 50 card: the architecture, the memory and the code

This page follows one model, Qwen3.8-Flash-Next (`qwen4exp`), on one family of cards, the NVIDIA RTX 50 series
(Blackwell, compute capability 12.0, `sm_120`). It explains the model's architecture part by part. For each part it
names the code that runs it, what the part costs in VRAM and RAM, and what keeps that cost low. It also lists every
change this fork made to the code that serves the model.

## Read this first: what was changed, and what was not

Three points come out of the source and the git history. They are stated before anything else so the rest of the
page is read correctly.

1. **There is no RTX 50-specific code change in this fork.** The engine came from upstream Strata, and upstream
   wrote it for `sm_120` first. `include/strata/core/device.hpp:51-54` still says "The engine targets sm_120
   specifically". `CMakeLists.txt:140` still defaults the build to arch `120`. `CMakeLists.txt:144-145` says the
   engine was "developed and measured on sm_120 (RTX 50)" before it was ported down to RTX 20, 30 and 40. Upstream's
   published speeds were measured on a 12 GB RTX 5070 ([docs/upstream/DETAILS.md](../../upstream/DETAILS.md),
   "Speed (measured)"). Blackwell is the engine's home architecture, not a port.
2. **The fork's changes for Flash-Next are about loading, serving and building, not about the GPU.** They are:
   - retrying reads from slow stores;
   - holding the n-gram table in RAM without `mlock`;
   - the C++ front end and the engine flags it chooses;
   - one patch to llama.cpp's tool-call parser;
   - a build that compiles `sm_120` beside five older architectures.

   [Every change, with its commit](#5-every-fork-change-that-touches-flash-next) is listed below.
3. **The low-memory machinery is upstream's.** The fork kept, unchanged:
   - the VRAM expert cache, the pinned RAM tier and the CPU/GPU split;
   - the compact KV formats and KV streaming;
   - the n-gram table reader;
   - the verify window that reuses one expert read for several tokens.

   The fork changed how two of these load data (the expert loader and the n-gram table), not how they compute.

**Tested status.** This fork has served Flash-Next on an RTX 3090 (container, 125 GB RAM) and an RTX 4090 (62 GB and
82 GB hosts). It has **not run on an RTX 50 card**:
- the README's status table: "Not tested | RTX 50 cards";
- the Known problems section: "the RTX 50 and Volta builds have not run on a card".

On an RTX 50 card the binary runs upstream's Blackwell kernels with this fork's front end and loaders. Section 7 lists
what to check on a first start.

Contents:

1. [The model in one picture](#1-the-model-in-one-picture)
2. [Where the bytes are](#2-where-the-bytes-are)
3. [Part by part: architecture, code, memory](#3-part-by-part-architecture-code-memory)
4. [The RTX 50 code paths](#4-the-rtx-50-code-paths)
5. [Every fork change that touches Flash-Next](#5-every-fork-change-that-touches-flash-next)
6. [Sizing for each RTX 50 card](#6-sizing-for-each-rtx-50-card)
7. [Building and running on an RTX 50 card](#7-building-and-running-on-an-rtx-50-card)

---

## 1. The model in one picture

Qwen3.8-Flash-Next has about 125B parameters ([README.md](../../../README.md)). It is a mixture-of-experts model:
each token uses only a small, token-dependent part of the weights. It combines five ideas that are rare in other
open models:
- linear-attention (Gated DeltaNet) layers;
- sparse attention picked by an indexer;
- a residual stream of four parallel copies (hyper-connections);
- a hashed n-gram embedding table;
- a built-in multi-token-prediction draft layer.

The geometry is in `struct ModelGeometry` (`include/strata/core/layout.hpp:38`). The engine reads it from the GGUF
at start-up; [core/model-geometry.md](../core/model-geometry.md) has every field.

```
 token id ──┬──> token embedding (248,320 x 2,560)            host RAM, one row over PCIe per token
            └──> n-gram hash of (token, 2 previous) -> 16 rows of the PLE table (28.8 GB, shard 2)
                                                                │
 residual stack R: 4 streams x 2,560  (hyper-connections, hc = 4)
                                                                │
 for layer l = 0 .. 47:                                         │
   read:   gr_read(R) -> one 2,560-wide input  ─────────────────┤ layer 1 only: PLE block adds the 16 rows into R
   mixer:  l % 4 != 3 -> Gated DeltaNet (36 layers)   fixed-size recurrent state, no KV cache
           l % 4 == 3 -> sparse attention "QSA" (12)   KV cache; an indexer picks <= 2,051 cells to attend to
   write:  gr_write(R, mixer out)
   read:   gr_read(R) -> MoE input
   MoE:    router over 512 experts -> top 10, plus 1 shared expert (each expert 2,560 -> 640 -> 2,560)
   write:  gr_write(R, MoE out)
 head:     gr_read(R) -> rms_norm -> output (2,560 x 248,320) -> logits
 MTP:      one more block (attention + MoE) drafts the next tokens; the 48 layers then check them together
```

| Part | Flash-Next value | Where it is declared |
| --- | --- | --- |
| Layers | 48: 36 Gated DeltaNet + 12 attention (`qsa_interval` 4) | `layout.hpp:38`, `is_qsa_layer` `:105` |
| Width | `n_embd` 2,560 | `layout.hpp:38` |
| Residual | `hc` 4 streams, low rank `hc_lr` 320 | `layout.hpp:38` |
| Gated DeltaNet | state 128, 16 key heads, 48 value heads, conv 4 taps over 10,240 channels, **sigmoid** output gate | `gdn_gate()` `layout.hpp:90` |
| Attention | 24 query heads, 2 KV heads, head dim 256, rotary on 64 dims | `qsa_real_shapes()` `include/strata/kernels/qsa.hpp:110` |
| Indexer | 4 heads x 128, block of 4 cells, top 2,048 cells | `qsa.hpp:113`, `has_indexer()` `layout.hpp:86` |
| Experts | 512 routed, 10 per token, 1 shared, FFN width 640 | `layout.hpp:72` |
| PLE | layer 1; 320,001,536 rows of 160 values | `has_ple` `layout.hpp:79` |
| Vocabulary | 248,320 | `layout.hpp:38` |

**Why the model fits at all: most of it is experts.** The arithmetic below is derived from the geometry, not
quoted from a model card.
- One expert has `3 x 2,560 x 640` = 4.9M parameters (gate, up and down projections).
- 512 experts x 48 layers = 24,576 experts, about 121B parameters. That is about 96% of the 125B.
- A token routes to 10 of them per layer: 480 experts, about 2.4B parameters.

So the per-token compute is small, but which 2% of the experts a token needs is decided only at that token's router.
Every memory technique on this page manages that one fact: almost all the bytes are experts, and any expert can be
needed next.

---

## 2. Where the bytes are

At the quantization this fork serves (ISTA-DASLab's IQ3_XXS GGUF, revision `ed59f920`), the model is about 78 GB on
disk ([README.md](../../../README.md), "Qwen3.8-Flash-Next").

| Piece | Size | Default home | What decides it |
| --- | --- | --- | --- |
| Routed experts, 24,576 | ~40 GiB (the arena a 4090 run pinned), so ~1.7 MiB each on average | VRAM slots for the hottest, pinned RAM for the rest | `--expert-cache auto`, `expert-profile.bin`, adaptive swaps |
| PLE n-gram table | 28.8 GB (shard 2) | RAM (the front end's `--ple-io ram`) | `--ple-io direct\|mmap\|ram` |
| Dense weights: GDN, attention, shared experts, routers, hyper-connection weights, output head | a few GB | VRAM | always on the card |
| Token embedding | 248,320 rows | mapped pinned host RAM, no VRAM | `NativeEmbed` (`include/strata/core/native_head.hpp:40-43`) |
| KV cache, 12 attention layers | 12,672 B per token in INT8 (3.3 GB at 262,144) | VRAM, or host RAM with a VRAM window | `--kv`, `--kv-resident` |
| GDN state, 36 layers | ~118 MB, independent of context length | VRAM | geometry |
| MTP draft head | 0.8 GB on disk | VRAM | `--mtp`, `draft_vocab.bin` |

Measured on the RTX 4090 (24 GB) with the service's command line, which is `--ctx 196608`, INT8 KV and
`--kv-resident 32768` ([docs/qwen36-port/step5-serve.md](../../qwen36-port/step5-serve.md), "Flash-Next
regression check"):
- the card held **10,368 of the 24,576 experts** (42%) with 430 MiB of VRAM left;
- decode ran at about 90 tokens/s;
- capped at `--expert-cache 4000`, which allocated 5,357 sized slots and 15.4 GB of the card, decode ran at
  54 tokens/s.

---

## 3. Part by part: architecture, code, memory

Each part below follows the same order:
1. what the architecture does;
2. the code that runs it;
3. what it costs and how the cost is kept down;
4. what this fork changed, if anything.

### 3.1 Token embedding and the 248,320-token vocabulary

**Architecture.** A token id selects one row of a 248,320 x 2,560 table. At the other end, the output head
multiplies the final hidden vector by a 2,560 x 248,320 matrix to get one logit per vocabulary entry.

**Code and memory.**
- **The embedding table.** `NativeEmbed` (`include/strata/core/native_head.hpp:40-43`) keeps the table "in mapped
  pinned host memory: a row is read over PCIe per token, so the table costs no VRAM". One token needs one row, a few
  KB, so the PCIe cost is negligible and the table's full size is VRAM the expert cache gets instead.
- **The output head.** It stays on the GPU (`lm_head`, [core/dense-and-head.md](../core/dense-and-head.md)) because
  every token needs all of it.
- **Greedy argmax.** It reads a 1 MB row of logits per token. On RTX 50 it runs on a thread-block cluster
  ([section 4](#4-the-rtx-50-code-paths)).
- **The draft head.** It is limited to a vocabulary subset (`draft_vocab.bin`, 106,299 ids) to save VRAM. Upstream
  puts the head at about 180 MiB with that subset ([DETAILS.md](../../upstream/DETAILS.md), "The draft layer's
  tokens"). [guides/data-files.md](data-files.md) describes the subsets.

**Fork changes.** None.

### 3.2 The residual stream: four hyper-connection streams

**Architecture.** A usual transformer carries one residual vector from layer to layer. Flash-Next carries a
*stack* of `hc = 4` vectors, each 2,560 wide (`R`, 10,240 floats). Each half-layer (mixer, then MoE) works like
this:
- `gr_read` collapses the stack to the one vector the sub-block consumes, through a norm and a rank-320 down/up
  projection, and computes a per-stream injection;
- `gr_write` adds the sub-block's output back into every stream, weighted per stream.

The header `include/strata/kernels/gr.hpp:6-8` states it plainly: "The residual is a STACK of `hc` streams ... not
one vector with a gate".

**Code.**
- `gr_read` / `gr_write` (`src/kernels/cuda/gr.cu:344`, `:413`);
- the fused form, which folds the previous write into the next read (`fused_gr.cu`, `layer_set_fused_gr`);
- the block wiring in [core/layer-forward.md](../core/layer-forward.md#residual-handling).

**Memory.** The stack itself is tiny. Its weights are not: `gr.hpp:10-14` records that the `hc_*` tensors are 288
of the shard's 483 BF16 tensors and 1.374 GiB of the 5.305 GiB dense file. Keeping them in BF16, their source type,
rather than widening them to F32, "is the difference between the dense weights fitting in VRAM beside the expert
cache and not fitting at all".

**Fork changes.** None. The Qwen3.6 port added a plain-residual branch (`hc == 1`) beside this one. Flash-Next still
takes the `hc > 1` path ([core/model-geometry.md](../core/model-geometry.md#model-specific-behaviour)).

### 3.3 PLE: the per-layer n-gram embedding on layer 1

**Architecture.** Layer 1 adds information from a learned table indexed by *hashed n-grams* of the input, not by the
token alone:
1. The host hashes the token and its two predecessors into **16 row indices** (`ngram_rows`,
   `src/kernels/ngram.cpp:68`; 64-bit multiply and XOR).
2. It gathers those 16 rows of 160 values from a table of 320,001,536 rows.
3. The GPU's `ple_block` (`src/kernels/cuda/ple.cu:228`) turns the rows into a key and a value, gates them against
   the hidden state (sigmoid of a signed square root), runs a short causal convolution over the history and adds the
   result into the residual stack. `include/strata/kernels/ple.hpp:1-14` has the formula.

**Memory problem.** The table is **28.8 GB**: IQ4_NL, 90 B per row. It sits in shard 2 of the GGUF and is the only
tensor the engine reads straight from the GGUF instead of the pack. A token needs only 16 rows, about 1.4 KB, but
which 16 depends on the text, so the access is random. The rows must also be fetched on the host *before* the GPU
graph for the token runs: a captured graph would freeze the indices (`layer.cpp` PLE notes in
[core/layer-forward.md](../core/layer-forward.md#ple-flash-next-layer-1)).

**How upstream keeps it small.** Three I/O modes (`PleTable`, [kernels/cpu/ngram.md](../kernels/cpu/ngram.md)):

| `--ple-io` | RAM used | How rows are read |
| --- | --- | --- |
| `direct` (upstream default) | ~95 MB row cache | `PleReader`: unbuffered 4 KiB reads from the SSD, issued as soon as the token id is known and collected before layer 1, with an SSD keep-alive ([libs/ngram-ple.md](../libs/ngram-ple.md)) |
| `mmap` | OS file cache, which the OS can reclaim | page faults on the mapped shard |
| `ram` | 28.8 GB | the whole table in memory |

**What the fork changed.**
- **`src/kernels/ngram.cpp:265-287`** (commit `9a53e42`). Upstream's `ram` mode mapped the file and `mlock`ed the
  mapping. Inside a container `ulimit -l` is typically 8 MB, so the lock failed. Upstream then touched every page
  instead, which on a network volume faulted 28.8 GB back from the store one page at a time. The fork:
  - allocates anonymous memory of the table's size;
  - fills it with `read_range` (16 parallel readers, retrying refused reads; see 3.6);
  - drops the file mapping;
  - tries `mlock` only as an extra.

  The memory is resident either way, no later page fault can reach the file, and nothing depends on `ulimit -l`.
  `generate.cpp:2728-2729` maps `ram` to `PleIo::Mmap` with `lock = true`, which is the branch the fork replaced.
- **`frontend/main.cpp:680`.** When the model path names shard 1 of 2, the front end passes `--ple-gguf <shard2>
  --ple-io ram`. This trades 28.8 GB of RAM for no disk I/O during decode. The README's measurement suggests the
  trade matters: upstream with `--ple-io ram` and a sane worker count ran at 78 tokens/s, upstream as its installer
  started it at 20 (per-token table reads plus 127 workers on 32 vCPUs).

**Low-RAM hosts.** A later `-- --ple-io mmap` or `-- --ple-io direct` on the command line overrides the front end's
choice, because user flags come last ([frontend/engine-process.md](../frontend/engine-process.md)). The RTX 4090
service on a 62 GB host runs with `--ple-io mmap`. `direct` wants the shard on an SSD, and `generate.cpp:2751-2758`
warns on a rotational disk.

### 3.4 Gated DeltaNet: 36 layers with a fixed-size state

**Architecture.** Three of every four layers replace attention with a *linear-attention recurrence*. For each of
the 48 value heads the layer keeps a 128 x 128 state matrix `S`. Each token:
1. runs a 4-tap causal convolution over `q|k|v` (10,240 channels) with SiLU;
2. L2-normalizes `q` and `k`;
3. computes a decay `g = exp(softplus(alpha + dt) * a)` in (0, 1) and a write strength `beta = sigmoid(b)`;
4. updates the state with the delta rule, `S <- g*S + k * (beta * (v - g*S^T k))^T`, and reads it out with `q`;
5. RMS-normalizes the output and gates it with **`sigmoid(z)`**. This is Flash-Next's gate; Qwen3.6 uses SiLU.

[kernels/cuda/gdn.md](../kernels/cuda/gdn.md#the-math) has the full math with line references.

**Code.**
- decode: `gdn_layer` (`src/core/layer.cpp:224`), with the three fused kernels in `fused_gdn.cu`;
- the speculative window: `gdn_*_multi` in `verify_kernels.cu`;
- the prompt path: `src/prefill/kernels.cu`.

**Why this matters for memory.** A GDN layer has **no KV cache**. Its state is the same size whether the
conversation is 10 tokens or 262,144: `36 x (128 x 48 x 128 + 10,240 x 3) x 4 B` = about 118 MB in total. That is
why only 12 of the 48 layers grow with context (3.5), and why a long context costs Flash-Next a quarter of what it
would cost a full-attention model of the same depth.

Two more savings follow from the fixed state:
- **Prompt-cache checkpoints are cheap.** A checkpoint is a copy of the running state, about 118 MB, kept in host
  RAM. The KV stays in place. Up to `--prompt-cache 6` are kept
  ([core/conversation/README.md](../core/conversation/README.md)).
- **Speculative decoding needs no state snapshot.** The verify kernels run the window's tokens from the stored state
  without writing it. A separate commit kernel replays only the accepted tokens. So "there is no snapshot of the
  113 MB recurrent state to keep" (`include/strata/kernels/verify_kernels.hpp:4-12`).

**Fork changes.** The Qwen3.6 port added the `GdnGate` parameter (sigmoid / SiLU) and made the kernels generic in
the number of value heads. The sigmoid path is "bitwise what it was before the parameter existed" (`gdn.hpp`, quoted
in [kernels/cuda/gdn.md](../kernels/cuda/gdn.md#gdngate)). Nothing in it is specific to RTX 50.

### 3.5 Sparse attention (QSA): 12 layers that only read 2,051 cells

**Architecture.** Layers 3, 7, 11, ..., 47 are full attention layers: 24 query heads sharing 2 KV heads of width
256, with an output gate `attn * sigmoid(gate)`. Flash-Next's addition is an **indexer** that makes them sparse:
1. Every 4 cells form a block. The indexer keeps one pooled, normalized and rotated key per block (4 heads x 128).
2. Each new token scores every block with `sum_h relu(dot(pooled, q_idx[h]))`.
3. It keeps the best blocks up to **2,048 cells**, plus the incomplete tail block: at most 2,051 cells.
4. Attention runs over those cells only.

A token at position 200,000 therefore reads about 1% of the KV cache, not all of it.

**Code.** [kernels/cuda/attention-qsa.md](../kernels/cuda/attention-qsa.md):

| Step | Kernel |
| --- | --- |
| Indexer append | `indexer_key_append` / `native_qsa_indexer_append` |
| Block scoring | `qsa_block_scores`; on sm_80+ in the prompt path, the TF32 `qsa_block_scores_tc` |
| Selection | `qsa_block_topk`: a radix select over blocks. **On RTX 50 a thread-block cluster version**, see section 4 |
| Attention | split-K `qsa_decode_attn_step`, which reads the KV pools in place through the page table |
| Prompt attention | `qsa_prompt_attn_batch`, on the tensor cores |

**Memory.** KV grows with context, but only on these 12 layers (2 KV heads x 256 x K and V):

| `--kv` | Bytes per token, 12 layers | At 196,608 | At 262,144 |
| --- | ---: | ---: | ---: |
| `fp16` | 24,576 | 4.8 GB | 6.4 GB |
| `int8` (the front end's choice above 8,192 context, `frontend/main.cpp:684`) | 12,672 | 2.5 GB | 3.3 GB |
| `k8v4` (INT8 K, rotated Q4_0 V) | 9,792 | 1.9 GB | 2.6 GB |
| `q4_0` (Walsh-Hadamard rotated) | 6,912 | 1.4 GB | 1.8 GB |

The per-cell sizes come from [kernels/cuda/kv-cache.md](../kernels/cuda/kv-cache.md#formats); the totals are those
times 12 layers. `kv_q8.hpp:3-8` records the reason for INT8: with FP16 and the FP32 pooled indexer keys, "29 KB per
token, 7.65 GB at 262K ... cannot fit 12 GB". That is the 12 GB RTX 5070 that upstream measured on.

**KV streaming is possible because the attention is sparse.** With `--kv-resident N` (Flash-Next only;
[kernels/cuda/kv-cache.md](../kernels/cuda/kv-cache.md#kv-streaming-kv_streamhpp-kv_streamcu)):
- the full K/V lives in pinned host RAM;
- VRAM holds only N cells per layer as 4-cell pages;
- a capturable CLOCK resolver copies the selected-but-missing pages over PCIe inside the decode graph.

Because a token selects at most 2,051 cells, the misses per token are bounded. Qwen3.6, whose attention is dense,
cannot use streaming: every page is needed every token. The RTX 4090 service streams with `--kv-resident 32768`.
VRAM then holds 32,768 cells per layer, 32,768 x 12,672 B = about 415 MB across the 12 layers, while the context is
196,608 and its full 2.5 GB of INT8 K/V sits in pinned RAM.

**Fork changes.** None to the sparse path. The Qwen3.6 port taught the decode kernel a dense mode (`ids == nullptr`)
and the group sizes 2 and 8 beside Flash-Next's 12. Flash-Next's sparse selection and its KV formats are upstream's.

### 3.6 Mixture of experts: 512 experts, 10 per token, and where they live

**Architecture.** After each mixer:
1. A BF16 router scores 512 experts, takes the top 10 and renormalizes their weights (`native_router_top10`,
   `src/kernels/cuda/native_router.cu:110`).
2. Each chosen expert is a small SwiGLU FFN, 2,560 -> 640 -> 2,560.
3. A shared expert of the same shape runs for every token, scaled by its own sigmoid gate.
4. The outputs are combined with the router weights and written back through `gr_write`.

**This is where limited memory is decided.** The [experts overview](../core/experts/README.md) has the full tier
diagram. In short, every `(layer, expert)` pair, 24,576 of them, lives in one of these places:

```
VRAM slot (ExpertCache)        -> GPU grouped kernel        "hit"
pinned RAM (ArenaExpertSource) -> CPU pool computes it in place, or the GPU reads it over PCIe (--pcie-frac)
mapped file (--mmap-experts)   -> page cache, then the CPU pool         (only in the low-RAM modes)
```

The mechanisms that make a 24 GB (or smaller) card work, each from upstream:

| Mechanism | What it does | Code / doc |
| --- | --- | --- |
| **Profile-filled VRAM cache** | At start, `--expert-cache auto` takes all free VRAM after a reserve and fills slots in the rank order of `expert-profile.bin` (48 x 512 ranking, [data-files.md](data-files.md)). **Sized slots** cut each slot to its expert's own blob size and fit about 30% more experts | `generate.cpp:3545-3843`, [program/expert-tiers.md](../program/expert-tiers.md#vram-cache-sizing-3545-3796), [core/experts/vram-cache.md](../core/experts/vram-cache.md) |
| **Pinned RAM arena** | Every expert is loaded into page-locked RAM. Once the VRAM fill is done, the RAM pages of VRAM-held experts are handed back to the OS (`srcp->release`) | [core/experts/expert-sources.md](../core/experts/expert-sources.md) |
| **CPU and GPU at the same time** | The router's ids go to the host through a mapped "doorbell". The CPU pool computes the missed experts in place in RAM while the GPU computes the cached ones and the shared expert. Neither waits for the other to fetch weights | `struct Doorbell` (`include/strata/core/layer.hpp:409`), [core/experts/dispatch.md](../core/experts/dispatch.md) |
| **PCIe share of the misses** | Up to 55% of a layer's distinct missed experts are read by the GPU straight from pinned RAM. The share is scaled down on a link measured below 20 GB/s | `pcie_frac_for_gbps` (`generate.cpp:1275`), [dispatch.md](../core/experts/dispatch.md) |
| **Adaptive swaps** | Every 4 verify rounds, up to 96 of the most-routed non-resident experts swap into the slots of the least-routed resident ones of the same layer, so the cache follows the conversation | [core/experts/resident-ram-and-adaptive.md](../core/experts/resident-ram-and-adaptive.md) |
| **Low-RAM modes** | `--mmap-experts` (file plus page cache), `--resident-experts` / `--resident-budget-gib N` (RAM copy of only what VRAM lacks, or the N GiB the profile ranks hottest), with router lookahead warming the next layer's pages | [resident-ram-and-adaptive.md](../core/experts/resident-ram-and-adaptive.md), [program/expert-tiers.md](../program/expert-tiers.md#resident-ram-mode-4881-4954) |

**What the fork changed, and why.**
- **`include/strata/platform/robust_read.hpp` (new, commit `9a53e42`).** Positional reads for slow or remote file
  systems:
  - `read_at` (`:101`) retries the errors a network store gives for a moment, with a growing pause: ENOMEM, EIO,
    ETIMEDOUT, the connection errors (`read_error_is_transient`, `:84`).
  - `ReadGate` (`:35`) halves the number of concurrent readers each time a read is refused and adds one back after
    32 good reads.
  - `read_range` (`:132`) loads one range with several readers, each on its own file handle, because such a store's
    speed is per stream.
- **`src/core/expert_source.cpp:2676`, inside `load_experts_gguf` (`:2580`).** The loader that copies the experts
  out of the GGUF into the RAM arena now:
  - reads through `read_at` with a shared `ReadGate`, replacing `fopen` / `fseek` / `fread`;
  - prints progress every 4 GiB;
  - on failure, names the file, the offset and the error instead of "short read or unreadable shard".

  It was measured on an RTX 3090 pod: the model loaded from an S3-backed RunPod volume in 12 minutes, through two
  refused reads that previously would have failed the start (README "Status", commit `9a53e42`).
- **`frontend/main.cpp:586` `default_threads()`.** The CPU half of the expert split runs one worker per core. In a
  container the engine saw the host's 256 CPUs while the quota was 32. The front end now reads the cgroup quota,
  halves it when SMT is on, and passes `--pool-workers N`. The README's table shows what this was worth. With upstream
  as its installer started it, at 127 workers, decode ran at 20 tokens/s. Upstream with `--pool-workers 15
  --ple-io ram` ran at 78, and this fork at 107. The README says the remaining gap is not explained.

None of these change which tier an expert is in or how it is computed.

### 3.7 The MTP draft layer and the verify window

**Architecture.** Flash-Next ships one extra multi-token-prediction block: attention plus the same 512-expert MoE,
fed by the trunk's last residual and the next token. Given the trunk's state it predicts a token two positions
ahead, and can be chained to draft several tokens ([core/mtp-draft-head.md](../core/mtp-draft-head.md)).

**Why it matters for memory.** In a memory-bound MoE, the cost of a decode step is mostly *reading experts*. The
verify window runs the last accepted token and the drafts through all 48 layers together. The front end passes
`--spec 4`: up to 3 drafts plus 1 token. Each layer routes all of them at once, **deduplicates the experts they
share**, and computes each needed expert once for the whole window
([core/experts/README.md](../core/experts/README.md#data-flow-of-one-verify-window-layer),
[program/speculative-decoding.md](../program/speculative-decoding.md)). Upstream reports 2.4 to 3.2 tokens per pass.
The 4090 regression run had 61-62% of offered drafts accepted. Fewer passes means fewer expert reads from RAM and
fewer CPU expert computations per token.

The draft head's own routed experts are packed separately (`<mtp>/experts.bin`). Its vocabulary is cut to
`draft_vocab.bin` to keep its VRAM small (3.1). `--spec-min-p 0.5` (front end) stops a draft chain once the drafter's
confidence falls below 0.5.

**Fork changes.** None to the drafter for Flash-Next. The Qwen3.6 port added a second drafter (`qwen35moe`) beside
it. The `[shared]` commits `6abffa7` and `b4ee41e` make the shared code skip the indexer only when a model has none.

### 3.8 Prompt processing

**Architecture.** It is the same model, run on many tokens at a time.

**Memory.** A prompt chunk of thousands of tokens touches nearly *every* expert of every layer, so the VRAM cache
alone cannot serve it. Upstream's prompt path, with `--prefill auto` from the front end
([prefill/README.md](../prefill/README.md)):
- **borrows the expert cache's last slots** for the chunk's buffers instead of reserving VRAM for them, and refills
  them afterwards ([program/expert-tiers.md](../program/expert-tiers.md#lending-cache-slots-to-the-prompt-path));
- **streams the missing experts over PCIe** in a ring while the current layer's attention runs
  ([prefill/expert-streaming.md](../prefill/expert-streaming.md));
- picks the largest chunk on a 256-token grid that keeps that ring at full size.

**RTX 50 relevance.**
- The expert ring is PCIe-bound, so a card's link width matters more here than in decode.
- The tensor-core prompt attention and the TF32 block scorer run on every sm_80+ card, RTX 50 included.
- The fused INT8 tensor-core expert kernels (`STRATA_PF_FUSED`, `src/prefill/moe_fused.cu:399-420`) are **on by
  default only for Q2_0 packs**. For the IQ packs this fork uses they are opt-in with `STRATA_PF_FUSED=1`. Upstream
  measured the IQ3 packs "about even" on an RTX 5070.

**Fork changes.** None to the Flash-Next prompt path. The Qwen3.6 port's `prefill.cpp` changes add its dense and
plain-residual branches.

---

## 4. The RTX 50 code paths

These are the places where the code takes a different path on compute capability 9.0 and up (Hopper and
Blackwell). All are upstream's. Among consumer cards only the RTX 50 series has compute capability ≥ 9.0, so on a
desktop these paths are, in practice, RTX 50-only.

| Path | Condition | What it does | Upstream's measurement (RTX 5070) | Off switch |
| --- | --- | --- | --- | --- |
| **Cluster top-k for the sparse attention** (`block_topk_cluster_kernel`, `src/kernels/cuda/qsa_select.cu:847`; dispatch `:1108`, `:1182-1188`) | compute capability 9.0+ **and** the kernel's own binary built for 9.0+ (`ptxVersion` and `binaryVersion` >= 90, `:1129`), at least one 8-CTA cluster resident, calls of at most 16 queries (decode and verify windows) | One query's block selection is spread over 8 CTAs of 1,024 threads that share histograms through distributed shared memory, instead of 1 CTA doing 4 radix passes over up to 65,538 blocks while the rest of the GPU idles. The ids are identical by construction (`:765-791`) | per call, at capacity = context: 21.9 -> 15.6 µs at 32K, 58 -> 18 at 128K, 200 -> 22 at 262K | `STRATA_QSA_CLUSTER=0` |
| **Cluster greedy argmax** (`sampler_greedy_cluster_kernel`, `src/kernels/cuda/sampler.cu:170-188`; launch `:955-962`; dispatch `:990-993`) | the same checks; greedy sampling with no repetition penalty | The 248,320-logit row (1 MB) is shared by 8 CTAs with 4 loads in flight per thread. Same token, bit for bit | 40 -> 6 µs per call, for every verify window and every draft step | `STRATA_ARGMAX_MULTI=0` |

Together, upstream measured decode at 4K going from 89 to 93.5 tokens/s, and at 128K from 64.5 to 76.4 (Q2_0,
[DETAILS.md](../../upstream/DETAILS.md), engine 0.1.36). The gain grows with context because the one-CTA top-k's
cost grows with the number of blocks. These two paths are the long-context advantage RTX 50 has over RTX 30/40 in
this engine.

**Shared with RTX 30/40 (sm_80+), also used on RTX 50:**
- the TF32 `mma.sync` block scorer (`qsa_block_scores_tc`) in the prompt path;
- the INT8 `cp.async` prompt attention (`launch_i8` in `qsa_prompt_attn.cu`);
- the fused INT8 expert prompt kernels, opt-in for IQ packs (3.8).

**Things RTX 50 does not get here:**
- **FP4.** Blackwell has FP4 tensor cores, and upstream's DETAILS.md mentions an "NVFP4 fork" measured on an
  RTX 5090. No FP4 kernel is in this repository: `include/strata/artifact/gguf_reader.hpp:108-110` only names the
  `MXFP4` and `NVFP4` GGUF types, and two comments in `generate.cpp` and `prefill.cpp` cite NVFP4 measurements. The
  IQ3_XXS experts run on the same `iq_kernels.cu` dot products as on any other card.
- **A faster PCIe share.** `pcie_frac_for_gbps` (`generate.cpp:1270-1277`) gives the full 0.55 share from 20 GB/s
  up. Its comment notes that "an x16 PCIe 4/5 link: ~26-28 GB/s" gets the same share. So a PCIe 5.0 x16 link does
  not raise the share. What it changes is how long that share takes, and the prompt path's ring speed.

**Build-time conditions.**
- **The toolkit.** `sm_120` code needs CUDA **12.8 or newer**; CUDA 12.4's nvcc rejects `compute_120`
  ([docs/qwen36-port/step4-integration.md](../../qwen36-port/step4-integration.md)).
- **The CUDA 12.8 warning.** `CMakeLists.txt:166-170` warns when building `sm_120` with a toolkit older than 13.0:
  "engines built with CUDA 12.8 crashed on long prompts on these cards (issues #220, #224)". The fork's README says
  "12.8+ for RTX 50", and its first Makefile said "upstream recommends 13.0+". That sentence was dropped in commit
  `1e40b06`. **Use CUDA 13.0 or newer for an RTX 50 build.**
- **Card check at start.** `src/core/device.cu:286` requires compute capability 7.5+ (6.0+ in the experimental
  build); 12.0 passes. `device_code_error()` (`device.hpp`) names the cause when the binary has no code for the card.

---

## 5. Every fork change that touches Flash-Next

`git diff 12a7825 HEAD -- src include CMakeLists.txt` is the complete record. Most of its 70 changed files are the
Qwen3.6 port. The rows below are the changes on Flash-Next's path.

| Commit | Files | Change | Why | Memory relevance |
| --- | --- | --- | --- | --- |
| `1b59e5e` | `CMakeLists.txt`, `Makefile`, `.gitmodules` | ggml from the pinned `third_party/llama.cpp` submodule (upstream's own pin, `3cf03257`) instead of a clone during CMake | builds offline, reproducibly | none |
| `1e40b06` | `Makefile:9-16` | `CUDA_ARCHS` default `61;70;75;86;89;120` (was `86;89;120`; upstream CMake's default is `120` alone), filtered against `nvcc --list-gpu-arch` with a note for each dropped arch | one binary for every single NVIDIA card | none. Note: on CUDA 12.4 the filter silently drops `120`; read the `make:` note |
| `e73adc6` | `CMakeLists.txt`, `Makefile` | the chat-parser patch applied at configure time; `make build` turns `STRATA_MMQ_KQUANTS` on | Qwen3.6's K-quant experts | none for Flash-Next |
| `1db37aa`, later front-end commits | `frontend/`, `CMakeLists.txt` | the C++ front end; the engine compiled into the same binary (`frontend/engine_entry.cpp` renames its `main`) and started as a child (`frontend/main.cpp:651`, `:676-692`) | replaces upstream's Python server | chooses the memory flags below |
| (in `1db37aa`, `9a53e42`, later) | `frontend/main.cpp:676-692` | engine flags: `--native <model>` (read the GGUF in place), `--expert-profile <pack>/expert-profile.bin --expert-cache auto`, `--prefill auto`, `--spec 4 --spec-min-p 0.5`, `--mtp`, `--max-context`, `--ple-gguf <shard2> --ple-io ram`, `--kv int8` above 8,192 context, `--pool-workers` from the cgroup quota | fixed, known-good defaults instead of an installer's per-machine choices | sets the PLE mode (3.3), the KV format (3.5) and the CPU worker count (3.6) |
| `9a53e42` | `include/strata/platform/robust_read.hpp` (new, 181 lines) | `ReadGate`, `read_at`, `read_range` | network volumes refuse reads now and then | loading only |
| `9a53e42` | `src/core/expert_source.cpp:2676` | expert loader through `read_at`, progress, a named error | the expert load failed at start on a network volume | loading only |
| `9a53e42` | `src/kernels/ngram.cpp:265-287` | `--ple-io ram`: anonymous memory filled by 16 readers; no dependence on `mlock` | `ulimit -l` in containers; page faults back to the store | same 28.8 GB, but resident and reliable |
| `9a53e42` | `frontend/main.cpp:586` | `default_threads()` from the cgroup quota | 256 visible CPUs, 32-CPU quota | CPU side of the expert split |
| `8df8936` | `patches/llama.cpp-qwen3-coder-any-argument-order.patch` | the Qwen3-Coder tool-call grammar accepts arguments in any order | Flash-Next often writes optional arguments first, and such calls were dropped silently ([build/third-party-and-patches.md](../build/third-party-and-patches.md)) | none |
| `121d23a` | `frontend/main.cpp` | leave the engine's 8-slot reserve out of `max_new` | requests without `max_tokens` were refused | none |
| Qwen3.6 port (`[shared]` commits and the rest) | `layer.cpp`, `verify.cpp`, `mtp.cpp`, `prefill.cpp`, `qsa_decode_attn.cu`, `gdn*.cu`, `generate.cpp`, ... | model geometry read from the GGUF; the GDN gate and attention group size as parameters; dense attention and plain residual branches; a VRAM planner used only for Qwen3.6 | a second model | Flash-Next unchanged. The 4090 regression check gave 90.4 vs 90.6 tokens/s and 2,091 vs 2,097 prompt tokens/s against the pre-port binary ([step5-serve.md](../../qwen36-port/step5-serve.md)) |

**Not in the list: anything for RTX 50.** No kernel, launch bound, arch check or tuning constant was changed for
`sm_120`. The only RTX 50 item the fork added is `120` in its default arch list. Upstream's CMake already built
`120` by default.

---

## 6. Sizing for each RTX 50 card

The fork's README asks for "24 GB or more (RTX 30, 40 or 50)" and 128 GB of RAM for Flash-Next. Nothing in the code
enforces either figure: `--expert-cache auto` takes what the card has, and the arena takes what RAM has. The README
lists the figures because they are the configurations this fork checked. Upstream ran the same model family on a
12 GB RTX 5070 with 64 GB of RAM.

### VRAM

The expert-slot counts below are **estimates**, derived from the two 4090 measurements in section 2 and not
measured on these cards:
- 5,011 more slots took about 8.2 GiB more of the card, so about 1.67 MiB per sized slot, which is the average
  expert size;
- that leaves about 6.7 GiB for everything else at 196,608 context, INT8 KV and `--kv-resident 32768`, including
  the CUDA context and the ~0.4 GiB left free.

So each card gets roughly `(VRAM - 7.1 GiB) / 1.67 MiB` slots.

A different context, KV setting or draft vocabulary moves these figures.

| Card | VRAM | Expert slots (estimate) | Share of 24,576 | Notes |
| --- | ---: | ---: | ---: | --- |
| RTX 5090 | 32 GB | ~15,000 | ~60% | the one desktop RTX 50 card that meets the README's 24 GB |
| RTX 5090 Laptop | 24 GB | ~10,000 | ~40% | the 4090's measured 10,368 is the closest reference (about 90 tokens/s there) |
| RTX 5080, 5070 Ti, 5060 Ti 16 GB | 16 GB | ~5,300 | ~22% | the 4090 capped to 15.4 GB of the card ran at 54 tokens/s with 5,357 slots |
| RTX 5070 | 12 GB | ~3,000 at the service's context; more with a shorter `--ctx` | ~12% | upstream's own measurement card: IQ3_XXS at 62 tokens/s (1K) and 49 (128K) on a Ryzen 5 7600 with 64 GB DDR5, Windows, engine 0.1.26, its installer's settings |
| RTX 5060 Ti 8 GB, 5060, 5050 | 8 GB | ~0 after the fixed set at long context | - | below what upstream supports ("12 GB VRAM or more (8 GB runs, slowly)"); expect mostly CPU experts |

On a card under 24 GB, these levers free VRAM for expert slots, in rough order of effect:
- **A shorter `--ctx`.** It shrinks the RoPE table and the KV allocations.
- **`-- --kv-resident 32768`.** With it, the KV beyond 32,768 cells lives in RAM.
- **`-- --kv k8v4` or `q4_0`.** These halve the KV again.
  [kernels/cuda/kv-cache.md](../kernels/cuda/kv-cache.md) has the quality notes.
- **A smaller draft vocabulary.** Copying `data/draft_vocab_en.bin` over `<mtp>/draft_vocab.bin` saves about
  110 MiB, at the cost of almost no drafts for CJK text ([data-files.md](data-files.md)). The engine reads whatever
  `draft_vocab.bin` the `--mtp` directory holds.

### Host RAM

| Item | Size | When |
| --- | ---: | --- |
| Expert arena (pinned) | ~40 GiB at load. The VRAM-held experts' pages are given back afterwards, so 40 GiB minus the card's share | default |
| PLE table | 28.8 GB | `--ple-io ram` (the front end's default); ~0.1 GB with `direct`; OS cache with `mmap` |
| KV host copy | context x 12,672 B (2.5 GB at 196,608) | with `--kv-resident` |
| Prompt-cache checkpoints | up to 6 x ~118 MB | serve mode |
| Engine, front end, OS | a few GB | always |

That comes to roughly 75-80 GB at peak with `--ple-io ram`. The README's 128 GB leaves room for the OS file cache
to hold the 76 GB of model files, which is what made the 19 s warm start possible. On 64 GB, use `-- --ple-io
direct` (the table on an SSD) or `mmap`, and consider `-- --resident-budget-gib N`
([program/expert-tiers.md](../program/expert-tiers.md#resident-ram-mode-4881-4954)). The 4090 service on 62 GB
runs this way (`--ple-io mmap`): its first two passes after a start ran at 60 tokens/s while the file cache filled.

**The PCIe link.** RTX 50 desktop cards use PCIe 5.0. The 5060 Ti and below use an x8 link, which is about PCIe 4.0
x16 in a PCIe 5 slot and half that in a PCIe 4 board. The engine measures the link at start (`probe_pcie_h2d_gbps`,
`generate.cpp:1212`) and prints the result with the PCIe share it chose. Below 20 GB/s it scales the share down in
proportion.

---

## 7. Building and running on an RTX 50 card

**Build.** CUDA 13.0 or newer, because of the CMake warning in section 4.

```
git submodule update --init
make build CUDA_ARCHS=120          # RTX 50 only; a short build
# or: make build                   # the full list; with CUDA 13, 61 and 70 are dropped with a note
```

- With a toolkit older than 12.8, the default `make build` drops `120` with the note `make: nvcc cannot compile CUDA
  arch(s) 120`. The binary then has no Blackwell code: it either fails at start with `device_code_error()`'s message
  or runs without the cluster kernels.
- When `CUDA_ARCHS` is given on the make command line, the Makefile's filter does not apply (a command-line variable
  overrides the makefile's assignment). `CUDA_ARCHS=120` with an old toolkit therefore fails in nvcc instead of
  being dropped ([build/makefile.md](../build/makefile.md)).
- `third_party/llama.cpp`'s patch is applied by `make build`; nothing is downloaded.

**Run.** The standard command line from the README; nothing is RTX 50-specific:

```
./build/unbiased-strata \
  --model /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --pack  /models/pack-iq3-xxs  --mtp /models/mtp  --ctx 196608 \
  -- --kv-resident 32768                 # optional: the KV window in VRAM, the rest in RAM
     # --ple-io mmap                     # optional: under ~96 GB of RAM
```

**What to check on the first start, since no RTX 50 card has run this fork:**

1. **The card and its code.** The engine prints `strata generate: GPU 0: <name>, compute capability 12.0`
   (`generate.cpp:3438`). A `device_code_error()` message means the build has no `sm_120` code.
2. **The cluster kernels.** Build and run the parity program:

   ```
   cmake --build build --target decode_cluster_parity
   ./build/decode_cluster_parity --selftest
   ```

   It is registered in any CUDA build (`CMakeLists.txt:835-840`), not gated by `STRATA_BUILD_TESTS`. It checks the
   cluster top-k and argmax against the one-CTA kernels, bitwise and under graph capture, and **skips below sm_90**.
   A pass on an RTX 50 card is the first evidence that upstream's Blackwell paths work in this build.
   `--bench` prints the per-call times quoted in section 4.
3. **The plan.** The start-up log shows:
   - the expert-cache slot count and the VRAM left;
   - the PCIe probe and the chosen share;
   - `PLE table loaded` with the time;
   - with `--kv-resident`, the resident KV cells.

   Compare them with section 6.
4. **Long prompts.** The crash in issues #220 and #224 appeared on long prompts with CUDA 12.8 builds. A 32K-token
   request is the check.
5. **Serving.** `make test-serve PORT=<port>` and the front-end checks in [guides/testing.md](testing.md).
6. **Speed and tiers.** `GET /health` reports VRAM hits, RAM reads and file reads per request
   ([frontend/http-api.md](../frontend/http-api.md)). For an A/B, `STRATA_QSA_CLUSTER=0 STRATA_ARGMAX_MULTI=0`
   turns the RTX 50 paths off with the same tokens.

---

## See also

- [core/model-geometry.md](../core/model-geometry.md), [core/layer-forward.md](../core/layer-forward.md): the
  architecture as the engine encodes it.
- [kernels/cuda/attention-qsa.md](../kernels/cuda/attention-qsa.md), [kernels/cuda/gdn.md](../kernels/cuda/gdn.md),
  [kernels/cuda/kv-cache.md](../kernels/cuda/kv-cache.md), [kernels/cpu/ngram.md](../kernels/cpu/ngram.md),
  [libs/ngram-ple.md](../libs/ngram-ple.md): the kernels behind each part.
- [core/experts/README.md](../core/experts/README.md), [program/expert-tiers.md](../program/expert-tiers.md): the
  expert tiers in full.
- [libs/platform.md](../libs/platform.md): `robust_read.hpp` among the platform helpers.
- [frontend/engine-process.md](../frontend/engine-process.md): the engine command line the front end builds.
- [UPSTREAM.md](../../../UPSTREAM.md): the fork's change list and how the pack and draft head were made.
- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md): upstream's RTX 5070 measurements and its "How it works".
