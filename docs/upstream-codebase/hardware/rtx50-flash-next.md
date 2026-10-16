[Upstream docs](../README.md) › [Hardware](README.md) › Qwen3.8-Flash-Next on an RTX 50 card

# Qwen3.8-Flash-Next on an RTX 50 card: the architecture, the memory and the code

**Purpose.** This page follows upstream Strata (engine 0.1.39, commit `6f32ec0`) running Qwen3.8-Flash-Next on one
NVIDIA RTX 50 card (Blackwell, compute capability 12.0, `sm_120`). It walks the model's architecture part by part
and, for each part, names the code that runs it, what it costs in VRAM and RAM and what keeps that cost low. It then
covers the code paths that differ on RTX 50, what the installer chooses for each RTX 50 card, sizing per card and
model variant, and how to build, run and check it.

**Files covered.** `setup.py`, `serve/server.py`, `CMakeLists.txt`, `Dockerfile`, `src/program/generate.cpp`,
`src/kernels/cuda/qsa_select.cu`, `src/kernels/cuda/sampler.cu`, `src/kernels/decode_cluster_parity.cpp`,
`src/prefill/moe_fused.cu`, `src/kernels/ngram.cpp`, `src/core/device.cu`, and the headers cited below. The
[area README](README.md#source-map) links each by permalink.

## Read this first

1. **RTX 50 is the engine's home architecture.** `CMakeLists.txt:139-141` defaults `CMAKE_CUDA_ARCHITECTURES` to
   `120`. `CMakeLists.txt:144-146` says the engine was "developed and measured on sm_120 (RTX 50)" before the port
   down to RTX 20, 30 and 40. `include/strata/core/device.hpp:51-53` still says "The engine targets sm_120
   specifically". Every speed in upstream's own tables was measured on a 12 GB RTX 5070.
2. **Upstream's measured machine.** RTX 5070 12 GB (PCIe 5.0 x16), Ryzen 5 7600 (6 cores, AVX-512), 64 GB
   DDR5-5200, Windows, with the settings `setup.py` writes ([DETAILS.md](../../upstream/DETAILS.md), "Speed
   (measured)"; `bench/results/2026-09-29-speed-0126/README.md`). A community member ran an RTX 5090 on Linux
   (`bench/results/2026-09-30-community-rtx-5090/README.md`). Section 6 has both.
3. **The ready-made engine is built with CUDA 13.0.** `setup.py:104-106`: "the CUDA libraries the ready-made engine
   loads (the same CUDA 13.0 it is built with)", from NVIDIA's pip wheels, and `MIN_DRIVER = 580`. CUDA 13 matters
   on RTX 50: `CMakeLists.txt:166-170` warns that `sm_120` engines built with CUDA 12.8 "crashed on long prompts on
   these cards (issues #220, #224). Use CUDA 13.0 or newer."
4. **Two decode kernels are RTX 50-only in practice.** The sparse-attention block top-k and the greedy argmax run on
   8-CTA thread-block clusters on compute capability 9.0 and up (section 4). Among consumer cards only RTX 50 has
   that. Same tokens, less time per call, and the gain grows with context.
5. **What runs the card.** The installer (`setup.py`, via `START-HERE.bat` or `setup.sh`) writes
   `strata-<model>.json`; `serve/server.py` starts `strata --serve` with its `"args"`. Nothing in the RTX 50 path is
   set by hand. Section 5 shows the command line the installer produces for each RTX 50 card.

Contents:

1. [The model in one picture](#1-the-model-in-one-picture)
2. [Where the bytes are](#2-where-the-bytes-are)
3. [Part by part: architecture, code, memory](#3-part-by-part-architecture-code-memory)
4. [The RTX 50 code paths](#4-the-rtx-50-code-paths)
5. [What the installer chooses on an RTX 50 card](#5-what-the-installer-chooses-on-an-rtx-50-card)
6. [Sizing for each RTX 50 card and model variant](#6-sizing-for-each-rtx-50-card-and-model-variant)
7. [Building, running and checking](#7-building-running-and-checking)
8. [Tests](#8-tests)
9. [Gotchas](#9-gotchas)
10. [Compared with UnbiasedStrata](#10-compared-with-unbiasedstrata)

---

## 1. The model in one picture

Qwen3.8-Flash-Next has about 125B parameters (README). It is a mixture-of-experts (MoE) model: each token uses a
small, token-dependent part of the weights. It combines:
- linear-attention (Gated DeltaNet) layers;
- sparse attention whose cells are picked by an indexer ("QSA" in the code);
- a residual stream of four parallel copies (hyper-connections, "gated residual" in the code);
- a hashed n-gram embedding table on layer 1 (PLE);
- a built-in multi-token-prediction (MTP) draft layer.

The shape is fixed in `struct ModelGeometry` (`include/strata/core/layout.hpp:27-59`); upstream does not read it from
the GGUF, except the MoE shape that "the model file overrides" (`src/program/generate.cpp:2060`) - that is how the
Coder runs with 256 experts per layer.

```
 token id --+--> token embedding (248,320 x 2,560)        mapped pinned host RAM, one row over PCIe per token
            +--> n-gram hash of (token, 2 previous) -> 16 rows of the PLE table (28.8 GB, GGUF shard 2, on the SSD)
                                                               |
 residual stack R: 4 streams x 2,560  (hc = 4)                 |
 for layer l = 0 .. 47:                                        |
   gr_read(R) -> one 2,560-wide input  ------------------------+ layer 1 only: the PLE block adds into R
   mixer:  l % 4 != 3 -> Gated DeltaNet (36 layers)   fixed-size recurrent state, no KV cache
           l % 4 == 3 -> sparse attention (12 layers)  KV cache; an indexer picks <= 2,051 cells per token
   gr_write(R, mixer out); gr_read(R) -> MoE input
   MoE:    router over 512 experts -> top 10, plus 1 shared expert (each 2,560 -> 640 -> 2,560)
   gr_write(R, MoE out)
 head:     gr_read(R) -> rms_norm -> output (2,560 x 248,320) -> logits
 MTP:      one more block drafts the next tokens; the 48 layers check them together (the verify window)
```

| Part | Value | Where it is declared |
| --- | --- | --- |
| Layers | 48: 36 Gated DeltaNet + 12 attention (`qsa_interval` 4) | `layout.hpp:28-30`, `is_qsa_layer` `layout.hpp:63` |
| Width | `n_embd` 2,560 | `layout.hpp:28` |
| Residual | `hc` 4 streams, low rank `hc_lr` 320 | `layout.hpp:48-49` |
| Gated DeltaNet | state 128, 16 key heads, 48 value heads, conv 4 taps over 10,240 channels, output gate `sigmoid(z)` | `layout.hpp:33-38`, `include/strata/kernels/gdn.hpp:90` |
| Attention | 24 query heads, 2 KV heads, head dim 256, rotary on 64 dims | `qsa_real_shapes()` `include/strata/kernels/qsa.hpp:109` |
| Indexer | 4 heads x 128, one pooled key per 4 cells, 2,048-cell budget + the incomplete tail = 2,051 | `qsa.hpp:112`, `qsa_selection_width` `qsa.hpp:133` |
| Experts | 512 routed (256 in the Coder), 10 per token, 1 shared, FFN width 640 | `layout.hpp:52-53` |

**Why it fits at all: most of it is experts.** Derived from the geometry, not quoted:
- one expert is `3 x 2,560 x 640` = 4.9M parameters;
- 512 x 48 = 24,576 experts, about 121B parameters, about 96% of the 125B;
- a token routes to 10 per layer: 480 experts, about 2.4B parameters.

So the per-token compute is small, but which 2% of the experts a token needs is decided only at its router. Every
memory technique below manages that fact.

---

## 2. Where the bytes are

The routed experts' size depends on the variant. The totals below are the native-pack tables in
`tests/data/native_experts/*.txt` (first line of each), the per-expert range is over the 48 layers, and "RAM"
is what `setup.py`'s `MODELS` table (`setup.py:122-153`) books for the arena and as the RAM to recommend.

| Variant | Routed experts | Bytes per expert | `arena_gb` / `ram_gb` in setup | Download |
| --- | --- | --- | --- | ---: |
| Q2_0 | 24,576, 33,973,862,400 B | 1.32 MiB, every layer | 34.0 / 48 | 66 GB |
| IQ2_XS (also Swift 1.5) | 24,576, 35,454,976,000 B | 1.12-1.44 MiB | 35.5 / 48 | 68 GB |
| IQ3_XXS | 24,576, 42,912,972,800 B | 1.25-2.22 MiB (1.67 average) | 42.9 / 60 | 76 GB |
| IQ3_S | 24,576, 50,292,326,400 B | 1.44-2.54 MiB (1.95 average) | 50.3 / 62 | 84 GB |
| Coder (IQ1_M) | 12,288 (256 per layer), 25,146,163,200 B | 1.44-2.54 MiB | 23.4 / 32 | 58 GB |
| Unsloth UD-IQ4_XS | - | - | 59.5 / 48 (RAM budget, rest from SSD) | 94 GB |
| Unsloth UD-Q4_K_XL (experimental) | - | - | 77.0 / 48 (RAM budget, rest from SSD) | 111 GB |

The other pieces, the same for every variant:

| Piece | Size | Default home | What decides it |
| --- | --- | --- | --- |
| PLE n-gram table | 28.8 GB (GGUF shard 2) | the SSD: `--ple-io direct`, a ~94 MB row cache (`--ple-row-cache` 1,048,576 rows of 90 B, `generate.cpp:306`) | `--ple-io direct\|mmap\|ram` (3.3) |
| Dense weights (mixers, routers, shared experts, hyper-connection weights, head) | a few GiB (the canonical dense file is 5.305 GiB, `include/strata/kernels/gr.hpp:12`; "~3.4 GB for the Coder", MULTI_GPU.md) | VRAM | always on the card |
| Token embedding | 248,320 rows | mapped pinned host RAM, no VRAM | `NativeEmbed` (`include/strata/core/native_head.hpp:40-41`) |
| KV cache, 12 attention layers | INT8: 12,672 B per token (`include/strata/kernels/kv_q8.hpp:8`) | VRAM, or RAM with a 32,768-cell VRAM window | `--kv`, `--kv-resident` (3.5) |
| Gated DeltaNet state, 36 layers | ~118 MB at any context | VRAM | geometry |
| MTP draft layer | "~0.8 GB" (MULTI_GPU.md); "839 MiB" plus the head (`generate.cpp:2729`) | VRAM | `--mtp`, `draft_vocab.bin` (3.7) |
| The expert cache | whatever VRAM is left after the rest and a 700 MiB reserve (`--vram-reserve-mib`, `generate.cpp:424`) | VRAM | `--expert-cache auto` (3.6) |

---

## 3. Part by part: architecture, code, memory

Each part follows the same order: what the architecture does; the code that runs it; what it costs and what keeps
the cost down; its configuration.

### 3.1 Token embedding, output head and the vocabulary

**Architecture.** A token id selects one row of a 248,320 x 2,560 table. At the end, the output head multiplies
the final hidden vector by a 2,560 x 248,320 matrix: one logit per vocabulary entry.

**Code and memory.**
- `NativeEmbed` (`native_head.hpp:40-41`) keeps the table "in mapped pinned host memory: a row is read over PCIe
  per token, so the table costs no VRAM".
- The output head stays in VRAM: every token needs all of it.
- Greedy argmax reads a 1 MB row of logits per call. On RTX 50 it runs on a thread-block cluster (section 4).
- The draft layer's head is limited to a token subset, `<mtp>/rt/draft_vocab.bin`. The default subset (`cjk`,
  106,299 ids) takes "~180 MiB of VRAM" ([DETAILS.md](../../upstream/DETAILS.md), "The draft layer's tokens");
  `setup.py:3241` books "up to" 348 MiB for it (IQ3_S, the largest), 133 MiB for `en`, 193 for `cyrillic`, 151 for
  `fr`. The expert cache is sized after the head is reserved (`include/strata/core/mtp.hpp:58`).

**Configuration.** `--draft-vocab cjk|en|cyrillic|fr` at setup (`refresh_draft_vocab`, `setup.py:3366`, copies
`data/draft_vocab_<x>.bin` over `<mtp>/rt/draft_vocab.bin`). Setup suggests `en` on cards under 14 GB but changes
nothing by itself (`draft_vocab_note`, `setup.py:3245`).

### 3.2 The residual stream: four hyper-connection streams

**Architecture.** The model carries a stack of `hc = 4` vectors of 2,560, not one residual vector. Each half-layer:
- `gr_read` collapses the stack to the one vector its sub-block consumes (a norm and a rank-320 down/up projection)
  and computes a per-stream injection;
- `gr_write` adds the sub-block's output back into every stream, weighted per stream.

`include/strata/kernels/gr.hpp:5-7`: "The residual is a STACK of `hc` streams, each `n_embd` wide - not one vector
with a gate."

**Code.** `gr_read` / `gr_write` (`src/kernels/cuda/gr.cu:344`, `:413`); a fused form that folds the previous write
into the next read (`--no-fused-gr` is the A/B arm, `generate.cpp:642`).

**Memory.** The stack is tiny; its weights are not. `gr.hpp:9-13`: the `hc_*` tensors are 288 of the shard's 483 BF16
tensors and 1.374 GiB of the 5.305 GiB dense file. Keeping them in BF16 "is the difference between the dense
weights fitting in VRAM beside the expert cache and not fitting at all".

**Across GPUs.** A layer split hands this stack, plus one input vector and the stream weights, to the next card:
`handoff_floats(g)` = `hc x n_embd + n_embd + hc` floats per token (`include/strata/core/verify.hpp:126`).

### 3.3 PLE: the n-gram embedding on layer 1

**Architecture.** Layer 1 adds information from a learned table indexed by hashed n-grams of the input:
1. the host hashes the token and its two predecessors into 16 row indices (`ngram_rows`,
   `src/kernels/ngram.cpp:67`);
2. it gathers those 16 rows of 160 values from a table of about 320 million rows;
3. the GPU's `ple_block` (`src/kernels/cuda/ple.cu:228`) turns them into a key and a value, gates them against the
   hidden state (sigmoid of a signed square root), runs a short causal convolution and adds the result to the
   stack. The formula is in `include/strata/kernels/ple.hpp:3-13`.

**Memory problem.** The table is 28.8 GB (IQ4_NL, 90 B per row). It is in GGUF shard 2, which the Coder shares with
the original. A token needs 16 rows, about 1.4 KB, but which ones depends on the text: the reads are random.

**How upstream keeps it small: the table stays on the SSD.** Three modes (`PleTable`, `ngram.cpp:138-290`):

| `--ple-io` | RAM used | How rows are read | Where it is used |
| --- | --- | --- | --- |
| `direct` (default, `generate.cpp:305`) | ~94 MB row cache | unbuffered reads straight from the SSD, up to 256 in flight (`--ple-inflight`), issued as soon as the token id is known; "the table never enters RAM or the file cache" (`generate.cpp:548-550`); the SSD is kept awake (`STRATA_SSD_KEEPALIVE`, `generate.cpp:2393-2403`) | every install on an SSD |
| `mmap` | OS file cache | page faults on the mapped shard | "A/B arm" |
| `ram` | 28.8 GB | the mapping with `madvise(WILLNEED)` and `mlock`; if the lock fails ("raise `ulimit -l`") it touches every page instead (`ngram.cpp:262-279`). Linux only: Windows refuses it, "no mlock; use --ple-io mmap" (`generate.cpp:1745-1750`) | setup adds it on Linux when the model is on a rotational disk and RAM holds the table (`setup.py:4238-4250`, #605) |

The engine warns at start when `direct` reads from a rotational disk (`generate.cpp:2410-2418`), and the stall report
names it ("the engine is waiting for the drive", `generate.cpp:950`).

### 3.4 Gated DeltaNet: 36 layers with a fixed-size state

**Architecture.** Three of every four layers replace attention with a linear-attention recurrence. Each of the 48
value heads keeps a 128 x 128 state matrix. Each token runs a 4-tap causal convolution over `q|k|v` (10,240
channels), L2-normalizes `q` and `k`, decays and updates the state with the delta rule, reads it with `q`, and gates
the RMS-normalized output with `sigmoid(z)` (`gdn.hpp:90`).

**Code.** Decode: `gdn_layer` (`src/core/layer.cpp:223`) and the fused kernels in `src/kernels/cuda/fused_gdn.cu`;
the verify window: `verify_kernels.cu`; prompts: the prompt path's own kernels ([prefill](../prefill/README.md)).

**Why it matters for memory.** A DeltaNet layer has no KV cache. Its state is the same size at 10 tokens or
262,144: `36 x (128 x 48 x 128 + 10,240 x 3) x 4 B` = about 118 MB (derived). So only 12 of 48 layers grow with
context. Two more savings follow:
- **Conversation checkpoints are cheap.** One is "~118 MB of host RAM" and the KV stays where it is; up to 6 are
  kept (`--prompt-cache`, `generate.cpp:498-500`, `:615`).
- **The verify window needs no state snapshot.** The verify kernels run the window's tokens without writing the
  state; a commit kernel replays only the accepted ones, so "there is no snapshot of the 113 MB recurrent state to
  keep" (`include/strata/kernels/verify_kernels.hpp:4-12`).

### 3.5 Sparse attention: 12 layers that read at most 2,051 cells

**Architecture.** Layers 3, 7, ..., 47 are attention layers: 24 query heads over 2 KV heads of 256, with an output
gate. An indexer makes them sparse:
1. every 4 cells form a block; the indexer keeps one pooled key per block;
2. each new token scores every block;
3. it keeps the best blocks up to 2,048 cells plus the incomplete tail: `min(n_kv, 2,048 + 4 - 1)` cells
   (`qsa_selection_width`, `qsa.hpp:127-136`);
4. attention runs over those cells only.

**Code.**

| Step | Kernel | Note |
| --- | --- | --- |
| Block scoring, prompt path | `qsa_block_scores_tc` (`qsa_select.cu:1015`) | TF32 `mma.sync`, sm_80+; older cards keep the warp kernel (`qsa_select.cu:1038`) |
| Selection, decode | `qsa_block_topk` (`qsa_select.cu:1174`) | on RTX 50 the cluster kernel (section 4); else one CTA per query |
| Prompt attention | `qsa_prompt_attn.cu` | MMA on sm_75+, `cp.async` INT8 path on sm_80+ (`qsa_prompt_attn.cu:30-32`) |

**Memory.** KV grows with context on these 12 layers only:

| `--kv` | Bytes per cell, one layer | 12 layers, per token | At 131,072 | At 262,144 |
| --- | ---: | ---: | ---: | ---: |
| `fp16` (the engine's default; setup's up to 8K) | 2,048 | 24,576 | 3.2 GB | 6.4 GB |
| `int8` (setup's default above 8K) | 1,056 | 12,672 | 1.7 GB | 3.3 GB |
| `k8v4` (INT8 K, rotated 4-bit V; PR #120) | 816 | 9,792 | 1.3 GB | 2.6 GB |
| `q4_0` (Hadamard-rotated 4-bit; PR #21) | 576 | 6,912 | 0.9 GB | 1.8 GB |

Per-cell sizes: `generate.cpp:555-560` and `setup.py:4254`; totals are derived. `kv_q8.hpp:3-8` gives the reason
for INT8: FP16 with the FP32 pooled indexer keys took "29 KB per token, 7.65 GB at 262K ... which cannot fit 12 GB" -
the RTX 5070 upstream measured on.

**KV streaming is possible because the attention is sparse.** `--kv-resident N` (N at least 20,480,
`generate.cpp:561`) keeps the full K/V in pinned host RAM and only N cells per layer in VRAM; the cells a token
selects but VRAM lacks are copied over PCIe inside the decode graph. A token selects at most 2,051 cells, so the
misses per token are bounded. Setup turns it on from 64K with `--kv-resident 32768` when RAM allows it, at
"~13.7 KB of RAM per context token" (13 x 1,056 B: the 12 layers plus the draft layer, `setup.py:4254`).
Measured on the RTX 5070: Q2_0 at 262K went "50.9 -> 62.6 tokens/s (1,589 -> 3,872 experts in VRAM); at 128K
about +6%" (DETAILS.md, "KV streaming"). Not with `k8v4` (`generate.cpp:1769`) and not under WSL (`setup.py:3002`).

### 3.6 Mixture of experts: 512 experts, 10 per token, and where they live

**Architecture.** After each mixer a BF16 router scores 512 experts and keeps the top 10
(`native_router_top10`, `src/kernels/cuda/native_router.cu:110`). Each is a small SwiGLU FFN; a shared expert runs
for every token. The outputs are combined and written back through `gr_write`.

**Where every expert lives.** Each `(layer, expert)` pair is in one of these tiers:

```
VRAM slot (the expert cache)        -> GPU grouped kernel                                   "hit"
pinned RAM (the expert arena)       -> the CPU pool computes it in place, or the GPU reads it over PCIe
mapped file (--mmap-experts, ...)   -> the OS file cache, then the CPU pool                 (low-RAM modes only)
```

The mechanisms that make a 12-32 GB card work:

| Mechanism | What it does | Code |
| --- | --- | --- |
| Profile-filled VRAM cache | `--expert-cache auto` takes the free VRAM after the reserve and fills it in the rank order of the expert profile (`data/expert-profile.bin`, or `expert-profile-coder.bin`). For a native pack each slot is cut to its own expert's blob size: "the same VRAM holds ~30% more IQ3_XXS experts than slots of the largest blob would" | `generate.cpp:3294-3345` |
| The arena in pinned RAM | every expert of the variant is copied into page-locked RAM and stays there, the VRAM-held ones included (`ArenaExpertSource`, `src/core/expert_source.cpp`); on Windows it is pinned in per-layer slices (a `cudaHostRegister ... out of memory` line is "Normal on Windows", DETAILS.md, "Troubleshooting") | `generate.cpp:3146-3163` |
| CPU and GPU at the same time | the router's ids reach the host through a mapped doorbell (`struct Doorbell`, `include/strata/core/layer.hpp:405`); the CPU pool computes the misses in place while the GPU computes the hits and the shared expert | the pool: one worker per physical core but the host's (`generate.cpp:341`) |
| PCIe share of the misses | for a native pack the GPU reads 55% of a layer's distinct misses from pinned RAM over PCIe, scaled down in proportion below 20 GB/s (`pcie_frac_for_gbps`, `generate.cpp:1222`; the probe `generate.cpp:1159`, used at `:2035-2050`); the canonical Q2_0 pack keeps 0.2, unprobed | `--pcie-frac`, `--calibrate` |
| Adaptive swaps | every 4 verify rounds up to 96 of the most-routed missing experts replace the least-routed resident ones (`--adapt-every 4`, `--adapt-swaps 96`, `generate.cpp:472-474`, `:497`) | the cache follows the conversation |
| Low-RAM modes (chosen by setup) | `--mmap-experts` maps the pack's `experts.bin`; `--resident-experts` copies only what the GPU lacks into RAM; `--resident-budget-gib N` keeps the hottest N GiB in RAM and reads the rest from the files (the Unsloth variants) | DETAILS.md, "Low-RAM mode" |

Upstream reports the effect as the decode hit rate: on one RTX 3090 with IQ3_XXS, 10,584 of 24,576 experts in
VRAM gave 94-97% hits (see [rtx3090-flash-next.md](rtx3090-flash-next.md)).
The ranking puts the experts routed most first, so the hit rate is far above the share of experts held.

### 3.7 The MTP draft layer and the verify window

**Architecture.** One extra block (attention plus a MoE) predicts the token after next from the trunk's last state;
chained, it drafts several. The 48 layers then check the last token and the drafts in one verify window.

**Why it matters for memory.** In a memory-bound MoE most of a step's cost is reading experts. In a verify window
each layer routes all of the window's tokens at once and computes each needed expert once for all of them. Setup
passes `--spec 4 --spec-min-p 0.5` (`setup.py:4226`): up to 3 drafts, stopped once the drafter's confidence falls
below 0.5. Prompt lookup (`--suffix-draft 3` by default, `generate.cpp:633`) drafts up to 5 tokens from an earlier
copy in the context. Upstream reports "2.4-3.2 tokens per pass on average" (DETAILS.md, "How it works").

The draft layer's experts are packed as Q2_0 for every variant (`tools/mtp_pack.py --experts q2_0`,
`setup.py:4205`) and live in VRAM with its head.

### 3.8 Prompt processing

**Architecture.** The same model run on many tokens at a time.

**Memory.** A chunk of thousands of tokens touches nearly every expert of every layer, so the VRAM cache cannot
serve it alone. With `--prefill auto` ([prefill](../prefill/README.md)):
- chunks of up to 8,192 tokens, "the largest chunk whose buffers fit in the expert-cache slots it borrows"
  (DETAILS.md, "Faster prompts"); the borrowed slots are refilled afterwards;
- the next layer's missing experts stream over PCIe in a ring while the current layer's attention runs; since
  0.1.39b the ring is sized in bytes per pack (DETAILS.md, "How it works");
- the experts are multiplied by llama.cpp's MMQ kernels or by the fused INT8 kernels.

**The fused INT8 prompt kernels (`STRATA_PF_FUSED`).** `src/prefill/moe_fused.cu:399-417`:
- `available()`: compute capability 8.0 or newer, so every RTX 30, 40 and 50 card;
- `enabled()`: on unless `STRATA_PF_FUSED=0`, and used for the canonical (non-native) Q2_0 pack, which setup
  builds only on an AVX-512 CPU (`setup.py:4175-4182`; `src/prefill/prefill.cpp:2268-2269`);
- `requested()`: the native packs' kernels run only with `STRATA_PF_FUSED=1`.

Measured on the RTX 5070 with engine 0.1.36: Q2_0 prompts "4K 1,294 -> 1,570, 32K 2,170 -> 2,653, 128K 2,123 ->
2,468 tokens/s (+16-22%)"; with `STRATA_PF_FUSED=1`, "IQ2_XS prompts +12% at 4K, +3% at 32K, the IQ3 packs about
even" (DETAILS.md, "Speed (measured)").

**RTX 50 relevance.** The streamed ring is PCIe-bound, so the link matters more here than in decode. RTX 50
desktop cards are PCIe 5.0: the RTX 5090 run probed 50.0 GB/s host-to-device, the 5070 is PCIe 5.0 x16.

---

## 4. The RTX 50 code paths

These paths change on compute capability 9.0 and up. Among consumer cards only RTX 50 has it, so on a desktop they
are RTX 50-only.

| Path | Condition | What it does | Measured (RTX 5070) | Off switch |
| --- | --- | --- | --- | --- |
| Cluster top-k for the sparse attention (`block_topk_cluster_kernel`, `qsa_select.cu:847`; `qsa_block_topk_cluster`, `:1108`; dispatch `:1181-1188`) | the device's compute capability 9.0+ and the kernel's own code built for it (`ptxVersion` and `binaryVersion` >= 90, `:1117-1130`), checked once per device; for a larger capacity, a cluster of 8 such CTAs must fit (`:1135-1153`); calls of at most 16 queries (`CL_MAXQ`, `:794`): decode and verify windows | one query's selection spread over 8 CTAs of 1,024 threads that share digit histograms through distributed shared memory, instead of one CTA doing four radix passes over up to 65,538 blocks "on ONE SM while the rest of the GPU idles" (`:768-775`). "IDENTICAL IDS, by construction" (`:778-785`) | per call at capacity = context: "21.9 -> 15.6 us at 32K, 58 -> 18 at 128K, 200 -> 22 at 262K" (`:775-776`) | `STRATA_QSA_CLUSTER=0`; `STRATA_TOPK_OLD=1` also skips it |
| Cluster greedy argmax (`sampler_greedy_cluster_kernel`, `sampler.cu:202`; `sample_greedy_cluster`, `:928`; dispatch `:989-997`) | the same checks; greedy decoding with no repetition-penalty window | the 248,320-logit row (1 MB) shared by 8 CTAs with four loads in flight per thread; the same token bit for bit (`sampler.cu:172-181`) | "40 us per call ... 6 us", for every verify window and every draft step (`:173-176`) | `STRATA_ARGMAX_MULTI=0` |

Together, upstream measured Q2_0 decode on the RTX 5070 with engine 0.1.36 "at 4K 89 -> 93.5, at 128K 64.5 -> 76.4
tokens/s" (DETAILS.md, "Speed (measured)"). The gain grows with context because the one-CTA top-k's cost grows with
the number of blocks. Both kernels compile to a trap below `__CUDA_ARCH__` 900 (`STRATA_SEL_CLUSTER`,
`qsa_select.cu:795-799`; `STRATA_AM_CLUSTER`, `sampler.cu:188-192`) and are never launched there. On HIP the cluster
path is compiled out.

**Shared with RTX 30/40 (sm_80+), also used on RTX 50:** the TF32 block scorer in the prompt path; the `cp.async`
INT8 prompt attention; the fused INT8 prompt-expert kernels (3.8).

**What RTX 50 does not get here.**
- **FP4.** Blackwell has FP4 tensor cores. No FP4 expert kernel is in this repository: the experts run on the same
  integer dot products as on any other card. DETAILS.md cites "the NVFP4 fork" only as the source of one RTX 5090
  measurement (`--draft-vocab cyrillic`, 83 -> 109 tokens/s).
- **A larger PCIe share.** `pcie_frac_for_gbps` (`generate.cpp:1216-1224`) gives the full 0.55 from 20 GB/s up: "an
  x16 PCIe 4/5 link: ~26-28 GB/s" gets the same share. PCIe 5.0 shortens the time the share takes and speeds up the
  prompt path's ring; it does not raise the share.

**Build-time conditions.**
- **Toolkit.** `sm_120` needs CUDA 12.8 or newer. `CMakeLists.txt:166-170` warns for any `sm_120` build with a
  toolkit older than 13.0. `cuda_choice` (`setup.py:601-624`) adds the same warning when a user forces `--cuda 12`
  on an RTX 50 card, and docs/OLDER_GPUS.md says to keep an RTX 50 card on its own model when another card needs the
  CUDA 12 engine.
- **The ready-made engine.** A zip from the GitHub release (`PREBUILT_URL`, `setup.py:101-103`) with `BUILD.json`
  naming its architectures; setup prints them ("ready-made engine X for sm_..., CUDA ...", `get_prebuilt`,
  `setup.py:1946-2030`) and compiles locally instead when one of the chosen cards is not covered. README and
  INSTALL.md say the ready-made engine covers RTX 20/30/40/50; the exact arch list is not in the repository (it is
  in the release's `BUILD.json`).
- **Docker.** `Dockerfile:41` builds on `nvidia/cuda:13.0.0-devel-ubuntu24.04` with `CUDA_ARCHITECTURES` default
  `75;80;86;89;120` (`Dockerfile:58`).
- **Card check at start.** `src/core/device.cu:261-291` requires 7.5+ (6.0+ in the experimental build); 12.0 passes.
  `device_code_error()` (`device.hpp:56-58`) makes the engine stop with "this engine has no code for ... rebuild it
  for this card" (`generate.cpp:3111-3115`). A CUDA runtime older than the build's headers is a warning (#542,
  `generate.cpp:3098-3107`).

---

## 5. What the installer chooses on an RTX 50 card

`setup.py` measures the card with `nvidia-smi` (`gpus()`, `setup.py:529`), the RAM in GiB (`ram_gb()`,
`setup.py:315`) and the CPU, then recommends. With `--yes` it takes every recommendation.

**Model and size (by RAM, not by card).** The first question offers the original, Swift 1.5, the Coder or Unsloth;
the default is the original (`setup.py:3865`). The default size is IQ3_XXS when the RAM reads 60 GiB or more,
else the first in the list, Q2_0 for the original (`setup.py:3891`). README, MODELS.md and AI_SETUP.md recommend
differently by RAM (32 GB: Coder; 48 GB: IQ2_XS or Q2_0; 64 GB: IQ2_XS; 96 GB+: IQ3_S or UD-IQ4_XS); that table is
advice to the user, not the code's default. When a variant's experts plus ~10 GB do not fit the RAM, setup picks a
low-RAM mode (`low_ram_needed`, `setup.py:2347-2350`).

**Per card (by VRAM).** The rules are in `setup.py:3937-3990` and `:4224-4320`:

| Card | VRAM | Recommended context | KV | KV streaming | Notes setup prints |
| --- | ---: | ---: | --- | --- | --- |
| RTX 5090 | 32 GB | 131,072 | INT8 | `--kv-resident 32768` if RAM >= the variant's `ram_gb` + 1.8 + 1 GiB | - |
| RTX 5090 Laptop | 24 GB | 131,072 | INT8 | as above | - |
| RTX 5080, 5070 Ti, 5060 Ti 16 GB | 16 GB | 65,536 | INT8 | `--kv-resident 32768` if RAM >= `ram_gb` + 0.9 + 1 GiB | - |
| RTX 5070 | 12 GB | 32,768 | INT8 | none (under 64K) | the `--draft-vocab en` tip (under 14 GB); with images, a `--vram-reserve-mib 1000` tip (up to 12.5 GB) |
| RTX 5060 Ti 8 GB, 5060, 5050 | 8 GB | 32,768 | INT8 | none | "less than 12 GB of VRAM: Strata will run, but most experts stay on the CPU and it will be slow" (`setup.py:3797-3798`); the draft-vocab tip |

The context rule uses the smallest chosen card and is then capped by a RAM rule for IQ3_XXS and IQ3_S
(`ram_ctx`, `setup.py:2514`); a longer context the user picks is kept with a warning (#406). The KV choice is FP16 up
to 8K and INT8 above, unless `--kv q4_0|k8v4` is given (`setup.py:3986-3995`). The PLE table stays on the SSD
(`direct`) unless the model is on a rotational disk on Linux (3.3). A hybrid CPU (more efficiency than performance
cores) gets `--pool-workers` (`hybrid_pool_workers`, `setup.py:375-387`).

**The engine command line.** `setup.py:4224-4227` starts every config with:

```
--pack <data>/packs/<tag> --native <shard 1> --ple-gguf <the shard with per_layer_token_embd>
--expert-profile <Strata>/data/expert-profile.bin --expert-cache auto
--prefill auto --spec 4 --spec-min-p 0.5 --mtp <data>/mtp/rt --max-context <ctx>
```

and adds `--kv int8` above 8K, `--kv-resident 32768` from 64K when RAM allows, `--mmap-experts` or
`--resident-experts` in a low-RAM mode, `--resident-budget-gib N` for Unsloth, `--vision --vram-reserve-mib 700`
with images, `--ple-io ram` in the rotational-disk case. Setup then writes `strata-<tag>.json` with `"exe"` (the
engine in `engine/`), `"args"`, `"lib_dirs"` (the pip CUDA 13 libraries) and, on a PC with several cards, `"gpu"`.

**How the server starts it.** `setup.py:3093` runs `serve/server.py --engine strata --config strata-<tag>.json
--port 8080`. The server builds the final arguments with `engine_args` (`serve/server.py:1461-1482`) and starts
`<exe> --serve <args>` (`serve/server.py:454`, `:4001`). `child_env` (`serve/server.py:1560-1576`) sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` from `"gpu"` and puts `"lib_dirs"` first on `PATH` or
`LD_LIBRARY_PATH`.

Example: an RTX 5080 (16 GB) with 64 GB of RAM and `--yes` gets IQ3_XXS (RAM 60 GiB or more), 65,536 context, INT8
KV and, when the RAM reads at least 61.9 GiB, `--kv-resident 32768`. This is the code's arithmetic, not a measured
install.

See [install/context-kv-and-options.md](../install/context-kv-and-options.md) and
[install/model-choice.md](../install/model-choice.md) for the installer's logic in full.

---

## 6. Sizing for each RTX 50 card and model variant

### What was measured on RTX 50 cards

| Card and machine | Variant, setting | Result | Source |
| --- | --- | --- | --- |
| RTX 5070 12 GB, Ryzen 5 7600, 64 GB DDR5-5200, Windows, engine 0.1.26, setup's settings | Q2_0 / IQ2_XS / IQ3_XXS / IQ3_S / Coder, output at 4K | 93.0 / 78.6 / 61.6 / 53.3 / 55.1 tokens/s | DETAILS.md, "Output (tokens/s)" |
| the same | the same, output at 128K | 73.7 / 62.7 / 49.0 / 45.5 / 43.0 tokens/s | the same |
| the same | the same, prompt at 32K | 2,171 / 2,092 / 1,745 / 1,624 / 2,177 tokens/s | DETAILS.md, "Prompt processing" |
| the same, engine 0.1.36 | Q2_0 decode with the cluster kernels | 4K 89 -> 93.5, 128K 64.5 -> 76.4 tokens/s | DETAILS.md |
| the same | UD-Q4_K_XL, `--resident-budget-gib 40` | 7-8.5 tokens/s; the cache held 1,280 experts (3.7 GiB) at 4K | DETAILS.md; UNSLOTH_Q4.md |
| RTX 5090 32 GB, Core Ultra 9 285K, 64 GB, Linux, engine 0.1.29 compiled for `sm_120` with CUDA 13.0.2 | IQ2_XS, 131,072 context, INT8 KV, 32,768-cell KV window, images on (GPU) | decode 179.4 / 175.7 / 165.0 tokens/s at 4,096 / 32,768 / 128,000 prompt tokens; expert cache 17,463 slots, 23.44 GiB, "no eviction"; PCIe probe 50.0 GB/s; 23 pool workers | `bench/results/2026-09-30-community-rtx-5090/README.md` |
| RTX 5080 16 GB (x16), Ryzen 9 9950X3D, 62 GB, Windows 11 | Coder, 32K, alone | decode 83-87 (story) / 88-105 (code) tokens/s; prompt 1,726-2,017 at 16K | MULTI_GPU.md, "Measured" |
| RTX 5070 Ti, Ryzen 7 9800X3D, NVMe, Windows | expert load at start | 1.42-1.52 GiB/s (~35 s) from a terminal; 0.05 GiB/s from Task Scheduler with its defaults | DETAILS.md, "Running it at startup" |

README.md also says "an RTX 3090 (24 GB) should write about 100-140 tokens per second"; that is an estimate.
DETAILS.md's "Other GPUs (estimated)" gives the RTX 5060 Ti 16 GB (±20%, and its prompt column predates engine
0.1.13): output ~87 / ~77 / ~65 tokens/s at 4K for Q2_0 / IQ2_XS / IQ3_XXS.

### Expert slots per card (estimates)

The counts below are **estimates**, not measurements. They take the expert cache as the card's VRAM less about
6.8 GiB for everything else (the dense weights, the head and drafter, a 32K-cell KV window, the CUDA context, the
reserve and the prompt path's loan). That figure comes from two measurements: one RTX 3090 held a 17,583 MiB cache
with 389 MiB free at 128K and INT8 KV (24,576 MiB card); the RTX 5070's Q2_0 cache at 262K held 3,872 slots, about
5.0 GiB. Slots are the cache divided by the variant's average expert (section 2); the profile fills hottest first,
so the real count moves with the blob sizes it meets.

| Card | VRAM | Cache (estimate) | Q2_0 (of 24,576) | IQ2_XS | IQ3_XXS | IQ3_S | Coder (of 12,288) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5090 | 32 GB | ~23.4 GiB (the 5090 run, images on) | ~18,000 (74%) | 17,463 measured (71%) | ~14,000 (59%) | ~12,000 (50%) | all, "an RTX 5090 holds all of the Coder's experts" (DETAILS.md) |
| RTX 5090 Laptop | 24 GB | ~17 GiB | ~13,000 (54%) | ~12,500 (52%) | ~10,500 (43%; the 3090 measured 10,584) | ~9,000 (37%) | ~9,000 (73%; the 3090 measured 8,973 at 32K) |
| RTX 5080, 5070 Ti, 5060 Ti 16 GB | 16 GB | ~9 GiB | ~7,000 (29%) | ~6,800 (28%) | ~5,500 (23%) | ~4,700 (19%) | ~4,700 (39%) |
| RTX 5070 | 12 GB | ~5 GiB | ~3,900 (16%) | ~3,700 (15%) | ~3,000 (13%) | ~2,600 (11%) | ~2,600 (21%) |
| 8 GB cards | 8 GB | ~1 GiB or less | under 1,000 | | | | |

The hit rate is much higher than the share held: the Coder's profile gives "72% of the expert reads hit the GPU on a
12 GB card" (DETAILS.md, "Or: the Coder").

**Levers that free VRAM for experts**, in rough order of effect: a shorter `--context`; `--kv-resident 32768` at 64K
and up (setup's default when RAM allows); `--kv k8v4` or `q4_0` (measured on an RTX 3090 with the Coder at 198K,
`k8v4` gave "99 instead of 85 tokens/s output", DETAILS.md); `--draft-vocab en` (about 110-215 MiB, see 3.1); images
off or on the CPU (the GPU encoder keeps ~1.4 GB free). `--vram-reserve-mib` does the opposite: it keeps VRAM free
for other programs (default 700).

### Host RAM

| Item | Size | When |
| --- | ---: | --- |
| The expert arena, pinned | the variant's experts: 34-50 GB (23 for the Coder) | default |
| PLE table | ~0.1 GB row cache; 28.8 GB with `--ple-io ram` | `direct` is the default |
| KV host copy | context x 13,728 B (1.8 GB at 128K) | with `--kv-resident` |
| Conversation checkpoints | up to 6 x ~118 MB | serve mode |
| Engine, server, OS | "~10 GB" beside the experts in setup's rule (`LOW_RAM_HEADROOM_GB`, `setup.py:2343`) | always |

That is why 64 GB runs every size and why the RTX 5090 run had 62.19 GiB usable for IQ2_XS at 128K. A 32 GB PC
with a 12-16 GB RTX 50 card runs the Coder in the low-RAM resident mode (MODELS.md, "A big graphics card and little
RAM").

---

## 7. Building, running and checking

**Install and run (the supported path).**

```
START-HERE.bat                      Windows: installs, then starts; later runs only start
./setup.sh                          Linux: the same
./setup.sh --setup --family qwen --model IQ2_XS --context 65536 --yes     no questions
./setup.sh --setup --build          compile the engine here instead of the ready-made one
./setup.sh --calibrate              measure --pcie-frac, --spec-min-p, --pool-workers on this PC (NVIDIA only)
```

A local compile (`build_engine`, `setup.py:2274`) runs CMake with `-DCMAKE_CUDA_ARCHITECTURES=<the chosen cards>`
(`120` for an RTX 50 card) and the `nvcc` it finds; on Linux setup looks on `PATH`, in `/usr/local/cuda*` and
`/opt/cuda*` (DETAILS.md, "Linux"). Use CUDA 13.0 or newer for `sm_120` (section 4).

**By hand.**

```
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --target strata -j
```

**What to check on a first start** (the engine log is `strata-<model>.log`):

1. **The card.** `strata generate: GPU 0: <name>, compute capability 12.0` (`generate.cpp:3095`). A "this engine has
   no code for" line means the build lacks `sm_120`.
2. **The cluster kernels.** `decode_cluster_parity` is built in every CUDA build with the GPU targets
   (`CMakeLists.txt:705-711`, inside `if(STRATA_ENABLE_CUDA)`):

   ```
   cmake --build build --target decode_cluster_parity
   ./build/decode_cluster_parity --selftest      # bitwise vs the one-CTA kernels, also under graph capture
   ./build/decode_cluster_parity --bench         # the per-call times quoted in section 4
   ```

   It skips below sm_90 (`CMakeLists.txt:705-706`).
3. **The plan.** The log shows the PCIe probe and the share it chose (`PCIe probe: ... GB/s ... -> pcie_frac`), the
   expert-cache size, the PLE mode, and `... MiB of VRAM free with everything loaded`, which names a
   `--vram-reserve-mib` to add when it is low (`generate.cpp:5670-5674`). The `INFO` line (`generate.cpp:5696`)
   carries `expert_slots`, `expert_cache_mib`, `kv_resident`, `vram_free_mib`, `pool_workers`, `pcie_frac`.
4. **Long prompts.** The crash in #220 / #224 appeared on long prompts with CUDA 12.8 builds; a 32K prompt is the
   check for a local build.
5. **Speed and tiers.** `GET /metrics` lists per request the hit rate, `pcie_share`, `ram_blobs`, `file_blobs`
   and the drafts offered and accepted (DETAILS.md, "How much came from where"). `STRATA_DECODE_TIMING=1` prints
   where a decode window's time goes. For an A/B of the RTX 50 paths:
   `STRATA_QSA_CLUSTER=0 STRATA_ARGMAX_MULTI=0` in the config's `"env"`.

---

## 8. Tests

| Test | What it checks | Run |
| --- | --- | --- |
| `decode_cluster_parity` (`src/kernels/decode_cluster_parity.cpp`) | the cluster top-k and argmax against the one-CTA kernels, bitwise, captured too; skips below sm_90 | `ctest -R decode_cluster_parity`, or the binary with `--selftest` / `--bench` |
| `sampler_parity` | the one-block and older sampler paths | `ctest -R sampler_parity` |
| `tools/test_setup_risk.py` | the context and RAM rules, `KvStreaming`, `SmallCard`, `SplitShortCard` | `python -m unittest tools.test_setup_risk` |
| `tools/test_setup_draft_vocab.py`, `tools/test_setup_older_gpus.py` | the draft-vocab note; the CUDA 12 / 13 choice | `python -m unittest tools.test_setup_<name>` |
| `serve/test_server.py` | `engine_args`, `layer_split_value`, `CUDA_VISIBLE_DEVICES` from `"gpu"` | `python -m unittest serve.test_server` |

Setup's tests need no GPU and no download (AGENTS.md).

## 9. Gotchas

- **CUDA 12.8 builds** for RTX 50 crashed on long prompts (#220, #224). The ready-made engine and Docker use CUDA 13.0.
- **Driver 580 or newer** is required for the CUDA 13 engine (`MIN_DRIVER`, `setup.py:106`).
- **Windows** cannot use `--ple-io ram`; the engine refuses it (`generate.cpp:1745-1750`). Starting from Task
  Scheduler with its defaults made the expert load 24x slower (DETAILS.md).
- **Windows page file.** A page file under 4 GB can make `cudaMalloc` fail although VRAM is free (#60); setup warns.
- **The PCIe share is capped at 0.55** whatever the link (section 4).
- **Hit rate is not the PCIe share.** Experts the GPU reads over PCIe are not counted as hits (DETAILS.md).
- **Images on a 12 GB card** can leave ~200 MiB free; setup prints a `--vram-reserve-mib 1000` tip.

---

## 10. Compared with UnbiasedStrata

The fork's page is [../../codebase/guides/rtx50-flash-next.md](../../codebase/guides/rtx50-flash-next.md). The RTX 50
kernels are the same: `qsa_select.cu`, `sampler.cu`, `moe_fused.cu` and `decode_cluster_parity.cpp` are unchanged in
the fork. What differs:

| | Upstream (this page) | UnbiasedStrata |
| --- | --- | --- |
| How it is installed and started | `setup.py` recommends by card and RAM and writes `strata-<model>.json`; `serve/server.py` starts the engine | `make build`, then one C++ binary (`unbiased-strata`) whose front end starts the engine with fixed flags; no installer, no Python |
| Engine build | ready-made engine with CUDA 13.0, or a local compile for the chosen cards | `CUDA_ARCHS` `61;70;75;86;89;120` filtered by the installed `nvcc`; built and tested with CUDA 12.4 and 12.8; README says "12.8+ for RTX 50" |
| Model variants | Q2_0 (canonical or native), IQ2_XS, IQ3_XXS, IQ3_S, Coder, Swift 1.5, Unsloth UD-IQ4_XS and UD-Q4_K_XL | the IQ3_XXS GGUF for Flash-Next (plus Qwen3.6-35B-A3B, which upstream does not run) |
| Context and KV | by VRAM: 32K / 64K / 128K; INT8 above 8K; KV streaming from 64K when RAM allows | `--ctx 32768` default; `--kv int8` above 8,192; streaming only when passed after `--` |
| n-gram table | `--ple-io direct` (SSD) by default; `ram` mmaps and `mlock`s, Linux only | `--ple-io ram` by default, read into anonymous memory by 16 readers, no dependence on `mlock` (commit `9a53e42`) |
| Slow stores | plain reads | `robust_read.hpp` retries refused reads (commit `9a53e42`) |
| CPU workers | engine default (physical cores but the host's) or setup's hybrid rule / `--calibrate` | the front end passes `--pool-workers` from the container's CPU quota |
| Tested on RTX 50 | RTX 5070 (upstream's own measurements), RTX 5090 (community), RTX 5080 and 5070 Ti runs in upstream docs | not run on an RTX 50 card (fork README) |

## See also

- [rtx3090-flash-next.md](rtx3090-flash-next.md): the same model on one and on two RTX 3090s.
- [install/README.md](../install/README.md), [install/hardware-detection.md](../install/hardware-detection.md):
  the installer's card and RAM detection.
- [build/gpu-backends.md](../build/gpu-backends.md), [build/cmake-options.md](../build/cmake-options.md): CUDA
  architectures, CUDA 12 vs 13, HIP.
- [program/cli-flags.md](../program/cli-flags.md): every engine flag named here.
- [serve/engine-process.md](../serve/engine-process.md): how the server starts and talks to the engine.
- [prefill/README.md](../prefill/README.md): the prompt path.
- [DETAILS.md](../../upstream/DETAILS.md): upstream's measurements and "How it works".
