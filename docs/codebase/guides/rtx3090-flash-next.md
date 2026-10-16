[Codebase docs](../README.md) › [Guides](README.md) › Qwen3.8-Flash-Next on RTX 3090 cards

# Qwen3.8-Flash-Next on RTX 3090 cards: one card, and two

This page follows Qwen3.8-Flash-Next (`qwen4exp`) on the NVIDIA RTX 3090: 24 GB, Ampere, compute capability 8.6,
`sm_86`. It has two parts:

- **[Part A: one RTX 3090](#part-a-one-rtx-3090).** For each part of the model's architecture it covers where that
  part lives on a single 24 GB card, what it costs, and the code that runs it. It also covers what was measured on
  this card and the code changes that came from that measurement.
- **[Part B: two RTX 3090s](#part-b-two-rtx-3090s).** It covers how the same architecture can be spread over two
  cards, the three multi-GPU modes the engine has for doing that, what each card holds in each mode, and what this
  fork's front end allows.

A companion page, [rtx50-flash-next.md](rtx50-flash-next.md), describes each architectural part in more depth. This
page repeats only what a 3090 owner needs.

## Read this first

1. **A single RTX 3090 is where this fork was first made to work.** It ran in a RunPod container: RTX 3090, 32 vCPUs
   of an EPYC 7763, 125 GB RAM, CUDA 12.8 ([README.md](../../../README.md), "Status").
   - The fork's engine changes for Flash-Next came out of that machine: retrying reads, the n-gram table in its own
     memory, and the worker count from the container's CPU quota.
   - Measured there: 107 tokens/s generation, 2,483 tokens/s prompt, a 19 s start from local disk and 12 minutes from
     a network volume.
2. **Two RTX 3090s are not supported or tested by this fork.**
   - The README's "Not supported" row says "more than one GPU". The front end refuses most multi-GPU engine flags
     (`frontend/main.cpp:632-634`).
   - The engine source still carries upstream Strata's three multi-GPU modes. One of them, the helper expert cache
     (`--expert-cache-device1`), is not in the front end's refusal list and passes through. The other two need a
     one-line change to the front end ([B.6](#b6-what-it-would-take-in-code)).
   - Nothing in Part B has run on two cards in this fork. Its capacity figures are derived from the measurements in
     Part A, and each one says so.
3. **The RTX 3090 takes the engine's sm_80 tensor-core paths, not its sm_90 ones.** Upstream added two
   thread-block-cluster kernels for RTX 50 (the sparse-attention top-k and the greedy argmax). On a 3090 the engine
   uses the one-CTA kernels they replace ([A.5](#a5-what-sm_86-gets-and-what-it-does-not)). The tokens are the same,
   but they take longer at long context.

---

# Part A: one RTX 3090

## A.1 The card as the engine sees it

| Property | RTX 3090 | What reads it |
| --- | --- | --- |
| Compute capability | 8.6 (`sm_86`) | in the default `CUDA_ARCHS` (`Makefile:9`); `src/core/device.cu:286` requires 7.5+ |
| VRAM | 24 GB GDDR6X | `--expert-cache auto` takes what is free after the fixed set |
| Tensor cores | Ampere: TF32 and INT8 `mma.sync`, `cp.async` | the sm_80+ prompt kernels (A.5) |
| Thread-block clusters | no (sm_90+) | `qsa_block_topk_cluster` and `sample_greedy_cluster` return false; the one-CTA kernels run |
| Host link | PCIe 4.0 x16, about 26 GB/s measured by such links | `probe_pcie_h2d_gbps` (`src/program/generate.cpp:1212`); at 20 GB/s and above the GPU takes the full 0.55 share of missed experts (`pcie_frac_for_gbps`, `:1275`) |
| CUDA toolkit | any 12.x; CUDA 13 also keeps `sm_86` | the fork was built with 12.4 and 12.8 ([README.md](../../../README.md), "Requirements") |

`make build CUDA_ARCHS=86` builds for the 3090 alone and is the shortest build. The default list (`61;70;75;86;89;120`)
also works on it.

## A.2 What was measured, and the code it led to

This is the fork's measured configuration. The front end ran with its defaults:
- `--ctx 32768`, so the front end adds `--kv int8`;
- `--ple-io ram`;
- `--expert-cache auto` from the pack's profile;
- `--spec 4`;
- the CPU worker count from the container's quota.

The README's comparison is one run each, greedy, reasoning off. Generation is the mean of 8 prompts of 400 tokens.
Prompt speed is the mean of 2 prompts of about 20,000 tokens.

| | Generation, tokens/s | Prompt, tokens/s |
| --- | --- | --- |
| This fork | 107 (87 to 116) | 2,483 |
| Upstream, `--pool-workers 15 --ple-io ram` | 78 (61 to 93) | 2,439 |
| Upstream, arguments as its installer writes them | 20 (16 to 21) | 2,368 |

Upstream's installer started 127 CPU workers on the container's 32 vCPUs and read the n-gram table from disk per
token. The README says the remaining gap between the first two rows is not explained.

Two problems seen on that container became the fork's engine changes (commit `9a53e42`; details in
[rtx50-flash-next.md, section 5](rtx50-flash-next.md#5-every-fork-change-that-touches-flash-next)):

| Problem on the 3090 pod | Change | Code |
| --- | --- | --- |
| The S3-backed network volume refused reads now and then (ENOMEM, EIO), and the expert load failed at start | retrying positional reads; the number of parallel readers halves while the store refuses; errors name the file and offset | `include/strata/platform/robust_read.hpp` (new); `src/core/expert_source.cpp:2676` |
| `--ple-io ram` locked the file mapping. That failed under the container's `ulimit -l`, and touching the pages instead faulted 28.8 GB back from the volume | the table is read into anonymous memory by 16 readers | `src/kernels/ngram.cpp:265-287` |
| The engine saw 256 host CPUs under a 32-CPU quota and started a worker per core it saw | the front end passes `--pool-workers` from the cgroup quota, halved with SMT | `frontend/main.cpp:586` |

With these changes, the model loaded from the network volume in 12 minutes, through two refused reads. From local
disk, with the files in the OS cache, it loaded in 19 s.

## A.3 The model, part by part, on one 3090

Each row is one architectural part: what it does, where it lives on a single 3090, and the code and doc that cover
it. The geometry is `struct ModelGeometry` (`include/strata/core/layout.hpp:38`). The model is 48 layers, width 2,560
and vocabulary 248,320.

| Part | What it does | Where it lives on one 3090 | Size | Code and doc |
| --- | --- | --- | --- | --- |
| Token embedding | one 2,560-float row per token id | mapped pinned host RAM; one row crosses PCIe per token | 0 VRAM | `NativeEmbed` (`include/strata/core/native_head.hpp:40-43`) |
| Hyper-connection residual | a stack of 4 streams x 2,560; each half-layer reads the stack into one vector (`gr_read`) and writes back per stream (`gr_write`) | VRAM | BF16 weights, 1.374 GiB of upstream's 5.305 GiB dense file (`include/strata/kernels/gr.hpp:10-14`) | `src/kernels/cuda/gr.cu:344`, `:413`; [core/layer-forward.md](../core/layer-forward.md#residual-handling) |
| PLE n-gram embedding (layer 1) | the token and its two predecessors hash to 16 rows of a 320M-row table; a gated, convolved update to the residual | host RAM (the front end's `--ple-io ram`); the 16 rows are gathered on the host before the token's GPU graph | 28.8 GB RAM | `ngram_rows` (`src/kernels/ngram.cpp:68`), `ple_block` (`src/kernels/cuda/ple.cu:228`); [kernels/cpu/ngram.md](../kernels/cpu/ngram.md) |
| Gated DeltaNet (36 layers) | a linear-attention recurrence: a 128 x 128 state per value head (48 heads), decayed and updated by the delta rule each token; sigmoid output gate | VRAM | ~118 MB of state at any context length | `src/core/layer.cpp:224`, `fused_gdn.cu`; [kernels/cuda/gdn.md](../kernels/cuda/gdn.md) |
| Sparse attention, QSA (12 layers) | an indexer scores 4-cell blocks and keeps at most 2,051 cells per token; 24 query heads over 2 KV heads of 256 | VRAM | KV 12,672 B per token in INT8: 415 MB at the default 32,768 context | `qsa_select.cu`, `qsa_decode_attn.cu`; [kernels/cuda/attention-qsa.md](../kernels/cuda/attention-qsa.md) |
| Routed experts | 512 per layer, top 10 per token, each 2,560 -> 640 -> 2,560 | **all 24,576 in pinned RAM**; copies of the most-routed (~42%, below) in VRAM slots; misses computed by the CPU pool, or read by the GPU over PCIe | 42,912,972,800 B = 39.97 GiB; 1.31 to 2.33 MB each by layer, 1.67 MiB average (`tests/data/native_experts/iq3_xxs.txt`) | [core/experts/README.md](../core/experts/README.md), [program/expert-tiers.md](../program/expert-tiers.md) |
| Shared expert, routers | one shared expert per layer, run for every token; the BF16 router | VRAM | part of the fixed set | `moe_shared`, `moe_route` (`src/core/layer.cpp`) |
| Output head | 2,560 x 248,320 | VRAM | part of the fixed set | [core/dense-and-head.md](../core/dense-and-head.md) |
| MTP draft layer | drafts up to 3 tokens; the 48 layers check them in one window | VRAM; its draft head is limited to `draft_vocab.bin` (106,299 ids) | 0.8 GB on disk | [core/mtp-draft-head.md](../core/mtp-draft-head.md) |

Four mechanisms make the 40 GiB of experts work with a 24 GB card. They are upstream's and run unchanged on a 3090:

1. **A profile-filled VRAM cache.** `--expert-cache auto` takes the free VRAM after a reserve and fills it in the
   rank order of `expert-profile.bin`. Each slot is cut to its expert's own size ("sized slots", about 30% more
   experts than uniform slots).
2. **CPU and GPU at the same time.** The router's choices reach the host through a mapped "doorbell"
   (`struct Doorbell`, `include/strata/core/layer.hpp:409`). The CPU pool computes missed experts in place in pinned
   RAM, while the GPU computes the cached ones and the shared expert. Up to 55% of a layer's distinct misses are read
   by the GPU over PCIe instead ([core/experts/dispatch.md](../core/experts/dispatch.md)).
3. **The verify window.** With `--spec 4`, the last token and up to 3 drafts pass through each layer together. The
   experts they share are fetched and computed once. Upstream reports 2.4 to 3.2 tokens per pass.
4. **Adaptive swaps.** Every 4 verify rounds, up to 96 of the most-routed non-resident experts replace the
   least-routed resident ones of the same layer
   ([core/experts/resident-ram-and-adaptive.md](../core/experts/resident-ram-and-adaptive.md)).

For prompts, the prompt path borrows the cache's last slots for its buffers. It streams the experts the card lacks
from pinned RAM over PCIe in a ring ([prefill/expert-streaming.md](../prefill/expert-streaming.md)).

## A.4 Memory budget on one 3090

**VRAM.** The 3090's slot count was not recorded in this repository. The nearest measurement is the RTX 4090, which
also has 24 GB. With the service's command line (`--ctx 196608`, INT8 KV, `--kv-resident 32768`), it held **10,368 of
the 24,576 experts with 430 MiB free**
([docs/qwen36-port/step5-serve.md](../../qwen36-port/step5-serve.md), "Flash-Next regression check").

The front end's default on a 3090 is `--ctx 32768` with INT8 KV and no streaming. That puts the same 32,768 cells of
KV in VRAM (415 MB) and a smaller RoPE table, so expect about the same: **roughly 10,300 slots, about 42% of the
experts.** This is derived, not measured.

The rest of the card, about 6.7 GiB, is the fixed set:
- the dense weights;
- the output head;
- the draft head;
- the KV pools;
- the GDN state;
- the CUDA context;
- the reserve.

**Host RAM.** The 3090 pod had 125 GB. What the engine holds:

| Item | Size | Note |
| --- | ---: | --- |
| Expert arena, pinned | 40 GiB | every expert, VRAM-held ones included. `ArenaExpertSource::release` (`src/core/expert_source.cpp:3005`) frees pages only for the file-backed arena of `STRATA_ARENA_MMAP=1` |
| PLE table | 28.8 GB | `--ple-io ram`; ~0.1 GB with `-- --ple-io direct` (unbuffered SSD reads); OS file cache with `mmap` |
| Prompt-cache checkpoints | up to 6 x ~118 MB | GDN state copies, in host RAM |
| KV host copy | context x 12,672 B | only with `--kv-resident` |
| Engine, front end, OS | a few GB | |

That is about 75 GB with `--ple-io ram`. The README asks for 128 GB so that the OS file cache can also hold the 76 GB
of model files, which is what made the 19 s warm start possible.

On a 64 GB host, use `-- --ple-io direct` with the model on an SSD, or `mmap`. The 4090 service runs on 62 GB with
`--ple-io mmap`, and its first passes after a start ran at about 60 tokens/s while the file cache filled.

## A.5 What sm_86 gets, and what it does not

| Code path | On a 3090 | Effect |
| --- | --- | --- |
| Sparse-attention top-k, decode (`qsa_block_topk`, `src/kernels/cuda/qsa_select.cu`) | one CTA per query: `block_topk_reg_kernel` (keys in registers, up to about 135K cells of context), then `block_topk_wide_kernel` | Upstream measured the RTX 5070's one-CTA kernel at 21.9 µs per call at 32K, 58 at 128K and 200 at 262K, against 15.6, 18 and 22 for the RTX 50 cluster kernel (`qsa_select.cu:765-791`). There is one call per attention layer per window, 12 per window. At 262K that is on the order of 2 ms per window that an RTX 50 card does not spend. These figures are the 5070's; the 3090 was not timed |
| Greedy argmax (`sample_tokens`, `src/kernels/cuda/sampler.cu`) | the one-block kernel over the 248,320 logits | 40 µs per call on the 5070, against 6 µs for the cluster kernel; one call per window and per draft step |
| Block scoring in the prompt path (`qsa_block_scores_tc`) | yes: TF32 `mma.sync`, sm_80+ | |
| Prompt attention (`qsa_prompt_attn_batch`) | yes: INT8 KV through `launch_i8` with `cp.async`, sm_80+ | |
| Fused INT8 expert prompt kernels (`STRATA_PF_FUSED`, `src/prefill/moe_fused.cu:399-420`) | available on sm_80+, but on by default only for Q2_0 packs; opt-in (`STRATA_PF_FUSED=1`) for this fork's IQ3_XXS pack | upstream measured the IQ3 packs "about even" |

At the default 32,768 context the missing cluster kernels cost little. They matter at the long contexts the 4090
service runs (196,608). On a 3090 at long context, a smaller KV frees VRAM for experts. Upstream measured
`--kv k8v4` on an RTX 3090 with the Coder model at 198K context at 99 tokens/s instead of 85, because more experts fit
([DETAILS.md](../../upstream/DETAILS.md), "Hybrid K8V4 KV cache"). k8v4 does not stream its KV.

## A.6 Running on one 3090

```
make build CUDA_ARCHS=86
./build/unbiased-strata \
  --model /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --pack  /models/pack-iq3-xxs  --mtp /models/mtp
# long context, as the 4090 service runs it:
#   --ctx 196608 -- --kv-resident 32768
# under ~96 GB of RAM:
#   -- --ple-io mmap            (or direct, with the model on an SSD)
```

On a first start, check:
- the start-up line `GPU 0: ... compute capability 8.6` (`generate.cpp:3438`);
- the PCIe probe and the share it chose;
- the expert slot count and the free VRAM (the `INFO` line's `expert_slots`);
- `PLE table ... (--ple-io ram) in N s`.

`GET /health` reports VRAM hits, RAM reads and file reads per request ([frontend/http-api.md](../frontend/http-api.md)).

---

# Part B: two RTX 3090s

## B.1 What this fork allows

The front end refuses any engine argument after `--` that **starts with** one of `--gpus`, `--layer-split`,
`--split-device`, `--peer-device`, `--peer-slots`, `--remote-expert`, `--vision` or `--batch`, and exits with code 2
(`frontend/main.cpp:629-640`). Its comment reads "one GPU only: the engine's multi-GPU and remote-expert modes are
not part of this fork".

| Engine mode | Flags | Through the front end |
| --- | --- | --- |
| Helper expert cache on the second card | `--expert-cache-device1 N\|auto`, `--expert-cache-remote-placement` | **passes**: no refused prefix matches. Untested |
| Helper cache, adaptive variant | `--remote-expert-opt` | refused (`--remote-expert` prefix) |
| Layer split | `--layer-split K\|auto`, `--split-device`, `--trim-stage-weights`, `--split-skip-if-fits` | `--layer-split` and `--split-device` refused |
| Peer expert tier | `--peer-device N`, `--peer-slots`, `--peer-reserve-mib`, ... | `--peer-device` and `--peer-slots` refused |

The engine can also be started directly (`unbiased-strata --engine --serve ...`), which bypasses the filter. That
gives only the engine's line protocol on stdin and stdout, which takes token ids. There is then no tokenizer, chat
template or HTTP API, because those live in the front end ([frontend/engine-process.md](../frontend/engine-process.md)).
For a usable server, the choice is between the helper cache, which needs no change, and a front-end change (B.6).

## B.2 How the architecture divides across two cards

Each part of Flash-Next is either **per layer**, **per expert**, or **once per model**. That decides which ways of
splitting are possible:

| Part | Granularity | Can go to the second card by |
| --- | --- | --- |
| Routed experts, 40 GiB | per (layer, expert): 24,576 independent blobs | any of the three modes, because an expert is computed wherever its bytes are |
| GDN state, ~118 MB | per GDN layer (36) | a layer split: each card keeps its own layers' rows (the "carve", `session_bytes(g, ctx, K, lb, le)`) |
| KV cache and indexer | per attention layer (12) | a layer split |
| Dense weights (attention, GDN projections, shared experts, routers, hyper-connection weights) | per layer | a layer split with `--trim-stage-weights` (without it, each card loads the whole pack's dense weights) |
| Output head and MTP draft layer | once | the last stage of a layer split |
| PLE (layer 1), token embedding | once, host side | stays on the host; layer 1 runs on the first card |

Two numbers make the modes comparable:
- **Experts are 40 GiB, and two 3090s have 48 GB.** With the fixed set on one or both cards, the experts left over
  roughly match the free VRAM. Either the helper mode or the layer split can hold **all or nearly all** 24,576
  experts in VRAM, against about 42% on one card. The estimates are in B.3.
- **What crosses between cards is small.** A layer split hands over `handoff_floats(g)` = `hc x n_embd + n_embd + hc`
  = 12,804 floats (51 KB) per token at each stage boundary (`include/strata/core/verify.hpp:133`), once per verify
  window. The hyper-connection stack makes this 4 times a plain model's residual, and it is still trivial over PCIe.
  The helper and peer modes instead move each layer's expert inputs and output rows: 2,560 floats per routed entry,
  48 times per window.

## B.3 The three modes

### Mode 1: helper expert cache, `--expert-cache-device1 auto` (passes the front end)

**How it works.** `RemoteExperts` (`include/strata/core/remote_experts.hpp`, `src/core/remote_experts.cpp`;
[core/experts/multi-gpu-tiers.md](../core/experts/multi-gpu-tiers.md#remoteexperts-static-helper-tiers-on-cuda13)):
- The second card gets an expert cache filled with the profile's next ranks after CUDA0's. With `auto` it fills until
  512 MiB is left free (`open`, `remote_experts.cpp:121-237`).
- In each verify window, for each layer, the dispatcher marks the entries CUDA0 does not hold. `begin` (`:239-326`)
  claims the ones the helper holds and launches `native_expert_grouped` on them. The helper reads its input from, and
  writes compact rows to, mapped host memory ("zero-copy", the default). No peer-to-peer access is used, so no NVLink
  is needed.
- The CPU computes whatever neither card holds. `finish` (`:328-339`) waits for the helper, then scatters its rows
  into the same output the CPU fills.
- CUDA0 runs every layer's dense compute, all 48 layers, plus the KV, the GDN state, the head and the drafter.

**What each card holds (estimate).**

| | CUDA0 | CUDA1 |
| --- | --- | --- |
| Fixed set | ~6.7 GiB, as on one card | CUDA context, staging, 512 MiB left free |
| Experts | ~10,300 slots, the hottest | the next ranks: ~23 GiB of room for the ~23.2 GiB of experts CUDA0 lacks |

Together that is all or nearly all of the 24,576. Any remainder is the profile's coldest pairs. A shorter `--ctx` or a
smaller KV format on CUDA0 leaves more slots.

**Limits.**
- Placement is static: it is the profile order, with no adaptive swaps on the helper. The adaptive variant,
  `--remote-expert-opt`, is refused by the front end. If everything fits, there is nothing to adapt.
- The helper is not used by the prompt path. `src/prefill/prefill.cpp` includes `remote_experts.hpp` but makes no
  use of it. Prompts stream CUDA0's missing experts from RAM over PCIe as on one card. **Expect prompt speed close to
  a single 3090's.**
- Each layer of each window pays a host round trip to the second card. `STRATA_REMOTE_SPIN` (on by default) makes
  its context spin rather than sleep while waiting.
- `RemoteExperts::begin` requires `k == 10` (Flash-Next's top-10).

**Run.**

```
./build/unbiased-strata --model ... --pack ... --mtp ... -- --expert-cache-device1 auto
```

The engine's `INFO` line reports `expert_slots` over every tier and `expert_slots_primary` for CUDA0 alone
(`generate.cpp:6031-6047`).

### Mode 2: layer split, `--layer-split auto` (needs the front-end change)

**How it works.** [program/layer-split.md](../program/layer-split.md):
- Consecutive layer ranges run on different cards. For two cards, `--layer-split 24` gives layers 0-23 to CUDA0 and
  24-47 to CUDA1. `auto` places the split with a cost model of one decode window (`generate.cpp:3031`, `:3079-3180`).
- Each card, a "stage", has its own:
  - dense weights (only its own layers' with `--trim-stage-weights`);
  - session state, carved to its layers' GDN rows and KV pools;
  - expert cache, filled with **its own layers'** profiled experts;
  - verifier and prompt path.
- The last stage also holds the output head and the MTP drafter. The auto search reserves 1,000 MiB for them.
- Each verify window runs stage 0's layers, hands over the residual (51 KB per token) through mapped pinned host
  buffers, then runs stage 1's layers.
- For prompts, each stage's prompt path gets the other card as a stream helper (`set_stage_helper`, through mapped
  host memory, not P2P).

**What each card holds (estimate).**

| | CUDA0, layers 0-23 | CUDA1, layers 24-47, head, drafter |
| --- | --- | --- |
| Fixed set | half the dense weights and session; context; reserve | the other half, plus ~1 GiB of head and drafter |
| Experts | its layers' 12,288 pairs, ~20 GiB | its layers' 12,288 pairs, ~20 GiB |

The fixed set split over two cards plus a second CUDA context comes to about 8 GiB. That leaves about 40 GiB of room
for 39.97 GiB of experts: all of them or very nearly. The split is balanced by layer, so a card that runs out keeps
CPU misses only in its own layers. **Without `--trim-stage-weights`, both cards load the whole pack's dense weights**,
and a few GiB of experts fall back to the CPU.

**What changes against one card.**
- Every expert can be on the card that runs its layer, so the CPU pool has little or nothing to do. Decode then
  depends on the two GPUs and the hand-off, not on the host CPU.
- The two stages run one after the other for one request. Pipelining needs batch groups (`--batch-groups`), and the
  front end refuses `--batch*`.
- The token graph is not captured under a split; decode always runs through verify windows.
- `--resident-cpu-experts` is not supported with a split. `--layer-split` requires `--serve` (the front end always
  passes it) and `--expert-profile` (the front end passes the pack's).
- Conversation checkpoints are saved per stage and assembled by the last stage
  ([program/prompt-cache.md](../program/prompt-cache.md)).

### Mode 3: peer expert tier, `--peer-device 1` (needs the front-end change)

**How it works.** `PeerExperts` (`include/strata/core/peer_experts.hpp`, `src/core/peer_experts.cpp`;
[core/experts/multi-gpu-tiers.md](../core/experts/multi-gpu-tiers.md#peerexperts-a-second-adaptive-tier)):
- Like the helper, the second card caches the profile's pairs that CUDA0 lacks, and CUDA0 runs all dense compute.
- Unlike the helper, it is **adaptive**. After CUDA0's swaps, it takes the next most-routed CPU misses
  (`--peer-adapt-swaps`).
- In decode it writes its rows straight into the host output rows ("direct mode", `scatter_rows_f32`), which needs no
  P2P.
- It supports native packs only, which this fork's IQ3_XXS pack is.

**P2P and NVLink.**
- `open` enables peer access when `cudaDeviceCanAccessPeer` allows it both ways (`peer_experts.cpp:97-110`).
- The **prompt path** uses the peer only with P2P: `Prefill::set_peer` refuses with "the two GPUs cannot access each
  other (no P2P)" (`src/prefill/prefill.cpp:1475`). The engine then prints "the prompt path stays on the primary GPU"
  and goes on (`generate.cpp:5339-5342`).
- GeForce cards normally do not offer P2P over plain PCIe. **Two 3090s with an NVLink bridge** are the case where
  the peer can also compute its experts' prompt rows.

**Limits.**
- It cannot be combined with `--layer-split` or `--expert-cache-device1..3` (`generate.cpp:2000-2007`).
- Capacity is the helper's, less a 600 MiB reserve (`--peer-reserve-mib`).

### Side by side

| | Helper cache | Layer split | Peer tier |
| --- | --- | --- | --- |
| Front end | passes | needs B.6 | needs B.6 |
| Experts in VRAM (estimate) | all or nearly all | all or nearly all, with `--trim-stage-weights` | slightly fewer than the helper (600 MiB reserve) |
| Expert placement | static (profile) | static per stage, adaptive within each stage's cache | adaptive across both cards |
| Dense compute | CUDA0, all layers | each card its layers | CUDA0, all layers |
| Per-window traffic between cards | expert inputs and rows, 48 layers, through host memory | one 51 KB-per-token hand-off | expert inputs and rows, 48 layers |
| Prompt path uses the second card | no | yes, as a stream helper | only with P2P (NVLink) |
| CPU expert work | the remainder, if any | the remainder, if any | the remainder, if any |
| Upstream's multi-GPU notes | "fill CUDA1 with N experts" | verified on a 4-GPU split (conversation parking); placement model tuned on an RTX 5080 | "a second GPU as an adaptive expert-cache tier" |

None of the three has been measured on two 3090s here. The one multi-GPU speed figure in this repository is
upstream's report that its setup option `--gpus 0,1`, which "shares the model across them", was measured 1.3 to 1.6
times faster than one card by two users ([DETAILS.md](../../upstream/DETAILS.md), "Several GPUs"). That figure was
on other cards and in upstream's low-RAM mapped mode.

## B.4 Host RAM with two cards

A second card does not shrink the default RAM use. The pinned arena still loads all 40 GiB, the experts the cards
hold included (A.4).

Upstream's option for "a PC whose GPUs hold most experts but whose RAM is small" is `STRATA_ARENA_MMAP=1`
([DETAILS.md](../../upstream/DETAILS.md), "A mapped arena for small RAM"). It maps the pack's `experts.bin`
read-only and hands back the pages of the experts a GPU holds:
- The first start writes `experts.bin`, about 43 GB, into the pack directory, if the drive has room.
- Run it with `--pcie-frac 0`: the GPUs get no mapped alias of a file-backed arena.
- The hand-back follows the residency table `host_res` (`generate.cpp:4615-4650`). That table holds CUDA0's cache
  and the layer-split stages' caches, **but not the helper or peer tiers**. With a layer split the RAM drops to about
  the experts no card holds. With the helper cache, the helper's experts stay mapped and are even prefetched.

Together with `--ple-io direct`, which keeps the 28.8 GB table on the SSD, a layer split over two 3090s is the
configuration that could run Flash-Next on a host with far less than 128 GB. This is a reading of the code, not a
tested configuration.

## B.5 The PCIe and the board

- **Lane split.** Many desktop boards give two cards x8 / x8, about 13 GB/s each on PCIe 4.0. The engine probes
  CUDA0's link at start and, below 20 GB/s, scales the GPU's share of missed experts down in proportion (to about 0.36
  at 13 GB/s). A layer split probes each stage's link separately (`generate.cpp:2966-2975`). The prompt path's expert
  ring is PCIe-bound and slows with the link.
- **NVLink.** This fork uses no NVLink-specific path. A bridge only enables P2P, which only the peer tier's prompt
  help uses (Mode 3).
- **Which card is CUDA0.** CUDA0 holds the fixed set in every mode. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
  `CUDA_VISIBLE_DEVICES` so CUDA0 is the card on the faster link.

## B.6 What it would take in code

To serve Modes 2 and 3 through the HTTP API, remove `--layer-split`, `--split-device`, `--peer-device` and
`--peer-slots` (and `--remote-expert`, for the adaptive helper) from the refusal list at `frontend/main.cpp:633-634`.
The front end does not otherwise depend on the engine using one GPU:
- it starts the engine with `--serve`;
- it reads `READY <context>`;
- it speaks the same line protocol whatever the engine does inside.

Leave `--batch` refused: the front end sends one request at a time. This change has not been made or tested in this
fork. The README's "one GPU" statement and the refusal list would then need updating together.

## B.7 What to check on a first two-card start

1. **Tiers.** The start-up log shows one of these:
   - helper: `CUDA1: N additional experts, X GiB; results return through pinned host rows` (`generate.cpp:3983`);
   - split: `layer split: CUDA0 runs layers 0-N` and each stage's slot count;
   - peer: `peer GPU 1: N experts, X GiB ...; with the primary's M that is T of 24576 on the GPUs`.

   The `INFO` line's `expert_slots` should be close to 24,576 when everything fits.
2. **CPU work.** `GET /health` and the engine's per-request `expert tiers` line show how many experts were computed on
   the CPU. With all experts on the cards this should be near zero.
3. **Timing.** `STRATA_SPLIT_TIMING=1` (split) prints per-stage wait, pool, host and commit time per window. For the
   helper, the per-request log reports the helper's computed rows, launches and wait time (`generate.cpp:7890-7895`).
4. **One card as the baseline.** Compare against the same start without the multi-GPU flag, the configuration of
   Part A.

---

## See also

- [rtx50-flash-next.md](rtx50-flash-next.md): the same model on RTX 50, with the full architecture walkthrough and
  every fork change.
- [core/experts/multi-gpu-tiers.md](../core/experts/multi-gpu-tiers.md), [program/layer-split.md](../program/layer-split.md):
  the multi-GPU code in full.
- [program/expert-tiers.md](../program/expert-tiers.md), [core/experts/README.md](../core/experts/README.md): the
  single-card expert tiers.
- [frontend/engine-process.md](../frontend/engine-process.md): the engine command line and the refused flags.
- [README.md](../../../README.md), "Status" and "Speed": the 3090 measurements.
