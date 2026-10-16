[Upstream docs](../README.md) › [Hardware](README.md) › Qwen3.8-Flash-Next on RTX 3090 cards

# Upstream Strata on RTX 3090 cards: one card, and two

This page covers upstream Strata at commit 6f32ec0 on the NVIDIA RTX 3090: 24 GB, Ampere, compute capability 8.6.
- **Part A** is one card.
- **Part B** is two cards. Upstream supports this officially with its layer split, unlike the fork.

The model's architecture, part by part, is on the [RTX 50 page](rtx50-flash-next.md); the parts are the same on
any card. This page is written from upstream's own guides, mainly
[docs/MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MULTI_GPU.md)
and
[docs/SECOND_GPU.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/SECOND_GPU.md).
Key claims were re-checked against the source at this commit; where the guide and the code differ, the code is quoted. Every number is upstream's, and each says what it was
measured on.

---

## Part A: one RTX 3090

**What the card gets in the engine.** Compute capability 8.6 is at or above the engine's 7.5 minimum.
- **It gets** the sm_80 tensor-core prompt paths: the TF32 block scorer, the INT8 `cp.async` prompt attention, and
  the fused INT8 expert prompt kernels.
- **It does not get** the sm_90+ thread-block-cluster decode kernels (the sparse-attention top-k and the greedy
  argmax). Those are RTX 50-only. The tokens are the same, but they are slower at long context.

The details are on the [RTX 50 page](rtx50-flash-next.md).

**What upstream says about a 3090.**
- [DETAILS.md](../../upstream/DETAILS.md), requirements: "Measured on an RTX 5070 and an RTX 3090".
- DETAILS.md, "Other GPUs (estimated)": a table of estimates, ±20%, scaled from the RTX 5070 runs. For an RTX 3090
  with 64 GB RAM:
  - IQ3_XXS: about 106 tokens/s output at 1K context, about 71 at 128K;
  - Q2_0: about 128 tokens/s output at 1K.

  DETAILS.md adds that "a 3090's 24 GB takes most of the CPU work away".
- Measured on an RTX 3090 (DETAILS.md, "Hybrid K8V4 KV cache"): the Coder at 198K context wrote 99 tokens/s with
  `--kv k8v4` instead of 85, because a smaller KV cache leaves more VRAM for experts.

**Memory.** Same as any 24 GB card: about 40 GiB of IQ3_XXS experts, all pinned in RAM. VRAM holds the dense set
plus as many of the most-routed experts as fit, about 42% on 24 GB (measured on the fork's RTX 4090 with the same
24 GB). The installer picks the rest:
- the KV format (8-bit above 4K);
- KV streaming from 64K;
- `--prefill auto`;
- `--ple-io direct` (the 28.8 GB n-gram table read from the SSD, not RAM);
- a low-RAM mode when RAM is short.

See [../install/README.md](../install/README.md).

**Running it.** `./setup.sh` (Linux) or `START-HERE.bat` (Windows) detects the card, chooses the model by RAM, and
writes `strata-<model>.json`. The Python server then starts the engine from that config. With one card there is
nothing multi-GPU to choose.

---

## Part B: two RTX 3090s

### What upstream does

Upstream's main multi-GPU mode is the **layer split**, which is pipeline parallelism. The 48 layers are cut into
contiguous ranges:
- card 0 runs layers 0 to K-1, card 1 runs K to 47;
- the last card also runs the output head and the draft (MTP) layer, about 0.8 GB;
- each card keeps an expert cache **for its own layers only**, so two cards hold about twice the experts one card
  holds.

A token crosses from one card to the next once per verify window, "a few hundred KB through pinned RAM", not twice
per layer. **No NVLink or peer-to-peer access is needed**, and cards on x4 or x1 slots work. Each card's PCIe share
is probed on its own link (MULTI_GPU.md).

### Setting it up

There is nothing to type: setup lists the NVIDIA cards and recommends the two best together. The newest
generation becomes the main card; for two identical 3090s the order follows the card numbers. To choose yourself:

```
./setup.sh --gpus 0,1              # both cards; the first is the main one; remembered
./setup.sh --gpus 0,1 --layer-split auto   # auto is the default
./setup.sh --gpu 0                 # one card only
```

You can also edit the config and restart: `"gpu": [0, 1], "layer_split": "auto"`. `layer_split` is "auto" or the
**first layer of the later card** (for example `24`), not a count of layers per card.

The engine flags behind it are `--layer-split K|auto` and `--split-device D` (MULTI_GPU.md). `auto` tries every
placement for two cards. MULTI_GPU.md says it keeps the placement whose caches hold the most of the profile; the code
at this commit ranks placements by a **predicted decode-window time**, and logs that time together with the caches'
share (`src/program/generate.cpp:2837-2839`):

```
strata generate: layer split auto: K=<k> - predicted <t> ms per decode window; the caches hold <n> of <m> profiled pairs (~<p>% of the routed mass)
strata serve: layer split: layers <ranges>, one hand-off per window
```

The second line comes from `generate.cpp:5131`.

### What each card holds

From MULTI_GPU.md, "What each card holds":

| | Card 0 | Card 1 |
| --- | --- | --- |
| Dense weights | a full copy by default; only its own layers' with `STRATA_STAGE_TRIM=1` or `--trim-stage-weights` and an explicit K. With `auto` the trim is ignored, with a message (`generate.cpp:2202-2213`) | the same |
| Session | its own state: the KV of the full context for its attention layers, its GDN state | the same, for its layers |
| Verify window and prompt buffers | its own (1.5 GB at the default 2,048-token chunk; `--prefill 1024` halves it) | its own |
| Expert cache | its layers' experts, from the profile | its layers' experts |
| Head and draft layer | no | yes, ~0.8 GB |

The expert arena sits in host RAM once, shared by both cards, and the CPU pool computes whatever neither card holds.

**Trimming the dense copy is the one 3090-relevant opt-in.** Upstream measured it on 2x MI50 16 GB with the Coder:
8,819 to 10,626 experts in VRAM, and decode from 39.2 to 41.7 tokens/s. On two 24 GB 3090s with Flash-Next IQ3_XXS
(about 40 GiB of experts), the two caches together come close to holding every expert. This is an estimate, not an
upstream measurement.

### What upstream measured with a 3090 in the pair

**Two RTX 3090s, UD-Q4_K_XL, no P2P.** This was the one-chunk prompt "helper" (opt-in, `STRATA_PREFILL_HELP=1`,
MULTI_GPU.md). While one card reads its layers of a short prompt, the idle card streams and computes a share of
them. Prompt tokens/s, without / with the helper:

| Prompt | 1.1K | 1.5K | 2K | 2.5K | 3K | 4K and 8K |
| --- | --- | --- | --- | --- | --- | --- |
| tokens/s | 496 / 723 | 674 / 833 | 878 / 1,128 | 1,017 / 1,296 | 1,260 / 1,440 | unchanged |

Decode is unchanged. The helper costs no VRAM, and about 110 KB of mapped host memory per token of the largest
chunk it helped. Its output is repeatable but not bit-identical to the default, which is why it is opt-in. It works
with native packs on the MMQ prompt path, not with `STRATA_PF_FUSED=1` or `--peer-device`.

**RTX 5080 + RTX 3090, the Coder, 32K context** (Ryzen 9 9950X3D):

| | Prompt 16K / 28K, tokens/s | Decode story / code, tokens/s |
| --- | --- | --- |
| 5080 alone | 1,726-2,017 / 1,970 | 83-87 / 88-105 |
| 5080 + 3090, best split (K=26) | 2,039 / 2,357 | 84 / 110 |
| 5080 + 3090, auto (K=22) | 2,037 / 2,073 | 80 / 109 |

Upstream's reading of these numbers:
- **Prompts gain the most**, +18 to 20%, because each card reads its own layers of a chunk while the other reads the
  next.
- **Decode is on par with the faster card alone**, and ahead on code. Once both caches hold nearly every routed
  expert, the per-layer GPU time decides.
- **The hand-off is bit-exact.**

No two-3090 decode number for Flash-Next IQ3_XXS is published at this commit. Upstream's general note (DETAILS.md,
"Several GPUs") is that two users measured `--gpus 0,1` at 1.3 to 1.6 times one card.

### Advice from upstream's guide that applies to two 3090s

- Two identical cards need no ordering. The guide's rule is "put the fastest card first".
- Don't add a third, slower card when two already hold the model. An RTX 2080 Ti as a third card made a 5080 + 3090
  pair slower (68 / 90 tokens/s decode), because every extra card costs its own round per window.
- `--mmap-experts` needs a canonical pack. A native IQ pack refuses it, with or without a split.
- On Windows and WSL2, only 8 GiB of the expert arena is pinned with a split (`STRATA_ARENA_PIN_GIB` overrides).
  Linux pins the whole arena.
- **The other two-card modes are experimental.**
  - The helper-GPU caches (`--expert-cache-device1..3`, `setup.py --gpu1-experts N`; SECOND_GPU.md) were measured
    slower on the test rig. A 2080 Ti helper's per-layer round trip cost more than the CPU pool needed for those
    experts.
  - `--peer-device` is an adaptive second tier. Its prompt help needs P2P, which on two 3090s means an NVLink
    bridge.
  - Prefer the layer split.
- Several conversations at once: with a split, `--batch N --batch-groups G --trim-stage-weights` pipelines
  conversations through the cards (BATCHING.md).

### Code behind it

| Piece | Where (upstream) | Doc |
| --- | --- | --- |
| Layer split: parsing, auto placement, stages, hand-off | `src/program/generate.cpp` | [../program/layer-split.md](../program/layer-split.md) |
| Expert tiers incl. helper caches and the peer | `src/program/generate.cpp`, `src/core/remote_experts.cpp`, `src/core/peer_experts.cpp` | [../program/expert-tiers.md](../program/expert-tiers.md) |
| Installer's GPU listing and `--gpus` | `setup.py` | [../install/README.md](../install/README.md) |

---

## Compared with UnbiasedStrata

The fork's page is [../../codebase/guides/rtx3090-flash-next.md](../../codebase/guides/rtx3090-flash-next.md).

| | Upstream | UnbiasedStrata |
| --- | --- | --- |
| One 3090 | supported; estimates in DETAILS.md; the installer picks the settings | the fork's tested configuration: 107 tokens/s, 2,483 prompt tokens/s (RunPod, 125 GB RAM), with fixed front-end flags and `--ple-io ram` |
| Two 3090s | supported: the layer split through `--gpus 0,1`, with no P2P needed | not supported. The C++ front end refuses `--layer-split` and `--peer-device`; only `--expert-cache-device1` passes, untested |
| Server | Python (`serve/server.py`) | C++ front end in the same binary |
| n-gram table | `--ple-io direct` (SSD) by default | `--ple-io ram` |
