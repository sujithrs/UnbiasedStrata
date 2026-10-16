[Upstream docs](../README.md) › [Guides](README.md) › Models

# Models: every model file and quantization upstream runs

## Purpose

Upstream Strata runs one architecture, Qwen3.8-Flash-Next (llama.cpp architecture `qwen4exp`), from several
publishers' GGUF files. This page lists every variant the installer offers and every one with a manual recipe: its
files, its pack type, its expert count and formats, its sizes, and what in the engine exists because of it. Facts are
taken from `setup.py` (the `MODELS` and `FAMILIES` tables), `docs/MODELS.md`, `docs/UNSLOTH_Q4.md`, `docs/ORCA*.md`,
the `tests/data/native_experts/*.txt` copies of the real packs' expert tables, and a source search for each name.

Files covered:

| Path | Lines | Role |
| --- | ---: | --- |
| [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py) | 4,422 | `HF_REVISIONS` (`:65-70`), `MODELS` (`:122-152`), `UNSLOTH_SHARDS` (`:155-173`), `FAMILIES` (`:183-215`), pack building (`:4172-4194`), engine arguments (`:4224-4228`) |
| [docs/MODELS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MODELS.md) | 155 | the user's guide: pick by RAM, speed, sizes, fit, versions |
| [include/strata/core/layout.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/layout.hpp) | | `ModelGeometry`, the compile-time shape every variant shares |
| [tests/data/native_experts/](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/tests/data/native_experts) | 6 x 49 | per-layer expert formats and sizes of six real packs |
| [data/](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/data) | | `expert-profile.bin`, `expert-profile-coder.bin`, `draft_vocab*.bin`, the speed-projection control vector |

---

## 1. The shared architecture

Every variant has the same dense shape; only the routed experts differ. From `include/strata/core/layout.hpp:27-55`
and the paper (`docs/paper/Strata-Paper.pdf`, section 2):

| Part | Value |
| --- | --- |
| Parameters | about 125B, about 6B active per token (paper abstract) |
| Layers | 48, width `n_embd` 2560; every 4th is full attention (layers 3, 7, ..., 47): 12 QSA layers, 36 GDN layers |
| GDN (Gated DeltaNet) | state 128, 16 key heads, 48 value heads, conv kernel 4 |
| QSA (Qwen Sparse Attention) | 24 heads, 2 KV heads, head dim 256; an indexer with 4 query heads of 128 picks about 2,048 positions |
| Hyper-connections | `hc` 4 residual streams, `hc_lr` 320 |
| MoE | 512 routed experts per layer, top-10, expert width `n_ff` 640, plus a shared expert; 48 x 512 = 24,576 experts |
| PLE | a per-layer token embedding whose 28.8 GB n-gram table (`per_layer_token_embd.weight`, rows of 160 values) is addressed by the last three token ids |
| MTP | one multi-token-prediction layer, the drafter for speculative decoding |
| Context | trained 262,144; setup offers up to 524,288 with rope scaling (`setup.py:181`) |

The engine refuses any other architecture: `Qwen4ExpGuard` in `include/strata/artifact/gguf_reader.hpp:573-578`
pins 48 blocks, width 2560, 24 heads and 2 KV heads, and treats the expert count as "presence-only" because "pruned
variants (GSQ-RCO Coder) legitimately ship fewer experts than the canonical 512".

## 2. Canonical and native packs

Every model needs a **pack**, a directory beside the GGUF that the engine loads with `--pack` (see
[glossary.md](glossary.md#models-files-and-packs)). Upstream has two kinds:

| | Canonical pack | Native pack |
| --- | --- | --- |
| Built by | `tools/strata_pack.py build` then `tools/pack_index.py` | `tools/iq_pack.py` |
| Experts | re-encoded into Strata's own **S2** form (Q2_0's 2-bit codes, decoded as `(code - 1) * d`), all in `experts.bin`, one 1,382,400-byte blob per expert | kept as the GGUF stores them; `native_experts.txt` says where each layer's gate, up and down rows are; `experts.bin` only with `--experts-bin` (low-RAM mode) |
| Rule | `decode(pack) == dequant_ggml(source)` bit for bit, enforced by `strata_pack.py verify` | conversions listed in `conversions.json`; exact BF16 rewrites only, unless `--compat-bf16` |
| Used for | Q2_0 of the original model, on a CPU with full AVX-512 (`setup.py:4174-4183`) | every other case |
| CPU expert kernels | AVX-512 VNNI only (`cpu_require_expert_support`, `src/program/generate.cpp:2053-2054`) | AVX-512 or AVX2 kernels for the i-quants, ggml-cpu `vec_dot` otherwise, an AVX-1 floor for the older-CPU build |
| Default PCIe share of misses | 0.2 | 0.55, scaled by a startup PCIe probe (`src/program/generate.cpp:2038-2051`) |

Setup's comment for the canonical choice: "the Q2_0 experts repacked for the AVX-512 kernel (the measured speed): a
one-time ~40 GB conversion" (`setup.py:4175-4176`). Code paths that depend on the kind: `--mmap-experts` needs an
`experts.bin` (a native pack without one gets a start-up error, `docs/MULTI_GPU.md:137`); the GGUF-in-place low-RAM
mode and `--resident-budget-gib` apply to native packs only (`docs/DETAILS.md:160-161`). The tools are documented in
[../tools/README.md](../tools/README.md); the expert tiers in [../core/experts/README.md](../core/experts/README.md).

---

## 3. Variants in setup's menu

`--family` picks the publisher, `--model` the size. Sizes are setup's own figures (`setup.py:122-152`): `download_gb`
(all shards), `ram_gb` (the RAM setup recommends), `arena_gb` (the routed experts that go into RAM).

| `--family` / `--model` | Publisher (pinned revision) | Shards | Download | Experts in RAM | Setup's RAM | Experts per layer | Pack |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen` / `Q2_0` | ISTA-DASLab GSQ-RCO (`ed59f920`) | 2 | 66.4 GB | 34.0 GB | 48 GB | 512 | canonical on AVX-512, else native |
| `qwen` or `swift` / `IQ2_XS` | ISTA-DASLab; UkisAI (`b22d729e`) | 2 | 68.0 GB | 35.5 GB | 48 GB | 512 | native |
| `qwen` or `swift` / `IQ3_XXS` | ISTA-DASLab; UkisAI | 2 | 75.8 GB | 42.9 GB | 60 GB | 512 | native |
| `qwen` / `IQ3_S` | ISTA-DASLab | 2 | 83.6 GB | 50.3 GB | 62 GB | 512 | native |
| `coder` / `IQ1_M` | ISTA-DASLab Coder (`5348543e`) | 2 (shard 2 shared with `qwen`) | 58.4 GB | 23.4 GB | 32 GB | 256 | native |
| `unsloth` / `UD-IQ4_XS` | Unsloth (`38bb39ee`) | 3 | 93.7 GB | 59.5 GB | 48 GB (RAM budget) | 512 | native, `--compat-bf16` |
| `unsloth` / `UD-Q4_K_XL` (experimental) | Unsloth (`38bb39ee`) | 4 | 111.3 GB | 77.0 GB | 48 GB (RAM budget) | 512 | native, `--compat-bf16` |

Models without a `families` key are offered to `qwen` and `swift` (`setup.py:1156, 3871`). Swift 1.5 has no IQ3_S, and
its Q2_0 is not offered because its files "split one layer's experts across the two shards, which the pack tool ...
cannot prepare yet (#171)" (`setup.py:123-124`). `docs/MODELS.md:58-66` gives the RAM+VRAM need of the four original
sizes as 37.6 / 39.2 / 47.0 / 54.8 GB.

### Expert formats and sizes per pack (from `tests/data/native_experts/`)

The header of each copied `native_experts.txt` names the model file, `n_expert` and the total expert bytes; each line
gives a layer's gate/up and down ggml types and blob size. ggml type numbers: 16 IQ2_XXS, 17 IQ2_XS, 18 IQ3_XXS, 20
IQ4_NL, 21 IQ3_S, 22 IQ2_S, 23 IQ4_XS, 29 IQ1_M, 42 Q2_0.

| Pack | Gate/up types (layers) | Down types (layers) | Blob bytes | Total expert bytes |
| --- | --- | --- | --- | ---: |
| Q2_0 | Q2_0 (48) | Q2_0 (48) | 1,382,400 | 33,973,862,400 |
| IQ2_XS (original and Swift) | IQ2_S (34), IQ2_XXS (11), IQ1_M (3) | Q2_0 (48) | 1,177,600-1,510,400 | 35,454,976,000 |
| IQ3_XXS | IQ3_S (13), IQ2_XS (10), IQ2_S (10), IQ2_XXS (9), IQ3_XXS (6) | Q2_0 (30), IQ4_NL (18) | 1,305,600-2,329,600 | 42,912,972,800 |
| IQ3_S | IQ2_S (20), IQ3_XXS (17), IQ3_S (10), IQ4_XS (1) | IQ4_NL (39), Q2_0 (9) | 1,510,400-2,662,400 | 50,292,326,400 |
| Coder IQ1_M | the same as IQ3_S | the same as IQ3_S | 1,510,400-2,662,400 | 25,146,163,200 (half of IQ3_S) |

So a file's name is not one format: "IQ3_XXS" mixes five gate/up types, and the Coder's "IQ1_M" is "the release's name
for 1.89 bits per original parameter" while its experts are stored like IQ3_S (`bench/results/2026-09-28-coder/README.md:18-21`).

---

## 4. Each variant and what the engine does for it

### Qwen3.8-Flash-Next, GSQ-RCO Q2_0 / IQ2_XS / IQ3_XXS / IQ3_S

- **Files.** `Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{1,2}-of-00002.gguf` from `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`,
  plus `mmproj-Qwen3.8-Flash-Next-BF16.gguf` for images. Shard 1 holds the experts and dense weights (37.6-47 GB);
  shard 2 is the 28.8 GB IQ4_NL n-gram table alone, read from the SSD (`src/kernels/ngram.cpp:196, 229-231`: alone in
  its shard it must fill the file exactly).
- **Q2_0.** The fastest. On AVX-512 the canonical pack and its S2 kernels (`dequant_s2`, `s2_gemv*`,
  `s2_expert_grouped`; CPU `src/kernels/cpu/expert.cpp`); the prompt path converts its blobs back to GGUF Q2_0 blocks for
  MMQ (`src/prefill/prefill.cpp:656-657`) and, from 0.1.36, runs them on Strata's fused int8 kernels on RTX 30 and newer
  (#136, `STRATA_PF_FUSED=0` keeps the previous ones; `docs/DETAILS.md:22-29`).
- **IQ2_XS, IQ3_XXS, IQ3_S.** Native packs. GPU: the grouped i-quant kernels in `src/kernels/cuda/iq_kernels.cu`
  (gate/up and down format lists at `:525-527`). CPU: `src/kernels/cpu/iq_avx512.cpp` and `iq_avx2.cpp` cover IQ2_XXS,
  IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S (AVX2 also IQ4_XS); IQ1_M "stays on ggml-cpu" (`iq_avx2.cpp:12`). The three IQ1_M layers
  of IQ2_XS also keep the FP16 prompt path, which MMQ does not cover (`src/prefill/prefill.cpp:656-657, 2260`). The
  fused int8 prompt kernels for these packs are opt-in (`STRATA_PF_FUSED=1`).
- **IQ3_S** is "the best quality (matches the full model), the slowest" (`setup.py:131-132`); setup recommends at most
  128K context with it and IQ3_XXS on 64 GB (`docs/DETAILS.md:62-66`).

### Coder (`--family coder`, IQ1_M)

- ISTA-DASLab's expert-pruned release: 256 of 512 experts per layer kept, chosen on code, agentic and vision data;
  10 still active per token. Its authors report 91.3% of the full model's SWE-bench Verified (`docs/DETAILS.md:253-260`).
  Weaker outside code, including CJK text (#438).
- **Files.** Its own shard 1 (`Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf`, 29.6 GB); shard 2 and the vision
  encoder are the original's, so setup downloads only shard 1 when the original is installed.
- **Engine.** The expert count comes from the model file: `qwen4exp.expert_count` and `expert_used_count`
  (`src/program/generate.cpp:2074-2078`) and the v3 `native_experts.txt` header's `(n_expert 256)`
  (`src/kernels/cpu/expert_layout.cpp:268-272`), overriding the compile-time 512. The MTP drafter keeps the 512-expert
  geometry, because the draft layer is the original model's (`src/program/generate.cpp:3011-3013`). On gfx906 the
  256-expert router runs a whole verify window in one projection (`src/core/verify.cpp:924-926`). Its expert profile
  is `data/expert-profile-coder.bin` (48 x 256), the shipped ranking mapped through the release's
  `rco-allocation.txt` (`FAMILIES["coder"]["profile"]`, `setup.py:205`).
- With half the experts, the 23 GB pins completely and a 12 GB card holds about 20% of them instead of about 10%
  (`bench/results/2026-09-28-coder/README.md:21-22`); it is the model of every multi-GPU result set.

### Swift 1.5 (`--family swift`)

- UkisAI's fine-tune that thinks shorter ("-63% thinking tokens, 1.8x sooner answers by its authors' numbers",
  `setup.py:191`); Swift Open License 1.0. Sizes IQ2_XS (its authors' recommendation) and IQ3_XXS; same speed per
  token as the original (4K, IQ2_XS: 465 / 78.7 vs 467 / 78.3 tokens/s, `docs/DETAILS.md:270-276`).
- **Files.** `Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{1,2}-of-00002.gguf` and its own `mmproj-Swift-...-BF16.gguf`.
- **Engine.** Its shards are cut differently: shard 1 holds layers 0-12 and the n-gram table, shard 2 layers 13-47.
  `iq_pack.py` writes a v3 `native_experts.txt` with a shard column on those layers (the `swift-iq2_xs.txt` fixture),
  `--ple-gguf` defaults to whichever shard holds `per_layer_token_embd.weight` (`src/program/generate.cpp:1786-1792`),
  the PLE reader accepts a table that shares its file (`src/kernels/ngram.cpp:229-231`), and the native dense loader
  reads every shard (`src/program/generate.cpp:1810-1813`). Its routers are stored as F32 whose values are exactly
  BF16, which `iq_pack.py` writes as BF16. It uses the original model's MTP draft layer.

### Unsloth UD-IQ4_XS and UD-Q4_K_XL (`--family unsloth`)

- **Formats.** UD-Q4_K_XL: experts Q4_K (Q5_K in layer 2) gate/up and Q5_1 (Q8_0 in layers 2, 4, 30, 46, 47) down,
  71.7 GiB; embedding, head, attention and shared-expert projections and the PLE key Q8_0; the PLE table IQ4_NL
  (`docs/UNSLOTH_Q4.md:8-12`). UD-IQ4_XS: IQ3_S gate/up (IQ4_XS in one layer) with IQ4_NL down (Q8_0 in five layers),
  59.5 GB; the dense side as UD-Q4_K_XL's, with a Q6_K head (`docs/UNSLOTH_Q4.md:224-230`).
- **Files.** `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf` and `...UD-IQ4_XS-0000{1..3}-of-00003.gguf`; shard 1
  holds only metadata (10,946,624 bytes). Setup checks every shard's size and SHA-256 against `UNSLOTH_SHARDS` /
  `UNSLOTH_IQ4_XS_SHARDS`. UD-Q4_K_XL puts layer 11's down in shard 2 and its gate/up in shard 3: the per-role shard
  column of `native_experts.txt` v4 (`include/strata/kernels/cpu/expert_layout.hpp:28`).
- **Pack.** Native with `--compat-bf16` (`FAMILIES["unsloth"]["pack_args"]`): the 195 Q8_0 hyper-connection and PLE
  value projections are rounded to BF16, 264 exactly-BF16 F32 tensors stored as BF16, every conversion listed in
  `conversions.json` (`docs/UNSLOTH_Q4.md:72-99`). Never `experts.bin`.
- **RAM budget.** The experts do not fit RAM on 64 GB, so the engine keeps `--resident-budget-gib N` of them (setup: the
  RAM less 24 GB, `UNSLOTH_RAM_LEFT_GB`) and reads the rest from the GGUF on the SSD while it answers, with a
  routing-aware prefetch (`STRATA_LOOKAHEAD`). One GPU by default: the budget has no layer split; two GPUs only with
  enough RAM to load everything (#498).
- **Engine.** GPU kernels for Q4_K, Q5_K, Q5_1 and Q8_0 experts (`src/kernels/cuda/iq_kernels.cu:309` onward); CPU
  multi-token AVX2 kernels in `src/kernels/cpu/kq_avx2.cpp` (opt-in `STRATA_KQ256=1`, measured no faster); prompt MMQ
  instances for `q4_k q5_k q5_1` only with `-DSTRATA_MMQ_KQUANTS=ON` on CUDA (`CMakeLists.txt:1056-1065`), so
  UD-Q4_K_XL is `nvidia_only` in setup; split-GGUF loading with a tensor-free shard 1 (`src/core/native_head.cpp:27`,
  `include/strata/artifact/gguf_reader.hpp:494`); a Q8_0 PLE key (`src/core/native_dense.cpp:27`).
- **Status.** UD-Q4_K_XL is experimental: 7-8.5 tokens/s on a 64 GB PC with an RTX 5070, same tokens as llama.cpp at
  97.5-99% of short-answer positions. UD-IQ4_XS is a regular choice from 0.1.39 (engine 0.1.38 or newer), measured
  only on a Strix Halo (gfx1151), not yet on NVIDIA (`docs/UNSLOTH_Q4.md:244-246`).

---

## 5. Variants with a manual recipe only

| Variant | Recipe | What is special |
| --- | --- | --- |
| OrcaRouter Uncensored IQ3_XXS (`orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF`, 2 shards, 85.20 GB) | [docs/ORCA.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/ORCA.md) | quantized hyper-connection, small attention and PLE projections, so `iq_pack.py --compat-bf16` (460 tensors converted); its BF16 PLE key stays in the arena (#326, `src/program/generate.cpp:2195-2199`, test `native_dense_ple_key_test`); PLE table in shard 1; 48 layers IQ3_XXS gate/up, IQ4_NL down; ~49.8 GiB of experts |
| OrcaRouter Uncensored Q4_K_S (3 shards, 112 GB) | [docs/ORCA_Q4_K_S.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/ORCA_Q4_K_S.md) | Q4_K gate/up with Q5_0 / Q5_1 down; a Q5_0 PLE table (110-byte rows, `src/kernels/ngram.cpp:196-200`) read with `--ple-io mmap`; `-DSTRATA_ORCA_Q4KS_MMQ=ON` adds the Q5_0 MMQ instance; real-weight checks `native_expert_parity` and `ple_q5_parity` |
| An FP8 n-gram table | `tools/ple_fp8_pack.py` (#291) | the table as I8 bytes tagged `strata.ple.format = f8_e4m3` with a scale, 160-byte rows (`src/kernels/ngram.cpp:201-211`); checked by `ple_fp8_parity` |
| A BF16 token embedding | `tools/embd_bf16_pack.py`, `--embd-gguf` | the embedding exactly as the checkpoint ships it, in mapped host memory (#290, `native_expert_parity_bf16_embd`) |

Both Orca recipes use the original model's MTP draft layer and say its acceptance on the fine-tune must be measured.
`docs/ORCA.md:85` notes that IQ3_M "additionally uses Q5_0 expert down matrices, which the native GPU expert path
does not support"; since engine 0.1.38 (#473) Q5_0 is in the GPU down-format list (`src/kernels/cuda/iq_kernels.cu:526`),
so that line and `docs/ORCA_Q4_K_S.md:5` are older than the code.

**NVFP4** appears only as a type name (`include/strata/artifact/gguf_reader.hpp:110`, `tools/gguf_reader.py:27`) and as
"the NVFP4 fork" in a measurement (`docs/DETAILS.md:109`) and in comments (`src/program/generate.cpp:1759, 4440`).
Upstream has no NVFP4 expert kernel; `tools/embd_bf16_pack.py:5` refers to that fork's `nvfp4_convert.py`, which is not
in the tree.

## 6. What every variant shares

- **MTP draft layer.** From the original `Qwen/Qwen3.8-Flash-Next` checkpoint, not from the GGUF: `tools/mtp_fetch.py`
  range-reads ~5 GB of MTP tensors, `tools/mtp_pack.py --experts q2_0` and `tools/mtp_rt.py` build `mtp/rt/`
  (`setup.py:4195-4210`). Its draft vocabulary is `data/draft_vocab.bin` (106,299 ids incl. CJK, the default),
  `draft_vocab_en.bin` (40,525), `draft_vocab_cyrillic.bin` (58,963) or `draft_vocab_fr.bin` (46,211)
  (`docs/DETAILS.md:102-113`).
- **Expert profile.** `data/expert-profile.bin` ranks all 48 x 512 pairs; the Coder has its own.
- **Engine arguments.** Setup writes `--pack`, `--native SHARD1`, `--ple-gguf` (two-shard files only), `--expert-profile`,
  `--expert-cache auto`, `--prefill auto`, `--spec 4 --spec-min-p 0.5`, `--mtp`, `--max-context`, plus `--kv` above 8K
  and the low-RAM flags (`setup.py:4224-4238`).
- **Optional control vector.** `data/experimental-speed-projection/` (47 directions, off by default) was made from the
  full model and loads on the Coder too (`bench/results/2026-09-28-coder/README.md:40-43`).

## Gotchas

- `setup.py` identifies an existing GGUF by its published first-shard name (`gguf_choice`, `:1151-1158`); the Coder's
  IQ1_M is named like the original's sizes, and Unsloth's UD-IQ3_XXS can be mistaken for IQ3_XXS by name (#444,
  `gguf_dir_problem`).
- The comment at `src/kernels/cpu/iq_avx512.cpp:9-10` says no shipped model has IQ1_M expert rows; the IQ2_XS fixture
  has three IQ1_M gate/up layers (`tests/data/native_experts/iq2_xs.txt`, type 29).
- `--compat-bf16` rounding put perplexity 6-9% above llama.cpp's on the same file on a Strix Halo
  (`docs/UNSLOTH_Q4.md:246-249`).

## In UnbiasedStrata

The fork supports only the original Flash-Next at IQ3_XXS among these (its README, "The supported models") and adds
Qwen3.6-35B-A3B (a different architecture, `qwen35moe`). It keeps the native pack path and the `tests/data/native_experts`
fixtures, removed `setup.py`, `data/expert-profile-coder.bin`, `draft_vocab_cyrillic.bin`, `draft_vocab_fr.bin` and the
speed-projection vector (8894a81 and later), and does not support the Coder, Swift or the Unsloth Flash-Next files.
See [../../codebase/guides/glossary.md](../../codebase/guides/glossary.md).

## See also

- [glossary.md](glossary.md) - quantization formats, packs, profiles.
- [benchmarks.md](benchmarks.md) - every measured number per model.
- [../install/README.md](../install/README.md) - how setup downloads, verifies and packs each model.
- [../tools/README.md](../tools/README.md) - `iq_pack.py`, `strata_pack.py`, `mtp_*.py`.
- [../kernels/cuda/README.md](../kernels/cuda/README.md), [../kernels/cpu/README.md](../kernels/cpu/README.md) - the
  expert kernels per format.
