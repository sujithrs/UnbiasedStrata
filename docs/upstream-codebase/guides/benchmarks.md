[Upstream docs](../README.md) › [Guides](README.md) › Benchmarks

# Benchmarks: what upstream measured, where, and on which hardware

## Purpose

Every speed or quality number in upstream's README, `docs/MODELS.md` and `docs/DETAILS.md` comes from a measurement
that is either written up in `bench/results/<date>-<topic>/` or quoted with its machine in a prose doc. This page
indexes all 171 files under `bench/`, the community report process in `docs/COMMUNITY_BENCHMARKS.md`, and the one
file in `docs/benchmarks/`. It says what each result set measured, on which hardware and engine version, with which
model, and which document quotes it. Pass/fail tests are in [testing.md](testing.md); the models in [models.md](models.md).

Files covered:

| Path | Files | Role |
| --- | ---: | --- |
| [bench/results/](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results) | 171 in 18 directories | result sets: a README (most), JSON per run, a few scripts |
| [docs/COMMUNITY_BENCHMARKS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/COMMUNITY_BENCHMARKS.md) | 170 lines | what to record, how to measure, how to submit a community report, and a template |
| [docs/benchmarks/2026-09-29-gfx1100.json](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/benchmarks/2026-09-29-gfx1100.json) | 7,186 bytes | the RX 7900 XTX control vs candidate runs behind `docs/AMD_HIP_PERFORMANCE.md` |

`bench/` holds only `results/`. There is no `bench/micro/` (the micro benchmarks and test fixtures the CMake file
looks for, `CMakeLists.txt:87-92`) and no `bench/prompts/` (the frozen code-agent prompts the speed READMEs name, e.g.
`bench/prompts/long/32k-01-code-agent.ids`). Both are left out of the published source.

---

## 1. The reference machine

Most upstream numbers come from one PC, called "the bench machine" in the READMEs:

- **RTX 5070 12 GB** (PCIe 5.0 x16), **Ryzen 5 7600** (6 cores, AVX-512), **64 GB DDR5-5200**, Windows 10
  (`bench/results/2026-09-28-speed-0114/README.md:3`, `docs/DETAILS.md:17`).
- Method for the speed matrices: the ready-made engine run one-shot with what setup writes (`--prefill auto`, the
  shipped expert profile, `--expert-cache auto`, 8-bit KV above 4K, `--kv-resident 32768` from 64K, MTP
  `--spec 4 --spec-min-p 0.5`), greedy, 256 generated tokens, one code-agent prompt per context length.
- "IQ2_XS" rows on this machine were measured with Swift 1.5's IQ2_XS, which runs at the original's speed (4K: 465 vs
  467 prompt tokens/s).

The second reference is AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM, Linux (`docs/MODELS.md:35-42`); its runs are
written up in `docs/AMD_HIP.md`, not under `bench/`.

### Which document quotes which result set

```
README.md "How fast is it?"  ->  docs/MODELS.md "How fast is each size"  ->  docs/DETAILS.md "Speed (measured)"
                                                                               |
     engine 0.1.26 tables  <-  bench/results/2026-09-29-speed-0126   <---------+
     0.1.36 deltas (#136)     (quoted in DETAILS.md:22-29, no result directory)
     older history         <-  2026-09-24-final (0.1.12-), 2026-09-28-speed-0114, 2026-09-29-speed-0122
```

---

## 2. The result directories

| Directory | Files | Question | Hardware | Engine | Model(s) |
| --- | ---: | --- | --- | --- | --- |
| [2026-09-24-final](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-24-final) | 4 | the first full speed matrix 1K-262K, plus estimates for other cards | RTX 5070 (estimates: RTX 5060 Ti 16 GB, RTX 3090 24 GB) | before 0.1.13 | Q2_0, IQ2_XS, IQ3_XXS |
| [2026-09-24-vision](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-24-vision) | 2 | speed with the image encoder on vs off, 4K | RTX 5070 | not stated | Q2_0, IQ2_XS, IQ3_XXS |
| [2026-09-27-cache-parity](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-27-cache-parity) | 1 | does the GPU expert cache make output worse? (issue #23) | RTX 5070 | not stated | Q2_0 |
| [2026-09-27-esp](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-27-esp) | 5 | the experimental-speed-projection control vector: kernel, teacher-forced KL, serve checks | RTX 5070 | not stated | Q2_0 |
| [2026-09-27-kv-q4](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-27-kv-q4) | 1 | 4-bit KV (PR #21): precision, needles, speed, memory | RTX 5070 | 0.1.8 | Q2_0 |
| [2026-09-28-coder](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-28-coder) | 2 | the GSQ-RCO Coder (256 experts): speed matrix, profile mapping, images, the vector | RTX 5070, Windows 10 | 0.1.15 + PR #54 | Coder IQ1_M |
| [2026-09-28-prefill-speed](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-28-prefill-speed) | 1 | the 0.1.13 prompt path, step by step | RTX 5070 | 0.1.12 -> 0.1.13 | Q2_0, IQ3_S |
| [2026-09-28-rope-scaling](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-28-rope-scaling) | 10 | needle recall past the trained 262,144 with linear and YaRN scaling (PR #84) | RTX 4070 Ti SUPER 16 GB, Ryzen 9 5900X (no AVX-512), 125 GB, Linux | 0.1.17 base, re-checked on 0.1.20 | IQ3_XXS |
| [2026-09-28-speed-0114](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-28-speed-0114) | 3 | the speed matrix after the 0.1.13 prompt path | RTX 5070 | 0.1.14 | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S |
| [2026-09-29-layer-split](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-29-layer-split) | 35 | the multi-GPU layer split: correctness, K sweep, auto placement | community "RIBPC": RTX 5080 16 GB + RTX 3090 24 GB + RTX 2080 Ti 22 GB, Ryzen 9 9950X3D, 62 GB, Windows 11 | 0.1.20 vs the split branch | Coder IQ1_M |
| [2026-09-29-layer-split-limits](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-29-layer-split-limits) | 15 | features that had been off across GPUs (KV streaming, control vector, images, mid-prompt checkpoints, helpers) | RIBPC | branch `mg/limits` (0.1.21 era) | Coder IQ1_M |
| [2026-09-29-rtx3090-epyc-milan](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-29-rtx3090-epyc-milan) | 60 | preliminary native Linux results, one and two 3090s (issue #165) | 2x RTX 3090 24 GB at 280 W, EPYC 7453 (20 vCPU, AVX2), 165 GiB, Ubuntu 24.04 KVM guest | 0.1.26 source build `ac8b251` | IQ3_XXS, IQ3_S, Coder IQ1_M |
| [2026-09-29-speed-0122](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-29-speed-0122) | 2 | the speed matrix | RTX 5070 | 0.1.22 | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, Coder |
| [2026-09-29-speed-0126](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-29-speed-0126) | 2 | the speed matrix behind the published tables | RTX 5070 | 0.1.26 | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, Coder |
| [2026-09-30-community-rtx-5090](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-30-community-rtx-5090) | 17 | a community report: decode, prompt, TTFT, memory, recall | RTX 5090 32 GB, Core Ultra 9 285K, 64 GB, Ubuntu 24.04 (container) | 0.1.29 (`d6708a4`) | IQ2_XS, 131,072 context |
| [2026-09-30-tc-gemv-sm75](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-30-tc-gemv-sm75) | 2 | can the decode BF16 GEMVs use Turing tensor cores? (a measured negative) | RTX 2070 8 GB (sm_75), i7-8700K | 0.1.27 tree | none (isolated kernel shapes) |
| [2026-10-03-community-2x-mi50](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-10-03-community-2x-mi50) | 8 | a community report on the gfx906 build (#638) with a layer split | 2x Instinct MI50 16 GB at 85 W, Xeon E5-2666 v3, 32 GB, Ubuntu 24.04 | 0.1.38 + #638 #639 #640 #637 | Coder IQ1_M, 131,072 context |
| [2026-10-03-v100-prompt-attn](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-10-03-v100-prompt-attn) | 1 | a Volta tensor-core QSA prompt attention: speed, nsys profile, accuracy, distribution | Tesla V100-PCIE-32GB, i9-7960X, 125 GB, Linux, CUDA 12.8 | `99f3dbd` (0.1.35 + 83 commits) | Unsloth UD-IQ4_XS (not a setup model then) |

File counts add up to 171.

### What each one found (quoted from its README)

- **2026-09-24-final.** `matrix.md`: Q2_0 prompt 389-571 tokens/s, output 88.7 (1K) / 94.6 (4K) / 56.3 (262K);
  IQ3_XXS output 64.6-44.8. `estimates.md` scales the GPU part by memory bandwidth and the CPU part by how many more
  experts fit (RTX 3090 Q2_0 output "~140" at 4K). The paper's numbers (95 tokens/s at 4K, prompts 285-571) are these.
  `docs/DETAILS.md:217-221` still quotes the estimates for other GPUs, marked ±20% and predating 0.1.13.
- **2026-09-24-vision.** Images on cost 2-8%: about 1,000 fewer expert slots (Q2_0 4,464 -> 3,402). `--vision` alone
  gives the same tokens. Quoted in `docs/DETAILS.md:923-933`.
- **2026-09-27-cache-parity.** Teacher-forced `--dump-logits`, cache off vs `--expert-cache auto` (5,130 experts on the
  GPU): same top-1 95.0-97.7%, perplexity difference -0.005 ± 0.005 nats over 2,557 tokens. The GPU and CPU compute
  the same quantized expert in a different floating-point order; neither is more accurate.
- **2026-09-27-esp.** The 47-direction control vector: `cvec_parity` max error 5e-7; teacher-forced mean KL 0.063,
  cost 0.2-0.4% per token; loaded-but-off is token-identical to stock. Scripts: `esp_kl.py`, `esp_serve.py`; raw
  output `teacher-forced.txt`, `serve-checks.txt`.
- **2026-09-27-kv-q4.** int8 KV is indistinguishable from fp16; q4_0 costs document perplexity +8% at 1K and +12% at
  8K, passes 5/5 needles, decodes 62.4 -> 67.4 tokens/s at 128K. Decision: q4_0 is an option, int8 the default.
- **2026-09-28-coder.** Coder prompt 599-1,350, output 42.8-53.3 tokens/s; 72% of a 4K code prompt's expert reads hit
  the GPU with the mapped profile (a wrong mapping would give about the 21% slot share).
- **2026-09-28-prefill-speed.** Seven steps (shared buffers, PLE prefetch, chunk sizing, helper copies, batched PLE
  block, MMQ experts, streaming ring): Q2_0 32K prompt 572 -> 1,290, IQ3_S 383 -> 1,208 tokens/s; quality judged
  against the change a different chunk size alone makes.
- **2026-09-28-rope-scaling.** 8 of 8 needles found, up to 512K (527,051 tokens) with linear and YaRN factor 2. Configs
  `config-*.json`, results `needle-*.json`, `kv-sweep-tgs.json`.
- **2026-09-28-speed-0114.** `matrix.md` has prompt, output, draft acceptance and experts in VRAM; IQ3_XXS 64K/128K
  needed `CUDA_MODULE_LOADING=EAGER`, which the engine now sets itself.
- **2026-09-29-layer-split.** Best two-GPU split for the Coder K=26-28: prompts +18-20% over the 5080 alone, decode on
  par to +7%; a slow third card costs more than it holds; `--layer-split auto` picked K=22. Tools `rib_pp.py`,
  `rib_norepeat.py` and the `rib_*.bat` drivers are "in the dev notes, not in this repository".
- **2026-09-29-layer-split-limits.** KV streaming, the control vector, images and mid-prompt checkpoints work across
  the split, byte-identical on one GPU; a helper card makes decode slower; `--mmap-experts` refuses a native pack.
- **2026-09-29-rtx3090-epyc-milan.** IQ3_XXS on one 3090: prompt 870-2,174, decode 93.0 (1K) to 67.3 (262K, median);
  Coder at 32K: 66.8 tokens/s on one card, 98.6 on two (auto K=19). Its own harness (`data/strata-bench.py`) and a
  source-tree corpus, not upstream's prompts: "indicative only". `hw.txt` records the guest's hardware facts.
- **2026-09-29-speed-0122 / -0126.** The tables in `docs/DETAILS.md:31-55` are 0.1.26's: Q2_0 prompt 536-2,171,
  output 73.7-93.0; the gains over 0.1.22 come from QSA selection on tensor cores (0.1.24), grouping tables in mapped
  memory and fused norms (0.1.25), the batched draft-layer pass (0.1.26).
- **2026-09-30-community-rtx-5090.** Decode 179.4 / 175.7 / 165.0 tokens/s at 4,096 / 32,768 / 128,000 prompt tokens,
  prompt 4,270-5,779 tokens/s, cache 17,463 slots with no eviction, 6/6 needles. Reproducible: `Dockerfile`,
  `benchmark.py`, `monitor.py`, the run config `strata-iq2_xs.json`, hashes of every artifact.
- **2026-09-30-tc-gemv-sm75.** Tensor-core GEMV 1.3-2.2x slower than the warp kernel on every decode shape; the warp
  kernel is already bandwidth-bound. `bench_tc_gemv.cu` reruns it (`nvcc -arch=sm_75 -O3`).
- **2026-10-03-community-2x-mi50.** All 12,288 Coder experts in VRAM across two MI50s: decode 50.1 / 47.8 / 45.7,
  prompt 321 / 523 / 517 tokens/s; 6/6 needles; `STRATA_ARENA_MMAP=1` with `--pcie-frac 0`. Uses the 5090 report's
  `benchmark.py` unchanged.
- **2026-10-03-v100-prompt-attn.** `prompt_attn_v70_kernel` (`mma.sync.m8n8k4`): prompts +15-18% (e.g. 28,650 tokens:
  1,063 -> 1,251 tokens/s); `qsa_prompt_attn_parity` error vs FP64 5e-6; needles 15/15 either way; the distribution
  shift is within two controls that only reorder rounding.

### Data formats

- `matrix.json` (2026-09-24-final, -coder, -speed-0114/0122/0126): a list of runs, each with `model`, `tier`
  (`1k` ... `262k`), `prompt_tokens`, `prefill_tok_s`, `ttft_s`, `decode_tok_s`, `decoded`, `spec_accept`,
  `tokens_per_round`, `vram_slots`, `wall_s`, `exit`. The harness that wrote them is not in the snapshot.
- `vision.json`: per model, `off` / `on` blocks and `flag_identical` / `on_identical`.
- Community reports: `results.json` (per run, with request hashes and output text), `summary.json` (aggregates),
  `telemetry.jsonl` (one host sample per second), `needles.json` (`tools/needle_bench.py --out`).
- `docs/benchmarks/2026-09-29-gfx1100.json`: `source_commit`, `date`, binary SHA-256s, a `note`, and `control` /
  `optimized` lists of `warmup` / `cold` / `followup` requests with `prompt_tokens`, `reused`, `fresh`, `prefill_ms`,
  `prefill_tps`, `generated`, `decode_ms`, `decode_tps`. It is the source of the table at
  `docs/AMD_HIP_PERFORMANCE.md:67-95`: RX 7900 XTX 24 GiB (gfx1100), Ryzen 9 7950X3D, 64 GiB, ROCm 7.1, 272 W cap,
  fresh-prompt prefill 1.80-2.02x the existing HIP runtime, revision `9568e78`.

---

## 3. Community reports: `docs/COMMUNITY_BENCHMARKS.md`

A guide for users who want to publish a measurement. It lists the two community reports above, then asks for:

- **What to record:** hardware (GPU, VRAM, CPU, RAM, storage, PCIe link, power limits), software (OS, commit, engine,
  driver, CUDA or ROCm, release or source build), model (repository, revision, quantization, GGUF names, custom
  packs or profiles with hashes), settings (context, KV type and streaming window, expert cache, low-RAM mode, prefill,
  CPU workers, MTP, reasoning, sampling, vision, calibration, speed projection) and workload.
- **How to measure:** at least three runs per configuration with a median and range; separate freshly read prompts
  from reused prefixes; keep prompt throughput, decode throughput, time to first token, total latency and memory
  apart; never divide generated tokens by total request time; failures stay in the report.
- **Correctness checks:** a small task check, and `tools/needle_bench.py` for long-context recall.
- **Submitting:** a `bench/results/YYYY-MM-DD-community-<hardware>/` folder with a README from the template and the
  per-run data, in a pull request separate from engine changes.

---

## 4. Benchmark programs (no `bench/micro/` in the snapshot)

The CMake file names 20 programs under `bench/micro/` (the 16 llama.cpp-oracle `native_*_parity`, `native_mmvq_multi`,
`graph_registry_test`, `pinned_capture`, `hit_cpu_order_parity`, and the `ple_in.bin` / `ple_out.bin` fixtures); none is
present. What remains in the tree:

| Program | Where | What it times |
| --- | --- | --- |
| `fused_gr_bench` | `src/kernels/fused_gr_bench.cpp` (116 lines), built with any GPU backend | the fused hyper-connection read |
| `native_expert_bench` | `src/kernels/native_expert_bench.cpp` (170) | native expert kernels |
| `qsa_select_bench` | `src/kernels/qsa_select_bench.cpp` (174), HIP CTest entry | QSA block scores and top-k, with an FP64 accuracy gate |
| `--bench` modes | `iq_multi_parity`, `native_grouped_parity`, `s2_expert_grouped_parity`, `gdn_rec_parity`, `gemm_bf16_parity`, `decode_cluster_parity`, `expert_multi_test` | old vs new kernel timings beside the bitwise check |
| SYCL benches | `sycl/src/kernels/{mmvq,mmvq_sg,q6k_align,gr,xmx_gemm}_bench.cpp` | the Intel port's kernel timings ([../sycl/README.md](../sycl/README.md)) |
| `tools/needle_bench.py` | 118 lines | recall through a running server; used by every recent report |
| `tools/calibrate.py` | | `START-HERE.bat --calibrate`: measures engine settings on the user's PC and keeps the fastest (NVIDIA) |
| `tools/conversation_restore_speed.py` | 182 | decode speed after a conversation-cache restore vs live |
| `tools/hip/bench_prefill.py`, `tools/hip/tune_hipblaslt.cpp` | | HIP prefill probe; the hipBLASLt tuning tables `tools/hip/gfx*-hipblaslt-*.txt` |
| `bench/results/2026-09-30-tc-gemv-sm75/bench_tc_gemv.cu` | | standalone sm_75 GEMV experiment |

## Gotchas

- Several READMEs name files that were not published with them: `logits-compare.txt` (cache-parity), `needles-q4.*`
  and `tools/kv_precision_compare.py` (kv-q4), "server and engine logs alongside" (rope-scaling), `engine.log`
  (both community reports), the `rib_*` tools (layer split).
- The source cites result sets that are not in `bench/results/`: `2026-09-23-attention-ple`
  (`src/program/generate.cpp:299`), `2026-09-23-mmvq-multi` (`include/strata/spec/controller.hpp:13`),
  `2026-09-23-p2-ssd-direct` (`include/strata/kernels/ngram.hpp:111`), `2026-09-23-p7-int-audit`
  (`include/strata/kernels/kv_q8.hpp:4`), `2026-09-23-ngram-io` (`include/strata/ngram/ple_reader.hpp:15`),
  `2026-09-27-spec` (`src/program/generate.cpp:1776`, `src/spec/draft_policy.cpp:11`), and by name only
  `2026-09-23-suffix-lookup`, `-spec-economics`, `-cpu-expert-multi`.
- Output speed moves several percent with the text, because speculative decoding accepts more or fewer drafts
  (`spec_accept`); the READMEs warn against comparing single runs.
- The README's headline rows mix engine versions: Q2_0 at 0.1.36, the others at 0.1.26 (`README.md:46`).

## In UnbiasedStrata

Removed: all of `bench/` (171 files), `docs/COMMUNITY_BENCHMARKS.md` and `docs/benchmarks/` in 8894a81 ("Remove
installer, Python, non-target platforms, benchmarks and marketing"). The fork keeps upstream's `docs/DETAILS.md` (with
its speed tables) as [../../upstream/DETAILS.md](../../upstream/DETAILS.md) and records its own measurements in its
README and port notes.

## See also

- [prose-docs.md](prose-docs.md) - the documents that quote these numbers.
- [testing.md](testing.md) - the parity tests the benchmarks lean on.
- [../hardware/README.md](../hardware/README.md) - per-GPU behaviour behind the community results.
- [../prefill/README.md](../prefill/README.md) - the prompt path measured in 2026-09-28-prefill-speed.
