[Upstream docs](../../README.md) › [Kernels](../README.md) › Parity tests

# Kernel parity, bench and unit tests (`src/kernels/*.cpp`)

The engine's correctness rests on parity tests: every GPU kernel is held to a reference that was itself validated
against ggml / llama.cpp or a numpy oracle, usually with a **negative control** (the plausible wrong reading is
computed too and must differ by a stated margin before the kernel is judged). The tests are standalone executables at
the top level of `src/kernels/`: one `*_parity.cpp`, `*_test.cpp` or `*_bench.cpp` per kernel family, plus two
generated oracle tables (`*.inc`). The top-level `CMakeLists.txt` builds them and registers most with CTest. They call
the kernels in `strata_kernels` ([../cuda/README.md](../cuda/README.md)) and `strata_kernels_cpu`
([../cpu/README.md](../cpu/README.md)), and use `strata/artifact/dequant.hpp`, ggml (from the llama.cpp checkout
the build fetches), `tools/iq_fixture.py` and real model files as references. Most are written for CUDA and also
compile for AMD through the HIP shims ([../hip/README.md](../hip/README.md)); a few take a different branch there.
The CPU-only tests beside the CPU kernels are in [../cpu/tests.md](../cpu/tests.md); the HIP-only tests in
`tests/hip/` are in [../hip/tests.md](../hip/tests.md).

## Contents

| Page | Covers |
|---|---|
| [fixtures.md](fixtures.md) | Build switches, how CTest is wired (Python probe, fixture test, working directories, environment), every input file and the `.inc` oracle vectors, exit-code conventions, how to run |

## Source map

| File | Lines | Role |
|---|---|---|
| [src/kernels/bf16_bits_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/bf16_bits_test.cpp) | 80 | f32 -> bf16 rounding vs ggml over all 2^32 inputs (CPU) |
| [src/kernels/bf16_gemv_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/bf16_gemv_parity.cpp) | 148 | BF16 GEMV (naive and split) |
| [src/kernels/cvec_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cvec_parity.cpp) | 223 | Control-vector kernel |
| [src/kernels/decode_cluster_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/decode_cluster_parity.cpp) | 424 | sm_90 cluster top-k / argmax vs the one-CTA kernels |
| [src/kernels/dequant_bf16_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/dequant_bf16_test.cpp) | 108 | Device dequantizers vs CPU on real shards (manual) |
| [src/kernels/dequant_s2_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/dequant_s2_parity.cpp) | 143 | S2 (canonical Q2_0) decode |
| [src/kernels/elementwise_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/elementwise_parity.cpp) | 475 | Layer glue (softplus, gates, SiLU, ...) |
| [src/kernels/fused_gr_bench.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/fused_gr_bench.cpp) | 116 | gfx906 fast GR kernels vs old, bitwise + timing (manual) |
| [src/kernels/gdn_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/gdn_parity.cpp) | 397 | Gated DeltaNet recurrence, conv, norms |
| [src/kernels/gr_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/gr_parity.cpp) | 770 | Gated residual / hyper-connection read and write |
| [src/kernels/iq_multi_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/iq_multi_parity.cpp) | 414 | Decode-once IQ kernels vs per-column ones |
| [src/kernels/iq_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/iq_parity.cpp) | 125 | i-quant dequant and MMVQ vs gguf-py fixtures |
| [src/kernels/kv_hybrid_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/kv_hybrid_parity.cpp) | 361 | K8V4 hybrid KV |
| [src/kernels/kv_q4_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/kv_q4_parity.cpp) | 346 | Hadamard-rotated Q4_0 KV |
| [src/kernels/kv_q8_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/kv_q8_parity.cpp) | 141 | INT8 KV |
| [src/kernels/kv_stream_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/kv_stream_parity.cpp) | 234 | Streamed vs resident KV |
| [src/kernels/mmvq_multi_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/mmvq_multi_parity.cpp) | 263 | `native_mmvq` multi-column exactness |
| [src/kernels/native_expert_bench.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/native_expert_bench.cpp) | 170 | Grouped native experts, one `STRATA_EXP_MODE` vs another on real rows (manual) |
| [src/kernels/native_expert_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/native_expert_parity.cpp) | 599 | One native expert: float reference vs CPU vs GPU |
| [src/kernels/native_grouped_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/native_grouped_parity.cpp) | 351 | `native_expert_grouped` group stride and fused SwiGLU + q8_1 vs v1 |
| [src/kernels/ple_fp8_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/ple_fp8_parity.cpp) | 84 | FP8 PLE table through `PleTable` (manual) |
| [src/kernels/ple_oracle_vectors.inc](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/ple_oracle_vectors.inc) | 105 | Generated hash / table oracle vectors for `ple_parity` |
| [src/kernels/ple_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/ple_parity.cpp) | 1184 | PLE hash, table read, PLE block |
| [src/kernels/ple_q5_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/ple_q5_parity.cpp) | 59 | Q5_0 PLE rows vs ggml (manual) |
| [src/kernels/q5_projection_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/q5_projection_parity.cpp) | 62 | Q5_1 shared-expert projection vs ggml (manual) |
| [src/kernels/qsa_oracle_vectors.inc](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/qsa_oracle_vectors.inc) | 134 | Generated f32 -> f16 bit pairs for `qsa_parity` |
| [src/kernels/qsa_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/qsa_parity.cpp) | 1555 | QSA cache, indexer, selection, attention |
| [src/kernels/qsa_prompt_attn_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/qsa_prompt_attn_parity.cpp) | 274 | Tensor-core (CUDA) / WMMA (RDNA4) prompt attention vs FP32 kernel and FP64 |
| [src/kernels/qsa_select_bench.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/qsa_select_bench.cpp) | 174 | Prompt-path QSA selection timing + accuracy gate (HIP ctest) |
| [src/kernels/qsa_topk_active_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/qsa_topk_active_parity.cpp) | 95 | Active-block-bounded top-k vs the capacity rule |
| [src/kernels/qsa_topk_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/qsa_topk_parity.cpp) | 139 | Dispatched QSA top-k vs the reference kernel |
| [src/kernels/quantize_act_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/quantize_act_parity.cpp) | 407 | Q8_0 activation quantizer bytes |
| [src/kernels/rope_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/rope_parity.cpp) | 442 | NEOX partial RoPE, scaled tables |
| [src/kernels/route_window_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/route_window_parity.cpp) | 75 | Batched verify-window routing vs per-token |
| [src/kernels/router_top10_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/router_top10_parity.cpp) | 167 | MoE router (512 experts, top-10) |
| [src/kernels/s2_expert_grouped_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/s2_expert_grouped_parity.cpp) | 679 | Rewritten S2 expert kernels vs previous |
| [src/kernels/s2_gemv_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/s2_gemv_parity.cpp) | 173 | S2 GEMV |
| [src/kernels/s2_gemv_q8_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/s2_gemv_q8_parity.cpp) | 181 | S2 GEMV with Q8_0 activations |
| [src/kernels/s_gemv_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/s_gemv_parity.cpp) | 634 | S2 / S4 / S8 GEMV (Q2_0, Q4_0, IQ4_NL, Q8_0) |
| [src/kernels/s_gemv_q8k_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/s_gemv_q8k_parity.cpp) | 343 | S-GEMV with Q8_K activations |
| [src/kernels/sampler_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/sampler_parity.cpp) | 1284 | Sampler chain order and its three paths |
| [src/kernels/shared_expert_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/shared_expert_parity.cpp) | 482 | Shared expert structure (SiLU on gate, scalar gate) |

## Conditions used in the table

- **GPU**: built inside `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` (`CMakeLists.txt:248-885`, and the native-expert
  block's `:1145`) with no `STRATA_BUILD_TESTS` guard. That block calls `enable_testing()` (`:424`, because
  `tests/core/file_expert_source_test.cpp` exists; `:500` in a HIP build), so the ctests are registered in every GPU
  build. `STRATA_HIP_GFX906` builds count as CUDA (it forces `STRATA_ENABLE_CUDA=ON`, `:65-70`).
- **CUDA**: additionally `if(STRATA_ENABLE_CUDA)`; **HIP**: `if(STRATA_ENABLE_HIP)` (the wave32 build).
- **TESTS**: `STRATA_BUILD_TESTS=ON` (default OFF in the published source: `tests/CMakeLists.txt` and `bench/micro/`
  are absent, `:87-92`).
- **NATIVE**: inside `if(STRATA_NATIVE_EXPERTS)` (default ON).
- **py**: a python3 that can `import numpy` was found at configure time (`_iq_python`, `:597-635`).
- "manual": an executable with no `add_test`; run it by hand.

## Tests at the top level of `src/kernels/`

| Test (CTest name, command) | CMake lines | Build condition | What it checks | Reference | Data |
|---|---|---|---|---|---|
| `bf16_bits_test` | `:912-925` | TESTS or `STRATA_BUILD_CONVERSATION_TESTS` (CPU, header-only; no GPU backend needed) | `bf16_bits.hpp` f32 -> bf16 equals ggml's rule on every f32 pattern; unchanged on non-NaN vs the old rule; NaN stays a quiet NaN of the same sign; every non-NaN bf16 survives a round trip | `ggml_compute_fp32_to_bf16` transcribed | None |
| `bf16_gemv_parity --selftest` | `:751-754` | GPU | Naive and split BF16 GEMV vs a host dot on the same bf16 bits; naive vs split; an fp16 activation (the rival) must differ by more than the tolerance | Host double | Synthetic |
| `cvec_parity` | `:812-815` | GPU | Control vector: project (`h - s(h.v)v`), add, switched-off is bitwise the fused GR write, layers without a direction untouched | Host double | Synthetic |
| `decode_cluster_parity --selftest` | `:705-712` | CUDA | `qsa_block_topk_cluster` and `sample_greedy_cluster` bitwise vs the one-CTA kernels, adversarial scores (ties, NaN, +-inf, -0, denormals), up to 262,144 cells, graph replay; prints SKIP and exits 0 below sm_90; `--bench` | `qsa_block_topk`, `qsa_block_topk_ref`, greedy sampler (via `STRATA_QSA_CLUSTER=0`, `STRATA_ARGMAX_MULTI=0`) | Synthetic |
| `dequant_bf16_test SHARD1 [SHARD2]` (manual) | `:1202-1207` | TESTS and GPU | Device dequantizers (FP32 path rel 1e-6, BF16 within half a BF16 ulp) on the first 4 rows of one real tensor per type | `strata/artifact/dequant.hpp` | **Real model shards** |
| `dequant_s2_parity --selftest` | `:577-583` | GPU | GPU canonical S2 decode bit-exact (`--blocks N`) | `strata::dequantize_q2_0` (cross-checked against ggml upstream) | Synthetic |
| `elementwise_parity --selftest` | `:746-749` | GPU | softplus large-x branch, negative `ssm_a` decay, SiLU in double, other glue conventions; rivals asserted observable | Host transcriptions of `ref/gdn.py` rules | Synthetic |
| `fused_gr_bench [iters] [Tmin] [Tmax]` (manual) | `:677-679` | GPU | `fused_gr_read_multi`'s fast kernels (`STRATA_GR_FAST`) bitwise vs the old ones on every output, plus timings. The fast kernels exist only in gfx906 builds (`src/kernels/cuda/fused_gr.cu:1045-1053`) | The previous kernels | Synthetic |
| `gdn_parity --selftest` | `:685-688` | GPU | GDN: modulo head pairing (`h % h_k`), decay before the rank-1 update, conv slide direction, `l2_norm` eps on the squared norm, the (S, h_v, S) state layout | Host reference of `ref/gdn.py` rules | Synthetic |
| `gr_parity --selftest` | `:680-683` | GPU | Hyper-connection read / write: per-stream RMSNorm, `/hc` inside SiLU, SiLU vs sigmoid placement, mean not sum, BF16 vs FP32 activation contracts with graph replay, the `2*sigmoid` gate | Host reference of `ref/gr.py` rules | Synthetic |
| `iq_multi_parity` | `:648-653` | GPU | `iq_mmvq` (1..8 and 11 columns) and `native_expert_grouped` (0..11 entries, scattered destinations) bitwise new vs old; a gross-error guard vs double; `--bench` | Old kernels (`iq_set_old_kernels`) | Synthetic blocks |
| `iq_parity <dir>` (+ fixture test `iq_parity_fixtures`) | `:590-635` | GPU; ctests only with py | i-quant dequant vs gguf-py (measured rel error; 0 on the fixtures); MMVQ vs float matvec for 1..8 columns, each column of a multi-column call bitwise the one-column call | gguf-py (Q2_0 via `tools/gguf_writer.py`) | `tools/iq_fixture.py --out <build>/iq_fixture` (exit 3 = skip) |
| `kv_hybrid_parity` | `:802-806` | GPU | K8V4: INT8 K codes / scales and rotated Q4_0 V bitwise; composed gathers; `qsa_decode_attn<3>` + inverse FWHT vs host attention; on HIP the prompt-attention step is expected to be refused (`kv_hybrid_parity.cpp:332-334`) | Host references (`kv_q8.hpp`, `kv_q4.hpp` rules) | Synthetic |
| `kv_q4_parity` | `:807-811` | GPU | FWHT-256 vs the exact orthonormal Hadamard; Q4_0 packing bitwise; paged append / gather; dot products invariant under rotation (PR #21) | Host FWHT | Synthetic |
| `kv_q8_parity` | `:790-794` | GPU | INT8 codes + fp16 scales bitwise; gather bitwise; vs the FP16 path within 0.624 quantization steps | Host reference | Synthetic |
| `kv_stream_parity` | `:796-800` | GPU | Streamed attention bitwise vs resident; residency map consistent; counters; ring restore (MTP layout); INT8, FP16, Q4_0 pools | Fully resident pool | Synthetic |
| `mmvq_multi_parity` (+ `mmvq_multi_parity_cuda_layout` with `STRATA_MMVQ_WAVE=0` on gfx906) | `:636-647` | CUDA | `native_mmvq` multi_exact: a T-column call bitwise T single-column calls on the same Q8_1 bytes; negative control (multi_exact off) asserted for T > 4. On gfx906's wave64 layout the control has no other layout to find, hence the second ctest | Single-column calls | Synthetic, finite scales |
| `native_expert_bench <shard1.gguf> <layers> <groups> <tok> <ref mode> <mode> [iters]` (manual) | `:1149-1151` | NATIVE and GPU | Grouped native experts under two `STRATA_EXP_MODE` layouts (the gfx906 AMD layouts), bitwise, with timings | Reference mode | **Real GGUF rows** |
| `native_expert_parity_q4_K_q5_1`, `_q4_K_q8_0`, `_q5_K_q8_0`, `_q8_0_q8_0`, `_q4_K_q5_0`, `_q5_0_q8_0` (`--synthetic GU/DOWN`), `native_expert_parity_q5_1_min`, `native_expert_parity_bf16_embd`, `native_expert_parity_refuses_q6_K` (`WILL_FAIL`) | `:1145-1172` | NATIVE and GPU builds the target; the ctests need TESTS and CUDA | One native expert three ways at 2560 x 640: (a) float reference from ggml `to_float`, (b) CPU path (and the Strata multi-token kernels: `iq512_rows` / `iq256_rows`, `kq256_*`, `q2_rows_any`), (c) GPU `native_expert_grouped` (both kernel variants bitwise); (d) GPU dequantizers and the BF16 embedding gather vs `to_float`. Sets `STRATA_IQ_MT_MIN=1` unless given (`native_expert_parity.cpp:537-543`) | ggml `to_float` + float SwiGLU | Synthetic ggml-quantized weights for the ctests; real shards by hand (`native_expert_parity <shard.gguf> [layer ...]`) |
| `native_grouped_parity` | `:666-671` | GPU | `native_expert_grouped` group stride (`grid_groups`) and fused SwiGLU + q8_1 bitwise vs v1, rows it must not write included; `--bench` | v1 launches (`native_grouped_set_v1`) | Synthetic |
| `ple_fp8_parity <ple-fp8.gguf> <reference.bin> [<iq4nl shard>]` (manual) | `:1180-1182` | NATIVE, CUDA and TESTS | FP8 PLE rows bit for bit through Direct, Mmap and `gather_batch`; correlation (~0.997) with the IQ4_NL table (#291) | Values decoded from the checkpoint (torch) | **Real files** |
| `ple_parity --selftest --in ... --out ...` (working dir: source root) | `:773-788` | GPU | A: hash vs `ref/ngram.py` vectors with rivals observable; B: table rows vs numpy over the original GGUF; C: the PLE block vs ggml's graph, per stage; NaN-safe comparisons. `--check-fixtures` validates the inputs on the CPU only | `ple_oracle_vectors.inc`; numpy; a ggml capture | **Required**: the Q2_0 model's PLE shard, `pack/full/dense.bin`, `ple_in.bin` / `ple_out.bin` (see [fixtures.md](fixtures.md)); missing input = exit 2 |
| `ple_q5_parity` (manual) | `:1173-1176` | NATIVE, CUDA and TESTS | Q5_0 PLE rows of a real artifact vs ggml's dequantizer (OrcaRouter Q4_K_S, #296) | ggml | **Real file** |
| `q5_projection_parity` (manual) | `:1177-1179` | NATIVE, CUDA and TESTS | Q5_1 shared-expert projection: Strata CUDA dequantization vs ggml (#296) | ggml | **Real file** |
| `qsa_parity --selftest` | `:761-764` | GPU | QSA: pooled key rotated at the block's first cell, pool-then-norm, spare slot from cell 0, per-head ReLU, `q / 12` head map, only selected cells attended, gate from the second half, paged KV with reversed page tables | Host reference of `ref/qsa.py` / `ref/model.py` rules; `qsa_oracle_vectors.inc` for f16 rounding | Synthetic |
| `qsa_prompt_attn_parity [ctx] [queries] [reps]` (manual on CUDA); HIP: ctest `hip_prompt_attn_wmma 32768 2048 3` | CUDA `:337-343`; HIP `:552-557` | CUDA: (TESTS or `STRATA_PARITY_PROMPT_ATTN`); HIP: every HIP build | Tensor-core prompt attention vs `qsa_decode_attn_batch` (rel 1e-4) and FP64 (error no worse than a small multiple of the old kernel's), INT8 / FP16 / Q4_0 pools; timing. HIP: the RDNA4 WMMA kernel (int8 KV only), skip (77) off gfx12 | FP32 kernel, FP64 host | Synthetic |
| `qsa_select_bench 32768 256 3 262144` | `:353-361` | (TESTS or `STRATA_PARITY_PROMPT_ATTN`) and HIP | Prompt-path QSA selection per stage, scorer vs scorer, FP64 accuracy gate; non-zero exit on a failed gate | FP64 host | Synthetic |
| `qsa_topk_active_parity` | `:344-352` | TESTS and CUDA | Top-k with the active-block bound (Turing's dispatch, forced via `STRATA_TOPK_ACTIVE_ANY=1`) selects the same ids as the capacity rule at capacity 262,144 (#512); exit 77 without a device | Capacity-rule dispatch | Synthetic |
| `qsa_topk_parity --selftest` | `:766-771` | GPU | Dispatched `qsa_block_topk` ids identical to `qsa_block_topk_ref` across the register kernel's reach; continuous, 64-level and all-equal scores; tail blocks of 0-3 cells | Reference kernel | Synthetic |
| `quantize_act_parity --selftest` | `:719-722` | GPU | Q8_0 activation quantizer compared on the 34-byte blocks, not values | Host transcription of `ref/quant.py::q8_0` with `f16_bits.hpp` conversions | Synthetic |
| `rope_parity --selftest` | `:714-717` | GPU | RoPE table vs float64; rotation bit-exact vs the host NEOX rotation with the same table; scaled tables (linear, YaRN) vs a float64 `rope_yarn` transcription; table path vs native fast-math path | `ref/qsa.py` formulas in double | Synthetic |
| `route_window_parity` | `:724-728` | GPU | `moe_route_window` (one multi-column BF16 projection + one n-token top-k) bitwise vs per-token calls, router (2560 -> 256, top-10) and shared-gate (2560 -> 1) shapes | Per-token kernels | Synthetic |
| `router_top10_parity --selftest` | `:730-733` | GPU | Expert ids exactly and weights to tolerance at 512 experts / top-10, ties and near-ties; the 2^-14 clamp proven unreachable for this geometry | Host `ref/moe.py::router` transcription | Synthetic |
| `s2_expert_grouped_parity --selftest` | `:817-822` | GPU | Every rewritten S2 expert entry point bitwise vs `moe_grouped_select_old(1)`, output and scratch; fallbacks on misaligned inputs verified via `moe_grouped_last_path`; double-precision sanity check; `--bench` | Previous kernels | Synthetic Q2_0 blobs |
| `s2_gemv_parity --selftest` | `:585-588` | GPU | S2 GEMV relative error (measured; FP16 tolerance 1e-3) with full-mantissa fp16 scales | Dot on `strata::dequantize_q2_0` | Synthetic |
| `s2_gemv_q8_parity --selftest` | `:690-693` | GPU | Q8_0-activation GEMV vs host; FP16 vs Q8_0 activations must differ (~1%); the gap is reported | Host `ref/quant.py::q8_0` rule | Synthetic |
| `s_gemv_parity --selftest` | `:741-744` | GPU | S2 Q2_0, S4 Q4_0 and IQ4_NL, S8 Q8_0 GEMV; the offset branch (Q4_K / Q5_K) is reported as not exercised | Raw GGUF block rebuilt from the canonical planes, decoded by `strata/artifact/dequant.hpp` | Synthetic |
| `s_gemv_q8k_parity --selftest` | `:756-759` | GPU | Q8_K activation path of `s_gemv` vs double on the same bytes; the Q8_K vs FP16 gap measured | Host double; `quantize_q8_K` | Synthetic |
| `sampler_parity --selftest`, `sampler_parity_one_block`, `sampler_parity_old` | `:695-704` | GPU | The order top_k -> top_p -> temperature is observable (the wrong order must pick differently) and the kernel follows it; the same fixtures on the one-block (`STRATA_SAMPLER_ONE_BLOCK=1`) and 0.1.20 (`STRATA_OLD_SAMPLER=1`) paths | Host reference chain | Synthetic |
| `shared_expert_parity --selftest` | `:672-675` | GPU | SiLU on gate (not up), scalar `ffn_gate_inp_shexp` gate (not elementwise or per-expert), each rival required to differ; tolerance set by the fp16 intermediate | Host reference | Synthetic |

## Related tests elsewhere

| Test | CMake lines | Where documented |
|---|---|---|
| `src/prefill/gdn_rec_parity.cu`, `src/prefill/gemm_bf16_parity.cu` (CUDA, not gfx906) | `:654-665` | The prompt-path area |
| `tests/cuda/prefill_fused_moe_test.cpp`, `prefill_fused_iq_test.cpp`, `prefill_mmq_kquant_test.cpp` (NATIVE, CUDA, TESTS) | `:1081-1099` | The prompt-path area |
| `tests/core/*.cpp` (expert layout, expert cache, profile, file expert source, gguf split) | `:225-234`, `:423-459` | The core area; `expert_layout_test` also in [../cpu/expert-layout.md](../cpu/expert-layout.md) |
| `tests/hip/*.cpp` and `tools/hip/tune_hipblaslt.cpp` | `:487-575`, `:1132-1144` | [../hip/tests.md](../hip/tests.md), [../hip/tools.md](../hip/tools.md) |
| `ple_reader_selftest`, `platform_memory_test`, `pinned_shared_test`, `cuda_device_selftest` / `hip_device_selftest` | `:832-856` | The core / platform areas |
| `bench/micro/*` targets (`native_*_parity` against a separate llama.cpp build, `native_mmvq_multi`, `graph_registry_test`, `pinned_capture`, `hit_cpu_order_parity`) | `:387-410`, `:824-878`, `:1186-1189` | Guarded by `EXISTS`; `bench/micro/` is not in the published source |

## Key concepts

- **The reference is a chain, not a second opinion.** Kernels are compared against code already validated against
  ggml (`strata/artifact/dequant.hpp`, `quantize_q8_K`, ggml's own `to_float`), or against generated oracle vectors,
  not against a second copy of themselves (`CMakeLists.txt:286-289`).
- **Negative controls.** Every structural reading with a plausible rival is computed the wrong way and required to
  differ before the kernel is judged; a test whose control has no power says so (for example
  `mmvq_multi_parity.cpp`'s T <= 4 cases).
- **Bitwise where the contract is bitwise.** Multi-token and rewritten kernels must equal their single-token or
  previous forms bit for bit (the verify window reproducing greedy decode depends on it); tolerances are used only
  where the arithmetic order legitimately differs, and the measured error is printed.
- **NaN-safe comparisons** (`ple_parity.cpp`): `x <= t` silently passes NaN, so comparisons require finiteness first.
- **Exit codes**: 0 pass, 1 fail, 2 missing required input or bad usage (a failure), 77 skip where CTest's
  `SKIP_RETURN_CODE 77` is set, 3 for the i-quant fixture generator without numpy / gguf-py. Details in
  [fixtures.md](fixtures.md).

## Model- and platform-specific notes

- All synthetic fixtures use the Flash-Next geometry (2560 hidden, 640 expert width, 512 experts top-10, 24 query /
  2 KV heads); none covers another model shape.
- `native_expert_parity`'s ctests cover the K-quant / Q5_x / Q8_0 pairs of Unsloth UD-Q4_K_XL and plain `_M`
  quantizations (`CMakeLists.txt:1153-1159`); the i-quant formats are covered by `iq_parity`, `iq_multi_parity` and
  `native_grouped_parity` on the GPU side.
- HIP builds run every GPU test above; `hip_prompt_attn_wmma` replaces the CUDA prompt-attention parity, and
  `qsa_select_bench` is HIP-only. gfx906 builds take the CUDA branches plus `mmvq_multi_parity_cuda_layout`.

## In UnbiasedStrata

| Kept unchanged | Changed (commit e41dfc7, the Qwen3.6-35B-A3B port) | Added by the fork | Removed |
|---|---|---|---|
| 33 of the 40 test programs and both `.inc` tables | `gdn_parity` (re-pinned at 32 value heads with the SiLU output gate), `router_top10_parity` (also 256 experts / top-8), `shared_expert_parity` (also the 2048 x 512 shared expert), `native_expert_parity` (`--geom HxFF`, Qwen3.6 pairs), `native_grouped_parity` (`--bench-down` at 2048 x 512), `iq_multi_parity` (an IQ3_XXS down case at Qwen3.6's width), `kv_hybrid_parity` (the sub-sm_70 refusal counted as a pass) | `qsa_dense_parity.cpp`, `qwen36_seam_parity.cpp`, `qsa_attn_bench.cpp` | None in `src/kernels/`; the HIP tests in `tests/hip/` were removed (8894a81) |

The fork's pages: [../../../codebase/kernels/parity/README.md](../../../codebase/kernels/parity/README.md).

## See also

- [fixtures.md](fixtures.md), [../cpu/tests.md](../cpu/tests.md), [../hip/tests.md](../hip/tests.md)
- [../../build/targets.md](../../build/targets.md), [../../build/cmake-options.md](../../build/cmake-options.md)
