[Upstream docs](../../README.md) › [Kernels](../README.md) › CUDA

# CUDA kernels (`src/kernels/cuda/`, `include/strata/kernels/`)

The CUDA kernels do the GPU part of every token: the hyper-connection residual, the Gated DeltaNet and QSA
attention layers, the router, the experts held in VRAM, the shared expert, the output head's GEMV and the sampler.
They are 44 `.cu` files, built into one static library, `strata_kernels` (`CMakeLists.txt:286-336`), together with
three host files (`src/kernels/ngram.cpp`, `src/ngram/ple_reader.cpp`, `src/platform/direct_file.cpp`). Every entry
point is a plain host function in `namespace strata::kernels`, declared in a header under `include/strata/kernels/`.
It takes device pointers and a `void* stream` (a `cudaStream_t` hidden behind `void*`, so the headers need no CUDA
types). The callers are the layer and session code in `src/core/` (`layer.cpp`, `verify.cpp`, `mtp.cpp`,
`session.cpp`, `peer_experts.cpp`, `remote_experts.cpp`, `native_dense.cpp`, `native_head.cpp`), the prompt path in
`src/prefill/` and the switches in `src/program/generate.cpp`. The kernels call only the CUDA runtime and ggml's block
definitions (`third_party/ggml/ggml-common.h`, added as an include directory at `CMakeLists.txt:363`). The same `.cu`
files are also compiled as HIP for AMD cards (two backends, see "Platforms" below), and the Intel port in `sycl/` has a
translated copy of each file (`sycl/src/kernels/cuda/*.dp.cpp`, see [SYCL](../../sycl/README.md)). Terms such as QSA,
GDN, MTP and PLE are defined in the [glossary](../../guides/glossary.md).

## Contents

| Page | Covers |
| --- | --- |
| [attention-qsa.md](attention-qsa.md) | The QSA full-attention layers: paged FP16 KV cache, the indexer (key pooling, block scores, top-k, including the sm_90+ cluster top-k), split-K decode attention, the tensor-core prompt attention (with the Volta and RDNA4 WMMA variants), and the pinned `native_qsa*` / `native_flash_attn` adapters |
| [gdn.md](gdn.md) | Gated DeltaNet layers: conv + SiLU + L2 norm, gate and beta, the delta-rule recurrence and the sigmoid-gated output norm, in reference, fused and pinned forms, plus the verify/commit kernels for the speculative window |
| [hyper-connections-ple.md](hyper-connections-ple.md) | The hyper-connection residual read/write (`gr`, `fused_gr`, `native_gr_*`), the PLE n-gram embedding block (`ple`, `native_ple_postops`) and control vectors (`cvec`) |
| [norms-rope-elementwise.md](norms-rope-elementwise.md) | RoPE (float64 tables, the analytic pinned kernel, M-RoPE, YaRN/linear scaling), the elementwise glue, weighted RMSNorm, the doorbell kernels and mapped-memory copies |
| [moe-and-experts.md](moe-and-experts.md) | The router (`router_top10`, `native_router`), the MoE combine, the shared expert, the S2 expert tier (`s2_expert_grouped`), the native GGUF expert kernels (`iq_kernels`) and the pinned llama.cpp MMVQ (`native_mmvq`) |
| [gemv-and-quant.md](gemv-and-quant.md) | GEMVs over the canonical S2/S4/S8 forms and over BF16, the pinned BF16 MMVF, activation quantizers (Q8_0, Q8_K) and GGUF dequantizers |
| [kv-cache.md](kv-cache.md) | KV formats beyond FP16 (INT8, Hadamard-rotated Q4_0, hybrid K8V4) and KV streaming from pinned host memory |
| [sampling-and-verify.md](sampling-and-verify.md) | The sampler (greedy argmax, the sm_90+ cluster argmax, the sampled chain, coupled draft sampling) and the verify-window / MTP helper kernels |
| [bits-and-arch-shims.md](bits-and-arch-shims.md) | `bf16_bits.hpp`, `f16_bits.hpp` (bit-exact conversions), `dp4a.hpp` (pre-sm_61 / pre-sm_70 fallbacks) and `q8_1_finite.hpp` |

## Source map

Line counts are from the pinned commit.

### Kernels (`src/kernels/cuda/`)

| File | Lines | Role | Page |
| --- | ---: | --- | --- |
| [bf16_gemv.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/bf16_gemv.cu) | 141 | BF16 weight x BF16 activation GEMV (naive, warp per row, split) | [gemv](gemv-and-quant.md) |
| [cvec.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/cvec.cu) | 166 | Control vector on the residual streams (project or add) | [hc/ple](hyper-connections-ple.md) |
| [dequant_bf16.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/dequant_bf16.cu) | 261 | GGUF blocks to BF16 / FP16 / FP32 for the prompt path's GEMMs | [gemv](gemv-and-quant.md) |
| [dequant_s2.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/dequant_s2.cu) | 51 | Naive S2 decode (reference) | [gemv](gemv-and-quant.md) |
| [elementwise.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/elementwise.cu) | 397 | Glue ops, embedding gather, weighted RMSNorm, doorbells, mapped-memory copies | [norms](norms-rope-elementwise.md) |
| [fused_gdn.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/fused_gdn.cu) | 166 | GDN conv+L2, alpha/beta, step+output norm, one kernel each | [gdn](gdn.md) |
| [fused_gr.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/fused_gr.cu) | 1371 | Two-kernel hyper-connection read, multi-token variants, per-card self-check, gfx906 fast path | [hc/ple](hyper-connections-ple.md) |
| [gdn.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/gdn.cu) | 248 | Reference GDN step, conv, L2 norm, beta gate, output norm | [gdn](gdn.md) |
| [gr.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/gr.cu) | 432 | Hyper-connection read/write, unfused | [hc/ple](hyper-connections-ple.md) |
| [iq_kernels.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/iq_kernels.cu) | 2578 | i-quant / K-quant MMVQ, q8_1 quantizer, dequantizers, grouped native experts (with gfx906 layouts) | [moe](moe-and-experts.md) |
| [kv_q4.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/kv_q4.cu) | 231 | 256-point Walsh-Hadamard transform, Q4_0 KV append/gather | [kv](kv-cache.md) |
| [kv_q8.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/kv_q8.cu) | 127 | INT8 KV append/gather | [kv](kv-cache.md) |
| [kv_stream.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/kv_stream.cu) | 276 | KV streaming: residency resolve (CLOCK), ring tables, staging | [kv](kv-cache.md) |
| [native_bf16.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/native_bf16.cu) | 199 | Pinned llama.cpp MMVF (BF16 weights, F32 activations, 1..8 columns) | [gemv](gemv-and-quant.md) |
| [native_flash_attn.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/native_flash_attn.cu) | 212 | Diagnostic pinned FlashAttention-vec, context <= 256 | [attention](attention-qsa.md) |
| [native_gdn.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/native_gdn.cu) | 122 | Pinned `gated_delta_net` recurrence | [gdn](gdn.md) |
| [native_gdn_preprocess.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/native_gdn_preprocess.cu) | 199 | Pinned conv+SiLU, L2 norm, beta, gate, output norm | [gdn](gdn.md) |
| [native_gr_norm.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/native_gr_norm.cu) | 102 | Pinned weighted F32 RMSNorm (hyper-connection read) | [hc/ple](hyper-connections-ple.md) |
| [native_gr_postops.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/native_gr_postops.cu) | 115 | Pinned hyper-connection down-SiLU, pre-gated mean, post (write) | [hc/ple](hyper-connections-ple.md) |
| [native_mmvq.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/native_mmvq.cu) | 1619 | Pinned llama.cpp MMVQ (Q2_0, Q4_0, Q5_0, Q8_0, Q3_K..Q6_K, IQ4_NL, IQ4_XS), q8_1 quantizers, type dispatcher, gfx906 wave layout | [moe](moe-and-experts.md) |
| [native_moe.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/native_moe.cu) | 86 | Pinned MoE weighted reduction (combine) | [moe](moe-and-experts.md) |
| [native_ple_postops.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/native_ple_postops.cu) | 221 | Pinned PLE arithmetic after its projections (one token and batch) | [hc/ple](hyper-connections-ple.md) |
| [native_qsa.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/native_qsa.cu) | 127 | Pinned weighted RMSNorm and sigmoid output gate | [attention](attention-qsa.md) |
| [native_qsa_indexer.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/native_qsa_indexer.cu) | 290 | Pinned indexer key cache, pooling, norm and IMRoPE | [attention](attention-qsa.md) |
| [native_qsa_score.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/native_qsa_score.cu) | 213 | Pinned TF32-MMA indexer scores (no engine caller) | [attention](attention-qsa.md) |
| [native_rope.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/native_rope.cu) | 147 | Pinned analytic IMRoPE; M-RoPE table and RoPE-table registry | [norms](norms-rope-elementwise.md) |
| [native_router.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/native_router.cu) | 127 | Pinned `topk-moe` router for 512 experts / top 10 | [moe](moe-and-experts.md) |
| [ple.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/ple.cu) | 386 | PLE block (the n-gram embedding layer) | [hc/ple](hyper-connections-ple.md) |
| [qsa.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/qsa.cu) | 821 | QSA cache append/gather, indexer pooling, per-cell scores, top-k, reference attention, gate | [attention](attention-qsa.md) |
| [qsa_decode_attn.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/qsa_decode_attn.cu) | 537 | Split-K decode attention straight from the KV pools (+ PR #540's pre-sm_75 kernel) | [attention](attention-qsa.md) |
| [qsa_prompt_attn.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/qsa_prompt_attn.cu) | 1479 | Tensor-core prompt attention (mma.sync, cp.async; Volta m8n8k4 and RDNA4 WMMA variants) | [attention](attention-qsa.md) |
| [qsa_select.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/qsa_select.cu) | 1237 | Block scores (warp, multi-query, TF32 tensor core, gfx12 WMMA) and radix top-k (register, wide, cluster) | [attention](attention-qsa.md) |
| [quantize_act.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/quantize_act.cu) | 313 | Q8_0 / Q8_K activation quantizers and their inverses | [gemv](gemv-and-quant.md) |
| [rope.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/rope.cu) | 136 | Host float64 RoPE tables (with scaling), table-driven NEOX rotation | [norms](norms-rope-elementwise.md) |
| [router_top10.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/router_top10.cu) | 497 | Softmax / top-k / renormalise router (with HIP fast and gfx906 wave kernels) | [moe](moe-and-experts.md) |
| [s2_expert_grouped.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/s2_expert_grouped.cu) | 1190 | GPU expert tier for S2 (Q2_0 canonical) packs: per-hit and grouped kernels | [moe](moe-and-experts.md) |
| [s2_gemv.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/s2_gemv.cu) | 63 | Naive S2 GEMV (reference) | [gemv](gemv-and-quant.md) |
| [s2_gemv_fast.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/s2_gemv_fast.cu) | 153 | Experiment kept as a negative result (not faster) | [gemv](gemv-and-quant.md) |
| [s2_gemv_q8.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/s2_gemv_q8.cu) | 96 | S2 GEMV over Q8_0 activations (the canonical pack's dense S2 path) | [gemv](gemv-and-quant.md) |
| [s2_gemv_quads.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/s2_gemv_quads.cu) | 93 | S2 GEMV, four elements per code byte | [gemv](gemv-and-quant.md) |
| [s_gemv.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/s_gemv.cu) | 694 | S-form GEMVs (S2/S4/S8) over FP16, Q8_K and Q8_0 activations | [gemv](gemv-and-quant.md) |
| [sampler.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/sampler.cu) | 1081 | Sampler chain, greedy argmax (cluster on sm_90+), coupled draft sampling | [sampling](sampling-and-verify.md) |
| [shared_expert.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/shared_expert.cu) | 384 | Shared expert (one token and 1..8 tokens) and `moe_combine` | [moe](moe-and-experts.md) |
| [verify_kernels.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/verify_kernels.cu) | 682 | Verify-window GDN multi/commit, MTP helpers, resident expert plans, small copies | [gdn](gdn.md), [sampling](sampling-and-verify.md) |

### Headers (`include/strata/kernels/`, without `cpu/`)

| File | Lines | Role | Page |
| --- | ---: | --- | --- |
| [bf16_bits.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/bf16_bits.hpp) | 61 | Host/device bf16 <-> f32 (round to nearest even, NaN quieting) | [bits](bits-and-arch-shims.md) |
| [bf16_gemv.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/bf16_gemv.hpp) | 56 | BF16 GEMV and pinned MMVF API | [gemv](gemv-and-quant.md) |
| [cvec.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cvec.hpp) | 61 | `Cvec` and the control-vector API | [hc/ple](hyper-connections-ple.md) |
| [dequant_bf16.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/dequant_bf16.hpp) | 27 | GGUF dequantizer API | [gemv](gemv-and-quant.md) |
| [dequant_s2.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/dequant_s2.hpp) | 18 | S2 decode API | [gemv](gemv-and-quant.md) |
| [dp4a.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/dp4a.hpp) | 45 | `STRATA_DP4A`, `strata_spin_pause` (sm_60 / pre-sm_70 fallbacks) | [bits](bits-and-arch-shims.md) |
| [elementwise.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/elementwise.hpp) | 128 | Glue ops and doorbell API | [norms](norms-rope-elementwise.md) |
| [f16_bits.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/f16_bits.hpp) | 104 | Host/device fp16 <-> f32 (RNE, subnormals, saturation) | [bits](bits-and-arch-shims.md) |
| [fused_gdn.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/fused_gdn.hpp) | 35 | Fused GDN API | [gdn](gdn.md) |
| [fused_gr.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/fused_gr.hpp) | 67 | `FusedGrArgs`, fused hyper-connection read API | [hc/ple](hyper-connections-ple.md) |
| [gdn.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/gdn.hpp) | 97 | `GdnShapes`, reference GDN API, state layout | [gdn](gdn.md) |
| [gr.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/gr.hpp) | 122 | `GrShapes`, `GrWorkspace`, hyper-connection read/write API | [hc/ple](hyper-connections-ple.md) |
| [iq_kernels.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/iq_kernels.hpp) | 80 | i-quant API, `NativeExpertLayout`, grouped experts | [moe](moe-and-experts.md) |
| [kv_q4.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/kv_q4.hpp) | 61 | `block_q4_0`, FWHT and Q4_0 KV API | [kv](kv-cache.md) |
| [kv_q8.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/kv_q8.hpp) | 41 | INT8 KV API | [kv](kv-cache.md) |
| [kv_stream.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/kv_stream.hpp) | 104 | `KvHostPools`, `KvStreamMap`, `KvFormat`, streaming API | [kv](kv-cache.md) |
| [mrope.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/mrope.hpp) | 64 | M-RoPE position table, opt-in RoPE angle table | [norms](norms-rope-elementwise.md) |
| [native_flash_attn.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/native_flash_attn.hpp) | 36 | Short-context FlashAttention adapter | [attention](attention-qsa.md) |
| [native_gdn.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/native_gdn.hpp) | 22 | Pinned GDN step and its switch | [gdn](gdn.md) |
| [native_gdn_preprocess.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/native_gdn_preprocess.hpp) | 37 | Pinned GDN pre/post-processing | [gdn](gdn.md) |
| [native_gr_norm.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/native_gr_norm.hpp) | 18 | Pinned weighted RMSNorm | [hc/ple](hyper-connections-ple.md) |
| [native_gr_postops.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/native_gr_postops.hpp) | 28 | Pinned hyper-connection post-ops | [hc/ple](hyper-connections-ple.md) |
| [native_mmvq.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/native_mmvq.hpp) | 115 | Pinned MMVQ API and type ids | [moe](moe-and-experts.md) |
| [native_moe.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/native_moe.hpp) | 21 | Pinned MoE combine | [moe](moe-and-experts.md) |
| [native_ple_postops.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/native_ple_postops.hpp) | 38 | Pinned PLE post-ops | [hc/ple](hyper-connections-ple.md) |
| [native_qsa.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/native_qsa.hpp) | 30 | Pinned QSA norm and gate | [attention](attention-qsa.md) |
| [native_qsa_indexer.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/native_qsa_indexer.hpp) | 46 | Pinned indexer append | [attention](attention-qsa.md) |
| [native_qsa_score.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/native_qsa_score.hpp) | 31 | Pinned indexer score | [attention](attention-qsa.md) |
| [native_rope.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/native_rope.hpp) | 20 | Pinned RoPE | [norms](norms-rope-elementwise.md) |
| [native_router.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/native_router.hpp) | 19 | Pinned 512/top-10 router | [moe](moe-and-experts.md) |
| [ngram.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/ngram.hpp) | 216 | PLE n-gram hash and table reader; host code in `src/kernels/ngram.cpp` (see [CPU kernels](../cpu/README.md)) | [hc/ple](hyper-connections-ple.md) |
| [ple.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/ple.hpp) | 123 | `PleWeights`, `PleOut`, PLE block API | [hc/ple](hyper-connections-ple.md) |
| [q8_1_finite.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/q8_1_finite.hpp) | 33 | Finite clamp of the q8_1 scale and sum (#606) | [bits](bits-and-arch-shims.md) |
| [qsa.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/qsa.hpp) | 369 | `QsaShapes`, `QsaStep`, cache layouts, QSA API | [attention](attention-qsa.md) |
| [qsa_decode_attn.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/qsa_decode_attn.hpp) | 46 | `QsaAttnPools`, split-K decode attention API | [attention](attention-qsa.md) |
| [qsa_prompt_attn.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/qsa_prompt_attn.hpp) | 32 | Tensor-core prompt attention API | [attention](attention-qsa.md) |
| [qsa_select.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/qsa_select.hpp) | 57 | Block scores / top-k API | [attention](attention-qsa.md) |
| [quantize_act.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/quantize_act.hpp) | 58 | Q8_0 / Q8_K quantizer API | [gemv](gemv-and-quant.md) |
| [rope.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/rope.hpp) | 64 | `rope_neox_pair`, table builders, `rope_neox_apply` | [norms](norms-rope-elementwise.md) |
| [rope_scaling.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/rope_scaling.hpp) | 193 | `RopeScaling` (none / linear / YaRN), `rope_scaled_angle` | [norms](norms-rope-elementwise.md) |
| [router_top10.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/router_top10.hpp) | 26 | Router API, HIP test variants | [moe](moe-and-experts.md) |
| [s2_expert_grouped.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/s2_expert_grouped.hpp) | 131 | S2 GPU expert tier API | [moe](moe-and-experts.md) |
| [s2_gemv.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/s2_gemv.hpp) | 25 | S2 GEMV API | [gemv](gemv-and-quant.md) |
| [s2_gemv_q8.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/s2_gemv_q8.hpp) | 22 | S2 x Q8_0 GEMV API | [gemv](gemv-and-quant.md) |
| [s_gemv.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/s_gemv.hpp) | 133 | `SForm`, `Codebook`, S-form GEMV API | [gemv](gemv-and-quant.md) |
| [sampler.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/sampler.hpp) | 86 | `SamplerParams`, sampler API, `penalty_rows` | [sampling](sampling-and-verify.md) |
| [shared_expert.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/shared_expert.hpp) | 103 | `NativeSharedWeights`, shared expert and `moe_combine` API | [moe](moe-and-experts.md) |
| [verify_kernels.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/verify_kernels.hpp) | 98 | Verify-window and MTP helper API, `kVerifyMaxT` | [gdn](gdn.md), [sampling](sampling-and-verify.md) |

## Key concepts

### The model the kernels are written for

Upstream Strata runs one architecture, llama.cpp's `qwen4exp` (Qwen3.8-Flash-Next and its Coder, Swift 1.5 and
Unsloth variants). Many kernels hard-code its geometry, and the headers state it:

| Part | Geometry | Where it is stated |
| --- | --- | --- |
| Layers | 48: 36 GDN and 12 QSA (every 4th layer, indices 3, 7, ..., 47) | `include/strata/core/graph.hpp:45` |
| Residual | hyper-connections: hc = 4 streams of n_embd 2,560, bottleneck rank hc_lr 320 | `gr.hpp:41-46`, `fused_gr.cu:20-23` |
| GDN | S = 128, h_k = 16, h_v = 48, 10,240 conv channels, d_conv 4, gate `sigmoid(z)` | `gdn.hpp:7-11` |
| QSA | 24 query heads, 2 KV heads, head_dim 256, n_rot 64; indexer 4 heads x 128, block 4, top-k 2,048 | `qsa.hpp:94-117` |
| MoE | 512 experts, top 10, n_ff 640, plus a shared expert | `s2_expert_grouped.cu:37-38`, `native_router.hpp` |
| PLE | one n-gram embedding layer feeding the residual stack | `ple.hpp:1-13` |

There is no dense-attention mode: every QSA layer runs the indexer and its top-k selection, and the attention kernels
always read an explicit `ids` selection. The decode and prompt attention kernels are compiled for exactly 12 query
heads per KV head (`qsa_decode_attn.cu:17`, `qsa_prompt_attn.cu:22`). The GDN output gate is always a sigmoid.

### The two kernel families: canonical and "native"

The files named `native_*.cu` reproduce llama.cpp's CUDA backend at commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`
operation for operation (same reduction trees, same `--use_fast_math` arithmetic). CMake compiles those 14 files
with `--use_fast_math` on CUDA (`CMakeLists.txt:365-379`; not on the gfx906 HIP build). Each is enabled by a CLI
switch that `src/program/generate.cpp:2276-2313` applies before any session or graph exists: `--native-gdn`,
`--native-qsa`, `--native-qsa-indexer`, `--native-rope`, `--native-router`, `--native-moe-combine`,
`--native-ple-postops`, `--gr-native-mmvf`, `--native-bf16`, `--native-bf16-extra`, `--native-flash-attn-short`.
`--native SHARD1` turns on all of them except the diagnostic `--native-flash-attn-short`
(`src/program/generate.cpp:1782-1824`, flags at `:1803-1807`), and the installer builds its engine arguments with
`--native` (`setup.py:4224`). The other kernels are the older "canonical" path, built for upstream's own packs
(S2/S4/S8 code planes, BF16-rounded activations). They remain the fallback, the reference the parity tests compare
against, and the path for S2 (Q2_0 canonical) experts.

### Per-token data flow (decode, one layer)

```
R (4 x 2560) --fused_gr_read (or gr_read)--> mixed (2560)
                   |
       +-----------+-----------------------------+
   GDN layer (36 of 48)                    QSA layer (12 of 48)
   native_mmvq / s_gemv (qkv, z)           native_mmvq / s_gemv (q, k, v); bf16 MMVF (indexer k, q)
   fused_gdn_conv_l2                       q/k norm + rope; kv_append_{_, q8, q4}_step
   fused_gdn_ab                            indexer_key_append | native_qsa_indexer_append
   fused_gdn_step_norm                     qsa_block_scores -> qsa_block_topk   [kv_stream_resolve]
   native_mmvq / s_gemv (ssm_out)          qsa_decode_attn_step -> gate -> attn_output GEMV
       +-----------+-----------------------------+
                   |
   fused_gr_read (ffn half) --> BF16 MMVF router logits --> native_router_top10 | router_top10
                   |
   doorbell_publish --> the CPU expert pool computes the misses (host side)
   VRAM experts: native_expert_grouped (GGUF packs) | moe_hit_grouped_s2_dev / moe_grouped_s2 (S2 packs)
   shared_expert(_multi) --> native_moe_combine | moe_combine --> residual write (folded into the next read)
   + cvec_apply on covered layers; layer 1 also runs ple_block
```

The speculative verify window (up to `kVerifyMaxT` = 8 tokens, `verify_kernels.hpp:21`) runs the same layer with
the multi-column forms: `fused_gr_read_multi`, `gdn_*_multi`, `native_mmvq` with `ncols` > 1,
`qsa_decode_attn_batch`, `shared_expert_multi`, `native_router_top10_multi`, `native_moe_combine_multi`. The prompt
path (`src/prefill/`, see [Prefill](../../prefill/README.md)) uses `qsa_block_scores_tc`, `qsa_block_topk` with an
active-block bound, `qsa_prompt_attn_batch`, the dequantizers (for cuBLAS / hipBLAS GEMMs) and
`native_ple_postops_batch`.

### Conventions every kernel follows

- Graph capture. Every per-token value (position, cell count, selection width, hit counts) is read from device
  memory, never passed as a host scalar, because a CUDA graph bakes kernel arguments in at capture time and a frozen
  scalar gives plausible but wrong output. The `QsaStep` record (`qsa.hpp:156-162`) is the canonical example; the
  long note at `qsa.hpp:27-58` lists what a frozen `pos` would do. Grids are sized by a constant capacity and the
  kernels skip the surplus.
- No allocation on the token path. Scratch is caller-owned and sized by helpers: `gr_workspace_bytes`,
  `shared_expert_scratch_bytes`, `ple_block_scratch_bytes`, `native_expert_scratch_bytes`,
  `moe_hit_grouped_scratch_bytes`, `qsa_decode_attn_scratch_floats`. The exception is the sampler's split scratch
  (`sampler.cu:881`), which is never used on a capturing stream.
- Kernel choice is fixed at capture. Every `*_set_enabled` / `*_set_*` switch must be set before a session is built,
  and existing graphs keep the kernels they captured.
- Bitwise multi-token forms. The `_multi` / `_batch` forms are written to equal the single-token call bit for bit, and
  the parity tests check it, because speculative decoding is only exact when a token's result does not depend on its
  window. The prompt attention (`qsa_prompt_attn_batch`) is the stated exception.
- Errors. The canonical entry points print and `std::exit(1)` on bad geometry. The `native_*` adapters throw
  `std::invalid_argument` / `std::runtime_error`, which `src/core/layer.cpp` catches and reports by layer.
- Many kernels are transcriptions of llama.cpp / ggml (MIT; license headers in the `native_*.cu` files and
  `third_party/ggml/LICENSE`), so a weight means exactly what it means there.

### Platforms

The same sources build for three GPU targets. Which branches exist:

| Target | How it builds these files | Kernel-level differences |
| --- | --- | --- |
| NVIDIA, default (`-DSTRATA_ENABLE_CUDA=ON`) | nvcc, `CMAKE_CUDA_ARCHITECTURES` default 120 (`CMakeLists.txt:139-141`); architectures below 75 refused | sm_80+: `mma.sync` m16n8k16, TF32 m16n8k8, `cp.async`. sm_75 (Turing): two m16n8k8 MMAs per k16 step, no `cp.async`, warp scorer, top-k active-block bound. sm_90+ with code built for it: thread-block-cluster top-k (`qsa_select.cu`) and greedy argmax (`sampler.cu`) |
| NVIDIA, experimental (`-DSTRATA_EXPERIMENTAL_SM60=ON`, CUDA 12.x) | allows sm_60 / sm_61 / sm_70 and defines `STRATA_EXPERIMENTAL_SM60` for every target (`CMakeLists.txt:138-176`) | `dp4a.hpp` fallbacks (`__dp4a` below sm_61, `__nanosleep` below sm_70); PR #540's pre-sm_75 decode attention kernel (`qsa_decode_attn.cu:183-393`); PR #600's Volta prompt attention on `mma.m8n8k4` (`qsa_prompt_attn.cu:740-1122`). Below sm_70 the prompt attention returns false and the decode kernel serves prompts |
| AMD RDNA, wave32 (`-DSTRATA_ENABLE_HIP=ON`, `cmake/hip_backend.cmake`) | the `.cu` files compiled as HIP (`__HIPCC__` defined); gfx1100 / gfx1201 validated | no `mma.sync` / `cp.async` / clusters: tensor-core and cluster paths compiled out, the warp/FP32 kernels run. gfx12 (RDNA4) opt-ins: `STRATA_HIP_WMMA=1` prompt attention and `STRATA_SELECT_WMMA=1` block scorer on `v_wmma`. HIP router fast kernel (default). `elementwise.cu` and `quantize_act.cu` built with `-ffp-contract=off` (`CMakeLists.txt:380-384`) |
| AMD gfx906, wave64 (`-DSTRATA_HIP_GFX906=ON`) | the CUDA targets through a compat layer (`include/strata/platform/hip_compat/`), so `STRATA_ENABLE_CUDA` is on and `STRATA_HIP_GFX906` is defined | wave64 layouts: MMVQ (`STRATA_MMVQ_WAVE`), grouped experts (`STRATA_EXP_MODE`), router (`router_top10_wave_kernel`), hyper-connection fast path (`STRATA_GR_FAST`, `STRATA_GR_SPLIT`); `qsa_prompt_attn_batch` returns false; no `--use_fast_math` on the `native_*` files |

`STRATA_EMULATE_CC` (`include/strata/core/emulate.hpp`, tests only) makes the run-time capability checks
(`cc_major_of`, `smem_optin_of`) answer as another generation would. The HIP build is documented in
[HIP kernels](../hip/README.md) and upstream's [AMD_HIP.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/AMD_HIP.md);
the pre-sm_75 build in [OLDER_GPUS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/OLDER_GPUS.md)
and [NVIDIA_V100.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/NVIDIA_V100.md).

### Environment variables read by the kernels

| Variable | Default | Effect | Page |
| --- | --- | --- | --- |
| `STRATA_SCORES_MULTI` | on | block-scores kernel that reads each key once for <= 8 queries | [attention](attention-qsa.md) |
| `STRATA_QSA_WARP` | off | `1`, `attn` or `select`: the pre-sm_80 kernels on any card | [attention](attention-qsa.md) |
| `STRATA_QSA_CLUSTER`, `STRATA_TOPK_OLD`, `STRATA_TOPK_CAPACITY_GUARD`, `STRATA_TOPK_ACTIVE_ANY` | on, off, off, off | top-k kernel choice | [attention](attention-qsa.md) |
| `STRATA_ATTN_PRE75` | on | PR #540 kernel below sm_75 (SM60 builds only) | [attention](attention-qsa.md) |
| `STRATA_PROMPT_ATTN_Q4`, `STRATA_PROMPT_ATTN_V1` | on, off | prompt attention for Q4_0 KV; the first INT8 version | [attention](attention-qsa.md) |
| `STRATA_HIP_WMMA`, `STRATA_SELECT_WMMA` | off, off | gfx12 matrix-core prompt attention / block scorer (HIP) | [attention](attention-qsa.md) |
| `STRATA_GDN_COMMIT_SPLIT` | on | lighter commit kernel when the commit writes no outputs | [gdn](gdn.md) |
| `STRATA_HC_SPLIT`, `STRATA_GR_V3`, `STRATA_GR_DOWN_MAX4` | staged, off, off | fused hyper-connection read variants | [hc/ple](hyper-connections-ple.md) |
| `STRATA_GR_FAST`, `STRATA_GR_SPLIT` | on, off | gfx906 only | [hc/ple](hyper-connections-ple.md) |
| `STRATA_ROPE_TABLE` | off | rotate by the float64 table instead of fast-math `cosf/sinf` | [norms](norms-rope-elementwise.md) |
| `STRATA_OLD_IQ_MMVQ`, `STRATA_IQ_STAGE_GRID`, `STRATA_IQ_STAGE_GRID_MMVQ` | off, on, off | i-quant kernel choice and codebook staging | [moe](moe-and-experts.md) |
| `STRATA_GROUPED_V1`, `STRATA_OLD_GROUPED`, `STRATA_GROUPED_PAIR_MIN_HITS` | off, off, 0 | older grouped-expert launches | [moe](moe-and-experts.md) |
| `STRATA_FUSED_SWIGLU_Q81`, `STRATA_DEC_BATCH` | on, on | shared-expert fusions | [moe](moe-and-experts.md) |
| `STRATA_HIP_ROUTER_OLD`, `STRATA_HIP_ROUTER_FAST`, `STRATA_ROUTER_BLOCK`, `STRATA_MMVQ_WAVE`, `STRATA_EXP_MODE`, `STRATA_HIP_SWIGLU_FUSED` | - | HIP / gfx906 router, MMVQ and expert layouts | [moe](moe-and-experts.md) |
| `STRATA_OLD_SAMPLER`, `STRATA_SAMPLER_ONE_BLOCK`, `STRATA_ARGMAX_MULTI` | off, off, on | sampler kernel choice | [sampling](sampling-and-verify.md) |

Engine-level variables that pick between these kernels (`STRATA_KV_ROT`, `STRATA_PROMPT_ATTN_OLD`,
`STRATA_SELECT_OLD`, `STRATA_KV_HOST_DMA`, `STRATA_DOORBELL_STORE`, `STRATA_SPEC_COUPLED`) are read by the callers and
named on the pages that use them.

## Tests

The kernel parity programs live next to the kernels in `src/kernels/*_parity.cpp` and are registered as CTest
entries inside the GPU block of `CMakeLists.txt` (most run with `--selftest`; run one with
`ctest --test-dir build -R <name>`). Each leaf page lists the ones that cover it, and
[Parity tests](../parity/README.md) documents them. Two more groups are named in `CMakeLists.txt` but are not in the
snapshot: the oracle-linked `native_*_parity` tests (`CMakeLists.txt:387-410`, built only with
`STRATA_ORACLE_SOURCE_DIR` / `STRATA_ORACLE_BUILD_DIR` pointing at a pinned llama.cpp build) expect their sources in
`bench/micro/`, and the published source leaves `bench/micro/` and `tests/CMakeLists.txt` out (`CMakeLists.txt:87-91`).
The HIP-only tests (`tests/hip/*.cpp`, `CMakeLists.txt:499-575`) are present.

## In UnbiasedStrata

The fork kept all 44 `.cu` files and every header in this area. Its tree after commit e41dfc7 (the fork's squashed
history on top of the import 12a7825) changes eight kernels and seven headers to add Qwen3.6-35B-A3B; nothing was
removed or added here. The fork does not build or support the HIP, gfx906 and SYCL paths, which remain in the files.
The fork's own pages are in [../../../codebase/kernels/cuda/](../../../codebase/kernels/cuda/README.md).

| File | Fork change |
| --- | --- |
| `gdn.hpp`, `gdn.cu`, `fused_gdn.*`, `native_gdn_preprocess.*`, `verify_kernels.*` | a `GdnGate` parameter on every output-norm kernel: SiLU gate for Qwen3.6, sigmoid unchanged for Flash-Next |
| `qsa.hpp`, `qsa.cu` | dense attention: `qsa_dense`, `qsa_shapes_valid`, `qsa_selection_cap`; the indexer entry points refuse a dense shape |
| `qsa_decode_attn.*`, `qsa_prompt_attn.cu` | the query-group size `G` becomes a template parameter (2, 8, 12); a null `ids` is the identity selection; a chunk cap of 128 blocks for dense shapes with a running rescale inside each block; `qsa_identity_ids` |
| `iq_kernels.*` | Q6_K MMVQ and dequantizer; IQ3_XXS, Q5_K and Q6_K added to the grouped-expert down list (Qwen3.6's n_ff 512) |

## See also

- [Kernels overview](../README.md), [CPU kernels](../cpu/README.md), [HIP](../hip/README.md),
  [parity tests](../parity/README.md), [Prefill](../../prefill/README.md), [SYCL](../../sycl/README.md)
- Upstream's [DETAILS.md](../../../upstream/DETAILS.md) ("How it works", the speed tables)
