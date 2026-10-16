[Upstream docs](../README.md) › Prefill

# Prefill (batched prompt processing)

The prefill subsystem (`strata::prefill`, library `strata_prefill`, plus `strata_mmq` for the int8 expert kernels)
reads a prompt in chunks of `T` tokens (typically 1,024 to 8,192; up to 32,768 on request) through every layer at
once, instead of one token at a time through the decode path. It leaves the session state exactly where the token
path would have left it: the Gated DeltaNet (GDN) recurrent and conv state of every linear-attention layer, the paged
K/V cells and the indexer state of every QSA (sparse full-attention) layer, the per-layer-embedding (PLE) n-gram
history, and, through a callback, the MTP draft layer's K/V. Decode then continues from the next position. The only
engine caller is `src/program/generate.cpp`: the one-shot `generate` path and the `--serve` loop build a
`strata::prefill::Prefill`, lend it the tail of the GPU expert cache as device memory, call `Prefill::run`, and
refill the lent slots afterwards. Prefill in turn calls the core layer (weights, layer views, session state, expert
source and cache, embedding), the shared kernels in `src/kernels/` (QSA prompt attention, indexer, KV formats, PLE,
rope scaling, i-quant dequantizers), cuBLAS / hipBLAS(Lt), and llama.cpp's MMQ kernels compiled from the pinned
llama.cpp checkout.

Upstream's prompt path is built for one geometry only: Qwen3.8-Flash-Next (`qwen4exp`): 2,560 wide, four
hyper-connection streams (`hc` 4, low rank 320), 512 experts with top-10 routing, expert width 640, GDN with 48 value
heads, 16 key heads and 10,240 conv channels, QSA attention with 24 query heads, 2 KV heads, head dim 256. These are
compile-time constants (`src/prefill/prefill.cpp:90-91`, `src/prefill/kernels.cu:18-19`), and `Prefill::init`
refuses a model whose widths differ (`src/prefill/prefill.cpp:737-739`). Every pack of that model goes through it:
the Strata Q2_0 pack, the native i-quant packs (IQ2_XS, IQ3_XXS, IQ3_S, the Coder's IQ1_M layers), Swift 1.5 and
Unsloth's UD-Q4_K_XL. See the [glossary](../guides/glossary.md) for the terms.

## Contents

| Page | What it covers |
| --- | --- |
| [pipeline.md](pipeline.md) | `Prefill` class: `init` / `carve` / `relayout` / `bytes_needed`, the chunk and layer loop of `run_impl`, embeddings, the PLE block, hooks (`on_chunk`, `on_stage_chunk`, `should_stop`, `embd_rows`), the state left for decode, `draft_kv`, timing and debug switches, how `generate.cpp` drives it |
| [expert-streaming.md](expert-streaming.md) | How experts reach the card during a prompt: borrowed cache slots, the staging ring, the stream-all walk, the `Stager` host-copy threads, the issuer thread, ring sizing in bytes, `--prefill auto` and its chunk scan, refill after the prompt |
| [moe.md](moe.md) | The MoE half: router, shared expert, the routed-expert paths (FP16 GEMM fallback, llama.cpp MMQ, fused Q2_0 int8, fused native IQ int8), how one is chosen per layer, the combine |
| [gemm.md](gemm.md) | `Gemm` (`gemm.cu`): cuBLAS BF16 / FP16 / dequantize-then-FP16 products, the sub-sm_80 BF16 paths, hipBLASLt with a tuning table (`hipblaslt_tuning.hpp`) |
| [ggml-bridge.md](ggml-bridge.md) | `moe_mmq.cu` and `ggml_cuda_host.cu`: llama.cpp's MMQ compiled into `strata_mmq`, `supported` / `fits`, the gathers, `STRATA_MMQ_KQUANTS`, `STRATA_ORCA_Q4KS_MMQ`, the HIP `STRATA_PREFILL_MMQ` build |
| [gdn-and-attention.md](gdn-and-attention.md) | `kernels.cu`: the hyper-connection kernels, the chunked GDN gates / conv / recurrence (all variants), the QSA half (norm, rope, `kv_append`, indexer, block selection, prompt attention), how the state lands for decode |
| [multi-gpu.md](multi-gpu.md) | The layer-split stage pipeline, the `--peer-device` peer (`set_peer`, peer streaming) and the opt-in layer-split helper (`STRATA_PREFILL_HELP=1`) |
| [tests.md](tests.md) | `gdn_rec_parity`, `gemm_bf16_parity`, the fused / MMQ expert tests, the HIP prompt tests, debug switches |

## Source map

| Path | Lines | Role |
| --- | ---: | --- |
| [src/prefill/prefill.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/prefill.cpp) | 3084 | `Prefill` implementation: buffer layout, ring sizing, expert streaming, the layer loop, peer and layer-split help, draft-layer K/V |
| [src/prefill/gemm.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/gemm.cu) | 556 | `Gemm`: `cublasGemmEx` BF16 / FP16 products, dequantize-then-GEMM for GGUF weights, Volta/Turing/Pascal BF16 fallbacks, hipBLASLt dispatch |
| [src/prefill/kernels.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/kernels.cu) | 1024 | Element-wise and stateful prompt kernels: hyper-connection, GDN gates / conv / recurrence, router, SwiGLU, combine, rope, KV append, conversions |
| [src/prefill/moe_fused.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/moe_fused.cu) | 465 | Fused int8 tensor-core experts for the Strata Q2_0 pack (sm_80+), GPU-side grouping |
| [src/prefill/moe_fused_iq.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/moe_fused_iq.cu) | 638 | Fused int8 experts for native GGUF i-quant packs (gate/up IQ2_XXS..IQ4_XS, down Q2_0 / IQ4_NL), opt-in |
| [src/prefill/moe_mmq.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/moe_mmq.cu) | 266 | Wrapper over llama.cpp's MMQ (`mul_mat_q_case`, `quantize_mmq_q8_1_cuda`), expert gathers, SwiGLU, iota |
| [src/prefill/ggml_cuda_host.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/ggml_cuda_host.cu) | 147 | The ggml-cuda host symbols MMQ needs (`ggml_cuda_info`, error hook, buffer pool) without linking ggml-cuda |
| [src/prefill/hipblaslt_tuning.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/hipblaslt_tuning.hpp) | 123 | Parser and lookup for the `STRATA_HIPBLASLT_TUNING_V1` solution table (HIP only) |
| [src/prefill/gdn_rec_parity.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/gdn_rec_parity.cu) | 886 | Test / bench program: the GDN recurrence variants bit for bit |
| [src/prefill/gemm_bf16_parity.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/gemm_bf16_parity.cu) | 155 | Test / bench program: `Gemm::bf16` against cuBLAS's own BF16 GEMM |
| [include/strata/prefill/prefill.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/prefill/prefill.hpp) | 184 | Public `Prefill` API and `PrefillStats` |
| [include/strata/prefill/gemm.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/prefill/gemm.hpp) | 66 | `Gemm` class |
| [include/strata/prefill/kernels.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/prefill/kernels.hpp) | 100 | Kernel launch functions |
| [include/strata/prefill/moe_fused.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/prefill/moe_fused.hpp) | 64 | `strata::prefill::fused` API (Q2_0 path), `Batch`, `kTileRows`, `kMaxBatch` |
| [include/strata/prefill/moe_fused_iq.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/prefill/moe_fused_iq.hpp) | 41 | `NativeGeom`, `native_supported`, `quantize_act_native`, `experts_native` |
| [include/strata/prefill/moe_mmq.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/prefill/moe_mmq.hpp) | 88 | `strata::prefill::mmq` API: `Product`, `Context`, `GatherGroup` |

Build (`CMakeLists.txt`): `strata_prefill` = `gemm.cu` + `kernels.cu` + `prefill.cpp` (`CMakeLists.txt:484`), linked
into the `strata` executable (`CMakeLists.txt:497`). On CUDA, `strata_mmq` = `moe_mmq.cu`, `moe_fused.cu`,
`moe_fused_iq.cu`, `ggml_cuda_host.cu`, ggml's `quantize.cu` and one `mmq-instance-<type>.cu` per covered type
(`CMakeLists.txt:1054-1064`); linking it defines `STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1` on `strata_prefill`
(`CMakeLists.txt:1079-1080`). On HIP, `-DSTRATA_PREFILL_MMQ=ON` builds `strata_mmq` from `moe_mmq.cu` and
`ggml_cuda_host.cu` only and defines `STRATA_PREFILL_MMQ=1` (`CMakeLists.txt:1100-1131`). Without these macros
`prefill.cpp:47-83` compiles stub namespaces, so the FP16 expert path is used everywhere. The Intel SYCL port keeps
its own migrated copies (`sycl/src/prefill/`, with `moe_fused_stub.cpp` in place of the fused kernels); that port is
documented in its own area.

## Data flow of one chunk

```
tokens[c0 .. c0+T)                                           (generate.cpp -> Prefill::run)
   |  embed: one batched device gather, or per token (image rows from embd_rows)
   v
 R [T x 4*2560]   gr_broadcast to the four hyper-connection streams
   |                                     PLE rows for the chunk gathered on a host thread (SSD reads)
   |  for layer l in [LB, LE):           experts of every layer streamed in (layer, id) order on the
   |    l == 1: PLE block over the chunk copy stream (chunks >= 1024 tokens)
   |    half 0: hc read -> mixer -> hc write
   |       GDN layer : qkv, z (FP16 GEMM) alpha, beta (BF16 GEMM) -> gates -> conv -> recurrence -> ssm_out
   |                   writes ss.gdn_state (recurrent state + conv history)
   |       QSA layer : K, V, Q (FP16 GEMM) -> k/q norm, rope -> kv_append into the paged pools
   |                   -> indexer append, block scores, top-k selection -> prompt attention -> gate -> attn_output
   |    half 1: hc read -> MoE -> hc write
   |       router (BF16 GEMM) -> top-10 of 512 -> shared expert (FP16 GEMMs)
   |       routed experts: grouped by expert -> fused int8 / MMQ / FP16 GEMM, each expert read from
   |                       a resident cache slot or a ring slot just copied from host RAM
   |       moe_combine -> bo;   gr_write (or gr_write_norm_rs, fused with the next half's norm)
   v
 R rows of the chunk -> on_chunk (MTP draft layer K/V) ; next chunk
```

## Key concepts

- **Chunk `T`.** Chosen by `generate.cpp` (`--prefill CHUNK` or `--prefill auto[:N]`; the engine's default is 0,
  the token path, and the installer passes `--prefill auto`). Every non-resident expert a chunk routes to (from
  1,024 tokens on: every non-resident expert, routed or not) is copied over PCIe once per chunk, so a bigger chunk
  streams fewer bytes per token. See [expert-streaming.md](expert-streaming.md).
- **Borrowed memory.** By default every prompt buffer is carved out of the top slots of the GPU expert cache
  (`init(..., borrow, borrow_bytes)`); the experts in those slots are marked non-resident for the prompt and the slots
  are refilled from host RAM after it. `--no-prefill-borrow` allocates instead.
- **One scratch region for three phases.** The GDN, QSA and MoE buffer sets are never live at once, so they share one
  region sized as the largest of the three (`src/prefill/prefill.cpp:635-728`, carved at `:867-906`).
- **Numerics differ from decode by design.** Dense projections take BF16 or FP16 activations and experts take
  q8_1 / int8 activations; decode keeps FP32 activations and its own kernels. The difference is screened by the tests
  in [tests.md](tests.md), and some switches (`STRATA_PREFILL_BF16X2`, `STRATA_PF_FUSED=0`) exist as A/B arms.
- **Native weights required.** Every quantized dense projection must carry its GGUF blocks (`--native`), otherwise
  `native_proj` fails (`src/prefill/prefill.cpp:1441-1446`).

## Upstream measurements

From `docs/DETAILS.md` (RTX 5070 12 GB, Ryzen 5 7600, 64 GB DDR5-5200, Windows, engine 0.1.26, `--prefill auto`):
prompt processing at 32K reads 2,171 tok/s on Q2_0, 2,092 on IQ2_XS, 1,745 on IQ3_XXS, 1,624 on IQ3_S and 2,177 on
the Coder. Engine 0.1.36 put Q2_0's prompt experts on the fused int8 kernels: "4K 1,294 -> 1,570, 32K 2,170 -> 2,653,
128K 2,123 -> 2,468 tokens/s" on the same PC. The ring-in-bytes change of 0.1.39b (#583) is described there too
("Prompts", in "How it works"). The fork keeps that file as [docs/upstream/DETAILS.md](../../upstream/DETAILS.md).

## In UnbiasedStrata

Every source file of this area is kept. The fork changed `prefill.cpp`, `kernels.cu` / `kernels.hpp`,
`gdn_rec_parity.cu`, `moe_fused_iq.cu` / `.hpp`, `moe_mmq.cu` and `prefill.hpp` (commits e41dfc7 and 16d0fb2): the
compile-time Flash-Next widths became a `Geom` read from the GGUF so the same path also runs Qwen3.6-35B-A3B
(`qwen35moe`: one residual stream instead of hyper-connections, SiLU-gated DeltaNet, dense attention without the
indexer, top-8 of 256 experts), an `on_seam` test hook was added, `draft_kv` gained a Qwen3.6 (`nextn`) branch, the
fused IQ kernels take `n_embd` / `n_ff` from `NativeGeom`, and MMQ covers Q6_K under `STRATA_MMQ_KQUANTS`. The HIP
prompt tests (`tests/hip/prefill_*.cpp`) and the SYCL copies were removed. The fork's own description is
[../../codebase/prefill/README.md](../../codebase/prefill/README.md).
