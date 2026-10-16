[Upstream docs](../README.md) › [Prefill](README.md) › ggml bridge

# The ggml bridge: llama.cpp's MMQ inside the prompt path

## Purpose

The prompt path's default expert kernels (and the draft layer's Q8_0 projections in `draft_kv`) are llama.cpp's MMQ
kernels (`ggml-cuda/mmq.cuh`, MIT): the weights stay quantized, the activations are rounded to q8_1, and the
products run on int8 tensor cores. They are compiled from the same pinned llama.cpp checkout the build already takes
ggml from, into a separate static library `strata_mmq`, without the rest of ggml-cuda. Two files make that work:
`moe_mmq.cu` (the wrapper the prompt path calls) and `ggml_cuda_host.cu` (the few host symbols of `ggml-cuda.cu` that
MMQ and its quantizer reference). The header comment gives the reason for MMQ: the dequantize-to-FP16 + cuBLAS path
"wrote ~10 MB of FP16 per expert and multiplied in FP16; this reads the ~1.4-2 MB expert once"
(`include/strata/prefill/moe_mmq.hpp:1-5`).

Files: [include/strata/prefill/moe_mmq.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/prefill/moe_mmq.hpp),
[src/prefill/moe_mmq.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/moe_mmq.cu),
[src/prefill/ggml_cuda_host.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/ggml_cuda_host.cu),
`CMakeLists.txt:1037-1131`.

## API (`strata::prefill::mmq`)

| Function | Line (`moe_mmq.cu`) | What it does |
| --- | ---: | --- |
| `built()` | 106 | true in a build with MMQ (the stub in `prefill.cpp:47-64` returns false) |
| `supported(type)` | 108-124 | Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q8_0; Q5_0 with `STRATA_ORCA_Q4KS_MMQ`; Q4_K, Q5_K, Q5_1 with `STRATA_MMQ_KQUANTS`. IQ1_M is not covered |
| `fits(type, w_rows)` | 126-152 | #420: `supported`, and on EVERY visible GPU some MMQ tile (J = 8..128) exists for this type with its shared memory within the card's limit, the same test llama.cpp's tile choice makes (which aborts the process with "J_best=0" when nothing fits). A refusal is printed once per type and keeps that product on the non-MMQ path |
| `matrix_bytes(type, rows, cols)` | 154-156 | bytes of one expert matrix |
| `q8_bytes(rows, cols)` | 158-160 | bytes of q8_1 activation rows (row padded to 512 values) plus 128 blocks of slack |
| `quantize(x, ids, xq, type, cols, ld, rows, stream)` | 162-167 | `quantize_mmq_q8_1_cuda`: row `i` of the output is row `ids[i]` of `x` |
| `Context` / `Context::run(Product, stream)` | 169-210 | wraps a `ggml_backend_cuda_context` (its scratch pool for the stream-k fixup); `run` fills an `mmq_args` and calls `mul_mat_q_case<type>` in MoE mode: `n` experts `expert_bytes` apart, rows `[bounds[e], bounds[e+1])`, destination rows through `ids` |
| `gather_native(gate, up, gu_half, down, d_bytes, gu_dst, d_dst, stream)` | 212-227 | copies one native expert's gate rows, up rows and down rows into a group slot (16-byte copies when everything is aligned, byte copies otherwise) |
| `gather_native_group(group, ...)` | 229-246 | the same for up to 16 experts in ONE launch (`blockIdx.y` = expert); false when anything is not 16-byte aligned |
| `gather_strata_q2(blob, gu_dst, d_dst, stream)` | 248-252 | converts a Strata Q2_0 blob (codes and FP16 scales in separate planes) into GGUF Q2_0 blocks `{f16 d; 16 code bytes}`; same values |
| `swiglu(gu, h, rows, n_ff, interleaved, stream)` | 254-258 | `silu(gate) * up` in FP32, gate / up interleaved (Strata pack) or split (GGUF) |
| `iota(dst, n, stream)` | 260-264 | identity row map MMQ's MoE mode writes through |

`Product` (`moe_mmq.hpp:36-48`): weights `w` of `type`, `w_rows x w_cols` per expert, `n` experts, `xq`, device
`bounds` (n + 1 entries), `ids`, `total_rows`, `max_rows` (launch grid), `dst`, `ld_dst`.

## `ggml_cuda_host.cu`

| Symbol | Line | Notes |
| --- | ---: | --- |
| `ggml_cuda_error` | 12 | prints and aborts |
| `ggml_cuda_get_device` | 17 | |
| `ggml_cuda_info()` | 62-114 | device count and per device `cc`, `smpbo`, `nsm`, `warp_size`, VRAM. CUDA: from `cudaDeviceGetAttribute` (#542: a `cudaDeviceProp` filled by an older libcudart than the headers is shifted, which once made every i-quant lose MMQ through `fits`), with `STRATA_EMULATE_CC` applied (`strata/core/emulate.hpp`). HIP: `cc` parsed from `gcnArchName` in ggml's AMD encoding (`ggml_cuda_parse_amd_id`, `:35-57`) |
| `CachingPool` / `new_pool_for_device` | 118-145 | a simple reuse-by-size pool of `cudaMalloc`s for MMQ's fixup buffers |
| `~ggml_backend_cuda_context` | 147 | empty |

## Build

| Build | What `strata_mmq` contains | Macros on `strata_prefill` |
| --- | --- | --- |
| CUDA (`STRATA_ENABLE_CUDA` with the ggml sources, `CMakeLists.txt:1043-1092`) | `moe_mmq.cu`, `moe_fused.cu`, `moe_fused_iq.cu`, `ggml_cuda_host.cu`, ggml's `quantize.cu`, `template-instances/mmq-instance-<t>.cu` for q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0 (+ q5_0, + q4_k q5_k q5_1) | `STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1` |
| HIP with `-DSTRATA_PREFILL_MMQ=ON` (`:1100-1131`) | `moe_mmq.cu`, `ggml_cuda_host.cu`, `quantize.cu`, the instances, compiled as HIP with `GGML_USE_HIP=1` | `STRATA_PREFILL_MMQ=1` only (no fused kernels) |
| Neither | none | none: `prefill.cpp` compiles stubs, the FP16 path everywhere |

CMake options (`CMakeLists.txt:72-79`): `STRATA_PREFILL_MMQ` (HIP only; requires `STRATA_NATIVE_EXPERTS`),
`STRATA_MMQ_KQUANTS` (CUDA: Q4_K / Q5_K / Q5_1 for Unsloth's UD-Q4_K_XL, see `docs/UNSLOTH_Q4.md`),
`STRATA_ORCA_Q4KS_MMQ` (Q5_0 for OrcaRouter Q4_K_S, #296). `STRATA_GGML_DIR` points at another ggml checkout. MSVC
needs `/bigobj` for these sources; they compile with `--extended-lambda -use_fast_math`. The llama.cpp checkout
itself (which provides `GGML_TYPE_Q2_0`, id 42) is described in the build area.

Runtime: `STRATA_PREFILL_MMQ=0` turns MMQ off in a build that has it (`src/prefill/prefill.cpp:673-674`).

## Platform notes

- CUDA: MMQ on every card with a tile for the type (`fits` checks all visible GPUs, so a mixed rig takes the most
  restrictive card's answer).
- HIP: MMQ is opt-in at build time; `docs/AMD_HIP_PERFORMANCE.md` describes the measured arm (`STRATA_PREFILL_MMQ=1`
  with a hipBLASLt table and `STRATA_PREFILL_RING=96`).
- gfx906 compiles the CUDA branch through the compat headers; `moe_fused.cu` reports its kernels unavailable there.

## Tests

`prefill_mmq_kquant_test` (CUDA, `STRATA_MMQ_KQUANTS`): Q4_K / Q5_K gate-up and Q5_1 / Q8_0 down through MMQ against
a double-precision product of ggml's dequantized weights; a screen for layout, stride, bound and row-id errors.
`hip_prefill_mmq_parity` (HIP with `STRATA_PREFILL_MMQ`): the same idea on synthetic layouts. Both in
[tests.md](tests.md).

## Gotchas

- `fits` prints its refusal once per type, naming the GPU, its cc and shared memory; the layer then runs the FP16
  path, which is much slower, so a missing MMQ on one card of a rig slows every card's prompt.
- `gather_native_group` needs 16-byte alignment of every blob and offset; otherwise the experts are gathered one at a
  time (`src/prefill/prefill.cpp:2713-2720`).

## In UnbiasedStrata

Kept; `moe_mmq.cu` adds Q6_K under `STRATA_MMQ_KQUANTS` for Qwen3.6's UD-Q4_K_XL down projections (commit e41dfc7).
The HIP build branch is unused there. See [../../codebase/prefill/ggml-bridge.md](../../codebase/prefill/ggml-bridge.md).

## See also

[moe.md](moe.md), [tests.md](tests.md), the build area (`docs/upstream-codebase/build/`) for the llama.cpp checkout.
