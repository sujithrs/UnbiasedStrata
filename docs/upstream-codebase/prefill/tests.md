[Upstream docs](../README.md) › [Prefill](README.md) › Tests

# Tests of the prompt path

## Purpose

The prompt path is not bit-identical to decode, so its tests check two things: that each optimized kernel matches
its simpler predecessor exactly (where it claims to), and that the int8 / MMQ expert products stay within the error
of a reference. None of them needs the model; all are synthetic. They are registered only with
`-DSTRATA_BUILD_TESTS=ON`, which defaults to OFF in this snapshot because `tests/CMakeLists.txt` and `bench/micro`
are not published (`CMakeLists.txt:86-92`); the GPU parity programs below are built unconditionally on CUDA.

## Inventory

| Test (CTest name) | Source | Registered | What it checks |
| --- | --- | --- | --- |
| `gdn_rec_parity` | [src/prefill/gdn_rec_parity.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/gdn_rec_parity.cu) | CUDA, not gfx906 (`CMakeLists.txt:656-660`) | `gdn_rec_kh_kernel` against `gdn_rec_cols_pipe_kernel` (both copied verbatim) and the output norm: output, state after the chunk and FP16 output bit for bit, for T = 1 .. 4,099 with a non-zero starting state; `--bench` times T = 8,192 and 32,768, then 1 to 6 blocks per SM (`scale`) and the engine's grids on fewer SMs (`fewer_sms`) |
| `gemm_bf16_parity` | [src/prefill/gemm_bf16_parity.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/gemm_bf16_parity.cu) | CUDA, not gfx906 (`:662-664`) | `Gemm::bf16` against `cublasGemmEx` BF16 over the prompt path's shapes, within FP32 rounding; on sm_7x it exercises the FP16 route, `STRATA_BF16_TC=2` forces it on any card; `--bench` adds timings |
| `prefill_fused_moe_test` | `tests/cuda/prefill_fused_moe_test.cpp` | CUDA + MMQ, `STRATA_BUILD_TESTS` (`:1081-1086`) | the fused Q2_0 kernels on random blobs and skewed routing against a double-precision reference and the MMQ path; the fused error must stay within 1.5x MMQ's RMS and 2x its worst row; part 2 times one layer at 2,048 tokens; exits 77 below sm_80 |
| `prefill_fused_iq_test` | `tests/cuda/prefill_fused_iq_test.cpp` | same (`:1087-1091`) | the same per native format pair (gate/up IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_XS, down Q2_0 IQ4_NL), against ggml's own `to_float`; sets `STRATA_PF_FUSED=1` itself; exits 77 below sm_80 |
| `prefill_mmq_kquant_test` | `tests/cuda/prefill_mmq_kquant_test.cpp` | CUDA with `STRATA_MMQ_KQUANTS` and tests (`:1093-1099`) | MMQ products for UD-Q4_K_XL's formats (Q4_K / Q5_K gate-up, Q5_1 / Q8_0 down) against ggml's dequantized weights; several experts, permuted rows, a zero row; a screen for layout / stride / bound / row-id errors |
| `hip_prefill_gemm` | `tests/hip/prefill_gemm.cpp` | HIP (`:502-505`) | BF16 / FP16 through `Gemm` against a CPU reference |
| `hip_prefill_hipblaslt_gemm` | `tests/hip/prefill_hipblaslt_gemm.cpp` | HIP with hipBLASLt (`:558-564`) | needs `STRATA_HIPBLASLT_TUNING` (exit 77 otherwise): the table loads, two rows exist, and `Gemm` matches hipBLAS on them; it does not prove the table's solution ran |
| `hip_prefill_mmq_parity` | `tests/hip/prefill_mmq_parity.cpp` | HIP with `STRATA_PREFILL_MMQ` (`:1133-1138`) | MMQ against a CPU FP32 product of ggml-dequantized weights on synthetic layouts; optional first-expert check with a pack |
| `hip_prefill_native_batch` | `tests/hip/prefill_native_batch.cpp` | HIP, when the file exists (`:541-546`) | the native QSA indexer append and the native embedding gather the prompt path batches |

Run them from a build directory, e.g. `ctest -R "gdn_rec_parity|gemm_bf16_parity|prefill_"`, or run the programs
directly (`./gdn_rec_parity --bench`). CTest treats exit code 77 as "skipped" where `SKIP_RETURN_CODE 77` is set.

## End-to-end checks

There is no CTest target that runs a real prompt. Upstream validates the whole path with the engine:

- `--prefill-until P` batches only `[0, P)` and runs the rest through the token path, so logits after `P` (which
  depend on the batched state) can be compared against a reference (`src/program/generate.cpp:441-445`).
- `STRATA_PREFILL_DUMP_R=<file>` writes every 64th position's final residual for A/B comparisons
  (`src/prefill/prefill.cpp:2942-2954`); `STRATA_STATE_HASH_GDN` prints a hash of every GDN state after the prompt.
- `docs/DETAILS.md` reports teacher-forced KL and top-1 agreement against the FP16 prompt path for the fused kernels
  and the 0.1.39b ring change.

## Debug switches

`STRATA_PREFILL_TIMING`, `STRATA_PF_STEP_SYNC`, `STRATA_DBG_NAN`, `STRATA_TRACE`, `STRATA_DRAFT_TIMING`,
`STRATA_SEL_OVERLAP`, `STRATA_IDX_FP16_CHECK`, `STRATA_QSA_DUMP`; see [pipeline.md](pipeline.md).

## In UnbiasedStrata

The CUDA tests are kept; `gdn_rec_parity.cu` was extended to both model geometries and a `prefill_qwen36_test` with
its fixture was added (commit e41dfc7). The HIP tests (`tests/hip/`) were removed. See
[../../codebase/prefill/tests.md](../../codebase/prefill/tests.md).

## See also

[gemm.md](gemm.md), [moe.md](moe.md), [gdn-and-attention.md](gdn-and-attention.md), the build area for
`STRATA_BUILD_TESTS`.
