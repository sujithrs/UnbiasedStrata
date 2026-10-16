[Upstream docs](../README.md) › [Prefill](README.md) › Dense GEMMs

# `Gemm`: the prompt path's dense products

## Purpose

Every dense projection of a chunk is `Y[T, N] = X[T, K] . W[N, K]^T` with `W` row-major (the GGUF / pack layout) and
FP32 outputs. `strata::prefill::Gemm` runs them on tensor cores through cuBLAS (hipBLAS on HIP, with an optional
hipBLASLt route), with fallbacks for GPUs that have no BF16 tensor cores.

Files: [include/strata/prefill/gemm.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/prefill/gemm.hpp),
[src/prefill/gemm.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/gemm.cu),
[src/prefill/hipblaslt_tuning.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/hipblaslt_tuning.hpp).

## API

| Member | Line (`gemm.hpp`) | What it does |
| --- | ---: | --- |
| `init(stream, scratch_elems, err)` | 23 | own cuBLAS handle, 32 MiB workspace, optional dequant scratch (`src/prefill/gemm.cu:349-379`) |
| `init_external(stream, scratch, elems, workspace, ws_bytes, err)` | 25 | the same over caller-owned buffers: the prompt path's borrowed region (`:313-333`) |
| `bf16(X, W, Y, T, N, K, ldy, beta)` | 31 | BF16 x BF16 -> FP32 (`:464-517`) |
| `f16(X, W, Y, T, N, K, ldy, beta)` | 35 | FP16 x FP16 -> FP32 (`:519-536`) |
| `native(X, ggml_type, W_blocks, Y, T, N, K, ldy, beta)` | 39 | dequantize the GGUF blocks to FP16 in the scratch (`strata::kernels::dequant_f16`), then `f16`; in row slices when `N * K` exceeds the scratch (`:538-554`) |
| `rebind(scratch, elems, workspace, ws_bytes)` | 43 | after `Prefill::relayout` moved the buffers (`:335-347`) |

`beta = 1` accumulates into `Y`; the prompt path uses it for the BF16 low parts of `STRATA_PREFILL_BF16X2`. All
products use `cublasGemmEx` with `CUBLAS_COMPUTE_32F` in a column-major view (`Y^T = W . X^T`, `OP_T` on `W`).
Setup failures that the engine survives (`cublasSetStream`, `cublasSetWorkspace`, `cublasSetMathMode`) are printed
and continue (`note`, `:67-69`); a failed GEMM exits the process (`ck`, `:42-47`).

## Who uses which product

| Product | Weights | Activations |
| --- | --- | --- |
| `native` (FP16 after dequantization) | quantized dense projections: GDN `attn_qkv`, `attn_gate`, `ssm_out`; QSA `attn_q/k/v`, `attn_output`; shared expert; the PLE key when it is GGUF-native | `mixed_h`, `y_h`, `attn_h`, `sh_h` (FP16 images) |
| `bf16` | the pack's BF16 tensors: hyper-connection down / up / inject, `ssm_alpha` / `ssm_beta`, indexer `k_proj` / `q_proj`, router, shared-expert gate, PLE key / value | BF16 images (`xn16`, `lo16`, `mixed_bf`) |
| `f16` | the FP16 expert fallback (dequantized experts) | gathered FP16 rows |

The dequant scratch is `GEMM_SCRATCH` = 32M FP16 elements (64 MiB) and the workspace 32 MiB
(`src/prefill/prefill.cpp:632-633`).

## Below sm_80 (`src/prefill/gemm.cu:381-462`)

Turing and Volta have no BF16 tensor cores (cuBLAS falls back to a SIMT kernel: "a fifth of a 4K prompt on an RTX
2080 Ti (#655) and ~10% of a V100's prompt (#540)"), and Pascal has no BF16 GEMM at all (#395). `bf16_path()`
(`:439-450`) picks:

| Path | When | How |
| --- | --- | --- |
| 0: cuBLAS BF16 | sm_80+, unknown cc, or sm_75 by default | `cublasGemmEx` on `CUDA_R_16BF` |
| 1: through FP16 | sm_70-7.4 by default; any 7.x with `STRATA_BF16_TC=1`; any card with `STRATA_BF16_TC=2` (test mode) | weight converted to FP16 once per call, activations in slices of 16M elements, FP16 tensor-core GEMM; values beyond FP16's range clamp to +-65504; single-row products and `beta = 1` keep cuBLAS |
| 2: through FP32 | sm_6x (only in the `STRATA_EXPERIMENTAL_SM60` build) | both operands widened to FP32 exactly, `cublasSgemm` |

`STRATA_BF16_TC=0` keeps cuBLAS BF16 on 7.x. The FP16 copies live in the instance's own buffers (`tc_w_`, `tc_x_`),
allocated on demand with `cudaMalloc`, not from the borrowed region. The compute capability is cached per device, so
a layer split can mix cards.

## HIP

- With `STRATA_HIPBLASLT_AVAILABLE` (CMake defines it when hipBLASLt is found, `CMakeLists.txt:487-489`) and
  `STRATA_HIPBLASLT_TUNING=<file>`, `bf16` and `f16` first try hipBLASLt with a calibrated solution
  (`try_hipblaslt`, `src/prefill/gemm.cu:246-295`). The table (`hipblaslt_tuning.hpp`) has a header
  `STRATA_HIPBLASLT_TUNING_V1 <arch> <version>` and rows `<f16|bf16> N K ldy T-bucket solution-id`; it is refused
  when the architecture or hipBLASLt version differs, a row is malformed or duplicated (`:26-96`). Lookup takes the
  row with the same type, N, K and ldy and the closest T bucket (`closest`, `:98-110`). A shape without a row, a
  solution the library rejects or that needs more workspace falls back to hipBLAS, counted per shape
  (`STRATA_HIPBLASLT_VERBOSE=1` prints a summary at exit). A failed launch with `beta != 0` exits rather than fall
  back.
- Windows HIP: hipBLAS can leave a stale `hipErrorInvalidValue` after a correct GEMM (#247 / #325); it is cleared
  after a successful call (`STRATA_ABSORB_HIPBLAS_STICKY`, `:49-64`).
- gfx906 builds compile this file through `include/strata/platform/hip_compat/cublas_v2.h`, which maps the cuBLAS
  names onto hipBLAS and turns `cublasSetMathMode` / `cublasSetWorkspace` into no-ops.

The upstream user docs: `docs/AMD_HIP.md` ("Tuning table") and `docs/AMD_HIP_PERFORMANCE.md`
([permalink](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/AMD_HIP.md)); the
tables ship as `tools/hip/<arch>-hipblaslt-<version>.txt` and are made with `tools/hip/tune_hipblaslt.cpp`.

## Tests

`gemm_bf16_parity` (CUDA, not gfx906): `Gemm::bf16` against cuBLAS's own BF16 GEMM over the prompt path's shapes
(hc down / up, PLE value, router, hc inject, a single row, `beta = 1`, a wide `ldy`), within FP32-accumulation
rounding; `--bench` adds timings. HIP: `hip_prefill_gemm` (BF16 / FP16 against a CPU reference) and the opt-in
`hip_prefill_hipblaslt_gemm` (needs `STRATA_HIPBLASLT_TUNING`, exits 77 otherwise). See [tests.md](tests.md).

## Gotchas

- `gemm.hpp`'s header comment says weights are dequantized "to BF16 ... (`dequant_bf16`)" and activations rounded to
  BF16; the code dequantizes native blocks to FP16 (`dequant_f16`) and multiplies FP16 activations
  (`src/prefill/gemm.cu:538-554`). Only the pack's BF16 tensors use the BF16 product.
- The CUDA path never uses cuBLASLt; the "Lt" route is HIP only.

## In UnbiasedStrata

Kept unchanged (no diff against 12a7825). The fork builds CUDA only, so the hipBLASLt code is compiled out there. See
[../../codebase/prefill/gemm.md](../../codebase/prefill/gemm.md).

## See also

[pipeline.md](pipeline.md), [gdn-and-attention.md](gdn-and-attention.md), [tests.md](tests.md).
