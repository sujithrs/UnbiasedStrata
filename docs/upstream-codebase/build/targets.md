[Upstream docs](../README.md) › [Build](README.md) › Targets

# Targets: libraries, programs and tools

## Purpose

What the top-level `CMakeLists.txt` builds: the 12 distinct library targets (two of them imported aliases for the
gfx906 build; `strata_hip_runtime` comes from `cmake/hip_backend.cmake`), the user-facing engine `strata`, the diagnostic programs, and how they link. Test executables are
listed in [tests-registration.md](tests-registration.md).

File covered: [CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/CMakeLists.txt)
(13 `add_library` calls, 102 `add_executable` calls; one of each pair `strata_mmq` / `pool_affinity_test` /
`hip_router_fast` is chosen by backend or OS).

## Which targets exist in which configuration

| Configuration | Libraries | Programs |
| --- | --- | --- |
| Always (even with no GPU backend) | `strata_warnings`, `strata_artifact`, `strata_plan`, `strata_spec`, `strata_kernels_cpu` | `strata-gguf`, `strata-dequant`, `strata-plan`; `router_dot_parity` when `STRATA_NATIVE_EXPERTS` |
| `STRATA_ENABLE_CUDA` or `STRATA_ENABLE_HIP` (gfx906 counts as CUDA) | + `strata_core`, `strata_kernels`, `strata_engine`, `strata_prefill` | + `strata`, `strata-device`, `strata-load`, `strata-concurrent`, `strata-overlap`, and the GPU parity programs |
| GPU + `STRATA_NATIVE_EXPERTS` (default ON) on CUDA, or HIP + `STRATA_PREFILL_MMQ` | + `strata_mmq` | + `native_expert_parity`, `native_expert_bench` |
| HIP with hipBLASLt found | (`strata_prefill` links `roc::hipblaslt`) | + `tune_hipblaslt` |
| `STRATA_HIP_GFX906` | + imported `CUDA::cudart`, `CUDA::cublas` (INTERFACE aliases of HIP) | as CUDA |
| `STRATA_ENABLE_SYCL` | none here: `sycl/CMakeLists.txt` defines its own (see [../sycl/README.md](../sycl/README.md)) | |

A CPU-only configure of this snapshot (no GPU option, `STRATA_NATIVE_EXPERTS=OFF`, `STRATA_BUILD_TESTS=ON`) was run
for this page with CMake 3.28 and GCC 13: it produced exactly the "Always" row plus the 14 CPU tests listed in
[tests-registration.md](tests-registration.md#registered-tests-per-configuration).

None of the GPU programs and parity tests is `EXCLUDE_FROM_ALL`, so a plain `cmake --build build` compiles all of
them; the installer and every recipe in upstream's docs build `--target strata` only
([installer-builds.md](installer-builds.md)).

## Libraries

### `strata_warnings` (INTERFACE)

`CMakeLists.txt:194-208`. Carries the warning flags; see [cmake-options.md](cmake-options.md#warning-flags-the-strata_warnings-interface-target).

### `strata_artifact` (INTERFACE, header-only)

`CMakeLists.txt:214-216`. Include path `include/` plus `strata_warnings`. Holds the GGUF reader and the scalar
dequantizers (`include/strata/artifact/`). The comment explains the choice: they "hold no state worth hiding", and a
header-only boundary cannot introduce duplicate-symbol bugs into validated code (`CMakeLists.txt:211-213`).

### `strata_plan` (INTERFACE, header-only)

`CMakeLists.txt:239-241`. The memory planner; header-only so its decision is "arithmetic you can check by hand"
(`CMakeLists.txt:237-238`).

### `strata_core` (STATIC, GPU builds)

`CMakeLists.txt:249-259`. Sources: `src/core/device.cu`, `src/core/pinned.cu`, `src/platform/memory.cpp`,
`src/core/graph.cpp`, `src/core/weights.cpp`, `src/core/layout.cpp`. PUBLIC links: `strata_plan`, `strata_warnings`,
and the selected GPU runtime target. The runtime is PUBLIC because `graph.cpp` is a host `.cpp` that calls
`cudaGraphLaunch` / `cudaEventQuery`, and `.cpp` files do not get cudart from the CUDA language rules
(`CMakeLists.txt:256-258`). Gets `STRATA_EXPERIMENTAL_SM60=1` privately for the runtime device check in
`src/core/device.cu:280-287`.

### `strata_kernels` (STATIC, GPU builds)

`CMakeLists.txt:290-336`. 44 `.cu` files under `src/kernels/cuda/` (dequantizers, GEMVs, router, sampler, RoPE,
DeltaNet (GDN), QSA attention and selection, KV formats, PLE, the 14 `native_*` llama.cpp-contract kernels, verify and
i-quant kernels) plus three host files: `src/kernels/ngram.cpp`, `src/ngram/ple_reader.cpp`,
`src/platform/direct_file.cpp`. PUBLIC links `strata_core` and `strata_artifact` (the validated scalar dequantizers
are the parity reference, `CMakeLists.txt:287-289`). PRIVATE include `third_party/ggml` for `ggml-common.h`
(`CMakeLists.txt:363`).

### `strata_engine` (STATIC, GPU builds)

`CMakeLists.txt:417-422`. The composed layer and the session: `src/core/layer.cpp`, `session.cpp`,
`expert_source.cpp`, `remote_experts.cpp`, `remote_expert_opt.cu`, `expert_cache.cpp`, `native_head.cpp`,
`native_dense.cpp`, `verify.cpp`, `mtp.cpp`, `conversation_snapshot.cpp`, `conversation_state.cpp`,
`conversation_memory.cpp`, `peer_experts.cpp`. PUBLIC links `strata_core`, `strata_kernels`,
`strata_kernels_cpu`, the GPU runtime. It is a third library because putting the layer in `strata_core` would make
core and kernels depend on each other (`CMakeLists.txt:413-416`).

### `strata_prefill` (STATIC, GPU builds)

`CMakeLists.txt:484-493`. Batched prompt processing: `src/prefill/gemm.cu`, `kernels.cu`, `prefill.cpp`. PUBLIC links
`strata_engine`, the GPU BLAS target(s) and runtime; on HIP with hipBLASLt also `roc::hipblaslt` and
`STRATA_HIPBLASLT_AVAILABLE=1`. Later gains `strata_mmq` as a PUBLIC dependency (`CMakeLists.txt:1079`, `1129`).

### `strata_spec` (STATIC, always)

`CMakeLists.txt:899-900`. Speculation components that need no GPU: `src/spec/suffix_drafter.cpp`, `controller.cpp`,
`draft_policy.cpp`. Include path only; no links.

### `strata_kernels_cpu` (STATIC, always)

`CMakeLists.txt:927-942`, extended at `1006-1029`. Base sources: `src/kernels/cpu/expert.cpp`, `pool.cpp`,
`expert_layout.cpp`, `q2_avx2.cpp`. With `STRATA_NATIVE_EXPERTS`: `native_expert.cpp`, `iq_avx512.cpp`, `iq_avx2.cpp`,
`kq_avx2.cpp`, `kq_avx1.cpp`, PUBLIC `ggml-cpu ggml-base`, PRIVATE include `third_party/ggml`. Links
`Threads::Threads`. The native sources are in this library rather than a separate one so that GNU ld's one-pass static
link has no cycle (`CMakeLists.txt:1004-1005`).

Per-file ISA flags (the pool and the probe stay generic x86-64; the engine picks kernels at run time):

| File | MSVC | GCC / Clang |
| --- | --- | --- |
| `expert.cpp`, `iq_avx512.cpp` | `/arch:AVX512` | `-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi -mf16c -mfma` |
| `q2_avx2.cpp`, `iq_avx2.cpp`, `kq_avx2.cpp` | `/arch:AVX2` | `-mavx2 -mfma -mf16c` |
| `kq_avx1.cpp` | `/arch:AVX` | `-mavx` |

The comment at `CMakeLists.txt:887-897` explains why the flag is on the source, not the target: a TU compiled with
`/arch:AVX512` may use AVX-512 anywhere in it, so the scalar fallback must live in a different TU. VBMI needs its own
flag on GCC/Clang (`-mavx512vnni` is not enough, `CMakeLists.txt:936`).

### `strata_mmq` (STATIC)

Two definitions, one per backend:

- **CUDA** (`CMakeLists.txt:1042-1099`): `src/prefill/moe_mmq.cu`, `moe_fused.cu`, `moe_fused_iq.cu`,
  `ggml_cuda_host.cu`, and from the llama.cpp checkout `ggml/src/ggml-cuda/quantize.cu` plus one
  `template-instances/mmq-instance-<type>.cu` per type in `q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0`
  (+ `q5_0` with `STRATA_ORCA_Q4KS_MMQ`, + `q4_k q5_k q5_1` with `STRATA_MMQ_KQUANTS`). Links `ggml-base` and the GPU
  runtime. Makes `strata_prefill` define `STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1`.
- **HIP** with `STRATA_PREFILL_MMQ` (`CMakeLists.txt:1100-1139`): `moe_mmq.cu`, `ggml_cuda_host.cu`, `quantize.cu` and
  the same instance list, all forced to `LANGUAGE HIP`, `GGML_USE_HIP=1`; links `ggml-base hip::host roc::hipblas`.
  Fails at configure if the checkout lacks `common.cuh` / `quantize.cu`. No `moe_fused*.cu` (comment at
  `CMakeLists.txt:1052-1053`: the fused int8 experts are CUDA only).

### Imported `CUDA::cudart` / `CUDA::cublas` (gfx906 only)

`CMakeLists.txt:113-120`: INTERFACE IMPORTED targets that link `hip::host` / `roc::hipblas`, define
`STRATA_HIP_GFX906=1 __HIP_PLATFORM_AMD__=1`, and put `include/strata/platform/hip_compat/` first on the include path.
Every CUDA target then builds unchanged. See [gpu-backends.md](gpu-backends.md#amd-gfx906-wave64-the-cuda-sources-as-hip).

### `strata_hip_runtime` (INTERFACE, HIP only)

Defined in `cmake/hip_backend.cmake:73-94`; see [gpu-backends.md](gpu-backends.md#amd-wave32-cmakehip_backendcmake).

## Programs (not tests)

| Target | Source | Built when | What it is |
| --- | --- | --- | --- |
| `strata` | `src/program/generate.cpp` | GPU | The engine process (`strata generate` / serve loop); links `strata_engine strata_prefill strata_spec` + runtime (`CMakeLists.txt:495-497`). The only target the installer builds |
| `strata-gguf` | `src/artifact/gguf_reader.cpp` | always | GGUF reader CLI |
| `strata-dequant` | `src/artifact/dequant.cpp` | always | Dequantize a GGUF tensor to F32 (main behind `STRATA_DEQUANT_SELFTEST`) |
| `strata-plan` | `src/plan/plan_main.cpp` | always | Print the memory plan "or say why it does not close" |
| `strata-device` | `src/core/device_main.cpp` | GPU | Report the GPU and plan, exercise the arena; also run by the `*_device_selftest` CTest. `tools/hip/build_windows.bat` ships it with `strata` |
| `strata-load` | `src/core/load_main.cpp` | GPU | Expert arena load timing (`expert_arena_load` test when `pack/full/experts.bin` exists) |
| `strata-concurrent` | `src/core/concurrent_main.cpp` | GPU | Do the CPU expert path and the GPU stream run at once? |
| `strata-overlap` | `src/core/overlap_main.cpp` | GPU | Does streaming an expert overlap with compute? |
| `tune_hipblaslt` | `tools/hip/tune_hipblaslt.cpp` | HIP + hipBLASLt | Calibrates the hipBLASLt solution table (`docs/AMD_HIP.md`, "Tuning table") |
| `fused_gr_bench` | `src/kernels/fused_gr_bench.cpp` | GPU | AMD fast GR kernels vs old, bitwise + timing (no CTest) |
| `native_expert_bench` | `src/kernels/native_expert_bench.cpp` | GPU + native experts | Grouped native experts on real GGUF rows, one `STRATA_EXP_MODE` against another |
| `hip_ple_iq4` | `tests/hip/ple_iq4.cpp` | HIP + native experts | Optional real-model PLE diagnostic (no CTest) |
| `dequant_bf16_test` | `src/kernels/dequant_bf16_test.cpp` | GPU + `STRATA_BUILD_TESTS` | Device vs CPU dequantizers on the real 38 GB shards; "not a ctest" (`CMakeLists.txt:1202-1207`) |
| `ple_q5_parity`, `q5_projection_parity`, `ple_fp8_parity` | `src/kernels/*.cpp` | CUDA + native + `STRATA_BUILD_TESTS` | Real-weight checks for OrcaRouter Q4_K_S and the FP8 PLE table; "not part of a release" |
| `qsa_prompt_attn_parity` | `src/kernels/qsa_prompt_attn_parity.cpp` | CUDA + (`STRATA_BUILD_TESTS` or `STRATA_PARITY_PROMPT_ATTN`) | Tensor-core prompt attention vs FP32 / FP64 (no CTest on CUDA) |
| `hit_cpu_order_parity` | `bench/micro/hit_cpu_order_parity.cu` | CUDA + tests + file exists | Absent in this snapshot |

The image encoder `strata-vision` is not part of this project: it is `tools/vision/CMakeLists.txt`, a separate CMake
project (min 3.21) that `add_subdirectory`s a whole llama.cpp checkout (`LLAMA_DIR`) with `LLAMA_BUILD_MTMD=ON` and
links `mtmd llama`; options `STRATA_VISION_CUDA` and its own `STRATA_PORTABLE`. The installer builds it next to the
engine ([installer-builds.md](installer-builds.md)).

## Link graph

```
strata ──> strata_engine ──> strata_core ──> strata_plan, strata_warnings, <gpu runtime>
   │            ├──> strata_kernels ──> strata_core, strata_artifact
   │            └──> strata_kernels_cpu ──> Threads, [ggml-cpu, ggml-base]
   ├──> strata_prefill ──> strata_engine, <gpu blas>, <gpu runtime>, [strata_mmq ──> ggml-base, <gpu runtime>]
   └──> strata_spec
```

`<gpu runtime>` is `CUDA::cudart` (CUDA and gfx906) or `strata_hip_runtime` (HIP); `<gpu blas>` is `CUDA::cublas` or
`roc::hipblas` (+ `roc::rocblas` with hipBLAS older than 1.0, `cmake/hip_backend.cmake:47-53`).

## Gotchas

- Host `.cpp` programs that call the CUDA runtime themselves need `_strata_gpu_include_directories` and the runtime
  target added explicitly; the CUDA language rules only cover `.cu` sources (`CMakeLists.txt:578-581`, `272`).
- The `.cu` suffix does not make CMake pick HIP: both AMD paths glob the `.cu` files and set `LANGUAGE HIP` on them
  (`cmake/hip_backend.cmake:96-103`, `CMakeLists.txt:1244-1254`).

## In UnbiasedStrata

All of these targets are kept. The fork adds the program `unbiased-strata` (`frontend/main.cpp`,
`frontend/engine_entry.cpp`; CUDA + native experts only), which links the engine libraries with llama.cpp's `llama`,
`llama-common`, `mtmd` and `cpp-httplib` from the submodule, plus test programs for Qwen3.6
(`model_geometry_test`, `vram_plan_test`, `pool_native_widths_test`, `pack_load_check`, `qwen36_*`, `qsa_dense_parity`,
`qsa_attn_bench`, `prefill_qwen36_test`). `strata-vision` / `tools/vision` and `tools/hip` are removed. See
[../../codebase/build/targets.md](../../codebase/build/targets.md).

## See also

[cmake-options.md](cmake-options.md), [gpu-backends.md](gpu-backends.md), [tests-registration.md](tests-registration.md),
[third-party.md](third-party.md)
