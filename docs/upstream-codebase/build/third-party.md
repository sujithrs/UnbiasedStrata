[Upstream docs](../README.md) › [Build](README.md) › Third-party code

# Third-party code: the pinned llama.cpp / ggml and `third_party/ggml/`

## Purpose

Where the build gets llama.cpp's code, which parts of it are compiled into Strata, and what is vendored in the
repository. Files covered:
[CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/CMakeLists.txt) (lines 944-1184),
[third_party/ggml/ggml-common.h](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/third_party/ggml/ggml-common.h),
[third_party/ggml/VERSION.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/third_party/ggml/VERSION.txt),
[third_party/ggml/LICENSE](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/third_party/ggml/LICENSE).

## The pinned commit

One llama.cpp commit is used everywhere:

| Place | Value |
| --- | --- |
| `CMakeLists.txt:996` (`FetchContent_Declare ... GIT_TAG`) | `3cf03257f219afbe7334045ff7c6a06ac68c627d` |
| `third_party/ggml/VERSION.txt` | `3cf03257f219afbe7334045ff7c6a06ac68c627d Sun Sep 20 16:08:11 2026 +0800` |
| `setup.py:93` (`LLAMA_CPP_COMMIT`) | same; the installer downloads `https://github.com/ggml-org/llama.cpp/archive/<commit>.zip` |
| `sycl/CMakeLists.txt:115` | same |

`setup.py`'s `source_hash()` seeds its engine fingerprint with this commit (`setup.py:2235-2243`), so changing the pin
forces a recompile of an installed engine.

## How ggml enters the build (`STRATA_NATIVE_EXPERTS=ON`, the default)

`CMakeLists.txt:957-1003`:

1. `enable_language(C)`; ggml options forced: static, no CUDA backend, no OpenMP, no dynamic backends, no tests or
   examples; the CPU ISA chosen by `STRATA_PORTABLE` / `STRATA_ISA_FLOOR` / native
   ([cmake-options.md](cmake-options.md#cpu--ggml-options)).
2. Source:
   - `STRATA_GGML_DIR` set: `add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)`.
     The installer always uses this path, with its downloaded `third_party/llama.cpp`.
   - Empty: `FetchContent_Declare(strata_llamacpp GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git GIT_TAG
     3cf0325... GIT_SHALLOW FALSE)`, `FetchContent_Populate` if not yet populated, then the same `add_subdirectory` of
     its `ggml/`. `GIT_SHALLOW FALSE` means a full-history clone of llama.cpp at configure time.
3. Only `ggml-cpu` and `ggml-base` are linked (PUBLIC on `strata_kernels_cpu`); `EXCLUDE_FROM_ALL` keeps the rest of
   ggml out of the default build.

What Strata uses from ggml-cpu: the CPU computes a native (IQ / K-quant) expert with ggml-cpu's own dot products
(`ggml_get_type_traits_cpu`), "so an IQ weight means exactly what it means in llama.cpp" (`CMakeLists.txt:944-948`).

### The ggml-cuda MMQ sources (prompt path)

With CUDA (and with HIP + `STRATA_PREFILL_MMQ`), `strata_mmq` compiles files straight out of the same checkout's
`ggml/src/ggml-cuda/` (`CMakeLists.txt:1043-1139`): `quantize.cu` and `template-instances/mmq-instance-<type>.cu` for
the pack types, with include paths `ggml/include`, `ggml/src`, `ggml/src/ggml-cuda`. The few host symbols those need
from `ggml-cuda.cu` are reimplemented in `src/prefill/ggml_cuda_host.cu`. No full ggml-cuda backend is built
(`GGML_CUDA=OFF`). The HIP branch checks that `common.cuh` and `quantize.cu` exist in the checkout and stops otherwise.

### Test oracles

`STRATA_ORACLE_SOURCE_DIR` / `STRATA_ORACLE_BUILD_DIR` point at a separately built shared llama.cpp (CUDA) whose
`ggml`, `ggml-base`, `ggml-cpu`, `ggml-cuda` libraries the `bench/micro/*_parity` programs link
(`CMakeLists.txt:387-410`). `bench/micro/` is not in the published tree.

## Vendored files: `third_party/ggml/`

| File | Lines | Content |
| --- | --- | --- |
| `ggml-common.h` | 1911 | ggml's quantization block structs (`block_q8_0`, `block_iq2_xs`, ...; `QK_K 256` at line 89) with `static_assert`ed sizes, and the i-quant codebook grids (`iq2xs_grid`, ...). Language selected by `GGML_COMMON_DECL_C`, `_CPP`, `_METAL`, `_CUDA`, `_HIP`, ... (top of file) |
| `VERSION.txt` | 1 | The commit and date it was taken from |
| `LICENSE` | 21 | MIT, "Copyright (c) 2023-2026 The ggml authors" |

The header is on the PRIVATE include path of `strata_kernels` (`CMakeLists.txt:362-363`, comment "plan v0.3 P6:
llama.cpp's block layouts and codebook grids for the i-quant formats") and of `strata_kernels_cpu`
(`CMakeLists.txt:1009`). It is included by:

| Source | Mode |
| --- | --- |
| `src/kernels/cuda/iq_kernels.cu:14-16` | `GGML_COMMON_DECL_CUDA` |
| `src/kernels/cpu/iq_avx512.cpp:13-15`, `iq_avx2.cpp:15-17`, `kq_avx2.cpp:16-18` | `GGML_COMMON_DECL_CPP` |
| `src/prefill/moe_fused_iq.cu:29-31` | `GGML_COMMON_DECL_CUDA`; this file is in `strata_mmq`, whose include path has the checkout's `ggml/src`, so it resolves to the checkout's own copy rather than the vendored one |
| `sycl/src/kernels/cuda/iq_kernels.dp.cpp:18` | SYCL port (see [../sycl/README.md](../sycl/README.md)) |

Whether the vendored copy is byte-identical to the checkout's `ggml/src/ggml-common.h` at that commit was not checked
here (no llama.cpp checkout was available).

`third_party/ggml` is part of `setup.py`'s `ENGINE_SOURCES` (`setup.py:2231`), so editing it triggers a rebuild of an
installed engine.

## Other llama.cpp uses outside this CMake project

- `tools/vision/CMakeLists.txt` adds the whole checkout (`LLAMA_DIR`) to build `mtmd` and `llama` for the image encoder.
- Python tools import `third_party/llama.cpp/gguf-py`; `tools/iq_fixture.py:34` puts it on `sys.path`, and the
  `iq_parity_fixtures` test exits 3 (a skip) when it or numpy is missing.
- `.gitignore` and `.dockerignore` both exclude `third_party/llama.cpp` and `third_party/*.zip`: the checkout is a
  download, not source.

Licensing summary from upstream: `docs/HOW_IT_WORKS.md:74-76` and the credits in
[DETAILS.md](../../upstream/DETAILS.md) name `third_party/ggml` (MIT, llama.cpp / ggml).

## Gotchas

- Without `STRATA_GGML_DIR`, configure needs network access to GitHub and clones the full llama.cpp history.
- `FetchContent_Populate` with a single argument is deprecated from CMake 3.30 (policy CMP0169), so newer CMake
  versions print a deprecation warning on this path.
- The `GGML_*` options are cache `FORCE` sets; they overwrite anything given on the command line.
- `STRATA_NATIVE_EXPERTS=OFF` drops ggml entirely, and with it `strata_mmq`, the IQ CPU kernels and the native-expert
  tests; `STRATA_PREFILL_MMQ` and `STRATA_ORCA_Q4KS_MMQ` refuse that combination.

## In UnbiasedStrata

`third_party/ggml/` is kept unchanged. The `FetchContent` branch is gone: `third_party/llama.cpp` is a git submodule
at the same commit `3cf0325` (`.gitmodules`, shallow), `STRATA_GGML_DIR` defaults to it, and configure fails with
"run `git submodule update --init` first" if it is empty (1b59e5e). The fork applies
`patches/llama.cpp-qwen3-coder-any-argument-order.patch` to the submodule at configure time (e73adc6) and adds the
whole llama.cpp (`llama`, `llama-common`, `mtmd`, `cpp-httplib`) to the build for its `unbiased-strata` front end. See
[../../codebase/build/third-party-and-patches.md](../../codebase/build/third-party-and-patches.md).

## See also

[cmake-options.md](cmake-options.md), [targets.md](targets.md), [installer-builds.md](installer-builds.md)
