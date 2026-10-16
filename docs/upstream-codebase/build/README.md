[Upstream docs](../README.md) › Build

# Build and packaging

Upstream Strata (engine 0.1.39) is built by one top-level CMake project, `CMakeLists.txt` (1,254 lines, CMake 3.24+,
C++20). It turns the engine sources (`src/`, `include/`) into a set of static libraries (`strata_core`,
`strata_kernels`, `strata_kernels_cpu`, `strata_engine`, `strata_prefill`, `strata_spec`, `strata_mmq`) and the
user-facing program `strata` (the engine process that `serve/server.py` starts), plus small tools and many parity tests.
It builds for one GPU backend per build tree: NVIDIA CUDA (the default target, sm_120, from sm_75 up; sm_60 / sm_70
behind an experimental flag), AMD HIP for wave32 RDNA cards (`cmake/hip_backend.cmake`), AMD gfx906 (wave64) by
compiling the CUDA sources as HIP through a compat layer, or Intel SYCL by delegating to the separate project in
`sycl/`. ggml's CPU backend and llama.cpp's CUDA MMQ kernels come from llama.cpp at a pinned commit
(`3cf03257f219afbe7334045ff7c6a06ac68c627d`), either a local checkout (`-DSTRATA_GGML_DIR=`) or a `FetchContent`
clone; one header, `ggml-common.h`, is vendored in `third_party/ggml/`. Users normally never run CMake themselves:
the installer (`setup.py`, see [../install/README.md](../install/README.md)) downloads a ready-made engine, or calls
CMake with a fixed set of definitions when it has to compile one. The `Dockerfile` compiles the engine into an image
the same way.

## Contents

| Page | What it covers |
| --- | --- |
| [cmake-options.md](cmake-options.md) | Every `option()` and cache variable, its default, mutual exclusions, the compile definitions it produces; global compile flags, MSVC specifics, `STRATA_PORTABLE` / `STRATA_ISA_FLOOR` and native CPU flags |
| [targets.md](targets.md) | The libraries and programs: sources, link graph, which backend builds which target, per-file CPU ISA flags |
| [gpu-backends.md](gpu-backends.md) | CUDA architecture handling (sm_120 default, the Turing floor, `STRATA_EXPERIMENTAL_SM60`, the CUDA 12.8 / RTX 50 warning), the HIP wave32 backend, the gfx906 compat build, the SYCL hand-off |
| [third-party.md](third-party.md) | The pinned llama.cpp / ggml (local dir or `FetchContent`), the ggml options forced on it, the MMQ sources taken from it, and the vendored `third_party/ggml/` files |
| [tests-registration.md](tests-registration.md) | All 94 `add_test` calls grouped by gate, the CTest fixtures and properties, which tests run in which configuration |
| [installer-builds.md](installer-builds.md) | The exact CMake commands `setup.py`, the Docker build and `tools/hip/build_windows.bat` run, and the hand-build recipes in upstream's docs |
| [docker.md](docker.md) | `Dockerfile`, `docker-entrypoint.sh`, `.dockerignore`: image build, env-var interface, volume, health check |

## Source map

| File | Lines | Role |
| --- | --- | --- |
| [CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/CMakeLists.txt) | 1254 | The whole build: options, backend selection, all libraries, programs and test registrations |
| [cmake/hip_backend.cmake](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/cmake/hip_backend.cmake) | 110 | Wave32 HIP backend: arch validation, ROCm packages, `strata_hip_runtime` shim target, `.cu` files as HIP |
| [third_party/ggml/ggml-common.h](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/third_party/ggml/ggml-common.h) | 1911 | Vendored ggml block layouts and i-quant codebook grids |
| [third_party/ggml/VERSION.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/third_party/ggml/VERSION.txt) | 1 | The llama.cpp commit the header was taken from |
| [third_party/ggml/LICENSE](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/third_party/ggml/LICENSE) | 21 | ggml's MIT license |
| [Dockerfile](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/Dockerfile) | 115 | CUDA 13.0 image; compiles the engine (and image encoder) at `docker build` time through `setup.py`'s helpers |
| [docker-entrypoint.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docker-entrypoint.sh) | 65 | Container start: env vars to `setup.py --setup`, then `setup.py` (which starts the server) |
| [.dockerignore](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/.dockerignore) | 27 | Keeps build trees, downloads, `third_party/llama.cpp/` and `.git` out of the image context |
| [sycl/CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/CMakeLists.txt) | 241 | Pointer only: the Intel SYCL project, reached through `-DSTRATA_ENABLE_SYCL=ON` (see [../sycl/README.md](../sycl/README.md)) |
| [tools/vision/CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/vision/CMakeLists.txt) | 44 | Pointer only: the separate `strata-vision` image-encoder project the installer also builds |
| [tools/hip/build_windows.bat](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/hip/build_windows.bat) | 86 | Builds and packages the Windows HIP release zip (the only in-tree caller of `STRATA_PORTABLE=ON` for the engine) |

The build-related functions of `setup.py` (`cmake_build`, `build_engine`, `build_engine_hip`, `engine_defs`,
`isa_floor_defs`, `source_hash`, `get_llama_cpp`) are quoted in [installer-builds.md](installer-builds.md); the
installer itself is documented in [../install/README.md](../install/README.md).

## Key concepts and data flow

```
                         configure-time choice (one backend per build tree)
  -DSTRATA_ENABLE_SYCL=ON ──> add_subdirectory(sycl) ; return()      (Intel, separate project)
  -DSTRATA_ENABLE_CUDA=ON ──> enable_language(CUDA), arch checks, CUDA::cudart / CUDA::cublas
  -DSTRATA_HIP_GFX906=ON  ──> forces CUDA=ON, enable_language(HIP), CUDA::* = aliases of hip::host / roc::hipblas
  -DSTRATA_ENABLE_HIP=ON  ──> include(cmake/hip_backend.cmake): strata_hip_runtime, .cu compiled as HIP
  (none)                  ──> CPU-only: strata_artifact, strata_plan, strata_spec, strata_kernels_cpu, tools, CPU tests

  libraries (GPU build)                         llama.cpp @ 3cf0325 (STRATA_GGML_DIR or FetchContent)
  strata_artifact (hdr) ─┐                         │ ggml/ (add_subdirectory, EXCLUDE_FROM_ALL)
  strata_plan     (hdr) ─┼─> strata_core ─> strata_kernels ─┐   ├─> ggml-base, ggml-cpu ──> strata_kernels_cpu
                         │                                  ├─> strata_engine ─> strata_prefill ─> strata  (engine exe)
  strata_kernels_cpu ────┴──────────────────────────────────┘          ^             ^
  strata_spec ─────────────────────────────────────────────────────────┼─────────────┘
  ggml/src/ggml-cuda mmq + quantize.cu  ──> strata_mmq ────────────────┘ (PUBLIC dep of strata_prefill)
```

- **One backend per tree.** CUDA and HIP are refused together (`CMakeLists.txt:81-83`), gfx906 and HIP are refused
  together (`CMakeLists.txt:65-68`), and SYCL refuses both (`CMakeLists.txt:15-18`).
- **Backend-neutral link names.** After backend selection three variables carry the choice:
  `_strata_gpu_runtime_target`, `_strata_gpu_blas_target`, `_strata_gpu_include_directories`
  (`CMakeLists.txt:123-125`, `178-180`, `182-184`). Every GPU target links these instead of naming CUDA or HIP.
- **The ggml CPU backend is always static and built from llama.cpp**, with its ISA chosen by `STRATA_PORTABLE`,
  `STRATA_ISA_FLOOR` or `GGML_NATIVE` (`CMakeLists.txt:957-1003`).
- **Strata's own CPU kernels are per-file ISA**: AVX-512 / AVX2 / AVX flags are set on single source files, and the
  engine picks a kernel at run time (`CMakeLists.txt:930-940`, `1010-1022`).
- **Tests are opt-in in this snapshot.** `STRATA_BUILD_TESTS` defaults to ON only when `tests/CMakeLists.txt` and
  `bench/micro/` exist; the published tree has neither, so it defaults to OFF (`CMakeLists.txt:87-92`). Many GPU
  parity tests are registered regardless of that option; see [tests-registration.md](tests-registration.md).
- **There is no CI.** `.github/` holds only `FUNDING.yml`; nothing in the snapshot runs the build automatically.

Terms such as MMQ, QSA, PLE, native pack and i-quant are defined in the [glossary](../guides/glossary.md).

## In UnbiasedStrata

| Item | In the fork |
| --- | --- |
| `CMakeLists.txt` | Kept and extended (1,444 lines at HEAD). `FetchContent` removed: ggml comes from the git submodule `third_party/llama.cpp` (pinned to the same commit `3cf0325`) and configure fails if it is missing; a chat-parser patch (`patches/llama.cpp-qwen3-coder-any-argument-order.patch`) is applied at configure time (e73adc6); `q6_k` added to the `STRATA_MMQ_KQUANTS` MMQ types; Qwen3.6 test targets and `cmake/qwen36_*.cmake` fixture scripts added; a new target `unbiased-strata` (the C++ front end in `frontend/`, linking llama.cpp's `llama`, `llama-common`, `mtmd`, `cpp-httplib`; 1db37aa). Commits that touch it include 1b59e5e (submodule instead of `FetchContent`), 1db37aa (front end), e73adc6 (configure-time patch) and the Qwen3.6 test commits (e41dfc7, 8fc3b20, bbc3532, bc0b297, 22d72e6). |
| `Makefile` | New in the fork (1b59e5e): `make build` configures CUDA + `STRATA_PORTABLE` + `STRATA_MMQ_KQUANTS` for archs `61;70;75;86;89;120` (filtered to what nvcc supports, 1e40b06) and builds `unbiased-strata` only. |
| `cmake/hip_backend.cmake`, `third_party/ggml/` | Kept unchanged. |
| `Dockerfile`, `docker-entrypoint.sh` | Removed (8894a81, with the installer and the Python server). |
| `sycl/`, `tools/vision/`, `tools/hip/`, `setup.py` | Removed. |

Fork-side docs: [../../codebase/build/README.md](../../codebase/build/README.md).
