[Upstream docs](../README.md) › [Build](README.md) › GPU backends

# GPU backends in the build: CUDA archs, HIP wave32, gfx906, SYCL

## Purpose

How the build selects and checks a GPU target: the CUDA architecture policy (sm_120 default, sm_75 floor, the
experimental sm_60 / sm_70 build, the CUDA 12.8 warning for RTX 50), the wave32 AMD backend in
`cmake/hip_backend.cmake`, the wave64 gfx906 build that compiles the CUDA sources as HIP, and the hand-off to the
Intel SYCL project. Runtime behaviour of the kernels on each platform belongs to the kernel and platform areas; this
page covers what CMake does.

Files covered:
[CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/CMakeLists.txt) (lines 12-22, 58-185, 337-385, 499-575, 1100-1144, 1244-1254),
[cmake/hip_backend.cmake](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/cmake/hip_backend.cmake),
[include/strata/kernels/dp4a.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/dp4a.hpp) (the SM60 intrinsic fallbacks),
[tools/hip/build_windows.bat](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/hip/build_windows.bat).

## Backend selection

```
STRATA_ENABLE_SYCL ─ yes ─> add_subdirectory(sycl); return()                  CMakeLists.txt:14-22
        │ no
STRATA_HIP_GFX906 ── yes ─> STRATA_ENABLE_CUDA=ON (FORCE); HIP branch below    CMakeLists.txt:64-70, 106-125
        │ no
STRATA_ENABLE_HIP ── yes ─> include(cmake/hip_backend.cmake)                    CMakeLists.txt:84-86, 181-185
        │ no
STRATA_ENABLE_CUDA ─ yes ─> enable_language(CUDA), arch checks                 CMakeLists.txt:126-180
        │ no
CPU-only targets (no strata_core / engine / strata)
```

The result of the branch is three variables used by every GPU target:

| Variable | CUDA | gfx906 | HIP wave32 |
| --- | --- | --- | --- |
| `_strata_gpu_runtime_target` | `CUDA::cudart` | `CUDA::cudart` (alias of `hip::host`) | `strata_hip_runtime` |
| `_strata_gpu_blas_target` | `CUDA::cublas` | `CUDA::cublas` (alias of `roc::hipblas`) | `${STRATA_HIP_BLAS_TARGETS}` |
| `_strata_gpu_include_directories` | `${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}` | `include/strata/platform/hip_compat` | `include/strata/hip_compat` |

## NVIDIA CUDA

### Default architecture and the per-arch checks

`CMakeLists.txt:139-171`:

1. If `CMAKE_CUDA_ARCHITECTURES` is not defined it is set to `120` (RTX 50) in the cache.
2. Each list entry is reduced to its number (`120-real` and `120-virtual` count as 120). `native`, `all` and
   `all-major` are skipped without checks.
3. Below 75 without `STRATA_EXPERIMENTAL_SM60`: fatal error, "Strata needs an NVIDIA GPU of compute capability 7.5 or
   newer (RTX 20 / 30 / 40 / 50 series)".
4. Below 60: fatal even with the experimental flag.
5. 120 or above with a CUDA compiler older than 13.0: a warning that engines built with CUDA 12.8 crashed on long
   prompts on RTX 50 cards (issues #220, #224). It is a warning, not an error.

The "Turing port" comment (`CMakeLists.txt:144-146`): development and measurement were on sm_120; the QSA scorer's
tf32 `mma` has a portable fp32-FMA fallback, so sm_75 (RTX 20), 86 (RTX 30), 89 (RTX 40) and 120 build too.

The runtime half of the policy is in `src/core/device.cu:280-290`: the engine refuses a card below compute capability
7.5 (6.0 in the experimental build), because a binary can be carried to another machine.

### The experimental Pascal / Volta build (`STRATA_EXPERIMENTAL_SM60`)

| Aspect | Detail |
| --- | --- |
| Option | `STRATA_EXPERIMENTAL_SM60`, OFF, "community build for GPUs below compute capability 7.5: Pascal sm_60, Volta sm_70 (#236)" (`CMakeLists.txt:138`) |
| Toolkit | CUDA 12.x: CUDA 13 dropped sm_60 and sm_70, and nvcc then fails CMake's compiler test with "compute_70" (`CMakeLists.txt:154-155`) |
| Definitions | `STRATA_EXPERIMENTAL_SM60=1` on every target (`CMakeLists.txt:172-176`), so the Pascal / Volta code paths (e.g. the Volta prompt attention, PR #600) are compiled into this build only |
| Intrinsics | `include/strata/kernels/dp4a.hpp`: `__dp4a` (6.1+) replaced below sm_61 by a bit-exact signed-byte loop (`STRATA_DP4A`, lines 26-35); `__nanosleep` (7.0+) replaced below sm_70 by a plain spin (`strata_spin_pause`, lines 39-45) |
| Windows | MSVC + this build links the shared CUDA runtime unless `CMAKE_CUDA_RUNTIME_LIBRARY` is given (`CMakeLists.txt:127-133`, #585: LNK1169 otherwise) |
| Who uses it | `setup.py`'s `engine_defs()` adds `-DSTRATA_EXPERIMENTAL_SM60=ON` when any target arch is below 75 or the toolkit is CUDA 12 (`setup.py:2254-2257`); the result goes to `engine-cuda12/` |

Hand-build recipes in upstream's docs: `docs/OLDER_GPUS.md` ("Building it yourself": archs `61;70` with CUDA 12.9's
nvcc) and `docs/NVIDIA_V100.md` (arch 70, CUDA 12.8 used). Both pass `-DSTRATA_BUILD_TESTS=OFF` and build
`--target strata`.

### CUDA-only per-file flags and targets

- `--use_fast_math` on the 14 `native_*.cu` kernels (not on gfx906) (`CMakeLists.txt:364-379`).
- `strata_mmq` with the fused int8 experts (`moe_fused*.cu`), `--extended-lambda -use_fast_math`
  (`CMakeLists.txt:1042-1099`); see [targets.md](targets.md#strata_mmq-static).
- CUDA-only tests: `qsa_topk_active_parity`, `mmvq_multi_parity`, `gdn_rec_parity`, `gemm_bf16_parity`,
  `decode_cluster_parity`, `cuda_device_selftest`, `prefill_fused_*`, `native_expert_parity_*`
  ([tests-registration.md](tests-registration.md)).
- `STRATA_CUDA_TESTS=ON` is set for `tests/CMakeLists.txt` (`CMakeLists.txt:880-884`), which is absent in this
  snapshot.

## AMD wave32: `cmake/hip_backend.cmake`

Included when `STRATA_ENABLE_HIP=ON`. Target: wave32 RDNA2 / RDNA3 / RDNA4 with 64 KiB LDS per workgroup and a signed
dot4 instruction (`cmake/hip_backend.cmake:1-3`).

| Step | Lines | What happens |
| --- | --- | --- |
| Default arch | 4-6 | `CMAKE_HIP_ARCHITECTURES=gfx1100` if empty or undefined |
| Arch lists | 12-14 | validated: `gfx1100 gfx1201`; community-validated: `gfx1101 gfx1200`; unvalidated (warning): `gfx1012 gfx1102 gfx1030 gfx1031` |
| Normalise | 15-35 | spaces become `;`; a feature suffix is dropped (`gfx1100:xnack-` -> `gfx1100`); any other arch is a fatal error |
| Runtime list | 40-42 | `STRATA_HIP_ARCHS="gfx1100,gfx1201"`; `src/core/device.cu:28-70` compares each card's `gcnArchName` with it so a binary on a card it has no code for stops with a clear message |
| Packages | 44-67 | `enable_language(HIP)`; `hip` and `hipblas` (CONFIG, required); `rocblas` too when hipBLAS is older than 1.0 (no workspace API); `hipblaslt` optional, sets `STRATA_HIPBLASLT_AVAILABLE` |
| PIC | 69-70 | `CMAKE_POSITION_INDEPENDENT_CODE ON` (ROCm links a PIE) |
| `strata_hip_runtime` | 72-94 | INTERFACE target: `include/strata/hip_compat` before `include/`, `STRATA_USE_HIP=1`, `STRATA_HIP_ARCHS`, links `hip::host`, force-includes `include/strata/hip_compat/cuda_runtime.h` into every CXX and HIP source (`/FI` on an MSVC-style front end, `-include` otherwise) |
| Language | 96-108 | every `src/**.cu`, `bench/**.cu`, `tests/**.cu` and `tests/hip/intrinsics.cpp`, `tests/hip/native_qsa_score.cpp` compiled as HIP |

HIP-specific parts of `CMakeLists.txt`:

- `-ffp-contract=off` on `elementwise.cu` and `quantize_act.cu` (`CMakeLists.txt:380-385`).
- `strata_mmq` as HIP only with `STRATA_PREFILL_MMQ=ON` (no fused int8 experts), `CMakeLists.txt:1100-1139`. The
  installer always passes it for AMD.
- `tune_hipblaslt` and `roc::hipblaslt` on `strata_prefill` when hipBLASLt is found (`CMakeLists.txt:487-493`).
- The HIP test block `CMakeLists.txt:499-575` (registered without `STRATA_BUILD_TESTS`), `qsa_select_bench`,
  `hip_prefill_mmq_parity`, `hip_device_selftest`.
- The oracle parity tests are CUDA only: "do not pull CUDA libraries into a HIP build" (`CMakeLists.txt:389-390`).

Windows: `tools/hip/build_windows.bat` installs ROCm from AMD's TheRock Python wheels into a private venv, uses
ROCm's clang/clang++ for host and device code with MSVC's linker and SDK from `vcvars64.bat` ("CMake refuses to mix
cl.exe with Clang for HIP", lines 9-11), configures with `STRATA_ENABLE_HIP=ON STRATA_PREFILL_MMQ=ON
STRATA_NATIVE_EXPERTS=ON STRATA_PORTABLE=ON` for `gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1030` by default, builds
`strata strata-device` (or everything with `tests`) and packages `dist/strata-windows-x64-hip.zip` through
`tools/hip/package_windows.py`.

## AMD gfx906 (wave64): the CUDA sources as HIP

`STRATA_HIP_GFX906=ON` targets Instinct MI50 / MI60 and Radeon VII. It is not the wave32 backend: it turns
`STRATA_ENABLE_CUDA` on, leaves `STRATA_ENABLE_HIP` off, and makes HIP impersonate CUDA, so every "CUDA" branch of the
file is the one taken (`CMakeLists.txt:59-63`).

| Step | Lines | What happens |
| --- | --- | --- |
| Arch | 107-109 | `CMAKE_HIP_ARCHITECTURES=gfx906` if undefined (no validation list) |
| Packages | 110-112 | `enable_language(HIP)`, `find_package(hip)`, `find_package(hipblas)`; `enable_language(CUDA)` is never called |
| Aliases | 113-118 | `CUDA::cudart` -> `hip::host` with `STRATA_HIP_GFX906=1 __HIP_PLATFORM_AMD__=1`; `CUDA::cublas` -> `roc::hipblas` |
| Shim | 116, 119-121 | `include/strata/platform/hip_compat/` (headers `cuda_runtime.h`, `cublas_v2.h`, `cuda_fp16.h`, `math_constants.h`, `strata_hip.h`) first on every include path |
| Language | 1244-1254 | every `src/**.cu`, `bench/micro/*.cu`, `tools/vision/*.cu`, and (when MMQ is built) the ggml-cuda `*.cu` and template instances, set to `LANGUAGE HIP`; `strata_mmq` gets `GGML_USE_HIP` |
| Differences | 379, 643-646, 656, 734-739 | no `--use_fast_math` on the native kernels; an extra `mmvq_multi_parity_cuda_layout` test with `STRATA_MMVQ_WAVE=0`; no `gdn_rec_parity` / `gemm_bf16_parity` (cp.async and the CUDA occupancy API); `hip_router_fast` built from `tests/hip/router_fast.cpp` (no CTest) |

The device check takes the gfx arch from `gcnArchName` (`src/core/device.cu:265-274`); it does not compare it with
the compiled list. `setup.py` does not build this variant: the recipe is in
[docs/AMD_HIP.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/AMD_HIP.md)
("gfx906 (Instinct MI50 / MI60, Radeon VII): wave64, built from source"):

```sh
cmake -S . -B build-906 -G Ninja -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR=/src/third_party/llama.cpp \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
ninja -C build-906 strata
```

The doc notes AMD's current ROCm no longer ships gfx906 libraries; the measured build used the community image
`mixa3607/rocm-gfx906:7.14-complete`. The community MI50 benchmark
(`bench/results/2026-10-03-community-2x-mi50/README.md`) adds `-DSTRATA_PORTABLE=ON`.

## Intel SYCL

`-DSTRATA_ENABLE_SYCL=ON` on the top-level project is a pass-through: it refuses CUDA / HIP, warns that the Intel Arc
engine is experimental, runs `add_subdirectory(sycl)` and returns (`CMakeLists.txt:12-22`). `sycl/CMakeLists.txt`
(241 lines) is its own project (`strata_sycl`, version 0.1.39) built with `icpx`; it reuses `src/` files unless
`sycl/src/` has a migrated copy, and fetches the same pinned llama.cpp for ggml-cpu. Details:
[../sycl/README.md](../sycl/README.md).

## Gotchas

- The header comment at `CMakeLists.txt:7-9` still says the kernels need sm_80 and that older arches are refused; the
  code below it sets the floor at sm_75.
- `native`, `all` and `all-major` bypass the configure-time arch checks; only the runtime check in `device.cu` then
  guards a too-old card.
- `cmake/hip_backend.cmake:7-11` says gfx1030 ran in a community report (#311), yet it is in the "unvalidated" list
  and gets the warning.
- The Docker image's default arch list (`75;80;86;89;120`) is compiled with CUDA 13.0, so it never triggers the
  RTX 50 warning; a CUDA 12.x host build for sm_120 does.

## In UnbiasedStrata

The CMake logic for CUDA, wave32 HIP and gfx906 is kept unchanged (`cmake/hip_backend.cmake` is byte-identical). The
fork builds only CUDA through `make build`, with a default arch list `61;70;75;86;89;120` filtered by what
`nvcc --list-gpu-arch` reports and `STRATA_EXPERIMENTAL_SM60=ON` added automatically when an arch below 75 remains
(1e40b06). The AMD paths are not built or tested there, `tools/hip/` and `sycl/` are removed, and the top-level
`STRATA_ENABLE_SYCL` branch still points at a `sycl/` directory that no longer exists. See
[../../codebase/build/hip-compat.md](../../codebase/build/hip-compat.md).

## See also

- [cmake-options.md](cmake-options.md), [targets.md](targets.md), [installer-builds.md](installer-builds.md)
- Upstream: [docs/OLDER_GPUS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/OLDER_GPUS.md),
  [docs/NVIDIA_V100.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/NVIDIA_V100.md),
  [docs/AMD_HIP.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/AMD_HIP.md),
  [docs/INTEL_ARC.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INTEL_ARC.md)
