[Upstream docs](../README.md) › [Build](README.md) › CMake options

# CMake options and cache variables

## Purpose

Every configuration knob of upstream's top-level build, what it defaults to, what it refuses, and which compile
definitions or flags it produces. Also the project-wide settings (language standard, build type, warning flags,
MSVC-specific link and runtime settings) and the CPU ISA selection for ggml-cpu.

Files covered: [CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/CMakeLists.txt),
[cmake/hip_backend.cmake](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/cmake/hip_backend.cmake).

## Project-wide settings

| Setting | Where | Effect |
| --- | --- | --- |
| `cmake_minimum_required(VERSION 3.24)` | `CMakeLists.txt:10` | CMake 3.24 or newer |
| `project(strata VERSION 0.1.39 LANGUAGES CXX)` | `CMakeLists.txt:11` | Only CXX at first; CUDA, HIP and C are enabled later by the backend / native-expert branches |
| `STRATA_VERSION="0.1.39"` | `CMakeLists.txt:23` | Global compile definition; printed by `strata` (e.g. `src/program/generate.cpp:3867`) |
| `CMAKE_CXX_STANDARD 20`, required, no extensions | `CMakeLists.txt:51-53` | C++20 everywhere |
| Default `CMAKE_BUILD_TYPE=Release` | `CMakeLists.txt:54-56` | Only for single-config generators when none is given |
| MSVC Release: `/Zi` (CXX) and `/DEBUG /OPT:REF /OPT:ICF` (link) | `CMakeLists.txt:26-29` | Issue #31: a PDB beside the exe so a minidump of a stalled engine can be symbolized; `/OPT:REF /OPT:ICF` keep the binary small despite `/DEBUG` |

### Warning flags: the `strata_warnings` interface target

Warnings are attached to an INTERFACE library that the project's own targets link, not set globally, so that
fetched third-party code (the comment names Catch2) is not compiled with them (`CMakeLists.txt:187-208`). Every flag is
wrapped in a `COMPILE_LANGUAGE` generator expression, because nvcc given bare `/W4 /utf-8` fails with "A single input
file is required for a non-link phase" (`CMakeLists.txt:191-193`).

| Compiler / language | Flags |
| --- | --- |
| MSVC, CXX | `/W4 /utf-8` (`/WX` with `STRATA_WERROR`) |
| GCC / Clang, CXX | `-Wall -Wextra` (`-Werror` with `STRATA_WERROR`) |
| CUDA with MSVC host | `-Xcompiler=/W4` |
| HIP | `-Wall -Wextra` (`-Werror` with `STRATA_WERROR`) |

Only targets that link `strata_warnings` (directly: `strata_artifact`, `strata_plan`, `strata_core`; transitively the
GPU libraries and programs) get these flags. `strata_spec` and `strata_kernels_cpu` do not link it.

## Backend options

| Option | Default | Where | Effect |
| --- | --- | --- | --- |
| `STRATA_ENABLE_SYCL` | OFF | `CMakeLists.txt:14-22` | EXPERIMENTAL. Refuses CUDA / HIP, prints a warning, `add_subdirectory(sycl)` and `return()`: nothing else in the top-level file is configured. See [../sycl/README.md](../sycl/README.md) |
| `STRATA_ENABLE_CUDA` | OFF | `CMakeLists.txt:58` | Builds the GPU targets with CUDA (`enable_language(CUDA)`, `find_package(CUDAToolkit)`) |
| `STRATA_HIP_GFX906` | OFF | `CMakeLists.txt:64-70` | AMD gfx906 (wave64) build: forces `STRATA_ENABLE_CUDA=ON` in the cache, refuses `STRATA_ENABLE_HIP`, compiles the CUDA sources as HIP. Defines `STRATA_HIP_GFX906=1` globally (`CMakeLists.txt:121`) |
| `STRATA_ENABLE_HIP` | OFF | `CMakeLists.txt:71` | Wave32 AMD backend; includes `cmake/hip_backend.cmake` (`CMakeLists.txt:84-86`). Refused together with CUDA unless gfx906 (`CMakeLists.txt:81-83`) |
| `STRATA_EXPERIMENTAL_SM60` | OFF | `CMakeLists.txt:138` (declared inside the CUDA branch) | Admits sm_60 / sm_70 in `CMAKE_CUDA_ARCHITECTURES`; adds `STRATA_EXPERIMENTAL_SM60=1` to every target (`CMakeLists.txt:172-176`) and privately to `strata_core` (`CMakeLists.txt:252-255`); on MSVC switches CUDA to the shared runtime (`CMakeLists.txt:131-133`) |
| `CMAKE_CUDA_ARCHITECTURES` | `120` if undefined | `CMakeLists.txt:139-141` | Validated per entry; see [gpu-backends.md](gpu-backends.md) |
| `CMAKE_CUDA_RUNTIME_LIBRARY` | CMake default (static), `Shared` for MSVC + SM60 | `CMakeLists.txt:131-133` | Issue #585: MSVC + CUDA 12.x failed with LNK1169 (cudart defined twice) |
| `CMAKE_HIP_ARCHITECTURES` | `gfx1100` (HIP) / `gfx906` (gfx906 build) if undefined | `cmake/hip_backend.cmake:4-6`, `CMakeLists.txt:107-109` | Validated by `hip_backend.cmake` for wave32; not validated for gfx906 |
| `STRATA_GFX1012_PORTABLE_DOT` | OFF | `cmake/hip_backend.cmake:77-80` | Defines `STRATA_GFX1012_PORTABLE_DOT=1` on `strata_hip_runtime` (the portable signed-byte dot on gfx1012) |

## Prompt-path (MMQ) options

| Option | Default | Where | Effect |
| --- | --- | --- | --- |
| `STRATA_PREFILL_MMQ` | OFF | `CMakeLists.txt:72` | HIP only (refused otherwise, `CMakeLists.txt:78-80`; also needs `STRATA_NATIVE_EXPERTS`, `CMakeLists.txt:951-953`): compiles llama.cpp's MMQ instances as HIP into `strata_mmq` and defines `STRATA_PREFILL_MMQ=1` on `strata_prefill` (`CMakeLists.txt:1100-1139`). On CUDA, MMQ is always built when native experts are on, without this option |
| `STRATA_MMQ_KQUANTS` | OFF | `CMakeLists.txt:76` | CUDA: adds the `q4_k q5_k q5_1` MMQ instances (Unsloth UD-Q4_K_XL experts) and defines `STRATA_MMQ_KQUANTS=1` on `strata_mmq` (`CMakeLists.txt:1060-1062`, `1073-1075`). Off because the engine loads every kernel eagerly (`CUDA_MODULE_LOADING=EAGER`) and these cost about 5 MB of VRAM, "1-3 expert slots" (comment at `CMakeLists.txt:73-75`) |
| `STRATA_ORCA_Q4KS_MMQ` | OFF | `CMakeLists.txt:77` | Adds the `q5_0` MMQ instance for OrcaRouter Q4_K_S (#296), defines `STRATA_ORCA_Q4KS_MMQ=1` on `strata_mmq`. Needs `STRATA_NATIVE_EXPERTS` (`CMakeLists.txt:954-956`) |

## CPU / ggml options

| Option | Default | Where | Effect |
| --- | --- | --- | --- |
| `STRATA_NATIVE_EXPERTS` | ON | `CMakeLists.txt:949` | Builds ggml (static) from llama.cpp, adds the IQ / K-quant CPU kernels to `strata_kernels_cpu`, defines `STRATA_NATIVE_EXPERTS=1` (PUBLIC), and on CUDA builds `strata_mmq`. See [third-party.md](third-party.md) |
| `STRATA_GGML_DIR` | empty | `CMakeLists.txt:950` | A llama.cpp checkout to take `ggml/` from; empty means `FetchContent` of the pinned commit |
| `STRATA_PORTABLE` | OFF | `CMakeLists.txt:34` | "build for other PCs": ggml-cpu for an AVX2 baseline (`GGML_NATIVE OFF`, `GGML_AVX/AVX2/FMA/F16C/BMI2 ON`, `CMakeLists.txt:966-970`) and, on MSVC, the static runtime `MultiThreaded` (`CMakeLists.txt:47-49`) so no VC++ redistributable is needed |
| `STRATA_ISA_FLOOR` | `""` | `CMakeLists.txt:39-46` | EXPERIMENTAL older-CPU build: `avx` (Sandy/Ivy Bridge) or `none` (SSE4.2 + POPCNT). Any other value is a fatal error; excludes `STRATA_PORTABLE`. Sets ggml-cpu to that floor with every newer feature forced off (`CMakeLists.txt:971-986`) and defines `STRATA_ISA_FLOOR_AVX=1` or `STRATA_ISA_FLOOR_NONE=1` on `strata_kernels_cpu` (`CMakeLists.txt:1023-1027`) |
| (neither) | | `CMakeLists.txt:987-989` | `GGML_NATIVE ON`: ggml-cpu is compiled for the build machine's CPU |

The three ggml-cpu modes, side by side:

| Mode | ggml settings | Typical caller |
| --- | --- | --- |
| default | `GGML_NATIVE=ON` | `setup.py` compiling on the user's PC |
| `STRATA_PORTABLE=ON` | `GGML_NATIVE=OFF`; AVX, AVX2, FMA, F16C, BMI2 ON | release zips (comment at `CMakeLists.txt:31-33`); `tools/hip/build_windows.bat:69` |
| `STRATA_ISA_FLOOR=avx` | `GGML_NATIVE=OFF`, `GGML_CPU_ALL_VARIANTS=OFF`, SSE42 + AVX ON, AVX2 ... AMX all OFF | `setup.py` on a CPU without AVX2 (`isa_floor_defs`) |
| `STRATA_ISA_FLOOR=none` | as above but AVX OFF too | `setup.py` on an SSE4.2-only CPU |

ggml is always forced to `BUILD_SHARED_LIBS=OFF`, `GGML_STATIC=ON`, `GGML_CUDA=OFF`, `GGML_OPENMP=OFF`,
`GGML_BACKEND_DL=OFF`, `GGML_BUILD_TESTS=OFF`, `GGML_BUILD_EXAMPLES=OFF` (`CMakeLists.txt:959-965`). These are
`CACHE ... FORCE` sets, so a user's `-DGGML_...` for those names is overwritten.

`STRATA_PORTABLE` affects only ggml-cpu and the MSVC runtime. Strata's own CPU kernels are compiled per file with
fixed ISA flags in every mode (see [targets.md](targets.md#strata_kernels_cpu-static-always)) and chosen at run time.

## Test options and cache variables

| Option / variable | Default | Where | Effect |
| --- | --- | --- | --- |
| `STRATA_BUILD_TESTS` | ON only if `tests/CMakeLists.txt` and `bench/micro/` exist; OFF in this snapshot | `CMakeLists.txt:87-92` | Gates most CPU tests and some GPU tests; adds `tests/` as a subdirectory if present (`CMakeLists.txt:1236-1242`) |
| `STRATA_BUILD_CONVERSATION_TESTS` | OFF | `CMakeLists.txt:93` | Conversation cache / memory / snapshot / validation / transfer tests (`CMakeLists.txt:96-104`, `460-478`) and the header-only CPU tests at `CMakeLists.txt:912-925` |
| `STRATA_WERROR` | OFF | `CMakeLists.txt:94` | Warnings as errors on `strata_warnings` |
| `STRATA_PARITY_PROMPT_ATTN` | unset (plain variable, no `option()`) | `CMakeLists.txt:339`, `356` | Builds `qsa_prompt_attn_parity` (CUDA) or `qsa_select_bench` (HIP) without the whole test suite (used by `docs/NVIDIA_V100.md`) |
| `STRATA_ORACLE_SOURCE_DIR`, `STRATA_ORACLE_BUILD_DIR` | empty (PATH cache) | `CMakeLists.txt:387-410` | With both set on CUDA: 16 `*_parity` programs from `bench/micro/` linked against an existing shared llama.cpp build. `bench/micro/` is absent in this snapshot, so setting them fails at configure time |
| `IQ_FIXTURE_PYTHON` | unset | `CMakeLists.txt:602-604` | Python interpreter for `tools/iq_fixture.py`; otherwise the configured Python3, `.venv`, `/usr/bin/python3`, `python3` are probed for numpy |
| `STRATA_PLE_FIXTURE_DIR` | `${CMAKE_SOURCE_DIR}/bench/micro` | `CMakeLists.txt:784-785` | Where `ple_parity` reads `ple_in.bin` / `ple_out.bin` |

## Compile definitions produced by the build

| Definition | Scope | Set by |
| --- | --- | --- |
| `STRATA_VERSION="0.1.39"` | global | always |
| `STRATA_EXPERIMENTAL_SM60=1` | global (+ `strata_core` PRIVATE) | `STRATA_EXPERIMENTAL_SM60` with CUDA |
| `STRATA_HIP_GFX906=1`, `__HIP_PLATFORM_AMD__=1` | global / via `CUDA::cudart` | `STRATA_HIP_GFX906` |
| `STRATA_USE_HIP=1`, `STRATA_HIP_ARCHS="gfx1100,..."` | `strata_hip_runtime` INTERFACE | `STRATA_ENABLE_HIP` (`cmake/hip_backend.cmake:76`) |
| `STRATA_GFX1012_PORTABLE_DOT=1` | `strata_hip_runtime` | `STRATA_GFX1012_PORTABLE_DOT` |
| `STRATA_HIPBLASLT_AVAILABLE=1` | `strata_prefill` PRIVATE | HIP and hipBLASLt found |
| `STRATA_NATIVE_EXPERTS=1` | `strata_kernels_cpu` PUBLIC | `STRATA_NATIVE_EXPERTS` |
| `STRATA_ISA_FLOOR_AVX=1` / `STRATA_ISA_FLOOR_NONE=1` | `strata_kernels_cpu` PRIVATE | `STRATA_ISA_FLOOR` |
| `STRATA_PREFILL_MMQ=1`, `STRATA_PREFILL_FUSED=1` | `strata_prefill` PUBLIC | CUDA with native experts (both); HIP with `STRATA_PREFILL_MMQ` (only the first) |
| `STRATA_MMQ_KQUANTS=1`, `STRATA_ORCA_Q4KS_MMQ=1` | `strata_mmq` PRIVATE | the options of the same name |
| `GGML_USE_HIP` / `GGML_USE_HIP=1` | `strata_mmq` PRIVATE | gfx906 build (`CMakeLists.txt:1251`) / HIP MMQ (`CMakeLists.txt:1124`) |
| `STRATA_DEQUANT_SELFTEST` | `strata-dequant` PRIVATE | always (`CMakeLists.txt:224`) |

## Per-file floating-point flags

- CUDA (not gfx906): the 14 `native_*.cu` kernels get `--use_fast_math` to match "the pinned llama.cpp CUDA
  arithmetic contract" (`CMakeLists.txt:364-379`); for gfx906 the generator expression yields no flag.
- HIP: `elementwise.cu` and `quantize_act.cu` get `-ffp-contract=off` so Q8_K's multiply and magic-bias rounding stay
  separate FP32 operations (`CMakeLists.txt:380-385`).
- `strata_mmq` (CUDA): `--extended-lambda -use_fast_math`, and `-Xcompiler=/bigobj` on MSVC (`CMakeLists.txt:1068-1071`).

## Windows / MSVC specifics (summary)

- Release PDBs: `/Zi` + `/DEBUG /OPT:REF /OPT:ICF` (`CMakeLists.txt:26-29`).
- Static MSVC runtime with `STRATA_PORTABLE` (`CMakeLists.txt:47-49`).
- Shared CUDA runtime for the SM60 build (`CMakeLists.txt:131-133`).
- `/arch:AVX512`, `/arch:AVX2`, `/arch:AVX` per CPU kernel file (`CMakeLists.txt:932-934`, `1010-1015`).
- `/bigobj` for the MMQ library.
- Oracle parity tests prepend the oracle's `bin/` to `PATH` on WIN32 (`CMakeLists.txt:405-408`).
- `pool_affinity_test` has a Windows source and a Linux source (`CMakeLists.txt:1192-1200`).
- The HIP backend force-includes its CUDA shim with `/FI` under an MSVC-style front end and `-include` otherwise
  (`cmake/hip_backend.cmake:82-94`); the Windows HIP build uses ROCm's clang++ for host code too.

## Gotchas

- `STRATA_HIP_GFX906` writes `STRATA_ENABLE_CUDA=ON` into the cache with `FORCE`; turning gfx906 off later in the same
  build tree leaves CUDA on.
- `STRATA_EXPERIMENTAL_SM60` is declared inside the CUDA branch, but it is read earlier (`CMakeLists.txt:131`) for the
  MSVC runtime choice; it only has effect there when passed with `-D` on the first configure.
- `STRATA_PORTABLE` and `STRATA_ISA_FLOOR` are mutually exclusive (`CMakeLists.txt:44-46`). Changing `STRATA_ISA_FLOOR`
  in an existing tree leaves stale ggml cache entries; `setup.py` deletes `CMakeCache.txt` in that case
  (`setup.py:2246-2251`).
- `STRATA_BUILD_TESTS` defaults OFF in the published tree; see [tests-registration.md](tests-registration.md).

## In UnbiasedStrata

Every option above is kept. The fork removes the `FetchContent` path of `STRATA_GGML_DIR` (empty now means the
submodule `third_party/llama.cpp`), and its `Makefile` always passes `STRATA_ENABLE_CUDA=ON STRATA_NATIVE_EXPERTS=ON
STRATA_PORTABLE=ON STRATA_MMQ_KQUANTS=ON STRATA_BUILD_TESTS=OFF`, plus `STRATA_EXPERIMENTAL_SM60=ON` when an arch below
75 is listed. See [../../codebase/build/cmake-options.md](../../codebase/build/cmake-options.md).

## See also

- [gpu-backends.md](gpu-backends.md), [third-party.md](third-party.md), [installer-builds.md](installer-builds.md)
- Upstream: [docs/OLDER_GPUS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/OLDER_GPUS.md),
  [docs/AMD_HIP.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/AMD_HIP.md),
  [docs/UNSLOTH_Q4.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/UNSLOTH_Q4.md)
