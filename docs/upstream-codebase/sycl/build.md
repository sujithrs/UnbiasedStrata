[Upstream docs](../README.md) › [Intel SYCL port](README.md) › Build

# Building the SYCL engine

## Purpose

How `sycl/CMakeLists.txt` turns the mixed tree (migrated copies in `sycl/`, everything else from upstream) into the
`strata` binary and the kernel tests, what the toolchain must provide, and how the top-level build hands off to it.

Files covered:

| File | Lines | Role |
|---|---|---|
| [sycl/CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/CMakeLists.txt) | 241 | The SYCL project |
| [CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/CMakeLists.txt#L12-L22) | lines 12-22 | `STRATA_ENABLE_SYCL` |
| [sycl/tools/build.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/build.sh) | 12 | The build as the port author runs it (see [migration.md](migration.md#buildsh)) |

## Requirements

From [docs/INTEL_ARC.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INTEL_ARC.md),
"What you need":

| Need | Detail |
|---|---|
| OS | Linux (Ubuntu 24.04 used). No Windows path: the CMake file uses GCC-style flags (`-mavx512f`, `-fp-model=precise`, `-qmkl`) and the runner is a bash/Docker script |
| GPU driver | `xe` or `i915` kernel driver plus the compute runtime / Level Zero (on Ubuntu `intel-opencl-icd libze1 libze-intel-gpu1`) |
| Compiler | Intel oneAPI DPC++ `icpx` 2025.3 or newer (2026.1 / 2026.1.1 used), with `icx` for C; about 5 GB |
| Libraries | oneMKL (the prompt GEMMs through `dpct::blas`) |
| Tools | CMake 3.24+ (`cmake_minimum_required(VERSION 3.24)`, `sycl/CMakeLists.txt:12`), Ninja, git (ggml is fetched), Python 3 |
| AOT only | `ocloc` (package `intel-ocloc`) |
| Serving through setup | Docker and the `strata-sycl-dev` image ([serving.md](serving.md)) |

INTEL_ARC.md gives the apt commands for Intel's oneAPI repository and installs
`intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel ninja-build cmake git`.

## Two ways to configure

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_SYCL=ON
#   or: cmake -S sycl -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl --target strata
```

- **Top level.** `option(STRATA_ENABLE_SYCL ... OFF)` (`CMakeLists.txt:14`). When on, it refuses to combine with
  `STRATA_ENABLE_CUDA` or `STRATA_ENABLE_HIP` (`FATAL_ERROR`, `:16-18`), prints a "experimental" warning,
  `add_subdirectory(sycl)` and `return()`s, so none of the CUDA/HIP build is configured (`:15-22`). The binary then
  lands at `build-sycl/sycl/strata`, which `setup_intel.py` does not look for: use `-S sycl` or set
  `STRATA_SYCL_BIN` (INTEL_ARC.md).
- **Directly.** `cmake -S sycl ...`; the binary is `build-sycl/strata`. `setup_intel.py` looks for
  `build-sycl-aot/strata` first, then `build-sycl/strata` (`sycl/setup_intel.py:85`).

## Project settings

| Setting | Value | Where |
|---|---|---|
| Project | `strata_sycl VERSION 0.1.39`, "the upstream engine version this port tracks"; `setup_intel.py` reads this version for its stub `BUILD.json` | `:13` |
| Engine version string | `STRATA_VERSION="0.1.39-sycl"` | `:21` |
| C++ | C++20, no extensions | `:15-17` |
| Default build type | `Release` | `:18-20` |
| `STRATA_ROOT` | the repository root (`..`), where unported sources come from | `:14` |

## Compiler flags

`_sycl_flags` go to every C++ translation unit and `_sycl_link` to every link (`:24-53`). Every file gets `-fsycl`
because the migrated host files create queues and USM allocations too.

| Flag | Why (from the comments) |
|---|---|
| `-fsycl` | SYCL compilation for all C++ |
| `-fsycl-targets=spir64_gen` plus `-Xsycl-target-backend=spir64_gen "-device <STRATA_SYCL_AOT> -options -cl-fp32-correctly-rounded-divide-sqrt"` (AOT) | Ahead-of-time device code for one GPU, through `ocloc` |
| `-Xsycl-target-backend=spir64 -cl-fp32-correctly-rounded-divide-sqrt` (JIT) | The Intel GPU's fp32 divide is not correctly rounded by default (OpenCL allows 2.5 ulp); Strata's quantizers are byte-exact against ggml through `amax / 127`. Measured: `quantize_act_parity` from 303k mismatches to byte-exact. The AOT build must pass it to `ocloc`; it was missing there until 2026-09-30 (`:27-40`). At run time the equivalent is `SYCL_PROGRAM_COMPILE_OPTIONS=-cl-fp32-correctly-rounded-divide-sqrt` |
| `-fsycl-default-sub-group-size=32` | The kernels assume 32-lane warps; Xe2's default is 16 (`:44-46`) |
| `-fsycl-device-code-split=per_kernel` (compile and link) | One device module per kernel, so a first launch loads only its own module; the embedding gather's first launch went from 245 ms to 1 ms (`:47-50`) |
| `-fp-model=precise` | icpx defaults to a fast model (reciprocal approximations, reassociation); the kernels are compared bit-for-bit with ggml's, which nvcc built value-safe (`:42-43`, `:51`) |
| `-Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-declarations` | dpct output is noisy |
| `-Wall -Wextra` | on `strata_includes` users (`:81`) |

## Targets

Sources are named as upstream names them and resolved with `strata_resolve()` (see
[layout.md](layout.md#source-resolution)).

| Target | Kind | Sources | Notes |
|---|---|---|---|
| `strata_includes` | interface | include dirs `sycl/include` then `include/` | `:77-81` |
| `strata_spec` | static | `src/spec/suffix_drafter.cpp`, `controller.cpp`, `draft_policy.cpp` (upstream) | `:84-86` |
| `strata_kernels_cpu` | static | `src/kernels/cpu/expert.cpp`, `pool.cpp`, `expert_layout.cpp`, `q2_avx2.cpp` (upstream); with native experts also `native_expert.cpp`, `iq_avx512.cpp`, `iq_avx2.cpp`, `kq_avx2.cpp` | `expert.cpp` and `iq_avx512.cpp` get `-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi -mf16c -mfma`; the AVX2 files `-mavx2 -mfma -mf16c`; links `Threads`, and with native experts `ggml-cpu ggml-base`, defines `STRATA_NATIVE_EXPERTS=1` (`:88-133`) |
| ggml (subproject) | static | llama.cpp's `ggml/` from `STRATA_GGML_DIR` or fetched at commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` | forced: static, no CUDA, no SYCL, no OpenMP, no backend DL, no tests/examples, `GGML_NATIVE ON` (`:97-122`) |
| `strata_core` | static | `src/core/device.cu`, `pinned.cu` (migrated), `src/platform/memory.cpp`, `src/core/graph.cpp`, `weights.cpp`, `layout.cpp` | `:136-139` |
| `strata_kernels` | static | the 44 kernel files of `src/kernels/cuda/` (list `_kernel_cu`, `:141-145`), `src/kernels/ngram.cpp`, `src/ngram/ple_reader.cpp`, `src/platform/direct_file.cpp`, plus the port-only `qsa_prompt_attn_xmx.dp.cpp` | includes `third_party/ggml` (upstream's directory, `:153`) |
| `strata_engine` | static | `src/core/layer.cpp`, `session.cpp`, `expert_source.cpp`, `remote_experts.cpp`, `expert_cache.cpp`, `native_head.cpp`, `native_dense.cpp`, `verify.cpp`, `mtp.cpp`, `conversation_snapshot.cpp`, `conversation_state.cpp`, `conversation_memory.cpp` (upstream), `peer_experts.cpp` (the stub), plus the port-only `gguf_expert_source.cpp` | private include `src/core` for `conversation_checked.hpp` (`:156-163`) |
| `strata_prefill` | static | `src/prefill/gemm.cu`, `kernels.cu` (migrated), `prefill.cpp`, plus `moe_fused_stub.cpp` | `-qmkl=sequential` at compile and link (cuBLAS became oneMKL through `dpct::blas`) (`:166-171`) |
| `strata` | executable | `src/program/generate.cpp` | links engine, prefill, spec (`:173-175`) |

No `strata-server` or other upstream binary is built; the server is the Python one
([serving.md](serving.md)).

## Options

| Option | Default | Effect |
|---|---|---|
| `STRATA_SYCL_AOT` | empty | Device name for AOT code (`bmg-g31` for the Arc Pro B70, `bmg-g21` for the B580 / B570 / Pro B60, per INTEL_ARC.md); empty means SPIR-V JIT-compiled at run time |
| `STRATA_NATIVE_EXPERTS` | `ON` | Native (IQ) CPU expert support through ggml-cpu |
| `STRATA_GGML_DIR` | empty | A llama.cpp checkout to take `ggml/` from instead of fetching |
| `STRATA_SYCL_PARITY` | `ON` | Build the kernel tests and benches |

JIT versus AOT: without AOT the runtime compiles every kernel on first use, about 47 s the first time a process runs
(INTEL.md, INTEL_ARC.md). INTEL.md says `SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=<dir>` keeps the result (15 MB) but
also, under "Things that matter on this GPU", that the persistent JIT cache segfaults on Xe2 during the first
compile, and the dev image sets `SYCL_CACHE_PERSISTENT=0`. The served build is AOT (`build-sycl-aot`).

## Tests

With `STRATA_SYCL_PARITY=ON` (`:178-241`):

| Registered with `add_test` | Command |
|---|---|
| `dequant_s2`, `s2_gemv`, `iq`, `shared_expert`, `gr`, `gdn`, `s2_gemv_q8`, `sampler`, `rope`, `quantize_act`, `router_top10`, `s_gemv`, `elementwise`, `bf16_gemv`, `s_gemv_q8k`, `qsa`, `ple`, `kv_q8`, `kv_stream`, `kv_q4`, `cvec` (each `<name>_parity`) | `<name>_parity --selftest` |
| `iq_multi_parity` | no argument (runs its checks by default) |
| `s2_expert_grouped_parity` | `--selftest` |
| `native_grouped_parity` | no argument |
| `conversation_snapshot_test` | no argument; skip code 77, timeout 120 s |

Built but not registered: `qsa_prompt_attn_parity` (takes a context length; needs the tensor-core prompt kernel or
the XMX one it now turns on), `kv_hybrid_parity`, the benches `mmvq_bench`, `mmvq_sg_bench`, `q6k_align_bench`,
`gr_bench`, `xmx_gemm_bench` (linked with `strata_prefill` for the oneMKL reference), and, with native experts,
`native_expert_parity <shard1.gguf> [layer ...]` (`NATIVE_BENCH=1` times it). Each parity test runs on the
selected SYCL device and compares against a CPU reference.

Run them on the card with `ctest --test-dir build-sycl`. To use a CPU device instead:
`ONEAPI_DEVICE_SELECTOR=opencl:cpu` (Intel's OpenCL CPU runtime).

Reported results (not re-run here):

| When | Device | Result | Source |
|---|---|---|---|
| 2026-09-29 | Arc Pro B70 | 19 of 22 pass; `iq_parity` and `ple_parity` need fixtures, `qsa_prompt_attn_parity` the tensor-core kernel | INTEL.md |
| 2026-10-01 (0.1.32 merge) | Arc Pro B70 | `kv_hybrid_parity` and `qsa_prompt_attn_parity` pass with the XMX prompt attention; `iq_multi_parity` (IQ2_XS) and `s2_expert_grouped_parity` still failing; IQ2_XS fixed later the same day | INTEL.md |
| 0.1.39 release | OpenCL CPU device (AMD Ryzen 5 7600), oneAPI DPC++ 2026.1.1 + oneMKL 2026.1, WSL2 | the whole project builds with 0 errors (the engine plus 157 targets); 14 of 25 tests pass, the same set PR #423's 0.1.38 port passes there; failures are tight float tolerances on CPU math, missing fixtures and two CPU timeouts. The AOT build was not tried (no `ocloc`) | INTEL_ARC.md |

The installer side has its own unit test, `tools/test_setup_sycl.py`, whose `CMakeOption` class checks that the
top-level option exists, is `OFF` by default and adds `sycl` (`tools/test_setup_sycl.py:66-71`); see
[serving.md](serving.md#tests).

## Platform notes

- **AVX-512 objects.** `expert.cpp` and `iq_avx512.cpp` are always compiled with AVX-512 flags, as in upstream's
  build; which CPU path runs is decided at run time by upstream's code (not changed by the port). `GGML_NATIVE ON`
  tunes ggml to the build machine.
- **Two GPUs.** The engine's `--layer-split` works with `ONEAPI_DEVICE_SELECTOR=level_zero:*` (INTEL_ARC.md; the
  dev image pins `level_zero:0`). `--peer-device` does not ([differences.md](differences.md#9-what-the-port-leaves-out)).
- **HIP and CUDA builds** are unaffected: the option is off by default and exclusive.

## Gotchas

- With `-DSTRATA_ENABLE_SYCL=ON` at the top level the binary path differs from what `setup_intel.py` searches.
- `third_party/ggml` on the include path is upstream's directory (`sycl/CMakeLists.txt:126`, `:153`);
  `iq_kernels.dp.cpp` defines `GGML_COMMON_DECL_SYCL` to get its SYCL declarations. `sycl/third_party/ggml/ggml-common.h`
  (the dpct-migrated copy) is not on any include path in this CMake file; whether anything uses it is not
  determined from the source.
- Seven migrated files have no target at all (`concurrent_main.cpp`, `overlap_main.cpp`, `dequant_bf16_test.cpp`,
  `q5_projection_parity.cpp`, `qsa_select_bench.cpp`, `moe_mmq.dp.cpp`, `ggml_cuda_host.dp.cpp`).

## In UnbiasedStrata

`sycl/CMakeLists.txt` was removed in commit `8894a81`. The top-level `STRATA_ENABLE_SYCL` option and its
`add_subdirectory(sycl)` are still in the fork's `CMakeLists.txt` (lines 12-22 at HEAD), so turning the option on
there fails because the directory is gone.

## See also

- [layout.md](layout.md): which file each target compiles.
- [migration.md](migration.md#buildsh): `build.sh`.
- [hardware.md](hardware.md): which `STRATA_SYCL_AOT` device to choose.
