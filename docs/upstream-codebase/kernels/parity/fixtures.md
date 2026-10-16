[Upstream docs](../../README.md) › [Kernels](../README.md) › [Parity tests](README.md) › Fixtures and running

# Test fixtures, oracle data and how to run the parity tests

## Purpose

Where each parity test's reference data comes from, how CTest is wired (the Python probe, the one fixture test,
working directories, environment variables, timeouts), which inputs are checked into the repository and which must
be supplied, the build switches and the exit-code conventions. Files covered: the two generated
`src/kernels/*.inc` tables, `tools/iq_fixture.py` (as the fixture generator), the `tests/data/native_experts/`
tables, and the CTest wiring in `CMakeLists.txt`.

## Files

| File | Lines | Role |
|---|---|---|
| [src/kernels/ple_oracle_vectors.inc](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/ple_oracle_vectors.inc) | 105 | Hash cases, probe rows of the PLE table, table geometry and dense-pack offsets (for `ple_parity`) |
| [src/kernels/qsa_oracle_vectors.inc](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/qsa_oracle_vectors.inc) | 134 | `{f32 bits, numpy float16 bits}` pairs (for `qsa_parity`) |
| [tools/iq_fixture.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/iq_fixture.py) | 137 | Deterministic i-quant fixtures for `iq_parity` (`<name>.bin` raw blocks + `<name>.f32` gguf-py reference) |
| [tests/data/native_experts/](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/tests/data/native_experts) | 6 files | Copies of real packs' `native_experts.txt` (for `tests/core/expert_layout_test.cpp`) |
| [CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/CMakeLists.txt) | 1254 | Builds and registers every test |

## Build switches

| Switch | Default | Effect on tests |
|---|---|---|
| `STRATA_BUILD_TESTS` | OFF in the published source: it defaults ON only if `tests/CMakeLists.txt` and `bench/micro/` exist (`CMakeLists.txt:87-92`), and neither is published | Gates the CPU tests, `qsa_topk_active_parity`, `qsa_prompt_attn_parity` (CUDA), the `native_expert_parity_*` ctests, the manual real-file programs, `dequant_bf16_test`, `bf16_bits_test`, `tests/core/*` and `tests/cuda/*` |
| `STRATA_BUILD_CONVERSATION_TESTS` | OFF | Also builds `bf16_bits_test` (`:912-925`) |
| `STRATA_ENABLE_CUDA` / `STRATA_ENABLE_HIP` / `STRATA_HIP_GFX906` | OFF | Any GPU backend builds the GPU parity executables and registers their ctests (no `STRATA_BUILD_TESTS` needed) |
| `STRATA_NATIVE_EXPERTS` | ON | `native_expert_parity`, `native_expert_bench`, `router_dot_parity`, the real-file PLE / Q5 programs |
| `STRATA_PARITY_PROMPT_ATTN` | unset (a plain variable, not an `option`) | Builds `qsa_prompt_attn_parity` (CUDA) or `qsa_select_bench` (HIP) without the rest of the tests (`:339`, `:356`) |
| `IQ_FIXTURE_PYTHON` | unset | Overrides the numpy-python search outright (`:602-604`) |
| `STRATA_PLE_FIXTURE_DIR` | `<source>/bench/micro` (cache path, `:784-785`) | Where `ple_parity`'s ctest looks for `ple_in.bin` / `ple_out.bin` |
| `STRATA_ORACLE_SOURCE_DIR`, `STRATA_ORACLE_BUILD_DIR` | empty | With CUDA, builds `bench/micro/native_*_parity.cpp` against a separate shared llama.cpp build (`:387-410`); those sources are not published |

## The Python probe and the one CTest fixture

`_iq_python` (`CMakeLists.txt:597-635`, inside the GPU block) is the first interpreter that can `import numpy`
among: `IQ_FIXTURE_PYTHON` if set, else the configured `Python3_EXECUTABLE`, `.venv/bin/python`,
`.venv/Scripts/python.exe`, `/usr/bin/python3`, `python3`. Without one, configure prints a status message and
neither `iq_parity_fixtures` nor `iq_parity` is registered.

```
iq_parity_fixtures  (python tools/iq_fixture.py --out <build>/iq_fixture)
        | FIXTURES_SETUP iq_fixture_data      SKIP_RETURN_CODE 3
        v
iq_parity <build>/iq_fixture                  FIXTURES_REQUIRED iq_fixture_data, SKIP_RETURN_CODE 3
```

The generator also needs the vendored **gguf-py** at `third_party/llama.cpp/gguf-py` (the copy setup downloads with
the pinned llama.cpp zip; `tools/iq_fixture.py:26-47`); without numpy or gguf-py it exits 3, which CTest counts as a
skip. Generated data goes into the build directory; the source tree is not touched.

## Environment, working directories and timeouts set by CTest

| Test | Setting | Lines |
|---|---|---|
| `sampler_parity` / `_one_block` / `_old` | `STRATA_OLD_SAMPLER` and `STRATA_SAMPLER_ONE_BLOCK` set to select each path | `:699-704` |
| `mmvq_multi_parity_cuda_layout` (gfx906) | `STRATA_MMVQ_WAVE=0` | `:643-646` |
| `router_dot_parity_avx` | `STRATA_FORCE_ISA=avx` via `cmake -E env` | `:1035` |
| `ple_parity`, `expert_parity`, `pool_test` | `WORKING_DIRECTORY` = source root (relative `pack/full/...` paths) | `:786-788`, `:1217-1222` |
| `native_expert_parity_*` | `TIMEOUT 120`; `refuses_q6_K` `WILL_FAIL TRUE` | `:1160-1170` |
| `qsa_topk_active_parity` | `SKIP_RETURN_CODE 77` | `:351` |
| HIP tests | `TIMEOUT 60` for the list, 120 for two, 600 for `hip_prompt_attn_wmma` | `:552-574` |

Some programs set variables themselves: `native_expert_parity` sets `STRATA_IQ_MT_MIN=1` unless given
(`native_expert_parity.cpp:537-543`); `decode_cluster_parity` sets `STRATA_QSA_CLUSTER=0` and
`STRATA_ARGMAX_MULTI=0` before the dispatchers' first call (`:378-379`); on AMD `qsa_prompt_attn_parity` sets
`STRATA_HIP_WMMA=1` after checking for gfx12 (`:243-258`).

## Inputs that are not generated

| Input | Needed by | Notes |
|---|---|---|
| `pack/full/experts.bin` (canonical Q2_0 pack) | `expert_parity`, `pool_test` (working dir = source root); `expert_arena_load` is registered only if the file exists (`:276-284`) | Missing = exit 2 (failure) for the first two |
| `pack/full/dense.bin` | `ple_parity` | Must cover the PLE key / value regions listed in `ple_oracle_vectors.inc:98-103` |
| The Q2_0 model's GGUF shard 2 (the IQ4_NL PLE table) | `ple_parity` | `--gguf`, `$STRATA_PLE_GGUF`, else `../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`; the file must be exactly 192 + 320,001,536 x 90 = 28,800,138,432 bytes (`ple_oracle_vectors.inc:79-81`) |
| `ple_in.bin`, `ple_out.bin` (a ggml capture of the PLE layer) | `ple_parity` | Written by `bench/micro/ple_layer_xcheck.cpp`, which is not published; `--check-fixtures` validates them on the CPU only |
| Real model shards / files | `dequant_bf16_test`, `native_expert_bench`, `native_expert_parity <shard>`, `ple_fp8_parity`, `ple_q5_parity`, `q5_projection_parity`, `hip_ple_iq4`, `hip_prefill_mmq_parity <pack>` | Manual runs only |

## The `.inc` oracle vectors

- `ple_oracle_vectors.inc` says it is generated by `tests/python/gen_ple_oracle.py` from `ref/ngram.py::ngram_rows`,
  the original GGUF's bytes read with numpy, and `pack/full/manifest.json`. It holds six hash cases, three probe
  rows of the table with their values, the table geometry (rows, data start, file size) and the dense-pack offsets of
  the PLE key / value weights.
- `qsa_oracle_vectors.inc` says it is generated by `scripts/_gen_f16_oracle.py`: f32 -> f16 conversions as numpy
  performs them (zeros, overflow to inf, subnormals, ties).
- Neither generator (nor `ref/ngram.py`, `ref/gdn.py`, `ref/gr.py`, `ref/qsa.py`, `ref/moe.py`, `ref/quant.py`,
  which the test comments cite) is in the published source: `ref/` holds only `load.py`, and there is no
  `tests/python/` or `scripts/`. The vectors are frozen data.

## Exit codes

| Code | Meaning | Where |
|---|---|---|
| 0 | Pass; also some explicit skips on missing CPU or GPU features (`expert_parity`, `pool_test`, `pool_stress`, `decode_cluster_parity` below sm_90, `hip_mapped_alias` whenever it ran) | all |
| 1 | Failure | all |
| 2 | Bad usage, a missing required input, or a HIP/CUDA API error in a HIP test's `CHECK` (a failure) | `expert_parity`, `pool_test`, `ple_parity`, real-file programs, `tests/hip/*` |
| 3 | Fixture generator without numpy / gguf-py (`SKIP_RETURN_CODE 3`); also `pool_stress`'s 10 s watchdog (there a failure) | `iq_parity_fixtures`, `iq_parity`; `pool_stress` |
| 77 | Skip: no device, unsupported architecture, or no tuning table (`SKIP_RETURN_CODE 77`) | `qsa_topk_active_parity`, `hip_prompt_attn_wmma`, `hip_mapped_alias`, `hip_prefill_hipblaslt_gemm`, some `tests/core` and `tests/cuda` tests |
| 7-12 | `hip_handoff`'s specific failures (timeouts, stale payloads) | `tests/hip/handoff.cpp` |

## How to run

```
# NVIDIA
cmake -B build -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure                 # everything registered
ctest --test-dir build -R parity -E 'ple_parity'            # the kernel parity tests without the model fixture
build/native_expert_parity --synthetic iq2_s/q2_0           # one case by hand
build/native_expert_parity /models/...-00001-of-00002.gguf 0 1   # real rows

# AMD wave32 (see ../hip/build.md)
cmake -B build-hip -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DSTRATA_BUILD_TESTS=ON
cmake --build build-hip -j2
ctest --test-dir build-hip --output-on-failure --timeout 60 -E '^(ple_parity|platform_memory_test)$'
```

The last line is upstream's own command for the AMD backend
([docs/AMD_HIP.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/AMD_HIP.md),
"Original backend validation"): `ple_parity` needs the external model fixture and `platform_memory_test` needs
256 MiB of locked memory. Most GPU parity tests take `--selftest` (their ctest form); several take `--bench` for
timings (`decode_cluster_parity`, `iq_multi_parity`, `native_grouped_parity`, `s2_expert_grouped_parity`).

## Gotchas

- Tests that need a pack or a model fail with exit 2 rather than pass when the input is absent; on a machine without
  the Q2_0 fixture `ple_parity` always fails (upstream's AMD reports list it as an expected failure).
- A CPU without AVX-512 VNNI fails `expert_multi_test` (no CPU check) while `expert_parity` and `pool_test` skip.
- Several comments cite paths that are not in the published source (`bench/micro/...`, `ref/*.py` except `load.py`,
  `scripts/`, `tests/python/`, `docs/semantics.md`, `docs/activation-contract.md`).
- `iq_parity` and the fixture both use exit 3 as "skip"; a missing build-directory fixture otherwise makes
  `iq_parity` fail.

## In UnbiasedStrata

The fork keeps both `.inc` tables and `tests/data/native_experts/` (adding two Qwen3.6 tables), vendors llama.cpp as
a git submodule at `third_party/llama.cpp` instead of fetching it, and adds four CMake fixture scripts
(`cmake/qwen36_*.cmake`) with a second Python probe for its Qwen3.6 harnesses; `tools/iq_fixture.py` was removed with
the other Python tools (8894a81). The fork's page: [../../../codebase/kernels/parity/fixtures.md](../../../codebase/kernels/parity/fixtures.md).

## See also

- [README.md](README.md), [../cpu/tests.md](../cpu/tests.md), [../hip/tests.md](../hip/tests.md)
- [../../build/cmake-options.md](../../build/cmake-options.md), [../../build/targets.md](../../build/targets.md)
