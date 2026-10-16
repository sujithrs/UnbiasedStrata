[Upstream docs](../README.md) › [Build](README.md) › Test registration

# Test registration (CTest)

## Purpose

How upstream's `CMakeLists.txt` registers its C++ tests with CTest: every `add_test` call grouped by the condition
that gates it, the CTest properties used (skip codes, fixtures, timeouts, environments), and which tests exist in the
configurations people actually build. What each test checks belongs to the test and kernel areas; this page is about
registration.

File covered: [CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/CMakeLists.txt).
The test sources it names live in `src/**/**_test.cpp`, `src/kernels/*_parity.cpp`, `tests/core/`, `tests/cuda/`,
`tests/hip/`, `tests/platform/` and `tests/data/native_experts/`. Python tests under `tools/test_*.py` are not
registered with CTest.

## Counts

- `CMakeLists.txt` contains **94 `add_test(` calls**. Two of them are in loops: the oracle parity loop
  (`CMakeLists.txt:404`, 16 names) and the native-expert pair loop (`CMakeLists.txt:1162`, 6 names). Two pairs are
  alternatives that never coexist: `cuda_device_selftest` / `hip_device_selftest` and the Windows / Linux
  `pool_affinity_test`.
- `STRATA_BUILD_TESTS` defaults to OFF in this snapshot because `tests/CMakeLists.txt` and `bench/micro/` are not
  published (`CMakeLists.txt:87-92`). Many GPU tests are registered regardless of it.
- `enable_testing()` is called at `CMakeLists.txt:97`, `424`, `433`, `500` and `1237`. In a GPU build the call at 424
  always runs in this snapshot (its guard file `tests/core/file_expert_source_test.cpp` exists), so the GPU
  registrations are active even with `STRATA_BUILD_TESTS=OFF`.

## The groups

| Group | Gate | `add_test` calls | Tests |
| --- | --- | --- | --- |
| A. Conversation | `STRATA_BUILD_CONVERSATION_TESTS` (the last three also need a GPU build) | 5 | `conversation_cache_test`, `conversation_memory_test` (`:96-104`, CPU); `conversation_snapshot_test` (skip 77), `conversation_validation_test`, `conversation_transfer_test` (`:460-478`; validation and transfer run with `CUDA_VISIBLE_DEVICES=-1`; transfer only on non-HIP Linux GCC/Clang, using `-Wl,--wrap=cudaMemcpy,...`) |
| B. Artifact | `STRATA_BUILD_TESTS` | 2 | `gguf_reader_test`, `gguf_split_test` (`:225-234`, synthetic GGUFs) |
| C. GPU, ungated | GPU build (+ the noted extra condition) | 43 | see the table below |
| D. GPU + tests | GPU build and `STRATA_BUILD_TESTS` | 9 | `qsa_topk_active_parity` (CUDA, skip 77), `qsa_select_bench` (HIP, or with `STRATA_PARITY_PROMPT_ATTN`), `expert_layout_test`, `native_dense_ple_key_test`, `expert_cache_per_layer_test`, `expert_cache_segmented_test` (these four also need `STRATA_NATIVE_EXPERTS`), `native_mmvq_multi`, `graph_registry_test`, `pinned_capture` (these three also need files under `bench/micro/`, absent) |
| E. HIP | `STRATA_ENABLE_HIP` (no `STRATA_BUILD_TESTS` gate) | 12 | `hip_prefill_gemm`, `hip_q2_zero`, `hip_expert_cache_staging`, `hip_intrinsics`, `hip_handoff`, `hip_mapped_alias` (skip 77), `hip_native_qsa_score`, `hip_prefill_native_batch` (120 s), `hip_router_fast`, `hip_prompt_attn_wmma` (600 s, skip 77 off gfx12), `hip_prefill_hipblaslt_gemm` (hipBLASLt; skip 77), `hip_prefill_mmq_parity` (`STRATA_PREFILL_MMQ`; 120 s). Default timeout 60 s (`:566-568`) |
| F. Oracle parity | CUDA and both `STRATA_ORACLE_*` dirs | 1 (x16) | `<name>_parity --selftest` for `native_mmvq native_bf16 native_shared native_gr_norm native_moe native_gdn native_gr_postops native_router native_gdn_preprocess native_gdn_pipeline native_qsa native_rope native_ple_postops native_flash_attn native_qsa_indexer native_qsa_score` (`:390-410`); sources in the absent `bench/micro/` |
| G. CPU | `STRATA_BUILD_TESTS` (some: or `STRATA_BUILD_CONVERSATION_TESTS`) | 15 | `suffix_drafter_test`, `controller_test`, `draft_policy_test` (`:901-911`); `conv_cache_test`, `coupled_draft_test`, `bf16_bits_test` (`:912-925`, header-only); `router_dot_parity`, `router_dot_parity_avx` (needs `STRATA_NATIVE_EXPERTS`; the second runs with `STRATA_FORCE_ISA=avx`, `:1033-1036`); `pool_affinity_test` (Windows or Linux source); `expert_multi_test`, `expert_parity`, `pool_test`, `pool_stress` (`pool_stress 20`, issue #29), `direct_file_async_test` (Unix, not Apple) (`:1191-1234`) |
| H. CUDA native experts | CUDA, `STRATA_NATIVE_EXPERTS`, `STRATA_BUILD_TESTS` | 7 | `prefill_fused_moe_test` (skip 77 below sm_80, 300 s), `prefill_fused_iq_test` (600 s), `prefill_mmq_kquant_test` (`STRATA_MMQ_KQUANTS`), `native_expert_parity_<pair>` for `q4_K/q5_1 q4_K/q8_0 q5_K/q8_0 q8_0/q8_0 q4_K/q5_0 q5_0/q8_0`, `native_expert_parity_q5_1_min`, `native_expert_parity_bf16_embd`, `native_expert_parity_refuses_q6_K` (`WILL_FAIL TRUE`) |
| Total | | 94 | |

### Group C in detail (registered in every GPU build)

| Test(s) | Extra condition | Notes |
| --- | --- | --- |
| `expert_arena_load` | `pack/full/experts.bin` exists | `strata-load --layers 1`, passes on "per-layer checksums"; a configure message says when it is not registered (`:276-284`) |
| `file_expert_source_test`, `expert_profile_save_test` | their source file exists (it does) | 60 s timeout |
| `dequant_s2_parity`, `s2_gemv_parity`, `iq_multi_parity`, `native_grouped_parity`, `shared_expert_parity`, `gr_parity`, `gdn_parity`, `s2_gemv_q8_parity`, `rope_parity`, `quantize_act_parity`, `route_window_parity`, `router_top10_parity`, `s_gemv_parity`, `elementwise_parity`, `bf16_gemv_parity`, `s_gemv_q8k_parity`, `qsa_parity`, `qsa_topk_parity`, `kv_q8_parity`, `kv_stream_parity`, `kv_hybrid_parity`, `kv_q4_parity`, `cvec_parity`, `s2_expert_grouped_parity` | none | GPU, synthetic data; most run `--selftest` |
| `sampler_parity`, `sampler_parity_one_block`, `sampler_parity_old` | none | one program, three sampler paths chosen by `STRATA_OLD_SAMPLER` / `STRATA_SAMPLER_ONE_BLOCK` in `ENVIRONMENT` (`:695-704`) |
| `iq_parity_fixtures`, `iq_parity` | a Python with numpy was found | fixture pair, see below |
| `mmvq_multi_parity` | CUDA (incl. gfx906) | |
| `mmvq_multi_parity_cuda_layout` | gfx906 | `STRATA_MMVQ_WAVE=0` |
| `gdn_rec_parity`, `gemm_bf16_parity` | CUDA, not gfx906 | |
| `decode_cluster_parity` | CUDA | skips below sm_90 |
| `ple_parity` | none | reads `ple_in.bin` / `ple_out.bin` from `STRATA_PLE_FIXTURE_DIR` (default `bench/micro`, absent), working dir = source root; a missing input is a hard failure by design (`:773-788`) |
| `platform_memory_test` | none | locks a 256 MiB host region (`docs/AMD_HIP.md`: fails under an 8 MiB memlock limit) |
| `pinned_shared_test` | Linux | |
| `ple_reader_selftest` | none | CPU and a temp file |
| `cuda_device_selftest` or `hip_device_selftest` | CUDA (incl. gfx906) / HIP | `strata-device --selftest`; fails, not skips, without a GPU (`:849-856`) |

## CTest properties used

| Property | Used for |
| --- | --- |
| `SKIP_RETURN_CODE 77` | GPU tests that cannot run on the present card (arch too old, no hipBLASLt table, not gfx12, ...) |
| `SKIP_RETURN_CODE 3` | `iq_parity_fixtures` / `iq_parity`: numpy or the gguf-py in `third_party/llama.cpp/gguf-py` missing |
| `FIXTURES_SETUP iq_fixture_data` / `FIXTURES_REQUIRED` | `tools/iq_fixture.py --out <build>/iq_fixture` runs before `iq_parity` even under `ctest -j` (`:593-635`) |
| `WILL_FAIL TRUE` | `native_expert_parity_refuses_q6_K`: a pair without GPU kernels must be refused |
| `ENVIRONMENT` | sampler variants, `CUDA_VISIBLE_DEVICES=-1`, `STRATA_MMVQ_WAVE=0` |
| `ENVIRONMENT_MODIFICATION PATH=path_list_prepend:` | oracle tests on Windows find the oracle DLLs |
| `PASS_REGULAR_EXPRESSION` | `expert_arena_load` |
| `TIMEOUT` | 60 to 600 s on the slower tests |
| `WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}` | `ple_parity`, `expert_parity`, `pool_test` (real-artifact inputs by relative path) |

The Python for `tools/iq_fixture.py` is the first of `IQ_FIXTURE_PYTHON`, or of the configured `Python3`,
`.venv/bin/python`, `.venv/Scripts/python.exe`, `/usr/bin/python3`, `python3`, that can `import numpy`
(`CMakeLists.txt:597-622`). If none can, both tests are left out and a configure message says so.

## Registered tests per configuration

Derived by reading the gates, except the CPU-only row, which was configured for this page.

| Configuration | Registered | Notes |
| --- | --- | --- |
| CPU only (no GPU option), `STRATA_NATIVE_EXPERTS=OFF`, `STRATA_BUILD_TESTS=ON`, Linux | 14 | Configured with CMake 3.28 / GCC 13: groups B and G minus `router_dot_parity*` |
| CPU only, defaults | 0 | `STRATA_BUILD_TESTS` is OFF; no GPU block |
| CUDA (not gfx906), defaults (`STRATA_BUILD_TESTS=OFF`), Linux, numpy found, no `pack/full` | 40 | group C only; 38 without numpy. This is the installer's configuration, though it builds only `--target strata` |
| HIP wave32, `STRATA_PREFILL_MMQ=ON`, hipBLASLt found, defaults otherwise, Linux, numpy found | 48 | group C (36 apply on HIP) + group E (12). `docs/AMD_HIP.md` reports "all 45 registered tests" on an R9700 for its own build; the difference was not determined |

## How to run them

```sh
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`docs/AMD_HIP.md` runs the HIP set as `cmake --build build-hip -j2` then
`ctest --test-dir build-hip --output-on-failure --timeout 60 -E '^(ple_parity|platform_memory_test)$'`, excluding
the test that needs an external model fixture and the one that needs a larger memlock limit.
`tools/hip/build_windows.bat tests` builds the Windows HIP tests for `ctest` in `build-hip-win`.

## Gotchas

- A plain `cmake --build` of a GPU tree compiles every parity program even with `STRATA_BUILD_TESTS=OFF`; only
  `--target strata` avoids that.
- `ple_parity` and `expert_parity` need real model artifacts and fail (not skip) without them; this is intentional per
  the comments at `CMakeLists.txt:777-780` and `1215-1216`.
- Most GPU tests use synthetic data and need only a card; the ones that need real artifacts or external builds are
  `expert_arena_load`, `ple_parity`, `expert_parity` and the oracle set.
- `tests/CMakeLists.txt` (the "model-level checks", `CMakeLists.txt:1238-1241`) is not in the published tree.

## In UnbiasedStrata

The registrations are kept, and the fork adds Qwen3.6 tests (`model_geometry_test`, `vram_plan_test`,
`pool_native_widths_test`, `pack_load_check`, `qwen36_block_test`, `qwen36_verify_test`, `qwen36_runner_check`,
`prefill_qwen36_test`, `prefill_qwen36_attn_test`, `qwen36_mtp_test`, `qwen36_seam_parity`, `qsa_dense_parity`,
`native_expert_parity_qwen36_*`) with `cmake -P` fixture scripts in `cmake/qwen36_*.cmake`. It replaces
`native_expert_parity_refuses_q6_K` with `native_expert_parity_refuses_q6_K_down_at_640` and adds the `q6_K/q8_0` pair
(Q6_K became a supported gate/up type). The fork's `make build` passes `STRATA_BUILD_TESTS=OFF`. See
[../../codebase/build/test-targets.md](../../codebase/build/test-targets.md) and
[../../codebase/build/test-fixtures-cmake.md](../../codebase/build/test-fixtures-cmake.md).

## See also

[targets.md](targets.md), [cmake-options.md](cmake-options.md), [gpu-backends.md](gpu-backends.md)
