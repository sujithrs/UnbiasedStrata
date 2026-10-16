[Upstream docs](../README.md) › [Guides](README.md) › Testing

# Testing: one index of every test in upstream Strata

## Purpose

Upstream's tests live in five places and run under three drivers: CTest (C++ and CUDA/HIP/SYCL programs registered
in `CMakeLists.txt` and `sycl/CMakeLists.txt`), Python `unittest` (`serve/test_*.py`, `tools/test_*.py`), and
hand-run harnesses that drive a built engine or a running server with a real model. This page lists all of them:
what each needs (CPU only, a CUDA or HIP device, Python with numpy, a real model file), how it is registered, how to
build and run it, and how it reports a skip. Kernel parity programs are counted and grouped here; each one is
described in [../kernels/parity/README.md](../kernels/parity/README.md). CMake options are in
[../build/README.md](../build/README.md).

There is no CI in the snapshot: `.github/` holds only `FUNDING.yml`. Every test is run by hand. `AGENTS.md` names the
setup tests as the ones that need no GPU or download (`python tools/test_setup_<name>.py`).

| Kind | Where | Driver | Count |
| --- | --- | --- | --- |
| Engine unit tests beside their code | `src/**/*_test.cpp`, `src/kernels/cpu/{pool_stress,expert_parity}.cpp` | CTest | 21 source files |
| Engine tests in `tests/` | `tests/core` (7 + 1 header), `tests/cuda` (3), `tests/hip` (12), `tests/platform` (1), `tests/data` (6 fixtures) | CTest | 23 programs |
| Kernel parity programs | `src/kernels/*_parity.cpp` (35), `src/prefill/*_parity.cu` (2), `src/kernels/cpu/router_dot_parity.cpp` | CTest (most) or by hand | 38 programs |
| SYCL port's kernel tests | `sycl/src/kernels/*_parity.cpp` (+ engine-tree sources) | CTest in `build-sycl` | 25 registrations |
| Python unit tests, server | `serve/test_*.py` | `python -m unittest` | 12 files, 268 `test_*` methods |
| Python unit tests, tools and setup | `tools/test_*.py` | `python -m unittest` or `python tools/test_X.py` | 32 files, 380 `test_*` methods |
| Live harnesses | `tools/batch_test.py`, `parking_test.py`, `conversation_cache_*.py`, ... | by hand, against an engine or server | 14 scripts |

Counts are from the snapshot (`grep -c "def test_"`, `ls`); a `test_*` method count includes every method so named.

---

## 1. How to build and run

### The C++ tests: configure with `-DSTRATA_BUILD_TESTS=ON`

`STRATA_BUILD_TESTS` defaults ON only when both `tests/CMakeLists.txt` and `bench/micro/` exist
(`CMakeLists.txt:87-92`; the comment says "the published source leaves out" both). Neither is in the snapshot, so
the default is OFF and the option must be given. A Linux CUDA build with the tests:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120            # add -DSTRATA_MMQ_KQUANTS=ON for prefill_mmq_kquant_test
cmake --build build -j                          # every target, not only strata
ctest --test-dir build --output-on-failure
```

AMD's own recipe (`docs/AMD_HIP.md:484-499`) builds every target, then runs
`ctest --test-dir build-hip --output-on-failure --timeout 60 -E '^(ple_parity|platform_memory_test)$'`, excluding the
two tests that need an outside fixture or a memlock limit above 8 MiB. Reported HIP totals: "all 45 registered
tests, R9700: 42 pass", one skip, two host failures (`docs/AMD_HIP.md:215`); "ctest 30/32" and "32/32" on community
cards (`docs/AMD_HIP.md:295-299`); "ctest 42 of 46" for the Windows HIP build (`docs/AMD_HIP.md:100`). On CUDA,
`bench/results/2026-09-27-kv-q4/README.md` records "all 70 ctest tests pass" at engine 0.1.8. By the guards below, a
Linux CUDA build with `-DSTRATA_BUILD_TESTS=ON`, Python with numpy, and no other option registers 72 tests (a count of
the guards, not a recorded run).

### Conversation-cache tests: `-DSTRATA_BUILD_CONVERSATION_TESTS=ON`

A separate opt-in (default OFF, `CMakeLists.txt:93`). It registers `conversation_cache_test` and
`conversation_memory_test` at the top level (`:96-104`, CPU, header-only plus one source), and inside the GPU block
`conversation_snapshot_test` (skip 77), `conversation_validation_test` and, on Linux with GCC or Clang and no HIP,
`conversation_transfer_test` (`:460-478`): the validation source relinked with `-Wl,--wrap=cudaMemcpy` (and
`cudaDeviceSynchronize`, `cudaGetLastError`) so the test counts transfers; both run with `CUDA_VISIBLE_DEVICES=-1`.
`conv_cache_test`, `coupled_draft_test` and `bf16_bits_test` build with either option (`:912-925`).

### The SYCL port's tests

`-DSTRATA_ENABLE_SYCL=ON` hands the whole configure to `sycl/CMakeLists.txt` (`CMakeLists.txt:14-22`; it refuses CUDA
or HIP beside it). `STRATA_SYCL_PARITY` (default ON, `sycl/CMakeLists.txt:177`) registers the port's tests; run them
with `ctest --test-dir build-sycl` on the card (`docs/INTEL_ARC.md:88-89`). See section 6 and
[../sycl/README.md](../sycl/README.md).

### Python tests

All `unittest`; no pytest configuration in the tree. From the repository root, with the setup's `.venv` (it has
jinja2, which `serve/frontend.py:24` imports):

```
python -m unittest serve.test_server -v               # one module; each file's docstring names its own line
python -m unittest tools.test_setup_choices
python tools/test_setup_parallel.py                   # every test file also ends in unittest.main()
.venv/bin/python -m unittest discover -s tools -p test_iq_pack.py   # needs numpy + gguf-py (docs/ORCA.md:87)
```

gguf-py is found by `tools/_paths.py:15-23`: `$STRATA_GGUF_PY`, then `third_party/llama.cpp/gguf-py` (which
`setup.py` clones; the snapshot's `third_party/` holds only `ggml/`), then a development tree's
`../../.ref/llama.cpp/gguf-py`; without one it exits with "llama.cpp's gguf-py was not found".

---

## 2. Rules that apply to every CTest entry

- **Skip codes.** Exit 77 is a skip (`SKIP_RETURN_CODE 77`) for: `qsa_topk_active_parity`, `native_dense_ple_key_test`,
  `expert_cache_per_layer_test`, `expert_cache_segmented_test`, `conversation_snapshot_test`, `hip_mapped_alias`,
  `hip_prompt_attn_wmma`, `hip_prefill_hipblaslt_gemm`, `prefill_fused_moe_test`, `prefill_fused_iq_test`. The reasons
  in the sources: no CUDA device, no virtual memory management (segmented cache), below sm_80 (fused prompt MoE),
  not gfx12 (WMMA attention), no `STRATA_HIPBLASLT_TUNING` table. Exit 3 is a skip for `iq_parity_fixtures` and
  `iq_parity` (numpy or gguf-py missing, `CMakeLists.txt:626-630`; `tools/iq_fixture.py:47`).
- **Expected failure.** `native_expert_parity_refuses_q6_K` is `WILL_FAIL TRUE` (`CMakeLists.txt:1169-1170`): a
  Q6_K gate/up pair has no GPU kernel and `native_expert_supported` must refuse it.
- **Pass by output.** `expert_arena_load` passes on the regular expression `per-layer checksums` (`:281`).
- **One binary, several environments.** `sampler_parity` runs three times (`STRATA_OLD_SAMPLER`,
  `STRATA_SAMPLER_ONE_BLOCK`, `:698-704`); `mmvq_multi_parity_cuda_layout` sets `STRATA_MMVQ_WAVE=0` (gfx906 only,
  `:644-646`); `router_dot_parity_avx` runs the same binary under `STRATA_FORCE_ISA=avx` (`:1035`).
- **Timeouts.** 60 s for the HIP tests and most `tests/core` tests, 120 s for `hip_prefill_native_batch`,
  `hip_prefill_hipblaslt_gemm`, `hip_prefill_mmq_parity`, the conversation tests and `native_expert_parity_*`, 300 s
  `prefill_fused_moe_test`, 600 s `prefill_fused_iq_test` and `hip_prompt_attn_wmma`.
- **Python at configure time.** The `iq_parity` pair is registered only when CMake finds an interpreter that imports
  numpy: `IQ_FIXTURE_PYTHON`, else `Python3_EXECUTABLE`, `.venv/bin/python`, `.venv/Scripts/python.exe`,
  `/usr/bin/python3`, `python3` (`:597-635`). Otherwise a configure message says the pair is not registered. The
  generator writes into `build/iq_fixture`, never the source tree; `FIXTURES_SETUP iq_fixture_data` orders it before
  the test.
- **Registration is not uniform.** Most kernel parity tests sit in the `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)`
  block (`:248-885`) with no `STRATA_BUILD_TESTS` guard. They are registered, and their executables built by the
  default target, whenever a GPU backend is on. `file_expert_source_test` and `expert_profile_save_test` are guarded
  only by their source file existing (`:423-437`).
- **Real artifacts are hard requirements.** `expert_parity` reads `pack/full/experts.bin`
  (`src/kernels/cpu/expert_parity.cpp:83`, working directory pinned to the source tree) and `ple_parity` reads
  `bench/micro/ple_in.bin` / `ple_out.bin` (`STRATA_PLE_FIXTURE_DIR`), a `pack/full` and the Q2_0 shard 2
  (`STRATA_PLE_GGUF`, default `../../Q2_0/...-00002-of-00002.gguf`, `src/kernels/ple_parity.cpp:253-256`). A missing
  input is a failure, by design (`CMakeLists.txt:775-780`: "A test that passes by finding nothing to compare is the
  failure mode this project keeps re-learning"). Neither input is in the snapshot.
- **GPU required by design.** `cuda_device_selftest` / `hip_device_selftest` run `strata-device --selftest`, which
  allocates real VRAM; a machine without a device fails it rather than skipping (`:849-856`).

---

## 3. The CTest registrations, counted and grouped

`CMakeLists.txt` has 94 `add_test(` calls. Two are inside `foreach` loops (16 llama.cpp-oracle parity programs, 6
native-expert format pairs), so 114 test names can exist; two pairs are mutually exclusive (CUDA vs HIP device
selftest, Windows vs Linux pool affinity), and many need an option, a backend or an absent file.

| Group | Lines | `add_test` calls | Tests | Guard |
| --- | --- | ---: | --- | --- |
| Conversation tests, top level | 96-104 | 2 | `conversation_cache_test`, `conversation_memory_test` | `STRATA_BUILD_CONVERSATION_TESTS` |
| GGUF reader | 225-234 | 2 | `gguf_reader_test`, `gguf_split_test` | `STRATA_BUILD_TESTS` (CPU, no GPU block) |
| Expert arena load | 278-284 | 1 | `expert_arena_load` (`strata-load --file pack/full/experts.bin --layers 1`) | GPU + `pack/full/experts.bin` exists |
| QSA top-k / select | 346-361 | 2 | `qsa_topk_active_parity` (CUDA), `qsa_select_bench` (HIP, `32768 256 3 262144`) | GPU + TESTS (or `STRATA_PARITY_PROMPT_ATTN` for the HIP bench) |
| llama.cpp-oracle parity | 390-410 | 1 (16 tests) | `native_{mmvq,bf16,shared,gr_norm,moe,gdn,gr_postops,router,gdn_preprocess,gdn_pipeline,qsa,rope,ple_postops,flash_attn,qsa_indexer,qsa_score}_parity` | CUDA + `STRATA_ORACLE_SOURCE_DIR` + `STRATA_ORACLE_BUILD_DIR`; sources in `bench/micro/` (absent) |
| Engine tests in `tests/core` | 423-459 | 6 | `file_expert_source_test`, `expert_profile_save_test`, `expert_layout_test`, `native_dense_ple_key_test`, `expert_cache_per_layer_test`, `expert_cache_segmented_test` | GPU; the last four also TESTS + `STRATA_NATIVE_EXPERTS` |
| Conversation tests, GPU | 460-478 | 3 | `conversation_snapshot_test`, `conversation_validation_test`, `conversation_transfer_test` | GPU + CONVERSATION (transfer: Linux, GCC/Clang, not HIP) |
| HIP tests | 499-575 | 11 | `hip_prefill_gemm`, `hip_q2_zero`, `hip_expert_cache_staging`, `hip_intrinsics`, `hip_handoff`, `hip_mapped_alias`, `hip_native_qsa_score`, `hip_prefill_native_batch`, `hip_router_fast`, `hip_prompt_attn_wmma`, `hip_prefill_hipblaslt_gemm` | `STRATA_ENABLE_HIP` (no TESTS guard); hipBLASLt one also `STRATA_HIPBLASLT_AVAILABLE` |
| Kernel parity, GPU block | 583-822 | 35 | the `src/kernels/*_parity.cpp` and `src/prefill/*_parity.cu` programs, `iq_parity_fixtures`, three sampler variants, `mmvq_multi_parity_cuda_layout` | GPU (several CUDA only or not gfx906, section 6) |
| bench/micro programs | 825-830, 861-878 | 3 | `native_mmvq_multi`, `graph_registry_test`, `pinned_capture` | TESTS + the `bench/micro/` source exists (absent) |
| Platform and PLE reader | 835-856 | 5 | `platform_memory_test`, `pinned_shared_test` (Linux), `ple_reader_selftest`, `cuda_device_selftest` or `hip_device_selftest` | GPU |
| Speculation | 901-911 | 3 | `suffix_drafter_test`, `controller_test`, `draft_policy_test` | TESTS (CPU) |
| Header-only CPU tests | 912-925 | 3 | `conv_cache_test`, `coupled_draft_test`, `bf16_bits_test` | TESTS or CONVERSATION |
| CPU router dot | 1033-1036 | 2 | `router_dot_parity`, `router_dot_parity_avx` | `STRATA_NATIVE_EXPERTS` + TESTS |
| Prompt-path MoE | 1081-1098 | 3 | `prefill_fused_moe_test`, `prefill_fused_iq_test`, `prefill_mmq_kquant_test` | NATIVE + CUDA + TESTS; the last also `STRATA_MMQ_KQUANTS` |
| HIP MMQ | 1136 | 1 | `hip_prefill_mmq_parity` | NATIVE + HIP + `STRATA_PREFILL_MMQ` |
| Native expert parity | 1152-1170 | 4 (9 tests) | `native_expert_parity_{q4_K_q5_1,q4_K_q8_0,q5_K_q8_0,q8_0_q8_0,q4_K_q5_0,q5_0_q8_0}`, `_q5_1_min`, `_bf16_embd`, `_refuses_q6_K` | NATIVE + GPU + TESTS + CUDA |
| CPU expert pool and files | 1191-1234 | 7 | `pool_affinity_test` (Windows or Linux source), `expert_multi_test`, `expert_parity`, `pool_test`, `pool_stress` (`20` rounds), `direct_file_async_test` (Unix, not Apple) | TESTS |

`add_subdirectory(tests)` (`:1236-1242`) is skipped because `tests/CMakeLists.txt` is absent; the comment at
`:1239` says "Model-level checks may be absent from source distributions". `STRATA_CUDA_TESTS` (`:882-884`) is set
for that missing file's Catch2 target and has no effect in the snapshot.

Built but never registered: `qsa_prompt_attn_parity` (CUDA, `:339-343`; also alone with
`-DSTRATA_PARITY_PROMPT_ATTN=ON`), `fused_gr_bench` (`:677`), `native_expert_bench` (`:1149`), `hip_ple_iq4`
(`:1142`, takes a real shard), `ple_q5_parity`, `q5_projection_parity`, `ple_fp8_parity` (`:1174-1183`, real-weight
checks for the OrcaRouter Q4_K_S files and the FP8 PLE table), `dequant_bf16_test` (`:1201-1206`: "not a ctest: needs
the 38 GB model; run it by hand or from tools/gpu_session.py"; that script is not in the snapshot),
`hit_cpu_order_parity` (`bench/micro`, absent), and the tools `strata-load`, `strata-concurrent`, `strata-overlap`,
`tune_hipblaslt`.

---

## 4. Engine unit tests beside the code (`src/`)

| File | Lines | CTest name | Registered when | What it checks |
| --- | ---: | --- | --- | --- |
| [src/artifact/gguf_reader_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/artifact/gguf_reader_test.cpp) | 93 | `gguf_reader_test` | TESTS | the reader refuses a duplicate tensor name at open, naming it (synthetic GGUF) |
| [src/core/conversation_cache_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/conversation_cache_test.cpp) | 305 | `conversation_cache_test` | CONV | the conversation cache's bookkeeping (header-only) |
| [src/core/conversation_memory_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/conversation_memory_test.cpp) | 66 | `conversation_memory_test` | CONV | `MemAvailable` parsing for the parking floor (missing, malformed, zero) |
| [src/core/conversation_snapshot_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/conversation_snapshot_test.cpp) | 324 | `conversation_snapshot_test` | GPU + CONV | snapshot capture and restore against the engine (skip 77) |
| [src/core/conversation_validation_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/conversation_validation_test.cpp) | 239 | `conversation_validation_test`, `conversation_transfer_test` | GPU + CONV | invalid restores return before the first CUDA call or destination write |
| [src/core/coupled_draft_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/coupled_draft_test.cpp) | 115 | `coupled_draft_test` | TESTS or CONV | coupled draft sampling's Philox counters and penalty ring (CPU) |
| [src/core/pinned_shared_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/pinned_shared_test.cpp) | 61 | `pinned_shared_test` | GPU, Linux | `PinnedArena`'s optional `MAP_SHARED` backing |
| [src/kernels/bf16_bits_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/bf16_bits_test.cpp) | 80 | `bf16_bits_test` | TESTS or CONV | f32 -> bf16 rounding vs ggml's rule over all 2^32 inputs, NaN included |
| [src/kernels/dequant_bf16_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/dequant_bf16_test.cpp) | 108 | none (by hand: `dequant_bf16_test SHARD1.gguf [SHARD2.gguf]`) | GPU + TESTS | device dequantizers vs the artifact's CPU ones on a real tensor of every type |
| [src/kernels/cpu/expert_multi_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/expert_multi_test.cpp) | 87 | `expert_multi_test` | TESTS | `s2_expert_vnni_multi` bitwise vs the per-token kernel, n = 1..8 (needs AVX-512) |
| [src/kernels/cpu/expert_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/expert_parity.cpp) | 184 | `expert_parity` | TESTS | the CPU expert path vs a scalar reference on the real `pack/full/experts.bin` |
| [src/kernels/cpu/pool_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_test.cpp) | 224 | `pool_test` | TESTS | the expert pool's small correctness claims, checked directly |
| [src/kernels/cpu/pool_stress.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_stress.cpp) | 99 | `pool_stress` | TESTS | tiny batches with sleeping workers (issue #29); a hang fails it |
| [src/kernels/cpu/pool_affinity_linux_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_affinity_linux_test.cpp) | 290 | `pool_affinity_test` | TESTS, Linux | worker CPU affinity, queried with `CPU_ALLOC` (CPUs past `CPU_SETSIZE`) |
| [src/kernels/cpu/pool_affinity_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_affinity_test.cpp) | 148 | `pool_affinity_test` | TESTS, Windows | the same through Windows CPU Sets |
| [src/ngram/ple_reader_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/ngram/ple_reader_test.cpp) | 298 | `ple_reader_selftest` | GPU | the direct SSD row reader on a synthetic table (90 and 110 B rows); `--gguf SHARD2` by hand on the real one |
| [src/platform/memory_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/platform/memory_test.cpp) | 72 | `platform_memory_test` | GPU | `lock_resident` on 256 MiB (fails under a low `ulimit -l`) and #577's buffered/unbuffered choice |
| [src/program/conv_cache_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/program/conv_cache_test.cpp) | 67 | `conv_cache_test` | TESTS or CONV | serve-loop checkpoint retention: the shared root survives, leaves rotate by LRU |
| [src/spec/controller_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/spec/controller_test.cpp) | 74 | `controller_test` | TESTS | the speculation controller's decisions |
| [src/spec/draft_policy_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/spec/draft_policy_test.cpp) | 63 | `draft_policy_test` | TESTS | when a lookup window beats the MTP's, with RTX 5070 costs |
| [src/spec/suffix_drafter_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/spec/suffix_drafter_test.cpp) | 141 | `suffix_drafter_test` | TESTS | suffix-lookup drafter; `--simulate FILE.ids` replays text offline |

"TESTS" = `STRATA_BUILD_TESTS`, "CONV" = `STRATA_BUILD_CONVERSATION_TESTS`, "GPU" = inside the CUDA/HIP block.

---

## 5. Engine tests in `tests/`

| File | Lines | CTest name | Registered when | Needs | What it checks |
| --- | ---: | --- | --- | --- | --- |
| [tests/core/gguf_fixture.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/core/gguf_fixture.hpp) | 103 | (header) | | | synthetic GGUF v3 writer for the loader tests (no gguf-py) |
| [tests/core/gguf_split_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/core/gguf_split_test.cpp) | 220 | `gguf_split_test` | TESTS | CPU | split shards and their joint view, UD-Q4_K_XL in miniature: missing shard, foreign split keys, duplicates, truncation |
| [tests/core/file_expert_source_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/core/file_expert_source_test.cpp) | 360 | `file_expert_source_test` | GPU (file exists) | CPU | the file expert tier (`FileExpertSource`) |
| [tests/core/expert_profile_save_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/core/expert_profile_save_test.cpp) | 80 | `expert_profile_save_test` | GPU (file exists) | CPU | #477: `--expert-profile-save`'s order and file, byte for byte `tools/make_profile.py`'s format |
| [tests/core/expert_layout_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/core/expert_layout_test.cpp) | 355 | `expert_layout_test` | GPU + TESTS + NATIVE | CPU | `native_experts.txt` v2-v4 on `tests/data/native_experts`, the arena filled from synthetic shards |
| [tests/core/native_dense_ple_key_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/core/native_dense_ple_key_test.cpp) | 169 | `native_dense_ple_key_test` | GPU + TESTS + NATIVE | CUDA (77) | #326: a `--compat-bf16` pack's BF16 PLE key stays in the arena (OrcaRouter IQ3_XXS) |
| [tests/core/expert_cache_per_layer_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/core/expert_cache_per_layer_test.cpp) | 44 | `expert_cache_per_layer_test` | GPU + TESTS + NATIVE | CUDA (77) | #369: per-layer admission on a sized cache uses each layer's own slot range |
| [tests/core/expert_cache_segmented_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/core/expert_cache_segmented_test.cpp) | 99 | `expert_cache_segmented_test` | GPU + TESTS + NATIVE | CUDA VMM (77) | #533: `--vram-elastic` shrink and grow keep every remaining slot's address and bytes |
| [tests/cuda/prefill_fused_moe_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/cuda/prefill_fused_moe_test.cpp) | 420 | `prefill_fused_moe_test` | NATIVE + CUDA + TESTS | sm_80 (77 below) | #136: fused int8 Q2_0 prompt experts vs a double reference and the MMQ path |
| [tests/cuda/prefill_fused_iq_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/cuda/prefill_fused_iq_test.cpp) | 633 | `prefill_fused_iq_test` | NATIVE + CUDA + TESTS | sm_80 (77 below) | the same for IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS gate/up and Q2_0, IQ4_NL down |
| [tests/cuda/prefill_mmq_kquant_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/cuda/prefill_mmq_kquant_test.cpp) | 168 | `prefill_mmq_kquant_test` | + `STRATA_MMQ_KQUANTS` | CUDA | MMQ for UD-Q4_K_XL's Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down vs ggml's dequantized weights; Q6_K must be refused |
| [tests/hip/prefill_gemm.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/hip/prefill_gemm.cpp) | 93 | `hip_prefill_gemm` | HIP | AMD GPU | BF16/F16 prefill through the engine GEMM path vs a CPU reference |
| [tests/hip/q2_zero.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/hip/q2_zero.cpp) | 50 | `hip_q2_zero` | HIP | AMD GPU | Q2_0 dequant keeps signed zero (caught HIP 5.7 / gfx1012 folding it) |
| [tests/hip/expert_cache_staging.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/hip/expert_cache_staging.cpp) | 34 | `hip_expert_cache_staging` | HIP | AMD GPU | sized slots, repeated pageable payloads, close and reopen through the staging buffer |
| [tests/hip/intrinsics.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/hip/intrinsics.cpp) | 236 | `hip_intrinsics` | HIP | AMD GPU | the HIP compatibility layer's CUDA intrinsics, numerically |
| [tests/hip/handoff.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/hip/handoff.cpp) | 122 | `hip_handoff` | HIP | AMD GPU | asynchronous CPU/GPU handoff and graph replay with changing payloads |
| [tests/hip/mapped_alias.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/hip/mapped_alias.cpp) | 108 | `hip_mapped_alias` | HIP | AMD GPU (77) | #325: what a `hipHostGetDevicePointer` alias is on this stack; a diagnosis, exits 0 whenever it ran |
| [tests/hip/native_qsa_score.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/hip/native_qsa_score.cpp) | 95 | `hip_native_qsa_score` | HIP | AMD GPU | the scalar HIP native QSA scorer |
| [tests/hip/prefill_native_batch.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/hip/prefill_native_batch.cpp) | 293 | `hip_prefill_native_batch` | HIP | AMD GPU | native QSA append and embedding gather ported from a community PR (#108) |
| [tests/hip/router_fast.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/hip/router_fast.cpp) | 101 | `hip_router_fast` | HIP (gfx906: built, not registered) | AMD GPU | the AMD fast router bitwise vs the portable one on 65,536 rows incl. ties and NaN, plus timings |
| [tests/hip/prefill_hipblaslt_gemm.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/hip/prefill_hipblaslt_gemm.cpp) | 197 | `hip_prefill_hipblaslt_gemm` | HIP + hipBLASLt | `STRATA_HIPBLASLT_TUNING` (77) | the calibrated hipBLASLt table loads and its two rows match hipBLASEx |
| [tests/hip/prefill_mmq_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/hip/prefill_mmq_parity.cpp) | 281 | `hip_prefill_mmq_parity` | HIP + `STRATA_PREFILL_MMQ` | AMD GPU | GGML MMQ vs a CPU FP32 product of ggml-dequantized blocks (a layout screen, not bitwise) |
| [tests/hip/ple_iq4.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/hip/ple_iq4.cpp) | 202 | none (`hip_ple_iq4 <shard1.gguf>`) | HIP | a real shard | PLE IQ3_XXS/IQ4_XS matrices: ggml CPU dequant vs HIP native MMVQ |
| [tests/platform/direct_file_async.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/platform/direct_file_async.cpp) | 60 | `direct_file_async_test` | TESTS, Unix non-Apple | CPU | `DirectFile` asynchronous reads on a temp file |

### Fixture data: `tests/data/native_experts/`

Six copies of real packs' `native_experts.txt` (49 lines each: a header and 48 layers), read by
`expert_layout_test`. The header names the model file, the expert count and the total bytes; [models.md](models.md)
uses them as the source for per-model expert geometry.

| File | Header | Experts | Total expert bytes |
| --- | --- | ---: | ---: |
| [q2_0.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/data/native_experts/q2_0.txt) | v2 | 512 | 33,973,862,400 |
| [iq2_xs.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/data/native_experts/iq2_xs.txt) | v2 | 512 | 35,454,976,000 |
| [iq3_xxs.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/data/native_experts/iq3_xxs.txt) | v2 | 512 | 42,912,972,800 |
| [iq3_s.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/data/native_experts/iq3_s.txt) | v3 | 512 | 50,292,326,400 |
| [coder-iq1_m.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/data/native_experts/coder-iq1_m.txt) | v3 | 256 | 25,146,163,200 |
| [swift-iq2_xs.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/data/native_experts/swift-iq2_xs.txt) | v3, shard column on layers 13-47 | 512 | 35,454,976,000 |

---

## 6. Kernel parity programs

38 programs hold a kernel to a reference: a host implementation, the previous kernel bit for bit, ggml's
dequantizers, or an FP64 product. 35 are `src/kernels/*_parity.cpp`, two are `src/prefill/gdn_rec_parity.cu` and
`gemm_bf16_parity.cu`, one is `src/kernels/cpu/router_dot_parity.cpp` (`expert_parity.cpp` is listed in section 4).
All need a GPU except `router_dot_parity`. Synthetic data, no model, except `ple_parity`, `ple_q5_parity`,
`q5_projection_parity`, `ple_fp8_parity` and `native_expert_parity` when given shards.

- 31 of the 35 `src/kernels/*_parity.cpp` are registered under their own name, most as `<name> --selftest`.
  Not registered: `qsa_prompt_attn_parity` (CUDA; on HIP it is registered as `hip_prompt_attn_wmma 32768 2048 3`),
  `ple_q5_parity`, `q5_projection_parity`, `ple_fp8_parity`.
- `native_expert_parity` is registered nine times with flags (section 3) and also takes real shards by hand
  (`docs/ORCA_Q4_K_S.md:22`: `native_expert_parity SHARD1 SHARD2 SHARD3 0 5 6 20 47`).
- Backend limits: `mmvq_multi_parity` and `decode_cluster_parity` are CUDA only; `gdn_rec_parity` and
  `gemm_bf16_parity` CUDA and not gfx906 (`cp.async`, the occupancy API); `qsa_topk_active_parity` CUDA + TESTS;
  `decode_cluster_parity` skips below sm_90 (`CMakeLists.txt:705-712`).
- Benchmarks without an assertion are separate targets: `fused_gr_bench`, `native_expert_bench`; `qsa_select_bench` is
  a HIP CTest entry that fails on its FP64 accuracy gate. Several parity programs take `--bench` for timings.

See [../kernels/parity/README.md](../kernels/parity/README.md) for each program.

### The SYCL port's kernel tests

`sycl/CMakeLists.txt:177-217` registers 25 tests (5 `add_test` calls): a `foreach` over 21 parity programs
(`dequant_s2 s2_gemv iq shared_expert gr gdn s2_gemv_q8 sampler rope quantize_act router_top10 s_gemv elementwise
bf16_gemv s_gemv_q8k qsa ple kv_q8 kv_stream kv_q4 cvec`, each `--selftest`), then `iq_multi_parity`,
`s2_expert_grouped_parity`, `native_grouped_parity` and `conversation_snapshot_test` (skip 77, 120 s). Each source is
resolved by `strata_resolve` (`sycl/CMakeLists.txt:56-70`): the port's own copy under `sycl/` first, else the
engine tree's file. Built, not registered: `qsa_prompt_attn_parity` and `kv_hybrid_parity` (both wait for the
tensor-core prompt kernel), `native_expert_parity`, and the benches `mmvq_bench`, `mmvq_sg_bench`, `q6k_align_bench`,
`gr_bench`, `xmx_gemm_bench`. `gdn_rec_parity` is not built ("an SM-holding NVIDIA bench"). The maintainers
compile the port and run its kernel tests but have no Intel GPU (`docs/INTEL_ARC.md:8-11`).

---

## 7. Python tests

### Server (`serve/test_*.py`): no GPU, no pack

All but `test_detok` run the real `Service` and HTTP server against `MockEngine` or a fake engine script.

| File | Lines | Tests | Covers |
| --- | ---: | ---: | --- |
| [serve/test_server.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_server.py) | 2,685 | 139 | the max-tokens budget over both APIs and most of the server's behaviour against the mock engine |
| [serve/test_mcp.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_mcp.py) | 447 | 25 | tools from MCP servers (`serve/mcp.py`) and the web app's tool loop, against `serve/mcp_fake_server.py` (no MCP SDK) |
| [serve/test_security.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_security.py) | 280 | 22 | the Host check (DNS rebinding) and the Origin check without an API key |
| [serve/test_responses.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_responses.py) | 361 | 19 | #451: the Responses API (`POST /v1/responses`) |
| [serve/test_parallel.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_parallel.py) | 382 | 11 | #465: `"parallel": N` batch slots, via a fake engine speaking `GEN / BGEN / BT / BDONE / BADM / BSTOP` |
| [serve/test_monitor.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_monitor.py) | 261 | 10 | request inspection through HTTP |
| [serve/test_runconfig.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_runconfig.py) | 165 | 9 | #564: the Settings view of the run config (`GET / POST /config`) |
| [serve/test_detok.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_detok.py) | 252 | 8 | the incremental detokenizer equals the old re-decode, token by token; skipped without a pack tokenizer |
| [serve/test_lifecycle.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_lifecycle.py) | 170 | 8 | model discovery, on-demand loading and safe unloading over HTTP |
| [serve/test_structured.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_structured.py) | 167 | 8 | the JSON response contract; two tests skip without `jsonschema` |
| [serve/test_vram.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_vram.py) | 136 | 7 | #533: `POST /v1/vram`, the engine command, the config key, the reserve kept across unload |
| [serve/test_winjob.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_winjob.py) | 67 | 2 | `serve/winjob.py`: a contained child ends with its parent; skipped outside Windows |

Fixtures beside them: `serve/mcp_fake_server.py` (stdio or Streamable HTTP MCP server with `echo`, `fail`, `sleep`,
`big`, `die`, `add` tools). `serve/chat_golden.json` (10 cases) is named by `serve/frontend.py:7` as the template
check's reference, but no script in the snapshot reads it.

### Tools and setup (`tools/test_*.py`): no GPU, no download

| File | Lines | Tests | Needs | Covers |
| --- | ---: | ---: | --- | --- |
| [tools/test_setup_unsloth.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_unsloth.py) | 577 | 37 | stdlib | UD-IQ4_XS and UD-Q4_K_XL: pinned shards, RAM budget, `--compat-bf16` pack, engine version, one GPU |
| [tools/test_setup_choices.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_choices.py) | 441 | 35 | stdlib | PC-dependent choices: image encoder (#331), `--gguf-dir` (#305), Pascal/Volta (#295), AMD CPU encoder (#304) |
| [tools/test_setup_risk.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_risk.py) | 414 | 34 | stdlib | "recommends, never forces": bigger choices kept with their risk (#406 #364 #384 #403) |
| [tools/test_setup_amd.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_amd.py) | 454 | 28 | stdlib | AMD detection on a mocked KFD topology, HIP numbering, TheRock index per family |
| [tools/test_strata_mcp.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_strata_mcp.py) | 590 | 23 | stdlib | `tools/strata_mcp.py`: JSON-RPC handshake, argument checks, install / start / stop against fakes |
| [tools/test_iq_pack.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_iq_pack.py) | 528 | 23 | numpy, gguf-py | `iq_pack.py`: split shards, v3/v4 tables, `--compat-bf16`, refusals |
| [tools/test_setup_older_gpus.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_older_gpus.py) | 287 | 19 | stdlib | Pascal/Volta on the CUDA 12 engine, `--cuda 12|13` |
| [tools/test_setup_rope.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_rope.py) | 121 | 19 | stdlib | automatic rope scaling from the final context (factor = final / 262144) |
| [tools/test_setup_pins.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_pins.py) | 280 | 16 | stdlib | #214 reproducible installs: engine release, pinned HF revisions, pinned requirements |
| [tools/test_calibrate.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_calibrate.py) | 234 | 14 | stdlib | `tools/calibrate.py` with a stand-in engine whose speed is a function of its settings |
| [tools/test_setup_config.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_config.py) | 244 | 13 | stdlib | what a rerun keeps in the run config (#629), browser choice (#609), image-token flag (#625) |
| [tools/test_conversation_cache_parity.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_conversation_cache_parity.py) | 170 | 12 | stdlib | the model-level gate cannot pass with incomplete evidence |
| [tools/test_setup_update.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_update.py) | 145 | 8 | stdlib | `setup.py --update` refreshes, never starts |
| [tools/test_mtp_fetch.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_mtp_fetch.py) | 201 | 8 | stdlib (mocked HTTP) | #327: range reads need a 206, sha256 per tensor, re-fetch on mismatch |
| [tools/test_setup_draft_vocab.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_draft_vocab.py) | 109 | 7 | stdlib | #287: a rerun keeps the draft subset the config chose |
| [tools/test_setup_oldcpu.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_oldcpu.py) | 95 | 7 | stdlib (one test Windows only) | the older-CPU (no AVX2) build floor |
| [tools/test_setup_prompts.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_prompts.py) | 89 | 7 | stdlib | prompts need input or `--yes`; EOF never authorizes installs (#393) |
| [tools/test_make_profile.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_make_profile.py) | 104 | 7 | stdlib | `make_profile.py` on synthetic profiles and traces |
| [tools/test_shards.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_shards.py) | 119 | 7 | stdlib | a missing or short shard is named; a pack's verify reads back its source hash |
| [tools/test_setup_remote_opt.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_remote_opt.py) | 74 | 6 | stdlib | #578: `--remote-expert-opt` recommended on 2+ GPUs only |
| [tools/test_setup_sycl.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_sycl.py) | 75 | 6 | stdlib | `--backend sycl` hands off to `sycl/setup_intel.py` on Linux, stops on Windows |
| [tools/test_conversation_cache_disabled.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_conversation_cache_disabled.py) | 104 | 6 | stdlib | the cache-disabled comparison rejects mismatches |
| [tools/test_setup_lowram.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_lowram.py) | 84 | 5 | stdlib | #250 low-RAM messages and the paging question |
| [tools/test_setup_golden.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_golden.py) | 210 | 4 | stdlib | `--yes` on several mocked PCs writes the run config of `tools/test_setup_golden.json` byte for byte |
| [tools/test_setup_hybrid.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_hybrid.py) | 65 | 4 | stdlib | #642: pool workers on a hybrid CPU |
| [tools/test_draft_vocab.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_draft_vocab.py) | 47 | 4 | stdlib | #597: `draft_vocab.py --corpus` coverage |
| [tools/test_conversation_cache_growth.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_conversation_cache_growth.py) | 79 | 4 | stdlib | negative controls for the incremental capture gate |
| [tools/test_conversation_cache_http_smoke.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_conversation_cache_http_smoke.py) | 71 | 4 | stdlib | the HTTP smoke test's orchestration and safety defaults |
| [tools/test_conversation_cache_isolation.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_conversation_cache_isolation.py) | 82 | 4 | stdlib (imports `tools.gguf_reader`) | image/steering isolation evidence and fixtures |
| [tools/test_conversation_cache_soak.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_conversation_cache_soak.py) | 73 | 4 | stdlib | the known-answer soak gate |
| [tools/test_setup_parallel.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_parallel.py) | 53 | 3 | stdlib | #465: setup's `"parallel"` recommendation |
| [tools/test_conversation_restore_speed.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_conversation_restore_speed.py) | 34 | 2 | stdlib | #528: the restore-speed bench's flag overrides and pass rule |

The 18 `test_setup_*.py` files (258 tests) import `setup.py` and mock every outside effect. Several reuse
`test_setup_golden.py`'s `PROFILES` and `install()`.

### Live harnesses (a built engine, a model, often a GPU)

Each exits non-zero on a failure. The `conversation_cache_*` ones are dry runs unless given `--run`.

| Script | Lines | Drives | What it checks |
| --- | ---: | --- | --- |
| `tools/batch_test.py` | 161 | engine (`--exe`, `--config`) | batch slots: every slot's greedy tokens equal its solo tokens (`docs/BATCHING.md:174-196`) |
| `tools/batch_interleave_test.py` | 237 | engine | a long prompt read while slots decode, `BYIELD`, a next turn from its slot |
| `tools/parking_test.py` | 89 | engine | a follow-up decodes the same tokens live or from the parking cache |
| `tools/early_close_test.py` | 47 | server URL | a client that stops reading early leaves no tokens to the next request |
| `tools/needle_bench.py` | 118 | server URL | needle-in-a-haystack recall at chosen lengths and depths (also used by every benchmark report) |
| `tools/conversation_cache_parity.py` | 253 | private engines | full-model A/B/A parity of the conversation cache |
| `tools/conversation_cache_disabled.py` | 196 | private engines | cache-disabled engines against untouched upstream |
| `tools/conversation_cache_growth.py` | 170 | private engines | full vs incremental snapshot capture during growth and rewind |
| `tools/conversation_cache_isolation.py` | 147 | private engines | image / steering cache isolation with synthetic embeddings and control vectors |
| `tools/conversation_cache_soak.py` | 169 | private engines | known-answer A/B/A soak |
| `tools/conversation_cache_http_smoke.py` | 137 | an exclusive idle server | evictions and a dropped stream over HTTP |
| `tools/conversation_restore_speed.py` | 182 | private engines | #528: decode speed after a restore vs live |
| `tools/canonical_xcheck.py` | 799 | a real GGUF | the canonical (S2) form round-trips bit-exactly against gguf-py / ggml |
| `tools/hip/bench_prefill.py` | 75 | server URL | matched fresh / follow-up prefill probe for HIP arms |

For exact comparisons the scripts want `--pcie-frac 0 --adapt-every 1000000` and set `STRATA_IQ_MT_MIN=1`
(`docs/BATCHING.md:186`). These harnesses belong to the [tools](../tools/README.md) area.

---

## 8. Named in the build or the code but absent from the snapshot

- `tests/CMakeLists.txt` and `bench/micro/` (hence the OFF default, the 16 oracle parity sources, `native_mmvq_multi`,
  `graph_registry_test`, `pinned_capture`, `hit_cpu_order_parity`, and `ple_parity`'s default fixture directory).
- `pack/full/experts.bin` (`expert_arena_load`, `expert_parity`) and the `../../Q2_0/` shard `ple_parity` defaults to.
- `tools/gpu_session.py` (named for `dequant_bf16_test`, `CMakeLists.txt:1203`).
- `third_party/llama.cpp` (gguf-py for the Python fixtures; `setup.py` clones it).

## Gotchas

- Configuring with `STRATA_ORACLE_SOURCE_DIR` and `STRATA_ORACLE_BUILD_DIR` fails on the snapshot: the loop adds
  `bench/micro/${parity}_parity.cpp`, which does not exist (`CMakeLists.txt:392`).
- A GPU build without `STRATA_BUILD_TESTS` still registers about 40 parity and platform tests, because they are not
  behind that option; `ctest` in such a tree runs them.
- `platform_memory_test` needs a memlock limit of at least 256 MiB; `expert_multi_test` needs AVX-512; `ple_parity` and
  `expert_parity` need real Q2_0 artifacts. These are the recorded host failures (`docs/AMD_HIP.md:215-218, 295`).
- `docs/ORCA_Q4_K_S.md` says Q5_0 down projections have no GPU kernel and the engine refuses such a layer, but
  `src/kernels/cuda/iq_kernels.cu:526` lists ggml type 6 (Q5_0) among the down formats and
  `native_expert_parity_q4_K_q5_0` / `_q5_0_q8_0` are registered (engine 0.1.38, #473, per `setup.py:118`); the doc
  predates that.
- `serve/test_vram.py` imports `ResidentEngine` from `serve/test_lifecycle.py`; run them from the repository root so
  `serve` is importable.

## In UnbiasedStrata

Kept: `tests/core`, `tests/cuda`, `tests/platform`, `tests/data/native_experts` (restored in c00942d after 8894a81
removed `tests/`), every `src/**/*_parity.*` and `*_test.cpp`. Changed: several of those (`expert_layout_test`,
`prefill_fused_iq_test`, `prefill_mmq_kquant_test`, seven parity programs) for Qwen3.6-35B-A3B; added Qwen3.6 tests,
fixtures and `tests/python/`. Removed: `tests/hip/` (12 files), all of `serve/` (and its 12 test files), 28 of the 32
`tools/test_*.py` (kept `test_draft_vocab`, `test_iq_pack`, `test_make_profile`, `test_mtp_fetch`), the live harnesses
and the SYCL port. The fork's index: [../../codebase/guides/testing.md](../../codebase/guides/testing.md).

## See also

- [../kernels/parity/README.md](../kernels/parity/README.md) - each parity program.
- [../build/README.md](../build/README.md) - `STRATA_BUILD_TESTS`, `STRATA_MMQ_KQUANTS`, `STRATA_PREFILL_MMQ` and the
  other options.
- [../serve/README.md](../serve/README.md), [../tools/README.md](../tools/README.md), [../sycl/README.md](../sycl/README.md).
- [benchmarks.md](benchmarks.md) - measurements, as opposed to pass/fail tests.
- [models.md](models.md) - the fixture files' models.
