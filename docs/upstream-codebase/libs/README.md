[Upstream docs](../README.md) › Libraries

# Support libraries

The support libraries are the small, mostly GPU-free building blocks under the engine; `src/program/generate.cpp`
uses them directly and through the core and kernel layers. **Artifact** (`strata_artifact`, header-only) parses GGUF
v3 model files and split shards, guards the architecture (`qwen4exp`, Qwen3.8-Flash-Next) and provides scalar
reference dequantizers. **Platform** wraps OS-specific I/O and memory: unbuffered asynchronous reads (Windows IOCP,
Linux `O_DIRECT`), locking the expert arena into RAM, the machine's RAM size, the Windows DXGI shared-memory budget,
and the gfx906 HIP compatibility headers. **ngram** (`PleReader`) streams the n-gram (PLE) table's rows straight from
the SSD for `PleTable` in `src/kernels/ngram.cpp`. **Spec** (`strata_spec`) holds the suffix-lookup drafter and the
online policy that chooses the MTP's window or a lookup window per verify round. **Plan** holds the P1.S9 memory
planner and the `strata-plan` CLI. Apart from `ngram/` and parts of `platform/`, which are compiled into the GPU
libraries, everything here builds and tests without CUDA or a model.

## Contents

| Page | Description |
| --- | --- |
| [spec.md](spec.md) | Speculation policy: `SuffixDrafter` (prompt lookup), `DraftPolicy` (MTP vs lookup window), the unused `Controller` |
| [platform.md](platform.md) | `DirectFile` (IOCP / `O_DIRECT` async reads), `memory.hpp` (lock resident, RAM, DXGI budget, file-cache arithmetic), `hip_compat/` (gfx906 CUDA-name mapping); no `robust_read.hpp` upstream |
| [plan.md](plan.md) | `plan.hpp` memory planner, `strata-plan`, and its use in `strata-device` |
| [artifact.md](artifact.md) | GGUF v3 reader, split-shard model view, architecture guard, scalar dequantizers, `strata-gguf` / `strata-dequant`, where pack loading lives |
| [ngram-ple.md](ngram-ple.md) | `PleReader`: ticketed, deduplicated, cached direct reads of the PLE table, SSD keep-alive |

## Source map

| Path | Lines | Role |
| --- | ---: | --- |
| [include/strata/spec/suffix_drafter.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/spec/suffix_drafter.hpp) | 53 | Suffix-lookup drafter API |
| [src/spec/suffix_drafter.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/spec/suffix_drafter.cpp) | 88 | Trigram hash index, longest-match proposal |
| [src/spec/suffix_drafter_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/spec/suffix_drafter_test.cpp) | 141 | Unit tests and an offline `--simulate` replay |
| [include/strata/spec/draft_policy.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/spec/draft_policy.hpp) | 52 | MTP-vs-lookup window policy API |
| [src/spec/draft_policy.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/spec/draft_policy.cpp) | 101 | Learned costs and acceptance, the choice rule |
| [src/spec/draft_policy_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/spec/draft_policy_test.cpp) | 63 | Policy tests on simulated rounds |
| [include/strata/spec/controller.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/spec/controller.hpp) | 73 | Cost-model speculation controller (tests only) |
| [src/spec/controller.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/spec/controller.cpp) | 73 | Controller implementation |
| [src/spec/controller_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/spec/controller_test.cpp) | 74 | Controller tests |
| [include/strata/platform/direct_file.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/platform/direct_file.hpp) | 60 | Unbuffered async file reads API |
| [src/platform/direct_file.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/platform/direct_file.cpp) | 365 | Windows IOCP and POSIX `O_DIRECT` thread-pool implementations |
| [include/strata/platform/memory.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/platform/memory.hpp) | 46 | Lock resident, RAM size, DXGI budget, `file_cache_keeps` |
| [src/platform/memory.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/platform/memory.cpp) | 138 | Windows / POSIX implementations |
| [src/platform/memory_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/platform/memory_test.cpp) | 72 | `file_cache_keeps` cases and a lock test |
| [include/strata/platform/hip_compat/](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/platform/hip_compat/strata_hip.h) | 329 (5 files) | `strata_hip.h` (295) and the `cuda_runtime.h`, `cuda_fp16.h`, `cublas_v2.h`, `math_constants.h` shims of the gfx906 build |
| [include/strata/plan/plan.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/plan/plan.hpp) | 168 | P1.S9 memory planner (header-only) |
| [src/plan/plan_main.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/plan/plan_main.cpp) | 73 | `strata-plan` CLI |
| [include/strata/artifact/gguf_reader.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/artifact/gguf_reader.hpp) | 606 | GGUF v3 reader, `GgufModel`, architecture guard |
| [include/strata/artifact/gguf_split.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/artifact/gguf_split.hpp) | 52 | Split-shard path resolution |
| [include/strata/artifact/dequant.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/artifact/dequant.hpp) | 292 | Scalar reference dequantizers |
| [src/artifact/gguf_reader.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/artifact/gguf_reader.cpp) | 67 | `strata-gguf` CLI |
| [src/artifact/dequant.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/artifact/dequant.cpp) | 123 | `strata-dequant` CLI |
| [src/artifact/gguf_reader_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/artifact/gguf_reader_test.cpp) | 93 | Duplicate-name refusal test |
| [include/strata/ngram/ple_reader.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/ngram/ple_reader.hpp) | 100 | `PleReader` API |
| [src/ngram/ple_reader.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/ngram/ple_reader.cpp) | 547 | PLE table direct reader |
| [src/ngram/ple_reader_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/ngram/ple_reader_test.cpp) | 298 | Synthetic selftest and real-table check |

Related tests outside these folders:
[tests/platform/direct_file_async.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/platform/direct_file_async.cpp)
and [tests/core/gguf_split_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tests/core/gguf_split_test.cpp)
(220 lines). The Intel SYCL port compiles `src/spec` as is and keeps its own copies of `ple_reader.cpp` and
`direct_file.cpp` under `sycl/src/` (`sycl/CMakeLists.txt:84`, `:150`).

## Build targets

| Target | Kind | Sources | Needs a GPU backend | `CMakeLists.txt` |
| --- | --- | --- | --- | --- |
| `strata_artifact` | INTERFACE (header-only) | `include/strata/artifact/` | no | 214-216 |
| `strata-gguf`, `strata-dequant` | executables | `src/artifact/*.cpp` | no | 218-224 |
| `strata_plan` | INTERFACE (header-only) | `include/strata/plan/` | no | 239-241 |
| `strata-plan` | executable | `src/plan/plan_main.cpp` | no | 243-244 |
| `strata_spec` | STATIC | `src/spec/{suffix_drafter,controller,draft_policy}.cpp` | no | 899-900 |
| (in `strata_core`) | | `src/platform/memory.cpp` | yes | 249 |
| (in `strata_kernels`) | | `src/ngram/ple_reader.cpp`, `src/platform/direct_file.cpp` | yes | 333-335 |

`strata_plan` is linked publicly into `strata_core` (`:259`) and `strata_spec` into the `strata` executable (`:497`).
The CPU-only tests (`gguf_reader_test`, `gguf_split_test`, the spec tests, `direct_file_async_test`) are registered
only with `-DSTRATA_BUILD_TESTS=ON`, which defaults to OFF in this snapshot because `tests/CMakeLists.txt` and
`bench/micro` are not published (`CMakeLists.txt:86-92`); `platform_memory_test` and `ple_reader_selftest` are
registered in every GPU build (`:833-847`). Each leaf page lists its tests.

## Key concepts and data flow

```
                     model files (.gguf, maybe split -00001-of-0000N)
                                       |
                 gguf_split_paths --> GgufModel / GgufFile  (artifact: mmap, parse, guard)
                    |                        |                         |
      native head / dense loaders     PLE table offset          expert source checks
      (core: native_head, native_dense)     |                  (tensor_payload_bytes)
                                     PleReader (ngram)  --issue/collect-->  PleTable (kernels/ngram.cpp)
                                            |
                                     DirectFile (platform: O_DIRECT / IOCP, STRATA_IO_THREADS)

  expert arena (core: pinned.cu, expert_source.cpp) --> lock_resident / unlock_resident (platform)
                                                        total_physical_memory, gpu_shared_memory_budget,
                                                        file_cache_keeps

  verify loop (generate.cpp):  SuffixDrafter.propose --> DraftPolicy.choose(T_mtp, k, match)
                               --> verify window --> DraftPolicy.observe(lookup, T, accepted, match, round_ms)
```

- **Refuse early.** The GGUF reader refuses duplicate names, inconsistent split keys and missing shards at open;
  `plan.hpp` throws `DoesNotClose` instead of returning a plan it cannot honour.
- **Keep the PLE table out of RAM.** `GgufFile` only supplies the table's offset; `PleReader` reads 4 KiB pages
  unbuffered, with a small bounded row cache, so neither the table nor the OS file cache holds it.
- **Output-neutral speculation.** The spec library only decides which drafts to verify; verification keeps the output
  identical to plain decoding.

## In UnbiasedStrata

| Part | Fork status |
| --- | --- |
| spec, ngram, plan (`plan.hpp`, `plan_main.cpp`), `direct_file`, `memory`, `dequant`, `gguf_split`, the CLIs and tests | kept unchanged (no diff against 12a7825) |
| `gguf_reader.hpp` | changed in e41dfc7: the architecture guard also accepts `qwen35moe` (Qwen3.6-35B-A3B), checking its keys for presence only |
| `include/strata/platform/robust_read.hpp` | added by the fork (commit 9a53e42: retrying reads for slow or remote stores); upstream has no such file |
| `include/strata/plan/vram_plan.hpp` | added by the fork (e41dfc7, extended in 6042a32): the Qwen3.6 VRAM planner |
| `hip_compat/` | kept in the tree; the fork does not build HIP |
| SYCL copies (`sycl/src/ngram`, `sycl/src/platform`) | removed with the SYCL port |

The fork's own pages: [../../codebase/libs/README.md](../../codebase/libs/README.md).

## See also

- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md), "How it works" (the memory tiers, the SSD-resident n-gram
  table, speculation).
- The [glossary](../guides/glossary.md); the core area (expert source, pinned arena, native head) and the kernels
  area (`PleTable`).
