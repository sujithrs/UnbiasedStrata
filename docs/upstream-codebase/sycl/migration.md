[Upstream docs](../README.md) › [Intel SYCL port](README.md) › Migration

# How the port is produced and kept current

## Purpose

The SYCL engine is generated from the CUDA engine, not written beside it. This page covers the toolchain that does
it: the development image, the CUDA headers, the dpct run, the normalisation, the scripted corrections, the 3-way
merge used after each upstream release, and the build script. The hand-written changes that came after the
mechanical steps are in [differences.md](differences.md).

Files covered:

| File | Lines | Role |
|---|---|---|
| [sycl/tools/Dockerfile](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/Dockerfile) | 9 | The `strata-sycl-dev` image |
| [sycl/tools/Dockerfile.unitrace](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/Dockerfile.unitrace) | 8 | The same image plus Intel's `unitrace` |
| [sycl/tools/get-cuda-headers.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/get-cuda-headers.sh) | 28 | CUDA headers for dpct, from pip wheels |
| [sycl/tools/migrate.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/migrate.sh) | 47 | The dpct run |
| [sycl/tools/normalize.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/normalize.sh) | 26 | A dpct output in the port's shape |
| [sycl/tools/fixups.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/fixups.py) | 305 | Scripted corrections of dpct's output |
| [sycl/tools/merge_upstream.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/merge_upstream.py) | 37 | 3-way merge of a re-migration |
| [sycl/tools/build.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/build.sh) | 12 | Configure and build in the image |
| [sycl/.gitignore](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/.gitignore) | 4 | dpct's by-products |

The procedure itself is written down in upstream's
[docs/INTEL.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INTEL.md),
sections "How it was made, so it can be redone" and "Keeping up with upstream".

## The pipeline

```
 get-cuda-headers.sh  -> cuda-headers/include   (mounted at /cuda-headers in the image)
 Dockerfile           -> strata-sycl-dev        (oneAPI compiler + MKL + Level Zero + dpct + ninja)

 migrate.sh  [repo] [out]
   compile_commands.json for: every src/**/*.cu  +  every src/**/*.cpp that includes a CUDA header
                              minus moe_mmq.cu, ggml_cuda_host.cu, native_expert_parity.cpp
   dpct -p db --in-root repo --out-root out ...   -> out/src/**/X.dp.cpp, out/src/**/X.cpp.dp.cpp, headers,
                                                      include/dpct/ (helper headers), migrate.log, dpct_report.*
 normalize.sh <out> <commit>
   X.cpp.dp.cpp -> X.cpp;  drop *.yaml, reports;  drop files byte-identical to upstream at <commit>;
   DPCTnnnn:<serial>: -> DPCTnnnn: ;  copy + run tools/fixups.py
 (first port)  hand edits on top -> sycl/
 (an upstream release)  migrate + normalize the OLD tree -> sycl-base, the NEW tree -> sycl-new
                        merge_upstream.py: git merge-file sycl/F sycl-base/F sycl-new/F
 build.sh    cmake -S sycl -B build-sycl -G Ninja -DCMAKE_CXX_COMPILER=icpx [-DSTRATA_SYCL_AOT=...]
```

## The development image

`sycl/tools/Dockerfile:4` starts `FROM ghcr.io/snailium/llama.cpp-sycl-intel-b70/llama-sycl-b70:stable`, a
community llama.cpp SYCL image that already carries the oneAPI 2026.1 compiler, oneMKL and Level Zero for Xe2
(`Dockerfile:2`). It adds `intel-oneapi-dpcpp-ct` (SYCLomatic, the "DPC++ Compatibility Tool", invoked as `dpct`),
`ninja-build`, `unzip`, `python3-pip`, `python3-venv` and `rsync` (`:5-7`), sets
`ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_CACHE_PERSISTENT=0 ZES_ENABLE_SYSMAN=1` (`:8`) and uses
`bash -lc` as entry point (`:9`). The same image is the runtime of the served engine (`strata-sycl.sh`, see
[serving.md](serving.md)). Upstream's
[docs/INTEL_ARC.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INTEL_ARC.md)
points out that the base is a community image, not an Intel or Strata one.

`Dockerfile.unitrace` builds `strata-sycl-dev:unitrace`: it clones `intel/pti-gpu` at its default branch
(`--depth 1`, unpinned), builds `tools/unitrace` and puts it on `PATH`. INTEL.md mentions a further
`strata-sycl-dev:metrics` image and `sycl/rank_kernels.py`; neither is in this snapshot.

## CUDA headers without a CUDA toolkit

dpct parses CUDA source, so it needs the CUDA headers but no GPU or toolkit. `get-cuda-headers.sh [out]` runs
`pip download --no-deps` for five NVIDIA wheels pinned to CUDA 12.8 (`nvidia-cuda-runtime-cu12==12.8.90`,
`nvidia-cuda-nvcc-cu12==12.8.93`, `nvidia-cublas-cu12==12.8.4.1`, `nvidia-cuda-cccl-cu12==12.8.90`,
`nvidia-curand-cu12==10.3.9.90`; `:7-8`), extracts every member under `/include/` (`:9-24`) and checks that
`cuda.h`, `cuda_runtime.h`, `cuda_runtime_api.h`, `crt/host_defines.h`, `cuda_fp16.h`, `cuda_bf16.h` and
`cublas_v2.h` are present (`:26-27`). The output is meant to be mounted at `/cuda-headers` in the image.
`PYTHON` selects the interpreter.

## migrate.sh: the dpct run

`migrate.sh [repo] [out]` (`repo` defaults to `/work/Strata_B70`, `out` to `$repo/sycl`; `CUDA_HEADERS` defaults
to `/cuda-headers/include`; `:8-10`). It sources `/opt/intel/oneapi/setvars.sh` and puts the dpct `bin` first on
`PATH` (`:11-12`).

1. **Compilation database** (`:14-35`). There is no `nvcc` in the image, so CMake cannot produce one. An inline
   Python script lists `find src -name '*.cu'` plus every `src/**/*.cpp` that mentions `cuda_runtime`,
   `cublas_v2`, `cuda_fp16` or `cuda.h`, drops three files (`moe_mmq.cu` and `ggml_cuda_host.cu` need llama.cpp's
   ggml-cuda sources; `native_expert_parity.cpp` needs `ggml-cpu.h`), and writes one entry per file: `nvcc` with
   `--cuda-gpu-arch=sm_80 -x cuda` for `.cu`, `g++` otherwise, both with `-std=c++20 -Iinclude -Ithird_party/ggml
   -I<headers> -DSTRATA_VERSION="port"`. The database lives in a `mktemp` directory removed on exit.
2. **dpct** (`:37-47`) with:
   - `--use-experimental-features=graph,matrix,logical-group,masked-sub-group-operation,occupancy-calculation,free-function-queries,local-memory-kernel-scope-allocation,bindless_images,virtual_mem,root-group,non-uniform-groups`
     (the comment names the two that matter most: `graph` turns CUDA graphs into `sycl_ext_oneapi_graph`, `matrix`
     turns `mma` into `joint_matrix`);
   - `--gen-helper-function` (copies the helper headers into the output, which is why `sycl/include/dpct/` exists);
   - `--always-use-async-handler --sycl-named-lambda --enable-ctad --optimize-migration --assume-nd-range-dim=3`;
   - `--report-type=all --report-format=csv --report-file-prefix=dpct_report`, `--stop-on-parse-err=false`;
   - `DPCT_EXTRA` appends further flags. Output goes to `$out/migrate.log`; the script prints `MIGRATE EXIT` and
     the number of files written.

INTEL.md records the first run: 86 CUDA-touching translation units, 85 migrated, no line dpct could not migrate,
about 1,400 advisory notes. The committed tree still carries about 1,500 `DPCTnnnn` comments; the most common are
DPCT1114 (a `cudaMemcpy` became an asynchronous copy, "call wait() if needed"), DPCT1065 (barrier fence space),
DPCT1010 / DPCT1009 (error codes became exceptions), DPCT1108 (masked sub-group shuffles are experimental) and
DPCT1124 (`cudaMemcpyAsync`). DPCT1114 is the one that caused a real bug (rule 7d below).

Gotchas:
- The header comment calls it `sycl/migrate.sh`, and `sycl/CMakeLists.txt:7` says `../../sycl/migrate.sh`; the
  script is at `sycl/tools/migrate.sh`.
- The skip list excludes `moe_mmq.cu` and `ggml_cuda_host.cu`, yet `sycl/src/prefill/moe_mmq.dp.cpp` and
  `ggml_cuda_host.dp.cpp` exist in the tree. They are not built (see [build.md](build.md)); how they were
  produced is not determined from the source.

## normalize.sh: the port's shape

`normalize.sh <dpct-out-dir> <upstream-commit>` (`:2-9` describe it) prepares a fresh migration for comparison
with the committed port:

1. deletes dpct's `.yaml` files and `dpct_report.*` (`:13`);
2. renames `X.cpp.dp.cpp` to `X.cpp`; `.cu` files keep their `X.dp.cpp` name (`:14`);
3. deletes every file that is byte-identical to the upstream original at `<commit>` (for `X.dp.cpp` the original is
   `X.cu`), because `sycl/CMakeLists.txt` falls back to the original (`:15-21`);
4. strips dpct's per-run serial numbers, `DPCT1065:123:` becoming `DPCT1065:` (`:23`), since they renumber on
   every run;
5. copies the repository's `sycl/tools/fixups.py` next to the output and runs it there, failing on error (`:24-25`).

It prints how many unchanged files it dropped and how many fixups applied. After this, two dpct runs over the same
source differ only in the kernel-name hashes dpct generates (INTEL.md).

## fixups.py: the scripted corrections

`fixups.py` edits files relative to its own parent directory (`root = parents[1]`, `:7`), so it always acts on the
`sycl/` tree it sits in. Every rule is a string or regex substitution through `edit(rel, fn)` (`:10-17`), which
writes only when the text changes, so the script is idempotent and can be re-run after every re-migration. "Upstream
files are never touched" (`:4`). The rules, in file order (rule numbers are the script's own):

| Rule | Lines | Files | What dpct got wrong, and the fix |
|---|---|---|---|
| 1 | 25-27 | `include/dpct/util.hpp` | Helper headers older than the 2026.1 compiler: `get_tangle_group` → `entangle`, `get_fixed_size_group` → `chunked_partition` |
| 2 | 31-32 | all `src/**/*.cpp` | `(dpct::queue_ptr)stream->memcpy(...)` cast bound to the call's result; re-parenthesised |
| 3 | 33-41 | all | `cudaStreamSynchronize(0)` → `dpct::get_in_order_queue().wait()`; other stream waits cast correctly; a null stream's `ext_oneapi_graph` → the default queue; `cudaGraphUpload` removed (a finalized command graph is resident) |
| (3) | 42-47 | all | `__fadd_rn(a, b)` → `(a + (b))`, applied three times for nesting; the second operand is parenthesised because `__fadd_rn(sum, c ? x : y)` had become `sum + c ? x : y` (found 2026-10-01 in the QSA indexer's pooled sum and the block scores); `__float_as_uint` → `sycl::bit_cast<uint32_t>` |
| 4 | 48-49 | all | `const volatile sycl::float4*` casts on kernel arguments → plain |
| 5 | 50-51 | all | `__isnanf` / `__isinff` → `sycl::isnan` / `sycl::isinf` |
| 6 | 52-53 | all | `__nanosleep(n);` in a doorbell spin → an empty spin |
| 7 | 54-55 | all | `a + b ? c[i] : 0.0f` → `a + (b ? c[i] : 0.0f)` |
| 2b | 57-68 | all | Every `(dpct::queue_ptr) x`, `((sycl::queue*)((dpct::queue_ptr)x))` and `static_cast<dpct::queue_ptr>(x)` → `strata::q_of(x)`, adding `#include "strata/sycl_queue.hpp"`; a null CUDA stream is the default queue, not a null pointer |
| 7b | 70-72 | `s_gemv.dp.cpp` | `__ldg((const float*) p)` had become `*p`, reading one byte of a float scale; the cast is restored (two sites) |
| 7c | 74-98, 128-129 | `elementwise.dp.cpp`, `verify_kernels.dp.cpp` | Doorbell loads and stores of host-mapped flags go through `strata::sys_load` / `sys_store` (system-scope atomics) and every wait is bounded by `strata::kSpinMax`; covers both upstream spellings (`while (*flag ...)` with and without `strata_spin_pause()`) |
| (7c) | 99-101 | `include/strata/kernels/dp4a.hpp` | The spin pause's `__nanosleep(100)` removed; the bounded spin is the back-off |
| — | 102-105 | `fused_gr.dp.cpp` | `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlock)` left untranslated → `dpct::get_device(dev).get_local_mem_size()` |
| — | 106-113 | `prefill/kernels.dp.cpp` | The key-head DeltaNet kernel (upstream PR #413) uses its plain-copy path (`#if 1`, no `cp.async`); `float4` → `sycl::float4` |
| — | 114-121 | `sampler.dp.cpp`, `qsa_select.dp.cpp` | sm_90 thread-block-cluster kernels (0.1.38) have no SYCL counterpart: the HIP branch's "not available" is taken with `#if 1` |
| — | 122-127 | `fused_gr.dp.cpp` | dpct could not deduce two templated kernel names (`dpct_placeholder /*Fix the type mannually*/`); filled in |
| 7d | 131-157 | all | `cudaMemcpy` / `cudaMemset` are synchronous but dpct's `get_in_order_queue().memcpy/memset(...)` was not waited on (DPCT1114). With the streaming expert source's ring buffers this filled the expert cache from reused buffers ("non-deterministic experts"). Every default-queue copy or fill gets `.wait()` |
| 7e | 159-167 | all | `dpct::dp4a(` → `strata::dp4a(` with `#include "strata/sycl_math.hpp"` |
| 7f | 169-176 | `native_mmvq.dp.cpp`, `s2_gemv_fast.dp.cpp` | Static `dpct::global_memory` / `constant_memory` objects were destroyed after the default queue at exit and crashed; they are heap-allocated and never freed |
| 8 | 178-180 | `iq_kernels.dp.cpp` | `GGML_COMMON_DECL_CUDA` / `IMPL_CUDA` → `_SYCL`, ggml-common.h's SYCL mode (`sycl::half`) |
| 8b | 182-188 | `iq_kernels.dp.cpp` | ggml tables are plain arrays in that mode: `.get_ptr()` and `.init(...)` removed; table parameters made `const` |
| 8c | 189-191 | `native_qsa_score.dp.cpp` | One nested `__fadd_rn` the generic rule misses |
| 8d | 193-205 | `iq_kernels.dp.cpp` | dpct threaded every ggml table through kernel parameters with one name per template regardless of the table a quant type needs; parameters, arguments and capture copies are removed so device code reads the `static const` globals (as ggml-sycl does); `get_int_from_table_16` keeps an explicit `kvalues_iq4nl` |
| 9 | 207-210 | `program/generate.cpp` | A ternary on the stream argument of `cudaMemcpyAsync` was migrated around the wrong operand |
| 10 | 212-215 | `verify_kernels.dp.cpp` | `%globaltimer` inline PTX → `t = 0`: the verify-window stage profiler reads zeros on SYCL |
| 11 | 217-236 | `qsa_prompt_attn.dp.cpp`, `qsa_select.dp.cpp`, `native_qsa_score.dp.cpp` | Inline PTX (`mma.sync`, `ldmatrix`, `cp.async`) cannot be migrated. dpct.hpp defines `DPCT_COMPATIBILITY_TEMP` as 900, which selected the sm_80 path; the PTX branches become `#if 0` / `#elif 0`, the scalar ones `#if 1`, `__trap()` a comment, and the launchers without an in-kernel fallback return `false` so callers use the older kernels (as on a pre-Ampere card). Includes the Turing m16n8k8 branch of 0.1.31 |
| 12 | 238-253 | `native_mmvq.dp.cpp` | IQ4_XS: the `__constant__` table had become a kernel argument and the trait a 3-argument `load()`; a `constexpr kIq4nlTable` restores 2-argument overloads |
| 13 | 255-256 | `qsa_prompt_attn_parity.cpp` | `sycl::free` of a const pointer: cast to `void*` |
| 14 | 258-276 | `core/verify.cpp` | Graph-node kernel-name introspection has no SYCL API (node named `"kernel"`); the `cudaLaunchHostFunc` lambda is rewritten as a `host_task` capturing a pointer |
| 15 | 277-280 | `core/remote_experts.cpp` | `cudaInitDevice(..., cudaDeviceScheduleSpin \| cudaDeviceMapHost)` has no equivalent; dropped |
| 16 | 281-283 | `include/strata/core/graph.hpp` | The graph's done timestamp is written from a `const` method; made `mutable` |
| — | 286-288 | all | (0.1.29) `((cudaStream_t)x)->` → `strata::q_of(x)->` |
| — | 289-290 | `core/native_head.cpp` | `sycl::free` of a const pointer |
| — | 292-299 | `src/kernels/*_parity.cpp` | `cudaMemcpy` on the legacy default stream waits for every blocking stream on the device; the default SYCL queue does not wait for other queues. In the parity tests every default-queue copy first drains all queues (`queues_wait_and_throw()`) |
| — | 301-305 | `program/generate.cpp` | (0.1.39, #423) dpct dropped the `cudaMemGetInfo` of `stage_room` (the expert cache of later `--layer-split` stages), so every later GPU saw 0 bytes free; `get_memory_info(fb, tb)` is re-inserted |

Gotcha: the `print("files changed:", changed)` at `:284` runs before the rules from `:286` on, so its count leaves
those out; `normalize.sh` counts `fixed` lines in `fixups.log` instead.

## merge_upstream.py: re-migration after an upstream release

`merge_upstream.py BASE_OUT NEW_OUT OLD_REV NEW_REV` (its header still calls it `merge_sel.py`) is run from the
repository root. It takes the files upstream changed between the two revisions (`git diff --name-only OLD NEW --
src include`, `:11`) and, for each one, maps `X.cu` to `sycl/X.dp.cpp` and decides (`:13-37`):

| Case | Output | Action |
|---|---|---|
| In `HAND` (`src/core/mtp.cpp`, `src/core/verify.cpp`, `src/ngram/ple_reader.cpp`, `src/kernels/native_expert_parity.cpp`; `:12`) | `HAND MERGED` / `HAND CONFLICT n` | `git merge-file` of the port's copy with upstream's own old and new file (no dpct in between) |
| No file in `NEW_OUT` | `NOT-MIGRATED (original used)` or `PORT-HAS-NO-DPCT?` | nothing |
| No copy in `sycl/` yet | `NEWCOPY` | the new migration is copied in |
| No file in `BASE_OUT` | `NO-BASE (new to dpct output)` | nothing |
| Base and new equal after hash canonicalisation | `NOCHANGE-after-dpct` | nothing |
| Otherwise | `MERGED` / `CONFLICT n` | `git merge-file sycl/F base new` |

dpct names kernels `name_<6 hex digits>`, and the hash changes between runs. The script canonicalises those
suffixes (`pat`, `:4`): the new migration takes the base's hashes, and both then take the port's hashes, so a
re-run of dpct does not produce spurious conflicts and the port keeps its own kernel names. It writes temporary
files to `/tmp/_b` and `/tmp/_n`.

## The update procedure

From INTEL.md "Keeping up with upstream" (the port lived on a branch called `b70` of the author's fork):

1. Merge upstream into the port's branch. No shared file should conflict, since all Intel code is in `sycl/`. Check
   that `sycl/setup_intel.py --check` and `sycl/serve/server_intel.py --engine mock` still run against the new
   `setup.py` and `serve/server.py`.
2. Migrate the old upstream tree (a `git archive` of the pre-merge commit) into `sycl-base` and the merged tree into
   `sycl-new` with `migrate.sh` (about three minutes each in the image).
3. `normalize.sh` both directories with their commits.
4. 3-way merge every file present in both (`merge_upstream.py`); files dpct never produced take upstream's diff by
   hand; new parity tests are copied in and listed in `sycl/CMakeLists.txt`.
5. A fixup whose pattern upstream changed shows up as a compile error: extend the rule, re-run it, rebuild.

INTEL.md logs the merges done this way: 0.1.25-0.1.27 (2026-09-30), 0.1.31 (132 upstream commits, 38 conflicts;
a "symbol audit" of the port's own identifiers became part of the procedure), 0.1.32, 0.1.33, 0.1.35 plus seven
then-open upstream PRs (2026-10-02; `merge_upstream.py` was written for that), and 0.1.38 (2026-10-03).
INTEL_ARC.md adds that the 0.1.39 port re-migrates 0.1.38's port onto the 0.1.39 sources and that nobody had run
it on an Arc at release time.

## build.sh

`build.sh [target...]` (`REPO` default `/work/Strata_B70`, `BUILD_DIR` default `$REPO/build-sycl`) configures once
(only if `build.ninja` is missing) with `cmake -S $REPO/sycl -G Ninja -DCMAKE_C_COMPILER=icx
-DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_AOT="$AOT"`, then builds with `-j ${JOBS:-12}` and `-k 0` (keep going),
tees the log to `$BUILD_DIR/build.log` and prints the failures, the error count and the failed objects. INTEL.md
uses `AOT=bmg-g31 BUILD_DIR=.../build-sycl-aot` for the build that is served. CMake details are in
[build.md](build.md).

## Configuration

| Variable | Used by | Meaning |
|---|---|---|
| `CUDA_HEADERS` | `migrate.sh` | CUDA include directory (default `/cuda-headers/include`) |
| `DPCT_EXTRA` | `migrate.sh` | extra dpct flags |
| `PYTHON` | `get-cuda-headers.sh` | interpreter with pip |
| `REPO`, `BUILD_DIR`, `AOT`, `JOBS` | `build.sh` | checkout, build directory, `STRATA_SYCL_AOT` device, parallel jobs |

## Tests

None of these scripts has a test. The migrated tree is checked by the parity tests and an end-to-end comparison of
greedy output tokens before and after each merge (INTEL.md); see [build.md](build.md#tests).

## In UnbiasedStrata

Removed with the rest of `sycl/` in commit `8894a81`.

## See also

- [layout.md](layout.md): which files the pipeline produced and which were written by hand.
- [differences.md](differences.md): the hand-written changes after the fixups.
- [build.md](build.md): what CMake does with the tree.
