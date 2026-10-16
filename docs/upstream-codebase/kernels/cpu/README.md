[Upstream docs](../../README.md) › [Kernels](../README.md) › CPU

# CPU kernels: routed experts, the worker pool and the PLE table

The CPU half of the engine computes every routed MoE expert that the GPU does not hold in VRAM, from expert bytes in
host RAM, at the same time as the GPU computes the experts it holds. It is the static library `strata_kernels_cpu`
(built in every configuration, with or without a GPU backend). It contains the Q2_0 expert kernel for canonical
Strata packs, the ggml-cpu based native-expert path plus hand-written multi-token AVX-512 / AVX2 kernels for native
(GGUF-form) packs, the per-layer expert layout table, the CPU feature probes, and `ExpertPool`, the pinned
worker-thread pool that runs all of them. Its callers are `src/core/expert_source.cpp` (the per-layer dispatch, driven
by the layer loop in `src/core/session.cpp` and by the speculative verify window in `src/core/verify.cpp`) and
`src/program/generate.cpp` (construction, CPU checks, layout load). It depends only on ggml-cpu / ggml-base (fetched
from llama.cpp, for native experts), the C++ thread library and two hooks in `strata/core/progress.hpp`. The n-gram
hash and PLE table reader (`src/kernels/ngram.cpp`) is documented here too, although it is host code built into the
GPU library `strata_kernels`.

All models upstream supports (Qwen3.8-Flash-Next and its Coder, Swift 1.5 and Unsloth variants) have the same expert
geometry: hidden width 2560, expert width 640, 512 experts (fewer in pruned packs), top-10 routing. Every CPU kernel
here assumes it.

## Contents

| Page | Covers |
|---|---|
| [q2-expert.md](q2-expert.md) | The Q2_0 expert kernel (`expert.cpp`, `q2_avx2.cpp`): VNNI row dot, activation quantizer, row-range and multi-token variants, the "oracle" contract, the scalar reference |
| [native-experts.md](native-experts.md) | Native experts through ggml-cpu (`native_expert.cpp`) and the multi-token kernels `iq_avx512`, `iq_avx2`, `kq_avx2`, `kq_avx1` (the last also the router-prefetch dot) |
| [expert-layout.md](expert-layout.md) | `native_experts.txt` and `ExpertLayout`; the ISA probes `cpu_avx512_ok` / `cpu_avx2_ok` / ... and `STRATA_FORCE_*` |
| [pool.md](pool.md) | `ExpertPool`: topology and pinning (Windows processor groups and CPU Sets, Linux masks, hybrid P/E-core CPUs), the epoch-tagged claim protocol, run modes, overlap with the GPU |
| [tests.md](tests.md) | `expert_parity`, `expert_multi_test`, `pool_test`, `pool_stress`, `pool_affinity_test`, `router_dot_parity` |
| [ngram.md](ngram.md) | The PLE n-gram hash and `PleTable` (`ngram.cpp`), `--ple-io direct\|mmap\|ram` |

## Source map

| File | Lines | Role |
|---|---|---|
| [include/strata/kernels/cpu/expert.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/expert.hpp) | 179 | Q2_0 expert geometry, `ActQ`, scratch, entry points, design notes |
| [src/kernels/cpu/expert.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/expert.cpp) | 683 | AVX-512 VNNI/VBMI Q2_0 expert, quantizers, oracle contract, scalar reference, CPUID probe |
| [src/kernels/cpu/q2_avx2.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/q2_avx2.cpp) | 130 | Q2_0 GGUF-layout rows and quantizer for AVX2-only CPUs |
| [include/strata/kernels/cpu/native_expert.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/native_expert.hpp) | 49 | `NativeFmt` and the native row API |
| [src/kernels/cpu/native_expert.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/native_expert.cpp) | 166 | ggml-cpu traits, `native_fmt`, kernel dispatch with env overrides |
| [include/strata/kernels/cpu/iq_avx512.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/iq_avx512.hpp) | 18 | AVX-512 i-quant API |
| [src/kernels/cpu/iq_avx512.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/iq_avx512.cpp) | 273 | Multi-token IQ2_XXS / IQ2_XS / IQ3_XXS / IQ3_S / IQ2_S rows, 512-bit |
| [include/strata/kernels/cpu/iq_avx2.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/iq_avx2.hpp) | 26 | AVX2 i-quant API |
| [src/kernels/cpu/iq_avx2.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/iq_avx2.cpp) | 493 | Multi-token i-quant rows (+ IQ4_XS) and IQ4_NL down rows, 256-bit |
| [include/strata/kernels/cpu/kq_avx2.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/kq_avx2.hpp) | 25 | K-quant and BF16 router API |
| [src/kernels/cpu/kq_avx2.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/kq_avx2.cpp) | 265 | Q4_K / Q5_1 / Q8_0 multi-token rows, bit-exact vs ggml's AVX2 `vec_dot`; BF16 router dot |
| [include/strata/kernels/cpu/kq_avx1.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/kq_avx1.hpp) | 18 | AVX1 router dot API |
| [src/kernels/cpu/kq_avx1.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/kq_avx1.cpp) | 71 | BF16 router dot for AVX-only CPUs |
| [include/strata/kernels/cpu/expert_layout.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/expert_layout.hpp) | 82 | `ExpertLayout`, ISA probes, dispatchers |
| [src/kernels/cpu/expert_layout.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/expert_layout.cpp) | 341 | Probes, dispatchers, `native_experts.txt` parser, no-ggml stubs |
| [include/strata/kernels/cpu/pool.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/pool.hpp) | 286 | `ExpertPool` and its job / topology / affinity types; the protocol notes |
| [src/kernels/cpu/pool.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool.cpp) | 779 | Topology, pinning, worker loop, claim protocol, run modes |
| [src/kernels/cpu/pool_affinity_linux.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_affinity_linux.hpp) | 69 | Linux affinity helpers (`CPU_ALLOC` masks of any size) |
| [src/kernels/cpu/pool_affinity_win.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_affinity_win.hpp) | 64 | Windows processor-group affinity and CPU Set helpers |
| [src/kernels/cpu/expert_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/expert_parity.cpp) | 184 | Test: VNNI kernel vs scalar oracle on a real pack blob |
| [src/kernels/cpu/expert_multi_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/expert_multi_test.cpp) | 87 | Test: multi-token bitwise vs single-token; bench |
| [src/kernels/cpu/pool_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_test.cpp) | 224 | Test: pool correctness and timing |
| [src/kernels/cpu/pool_stress.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_stress.cpp) | 99 | Test: issue #29 sleep/wake stress |
| [src/kernels/cpu/pool_affinity_linux_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_affinity_linux_test.cpp) | 290 | Test: Linux affinity round trips |
| [src/kernels/cpu/pool_affinity_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_affinity_test.cpp) | 148 | Test: Windows CPU Set round trips |
| [src/kernels/cpu/router_dot_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/router_dot_parity.cpp) | 70 | Test: router dot (AVX2, AVX1) vs double |
| [include/strata/kernels/ngram.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/ngram.hpp) | 216 | PLE hash, dequantizers, `PleTable`, I/O options |
| [src/kernels/ngram.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/ngram.cpp) | 506 | PLE hash and table reader (built into `strata_kernels`) |

## Key concepts

### Two expert forms

A **canonical pack** stores each expert in Strata's own planar Q2_0 form; a **native pack** keeps each expert's raw
GGUF bytes (see the [glossary](../../guides/glossary.md) for both terms).

| | Canonical Q2_0 pack | Native pack |
|---|---|---|
| Written by | `tools/strata_pack.py` | `tools/iq_pack.py` (writes `native_experts.txt`) |
| Blob | Fixed 1,382,400 B planar Strata form | Raw GGUF slices `[gate \| up \| down]`, size per layer |
| Typical models | The Q2_0 model | IQ2_XS / IQ3_XXS / IQ3_S packs, Coder IQ1_M, Swift 1.5, Unsloth UD-Q4_K_XL, and Q2_0 as a native pack |
| CPU arithmetic | `s2_expert_*` (AVX-512 VNNI + VBMI required) | ggml-cpu `vec_dot`, or Strata multi-token kernels for i-quants, Q2_0, IQ4_NL and (opt-in) K-quants; AVX2 floor |
| Pool entry | `run`, `run_split`, `run_split_multi` | `run_split_multi_native` (verify windows only) |

A native pack runs its experts in speculative verify windows only (`src/program/generate.cpp:1994`, `:3823`), so
its CPU experts always go through the multi-token path.

### Per-layer overlap with the GPU

```
          GPU stream                              host thread (pinned) + pool workers
  pre[l]: mixer, router, top-k
     |  doorbell ring (x_f, ids, weights in mapped pinned memory)
     v                                         host sees the ring
  experts held in VRAM                <-----   enqueues the GPU's share, does not wait
     |                                         expert_pool_dispatch: quantize x_f once,
     |                                         one job per expert NOT in VRAM -> ExpertPool
     |                                         (all cores, the host included) -> y_miss
     |                                <-----   H2D copy of y_miss
  post[l]: moe_combine (router weights applied here), residual
  pre[l+1] ...
```

In a speculative verify window the same happens for up to 8 tokens: every distinct CPU expert becomes one
multi-token job whose weight rows are read once for all tokens routed to it. Details: [pool.md](pool.md).

### ISA tiers

| Probe | Requires | Unlocks |
|---|---|---|
| `cpu_features().usable()` / `cpu_avx512_ok()` | AVX512 F, BW, VL, VNNI, VBMI (+ OS ZMM state for the second) | Canonical packs; `iq_avx512`; AVX-512 Q2_0 rows |
| `cpu_avx2_ok()` | AVX, AVX2, FMA, F16C + OS YMM state | Native packs' Strata kernels (`iq_avx2`, `q2_avx2`, `kq_avx2`); the normal engine's floor |
| `cpu_avx1_ok()` | AVX | AVX1 router dot; ggml-cpu in a `STRATA_ISA_FLOOR=avx` build |
| `cpu_sse42_ok()` | SSE4.2 + POPCNT | ggml-cpu in a `STRATA_ISA_FLOOR=none` build |

`src/program/generate.cpp:1961-1985` refuses a CPU without AVX2 unless the engine is an older-CPU build, and
`:2054-2059` refuses a canonical pack on a CPU without the VNNI features (`cpu_require_expert_support`).

## Build

`CMakeLists.txt:927-1036`:

- `strata_kernels_cpu` = `expert.cpp pool.cpp expert_layout.cpp q2_avx2.cpp` (`:927-928`), plus, with
  `STRATA_NATIVE_EXPERTS=ON` (default, `:949`), `native_expert.cpp iq_avx512.cpp iq_avx2.cpp kq_avx2.cpp kq_avx1.cpp`
  (`:1006-1008`), linking `ggml-cpu ggml-base` and defining `STRATA_NATIVE_EXPERTS=1` publicly (`:1028-1029`).
- ISA flags are per source file, never per target (`:930-940`, `:1010-1022`): `expert.cpp` and `iq_avx512.cpp` get
  AVX-512 (F/BW/VL/DQ/VNNI/VBMI + F16C + FMA; MSVC `/arch:AVX512`), `q2_avx2.cpp`, `iq_avx2.cpp`, `kq_avx2.cpp` get
  `-mavx2 -mfma -mf16c` (`/arch:AVX2`), `kq_avx1.cpp` gets `-mavx` (`/arch:AVX`). `pool.cpp` and `expert_layout.cpp`
  (the probes) get none, so they run on any x86-64.
- ggml is built static from a llama.cpp checkout: `-DSTRATA_GGML_DIR=...`, otherwise CMake's `FetchContent` clones
  llama.cpp at the pinned commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` (`:990-1003`).
  `third_party/ggml/` holds only `ggml-common.h`, its licence and `VERSION.txt`.
- `STRATA_PORTABLE` builds ggml-cpu for an AVX2 baseline (the release engine, `:966-970`); `STRATA_ISA_FLOOR=avx|none`
  builds it for an older CPU and defines `STRATA_ISA_FLOOR_AVX` / `_NONE` (`:971-986`, `:1023-1027`).

See also [../../build/cmake-options.md](../../build/cmake-options.md).

## Configuration summary

CLI (`src/program/generate.cpp`): `--pool-workers`, `--pool-affinity`, `--no-host-worker`, `--no-split-rows`,
`--cpu-oracle-q8-0`, `--ple-io` and the other `--ple-*` flags. Environment: `STRATA_CPU_YMM`, `STRATA_IQ_MT_MIN`,
`STRATA_NO_IQ512`, `STRATA_NO_IQ256`, `STRATA_NO_IQ4NL`, `STRATA_KQ256`, `STRATA_IQ_PREFETCH`, `STRATA_IQ_GATHER`,
`STRATA_IQ256_GATHER`, `STRATA_FORCE_AVX2`, `STRATA_FORCE_ISA`, `STRATA_POOL_SPIN_US`, `STRATA_SSD_KEEPALIVE*`. Each
is described on the page that owns it.

## In UnbiasedStrata

| Kept unchanged | Changed | Removed |
|---|---|---|
| `expert.hpp/.cpp`, `q2_avx2.cpp`, `native_expert.hpp`, the `iq_*` / `kq_*` kernels, `pool.hpp`, both affinity headers, every CPU test, `ngram.hpp` | `expert_layout.hpp/.cpp` (commit e41dfc7, the Qwen3.6-35B-A3B port: `expert_layout_load` takes the model file's `n_embd` / `n_ff` and refuses a canonical pack at another geometry); `native_expert.cpp` (22d72e6: `native_fmt` refuses widths above 2560 x 640); `pool.cpp` (22d72e6: modes 5/6 use the layer's own widths); `ngram.cpp` (9a53e42: `--ple-io ram` copies the table into its own anonymous memory instead of `mlock`ing the file mapping) | Nothing in this directory |

The fork's own description is in [../../../codebase/kernels/cpu/README.md](../../../codebase/kernels/cpu/README.md).

## See also

- [../README.md](../README.md) (all kernels), [../parity/README.md](../parity/README.md) (GPU and cross-device tests),
  [../cuda/README.md](../cuda/README.md) (the GPU side of the expert split)
- [docs/upstream/DETAILS.md](../../../upstream/DETAILS.md) (upstream's own "How it works")
