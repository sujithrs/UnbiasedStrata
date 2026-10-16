[Upstream docs](../../README.md) › [Kernels](../README.md) › [CPU](README.md) › Tests

# CPU kernel and pool tests

## Purpose

The test programs that live beside the CPU kernels in `src/kernels/cpu/`. None needs a GPU. The GPU-side and
cross-device parity tests (including `native_expert_parity`, which also checks the CPU native path) are in
[../parity/README.md](../parity/README.md).

## Files

| File | Lines | Role |
|---|---|---|
| [src/kernels/cpu/expert_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/expert_parity.cpp) | 184 | VNNI Q2_0 expert vs the scalar oracle on a real pack blob |
| [src/kernels/cpu/expert_multi_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/expert_multi_test.cpp) | 87 | Multi-token Q2_0 expert bitwise vs single-token; optional bench |
| [src/kernels/cpu/pool_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_test.cpp) | 224 | Pool correctness (serial equality, exactly-once, repeated batches) and a layer timing |
| [src/kernels/cpu/pool_stress.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_stress.cpp) | 99 | Issue #29 stress: tiny batches with sleeping workers |
| [src/kernels/cpu/pool_affinity_linux_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_affinity_linux_test.cpp) | 290 | Linux affinity save / pin / restore |
| [src/kernels/cpu/pool_affinity_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_affinity_test.cpp) | 148 | Windows CPU Set host pinning and worker hard pins |
| [src/kernels/cpu/router_dot_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/router_dot_parity.cpp) | 70 | BF16 router dot (AVX2, AVX1) vs double |

## Test table

All are registered in the top-level `CMakeLists.txt`. `STRATA_BUILD_TESTS` defaults to ON only when
`tests/CMakeLists.txt` and `bench/micro/` exist (`CMakeLists.txt:87-92`); the published source has neither, so pass
`-DSTRATA_BUILD_TESTS=ON`.

| CTest name (target) | Registered at | Condition | What it checks | Reference | Inputs |
|---|---|---|---|---|---|
| `expert_parity` (`expert_parity --selftest`, working dir = source root) | `CMakeLists.txt:1213-1218` | `STRATA_BUILD_TESTS` | (1) `s2_expert_vnni` vs `s2_expert_scalar(quant_acts=true)`: rel L1 <= 3e-3 (P2.S3); (2) vs exact FP32 activations: reported, not asserted; the activation contract must be observable; the blob must decode to finite non-zero scales and another layer's blob must differ (a wrong stride) | `s2_expert_scalar`, a transcription of ggml's formula | **Required**: `pack/full/experts.bin` (a canonical pack; `--file`, `--layer`, `--expert`); exit 2 if unreadable. On a CPU without AVX-512 VNNI / VBMI it prints SKIPPED and exits 0 |
| `expert_multi_test` | `:1209-1211` | `STRATA_BUILD_TESTS` | `s2_expert_vnni_multi` bitwise equal to `s2_expert_vnni_q` per token, n = 1..8; `--bench [E]` times n = 1..5 over E synthetic experts (default 256 = 354 MB, beyond L3) on one thread, the speculation model's `extra_use_cost` | The single-token kernel | Synthetic blobs |
| `pool_test` (`pool_test --selftest`, working dir = source root) | `:1220-1222` | `STRATA_BUILD_TESTS` | Every job bitwise equal to the serial run; every job exactly once (sentinel outputs); 200 consecutive batches; batch sizes either side of the worker count; then one layer's timing of 10 experts | Serial `s2_expert_vnni_q` | **Required**: `pack/full/experts.bin` (`--file`, `--layer`); skips without VNNI |
| `pool_stress` (`pool_stress 20`) | `:1223-1226` | `STRATA_BUILD_TESTS` | 20 s of random 0-4-job batches through `run` and `run_split`, with 21-30 ms gaps so the workers fall asleep; every output written; a watchdog thread exits 3 (`std::_Exit(3)`) if a batch does not return within 10 s | Sentinel check | Synthetic; skips without VNNI |
| `pool_affinity_test` | `:1192-1200` | `STRATA_BUILD_TESTS` and Windows (`pool_affinity_test.cpp`) or Linux (`pool_affinity_linux_test.cpp`) | Linux: full-mask, sparse / nested and high-CPU-id (>= 64) round trips of `pin_current_thread` / `restore_thread_affinity`, topology respects the current mask, worker pin, negative CPU is a no-op, a failed pin keeps the mask; `--require-high-cpu` makes the high-id case mandatory. Windows: CPU Sets cover all active groups, host selection round trips, worker hard pin preserves the caller's hard affinity, inherited defaults | The OS's own affinity query | None |
| `router_dot_parity`, `router_dot_parity_avx` (`STRATA_FORCE_ISA=avx` via `cmake -E env`) | `:1030-1036` | `STRATA_NATIVE_EXPERTS` builds the target; ctests need `STRATA_BUILD_TESTS` | `bf16_rows_dot_multi` (AVX2) and `bf16_rows_dot_multi_avx1` (AVX) on random BF16 rows of the router's shape 512 x 2560, 1..8 tokens: rel L1 < 1e-5; prints one layer's time | Double-precision dot | Synthetic |

Related tests outside this directory: `tests/core/expert_layout_test.cpp` (the `native_experts.txt` reader over
`tests/data/native_experts/*.txt`, registered at `CMakeLists.txt:438-444` inside the GPU block, needs
`STRATA_BUILD_TESTS` and `STRATA_NATIVE_EXPERTS`), and the `native_expert_parity_*` ctests
([../parity/README.md](../parity/README.md)).

## How to run

```
cmake -B build -DSTRATA_BUILD_TESTS=ON            # no GPU backend needed for these
cmake --build build --target expert_multi_test pool_stress pool_affinity_test router_dot_parity
ctest --test-dir build -R 'expert_multi_test|pool_stress|pool_affinity_test|router_dot_parity'
build/expert_multi_test --bench 256                # timing, not a ctest
```

`expert_parity` and `pool_test` also need a canonical pack at `pack/full/` in the source tree. The default
`STRATA_NATIVE_EXPERTS=ON` fetches llama.cpp at configure time (or uses `-DSTRATA_GGML_DIR`).

## Gotchas

- A missing pack is deliberately a failure (exit 2), not a skip: "a missing file would otherwise be reported as a
  pass by a test that skipped everything" (`CMakeLists.txt:1215-1216`).
- A CPU without AVX-512 VNNI / VBMI makes `expert_parity`, `pool_test` and `pool_stress` print SKIPPED and exit 0,
  which CTest reports as passed. `expert_multi_test` has no such check, so it fails on such a CPU: upstream's AMD
  report lists it among the two expected ctest failures on a Ryzen 9 3900X ("needs an AVX-512 CPU",
  [docs/AMD_HIP.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/AMD_HIP.md)).
- The Linux affinity test needs `_GNU_SOURCE` and compiles to a SKIP stub on non-Linux.
- On macOS (neither Windows nor Linux) `pool_affinity_test` is not registered.

## In UnbiasedStrata

Kept unchanged (all seven files). The fork's page: [../../../codebase/kernels/cpu/tests.md](../../../codebase/kernels/cpu/tests.md).

## See also

- [q2-expert.md](q2-expert.md), [pool.md](pool.md), [native-experts.md](native-experts.md)
- [../parity/README.md](../parity/README.md), [../parity/fixtures.md](../parity/fixtures.md)
- [../../build/targets.md](../../build/targets.md)
