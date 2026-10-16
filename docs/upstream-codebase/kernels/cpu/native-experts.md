[Upstream docs](../../README.md) › [Kernels](../README.md) › [CPU](README.md) › Native experts

# Native (GGUF-form) experts on the CPU

## Purpose

A **native pack** (`tools/iq_pack.py`) keeps each expert as its raw GGUF slices `[gate rows | up rows | down rows]` in
the ggml quant types of the model file (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q4_K, Q5_K, Q8_0,
Q2_0 for gate/up; Q2_0, IQ4_NL, Q5_1, Q8_0 and others for down). These values cannot be re-expressed in Strata's Q2_0
form, so the CPU computes them with **ggml-cpu's own arithmetic** (`ggml_get_type_traits_cpu`): the activation is
quantized with the weight type's `vec_dot_type` and each row is one `vec_dot`, which is what llama.cpp's CPU backend
computes for the same tensor (`include/strata/kernels/cpu/native_expert.hpp:1-7`).

On top of that baseline, hand-written **multi-token** kernels decode a block of weights once and apply it to every
token of a speculative verify window that was routed to the same expert (up to 8 tokens).

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/kernels/cpu/native_expert.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/native_expert.hpp) | 49 | `NativeFmt`, buffer limits, the row / quantizer entry points |
| [src/kernels/cpu/native_expert.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/native_expert.cpp) | 166 | ggml-cpu trait lookup, `native_fmt`, kernel dispatch with env overrides |
| [include/strata/kernels/cpu/iq_avx512.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/iq_avx512.hpp) | 18 | AVX-512 i-quant API |
| [src/kernels/cpu/iq_avx512.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/iq_avx512.cpp) | 273 | IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22) rows vs Q8_K, 512-bit |
| [include/strata/kernels/cpu/iq_avx2.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/iq_avx2.hpp) | 26 | AVX2 i-quant API |
| [src/kernels/cpu/iq_avx2.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/iq_avx2.cpp) | 493 | The same formats plus IQ4_XS (23) vs Q8_K, and IQ4_NL (20) down rows vs Q8_0, 256-bit |
| [include/strata/kernels/cpu/kq_avx2.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/kq_avx2.hpp) | 25 | K-quant / BF16 router API |
| [src/kernels/cpu/kq_avx2.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/kq_avx2.cpp) | 265 | Q4_K (12) gate/up vs Q8_K, Q5_1 (7) / Q8_0 (8) down; bit-exact vs ggml's AVX2 `vec_dot`; the BF16 router dot |
| [include/strata/kernels/cpu/kq_avx1.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/kq_avx1.hpp) | 18 | AVX1 router dot API |
| [src/kernels/cpu/kq_avx1.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/kq_avx1.cpp) | 71 | `bf16_rows_dot_multi_avx1` for AVX-only CPUs (from the Strata_Dirigo fork) |

These files are added to `strata_kernels_cpu` only with `STRATA_NATIVE_EXPERTS=ON` (the default), which also builds
ggml (`CMakeLists.txt:957-1029`). Per-source flags: `iq_avx512.cpp` AVX-512 (+ VNNI / VBMI), `iq_avx2.cpp` and
`kq_avx2.cpp` `-mavx2 -mfma -mf16c`, `kq_avx1.cpp` `-mavx` (`kq_avx1.cpp:16-18` has an `#error` if it is built
without AVX). Without `STRATA_NATIVE_EXPERTS`, stubs in `src/kernels/cpu/expert_layout.cpp:223-231` refuse or abort,
and `expert_layout_load` refuses a native pack (`:243-245`).

## Main types and functions

| Symbol | Where | What it does |
|---|---|---|
| `kNativeActBytes = 4096`, `kNativeHBytes = 1024` | `native_expert.hpp:17-19` | Largest quantized gate/up activation (2,560 values as Q8_K = 2,920 B) and down activation the pool's buffers hold |
| `NativeFmt` | `native_expert.hpp:22` | Per-layer types (`gu_type`, `d_type`), their `vec_dot_type`s (`gu_act`, `d_act`), `n_embd`, `n_ff`, row bytes, `up_off`, `down_off`, blob `bytes`, activation sizes |
| `native_fmt(gu, d, n_embd, n_ff, f, err)` | `native_expert.cpp:32` | Fills `NativeFmt` from ggml-cpu traits; refuses a type without `vec_dot` / `from_float`, a geometry that is not whole blocks, or activations bigger than the pool's buffers |
| `native_quant_act` / `native_quant_h` | `native_expert.cpp:71`, `:75` | ggml `from_float` into the layer's activation format |
| `native_gu_rows(f, blob, act, nt, ff, r0, r1)` | `native_expert.cpp:79` | `ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t])`; picks a kernel (below) |
| `native_down_rows(f, blob, hq, nt, out, r0, r1)` | `native_expert.cpp:136` | `out[t][r] = down_r . hq[t]`; picks a kernel (below) |
| `iq512_supported` / `iq512_gu_rows` / `iq512_rows` | `iq_avx512.cpp:245`, `:249`, `:261` | 512-bit i-quant rows: 64 values decoded once (grid lookups, a 64-bit sign mask, scales), each token applies them with a masked subtract + `maddubs` + `madd` (`iq_avx512.cpp:1-7`) |
| `iq256_supported` / `iq256_gu_rows` / `iq256_rows` | `iq_avx2.cpp:410`, `:414`, `:427` | 256-bit version over 32-value halves; IQ2_XS has a dedicated sign decode (`row_dot_iq2xs`, `iq_avx2.cpp:266`) |
| `iq4nl256_down_rows` | `iq_avx2.cpp:477` | IQ4_NL down rows vs Q8_0: nibble decode and `\|w\|` once per block, then sign + `maddubs` + `madd` + FMA per token |
| `kq256_supported` / `kq256_gu_rows` / `kq256_rows` | `kq_avx2.cpp:202`, `:239`, `:253` | ggml's `__AVX2__` branches of `ggml_vec_dot_q4_K_q8_K`, `_q5_1_q8_1`, `_q8_0_q8_0` with the token loop moved inside the block loop |
| `bf16_rows_dot_multi` / `bf16_rows_dot` | `kq_avx2.cpp:204`, `:219` | BF16 router rows x fp32 activations for up to 8 tokens; the routing-aware prefetch's router (an estimate only) |
| `bf16_rows_dot_multi_avx1` | `kq_avx1.cpp:45` | Same arithmetic with separate mul + add and 128-bit widening for AVX-only CPUs (2.6 ms vs 362 ms per layer for the scalar `std::fma` fallback on a Xeon E5-2665, `kq_avx1.cpp:4-6`) |

## Kernel selection

```
native_gu_rows (native_expert.cpp:79-134)
  STRATA_KQ256=1 and AVX2 and gu_type == Q4_K (12) and nt >= 2   -> kq256_gu_rows (bit-exact vs ggml)
  nt >= STRATA_IQ_MT_MIN (default 2) and the type has a kernel:
     AVX-512 CPU, not STRATA_NO_IQ512, iq512_supported         -> iq512_gu_rows
     AVX2 CPU, not STRATA_NO_IQ256, iq256_supported            -> iq256_gu_rows
       (IQ4_XS: only on a CPU WITHOUT AVX-512, :108-112)
  otherwise                                                    -> ggml vec_dot, token by token

native_down_rows (native_expert.cpp:136-164)
  AVX2 and STRATA_KQ256=1 and nt >= 2 and d_type in {Q5_1, Q8_0} -> kq256_rows
  AVX2 and nt >= STRATA_IQ_MT_MIN and d_type == IQ4_NL (20), not STRATA_NO_IQ4NL -> iq4nl256_down_rows
  otherwise                                                    -> ggml vec_dot, token by token
```

Q2_0 layers of a native pack bypass these two functions: the pool calls the Q2_0 kernels directly when
`q2_native_kernels(type)` is true (type 42 on an AVX2 CPU; `include/strata/kernels/cpu/expert_layout.hpp:65`). See
[q2-expert.md](q2-expert.md).

The multi-token i-quant kernels were measured at "2.0-2.4x ggml-cpu at three tokens, no faster at one"
(`native_expert.cpp:81-82`, measured by `native_expert_parity`). IQ1_M (type 29) has no hand-written kernel;
`iq_avx512.cpp:9-10` says "no shipped model has IQ1_M expert rows", but the IQ2_XS and Swift IQ2_XS tables in
`tests/data/native_experts/` list type 29 gate/up on three layers each, which therefore run on ggml-cpu.
`iq512_rows` and `iq256_rows` (plain row dots) have no caller in the engine; `native_expert_parity` calls them.

## Invariants

- The i-quant multi-token kernels use ggml's integer arithmetic but a **different float addition order**, so their
  results differ from ggml's in the last bits; a token's rows therefore round differently alone (`nt == 1`, ggml)
  than in a group (#152). `STRATA_IQ_MT_MIN=1` (opt-in since 0.1.30) routes every group through the multi-token
  kernels, so the output does not depend on how many drafts a window held, "at a measured -1..-3% decode on IQ3_S"
  (`native_expert.cpp:95-99`).
- The K-quant kernels are bit-exact per token against ggml-cpu's AVX2 `vec_dot` for any group size
  (`kq_avx2.cpp:1-13`). The comment assumes ggml-cpu was built for AVX2 (`STRATA_PORTABLE`: `GGML_AVX2=ON`,
  `GGML_AVX512=OFF`); a `GGML_NATIVE` build on an AVX-512 machine selects other `vec_dot` branches.
- Each kernel is reached only for the types it implements: falling through an empty `switch` would leave `ff`
  unwritten (`native_expert.cpp:108-110`).
- AVX2 kernels are gated on `cpu_avx2_ok()` as well as the env switches: reaching `iq256_gu_rows` on a pre-Haswell
  CPU was an illegal instruction (`native_expert.cpp:87-94`, `:143-145`).
- `iq_avx2.cpp` builds its sign table at compile time (`constexpr EvenSigns`, `iq_avx2.cpp:56-82`) so the AVX2
  translation unit has no static constructor that could execute AVX2 / BMI2 before `main()` on an older CPU (#391: a
  Sandy Bridge Xeon exited 0xC000001D before printing anything).

## Callers and dependencies

- [pool.md](pool.md): `ExpertPool::run_split_multi_native` (modes 5/6 in `drain`, `src/kernels/cpu/pool.cpp:561-597`)
  calls `native_gu_rows`, `native_quant_h`, `native_down_rows`.
- [expert-layout.md](expert-layout.md): `expert_layout_load` calls `native_fmt` for every layer line
  (`expert_layout.cpp:284`).
- `src/core/expert_source.cpp:2146-2147` quantizes each verify-window token with `native_quant_act`.
- `src/core/expert_source.cpp:1224-1240` runs the routing-aware prefetch's router with `bf16_rows_dot_multi` (AVX2),
  `bf16_rows_dot_multi_avx1` (AVX) or a scalar mul + add loop.
- ggml-cpu (`ggml.h`, `ggml-cpu.h`, `ggml-common.h`); `ggml_cpu_init()` is called once (`native_expert.cpp:23-26`).

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `STRATA_IQ_MT_MIN` | 2 | Smallest group that uses the multi-token i-quant / IQ4_NL kernels |
| `STRATA_NO_IQ512` | unset | An AVX-512 CPU uses the AVX2 i-quant kernel instead |
| `STRATA_NO_IQ256` | unset | Drop the AVX2 i-quant kernel (with `STRATA_NO_IQ512` too: ggml-cpu only) |
| `STRATA_NO_IQ4NL` | unset | IQ4_NL down rows on ggml-cpu |
| `STRATA_KQ256` | 0 | `=1`: K-quant multi-token kernels (opt-in; "measured no faster in the engine", 1.01-1.13x in `native_expert_parity`, `native_expert.cpp:100-102`) |
| `STRATA_IQ_PREFETCH` | 2048 | Software prefetch distance in bytes for both i-quant kernels (0 = off; `iq_avx512.cpp:34-39`, `iq_avx2.cpp:92-101`) |
| `STRATA_IQ_GATHER` | unset | AVX-512: IQ3 grids via a gather ("3-5% SLOWER" on a Ryzen 5 7600, `iq_avx512.cpp:31-33`) |
| `STRATA_IQ256_GATHER` | 0 | AVX2: IQ3 grids via a gather (~1.3x on an i7-12850HX for the two 32-bit-grid formats, `iq_avx2.cpp:109-112`) |
| `STRATA_FORCE_AVX2=1`, `STRATA_FORCE_ISA` | unset | Cap the CPU probes ([expert-layout.md](expert-layout.md)) |

CMake: `STRATA_NATIVE_EXPERTS` (ON), `STRATA_GGML_DIR`, `STRATA_PORTABLE`, `STRATA_ISA_FLOOR`.

## Model-specific behaviour

Per the layer tables shipped as test data (`tests/data/native_experts/`, Flash-Next family, 48 layers):

| Table | Gate/up types (layers) | Down types (layers) | CPU kernels |
|---|---|---|---|
| `iq2_xs.txt`, `swift-iq2_xs.txt` | IQ2_S (34), IQ2_XXS (11), IQ1_M (3) | Q2_0 (48) | i-quant multi-token; IQ1_M on ggml-cpu; Q2_0 rows |
| `iq3_xxs.txt` | IQ3_S (13), IQ2_XS (10), IQ2_S (10), IQ2_XXS (9), IQ3_XXS (6) | Q2_0 (30), IQ4_NL (18) | i-quant multi-token; Q2_0 rows; `iq4nl256_down_rows` |
| `iq3_s.txt`, `coder-iq1_m.txt` | IQ2_S (20), IQ3_XXS (17), IQ3_S (10), IQ4_XS (1) | IQ4_NL (39), Q2_0 (9) | as above; IQ4_XS on AVX2-only CPUs, ggml-cpu on AVX-512 |
| `q2_0.txt` | Q2_0 (48) | Q2_0 (48) | Q2_0 rows for all three projections |

Unsloth UD-Q4_K_XL packs are Q4_K gate/up with Q5_1 / Q8_0 down (`kq_avx2.hpp:1-3`): ggml-cpu by default, the
`kq256_*` kernels with `STRATA_KQ256=1`.

## Tests

`native_expert_parity` (CPU path vs a ggml `to_float` float reference, plus the GPU path; its ctests use synthetic
K-quant weights) and `router_dot_parity` (AVX2 and AVX1 router dot vs double). See [tests.md](tests.md) and
[../parity/README.md](../parity/README.md).

## Gotchas

- `kq_avx2.cpp` has its own `MAXT = 8` (`kq_avx2.cpp:28`); the i-quant `dot_rows_nt` / `gu_rows_nt` split groups
  above 4 tokens into groups of 4 (`iq_avx512.cpp:214-240`, `iq_avx2.cpp:379-405`).
- `bf16_rows_dot` (single-token) has no caller in `src/`.
- `native_fmt` does not bound `n_embd` / `n_ff` against the pool's buffer widths; upstream only ever passes the
  compile-time 2560 / 640.

## In UnbiasedStrata

`native_expert.cpp` changed (22d72e6): `native_fmt` refuses an expert geometry wider than 2560 x 640, the pool's
buffer size, because the fork passes each model's own widths (Qwen3.6-35B-A3B: 2048 x 512). The kernel files and
headers are unchanged. The fork's page: [../../../codebase/kernels/cpu/native-experts.md](../../../codebase/kernels/cpu/native-experts.md).

## See also

- [q2-expert.md](q2-expert.md), [pool.md](pool.md), [expert-layout.md](expert-layout.md)
- [../cuda/README.md](../cuda/README.md) (the GPU's grouped native-expert kernels)
