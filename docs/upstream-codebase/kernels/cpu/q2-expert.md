[Upstream docs](../../README.md) › [Kernels](../README.md) › [CPU](README.md) › Q2_0 expert kernel

# The Q2_0 expert kernel (AVX-512 VNNI and AVX2)

## Purpose

Computes one routed expert of a **canonical Strata Q2_0 pack** (and the Q2_0 rows of a native pack) on the CPU:
`out(H) = down(swiglu(gate(x), up(x)))`, all three projections 2-bit Q2_0, against an int8-quantized activation. It
is the kernel the [expert pool](pool.md) runs for every expert the GPU does not hold. The code was promoted from
upstream's micro-benchmark `bench/micro/cpu_s2.cpp` (not in the published source) and is DRAM-bandwidth bound by
design. The header records the measurement that motivated it: 15.03 ms for one token's full routed set on 6 cores,
DRAM-bound at 44.14 GB/s (`include/strata/kernels/cpu/expert.hpp:3-5`, quoting the project's ledger).

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/kernels/cpu/expert.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/expert.hpp) | 179 | Geometry constants, `ActQ`, scratch structs, every entry point, the design notes |
| [src/kernels/cpu/expert.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/expert.cpp) | 683 | AVX-512 (F/BW/VL/VNNI/VBMI) implementation, the "oracle" Q8_0 contract, the scalar reference, the CPUID probe |
| [src/kernels/cpu/q2_avx2.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/q2_avx2.cpp) | 130 | The Q2_0 GGUF-layout rows and the activation quantizer for AVX2-only CPUs |

`expert.cpp` is compiled with `-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi -mf16c -mfma`
(MSVC `/arch:AVX512`), `q2_avx2.cpp` with `-mavx2 -mfma -mf16c` (MSVC `/arch:AVX2`), per source file
(`CMakeLists.txt:932-940`), so no other translation unit gets the wider ISA. The CMake comment at `:887-897`
explains why: a translation unit built for AVX-512 may use AVX-512 anywhere in it, so the probes and the test
drivers must live elsewhere.

## Geometry (compile-time)

`expert.hpp:34-51` fixes the canonical blob: `H = 2560`, `FF = 640`, `NE = 512`, `QK = 64` weights per fp16 weight
scale, `QKA = 32` activation elements per activation scale, `BB = 18` bytes per 64-weight block,
`BLOB = 1,382,400` bytes per expert. The blob is planar: gate/up codes interleaved by row (`2r` = gate row r,
`2r+1` = up row r) at `O_GU_CODES`, down codes at `O_D_CODES`, then the gate/up fp16 scales (`O_GU_SCALES`) and the
down scales (`O_D_SCALES`).

**The contract that matters** (`expert.hpp:22-25`): one 64-weight block spans **two** 32-element activation chunks,
each with its own scale. Using one activation scale per weight block is "the natural-looking mistake".

Every upstream model has this expert geometry (2560 x 640), so the constants are never wrong upstream; the
`native_experts.txt` reader also passes them as the expected widths (see [expert-layout.md](expert-layout.md)).

## Main types and functions

| Symbol | Where | What it does |
|---|---|---|
| `ActQ` | `expert.hpp:64` | int8 activation `q[H]`, per-chunk `scale`, integer `sum`, and `hx = scale*sum` (the code-offset correction, precomputed once per layer, not per row) |
| `ExpertScratch`, `ExpertScratchMulti` | `expert.hpp:74`, `:147` | Caller-owned per-worker scratch; the token path allocates nothing |
| `CpuFeatures`, `cpu_features()` | `expert.hpp:83-93`, `expert.cpp:352` | CPUID leaf 7 probe of AVX512 F/BW/VL/VNNI/VBMI; `reason()` names each missing bit (`expert.cpp:341`) |
| `cpu_require_expert_support()` | `expert.cpp:375` | Exits with a message ("Intel Ice Lake / AMD Zen 4 or newer") if the CPU cannot run the VNNI kernel. Called at `src/program/generate.cpp:2054` for canonical packs only |
| `expert_set_oracle_q8_0(bool)` | `expert.cpp:453` | Switches both quantization boundaries to the pinned ggml x86 Q8_0 contract and the generic CPU Q2_0 dot order (`--cpu-oracle-q8-0`, `generate.cpp:1955`) |
| `act_quant_q8_1(x, n, a)` | `expert.cpp:386` | Default ("legacy") activation quantizer: FP32 scale `amax/127`, round half away from zero, clamp to +-127; AVX-512 body. The name is historical, not a ggml block layout |
| `s2_expert_vnni` / `s2_expert_vnni_q` | `expert.cpp:457`, `:462` | Whole expert from float `x` / from an already-quantized shared `ActQ` (the pool uses `_q` so a layer's ten experts share one quantization) |
| `s2_expert_gu_rows` / `s2_expert_down_rows` | `expert.cpp:484`, `:494` | Row ranges of the gate/up and down phases, so several threads can share one expert (`run_split`) |
| `s2_expert_gu_rows_multi` / `s2_expert_down_rows_multi` / `s2_expert_vnni_multi` | `expert.cpp:522`, `:536`, `:611` | Up to `MAXT = 8` tokens routed to the same expert: the codes are unpacked once per block for all tokens |
| `q2_0_gguf_rows_multi` | `expert.cpp:597` | Q2_0 rows in the GGUF block layout (2 B fp16 `d` + 16 code bytes per 64 weights, interleaved), for native packs' Q2_0 layers |
| `q2_0_gguf_rows_multi_avx2`, `act_quant_q8_1_avx2` | `q2_avx2.cpp:75`, `:90` | AVX2 versions of the two functions above; above 4 tokens the rows run in groups of 4 (`q2_avx2.cpp:84-88`) |
| `s2_expert_scalar(blob, x, out, quant_acts)` | `expert.cpp:629` | Scalar transcription of ggml's formula: the oracle `expert_parity` compares against; `quant_acts` quantizes at **both** stages |

The AVX-512 / AVX2 choice for native packs goes through `q2_rows_any` and `act_quant_any`
(`src/kernels/cpu/expert_layout.cpp:212-221`), keyed on `cpu_avx512_ok()`.

## How the row dot works

`row_dot` (`expert.cpp:232`) evaluates the identity
`sum (c-1) d_w xhat d_x = d_w * (d_x*sum(c*xhat) - d_x*sum(xhat))` per 32-element chunk:

- `unpack_q2_0` (`expert.cpp:125`) unpacks 2-bit codes with `vpmultishiftqb`; the argument order is
  `(control, data)`, determined empirically (the Intel guide's order "produced 43 of 64 wrong codes",
  `expert.cpp:118`).
- `vpdpbusd` (VNNI) does the code x int8 products; the accumulator stays a float vector and is reduced once per row
  (reducing per block made the kernel "compute-bound at 21 GB/s instead of DRAM-bound at 32", `expert.cpp:228-231`).
- By default (`kZmm`, `expert.cpp:153`) the 512-bit variant `row_dot_z` (`expert.cpp:174`) is used: one unpack of 64
  codes, one VNNI dot and one FMA per block, scales built eight blocks at a time. Setting `STRATA_CPU_YMM` (any value)
  selects the older 256-bit loop. The two differ in the last bits; plain decode and the verify window use the same
  one, so speculative decoding still reproduces plain decode (`expert.cpp:145-152`).
- The oracle contract uses `row_dot_oracle` (`expert.cpp:261`): integer chunk dots with the code offset removed before
  conversion (`chunk_dot_oracle`, `:254`), reduced per 64-block in ggml's generic order, and `quantize_oracle_q8_0`
  (`expert.cpp:77`, adapted from llama.cpp 3cf03257 under MIT, `:51-76`): FP16-rounded scales, round to nearest even.

SwiGLU is `silu(gate) * up` (`expert.cpp:472-474`); the intermediate is re-quantized with `act_quant_q8_1` before
the down projection.

## Invariants

- `s2_expert_vnni_multi` and the `_rows_multi` functions produce, for each token, **bitwise** the single-token result
  (same per-token operation order), checked by `expert_multi_test`.
- `s2_expert_gu_rows` + `act_quant_q8_1(ff, FF)` + `s2_expert_down_rows` is bitwise `s2_expert_vnni_q`
  (`expert.hpp:132-135`): this is what lets the pool split one expert by rows.
- The multi-token path is legacy-contract only: with the oracle contract on, `s2_expert_vnni_multi` falls back to one
  `s2_expert_vnni_q` per token (`expert.cpp:613-616`), and the pool's `run_split*` fall back to `run`.
- The router weight is never applied here; the GPU's `moe_combine` applies it.

## Callers and dependencies

- [pool.md](pool.md): `ExpertPool::drain` calls `s2_expert_vnni_q` (mode 0), `s2_expert_gu_rows` /
  `s2_expert_down_rows` (modes 1-2), the `_rows_multi` pair (modes 3-4) and `q2_rows_any` (native Q2_0 layers in
  modes 5-6) (`src/kernels/cpu/pool.cpp:542-620`); `ExpertPool::run` also calls `s2_expert_vnni_q` on the host
  thread (`pool.cpp:723`, `:760`).
- `src/core/expert_source.cpp:2144-2149` quantizes each verify-window token with `act_quant_any` (native Q2_0 pack),
  `native_quant_act` (other native packs) or `act_quant_q8_1` (canonical).
- No dependency on ggml: only `<immintrin.h>` and CPUID.

## Configuration

| Knob | Effect |
|---|---|
| `--cpu-oracle-q8-0` (`generate.cpp:1335`) | `expert_set_oracle_q8_0(true)`: the experimental pinned ggml contract. Refused together with `--expert-cache` or `--expert-profile` (`generate.cpp:1947-1949`), because GPU expert-cache parity is not established for it (`expert.hpp:103-104`) |
| `STRATA_CPU_YMM` (env, any value) | Use the 256-bit row dot instead of the 512-bit one |
| CMake per-source `-mavx512*` / `-mavx2` flags | See [README.md](README.md#build) |

## Model- and platform-specific behaviour

- **Q2_0 canonical pack** (`tools/strata_pack.py`): every function here; requires an AVX-512 VNNI + VBMI CPU.
- **Native packs**: only the GGUF-layout rows, for Q2_0 (ggml type 42) layers. In the tables shipped as test data
  (`tests/data/native_experts/`) every down projection of the IQ2_XS and Swift IQ2_XS packs is Q2_0, as are 30 of
  48 in IQ3_XXS and 9 of 48 in IQ3_S and the Coder IQ1_M; the `q2_0.txt` native pack is Q2_0 throughout, so its
  gate/up rows take these kernels too.
- **Windows and Linux** run the same code; MSVC gets `/arch:` flags, GCC/Clang `-m` flags.

## Tests

| Test | What |
|---|---|
| `expert_parity` (`src/kernels/cpu/expert_parity.cpp`) | VNNI vs `s2_expert_scalar` on a real blob from `pack/full/experts.bin`: <= 3e-3 with quantized activations; the FP32-activation gap is reported |
| `expert_multi_test` (`src/kernels/cpu/expert_multi_test.cpp`) | `s2_expert_vnni_multi` bitwise vs `s2_expert_vnni_q` for n = 1..8, synthetic blobs |
| `native_expert_parity` (`src/kernels/native_expert_parity.cpp`) | Exercises `q2_rows_any` / `act_quant_any` on native Q2_0 down rows |

Details and run conditions: [tests.md](tests.md), [../parity/README.md](../parity/README.md).

## Gotchas

- In `act_quant_q8_1` the AVX-512 block returns unconditionally (`expert.cpp:395-420`), so the scalar loop after it
  (`:421-450`) is unreachable; it documents the rule the vector code reproduces bit for bit (and why `std::lround`
  was replaced: 51.3 us per 2,560 values, `:430-439`).
- The scalar oracle's `quant_acts` quantizes both stages; the original in `bench/micro/cpu_s2.cpp` quantized only
  the intermediate, which showed up as a 1.19e-02 gap that "read as a kernel bug when it was a fixture bug"
  (`expert.hpp:164-167`, `expert.cpp:630-637`).
- `h2f` is written out by hand rather than with `_cvtsh_ss` because of subnormal handling (`expert.cpp:26-50`).
- `cpu_features()` (CPUID only) does not check the OS's XSAVE state; `cpu_avx512_ok()` in `expert_layout.cpp` does.
  The canonical-pack check uses the former.

## In UnbiasedStrata

Kept unchanged (`expert.hpp`, `expert.cpp`, `q2_avx2.cpp` are byte-identical at the fork's HEAD). The fork's page:
[../../../codebase/kernels/cpu/q2-expert.md](../../../codebase/kernels/cpu/q2-expert.md).

## See also

- [native-experts.md](native-experts.md), [pool.md](pool.md), [expert-layout.md](expert-layout.md), [tests.md](tests.md)
- [docs/upstream/DETAILS.md](../../../upstream/DETAILS.md)
