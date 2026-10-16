[Upstream docs](../../README.md) › [Kernels](../README.md) › [CPU](README.md) › Expert layout and CPU probes

# Expert layout table and CPU feature probes

## Purpose

`expert_layout.cpp` holds two process-wide things every CPU expert path asks about:

1. **The expert layout**: where each routed expert's blob lives in `experts.bin` (or in the GGUF shards), how big it
   is, and in which formats, per layer. A canonical Q2_0 pack has one fixed blob size; a native pack has a per-layer
   table in `native_experts.txt`. "Everything that touches an expert blob - the arena, the VRAM tier, the prompt path,
   the CPU pool, the GPU window - asks this table" (`include/strata/kernels/cpu/expert_layout.hpp:1-6`).
2. **The CPU ISA probes** (`cpu_avx512_ok`, `cpu_avx2_ok`, ...), compiled **without** AVX flags so asking is safe on
   any x86-64 CPU, plus the `q2_rows_any` / `act_quant_any` dispatchers.

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/kernels/cpu/expert_layout.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/expert_layout.hpp) | 82 | `ExpertLayout`, probe declarations, `kExpertLayoutVersion` |
| [src/kernels/cpu/expert_layout.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/expert_layout.cpp) | 341 | Probes, dispatchers, `native_experts.txt` parser, no-ggml stubs |

## Main types and functions

| Symbol | Where | What it does |
|---|---|---|
| `ExpertLayout` | `expert_layout.hpp:18` | `native`, `n_layers`, `n_expert`; per layer `fmt` (`NativeFmt`), `offset`, `bytes`; per layer and role `gguf_off` and `gguf_file` (3 x n_layers); `version`, `max_blob`, `total`. Accessors `blob_bytes`, `layer_offset`, `blob_offset` (`:35-41`) |
| `expert_layout()` | `expert_layout.cpp:22` | The global (canonical Q2_0 until a native table is loaded) |
| `expert_layout_load(pack_dir, n_layers, n_expert, err)` | `expert_layout.cpp:233` | Parses `<pack_dir>/native_experts.txt`, or sets the canonical layout when the file is absent |
| `kExpertLayoutVersion = 4` | `expert_layout.hpp:80` | Newest table version read; a newer header is refused |
| `cpu_isa_cap()` | `expert_layout.cpp:24` | `STRATA_FORCE_ISA=avx2\|avx\|sse` (also `sse4.2`, `none`) caps the probes (3 = no cap); an unknown value is ignored with a message |
| `cpu_avx512_ok()` | `expert_layout.cpp:40` | F + BW + VL + VNNI + VBMI and OS XSAVE support for ZMM state (`xcr0 & 0xE6`). `STRATA_FORCE_AVX2=1` or a cap below 3 answers no |
| `cpu_avx512bw_ok()` | `expert_layout.cpp:73` | F + DQ + BW + VL (no VNNI / VBMI). The header says it is for `iq_avx512.cpp`, but nothing outside this file calls it |
| `cpu_avx2_ok()` | `expert_layout.cpp:105` | AVX + AVX2 + FMA + F16C with OS YMM support: the floor of every Strata expert kernel |
| `cpu_avx1_ok()`, `cpu_sse42_ok()` | `expert_layout.cpp:138`, `:164` | Floors of the experimental older-CPU builds (`cpu_avx1_ok` from the Strata_Dirigo fork) |
| `isa_floor_build()` | `expert_layout.cpp:179` | `"avx"`, `"sse4.2"` or `""` from `STRATA_ISA_FLOOR_AVX` / `_NONE` |
| `q2_native_kernels(type)` | `expert_layout.hpp:65` | `type == 42 && cpu_avx2_ok()`: a native layer of Q2_0 runs on Strata's Q2_0 kernels |
| `cpu_name()` | `expert_layout.cpp:189` | CPUID brand string, for messages |
| `q2_rows_any`, `act_quant_any` | `expert_layout.cpp:212`, `:218` | AVX-512 or AVX2 variant of the Q2_0 rows / quantizer ([q2-expert.md](q2-expert.md)) |

All probes cache their answer in a function-local `static`, so environment overrides must be set before the first
call.

## `native_experts.txt`

Written by `tools/iq_pack.py`. One header line and one line per layer; for example the first lines of
`tests/data/native_experts/iq3_s.txt`:

```
# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] (n_expert 512, total 50292326400; ...)
0 18 20 0 2176000 1386491904 1711166464 913711104
```

Parsing rules (`expert_layout.cpp:233-339`):

- No file: canonical Q2_0 layout, `total = n_layers * n_expert * BLOB` (`:238-242`).
- A file, but a build without `STRATA_NATIVE_EXPERTS`: refused (`:243-245`).
- `# strata native experts vN`: `N > 4` is refused ("the pack was written by a newer tools/iq_pack.py", `:258-266`).
  `(n_expert K)` anywhere in a comment line overrides the caller's expert count, because a pruned model such as the
  GSQ-RCO Coder ships fewer experts (`:268-272`; `coder-iq1_m.txt` says `n_expert 256`).
- Each layer line: `native_fmt(gu, d, H, FF)` must succeed and its byte count must equal `blob_bytes` (`:283-289`).
  The widths are the compile-time 2560 / 640 of `expert.hpp`.
- v2 adds the three absolute GGUF offsets of gate / up / down; v3 a shard file name; v4 allows `gate,up,down` (three
  names, an empty field meaning the `--native` shard) when a shard boundary falls inside a layer (Unsloth UD-Q4_K_XL,
  #255). One name is replicated to all three roles; any other count, or a trailing column, is refused (`:290-321`).
- Layers must be present and contiguous: `offset[l]` must equal the running sum of `bytes * n_expert` (`:327-334`).

## Callers and dependencies

- `src/program/generate.cpp:1997` calls `expert_layout_load(o.pack, g0.n_layers, g0.n_expert, err)` with the
  default `ModelGeometry` (48 layers, 512 experts); `:2016` then reads `expert_layout().native`.
- `src/core/expert_source.cpp` (12 uses) reads `expert_layout()` for blob addressing, `gguf_off` / `gguf_file`
  (filling the arena from GGUF shards, `expert_source.cpp:636`, `:685`, `:2466`) and the verify window's native
  dispatch; `src/prefill/prefill.cpp`, `src/core/verify.cpp`, `src/core/remote_experts.cpp`,
  `src/core/peer_experts.cpp` and `src/core/remote_expert_opt.cu` read it too.
- [pool.md](pool.md) uses `q2_native_kernels`, `q2_rows_any`, `act_quant_any`.
- `src/program/generate.cpp:1961-1985` and `:2054-2059` use the probes to refuse a CPU without AVX2 (unless the
  engine is an older-CPU build) and to report which kernels a native pack will run on.
- Depends on `native_fmt` ([native-experts.md](native-experts.md)) and on CPUID / `xgetbv`.

## Configuration

| Knob | Effect |
|---|---|
| `STRATA_FORCE_AVX2=1` | `cpu_avx512_ok()` and `cpu_avx512bw_ok()` return false (tests run the AVX2 kernels on an AVX-512 CPU) |
| `STRATA_FORCE_ISA=avx2\|avx\|sse` | Caps `cpu_avx512_ok` / `cpu_avx2_ok` / `cpu_avx1_ok`; ggml-cpu is not affected (it runs what it was compiled for) |
| CMake `STRATA_ISA_FLOOR` | `avx` or `none`: defines `STRATA_ISA_FLOOR_AVX` / `STRATA_ISA_FLOOR_NONE` on `strata_kernels_cpu` (`CMakeLists.txt:1023-1027`) |
| CMake `STRATA_NATIVE_EXPERTS` | OFF: native tables are refused and the native entry points are stubs (`expert_layout.cpp:223-231`) |

## Model- and platform-specific behaviour

The Q2_0 canonical pack has no table (canonical layout). The IQ2_XS, IQ3_XXS, IQ3_S, Coder IQ1_M, Swift 1.5 and
Unsloth UD-Q4_K_XL packs are native; copies of six real tables are in
[tests/data/native_experts/](https://github.com/Niko1221/Strata/tree/6f32ec070f23ced9f50e704d854d775da52591ab/tests/data/native_experts)
(Q2_0 as a native pack, IQ2_XS, IQ3_XXS, IQ3_S, Coder IQ1_M, Swift IQ2_XS; v2 and v3 headers). The CPUID code has an
MSVC branch (`__cpuidex`, `_xgetbv`) and a GCC/Clang branch (`__cpuid_count`, inline `xgetbv`).

## Tests

- `tests/core/expert_layout_test.cpp` (ctest `expert_layout_test`, `CMakeLists.txt:438-444`: `STRATA_BUILD_TESTS`
  and `STRATA_NATIVE_EXPERTS`, inside the GPU block): parses the six real tables against the pre-v4 reader's rules,
  the v4 shard column and its refusals, and fills a synthetic arena from GGUF shards at 2560 / 640.
- `router_dot_parity_avx` runs `router_dot_parity` under `STRATA_FORCE_ISA=avx` ([tests.md](tests.md)).

## Gotchas

- The probes are `static`-cached: setting `STRATA_FORCE_*` after the first call has no effect.
- `v4` is written only when some layer's roles span shards; every other pack stays v3 byte for byte
  (`expert_layout.hpp:78-79`).
- The v4 comma split copies the single name before `assign` to avoid aliasing (`expert_layout.cpp:309-312`).
- `cpu_avx512bw_ok()` ignores `STRATA_FORCE_ISA` (it checks only `STRATA_FORCE_AVX2`).

## In UnbiasedStrata

Changed in commit e41dfc7 (the Qwen3.6-35B-A3B port): `expert_layout_load` gained `n_embd` / `n_ff` parameters (the
model file's widths), builds each layer's `NativeFmt` with them, and refuses a canonical pack (no
`native_experts.txt`) whose model is not 2560 x 640. The fork's page:
[../../../codebase/kernels/cpu/expert-layout.md](../../../codebase/kernels/cpu/expert-layout.md).

## See also

- [native-experts.md](native-experts.md), [q2-expert.md](q2-expert.md), [pool.md](pool.md)
- [../../tools/pack-format.md](../../tools/pack-format.md) (the pack files)
