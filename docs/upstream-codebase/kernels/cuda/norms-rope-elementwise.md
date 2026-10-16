[Upstream docs](../../README.md) › [Kernels](../README.md) › [CUDA](README.md) › RoPE, norms, elementwise

# RoPE, norms, elementwise glue and the doorbell kernels

## Purpose

The small kernels around the mixers and the MoE:

- RoPE: float64 cos/sin tables built on the host with the scaling folded in, the table-driven NEOX rotation, the
  analytic pinned kernel (`--native-rope`), the vision path's M-RoPE position table, the opt-in angle table, and the
  rope-scaling configuration (none / linear / YaRN);
- elementwise glue: the GDN softplus gate, scale/add/SiLU, conversions, the canonical embedding gather, the
  weighted RMSNorm used by QSA;
- the doorbell kernels and mapped-memory copies that hand each layer's routed-expert work to the CPU expert pool
  and read the results back inside a captured graph.

Files: `src/kernels/cuda/rope.cu`, `native_rope.cu`, `elementwise.cu`; headers `include/strata/kernels/rope.hpp`,
`rope_scaling.hpp`, `mrope.hpp`, `native_rope.hpp`, `elementwise.hpp`. The hyper-connection norms and the native QSA
norm are on [hyper-connections-ple.md](hyper-connections-ple.md) and [attention-qsa.md](attention-qsa.md).

## RoPE

The model rotates the first `n_rot` = 64 of each head's 256 dims (QSA q and k) and of the indexer's 128 dims, in NEOX
pairs `(i, i + 32)`, with `freq_base` 1e7 (`qsa.hpp:120-125`). The vision path uses interleaved M-RoPE with sections
11/11/10/0 (`mrope.hpp:3-7`).

### Table path (`rope.hpp`, `rope.cu`)

- `build_rope_table(n_rot, theta, max_pos, cos, sin)` (`rope.cu:48`) and `build_rope_table(n_rot, RopeScaling, ...)`
  (`rope.cu:61`) fill `[max_pos][n_rot/2]` cos and sin on the host in float64, in the reference's order. With no
  scaling the scaled builder is bit-identical to the plain one.
- `rope_neox_apply` (`rope.cu:117`): one thread per row, 128 threads per block; partial rotation (64 of 256 dims,
  the rest copied). `pos` is a device array (graph capture). The pair arithmetic is `rope_neox_pair` (`rope.hpp`,
  host and device), which the indexer pooling kernel in `qsa.cu` also calls. Every rope kernel takes the M-RoPE table
  as an argument.

### Analytic path (`native_rope.hpp`, `native_rope.cu`)

`native_rope_apply` (`native_rope.cu:118`, `--native-rope`): ggml `rope_multi` / `rope_yarn` for text IMRoPE. Rows
of F32, head_dim 128 or 256, `n_rot` 64; grid `(ceil(head_dim/2 / 128), rows)`, 128 threads. `theta_scale` =
`powf(freq_base, -2 / n_rot)` is computed on the host to match the pinned host `powf`; the kernel computes the angle
with fast-math `powf/cosf/sinf` through `rope_scaled_angle`, unless the opt-in table is active. Bad arguments throw.

### Rope scaling (`rope_scaling.hpp`)

`RopeScaling` (`rope_scaling.hpp:66`) holds `type` (`None | Linear | YaRN`, `:48`), `freq_base` (1e7), `factor`,
`freq_scale_in`, `orig_ctx` (262,144), `ext_factor`, `attn_factor`, `beta_fast` (32) and `beta_slow` (1), transcribed
from ggml (`rope_scaling.hpp:13-19`). `freq_scale()`, `mscale()` and `corr_dims()` give ggml's quantities; the ramp
takes a pair index instead of ggml's even dim index (same numbers). `kernel_args(n_rot)` returns the float constants
for the analytic kernels (the identity for `None`). `rope_scaling_invalid` (`:169`) is the single validity rule.
`rope_scaling_set` (`rope.cu:45`, declared at `rope_scaling.hpp:190`) is called once at start-up
(`src/program/generate.cpp:2141`), before any table is built or graph captured, because K is cached after RoPE and a
mid-run change would mix two scalings (`rope_scaling.hpp:29-34`).

### M-RoPE and the angle table (`mrope.hpp`, implemented in `native_rope.cu:72-114`)

- `mrope_table_set(device_table)` (per device): null (the default) means the position passed is the rotary position.
  When set, it is an int32 `[cells][3]` (t, h, w) table and the passed position is read as a cell index; pair i
  rotates by `tab[cell * 3 + i % 3]` (`mrope_pos`). It must be set before capture; the contents may change between
  requests. `src/program/generate.cpp:2296-2311` sets it for `--vision` (initially the identity) and `:2624` per
  session.
- `rope_table_set` / `rope_table_for` (`native_rope.cu:103`, `:111`): with `STRATA_ROPE_TABLE=1` (#280), the analytic
  kernels read the session's float64-derived table (registered by `src/core/layer.cpp:698`) instead of computing
  fast-math angles. The header records 0.0014 rad of error at 32K and about 0.02 at 262K for the fast-math path, and
  that the prompt path (precise libm) and decode (fast math) otherwise disagree (`mrope.hpp:29-34`). Off by default.

## Elementwise glue (`elementwise.hpp`, `elementwise.cu`)

256-thread kernels unless noted. Calls with a null stream synchronise.

| Function | What |
| --- | --- |
| `embedding_gather` (`elementwise.cu:133`) | decode one canonical (S2/S4/S8) embedding row: `code + bias` times the group scale (`__fmul_rn`), plus the offset (`__fadd_rn`), rounded separately so it matches the CPU |
| `gdn_gate` (`:142`) | `softplus(alpha + dt) * ssm_a`, softplus with ggml's `x > 20` branch |
| `scale_inplace`, `add_inplace`, `silu_inplace` (`:151`, `:158`, `:179`) | elementwise |
| `f32_to_f16_bulk`, `f32_to_bf16_bulk` (`:165`, `:172`) | conversions through `f16_from_f32` / `bf16_from_f32` ([bits-and-arch-shims.md](bits-and-arch-shims.md)); separate entry points on purpose (`elementwise.hpp:60-64`) |
| `rms_norm_weighted` (`:386`) | `x / sqrt(mean(x^2) + eps) * w`, a warp per row, 4 warps per block; QSA's q / k / indexer-q norm on the canonical path. Not `gdn_l2_norm` (`elementwise.hpp:69-82`) |
| `doorbell_publish` (`:353`) | one 1,024-thread block copies x (n floats), ids and weights (k each) into mapped host memory, fences (`__threadfence_system`), and increments the mapped sequence number: the signal for the CPU expert pool to start on this layer's misses. It increments rather than stores so a captured graph rings a new value on each replay (`elementwise.hpp:90-93`) |
| `doorbell_publish_value` (`:345`) | #649 (HIP, `STRATA_DOORBELL_STORE=1`, read in `src/core/verify.cpp:82`): the ring stored as the step's own number instead of a read-modify-write over PCIe |
| `doorbell_publish_res` (`:361`) | the publish plus the residency row, for the device-side hit plan |
| `doorbell_ring` (`:379`), `doorbell_wait` (`:220`) | one-thread kernels. `doorbell_wait` spins (`strata_spin_pause`) until the host flag equals the ring, then fences; it lets one graph hold the whole token (`elementwise.hpp:95-99`). An event record inside a capture is silently dropped, which is why these are kernels (`elementwise.hpp:84-87`) |
| `copy_from_mapped` (`:281`), `copy_rows_from_mapped` (`:271`), `copy_i32_from_mapped` (`:373`), `scatter_rows_f32` (`:263`) | float4 / int32 copies from mapped pinned memory, kept on the compute queue instead of copy-engine memcpy nodes (67 flushes per token under WDDM, `elementwise.hpp:101-103`) |

On HIP, CMake compiles `elementwise.cu` and `quantize_act.cu` with `-ffp-contract=off` so separately rounded
operations are not fused into FMAs (`CMakeLists.txt:380-384`).

## Control and data flow

```
QSA layer: q, k, indexer q --rms_norm_weighted | native_qsa_rms_norm_weighted-->
           --rope_neox_apply(cos_tab, sin_tab, pos_dev) | native_rope_apply(scaling, pos_dev)-->   (layer.cpp:878-882)
MoE:       router ids/weights --doorbell_publish--> mapped host memory --> CPU pool --> host flag
           --doorbell_wait--> copy_from_mapped (expert outputs) --> combine
```

Callers: `src/core/layer.cpp` (QSA norm/rope in `qsa_layer`, the step-state mapped copies at `:913-914`, the MoE
doorbell), `src/core/verify.cpp`, `src/core/mtp.cpp`, `src/core/session.cpp`, `src/prefill/`.

## Configuration

- CLI: `--native-rope`; `--rope-scaling none|linear|yarn`, `--rope-scale`, `--rope-freq-base`, `--rope-freq-scale`,
  `--yarn-orig-ctx`, `--yarn-ext-factor`, `--yarn-attn-factor`, `--yarn-beta-fast`, `--yarn-beta-slow`
  (`src/program/generate.cpp:591-600`, parsed from `:1284`); `--vision`.
- Environment: `STRATA_ROPE_TABLE=1`; caller-side `STRATA_DOORBELL_STORE=1` (HIP).
- Measured effect of scaling: `bench/results/2026-09-28-rope-scaling` in the snapshot.

## Model- and platform-specific behaviour

- All model variants share n_rot 64, freq_base 1e7 and the trained 262,144-cell context (`orig_ctx`); scaling is a
  pure runtime feature because the GGUFs ship no `rope.scaling` keys (`rope_scaling.hpp:3-5`).
- gfx906 builds include `hip/hip_runtime.h` in `mrope.hpp` (PR #638); the RoPE headers mark their helpers
  `__host__ __device__` under both nvcc and hipcc.

## Tests

`rope_parity` (tables against a float64 transcription for none / linear / YaRN, bit-exact; the native kernel),
`elementwise_parity` (including the `gdn_l2_norm` vs `rms_norm_weighted` distinction and `dequant_bf16`). Oracle-linked
(source not in the snapshot): `native_rope_parity`.

## Gotchas

- `rope_neox_apply` reads `pos` on the device; a host pointer is an illegal access (`qsa.hpp:205-209`).
- `mrope_table_set` and `rope_scaling_set` must run before any graph capture; captured graphs keep the arguments they
  saw.
- `rope_table_release` must run before the table's memory is freed (`mrope.hpp:42-43`).

## In UnbiasedStrata

Kept unchanged: `rope.cu`, `native_rope.cu`, `elementwise.cu` and their headers are identical in the fork. The fork's
Qwen3.6 path additionally uses `add_inplace` as its plain residual add. See the fork's
[norms-rope-elementwise.md](../../../codebase/kernels/cuda/norms-rope-elementwise.md).

## See also

- [attention-qsa.md](attention-qsa.md), [gdn.md](gdn.md), [hyper-connections-ple.md](hyper-connections-ple.md),
  [moe-and-experts.md](moe-and-experts.md) (the doorbell's payload), [bits-and-arch-shims.md](bits-and-arch-shims.md)
- [CPU kernels](../cpu/README.md) (the expert pool on the other side of the doorbell)
