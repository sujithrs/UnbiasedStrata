[Upstream docs](../../README.md) › [Kernels](../README.md) › [CUDA](README.md) › Gated DeltaNet

# Gated DeltaNet (GDN) kernels

## Purpose

The Gated DeltaNet ("linear attention") layers are 36 of the model's 48 layers (`include/strata/core/graph.hpp:45`).
Each keeps a fixed-size recurrent state per value head instead of a KV cache. These kernels compute everything
between the layer's projections (`attn_qkv`, `attn_gate` (z), `ssm_alpha`, `ssm_beta`, `ssm_out`): the 4-tap causal
convolution with SiLU, the L2 norm of q and k, the decay gate and beta, the delta-rule recurrence on the
128 x 128 state of each value head, and the gated RMS output norm. The projections are GEMVs
([gemv-and-quant.md](gemv-and-quant.md), [moe-and-experts.md](moe-and-experts.md)).

Files:

- `src/kernels/cuda/gdn.cu`, `include/strata/kernels/gdn.hpp`: the reference ("canonical") kernels and `GdnShapes`.
- `src/kernels/cuda/fused_gdn.cu`, `include/strata/kernels/fused_gdn.hpp`: one kernel per stage, the default with
  the native contract.
- `src/kernels/cuda/native_gdn.cu`, `native_gdn_preprocess.cu` and their headers: the pinned llama.cpp adapters.
- `src/kernels/cuda/verify_kernels.cu` (GDN part), `include/strata/kernels/verify_kernels.hpp`: the verify-window
  kernels and the commit.
- `gdn_gate` (softplus) is in `src/kernels/cuda/elementwise.cu:142`.

The prompt path's GDN kernels are in `src/prefill/kernels.cu` (see [Prefill](../../prefill/README.md)).

## The math

From `gdn.hpp:19-30` and `fused_gdn.hpp:10-12`. For value head h, with key head `idx[h] = h % h_k` (modulo, not
interleave; `gdn.hpp:54-55`):

```
g      = exp(gate[h])                  gate = softplus(alpha + dt) * ssm_a  (ssm_a < 0, so g is in (0, 1))
beta   = sigmoid(beta_raw[h])
S     <- g * S + k (beta * (v - g * S^T k))^T              (decay before the rank-1 update)
o      = S^T q / sqrt(128)                                 (reference: q is pre-scaled instead)
y      = rms_norm(o) * ssm_norm * sigmoid(z)
```

The output gate is always `sigmoid(z)`. `gdn.hpp:90-93` stresses it: "SIGMOID, not SiLU - qwen3.5 used SiLU and this
artifact does not". Each `(j, h)` column of the state is independent, so one thread or warp can own a column for the
whole step with no barrier inside the recurrence (`gdn.hpp:28-30`).

### Layouts

- Recurrent state: `(S, h_v, S)`, i.e. `state[(i * h_v + h) * S + j]`, j fastest, so a warp walking j reads
  contiguous memory. This differs from `ref/gdn.py`'s `(S, S, H_v)` on purpose (`gdn.hpp:13-17`).
- Conv weight: GGML-native `kW[c * d_conv + i]` (ne0 = d_conv fast), tap 0 reading the oldest history row
  (`gdn.hpp:59-71`).
- Conv history: the reference `gdn_conv_step` keeps `(d_conv - 1, channels)`, oldest first. The fused, native and
  verify kernels keep `[channel][3]` (`fused_gdn.hpp:19-21`, `native_gdn_preprocess.hpp:10-15`). A session uses one
  family throughout.

### Geometry

`GdnShapes` (`gdn.hpp:38-42`): `S` 128, `h_k` 16, `h_v` 48 (3 value heads per key head), so 2 x 16 x 128 + 48 x 128 =
10,240 conv channels and d_conv 4 (`gdn.hpp:7-11`). `h_v % h_k == 0` is required everywhere. The fused and native
kernels hard-code S = 128. The recurrent state of all 36 GDN layers is 113 MB (`verify_kernels.hpp:12`).

## Main entry points

### Reference kernels (`gdn.cu`)

| Function | Kernel / launch | Notes |
| --- | --- | --- |
| `gdn_step` (`gdn.cu:153`) | `gdn_step_kernel` (`:43`), grid `(S/32, h_v)`, 32 threads | One thread per `(h, j)` column; q must arrive scaled by `1/sqrt(S)` (`gdn.hpp:47-48`). `MAX_H` = 1 head per block: the comment at `:16-32` records that 8 heads per block left 24 of 48 SMs idle, measured 0.28 ms per GDN block (14% of a 71.5 ms token) |
| `gdn_conv_step` (`:174`) | `gdn_conv_kernel` (`:99`), 256 threads | `ggml_ssm_conv` with the carried state prepended; updates the state |
| `gdn_l2_norm` (`:189`) | `gdn_l2_kernel` (`:118`), one warp per row, cols <= 1,024 | `x / sqrt(sum(x^2) + eps)` (sum, not mean; `gdn.hpp:75-80`) |
| `gdn_beta_gate` (`:222`) | 128 threads | `beta = sigmoid(beta)`; the step kernel does not do it and the layer must (`gdn.hpp:82-88`) |
| `gdn_out_norm` (`:235`) | `gdn_out_norm_kernel` (`:133`), one warp per head | per-head RMS, `ssm_norm`, `sigmoid(z)` |
| `gdn_gate` (`elementwise.cu:142`) | 256 threads | `softplus(alpha + dt) * ssm_a`, softplus with ggml's `x > 20` branch |

### Fused kernels (`fused_gdn.cu`)

| Function | Kernel / launch | Replaces |
| --- | --- | --- |
| `fused_gdn_conv_l2` (`fused_gdn.cu:127`) | `gdn_conv_l2_kernel` (`:73`), one 128-thread block per 128-channel head | conv + SiLU over all channels, then the L2 norm of the first `qk_heads` (= 2 h_k: q then k) heads; replaces three launches. Needs `channels % 128 == 0` |
| `fused_gdn_ab` (`:138`) | `gdn_ab_kernel` (`:95`), one warp per row, 8 rows per 256-thread block | both BF16 `(h_v, n_embd)` projections against the F32 activation, with the softplus gate and sigmoid beta epilogues; replaces two MMVF and two elementwise launches |
| `fused_gdn_step_norm` (`:150`) | `gdn_step_norm_kernel` (`:16`), grid `h_v`, block `(128, 4)` | One block owns a whole head: 512 threads = (column, row group of 32), each holding 32 state rows in registers; a row's 128 columns are one 512 B load. Step, readout scale and output norm with `sigmoid(z)` in one kernel. `fused_gdn.hpp:3-8` contrasts it with the pinned step's strided loads (4 useful bytes per 32 B sector, 47.6 us per layer for 6.3 MB of state traffic) |

### Pinned adapters (`native_gdn*.cu`, `--use_fast_math`)

| Function | Notes |
| --- | --- |
| `native_gdn_step` (`native_gdn.cu:96`) | llama.cpp `gated_delta_net` arithmetic with Strata's state layout: grid `(h_v, 1, S/4)`, block `(32, 4)`. q must not be pre-scaled; the readout is scaled by `1/sqrt(128)`. Throws on bad spans or geometry |
| `native_gdn_conv_silu` (`native_gdn_preprocess.cu:142`) | conv (with ggml's zero-bias add) and `expf` SiLU; writes the raw and the SiLU output |
| `native_gdn_l2_norm` (`:160`) | `scale(rms_norm(x, eps / 128), 1 / sqrt(128))`, ggml's L2 norm as its CUDA graph computes it |
| `native_gdn_beta_gate` (`:166`), `native_gdn_gate` (`:172`) | sigmoid; softplus with threshold 20 |
| `native_gdn_out_norm` (`:185`) | fused RMSNorm x gamma, then `sigmoid(z) * mul` |

`native_gdn_set_enabled` / `native_gdn_enabled` (`native_gdn.cu:93-94`) is the switch the layer reads.

### Verify window (`verify_kernels.cu`)

The recurrent state cannot be overwritten by a window that might be rejected, so the GDN verify kernels come in two
halves (`verify_kernels.hpp:4-12`). The verify half runs all T tokens in order from the stored state, writes only the
per-token outputs and leaves the state untouched. The commit half, launched once acceptance is known, replays the first
`*n_keep` tokens and writes the state. No snapshot of the 113 MB state is kept, and rejected drafts never touch it.

| Function | Kernel / launch | Bitwise equal to |
| --- | --- | --- |
| `gdn_conv_l2_multi` (`verify_kernels.cu:439`) | `gdn_conv_l2_multi_kernel` (`:25`), grid `(channels/128, n_tok)`, 128 threads; reads the history, does not write it | `fused_gdn_conv_l2` per token |
| `gdn_conv_commit` (`:450`) | `gdn_conv_commit_kernel` (`:53`), 256 threads | history <- last 3 of `[history, qkv_0 .. qkv_{n_keep-1}]` |
| `gdn_ab_multi` (`:456`) | `gdn_ab_multi_kernel<MAX_T>` (`:71`), 64 threads, `MAX_T` 4 or 8 | `fused_gdn_ab` per column |
| `gdn_step_norm_multi` (`:473`) | verify: `gdn_step_norm_multi_kernel` (`:160`), grid `h_v`, block `(128, 4)`; a commit with no outputs: `gdn_step_commit_kernel` (`:118`), grid `(h_v, 4)`, block `(32, 4)` | `fused_gdn_step_norm` |

`n_keep == nullptr` means verify (state read only); a non-null `n_keep` (device memory) means commit.
`STRATA_GDN_COMMIT_SPLIT=0` keeps the full kernel for a commit that writes no outputs (`t_out_begin >= n_tok`,
`verify_kernels.cu:481-491`). T is at most `kVerifyMaxT` = 8.

## Control and data flow

Decode (`gdn_layer`, `src/core/layer.cpp:223-354`):

```
attn_qkv GEMV --> fused_gdn_conv_l2                        when fused_pre (layer.cpp:247): fused GDN on, native GDN on,
                 | native_gdn_conv_silu -> native_gdn_l2_norm x2     native BF16 projections, d_conv 4
                 | gdn_conv_step -> gdn_l2_norm x2 -> scale_inplace(q, 1/sqrt(S))
alpha / beta  --> fused_gdn_ab | BF16 projections -> native_gdn_beta_gate + native_gdn_gate
                                                  | gdn_beta_gate + gdn_gate
attn_gate GEMV (z)
step + norm   --> fused_gdn_step_norm                      when native GDN on and S = 128 (layer.cpp:306)
                 | native_gdn_step -> native_gdn_out_norm | gdn_step -> gdn_out_norm
ssm_out GEMV
```

The verify window calls `gdn_conv_l2_multi`, `gdn_ab_multi` and `gdn_step_norm_multi` (`src/core/verify.cpp:726-748`)
and the commit `gdn_conv_commit` + `gdn_step_norm_multi` with `n_keep` (`:1293-1294`, `:1844-1845`). The kernels
depend only on the CUDA runtime.

## Configuration

- CLI: `--native-gdn` (`native_gdn_set_enabled`), `--native-bf16` (needed for `fused_gdn_ab`), `--no-fused-gdn`
  (`layer_set_fused_gdn`, `src/program/generate.cpp:1521`, `:2282`). `--native SHARD1` turns on the first two.
- Environment: `STRATA_GDN_COMMIT_SPLIT` (default on).
- Build: the `native_gdn*.cu` files compile with `--use_fast_math` on CUDA (`CMakeLists.txt:365-379`).

## Model- and platform-specific behaviour

- Every supported model (Flash-Next, the Coder, Swift 1.5, Unsloth) has the same GDN geometry and the sigmoid gate;
  only the projection types differ between packs.
- No architecture-specific branches in these files; they compile unchanged for HIP. `gpu_stamp` (the verify
  profiler in the same file) reads `%globaltimer` on CUDA and `wall_clock64()` on HIP (x10 on RDNA, x40 on gfx906;
  `verify_kernels.cu:667-676`).

## Tests

- `gdn_parity` (`src/kernels/gdn_parity.cpp`): the reference kernels against `ref/gdn.py`'s properties (decay before
  update, modulo head pairing, L2 vs RMS norm) and the fused / native / verify kernels against each other.
- Oracle-linked (sources not in the snapshot): `native_gdn_parity`, `native_gdn_preprocess_parity`,
  `native_gdn_pipeline_parity`.

## Gotchas

- `gdn_l2_norm` divides by `sqrt(sum + eps)`, while `rms_norm_weighted` divides by `sqrt(mean + eps)` and multiplies
  by w; they differ by `sqrt(S)` = 11.3x and the tests assert they are distinguishable (`gdn.hpp:75-80`,
  `elementwise.hpp:73-77`).
- `gdn_step` expects q already scaled by `1/sqrt(S)`; `native_gdn_step` and `fused_gdn_step_norm` expect it unscaled.
  The layer applies `scale_inplace` only when native GDN is off (`layer.cpp:276`).
- `gdn.hpp:7` says "48 layers are GDN, 12 are QSA"; `include/strata/core/graph.hpp:45` and `session.hpp:7-8` give
  36 GDN and 12 QSA of 48 layers. The header line is stale.
- `gdn_step` refuses S > 128 (its staged rows are 128 wide).

## In UnbiasedStrata

Changed in the fork (commit e41dfc7): `gdn.hpp` / `gdn.cu`, `fused_gdn.*`, `native_gdn_preprocess.*` and
`verify_kernels.*` gained a `GdnGate` argument (`Sigmoid` or `Silu`) on every output-norm kernel, so Qwen3.6-35B-A3B's
SiLU-gated DeltaNet (h_v 32) runs on the same kernels; the sigmoid arithmetic is bitwise unchanged. `native_gdn.cu`
is unchanged. See the fork's [gdn.md](../../../codebase/kernels/cuda/gdn.md).

## See also

- [attention-qsa.md](attention-qsa.md), [sampling-and-verify.md](sampling-and-verify.md) (the other verify helpers),
  [norms-rope-elementwise.md](norms-rope-elementwise.md) (`gdn_gate`, `scale_inplace`)
- [Prefill](../../prefill/README.md) (the prompt path's GDN kernels)
