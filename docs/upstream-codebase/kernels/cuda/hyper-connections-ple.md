[Upstream docs](../../README.md) › [Kernels](../README.md) › [CUDA](README.md) › Hyper-connections, PLE, control vectors

# Hyper-connections, the PLE block and control vectors

## Purpose

The model's residual stream is not one vector but a stack of `hc` = 4 streams of 2,560 values ("hyper-connections").
Before each mixer and each MoE block, a read collapses the stack into one input vector (`mixed`) plus a per-stream
injection weight; after the block, a write adds the block output back into every stream with that weight. This page
covers those kernels, the PLE ("per-layer embedding") n-gram block that layer 1 adds to the stack, and control vectors,
which edit the stack after a layer's write.

Files: `src/kernels/cuda/gr.cu`, `fused_gr.cu`, `native_gr_norm.cu`, `native_gr_postops.cu`, `ple.cu`,
`native_ple_postops.cu`, `cvec.cu`; headers `include/strata/kernels/gr.hpp`, `fused_gr.hpp`, `native_gr_norm.hpp`,
`native_gr_postops.hpp`, `ple.hpp`, `native_ple_postops.hpp`, `cvec.hpp`, and `ngram.hpp` (host side of PLE).

## Hyper-connection read and write

Geometry `GrShapes` (`gr.hpp:41-46`): n_embd 2,560, hc 4, hc_lr 320. The weights are BF16 in manifest layout
(`gr.hpp:81-90`): `w_down[k * hc*n_embd + i]` (hc_lr rows), `w_up[i * hc_lr + k]`, `w_inject` (hc rows, null for the
final mixer before the head). `w_norm` is stored as `(1 + w)` by the converter. The arithmetic (`gr.hpp:86-97`,
`fused_gr.hpp:6-13`):

```
xn       = R * w_norm * rsqrt(mean_d(R_c^2) + eps)            per stream c
lo       = silu((w_down . xn) / hc)                           the division by hc is inside the silu
mixed[d] = mean_c( xn[c,d] * sigmoid(w_up[c*n_embd + d] . lo) )
inject   = w_inject . xn                                      (absent for the final mixer)
gr_write: R_out[c,d] = R[c,d] + block_out[d] * 2 * sigmoid(inject[c] / hc)
```

| Function | What |
| --- | --- |
| `gr_read` (`gr.cu:344`) | The unfused read. Default: `gr_norm_kernel` (hc blocks), `gr_down_kernel` (hc_lr blocks), `gr_gate_kernel` (warp per row), `gr_mean_kernel`, `gr_inject_kernel`, with activations rounded to BF16. `gr_set_fp32_activations(true)` (`--gr-fp32-activations`) keeps them F32. `gr_set_native_mmvf(true)` (`--gr-native-mmvf`) runs the pinned sequence instead: `native_gr_rms_norm_weighted`, `bf16_gemv_fp32_mmvf` (down), `native_gr_down_silu`, MMVF (up), `native_gr_pre_gated`, MMVF (inject) (`gr.cu:363-370`) |
| `gr_write` (`gr.cu:413`) | `gr_write_kernel`, or `native_gr_post` when native MMVF is on |
| `gr_workspace_init` (`gr.cu:304`) | carves a caller-owned device buffer into xn, xq, lq, gated, lo and records its size; `gr_read` checks it |
| `fused_gr_read` (`fused_gr.cu:1168`) | Two kernels: a down kernel (41 blocks: 40 x 8 down rows plus the inject rows) that also applies the previous half's pending write on the fly (`apply`, `bo_prev`, `inj_prev`), then norm, down, inject; an up kernel (80 blocks of 32 columns x 4 streams) that writes R in place and computes `mixed`. F32 activations, BF16 weights; another summation order than the native MMVF read. Geometry hard-coded: `fused_gr_supported` (`:1164`) accepts only 2560 / 4 / 320 |
| `fused_gr_read_multi` (`fused_gr.cu:1056`) | Up to `kFusedGrMaxT` = 8 tokens sharing the weights (verify window, MTP drafter); each token's output is bitwise `fused_gr_read`. Variants: plain, split (norm per token and stream) and staged (the down projection's activations staged by `cp.async` on sm_80+, plain copies below). The down kernel's dynamic shared memory (`n_tok x TILEV` floats; TILEV 1,280 on sm_75 and RDNA, 2,560 otherwise) is opted in per device; tokens are sliced when it does not fit (`down_chunk`, `:687-765`) |
| `fused_gr_check` (`fused_gr.cu:1336`), `fused_gr_variant` (`:1326`) | Run once per card (`src/core/verify.cpp:311`): runs split and staged against plain on random data (`fused_gr_selftest`, `:1192`) and keeps the newest variant that agrees bit for bit; prints the choice. On an unchecked card the plain read runs unless `STRATA_HC_SPLIT` names a variant |

Pinned adapters (`--use_fast_math`): `native_gr_rms_norm_weighted` (`native_gr_norm.cu:85`; 256 threads below 1,024
columns, else 1,024, the pinned XOR tree), `native_gr_down_silu` (`native_gr_postops.cu:91`; scale by 1/hc, then
SiLU), `native_gr_pre_gated` (`:97`; sigmoid gate and mean over streams) and `native_gr_post` (`:107`; scale,
sigmoid, x2, add).

The layer uses `fused_gr_read` when `--gr-native-mmvf` is on and `--no-fused-gr`, `--gpu-stages`, layer dumps and
stage timing are all off (`src/program/generate.cpp:2284-2285`); `--native SHARD1` turns `--gr-native-mmvf` on.

### Platform branches in `fused_gr.cu`

- `cp.async` in the staged variant only on sm_80+ and not on HIP (`fused_gr.cu:563`).
- The down tile is 1,280 floats on sm_75 (all 8 tokens fit one 40 KiB launch) and on RDNA (gfx1100's 64 KiB LDS),
  2,560 on gfx906 and other CUDA cards (`:700-721`). Below sm_70 the per-block limit, not the opt-in, sizes the
  token slices (`:731-744`).
- `STRATA_GR_V3=1`: another two-kernel multi read (another summation order, opt-in); a 64 KiB opt-in card such as
  Turing is refused because its two-half split failed `gr_parity` (#375, `:1077-1100`).
- gfx906 only: `STRATA_GR_FAST` (default 1) latency-hidden norm/up kernels with the same sums, and `STRATA_GR_SPLIT=1`
  a K-split down that is about 3% faster per verify window but no longer bitwise equal to T single-token reads
  (`:1046-1153`).

## PLE block

PLE is the model's n-gram embedding layer (`ple.layers = [1]`, `ple.hpp:44`). The host side (`ngram.hpp`: the 64-bit
XOR n-gram hash, the IQ4_NL / FP8 row decoders, `PleTable`, the SSD reader) is CPU code in `src/kernels/ngram.cpp`
and `src/ngram/ple_reader.cpp` (see [CPU kernels](../cpu/README.md)). It gathers sixteen rows of 160 values per token
into `emb` (2,560). The GPU half computes (`ple.hpp:3-13`):

```
key    = grouped_norm(ple_key @ emb)          hc_dim = 10,240, per-stream RMS then a [n_embd, hc] gamma
query  = grouped_norm(hidden, ple_norm_query)
value  = ple_value @ emb                      n_embd
s[c]   = sum_d(key[c] * query[c]) / sqrt(n_embd)
gate   = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))
gated  = value * gate[c]        normalized = grouped_norm(gated, ple_norm_conv)    <- the conv input
conv   = 4-tap, dilation 3 conv over the 9-row history of normalized rows
out    = hidden + gated + silu(conv)          written in place into the residual stack
```

| Function | Notes |
| --- | --- |
| `ple_block` (`ple.cu:228`) | Key projection in one of three forms: `key_bf16` (BF16 MMVF; the IQ model files keep `ple_key` in BF16), a native GGUF key (`native_quantize_q8_1` + `native_mmvq`, types Q2_0 / IQ3_XXS / IQ4_XS / Q8_0; `--native-ple-key`), or the canonical S2 key (`quantize_q8_0` + `s2_gemv_q8`). Value: `bf16_gemv_fp32_mmvf` with `ple_set_native_bf16` (`--native-bf16-extra`), otherwise an internal BF16 GEMV. Then the canonical post-ops (`gnorm_kernel`, `gate_kernel`, `bcast_kernel`, `conv_kernel`, `add3_kernel`) or `native_ple_postops`. No allocation, no sync; scratch from `ple_block_scratch_bytes` (`:219`) |
| `ple_block_projected` (`ple.cu:369`) | the same after the projections, with the native post-ops |
| `ple_history_advance` (`ple.cu:204`) | shift the 9-row normalized history and append one row; one thread per channel |
| `native_ple_postops` (`native_ple_postops.cu:175`) | `--native-ple-postops`: pinned arithmetic after the projections; fixed geometry 2,560 x 4, history 9 (`:35`) |
| `native_ple_postops_batch` (`:205`) | T tokens for the prompt path, element for element T single calls with the history advanced |

`ple_set_native_postops` is applied at `src/program/generate.cpp:2313`.

## Control vectors (`cvec.hpp`, `cvec.cu`)

The engine side of llama.cpp's `llama_adapter_cvec`, with the `--cvec-mode` / `--cvec-dir` extension of the
`experimental-speed-projection` package (`cvec.hpp:1-16`). After layer l's FFN write, every residual stream h of every
token becomes `h - s_l (h . v_l) v_l` (project) or `h + d_l` (add), for l in [first, last].

- `cvec_upload` (`cvec.cu:101`) builds the per-layer tables from the GGUF directions; `cvec_replicate` (`:135`) copies
  them to the current device (one table set per device, up to 64).
- `cvec_set_enabled` (`:137`) flips a device flag between requests (it synchronises), so captured graphs are the same
  with the vector on or off. With the flag off the result is bit-identical to no vector.
- `cvec_apply` (`:154`): one 256-thread block per (stream, token). With `write` it first applies the pending
  `R += bo * 2 sigmoid(inj / hc)`, so it can replace the folded write of the fused read. `MAXK` = 16 limits n_embd to
  4,096 (`:12`).

Callers: `src/program/generate.cpp:1139` (upload), `:6598` (per-request switch); `src/core/layer.cpp:1330-1336`,
`src/core/verify.cpp:1097-1100`, `src/prefill/prefill.cpp:2908`.

## Control and data flow

```
block_layer_pre (src/core/layer.cpp:1170):
  [layer 1] ple_block(emb, R) -> R in place ; ple_history_advance
  fused_gr_read(attn half, apply = previous write)  | gr_read      -> mixed, inject   (layer.cpp:1253 / :1255)
  mixer (GDN or QSA)
  fused_gr_read(ffn half, apply = mixer write)       | gr_write + gr_read           (layer.cpp:1276 / :1278)
  router, doorbell, experts ...
block_layer_post (layer.cpp:1319): moe_combine -> [gr_write] -> cvec_apply on covered layers
```

The verify window uses `fused_gr_read_multi` (`src/core/verify.cpp:693`) and the MTP drafter too
(`src/core/mtp.cpp:510`, `:571`, `:620`). The prompt path has its own `gr_write` and PLE batch calls
(`src/prefill/kernels.cu:884`, `src/prefill/prefill.cpp:1956`).

## Configuration

- CLI: `--gr-fp32-activations`, `--gr-native-mmvf`, `--no-fused-gr`, `--native-ple-postops`, `--native-ple-key`,
  `--native-bf16-extra`, `--no-ple`, `--ple-gguf`, `--ple-io`, `--control-vector`, `--control-vector-scaled`,
  `--control-vector-layer-range`, `--cvec-mode`, `--cvec-dir` (control-vector help at `src/program/generate.cpp:636-640`,
  `--cvec-mode` / `--cvec-dir` parsing at `:1458-1468`).
- Environment: `STRATA_HC_SPLIT` (unset or `2` staged if the check passes, `1` at most split, `0` plain;
  `fused_gr.cu:821-828`), `STRATA_GR_V3`, `STRATA_GR_DOWN_MAX4` (4-token accumulators for launches of <= 4 tokens,
  same bits); gfx906: `STRATA_GR_FAST`, `STRATA_GR_SPLIT`.

## Model-specific behaviour

All supported model variants share hc 4 / 2,560 / 320 and one PLE layer. The PLE key's type depends on the pack: the
canonical S2 pack uses the S2 key, GGUF packs with `--native` use the native key, and the IQ model files (IQ2_XS,
IQ3_XXS, IQ3_S) keep it in BF16 (`ple.hpp:71-73`). Where the PLE table lives (second GGUF shard of the ISTA files and
of Unsloth's UD-Q4_K_XL, first shard of Swift's) is resolved by `generate.cpp:1787-1793`.

## Tests

`gr_parity` (canonical, fused, multi and the variants against each other, with `STRATA_HC_SPLIT` variants),
`cvec_parity` (project and add, on and off), `ple_parity` (hash, IQ4_NL table, the block against a captured ggml graph;
needs the real artifacts), `ple_fp8_parity`, `ple_q5_parity`, `fused_gr_bench` (benchmark). HIP: `tests/hip/ple_iq4.cpp`.
Oracle-linked (sources not in the snapshot): `native_gr_norm_parity`, `native_gr_postops_parity`,
`native_ple_postops_parity`.

## Gotchas

- `GrWorkspace` must be device memory; a host buffer gives an illegal access with a truncated address
  (`gr.hpp:51-56`).
- `inj_prev` and `inject_out` of `FusedGrArgs` must be different buffers (`fused_gr.hpp:16-17`).
- `ple.hpp:15-20`: the conv input is `normalized`, not the gated values as `ref/ngram.py`'s docstring says.
- `ple_block`'s outputs must not overlap its scratch; it throws before launching otherwise.

## In UnbiasedStrata

Kept unchanged: none of these files differs between the import (12a7825) and the fork's HEAD. The fork's Qwen3.6
support does not use them (that model has a plain residual and no PLE); Flash-Next still does. See the fork's
[norms-rope-elementwise.md](../../../codebase/kernels/cuda/norms-rope-elementwise.md).

## See also

- [gdn.md](gdn.md), [attention-qsa.md](attention-qsa.md), [gemv-and-quant.md](gemv-and-quant.md) (BF16 MMVF),
  [norms-rope-elementwise.md](norms-rope-elementwise.md), [bits-and-arch-shims.md](bits-and-arch-shims.md)
- [CPU kernels](../cpu/README.md) (n-gram hash and table reader)
