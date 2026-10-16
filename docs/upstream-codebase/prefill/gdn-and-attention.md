[Upstream docs](../README.md) › [Prefill](README.md) › GDN and attention

# The mixer half: hyper-connections, Gated DeltaNet and QSA attention for prompts

## Purpose

Half 0 of every layer is a mixer: on 36 of the 48 layers a Gated DeltaNet (GDN, linear attention with a recurrent
state), on the other 12 QSA (sparse full attention whose indexer picks which earlier blocks each query reads). Both
halves are wrapped by the hyper-connection read and write of Flash-Next's four residual streams. This page covers the
kernels in `kernels.cu` and the order in which `prefill.cpp` calls them; the QSA attention, indexer and KV-format
kernels themselves live in `src/kernels/` (kernels area).

Files: [src/prefill/kernels.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/kernels.cu),
[include/strata/prefill/kernels.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/prefill/kernels.hpp),
[src/prefill/prefill.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/prefill.cpp)
(`:1976-2234`, `:2888-2908`).

All kernels take `T` tokens at a time, row-major with the token as the slow index; where a step is a recurrence in
time (GDN conv and state, PLE conv) the kernel walks the chunk's tokens in order inside one launch
(`kernels.hpp:1-5`). The arithmetic mirrors the decode kernels, batched.

## Hyper-connection kernels (`hc` 4, width 2,560, low rank 320)

| Function | Kernel line | What it computes |
| --- | ---: | --- |
| `gr_broadcast` | 199 | `R[t, c, :] = emb[t, :]` for the four streams (after the embedding) |
| `gr_norm` | 66 | per stream RMS norm times `w_norm` into FP32 `xn` and its BF16 image (only with `STRATA_GR_UNFUSED=1`) |
| `gr_norm_rs` | 85 | F-1: only the row scales `rs[t*4 + c]` and the BF16 image; no FP32 copy of the normalized rows |
| `gr_silu` | 161 | `lo16 = bf16(silu(lo / hc))` |
| `gr_mix_r` | 103 | `mixed[t, d] = mean_c xn[t, c, d] * sigmoid(gated[t, c, d])`, with `xn` recomputed from `R`, `rs`, `w_norm` (same bits as `gr_norm`); writes FP32, BF16 and FP16 images |
| `gr_mix` | 171 | the same from an FP32 `xn` (unfused arm) |
| `gr_write` | 192 | `R[t, c, :] += bo[t, :] * 2 sigmoid(inj[t, c] / hc)` |
| `gr_write_norm_rs` | 129 | F-2: `gr_write` and the next half's `gr_norm_rs` in one pass, the same bits, without reading `R` back |

Sequence per half (`src/prefill/prefill.cpp:1977-1994`): norm (unless the previous write already did), BF16 GEMMs
`hc_{attn,ffn}_down` (2,560 x 4 -> 320), `gr_silu`, `hc_*_up` (320 -> 10,240), `hc_*_inject` (10,240 -> 4), then
`gr_mix_r`. With `STRATA_PREFILL_BF16X2` the BF16 images carry a second low-part image and each GEMM runs twice
(`bf_lo`, `src/prefill/kernels.cu:36-38`).

## Gated DeltaNet (state 128, 16 key heads, 48 value heads, 10,240 conv channels, 4 taps)

Call order (`src/prefill/prefill.cpp:1996-2018`):

1. `attn_qkv` (10,240 outputs) and `attn_gate` (`z`, 6,144) as FP16 GEMMs of the native blocks; `ssm_alpha` and
   `ssm_beta` (48 each) as BF16 GEMMs into one `[T, 96]` buffer.
2. `gdn_gates` (`kernels.cu:207`): `gate = softplus(alpha + dt_bias) * ssm_a`, `beta = sigmoid(beta_raw)`.
3. `gdn_conv` (`:896-906`): the 4-tap causal conv with SiLU over the chunk, reading and updating the history
   (`[10240][3]` floats, the last three inputs). For chunks above 64 tokens the tiled variant
   (`gdn_conv_tiled_kernel`, `:236`, C-3: tiles of 64 tokens read their three predecessors, then
   `gdn_conv_hist_kernel` writes the history) - the same expression per element, the same bits.
   `STRATA_GDN_CONV_SERIAL` keeps the serial walk. Then `gdn_l2_kernel` L2-normalizes the q and k heads.
4. `gdn_recurrence` (`:908-926`): the delta rule per value head, state kept in registers across the chunk:
   `S = g S + k (beta (v - g S^T k))`, `o = S^T q / sqrt(128)`, with `g = exp(gate)`; value head `h` uses key head
   `h % 16`. Output `y16 = fp16(rmsnorm(o) * gamma * sigmoid(z))`, the input of the out projection.
5. `ssm_out` (FP16 GEMM) into `bo`.

The state is `[128 k rows][48 heads][128 v cols]` FP32 followed by the conv history, in
`ss.gdn_state + (gdn_index - ss.gdn_ord0) * gdn_floats` (`src/prefill/prefill.cpp:2005-2006`), the layout the decode
kernels read.

Recurrence variants (all the same arithmetic in the same order per value head and column, so the same bits; checked
by [`gdn_rec_parity`](tests.md)):

| Kernel | Line | Used when |
| --- | ---: | --- |
| `gdn_rec_kernel` | 278 | `STRATA_GDN_REC_HEADS` (one block per head, A/B) |
| `gdn_rec_cols_kernel` | 338 | `STRATA_GDN_PIPELINE=0` (D-2: value columns split over 4 blocks per head, output norm in its own kernel) |
| `gdn_rec_cols_pipe_kernel` | 381 | default where the key-head kernel does not pay, and always on HIP (next token's inputs loaded while this one computes) |
| `gdn_rec_kh_kernel` | 479 | default on CUDA sm_80+ when the card holds all 64 blocks at once (`gdn_keyhead_ok`, `:579-597`); one thread serves the three value heads that share a key head, inputs staged 8 tokens at a time with `cp.async`; `STRATA_GDN_KEYHEAD=0` turns it off |
| `gdn_out_norm_kernel` | 600 | the output norm after the column-split kernels |

The `gdn_keyhead_ok` comment records the measurement: 1.40-1.42x on a 4080 SUPER and 1.28-1.29x on a 3090 at 64 SMs
or more, a draw at 48 to 63 SMs.

## QSA attention (24 query heads, 2 KV heads, head dim 256)

Call order (`src/prefill/prefill.cpp:2019-2234`):

1. FP16 GEMMs `attn_k`, `attn_v` (512 each), `attn_q` (12,288: query and gate per head); BF16 GEMMs of the indexer's
   `k_proj` (128) and `q_proj` (512 = 4 heads x 128).
2. `rms_rows` (`kernels.cu:717`) with `attn_k_norm`, then `rope` (`:994-1013`): NEOX rotary on the first 64 dims of
   each head at positions `pos0 + t`, under the session's rope scaling (`rope_scaling.hpp`; `STRATA_ROPE_TABLE=1`
   reads angles from a float64 table, the `<true>` kernel variant).
3. KV streaming (`kv_mode == 1`): the layer's cells `[0, p0)` are staged from the host copy into the staging pool
   (`kv_stage_from_host`).
4. The append (`:2061-2084`): K8V4 hybrid = INT8 K unrotated and V rotated (FWHT-256) Q4_0; otherwise optional FWHT
   rotation of K and V, then `kv_append_q4` (Q4_0) or `kv_append` (`kernels.cu:766-836`: FP16 pools, or INT8 codes
   with an FP16 scale per 64 values, the decode append's arithmetic). Only blocks resident in VRAM (`table >= 0`) are
   written to the pools; the host copy and the staging pool get every cell. On HIP, `STRATA_KV_HOST_DMA=1` writes
   the host copy with one DMA from the staging pool instead.
5. `split_q` (`:750`) separates the query from its gate; `rms_rows` with `attn_q_norm`; `rope`; FWHT when rotated.
   The indexer query: `rms_rows` with `indexer.q_norm`, `rope` on 128-dim heads.
6. Indexer append (C-2): the chunk's keys in three launches (`native_qsa_indexer_append_batch`;
   `STRATA_INDEXER_PER_TOKEN=1` appends token by token).
7. Block selection in batches of 256 queries: `qsa_block_scores_tc` (tensor cores, 3xTF32, FP32-level but not
   bitwise; `STRATA_SELECT_OLD=1` the warp kernel), then `qsa_block_topk` into `sel_ids` (`:2112-2128`).
8. Prompt attention (D-1): `qsa_prompt_attn_batch` over the whole chunk on tensor cores, one block per (query, KV
   head); Q4_0 KV or `STRATA_PROMPT_ATTN_OLD=1` use the decode kernel 32 queries at a time
   (`qsa_decode_attn_batch`) (`:2214-2229`). A rotated output is rotated back.
9. `gate_attn` (`kernels.cu:756`): `attn * sigmoid(gate)` to FP16, then `attn_output` (FP16 GEMM) into `bo`.

The selection width and block count come from `qsa_selection_width(kTopkMaxCells, s)` and
`max_cells / idx_block + 2` (`src/prefill/prefill.cpp:865-866`). Debug measurements: `STRATA_SEL_OVERLAP` (how much
neighbouring queries' selections share), `STRATA_IDX_FP16_CHECK` (would FP16 pooled keys select the same cells),
`STRATA_QSA_DUMP=<file>` (selected cells for `tools/qsa_locality.py`).

## The PLE block (layer 1)

Not in `kernels.cu`: the per-layer embedding block of layer 1 uses `native_ple_postops_batch` / `ple_block` from
`src/kernels/` (see [pipeline.md](pipeline.md) for the batching and `STRATA_PLE_BATCH`). It runs before the two halves
of layer 1 and reads the n-gram rows gathered for the chunk.

## Other helpers in `kernels.cu`

`route` (router top-10), `blob_dequant` / `blob_dequant_f16` (Strata Q2_0 blob to BF16 / FP16, the FP16 expert
path), `swiglu_interleaved` / `swiglu_pair` (saturated at +-65504 so a massive activation cannot become inf and then
NaN, `:41-44`), `gather_rows16`, `moe_combine`, `copy_i32` / `copy_f32_wide` (copies as kernels, so mapped host memory
is read over PCIe instead of queuing on the copy engine), `to_f16`, `to_bf16`, `round_f16`. See [moe.md](moe.md).

## Platforms

`gdn_rec_kh_kernel` is CUDA only (`#if !defined(__HIPCC__)` around its launch); HIP runs the pipelined column kernel.
The prompt attention and selection kernels have their own HIP variants in the kernels area.

## Gotchas

- `rope` exits the process on an invalid rope-scaling config (`kernels.cu:997-1000`); the engine validates the config
  at startup, so this only fires for a caller that bypassed it.
- The QSA layer index used for the state is counted from layer 0 even on a later layer-split stage
  (`src/prefill/prefill.cpp:1723-1724`).

## In UnbiasedStrata

Kept and changed (commit e41dfc7): a `GdnGeom` replaces the compile-time GDN widths so the kernels also run Qwen3.6
(32 value / 16 key heads, 8,192 channels, a SiLU output gate), plain-block kernels (`rms_norm_images`,
`residual_add`) were added for a single residual stream, and the attention half has a dense branch without the
indexer. See [../../codebase/prefill/gdn-and-attention.md](../../codebase/prefill/gdn-and-attention.md).

## See also

[pipeline.md](pipeline.md), [moe.md](moe.md), [tests.md](tests.md), the kernels area for `qsa_prompt_attn`,
`qsa_select`, `native_qsa_indexer`, `kv_q4`, `kv_stream` and `ple`.
