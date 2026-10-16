[Upstream docs](../../README.md) › [Kernels](../README.md) › [CUDA](README.md) › Attention (QSA)

# Attention: the QSA cache, indexer, selection and attention kernels

## Purpose

QSA is the model's full-attention layer type (12 of the 48 layers). Each query attends only to a selected subset of
the cached cells: a small "indexer" scores 4-cell blocks of the context and a top-k picks up to 2,051 cells. These
kernels implement everything in a QSA layer between its projections (`attn_q`, `attn_k`, `attn_v`, `attn_output`,
`indexer.q_proj`, `indexer.k_proj` are GEMVs from [gemv-and-quant.md](gemv-and-quant.md) and
[moe-and-experts.md](moe-and-experts.md)):

- the paged KV cache (FP16 here; INT8, Q4_0, K8V4 and streaming are on [kv-cache.md](kv-cache.md));
- the indexer: raw-key tail, pooled block keys, per-cell or per-block scores, top-k cell selection;
- attention over the selected cells, for decode, the verify window and the prompt;
- the output gate `attn * sigmoid(gate)`.

Files: `src/kernels/cuda/qsa.cu`, `qsa_select.cu`, `qsa_decode_attn.cu`, `qsa_prompt_attn.cu`, `native_qsa.cu`,
`native_qsa_indexer.cu`, `native_qsa_score.cu`, `native_flash_attn.cu`; headers `include/strata/kernels/qsa.hpp`,
`qsa_select.hpp`, `qsa_decode_attn.hpp`, `qsa_prompt_attn.hpp`, `native_qsa.hpp`, `native_qsa_indexer.hpp`,
`native_qsa_score.hpp`, `native_flash_attn.hpp`.

## Geometry and the step record

`QsaShapes` (`include/strata/kernels/qsa.hpp:96-106`) carries `n_head`, `n_head_kv`, `head_dim`, `n_rot`,
`idx_n_head`, `idx_dim`, `idx_block`, `idx_top_k` and `page_size`. `qsa_real_shapes()` (`qsa.hpp:109-117`) gives the
model's values: 24 query heads, 2 KV heads, head_dim 256, n_rot 64 (NEOX pairs), indexer 4 heads x 128 dims, one
pooled key per 4 cells, top-k budget 2,048 cells, page size 4 cells (one indexer block, which is also the KV-streaming
granule). `qsa_rms_eps()` is 1e-6 and `qsa_freq_base()` 1e7 (`qsa.hpp:120-125`).

- `qsa_selection_width(n_kv, s)` (`qsa.hpp:133-136`) = `min(n_kv, idx_top_k + idx_block - 1)` = min(n_kv, 2,051).
  The extra 3 cells are the incomplete tail block. Below the bound the selection is the identity.
- `QsaStep` (`qsa.hpp:156-162`): a 4 x int32 device record `{pos, n_kv = pos + 1, n_bid = n_kv / 4, width}`. One
  writer computes all four (`qsa_step_fill`, `qsa.cu:699`) and every capturable kernel reads its per-token counts
  from it. The header comment (`qsa.hpp:27-58`, `:138-155`) explains why: a CUDA graph that captured a host-scalar
  `pos` would write every token to cell 0 and attend over one cell, with plausible output.
- The query-to-KV head map is integer division `q / (n_head / n_head_kv)`, not modulo (`qsa.hpp:98`, `:293-294`).
- `validate` (`qsa.cu:90-117`) refuses an unset geometry, `n_head % n_head_kv != 0`, `head_dim % 4 != 0`, an odd or
  oversized `n_rot`, and `idx_n_head > 32`. Every geometry must have the indexer: there is no dense-attention mode.

KV layout (FP16): `[page][kv_head][page_size][head_dim]`. The row of cell t and head h is
`(table[t / page_size] * n_head_kv + h) * page_size + t % page_size` (`qsa.hpp:63-77`), so one head's consecutive cells
are contiguous within a page. The page table is never assumed to be the identity, and "nothing outside this header may
assume the mapping" (`qsa.hpp:74-77`). With KV streaming the table is a residency map in which -1 means "not in VRAM".

Indexer state (`QsaIndexerBuffers`, `qsa.hpp:215-220`): `tail` (3 x 128 f32, the raw keys of the block being filled),
`dead` (128 f32, the spare slot's key = rotated `rms_norm(raw key of cell 0)`, constant for a sequence), `pooled`
(`max_cells / 4 + 1` rows; block b at row b, the spare at row `n_bid`), `block_pos` (the last completed block's first
position). The keys are FP32 because the selected ids carry no tolerance (`qsa.hpp:79-87`).

## Main entry points

### Reference path (`qsa.cu`)

| Function | Kernel / launch | What it does |
| --- | --- | --- |
| `kv_append_step` (`qsa.cu:595`) | `kv_append_kernel` (`:122`), 128 threads | K and V of the new cell to FP16 with `f16_from_f32` (not `__float2half`); writes the VRAM page only when it is resident, and always writes the host copy when a `KvHostPools` is given |
| `indexer_key_append` (`qsa.cu:722`) | `indexer_key_append_kernel` (`:155`), one block, one thread per indexer dim | Stores the raw key in the tail; at `pos == 0` computes `dead`; on a block completion pools the mean of the 4 raw keys, applies RMSNorm x `w_k_norm`, writes `pooled[b]`, moves the spare to `b + 1`, and rotates the row in place at the block's first position `pos_base + 4b`. Double arithmetic with `__dadd_rn` / `__dmul_rn` so the pooled key is bit-exact against the reference (`qsa.cu:28-40`). The rotation lives inside the kernel so the call is capturable (`qsa.cu:148-153`) |
| `qsa_index_step` (`qsa.cu:608`) | `qsa_index_kernel` (`:249`), one block per pooled row, one warp per indexer head, FP64 dots | `score = bias + sum_h relu(dot(pooled[b], q_idx[h]))`, written to each of the block's cells; incomplete-tail cells get `+1e9f` (llama.cpp's causal decode input), even with a null bias. A block past `n_bid` returns, so the grid can be a constant `max_blocks` |
| `topk_512_step` (`qsa.cu:622`) | `topk_kernel` (`:307`), 1 block x 256 threads | Exact top-`width` cells, ties to the lower index, output ascending. The width-th largest score is found as a 32-bit order key by 32 monotone passes ("binary lifting"), then one ascending walk emits the set (`qsa.cu:6-15`). NaN never selected. Limited to `n_kv <= kTopkMaxCells` = 32,768 (`qsa.hpp:282`) |
| `kv_gather_step` (`qsa.cu:641`) | `kv_gather_kernel` (`:406`), 256 threads, `uint2` copies | Selected cells through the page table into contiguous FP16 scratch `[id][kv_head][head_dim]` |
| `qsa_attend_step` (`qsa.cu:657`) | `qsa_attend_kernel` (`:439`), grid `n_head`, block `head_dim`, dynamic shared `(max_ids + 32) * 4` B | Softmax GQA over the gathered cells, scale `1/sqrt(head_dim)`; optional per-head weights out (`L{n}.qsa_scores` captures). The shared-memory opt-in is per device (`qsa.cu:672-695`) |
| `qsa_gate_apply_f32` / `qsa_gate_apply` (`qsa.cu:803`, `:812`) | 256-thread elementwise | `attn * sigmoid(q_full[h][head_dim + d])` with the multiply in double; the gate is the second half of each head's `2*head_dim` block, not interleaved. The FP16 form rounds once at the end; the F32 form feeds the Q8_K quantizer for a K-quant `attn_output` (`qsa.hpp:314-324`) |

The host-scalar wrappers `kv_append`, `qsa_index`, `topk_512`, `kv_gather`, `qsa_attend` (`qsa.cu:716-791`) fill a
module-level step buffer with a synchronous copy and are for tests and direct launches only. The header says the layer
must use the `_step` forms (`qsa.hpp:326-341`).

### Block scores and selection at scale (`qsa_select.cu`)

These replace `qsa_index_step` + `topk_512_step` on the fast path (default; `--no-fast-select` keeps the old pair), in
the verify window and in the prompt path. They score once per 4-cell block instead of once per cell
(`qsa_select.hpp:4-17`). Geometry is fixed: idx_dim 128, 4 heads, block 4 (`qsa_select.cu:17`).

| Function | Kernels | Notes |
| --- | --- | --- |
| `qsa_block_scores` (`qsa_select.cu:991`) | `block_scores_kernel` (`:28`, a warp per (query, block), 8 warps per CTA); `block_scores_multi_kernel` (`:725`, a fixed 256-CTA grid that reads each key once for up to `MQ` = 8 queries) | FP32. Block `n_bid` (the tail) scores the `dead` key and gets +1e9 when it has cells. The multi kernel runs for every call with no active-block count and <= 8 queries (decode, the verify window, prefill's pooled16 call) unless `STRATA_SCORES_MULTI=0`. With `active_blocks > 0` the grid only reaches that many blocks (not for captured graphs) |
| `qsa_block_scores_tc` (`:1015`) | CUDA: `block_scores_tc_kernel` (`:195`, 128 threads, `mma.sync m16n8k8` TF32 with a 3xTF32 split) + `block_scores_tail_kernel` (`:438`). HIP: `block_scores_wmma_kernel` (`:329`, gfx12 `v_wmma_f32_16x16x16_bf16`, a three-way bf16 split, six products) | Prompt path only. FP32-level accuracy, another summation order (not bitwise). Returns false (nothing launched) on another geometry; on CUDA below sm_80 (or with `STRATA_QSA_WARP=1` / `select`); on HIP unless `STRATA_SELECT_WMMA=1` on a gfx1200/gfx1201 device |
| `qsa_block_topk` (`:1174`) | cluster, register, wide or reference kernel | 4-pass radix select over the query's `n_bid + 1` blocks weighted by their cell counts, then cells emitted ascending with ties to the lowest index. The ids equal `topk_kernel`'s |
| `qsa_block_topk_cluster` (`:1108`) | `block_topk_cluster_kernel` (`:847`), a cluster of `CL_N` = 8 CTAs x 1,024 threads per query (`cudaLaunchKernelEx`), keys of `ceil(max_blocks / 8)` blocks per CTA in shared memory, histograms pushed through distributed shared memory (`mapa`, `barrier.cluster`) | sm_90+ (or `STRATA_EMULATE_CC`) and code built for sm_90 (`ptxVersion` and `binaryVersion` >= 90; otherwise the kernel body is a trap); occupancy checked with `cudaOccupancyMaxActiveClusters`. Calls of up to `CL_MAXQ` = 16 queries. False on HIP. Capturable. Identical ids by construction (`qsa_select.cu:780-786`) |
| `qsa_block_topk_ref` (`:1078`) | `block_topk_kernel` (`:57`), 256 threads, one shared histogram | the original kernel; `STRATA_TOPK_OLD=1` |

`qsa_block_topk` dispatch (`qsa_select.cu:1174-1235`):

1. CUDA: the cluster kernel when `STRATA_QSA_CLUSTER` is not 0, `STRATA_TOPK_OLD` is unset, nq <= 16 and the cluster
   path accepts the call.
2. The "reach" (blocks a query can have) is the capacity `max_blocks`, except on sm_75 (or with
   `STRATA_TOPK_ACTIVE_ANY=1`) when the caller passes `active_blocks`, where that bound is used (issue #512; other CUDA
   cards keep the capacity rule because RTX 5070 64K prompts were 1-3% slower with it, per the comment at `:1195-1198`).
   `STRATA_TOPK_CAPACITY_GUARD` turns the bound off. HIP always uses the bound when given.
3. The reference kernel when `STRATA_TOPK_OLD` is set, or for small prompt batches: CUDA, past the register reach with
   fewer than 4,608 active blocks; HIP, fewer than 7,168 blocks (gfx1201) or past the reach.
4. Otherwise `block_topk_wide_kernel` (`:606`, CUDA only) past the register reach, else
   `block_topk_reg_kernel<TK_PER>` (`:507`, 1,024 threads, keys in registers). `TK_PER` is 33 blocks per thread
   (contexts up to 4 x 1,024 x 33 = 135,168 cells); HIP may use 66 (`TK_PER_MAX`, `:473-478`, up to 270,336 cells,
   because RDNA has 192 VGPRs per lane at 1,024 threads).

The cluster kernel's measured gain, quoted from `qsa_select.cu:778-779` (RTX 5070, `decode_cluster_parity --bench`, per
call at capacity = context): 21.9 -> 15.6 us at 32K, 58 -> 18 us at 128K, 200 -> 22 us at 262K.

### Split-K decode attention (`qsa_decode_attn.cu`)

`qsa_decode_attn_step` (`qsa_decode_attn.cu:495`) and `qsa_decode_attn_batch` (`:450`) read the KV pools directly
through the page table and the selection (no gather copy). Constants: `HD` = 256, `G` = 12 query heads per KV head,
`CHUNK` = 64 cells per block, 256 threads (`:16-20`).

- Kernel 1: `attn_chunk_kernel<KV_MODE>` (`:84`), grid `(ceil(cap / 64), n_head_kv[, n_q])`. Each block takes one
  64-cell chunk of the selection for all 12 query heads of one KV head: scores (each warp takes cells, each lane 8 of
  256 dims), the chunk's max and exp-sum per head, then `p . V` with one thread per dimension. It writes one partial
  (acc, m, l) per chunk.
- Kernel 2: `attn_merge_kernel` (`:395`), grid `(n_head[, n_q])`, 256 threads: log-sum-exp merge of the chunks.
- `KV_MODE` from the pools (`:462-463`, `:503-504`): 0 FP16, 1 INT8 (+ fp16 scale per 64), 2 Q4_0 (rotated; the
  caller rotates q and the output with the FWHT), 3 hybrid K8V4 (INT8 K, rotated Q4_0 V). `QsaAttnPools`
  (`qsa_decode_attn.hpp:23-33`) holds the pointer set.
- A page of -1 (a block KV streaming could not make resident) is masked: score `-FLT_MAX`, weight 0 (`:118-127`).
- Scratch: `qsa_decode_attn_scratch_floats(cap, s)` = `ceil(cap / 64) x n_head x 258 + 64` floats per query
  (`:490-493`). At the QSA layers' cap of 2,051 that is 33 chunks.
- Both entry points require `head_dim == 256`, `n_head == 12 * n_head_kv` and a non-null `ids` (`:452-458`,
  `:497-501`); the batch form takes up to 65,535 queries. The batch form is bitwise the single-query form per query.
- `STRATA_EXPERIMENTAL_SM60` builds add `attn_chunk_kernel_pre75` (PR #540, `:183-393`): the same arithmetic with fewer
  shuffles, bit for bit equal, more registers (80 vs 38 on sm_70), "+7% prompt speed on a V100" per `:184`. It runs
  only on cards below sm_75; `STRATA_ATTN_PRE75=0` turns it off (`pre75_attn`, `:420-446`).

### Tensor-core prompt attention (`qsa_prompt_attn.cu`)

`qsa_prompt_attn_batch` (`qsa_prompt_attn.cu:1407`) takes the same arguments as `qsa_decode_attn_batch` minus the
scratch. It returns false (nothing launched) when it cannot run, and `src/prefill/prefill.cpp:2222-2226` then uses
the decode kernel one query at a time.

- One block of 128 threads per (query, KV head), grid `(n_q, n_head_kv)` in slices of 65,535 queries. It walks the
  selection in chunks of `CH` = 32 cells with an online softmax: no split-K scratch, no merge kernel.
- q.k and p.v are `mma.sync` FP16 MMAs with FP32 accumulation over the 12 query heads (+ 4 pad rows). q and p are split
  into FP16 hi + lo halves (two MMAs each, about 22 bits). INT8 codes enter exactly; their per-64 scales are applied in
  FP32 (`qsa_prompt_attn.hpp:8-18`). Not bitwise equal to the decode kernel; `qsa_prompt_attn_parity` bounds the
  difference. The header gives the motivation: the decode kernel took 21% of a 32K prompt (5.6 s of 26 s on an RTX
  5070, Q2_0, int8 KV; `qsa_prompt_attn.hpp:3-6`).
- Kernels: `prompt_attn_kernel<KV_MODE>` (`:106`, modes 0 FP16, 1 INT8, 3 K8V4 with V dequantized at gather, 4 Q4_0 with
  a scale per 32), `prompt_attn_i8_kernel` (`:463`, INT8 "v2": each warp gathers its own 64-dim slice with
  double-buffered `cp.async`), `prompt_attn_v70_kernel<KV_MODE>` (`:793`, Volta `mma.m8n8k4`, SM60 builds only,
  PR #600), `prompt_attn_wmma_kernel` (`:1172`, HIP gfx12 `v_wmma_f32_16x16x16_f16`, INT8 KV only).
- `mma16816` (`:46-61`): sm_80 uses m16n8k16; sm_75 issues two m16n8k8 MMAs with the same register mapping (rounds
  twice, so Turing and Ampere+ differ in the last bits); pre-sm_75 and HIP builds compile it to a trap.

Dispatch (`qsa_prompt_attn.cu:1407-1477`), by the current device's compute capability (cached per device; a layer
split can mix cards):

| KV format | sm_80+ | sm_75 | sm_70 (SM60 build) | below sm_70 |
| --- | --- | --- | --- | --- |
| FP16 (mode 0) | `launch<0>` | `launch<0>` | `launch70<0>` | false |
| INT8 (mode 1) | `launch_i8` (`STRATA_PROMPT_ATTN_V1=1`: `launch<1>`) | `launch<1>` | `launch70<1>` | false |
| K8V4 (mode 3) | `launch<3>` | `launch<3>` | `launch70<3>` | false |
| Q4_0 (mode 4) | `launch<4>` (`STRATA_PROMPT_ATTN_Q4=0`: false) | false (no parity run for pairs of m16n8k8) | false | false |

`STRATA_QSA_WARP=1` or `attn` makes any card act as sm_75. In a build without `STRATA_EXPERIMENTAL_SM60`, `launch70`
returns false (`:1116-1121`). On HIP the function returns false except for INT8 KV with `STRATA_HIP_WMMA=1` on a gfx12
device (`hip_wmma_usable`, `:1364-1385`); a gfx906 build always returns false (`:1415-1417`). Geometry must be
head_dim 256 with `n_head == 12 * n_head_kv`.

Upstream's [AMD_HIP.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/AMD_HIP.md)
(around line 270) reports the WMMA kernel at 7.2-7.5x the portable kernel, R9700 prompts 4K 1,784 -> 2,427 tok/s, and
that greedy text differs from token ~50 on, as with the CUDA tensor-core attention.

### Pinned llama.cpp adapters ("native")

| Function | Enabled by | Contract |
| --- | --- | --- |
| `native_qsa_rms_norm_weighted` (`native_qsa.cu:108`) | `--native-qsa` (`native_qsa_set_enabled`) for the q and k norms | ggml `rms_norm` then `mul`: 256 threads below 1,024 columns, else 1,024, the pinned XOR tree; `(scale * x) * gamma` |
| `native_qsa_gate_apply` (`native_qsa.cu:120`) | `--native-qsa` | `attn * sigmoid(q_full second half)` in F32 |
| `native_qsa_indexer_append` (`native_qsa_indexer.cu:228`), `_append_batch` (`:255`) | `--native-qsa-indexer` | Pinned F16 raw-key cache (the tail holds half-rounded raws expanded to F32), F32 pooling, weighted RMSNorm, text IMRoPE with `rope_scaling` arguments and the optional angle table. Fixed geometry 128 / 4 / n_rot 64; `pos_base` must be a non-negative multiple of 4. The relative position is a cell index from `kStepPos`. The batch form is the prompt path's (host-side positions, not capturable) |
| `native_qsa_score` (`native_qsa_score.cu:191`) | `native_qsa_score_set_enabled` | Pinned MMF dispatch for budget 2,048 with n_kv padded to 256: `ldmatrix` + TF32 `mma.sync` on raw F32 bits (no `cvt.rn.tf32`). Below sm_80 an FP32-FMA path (rounds differently); on HIP an ordered scalar `scalar_score_kernel` (`:148`). No caller in `src/core`, `src/prefill` or `src/program`; tested by `tests/hip/native_qsa_score.cpp` and the oracle parity test |
| `native_flash_attn_short_step` (`native_flash_attn.cu:192`) | `--native-flash-attn-short` (diagnostic; `--max-context <= 256`, not part of `--native`) | FlashAttention-vec for exactly Q24x256 / KV2x256, scale 1/16, 24 blocks x (32, 4) threads over the gathered FP16 scratch. An invalid step writes `status = kNativeFlashAttnUnsupportedStep` and NaN output; the caller copies `status` back (`src/core/layer.cpp:993-1000`) |

## Control and data flow

Decode, per QSA layer (`qsa_layer`, `src/core/layer.cpp:876-1032`):

```
x --> bf16 MMVF / bf16_gemv: indexer.k_proj (raw key)          [step record: one H2D or mapped copy, :908-917]
x --> attn_k, attn_v GEMVs --> k norm + rope (rms_norm_weighted | native_qsa_rms_norm_weighted;
                                             rope_neox_apply | native_rope_apply)
kv_append_step | kv_append_q8_step | kv_append_q4_step        (rotated formats: fwht256 first; kv-cache.md)
indexer_key_append | native_qsa_indexer_append
x --> attn_q GEMV --> split q / gate (cudaMemcpy2DAsync) --> q norm + rope
x --> indexer.q_proj --> norm + rope
fast select (default): qsa_block_scores -> qsa_block_topk      (--no-fast-select: qsa_index_step -> topk_512_step)
qsa_kv_resolve -> kv_stream_resolve                            (only when the layer streams its KV)
[fwht256 on q when K is rotated]
qsa_decode_attn_step                                           (default, no dumps, no short flash)
  | kv_gather_*_step -> qsa_attend_step | native_flash_attn_short_step
[fwht256 on the output when V is rotated]
native_qsa_gate_apply | qsa_gate_apply_f32 -> attn_output (native_mmvq | Q8_K + s_gemv_q8k_split)
```

The verify window (`src/core/verify.cpp:819-878`) uses `qsa_block_scores` / `qsa_block_topk` and
`qsa_decode_attn_batch`. The MTP drafter (`src/core/mtp.cpp:530-556`) has no indexer: it attends with
`qsa_decode_attn_batch` over an identity id list (`ident_`, built at `mtp.cpp:263-301`), optionally narrowed to the last
`window` cells by `window_ids` ([sampling-and-verify.md](sampling-and-verify.md)); its capacity is the window or
`max_cells`, rounded up to 64 (`mtp.cpp:248-249`). The prompt path (`src/prefill/prefill.cpp:2101-2226`) appends the
indexer keys with `native_qsa_indexer_append_batch`, scores with `qsa_block_scores_tc` (or the warp kernel with
`STRATA_SELECT_OLD=1`), selects with `qsa_block_topk` and an active-block count, and attends with
`qsa_prompt_attn_batch` (`STRATA_PROMPT_ATTN_OLD=1`: the decode kernel).

## Configuration

- CLI: `--native-qsa`, `--native-qsa-indexer`, `--native-flash-attn-short`, `--no-fast-select`, `--no-fast-attn`
  (`src/program/generate.cpp:1519-1522`, applied at `:2280-2283`), `--kv`, `--kv-resident`, `--max-context` (sets
  the capacities), `--rope-*` (see [norms-rope-elementwise.md](norms-rope-elementwise.md)).
- Environment (kernel-side): `STRATA_SCORES_MULTI`, `STRATA_QSA_WARP`, `STRATA_QSA_CLUSTER`, `STRATA_TOPK_OLD`,
  `STRATA_TOPK_CAPACITY_GUARD`, `STRATA_TOPK_ACTIVE_ANY`, `STRATA_ATTN_PRE75`, `STRATA_PROMPT_ATTN_V1`,
  `STRATA_PROMPT_ATTN_Q4`, `STRATA_HIP_WMMA`, `STRATA_SELECT_WMMA`, `STRATA_EMULATE_CC`. Caller-side:
  `STRATA_SELECT_OLD`, `STRATA_PROMPT_ATTN_OLD`.
- Compile time: `STRATA_EXPERIMENTAL_SM60` (pre-sm_75 decode kernel, Volta prompt kernel); `D1_CH`, `D1_NO_QLO`,
  `D1_NO_PLO` are experiment macros in `qsa_prompt_attn.cu` (chunk size, dropping the lo halves).

## Model- and platform-specific behaviour

- All model variants (Flash-Next, the Coder, Swift 1.5, Unsloth) share the geometry: the kernels hard-code
  12 query heads per KV head, head_dim 256 and the 128 / 4 / 4 indexer. The packs differ only in the projection weight
  types, which this page does not touch.
- RTX 50 (sm_120) and other sm_90+ builds get the cluster top-k. Upstream's DETAILS.md (engine 0.1.36) reports the
  cluster top-k plus the cluster argmax raising Q2_0 decode at 4K from 89 to 93.5 and at 128K from 64.5 to 76.4
  tokens/s on RTX 50 ([DETAILS.md](../../../upstream/DETAILS.md), "Engine 0.1.36").
- Turing (sm_75): no `cp.async`, so INT8 prompt attention runs the v1 kernel; Q4_0 prompt attention falls back to the
  decode kernel; the block scores use the warp kernel; the top-k uses the active-block bound.
- Volta (sm_70) and Pascal (sm_60/61) need the experimental build; Pascal has no prompt attention kernel at all.
- AMD: no tensor-core or cluster path by default; RDNA4 opt-ins above.

## Tests

- `qsa_parity` (`src/kernels/qsa_parity.cpp`; cache, indexer, scores, top-k, attention, gate; page sizes 1/4/512 with a
  permuted table must give identical attention; `f16_from_f32` against numpy vectors).
- `qsa_topk_parity`, `qsa_topk_active_parity` (the sm_75 active-block dispatch, forced with `STRATA_TOPK_ACTIVE_ANY`),
  `qsa_prompt_attn_parity` (tensor-core vs FP32 kernel vs FP64; CUDA, built with the tests or
  `-DSTRATA_PARITY_PROMPT_ATTN=ON`), `decode_cluster_parity` (cluster top-k and argmax, bitwise and under graph
  capture; skips below sm_90), `kv_hybrid_parity`, `kv_stream_parity`, `qsa_select_bench` (HIP: scorer vs scorer and
  an FP64 gate at 32K under a 262,144-cell capacity).
- HIP: `hip_native_qsa_score`, `hip_prompt_attn_wmma` (skips off gfx12).
- Oracle-linked (sources not in the snapshot): `native_qsa_parity`, `native_qsa_indexer_parity`,
  `native_qsa_score_parity`, `native_flash_attn_parity`.

## Gotchas

- The tail bias is `+1e9f` on incomplete-tail cells even when `bias` is null (`qsa.hpp:258-261`).
- `topk_512` refuses more than 32,768 cells; longer contexts need `qsa_block_topk`.
- `qsa_attend` sizes its shared memory from `max_ids` and exits if the card cannot hold it.
- The prompt kernel and the decode kernel do not agree bit for bit; Turing and sm_80+ prompt kernels do not agree
  either (two k8 MMAs round twice).
- `qsa.hpp:17-25` states the reference kernels are decode-only by design: the per-query block-causal mask is a prefill
  property, implemented in the prompt path.
- The block-scores and top-k kernels hard-code the indexer geometry and exit on any other.

## In UnbiasedStrata

Changed in the fork (commit e41dfc7): `qsa.hpp` / `qsa.cu` gained a dense-attention geometry (`qsa_dense`,
`qsa_shapes_valid`, `qsa_selection_cap`, indexer entry points refuse it); `qsa_decode_attn.*` and `qsa_prompt_attn.cu`
template the group size (2, 8, 12), accept a null `ids` as the identity, cap dense shapes at 128 blocks per KV head
and add `qsa_identity_ids`. `qsa_select.cu`, `native_qsa*.cu` and `native_flash_attn.cu` are unchanged. See the fork's
[attention-qsa.md](../../../codebase/kernels/cuda/attention-qsa.md).

## See also

- [kv-cache.md](kv-cache.md), [norms-rope-elementwise.md](norms-rope-elementwise.md) (RoPE, scaling, M-RoPE),
  [sampling-and-verify.md](sampling-and-verify.md) (`window_ids`), [bits-and-arch-shims.md](bits-and-arch-shims.md)
- [Prefill](../../prefill/README.md), [HIP kernels](../hip/README.md)
