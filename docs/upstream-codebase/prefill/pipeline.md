[Upstream docs](../README.md) › [Prefill](README.md) › Pipeline

# The `Prefill` pipeline

## Purpose

`strata::prefill::Prefill` owns everything a batched prompt needs on one GPU: the chunk buffers, a cuBLAS handle,
the expert staging ring with its events, a copy stream, host staging threads, and optional multi-GPU helpers. It runs
the prompt's positions `[pos0, pos0 + n)` in chunks through all layers (or a layer range on a layer split), writes the
recurrent and attention state into the shared `core::SessionState`, and hands each chunk's final residual rows to the
caller for the MTP draft layer.

Files: [include/strata/prefill/prefill.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/prefill/prefill.hpp),
[src/prefill/prefill.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/prefill.cpp).

## Main types

| Type / function | Where | What it does |
| --- | --- | --- |
| `PrefillStats` | `include/strata/prefill/prefill.hpp:29` | tokens, chunks, total ms, host staging ms, experts streamed / by DMA / resident, PLE ms; printed by `generate.cpp` |
| `Prefill` | `include/strata/prefill/prefill.hpp:44` | the public object; non-copyable, pimpl (`Impl`) |
| constants | `src/prefill/prefill.cpp:89-91` | `N = 2560`, `HC = 4`, `D = 10240`, `LR = 320`, `K = 10`, `NE = 512`, GDN `C = 10240`, `ZV = 6144`, `HV = 48` |
| `Alloc` | `src/prefill/prefill.cpp:227` | bump allocator: counts only (`bytes_needed`), carves from a borrowed region, or `cudaMalloc`s; each take is `n * sizeof(T) + 256` rounded up to 256 bytes; overflow sets `ok = false` |
| `Stager` | `src/prefill/prefill.cpp:256` | host threads copying unpinned expert blobs into a pinned ring; see [expert-streaming.md](expert-streaming.md) |
| `PeerPrefill` | `src/prefill/prefill.cpp:388` | the second GPU's share of a chunk's experts; see [multi-gpu.md](multi-gpu.md) |
| `Prefill::Impl` | `src/prefill/prefill.cpp:478` | every device pointer of a chunk layout, the ring (`stage_dev`, `copied`, `used`, `used_of`, `stage_live`), the mapped grouping tables, PLE buffers, KV staging pool, hand-off buffers |
| `PfTimer` / `PeTimer` | `src/prefill/prefill.cpp:1469`, `:1505` | `STRATA_PREFILL_TIMING=1`: CUDA events per phase on the compute stream (and the peer's) |

### Public API (`prefill.hpp`)

| Member | Line | Notes |
| --- | ---: | --- |
| `draft_kv(mtp, R_rows, next_tokens, n, cell0, err)` | 51 | the MTP draft layer's K/V for prompt cells through this path's GEMMs and idle scratch (below) |
| `reset()` | 61 | frees everything `init` made; used to retry `init` with a smaller chunk |
| `init(wt, g, ss, src, cache, host_res, chunk, stream, err, borrow, borrow_bytes)` | 66 | see "Initialization" |
| `relayout(chunk, borrow, borrow_bytes, err)` | 72 | borrowed buffers only: lay them out again for a chunk of at most `init`'s in a new region (serve, per request) |
| `chunk()` | 73 | the current layout's chunk |
| `set_pinned_share`, `set_ring_override`, `set_ring_budget`, `ring_*`, `ring_bytes_enabled` | 77-111 | static ring-sizing knobs; see [expert-streaming.md](expert-streaming.md) |
| `bytes_needed(g, ss, chunk)` / `bytes_needed_no_ring` | 87, 91 | the allocation sequence counted (`src/prefill/prefill.cpp:1358`), plus 8 MiB slack |
| `run(tokens, n, pos0, err)` | 115 | `run_impl` then `drain_pipeline` (`src/prefill/prefill.cpp:3070`) |
| `stats()` | 117 | `PrefillStats` |
| `set_peer(peer, cap_rows, err)` | 122 | the `--peer-device` GPU computes the rows of experts it holds |
| `on_chunk` | 127 | after each chunk (stream synchronized): the chunk's final residual rows (`T x 4*2560`), `T`, `pos0` |
| `on_stage_chunk` | 133 | layer split: every stage calls it after a chunk, for mid-prompt checkpoints |
| `should_stop` | 136 | checked before every chunk; true makes `run` fail with `"cancelled"` |
| `embd_rows` | 140 | vision: host rows by absolute position, used in place of the token embedding |
| `set_stage(lb, le, next)` | 147 | layer split: this object runs layers `[lb, le)` and hands rows to `next` |
| `set_stage_helper(helper, err)` | 158 | layer split, opt-in `STRATA_PREFILL_HELP=1`: the next stage's idle GPU takes a share of this stage's experts |

## Initialization and memory layout

`Prefill::init` (`src/prefill/prefill.cpp:731`):

1. Geometry refusal: `n_embd != 2560`, `hc != 4`, `hc_lr != 320`, `n_expert < 1` or `ss.k != 10` fails with
   "geometry differs from the artifact's" (`:737-739`). The stage range must be valid and a stage that ends before
   the last layer must have a `next` (`:741-745`).
2. Layer split: two pinned portable hand-off buffers of `chunk x D` floats (`:746-750`); the token-id buffer
   (`:751-758`).
3. The copy stream (non-blocking), `ring_cap()` pairs of `copied` / `used` events, the `Stager` (thread and ring
   counts in [expert-streaming.md](expert-streaming.md)) (`:759-782`).
4. The grouping tables `[ids | slot | src | bounds]` in mapped pinned memory, portable under `--peer-device`
   (`:785-804`; `STRATA_GROUP_COPY=1` keeps plain `cudaMemcpyAsync`s instead). Kernels read them over PCIe, so they
   never queue behind expert blobs on the copy engine (measured in the comment: 4.2 s of a 128K prompt's 70 s).
5. PLE double buffers (pinned, with a pageable fallback) and their events (`:805-815`); with KV streaming
   (`kv_mode == 1`) an identity page table for the staging pool (`:816-825`).
6. From one `Alloc` (borrowed region or `cudaMalloc`): the GEMM scratch (`GEMM_SCRATCH`, 32M FP16 elements) and the
   cuBLAS workspace (`GEMM_WS`, 32 MiB) (`:632-633`, `:831-836`), then `carve(T)`.

`Prefill::carve` (`src/prefill/prefill.cpp:846`) takes, in this order: `emb`, `R`, `xn` (only with
`STRATA_GR_UNFUSED=1`), `grs`, `xn16`, `lo`, `lo16`, `gated`, `inj`, `mixed` (FP32) with its BF16 and FP16 images,
`bo`, the optional BF16 low parts (`STRATA_PREFILL_BF16X2`), the QSA step records, then ONE region of
`max(gdn_set_bytes, qsa_set_bytes, moe_set_bytes)` holding the three phase-local sets (three `Alloc`s over the same
base, `:867-906`), two dequantized-expert buffers (`DQ = 2`, FP16 path, `:907`), the MMQ identity row map, bounds and
two 16-expert group buffers with a 4,096-byte zero tail (`MMQ_TAIL`, `:908-917`), `ring_slots(T)` blob slots
(`:918-923`), the PLE buffers, and the KV staging pool (`take_stage`, KV streaming only). The order must match
`bytes_needed` (`:1358-1405`); a mismatch makes `init` fail with "do not fit" rather than overlap.

`moe_bufs` (`:703-710`) shrinks the MoE buffers when the layout runs the fused Q2_0 kernels (`fused_layout`,
`:697-699`): about 100 KB per token less, while each buffer keeps MMQ's size for a final chunk below
`stream_all_min()`.

## The chunk loop (`run_impl`, `src/prefill/prefill.cpp:1545`)

Before the first chunk: a one-chunk prompt on a layer split may swap in the stage helper (`:1565-1579`); the MMQ
identity row table is rewritten (`:1586`, it lives in borrowed slots that a refill overwrote); the PLE gather lambda
is set up (`:1601-1613`).

Per chunk of `T = min(m.T, n - c0)` tokens at `p0 = pos0 + c0` (`:1618`):

1. `should_stop` (`:1619`), progress text for the serve watchdog (`core::progress_at`).
2. **Embeddings** (`:1626-1679`): a later layer-split stage uploads the previous stage's rows (`hand_in_`); otherwise
   one batched device gather (`NativeEmbed::gather_dev` or `embedding_gather_dev` for the pack's quantized table)
   unless a row is an image row or out of range, in which case per-token `core::embed_row`. Then `gr_broadcast`
   copies the embedding into the four streams.
3. **PLE rows** (when `ss.ple.ready()` and the stage covers layer 1): the chunk's 16 n-gram rows per token are
   gathered on a `std::async` thread (`ple_gather`: `ngram_rows`, then `PleTable::gather_batch`, an SSD read per
   missing row), double-buffered so chunk `c+1`'s gather runs during chunk `c`; `ple_land` uploads them just before
   layer 1 (`:1683-1716`).
4. **QSA step records** for every position (`qsa_step_fill`), uploaded to `steps_dev` (`:1718-1721`).
5. **Expert stream plan** for chunks of `stream_all_min()` (1,024) tokens or more: see
   [expert-streaming.md](expert-streaming.md) (`:1725-1898`).
6. **Layer loop** over `[LB, LE)` (`:1915`):
   - Layer 1 with PLE: the PLE block batched over the chunk in sub-batches carved from the idle scratch region
     (key and value projections as GEMMs, then `native_ple_postops_batch`), or token by token
     (`ple_block` + `ple_history_advance`) with `STRATA_PLE_BATCH=0` or when the region is too small (`:1922-1975`).
   - Two halves (`:1976`), each starting with the hyper-connection read: `gr_norm_rs` (skipped when the previous
     write already normed `R`), BF16 `hc_*_down`, `gr_silu`, `hc_*_up`, `hc_*_inject`, `gr_mix_r` (`:1977-1994`).
   - Half 0, GDN layer (`!is_qsa_layer`) or QSA layer: [gdn-and-attention.md](gdn-and-attention.md)
     (`:1996-2234`).
   - Half 1, MoE: [moe.md](moe.md) (`:2235-2887`).
   - Write (`:2888-2906`): `gr_write`, or the fused `gr_write_norm_rs` that also norms `R` for the next half when
     nothing touches `R` in between (not the stage's last half, not before layer 1's PLE block, not under a control
     vector). After half 1, `--control-vector-scaled` (`cvec_apply`) where it covers the layer (`:2907-2908`).
7. After the last layer (`:2911-2966`): `ple_land` (a stage that ends before layer 1), join the issuer thread and
   add its counters, then either hand the rows to the next layer-split stage (async `next_->run_impl`) or
   synchronize and call `on_stage_chunk` / `on_chunk`.

After all chunks (`:2970-3046`): `ss.ple_prev` is advanced to the last two tokens; both streams are synchronized
separately, so a failed expert copy fails this prompt (PR #121, `:2994-2998`); `STRATA_PREFILL_TIMING` prints the
per-phase GPU timeline and the host shares.

## What prefill leaves for decode

| State | Written by | Where |
| --- | --- | --- |
| GDN recurrent state `[128][48][128]` and conv history `[10240][3]` per GDN layer | `gdn_conv`, `gdn_recurrence` | `ss.gdn_state + (gdn_index - ss.gdn_ord0) * gdn_floats` (`src/prefill/prefill.cpp:2005-2006`) |
| K/V of every QSA layer's prompt cells | `kv_append` / `kv_append_q4` into the `QsaState` pools via its page table (FP16, INT8 + scales, Q4_0, K8V4 hybrid; optional FWHT rotation); with KV streaming also the host copy | `:2040-2084` |
| Indexer state (tail, dead, pooled keys, block positions) | `native_qsa_indexer_append_batch` | `:2094-2111` |
| PLE history | `native_ple_postops_batch` / `ple_history_advance` | `ss.ple.hist` |
| `ss.ple_prev` | end of `run_impl` | `:2970-2971` |
| MTP draft layer K/V | `on_chunk` -> `draft_kv` or `MtpDrafter::prefill` | the drafter's `QsaState` |

The one-shot path batches positions `[0, n_prompt - 1)` (or `[0, P)` with `--prefill-until P`) and starts the token
loop at the next position; for a native pack that last prompt token is the first verify window
(`src/program/generate.cpp:7479`, `:7564`, `:7579`). `ss.ple_prev` is then reset from the token list for the token path
(`:7567-7568`).

## `draft_kv` (`src/prefill/prefill.cpp:958`)

Computes the MTP draft layer's K/V for prompt cells from each chunk's final residual rows, in batches carved from the
idle scratch region (at least 64 rows, a multiple of 64). It returns false with `err` empty when not applicable
(`STRATA_MTP_BATCH=0`, a KV stream mode other than 0 or 2, a ring with `STRATA_MTP_BATCH_RING=0`, a K8V4 hybrid K/V,
another device, missing drafter tensors, an unsupported embedding table, too little scratch), and the caller then runs
`MtpDrafter::prefill`. The arithmetic (`:1054-1108`): the next token's embedding, `pre_fc_norm_embedding`,
`fc_embedding`; the residual rows through `pre_fc_norm_hidden` and `fc_hidden` per stream; `add_streams_broadcast`;
the attention hyper-connection read; `k_proj` / `v_proj`; `k_norm`, rope at the cell's position, optional FWHT
rotation; then the append into the drafter's cache. The drafter's Q8_0 matrices go through Q8_1 x Q8_0 MMQ when it
fits (`STRATA_MTP_BATCH_F16=1`: FP16 GEMMs). Only cells the drafter's window can still reach are written
(`mtp.first_needed()`). Not bit-identical to the drafter's own pass; only drafts can differ, never the target's
logits (`prefill.hpp:46-50`). `STRATA_DRAFT_TIMING` prints where the time goes.

## How `generate.cpp` drives it

| Site | What happens |
| --- | --- |
| `src/program/generate.cpp:431-445` | options: `prefill_chunk` (default 0 = token path), `prefill_auto`, `prefill_auto_max` (8192), `no_prefill_borrow`, `prefill_until` |
| `:1379-1393` | `--prefill CHUNK|auto|auto:N`, `--no-prefill-borrow`, `--prefill-until P` (`STRATA_PREFILL_AUTO_MAX` raises a bare `auto`'s ceiling to 16384 / 32768) |
| `:4388` | `set_pinned_share` from the arena's pinned bytes |
| `:4411-4529` | `lend_slots`, `request_chunk` (round to 256), `kAutoLendPct`, `biggest_chunk`, `plan_lend` (the auto chunk scan) |
| `:7480-7578` | one-shot: lend slots, `init`, `on_chunk` -> `draft_kv` / `mtp.prefill`, `run`, refill lent slots (queued, one wait; `STRATA_REFILL_BLOCKING=1` waits on each), print `PrefillStats` |
| `:4607-4613` | `--serve` requires `--prefill`, `--spec`, `--mtp` |
| `:4889-4985` | serve: per stage `set_stage` + `init`, stepping the chunk down 6144 .. 512 on "do not fit" |
| `:4989-5010` | serve: `set_peer` (`--peer-prefill-rows`), `set_stage_helper` around the stages |
| `:5375-5416` | serve's `on_chunk`: draft K/V (batched `draft_kv` only on one GPU), the `PP` progress line for the server, checkpoints; on a layer split moved to the last stage |
| `:6795-6830` | serve per request: `lend` the slots this prompt needs and `relayout` each participant |
| `:6863-6870` | serve: `sp.run` whole, or one chunk at a time between decode slots with `--batch` |

`--short-read N` (default 64) lets serve read a short fresh text segment through the decode windows instead of the
prompt path (`src/program/generate.cpp:631`, `:6677`).

## Configuration

| Knob | Effect |
| --- | --- |
| `--prefill CHUNK`, `--prefill auto[:16384|:32768]` | chunk size; auto = the largest chunk whose buffers the cache can lend |
| `--no-prefill-borrow` | own buffers reserved for the session (estimated as `160 + chunk * 680 / 1024` MiB, `src/program/generate.cpp:3213`) |
| `--prefill-until P` | validation: batch only `[0, P)`, the rest through the token path |
| `STRATA_PREFILL_TIMING=1` | per-phase GPU timeline and host shares |
| `STRATA_PF_STEP_SYNC=1` | synchronize after each step and log any over 250 ms (stall diagnosis, #579 #613) |
| `STRATA_PREFILL_BF16X2=1|2` | adds each BF16 activation's remainder as a second GEMM (about 16 mantissa bits): 1 includes the hyper-connection, 2 all but it; changes numbers, off by default (`:202-215`) |
| `STRATA_GR_UNFUSED=1` | keep the FP32 normalized rows and separate norm / mix / write (A/B of F-1 / F-2) |
| `STRATA_PLE_BATCH=0` | per-token PLE block |
| `STRATA_MTP_BATCH=0`, `STRATA_MTP_BATCH_RING=0`, `STRATA_MTP_BATCH_F16=1`, `STRATA_DRAFT_TIMING` | `draft_kv` controls |
| `STRATA_INDEXER_PER_TOKEN`, `STRATA_SELECT_OLD`, `STRATA_PROMPT_ATTN_OLD` | older QSA kernels (A/B); see [gdn-and-attention.md](gdn-and-attention.md) |
| `STRATA_TRACE`, `STRATA_PREFILL_DUMP_R=<file>`, `STRATA_DBG_NAN`, `STRATA_STATE_HASH_GDN`, `STRATA_SEL_OVERLAP`, `STRATA_IDX_FP16_CHECK`, `STRATA_QSA_DUMP=<file>` (`STRATA_QSA_DUMP_LAST`) | debugging and measurement prints |

## Platforms

- CUDA: the full path. HIP (RDNA, `STRATA_ENABLE_HIP`): no fused int8 kernels, MMQ only with
  `-DSTRATA_PREFILL_MMQ=ON`, optional hipBLASLt, `STRATA_KV_HOST_DMA=1` (HIP only, `:2049-2060`) and a 96-slot ring
  with `STRATA_HIP_WMMA=1`. gfx906 (`STRATA_HIP_GFX906`) compiles the CUDA sources through the compat headers and
  reports the fused kernels unavailable. Older NVIDIA cards: see [gemm.md](gemm.md) for the BF16 fallbacks.
- One GPU or several: [multi-gpu.md](multi-gpu.md).

## Gotchas

- `bytes_needed` once over-counted (`xn` always, `grs` never); the corrected count is used unless
  `STRATA_RING_BYTES=0`, which restores the 0.1.39 count (`:1367-1378`).
- The MMQ identity table lives in borrowed slots that the post-prompt refill overwrites, so it is rewritten at every
  `run` (`:1584-1586`); `carve` deliberately does not write it (`:914-915`).
- A PLE gather failure surfaces at `ple_land` with the SSD reader's message.
- `rope` in `kernels.cu` exits the process on an invalid rope-scaling config; the engine validates it at startup.

## In UnbiasedStrata

Kept and changed (commits e41dfc7, 16d0fb2): the widths come from a `Geom` built from `ModelGeometry`, `init`
accepts Qwen3.6 as well, the layer loop has a plain-block branch, `draft_kv` has a Qwen3.6 branch, and an `on_seam`
hook exposes per-layer rows to tests. See [../../codebase/prefill/pipeline.md](../../codebase/prefill/pipeline.md).

## See also

[expert-streaming.md](expert-streaming.md), [moe.md](moe.md), [gdn-and-attention.md](gdn-and-attention.md),
[gemm.md](gemm.md), [multi-gpu.md](multi-gpu.md), [tests.md](tests.md),
[docs/upstream/DETAILS.md](../../upstream/DETAILS.md), the program area (`docs/upstream-codebase/program/`) for
`generate.cpp`, and the core area for `SessionState`, `ExpertCache` and `ExpertSource`.
