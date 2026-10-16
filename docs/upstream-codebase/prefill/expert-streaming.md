[Upstream docs](../README.md) › [Prefill](README.md) › Expert streaming

# Getting experts onto the card during a prompt

## Purpose

A prompt chunk of thousands of tokens routes to nearly every expert of every MoE layer (at 1,024 tokens x top-10 of
512 almost all are hit). Experts that are not resident in the GPU expert cache must be copied host -> device once
per chunk. This page covers where the prompt path's memory comes from (borrowed cache slots), how the copies are
scheduled so the copy engine overlaps compute, and how the ring and the chunk are sized. It is the mechanism that
makes prompt speed depend on VRAM: with few resident slots, and those partly lent to the prompt buffers, most experts
of every layer cross PCIe once per chunk.

Files: [src/prefill/prefill.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/prefill.cpp)
(ring, `Stager`, streamed walk, sizing),
[src/program/generate.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/program/generate.cpp)
(lending, chunk choice, refill).

## Borrowing cache slots

By default (`--no-prefill-borrow` not given, a residency table loaded) the prompt path does not allocate. The one-shot
path (`src/program/generate.cpp:7480-7512`) runs `plan_lend` to pick the chunk and the slot count `k`, shrinks the
chunk to what this prompt needs (`request_chunk`: the token count rounded up to 256), marks every expert in the last
`k` slots non-resident (`host_res[i] = kNotResident`, uploaded to `d_res`), and passes
`xcache.device_slot(first)` and the bytes to the end of the cache as `borrow` / `borrow_bytes` to `Prefill::init`.
`Alloc` then carves every buffer from that region. After `run`, the lent slots are refilled from the expert source
with queued copies and one wait (`STRATA_REFILL_BLOCKING=1` waits on each) and marked resident again
(`:7542-7560`), and the engine prints "N lent slots refilled in X ms".

In `--serve` the loan is per request and per participant: CUDA0 and every layer-split stage lend from their own
cache, and only the rows of their own layers (`PfPart`, `src/program/generate.cpp:4628`). `lend(tokens)`
(`:6795`) rounds the request to 256 tokens, caps it at the configured chunk, refills only this participant's earlier
loan if it is too small, and calls `Prefill::relayout` into the slots it needs (`:6821`), so a short prompt lends
little. `STRATA_SPLIT_SMALL_OWN` (layer split) keeps a small chunk's buffers per stage so short requests lend nothing.

The lending rule: at least 128 slots stay with the cache (`k + 128 <= slots`); with `--prefill auto` at most
`kAutoLendPct` percent of the slots may be lent, 90 when (nearly) every expert byte is pinned and 85 otherwise
(`STRATA_PREFILL_LEND_PCT` overrides, `src/program/generate.cpp:4432-4436`). A fixed `--prefill CHUNK` is halved until
its loan fits; if the one-shot path had to halve it, it allocates its own buffers at the requested chunk instead
(`:7496-7498`). An expert whose slot is lent is streamed by the prompt path (and computed on the CPU by decode if
decode runs meanwhile).

## Two ways a layer's experts are read

| Mode | When | How |
| --- | --- | --- |
| Routed-only staging | chunks below `stream_all_min()` (1,024 tokens), a ring of `STAGE` (8) slots, or no `ExpertSource` | after routing, the host groups the `(token, k)` pairs (one host sync per layer); then for each routed expert in id order: resident -> read from its cache slot; else DMA'd into one of 8 staging slots, up to 7 ahead (`stage_one`, `src/prefill/prefill.cpp:2660-2688`; loop `:2798-2815`) |
| Stream-all walk | `m.ring > STAGE && T >= stream_all_min() && m.src` (`:1729`) | at chunk start a sequence of EVERY non-resident expert of every layer in `[LB, LE)` is built in `(layer, id)` order (`:1759-1805`); entry `k` lands in ring slot `k % ring`; the copy stream runs ahead through the attention halves instead of waiting for each layer's routing |

The sequence skips experts resident in the cache (`host_res >= 0`) and, with `--peer-device`, those the peer holds;
under peer streaming every `ps_frac`-th pinned expert goes to the peer's ring instead (see
[multi-gpu.md](multi-gpu.md)). Experts the router did not pick still occupy a ring entry; their slots are given back
without compute (`release_to`, `:2821-2830`).

`stream_all_min()` is 1,024 since 0.1.30 (was 2,048), `STRATA_PREFILL_STREAM_MIN` overrides (`:100-107`). The
comment's measurement (RTX 5070, Q2_0 / IQ2_XS, fixed cache): 1,500-token prompts 621 -> 785 / 612 -> 735 tok/s,
2,000 727 -> 934 / 712 -> 892, "the same output"; below about 1,000 tokens the output changed on Q2_0, "so 1024 is the
floor".

### Slot reuse protocol

Each ring slot has a `copied` event (recorded on the copy stream after its DMA) and a `used` event (recorded on the
compute stream once its blob has been read). Before a slot is refilled, the copy stream waits on
`used[used_of[slot]]` (`:1817`, `:1864`). `used_of` lets one event release several slots: with group gathers
(`STRATA_PREFILL_GROUP_GATHER`, default on for native packs on MMQ in the streamed walk) a 16-expert MMQ group is
gathered in one launch after one wait on the group's last copy, and its slots share one `used` record
(`:2689-2728`). The comment gives the reason: a wait and a record per expert cost about 10 us of GPU bubble each
under WDDM, "226 ms of a 32K prompt on an NVFP4 pack". A ring entry the routing skipped inside an open group first
flushes the group, so no more than a group's entries are held back from the issuer. The fused int8 path releases a
whole launch batch the same way (`:2321-2328`).

### The issuer thread (D-5)

In the stream-all walk the copies are issued by a dedicated host thread (`:1837-1882`), never by the thread that
launches kernels: it issues entry `idx` once `idx < consumed + ring`, waiting for an unpinned blob's host copy if
needed. The consumer calls `wait_issued(k)` before waiting on `copied` and `give_back(consumed)` after
(`:1884-1891`). `STRATA_PREFILL_ISSUER=0` issues inline (`issue_until`, `:1810-1832`). Same copies, same order, same
results.

### The `Stager` (unpinned blobs)

Experts in pinned host memory are DMA'd straight from the arena (counted in `experts_dma`). The rest (the share the
arena could not pin; a GGUF read in place, `ExpertSource::transient`) are first copied into a ring of pinned host
buffers by `Stager` threads (`:256-382`). Job `j` uses buffer `j % kRing`, free again once job `j - kRing`'s DMA
(recorded by `issued_one`) is done. A claim is a CAS on `gen << 32 | n << 16 | next`, so a thread late from the
previous layer cannot take a job of the next ("the expert pool's issue #29 lesson"). A transient blob is assembled
by the source itself (`copy_blob`); a failure aborts the process (`:333-336`).

Sizes (`Prefill::init`, `:769-782`): `kRing` = `STRATA_STAGER_RING` (2..256, default 16); threads =
`STRATA_STAGER_THREADS` (1..32), default `max(2, min(4, hw/4))`, or 32 with a ring of `4 x threads` when any expert of
the source is transient (a GGUF read in place, e.g. UD-Q4_K_XL beyond its RAM budget). The comment's measurement:
"32 threads and a 128-deep ring read a 4K chunk in 29 s instead of 71 s on an RTX 5070 / NVMe PC".

## Sizing the ring

`ring_slots(T)` (`:178-200`) returns `STAGE` (8) below `stream_all_min()`, else a count clamped to
`[16, ring_cap()]`, chosen in this order:

1. `STRATA_PREFILL_RING` if set (exactly 8 keeps routed-only staging even for large chunks);
2. else `set_ring_override` (a layer split without borrowing uses 96, `src/program/generate.cpp:2718`);
3. else, with `STRATA_RING_BYTES` not 0, the auto scan's byte budget (`set_ring_budget`) for chunks above the size
   0.1.39's rule would have picked;
4. else the pinned rule: 1,024 with the fused ring, 384 otherwise, 96 when less than 90% of the streamed expert bytes
   are pinned.

On HIP with `STRATA_HIP_WMMA=1` (and no override) the ring is 96 (`:180-188`; measured there on a 9070 XT: 4K
prompts 718 -> 1,211 tok/s).

`ring_cap()` (`:153`) is 1,024 for the fused Q2_0 path, else 512. `fused_ring()` (`:135-150`) is true when the fused
experts are enabled, there is no `--peer-device`, and either the pack is Q2_0 or every layer of a native pack is
covered by the fused IQ kernels (a pack with an uncovered layer keeps MMQ's buffer sizes).

Because a slot is one whole `max_blob`, the budget is kept in bytes: `ring_bytes()` (`:157-160`) is the measured slot
count (1,024 fused / 384, a quarter of that below 90% pinned) times the Q2_0 blob (1,382,400 B), and
`ring_budget_slots()` (`:172-177`) divides it by this pack's `max_blob`. The comment's motivation (`:109-120`): 384
slots are 506 MiB on Q2_0 but 1,912 MiB on a Q8_0 pack (5,222,400 B blobs), "what held that card at a 1,024-token
chunk when its own buffers would have fitted 6,144"; on the 4-way rig, "chunk 1,024 -> 6,144, prefill 87 -> 402
tok/s". The static accessors:

| Function | Line | Returns |
| --- | ---: | --- |
| `ring_default_slots()` | `:1411` | 0.1.39's ring for this PC (1,024 fused / 384 / 96), within `ring_cap()` |
| `ring_max_slots()` | `:1426` | the byte budget as slots when >= 90% pinned, else `min(96, ring_cap())` |
| `ring_cap_for(old_chunk)` | `:1416` | `ring_default_slots()` when 0.1.39's chunk was >= 6,144 (`kKeepRingChunk`), else `ring_max_slots()` |
| `ring_slots_for(chunk)` | `:1430` | what `init` will lay out (reported on the engine's INFO line) |
| `bytes_needed_no_ring(chunk)` | `:1407` | `bytes_needed` minus the ring's blobs |

Measured reasons in the comments: Q2_0 at 8,192-token chunks, 96 slots 1,153 tok/s vs 384 slots 1,294; IQ3_S with a
third unpinned, 96 slots 1,216 vs 256 slots 1,070 (host copies are the limit); fused at 512 vs 1,024 slots, 4K 1,512
/ 1,605 and 32K 2,485 / 2,660 tok/s (`:127-134`). `STRATA_RING_BYTES=0` restores 0.1.39's ring, loan count and auto
chunk list (`:164-171`).

## Choosing the chunk (`plan_lend`)

`plan_lend` (`src/program/generate.cpp:4457-4529`):

- A fixed `--prefill CHUNK`: halved until `k + 128 <= slots`.
- `--prefill auto` with `STRATA_RING_BYTES=0`: 0.1.39's list (32768, 16384, 8192, 6144, ... 256; above 8192 only when
  asked and the context allows): the first chunk whose loan fits both caps.
- `--prefill auto` (the default since 0.1.39b, #583): chunk and ring are one byte budget. It computes 0.1.39's chunk
  (`small`), the full ring `ring_max = ring_cap_for(small)`, and the bytes the cache may lend. `biggest_chunk`
  bisects on the 256-token grid up to `auto_ceiling` (8,192, or 16,384 / 32,768 when asked and the context allows,
  `:4441`) for the largest chunk whose non-ring buffers leave room for a full ring; failing that, for one with at
  least 16 slots (`kRingMin`). If the scan buys nothing over `small`, 0.1.39's choice and ring stand. Otherwise
  `set_ring_budget(room, small)` hands the ring the remaining bytes for chunks above `small`.

The comments record why (`:4475-4485`): on a 4-way IQ3_S rig a ring slot was worth about 0.53 tok/s against 0.05 for
a chunk token. `docs/DETAILS.md` ("How it works", "Prompts") gives the user-facing summary and the RTX 5070
measurements (IQ3_XXS 32K +18.5% at a 1,500-slot cache, the Coder +3%, IQ3_S unchanged, 4K unchanged).

## Statistics

`PrefillStats::experts_streamed`, `experts_dma`, `ms_experts_host`, `experts_resident` (counted per expert-layer
group read from VRAM; in the fused walk counted whether routed or not). The one-shot path prints them on its
`strata generate: prefill ... experts streamed N (M by DMA, host X ms), resident R` line; `STRATA_PREFILL_TIMING=1`
shows the "wait copy" phase.

## Configuration

| Knob | Effect |
| --- | --- |
| `--prefill CHUNK|auto|auto:N`, `STRATA_PREFILL_AUTO_MAX` | chunk choice |
| `--no-prefill-borrow` | own buffers instead of a loan |
| `STRATA_PREFILL_LEND_PCT` | auto lend cap in percent |
| `STRATA_PREFILL_STREAM_MIN` | chunk size from which the stream-all walk runs (default 1,024) |
| `STRATA_PREFILL_RING` | ring slots |
| `STRATA_RING_BYTES=0` | 0.1.39's ring, loan and chunk list |
| `STRATA_PREFILL_ISSUER=0` | issue copies inline |
| `STRATA_PREFILL_GROUP_GATHER=0` | one gather / wait / record per expert |
| `STRATA_STAGER_RING`, `STRATA_STAGER_THREADS` | unpinned-blob staging |
| `STRATA_GROUP_COPY=1` | grouping tables via `cudaMemcpyAsync` instead of mapped memory (A/B) |
| `STRATA_REFILL_BLOCKING=1`, `STRATA_REFILL_SERIAL=1` | refill lent slots one blocking copy at a time / stage after stage |
| `STRATA_KV_STAGE_OWN` | KV streaming's staging pool in its own allocation instead of borrowed slots (A/B) |
| `STRATA_SPLIT_SMALL_OWN`, `STRATA_SPLIT_SMALL_MAX` | layer split: keep a small chunk's buffers per stage |
| `STRATA_HIP_WMMA=1` | HIP: 96-slot ring |

## Gotchas

- Lent slots hold prompt buffers, not experts, until refilled; in serve only the lending participant's loan is
  returned (`refill_one`), never another stage's (`src/program/generate.cpp:6813-6816`).
- With `STRATA_KV_STAGE_OWN` unset, KV streaming's staging pool is also carved from the loan.
- A ring smaller than the per-batch need of the fused path waits for slots its own first batch frees; that is why the
  fused path uses up to 1,024 slots.

## In UnbiasedStrata

Kept; the mechanism is unchanged apart from widths taken from the model geometry (commit e41dfc7). See
[../../codebase/prefill/expert-streaming.md](../../codebase/prefill/expert-streaming.md).

## See also

[pipeline.md](pipeline.md), [moe.md](moe.md), [multi-gpu.md](multi-gpu.md),
[docs/upstream/DETAILS.md](../../upstream/DETAILS.md). The expert cache, the residency table and `ExpertSource`
(`blob`, `pinned`, `transient`, `copy_blob`) belong to the core area.
