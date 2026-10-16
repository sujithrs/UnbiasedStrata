[Upstream docs](../README.md) › [Program](README.md) › Layer split

# Multi-GPU layer split (`--layer-split`)

**Purpose:** run the 48 layers on several GPUs. CUDA0 runs `[0, K1)`, the next GPU `[K1, K2)`, and so on. Each GPU has
its own copy of the dense weights (or only its own layers' with trimming), a session for its layer range, an expert
cache filled from its share of the profile, a verifier and a prompt path. The last stage also holds the head and the
MTP drafter. The user guide is upstream
[docs/MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MULTI_GPU.md).
This is one of three multi-GPU options. The others are helper expert caches (`--expert-cache-device1..3`) and a peer
GPU (`--peer-device`), both on [expert-tiers.md](expert-tiers.md).

**Covers:** option rules `src/program/generate.cpp:1537-1706`; `SplitDrive` and `GpuStage` (`:806-852`); stage loading
(`:2508-2640`); placement (`:2691-2864`); sessions (`:2866-2927`); stage caches (`:3498-3552`); serve-side loans,
hand-offs and checkpoints (`:4618-5046`, `:5068-5132`, `:5319-5433`).

## Rules (`:1549-1597`)

- `--layer-split K[,K2,..]|auto` needs `--serve`. Each K is the **first layer** of a later GPU, rising, from 2. For
  four GPUs that is `24,36,42`, not `24,12,6,6`; the error message says so.
- `--split-device D1,D2,..` names the later stages' GPUs. They must be distinct and in `1..n_visible-1`. Default: the
  next visible GPUs. `--split-device 0` with one K (`split_same`) runs both stages on CUDA0, sharing weights, session
  and cache. That is the bit-exact A/B of the hand-off; it refuses conversation parking.
- At most `SplitDrive::kMax - 1` = 7 later stages.
- `auto` with only one GPU visible: no split, with a message.
- A split across GPUs (`multi_gpu`) needs `--expert-profile` (`:1629-1632`). It refuses `--gpu-only-full` and
  `--gpu-stages` (`:1929-1933`), `--peer-device` (`:1911-1916`) and an explicit `--resident-cpu-experts`. Setup's
  `--resident-experts` falls back to plain mmap.
- Helper caches with a split go to the visible GPUs no stage uses, in order. If none is left: exit 2 (`:1687-1701`).
- Not yet across GPUs, per the comment at `:1538-1542`: control vectors, the helper caches, and lending cache slots to
  the prompt path. The code below replicates the control vector to each stage (`cvec_replicate`, `:2611`) and gives
  each stage its own loan.

## Own prompt buffers or loans (`STRATA_SPLIT_OWN`, #340, `:1633-1686`)

A stage's prompt path borrows from that **stage's own** cache by default. Opt-in `STRATA_SPLIT_OWN`:

- `1`: every stage keeps its own buffers, for 2048-token chunks or the `--prefill` chunk;
- `auto`: own buffers only when they take at most 12% of every card's VRAM (16 GB cards and larger);
- unset or `0`: borrow.

The reserve priced for own buffers is `160 + chunk * 680 / 1024 + 96 * 4` MiB. Own buffers change which experts a
full card keeps, so the output differs from borrowing. The comment's measurement: 2x RX 9070 XT / R9700, 2K 1588
tok/s, 16K 2006 tok/s with own buffers. `--split-skip-if-fits` keeps the loans.

## Loading the stages (`:2555-2640`)

For each later stage:

1. `RemoteExperts::preflight` creates its context.
2. Its weight arena and `WeightTable` are loaded. With `stage_trim`, only its own layers' dense weights are loaded
   (PLE tensors stay everywhere). Trimming needs an explicit split (not `auto`) and `--trim-stage-weights` or
   `STRATA_STAGE_TRIM=1`; the comment measured it on 2x MI50 (PR #639).
3. Its native dense projections, two streams (main and adapt) and an event are set up.
4. The control vector tables are replicated, and with `--vision` the M-RoPE table is too.
5. Its own PCIe probe sets `st.pcie_frac`, unless `--pcie-frac` was given; then one share is used for every stage
   (#485).

Sessions wait for the placement, because a session is priced by its layer range.

## `--split-skip-if-fits` (`:2508-2553`)

With `auto`, a profile and no helper caches, the engine sums the profile's pair bytes. If CUDA0's free VRAM, minus
the whole-model session, the reserve, 1000 MiB for the drafter and head, and 96 MiB for the windows, holds them all,
it clears the split and runs on one GPU. The comment cites two RDNA4 cards: 1,384 tok/s split against 1,794 tok/s
alone at 4K.

## Automatic placement (`:2691-2840`)

A cost model of one decode window, fitted on the 5080 + 3090 rig (bench/results/2026-09-29-layer-split):

- a layer costs `0.33 * (84 * 2.617) / (SMs x GHz)` ms on its GPU (0.33 ms on an RTX 5080);
- an expert no cache holds costs `STRATA_SPLIT_MISS_MS` (default 190) ms per unit of routed mass;
- the routed mass of profile rank r is `(r+1)^-1.2`;
- each stage's cache takes its layers' profiled pairs, hottest first, until its room is full. The room is the free
  VRAM minus the reserve, the prompt buffers (zero when borrowing), 96 MiB of windows on later stages, 1000 MiB for
  the drafter on the last stage (`stage_room`, `:2739-2749`), and the session for its layer range (`session_bytes`,
  pure arithmetic).

For 2 and 3 GPUs every placement is tried; with more, the layers are shared in proportion to speed. The log prints
the predicted window time and the routed mass the caches hold. With own buffers chosen by `auto`, the search ignores
their reserve, unless `STRATA_SPLIT_OWN_PLACE=reserve`. The comment: K=28 decoded 58.4 tok/s with own buffers and
58.5 borrowing, while placing with the reserve moved K to 32 and decode to 54.8.

After the search, the profile is divided by stage (`stage_of`), and each stage records `[lb, le)`.

## Sessions and caches

- **Sessions** (`:2866-2927`): CUDA0's covers `[0, K1)`; each stage's covers its range (the carve). The last stage
  also loads the native head. With `--batch`, every stage carves the same number of slot sessions.
- **Stage caches** (`:3498-3552`): each stage's room comes from `stage_room` after the drafter and head are loaded.
  Slots are sized per pair for a native pack, else `max_blob`. They are filled from the stage's profile share and one
  slot is verified. The log prints layers, slots, GiB and pairs.
- **Residency table**: one `host_res` for all layers. A value is a slot index into the cache of the stage that owns
  the layer, and each stage keeps a device copy (`st->d_res`). Adaptive swaps copy into the owning stage's cache on
  its adapt stream (`:5532-5544`).

## Serve-side plumbing

- **Verifiers and hand-offs (`:5068-5132`)**: one mapped, portable pinned buffer per boundary, sized
  `kVerifyMaxT * Verifier::handoff_floats(g)` floats. `stage_ver(st).set_stage(lb, le, in, out)` and `set_next` chain
  the verifiers. The verify windows call `drive_pool_split` (`:818-829`), which points the shared `Drive` at the
  layer's stage: its GPU plan sink, cache base, slot offsets and PCIe share. The log reads "layers 0-23 (CUDA0),
  24-47 (CUDA1), one hand-off per window".
- **Commit**: `Verifier::set_commit_async(!multi_gpu)` (`:2554`): with a split, the commit is synchronous.
- **Prompt paths (`:4884-5008`)**: `st.sp.set_stage(lb, le, next)` makes a pipeline; each stage's path is initialised
  with its own loan (`PfPart`) or its own buffers. A chunk that does not fit steps down through
  `{6144, 4096, 3072, 2048, 1536, 1024, 512}` instead of exiting. Each path gets the next stage's GPU as a helper for
  a one-chunk prompt (`set_stage_helper`).
- **Small-card warning (#448)**: when one stage's cache cannot fund the chunk CUDA0 alone would read in, the log names
  the card and suggests using it as a helper cache instead.
- **Own buffers per stage at serve time**: a stage whose card has the buffers plus 1.5 GiB free keeps its own and
  borrows nothing (`STRATA_SPLIT_OWN_BUFFERS=0` turns this off, `:4823-4847`).
- **Streamed ring**: with at least 75% of the pairs resident, the prompt path's ring is 96 slots
  (`STRATA_SPLIT_RING=N` overrides; `:4660-4675`).
- **`STRATA_SPLIT_SMALL_OWN=S` / `STRATA_SPLIT_SMALL_MAX=M` (`:5010-5046`)**: every borrowing stage keeps S-token
  buffers carved from its cache tail for the whole session. Requests up to M tokens then read in S-token chunks and
  lend nothing.
- **Refill (#340)**: every stage's refill copies are issued first, then each is waited for. `STRATA_REFILL_SERIAL=1`
  goes stage after stage.
- **Checkpoints**: stages run a chunk apart, so each stage saves its own part at a checkpoint position
  (`on_stage_chunk`), and the last stage joins them (`part_at`, `:5319-5433`). Parking saves one image per stage; the
  draft layer's K/V is saved with the last stage's image.
- **`--batch-groups G`**: the slots in G groups pipelined through the stages (`pump`, [serve-mode.md](serve-mode.md)).
- **`STRATA_SPLIT_TIMING`** prints each stage's host time per window after each request.
- **PCIe share per request**: a request's `pcie_frac=` key sets CUDA0's share; later stages keep their own probe's
  share unless the request sets one (`:6957-6959`).

## Platform notes

Under WDDM (Windows, WSL2), a multi-GPU run pins at most 8 GiB of the expert arena (`:3153-3161`): pinning everything
into two contexts left WDDM refusing later allocations on the 5080 + 3090 rig. Linux pins the whole arena (#253). HIP
checks that every stage's GPU architecture is in the binary (`:1707-1722`).

## In UnbiasedStrata

The logic is kept. The fork's VRAM plan and `--vram-cap` change the free figures the placement reads (`vram_info`).
See [../../codebase/program/layer-split.md](../../codebase/program/layer-split.md).

## See also

[expert-tiers.md](expert-tiers.md), [serve-mode.md](serve-mode.md), [prompt-cache.md](prompt-cache.md).
