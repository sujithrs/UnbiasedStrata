[Upstream docs](../README.md) › [Program](README.md) › Expert tiers

# Expert tiers as the driver wires them

**Purpose:** where each routed expert's bytes live and who computes it. An expert is computed in one of these places:

- a VRAM slot on CUDA0 (the R4 hit path);
- a VRAM slot on a helper GPU (CUDA1..3), on the peer GPU, or on a layer-split stage;
- over PCIe by the GPU, for a share of the misses;
- by the CPU expert pool, reading RAM (the pinned arena or the resident copy) or the model files (mmap).

This page covers how `main` builds these tiers and how the serve loop moves experts between them. The engine-side
mechanics belong to the core library (`ExpertCache`, `FileExpertSource`, `ArenaExpertSource`, `RemoteExperts`,
`PeerExperts`).

**Covers:** `src/program/generate.cpp:3056-3746` (sources, cache, helpers, hit path), `:4263-4332` (the residency
table), `:4379-4606` (prompt-path lending, resident RAM mode), `:5434-5589` and `:7856-7971` (adaptive swaps),
`:5760-5878` (`VRAM`).

## Expert source (`:3118-3189`)

| Mode | Chosen by | What it does |
| --- | --- | --- |
| Resident arena (`ArenaExpertSource`) | default | Copies all experts into anonymous host memory with 6 loader threads and pins as much as the driver allows. The comment's measurement: 19.41 ms/token pool time against 34.78 (warm) or 71.97 (cold) through mmap (`:3058-3066`). `--shared-expert-arena FILE` backs it with a `MAP_SHARED` file (Linux). A native pack may take its experts from GGUF shard 1 (`set_gguf`). Pinning limit: `STRATA_ARENA_PIN_GIB` (read by `arena_pin_cap_gib()`), else 8 GiB for multi-GPU under WDDM, else no limit (`:3153-3161`). It logs the load rate and any RAM warning (#633) |
| Mapped file (`FileExpertSource`) | `--mmap-experts` (also set by `--resident-experts` and `--resident-budget-gib`) | Maps `experts.bin`. A native pack without `experts.bin` maps the GGUF shards in place, using `native_experts.txt`'s spans ("CS-T", `:3125-3127`). `STRATA_FETCH_THREADS` sets its fetch threads. With a RAM budget or `STRATA_UNBUFFERED_LOAD`, the file tier may read unbuffered (`set_unbuffered`, `:3139-3144`) |

`ExpertPool pool(workers, pin=true, host_works=!--no-host-worker, affinity)` (`:3190`) runs the CPU experts. On a
hybrid CPU with a non-default affinity, it logs the P-core and E-core counts.

## VRAM expert cache sizing (`:3198-3449`)

1. **`--expert-cache auto`** (or 0 with a profile): slots = (free VRAM - reserve) / `max_blob`, capped at the number of
   profile pairs (`:3206-3225`). The reserve is `--vram-reserve-mib` (700) plus the prompt path's buffers when they
   are not borrowed (`160 + chunk * 680 / 1024` MiB) plus the draft head's binding bytes (`mtp.bind_bytes`, #199).
2. **Small cards (#496):** a working cache needs `min_slots`: the buffers of a 256-token chunk plus 128 slots when the
   prompt path borrows, else 1. If the default reserve leaves fewer slots, the reserve shrinks toward 300 MiB
   (`kSmallReserveMib`). An explicit `--vram-reserve-mib` is never shrunk. If no slot fits, the engine names what
   is short and what makes room: a smaller context, `--kv q4_0`, the English draft subset, a smaller reserve
   (`:3236-3275`).
3. **Layer split with an explicit size:** the size is lowered so the reserve and the prompt buffers fit, with a
   WARNING (`:3276-3293`).
4. **Sized slots** (native pack, profile, not per-layer): each slot is cut to its pair's blob size, rounded to 256 B,
   until the budget is used. The comment says this holds about 30% more IQ3_XXS experts in the same VRAM
   (`:3294-3314`).
5. **`--vram-elastic`** (serve, one GPU, no helpers, segments of at least 64 MiB, NVIDIA): `set_segment_bytes`
   (`:3315-3332`).
6. **Open and shrink loop** (`:3333-3423`): `open` or `open_sized`. On failure, an automatic cache retries at 3/4 of
   the size up to 8 times (#60, Windows commit). An automatic cache then zeroes the slots, reads the free figure
   again, and reopens smaller while it is below the reserve minus 64 MiB. Under WDDM the figure read before the
   allocation can be about 1 GB too high. `STRATA_TEST_CACHE_FAIL=N` fakes N failures to test the retry.

The log then states the policy:

- `PER-LAYER` (`--expert-cache-per-layer`);
- `compulsory-miss` (no profile: the cache fills with whatever is routed first);
- `PROFILE, ranked by routing frequency, no eviction`.

It also notes that GPU and CPU round differently, so a reply can differ slightly from a run without the cache. The
comment cites a teacher-forced measurement on 2,557 tokens: 95-98% the same top-1 token and equal perplexity
(`:3429-3438`, bench/results/2026-09-27-cache-parity).

## Profile fill (`:3450-3496`)

The pairs of `--expert-profile` are admitted in rank order. Each slot is filled with `fill_slot_blocking` from
`srcp->blob(layer, expert)`. With per-layer admission, a full layer skips only its own pairs (#369). An unbuffered
file tier reads the pairs in batches of 64, the next batch in the background (#286). One slot is then read back and
compared (`verify_slot`). A residency table that is right about indices but wrong about bytes would give plausible
but wrong tokens.

With `--peer-device`, `STRATA_PEER_HOT=f` (default 0.45) moves a share f of the first `STRATA_PEER_HOT_AT` (default
8700) ranked pairs behind the primary's fill point, so the peer gets hot experts too (`:2654-2683`).

## The R4 hit path and the residency table

- **Hit path (`:3698-3746`)**, with the cache on and the pool on: device buffers for the slot and destination lists,
  the Q8 activation codes and their **fp32** scales (R4.2h: fp16 scales disagreed with the CPU by 4.761e-04), and
  the hit outputs. `drive.d.cache*` point at the cache. `hit_poke` is on unless `--no-hit-poke`.
- **Residency table (`:4263-4332`)**: with a profile-filled cache and the token graph possible, `host_res` holds each
  pair's slot. With a split, the slot is in the cache of the stage that owns the layer. The table is copied to the
  device (`d_res`) and to every stage. Then two passes run over the arena:
  - the experts a GPU holds are handed back to the OS (`srcp->release`, skipped with `STRATA_ARENA_RELEASE=0`);
  - the rest get prefetch hints.

  `rss_probe` logs `RssFile` around both passes with `STRATA_RSS_TRACE`.
- **Routing-aware prefetch (`:3672-3697`)**: when the file tier warms pages, every layer's BF16 router is copied to
  the host. `RouterLookahead` then predicts the next layer's experts from this layer's MoE input and warms their
  pages. `STRATA_LOOKAHEAD=0` turns it off; `STRATA_LOOKAHEAD_K` sets the experts per token (default 10).

## Helper caches on CUDA1..3 (`:3554-3640`)

`--expert-cache-device1..3 N|auto` (upstream
[docs/SECOND_GPU.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/SECOND_GPU.md)).
They need `--expert-profile`, a CUDA0 cache and the pool.

- The ranking is the profile; with a split, the later stages' pairs no stage cache holds are appended.
- With several helpers or `auto`, every unranked pair is appended in expert-then-layer order, so the tail is spread
  over all layers.
- Pairs are striped across the helpers or grouped by layer (`--expert-cache-remote-placement`).
- `RemoteExperts::open` fills each helper and skips pairs a stage already holds (`claimed`).
- Results come back through pinned host rows.
- `--remote-expert-opt` attaches a `RemoteExpertOpt`, which takes part in the adaptive tier.
- Contexts: CUDA1 is created before CUDA0's weights (`:2022-2032`); CUDA2 and CUDA3 are created after the drafter
  and head, unless `STRATA_EARLY_REMOTE_CONTEXTS=1` (needed under WSL2, where the mapped-host budget runs out).

## Peer GPU (`:3642-3659`)

`PeerExperts::open(peer_device, profile, xcache, src, ..., peer_reserve_mib, peer_slots)` fills the second card with
the ranked pairs the primary does not hold, then logs both counts against `n_layers x n_expert`. In serve mode, the
prompt path hands the peer `--peer-prefill-rows` rows per layer (`sp.set_peer`, `:4989-4995`). The adaptive tier
gives the peer the next most-routed CPU misses (`peer.adapt`, `:5560-5569`).

## Lending cache slots to the prompt path

The batched prompt path needs device buffers for a chunk. By default it **borrows the last slots of the expert
cache**: those experts become non-resident during the prompt and are streamed like misses. The slots are then
refilled from the arena (`fill_slot_queued`, one wait; `STRATA_REFILL_BLOCKING=1` waits on each).

- `plan_lend` (`:4457-4532`) picks the chunk for one GPU. With `--prefill auto` (since 0.1.39b, #583), it bisects on
  a 256-token grid (`biggest_chunk`) for the largest chunk that still leaves the prompt path's streamed ring at its
  full byte budget. A loan must keep 128 slots and lend at most `kAutoLendPct` % of the slots: 90 when the arena is
  at least 90% pinned, else 85, or `STRATA_PREFILL_LEND_PCT`. `STRATA_RING_BYTES=0` (read in `prefill.cpp`) restores
  0.1.39's fixed list `{32768, ..., 256}`. An explicit chunk is halved until it fits.
- The serve path does the same per participant (`PfPart`), one per stage cache ([serve-mode.md](serve-mode.md),
  [layer-split.md](layer-split.md)).
- `Prefill::set_pinned_share` (`:4379-4389`) passes on the pinned share of the expert bytes, which sizes the ring and
  the lend cap.

## Resident RAM mode (`:4533-4606`)

With `--resident-cpu-experts` (also set by `--resident-experts` and `--resident-budget-gib`),
`src.pin_cache_complement` copies into RAM the experts the GPU cache does not hold, plus the lend region as far as RAM
allows. It respects the headroom and, with a budget, the hottest experts by profile.

- The soft mode (#467): if the whole set does not fit, it keeps what fits (`kResidentWhatFits`) with a WARNING; if
  even that fails, plain mmap.
- A budget that cannot be kept falls back to the files (#403).
- Otherwise exit 1.
- Adaptive swaps then **exchange** experts instead of reading the file: `reserve_exchanges(min(adapt_swaps, 96))`.
  `resident_stage_swaps` (`:212-241`) copies the outgoing expert out of its VRAM slot into an exchange buffer, and
  `commit_exchanges` puts it in RAM once the swap lands.
- Windows: after the copy is built, `recheck_unbuffered` decides the file tier's buffering again (#577).

## Adaptive swaps (the VRAM tier follows the conversation)

The same rule runs in serve (`adapt`, `:5491-5572`) and in the one-shot speculative loop (`:7898-7971`). Every
`--adapt-every` verify rounds it runs on a thread beside the commit and the draft:

1. Per layer, the candidates are missing experts routed at least 2.0 times (decayed counts in `drive.d.usage`). In
   serve, experts the peer or the helper optimisation already owns are excluded. The victims are resident experts,
   least-routed first.
2. A pair is swapped when the candidate's count is at least the victim's plus 1.5. The pairs are sorted by gain and
   capped at `--adapt-swaps`.
3. Resident RAM mode exchanges are staged first. With `pin_blobs_on()` (AMD, Linux, `STRATA_ARENA_MMAP=1`) the source
   pages are locked.
4. The copies are issued with `cudaMemcpyAsync` on the adapt stream: on CUDA0, or on the owning stage's stream with a
   split. The victim becomes non-resident at once (the CPU computes it meanwhile) and its pages get a prefetch. The
   newcomer goes into `pending`.
5. `apply_pending` admits `pending` once the event has completed, releases the newcomers' RAM pages and uploads the
   table. Since #463, the next window **waits** for the copies (`STRATA_ADAPT_NOWAIT=1` restores the non-blocking
   query of 0.1.37). Whether a swapped-in expert runs on the GPU or the CPU changes its rounding.
6. Counts are multiplied by `--adapt-decay`. With `--expert-profile-save`, the counts also accumulate into `heat`. In
   serve, the peer then adapts too.

`STRATA_TRACE_ADAPT=1` (one-shot loop) logs why each round swapped or skipped, and whether the copies had landed.

## Saving what was learned (`--expert-profile-save`, #477)

`save_profile` (`:5575-5589`) ranks the resident experts first, then the accumulated routing (`rank_learned_profile`,
with the loaded profile as prior), and writes a profile with `write_expert_profile`. It runs on `QUIT` or end of
stdin, and before a request once `--expert-profile-save-every` minutes have passed. It needs the adaptive tier and a
profile, else the engine says nothing will be saved.

## Giving VRAM back: the `VRAM` command (#533, `:5760-5878`)

Only with `--vram-elastic` and between requests. It is refused while batch slots decode, with the resident RAM mode,
with a peer, or while a prompt loan is out.

- **Shrink:** when less than `<reserve_mib>` is free, the cache gives back its last segments. It keeps at least 128
  slots plus a 256-token loan. Experts in the lost slots become CPU misses and are remembered in `vram_evicted`.
- **Grow:** when more is free, the segments are mapped again and each slot gets its old expert back. If the adaptive
  tier already brought that expert back, the layer's most-routed missing expert takes the slot.
- `relend` moves the prompt path's loan to the end of what is left, with a smaller chunk if needed.
- The reply is `VRAM reserve_mib=.. expert_slots=.. expert_slots_full=.. expert_cache_mib=.. expert_cache_full_mib=..
  vram_free_mib=.. prompt_chunk=..`. A bare `VRAM` uses the start-up reserve.

## Platform and model notes

- Native packs have per-layer blob sizes, so the code converts slots to bytes with `slot_offsets()` wherever it
  matters (`slots_from_bytes`, `bytes_from_slots`, `cache_slots_for`).
- With a layer split, the pinned-arena limit under WDDM, `STRATA_SPLIT_RING` and the per-stage loans apply; see
  [layer-split.md](layer-split.md).
- HIP: `pin_blobs` and the `GPU_PINNED_MIN_XFER_SIZE` setting exist for file-backed arenas on AMD. `--vram-elastic` is
  NVIDIA-only.

## Tests

The driver's wiring has no unit test. The core library's own tests cover the cache, the sources and the helpers.
`serve/test_vram.py` tests the server side of the `VRAM` command against a mocked engine.

## In UnbiasedStrata

Kept, with these changes: the fork's VRAM plan can choose the resident RAM tier ("the RAM tier chosen by the plan",
fa69e82), and the free-VRAM figures go through `vram_info` so that `--vram-cap` can clamp them. See
[../../codebase/program/expert-tiers.md](../../codebase/program/expert-tiers.md).

## See also

[startup.md](startup.md), [layer-split.md](layer-split.md), [serve-mode.md](serve-mode.md),
[env-vars.md](env-vars.md).
