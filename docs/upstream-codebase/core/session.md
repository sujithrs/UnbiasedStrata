[Upstream docs](../README.md) › [Core](README.md) › Session and decode drivers

# The session state and the decode drivers

## Purpose

`SessionState` is everything one sequence needs that is not a weight: per-layer GDN state, per-QSA-layer KV and
indexer state, the shared scratch of the layer functions, the hyper-connection residual, the PLE history and token
window, and a pointer to the doorbell. `session.cpp` sizes and carves it from one device allocation, zeroes it, and
runs one token through all 48 layers in three ways: direct launches, per-layer graphs with a host loop, or one
graph per token. The speculative verify window ([verify.md](verify.md)), the prompt path (`src/prefill/`) and the
conversation cache ([conversation/](conversation/README.md)) work on the same `SessionState`.

Files covered:
[include/strata/core/session.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/session.hpp),
[src/core/session.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/session.cpp),
[include/strata/core/hit_hook.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/hit_hook.hpp),
[include/strata/core/progress.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/progress.hpp).

## `SessionState` (`session.hpp:31-89`)

| Field | Meaning |
| --- | --- |
| `max_cells` | Context capacity (`--max-context`, default 4096 in `generate`; setup passes the chosen context) |
| `gdn`, `gdn_state` | Shared GDN scratch; `gdn_alloc` rows of per-layer recurrent state (`128 * 48 * 128` floats) plus conv state (`10240 * 3` floats) |
| `qsa_states`, `qsa_bufs` | One `QsaState` per QSA layer (array of 12, only the owned range initialised); shared QSA scratch |
| `moe`, `block`, `R` | MoE buffers, block buffers; `R` aliases `block.R` (4 x 2560 floats) |
| `k` | Experts per token (10; the Coder's GGUF also says 10) |
| `layer_lo`, `layer_hi`, `qsa_ord0`, `qsa_alloc`, `gdn_ord0`, `gdn_alloc` | The **layer-range carve** for a multi-GPU stage: only `[layer_lo, layer_hi)` gets state, indexed by global ordinals; `qsa_primary()` is the first owned QSA state, which owns the RoPE table |
| `db` | The doorbell the host loop polls (null until set) |
| `ple_prev`, `ple_token`, `ple_hist`, `ple` | PLE sequence state: two predecessor tokens (oldest first, -1 = none), the current token, the normalized history, and the whole `PleRun` |

## Sizing and lifetime

| Function | Lines | What it does |
| --- | --- | --- |
| `session_bytes(g, max_cells, k, lo, hi)` | `session.cpp:54-73` | Pure arithmetic: GDN scratch + owned GDN rows + one QSA state with the RoPE table and `q_n - 1` without + QSA scratch + MoE + block + PLE history, rounded, + 4096. Safe to call for candidate ranges (the layer-split placement search does, `generate.cpp:2784`) |
| `session_init(g, max_cells, k, base, s, lo, hi)` | `:75-129` | Carves `base` (device memory). Returns 0 when a QSA state cannot be built, which happens when the KV streaming host copy cannot be pinned (`:112-118`) |
| `session_release(s)` | `:131-137` | Before freeing: unregisters the RoPE table from the rope kernels (#280) |
| `session_zero(s, g, R_init, stream)` | `:139-165` | Residual to `hc` copies of `R_init` (or zero), every owned GDN and QSA state to zero, the PLE history zeroed and the token window reset |
| `gdn_point_at(g, layer, s)` | `:167-175` | Points the shared GDN scratch at layer `l`'s own state row |
| `stage_token(g, pos, pos_base, s)` | `:177-189` | Writes each owned QSA state's pinned step and position staging; must run before every replay |

`generate.cpp:2866-2890` allocates and initialises CUDA0's session (owning `[0, split_at[0])` under a layer split),
reports KV streaming, wires `ss.ple.hist`, and zeroes it. Each later stage gets its own session for its range
(`:2908-2909`); `--batch` slots get sessions of the same shape (`:2961-2977`).

## The three decode drivers

`generate.cpp:7636-7655` chooses, per plain decode token on the canonical pack:

```
--no-capture            -> session_token        (direct launches; --sync-every-layer for debugging)
token graph captured    -> session_run_token    (one graph for all 48 layers)
otherwise               -> session_loop         (two graphs per layer, the host loop in between)
```

A native pack captures neither graph form (`generate.cpp:3822`, `:4333-4334`) and decodes through verify windows
only; so does `--serve` with `--spec` / `--mtp` on any pack.

### `session_token` (`session.cpp:781-827`)

All 48 layers as `block_layer` calls with a caller-filled `parts`. `sync_every_layer` synchronises after every layer
to surface an asynchronous fault at the layer that caused it.

### `session_loop` (`session.cpp:568-779`): per-layer graphs and the host loop

```
stage_token; parts <- y_miss (zeros); doorbell_reset; launch pre[0]; record probe event
for l in 0..47:
    spin: cudaEventQuery(probe) + volatile read of h_seq until the ring for l is seen
    hits(user, stream, Launch, h_ids, k)        [VRAM experts: quantize + grouped kernel, async]
    pool(user, h_x_f, h_ids, h_weights, ...)    [CPU experts into pinned y_miss]
    cudaMemcpyAsync(parts_dev <- y_miss)
    hits(user, stream, Combine, ...)            [parts += hit_out]
    launch post[l]; [dump R]; [!overlap: sync]
    launch pre[l+1]; record probe
final cudaStreamSynchronize
```

* The per-iteration `cudaEventQuery` is required: the comment at `session.cpp:674-681` records three experiments
  where throttling it broke tests, because the device's write to mapped memory became host-visible only when the
  driver was entered.
* `rings_mid_graph`, `ms_to_ring`, `ms_host` and `calls_total` in `SessionGraphs` measure whether the ring was seen
  while the graph still ran, and where the host round trip goes (`session.hpp:156-225`).
* `SessionLoopScratch` (`session.hpp:369-380`, `session.cpp:524-566`): the pinned, mapped `y_miss` staging and the
  probe event, allocated once per session (passing null makes the loop allocate per call, and `cudaFreeHost` then
  synchronises the device every token). `init` also pins the host thread to the first physical core, so the spin does
  not land on a pool worker's core.
* `PoolFn` (`session.hpp:314-315`) and `HitFn` / `HitPhase` (`hit_hook.hpp:28-39`) are the callbacks; the expert side
  implements them as `expert_pool_dispatch` and `expert_hit_run` ([experts/dispatch.md](experts/dispatch.md)). The
  hook's two phases exist so the GPU's resident experts run during the CPU pool, not after it (`hit_hook.hpp:13-27`).

### `session_run_token` (`session.cpp:903-963`): the whole token as one graph

`session_capture_token` (`:829-901`) records, for every layer: `pre[l]`; optionally the device-side hit path
(`moe_hit_select` against a static residency table, `quantize_q8_0_scaled`, `moe_hit_grouped_s2_dev`, when
`TokenHits` is complete); `doorbell_wait` (a one-thread kernel spinning until `h_flag` equals the ring);
`copy_from_mapped` of the pool's answer into `parts` (a kernel, not a memcpy node, because a copy-engine node split
the WDDM submission: 67 flushes per token, `:874`); `moe_hit_add`; `post[l]`.

`session_run_token` launches it once, then for each layer spins on `h_seq` (with `_mm_pause`), runs the pool and
writes `h_flag`. Every 1024 spins it checks the clock: after `STRATA_TG_FLUSH_US` microseconds (default 2000) it calls
`cudaStreamQuery` once to flush submission and detect a dead graph, and after 20 s it fails (`:914-945`). It reports
progress through `progress_at` for the serve watchdog. The header (`session.hpp:386-398`) gives the motivation: under
WDDM each graph launch cost 0.3-0.4 ms, and the per-layer loop made 96 per token (48.6 ms/token measured for 12 ms of
GPU kernels and 27 ms of pool).

Capture conditions (`generate.cpp:4333-4334`): not `--no-capture`, not `--no-token-graph`, no dumps, the hit hook
either absent or fully on the device, not a native pack, not a layer split.

### Measurement replays

`session_replay` (pre graphs only), `session_replay_full` (pre and post, the true GPU floor),
`session_replay_stages`, `session_replay_stages_per_layer`, `session_replay_stage_prefixes` and
`session_replay_stage_sweep` (`session.cpp:278-485`) time captured stages from outside the capture. They need
`session_capture(..., split = true)` (`--gpu-stages`, `generate.cpp:3827`). See [cuda-graphs.md](cuda-graphs.md).

## Watchdog progress (`progress.hpp`)

`Progress` (`progress.hpp:17-25`) is a process-wide heartbeat: `beats`, `busy`, `where`, `detail`, `chunk`,
`since_ms`, `ticks`. `progress_at` / `progress_beat` / `progress_tick` are relaxed atomic stores. `--serve` runs a
watchdog that ends the engine when a request's heartbeat stops for `STRATA_WATCHDOG_S` seconds (default 60, 0 = off,
`generate.cpp:5717-5720`); `diag_pool_fn`, `diag_verify_fn` print diagnostics then, and `release_gpu_fn` first
releases the verify window's GPU spin waits (#267).

## Configuration

| Flag / env | Effect |
| --- | --- |
| `--max-context N` | `max_cells` |
| `--kv`, `--kv-resident N` | KV format and streaming ([layer-forward.md](layer-forward.md)) |
| `--no-capture`, `--sync-every-layer` | Direct launches; per-layer sync |
| `--no-token-graph` | Two graphs per layer instead of one per token |
| `--gpu-stages` | Capture the split measurement graphs |
| `STRATA_TG_FLUSH_US` | Token graph: microseconds between submission flushes while a ring is late (default 2000) |
| `STRATA_WATCHDOG_S` | Serve watchdog timeout |

## Platform notes

* The spin uses `_mm_pause` only on x86 / MSVC (`session.cpp:21-29`); `session_run_token` calls `_mm_sfence`
  unconditionally (`:951`), so this file assumes an x86 host.
* The WDDM measurements behind the token graph are Windows-specific; the same code runs on Linux and on HIP.
* `session_capture_token` indexes `qsa_states[0]` and walks every layer, so it is whole-model only; a layer split
  never captures it.

## Tests

No unit test of `session.cpp` in this snapshot. `graph_registry_test` and `pinned_capture` (`CMakeLists.txt:858-878`)
are registered only when `bench/micro/*.cu` exists, and `bench/micro/` is not in this snapshot.

## Gotchas

* The QSA position is data: forgetting `stage_token` before a replay replays position 0 (the bug recorded at
  `session.cpp:582-592`).
* `session_loop` refuses to run without a doorbell or a captured `parts` pointer.
* A whole-model replay on a carved session is refused rather than reading another stage's state
  (`generate.cpp:3823-3826`).

## In UnbiasedStrata

Changed slightly: the fork carves and zeroes the PLE history only when the model has a PLE, and builds the QSA
shapes through its `qsa_shapes.hpp`. The drivers are unchanged. Fork page: [session.md](../../codebase/core/session.md).

## See also

[layer-forward.md](layer-forward.md) · [cuda-graphs.md](cuda-graphs.md) · [verify.md](verify.md) ·
[experts/dispatch.md](experts/dispatch.md)
