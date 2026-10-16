[Upstream docs](../README.md) › [Prefill](README.md) › Several GPUs

# The prompt path on several GPUs

## Purpose

Upstream has two multi-GPU modes, and the prompt path supports both: a **layer split** (`--layer-split`, each GPU a
stage that runs a range of layers) and a **peer tier** (`--peer-device`, a second card that caches experts and
computes their rows). On top of the layer split there is an opt-in **stage helper**: the idle next stage streams and
computes a share of this stage's experts for a one-chunk prompt. All three are in
[src/prefill/prefill.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/prefill.cpp);
the user-facing descriptions are in `docs/MULTI_GPU.md` and `docs/SECOND_GPU.md`
([MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MULTI_GPU.md),
[SECOND_GPU.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/SECOND_GPU.md)).

## Layer split: a pipeline of stages

- `set_stage(lb, le, next)` (`include/strata/prefill/prefill.hpp:147-149`): this object runs layers `[lb, le)` on the
  device `init` ran on. A stage that does not end at the last layer needs a `next`; `init` allocates two pinned
  portable hand-off buffers of `chunk x 10240` floats (`src/prefill/prefill.cpp:746-750`).
- In `run_impl`, a non-final stage copies its chunk rows device -> host into one of the two hand-off buffers
  (alternating), calls `on_stage_chunk`, waits only for its direct successor's previous chunk, then starts
  `next_->run_impl` for this chunk on a `std::async` thread and continues with its next chunk (`:2921-2941`). So stage
  `i+1` reads chunk `c` while stage `i` reads chunk `c+1`.
- A later stage reads `hand_in_` instead of embedding the tokens (`:1628-1634`). Only the last stage calls
  `on_chunk`; `generate.cpp` moves the callback there (`src/program/generate.cpp:5414-5416`).
- `run` drains the chain at the end (`drain_pipeline`, `:3049-3068`) and reports the first error.
- Serve inits every stage's prompt path from its own cache's loan, stepping the chunk down through 6144 .. 512 when
  one does not fit (`src/program/generate.cpp:4889-4985`). Without borrowing, a layer split sizes the ring at 96
  (`set_ring_override(96)`, `:2718`).

## Peer tier: `set_peer` (`src/prefill/prefill.cpp:1251-1348`)

Called by serve when `--peer-device` is valid and `--peer-prefill-rows` is not 0 (`src/program/generate.cpp:4989-4995`;
-1 means half of `chunk x top-k`). It needs P2P between the two cards and the MMQ prompt path; a failure only prints
and keeps the prompt on the primary.

Per MoE layer (`:2438-2629`): the host grouping orders rows local-first, so the peer's experts' rows are one block at
the end (`on_peer` 1 = peer-held, 2 = peer-streamed). The primary records `ev_in`; on the peer's stream the
activations come over P2P (`cudaMemcpyPeerAsync`), the peer quantizes its rows, gathers its experts from its own
slots (`PeerExperts::slot_ptr`) or its ring, runs the same MMQ products and copies the result rows back into the
primary's `Dm`. The primary waits on `ev_done` before the combine (`:2858-2862`).

| Knob | Default | Effect |
| --- | --- | --- |
| `STRATA_PF_PEER_COMPACT` | on | per-group buffers (up to 16 experts and at most `T` rows) instead of layer-sized ones: "~0.9 GB -> ~0.26 GB on the helper card at 8192-token chunks, and no row cap" (`:401-405`); with compact on, `cap_rows` becomes `T_max x K`, so `--peer-prefill-rows` only caps the non-compact arm |
| `STRATA_PF_PEER_OUT_PIPE` | on (always with compact) | result rows go back group by group on a second stream while the next group computes |
| `STRATA_PF_PEER_STREAM` | 0.35 | share of the primary's streamed pinned experts that the peer streams over its own PCIe link into its own ring and computes; "measured: 0.25-0.5 all ~1950-1970 at 32K, 0.35 best" (`:1300-1301`); needs a chunk of at least `stream_all_min()` and MMQ on every layer |
| `STRATA_PF_PEER_RING` | 48 | the peer ring's slots |

With `--peer-device` the prompt path keeps MMQ (the fused kernels are not combined with the peer's rows,
`fused_ring()` returns false), and the mapped grouping tables become portable (`core::peer_portable()`).
`STRATA_PREFILL_TIMING=1` adds the peer's own timeline and "rows/layer" figures.

## Stage helper: `set_stage_helper` (opt-in `STRATA_PREFILL_HELP=1`)

A prompt of one chunk runs the stages one after the other, so the next stage's GPU idles while this one reads.
`set_stage_helper(helper)` (`:1146-1184`) prepares a stream-only `PeerPrefill` on the helper's device (streams,
events, a 48-slot ring's events) and serve wires every stage to the next one, the last to the first
(`src/program/generate.cpp:4996-5010`). `bind_stage_helper(T)` (`:1188-1249`) binds it to the helper stage's own
(lent) prompt buffers for one run, only when:

- the prompt is a single chunk of at least `stream_all_min()` tokens and the share `split_help_frac(T)` is positive;
- this stage streams (a source, a ring larger than `STAGE`) and does not use the fused ring;
- every layer of the stage is on MMQ and none is covered by the fused native kernels;
- the helper's buffers are laid out for at least this chunk.

The share (`split_help_frac`, `:1135-1143`) is `min(0.45, 0.5 - T/16384)` and 0 below 0.3, so about 0.41 at 1.5K
tokens, 0.32 at 3K and off from about 3.3K; `STRATA_PREFILL_HELP_FRAC` fixes it. Without P2P the activations and the
result rows go through mapped portable host buffers read by copy kernels (`copy_f32_wide`, `copy_i32`), grown once to
the largest chunk helped. The comment's measurement on 2x RTX 3090 (UD-Q4_K_XL, no P2P): fixed shares 0.2 / 0.3 / 0.4
against none at 1.5K tokens 749 / 820 / 908 (none 675), at 8K 1,895 / 1,804 / 1,663 (none 2,230). The helped rows
round like a different MMQ grouping, so the output is not bit-identical to the default (`docs/MULTI_GPU.md`), which is
why it is opt-in.

## Gotchas

- `--peer-device` and `--layer-split` refuse each other at startup (`docs/SECOND_GPU.md`), so `set_peer` and the
  stage helper never meet; the helper takes the place of `pp` for the run it helps (`std::swap`, `:1565-1579`).
- The peer's buffers are plain `cudaMalloc`s on the peer card; `--peer-reserve-mib` must leave room for them
  (the `set_peer` error says so).
- The one-shot `generate` path does not call `set_peer`; only serve shares prompt rows with the peer.

## In UnbiasedStrata

Kept in the source (only width changes, commit e41dfc7). The fork's docs describe the same three mechanisms:
[../../codebase/prefill/pipeline.md](../../codebase/prefill/pipeline.md) ("Multi-GPU").

## See also

[pipeline.md](pipeline.md), [expert-streaming.md](expert-streaming.md), [moe.md](moe.md); the program area for
`--layer-split`, `--peer-device` and the serve loop.
