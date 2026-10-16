[Upstream docs](../README.md) › [Program](README.md) › Diagnostics

# Dumps, measurement modes and traces

**Purpose:** the engine was built against a reference (llama.cpp on the same token ids, "Gate C1"). Most diagnostics
exist to bisect a disagreement: dump the logits, the residual, every layer or every half-layer, the routing, and time
the GPU in isolation. This page lists them, with the rules that refuse combinations that would produce plausible but
wrong files.

**Covers:** `include/strata/program/logits_selection.hpp`; in `src/program/generate.cpp`: dump options (`:253-329`),
the half dump (`:2345-2367`), stage timing and the routing dump (`:3747-3796`), the `--no-capture` refusals
(`:3834-3864`), the logits and layer dump set-up (`:3929-3974`), the measurement modes (`:4005-4234`), the dump writes
in the token loop (`:7656-7725`), the report (`:8180-8348`) and the serve-side traces.

## Dump files

| Flag | Format | Rules |
| --- | --- | --- |
| `--dump-logits PATH` | int32 header `{n_vocab, n_rows}`, then n_rows x n_vocab f32 | The header is written **with the first row**, because a native pack never reaches the per-token dump, and a header promising rows that were never written is worse than no file (`:3930-3932`). `n_rows = row_count(n_prompt - 1 + max_new, stride)`. The count is `n_prompt - 1 + max_new`, not `n_prompt + max_new`: the comment says `tools/logits_identical.py` caught that off-by-one (`:3945-3951`); that tool is not in this snapshot. Both dimensions must fit int32 |
| `--logits-stride N` | rows `0, N, 2N, ...` plus the final input position | `parse_stride`: a positive decimal int64, no sign, space or suffix. N > 1 requires `--max-new 1` and `--dump-logits`. Storage selection only: every prompt token still conditions the output |
| `--dump-residual PATH` | int32 `{hc, n_embd}`, then the final R (f32) | Written after the run, with nonfinite count, mean and max printed. Lets the head be checked in Python without the layers |
| `--dump-mixed PATH` | n_embd f32 of `bb.mixed` | The head input; only the 794 MB GEMV cannot be recomputed cheaply |
| `--dump-layers PATH` | per position `(n_layers + 1) x hc x n_embd` f32 | The C1 bisection ladder, written by `session_loop` into a pinned staging buffer **after** capture. Refused with `--no-capture` |
| `--dump-halves PATH` | per position `n_layers x (2 n_embd + 2 hc + n_head head_dim + 5 n_head_kv head_dim + 8)` f32 | Captured into the layer graphs, so it must be armed **before** `session_capture` (`ss.block.dump`, `:2345-2367`). Refused with `--no-capture` |
| `--dump-routing PATH` | records `int32 layer, int32 k, k x int32 ids, k x f32 weights` | Written by `drive_pool` after each dispatch. The layer index is `d.layers - 1`, because the adapter has already advanced it (the first trace was off by one, `:764-770`); out-of-range layers are dropped with a message. In verify windows (`drive_pool_multi`) the weights are written as zeros, read as unit weights (`:791-802`). Needs the pool. In serve it is flushed after every request |
| `--dump-final-r PATH` | per position `int64 pos, int64 tok`, then hc x n_embd f32 | The MTP head's input; token path only |

Disabling the token graph: any per-layer dump keeps the loop on the two-graphs-per-layer path (`:4268`, `:4333`).
`--dump-layers`, `--dump-halves`, `--gpu-stages` and `--stage-timing` also turn off the fused GR read (`:2284-2285`),
which needs the unfused layout of R.

## Measurement modes (print, then exit)

All three refuse a layer split across GPUs (`:1929-1933`) and `--native-flash-attn-short` (`:1922-1925`).

| Flag | What it times |
| --- | --- |
| `--gpu-stages` (`:4006-4152`) | R0.9: each layer captured as three graphs (mixer, ffn+router, post), timed from outside the capture over 20 reps. Then: the mixer split by layer kind (36 GDN and 12 QSA layers, plus the worst layer); R0.11, five stages by differencing prefixes; R3.5c, the same five by sweeping each prefix (the bias column compares the two). A native pack has no per-layer graphs: the error points to `STRATA_VERIFY_PROFILE=1 STRATA_DECODE_TIMING=1` with `--serve` (#610) |
| `--graph-only` (`:4154-4182`) | Only the 48 `pre` graphs back to back, 20 reps after a warm pass. **Not** the GPU floor: it omits the `post` graphs and the head |
| `--gpu-only-full` (`:4184-4234`) | pre + post for all layers plus the LM head, no pool, 20 reps after a warm pass: the per-token GPU floor. The comment says to quote this one, not `--graph-only` (Memory/ERRORS.md A4/A5) |

`--stage-timing` (`:3753-3782`) is not an exit mode. It records CUDA events inside the layer path and needs
`--no-capture` (an event inside a capture is dropped silently). Its table is a share of kernel count, not a time
profile, because uncaptured intervals include host gaps (`usage()` says so). Stage names include the eight GDN
internals.

## Refusals that prevent misleading output

- `--no-capture` without `--no-pool`: `session_token` has no pool hook, so the routed experts would add nothing
  (`:3843-3849`).
- `--no-capture` with `--dump-layers` or `--dump-halves` (`:3853-3864`).
- `--dump-routing` with `--no-pool`: the ids reach the host through the doorbell the pool reads (`:3785-3789`).
- Non-finite logits stop the run, naming the count and the position (`:7707-7713`).

## `--stats` (`:8263-8348`)

Wall-clock ms per token and six host phases: PLE, embed, layers, head, readback and sample. Then the PLE I/O report,
the CPU pool per position with its phases (wait-park, drain, re-park), blobs read, helper-cache counts, R4 overlap and
hit rate, token-graph waits, rings seen mid-graph, ring latency, and host time after the ring. The comments record
several past reporting bugs and the denominators chosen to avoid them (A6: positions, not decoded tokens).

## Serve-side and environment traces

| Variable | Output |
| --- | --- |
| `STRATA_TRACE=1` | `strata trace: N MiB free after <step>` at start-up (`mem_mark`); one line per request step in serve; lend and refill timings |
| `STRATA_RSS_TRACE=1` | `RssFile` (mapped file pages) at start-up steps (Linux `/proc/self/status`) |
| `STRATA_DECODE_TIMING=1` | Per request: windows, average T, ms per window split into verify (GPU wait, host plan/actq/jobs/CPU, stage), commit and draft; CPU experts, hits and PCIe per layer-window; the verifier's GPU-stage profile |
| `STRATA_SPLIT_TIMING=1` | Per stage after each request: wait for the GPU, pool + plan, host staging, commit per window |
| `STRATA_TRACE_ADAPT=1` | One-shot loop: whether each adaptive round swapped or skipped, and whether the copies had landed |
| `STRATA_STATE_HASH=1` (+ `STRATA_STATE_HASH_GDN=1`) | After each request: FNV-1a fingerprints of the GDN state, PLE history, indexer tails, pooled keys, K/V (`[0, L)` and the stale tail), drafter K/V, `dead` key; per GDN layer with `_GDN` |
| `STRATA_LOGPOS=<path>` (+ `STRATA_LOGPOS_EXTRA=<id>`) | During window-read prompt parts: the teacher-forced log-probability of every token read, plus the log-probability of an extra token (default 248046, `<\|im_end\|>`), appended to the file |
| `STRATA_DUMP_FIRST_LOGITS=<path>` | One-shot speculative loop: the first window's logits row (raw f32), so prompt paths can be compared by output on a native pack |
| `STRATA_CKPT_REREAD=1` | Instead of restoring a checkpoint, read its tokens again from 0 (and never use short reads); with `--adapt-swaps 0` the answer must match the restore |
| `STRATA_SNAPSHOT_VERIFY=1` | Logs the drafter-prefill path per chunk and verifies a restored parked conversation's draft K/V by hash |
| `STRATA_TEST_CACHE_FAIL=N` | The first N expert-cache opens fail (tests the shrink-and-retry) |

The watchdog's `stall_report` is described on [serve-mode.md](serve-mode.md).

## Tests

`logits_selection.hpp` has no test in the upstream tree: a search for `logits_selection` and `logits-stride` finds
only `generate.cpp`. `conv_cache_test` is the program directory's only unit test.

## In UnbiasedStrata

Kept, with these additions:

- `--teacher-force` (one row per position along `--spec-oracle`);
- `STRATA_SERVE_LOGIT_TRACE` and `STRATA_SERVE_LOGIT_DUMP` (per-token logits from `--serve`, c7825b4);
- one logits row per emitted position in the spec loop, so a native pack's `--dump-logits` is no longer empty.

See [../../codebase/program/diagnostics.md](../../codebase/program/diagnostics.md).

## See also

[generation-loop.md](generation-loop.md), [env-vars.md](env-vars.md).
