[Upstream docs](../README.md) › [Program](README.md) › Generation loop

# The one-shot path: prefill, token loop, report

**Purpose:** what `strata` does without `--serve`. It reads the prompt (batched, then one token at a time), generates
`--max-new` tokens, and prints a report on stdout. This is the path that tok/s figures, logit comparisons against
llama.cpp (Gate C1) and the A/B arms are measured on. Prompts are token ids; there is no tokenizer.

**Covers:** `src/program/generate.cpp:3866-3992` (head, input, sampler, dump files), `:4236-4357` (loop scratch, token
graph, residency table), `:7470-7797` (batched prefill and token loop), `:8180-8361` (report and `--stats`). The
speculative loop that follows is on [speculative-decoding.md](speculative-decoding.md).

## Building blocks set up before the loop

- **`run_head(stream)` (`:3868-3873`)**: the canonical `lm_head` writes the logits, or, with a native head,
  `lm_head_mix` and then `NativeHead::run`.
- **`put_input(tok, pos)` (`:3894-3910`)**: `embed_row` of the token. At position 0 this is `session_zero` with the
  embedding (the reference's initial state: the embedding broadcast to all `hc` streams). Later positions copy the
  embedding into each of the `hc` residual streams directly, because `session_zero` would also wipe the recurrence.
- **Sampler (`:3912-3927`)**: `SamplerParams` from `--greedy`, `--seed`, `--top-k`, `--top-p` and `--temperature`.
  `sample_tokens` takes device pointers. The comment records the bug this caused once: passing a host pointer
  faulted, and the fault surfaced one token later.
- **Loop scratch (`:4242-4253`)**: `SessionLoopScratch` is allocated once. Allocating it per token made every token end
  with a device-wide sync (review finding H3).
- **Token graph (`:4254-4341`)**: the whole token as **one** CUDA graph. It is used when nothing needs a host step
  between a layer's ring and its `post[l]` graph: capture is on (`--no-capture` off), `--no-token-graph` is off, there
  are no per-layer dumps, the pack is canonical, there is no split, and the hit path (if any) is decided on the device
  (`TokenHits` with the residency table). Otherwise `session_loop` runs two graphs per layer: `pre[l]` ending at the
  router and doorbell, the CPU pool, then `post[l]`.

## Batched prefill (`:7470-7575`)

With `--prefill` and more than one prompt token:

1. `n_batched = n_prompt - 1`, or `--prefill-until P` when that is smaller (validation: the rest of the prompt goes
   through the token path, teacher-forced).
2. **Borrowing** (unless `--no-prefill-borrow`, with a residency table): `plan_lend(chunk)` chooses the chunk and the
   slots, shrunk to what this prompt needs (`request_chunk`, rounded up to 256). The lent slots' experts become
   non-resident and the table is uploaded. The buffers are laid out from `xcache.device_slot(first)`. A fixed chunk
   that does not fit falls back to the path's own buffers. With `auto`, the log states the chunk.
3. `Prefill::init(wt, g, ss, src, cache, host_res, chunk, stream, err, borrow, borrow_bytes)`.
4. With `--mtp`: `mtp.bind(wt, &native_head, nullptr)`. `prefill.on_chunk` hands the drafter each chunk's residual
   rows and the next tokens. `draft_kv` batches the drafter's K/V; otherwise `mtp.prefill` does it token by token.
5. `prefill.run(tokens, n_batched, 0)`. Then the lent slots are refilled (`fill_slot_queued` plus `sync_queued`, or
   blocking with `STRATA_REFILL_BLOCKING=1`) and the table is uploaded again.
6. The token loop starts at `pos_start = n_batched`. The PLE window `ple_prev` is set to the two preceding prompt
   tokens. The log line gives tokens, chunks, ms, tok/s, experts streamed, how many by DMA, resident, and PLE ms.

## The token loop (`:7577-7797`)

Each position `pos` from `pos_start`:

1. A **native pack stops here** and hands position `pos` to the speculative loop (`spec_pos = pos`). Its last prompt
   token is the first verify window.
2. The timer starts before any of the token's work (A7: the published rate is wall-clock per token).
3. **PLE**: `ss.ple_token = tok`, `ple_issue_token` (the 16 SSD row reads start), `put_input`, then
   `ple_finish_token`. Both PLE steps run outside the capture because they are host hashing and host reads.
4. `drive.d.layers = 0`. The CPU adapter uses this as the blob's layer index, so it must restart at 0 every token.
5. **Layers**:
   - `--no-capture`: `session_token`, with no pool;
   - token graph: `session_run_token`;
   - otherwise: `session_loop(..., pool_fn, hit_fn, overlap=true, ..., layer_stage, &loop_scratch)`.
6. `--dump-final-r` appends `{pos, tok}` and R. A failed pool exits 1, naming the layer and expert.
7. `--dump-layers` and `--dump-halves` write this position's staging buffers.
8. `run_head`, then a `cudaDeviceSynchronize` unless `--stream-token`. An asynchronous fault is sticky and would
   otherwise be reported by the next token's first call.
9. **Logits**: read back when not in stream-token mode, with `--check-logits`, or when this row is dumped. Non-finite
   logits exit 1. The dump header is written with the first row. Rows are kept by
   `logits_selection::selected(pos, total, stride)`.
10. **Sample**: `sp.counter = pos` (Philox(seed, position), the same draw as row t of a verify window), then
    `sample_tokens` and a 4-byte readback. With `--native-flash-attn-short`, every QSA layer's status is checked.
11. Time goes to decode (from `n_prompt - 1`) or prefill. TTFT is measured at the last prompt position.
12. From `n_prompt - 1`, the sampled id goes to `produced`. The prediction at the last prompt position is the first
    generated token.
13. Stop at `--max-new`, or at an end-of-turn id with `--stop-eos`.
14. **Teacher forcing** while the prompt lasts: the next input is the prompt's own next token. The PLE window shifts.
15. With `--spec > 0`, once the first token is generated the loop hands over (`spec_pos = pos + 1`).

The six host-phase accumulators (`ms_ple`, `ms_embed`, `ms_layers`, `ms_head`, `ms_readback`, `ms_sample`) feed
`--stats`.

## Report (`:8180-8361`)

On stdout:

- the speculative report (see [speculative-decoding.md](speculative-decoding.md));
- the dump summaries (layers, halves, routing);
- `--stage-timing`'s table (`stage_timing_report`);
- `prompt  : ...` and `output  : ...` (the ids);
- `decode N tokens in X ms -> Y tok/s`;
- `prefill N tokens ... (time to first token Z ms)`;
- `--dump-mixed` and `--dump-residual`, with their summary numbers.

`--stats` adds:

- the wall-clock ms per token and the six host phases;
- the PLE table's I/O report;
- the CPU pool per position, with its denominator being the positions the pool ran on, not the decoded tokens (A6);
- the pool phases (wait-park, drain, re-park);
- blobs read and helper-cache entries;
- R4 overlap and hit rate (hits, admitted, refused, how full);
- token-graph waits;
- ring latency and host time after the ring.

Then everything is freed and `main` returns 0.

## Model and pack notes

- **Canonical Q2_0 pack:** the full path above. Prefill is optional: without `--prefill`, the whole prompt runs
  through the token loop.
- **Native packs:** no per-layer graphs (`:3823`). The token loop body never runs, and `--prefill` is required for a
  prompt longer than one token. So `--dump-logits` writes nothing from the token loop, and `--gpu-stages` cannot run
  ([diagnostics.md](diagnostics.md)). `STRATA_DUMP_FIRST_LOGITS` dumps the first window's logits instead.
- **Layer split:** one-shot generation does not apply, because `--layer-split` requires `--serve`.

## Tests

None in the repository for this path itself. The comments cite measurements in `bench/results/` and a
`tools/logits_identical.py` that read the `--dump-logits` header; that tool is not in this snapshot.

## In UnbiasedStrata

Changed for Qwen3.6:

- the routing trace bound is `n_layers` instead of a literal 48;
- without a draft source the spec loop runs one-token windows;
- `--teacher-force` (with `--spec-oracle`) commits every window row and dumps one logits row per position.

See [../../codebase/program/generation-loop.md](../../codebase/program/generation-loop.md).

## See also

[speculative-decoding.md](speculative-decoding.md), [diagnostics.md](diagnostics.md), [startup.md](startup.md).
