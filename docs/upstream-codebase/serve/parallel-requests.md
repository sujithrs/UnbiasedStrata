[Upstream docs](../README.md) › [Serve](README.md) › Parallel requests

# Parallel requests: the server side of batch slots

## Purpose

By default the server runs one request at a time behind `Service.fifo`; others wait. With `"parallel": N` in the
run config (the engine's `--batch N`, also spelled `--slots N`; #465, engine part PR #559) the engine keeps up to N
conversations in batch slots and decodes them together, one token of each per verify window. This page covers how
`StrataEngine.generate_batched` schedules requests onto the slots and speaks the batch protocol. The engine side
(slot sessions, pipeline groups, exactness) belongs to the engine area; upstream's own guide is `docs/BATCHING.md`.

Files: `serve/server.py` (`parallel_args`, `serve/server.py:1485`; `StrataEngine.__init__` slot state,
`serve/server.py:488-513`; `_control`, `_drain_control`, `_release_slot_when_done`, `_take_control`,
`_may_go_solo`, `_shorter_waiting`, `generate_batched`, `pick_slot`, `slots_view`, `serve/server.py:717-1107`;
`Service.run`'s per-request status, `serve/server.py:2263-2290`); `docs/BATCHING.md`; `serve/test_parallel.py`.

## Turning it on

`parallel_args` (`serve/server.py:1485-1503`): `"parallel": N` adds `--batch N` unless `--batch` / `--slots` is
already in `args`; 1, absent or `false` changes nothing; a non-integer is ignored with a message; more than 8
(`PARALLEL_MAX`, "the engine's batch window holds at most 8 rows (kVerifyMaxT)") is passed on with a warning. The
engine never refuses a count: it runs what fits and reports `INFO ... batch_slots=N slot_cache=0/1`; the server uses
that number (`serve/server.py:490-494`) and says when it differs from the request. `/v1/status` reports it as
`concurrency.serving`. With a layer split, `--batch-groups G` in `args` makes slot order spread consecutive requests
over the pipeline's groups (`slot_order`, `serve/server.py:495-499`).

Setup recommends `parallel` only where the experts mostly fit in VRAM; `docs/BATCHING.md` measured on one RTX 5070
(12 GB), Q2_0, 32K: four concurrent requests got their first token after 1.0 s / 1.8 s (median / last) with
`"parallel": 4` against 6.0 s / 11.2 s one at a time, but decoded 11-24% slower in total, and a request alone lost
11% (2 slots) or 24% (4 slots).

## Batch protocol (from `docs/BATCHING.md` and `generate.cpp`)

| Line | Direction | Meaning |
| --- | --- | --- |
| `BGEN <slot> <max_new> [keys] <ids>` | in | read the prompt as a `GEN 1`, then continue in `<slot>` (`generate.cpp:6212-6227`) |
| `BGENI <slot> <max_new> [keys] <file> <ids>` | in | the same with images |
| `BADM <slot> <1/0>` | out | after the admission's `DONE`: 1 = it continues in the slot, 0 = it ended (`generate.cpp:7383`) |
| `BT <slot> <id>` | out | a token of that slot (`generate.cpp:6023`, `6105`) |
| `BDONE <slot> <generated> <stop/length/cancel> <ms>` | out | the slot is free again and keeps its conversation (`generate.cpp:6030`, `6114`) |
| `BSTOP <slot>` | in | end that slot at its next window (`generate.cpp:6197`) |
| `BYIELD <slot>` | in | the prompt being read gives way at its next chunk boundary; its part read waits in `<slot>` (`generate.cpp:6877-6890`) |
| `YIELDED <slot> <tokens>` | out | before the `DONE cancel` of a read that gave way (`generate.cpp:7343`) |

`_pump` routes `BT` and `BDONE` lines to per-slot queues; everything else goes to the shared "control lines" queue.

## Scheduling (`generate_batched`, `serve/server.py:864-1078`)

- **Control lines, one at a time.** Prompt reads and admissions use the shared stdout queue, so only one request
  holds them (`ctl` lock, `_take_control`). Waiting requests are counted (`waiting`, `wait_lens`) and get a
  heartbeat every 10 s.
- **Alone: the solo path.** With no slot busy and nobody waiting, a request is sent as a plain `GEN` and decodes with
  MTP drafts, the fastest single stream. If another request starts waiting meanwhile (`stop_when=others`), the solo
  request is `STOP`ped and **promoted**: it continues with `BGEN` into a slot with prompt + tokens so far (the
  engine's prompt cache holds exactly that).
- **Otherwise: a slot.** `pick_slot` (`serve/server.py:1080-1093`) prefers the free slot whose held tokens are the
  longest prefix of the prompt (its conversation's last turn, so the engine copies its state back instead of
  re-reading), else an empty one, else the one used longest ago. `BGEN` is sent; the request reads the control lines
  until `BADM`; with `BADM ... 1` it releases the control lines and reads its own slot queue (`BT` ... `BDONE`).
- **A long read gives way to a short one (#656).** While a prompt is read and a request with a prompt under half as
  long waits (`_shorter_waiting`), the server sends `BYIELD <slot>` (a reserved free slot for a solo read); at the next
  chunk boundary the engine stops the read, keeps the part read in the slot and answers `YIELDED`. The request then
  lets the waiting ones go first (`after_epoch`) and continues from its slot. At most `YIELDS_MAX = 2` times per
  request; not with images.
- **Left alone in a slot: back to the solo path.** When a slot request is the only busy one and nobody waits
  (`_may_go_solo`: engine reports `slot_cache=1`, at least 32 tokens left, at most `SOLO_AGAIN_MAX = 2` times, no
  images, `STRATA_PARALLEL_SOLO` not `0`), the server sends `BSTOP`; at `BDONE ... cancel` it continues with a `GEN`
  whose prompt is the original prompt plus everything generated, and the engine copies the slot's sessions back.
- **Leaving early.** The `finally` block keeps the engine in step: a solo request gets `STOP` and is drained to its
  `DONE`; an admission is drained to its `BADM`; a decoding slot gets `BSTOP` and is freed in the background at its
  `BDONE` (`_release_slot_when_done`), recording what the slot holds for the conversation's next turn.

`slot_held[b]` (what each slot's sessions hold: prompt plus every token fed, all but the last) and `slot_used[b]`
drive `pick_slot`; `slot_live[b]` feeds `/metrics` (`slots_view`: idle with `held_tokens`, or reading / decoding with
`prompt_tokens`, `generated`, `elapsed_s`, `tok_s`).

## The service with parallel requests

- `Service.run` does not hold `fifo` around generation when the engine has slots (`serve/server.py:2272`); only
  `ensure_loaded` takes it, so one request restarts a dead engine while the others wait (`serve/server.py:2281`).
- Each running request keeps its own status dict and rate window in `live_reqs`; `self.status` mirrors the newest
  one; the live tok/s is the sum of all (`Service._tok_s`, `serve/server.py:1957`).
- With batch slots `StrataEngine.last` is thread-local, so each request records its own `DONE` / `BDONE` figures
  (`serve/server.py:403-419`).
- `/metrics` `live` gains `parallel`, `running`, `slots`, `outside_slots`, `waiting` (`serve/server.py:2054-2064`).
- `/v1/vram` also takes the control lines; the engine refuses a resize while slots decode ("VRAM: not while batch
  slots are decoding", `generate.cpp:6205`).

## Limits (from `docs/BATCHING.md`)

Batch windows carry no MTP drafts (one token per window in a slot); repetition / frequency / presence penalties are
not applied in batch windows; admissions are one at a time; a read gives way only at a chunk boundary and not for
pictures; `--batch-groups` needs every stage on its own GPU and a pipelined slot is not kept as a conversation cache;
slot sessions take VRAM (0.56 GiB per slot at 32K with 8-bit KV, measured there) and, with KV streaming, pinned RAM.

## Tests

`serve/test_parallel.py` runs the real `StrataEngine` and `Service` against `FAKE_BATCH`, a Python script that speaks
`GEN` / `BGEN` / `BT` / `BDONE` / `BADM` / `BSTOP` / `BYIELD` (one token every 0.02 s per active slot):
`ParallelArgs.test_parallel_config_key`, `PickSlot` (the slot that holds the conversation, `slots_view`),
`ParallelService` (the engine reports fewer slots, the engine turns batching off, concurrent requests share the
slots, a long prompt gives way to a short one, an admission gives way too, no way given to a long one, left alone it
goes back to the solo path, the next turn goes back to its slot). Run: `python -m unittest serve.test_parallel -v`.
Engine-level checks are `tools/batch_test.py`, `tools/batch_interleave_test.py`, `tools/parking_test.py`,
`tools/early_close_test.py` (need a built engine or a running server; `docs/BATCHING.md` "Testing").

## In UnbiasedStrata

Removed with the server (commit 8894a81). The fork's front end refuses `--batch` after `--` (exit code 2) and serves
one request at a time behind a mutex; `docs/BATCHING.md` was removed too.

## See also

- [engine-process.md](engine-process.md), [monitoring-and-telemetry.md](monitoring-and-telemetry.md)
- [docs/BATCHING.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/BATCHING.md) (upstream)
