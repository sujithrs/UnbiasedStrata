[Upstream docs](../README.md) › [Program](README.md) › Serve mode

# `--serve`: the persistent engine and its line protocol

**Purpose:** load everything once, then take requests on stdin and stream tokens on stdout. This is how the Python
server uses the engine. `serve/server.py:454` starts `strata --serve <args>` with stderr going to the log. It reads
`INFO` lines until `READY <ctx> stop` (`:458-467`), then writes `GEN` / `GENI` / `BGEN` / `STOP` / `BSTOP` / `BYIELD` /
`VRAM` / `QUIT` lines and parses `T`, `PP`, `RESUME`, `DONE`, `BT`, `BDONE`, `BADM`, `YIELDED` and `ERR`. The batch
part of the protocol is also tabled in upstream
[docs/BATCHING.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/BATCHING.md#engine-protocol---serve).

**Covers:** `src/program/generate.cpp:4359-4378` (protocol comment) and `:4607-7468` (the serve block). Prompt-cache
logic is on [prompt-cache.md](prompt-cache.md), decoding on [speculative-decoding.md](speculative-decoding.md),
layer-split plumbing on [layer-split.md](layer-split.md), and adaptive swaps and `VRAM` on
[expert-tiers.md](expert-tiers.md).

## Requirements (`:4608-4613`)

`--spec T` (T >= 2), `--mtp DIR`, `--prefill CHUNK|auto` and, when the graphed hit path is in use, a filled residency
table. Otherwise the engine prints `strata serve: needs --spec T, --mtp DIR and --prefill CHUNK ...` and exits 2.

## Set-up inside the serve block

| Lines | Step |
| --- | --- |
| 4614-4882 | Prompt-path loan planning per participant (`PfPart`: CUDA0 and each stage). Chunk and ring choice by bisection under `auto`; an explicit chunk halved until it fits; the #448 small-card warning; own buffers on a split stage with room |
| 4884-4988 | `init_prompt_paths`, with a chunk step-down through `{6144, ..., 512}` when buffers do not fit; on final failure, advice (#85) and exit 1 |
| 4989-5008 | Peer prompt rows; stage helpers for one-chunk prompts |
| 5010-5046 | `STRATA_SPLIT_SMALL_OWN` / `_MAX` |
| 5047-5059 | Penalty-history buffer `d_hist` (`4096 x kVerifyMaxT` ids) on the head's device |
| 5060-5168 | `Verifier ver` init (`VerifyHits`), layer-split hand-offs, `mtp.bind`, batch-slot sessions per stage (`init_slots`), the pool callback (`drive_pool_multi` or `drive_pool_split`), split and PCIe mode |
| 5169-5433 | Conversation-cache state, `park_current`, `checkpoint_at`, `sp.on_chunk` (drafter K/V, `PP` lines, periodic checkpoints), per-stage checkpoint parts |
| 5434-5589 | Adaptive tier (`apply_pending`, `adapt`, `save_profile`) |
| 5590-5645 | stdin reader thread (`read(2)` / `_read`, not `std::cin`, so `exit()` cannot hang on stdin's lock). `STOP` sets `stop_req` at once; other lines are queued |
| 5646-5652 | `STRATA_TRACE`: one stderr line per step of a request |
| 5653-5675 | Free-VRAM check: >= 256 MiB is fine; below that, a LOW warning suggesting `--vram-reserve-mib` (or saying the reserve was already lowered, #496) |
| 5676-5714 | `INFO` line |
| 5715-5742 | Watchdog thread |
| 5743 | `READY <max_context> stop` |
| 5745-5759 | `S = --spec`, `S_mtp`, `SuffixDrafter sfx`, `DraftPolicy policy`, vision constants (`kImagePad = 248056`) |
| 5760-5878 | `VRAM` command (`relend`, `vram_command`) |
| 5879-6183 | Batch slots: `BSlot`, `copy_to_slot`, `copy_from_slot`, `batch_step`, pipelined groups (`pump`, `pipe_drain`) |
| 6184-7465 | The request loop |
| 7466-7467 | `save_profile("exit")`, return 0 |

## Protocol

### Lines to the engine (stdin)

| Line | Meaning |
| --- | --- |
| `GEN <max_new> [key=value ...] <id,id,...>` | A text request |
| `GENI <max_new> [key=value ...] <embeddings file> <id,id,...>` | A request with images; needs `--vision` |
| `STOP` | End the running request at its next step (checked between prompt chunks and between windows). A `STOP` that arrives between requests is ignored |
| `QUIT` | Leave the loop: save the learned profile, exit 0. End of stdin does the same |
| `VRAM [reserve_mib]` | `--vram-elastic` only, between requests: shrink or grow the expert cache ([expert-tiers.md](expert-tiers.md)) |
| `BGEN <slot> <max_new> [keys] <ids>` / `BGENI <slot> <max_new> [keys] <file> <ids>` | `--batch`: read the prompt and first token as `GEN 1`, then continue in batch slot `<slot>` |
| `BSTOP <slot>` | End that slot at its next batch window |
| `BYIELD <slot>` | The prompt being read gives way at its next chunk boundary; the part read so far waits in `<slot>` |

**Request keys** (`:6246-6284`): `temperature=F` (absent = greedy), `top_p=F`, `top_k=N` (default 20; the sampled path
needs 1..64), `min_p=F`, `penalty_last_n=N`, `penalty_repeat=F`, `penalty_freq=F`, `penalty_present=F`, `seed=N`
(absent = time-based), and `cvec=0|1` (a loaded control vector for this request; on when absent). Two tuning keys are
used by `tools/calibrate.py`: `pcie_frac=F` and `spec_min_p=F`, both clamped to `[0, 1]`. Unknown keys are skipped.
The ids start at the first token without `=`. `GENI` takes the same keys (#75); its file path is the first token
without `=`.

### Lines from the engine (stdout)

| Line | When | Fields |
| --- | --- | --- |
| `INFO key=value ...` | once, before `READY` | `context kv kv_resident expert_slots expert_cache_mib expert_slots_primary expert_cache_primary_mib spec mtp_max lookup vram_free_mib cvec arena_mib pool_workers pcie_frac spec_min_p conversation_cache_mib conversation_cache_slots conversation_cache_min_free_mib tail_role_token vram_elastic [batch_slots slot_cache] engine` (`:5696-5713`). The expert slots add up every tier: CUDA0, the stages and the helpers |
| `READY <max_context> stop` | once | `stop` means this engine honours `STOP`; the server reads `max_context` from here |
| `RESUME <n>` | per request, before reading | Prompt tokens reused (live session, checkpoint, slot or parked conversation) |
| `PP <pos> <prompt_tokens> <ms> <tok/s>` | after each batched chunk and each window-read part | Progress; tok/s over fresh tokens since `RESUME` |
| `REUSED <n>` | prompt read, first window next | The same `n` |
| `T <id>` | each generated token, as its window commits | |
| `YIELDED <slot> <tokens>` | a read that gave way (`BYIELD`) | Comes before `DONE ... cancel` |
| `DONE ...` | end of a request | See below |
| `BADM <slot> <0\|1>` | after a `BGEN`'s `DONE` | 1 = it continues in the batch slot |
| `BT <slot> <id>` | batch windows | A token of a slot |
| `BDONE <slot> <produced> <stop\|length\|cancel> <ms>` | a slot finishes | The slot keeps its conversation |
| `VRAM reserve_mib=.. ...` | reply to `VRAM` | |
| `ERR <message>` | a request or command cannot run | After many errors the engine exits 1 (fatal); a bad request just continues |

`DONE` (`:7344-7350`) is space-separated. Fields were appended over time, so an older parser reads a prefix:

```
DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length|cancel> <drafts_accepted> <drafts_offered>
     <reused> <cache_hits> <cache_lookups> <ram_blobs> <file_blobs> <file_MB> <prompt_tokens_read> <offloaded>
```

`prompt_tokens_read` (#471) is less than `prompt - reused` when a cancel stopped the read. `offloaded` (#588) counts
the routed experts the GPU read over PCIe or another GPU computed; these are not part of the lookups.

## One request (`:6184-7465`)

1. **Dispatch (`:6185-6231`).** While batch slots are active, an empty input queue means one more batch window
   (`batch_step`, or `pump` for groups), and the loop continues. `BSTOP`, `BYIELD` (stale), `VRAM` and `BGEN` are
   handled here; `BGEN` is rewritten into a `GEN 1`. A due periodic profile save runs before the request. `QUIT`
   breaks out.
2. **Parse (`:6232-6297`).** `BusyScope` marks the request for the watchdog. The stale `STOP` flag is cleared. The
   keys are parsed. `max_new < 1` or bad ids give `ERR bad request`.
3. **Images (`:6298-6389`).** For `GENI`, or for the first text request after one: the M-RoPE table is rebuilt.
   Images are read from strata-vision records (`int32 'SVE1', n, nx, ny, n_embd`, then floats). Each image's rows go
   to its run of `<|image_pad|>` tokens, at positions `t = p`, `h = p + y`, `w = p + x`; the text after an image
   continues at `p + max(nx, ny)`. Each picture's grid and embeddings are hashed (FNV-1a) for the conversation cache.
   The table is uploaded to CUDA0 and every stage. Errors restore the identity table.
4. **Checks.** Prompt + `max_new` + 8 must fit the context. Every id must be in the vocabulary.
5. **Resume source** ([prompt-cache.md](prompt-cache.md)): the live session, a checkpoint, an idle batch slot, or a
   parked conversation, whichever covers the longest prefix (at most `n - 1` tokens). The outgoing conversation is
   parked first when it would be overwritten. Then restore, or `session_zero` on every stage, and `RESUME <n>`.
6. **Prompt read (`:6668-7046`).** The fresh part is cut at up to three boundaries: the root (the end of the system
   prompt, for a read from 0), the last turn (`--turn-token`, adjusted by `--tail-role-token`), and `n - 1`. Each part
   is read either:
   - through the **verify windows** (`read_windows`, S tokens at a time, the picks thrown away, no head sampling),
     when it has at most `--short-read` tokens and no image rows; or
   - through the **batched prompt path** (`lend`, then `read_part`), interleaved with batch windows and checked for
     `BYIELD`.

   The comment's measurements behind `--short-read`: the batched path costs about 300 ms per run plus about 180 ms of
   refill, a window about 16 ms per token. A checkpoint is taken at the root and turn boundaries. The lent slots are
   refilled, then `REUSED`.
7. **Sampling set-up (`:6938-6962`).** `ver.set_sampling(req_sp)`, `mtp.set_draft_sampling`, the request's PCIe share
   for CUDA0 and the stages, and the penalty history.
8. **Decode (`:7047-7192`):** verify windows with MTP or suffix drafts ([speculative-decoding.md](speculative-decoding.md)).
   `T <id>` per token; the finish is `stop` (end-of-turn id), `length`, or `cancel` (`STOP`).
9. **After (`:7193-7464`).** `ver.wait_commit`. `STRATA_DECODE_TIMING` prints one line per request. The live tokens
   are recorded (`live`), unless cancelled mid-prompt. `STRATA_STATE_HASH` prints fingerprints. Then `YIELDED` and
   `DONE`, and for a `BGEN`, the admission into the slot (`copy_to_slot`, `set_slot_sampling`) and `BADM`. The
   routing trace is flushed. stderr gets one summary line per request: prompt reused and read, generated tokens and
   rates, drafts, checkpoints. More stderr lines follow: decode hit rate and offload share, resident RAM, expert
   tiers, `STRATA_SPLIT_TIMING`, KV streaming hit share, suffix drafts, and helper-cache counters.

Errors that print `ERR` and then `return 1` end the process. Examples: a failed window, commit or refill; a failed
checkpoint restore; a restore of a parked conversation that fails after validation. The server starts the engine
again. A CUDA fault during the prompt calls `std::_Exit(1)` at once (#224): the context is poisoned, and unwinding
could hang until the watchdog.

## Batch slots (`--batch N`, `:5879-6183`)

Upstream [docs/BATCHING.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/BATCHING.md)
is the user guide. Each slot has its own session on every stage, carved at start-up
([startup.md](startup.md)).

- **Admission:** `BGEN` reads the prompt and first token on the main session as a `GEN 1`. `copy_to_slot` then copies
  the running state and the K/V up to the live length into the slot's sessions, on every stage. The slot gets its own
  sampling (`set_slot_sampling`) and the request's best turn checkpoint. `BADM <slot> 1`.
- **`batch_step`:** one window over the active slots only (row t = the t-th active slot). `run_slot_rows`, then
  `commit_slots`, `BT` per row, `BDONE` on an end-of-turn id, `BSTOP`, `max_new` or the context limit. When all
  slots are idle, one stderr line gives windows, average rows, ms per window split into run, commit and emit,
  CPU experts, VRAM hits and PCIe per layer-window, and rows per second.
- **Prompt reads beside decoding slots** (`read_part`): the prompt is read one chunk at a time. Between chunks the
  slots decode for `STRATA_BATCH_DECODE_SHARE` (default 0.5) of the chunk's time.
- **`BYIELD` (#656):** at a chunk boundary with more than a chunk and more than `--short-read` left, a text prompt
  read is copied into the given slot (`partial`), with its checkpoints. The request ends with `YIELDED` and
  `DONE ... cancel`. The server sends it again later and it continues from the slot.
- **Slot as cache:** an idle slot keeps its conversation (`cached`). A later request whose prompt starts with it is
  copied back into the main session (`copy_from_slot`), from the slot's end or from its turn checkpoint. Slots with
  pictures are never reused.
- **Pipelined groups (`--batch-groups G`, layer split only):** the slots form G groups of `batch / G`. `pump`
  launches and polls `batch_launch` / `batch_poll` per stage, so stage k runs one group while stage k+1 runs
  another. Pipelined slots are never reused as caches, because their pad rows write state. A new request first
  drains every group in flight.

## Watchdog (`:5715-5742`, issue #29 and #31)

A thread checks `progress().beats` every second. When a request is busy and nothing has moved for
`STRATA_WATCHDOG_S` seconds (default 60; 0 = off), it:

1. prints `no progress for N s ... (stage)`;
2. runs `stall_report` twice, 2 s apart: the pool's and the verifier's diagnostics, the threads in disk wait (Linux),
   resident/swap/available memory and page faults; on Windows it also writes a minidump, `strata-stall-<pid>.dmp`;
3. calls `release_gpu_waits` so that no spin kernel outlives the process;
4. calls `abort()`. The server then restarts the engine.

## Platform notes

- stdin is read with `read(2)` on POSIX and `_read` on Windows. A `\r` at the end of a line is dropped.
- `--vram-elastic` (`VRAM`) is NVIDIA-only.
- The Windows minidump needs `dbghelp.dll`. Linux reports processes in disk wait instead.

## Tests

- `serve/test_parallel.py` and `serve/test_server.py` run fake engine scripts that speak this protocol;
  `serve/test_lifecycle.py` and `serve/test_vram.py` use mocked engines.
- `tools/batch_test.py` drives the real engine (GEN against BGEN, token for token).
- `tools/conversation_restore_speed.py` checks parking and restore against the live session.

## In UnbiasedStrata

Kept, with three additions:

- a `STATS` command that answers one JSON line of tier counters (the fork's `/health`);
- `STRATA_SERVE_LOGIT_TRACE` and `STRATA_SERVE_LOGIT_DUMP` traces of the per-token logits (c7825b4);
- `READY` reports the context, which the fork's VRAM plan may have lowered.

Upstream's client, the Python server, is not in the fork; the fork's C++ front end speaks the `GEN` / `T` / `DONE`
subset. See [../../codebase/program/serve-mode.md](../../codebase/program/serve-mode.md).

## See also

[prompt-cache.md](prompt-cache.md), [speculative-decoding.md](speculative-decoding.md),
[expert-tiers.md](expert-tiers.md), [diagnostics.md](diagnostics.md).
