[Upstream docs](../README.md) › [Program](README.md) › Prompt cache

# The conversation cache: checkpoints, parking and slot caches

**Purpose:** a chat client or an agent sends the whole conversation again with every request. Reading it all again
made long sessions wait minutes per turn. `--serve` therefore remembers where the session is and reuses as much of
each new prompt as it can. There are four sources of reuse:

1. **the live session**: what the session holds right now;
2. **checkpoints**: snapshots of the running state at chosen positions of the current branch;
3. **idle batch slots**: their sessions still hold the conversation they served last;
4. **parked conversations**: whole snapshots of other conversations in host RAM.

**Covers:** `include/strata/program/conv_cache.hpp`, `src/program/conv_cache_test.cpp`, and in
`src/program/generate.cpp` the helpers at `:1026-1072`, the serve block's state and lambdas at `:5169-5433`, the
resume logic at `:6413-6667`, the boundaries at `:6963-7035`, and the end of a request at `:7214-7220` and
`:7352-7386`.

## What a checkpoint holds (`:1026-1037`)

A finished sequence leaves two kinds of state behind:

- **Positional state**: the KV cache of the 12 QSA layers and their pooled indexer keys, and the draft layer's K/V. A
  cell is written once for its position and read only by later positions. Going back to a position only means writing
  from there again.
- **Running state**: the 36 GDN recurrences and conv histories, each QSA layer's indexer tail (the unfinished block's
  raw keys) and the PLE's normalised history. It describes "everything so far" and cannot be rewound.

A checkpoint (`ConvCheckpoint`, an alias of `strata::core::ConversationCheckpoint`) copies only the running state:
about 118 MB, the same set the verifier snapshots to roll back rejected drafts. `checkpoint_save` and
`checkpoint_restore` wrap `conversation_checkpoint_save` / `_restore` from the core library. A checkpoint is only valid
while the positional cells below it still hold **its** tokens. So the serve loop keeps only checkpoints that are
prefixes of what the session holds, and they always form a **chain**.

## Retention policy (`conv_cache.hpp`)

`eviction_victim(stamps, n, cap)` returns the index of the chain item to drop when there are more than `cap` items.

- Index 0, the **root**, is pinned. It is the deepest point every request so far has shared: in practice the end of
  the system prompt, which every new chat of the same client goes through. Dropping it would cost every later
  conversation a full re-read of that prefix. The header's example: a 30K system prompt is about 30 s at about 1,000
  tok/s.
- Every other item rotates by least recent use. A stamp advances when the checkpoint is created, when a new
  checkpoint lands on its length, and when a request resumes from it (`check_clock`).
- With a budget below 2 slots, the pin is off and the oldest item leaves. This is the old behaviour: one slot holds the
  newest point only.

The serve loop calls it from `checkpoint_at` whenever there are more than `--prompt-cache` checkpoints (`:5365-5372`).

## Where checkpoints are taken

| Where | Rule |
| --- | --- |
| **Turn boundary** (`:6968-6980`) | The last `--turn-token` (`<\|im_start\|>`, 248045) after the resume point: the history before the new turn's header. The next request of the same chat renders the same history, but not always the same header or this reply's thinking. With `--tail-role-token R` (#458), when the turn before it starts with role R (the server's trailing reasoning-effort turn), the boundary moves in front of that turn |
| **Root** (`:6986-6992`) | For a read from token 0 (or one resumed from a yielded read that started at 0): the **first** turn boundary, when it is at least `--prompt-cache-root` (2048) tokens in. A smaller system prompt is cheaper to read again than the extra part costs (about 0.3 s). PR #65 |
| **Periodic** (`:5393-5411`) | Every `--prompt-cache-every` (16384) freshly read tokens, from `sp.on_chunk`. With a layer split, each stage saves its part when it reaches the position (`on_stage_chunk`) and the last stage joins the parts; an incomplete set is skipped |

`checkpoint_at(L)` (`:5337-5374`) reuses an existing checkpoint of length L by advancing its stamp. Otherwise it syncs
the device (a failed sync is reported as a GPU hang, #613) and saves CUDA0's part plus one part per stage.

## Choosing where a request starts (`:6413-6667`)

1. `starts_with(prefix, imgs)`: the prefix is at most `n - 1` tokens long (the last token is always the first
   window), matches the prompt id for id, and has the same pictures (each image keyed by its start and hash).
2. If the control-vector state matches (`want_cvec == cvec_cached`), the candidates are the live session and every
   checkpoint, longest first.
3. Then idle batch slots (`cached`, same control-vector state, a text request): the slot's whole history, or one of
   its turn checkpoints.
4. Then parked conversations: `conversations.best(ids, imgs, cvec)`. A parked image longer than the others is taken
   out of the cache, its stage count is checked, and it is validated on every stage before anything is overwritten.
5. **Park the outgoing conversation** (`park_current`) when it would be overwritten. That is the case unless the
   request continues the live session from its end without a parked or slot source.
6. Restore:
   - from a slot: `copy_from_slot`; on failure the prompt is read from the start;
   - from a parked image: `conversation_snapshot_restore` on CUDA0 and each stage, then its checkpoints are merged
     back;
   - from a checkpoint: `checkpoint_restore` on CUDA0 and each stage's part.

   `STRATA_CKPT_REREAD` instead reads the checkpoint's tokens again from 0, to check the checkpoint.
7. Checkpoints longer than the resume point, or not on this prompt's path, are dropped (`:6599-6603`). With
   `resume == 0`, every stage's session is zeroed and the checkpoints are cleared.
8. KV streaming: the drafter's ring may hold cells past `resume` from a longer turn, so `mtp.kv_restore(resume)` is
   called.
9. `RESUME <resume>` is printed. At the end of a request that was not cancelled mid-prompt, `live` becomes the
   consumed tokens and `live_ok` is set when `--prompt-cache > 0`.

A request whose control-vector setting differs from the cached one drops `live` and the checkpoints (`:6593-6597`).

## Parked conversations (`--conversation-cache-mib`, `:5183-5307`)

`strata::core::ConversationCache` holds whole `SavedConversation` images in host RAM: running state, positional K/V up
to the live length, the draft layer's K/V and the checkpoint chain. It is limited to `--conversation-cache-mib` and
`--conversation-cache-slots`, and is enabled only with `--prompt-cache > 0`.

`park_current(held)`:

- drops superseded copies of this conversation (#342);
- with a split, moves the checkpoints apart into stage parts (and back together however the function ends);
- estimates every stage's snapshot, reusing K/V buffers it still has from the last restore (`take_reuse`; disable
  with `STRATA_SNAPSHOT_FULL_CAPTURE`);
- `make_room` evicts until the image fits;
- `conversation_memory_admit` checks physical RAM against `--conversation-cache-min-free-mib` (2560) before and after
  the capture;
- saves CUDA0's image (with the draft K/V when there is no split) and one image per stage (the last with the draft
  K/V).

Each step can skip parking with a log line instead of failing the request. A `bad_alloc` is also caught.
`STRATA_SNAPSHOT_VERIFY` logs the drafter-prefill path and verifies a restored draft K/V by hash.

Parking needs each stage on its own GPU: `--split-device 0` with parking exits 2 (`:1573-1577`).

## Batch-slot caches

A slot that finishes keeps `ids` and, unless it has pictures, `cached = true` (`:6032`). Its best turn checkpoint is
copied in at admission (`:7374-7379`). A later request can continue from the slot's end or from that checkpoint
(`copy_from_slot`). A yielded read (`BYIELD`) parks its partial prompt and checkpoints in a slot and continues there
(`partial`). See [serve-mode.md](serve-mode.md).

## Tests

`conv_cache_test` walks five scenarios by hand:

1. FIFO with fresh stamps;
2. the root survives a stale stamp;
3. a mount moves the rotation;
4. budgets of two and one slots;
5. ties are resolved deterministically.

It is built when `STRATA_BUILD_TESTS` or `STRATA_BUILD_CONVERSATION_TESTS` is set (`CMakeLists.txt:912-916`). Run it
with `ctest -R conv_cache_test`, or run the `conv_cache_test` binary, which prints `PASS` or `FAIL`. End to end:
`tools/conversation_restore_speed.py` (#528) checks that restored turns decode with the same tokens and speed as live
ones. Its docstring cites the RTX 5070 at 0.1.38.

## Gotchas

- A cancelled prompt leaves the session between two chunks: `live` is not updated, but the checkpoints taken during
  the read stay valid (`:7214-7220`).
- Pictures are keyed by grid plus FNV-1a of the embeddings, so the same image re-encoded differently will not match.

## In UnbiasedStrata

`conv_cache.hpp` and its test are unchanged, and the serve-side cache logic is kept. See
[../../codebase/program/prompt-cache.md](../../codebase/program/prompt-cache.md).

## See also

[serve-mode.md](serve-mode.md), [layer-split.md](layer-split.md), [env-vars.md](env-vars.md).
