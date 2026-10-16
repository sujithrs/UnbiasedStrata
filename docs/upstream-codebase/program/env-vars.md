[Upstream docs](../README.md) › [Program](README.md) › Environment variables

# Environment variables read by the driver

**Purpose:** every environment variable `src/program/generate.cpp` reads with `getenv`, with its default, its effect
and where it is read. Most are A/B arms, tuning knobs or traces. None is needed for normal use; the setup and the
server pass flags instead.

**Count:** 45 distinct names: **44 `STRATA_*`** plus `CUDA_MODULE_LOADING`. Counted with
`grep -o 'getenv("[A-Z_0-9]*")' src/program/generate.cpp | sort -u`. Unless a row says otherwise, a variable is read
once (often into a `static`) and "set" means set to anything, even an empty string.

## Start-up and memory

| Variable | Read at | Default | Effect |
| --- | --- | --- | --- |
| `CUDA_MODULE_LOADING` | 1245 | unset, so the engine sets `EAGER` | When unset, `main` sets it to `EAGER` (`_putenv_s` on Windows), so every kernel loads before the expert cache takes the free VRAM. An explicit value wins |
| `STRATA_ARENA_MMAP` | 145, 1239 | off | `=1`: a file-backed expert arena (the arena itself is in `src/core/expert_source.cpp:2730-2761`). In the driver, on AMD under Linux only: it sets `GPU_PINNED_MIN_XFER_SIZE=1000000` (`:1239-1240`) and turns on `pin_blobs`, which locks the pages of adaptive swaps (`:143-150`) |
| `STRATA_ARENA_RELEASE` | 4291 | release | `=0`: keep the experts a GPU holds mapped in the arena instead of handing their pages back to the OS |
| `STRATA_RSS_TRACE` | 185 | off | Print `RssFile` (mapped file pages) at start-up steps (Linux) |
| `STRATA_TRACE` | 1006, 5647 | off | `strata trace: N MiB free after <step>` at start-up; one stderr line per serve request step; lend and refill timings |
| `STRATA_EARLY_REMOTE_CONTEXTS` | 2020 | off | `=1`: create every helper GPU context (CUDA1..3) before CUDA0's weights, not only CUDA1's. Needed under WSL2, where the mapped-host budget runs out |
| `STRATA_NO_IQ256` | 2059 | unset | Only the log text: the native-pack CPU experts are reported as running on ggml-cpu `vec_dot` instead of AVX-2. The switch itself is read in the CPU kernels |
| `STRATA_KV_ROT` | 1760 | off | `=1`: INT8 K/V through the Hadamard rotation that `--kv q4_0` uses (opt-in) |
| `STRATA_SSD_KEEPALIVE` | 2399 | 100 | ms without a PLE row read before one page is read anyway (clamped to 0..10000; 0 = off). Only with `--ple-io direct` and the I/O thread |
| `STRATA_SSD_KEEPALIVE_WINDOW` | 2400 | 60 | Seconds after the last row request during which the keep-alive goes on (1..86400) |
| `STRATA_TEST_CACHE_FAIL` | 3362 | 0 | The first N expert-cache opens fail like an out-of-commit `cudaMalloc` (tests the retry) |

## Expert source and tiers

| Variable | Read at | Default | Effect |
| --- | --- | --- | --- |
| `STRATA_RESIDENT_PIN` | 1498, 1509 | pinned | With `--resident-experts` or `--resident-budget-gib`: `=0` keeps the RAM copy pageable |
| `STRATA_RESIDENT_HEADROOM_GIB` | 1500, 1511 | 4 | GiB of available RAM the resident mode leaves free (`>= 0`) |
| `STRATA_FETCH_THREADS` | 3128 | the source's own | With `--mmap-experts`: the file tier's fetch threads (`> 0`) |
| `STRATA_UNBUFFERED_LOAD` | 3139 | off | Let the file tier read unbuffered even without `--resident-budget-gib` |
| `STRATA_LOOKAHEAD` | 3676 | on | `=0`: no routing-aware prefetch of the file tier |
| `STRATA_LOOKAHEAD_K` | 3687 | 10 | Experts per token the router lookahead warms |
| `STRATA_PEER_HOT` | 2660 | 0.45 | With `--peer-device`: the share of the hottest ranked pairs moved to the peer (0 or >= 1 = off) |
| `STRATA_PEER_HOT_AT` | 2662 | 8700 | How many of the first ranked pairs that share is taken from |
| `STRATA_ADAPT_NOWAIT` | 1015 | off | `=1`: a verify window does not wait for the adaptive tier's copies (0.1.37's behaviour, #463) |
| `STRATA_TRACE_ADAPT` | 7875, 7904 | off | One-shot speculative loop: log why each adaptive round swapped or skipped, and whether the copies had landed |
| `STRATA_REFILL_BLOCKING` | 199 | off | Refill lent slots one blocking copy at a time instead of queued copies and one wait |

## Prompt path

| Variable | Read at | Default | Effect |
| --- | --- | --- | --- |
| `STRATA_PREFILL_AUTO_MAX` | 1383 | 8192 | With a bare `--prefill auto`: the chunk ceiling, snapped to 8192, 16384 or 32768 (#282). `auto:N` takes precedence |
| `STRATA_PREFILL_LEND_PCT` | 4433 | 90 when at least 90% of the expert bytes are pinned, else 85 | The largest share of the cache's slots an automatic chunk may borrow |
| `STRATA_BATCH_DECODE_SHARE` | 6851 | 0.5 | `--batch`: while a prompt is read in chunks, the active slots decode for this share of each chunk's time |

## Layer split

| Variable | Read at | Default | Effect |
| --- | --- | --- | --- |
| `STRATA_SPLIT_OWN` | 1654 | borrow | `1`: every stage keeps its own prompt buffers; `auto`: only when they are at most 12% of every card; `0` or unset: borrow from each stage's cache (#340) |
| `STRATA_SPLIT_OWN_PLACE` | 2736 | off | `=reserve`: the automatic placement search counts the own-buffer reserve |
| `STRATA_SPLIT_MISS_MS` | 2768 | 190 | The placement model's cost (ms) per unit of routed mass no cache holds; a slower CPU needs a higher value |
| `STRATA_STAGE_TRIM` | 2209 | off | Not `0`: every GPU of an explicit split loads only its own layers' dense weights (the same as `--trim-stage-weights`) |
| `STRATA_SPLIT_RING` | 4668 | 96 when >= 75% of the pairs are resident, else the pinned-share rule | The prompt path's streamed ring in slots on a split; 0 = the pinned-share rule |
| `STRATA_SPLIT_OWN_BUFFERS` | 4830 | on | `=0`: a split stage with room never keeps its own prompt buffers; every stage borrows |
| `STRATA_SPLIT_SMALL_OWN` | 5018 | 0 | S tokens: every borrowing stage keeps S-token buffers for the whole session (its experts there become non-resident) |
| `STRATA_SPLIT_SMALL_MAX` | 5020 | S | Requests up to M tokens read in S-token chunks on those stages instead of borrowing |
| `STRATA_REFILL_SERIAL` | 6768 | off | Refill the stages' lent slots stage after stage (0.1.30/0.1.31) instead of issuing all copies first |
| `STRATA_SPLIT_TIMING` | 7429 | off | After each request: per-stage host time per window |

## Serve diagnostics and checks

| Variable | Read at | Default | Effect |
| --- | --- | --- | --- |
| `STRATA_WATCHDOG_S` | 5720 | 60 | Seconds without progress during a request before the engine reports and aborts; 0 = off |
| `STRATA_DECODE_TIMING` | 7067 | off | One line per request: where the decode time went, plus the verifier's GPU-stage profile |
| `STRATA_CKPT_REREAD` | 6619, 6675 | off | Read a checkpoint's tokens again instead of restoring it, and never use short reads (a checkpoint check) |
| `STRATA_SNAPSHOT_VERIFY` | 5381, 6555 | off | Log the drafter-prefill path per chunk; verify a restored parked conversation's draft K/V by hash |
| `STRATA_SNAPSHOT_FULL_CAPTURE` | 6582 | off | Do not keep a restored conversation's K/V buffers for reuse at its next park |
| `STRATA_STATE_HASH` | 7221 | off | Print state fingerprints after each request |
| `STRATA_STATE_HASH_GDN` | 7263 | off | With `STRATA_STATE_HASH`: one hash per GDN layer |
| `STRATA_LOGPOS` | 6712 | off | Path: append the teacher-forced log-probabilities of window-read prompt tokens |
| `STRATA_LOGPOS_EXTRA` | 6716 | 248046 | The extra token whose log-probability is added as the last column |
| `STRATA_DUMP_FIRST_LOGITS` | 8057 | off | Path: the first verify window's logits row (one-shot speculative loop) |

## Variables the driver sets or names but does not read

| Variable | Where | What |
| --- | --- | --- |
| `GPU_PINNED_MIN_XFER_SIZE` | set at `:1240` (`setenv(..., 0)`, so an explicit value wins) | ROCclr: stage pageable copies of 1 MiB or more instead of locking the pages in place |
| `STRATA_SPEC_COUPLED` | `include/strata/core/coupled_draft.hpp` (`coupled_draft_env()`) | The default of `--coupled-draft` |
| `STRATA_ARENA_PIN_GIB` | `src/core/pinned.cu` (`arena_pin_cap_gib()`, used at `:3153`) | How much of the expert arena is pinned (0 = all); overrides the 8 GiB multi-GPU cap under WDDM |
| `STRATA_RING_BYTES`, `STRATA_PREFILL_RING` | `src/prefill/prefill.cpp` | `STRATA_RING_BYTES=0` restores 0.1.39's fixed auto-chunk list; `STRATA_PREFILL_RING` forces the ring size |
| `STRATA_VERIFY_PROFILE` | `src/core/verify.cpp` | The verify window's GPU-stage profile, suggested by the `--gpu-stages` error for native packs |
| `STRATA_COMMIT_SYNC` | `src/core/verify.cpp` | Keep the commit synchronous (see `Verifier::set_commit_async`) |
| `STRATA_EMULATE_CC` | `include/strata/core/emulate.hpp` | An emulated compute capability; the GPU line says so |
| `STRATA_ISA_FLOOR` | a build setting (`isa_floor_build()`) | The older-CPU build (`avx` or `none`) |

## Gotchas

- Some variables test only the first character (`[0] == '1'`, or `!= '0'`), others only that the variable is set. The
  table says which where it matters.
- Most are read into a function-local `static`, so they cannot change while the engine runs.

## In UnbiasedStrata

All 45 are kept. The fork adds `STRATA_SERVE_LOGIT_TRACE` and `STRATA_SERVE_LOGIT_DUMP` (c7825b4), for 47 `STRATA_*`
names. See [../../codebase/program/env-vars.md](../../codebase/program/env-vars.md).

## See also

[cli-flags.md](cli-flags.md), [diagnostics.md](diagnostics.md), [layer-split.md](layer-split.md),
[expert-tiers.md](expert-tiers.md).
