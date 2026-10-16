[Upstream docs](../README.md) › Program

# Program: the engine driver (`strata`, `--serve`)

`src/program/generate.cpp` is the engine's `main`. It is the only file that puts every engine part together and turns
them into tokens: the weight arena, the session state, the CPU expert pool, the VRAM expert cache and its helper tiers,
the batched prompt path, the verify windows, the MTP drafter, the prompt-lookup drafter and the conversation cache.
CMake builds it as the `strata` executable (`CMakeLists.txt:495-497`). The program has **no tokenizer**. Prompts
arrive as token ids, through `--tokens`, `--tokens-file` or the `GEN` lines of serve mode. It runs in one of three
shapes:

- a **one-shot generation**: batched prefill, then the per-token graph loop, then the speculative loop, then a report;
- a persistent **`--serve`** process that reads requests on stdin and writes tokens on stdout;
- a **measurement mode** that prints a GPU timing table and exits.

Its main client is the Python server: `serve/server.py:454` starts `strata --serve <args>`, and the args come from
the `strata-<model>.json` config that `setup.py:4224-4227` writes. Downward it calls the core library (sessions,
verifier, MTP drafter, expert sources and caches, conversation snapshots), the kernels (sampler, PLE table, KV formats,
rope scaling, control vectors, CPU expert pool), the prompt path (`strata::prefill::Prefill`) and the speculation
helpers (`strata::spec::SuffixDrafter`, `strata::spec::DraftPolicy`). The Intel port has its own copy of this file,
converted by DPCT (`sycl/src/program/generate.cpp`, 8,380 lines). That copy is documented with the SYCL area, not
here. Terms such as GDN, QSA, PLE, MTP and "pack" are defined in the [glossary](../guides/glossary.md).

## Contents

| Page | What it covers |
| --- | --- |
| [cli-flags.md](cli-flags.md) | Every flag `main` parses (148 `--` spellings including `--help`, plus `-h`), with its default and effect, grouped by subsystem |
| [startup.md](startup.md) | Start-up from argv to "session is up": validation, CPU and GPU checks, pack kind and model variant, rope resolution, weights, PLE table, sessions, drafter, head, expert source, expert-cache sizing |
| [expert-tiers.md](expert-tiers.md) | Where experts live: the pinned arena or mmap, the resident RAM mode, the VRAM cache and profile fill, the hit path, helper caches on CUDA1..3, the peer GPU, prompt-path slot lending, adaptive swaps, `--vram-elastic` |
| [layer-split.md](layer-split.md) | Multi-GPU `--layer-split`: option rules, the automatic placement search, per-stage weights, sessions and caches, hand-offs, own vs borrowed prompt buffers |
| [generation-loop.md](generation-loop.md) | The one-shot path: batched prefill, the per-token loop (token graph, per-layer graphs, `--no-capture`), sampling, the final report and `--stats` |
| [speculative-decoding.md](speculative-decoding.md) | Verify windows, MTP drafts, suffix (prompt-lookup) drafts and the draft policy, `--spec-min-p`, acceptance and commit, oracle and corrupt drafts |
| [serve-mode.md](serve-mode.md) | The `--serve` loop and its line protocol (`GEN`, `GENI`, `STOP`, `QUIT`, `VRAM`, `BGEN`, `T`, `DONE`, `INFO`, `READY`, ...), batch slots, pipelined groups, the watchdog |
| [prompt-cache.md](prompt-cache.md) | Conversation checkpoints and their retention policy (`conv_cache.hpp`), turn and root boundaries, parked conversations, batch-slot caches, `conv_cache_test` |
| [diagnostics.md](diagnostics.md) | Dumps and measurement modes: logits (`logits_selection.hpp`), residual, layer, half, routing and final-R dumps, `--gpu-stages`, `--graph-only`, `--gpu-only-full`, `--stage-timing`, serve-side traces |
| [env-vars.md](env-vars.md) | Every environment variable `generate.cpp` reads (44 `STRATA_*` names plus `CUDA_MODULE_LOADING`) and the ones it sets |

## Source map

| File | Lines | Role |
| --- | --- | --- |
| [src/program/generate.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/program/generate.cpp) | 8,361 | The driver: options, start-up, measurement modes, serve loop, one-shot generation and report. `main` is lines 1228-8361 |
| [src/program/conv_cache_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/program/conv_cache_test.cpp) | 67 | Unit test of the checkpoint retention policy (`conv_cache_test`, `CMakeLists.txt:912-916`) |
| [include/strata/program/conv_cache.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/program/conv_cache.hpp) | 43 | `conv_cache::eviction_victim`: the chain's root is pinned, the other checkpoints rotate by least recent use |
| [include/strata/program/logits_selection.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/program/logits_selection.hpp) | 30 | `--logits-stride` parsing and row selection for `--dump-logits` |

## Layout of `generate.cpp`

The file is one anonymous namespace of helpers, followed by one very long `main`. The line ranges below are section
boundaries.

| Lines | Section |
| --- | --- |
| 1-104 | Header comment ("THE DRIVER"), includes |
| 106-241 | `under_wddm`, `vram_free_note`, AMD page pinning for adaptive swaps (`pin_blobs`), `rss_probe`, `refill_blocking`, `resident_stage_swaps` |
| 243-536 | `struct Options`: every flag's storage and default |
| 538-719 | `usage()` |
| 721-739 | `parse_i64_list` (token id lists) |
| 741-852 | `Drive`, `drive_pool`, `drive_pool_multi` (CPU-pool adapters and routing trace), `SplitDrive`, `GpuStage` |
| 854-1024 | Watchdog support (`mem_sample`, `stage_text`, rotational-disk and disk-wait probes, `stall_report`), `mem_mark`, `adapt_nowait`, `argmax` |
| 1026-1072 | Conversation-cache helpers (`ConvCheckpoint` alias, `fnv1a`, `checkpoint_save`, `checkpoint_restore`) |
| 1074-1146 | `load_control_vectors` (llama.cpp control-vector GGUFs) |
| 1148-1224 | `probe_pcie_h2d_gbps`, `pcie_frac_for_gbps` |
| 1228-1531 | `main`: unbuffered stdout, environment defaults, argv parsing (two `if` chains because of MSVC's C1061 limit) |
| 1532-2059 | Option validation, layer-split parsing, `--native` expansion, CPU ISA checks, expert layout and pack kind, PCIe probe |
| 2060-2499 | Model shape and rope keys from the GGUF, rope resolution, native embedding, weight arena, native dense, kernel switches, control vectors, half dump, doorbell, PLE table |
| 2501-3054 | `--split-skip-if-fits`, later stages' weights, profile, split search, sessions (the carve), batch slot sessions, MTP drafter, head, logits |
| 3056-3746 | GPU check, expert source (arena or mmap), CPU pool, expert-cache sizing and fill, stage caches, helper caches, peer GPU, router lookahead, R4 hit path |
| 3747-3992 | Stage timing, routing dump, graph capture and refusals, `run_head`, `put_input`, sampler set-up, dump files |
| 4005-4234 | Measurement modes that print and exit: `--gpu-stages`, `--graph-only`, `--gpu-only-full` |
| 4236-4606 | Loop scratch, token graph, residency table, prompt-path lend planning (`plan_lend`), resident RAM mode |
| 4607-7468 | `--serve` (returns at 7467) |
| 7470-7575 | One-shot batched prefill |
| 7577-7797 | One-shot token loop |
| 7799-8178 | One-shot speculative loop and its report |
| 8180-8361 | Dumps, report, `--stats`, clean-up |

## Key concepts and data flow

```
argv ──► Options ──► validation ──► expert layout (pack kind: canonical Q2_0 or native IQ)
                                         │
             ┌───────────────────────────┴── with --native: expert_count / expert_used_count and rope keys
             ▼                                from the GGUF's qwen4exp.* metadata
   weights (pack arena + native GGUF projections, head, embedding) ─ PLE table ─ session(s) ─ MTP drafter ─ head
             │
   expert source (pinned arena | mmap of experts.bin or the GGUF) ─► CPU ExpertPool
   VRAM expert cache (filled from --expert-profile) ─► host_res / d_res residency table
   (+ helper caches on CUDA1..3, or a peer GPU, or later layer-split stages)
             │
   ┌─────────┴───────────────┬──────────────────────────────────┐
   ▼                         ▼                                  ▼
 measurement modes        --serve loop                     one-shot generation
 (print, exit)            stdin GEN ─► resume source       Prefill.run (chunks)
                          ─► Prefill / short windows        ─► token loop (graphs; canonical pack)
                          ─► verify windows + MTP/suffix    ─► verify windows + MTP/suffix
                          ─► T / DONE on stdout             ─► report on stdout
```

Ideas that recur across the pages:

- **Pack kind.** `expert_layout_load` reads the pack (`src/program/generate.cpp:1997`), and `expert_layout().native`
  tells the two kinds apart (`:2016`). The canonical Q2_0 pack (`pack/full`, `experts.bin`) runs the captured
  per-layer graphs. A native pack (`tools/iq_pack.py`: IQ2_XS, IQ3_XXS, IQ3_S, the Coder, Swift, Unsloth) keeps the
  GGUF's tensors in their GGUF form. It needs `--native SHARD1`, `--spec >= 2` and `--prefill`, and it decodes only
  through verify windows (`:2164-2170`).
- **Model variants.** There is one architecture (`qwen4exp`, 48 layers). Variants differ only in what the GGUF metadata
  and the shard layout say: a pruned Coder ships 256 experts per layer, read from `qwen4exp.expert_count` (`:2074-2079`).
  The PLE table is in shard 2 of the ISTA and Unsloth files but in shard 1 of Swift's, and `--native` finds it by
  tensor name (`:1787-1793`). The MTP draft layer always uses the canonical 512-expert geometry (`:3011-3013`).
- **Residency table.** `host_res[layer * n_expert + expert]` holds the VRAM slot of a resident expert, or
  `kNotResident`. It is mirrored to the device (`d_res`) and to each layer-split stage. Lending slots to the prompt
  path, adaptive swaps and `VRAM` shrinks all work by editing this table and uploading it again (`:4263-4332`).
- **Running vs positional state.** A conversation checkpoint copies only the running state: the GDN recurrence, the
  PLE history and the indexer tails. The KV cache is positional, so going back to a position only means writing from
  that position again (`:1026-1037`).

## Tests

- `conv_cache_test` (CTest; built with `STRATA_BUILD_TESTS` or `STRATA_BUILD_CONVERSATION_TESTS`) tests the retention
  policy. Run it from the build directory with `ctest -R conv_cache_test`.
- The driver itself has no unit test. These exercise it from outside:
  - `tools/batch_test.py` drives the engine binary with `GEN` and `BGEN` and compares the tokens;
  - `tools/conversation_restore_speed.py` checks parking and restore against the live session;
  - `tools/calibrate.py` sends per-request `pcie_frac=` and `spec_min_p=` keys;
  - the server tests stand in for the engine: `serve/test_parallel.py` and `serve/test_server.py` run fake engine
    scripts that speak this protocol, and `serve/test_vram.py` and `serve/test_lifecycle.py` use mocked engines.

## In UnbiasedStrata

The fork keeps `conv_cache.hpp`, `logits_selection.hpp` and `conv_cache_test.cpp` unchanged. Its `generate.cpp`
changes by 535 added and 53 removed lines (`git diff 12a7825 HEAD`), all for the Qwen3.6-35B-A3B port:

- the model geometry is read from the GGUF (`qwen4exp` or `qwen35moe`);
- a VRAM planner (`strata::plan`) can stop the start or lower `--max-context`;
- three new flags: `--vram-cap`, `--ram-cap` and `--teacher-force`;
- the PLE is optional for models without one;
- the drafter's geometry follows the target;
- the native dense projections are bounded to the trunk;
- a `STATS` serve command and `STRATA_SERVE_LOGIT_TRACE` / `STRATA_SERVE_LOGIT_DUMP`.

The fork's history has two root commits: 12a7825, the import of this snapshot, and e41dfc7, the first commit of
the Qwen3.6 port branch (merged in 21b2aa2), whose tree already carries most of the port's changes to this file. Six
later commits change the file further: 6abffa7, 1f5ef7d, 6042a32, fa69e82, 7ebc9e5 and c7825b4. The fork also compiles the file a second time
into its HTTP front end: `frontend/engine_entry.cpp` (added in 1db37aa) does `#define main strata_engine_main` and
`#include`s the file. The fork removed the Python server that starts the engine upstream. See the fork's own pages in
[../../codebase/program/README.md](../../codebase/program/README.md).

## See also

- [docs/upstream/DETAILS.md](../../upstream/DETAILS.md) (the fork's copy of upstream `docs/DETAILS.md`): speeds,
  settings and troubleshooting as users see them.
- Upstream [docs/BATCHING.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/BATCHING.md),
  [docs/MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MULTI_GPU.md),
  [docs/SECOND_GPU.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/SECOND_GPU.md),
  [docs/MODELS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MODELS.md).
