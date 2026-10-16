[Upstream docs](../README.md) › [Tools](README.md) › Conversation-cache and batch gates

# Conversation-cache, batching and stream-close gates

## Purpose

Scripts that run a real engine (or a running server) and check that an optimization does not change the output:
the **conversation cache** (parking up to N conversations' engine state in host RAM and restoring it when a
conversation returns, `--conversation-cache-mib` / `--conversation-cache-slots`, opt-in, see
[docs/DETAILS.md](../../upstream/DETAILS.md) "Multiple conversations") and **batch slots** (several conversations
decoding together, `--batch`, see
[docs/BATCHING.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/BATCHING.md)).
They need a GPU and a model; none runs in a CI. Their logic (the pass/fail rules) is unit-tested offline.

| File | Lines | Drives | Checks |
| --- | ---: | --- | --- |
| [tools/conversation_cache_parity.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/conversation_cache_parity.py) | 253 | 2 private engines (cache 0 vs N) | A/B/A reuse: same tokens and, at `--spec 1`, byte-identical main-model state; scenarios `reuse`, `pressure`, `oversized`, `exchange`, `admission` |
| [tools/conversation_cache_soak.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/conversation_cache_soak.py) | 169 | 2 private engines | 30+ known-answer A/B/A cycles at three prompt lengths up to near the context limit, streamed KV covered, retained bytes reach a plateau |
| [tools/conversation_cache_disabled.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/conversation_cache_disabled.py) | 196 | an "upstream" engine and one or more candidates | with the cache disabled, candidates equal the reference engine: tokens, state, reuse, known answers |
| [tools/conversation_cache_growth.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/conversation_cache_growth.py) | 170 | 1 engine, 2 runs (full vs incremental capture) | growth, rewind and branch give the same output and state; repeated parks reuse unchanged KV bytes |
| [tools/conversation_cache_isolation.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/conversation_cache_isolation.py) | 147 | 2 private engines | a parked state is reused only for the same image / steering identity (`image`, `add`, `project` scenarios) |
| [tools/conversation_cache_http_smoke.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/conversation_cache_http_smoke.py) | 137 | an idle, exclusive running server | HTTP A/B/A reuse, slot eviction, stream cancellation and recovery |
| [tools/conversation_restore_speed.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/conversation_restore_speed.py) | 182 | 2 private engines | a restored turn decodes like a live one (#528): same tokens, speed within `--tolerance` (15%) |
| [tools/parking_test.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/parking_test.py) | 89 | 2 engine runs | a follow-up decodes the same tokens whether A stayed live or was parked by B and restored; the second park reuses K/V bytes |
| [tools/batch_test.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/batch_test.py) | 161 | 1 engine with `--batch` | every slot's greedy tokens equal its solo tokens; aggregate tok/s |
| [tools/batch_interleave_test.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/batch_interleave_test.py) | 237 | 1 engine, 3 slots | a long prompt read while slots decode, `BYIELD`, next turn from a slot, slot -> solo path, history without thinking from a turn checkpoint (#465) |
| [tools/early_close_test.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/early_close_test.py) | 47 | a running server | a client that closes a stream early leaves no tokens to the next request, solo and with a second request running |

## Two ways of driving the engine

```
conversation_cache_*.py, conversation_restore_speed.py
   └─ serve.server.StrataEngine(exe, args, cwd, log, env)   .generate(ids, n, sampling, cancel[, embeddings])
                                                            .info (engine settings)  .last (the DONE line's fields)
batch_test.py, batch_interleave_test.py, parking_test.py
   └─ batch_test.Engine: subprocess "<exe> --serve <cfg args>", waits for READY, speaks the --serve line protocol:
        GEN <n> [keys] <ids>  ->  T <id> ... DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish>
                                                  <drafts_accepted> <drafts_offered> <reused> ...
        BGEN <slot> <n> [keys] <ids> -> BADM <slot> <0|1>, BT <slot> <id>, BDONE <slot> ...
        BYIELD <slot> -> YIELDED ..., BSTOP <slot>, QUIT
conversation_cache_http_smoke.py, early_close_test.py
   └─ HTTP: /v1/chat/completions (stream or not), /metrics
```

The DONE fields are parsed by `StrataEngine._parse_done` (`serve/server.py:608-621`); the line protocol is
documented in docs/BATCHING.md ("Engine protocol") and on [../serve/README.md](../serve/README.md).

Every script takes a run config (`strata-<model>.json`, written by setup): `args` (engine arguments), `cwd`,
`tokenizer` (the pack's `tokenizer/`, loaded with `load_tokenizer` / `batch_test.tokenizer`), `lib_dirs`, `gpu`,
`layer_split`. The chat template comes from `<tokenizer>/chat_template.jinja` through `serve.frontend.ChatTemplate`.

## Evidence from the engine log

The private-engine gates set environment variables that make the engine print diagnostics to its log, then parse
them:

| Variable | Effect used | Set by |
| --- | --- | --- |
| `STRATA_STATE_HASH=1` | one `STATE_HASH L=... gdn=... ple=... tail=... pooled=... kv=... ple_prev=...` line per request: hashes of the main model's state (`src/program/generate.cpp:7324`) | all `conversation_cache_*` gates |
| `STRATA_SNAPSHOT_VERIFY=1` | after a restore, the draft ring is read back (`SNAPSHOT_VERIFY draft=`) | growth, `parking_test.py` |
| `STRATA_SNAPSHOT_FULL_CAPTURE=1` | forces a full capture on every park (the reference arm) | growth |
| `STRATA_MTP_BATCH=1` | keeps the drafter's batched K/V pass during prompts on (only `0` turns it off, `src/prefill/prefill.cpp:961`), overriding an inherited `0` | disabled, growth |
| `STRATA_IQ_MT_MIN=1` | the multi-token CPU expert kernel choice that makes batched tokens equal solo ones (`include/strata/core/verify.hpp:139`; `batch_test.py --mt-min` help: "1: exact") | batch, interleave, parking |
| `STRATA_DECODE_TIMING=1`, `STRATA_VERIFY_PROFILE=1` (with `PROF`) | timing output | `batch_test.py` |

Log lines parsed: `conversation cache: parked <n> tokens in <ms> ms; parked=<k> bytes=<b> evictions=<e>
snapshot_bytes=<s> reused_kv_bytes=<r>`, `conversation cache: skip parking (snapshot ...` and `(physical RAM
admission; ...`, `restored <n> tokens`, `disk cache: spilled`.

The gates compare state fields `L, gdn, ple, tail, dead, pooled, pooled_full, kv, ple_prev` (`STATE_KEYS`,
`conversation_cache_parity.py:19`); `conversation_cache_disabled.py` compares only the fields an older engine also
prints (`L, gdn, ple, tail, pooled, kv, ple_prev`) and requires `pooled_full` from the candidate as evidence it is
the new build (`conversation_cache_disabled.py:26-43`).

Engine arguments the private gates add for determinism (`engine_args`, `conversation_cache_parity.py:45-50`;
`common_args`, `conversation_cache_disabled.py:46-59`): `--prompt-cache 6 --adapt-swaps 0 --spec max(2, S)
--mtp-max-t S --suffix-draft 0 --spec-min-p 0` (a native IQ pack needs a verify capacity of at least 2 even for one
token). The batch scripts recommend `--pcie-frac 0 --adapt-every 1000000` (and `--no-prefill-borrow` for the
interleave test) for exact comparisons.

## Safety conventions

- Dry run by default: every `conversation_cache_*` gate and `conversation_restore_speed.py` prints what it would do
  and exits unless `--run` is given; the tests check that a dry run reads no config, starts no engine and creates no
  output.
- `--output` must be a new directory (or, for the HTTP smoke test, a new file, opened with mode `x`); existing paths
  are refused. Directories are created with mode 0700.
- The HTTP smoke test refuses a busy server, a disabled cache, more than 16 slots, or a server with an API key, and
  checks before each request that no other client is active (`conversation_cache_http_smoke.py:39-54`). Its default
  URL is `http://127.0.0.1:11434`.
- Verification functions use `require()` that raises instead of `assert`, so `python -O` cannot skip them (tested).

## Configuration (command lines)

```
python tools/conversation_cache_parity.py --config C --engine EXE --output DIR [--cache-mib 8192] [--paragraphs 128]
       [--scenario reuse|pressure|oversized|exchange|admission] [--min-free-mib 2560] [--spec 1..8] --run
python tools/conversation_cache_soak.py --config C --engine EXE --output DIR [--prompt-tokens 2048,40000,120000]
       [--cycles 30] [--cache-mib 8192] --run
python tools/conversation_cache_disabled.py --config C --upstream EXE --candidate EXE [--candidate EXE2] --output DIR
       [--paragraphs 128] [--spec 1|4] --run
python tools/conversation_cache_growth.py --config C --engine EXE --output DIR [--spec 1|4] [--cache-mib 8192] --run
python tools/conversation_cache_isolation.py --config C --engine EXE --output DIR --scenario image|add|project --run
python tools/conversation_cache_http_smoke.py [--url URL] --output FILE --run
python tools/conversation_restore_speed.py --config C --engine EXE --output DIR [--tokens 20000] [--gen 160] [--turns 3]
       [--cache-mib 8192] [--set=--flag=value ...] [--tolerance 0.15] [--only baseline|candidate]
       [--mode switch|same|pair] [--trim 8] --run
python3 tools/parking_test.py --exe EXE --config C [--repeat 60] [--max-new 120] [--extra "..."]
python3 tools/batch_test.py --exe EXE --config C [--batch 4] [--n 4] [--max-new 64] [--skip-solo] [--keys "..."]
       [--extra "..."] [--mt-min 1]
python3 tools/batch_interleave_test.py --exe EXE --config C [--max-new 200] [--long 5000] [--extra "..."]
STRATA_KEY=... python3 tools/early_close_test.py http://127.0.0.1:8090
```

`BATCH_TEST_LOG` sets the engine log path of the batch scripts (default `<tmp>/batch_test_engine.log`).

## Model- and platform-specific notes

- `conversation_cache_isolation.py` reads `qwen4exp.embedding_length` and `qwen4exp.block_count` from the `--native`
  GGUF and needs more than 20 layers (control-vector fixture on layers 1 and 20); its image scenario uses synthetic
  `.sve` embeddings and `--vision` without an image encoder.
- `conversation_cache_soak.py` reads `/proc/<pid>/status` for RSS (Linux only, optional).
- Measured by upstream with `conversation_restore_speed.py` (RTX 5070, 0.1.38, #528): Q2_0 37K tokens with
  `--kv-resident 16384`, IQ3_XXS 50-75K tokens with `--kv-resident 32768`, in all three modes - restored turns decode
  at the live turns' speed with the same tokens (`conversation_restore_speed.py:13-14`).
- Measured by upstream with `parking_test.py` (DETAILS.md): on a 4-GPU layer split a 9,276-token conversation restored
  in 51 ms and its follow-up decoded exactly the live tokens.

## Tests

Offline unit tests (no GPU, no model), run from the repository root because they import `tools.<module>`:

```
python -m unittest tools.test_conversation_cache_parity tools.test_conversation_cache_soak \
    tools.test_conversation_cache_disabled tools.test_conversation_cache_growth \
    tools.test_conversation_cache_isolation tools.test_conversation_cache_http_smoke \
    tools.test_conversation_restore_speed
```

Each builds a complete evidence fixture and checks that it passes and that every missing or altered piece fails,
including under `python -O`, and that the dry run touches nothing. `test_conversation_cache_http_smoke.py` uses a
`FakeClient`. `batch_test.py`, `batch_interleave_test.py`, `parking_test.py` and `early_close_test.py` have no unit
tests.

## Gotchas

- `parking_test.py` reads the log from the hardcoded `/tmp/batch_test_engine.log` (`parking_test.py:55`), while
  `batch_test.Engine` writes to `BATCH_TEST_LOG` or the platform's temp directory; on Windows or with
  `BATCH_TEST_LOG` set the parking test reads the wrong file.
- `conversation_cache_disabled.py` names its reference arm "upstream" and its docstrings refer to "this PR": the
  conversation-cache scripts came with a contribution and compare a build with the cache code against one without it.
- `conversation_cache_soak.py` requires the longest prompt to exceed `kv_resident` and reach 90% of the context, so
  the config must allow it.

## In UnbiasedStrata

All of these scripts and their tests were removed in fork commit 8894a81 and not restored (the fork's restore commit
c00942d leaves the conversation-cache tests out explicitly). The fork has no Python server for them to import.

## See also

- [calibration-and-benchmarks.md](calibration-and-benchmarks.md), [python-tests.md](python-tests.md)
- [../serve/README.md](../serve/README.md) (`StrataEngine`, the engine protocol, `/metrics`)
