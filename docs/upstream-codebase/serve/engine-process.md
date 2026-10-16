[Upstream docs](../README.md) › [Serve](README.md) › Engine process and protocol

# Engine process, line protocol and supervision

## Purpose

How `serve/server.py` starts the native engine (`strata --serve`), the exact line protocol both sides speak over
the engine's stdin and stdout, how a request runs on it (heartbeats, cancellation, the #481 silence timeout), and how
the server restarts, unloads and closes it. Also the image encoder process (`strata-vision`), which the server
drives the same way.

Files covered: `serve/server.py` (`Engine`, `MockEngine`, `StrataEngine`, `Vision`, the argument and environment
builders, `Service.ensure_loaded` / `load` / `unload`), and the engine's side in `src/program/generate.cpp` (the
`--serve` block starts at `src/program/generate.cpp:4607`).

## The engine boundary

`Engine` (`serve/server.py:98-100`) is a `Protocol`: an object with `max_context` and
`generate(ids, max_new, sampling, cancel) -> Iterator[int]`. Two implementations:

| Class | Where | What it is |
| --- | --- | --- |
| `MockEngine` | `serve/server.py:103-126` | Replays a scripted completion as token ids, then `<\|im_end\|>`; a list of scripts is used one per request (the last repeats). `--engine mock` (the default of `main`). Used by nearly every test |
| `StrataEngine` | `serve/server.py:391-1259` | The resident native engine process |

`Service` (`serve/server.py:1650`) only calls `generate` and, when present, `alive`, `restart`, `unload`,
`exit_code`, `death_note`, `vram`, `close`, `info`, `last`, `progress`. Checks such as `hasattr(engine, "restart")`
decide what `/v1/status` and `/props` advertise (`auto_load`, `models_autoload`).

## Starting the engine (`StrataEngine.__init__`, `serve/server.py:421-515`)

1. `self.spawn` keeps `(exe, args, cwd, log, env)` so `restart()` can run `__init__` again.
2. `model_path` is the value after `--native` (else `--pack`); `max_context` starts as `--max-context` (4096 if
   absent). The version comes from `BUILD.json` next to the exe, overridden later by `INFO engine=` (engine 0.1.8+).
3. With `lazy=True` (`--lazy` / `"lazy_load": true`) it returns here: nothing is started, `unloaded` stays true.
4. The log file (`"log"` in the config) is opened for append; `log_start` records its size so later readers only
   look at this start's lines. Two daemon threads read the log: `narrate_start` (`serve/server.py:258`) prints what
   the start is doing (weights, "loading the experts into RAM", "YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3
   MINUTES NOW", "filling the GPU's expert cache", "almost ready", a heartbeat every 20 s), and, with
   `STRATA_REQUEST_LINES` set, `echo_requests` (`serve/server.py:214`) restates each engine request summary as one
   stdout line.
5. `subprocess.Popen([exe, "--serve", *args], cwd, stdin=PIPE, stdout=PIPE, stderr=<log>, text=True,
   encoding="utf-8", bufsize=1, env)` (`serve/server.py:454-455`); on Windows `contain(proc)` puts it in a
   kill-on-close job ([windows.md](windows.md)).
6. It reads stdout lines until `READY`: every `INFO k=v ...` line is parsed into `self.info` (ints where numeric);
   `READY <ctx> [stop]` sets `max_context` and `can_stop` (`serve/server.py:458-467`).
7. If no `READY` came (the engine exited), it raises `RuntimeError("the engine exited before it was ready (see
   <log>)" + start_failure_hint + start_log_tail)`: `start_failure_hint` (`serve/server.py:174`) relays the engine's
   `strata mtp:` advice when the MTP draft head did not fit VRAM (#474), and `start_log_tail` (`serve/server.py:193`)
   appends the last 20 lines this start wrote (#496).
8. `known_ctx = max_context` (survives a failed restart), `ended = False`, a line queue and the `_pump` thread
   (`serve/server.py:517`) that moves stdout lines into it (`BT`/`BDONE` lines go to per-slot queues). When stdout
   closes the pump sets `ended` and puts `None` (end marker) in every queue.
9. Batch-slot state is set up from `INFO batch_slots=N` ([parallel-requests.md](parallel-requests.md)).

### The engine's command line

The run config's `"args"` (written by `setup.py`) are the engine's arguments. `engine_args(cfg)`
(`serve/server.py:1461-1479`) adds:

| Added | When | Source |
| --- | --- | --- |
| `--layer-split <value>` | more than one GPU in `"gpu"` and no `--layer-split` in args | `layer_split_value` (`serve/server.py:1428`): `"auto"` or rising first-layer numbers, e.g. `"24,36,42"`; per-card counts are refused with an example |
| `--split-skip-if-fits` | several GPUs and `"split_skip_if_fits"` | config |
| `--vram-elastic` [`--vram-segment-mib N`] | `"vram_elastic": true` | config (#533) |
| `--batch N` | `"parallel": N`, 2..8 (more is passed on with a warning) | `parallel_args` (`serve/server.py:1485`) |
| `--expert-profile-save P` [`--expert-profile-save-every M`] | `"expert_profile_save"` | `learned_profile_args` (`serve/server.py:1521`); if the saved profile's header matches the configured `--expert-profile`'s shape (`profile_shape`, magic `STRP`), the start uses the saved one instead (#477) |
| `--tail-role-token <id>` | `"effort_position": "end"` and the exe contains that flag string | `effort_end_args` (`serve/server.py:1392`, #458) |

The final command is `exe --serve <args>`. A relative `"exe"` is made absolute against the config's `"cwd"`
(`serve/server.py:3992`) because Windows' `CreateProcess` would not resolve it.

### The engine's environment (`child_env`, `serve/server.py:1560-1575`)

- `"backend": "hip"` with GPUs: `HIP_VISIBLE_DEVICES` from `hip_visible(cfg)` (`serve/server.py:1544`), which prefers
  `"hip_ordinal"` for one card (#325: an integrated Radeon can take HIP ordinal 0 on Windows).
- Otherwise, with `"gpu"`: `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<gpus>` (numbered as nvidia-smi,
  issue #51).
- Every `"env"` entry of the config (e.g. AMD's GEMM tuning table).
- `"lib_dirs"` that exist are prepended to `PATH` (Windows) or `LD_LIBRARY_PATH`.

The image encoder gets the same environment unless the config's `"vision"` names `"cuda_device"`, which pins it to
that card alone (`vision_env`, `serve/server.py:1578`, #408).

## The line protocol

Line-oriented UTF-8 over the engine's stdin (server to engine) and stdout (engine to server). The engine's stderr
is its log. Upstream engine 0.1.39.

### Server to engine

| Line | Meaning | Server code | Engine code |
| --- | --- | --- | --- |
| `GEN <max_new> [key=value ...] <id>,<id>,...` | a text request | `generate`, `serve/server.py:1119-1122` | parsed at `src/program/generate.cpp:6239-6297` |
| `GENI <max_new> [key=value ...] <file> <id>,...` | with images: `<file>` is the combined embeddings file | same | same; refused without `--vision` (`generate.cpp:6299`) |
| `STOP` | stop the running request at its next check | `generate`'s `finally`, `serve/server.py:1186`; `_control` for batch | handled on the reader thread (`generate.cpp:5628`); a STOP between requests is cleared when the next one starts (`generate.cpp:6237`) |
| `VRAM [<reserve_mib>]` | resize the expert cache between requests (#533, `--vram-elastic`) | `StrataEngine.vram`, `serve/server.py:623-652` | `generate.cpp:6203-6210` |
| `QUIT` | leave the serve loop; a learned expert profile is saved after it (`generate.cpp:7466`) | `close`, `serve/server.py:1234` | `generate.cpp:6231` |
| `BGEN`, `BGENI`, `BSTOP`, `BYIELD` | batch slots | [parallel-requests.md](parallel-requests.md) | `generate.cpp:6197-6227` |

Sampling keys between `max_new` and the ids, built by `StrataEngine.sampling_keys` (`serve/server.py:655-699`):

| Key | Sent when | Engine default |
| --- | --- | --- |
| `temperature=F` | `temperature > 0` (0 or absent means greedy, not sent) | 0 (greedy) |
| `top_p=F` | `top_p < 1` | 1.0 |
| `top_k=N` | an int `>= 0`; 0 or more than 64 becomes 64 ("the engine's sampled path keeps at most 64 candidates") | 20 |
| `min_p=F` | `0 < min_p <= 1` | 0 |
| `penalty_repeat=F`, `penalty_freq=F`, `penalty_present=F` | `repetition_penalty != 1`, `frequency_penalty != 0`, `presence_penalty != 0` | off |
| `penalty_last_n=N` | with any penalty: the request's value, else 64 | 0 |
| `seed=N` | a positive int | 0 |
| `pcie_frac=F`, `spec_min_p=F` | from `strata_tune: {...}` in the request (setup's calibration, `tools/calibrate.py`) | the engine's flags |
| `cvec=0/1` | `experimental_speed_projection` is a bool (`projection_key`, `serve/server.py:702`) | 1 |

The engine skips unknown keys; the ids start at the first token without `=` (`generate.cpp:6262-6284`).

### Engine to server

| Line | Meaning | Engine code | Server handling |
| --- | --- | --- | --- |
| `INFO context=... kv=... kv_resident=... expert_slots=... expert_cache_mib=... expert_slots_primary=... expert_cache_primary_mib=... spec=... mtp_max=... lookup=... vram_free_mib=... cvec=... arena_mib=... pool_workers=... pcie_frac=... spec_min_p=... conversation_cache_mib=... conversation_cache_slots=... conversation_cache_min_free_mib=... tail_role_token=... vram_elastic=... [batch_slots=N slot_cache=0/1] engine=<version>` | start-up facts | `generate.cpp:5696-5717` | `self.info` (Monitor, About, `/v1/status`); `arena_mib` feeds `warn_tight_ram`, `vram_free_mib` feeds `desktop_vram_note` |
| `READY <max_context> stop` | ready; `stop` = honours `STOP` | `generate.cpp:5743` | ends the start-up read |
| `RESUME <n>` | `n` prompt tokens are reused from a checkpoint | `generate.cpp:6665` | sets where the first chunk starts (silence timeout) |
| `PP <reached> <total> <ms> <tok/s>` | prompt-reading progress, one per chunk | `generate.cpp:6728` (also `5388`, batched) | `progress`, `prefill_tok_s_mean`; a heartbeat (keep-alive); sets the silence allowance |
| `REUSED <n>` | prompt read, decoding starts | `generate.cpp:7045` | ignored (any line resets the silence clock) |
| `T <id>` | one generated token | `generate.cpp:7155` | yielded to `Service.run` |
| `DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> <accepted> <offered> <reused> <hits> <lookups> <ram_blobs> <file_blobs> <file_mb> <prompt_read> <offloaded>` | the request ended; `finish` is `stop`, `length` or `cancel` | `generate.cpp:7344-7349` | `_parse_done` (`serve/server.py:608-621`) reads every field, tolerating older engines with fewer |
| `ERR <message>` | the request (or a command) failed | many, e.g. `generate.cpp:6391` (`prompt (N tokens) + max_new (M) exceeds the context (C)`) | raised as `ValueError` |
| `VRAM reserve_mib=... expert_slots=... expert_slots_full=... expert_cache_mib=... expert_cache_full_mib=... vram_free_mib=... prompt_chunk=...` | reply to `VRAM` | `generate.cpp:5872` | `StrataEngine.vram` |
| `BADM`, `BT`, `BDONE`, `YIELDED` | batch slots | `generate.cpp:7383`, `6023`, `6030`, `7343` | [parallel-requests.md](parallel-requests.md) |

Upstream's engine has no `STATS` command (the fork added one); upstream's server gets the same figures from `INFO`,
the `DONE` fields and the engine log.

The engine reads stdin on its own thread with `read(2)` (`_read` on Windows) rather than `std::cin`, because glibc's
`exit()` would otherwise wait for stdin's lock forever (`generate.cpp:5591-5635`). It refuses a request when
`prompt + max_new + 8 > context` (`generate.cpp:6390`); the server keeps the same 8-token margin (`CTX_SLACK`,
`serve/server.py:80`).

## A request on the engine (`StrataEngine.generate`, `serve/server.py:1109-1209`)

1. With batch slots, delegate to `generate_batched`. Otherwise reset `progress`, build the header and write
   `header + ",".join(ids) + "\n"`. A failed write is `EngineDied("the engine stopped unexpectedly (exit code N)")`.
2. Read the line queue with a timeout. A quiet 10 s yields `None` (a heartbeat the HTTP layer turns into an SSE
   comment). `T` yields the id; `PP` updates progress and yields `None`; `DONE` is parsed and ends the request;
   `ERR` raises `ValueError`; the end marker raises `EngineDied`. A set `cancel` returns at the next line or tick.
3. `finally`: if the request did not reach its `DONE` (the consumer stopped early: client gone, stop token,
   thinking budget, repeat stop), the server writes `STOP` (when the engine said `stop` in `READY`) and drains the
   queue up to the request's own `DONE` or `ERR`, so the next request never reads this one's leftovers.

`Service.run` closes the generator explicitly while still holding the FIFO (`serve/server.py:2343-2352`) for the
same reason.

### Silence timeout (#481)

A request whose engine prints nothing for too long has "lost step" with the server: the engine waits for a command,
the server for a `DONE`. `generate` keeps an allowance (`serve/server.py:1127-1166`):

- until the first line: `engine_silence_s + min(prompt tokens, 32768) / 50` seconds (`ENGINE_SILENCE_S = 300`,
  `PP_CHUNK_MAX = 32768`, `PP_FLOOR_TOK_S = 50`, `serve/server.py:88-94`);
- after a `PP`: `max(engine_silence_s, 3 x chunk tokens / rate)` (`PP_SLACK = 3`);
- after a `T`: `engine_silence_s`.

Past it, `_silent()` (`serve/server.py:1211`) kills the process, sets `silent_note`, and raises `EngineSilent`
(an `EngineDied`): the request fails and the next one restarts the engine. The same applies to a `STOP` the engine
never acknowledges ("did not finish the request after it was stopped (STOP) within N s"). `"engine_silence_s": 0`
waits forever (`engine_silence_s()`, `serve/server.py:1417`).

### Death, restart and messages

- `death_note()` (`serve/server.py:534`) explains a dead engine from its log tail: its own watchdog line
  (`issue #29`), its last `strata`/`ERR` line with the exit code, or "the usual cause is running out of RAM".
  `Service._say_died` prints it.
- The engine's own stall watchdog (`STRATA_WATCHDOG_S`, default 60 s, `generate.cpp:5718-5741`) aborts a request
  that stops making progress; the server sees the process end.
- `restart(tries=3)` (`serve/server.py:577-606`): `close()` the old process, set `starting` (requests answer 503
  "the engine is starting" meanwhile, #344), run `__init__` again; a start that fails before `READY` is retried
  after 15 s (`RESTART_RETRY_S`, PR #637: a dead engine's VRAM can take a while to come back, notably on ROCm).
  `info` keys from before are kept under the new ones.
- `Service.ensure_loaded` (`serve/server.py:1773-1811`), called by every request under the FIFO: if the engine is
  not alive (unloaded, died) or the vision encoder is down, run `before_load`, wait up to 15 s for
  `min_free_vram_mib` (else `GpuBusy`, a 503), restart the encoder first, then `engine.restart()`, then re-apply the
  last `/v1/vram` reserve.
- `Service.prepare` uses `known_ctx` when `max_context` is 0 after a failed restart, so a request still reaches
  `run()` and restarts the engine instead of failing with "exceeds the context (0)" (`serve/server.py:2184-2192`).

### Unload and close

- `close()` (`serve/server.py:1226-1259`): `QUIT`, close stdin (so Windows' detached stdin reader sees EOF), wait
  20 s; then `terminate` and wait 20 s; then `kill` and wait 20 s; still running raises `EngineStuck` ("Strata is
  still releasing GPU/RAM; retry unloading after it exits"), and the server keeps ownership instead of claiming the
  memory is free. Once exited: join the pump, close pipes and the log.
- `unload()` = `close()` + `unloaded = True` (spawn arguments kept). `Service.unload` (`serve/server.py:1874`)
  refuses with `"busy"` when a request runs or waits, also unloads the vision encoder, and is used by
  `POST /unload`, `POST /v1/unload` and the idle-unload thread.
- `main()` shuts down on Ctrl+C or SIGTERM (SIGTERM is mapped to `KeyboardInterrupt`, #96 for `docker stop`):
  `httpd.shutdown`, `engine.close`, `vision.close`, `hub.close`; a second Ctrl+C kills the engine at once
  (`serve/server.py:4121-4140`).

## The image encoder process (`Vision`, `serve/server.py:1262-1379`)

With a `"vision"` section in the config, `main` starts `strata-vision` (built from `tools/vision/strata_vision.cpp`)
before the engine (`serve/server.py:3974-3981`), so a GPU encoder takes its VRAM before the engine sizes its expert
cache:

    <exe> --mmproj <mmproj> --model <model> [--gpu] [--threads N] [--max-tokens N]

Protocol (`tools/vision/strata_vision.cpp:10-13`): it prints `READY <n_embd>`; then each `ENC <image> <out>` line
is answered `OK <n_tokens> <nx> <ny> <ms>` or `ERR <message>`; `QUIT` ends it. The output file is `SVE1`, `n`, `nx`,
`ny`, `n_embd` (int32) then `n x n_embd` floats.

- `Vision.load` takes a `data:` URL, an http(s) URL (60 s timeout) or a local path; `Vision.normalize` passes JPEG,
  PNG, BMP and GIF through and converts anything else to PNG with Pillow (transparent areas become white).
- Encodings are cached by SHA-256 of the image bytes (64 entries, oldest dropped), so a picture sent again every
  turn is encoded once. Encoding takes the FIFO: "the engine and the image encoder must not run on the GPU at the
  same time" (`serve/server.py:2157-2162`).
- `Service.prepare` (`serve/server.py:2150-2183`) expands each template-written `<|vision_start|><|image_pad|>`
  into one `<|image_pad|>` per image token; a literal `<|image_pad|>` in text (#150) is re-encoded as plain text.
  The images' files are concatenated into `req-<id>.sve` and sent with `GENI`; the file is deleted after the
  request.
- Lazy start is refused with vision (`serve/server.py:3972`).

## Configuration

| Knob | Where | Effect |
| --- | --- | --- |
| `"exe"`, `"args"`, `"cwd"`, `"log"`, `"env"`, `"lib_dirs"`, `"backend"`, `"gpu"`, `"hip_ordinal"` | run config | the engine's command line, directory, log and environment |
| `"layer_split"`, `"split_skip_if_fits"`, `"vram_elastic"`, `"vram_segment_mib"`, `"parallel"`, `"expert_profile_save"`, `"expert_profile_save_every"`, `"effort_position"` | run config | engine flags added by the server (table above) |
| `"engine_silence_s"` | run config | #481 timeout, default 300, 0 = off |
| `"vision"` (`exe`, `mmproj`, `model`, `gpu`, `threads`, `max_tokens`, `cuda_device`) | run config | the image encoder |
| `--gpu` | `server.py` flag | overrides `"gpu"` |
| `--lazy` / `"lazy_load"` | flag / config | start without the engine |
| `STRATA_REQUEST_LINES=1` | env | one `[strata] request ...` line per finished request |
| `STRATA_BATCH_TRACE=1` | env | `[batch-trace]` lines on stderr (`btrace`, `serve/server.py:380`) |
| `STRATA_DEBUG=1` | env | per-request summary and the raw model output in the server window |
| `STRATA_WATCHDOG_S`, `STRATA_TRACE` | env (inherited by the engine) | the engine's stall watchdog and per-step trace |

## Platform notes

- Windows: job objects (`contain`), stdin closed after `QUIT`, absolute exe path, `PATH` for DLLs; see
  [windows.md](windows.md).
- AMD (HIP): device numbering through `HIP_VISIBLE_DEVICES` and `hip_ordinal`; a Linux desktop on the same AMD
  card triggers `desktop_vram_note` (`serve/server.py:3625`, #560 #516) recommending `--vram-reserve-mib 3072`.
- Several GPUs: the layer split is added by the server; the Monitor reads every card ([monitoring-and-telemetry.md](monitoring-and-telemetry.md)).

## Tests

- `serve/test_server.py`: `EngineDeath`, `SilentEngine`, `LostStep` (#481), `RestartWindow` (#344, retries,
  `known_ctx`), `StartFailureLog`, `DraftHeadHint`, `StartNarrator`, `CancelledRead`, `LayerSplit`, `LearnedProfile`,
  `GpuChoice`, `SamplingKeys`, `ClientHangUp`, `SharingTheGpu`.
- `serve/test_lifecycle.py`: `close()` order (EOF before waiting, terminate before kill, a process still exiting),
  lazy construction, unload and reload over HTTP.
- `serve/test_vram.py`: the `VRAM` command and `/v1/vram`.
- `serve/test_parallel.py`: the batch protocol with a scripted Python engine.
- Run: `python -m unittest serve.test_server serve.test_lifecycle -v` from the repository root ([tests.md](tests.md)).

## Gotchas

- `generate` yields `None` for every `PP` line too, so a long prompt read sends one keep-alive per chunk plus one per
  quiet 10 s.
- Only a `DONE` replaces `engine.last`; `Service.run` compares object identity so a request that died or was
  disconnected never records the previous request's figures as its own (`serve/server.py:2258-2260`).
- A `RuntimeError` from a failed start inside `restart()` propagates to the request (it is not one of the handled
  exception types in `do_POST`).

## In UnbiasedStrata

Removed with `serve/` (commit 8894a81). The fork's C++ front end keeps the model (fork/exec of the engine in
`--serve` mode, `GEN`/`GENI`/`STOP`/`QUIT` in, `INFO`/`READY`/`T`/`DONE`/`ERR` out) but reads only the first seven
`DONE` fields, never sends `VRAM` or the batch commands, adds a `STATS` command (engine change e41dfc7), and exits
for a supervisor restart instead of restarting the engine itself; see
[the fork's engine-process page](../../codebase/frontend/engine-process.md).

## See also

- [http-api.md](http-api.md), [parallel-requests.md](parallel-requests.md), [monitoring-and-telemetry.md](monitoring-and-telemetry.md)
- [../../upstream/DETAILS.md](../../upstream/DETAILS.md) - Troubleshooting (#29, #481), "Sharing the GPU with other programs".
