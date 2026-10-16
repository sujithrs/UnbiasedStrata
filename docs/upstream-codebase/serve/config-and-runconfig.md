[Upstream docs](../README.md) › [Serve](README.md) › Configuration

# Configuration: flags, `strata-<model>.json`, environment, and `runconfig.py`

## Purpose

Everything that configures `serve/server.py`: its command-line flags, the keys of the run config
`strata-<model>.json` it reads (the file `setup.py` writes and the run scripts pass with `--config`), environment
variables, the shared Chat settings file, and `serve/runconfig.py`, which lets the web app's About tab change a few
documented keys of the run config.

Files: `serve/server.py` (`main`, `serve/server.py:3898-4141`; `sampling_defaults_from_config`,
`clean_shared_defaults`, `engine_args` and friends), `serve/runconfig.py`, `serve/mcp.py` (`settings_from`,
`hub_from_config`).

## How the server is started

`setup.py`'s `write_run_script` (`setup.py:3453-3467`) writes `run-<model>.bat` / `run-<model>.sh`:

    <python> serve/server.py --engine strata --config strata-<model>.json --port <port> [--open]

`setup.py:3093` starts the same command directly after an install. `tools/strata_mcp.py` runs it without `--open`.
The config is read as UTF-8 with an optional BOM (`utf-8-sig`, "Notepad adds a BOM", `serve/server.py:3937`).

## Command-line flags (`main`, `serve/server.py:3899-3936`)

| Flag | Default | Meaning | Config equivalent |
| --- | --- | --- | --- |
| `--engine mock\|strata` | `mock` | the scripted test engine or the real one (`strata` needs `--config`) | |
| `--config PATH` | | the run config | |
| `--host ADDR` | `127.0.0.1` | listen address (`0.0.0.0`: also other devices; set a key) | `"host"` |
| `--port N` | 8095 | listen port | (setup's `"port"` goes into the run script) |
| `--gpu N\|"0,2"` | | GPU(s) as nvidia-smi numbers them; several = layer split | `"gpu"` |
| `--tokenizer DIR` | `pack/full/tokenizer` | pack tokenizer (byte tokenizer if absent; required for `strata`) | `"tokenizer"` |
| `--script TEXT` (repeatable) | a short greeting | the mock engine's answers, in turn | |
| `--open` | off | open the browser when ready, unless `"open_browser": false` (#609) | `"open_browser"` |
| `--fit-max-tokens` | off | clamp `max_tokens` to the room left instead of 400 | `"fit_max_tokens"` |
| `--api-key KEY` | `$STRATA_API_KEY` | require this key | `"api_key"` |
| `--mcp-config PATH` | | a Claude Desktop style `{"mcpServers": {...}}` file; its entries win over the config's | `"mcp_servers"` / `"mcpServers"` |
| `--lazy` | off | start unloaded; the first request loads (text only) | `"lazy_load"` |
| `--api-monitor` | off | keep the last 100 requests for `/api-monitor` | `"api_monitor"` |
| `--idle-unload SECONDS` | never | unload after that long without requests | `"idle_unload_s"` |
| `--min-free-vram-mib M` | 0 | load only with M MiB free | `"min_free_vram_mib"` |
| `--before-load CMD` | | run before a reload | `"before_load"` |

Flags win over config keys where both exist.

## Run config keys read by the server

Setup writes some keys itself and rewrites them on every setup run (`SETUP_KEYS`, `setup.py:2694-2695`: `exe`,
`args`, `cwd`, `tokenizer`, `model_name`, `log`, `lib_dirs`, `port`, `backend`, `env`, `gpu`, `gpus_asked`,
`layer_split`, `host`, `api_key`, `draft_vocab`, `vision`); every other key is the user's and is kept (#629).

| Key | Default | Read at | Meaning |
| --- | --- | --- | --- |
| `exe` | required | `serve/server.py:3992` | engine executable (relative to `cwd`) |
| `args` | required | `engine_args` | engine arguments |
| `cwd`, `log` | | `StrataEngine` | engine working directory; engine log (stderr), also read by the start narrator and the Monitor |
| `env`, `lib_dirs`, `backend`, `gpu`, `hip_ordinal` | | `child_env` | engine environment ([engine-process.md](engine-process.md)) |
| `tokenizer`, `model_name` | `pack/full/tokenizer`, `qwen3.8-flash-next` | `main` | tokenizer folder; the reported model id |
| `vision` | none | `main`, `Vision` | image encoder: `exe`, `mmproj`, `model`, `gpu`, `threads`, `max_tokens`, `cuda_device` |
| `layer_split`, `split_skip_if_fits` | `auto`, off | `engine_args` | multi-GPU split (#644) |
| `vram_elastic`, `vram_segment_mib` | off | `engine_args` | elastic expert cache for `/v1/vram` (#533) |
| `parallel` | 1 | `parallel_args` | batch slots, 2..8 (#465) |
| `expert_profile_save`, `expert_profile_save_every` | off, 10 min | `learned_profile_args` | save and reuse the learned expert profile (#477) |
| `effort_position` | `start` | `effort_end_args` | `end`: the effort goes before the answer (#458) |
| `engine_silence_s` | 300 | `engine_silence_s()` | #481 timeout; 0 = wait forever |
| `lazy_load` | false | `main` | start unloaded |
| `sampling` | none (greedy) | `sampling_defaults_from_config`, `serve/server.py:3840` | defaults for requests that leave fields out: `temperature` (>= 0), `top_p` (0 < x <= 1), `top_k` (1..64), `min_p` (0..1), `presence_penalty`, `frequency_penalty` (>= 0), `repetition_penalty` (> 0), `penalty_last_n` (>= 0), `seed` (> 0), `experimental_speed_projection` (bool). A bad value stops the start; unknown keys are named and ignored |
| `fit_max_tokens` | false | `main` | #545 |
| `reasoning_budget_tokens` | 0 (none) | `main` | default thinking cap for every request (#123) |
| `repeat_stop_tokens` | 256 | `main` | #606; 0 = off |
| `anthropic_thinking` | `model` | `main` | `on_request`: Anthropic requests think only when asked (#278) |
| `aliases` | none | `Service.set_aliases` | other model names (#297) |
| `host`, `api_key`, `allowed_hosts`, `trusted_origins`, `cors_origins` | | `main` | [security.md](security.md) |
| `api_monitor` | false | `main` | the API request monitor (#332) |
| `idle_unload_s`, `min_free_vram_mib`, `before_load` | off | `main` | sharing the GPU with other programs |
| `open_browser` | true | `main` | `false` overrides `--open` (#609) |
| `mcp_servers` (or `mcpServers`), `mcp` | none | `hub_from_config` | MCP servers and their limits ([tools-and-mcp.md](tools-and-mcp.md)) |

Validation happens before the minutes-long engine start wherever possible (aliases, `engine_silence_s`,
`effort_position`, `layer_split`, `repeat_stop_tokens`, `anthropic_thinking`, `reasoning_budget_tokens`, origins,
allowed hosts, MCP entries): a bad value exits with `[strata] config ...`.

## Environment variables

| Variable | Effect |
| --- | --- |
| `STRATA_API_KEY` | default for `--api-key` (an empty value is refused) |
| `STRATA_ALLOWED_HOSTS` | extra Host names, comma-separated |
| `STRATA_REQUEST_LINES=1` | one `[strata] request prompt P cached C output O prompt_read R ms total S ms prefill X tok/s decode Y tok/s` line per request on stdout, for supervisors that see only the server's output (a tray, llama-swap) |
| `STRATA_DEBUG=1` | per-request summary line (`_debug_req`), the raw model text after each request, MCP servers' stderr |
| `STRATA_BATCH_TRACE=1` | `[batch-trace]` lines for the batch path |
| `STRATA_PARALLEL_SOLO=0` | keep a lone request in its batch slot ([parallel-requests.md](parallel-requests.md)) |
| engine variables (`STRATA_WATCHDOG_S`, `STRATA_TRACE`, ...) | inherited by the engine through `child_env` |

## Shared Chat settings (`/settings`)

The web app's Sampling drawer can make its settings the defaults for other API clients ("Use for other apps too").
`clean_shared_defaults` (`serve/server.py:3803`) accepts only `SHARED_KEYS` (`serve/server.py:3800`):
`reasoning_effort` (none / low / medium / high), `temperature` (0..2), `top_p` (0 < x <= 1), `top_k` (1..64),
`seed` and `max_tokens` (positive ints), `experimental_speed_projection` (bool). They are saved next to the config
as `strata-<model>.shared-settings.json` (removed when cleared) and loaded at start (`serve/server.py:4077-4085`).
`Service.with_shared` applies `max_tokens` and the effort to requests that have none; `Service.run` merges the
sampling keys under the request's own.

## `serve/runconfig.py` - the About tab's Model settings (#564)

A short list of documented keys can be read and changed from Strata's own page; everything else in the file
(setup's keys, network and security keys, MCP servers, `before_load`, other sampling keys) is never touched.

`EDITABLE` (`serve/runconfig.py:19-39`):

| Key | Kind |
| --- | --- |
| `sampling.temperature` | number >= 0 |
| `sampling.top_p` | 0 < x <= 1 |
| `sampling.top_k` | integer 1..64 |
| `sampling.min_p` | 0 <= x <= 1 |
| `reasoning_budget_tokens` | integer >= 0 |
| `fit_max_tokens` | bool |
| `anthropic_thinking` | `model` / `on_request` |
| `effort_position` | `start` / `end` |
| `aliases` | list of names (or comma-separated; each at most 128 characters) |
| `idle_unload_s` | number >= 0 |
| `lazy_load` | bool (refused when the config has `vision`) |
| `engine_silence_s` | number >= 0 |
| `api_monitor` | bool |
| `open_browser` | bool |
| `vram_reserve_mib` | integer, kept as `--vram-reserve-mib N` inside `args` |

Functions: `value_of` (`serve/runconfig.py:51`), `view` (61: the keys with value, help, kind, choices, file name),
`check` (81: the value to store or a `ValueError`; `None` removes the key), `apply` (121: every value checked first,
so one bad value changes nothing; returns the new config and the keys that changed), `save` (164: copies the old file
to `strata-<model>.json.bak`, writes a `.tmp`, `os.replace`), `load` (175).

HTTP (`serve/server.py:3260-3296`): `GET /config` (key) returns `view(...)` or 404 "this server was started without a
run config"; `POST /config` with `{"set": {"<key>": <value or null>}}` needs the key and Strata's own page (JSON, no
foreign Origin), runs under `config_lock`, and prints which keys changed. Changes take effect at the next start of
the server.

## Tests

`serve/test_runconfig.py` (`Apply`: only the named keys change, null restores the default, refusals, the view;
`Http`: read and write from the page, own page only, the API key, no run config, the page has the view);
`serve/test_server.py` (`SharedSettings`, `ModelAliases`, `EffortAtTheEnd.test_the_config`,
`SilentEngine.test_config`, `LayerSplit`, `RepeatStop`); `serve/test_parallel.py` (`ParallelArgs`).

## Gotchas

- The server ignores the config's `"port"`; the port comes from `--port` (setup writes it into the run script).
- Config changes made through `/config` are not applied to the running server.
- `before_load` as a string runs through the shell; as a list it does not.

## In UnbiasedStrata

Removed (commit 8894a81) together with `setup.py` and the run scripts: the fork has no run config. Its
`unbiased-strata` binary takes command-line options only (`--model`, `--pack`, `--mtp`, `--ctx`, `--host`,
`--port`, `--vram-cap`, `--ram-cap`, `--mmproj`, `-- <engine flags>`); see
[the fork's HTTP API page](../../codebase/frontend/http-api.md).

## See also

- [engine-process.md](engine-process.md), [security.md](security.md), [web-app.md](web-app.md)
- [../../upstream/DETAILS.md](../../upstream/DETAILS.md#using-it) - "Model settings in the web page (0.1.39, #564)".
