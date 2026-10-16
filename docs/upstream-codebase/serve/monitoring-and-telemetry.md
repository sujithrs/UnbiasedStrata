[Upstream docs](../README.md) › [Serve](README.md) › Monitoring and telemetry

# Monitoring: `/metrics`, `/status`, `/v1/status`, telemetry and the request monitor

## Purpose

What the server knows about itself and exposes: the live state of the running request, the history of finished
requests and totals since start, the engine's start-up facts, the conversation cache, hardware readings (GPU, CPU,
RAM, disk) with a minute of history, the opt-in API request monitor, and the lines it prints in its window. The web
app's Monitor and About tabs, `tools/strata_mcp.py` and external front ends read these.

Files: `serve/server.py` (`Service.metrics`, `v1_status`, `_tok_s`, `_note`, `_progress`, the history in
`Service.run`, `ConvCacheLog`, `begin_request` / `request_records`, `warn_tight_ram`, `desktop_vram_note`,
`echo_requests`, `narrate_start`), `serve/telemetry.py`.

## `GET /metrics` (`Service.metrics`, `serve/server.py:2024-2073`)

Needs the API key when one is set. `?requests=all` returns every kept request instead of the last 12.

| Field | Content |
| --- | --- |
| `engine` | `model`, `max_context`, `images`, plus every `INFO` field of the engine (`kv`, `kv_resident`, `expert_slots`, `expert_cache_mib`, `spec`, `mtp_max`, `lookup`, `vram_free_mib`, `cvec`, `arena_mib`, `pool_workers`, `pcie_frac`, `spec_min_p`, `conversation_cache_*`, `tail_role_token`, `vram_elastic`, `batch_slots`, `slot_cache`, `engine`), `version`, and `vram` after a resize |
| `live` | `state` (`reading`, `generating`, `unloaded`, `idle`), `queued`, `phase`, `prompt_tokens`, `prompt_read` / `prompt_total` (from `PP` lines while reading), `generated`, `max_tokens`, `elapsed_s`, `tok_s`, `tok_s_mean`, `prefill_tok_s_mean`, `tok_s_window_s`; with batch slots also `parallel`, `running`, `slots`, `outside_slots`, `waiting` |
| `requests` | finished requests, newest first (below) |
| `requests_kept` | how many the server keeps (at most 500) |
| `totals` | since start: `since`, `requests`, `prompt_tokens`, `reused`, `output_tokens`, `prompt_ms`, `decode_ms`, `drafts_offered`, `drafts_accepted` (#35, #457) |
| `conversation_cache` | the card's data (below) |
| `hardware`, `hardware_static`, `history` | `Telemetry.snapshot()` (below) |
| `time` | server time |

### A request's history row (`Service.run`'s `finally`, `serve/server.py:2383-2457`)

Recorded while the request still holds the FIFO (#266), so the next request's status is never mixed in:
`time`, `duration_s`, `finish` (`stop`, `length`, `cancel`, `disconnect`, `error`), `prompt_tokens` (the tokens it
got through: fewer than the prompt when a cancel stopped the read, #471, `prompt_tokens_seen`), `prompt_total`,
`prompt_read`, `reused`, `output_tokens`, `engine_generated`, `prompt_ms`, `decode_ms`, `decode_tok_s`, `hit_rate`
(expert-cache hits / lookups during decoding), `pcie_share` (routed experts read over PCIe or computed by another
GPU, #588), `ram_blobs`, `file_blobs`, `file_mb` (the RAM / file expert tiers), `drafts_offered`, `drafts_accepted`,
`projection` (whether the experimental speed projection was on, when the engine has one). Engine figures come only
from this request's own `DONE` line; a request that died or disconnected records none.

### Live rates (`serve/server.py:1957-1991`)

- `tok_s`: tokens over the last `RATE_WINDOW_S = 2.0` s of `(time, generated)` samples; younger than
  `RATE_MIN_SPAN_S = 0.25` s it is the mean so far with the span floored. "A mean reads ~1/elapsed at the first
  token (the Monitor showed five-digit numbers)" (`serve/server.py:81-84`). With parallel requests: the sum.
- `tok_s_mean`: the whole-request mean since the first token, kept for comparison.
- `prefill_tok_s_mean`: the engine's rate over newly read tokens from the last `PP` line (the cached prefix
  excluded).

## `GET /status` and `GET /v1/status`

- `/status` (`serve/server.py:3034-3047`, key required because `tail` shows the end of the answer, #212): the
  current status dict (`busy`, `queued`, `phase`, `prompt_tokens`, `generated`, `max_tokens`, `tool`, `tail`) plus
  `elapsed_s`, `tokens_per_s`, `tokens_per_s_mean`. The server window's progress lines say the same.
- `/v1/status` (`Service.v1_status`, `serve/server.py:2075-2115`): "for a client that would rather ask than guess":
  `service: "strata"`, `model`, `loaded`, `auto_load`, `structured_output`, `engine` (version), `started`,
  `uptime_s`, `cache_max_tokens` and `context.native` / `max_positions` (the engine context), `concurrency.serving`,
  `dialects` (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`), `vision`, `activity` (`requests`,
  `in_flight`, `last_request_at`), `last_timings` (llama.cpp names, with `at`), `vram` (the last resize and
  `elastic`), `machine` (`gpu`: name, used / total MiB, util, temperature, power; `ram`: used / total GiB).

## `serve/telemetry.py`

`Telemetry` (`serve/telemetry.py:266-363`) runs a daemon thread that samples once a second and keeps the last 60
readings (`HISTORY`) of `gpu_util`, `gpu_mem_used`, `gpu_temp`, `gpu_power`, `gpu_pcie_rx_mb`, `cpu`, `ram_used`,
`disk_read_mb`, `tok_s`, `prefill_tok_s_mean` for the sparklines. `Service.start_telemetry` (`serve/server.py:1947`)
creates it with an `extra()` callback that adds the server's rates. Nothing in it can stop the server: every reading
that fails is `None`.

| Source | Class / function | Readings |
| --- | --- | --- |
| NVIDIA | `_Nvml` (`serve/telemetry.py:25`) | NVML through `ctypes` (`nvml.dll`, also from `Program Files\NVIDIA Corporation\NVSMI`; `libnvidia-ml.so.1`), no pip package: utilization, memory used / total, temperature, power and enforced limit, PCIe generation (current and max; it drops at idle), width, RX / TX throughput |
| AMD (`"backend": "hip"`, #301) | `_Amd` (line 136), `amd_device_dir` (line 111) | Linux amdgpu sysfs: `gpu_busy_percent`, `mem_info_vram_used` / `_total`, hwmon `temp1_input`, `power1_average` (or `power1_input`), `power1_cap`; the card is found through the KFD topology nodes in HIP's order and `drm_render_minor` |
| CPU, RAM, disk | `psutil` when installed (setup installs it) | `cpu_percent`, `virtual_memory`, disk read / write MB/s |
| CPU, RAM without psutil | `_CpuRamFallback` (line 217) | Windows `GetSystemTimes` / `GlobalMemoryStatusEx`; Linux `/proc/stat`, `/proc/meminfo`; no disk rate |
| static | `Telemetry.static` | `gpu_name` (joined with `+` for several cards), `gpu_count`, `cpu_name` (registry on Windows, `/proc/cpuinfo` on Linux), `cores`, `threads`, `psutil` |

Several GPUs (a layer split, #112): `gpu_indices` from the config's `"gpu"`; the `gpu_*` values are the total
(memory, power, PCIe), the mean (load) or the hottest (temperature), and `hardware.gpus` lists each card. GPU numbers
are nvidia-smi's (PCI bus order) or HIP's. `free_vram_mib(index, amd)` (line 190) also backs `min_free_vram_mib`.

## Conversation cache card (#596)

`ConvCacheLog` (`serve/server.py:310-364`) reads the engine log from where the current engine run started (at most
the last MiB of new lines per poll) and parses the engine's `conversation cache: parked|skipped|restored N tokens ...
parked=P bytes=B [evictions=E]` and `dropped ... superseded ... parked=P` lines. `conversation_cache_view`
(`serve/server.py:367`) combines that with the engine's `INFO conversation_cache_mib` / `_slots` and the request
history: `enabled`, `budget_mib`, `slots`, `parked`, `bytes`, `evictions`, `parks`, `restores`, `last_event`,
`last_tokens`, `last_at`, `requests`, `requests_reused`, `reused_tokens`, `prompt_tokens`, `last_reused`,
`last_prompt`. The parking itself is the engine's opt-in `--conversation-cache-mib N --conversation-cache-slots K`.

## The API request monitor (#332)

Off by default; `"api_monitor": true` or `--api-monitor`. When on, every `/v1/chat/completions`, `/v1/messages` and
`/v1/responses` request gets a record (`Service.begin_request`, `serve/server.py:1993-2009`) in a deque of 100
(`api_requests`):

- `id` (12 hex), `path`, `model`, `started_at`, `state` (`queued` -> `loading` -> `generating` -> `completed`,
  `error` or `disconnected`), `stream`, `response_format`;
- `input` (the request body as indented JSON), `output`, `reasoning`, `response` (non-streamed body), each capped
  at 262,144 characters with `*_truncated` flags; headers and keys are never recorded;
- timings: `queue_s` (FIFO waits), `load_s` (automatic loading), `first_token_s`, `wallclock_s`, `finished_at`;
  `http_status`, `usage`, `timings`, `error`, `outcome`.

The handler's `_capture` (`serve/server.py:3322-3359`) copies content from the outgoing chunks of all three dialects
without changing them. `GET /api/requests` returns summaries (without the large fields) plus `retention: 100`,
`persistent: false`, `loaded`, `auto_load`; `GET /api/requests?id=<id>` one full record or 404 "request no longer
retained". The page is described in [web-app.md](web-app.md).

## The server window

| Line | Where |
| --- | --- |
| start narration ("reading the model's weights", "loading the experts into RAM ... YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES NOW", "experts loaded", "filling the GPU's expert cache", "almost ready", "still starting (N s)") | `narrate_start`, `serve/server.py:258` (#505: the words follow the engine flags `--resident-budget-gib`, `--resident-experts`, `--mmap-experts`) |
| `WARNING: RAM is tight ...` when the experts leave under 6 GB of RAM (needs psutil) | `warn_tight_ram`, `serve/server.py:3595` (issue #27) |
| AMD + Linux desktop + under 2048 MiB VRAM free: a note recommending `--vram-reserve-mib 3072` | `desktop_vram_note`, `serve/server.py:3625` (#560, #516) |
| `ready: http://.../v1 (...)`, the LAN addresses, firewall hints | `main`, `serve/server.py:4096-4114` |
| `reading the prompt: X of Y tokens, N s so far` / `<phase>: N of max M tokens, R tok/s, N s` (at most once a second) | `Service._progress`, `serve/server.py:2226` |
| `done: N tokens in S s (R tok/s) (finish, cancel=...), expert cache H% hit (+P% ... over PCIe)` | `Service.run`, `serve/server.py:2439-2440` |
| "the reply reached max tokens while still thinking" (#530), repeat stop (#606), thinking budget reached (#123), engine errors and deaths with `death_note` | `Service.run` |
| `[strata] request prompt P cached C output O prompt_read R ms total S ms prefill X tok/s decode Y tok/s` | `echo_requests`, with `STRATA_REQUEST_LINES=1` |

The engine's own log is the `"log"` file of the config (`strata-<model>.log` per upstream's Troubleshooting).

## Tests

`serve/test_server.py`: `LiveRate` (the live number is a rate; prefill excludes cached tokens; no counters without a
`DONE`), `DraftCounts`, `PcieShare`, `CancelledRead`, `UsageAndStatus.test_v1_status`, `StatusHandover` (#266),
`StatusNeedsTheKey`, `WebApp.test_metrics`, `AmdTelemetry` (sysfs readings from a fake tree, free VRAM on HIP),
`DesktopVramNote`, `StartNarrator`. `serve/test_monitor.py`: monitor off by default, output and reasoning for every
dialect and stream mode, errors and engine death are not successes, history needs the key and records no headers,
bounded capture and eviction, queue wait and loading attributed separately, the conversation cache log and card.

## Gotchas

- `/metrics` and `/v1/status` read hardware from the telemetry thread, which `serve()` starts; a `Service` used
  without `serve()` (tests) reports empty hardware.
- The disk rate needs psutil.
- `pcie_share` is not part of `hit_rate`: the hit rate is the VRAM share of the experts not sent over PCIe.

## In UnbiasedStrata

Removed (commit 8894a81). The fork's front end has only `GET /health`, which returns the engine's `STATS` reply
(tier census and counters, a command the fork added) and logs one start / end line per request to stderr; there is
no hardware telemetry, history or monitor.

## See also

- [web-app.md](web-app.md), [engine-process.md](engine-process.md), [parallel-requests.md](parallel-requests.md)
- [../../upstream/DETAILS.md](../../upstream/DETAILS.md#api-request-monitor) - "API request monitor".
