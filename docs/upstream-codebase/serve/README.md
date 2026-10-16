[Upstream docs](../README.md) › Serve

# Serve: the Python server and web app (`serve/`)

`serve/` is the program users of upstream Strata actually talk to. `serve/server.py` is a Python HTTP server
(standard library `ThreadingHTTPServer`, plus `jinja2`) that speaks three client dialects - OpenAI Chat Completions,
Anthropic Messages and OpenAI Responses - and serves a browser app (Chat, Monitor, About) and an optional API
request monitor. It renders the model's Jinja chat template itself, tokenizes with the pack's tokenizer
(`tools/strata_tokenizer.py`), and keeps one native engine process (`strata --serve`, built from
`src/program/generate.cpp`) resident, talking to it over stdin/stdout in a line protocol (`GEN ...` in,
`T <id>` / `DONE ...` out). Tokens coming back are decoded and parsed into reasoning, answer text and tool calls by
`serve/frontend.py`. Optional extras: an image encoder process (`strata-vision`), tools from MCP servers for the web
app's chat, JSON response formats, batch slots for parallel requests, unloading the model to give the GPU back, and
hardware telemetry. Callers: HTTP clients (chat apps, coding agents such as Claude Code, Codex CLI, OpenCode, omp),
the web app, `chat.py`, `tools/strata_mcp.py`; the server is started by the run scripts `setup.py` writes
(`run-<model>.bat` / `.sh`). Callees: the engine, `strata-vision`, MCP servers, NVML / amdgpu sysfs / psutil.

## Contents

| Page | What it covers |
| --- | --- |
| [engine-process.md](engine-process.md) | `StrataEngine`: how the engine is started, the exact stdin/stdout line protocol (both sides, with `generate.cpp` lines), request lifecycle, STOP and drain, the #481 silence timeout, restart, unload and close; the `strata-vision` process |
| [http-api.md](http-api.md) | Every route, `POST /v1/chat/completions` in full (request fields, thinking levels, `max_tokens` and the context, sampling defaults, SSE, errors), model discovery (`/v1/models`, `/props`, `/slots`), load / unload / lazy start / `/v1/vram` |
| [anthropic-and-responses.md](anthropic-and-responses.md) | `POST /v1/messages` and `count_tokens` (Anthropic), `POST /v1/responses` (OpenAI Responses, stateless, for Codex CLI) |
| [chat-template-and-parsing.md](chat-template-and-parsing.md) | `serve/frontend.py`: request normalization, the Jinja template, effort levels, literal `<think>` tags, the streaming `OutputParser` (reasoning, text, XML tool calls), `Detokenizer`, `chat_template.jinja`, `chat_golden.json` |
| [structured-output.md](structured-output.md) | `serve/structured.py`: `response_format` `json_object` / `json_schema` - prompt, then validate; buffered SSE; 502 on failure |
| [security.md](security.md) | API key, the Host check (DNS rebinding), the Origin check, "own page" rules, CORS, `allowed_hosts` / `trusted_origins` / `cors_origins` |
| [config-and-runconfig.md](config-and-runconfig.md) | Command-line flags of `server.py`, every key of `strata-<model>.json` the server reads, environment variables, the shared Chat settings file, and `serve/runconfig.py` (the About tab's Model settings, `GET`/`POST /config`) |
| [tools-and-mcp.md](tools-and-mcp.md) | `serve/mcp.py`: Strata as an MCP client (stdio and Streamable HTTP), the tool loop `run_with_mcp`, limits and safety |
| [mcp-server.md](mcp-server.md) | `tools/strata_mcp.py`: Strata's own MCP server for AI assistants (install, start, stop, status, logs, benchmark) |
| [parallel-requests.md](parallel-requests.md) | `"parallel": N`: the server side of the engine's batch slots (`BGEN`, `BT`, `BDONE`, `BYIELD`, slot choice, solo path), from `docs/BATCHING.md` |
| [web-app.md](web-app.md) | `serve/web/`: the Chat, Monitor and About tabs, the Sampling drawer, attachments, MCP tool blocks, the design tokens; the separate API request monitor page |
| [monitoring-and-telemetry.md](monitoring-and-telemetry.md) | `GET /metrics`, `/status`, `/v1/status`, `serve/telemetry.py` (NVML, amdgpu sysfs, psutil), request history and totals, the conversation cache card, the API request monitor, the server window's lines |
| [windows.md](windows.md) | `serve/winjob.py` (job objects) and every other Windows-specific behaviour of the server, MCP client and telemetry |
| [tests.md](tests.md) | `serve/test_*.py`, `serve/mcp_fake_server.py`, `tools/test_strata_mcp.py`: what each covers and how to run them |

## Source map

All paths are in the upstream repository at commit 6f32ec0.

| File | Lines | Role |
| --- | ---: | --- |
| [serve/server.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/server.py) | 4145 | The server: engines (`MockEngine`, `StrataEngine`), `Vision`, `Service` (queue, load/unload, prepare, run, status, metrics), the dialect adapters, the HTTP handler, host/origin checks, `main()` |
| [serve/frontend.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/frontend.py) | 676 | Text layer: `ChatTemplate`, OpenAI/Anthropic request normalization, effort mapping, `<think>` literal marking, `OutputParser` |
| [serve/responses.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/responses.py) | 538 | OpenAI Responses API: input items to chat messages, tools, `Assembler` (output items and typed SSE events) |
| [serve/structured.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/structured.py) | 147 | JSON response formats: schema prompt, validator, `validated_json` |
| [serve/runconfig.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/runconfig.py) | 179 | The editable subset of `strata-<model>.json` (`GET`/`POST /config`), checked and saved with a `.bak` |
| [serve/telemetry.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/telemetry.py) | 363 | Hardware sampler: NVML via ctypes, amdgpu sysfs, psutil or OS fallbacks; 60 s of history |
| [serve/mcp.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/mcp.py) | 676 | MCP client: stdio and Streamable HTTP transports, `McpServer`, `McpHub`, config parsing |
| [serve/winjob.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/winjob.py) | 74 | Windows: put child processes in a kill-on-close job object |
| [serve/__init__.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/__init__.py) | 1 | Package marker |
| [serve/chat_template.jinja](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/chat_template.jinja) | 183 | Fallback chat template (the original model's), used when the pack's tokenizer folder has none |
| [serve/chat_golden.json](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/chat_golden.json) | 196 | 10 template rendering cases with Hugging Face's output (`rendered` and `hf`) |
| [serve/web/index.html](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/web/index.html) | 243 | The web app's page: Chat, Monitor, About, Sampling drawer |
| [serve/web/app.js](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/web/app.js) | 1073 | The web app's logic: polling `/metrics`, chat streaming, Markdown, attachments, MCP blocks, settings |
| [serve/web/app.css](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/web/app.css) | 184 | The app's layout |
| [serve/web/components.css](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/web/components.css) | 128 | UI kit components (`st-*` classes) |
| [serve/web/tokens.css](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/web/tokens.css) | 87 | Design tokens (light / dark), the Outfit `@font-face` rules |
| [serve/web/sprite.svg](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/web/sprite.svg) | 33 | Icon sprite (`#i-chat`, `#i-activity`, ...) |
| [serve/web/monitor.html](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/web/monitor.html) | 37 | The API request monitor page (`/api-monitor`) |
| [serve/web/monitor.js](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/web/monitor.js) | 111 | Its logic: `/v1/status`, `/api/requests`, load / unload buttons |
| [serve/web/fonts/](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/web/fonts/OFL.txt) | 93 (OFL.txt) | Outfit variable font (`outfit-latin-wght.woff2`, `outfit-latin-ext-wght.woff2`, binary) and its SIL OFL 1.1 licence |
| [serve/test_server.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_server.py) | 2685 | The main test file (budgets, parsing, client shapes, engine death, monitor data, GPU sharing, thinking budget, ...) |
| [serve/test_mcp.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_mcp.py) | 447 | MCP client and tool loop |
| [serve/test_parallel.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_parallel.py) | 382 | Batch slots against a scripted engine |
| [serve/test_responses.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_responses.py) | 361 | Responses API |
| [serve/test_security.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_security.py) | 280 | Host and Origin checks |
| [serve/test_monitor.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_monitor.py) | 261 | API request monitor, conversation cache card |
| [serve/test_detok.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_detok.py) | 252 | Incremental detokenizer and tokenizer speed (needs a pack tokenizer) |
| [serve/test_lifecycle.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_lifecycle.py) | 170 | Lazy start, load, unload, close |
| [serve/test_structured.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_structured.py) | 167 | JSON response formats |
| [serve/test_runconfig.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_runconfig.py) | 165 | `GET`/`POST /config` |
| [serve/test_vram.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_vram.py) | 136 | `POST /v1/vram` |
| [serve/test_winjob.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/test_winjob.py) | 67 | Job objects (Windows only) |
| [serve/mcp_fake_server.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/mcp_fake_server.py) | 172 | A fake MCP server (stdio or HTTP) for `test_mcp.py` |
| [tools/strata_mcp.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/strata_mcp.py) | 1738 | Strata's own MCP server (stdio) for AI assistants |
| [tools/test_strata_mcp.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_strata_mcp.py) | 590 | Its tests, with a fake setup and server |
| [docs/MCP_SERVER.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MCP_SERVER.md) | 205 | Upstream's user guide for `tools/strata_mcp.py` |
| [docs/BATCHING.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/BATCHING.md) | 215 | Upstream's guide to batch slots (`"parallel"`), including the engine protocol lines |

Closely related files documented elsewhere: `src/program/generate.cpp` (the engine's `--serve` loop, from line
4607), `tools/strata_tokenizer.py` (331 lines, the pack tokenizer the server imports), `tools/vision/strata_vision.cpp`
(the image encoder), `setup.py` (writes `strata-<model>.json` and the run scripts), `chat.py` (116 lines, a terminal
chat client of this server).

## Key concepts and data flow

```
 HTTP client / web app / chat.py / strata_mcp.py
    |  POST /v1/chat/completions | /v1/messages | /v1/responses   GET /metrics /health /v1/models /props ...
    v
 +----------------------------- serve/server.py (one Python process) -----------------------------+
 | Handler (one thread per connection, HTTP/1.0)                                                   |
 |   parse_request: Host check -> _authorized (API key) -> _foreign_page (Origin) -> route         |
 |   dialect -> frontend.openai_to_messages / anthropic_to_messages / responses.input_messages     |
 |   [structured.prepare_format]  [McpHub tools when "strata_mcp": true]                           |
 | Service.prepare: ChatTemplate.render -> Tokenizer.encode -> image pads -> max_new vs context    |
 | Service.run (holds Service.fifo: one request at a time unless "parallel")                       |
 |   ensure_loaded (restart / reload / before_load / min_free_vram)                                |
 |   StrataEngine.generate(ids, max_new, sampling) -> token ids (None = heartbeat)                 |
 |   Detokenizer.push -> OutputParser.feed -> Events (reasoning | content | tool_start | tool_args | |
 |   tool_call) -> openai_chunks / anthropic_events / responses.Assembler -> JSON or SSE           |
 | Telemetry thread (1 s), idle-unload thread, MCP server threads, client-watch threads            |
 +---------------|------------------------------------------^--------------------------------------+
   stdin:        | GEN/GENI/STOP/VRAM/QUIT, BGEN/BGENI/       | stdout: INFO, READY <ctx> stop, RESUME, PP,
                 | BSTOP/BYIELD (parallel)                    | REUSED, T <id>, DONE ..., ERR, VRAM ..., BADM,
                 v                                            | BT, BDONE, YIELDED      (stderr -> engine log)
 +---------------- strata --serve <config args>  (engine process, src/program/generate.cpp) ------+
 | stdin reader thread (STOP sets a flag; other lines are queued), serve loop, stall watchdog      |
 +-------------------------------------------------------------------------------------------------+
   (optional) strata-vision --mmproj ...: "ENC <image> <out>" -> "OK <n> ..." ; embeddings file -> GENI
```

- **One resident model, one conversation at a time.** `Service.fifo` (`serve/server.py:1660`) serializes requests;
  others wait (`status["queued"]`). Opt-in `"parallel": N` lets up to N decode together in engine batch slots
  ([parallel-requests.md](parallel-requests.md)).
- **Prompts are never truncated.** A prompt plus `max_tokens` over the context is a 400 unless `fit_max_tokens`
  (`Service.prepare`, `serve/server.py:2145-2206`).
- **Crash and restart inside the server.** A dead or silent engine fails the current request; the next request
  starts it again (`Service.ensure_loaded`, `serve/server.py:1773`). The server process itself keeps running.
- **Everything is testable without a GPU.** `--engine mock` (`MockEngine`, `serve/server.py:103`) and
  `ByteTokenizer` (`serve/server.py:1594`) stand in for the engine and tokenizer in almost every test.
- **The model's own formats.** Reasoning is `<think>...</think>`; tool calls are the Qwen XML form
  `<tool_call><function=NAME><parameter=P>VALUE</parameter></function></tool_call>`, parsed by `OutputParser`.

## In UnbiasedStrata

Removed. Fork commit 8894a81 ("Remove installer, Python, non-target platforms, benchmarks and marketing") deleted
`serve/` entirely, `tools/` (so `tools/strata_mcp.py`, `tools/strata_tokenizer.py`, `tools/vision/`), `chat.py`,
and every upstream doc except `DETAILS.md` and the paper (so `docs/MCP_SERVER.md` and `docs/BATCHING.md` are gone).
The fork replaced the server with a C++ front end, `frontend/` (the `unbiased-strata` binary; see
[the fork's front-end docs](../../codebase/frontend/README.md)), which keeps the same process model - the engine
started as a child in `--serve` mode and driven over pipes with the same `GEN` / `T` / `DONE` protocol - but uses
llama.cpp's Jinja templates, GGUF tokenizer and `common_chat_parse` instead of `serve/frontend.py`.

| Upstream feature | Fork front end |
| --- | --- |
| `POST /v1/chat/completions` (stream and not, tools, reasoning) | Kept, reimplemented in C++ |
| `GET /v1/models`, `GET /health` | Kept (different JSON; `/health` carries the engine's `STATS` tiers, a command the fork added to the engine in e41dfc7) |
| `/v1/messages`, `/v1/messages/count_tokens`, `/v1/responses` | Not present |
| `/props`, `/slots`, `/status`, `/v1/status`, `/metrics`, `/mcp`, `/settings`, `/config`, `/api/requests` | Not present |
| Web app (Chat, Monitor, About), `/api-monitor` | Not present |
| API key, Host / Origin checks, CORS | Not present (no authentication options) |
| Images | Present, but in-process (`--mmproj`, llama.cpp `mtmd` on the CPU) instead of the `strata-vision` process |
| `response_format` (JSON), MCP tools, `reasoning_budget_tokens`, repeat stop, `effort_position`, aliases, shared settings | Not present |
| Unload / load / lazy start / idle unload / `min_free_vram_mib` / `before_load` / `/v1/vram` | Not present |
| `"parallel"` batch slots, layer split across GPUs | Refused (`--batch`, `--gpus`, `--layer-split` after `--` exit with code 2) |
| Engine restart after a crash; #481 silence timeout | Replaced: the front end exits (code 1) for a supervisor such as systemd to restart; no silence timeout |
| `tools/strata_mcp.py` (MCP server for assistants) | Not present |

## See also

- [../../upstream/DETAILS.md](../../upstream/DETAILS.md) - upstream's user-facing notes: "Using it", the Responses
  API, MCP tools, "Start the text API without occupying the GPU", "JSON response formats", "API request monitor".
- [../../codebase/frontend/http-api.md](../../codebase/frontend/http-api.md) and
  [../../codebase/frontend/engine-process.md](../../codebase/frontend/engine-process.md) - the fork's replacement.
- [Glossary](../guides/glossary.md).
