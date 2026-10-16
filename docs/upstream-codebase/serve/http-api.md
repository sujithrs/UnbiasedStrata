[Upstream docs](../README.md) › [Serve](README.md) › HTTP API

# HTTP API: routes, chat completions, model discovery and control

## Purpose

Every route `serve/server.py` answers, and the OpenAI Chat Completions path in full: request fields, thinking
levels, the output budget against the context, sampling defaults, streaming, errors. Also model discovery
(`/v1/models`, `/props`, `/slots`, `/health`), and the control routes that load, unload and resize the model. The
Anthropic and Responses dialects are in [anthropic-and-responses.md](anthropic-and-responses.md); the checks every
request passes first (API key, Host, Origin, CORS) in [security.md](security.md).

Files: `serve/server.py` (`make_handler`, `serve/server.py:2825-3582`; `Service`, `serve/server.py:1650-2464`;
`openai_chunks` / `openai_collect`, `serve/server.py:2602-2699`), `serve/frontend.py` for request normalization.

## The HTTP server

- `Server(ThreadingHTTPServer)` (`serve/server.py:3585`): one thread per connection; `allow_reuse_address` is off on
  Windows so a second server on the same port fails loudly; client hang-ups print no stack trace.
- The handler speaks HTTP/1.0 (`protocol_version`, `serve/server.py:2827`): an SSE stream ends by closing the
  connection. Its own access log is silenced (`log_message`).
- `serve(svc, host, port)` (`serve/server.py:3770`) computes the Host names, starts telemetry and runs
  `serve_forever` on a daemon thread. Default port of `server.py` is 8095; the run scripts pass `--port 8080`.
- Before loading the engine `main` binds the port once to fail fast with "port N is already in use"
  (`serve/server.py:3941-3945`).

## Routes

Auth column: "key" = needs the API key when one is configured (`Authorization: Bearer <key>` or `x-api-key`).
Every request without a key configured first passes the Host check ([security.md](security.md)).

| Method, path | Auth | What | Code |
| --- | --- | --- | --- |
| `GET /` | - | the web app (`serve/web/index.html`) | `serve/server.py:3023-3029` |
| `GET /web/<file>` | - | `.css`, `.js`, `.svg` from `serve/web/` (no subfolders), `Cache-Control: no-cache` | `serve/server.py:2978-2995` |
| `GET /fonts/<file>.woff2` | - | the Outfit font, cached a day | `serve/server.py:2963-2977` |
| `GET /api-monitor` | - | the API request monitor page, only with `api_monitor` on (else 404) | `serve/server.py:3023` |
| `GET /health`, `GET /api/health` | - | `{"status":"ok","max_context","model","images","api_key":bool,"loaded","service":"strata"}`, always 200 | `serve/server.py:3030-3033` |
| `GET /status` | key | what the model is doing: `busy`, `queued`, `phase`, `prompt_tokens`, `generated`, `max_tokens`, `tool`, `tail` (the last 600 characters of output), `elapsed_s`, `tokens_per_s`, `tokens_per_s_mean` | `serve/server.py:3034-3047` |
| `GET /v1/models`, `GET /models` | key | the model (and its aliases) | `serve/server.py:3048-3061` |
| `GET /props` (`?model=`) | key | llama.cpp-style properties | `_props`, `serve/server.py:3196-3219` |
| `GET /slots` | key | `[{"id":0,"n_ctx","is_processing"}]`, `[]` when the engine is not running | `serve/server.py:3065-3071` |
| `GET /v1/status` | key | service summary ([monitoring-and-telemetry.md](monitoring-and-telemetry.md)) | `Service.v1_status`, `serve/server.py:2075` |
| `GET /metrics` (`?requests=all`) | key | everything the Monitor tab shows | `Service.metrics`, `serve/server.py:2024` |
| `GET /api/requests` (`?id=`) | key | API monitor records (only with `api_monitor`) | `serve/server.py:3001-3009` |
| `GET /settings`, `POST /settings` | key; POST own page | the Chat settings shared with other apps | `_settings`, `serve/server.py:3298` |
| `GET /config`, `POST /config` | key; POST own page | the editable run-config keys ([config-and-runconfig.md](config-and-runconfig.md)) | `serve/server.py:3260-3296` |
| `GET /mcp` | key | MCP servers, states and tools | `serve/server.py:3018-3022` |
| `POST /v1/chat/completions` | key, Origin | OpenAI Chat Completions | `_openai`, `serve/server.py:3361` |
| `POST /v1/messages` (`?beta=true` accepted) | key, Origin | Anthropic Messages | `_anthropic`, `serve/server.py:3545` |
| `POST /v1/messages/count_tokens` | key, Origin | Anthropic token count | `_count_tokens`, `serve/server.py:3538` |
| `POST /v1/responses` | key, Origin | OpenAI Responses (stateless) | `_responses`, `serve/server.py:3417` |
| `POST /v1/responses/<anything>` | key | 404 "this server keeps no responses (stateless)" | `serve/server.py:3114-3117` |
| `POST /unload`, `POST /load` | key, own page, JSON | unload now (409 while busy) / load now | `serve/server.py:3090-3109` |
| `POST /v1/unload`, `POST /v1/load` | key, own page, JSON | the same for integrations, with model matching and `v1_status` in the reply | `serve/server.py:3118-3141` |
| `POST /v1/vram` | key, own page, JSON | give VRAM back / take it back (#533) | `serve/server.py:3142-3152` |
| `OPTIONS *` | - | CORS preflight (headers only for `cors_origins`) | `do_OPTIONS`, `serve/server.py:2927` |
| anything else | | 404 `{"error":{"message":"not found"}}` | |

Paths are matched without the query string and without a trailing `/` (`serve/server.py:2962`, `3081`; issue #55:
Claude Code posts `/v1/messages?beta=true`).

## `POST /v1/chat/completions`

`_openai` (`serve/server.py:3361-3415`), in order:

1. `svc.with_shared(req, "openai")` fills `max_tokens` and `reasoning_effort` from the shared Chat settings where
   the request has none (`serve/server.py:1925`).
2. `openai_to_messages(req)` (`serve/frontend.py:260`) -> template messages, tools, template kwargs.
3. `prepare_format(response_format, messages)` ([structured-output.md](structured-output.md)); a JSON format
   together with tools or `strata_mcp` is a 400.
4. `svc.load()`: start the engine if it is unloaded or dead (may wait in the FIFO; `GpuBusy` -> 503).
5. `max_new = max_completion_tokens or max_tokens or 0` (0 or -1: "the rest of the context").
6. `"strata_mcp": true` with MCP servers configured: own page only, MCP tools appended (the request's own names
   win) ([tools-and-mcp.md](tools-and-mcp.md)).
7. `svc.reasoning_budget(req)` validates `reasoning_budget_tokens` (400 before anything is sent).
8. `svc.prepare(...)` -> prompt ids, thinking on/off, `max_new` (below).
9. A client-watch thread (`_watch_client`, `serve/server.py:2875`) cancels the request when the socket closes
   (#430 #431), so a client that hangs up during a long prompt read frees the engine.
10. `openai_chunks` (or the MCP loop) -> optional `structured_chunks` -> optional monitor capture -> JSON or SSE.

### Request fields read

| Field | Use |
| --- | --- |
| `messages` | required in practice (the fallback template raises "No messages provided."; a template refusal such as "No user query found in messages." is a 400, #365); also accepted as a JSON string (#460). Roles `system`, `developer` (-> system), `user`, `assistant`, `tool`; a system/developer message after the first becomes a user message in place (#56) |
| content parts | `text`, `input_text`; images `image_url` (`{"url": ...}` or a string), `input_image`, `image` ([chat-template-and-parsing.md](chat-template-and-parsing.md)) |
| `reasoning_content` on assistant messages | rendered back into the prompt |
| `tool_calls` on assistant messages | `arguments` as a JSON string or object |
| `tools` | `{"type":"function","function":{...}}` or bare `{"name":...}`; malformed tools are a 400 (#592) |
| `tool_choice`, `parallel_tool_calls`, `n`, `stop`, `logprobs`, `stream_options`, `user` | not read |
| `model` | not used to choose anything; answered under the model's name, or under the alias it names (#297) |
| `reasoning_effort`, `reasoning: {"effort"}`, `chat_template_kwargs: {"enable_thinking", "reasoning_effort"}` | thinking level (below) |
| `max_completion_tokens`, `max_tokens` | output cap; the first truthy one wins |
| `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `penalty_last_n` | sampling ([engine-process.md](engine-process.md#server-to-engine)) |
| `experimental_speed_projection` | per-request switch of the engine's control vector (`cvec=0/1`) |
| `strata_tune` | `{"pcie_frac", "spec_min_p"}` for this request (setup's calibration) |
| `reasoning_budget_tokens` | hard thinking cap (below) |
| `response_format` | JSON formats |
| `strata_mcp` | `true`: run MCP tools on the server (the web app sends it) |
| `stream` | SSE when true |

The request dict itself is the sampling dict passed to `Service.run`, so any of these keys anywhere at the top level
count.

### Thinking levels

`effort_kwargs` (`serve/frontend.py:74`) maps client spellings onto the template's levels (`EFFORT`,
`serve/frontend.py:70`):

| Client value | Template kwargs |
| --- | --- |
| `none`, `off`, `minimal`, `disabled`, `false`, `False` | `enable_thinking=False` |
| `low` | `reasoning_effort="low"` |
| `medium` | `reasoning_effort="medium"` |
| `high`, `xhigh`, `max`, `maximum` | `reasoning_effort="xhigh"` (the template's default) |
| anything else | 400 "unknown reasoning effort" |

`chat_template_kwargs.enable_thinking: false` overrides everything. Without a level the template's default applies
(the model thinks at its highest level).

`"effort_position": "end"` in the config (#458, opt-in, engine 0.1.39+) renders a non-default level's request as a
default one and puts the level's sentence (`EFFORT_TEXT`, `serve/server.py:68-71`) in a short system turn right
before the answer (`Service.render_prompt`, `serve/server.py:2117-2133`); the engine, started with
`--tail-role-token <id of "system">`, checkpoints in front of that turn, so changing the level keeps the cached
conversation.

### Output budget and the context (`Service.prepare`, `serve/server.py:2145-2206`)

`room = context - 8 - prompt tokens` (`CTX_SLACK = 8`, the engine's own margin):

| Request | Result |
| --- | --- |
| no `max_tokens` (or 0 / negative) | `max_new = room`; 400 "leaves no room to answer" when `room < 1` |
| `max_tokens <= room` | as asked |
| `max_tokens > room` | 400 "prompt (N tokens) + max tokens (M) exceeds the context (C); requests are never truncated ...", naming the most that fits and `fit_max_tokens` |
| `max_tokens > room` with `fit_max_tokens` (`--fit-max-tokens` or config, #545) | `max_new = max(1, room)` |

While the engine is (re)starting and reports no context, `prepare` raises `EngineStarting` (503, #344).

### Thinking budget (`reasoning_budget_tokens`, #123)

`Service.reasoning_budget` (`serve/server.py:1739`): the request's value, else the config's; 0 or less means none;
a non-integer is a 400. In `Service.run` (`serve/server.py:2326-2372`), once the thinking reaches the budget at a
clean point (no tag held back, no split UTF-8 character), the server stops that pass and starts a new engine request
whose prompt is the previous prompt plus the generated tokens plus `REASONING_WRAP_UP` (`"\n\nI have thought about
this long enough; time to give my answer.\n</think>\n\n"`, `serve/server.py:78`), so the engine continues from its
cached prefix and the model answers. The wrap-up is streamed as reasoning and counts as output. If no room is left
it ends as `length`.

### Repeat stop (#606)

A reply that repeats the same token `repeat_stop_tokens` times in a row (default 256, `REPEAT_STOP_TOKENS`,
`serve/server.py:76`; config `"repeat_stop_tokens"`, 0 = off) is ended with `finish_reason` `length` and a line in
the server window (`serve/server.py:2316-2320`, `2375-2379`).

### Sampling defaults

`Service.run` merges, lowest to highest: the config's `"sampling"` block (`sampling_defaults_from_config`,
`serve/server.py:3840`), the shared Chat settings (`POST /settings`), then the request's non-null fields
(`serve/server.py:2247-2250`). Without a `"sampling"` block a request with no sampling keys decodes greedy (the
engine's default). An explicit `temperature: 0` stays greedy.

### Non-streaming response (`openai_collect`, `serve/server.py:2666-2699`)

```
{"id": "chatcmpl-<24 hex>", "object": "chat.completion", "created": <s>, "model": <name or alias>,
 "choices": [{"index": 0, "message": {"role": "assistant", "content": <text or null>,
              "reasoning_content": <if any>, "tool_calls": [...], "strata_mcp": [<MCP events, if any>]},
              "finish_reason": "stop" | "length" | "tool_calls"}],
 "usage": {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details": {"cached_tokens"}},
 "timings": {"cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second", "predicted_n",
             "predicted_ms", "predicted_per_token_ms", "predicted_per_second", "draft_n", "draft_n_accepted"}}
```

`timings` uses llama.cpp's names, from the engine's own clock (`request_timings`, `serve/server.py:2478`);
`prompt_n` excludes the tokens the conversation cache held (`cache_n`); `draft_n` / `draft_n_accepted` appear when
the engine reported MTP drafts (PR #83). Tool calls whose arguments do not parse as JSON are dropped unless the
finish is `tool_calls` (a call the output ended inside, #211).

### Streaming response (`openai_chunks`, `serve/server.py:2602-2655`)

Headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no` (nginx-style proxies
pass each event at once), CORS when configured. Events are `data: <json>\n\n`:

1. `chat.completion.chunk` with `delta: {"role":"assistant","content":""}`.
2. `delta.reasoning_content` and `delta.content` pieces as they are parsed.
3. Tool calls stream: `tool_start` sends `{"index","id","type":"function","function":{"name","arguments":""}}`,
   then each `tool_args` piece `{"index","function":{"arguments":<piece>}}` (string parameters character by
   character). A call that was not announced while written is sent whole at its end.
4. MCP activity (web app only) as chunks with an empty delta and a `strata_mcp` field.
5. The final chunk: empty delta, `finish_reason`, `usage`, `timings`.
6. `data: [DONE]`.

Heartbeats (`None` from the engine every 10 s of silence and once per prompt chunk) become the SSE comment
`: keep-alive\n\n`, which keeps clients' idle timers calm during a long prompt read.

`finish_reason`: `tool_calls` when calls were made, every streamed call was completed and the engine stopped at an
end token; `cancel` is reported as `stop`; otherwise the engine's `stop` or `length` (`serve/server.py:2644-2646`).

A client that disconnects mid-stream (write fails) sets `cancel` and closes the chunk generator, which `STOP`s the
engine and drains to its `DONE`.

### Errors

| Status | Body `error.type` | When |
| --- | --- | --- |
| 400 | `invalid_request_error` | `ValueError` before or during a non-streamed answer: bad JSON, not an object, template refusals (#365), bad tools, unknown effort, bad budget, context overflow, images without vision, image errors, and an engine `ERR` in a non-streamed request |
| 401 | `authentication_error` | missing or wrong key |
| 403 | `forbidden` | Host or Origin refused ([security.md](security.md)) |
| 409 | `model_busy` | `/v1/load`, `/v1/unload` while a request runs or waits; `/v1/vram` after 300 s |
| 415 | - | a JSON content type is required (own-page routes, browser pages without a key) |
| 502 | `structured_output_failed` | the answer failed its JSON format |
| 503 | `server_error` | `GpuBusy` (min free VRAM), `EngineStarting`, `EngineDied` ("...; the next request restarts it"), `EngineStuck` |
| 200 + `data: {"error":{...}}` + `[DONE]` | `server_error` / `structured_output_failed` | an error after the stream started |

## Model discovery

- `GET /v1/models` (`serve/server.py:3048-3061`): `{"object":"list","data":[{"id", "object":"model",
  "status":{"value":"loaded"|"unloaded"}, "meta":{"n_ctx"}, "architecture":{"input_modalities":["text"(,"image")],
  "output_modalities":["text"]}, "aliases":[...]}, <one entry per alias with "alias_of">]}`. An unloaded model
  (unloaded on purpose, lazy, or idle-unload configured) is still listed with status `unloaded`, like llama-server's
  router; an engine that died is not listed (empty `data`).
- `GET /props` (`serve/server.py:3196-3219`): `default_generation_settings: {n_ctx, params}` (configured sampling
  defaults and shared settings, renamed to llama.cpp's `repeat_penalty` / `repeat_last_n`; `n_predict` is the
  shared `max_tokens` or -1), `total_slots: 1`, `model_alias`, `chat_template` (the template's source),
  `modalities.vision`, `models_autoload`, `is_sleeping` (unloaded), `model_path`, `build_info: "Strata <version>"`.
  An unknown `?model=` is 404; a dead (not unloaded) engine 503. Discovery never loads or restarts the model.
- Model aliases (#297): `"aliases"` in the config (a list or a comma-separated string, `Service.set_aliases`,
  `serve/server.py:1715`); a request naming one is answered under that name; any other name is still served.

## Load, unload, lazy start, VRAM

| Route / option | Behaviour |
| --- | --- |
| `POST /unload` | `Service.unload()`: 200 `{"status":"unloaded"\|"not loaded"\|"unsupported"}`, 409 `{"status":"busy"}`, 503 on `EngineStuck`. Ends the engine and the vision encoder; the next request loads again |
| `POST /load` | `Service.load()`: 200 `{"status":"loaded"}`, 503 on `GpuBusy` |
| `POST /v1/unload`, `/v1/load` | body `{}` or `{"model": "<configured model>"}` (another name: 404); 409 `model_busy` while busy or queued; reply `{"status", ...v1_status}` |
| `--lazy` / `"lazy_load": true` | start the HTTP API without the engine; the first generation request loads it through the same path (with `before_load` and `min_free_vram_mib`). Text only: refused with `"vision"` |
| `--idle-unload S` / `"idle_unload_s"` | a thread unloads after S seconds without requests (checked every `min(30, S/4)` s, at least 1 s; `Service.start_idle_unload`, `serve/server.py:1898`) |
| `--min-free-vram-mib M` / `"min_free_vram_mib"` | load only when M MiB are free on the (first) GPU, waiting up to 15 s; else 503 "the GPU is in use by another program". Unreadable VRAM never refuses |
| `--before-load CMD` / `"before_load"` (string or list) | run before every reload (120 s timeout; a string goes through the shell) |
| `POST /v1/vram {"reserve_mib": N or null}` | `Service.vram` (`serve/server.py:1815`): waits up to 300 s for the FIFO (and the batch control lines), sends `VRAM N`, returns the engine's figures; on an unloaded engine the reserve is kept and applied at the next load; an engine without `--vram-elastic` gives 400 with its reason |

`/unload` and `/load` read the (unused) body first (`_control_body`, `serve/server.py:3221`, at most 64 KiB, else
413), because on Windows replying before reading it sends a TCP reset.

## Model- and platform-specific behaviour

- The HTTP layer knows the model only through its template and token strings (`<|im_end|>`, `<|endoftext|>`,
  `<|image_pad|>`, `<|vision_start|>`, `<think>`); the same server serves Qwen3.8-Flash-Next, the Coder, Swift 1.5
  and the Unsloth packs with each pack's own `chat_template.jinja`. `"model_name"` in the config is the reported id
  (default `qwen3.8-flash-next`).
- `min_free_vram_mib` reads NVML on NVIDIA and amdgpu sysfs with `"backend": "hip"` (#301); with several GPUs it checks
  the first.

## Tests

`serve/test_server.py` covers most of this: `MaxTokens`, `FitMaxTokens`, `ClientShapes`, `ToolCallTerminators`,
`UnfinishedToolCall`, `SamplingKeys`, `UsageAndStatus`, `TimingsDrafts`, `WebApp` (discovery, props, slots),
`ModelAliases`, `ThinkingBudget`, `RepeatStop`, `EffortAtTheEnd`, `SharedSettings`, `SharingTheGpu`, `ClientHangUp`;
`serve/test_lifecycle.py` (load/unload/lazy) and `serve/test_vram.py`. See [tests.md](tests.md).

## Gotchas

- An engine `ERR` in a non-streamed request becomes a 400 `invalid_request_error`, because both are `ValueError`
  (`serve/server.py:3165-3169`); streamed, it is a `server_error` event.
- `/health` says `"status":"ok"` even while the engine is down; read `"loaded"`.
- `/v1/load` accepts only the configured model name, not an alias.
- `tool_choice` is not read on the chat path: tools cannot be forced or hidden there (the Responses path honours
  `"none"`).

## In UnbiasedStrata

Removed (commit 8894a81). The fork's front end serves only `GET /health`, `GET /v1/models` and
`POST /v1/chat/completions`; it clamps `max_tokens` to the room left instead of refusing, ignores `response_format`,
`stop` and `tool_choice`, and has no authentication. See [the fork's HTTP API](../../codebase/frontend/http-api.md).

## See also

- [anthropic-and-responses.md](anthropic-and-responses.md), [chat-template-and-parsing.md](chat-template-and-parsing.md), [security.md](security.md), [config-and-runconfig.md](config-and-runconfig.md)
- [../../upstream/DETAILS.md](../../upstream/DETAILS.md#using-it) - "Using it" and "Sharing the GPU with other programs".
