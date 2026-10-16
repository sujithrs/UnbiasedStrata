[Upstream docs](../README.md) › [Serve](README.md) › Anthropic and Responses APIs

# Anthropic Messages and OpenAI Responses

## Purpose

The two dialects besides Chat Completions: Anthropic's Messages API (`POST /v1/messages`, `POST
/v1/messages/count_tokens`), which Claude Code uses, and OpenAI's Responses API (`POST /v1/responses`), which Codex
CLI uses. Both run on the same `Service.prepare` / `Service.run` path as `/v1/chat/completions`
([http-api.md](http-api.md)): the same template, thinking levels, thinking budget, conversation cache, API key, Host
and Origin checks.

Files: `serve/frontend.py` (`anthropic_to_messages`, `budget_effort`), `serve/server.py` (`anthropic_events`,
`anthropic_collect`, `_anthropic`, `_count_tokens`, `_responses`, `_responses_prepare`), `serve/responses.py`.

## Anthropic Messages

### Request (`anthropic_to_messages`, `serve/frontend.py:297-353`)

| Field | Use |
| --- | --- |
| `system` | a string or text blocks; becomes the first system message |
| `messages[].content` string | the message as is |
| user content with image blocks and no `tool_result` | text and `image` parts (`source.type` `base64` -> a `data:` URL, or `url` / `path`) |
| `text` blocks | joined into the message text |
| `thinking` blocks | the assistant message's `reasoning_content` |
| `tool_use` blocks | `tool_calls` (`name`, `input`) |
| `tool_result` blocks | each becomes a `tool` message (its text content), placed before the rest of that message |
| `tools` | `{name, description, input_schema}` -> template tools `{name, description, parameters}`; malformed tools are a 400 (#592) |
| `thinking: {"type":"disabled"}` | no thinking |
| `output_config: {"effort": low\|medium\|high}` | the level (high -> the template's xhigh) |
| `thinking: {"type":"enabled","budget_tokens": N}` | chooses a level only: under 2048 low, under 8192 medium, else high (`budget_effort`, `serve/frontend.py:87`) |
| none of these | the template's default (thinking on), unless the config says `"anthropic_thinking": "on_request"` (#278): then no thinking, unless the request sets `reasoning_budget_tokens` |
| `reasoning_budget_tokens` | Strata's hard thinking cap, as on the chat path |
| `max_tokens` | output cap (0 or absent: the rest of the context) |
| `temperature`, `top_p`, `top_k`, ... | sampling (the request dict is the sampling dict) |
| `model` | answered under it if it is an alias, else the model's name |

The shared Chat settings apply first (`Service.with_shared(req, "anthropic")`, `serve/server.py:1925`): a shared
level becomes `thinking: disabled` or `output_config.effort` when the request has neither. `"on_request"` exists
because Claude Code's short helper calls (a session title) allow a few dozen tokens, which the model otherwise spent
thinking and answered with no text (`serve/frontend.py:345-352`).

Claude Code setup (from upstream's DETAILS.md): `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, `ANTHROPIC_MODEL` set to
a Claude model name Claude Code accepts (Strata ignores it), any `ANTHROPIC_AUTH_TOKEN` (or the server's key).

### Response (`anthropic_events`, `serve/server.py:2727-2790`)

Events, as `event: <name>\ndata: <json>\n\n` when streaming:

1. `message_start`: `{id: "msg_<24 hex>", type: "message", role: "assistant", model, content: [], stop_reason: null,
   usage: {input_tokens: <prompt tokens>, output_tokens: 0}}`.
2. For each run of one kind: `content_block_start` (`thinking` with `signature: ""`, `text`, or `tool_use` with
   `id`, `name`, `input: {}`), its `content_block_delta`s (`thinking_delta`, `text_delta`, `input_json_delta` with
   `partial_json` pieces as the call is written), `content_block_stop`. Each tool call is its own block.
3. `message_delta`: `stop_reason` and the final usage: `input_tokens` = prompt minus what the conversation cache
   held, `cache_read_input_tokens` = that cached part, `output_tokens`.
4. `message_stop`.

Heartbeats are written as `: keep-alive` comments (no `ping` events). `stop_reason`: `tool_use` when a call was made,
every streamed call completed and the model stopped at an end token; otherwise `stop` -> `end_turn`, `length` ->
`max_tokens`, `cancel` -> `end_turn`.

Non-streaming (`anthropic_collect`, `serve/server.py:2793-2821`) assembles the same events into one message object;
a `tool_use` block whose JSON does not parse (a call the output ended inside, #211) is dropped.

An error after the stream started is `event: error` with `{"type":"error","error":{"type":"api_error","message"}}`.
Errors before it use the server's common body `{"error":{"type":..., "message":...}}` and status
([http-api.md](http-api.md#errors)), not Anthropic's `{"type":"error",...}` envelope.

### `POST /v1/messages/count_tokens` (`_count_tokens`, `serve/server.py:3538-3543`)

Renders and tokenizes the request exactly as `/v1/messages` would and returns `{"input_tokens": N}`; the model does
not run and is not loaded. Images count as their single placeholder token here (the per-image expansion happens only
in `Service.prepare`).

## OpenAI Responses (`POST /v1/responses`, #451)

`serve/responses.py` (based on PR #630) converts a Responses request into chat messages, tools and template kwargs,
runs `Service.run`, and turns its events into Responses output items and typed SSE events. It is **stateless**:
`store` is always false, and the client sends the whole conversation in `input` every time (Codex does).

### Request

| Field | Handling | Code |
| --- | --- | --- |
| `previous_response_id`, `conversation`, `background`, `prompt` | refused, `unsupported_parameter` | `check_request`, `serve/responses.py:307` |
| `max_output_tokens` | a positive int; the output cap, thinking included | same |
| `instructions` | the system message | `input_messages`, `serve/responses.py:128` |
| `input` as a string | one user message | same |
| `message` items | roles `user`, `assistant`, `system`, `developer` (-> system); parts `input_text`, `output_text`, `text`, `summary_text`, `reasoning_text`, `refusal`, `input_image` / `image_url` (URLs only: "this server keeps no files") | `_content`, `serve/responses.py:57` |
| `reasoning` items | their `reasoning_text` content, else Strata's own `encrypted_content`, becomes the next assistant turn's `reasoning_content`; another server's encrypted content and summaries alone carry nothing | `_reasoning_text`, `serve/responses.py:88` |
| `function_call`, `custom_tool_call` | tool calls of the open assistant turn (`namespace.name` when namespaced); unparseable arguments are kept as `{"arguments": "<text>"}` | `serve/responses.py:177-189` |
| `function_call_output`, `custom_tool_call_output` | `tool` messages, reordered to match the calls' order by `call_id` | `_order_tool_results`, `serve/responses.py:211` |
| `item_reference` | refused (needs stored responses) | |
| leading system/developer messages | merged into one system message, so the prompt start stays stable for the conversation cache; later ones become user messages | `serve/responses.py:201-208` |
| `tools` | `function`; `namespace` (members flattened to `namespace.name`, the namespace description prepended); `custom` (one free-form `input` string, a grammar definition appended to the description); hosted tools (`web_search`, `file_search`, `code_interpreter`, `mcp`, ...) left out with one log line each | `request_tools`, `serve/responses.py:234` |
| `tool_choice: "none"` | hides the tools; anything else lets the model choose | same |
| `reasoning.effort` | `none`/`minimal`, `low`, `medium`, `high`/`xhigh`; else the shared setting; else the model's default | `template_kwargs`, `serve/responses.py:329` |
| `text.format` | `json_object` / `json_schema` -> [structured output](structured-output.md); with tools refused | `text_format`, `serve/responses.py:284` |
| `include: ["reasoning.encrypted_content"]` | reasoning items also carry `encrypted_content` | |
| `temperature`, `top_p`, `reasoning_budget_tokens`, ... | as on the chat path | |

### Output (`Assembler`, `serve/responses.py:340-529`)

- The response object (`resp_<hex>`) echoes the request's settings; `status` goes `in_progress` -> `completed`,
  `incomplete` (`max_output_tokens` ran out, or a call was cut off) or `failed`.
- Output items: `reasoning` (`rs_...`, `content: [{"type":"reasoning_text"}]`, `summary: []` - the model writes no
  summaries), `message` (`msg_...`, one `output_text` part), `function_call` (`fc_...`, `call_id`, `name`,
  `arguments`, `namespace` when namespaced), `custom_tool_call` (`ctc_...`, `input`).
- `encrypted_content` is `"strata.r1:" + base64(JSON {"text": ...})`: opaque but not encrypted (the client already
  holds the text); sent back, it restores the thinking into the prompt so the conversation cache matches.
- `usage`: `input_tokens`, `input_tokens_details.cached_tokens`, `output_tokens`,
  `output_tokens_details.reasoning_tokens` (counted in `Service.run` while the parser is in its reasoning state),
  `total_tokens`.

Streaming events, in order, each `event: <type>\ndata: <json>` with a `sequence_number`: `response.created`,
`response.in_progress`, then per item `response.output_item.added` (and `response.content_part.added` for a
message), its deltas (`response.reasoning_text.delta`, `response.output_text.delta`,
`response.function_call_arguments.delta`, `response.custom_tool_call_input.delta`), its done events
(`response.reasoning_text.done`, `response.output_text.done` + `response.content_part.done`,
`response.function_call_arguments.done`, `response.custom_tool_call_input.done`), `response.output_item.done`;
finally `response.completed` or `response.incomplete`.

Codex's idle timer counts events, not SSE comments, so while a long prompt is read the server sends
`response.in_progress` every 15 s and `: keep-alive` comments in between (`serve/server.py:3466-3479`). With a JSON
`text.format` the answer is held until it validates and then sent as one delta.

### Errors

Responses-format bodies `{"error":{"message","type","param","code"}}` (`error_body`, `serve/responses.py:48`):
`ResponsesError` with its own status (400 by default; context overflow has `code: context_length_exceeded`,
`param: input`), 409 `model_busy`, 503 `server_error` (GPU busy, starting, stuck, died), 502
`structured_output_failed`, 500 for an engine `ERR` in a non-streamed request. After the stream started:
`response.failed` with `response.error = {code, message}`.

### Codex CLI

Upstream's DETAILS.md gives a `~/.codex/config.toml` with `wire_api = "responses"`, `base_url =
"http://127.0.0.1:8080/v1"`, `stream_idle_timeout_ms = 600000` and `show_raw_agent_reasoning = true`. Measured there
with Codex CLI 0.160.0 and Q2_0 on an RTX 5070: the first prompt (instructions and tool descriptions) was 9,443
tokens, read in 10 s; later turns of a tool loop reused about 96% of the prompt from the cache and read the rest in
1-2 s.

## Tests

- `serve/test_responses.py`: parsing (`Parsing`: string input, developer merging, Codex history, encrypted content,
  cut-off arguments, unsupported items, namespaces / custom / hosted tools, effort and text format), over HTTP
  (`OverHttp`: event order, `max_output_tokens` -> incomplete, errors, context overflow, key and foreign pages, JSON
  schema, monitor), and tool round trips.
- `serve/test_server.py`: `MaxTokens.test_anthropic_thinks_only_when_asked`,
  `MaxTokens.test_count_tokens_is_the_prompt_messages_reads`, `ClientShapes` (Anthropic mid-conversation system,
  `?beta=true`), `ThinkingBudget.test_anthropic_stream`, `UsageAndStatus.test_anthropic`.
- Run: `python -m unittest serve.test_responses -v` ([tests.md](tests.md)).

## Gotchas

- `/v1/messages` loads the model (`svc.load()`) before the request is validated, unlike the chat path, so an invalid
  Anthropic request can still trigger a load.
- `budget_tokens` is not a hard limit: use `reasoning_budget_tokens` for that.

## In UnbiasedStrata

Removed (commit 8894a81). The fork's front end has neither `/v1/messages` nor `/v1/responses`; Anthropic and
Responses clients need a translating proxy in front of it.

## See also

- [http-api.md](http-api.md), [chat-template-and-parsing.md](chat-template-and-parsing.md), [structured-output.md](structured-output.md)
- [../../upstream/DETAILS.md](../../upstream/DETAILS.md#the-responses-api-and-codex-cli) - user notes for both APIs.
