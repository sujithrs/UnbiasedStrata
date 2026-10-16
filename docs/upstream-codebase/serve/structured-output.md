[Upstream docs](../README.md) › [Serve](README.md) › Structured output

# JSON response formats (`serve/structured.py`)

## Purpose

`response_format` on `/v1/chat/completions` (and `text.format` on `/v1/responses`) asks for one JSON object,
optionally matching a JSON Schema. The engine has no grammar-constrained decoder, so the server does **schema
prompting followed by server validation**: it adds an instruction with the schema to the system message, generates
once, and checks the answer before any of it reaches the client. A failure is an error, never a silently retried or
unchecked answer.

Files: `serve/structured.py` (147 lines); its callers in `serve/server.py` (`_openai`, `structured_chunks`,
`_responses_prepare`, the `Assembler`'s JSON mode in `serve/responses.py`).

## Main functions

| Name | Line | What |
| --- | --- | --- |
| `StructuredOutputError` | `serve/structured.py:14` | a generation that failed its format: HTTP 502 `structured_output_failed` (or an SSE error event) |
| `jsonschema_modules()` | 35 | imports `jsonschema` and `referencing` on first use; when missing, prints once that answers are only checked to be one JSON object (`python -m pip install "jsonschema>=4.23,<5"`). Setup does not install it |
| `prepare_format(response_format, messages)` | 54 | validates the format and returns `(messages with the directive, validator)`, or `(messages, None)` for none / `text` |
| `validated_json(text, validator, finish)` | 121 | checks one answer and returns it in canonical compact JSON |

### `prepare_format`

| `type` | Validator | Rules |
| --- | --- | --- |
| `text` or absent | none | nothing changes |
| `json_object` | `_ObjectOnly` (one JSON object) | schema `{"type":"object"}` |
| `json_schema` | `jsonschema`'s validator class for the schema (`validator_for`, `check_schema`), with a `Registry` whose retrieve refuses every remote URI; `_ObjectOnly` without the package | needs `json_schema.name` (non-empty string) and `json_schema.schema` (object with `"type": "object"` at its root); `strict` must be boolean if present; `$ref` / `$dynamicRef` must be local (`#...`); an invalid schema is a 400 |
| anything else | | 400 "response_format.type must be text, json_object or json_schema" |

The directive (`serve/structured.py:104-108`): "OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching
the JSON Schema below. No Markdown, headings, code fences, commentary, or text outside the JSON. ..." followed by the
schema as compact JSON. It is appended to the first system message (as a text part when that message's content is a
list), or inserted as a new system message.

### `validated_json`

1. `finish` must be `stop`; anything else (e.g. `length`) is "structured output was incomplete
   (finish_reason=...); increase the output budget".
2. `json.loads` with duplicate keys refused and `NaN` / `Infinity` refused; must re-serialize with
   `allow_nan=False`.
3. The first validation error is reported with its JSON path ("model output failed the JSON Schema at /answer:
   ...").
4. Returns the canonical text (`separators=(",", ":")`, non-ASCII kept), which is what the client receives.

## Flow

```
_openai: prepare_format -> (refuse with tools or strata_mcp: 400) -> svc.load -> prepare -> openai_chunks
         -> structured_chunks(chunks, validator)
              buffers every chunk; yields None (an SSE keep-alive) at least once a second
              at the end: openai_collect(buffer) -> validated_json(content, validator, finish_reason)
              -> first chunk (role), one chunk with the whole validated content (+ reasoning_content), last chunk
```

`structured_chunks` (`serve/server.py:2702-2723`): a streamed structured answer is held until validated, so a client
never sees unvalidated content; the reasoning is sent together with the content at the end. Non-streamed answers go
through the same buffering via `openai_collect`. On `/v1/responses`, the `Assembler` with `json_mode` holds the
`content` text and sends it as one `response.output_text.delta` after `validate` (`serve/responses.py:441-444`,
`500-507`).

The format check runs before `svc.load()`, so a rejected schema never loads the engine.

`GET /v1/status` advertises: `structured_output: {"formats": ["json_object", "json_schema"], "method":
"prompt_and_validate", "constrained_decoding": false, "stream_buffered": true}` (`serve/server.py:2093-2094`).

## Errors

| Case | Result |
| --- | --- |
| bad `response_format` / schema / remote `$ref` | 400 `invalid_request_error` (Responses: `param: text.format`) |
| JSON format with `tools` or `strata_mcp` | 400 "structured response_format with tools/MCP is not supported" |
| malformed JSON, duplicate keys, non-finite numbers, schema violation, incomplete generation | 502 `{"error":{"type":"structured_output_failed","code":"structured_output_failed","message"}}`; streamed: `data: {"error":...}` then `[DONE]`, with no content deltas before it |

One generation per request; no hidden retry.

## Tests

`serve/test_structured.py` (`Structured`): a strict "Story" schema (H3) streamed with reasoning, invalid generations
never stream content, `json_object` strictness and truncation, a rejected schema does not load the engine or retry,
invalid schema 400 and local refs, malformed formats and formats with tools rejected before loading, an object still
required without `jsonschema`, and that the server does not import `jsonschema` at start. `serve/test_responses.py`
`test_json_schema_text_format` covers the Responses path. Run: `python -m unittest serve.test_structured -v`.

## In UnbiasedStrata

Removed (commit 8894a81). The fork's front end does not read `response_format`.

## See also

- [http-api.md](http-api.md), [anthropic-and-responses.md](anthropic-and-responses.md)
- [../../upstream/DETAILS.md](../../upstream/DETAILS.md#json-response-formats) - upstream's user notes.
