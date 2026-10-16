[Upstream docs](../README.md) › [Serve](README.md) › Chat template and output parsing

# Chat template, tokenization and output parsing

## Purpose

The text layer between API clients and the engine, which sees only token ids: how requests become the model's
messages, how the Jinja chat template is rendered and tokenized, and how the generated tokens are turned back into
reasoning, answer text and tool calls without ever showing a client half a tag. `serve/frontend.py` is pure text (no
model, no GPU); `serve/server.py` adds the tokenizer, the incremental detokenizer and image placeholders.

Files: `serve/frontend.py`, `serve/chat_template.jinja`, `serve/chat_golden.json`, and in `serve/server.py`:
`ByteTokenizer`, `Detokenizer`, `Service.render_prompt` / `encode_prompt` / `prepare`, the tokenizer loading in
`main()`. The tokenizer itself is `tools/strata_tokenizer.py` (outside this area).

## Template

`ChatTemplate` (`serve/frontend.py:34-53`) renders the model's template "with the same Jinja settings as
transformers' apply_chat_template": an `ImmutableSandboxedEnvironment` with `trim_blocks`, `lstrip_blocks` and the
`loopcontrols` extension, a `tojson` filter that keeps non-ASCII characters (`ensure_ascii=False`), and
`raise_exception` raising `TemplateRequestError`, which is a `ValueError` so the client gets a 400 with the
template's message (#365). `render(messages, tools, add_generation_prompt=True, **kwargs)`; `source` keeps the text
(served by `/props`).

Which file (`serve/server.py:4013-4015`): `<tokenizer dir>/chat_template.jinja` (the pack exports the model's own
template with its tokenizer), else `serve/chat_template.jinja` (the original model's).

`serve/chat_template.jinja` (183 lines), what it does:

- Leading `system` / `developer` messages are merged into one system turn; a system message later raises "System
  message must be at the beginning." (the server turns late ones into user messages first, below).
- Thinking on (default): a system sentence for the level - `xhigh` (the default; `high` maps to it) or `low`;
  `medium` has no sentence of its own here. Other values raise "Unexpected reasoning effort".
- Tools: a `# Tools` system block listing each tool as JSON inside `<tools>...</tools>`, then the call format
  instructions.
- Assistant turns: `<think>\n{reasoning_content}\n</think>\n\n{content}` (all history turns keep their reasoning
  unless `preserve_thinking` is false), then each call as `<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n
  </parameter>\n</function>\n</tool_call>`; string values raw, others as JSON.
- Tool results: consecutive `tool` messages go into one user turn of `<tool_response>...</tool_response>` blocks.
- Images: `<|vision_start|><|image_pad|><|vision_end|>` per image item (video items likewise with
  `<|video_pad|>`, which the server does not feed).
- Generation prompt: `<|im_start|>assistant\n<think>\n`, or `<think>\n\n</think>\n\n` with `enable_thinking=false`.

`serve/chat_golden.json` (196 lines) holds 10 cases (`single user`, `system + user`, `multi-turn`, `assistant with
thinking`, `no generation prompt`, `thinking disabled`, `unicode content`, `tool call and response`, `empty user`,
`whitespace and newlines`), each with the input `messages`, Strata's `rendered` string and Hugging Face's `hf`
string; they are equal in all 10. The `frontend.py` docstring cites it as the check; no test file in this snapshot
loads it.

## Request normalization (`serve/frontend.py:56-353`)

| Function | Line | What |
| --- | --- | --- |
| `_text_of(content)` | 57 | a string, or the text of `text` / `input_text` parts joined |
| `_parts_of(content)` | 116 | a plain string when there is no image (unchanged behaviour), else the template's list of `{"type":"text"}` and `{"type":"image","source":...}` items |
| `_image_source(part)` | 100 | OpenAI `image_url` (object or string), Responses `input_image`, Anthropic `image` with `base64` (-> `data:` URL) / `url` / `path` |
| `images_of(messages)` | 132 | the image sources in prompt order |
| `_late_system_to_user` | 214 | a system message after the first position becomes a user message in place (#56: Claude Code's hook context, late `developer` messages); merging it into the first would change the prompt start and lose the conversation cache every turn |
| `_object_list` | 222 | `messages` / `tool_calls` as a list of objects; a JSON string is decoded (#460) |
| `_tool_list` | 239 | `tools` must be objects with a name, OpenAI or Anthropic shape, else a 400 naming the field (#592) |
| `openai_to_messages` | 260 | see [http-api.md](http-api.md#request-fields-read) |
| `anthropic_to_messages` | 297 | see [anthropic-and-responses.md](anthropic-and-responses.md) |
| `effort_kwargs`, `budget_effort` | 74, 87 | thinking levels (`EFFORT`, line 70) |

Tool-call arguments given as a JSON string are decoded, because the template requires a mapping
(`serve/frontend.py:277-278`).

### Literal `<think>` tags (#537)

The tokenizer matches `<think>` and `</think>` as special tokens everywhere (GGUF token type 4, as llama.cpp does),
so a user quoting `</think>` used to hand the model a real end-of-reasoning token. `mark_think_literals`
(`serve/frontend.py:173`) swaps every literal tag in message text and tools for private-use characters (`THINK_TAGS`,
U+F0E01 / U+F0E02) before rendering - except a leading `<think>...</think>` block that a client sent inline in an
assistant message, which stays the model's marker. After rendering, `unmark_think_literals` (line 198) puts the tag
text back and returns the spans, which the server encodes as plain text (`tok.encode(prompt, parse_special=True,
plain=spans)`, `Service.encode_prompt`, `serve/server.py:2135-2143`). A prompt without such tags renders exactly as
before.

## Tokenization

`main()` (`serve/server.py:3946-3960`) loads the pack's tokenizer directory (`--tokenizer`, or `"tokenizer"` in the
config; default `pack/full/tokenizer`): `vocab.json`, `merges.txt`, `token_type.json` -> `strata_tokenizer.Tokenizer`
(imported from `tools/`). The engine mode refuses to start without `vocab.json` ("run setup again"). The interface
the server uses: `encode(text, parse_special, plain)`, `decode(ids)`, `token_bytes(id)`.

`ByteTokenizer` (`serve/server.py:1594-1618`) is the tests' stand-in: one id per UTF-8 byte, the specials
`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<|vision_start|>`, `<|image_pad|>`, `<|vision_end|>` as ids 256-261.

End tokens: `Service.stop_ids` are `<|im_end|>` and `<|endoftext|>` (`serve/server.py:1709`); `EOS_IDS = {248044,
248046}` (`serve/server.py:388`) is the same pair for the real vocabulary, used by the batch path.

### Incremental detokenization (`Detokenizer`, `serve/server.py:1622-1647`)

Each token's bytes go through Python's incremental UTF-8 decoder, which emits complete characters and holds a
character split across tokens (invalid bytes become U+FFFD). Constant time per token: the earlier re-decode of all
generated ids "cost 2 ms per token after 8K tokens and 4 ms after 16K" (perf-review F-1, quoted from the
docstring). `pending()` says a character is split, which the thinking budget checks before cutting. A tokenizer
without `token_bytes` (the byte tokenizer) uses the old re-decode.

## Output parsing (`OutputParser`, `serve/frontend.py:464-676`)

`OutputParser(thinking, tools, stream_tools)` is fed text deltas and returns `Event`s (`serve/frontend.py:365`):

| `Event.kind` | Carries |
| --- | --- |
| `reasoning` | text before `</think>` |
| `content` | answer text |
| `tool_start` | a `ToolCall` (`id` = `call_<24 hex>`, `name`) as soon as `<function=NAME>` is complete (with `stream_tools`) |
| `tool_args` | a piece of the call's JSON arguments (with `stream_tools`) |
| `tool_call` | the finished `ToolCall` (`name`, parsed `arguments`, same `id`) |

State machine (`feed`, `serve/frontend.py:600-658`):

- `reasoning` (the start state when thinking is on, since the prompt ends in `<think>\n`): text up to `</think>`;
  a possible partial `</think>` at the end is held back (`_hold`).
- `content`: newlines right after `</think>` or after a call are dropped; text up to `<tool_call>` is content; a
  partial `<tool_call>` and the newlines before it are held, so streamed and whole outputs match.
- `call`: the body up to its end (`call_end`, line 395). `param_end` (line 380) ends a parameter only at a
  `</parameter>` followed by `<parameter=` or `</function>`, so a value may contain the closing tags (a file that
  documents the call format, #210).

`parse_tool_call(body, schema)` (line 431): parameters declared `string` in the tool's schema stay raw text; others
are JSON-decoded when they parse (else kept as text); one leading and one trailing newline are removed from each
value.

With `stream_tools=True` (always, from `Service.run`, `serve/server.py:2251`) `_scan` (line 488) emits the
arguments while they are written: `{`, `"name":`, string values character by character as JSON-escaped text,
non-string values whole once complete, `}`. `_close_scan` closes the JSON of a call whose body ended without a clean
`</function>`. This exists because "for a large file write" a client otherwise saw nothing "for many minutes".

`finish()` (line 660) flushes what is held at the end of generation. A call already announced but cut off
(`max_tokens`) stays unfinished: no `tool_call` event follows it and its JSON is not closed (#211); the adapters then
report `length` and drop it from non-streamed answers. An unterminated call that was never announced is returned as
content.

`Service._note` (`serve/server.py:2208`) maps events to the live status phase (`thinking`, `answering`, `writing a
tool call: NAME`, `tool call complete`) and keeps the last 600 characters as `tail` for `/status`.

## Image placeholders

See [engine-process.md](engine-process.md) ("The image encoder process"): only an
`<|image_pad|>` right after `<|vision_start|>` is an image; it is expanded to one pad per image token.

## Model-specific behaviour

The parser knows the Qwen3.x formats (`<think>`, `<tool_call>` XML). All upstream model variants (Qwen3.8-Flash-Next,
the Coder, Swift 1.5, the Unsloth packs) use them; each pack ships its own `chat_template.jinja` and tokenizer files.

## Tests

`serve/test_server.py`: `LiteralThinkTags` (#537, including "the real tokenizer reads the tag as text" when a pack
is present), `ImageMarkers` (#150), `ToolCallTerminators` (#210), `UnfinishedToolCall` (#211), `ClientShapes`
(#56, #365, #460, #592), `EffortAtTheEnd` (#458). `serve/test_detok.py`: the incremental detokenizer equals the old
re-decode token by token, its cost per token does not grow, and the tokenizer's speed (needs a pack tokenizer;
skipped without one). See [tests.md](tests.md).

## Gotchas

- The fallback template has no sentence for `medium`; the server's `effort_position: end` path supplies one
  (`EFFORT_TEXT["medium"]`, `serve/server.py:70-71`).
- `stop` sequences are not implemented: generation ends at `<|im_end|>` / `<|endoftext|>`, `max_tokens`, the
  thinking budget or the repeat stop.

## In UnbiasedStrata

Removed (commit 8894a81), together with `tools/strata_tokenizer.py`. The fork's front end renders the GGUF's own
template with llama.cpp's `common_chat_templates_apply`, tokenizes with the GGUF vocabulary, and parses output with
llama.cpp's `common_chat_parse` (patched for Qwen3-Coder argument order); see the fork's
[chat-and-parsing page](../../codebase/frontend/chat-and-parsing.md).

## See also

- [http-api.md](http-api.md), [anthropic-and-responses.md](anthropic-and-responses.md), [engine-process.md](engine-process.md)
