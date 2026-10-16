[Upstream docs](../README.md) › [Serve](README.md) › Tools from MCP servers

# Tools from MCP servers (`serve/mcp.py`) and the server-side tool loop

## Purpose

The web app's chat can give the model tools from [MCP](https://modelcontextprotocol.io) (Model Context Protocol)
servers - reading files, fetching pages, searching - as LM Studio and Claude Desktop do. Strata is the MCP *client*
here: `serve/mcp.py` starts or connects to the configured servers, lists their tools, offers them to the model, and
runs the calls the model writes; `run_with_mcp` in `serve/server.py` feeds each result back and lets the model
continue. Only requests that opt in with `"strata_mcp": true` (the web app does) see these tools; every other API
client sees the API unchanged and keeps its own tools.

This is the opposite direction from [mcp-server.md](mcp-server.md) (`tools/strata_mcp.py`, an MCP *server* that lets
an assistant manage Strata).

Files: `serve/mcp.py` (676 lines), `serve/server.py` (`run_with_mcp`, `serve/server.py:2514-2598`; the MCP part of
`_openai`, `serve/server.py:3369-3384`; `GET /mcp`; hub start in `main`, `serve/server.py:4086-4092`),
`serve/mcp_fake_server.py` (tests).

## Configuration

In `strata-<model>.json` (Claude Desktop's block can be pasted as is under `"mcpServers"`), or in a separate file
passed with `--mcp-config` (`{"mcpServers": {...}}`; its entries win on a name clash):

```json
"mcp_servers": {
  "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\me\\notes"]},
  "search": {"url": "http://127.0.0.1:3000/mcp", "headers": {"Authorization": "Bearer ..."}}
},
"mcp": {"timeout_s": 60, "max_result_chars": 20000, "max_rounds": 8}
```

| Entry key | Meaning |
| --- | --- |
| `command`, `args`, `env`, `cwd` | a program Strata starts (stdio transport); `npx`, `uvx` are found on `PATH` (on Windows `.cmd` files only with `shutil.which`) |
| `url`, `headers` | Streamable HTTP transport; `"type": "sse"` (the old SSE-only transport) is refused at start |
| `disabled: true` | entry left out |

`"mcp"` block (`settings_from`, `serve/mcp.py:630`; `DEFAULTS`, `serve/mcp.py:37`): `timeout_s` (60, per call),
`max_result_chars` (20000), `max_rounds` (8, calls in a row per answer), `start_timeout_s` (120, `initialize` and
`tools/list`). A malformed entry stops the server's start with a `[strata] ...` message (`_check_server`,
`serve/mcp.py:601`). Restart Strata after a change. `python -m serve.mcp config.json` lists what a config offers.

## Main types

| Name | Line | What |
| --- | --- | --- |
| `StdioTransport` | `serve/mcp.py:79` | newline-delimited JSON-RPC 2.0 on the child's stdin/stdout; stderr kept (last 40 lines) for error messages; requests from the server are answered (`ping`) or refused (-32601); notifications ignored; on POSIX a new session so `close()` can end the whole process group; on Windows `CREATE_NO_WINDOW` and `taskkill /T /F`; `contain()` ties it to the server's lifetime |
| `HttpTransport` | `serve/mcp.py:251` | every message is a POST; the answer is JSON or an event stream; `Mcp-Session-Id` from `initialize` rides on later requests, `MCP-Protocol-Version` once negotiated; a 404 with a session marks it broken (a restart starts a new one); `close()` sends `DELETE` |
| `_wait` | `serve/mcp.py:61` | waits for a response, checking the chat's stop (`McpCancelled`) and the deadline (`McpTimeout`) 20 times a second; a timed-out or cancelled request sends `notifications/cancelled` |
| `McpServer` | `serve/mcp.py:380` | one configured server: `initialize` (protocol `2025-06-18`, client info `strata`), `notifications/initialized`, `tools/list` with pagination (up to 100 pages; -32601 = a server without tools). States `idle` -> `starting` -> `ready` / `failed`; `stopped` when it ends. A stopped server is started again at its next call; a failed one at most every 10 s |
| `result_text` | `serve/mcp.py:465` | MCP content blocks -> the text the model reads: text as is, resources' text or a placeholder, images and audio named with their size ("not shown to the model"), else JSON; `structuredContent` when there are no blocks |
| `McpHub` | `serve/mcp.py:494` | all servers; `start()` starts each on its own thread (an `npx` download must not delay the chat); `routes()` names tools `<server>__<tool>` (cleaned to `[A-Za-z0-9_-]`, at most 64 characters, de-duplicated); `template_tools(exclude)`; `call()` -> `{ok, text, chars, truncated, ms, server, tool}`; `status()` for `GET /mcp` |
| `hub_from_config` | `serve/mcp.py:648` | `mcp_servers` + `mcpServers` + the `--mcp-config` file -> `McpHub` or `None` |

`McpHub.call` (`serve/mcp.py:551-577`) never raises for a tool problem: the tool's own error (`isError`), a crash, a
timeout or an unknown name become a result starting with `error:`, so the model can react. Only `McpCancelled`
propagates. Results longer than `max_result_chars` are cut with the note "[... truncated: the tool returned N
characters; only the first M are shown]".

## The tool loop (`run_with_mcp`, `serve/server.py:2514-2598`)

```
_openai: "strata_mcp": true and a hub -> own page only -> hub.wait(10) for servers still starting
         -> tools = request tools + hub.template_tools(exclude=request tool names)
run_with_mcp loop:
   Service.run(ids, ...)            events stream to the client as usual, except:
     MCP tool_start  -> ("mcp", {"event":"start", id, name})      (its argument pieces are not streamed)
     MCP tool_call   -> collected
     request's own tool_call -> counted (the client must answer it)
   after the pass: run the collected calls only if no own call, finish == "stop", not cancelled,
                   and rounds < max_rounds (else ("mcp", {"event":"limit"}))
     each call: ("mcp", {"event":"call", id, name, server, tool, arguments, round})
                hub.call on a worker thread; the generator yields heartbeats every second meanwhile
                ("mcp", {"event":"result", id, ok, text, chars, truncated, ms})
   append the assistant turn (content, reasoning_content, tool_calls) and one "tool" message per result,
   Service.prepare again (same max_tokens rule), next pass
   announced but not run calls -> ("mcp", {"event":"result", ..., "skipped": true, "text": "not run"})
final ("done", {... completion_tokens summed over all rounds, prompt_tokens of the last round})
```

`openai_chunks` turns each `("mcp", x)` into a chunk with an empty delta and a `strata_mcp` field; a non-streamed
answer collects them into `message.strata_mcp`. The web app shows each call as a collapsible block (tool, arguments,
result) and sends the calls and results back in later turns as the model read them (`assistantMessages` in
`serve/web/app.js`).

Cancellation: the client's Stop closes the connection; the heartbeats during a slow tool are what detect it
(`GeneratorExit` sets `cancel`, which `_wait` turns into `McpCancelled` and a `notifications/cancelled` to the
server).

## Safety

From upstream's DETAILS.md and the code:

- MCP tools run on the PC with the user's rights, and the model decides when to call them, also because of what it
  reads (a page or a file can contain instructions).
- A request with `"strata_mcp": true` must pass `_own_page` (JSON, no foreign Origin), whatever the API key
  ([security.md](security.md)). The request's own tools win a name clash (#592: "a second line of defence").
- Without servers configured, `"strata_mcp": true` is a plain chat (`NoServers` test).

## Tests

`serve/test_mcp.py` against `serve/mcp_fake_server.py` (tools `echo`, `fail`, `sleep`, `big`, `die`, `add`;
`tools/list` pages two at a time; `--http PORT` for Streamable HTTP; `--crash-at-start`; `FAKE_MCP_LOG` records
calls and cancellations): `Client` (initialize and every page, namespaced tools, a failing server left out, calls,
errors as text, timeout, truncation, crash mid-call then restart, cancel), `HttpClient`, `Config` (both spellings and
the file; bad entries stop the start), `ToolLoop` (call then answer, a call the output ended inside is not run,
non-stream, plain requests get no MCP tools, own tools go to the client, tool errors, truncation, `max_rounds`, Stop
during a tool, own page only, `GET /mcp`), `NoServers`. Run: `python -m unittest serve.test_mcp -v`.

## Gotchas

- Only the chat path (`/v1/chat/completions`) runs MCP tools; `/v1/messages` and `/v1/responses` do not.
- MCP and a JSON `response_format` cannot be combined (400).
- When the model writes both an MCP call and one of the request's own calls in one answer, the MCP calls are not
  run (their results could not reach the model before the client's).

## In UnbiasedStrata

Removed (commit 8894a81). The fork's front end passes the request's `tools` to the model and returns its tool calls
to the client; it runs no tools itself.

## See also

- [web-app.md](web-app.md), [http-api.md](http-api.md), [mcp-server.md](mcp-server.md)
- [../../upstream/DETAILS.md](../../upstream/DETAILS.md#tools-from-mcp-servers) - upstream's user guide.
