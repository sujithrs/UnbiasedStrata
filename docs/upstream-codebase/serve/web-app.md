[Upstream docs](../README.md) › [Serve](README.md) › Web app

# The web app (`serve/web/`) and the API request monitor page

## Purpose

The browser app at `http://127.0.0.1:8080/` that `START-HERE.bat` / the run scripts open: **Chat** (streaming
answers with folded thinking, Markdown and code, attachments, MCP tool blocks), **Monitor** (what the model is doing,
hardware, context, recent requests, the conversation cache, MCP servers) and **About** (model and engine facts, the
PC, connection addresses, the API key, the Model settings editor). Plus a separate page, `/api-monitor`, for
inspecting API traffic. Everything is plain HTML, CSS and JavaScript served by `serve/server.py` from the same
origin: no build step, no CDN, no framework.

Files: `serve/web/index.html` (243 lines), `app.js` (1073), `app.css` (184), `components.css` (128), `tokens.css`
(87), `sprite.svg` (33), `monitor.html` (37), `monitor.js` (111), `fonts/` (Outfit variable font, SIL OFL 1.1).

## Serving

`do_GET` (`serve/server.py:2961-3029`): `/` -> `index.html`; `/web/<name>` -> a `.css`, `.js` or `.svg` file from
`serve/web/` (no subfolders, `Cache-Control: no-cache`); `/fonts/<name>.woff2` (cached a day); `/api-monitor` ->
`monitor.html` when the monitor is on. All app requests use relative URLs (`fetch("metrics")`, `fetch("v1/chat/...")`),
so the app works behind a reverse proxy or tunnel too.

## Structure of `index.html`

| Part | Elements |
| --- | --- |
| Header | brand, tabs Chat / Monitor / About (`role="tab"`, also selectable by `#chat` / `#monitor` / `#about`), the status pill (Idle, "Reading prompt · N%", "Generating · N tok/s", "N queued", "API key needed", "Server not reachable"), light/dark button |
| Chat view | message list (empty state "Ask anything"), composer: attachments row, textarea, buttons attach file, new chat, save chat as Markdown, Sampling, Stop, Send (Enter sends, Shift+Enter is a new line) |
| Monitor view | Model state card (badges Idle / Reading / Generating / Queued / Error, label, progress bar); eight metric cards with 60 s sparklines (Speed with Decode and Prefill, GPU load, VRAM, GPU temp, Power, PCIe, CPU, Disk read); Context fill gauge with Experts in VRAM, System RAM and GPU temperature bars; Recent requests table (time, status, prompt, reused, output, tok/s, expert cache hit rate with the PCIe share, duration; "Show all"); totals since start; Conversation cache card; MCP servers card |
| About view | Model and engine (model, engine version, context, KV cache type and streaming, experts in VRAM, speculation, images, the experimental speed projection), This PC (GPU, CPU, RAM), Connect your tools (OpenAI base URL `<origin>/v1`, Anthropic base URL `<origin>`, model name, copy buttons), Settings (API key, dark theme), Model settings (the run config editor) |
| Sampling drawer | Thinking (Off / Low / Medium / High), Temperature (0-1.5), Top-p, Top-k (1-64), Max tokens, Seed, Show thinking, Use tools from MCP servers (when servers exist), Experimental speed projection (when the engine has a control vector), Use for other apps too; Reset / Apply |

## Logic (`app.js`)

| Area | Lines | What |
| --- | --- | --- |
| storage | `serve/web/app.js:15-18` | `localStorage` keys `strata.theme`, `strata.apikey`, `strata.sampling`, `strata.chat` (chats stay in the browser); private mode falls back to memory |
| `headers()` | 89 | `Authorization: Bearer <key>` when a key is saved; JSON content type for POSTs |
| `loadHealth` | 100 | `GET health`: model name, whether images are on (attach button text) |
| `poll` / `render` | 155-198 | `GET metrics` (or `metrics?requests=all`) every second; a 401 shows "API key needed"; three failures "Server not reachable"; reloads `/mcp` every 10 s on the Monitor tab |
| `renderMonitor` | 240-354 | the cards from `/metrics`: decode rate live or of the last request; prefill rate live (`prefill_tok_s_mean`) or computed from the last request's read tokens; with several GPUs each card's own values as a subline (#112) |
| `renderConvCache` | 205-229 | the Conversation cache card (#596): parked conversations against slots and RAM budget when the engine's opt-in cache is on, how many requests reused part of their prompt, the last switch; otherwise a note on `--conversation-cache-mib` |
| `renderAbout` | 369-400 | the About facts from `engine` (the `INFO` fields) and the hardware |
| `loadMcp` / `renderMcp` | 403-435 | `GET mcp`: server badges (Connected, Starting, Failed, Stopped, Waiting), tools as chips; retried every 3 s while a server is starting |
| `loadConfig` / save | 438-489 | `GET`/`POST config`: a form built from the server's key list (selects for bool / enum, inputs otherwise); only changed keys are sent; the reply names the `.bak` |
| Markdown | 491-555 | escape first, then headings, rules, quotes, tables, lists, paragraphs; fenced code blocks with a copy button (an unterminated fence while streaming renders as code) |
| messages | 557-720 | the thinking block (`details.think`, "Thinking…", "Thought for N s"), tool blocks (`toolHtml`: Writing / Running / Done / Error / Skipped, arguments and result), auto-scroll |
| `send` | 761-863 | `POST v1/chat/completions` with `stream: true`, `reasoning_effort`, sampling (temperature 0 sends only `temperature: 0`), `seed`, `max_tokens`, `experimental_speed_projection` when the engine has a vector, `strata_mcp: true` when MCP tools exist and the switch is on; reads the SSE stream (skips `: keep-alive`), appends `reasoning_content` / `content`, handles `strata_mcp` events (`onTool`), an `error` event throws; Stop aborts the fetch, which the server sees as a disconnect |
| `apiMessages` / `assistantMessages` | 721-753 | the history sent each turn: user text with attached files inlined and images as `image_url` parts; assistant answers as content only (their thinking is not sent back), with MCP rounds rebuilt as `tool_calls` and `tool` messages |
| attachments | 893-972 | pictures (only when the server has images; at most 20 MB; read as `data:` URLs; pasting works) and text files (by MIME type or extension list, at most 512 KB, no NUL bytes) inlined as `File: <name>` followed by a fence longer than any backtick run in the file |
| export | 878-890 | the chat as `strata-chat-<date>.md` (thinking and tool calls in `<details>` blocks); "New chat" clears with an Undo toast |
| sampling drawer | 974-1066 | `DEFAULTS` (thinking `high`, temperature 0.6, top_p 0.95, top_k 20); "Use for other apps too" posts the settings to `/settings` (shared defaults for API clients) |
| start | 1067-1073 | `?q=<question>` starts a new chat with that question; the tab from the URL hash |

## Design system

`tokens.css` ("derived from the Gainsty design system") defines `--st-*` custom properties (type scale, 4-pt
spacing, colours for light and `[data-theme="dark"]`) and the Outfit `@font-face` rules (`../fonts/...`, so served
from `/fonts/`). `components.css` holds the UI kit (`st-header`, `st-tab`, `st-btn`, `st-card`, `st-metric`,
`st-progress`, `st-gauge`, `st-badge`, `st-drawer`, `st-toggle`, `st-code`, `st-table`, ...); `app.css` the layout.
`sprite.svg` has 32 icons (`#i-chat`, `#i-activity`, `#i-gpu`, `#i-cpu`, `#i-thinking`, `#i-tool`, ...). The theme
is chosen before first paint from `strata.theme` or the system preference. The Monitor tab "started from
@code-martin's dashboard idea (PR #22)" (upstream DETAILS.md, Credits).

## The API request monitor page (`monitor.html`, `monitor.js`, #332)

Off by default (it keeps prompts and answers in memory); on with `"api_monitor": true` or `--api-monitor`.
`monitor.html` is a self-contained page (its own inline styles on top of `tokens.css`, dark by default); `monitor.js`:

- polls `GET /v1/status` and `GET /api/requests` every 2 s; selecting a request fetches `GET /api/requests?id=<id>`;
- cards: model state (Loaded / Unloaded / Loading / In use), requests retained, last wall-clock (queue and loading
  included), last decode speed (`timings.predicted_per_second`);
- a request list with search and a status filter, and a detail pane: wall-clock, model load, queue wait, first
  token, prompt and output tokens, decode speed, format, error, and tabs Input (the original body), Output, Reasoning,
  Response, with a copy button and truncation notes;
- Load model / Unload model buttons (`POST /load`, `POST /unload`), disabled while requests are in flight;
- the API key is kept in `sessionStorage` (`strata.monitor.key`) for this tab only.

The data side is in [monitoring-and-telemetry.md](monitoring-and-telemetry.md#the-api-request-monitor-332).

## Tests

`serve/test_server.py` `WebApp` (`test_page_and_files`, `test_only_the_app_files_are_served`, `test_metrics`, ...),
`serve/test_monitor.py` (`test_page_and_relative_assets_are_served`, `ConversationCacheCard.test_the_card_is_on_the_page`),
`serve/test_runconfig.py` `Http.test_the_page_has_the_view`. There are no browser tests of `app.js`.

## In UnbiasedStrata

Removed (commit 8894a81). The fork's front end serves no web page; any OpenAI-compatible chat UI can be pointed at
its `/v1`.

## See also

- [monitoring-and-telemetry.md](monitoring-and-telemetry.md), [tools-and-mcp.md](tools-and-mcp.md), [config-and-runconfig.md](config-and-runconfig.md)
- [../../upstream/DETAILS.md](../../upstream/DETAILS.md) - "Windows" (the app's three tabs), "API request monitor".
