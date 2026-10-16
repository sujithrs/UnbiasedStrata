[Upstream docs](../README.md) › [Serve](README.md) › Strata's MCP server

# Strata's own MCP server (`tools/strata_mcp.py`)

## Purpose

An MCP server for the AI assistant a user already has (Claude Code, Claude Desktop, Cursor, VS Code Copilot agent
mode, Codex CLI, ...), so the assistant can install, start, check and stop Strata on request ("install Strata for
this PC", "is Strata running?"). It drives `setup.py` and `serve/server.py` as background processes and talks to a
running server over its HTTP API. One file, Python standard library only (it must run before `setup.py` has created
`.venv`), Python 3.10 or newer.

Not to be confused with `serve/mcp.py` ([tools-and-mcp.md](tools-and-mcp.md)), where Strata is the MCP *client* and
the model calls the user's tools.

Files: `tools/strata_mcp.py` (1738 lines), `tools/test_strata_mcp.py` (590 lines), `docs/MCP_SERVER.md` (the user
guide with every client's config snippet).

## Running it

```
claude mcp add strata -- python C:\Users\you\Strata\tools\strata_mcp.py      # Windows
claude mcp add strata -- python3 /home/you/Strata/tools/strata_mcp.py        # Linux
python tools/strata_mcp.py --root D:\Strata                                   # manage another Strata folder
```

`main` (`tools/strata_mcp.py:1711-1734`) refuses Python older than 3.10 and a folder without `setup.py` and
`serve/server.py` (exit code 2), redirects `sys.stdout` to stderr so stray prints never corrupt the protocol
stream, prints `[strata-mcp] ready (Strata folder ...)` to stderr and serves stdin until it closes. The hidden
`--install-job SPEC` mode is the background half of an install (`install_job`, `tools/strata_mcp.py:1665`).

## Protocol (`McpServer`, `tools/strata_mcp.py:1559-1662`)

- stdio, one JSON-RPC 2.0 message per line; batches (JSON arrays) supported.
- Protocol revisions `2025-06-18` (default), `2025-03-26`, `2024-11-05`: `initialize` answers in the client's
  revision when it is one of these (`PROTOCOLS`, `tools/strata_mcp.py:42`).
- Methods: `initialize` (capabilities `tools`, server info `strata` 0.1.0, and `INSTRUCTIONS` telling the assistant
  to call `strata_status` first and to get the user's OK before a 58-111 GB download), `ping`, `tools/list`,
  `tools/call`, `resources/list` and `prompts/list` (empty); notifications are ignored; others -32601.
- `tools/call` runs on its own thread, so pings and status calls are answered while a start waits for the model.
- Each result is JSON with a one-line `summary`, sent as text content and, from revision 2025-06-18, also as
  `structuredContent`. Tool errors (`ToolError`) are results with `isError: true`; an internal exception becomes an
  `isError` result too ("a bug must not end the server").

## The tools (`Tools`, `tools/strata_mcp.py:1042-1550`; schemas in `tool_defs`, line 925)

| Tool | What it does |
| --- | --- |
| `strata_status` | The running model (via `GET /health`, `/v1/status`, `/status`, and `/metrics` on older servers): model, engine version, context, busy or idle, VRAM and RAM. What is installed (configs, run scripts, engine version, data folder, downloads), an install in progress, the PC's hardware (OS, CPU, RAM, GPUs via `nvidia-smi` / ROCm / Windows video controllers, page file, free disk), and the size and context setup would recommend |
| `strata_models` | every family and size setup offers, with download size, RAM needed, fits / installed, contexts and the recommendation |
| `strata_install` | without `confirm: true`: the plan (download size, free disk, RAM check, the exact setup command). With it: runs `setup.py --yes --no-start --family ... --model ... --context ... --vision ...` in the background (creating `.venv` first if missing), output to `.strata-mcp/setup.log`. A second call shows step (1-7), download percentage or setup's error; `cancel: true` stops it. Refused while a model this server started is running |
| `strata_start` | runs `<venv python> serve/server.py --engine strata --config strata-<model>.json --port N [--gpu G]` detached (no `--open`, no console window), output to `.strata-mcp/server.log`, then polls `GET /health` for up to `wait_seconds` (default 50, max 600); returns "loading" if not ready. Refuses when a model already runs or another program holds the port |
| `strata_stop` | `POST /unload` (the engine frees its VRAM and pinned RAM itself), then ends the server it started: SIGTERM on POSIX (the server's own Ctrl+C path), `TerminateProcess` on Windows (the server's job object ends the engine, encoder and MCP servers); then a tree kill if needed. Refuses while a request runs unless `force: true`. A Strata started elsewhere is only unloaded over the API |
| `strata_logs` | the last 1-500 lines of the server, engine or setup log |
| `strata_benchmark` | one fixed request (a ~300-word story prompt, greedy, thinking off, `max_tokens` 16-512) and its output / prompt tokens per second from the response's `timings`; refused while busy |
| `strata_connect_info` | base URLs (OpenAI and Anthropic), model id, whether a key is needed, and settings for Claude Code, Codex, the OpenAI SDK, curl, Cursor, Continue and Open WebUI |

Argument choices (`model`, `family`, `context`, `kv`, `backend`, ...) come from `setup.py`'s own tables:
`Strata.setup_module` (`tools/strata_mcp.py:401`) imports `setup.py` as a module (stdout redirected, its command
runner replaced) and uses `MODELS`, `FAMILIES`, `CONTEXTS`; without it, the built-in `FALLBACK_MODELS` /
`FALLBACK_FAMILIES` / `FALLBACK_CONTEXTS` (lines 51-81), which a test keeps equal to `setup.py`'s.

## Safety (from `docs/MCP_SERVER.md` and the code)

- **No shell, no free paths.** `check_schema` (`tools/strata_mcp.py:320`) checks every argument against a fixed
  schema; unknown arguments and values setup does not offer are refused; `gpus` must match `all|\d{1,2}(,\d{1,2}){1,7}`
  (`GPUS_PATTERN`, line 921). Commands are argument lists.
- **One path argument**, `data_dir` (`check_data_dir`, line 1129): the current data folder, a folder inside the
  Strata folder, or a new absolute folder named `Strata*` with an existing parent outside system folders; `..`,
  relative and UNC paths are refused. Logs are read only from `.strata-mcp/` and an engine log inside the Strata or
  data folder.
- **Only its own processes.** `proc_identity` (line 111) records a process's start time (`/proc/<pid>/stat` field 22
  on Linux, `GetProcessTimes` on Windows) with its pid in `.strata-mcp/*.json`; `proc_terminate` / `proc_kill_tree`
  act only when both match, so a reused pid is never touched. POSIX processes are started as session leaders, and
  `end_group_leftovers` ends what is left of the group (an engine the server could not close).
- **Detached children.** `spawn_detached` (line 222): own session on POSIX; `CREATE_NEW_PROCESS_GROUP |
  CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB` on Windows (falling back without breakaway when the AI client's job
  forbids it; then the model ends with the app).
- **Local only.** Probes `127.0.0.1` only (proxies bypassed); never installs with `--host 0.0.0.0` or an API key, and
  leaves experimental options (speed projection, rope scaling past 262K, `--build`) to a manual setup. A key set in a
  config of the folder is used for its own requests (`api_key_for`, line 740) and never shown.
- **Big downloads need a yes**: `confirm: true`, after a plan that shows size and free space; refused when the disk is
  too small.

## Interfaces with the server

| Server endpoint | Used for |
| --- | --- |
| `GET /health` (`"service": "strata"`) | is a Strata server on the port, model, context, `loaded`, `api_key` |
| `GET /v1/status` | engine version, uptime, requests served, GPU and RAM, last timings |
| `GET /status` | busy / phase / tokens per second (refuse stop or benchmark while busy) |
| `GET /metrics` | fallback for older servers |
| `POST /unload` | free VRAM and RAM before ending the server |
| `POST /v1/chat/completions` | the benchmark |

## Tests

`tools/test_strata_mcp.py` builds a temporary fake Strata folder (a stand-in `setup.py` and `serve/server.py`, a
fake HTTP server), so nothing is downloaded and no GPU is used: `Protocol` (handshake, older revisions and errors, a
real stdio process, not a Strata folder), `Validation` (install arguments, `data_dir` paths, other tools' arguments,
an engine log outside the folders is not read), `Install` (plan then install, progress and cancel, failure, disk
space), `StartStop` (start / status / benchmark / stop, a crash while starting, loading then ready, stop while busy
needs force, an external server is only unloaded, an unrelated process is never killed, other programs on the
port), `Helpers` (progress parsing, the built-in table matches `setup.py`, Unsloth sizes, the recommendation
follows RAM). Run: `python -m unittest tools/test_strata_mcp.py -v`.

## In UnbiasedStrata

Removed with `tools/` and `docs/MCP_SERVER.md` (commit 8894a81). The fork has no installer or MCP server.

## See also

- [tools-and-mcp.md](tools-and-mcp.md), [http-api.md](http-api.md), [monitoring-and-telemetry.md](monitoring-and-telemetry.md)
- [docs/MCP_SERVER.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MCP_SERVER.md) (upstream)
