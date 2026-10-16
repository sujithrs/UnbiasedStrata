[Upstream docs](../README.md) › [Serve](README.md) › Security

# API key, Host and Origin checks, CORS

## Purpose

What protects a server that runs a model (and, with MCP, tools) on the user's PC: the optional API key, the Host
check against DNS rebinding, the Origin check for browser pages, the stricter "own page" rule for routes that change
settings or run tools, and opt-in CORS. All in `serve/server.py`: the handler methods `parse_request`,
`_authorized`, `_foreign_page`, `_own_page`, `_cors`, and the module functions `host_name`, `allowed_hosts_of`,
`host_names_for`, `host_allowed`, `origin_allowed`, `origins_of` (`serve/server.py:3645-3797`).

Upstream's rule for installers (AGENTS.md): never expose the server beyond `127.0.0.1` without `--api-key`.

## The checks, in the order a request meets them

```
parse_request        no API key configured: Host must be a name this server answers to   -> 403
do_GET / do_POST     routes marked "key": _authorized (Bearer or x-api-key)            -> 401
do_POST /v1/*        no API key and an Origin header: _foreign_page                     -> 403 / 415
settings, config,    _own_page: JSON content type, and an Origin (if any) equal to the  -> 415 / 403
load/unload, vram,   request's own Host or listed in trusted_origins
MCP tools
```

### API key (`_authorized`, `serve/server.py:2951-2959`)

- Set by `--api-key`, `$STRATA_API_KEY`, or `"api_key"` in the config (`serve/server.py:4031`).
- Sent as `Authorization: Bearer <key>` or `x-api-key: <key>` (Anthropic clients); compared with
  `hmac.compare_digest` (#213: constant time).
- Every `POST`, and the `GET` routes `/metrics`, `/status`, `/v1/models`, `/models`, `/props`, `/slots`,
  `/v1/status`, `/settings`, `/config`, `/mcp`, `/api/requests` need it. Open without it: `/`, `/web/*`, `/fonts/*`,
  `/health` (which says whether a key is required), `/api-monitor` (the page; its data needs the key).
- An empty key given explicitly (`--api-key ""`, `--api-key=`, or an empty `STRATA_API_KEY`) stops the start with
  exit code 2 instead of silently disabling authentication (`serve/server.py:4025-4030`).
- With a key set, the Host and Origin checks are skipped: a rebinding page cannot send the key, and tunnels or
  proxies that pass their own Host keep working.

### Host check, DNS rebinding (`parse_request`, `serve/server.py:2835-2852`)

A page of another site can point its own DNS name at `127.0.0.1` and then talk to the server as same-origin. Its
requests carry that name in `Host`, so without an API key only known names pass (`host_allowed`,
`serve/server.py:3737`):

- any IP address (rebinding needs a name), `localhost` and `*.localhost`, a request without `Host` (HTTP/1.0);
- the names in `svc.host_names` (`host_names_for`, `serve/server.py:3717`): `localhost`, `127.0.0.1`, `::1`, the bind
  address; when listening beyond the PC (`0.0.0.0` or a LAN address) also the PC's name and `<name>.local`,
  `host.docker.internal` and the LAN IPv4 addresses; the hosts of `allowed_hosts` and `trusted_origins`;
- `allowed_hosts` entries starting with a dot match that name and every name below it; `"*"` turns the check off.

`allowed_hosts` comes from the config (a name or a list) plus `$STRATA_ALLOWED_HOSTS` (comma-separated); a scheme,
port or path is stripped, and a malformed entry stops the start (`allowed_hosts_of`, `serve/server.py:3695`). A
refused request gets 403 `forbidden` naming both ways out (add the name, or set a key) and one line in the server
window.

### Origin check for the API (`_foreign_page`, `serve/server.py:2854-2873`)

Any web page can POST `text/plain` to a local server without a CORS preflight. Without an API key, a `POST /v1/...`
that carries an `Origin` header passes only if `origin_allowed` (`serve/server.py:3750`) says so, and then only with
a JSON content type (else 415):

- the Origin's `host[:port]` equals the request's `Host` (the server's own page);
- an `http(s)` origin on a loopback name, `*.localhost` or a known host name (any port); unlike the Host check, an
  IP is not trusted by itself here (a page from another IP is another site);
- an origin listed in `trusted_origins` or `cors_origins` (`"*"` in `cors_origins` allows every page);
- a non-web scheme (`chrome-extension://`, `moz-extension://`, an Electron `app://`): no web site can send one.

`Origin: null` (sandboxed frames, `file://` pages) is refused. Requests without an `Origin` (curl, the OpenAI and
Anthropic SDKs, other servers) are not affected.

### Own page only (`_own_page`, `serve/server.py:3243-3258`)

`POST /settings`, `POST /config`, `/load`, `/unload`, `/v1/load`, `/v1/unload`, `/v1/vram` and requests with
`"strata_mcp": true` additionally require, with or without a key:

- `Content-Type: application/json` (a form or a "simple" cross-site request cannot send it without a preflight,
  which this server never grants for these paths) - else 415;
- no `Origin`, or an `Origin` whose `host[:port]` is exactly the request's `Host`, or one listed in
  `trusted_origins` - else 403. Proxy headers (`X-Forwarded-*`, `CF-Ray`, `CF-Connecting-IP`) open nothing.

`trusted_origins` exists for the web app reached through a reverse proxy or tunnel whose address differs (#321);
it never takes a wildcard.

### CORS (`_cors`, `do_OPTIONS`, `serve/server.py:2906-2932`)

Off unless `"cors_origins"` is configured (#321). Then, for `/v1/*` paths only, the server sends
`Access-Control-Allow-Origin` (the origin itself with `Vary: Origin`, or `*`), `Allow-Methods: GET, POST, OPTIONS`,
`Allow-Headers` (what the preflight asked for, else `Authorization, Content-Type, x-api-key, anthropic-version,
anthropic-beta`) and `Max-Age: 600`. `OPTIONS` answers 204 without needing the key (preflights carry no
credentials). `/settings`, `/unload` and the MCP tools never get CORS headers. `["*"]` without a key prints a
warning at start.

`origins_of` (`serve/server.py:3778`) accepts only `http(s)://host[:port]` without a path; `"*"` only for
`cors_origins`; a wrong entry stops the start.

## Other safety measures in the server

- `GET /status` needs the key because it shows the end of the last answer (#212); finished requests drop `tail`.
- The API request monitor is off by default because it keeps prompts and answers in memory; with it on and no key,
  the start prints that anyone who can reach the server can read them.
- Static files: `/web/` and `/fonts/` serve only plain names with known extensions from `serve/web/` (no `/` or `\`).
- `/v1/load` and `/v1/unload` accept only the configured model name (404 otherwise), and read their body before
  acting (`/load`, `/unload`: at most 64 KiB, `_control_body`).
- MCP tools run with the user's rights and the model decides when to call them; only the web app's own page can
  request them ([tools-and-mcp.md](tools-and-mcp.md)).
- On a non-loopback bind the start prints the LAN addresses and, without a key, "WARNING: no API key - anyone on
  your network can use this model" (`serve/server.py:4100-4114`).

## Configuration

| Key / flag / env | Meaning |
| --- | --- |
| `"host"` / `--host` | bind address, default `127.0.0.1` (#26: the run scripts pass none, the config can) |
| `"api_key"` / `--api-key` / `STRATA_API_KEY` | the key |
| `"allowed_hosts"` / `STRATA_ALLOWED_HOSTS` | extra Host names, `.suffix`, or `*` |
| `"trusted_origins"` | origins that count as Strata's own page |
| `"cors_origins"` | origins that get CORS headers on `/v1/*` (and pass the Origin check) |

## Tests

`serve/test_security.py` (`HostNames`, `HostCheck`, `OriginCheck`, `OverHttp`: a rebinding name refused everywhere,
local names and IPs pass, no Host passes, allowed hosts and the wildcard, the refusal text, the key turns the Host
check off, trusted origins' hosts, SDK clients without Origin, cross-site pages refused, configured origins pass,
the control body is read before the reply, `/load` and `/unload` need JSON from the own page). Also
`serve/test_server.py` `SharedSettings` (proxy headers do not make a page Strata's own), `StatusNeedsTheKey`,
`UsageAndStatus` (CORS), `WebApp.test_metrics_need_the_key_when_one_is_set`, and the "only from the app's own page"
tests in `test_mcp.py`, `test_runconfig.py`, `test_vram.py`, `test_lifecycle.py`.

## In UnbiasedStrata

Removed (commit 8894a81). The fork's front end has no API key, Host, Origin or CORS handling; it listens on
`127.0.0.1` by default (`--host`).

## See also

- [http-api.md](http-api.md), [config-and-runconfig.md](config-and-runconfig.md)
- [../../upstream/DETAILS.md](../../upstream/DETAILS.md#using-it) - "From other devices", "From the internet",
  "From web apps in a browser (CORS)", "Host names (DNS rebinding)", "Web pages without an API key".
