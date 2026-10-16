[Upstream docs](../README.md) › [Serve](README.md) › Windows

# Windows-specific behaviour of the server (`serve/winjob.py` and the rest)

## Purpose

Upstream Strata's main platform is Windows (`START-HERE.bat`). This page collects what the Python server, its MCP
client, telemetry and `tools/strata_mcp.py` do differently on Windows, starting with `serve/winjob.py`, which ties
every child process to the server's lifetime.

## `serve/winjob.py` (74 lines)

Closing the console window, ending the server in Task Manager, or a crash end the server without running its
cleanup, and the engine, the vision encoder and the MCP servers used to keep running on their own (holding tens of
GB of RAM and the GPU). `contain(proc)` (`serve/winjob.py:61-74`):

- creates, once per process, a job object (`CreateJobObjectW` with no security attributes, so the handle is not
  inherited) whose limit is `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (`0x2000`, set with
  `SetInformationJobObject(JobObjectExtendedLimitInformation)`; `_make_job`, `serve/winjob.py:48-58`);
- assigns the child to it (`AssignProcessToJobObject(job, proc._handle)`); processes the child starts later join the
  same job;
- returns `True` when it is in, `False` (harmlessly) elsewhere, without a process, or when the job cannot be made.

The only handle to the job is the server's, which the OS closes however the server ends, so everything in the job is
killed then. The server itself stays out of the job, so a browser that `--open` starts is not tied to it.

Callers: `StrataEngine.__init__` (`serve/server.py:456`), `Vision._start` (`serve/server.py:1286`),
`StdioTransport.start` (`serve/mcp.py:110`). `tools/strata_mcp.py`'s stop relies on it: ending the server process
ends its engine, encoder and MCP servers.

## Other Windows behaviour

| Where | What | Why |
| --- | --- | --- |
| `Server.allow_reuse_address = os.name != "nt"` (`serve/server.py:3588`) | no `SO_REUSEADDR` on Windows | there it lets a second server bind a port that is already serving, and requests land on either (a forgotten second start of `run-<model>.bat`) |
| `StrataEngine.close` (`serve/server.py:1234-1237`) | close the engine's stdin after `QUIT`, before waiting | "Windows' detached stdin reader must see EOF before shutdown" (the engine reads stdin with `_read(0, ...)` on its own thread, `src/program/generate.cpp:5611`) |
| `main` (`serve/server.py:3990-3992`) | a relative `"exe"` is made absolute against `"cwd"` | `CreateProcess` resolves `engine/strata.exe` against nothing it is told about (WinError 2) |
| `child_env` (`serve/server.py:1571-1574`) | `"lib_dirs"` go on `PATH` (Linux: `LD_LIBRARY_PATH`) | DLL search path for the CUDA or ROCm libraries setup installed |
| `hip_visible` (`serve/server.py:1544-1557`) | one-card AMD configs use `"hip_ordinal"` | an integrated Radeon that HIP also enumerates takes ordinal 0 and pushes the discrete card to 1 (#325) |
| `_control_body` (`serve/server.py:3221-3241`) | read the body of `/load` / `/unload` before replying | otherwise Windows sends a TCP reset |
| `main` (`serve/server.py:4127-4129`) | the main thread sleeps in 1 s steps | "Windows never delivers Ctrl+C to an untimed Event.wait()" |
| `main` (`serve/server.py:4110-4114`) | on a non-loopback bind, prints the Windows Firewall hint and a `New-NetFirewallRule` command | inbound connections are blocked until allowed |
| `main` (`serve/server.py:3937`) | the config is read as `utf-8-sig` | Notepad adds a BOM |
| `warn_tight_ram` (`serve/server.py:3610-3611`) | "Windows will slow down (paging to disk)" instead of the Linux OOM warning | |
| `StdioTransport.start` (`serve/mcp.py:96-103`) | `shutil.which` finds `npx.cmd`, `uvx.cmd`; `CREATE_NO_WINDOW` | `.cmd` files are found by `CreateProcess` only with their extension; no console windows |
| `StdioTransport._kill_tree` (`serve/mcp.py:231-248`) | `taskkill /PID <pid> /T /F` | `npx.cmd` -> `node`: end the whole tree (POSIX uses the process group) |
| `serve/telemetry.py` | NVML from `nvml.dll` or `Program Files\NVIDIA Corporation\NVSMI\nvml.dll`; CPU from `GetSystemTimes`, RAM from `GlobalMemoryStatusEx`, CPU name from the registry when psutil is missing | |
| `tools/strata_mcp.py` | process identity from `GetProcessTimes`; `TerminateProcess` to stop; `taskkill /T /F` for trees; `CREATE_NEW_PROCESS_GROUP \| CREATE_NO_WINDOW \| CREATE_BREAKAWAY_FROM_JOB` for detached starts; config under `%APPDATA%\Strata` | the server must outlive the AI client unless its job forbids breakaway |

Signals: `SIGTERM` is mapped to Ctrl+C's path where the platform allows it (`serve/server.py:4121-4126`); on Windows
`strata_mcp.py` stops the server with `TerminateProcess`, so cleanup happens through the job object, not through the
server's own shutdown code.

Task Scheduler note (from upstream's DETAILS.md, "Running it at startup"): a serve started by Task Scheduler with its
defaults loaded experts at 0.05 GiB/s (821-841 s) instead of 1.42-1.52 GiB/s (~35 s), measured on an RTX 5070 Ti +
Ryzen 7 9800X3D + NVMe; setting "Priority level: Normal" and "Run with highest privileges" restored 1.42 GiB/s.

## Tests

`serve/test_winjob.py` (Windows only; skipped elsewhere): a stand-in server process contains a child that starts a
grandchild, is killed with `TerminateProcess` (no `atexit`, no `finally`, like closing the window), and both must
end within 10 s; `contain(None)` is `False`. Run: `python -m unittest serve.test_winjob`. `serve/test_lifecycle.py`
`test_close_sends_eof_before_waiting_for_windows_reader` checks the stdin close order on every platform.

## In UnbiasedStrata

Removed. The fork dropped Windows support altogether (commit 8894a81 removed `serve/` with `winjob.py`, the `.bat`
files and the installer); its front end is Linux-only (fork/exec, `/proc/self/exe`, cgroup CPU quotas).

## See also

- [engine-process.md](engine-process.md), [tools-and-mcp.md](tools-and-mcp.md), [mcp-server.md](mcp-server.md)
- [../../upstream/DETAILS.md](../../upstream/DETAILS.md#windows) - upstream's Windows notes.
