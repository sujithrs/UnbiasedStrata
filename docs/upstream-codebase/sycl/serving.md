[Upstream docs](../README.md) › [Intel SYCL port](README.md) › Setup and serving

# Setup and serving on an Intel GPU

## Purpose

The port changes no shared file, so the installer and the server reach the Intel engine through wrappers that
import upstream's `setup.py` and `serve/server.py` and replace a few functions at run time ("monkey-patching").
This page covers the chain from `./setup.sh --backend sycl` to a running engine in a container, and the Intel
additions to the server: GPU telemetry, a model menu and OpenAI logprobs.

Files covered:

| File | Lines | Role |
|---|---|---|
| [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py#L3507-L3531) | `sycl_setup`, lines 3507-3531 | `--backend sycl` hand-off |
| [tools/test_setup_sycl.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_sycl.py) | 75 | Tests of the hand-off |
| [sycl/setup_intel.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/setup_intel.py) | 246 | Upstream's installer with the Intel steps swapped in |
| [sycl/serve/strata-sycl.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/serve/strata-sycl.sh) | 27 | The engine `exe`: the binary in Docker |
| [sycl/serve/server_intel.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/serve/server_intel.py) | 242 | Upstream's server plus the Intel pieces |
| [sycl/serve/xe_telemetry.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/serve/xe_telemetry.py) | 103 | Arc readings for the Monitor tab |
| [sycl/serve/web/switcher.js](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/serve/web/switcher.js) | 60 | The Model menu |
| [sycl/serve/web/switcher.css](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/serve/web/switcher.css) | 7 | Its style (uses the web app's `--st-*` tokens) |
| [sycl/tools/gpustat.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/gpustat.py) | 182 | Root telemetry sampler |
| [sycl/tools/gpustat.service](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/gpustat.service) | 10 | Its systemd unit |

## Control flow

```
./setup.sh --backend sycl [opts]          setup.sh:39  exec .venv/bin/python setup.py "$@"
  setup.py main()                          setup.py:3625  a.backend == "sycl" -> sycl_setup(sys.argv[1:])
    sycl_setup()                           warn "EXPERIMENTAL"; Windows -> fail; drop --backend; subprocess.call
      python sycl/setup_intel.py [opts]
        install(): find the Arc in sysfs, find the SYCL build + Docker image, patch setup.py's steps,
                   sys.argv = setup.py [opts] --backend hip --vision none --low-ram off; S.main()
          setup.py's normal flow (model, download, pack, tokenizer, MTP, context, KV questions)
          write_run_script -> strata-<model>.json rewritten by to_sycl(); run-<model>.sh -> server_intel.py
          start -> exec /bin/sh run-<model>.sh
run-<model>.sh
  python sycl/serve/server_intel.py --engine strata --config strata-<model>.json --port N
    install_xe_reader(); install_switcher(argv); install_logprobs(); serve.server.main()
      StrataEngine spawns cfg["exe"] = sycl/serve/strata-sycl.sh <args>   (env from cfg["env"])
        docker run --rm -i strata-sycl-dev "cd /work/<repo> && exec build-sycl-aot/strata <args>"
          engine serve mode on stdin/stdout (GEN ... / T id / LP ... / DONE)
```

## setup.py: the hand-off

`setup.py --backend {cuda,hip,sycl}` (`setup.py:3619-3622`). For `sycl`, `main()` returns `sycl_setup(sys.argv[1:])`
before any other option is checked (`:3625-3626`). `sycl_setup` (`:3507-3531`):

1. prints a warning that the Intel engine is an experimental community port not tested by the maintainers;
2. on Windows fails with "the Intel Arc engine has no Windows setup yet" and a pointer to `docs/INTEL_ARC.md`;
3. says the engine is built from source with oneAPI;
4. removes `--backend X` and `--backend=X` from the arguments;
5. fails if `sycl/setup_intel.py` is missing ("use a full Strata checkout");
6. returns `subprocess.call([sys.executable, "sycl/setup_intel.py", *rest])`.

No CUDA or HIP code path runs. There is no ready-made Intel engine in the release zips (INTEL_ARC.md).

## sycl/setup_intel.py

It inserts the repository root into `sys.path` and `import setup as S` (`:33-35`), so it drives the very same
`setup.py`.

**Constants.** `SYCL_WRAPPER` = `sycl/serve/strata-sycl.sh`, `SYCL_IMAGE` = `strata-sycl-dev`, `SERVER` =
`sycl/serve/server_intel.py`, `MOUNT` = `STRATA_SYCL_ROOT` or the folder above the checkout (`:37-40`).
`INTEL_ARC` (`:44-47`) maps PCI device ids to a name and VRAM size, because `lspci` names lag new cards; see
[hardware.md](hardware.md#cards-setup-recognises).

**Functions.**

| Function | Lines | What it does |
|---|---|---|
| `intel_gpus()` | 50-78 | Scans `/sys/class/drm/card*` (skipping connector entries) for vendor `0x8086` under the `xe` or `i915` driver. Unknown ids are kept only under `xe` (an unknown `i915` device is taken to be an integrated GPU); their VRAM comes from the size of PCI resource line 2 (the VRAM BAR). Returns setup's GPU dicts with `arch` `"xe"` |
| `sycl_engine()` | 81-92 | `(binary, why-not)`: Linux only; `build-sycl-aot/strata` or `build-sycl/strata`; `docker` on `PATH`; `docker image inspect strata-sycl-dev` succeeds |
| `sycl_path(p)` | 95-103 | A host path as `/work/...` inside the container; fails if the path is outside `MOUNT` (models and data must live under it) |
| `sycl_version()` | 106-108 | The `project(... VERSION x)` of `sycl/CMakeLists.txt` |
| `to_sycl(cfg, exe, ram, keep)` | 121-156 | Rewrites the config setup wrote for its HIP path (below) |
| `install(argv)` | 159-238 | Checks, patches and runs setup (below) |

**What `install()` replaces** (`:166-229`). It first fails with a clear message if any of `gpus`, `amd_gpus`,
`amd_problem`, `hip_vision`, `build_engine_hip`, `hipblaslt_table`, `ram_gb`, `write_run_script`, `start`, `say`,
`main` is no longer a callable in `setup.py` (`:166-169`). Then:

| `setup.py` function | Replacement | Effect |
|---|---|---|
| `gpus` | `[]` | no NVIDIA card |
| `amd_gpus` | the Intel list | the Arc goes through setup's AMD path, the one that builds locally and has no images |
| `amd_problem` | `None` | no AMD driver complaints |
| `hip_vision` | `"none"` | images are not wired on the port |
| `build_engine_hip` | a temp dir with a stub `BUILD.json` (`version` from CMake, `source: local`, no `lib_dirs`) | no compile step |
| `hipblaslt_table` | `None` | no AMD GEMM tuning |
| `ram_gb` | `max(real RAM, 1024)` | setup's "RAM needed" rule does not block, since experts live in VRAM |
| `say` | `say_intel` | rewrites the AMD wording to Intel, shows the real RAM, and replaces `--check`'s per-model RAM verdict with a VRAM one: weights = download size minus the 28.8 GB per-layer embedding table (read from disk); "fits in VRAM" if within VRAM - 1.5 GB, "fits with part of its experts mirrored in RAM (slower)" if within VRAM - 3 + (RAM - 10), else "does not fit" |
| `write_run_script` | wrapper | rewrites the config with `to_sycl`, then calls the original and replaces `serve/server.py` with `sycl/serve/server_intel.py` in the script |
| `start` | wrapper | for a `"backend": "sycl"` config, `exec`s `/bin/sh run-<model>.sh` |

Finally it sets `sys.argv` to `setup.py` plus the user's options, adding `--backend hip`, `--vision none` and
`--low-ram off` unless given, and exits with `S.main()` (`:231-238`). Before patching it reads every existing
`strata-*.json` and keeps their `model_switcher` and `sampling` keys across the rewrite (`:172-178`).

**`to_sycl()`** (`:121-156`), the config rewrite:

- drops `--resident-experts` and `--mmap-experts` (the CUDA engine's low-RAM modes) and any `--kv-resident`;
- maps `--pack`, `--native`, `--ple-gguf`, `--expert-profile`, `--mtp` and `--control-vector-scaled` paths to
  container paths;
- estimates the KV size as `ctx * 13 * (576 for q4_0, else 1056) / 1e9` GB and adds `--kv-resident 32768` (KV
  streaming) when the KV type is not `k8v4`, the context is at least 65,536 and RAM is at least the KV plus 6 GB;
- adds `--stream-experts` and `--vram-reserve-mib 1024` (context up to 32,768) or `2048` (above), and for contexts
  above 32,768 replaces `--prefill` with `4096`;
- removes `lib_dirs`, `env`, `vision`, `gpus`; sets `"backend": "sycl"`, `"exe"` (the wrapper), `"sycl_root"`;
- sets `env` `STRATA_SYCL_BIN` when the binary is not `build-sycl-aot/strata` and `STRATA_SYCL_ROOT` when the mount
  is not the default; `serve/server.py` copies `cfg["env"]` into the engine's environment (`serve/server.py:1569-1570`);
- merges back the kept `model_switcher` / `sampling` keys.

## sycl/serve/strata-sycl.sh

A drop-in `exe` for `serve/server.py --engine strata`. The serve protocol is lines on stdin/stdout, so the script
`exec`s `docker run --rm -i` with the pipes attached and stderr going to the server's log (`:2-5`, `:22-27`):

- `--device /dev/dri`, `--oom-score-adj 1000` (the container is the first the kernel kills), `--stop-timeout 30`,
  `--no-healthcheck`, `-v <root>:/work`;
- `-e STRATA_VERIFY_DEVICE_PLAN=1 -e STRATA_VERIFY_NO_HOST=1 -e STRATA_STAGER_THREADS=12`
  ([differences.md](differences.md#5-decode-loop-and-verify-window));
- `ONEAPI_DEVICE_SELECTOR` passed through when set (the image pins `level_zero:0`; `level_zero:*` for a two-card
  `--layer-split`);
- runs `cd /work/<repo-name> && exec <STRATA_SYCL_BIN> <args>` with each argument quoted by `printf %q`.

It first removes a container of the same name left by a killed server (`:16`).

| Variable | Default |
|---|---|
| `STRATA_SYCL_ROOT` | the directory above the checkout, mounted at `/work` |
| `STRATA_SYCL_IMAGE` | `strata-sycl-dev` |
| `STRATA_SYCL_BIN` | `build-sycl-aot/strata` (relative to the repo) |
| `STRATA_SYCL_NAME` | `strata-sycl-serve` (container name; one engine per name) |
| `ONEAPI_DEVICE_SELECTOR` | passed in only when set |

## sycl/serve/server_intel.py

Usage: `python sycl/serve/server_intel.py --engine strata --config strata-<model>.json --port 8080 [...]`, every
argument being `serve/server.py`'s. At start it calls three installers and then `serve.server.main()`
(`:238-242`).

**`install_xe_reader()`** (`:36-47`) wraps `serve.telemetry.gpu_reader(index, amd)`: if upstream's reader is not
`ok()` (no NVML card), it returns an `_XeGpu(index)` when that finds an Intel card.

**`install_switcher(argv)`** (`:69-131`) does nothing unless the config has a `"model_switcher"` URL. Then it wraps
`serve.server.make_handler` with a subclass that adds:

| Route | Behaviour |
|---|---|
| `GET /switcher` (authorised) | `model_switcher(url, port, model)`: POSTs `{}` to the switcher RPC and returns `{"enabled", "mode", "up", "starting", "choices", "model"}`, the choices filtered to models served on this server's port |
| `POST /switcher` (authorised) | the same with `{"mode": m}`, asking the host to load model `m` |
| `GET /sycl-web/<file>` | serves `switcher.js` / `switcher.css` from `sycl/serve/web/` (only those suffixes, only files directly in that folder) |
| `GET /` | `serve/web/index.html` with the stylesheet link and the script injected before `</head>` and `</body>` |

The RPC is an external program on the host that swaps models on one card; it is not part of this repository.
`switcher.js` adds a "Model" select to the header (after `.st-header__spacer`), hidden unless there are at least two
choices; changing it asks for confirmation, POSTs the switch, then polls `/health` every 3 s until the model name
changes and reloads (giving up after 6 minutes); it refreshes every 15 s so a switch started elsewhere shows.

**`install_logprobs()`** (`:134-223`) adds OpenAI `logprobs` / `top_logprobs` to `/v1/chat/completions`, streamed or
not:

- replaces `StrataEngine._pump` with a copy that diverts `LP ` lines into a per-engine deque `lp_lines`, each paired
  with the id of the preceding `T` line;
- wraps `StrataEngine.sampling_keys` to append ` logprobs=K` to the engine's `GEN` line when the request asks;
- wraps `StrataEngine.generate` to clear `lp_lines` per request;
- wraps `openai_chunks` to attach `choices[0].logprobs = {"content" | "reasoning_content": [...]}` to each chunk
  (`token`, `logprob`, `bytes`, `top_logprobs`), skipping stop tokens; thinking tokens are listed under
  `reasoning_content`;
- wraps `openai_collect` to merge them for non-streamed replies.

`top_logprobs(req)` (`:226-235`): `logprobs: true` with `top_logprobs` clamped to 0..20 (default 0), or the legacy
integer form `logprobs: N` (capped at 20); otherwise no logprobs. INTEL.md: not on `/v1/completions` or the
Anthropic endpoint yet; each token costs one 1 MB logits-row copy when asked.

## Telemetry: xe_telemetry.py and gpustat.py

`serve/telemetry.py` reads NVIDIA cards through NVML. The xe driver exposes VRAM use and busy cycles only per DRM
client in `/proc/*/fdinfo`, which only root can read, so the port splits the job:

- **`sycl/tools/gpustat.py`** runs as root (`gpustat.service`, `Restart=always`), every `GPUSTAT_INTERVAL` seconds
  (default 3) scanning all processes' DRM file descriptors. It writes `/run/gpustat.json` atomically (mode 644) with
  `name`, `pci`, `ts`, `vram_used_mb` (sum of `drm-resident-vram0` over unique clients), `vram_total_mb` (largest PCI
  BAR, or `GPUSTAT_VRAM_MB`), `busy_pct` (cycle deltas of `rcs`/`ccs` engines), `temp_pkg`, `temp_vram`,
  `temp_vram_max` (hottest VRAM channel), `temp_pcie`, `temp_mctrl`, `power_w` and `pkg_power_w` (energy deltas over a
  `GPUSTAT_POWER_WINDOW`-second window, default 15, because the counters update in bursts), `power_cap_w`, `pcie`
  (current link, card max, slot max; re-read every 30 s), `host_load1`, `host_cpus`, `energy_j`, `fan_rpm`,
  `freq_mhz`. `GPUSTAT_PCI` picks the card (default the first Intel GPU with `tile*/gt*`). Install:
  `sudo install -m 755 sycl/tools/gpustat.py /usr/local/sbin/`, `sudo install -m 644 sycl/tools/gpustat.service
  /etc/systemd/system/ && sudo systemctl enable --now gpustat` (`gpustat.py:13-14`).
- **`sycl/serve/xe_telemetry.py`** `_XeGpu` has NVML's reader shape (`ok()`, `name()`, `read()`). It finds the first
  `0x8086` card with a `tile0` directory and its hwmon. `read()` takes `util`, `mem_used`, `mem_total`, `temp`,
  `power`, `power_limit` and `vram_temp` from `/run/gpustat.json` if it is less than 15 s old; without the sampler
  it still gets temperature, power (from `energy1_input` deltas) and power cap from hwmon. `_link()` walks up the
  PCI path to the first port wider than x1 (the card's own functions sit behind an internal x1 switch) and reports
  current generation, max generation (the lower of card and root port) and width. PCIe traffic counters are `None`
  (xe has none).

## Configuration

| Key or variable | Where | Meaning |
|---|---|---|
| `"backend": "sycl"` | config | marks a SYCL config for `setup_intel.py`'s `start` |
| `"exe"` | config | `sycl/serve/strata-sycl.sh` |
| `"sycl_root"` | config | the mounted data root (informational; `strata-sycl.sh` reads `STRATA_SYCL_ROOT`) |
| `"env"` | config | `STRATA_SYCL_BIN`, `STRATA_SYCL_ROOT` when not default |
| `"model_switcher"` | config, hand-set | URL of the host's model-swap RPC; enables the Model menu |
| `"sampling"` | config, hand-set | kept across a rerun of `setup_intel.py` |
| `STRATA_SYCL_ROOT`, `STRATA_SYCL_IMAGE`, `STRATA_SYCL_BIN`, `STRATA_SYCL_NAME` | `strata-sycl.sh` (`STRATA_SYCL_ROOT` also `setup_intel.py`) | see above |
| `GPUSTAT_INTERVAL`, `GPUSTAT_POWER_WINDOW`, `GPUSTAT_PCI`, `GPUSTAT_VRAM_MB` | `gpustat.py` | sampler settings |

An example config, from INTEL.md "Serving the port":

```json
{"engine": "strata", "exe": "<repo>/sycl/serve/strata-sycl.sh",
 "args": ["--pack", "/work/pack", "--native", "<shard 1>", "--ple-gguf", "<shard 2>",
          "--expert-profile", "data/expert-profile-coder.bin", "--expert-cache", "auto", "--stream-experts",
          "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/work/mtp/rt",
          "--max-context", "32768", "--kv", "int8", "--vram-reserve-mib", "1024"]}
```

## Tests

`python -m unittest tools.test_setup_sycl` (no GPU, no network). It patches `setup.WIN` and
`setup.subprocess.call` and checks (`tools/test_setup_sycl.py:33-71`):

| Test | Checks |
|---|---|
| `test_windows_stops_with_the_docs_pointer` | exit 1, "EXPERIMENTAL" and `docs/INTEL_ARC.md` printed, nothing launched |
| `test_linux_hands_over_to_setup_intel_without_the_backend_flag` | `sycl/setup_intel.py` called with `--model IQ2_XS --port 8085` (the `--backend sycl` pair removed) |
| `test_backend_equals_form_is_dropped_too` | `--backend=sycl` removed as well |
| `test_the_parser_accepts_sycl` | `setup.py` source has `choices=["cuda", "hip", "sycl"]` |
| `test_setup_intel_finds_the_setup_functions_it_replaces` | every name in `setup_intel.py`'s `for name in (...)` list is a callable in `setup.py` |
| `CMakeOption.test_sycl_is_off_by_default_and_exclusive` | `STRATA_ENABLE_SYCL` exists, defaults to `OFF`, and the file has `add_subdirectory(sycl)` |

INTEL.md adds a manual check after every upstream merge: `sycl/setup_intel.py --check` and
`sycl/serve/server_intel.py --engine mock` must still run.

## Gotchas

- **WSL2.** `intel_gpus()` reads `/sys/class/drm`, which WSL2 does not have, so setup cannot find the card there
  (INTEL_ARC.md).
- **Paths.** Models, packs, the data folder and the checkout must be under the mounted root, or `sycl_path` fails.
- **`STRATA_VERIFY_NO_HOST=1` is always set** by `strata-sycl.sh` although it is only valid with every expert
  resident (INTEL_ARC.md; see [differences.md](differences.md#gotchas)).
- **`_pump` is replaced, not wrapped.** The copy in `server_intel.py` lacks upstream's routing of batch-slot lines
  (`BT`, `BDONE`) and the end-of-stream signal to slot queues (`serve/server.py:517-531`). The port's engine has no
  batch slots, so this matters only if a config asks for `"parallel"` (see
  [differences.md](differences.md#9-what-the-port-leaves-out)).
- **Docker only.** Setup's path always runs the engine in the `strata-sycl-dev` image; running it natively is the
  "How to run it by hand" recipe in INTEL.md.
- The first start of a 30 GB model takes about two minutes (INTEL.md "Setup"); the cache fill was later pipelined
  ([differences.md](differences.md#3-expert-storage---stream-experts-and-the-host-mirror)).

## In UnbiasedStrata

Removed: `sycl/setup_intel.py`, `sycl/serve/*` and `sycl/tools/gpustat*` with `sycl/` in commit `8894a81`; the same
commit removed `setup.py`, `setup.sh`, `serve/` and `tools/` (including `tools/test_setup_sycl.py`), so nothing of
this chain remains in the fork.

## See also

- [differences.md](differences.md#6-serve-protocol-addition-logprobs): the engine side of logprobs.
- [build.md](build.md): building the binary that `strata-sycl.sh` runs.
- [hardware.md](hardware.md): card ids and VRAM sizes.
