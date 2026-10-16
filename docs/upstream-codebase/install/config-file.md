[Upstream docs](../README.md) › [Install](README.md) › Config file

# The run config: strata-&lt;model&gt;.json

## Purpose

The run config is the only thing that connects setup to the server. Setup writes it at the end of step 7, changes it
on later starts, and `serve/server.py --config` reads it. This page covers every key setup writes, what each one
becomes for the engine or the server, how a second setup run keeps what the user added, and the files written next
to it.

Files covered: [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py)
(`setup.py:4217-4385`, `setup.py:2684-2830`, `setup.py:3453-3466`). How the server reads the rest of the file is in
[serve](../serve/README.md).

## Name and place

`ROOT/strata-<tag>.json`, where `<tag>` is the family tag plus the size in lower case: `strata-iq3_xxs.json`,
`strata-swift-iq2_xs.json`, `strata-coder-iq1_m.json`, `strata-unsloth-ud-iq4_xs.json` (`setup.py:4375`). The config
lives in the Strata folder. The model files it points to live in the data folder.

## An example

The golden baseline's case `64GB-1x32GB qwen recommended` (`tools/test_setup_golden.json`). The temp folder shows
as `<T>` and the exe name as `<EXE>`:

```json
{
 "args": ["--pack", "<T>/data/packs/iq3_xxs",
          "--native", "<T>/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
          "--ple-gguf", "<T>/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
          "--expert-profile", "<T>/data/expert-profile.bin", "--expert-cache", "auto", "--prefill", "auto",
          "--spec", "4", "--spec-min-p", "0.5", "--mtp", "<T>/data/mtp/rt", "--max-context", "131072",
          "--kv", "int8", "--kv-resident", "32768"],
 "cwd": "<T>", "exe": "<T>/engine/<EXE>", "gpu": 0, "gpus_asked": true, "lib_dirs": ["<cuda>"],
 "log": "<T>/strata-iq3_xxs.log", "model_name": "qwen3.8-flash-next-iq3_xxs", "port": 8080,
 "tokenizer": "<T>/data/packs/iq3_xxs/tokenizer"
}
```

(The golden harness passes `--models-dir <T>/models`. The shards go to `<models-dir>/<family tag><SIZE>` with the
size's case kept, here `IQ3_XXS` (`setup.py:4061`). The pack, config, log and run script use the lower-case tag.)

## Keys setup writes (`SETUP_KEYS`, `setup.py:2694-2695`, and a few more)

| Key | Written when | Value | Read by | Becomes |
| --- | --- | --- | --- | --- |
| `exe` | always | `engine/strata(.exe)` or `engine-cuda12/...` | server | The engine program it spawns. A relative path resolves against `cwd` |
| `args` | always | The engine's argument list (next table) | server | The engine's command line, plus what the server adds: `--layer-split`, `--batch`, `--split-skip-if-fits`, `--vram-elastic`, `--expert-profile-save` (`serve/server.py:1461-1479`) |
| `cwd` | always | The Strata folder | server | The engine's working directory |
| `tokenizer` | always | `<pack>/tokenizer` | server | The server's tokenizer files |
| `model_name` | always | `<family name>-<size>` | server | The model name in API answers |
| `log` | always | `ROOT/strata-<tag>.log` | server | The engine log, also used for calibration errors |
| `lib_dirs` | always | CUDA wheel or toolkit folders, or `rocm/bin` | server | Put first on `PATH` (Windows) or `LD_LIBRARY_PATH` (`child_env`, `serve/server.py:1560-1576`) |
| `port` | always | `--port` or 8080 | setup, server | The run script's and `start()`'s `--port` |
| `cuda` | CUDA 12 engine | `12` | setup only | `config_toolkit()`: which engine folder this model uses |
| `backend` | AMD | `"hip"` | setup, server | `HIP_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES`; AMD readings |
| `env` | AMD | `STRATA_HIPBLASLT_TUNING=<table>`, `STRATA_RESIDENT_PIN=0` (`SETUP_ENV`) | server | The engine's environment |
| `gpu` | several cards on the PC, `--gpu`, or a split | an index, or a list | server | `CUDA_DEVICE_ORDER=PCI_BUS_ID` plus `CUDA_VISIBLE_DEVICES` (or `HIP_VISIBLE_DEVICES`). A list also means `--layer-split` |
| `gpus_asked` | with `gpu` | `true` | setup only | `offer_together()` does not ask again |
| `layer_split` | a split | `--layer-split` value or `"auto"` | server | `--layer-split <value>` (`layer_split_value()`, `serve/server.py:1428-1458`) |
| `host` | `--host` | e.g. `0.0.0.0` | server | Listen address (default `127.0.0.1`) |
| `api_key` | `--api-key` | a secret | server | Required bearer key |
| `draft_vocab` | `--draft-vocab` or kept | `cjk`, `en`, `cyrillic`, `fr` | setup | Which `data/draft_vocab*.bin` is copied to `mtp/rt/draft_vocab.bin` at each start |
| `vision` | images on | `{exe, mmproj, model, gpu, max_tokens[, threads]}` (`SETUP_VISION`) | server | `strata-vision --mmproj <f> --model <shard1> [--gpu] [--threads N] --max-tokens N` (`serve/server.py:1267-1274`) |

Written only when given, and otherwise carried over from an earlier config:

| Key | From | Read by | Becomes |
| --- | --- | --- | --- |
| `open_browser` | `--no-browser` (false) / `--browser` (true) | setup, server | No `--open` in `start()` and the run script; the server also honors `false` (#609) |
| `parallel` | `--parallel N` | server | `--batch N` (`parallel_args()`, `serve/server.py:1485-1503`) |
| `remote_expert_opt` | the user (`false`) | setup | Keeps `--remote-expert-opt` out |

## The engine arguments in `args`

| Flag | When | From | Page |
| --- | --- | --- | --- |
| `--pack <dir>` | always | `<data>/packs/<tag>` | [downloads-and-packs.md](downloads-and-packs.md#the-pack) |
| `--native <shard1>` | always | the first GGUF shard | |
| `--ple-gguf <shard>` | 2 shards or fewer | the shard with `per_layer_token_embd.weight` | [context-kv-and-options.md](context-kv-and-options.md#fixed-engine-flags) |
| `--expert-profile <bin>` | always | `data/expert-profile.bin` (Coder: `expert-profile-coder.bin`) | |
| `--expert-cache auto`, `--prefill auto`, `--spec 4`, `--spec-min-p 0.5` | always | fixed (calibration may change `--spec-min-p`) | |
| `--mtp <dir>` | always | `<data>/mtp/rt` | [downloads-and-packs.md](downloads-and-packs.md#the-mtp-draft-head) |
| `--max-context N` | always | the context | |
| `--rope-scaling M --rope-scale F` | context > 262144, or a method given | `resolve_rope()` | |
| `--kv int8\|q4_0\|k8v4` | context > 8192 | `--kv` / question | |
| `--resident-experts` / `--mmap-experts` | low-RAM mode | variant | [low-ram-and-multi-gpu.md](low-ram-and-multi-gpu.md) |
| `--ple-io ram` | PLE shard on a rotational disk and enough RAM | | |
| `--kv-resident 32768` | KV streaming | | |
| `--resident-budget-gib N` | budget model on one GPU | `budget_choice()` | |
| `--vision --vram-reserve-mib 700` | images | `VISION` | |
| `--vram-reserve-mib N` | `--vram-reserve-mib` | | |
| `--control-vector-scaled <gguf>:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer` | speed projection on | | |
| `--remote-expert-opt` | two or more GPUs | `recommend_remote_expert_opt()` | |
| `--pool-workers N` | hybrid CPU rule, or calibration | | [calibration.md](calibration.md) |
| `--pcie-frac F` | calibration | | [calibration.md](calibration.md) |

The engine parses all of these in `src/program/generate.cpp` (the option loop around `src/program/generate.cpp:1290-1520`).

## Writing and re-writing (#459 #629)

- `write_config(path, cfg)` (`setup.py:2684-2689`) writes `<name>.tmp` and then renames it over the file, so a
  setup stopped half-way never leaves an empty config.
- `write_setup_config(cfg_path, cfg, source)` (`setup.py:2739-2769`) handles a second setup run for the same model,
  or the first run in a copy that adopts an earlier install's config:
  - `carry_over(old, cfg)` (`setup.py:2700-2726`) copies every key not in `SETUP_KEYS` and not already in the new
    config. Examples are `sampling`, `mcp_servers`, `mcp`, `cors_origins`, `allowed_hosts`, `reasoning_effort`,
    `parallel` and `open_browser`. It also copies `env` entries not in `SETUP_ENV`, `vision` keys not in
    `SETUP_VISION`, and a user's own `mmproj` that still exists (#625). It prints "kept from your earlier ...".
  - If the file changes, the earlier one is kept as `strata-<tag>.json.bak`.
  - Engine options added by hand to `args` are not merged. `args_dropped()` (`setup.py:2729-2736`) names the flags
    the new `args` lacks.
- `readable_config()` / `model_config()` (`setup.py:2772-2785`, `setup.py:2833-2846`) skip a cut-off or foreign
  `strata-*.json` with a warning instead of stopping (#459 #549).
- `choices_from_config(cfg_path)` (`setup.py:2799-2821`) reads the setup answers back from a config (family and size
  from the file name, context, KV, vision, speed projection, host, key, port, GPUs, `layer_split`, CUDA 12, a
  non-default `--vram-reserve-mib`) to set up a new copy the same way.

## Changes made at a start

`start()` writes into the config: `--vram-reserve-mib`, `host`, `api_key`, `draft_vocab`, `open_browser` given on the
start; `gpu` / `gpus_asked` / `layer_split` from `--gpus` or `offer_together()`; `--remote-expert-opt`;
`split_mmap`/`split_budget` edits; `exe`/`cuda`/`lib_dirs` from `ensure_engine_for`/`use_cuda12`; and
`upgrade_config()` edits ([flow.md](flow.md#start-setuppy3066-3195)). Calibration rewrites `args`
([calibration.md](calibration.md)).

## Files next to it

| File | Written by | Content |
| --- | --- | --- |
| `run-<tag>.bat` / `run-<tag>.sh` | `write_run_script()` (`setup.py:3453-3466`) | `cd` to the Strata folder, then `<venv python> serve/server.py --engine strata --config <cfg> --port <port> [--open]`. The `.bat` sets the window title `Strata <tag>` and pauses on an error. The `.sh` uses `exec` and is made executable |
| `strata-<tag>.log` | the engine (via the server) | The engine log |
| `strata-<tag>.json.bak` | `write_setup_config()` | The previous config |
| `strata-<tag>.shared-settings.json` | the server ("Use for other apps too", per INSTALL.md) | Not written by setup |
| `<settings dir>/settings.json` | `save_settings()` (`setup.py:2520-2538`) | Per user: `data_dir`, `installs` (up to 20 Strata folders), `calibration` (per hardware key). Windows `%APPDATA%\Strata\settings.json`, Linux `$XDG_CONFIG_HOME/strata/settings.json` (default `~/.config/strata/settings.json`) |
| `.venv/.strata-pip.json`, `.venv/.strata-rocm.json` | `pip_install()`, `rocm_root()` | What was pip-installed (the skip check) |

## Gotchas

- `cuda` is not in `SETUP_KEYS`. When a model that ran the CUDA 12 engine is set up again and setup now picks
  CUDA 13, the new config has no `cuda` key, so `carry_over` copies the old `"cuda": 12` back in, next to an `exe`
  in `engine/`. `config_toolkit()` then treats the model as a CUDA 12 one. This was checked with the golden test's
  `install()` harness (64 GB / RTX 5090 profile, an earlier `strata-q2_0.json` with `"cuda": 12`,
  `--setup --family qwen --model Q2_0`): the new config kept `"cuda": 12` and setup printed "kept from your earlier
  strata-q2_0.json: cuda". OLDER_GPUS.md says "Setting the model up again chooses again". No upstream test covers
  this case.
- Engine options added by hand to `args` disappear on a re-run, and are only named in the `.bak` line. Persistent
  options belong in server keys (for example `parallel`, `vram_elastic`) where those exist.
- `settings_summary()` (`setup.py:3041-3063`) prints the effective settings at each start. It leaves out values that
  look like paths and positional arguments (#564).

## Tests

`tools/test_setup_config.py` (`CarryOver`, `NoBrowser`, `VisionTokens`), `tools/test_setup_risk.py`
(`BrokenEarlierConfig`: empty configs skipped, the atomic write), `tools/test_setup_update.py` (`SettingsLine`, a
`strata-*.json` that is not a model config), and `tools/test_setup_golden.py` (the whole config, byte for byte).

## In UnbiasedStrata

Removed. There is no config file. The fork's binary takes `--model`, `--pack`, `--mtp`, `--mmproj`, `--ctx`,
`--threads`, `--host`, `--port`, `--vram-cap`, `--ram-cap` and engine options after `--` on its command line
(`frontend/main.cpp:618-645`), and builds the engine's argument list itself (`frontend/main.cpp:676-692`).

## See also

[context-kv-and-options.md](context-kv-and-options.md) · [low-ram-and-multi-gpu.md](low-ram-and-multi-gpu.md) ·
[serve](../serve/README.md) ·
[INSTALL.md "Where things are stored"](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INSTALL.md#where-things-are-stored)
