[Upstream docs](../README.md) › [Install](README.md) › Flags and environment

# setup.py flags and environment variables

## Purpose

This page lists every command-line flag of `setup.py`. The launchers pass all of their arguments through, so the same
flags work with `START-HERE.bat`, `SETUP.bat`, `UPDATE.bat`, `./setup.sh` and `./update.sh`. It also lists every
environment variable setup reads or sets. `START-HERE.bat --help` prints argparse's own list.

Files covered: [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py)
(`main()`'s parser, `setup.py:3534-3639`).

## Flags

"Saved" means the value goes into the model's `strata-<tag>.json` ([config-file.md](config-file.md)).

### What to install

| Flag | Values / default | Effect |
| --- | --- | --- |
| `--family` | `qwen`, `swift`, `coder`, `unsloth` | The model family; skips "Which model?" and makes the run a setup run |
| `--model` | `Q2_0`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `IQ1_M`, `UD-Q4_K_XL`, `UD-IQ4_XS` | The size. With `--yes` it is also consent to that size's RAM risks ([risk-prompts.md](risk-prompts.md)) |
| `--context` | integer | The context in tokens (`--max-context`) |
| `--rope-scaling` | `none`, `linear`, `yarn` | Rope method. Past 262144 the default is `yarn`, and `none` is refused there |
| `--rope-scale` | float | Rope factor (default `max(1, context/262144)`) |
| `--kv` | `int8`, `q4_0`, `k8v4` | KV precision above 8K (default `int8`) |
| `--kv-streaming` | `auto` (default), `on`, `off` | `--kv-resident 32768` from 64K when RAM allows (`on` overrides the RAM test, except k8v4 and WSL) |
| `--vision` | `yes`, `no`, `none`, `gpu`, `cpu` | Images (`yes` = encoder on the GPU) |
| `--vision-tokens N` | >= 1 | The most image tokens per picture (default 1024 GPU / 300 CPU). Saved |
| `--experimental-speed-projection` | `on`, `off`, or a GGUF path | EXPERIMENTAL control vector on layers 4-44 (qwen, coder) |
| `--draft-vocab` | `cjk` (default), `en`, `cyrillic`, `fr` | The draft head's token subset. Saved, also on a start |
| `--low-ram` | `auto` (default), `on`, `off`, `resident`, `mmap` | The low-RAM mode and its variant ([low-ram-and-multi-gpu.md](low-ram-and-multi-gpu.md)) |
| `--resident-budget-gib N` | > 0 | UD-Q4_K_XL / UD-IQ4_XS: GiB of experts kept in RAM (default `RAM - 24`) |
| `--vram-reserve-mib N` | >= 0 | VRAM left free for other programs (the engine's default is 700). Saved, also on a start |
| `--parallel N` | integer | Requests decoded together (`"parallel"`, the server's `--batch`). Written only when given |

### GPUs and backend

| Flag | Values / default | Effect |
| --- | --- | --- |
| `--backend` | `cuda`, `hip`, `sycl` | `cuda` = NVIDIA (default). `hip` = AMD (chosen automatically when no NVIDIA card is usable). `sycl` = Intel Arc, handed to `sycl/setup_intel.py` (Linux, experimental) |
| `--gpu N` | an nvidia-smi index (AMD: setup's list) | One card. With `--setup` it is saved; on a start it applies to that start only. `--gpu 0,2` is read as `--gpus 0,2` |
| `--gpus` | `"0,2"` or `"all"` | Several cards sharing one model (a layer split), the first being the main card. Saved, also on a start |
| `--layer-split` | `"18"`, `"16,32"`, default `auto` | With `--gpus`: the first layer of each later GPU (one rising number per extra GPU) |
| `--no-remote-expert-opt` | flag | Leaves out the `--remote-expert-opt` setup adds for two or more GPUs |
| `--cuda` | `12`, `13`, `auto` (default from `STRATA_CUDA`) | NVIDIA toolkit of this model's engine ([engine.md](engine.md#cuda-13-or-cuda-12)) |

### Network and start

| Flag | Values / default | Effect |
| --- | --- | --- |
| `--port N` | default: the installed config's port, else 8080 | The server port. Saved at setup. On a start it is passed for that start (#32) |
| `--host` | `127.0.0.1` (server default), `0.0.0.0` | Where the server listens. Saved, also on a start (#26 #179) |
| `--api-key` | a secret | Required from clients (recommended with `--host 0.0.0.0`). Saved, also on a start |
| `--no-browser` / `--browser` | default: open | Whether the chat page opens when ready. Saved as `open_browser`, also in the run script (#609) |

### Mode

| Flag | Effect |
| --- | --- |
| `--yes` | Take every recommended answer, no prompts. Not consent to risks by itself |
| `--setup` | Install another model or change settings instead of starting (what `SETUP.bat` passes) |
| `--no-start` | Install only. It also routes a run with configs present into setup instead of a start |
| `--update` | Packages, engine, config upgrades and draft subset for every installed model; never starts (what `UPDATE.bat` / `update.sh` pass) |
| `--check` | Only check this PC (step 1 plus the per-size verdicts), then exit |
| `--calibrate` | Tune `--pcie-frac` / `--spec-min-p` / `--pool-workers` on this PC (5-10 minutes), then start ([calibration.md](calibration.md)) |
| `--build` | Compile the engine instead of using the ready-made one. Also skips the engine update before a start |
| `--prebuilt URL\|DIR` | Where the ready-made engine is (default `STRATA_PREBUILT_URL` or the GitHub releases) |
| `--skip-build` | Hidden (`argparse.SUPPRESS`). Parsed but not read anywhere in `setup.py` |

### Folders

| Flag | Effect |
| --- | --- |
| `--data-dir DIR` | Where the model files go (default `Strata-data` next to the Strata folder). Remembered for every Strata folder of this user |
| `--models-dir DIR` | Where the GGUF files go (default `<data folder>/models`) |
| `--gguf-dir DIR` | Use GGUF files already present: a folder with every shard |

## Examples (from upstream's docs)

```
START-HERE.bat --model IQ2_XS --context 32768 --vision yes --yes
START-HERE.bat --yes --family qwen --model IQ2_XS --no-start           (AI_SETUP.md, for agents)
START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>
START-HERE.bat --gpus 0,2
./setup.sh --setup --backend hip --gpus 1,0
./setup.sh --vram-reserve-mib 3072                                       (an AMD card driving a Linux desktop)
```

## Environment variables

| Variable | Read where | Effect |
| --- | --- | --- |
| `HF_ENDPOINT` | `setup.py:76-79` | Hugging Face mirror for model files (and `mtp_fetch.py`) |
| `STRATA_PREBUILT_URL` | `setup.py:3592`, `setup.py:3441` | Default for `--prebuilt` |
| `STRATA_CUDA` | `setup.py:3588` | Default for `--cuda` |
| `STRATA_NVCC` | `setup.py:955-966` | The only nvcc considered (#601) |
| `CUDA_PATH`, `CUDA_HOME` | `setup.py:968-970` | Extra places to look for nvcc |
| `STRATA_EXPERIMENTAL_SM60=1` | `setup.py:560-580` | Admit Pascal/Volta cards (#295) |
| `STRATA_ISA_FLOOR=avx\|none` | `setup.py:450-452` | Build the older-CPU engine on any PC |
| `ROCM_PATH` | `setup.py:1776` | System ROCm root (default `/opt/rocm`) |
| `STRATA_ROCM_VERSION`, `STRATA_ROCM_INDEX` | `setup.py:1357`, `setup.py:1372` | TheRock wheel version and index |
| `APPDATA`, `XDG_CONFIG_HOME` | `setup.py:2520-2523` | Where `settings.json` lives |
| `STRATA_EXECV` | `setup.py:3190` | Linux: `os.execv` into the server instead of a child (set by the Docker image) |
| `WAYLAND_DISPLAY`, `DISPLAY` | `setup.py:3340-3343` | Linux desktop detection (AMD VRAM reserve tip) |
| `ProgramFiles(x86)` | `setup.py:989` | Where `vswhere.exe` is |

Set by setup:

| Variable | Where | For |
| --- | --- | --- |
| `STRATA_GGUF_PY=<third_party/llama.cpp/gguf-py>` | `setup.py:4174` | The pack and MTP tools' environment |
| `HIP_PLATFORM`, `HIP_COMPILER`, `HIP_RUNTIME`, `ROCM_PATH`, `HIP_PATH`, `LD_LIBRARY_PATH`, `PATH` | `setup.py:1879-1882` | The HIP compile (setup's own process) |
| `STRATA_HIPBLASLT_TUNING`, `STRATA_RESIDENT_PIN=0` | config `env` (`setup.py:4334-4338`) | The engine, via the server |

## Gotchas

- `--port` on a start is not saved. `--host` and `--api-key` on a start are saved (#179).
- `--gpu` and `--gpus` are numbered differently on AMD: setup's list, which is HIP's order on Linux. On Windows,
  setup's display-adapter order is re-mapped to HIP's after the engine is installed.
- `--kv` at a context of 8192 or less is silently ignored, because the cache stays fp16.

## Tests

`tools/test_setup_sycl.py` (the parser accepts `sycl`), `tools/test_setup_choices.py` (`VramReserve`: a negative value
is refused), `tools/test_setup_unsloth.py` (`--resident-budget-gib` needs a positive number),
`tools/test_setup_config.py` (`--no-browser`, `--vision-tokens`).

## In UnbiasedStrata

Removed. The fork's binary has its own short option list: `--model`, `--pack`, `--mtp`, `--mmproj`,
`--vision-tokens`, `--ctx`, `--threads`, `--host`, `--port`, `--vram-cap`, `--ram-cap`, and `-- <engine options>`
(`frontend/main.cpp:70-82`, `frontend/main.cpp:618-645`).

## See also

[risk-prompts.md](risk-prompts.md) · [launchers-and-chat.md](launchers-and-chat.md) ·
[INSTALL.md "Options without questions"](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INSTALL.md#options-without-questions) ·
[AI_SETUP.md step 4](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/AI_SETUP.md#4-run-setup-without-questions)
