[Upstream docs](../README.md) › Install

# The one-click installer and launchers

Upstream Strata has a one-click installer. It is a single Python program, `setup.py` (4,422 lines). Thin launchers
start it: `START-HERE.bat`, `SETUP.bat` and `UPDATE.bat` on Windows, and `setup.sh` and `update.sh` on Linux. On a
first run it checks the PC (GPUs, driver, RAM, CPU, disk) and asks a few questions (model, size, context, KV cache,
images). It then installs the Python packages into `.venv/` and gets the engine, either a ready-made zip from the
GitHub release or a local compile with CMake. Next it downloads the model's GGUF shards from Hugging Face at pinned
revisions and runs the pack tools in `tools/` to prepare the pack and the MTP draft head. Last, it writes a run config
`strata-<model>.json` and a start script `run-<model>.bat`/`.sh`. It then starts the Python server
(`serve/server.py --engine strata --config strata-<model>.json`), and the server starts the engine with the config's
`args` ([serve](../serve/README.md)). Every later run skips straight to that start, after a quick engine-update check.
The fork (UnbiasedStrata) deleted all of this.

## Contents

| Page | What it covers |
| --- | --- |
| [flow.md](flow.md) | `main()` end to end: step 0 (start an installed model) through step 7 (config, start script, start), and `start()` |
| [hardware-detection.md](hardware-detection.md) | NVIDIA (`nvidia-smi`), AMD (KFD sysfs, Windows WMI and registry, `strata-device --list-devices`), Intel (handoff), RAM, page file, CPU features, hybrid cores, disk, WSL |
| [model-choice.md](model-choice.md) | The families and sizes setup offers, their numbers, the recommended size by RAM, `--check` verdicts, Unsloth budget models |
| [context-kv-and-options.md](context-kv-and-options.md) | Context, rope scaling, KV precision, KV streaming, prefill, images, draft vocabulary, VRAM reserve, parallel slots, speed projection, `--ple-io ram`, pool workers |
| [low-ram-and-multi-gpu.md](low-ram-and-multi-gpu.md) | The low-RAM mode (resident and mapped variants), the Unsloth RAM budget, choosing several GPUs (`--gpus`), what changes at start |
| [engine.md](engine.md) | Ready-made engine vs local compile: CUDA 13 / CUDA 12, HIP on Windows and Linux, older CPUs, `BUILD.json`, build tools, the image encoder |
| [downloads-and-packs.md](downloads-and-packs.md) | Resumable downloads, pinned revisions, `HF_ENDPOINT`, SHA-256 checks, `--gguf-dir`, disk need, building the pack and the MTP draft head |
| [config-file.md](config-file.md) | Every key of `strata-<model>.json`, what engine flag or server setting each becomes, carry-over on a re-run, the run script, `settings.json` |
| [risk-prompts.md](risk-prompts.md) | "Setup recommends, it never forces": `ask`, `confirm_risk`, every question, warning and hard stop |
| [cli-flags.md](cli-flags.md) | Every `setup.py` flag and environment variable |
| [update-and-data-folder.md](update-and-data-folder.md) | `--update` / `UPDATE.bat`, the engine update on each start, config upgrades, the shared `Strata-data` folder, adopting an earlier install |
| [calibration.md](calibration.md) | `--calibrate` (measured `--pcie-frac`, `--spec-min-p`, `--pool-workers`), and the hybrid-CPU worker rule |
| [launchers-and-chat.md](launchers-and-chat.md) | The `.bat` / `.sh` launchers, `requirements.txt`, `chat.py`, the generated run scripts, Docker and other programs that drive `setup.py` |
| [tests.md](tests.md) | `tools/test_setup_*.py` and `test_setup_golden.json`: what each checks and how to run them |

## Source map

All paths are relative to the upstream root at commit `6f32ec0`.

| File | Lines | Role |
| --- | ---: | --- |
| [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py) | 4422 | The installer and starter: checks, questions, downloads, engine, packs, config, start, update |
| [START-HERE.bat](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/START-HERE.bat) | 52 | Windows: finds or installs Python 3.10+, makes `.venv`, runs `setup.py %*` |
| [SETUP.bat](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/SETUP.bat) | 3 | `START-HERE.bat --setup %*` |
| [UPDATE.bat](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/UPDATE.bat) | 36 | `git pull --ff-only` (in a clone), then `START-HERE.bat --update %*` |
| [setup.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.sh) | 39 | Linux: finds or installs Python 3.10+ with venv and ensurepip, makes `.venv`, `exec`s `setup.py "$@"` |
| [update.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/update.sh) | 31 | `git pull --ff-only` (in a clone), then `setup.sh --update "$@"` |
| [requirements.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/requirements.txt) | 23 | The pinned Python packages setup installs into `.venv` (#214) |
| [chat.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/chat.py) | 116 | A terminal chat client for a running server (standard library only) |
| [tools/test_setup_amd.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_amd.py) | 454 | Tests: AMD detection, HIP numbering, ROCm choice, Windows HIP engine |
| [tools/test_setup_choices.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_choices.py) | 441 | Tests: encoder fallback, `--gguf-dir`, Pascal/Volta gate, HIP vision, rotational disk, VRAM reserve, tips |
| [tools/test_setup_config.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_config.py) | 244 | Tests: config carry-over (#629), `--no-browser` (#609), `--vision-tokens` (#625) |
| [tools/test_setup_draft_vocab.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_draft_vocab.py) | 109 | Tests: the draft subset choice and copy |
| [tools/test_setup_golden.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_golden.py) | 210 | Tests: `main()` on five mocked PCs writes the baseline configs byte for byte; also the shared `install()` harness |
| [tools/test_setup_golden.json](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_golden.json) | 968 | The baseline: 25 PC x model cases, exit code and normalized config |
| [tools/test_setup_hybrid.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_hybrid.py) | 65 | Tests: `--pool-workers` on hybrid CPUs (#642) |
| [tools/test_setup_lowram.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_lowram.py) | 84 | Tests: low-RAM messages and the paging question (#250) |
| [tools/test_setup_oldcpu.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_oldcpu.py) | 95 | Tests: the older-CPU ISA floor build (#394 #595 #623) |
| [tools/test_setup_older_gpus.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_older_gpus.py) | 287 | Tests: the CUDA 12 engine for Pascal / Volta |
| [tools/test_setup_parallel.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_parallel.py) | 53 | Tests: the `parallel` recommendation (#465) |
| [tools/test_setup_pins.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_pins.py) | 280 | Tests: pinned Hugging Face revisions, release engine first, pinned requirements (#214) |
| [tools/test_setup_prompts.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_prompts.py) | 89 | Tests: EOF never consents (#393) |
| [tools/test_setup_remote_opt.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_remote_opt.py) | 74 | Tests: `--remote-expert-opt` on two or more GPUs (#578) |
| [tools/test_setup_risk.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_risk.py) | 414 | Tests: "recommend, never force" (#403 #406 #364 #384) |
| [tools/test_setup_rope.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_rope.py) | 121 | Tests: rope scaling resolution |
| [tools/test_setup_sycl.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_sycl.py) | 75 | Tests: `--backend sycl` handoff |
| [tools/test_setup_unsloth.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_unsloth.py) | 577 | Tests: UD-Q4_K_XL and UD-IQ4_XS (pins, budget, SHA-256, split) |
| [tools/test_setup_update.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_update.py) | 145 | Tests: `--update` (#475) and the settings line (#564) |

Related files that call `setup.py` and belong to other areas: `Dockerfile` (115 lines) and `docker-entrypoint.sh`
(65 lines) build and run it in a container ([launchers-and-chat.md](launchers-and-chat.md)).
`sycl/setup_intel.py` imports it for Intel Arc. `tools/strata_mcp.py` imports its tables and runs it for AI
assistants. `tools/calibrate.py` is the tuning that `--calibrate` runs ([tools](../tools/README.md)).

The user-facing background is in upstream's prose docs:
[INSTALL.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INSTALL.md),
[AI_SETUP.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/AI_SETUP.md),
[MODELS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MODELS.md),
[OLDER_GPUS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/OLDER_GPUS.md),
[UNSLOTH_Q4.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/UNSLOTH_Q4.md),
and the "Before you start" / "Windows" / "Linux" sections of [DETAILS.md](../../upstream/DETAILS.md).

## Key concepts and data flow

```
 START-HERE.bat / setup.sh          UPDATE.bat / update.sh (git pull first)
        |  (Python 3.10+, .venv)            |
        v                                   v
   setup.py main() ----------------- --update: packages, engine, config upgrades, draft subset; never starts
        |
        |-- a model is installed (strata-*.json) and no --setup/--model/--family/--check/--no-start?
        |       yes: update_installed_engine() -> start(cfg)  ------------------------------+
        |                                                                                   |
        |-- 1 check PC: gpus() / amd_gpus(), ram_gb(), cpu_info(), cpu_floor(), page file   |
        |-- 2 questions: family, size, context (+rope), KV, images, speed projection        |
        |        low-RAM mode / RAM budget / GPUs, disk space                              |
        |-- 3 pip install -r requirements.txt  (into .venv)                                |
        |-- 4 engine: get_prebuilt() zip from GitHub release | build_engine() via CMake     |
        |        (CUDA 13 engine/, CUDA 12 engine-cuda12/, HIP; BUILD.json)                 |
        |-- 5 download GGUF shards (Hugging Face, pinned revision, resumable, .done marks)  |
        |-- 6 pack: tools/iq_pack.py | strata_pack.py + pack_index.py + strata_tokenizer.py |
        |        MTP: tools/mtp_fetch.py -> mtp_pack.py -> mtp_rt.py; draft_vocab.bin       |
        |-- 7 write strata-<tag>.json + run-<tag>.bat/.sh  (optional calibration)           |
        v                                                                                   |
   start(cfg): python serve/server.py --engine strata --config strata-<tag>.json --port N <-+
        |
        v
   serve/server.py: engine_args(cfg) (+ --layer-split, --batch), child_env(cfg)
        (CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES, lib_dirs, env) -> engine process
```

- **Run config** (`strata-<tag>.json`): the only interface between the installer and the server. `args` holds the
  engine's command line. The other keys are read by the server or by setup itself. See [config-file.md](config-file.md).
- **Tag**: the family's tag plus the size: `iq3_xxs`, `swift-iq2_xs`, `coder-iq1_m`, `unsloth-ud-iq4_xs`. It names the
  pack folder, the config, the log and the run script (`setup.py:3936`, `setup.py:4375`).
- **Finish marks**: `<file>.done` beside a download or step. A step with its mark is skipped, so an interrupted
  setup can be run again (`setup.py:291-297`).
- **Data folder**: `Strata-data` next to the Strata folder holds `models/`, `packs/` and `mtp/` (70-120 GB). Every
  Strata copy on the PC finds it through a per-user `settings.json`
  ([update-and-data-folder.md](update-and-data-folder.md)).
- **"Setup recommends, it never forces"**: a choice bigger than the recommendation is kept, with a warning. A choice
  setup expects to fail is asked about, defaulting to "n". With `--yes`, an explicit flag counts as consent and
  `--yes` alone does not ([risk-prompts.md](risk-prompts.md)).
- For terms like expert cache, pack, MTP draft head and KV streaming, see the [glossary](../guides/glossary.md).

## In UnbiasedStrata

The fork removed this whole area in commit `8894a81` ("Remove installer, Python, non-target platforms, benchmarks and
marketing"). That commit deleted `setup.py`, every `.bat`/`.sh` launcher, `requirements.txt`, `chat.py`, the Docker
files and `tools/`, including all `test_setup_*` files. Here is what replaces each piece:

| Upstream piece | In the fork |
| --- | --- |
| Engine download / compile (`get_prebuilt`, `build_engine`) | `make build`: one CMake build of the binary `unbiased-strata` for NVIDIA archs `61;70;75;86;89;120`, with ggml from the `third_party/llama.cpp` submodule, nothing downloaded ([makefile](../../codebase/build/makefile.md)) |
| Questions and the run config's `args` | Fixed flags in the C++ front end, `frontend/main.cpp:676-692`: `--pack`, `--native`, `--expert-profile <pack>/expert-profile.bin`, `--expert-cache auto`, `--prefill auto`, `--spec 4`, `--spec-min-p 0.5`, `--mtp`, `--max-context <--ctx>`, `--ple-gguf <shard 2> --ple-io ram`, `--kv int8` above 8K, `--vision` with `--mmproj`, `--pool-workers` from `--threads` (or half a container's CPU quota), plus `--vram-cap` / `--ram-cap` and anything after `--` ([engine process](../../codebase/frontend/engine-process.md)) |
| Model download, pack and draft head (steps 5-6) | Done by hand, once, with upstream's tools at `6f32ec0`: the `iq_pack.py` / `mtp_fetch.py` / `mtp_pack.py` / `mtp_rt.py` commands listed in the fork's `UPSTREAM.md` ("Prepared model files") |
| Python server started by `start()` | The C++ front end in the same binary starts the engine as a child process |
| Multi-GPU, AMD, Windows, low-RAM questions | Not offered: the front end refuses `--gpus`, `--layer-split`, `--peer-device`, `--remote-expert*`, `--batch` after `--` (`frontend/main.cpp:629-640`) |
| `chat.py`, `UPDATE.bat`, calibration, tests | No equivalent (the fork's own tests are `make test-frontend` and friends) |
