[Upstream docs](../README.md) › [Install](README.md) › Flow

# setup.py end to end

## Purpose

This page follows `main()` (`setup.py:3533-4414`) and `start()` (`setup.py:3066-3195`) from top to bottom, the way
one run goes. The later pages cover each part in detail.

Files covered: [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py).

## The building blocks

| Function | Lines | What it does |
| --- | --- | --- |
| `say`, `step`, `ok`, `warn` | `setup.py:231-245` | Print a line; `=== Step n: title ===`; `  [ok] ...`; `  [!]  ...` |
| `fail(msg, hint)` | `setup.py:248-253` | Prints `[X] msg`, the hint and "Setup stopped. Fix the item above and run it again - everything already done is kept and skipped.", then `sys.exit(1)` |
| `ask(question, choices, default, yes)` | `setup.py:256-269` | With `yes` it returns the default and never reads input. Enter takes the default. An answer is matched case-insensitively against `choices`. EOF calls `fail` (#393) |
| `run(cmd, ...)` | `setup.py:272-281` | Prints `  > cmd`, runs it, and calls `fail` on a non-zero exit when `check` is set |
| `out(cmd)` | `setup.py:284-288` | Captured stdout with a 60 s timeout; `""` on error |
| `done(path)`, `mark(path)` | `setup.py:291-297` | The finish mark `<path>.done`, written only after a step completed; `mark` stores a timestamp or a note such as `sha256 <hex>` |

Global state: `ROOT` (the Strata folder), `WIN`, `GPU_PICK` (the main NVIDIA card) and `OLD_GPUS` (why Pascal/Volta
cards are admitted; see [engine.md](engine.md#cuda-13-or-cuda-12)).

## main(): argument parsing and early exits

1. **Parse** (`setup.py:3534-3624`). Every flag is listed in [cli-flags.md](cli-flags.md).
2. **`--backend sycl`** goes to `sycl_setup()` (`setup.py:3507-3530`), which warns that it is experimental, stops
   on Windows, and on Linux runs `sycl/setup_intel.py` with the remaining arguments minus `--backend`.
3. **Validation** (`setup.py:3627-3639`): `--resident-budget-gib` must be above 0, `--vision-tokens` at least 1 and
   `--vram-reserve-mib` at least 0. `--gpu 0,2` is rewritten as `--gpus 0,2`, and any other `--gpu` value that is not
   a number is an error.
4. **Data folder** (`setup.py:3641-3644`): `data_folder(--data-dir)` returns the data folder and the folders on other
   drives that still hold model files. `--models-dir` defaults to `<data>/models`
   ([update-and-data-folder.md](update-and-data-folder.md)).
5. **Installed models** (`setup.py:3647`): `installed_configs()` lists every `strata-*.json` in `ROOT` that is a model
   config (has `exe` and a list `args`), newest first (`setup.py:2833-2851`).
6. **`--update`** returns `update_install()` and never starts a model (`setup.py:3648-3649`).
7. **Adopt an earlier install** (`setup.py:3650-3675`). This applies when this folder has no config and none of
   `--setup/--model/--family/--check/--no-start` was given. `previous_config()` then finds the newest readable
   `strata-*.json` of another Strata folder. `choices_from_config()` reads back its family, size, context, KV, vision,
   speed projection, host, API key, port, VRAM reserve, CUDA 12 choice and GPUs, and the run continues with
   `a.yes = True`. The pack and model files are found again in the data folder, so nothing big is downloaded.
8. **Start GPUs** (`setup.py:3680`): `run_gpu = start_gpus(--gpus) or --gpu`, for starting an installed model.
9. **`--calibrate` with an installed model** (`setup.py:3682-3697`): update the engine, pick the model (asked when
   several), run `calibrate_config()`, then `start()` unless `--no-start` ([calibration.md](calibration.md)).
10. **Start an installed model** (`setup.py:3698-3713`). This happens when a config exists and none of
    `--setup/--model/--family/--check/--no-start` was given. Unless `--build` is set, `update_installed_engine()`
    runs first. One config is started right away. With several, a menu asks "Which one?", whose last entry is
    "install another model / change settings" and continues into a fresh setup.

## Step 1: checking your PC (`setup.py:3716-3854`)

- NVIDIA cards from `gpus()`, AMD cards from `amd_gpus()`. `OLD_GPUS` is set when Pascal/Volta cards are opted in.
- **Backend choice**: HIP when `--backend hip` is given, or when no NVIDIA card is usable but an AMD one is. A PC
  with both asks "Which cards?" (1 = NVIDIA, recommended).
- **AMD path** (`setup.py:3741-3770`): lists the cards. It stops if none is supported, and stops on `--gpus` under
  Windows. It picks `--gpus`, `--gpu` or the card with the most VRAM, then records `gpu["archs"]` = every chosen
  card's arch.
- **NVIDIA path** (`setup.py:3771-3796`): stops if `nvidia-smi` gives nothing. It prints the GPU table when there are
  several cards or one unusable card, then `choose_gpus()` (may ask). Next it sets `GPU_PICK`, `gpu["archs"]` and the
  CUDA toolkit (`cuda_choice`), and checks the driver (580, or 528 Windows / 525 Linux for CUDA 12).
- Under 11 GB of VRAM: a warning that it will be slow.
- **RAM** (`setup.py:3799-3816`): below the smallest model's need minus 4 GB (32 - 4 = 28 GB), and with the low-RAM
  mode unable to help, a `confirm_risk` follows ([risk-prompts.md](risk-prompts.md)).
- **Windows page file** under 4 GB: a warning (#60).
- **CPU** (`setup.py:3822-3837`): the name and AVX2/AVX-512. `cpu_floor()`: "unsupported" stops; no AVX2 warns and
  sets `a.build = True` (an older-CPU engine is compiled; stops for HIP on Windows).
- **`--check`** (`setup.py:3838-3854`) prints a verdict per model and exits 0
  ([model-choice.md](model-choice.md#the---check-verdicts)).

Details: [hardware-detection.md](hardware-detection.md).

## Step 2: your choices (`setup.py:3857-4088`)

In order:

1. **Family** (question "Which model?", default 1 = `qwen`), and the license line for Swift 1.5.
2. **Size** ("Which size?"), recommended by RAM. A `--model` that the family does not have stops with the families
   that have it ([model-choice.md](model-choice.md)).
3. **Budget models** (Unsloth `UD-*`): experimental warning, the AMD question, the RAM question, `budget_choice()`,
   and `unsloth_together()` when several GPUs were chosen.
4. **Low-RAM mode** decision, and `low_ram_together()` when several GPUs were chosen. Otherwise `confirm_paging()`
   when RAM is short ([low-ram-and-multi-gpu.md](low-ram-and-multi-gpu.md)).
5. **Context** ("Context?"), then **rope scaling** past 262144 ("RoPE extension method?" when interactive).
6. **KV cache** ("KV cache?", only above 8K and without `--kv`/`--yes`).
7. **Images** ("Do you want images?", default n), forced off for UD-Q4_K_XL, CPU-only (Linux) for HIP.
8. **Low-RAM variant**: resident or mapped, and the GPU's share of the experts.
9. **Experimental speed projection** (qwen and coder only; "Turn on ...?", default n).
10. **Model files location** (`--gguf-dir`, the data folder, or another drive's Strata folder), whole-file marks, and
    the **free disk** check ([downloads-and-packs.md](downloads-and-packs.md)).

## Step 3: Python packages (`setup.py:4091-4093`)

`pip_install(requirement_lines())` installs `requirements.txt` into `.venv` (falling back to the bare `PY_PACKAGES`
list when the file is missing). The step is skipped when `.venv/.strata-pip.json` already lists the same pins
([launchers-and-chat.md](launchers-and-chat.md)).

## Step 4: the engine (`setup.py:4096-4129`)

`get_llama_cpp()` fetches llama.cpp at the pinned commit into `third_party/llama.cpp`. Then:

- Windows HIP: `get_prebuilt_hip()` (it stops if none is published), then `hip_card()` re-numbers the card as HIP
  numbers it.
- NVIDIA without `--build`: `get_prebuilt()`, for CUDA 13 or 12. When a ready-made engine is used, `pip_cuda_libs()`
  installs cuBLAS and the runtime, and `prebuilt_vision()` may move images to the CPU. A missing encoder forces a
  compile.
- Otherwise: `build_engine_hip()` (Linux AMD) or `build_engine()` (NVIDIA).
- `lib_dirs` comes from `BUILD.json`. A budget model whose engine is too old stops here, before the 94-111 GB
  download.

Details: [engine.md](engine.md).

## Step 5: downloading the model (`setup.py:4132-4169`)

The step prints where the files go (and the `HF_ENDPOINT` mirror). It downloads each missing shard (the original's
shard 2 is hard-linked from another size when possible), runs `check_shards()` and the Unsloth SHA-256 checks, and
fetches the vision encoder (`mmproj`) when images are on ([downloads-and-packs.md](downloads-and-packs.md)).

## Step 6: preparing the model (`setup.py:4172-4214`)

The pack goes to `<data>/packs/<tag>`, built with `strata_pack.py` + `pack_index.py` + `strata_tokenizer.py` (Q2_0 on
an AVX-512 CPU) or `iq_pack.py`. Low-RAM adds `iq_pack.py --experts-bin`. The MTP draft layer goes to `<data>/mtp/rt`
(`mtp_fetch.py`, `mtp_pack.py`, `mtp_rt.py`), rebuilt if `mtp_fetch.py verify` reports corrupt tensors. Then the
draft-vocabulary subset is copied, with a small-card tip.

## Step 7: the run config and start script (`setup.py:4217-4393`)

`args` is assembled ([config-file.md](config-file.md)) along with the rest of `cfg`. A saved calibration is applied,
or else the hybrid-CPU `--pool-workers` rule. `write_setup_config()` writes `strata-<tag>.json` (keeping user keys
and a `.bak`), and `write_run_script()` writes `run-<tag>.bat`/`.sh`. An interactive NVIDIA install without `--yes`
or `--no-start`, and with no saved calibration, asks "Tune Strata for this PC now?". The run ends with "All set." and the API addresses
(`http://127.0.0.1:<port>/v1` and `/v1/messages`), plus the `parallel` note. Without `--no-start` it then calls
`start(cfg_path, port)`.

## start() (`setup.py:3066-3195`)

1. `upgrade_config()` rewrites an old `--prefill 2048`, and drops `--kv-resident` under WSL
   ([update-and-data-folder.md](update-and-data-folder.md#config-upgrades-on-every-start)).
2. If the engine exe or any `.gguf` argument is missing, it stops: "run it again with --setup to repair".
3. Settings given on this start and kept from now on: `--vram-reserve-mib` (written into `args`), `--host`,
   `--api-key`, `--draft-vocab`, and `--no-browser`/`--browser` (`open_browser`) (#179 #493 #609).
4. Touches the config (it becomes the "most recently used"), and copies the draft-vocabulary subset again.
5. Builds `[python, serve/server.py, --engine, strata, --config, <cfg>, --port, <port>]`.
6. GPUs:
   - HIP: `--gpus` is saved into the config (Linux; checks that the engine has code for each card), and `--gpu N`
     becomes the server's `--gpu`.
   - NVIDIA: `--gpus` is checked and saved, with `--remote-expert-opt` added. `--gpu N` applies to this start only.
     With neither, `offer_together()` asks once.
   - For a list of cards: `split_mmap()` / `split_budget()` adjust the args, and `ensure_engine_for()` compiles for
     new archs or moves the model to the CUDA 12 engine.
7. `--open` is added unless the config says `"open_browser": false`.
8. Prints the banner: "it loads N GB into RAM ... YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES". The
   mapped low-RAM mode gets a different first line (#505). After it comes the `Settings (...)` line from
   `settings_summary()`, which lists the config's engine flags without paths, plus the server fields (#564).
9. On Linux with `STRATA_EXECV` set (the Docker image), `os.execv` replaces setup with the server. Otherwise
   `subprocess.call` runs it, and the exit code is setup's.

## Callers and dependencies

- Called by: the launchers ([launchers-and-chat.md](launchers-and-chat.md)), `docker-entrypoint.sh`,
  `sycl/setup_intel.py` (imports `setup` and runs `main()`), `tools/strata_mcp.py` (imports its tables, runs it as a
  subprocess), and the tests.
- Calls: `tools/gguf_reader.py`, `tools/iq_pack.py`, `strata_pack.py`, `pack_index.py`, `strata_tokenizer.py`,
  `mtp_fetch.py`, `mtp_pack.py`, `mtp_rt.py`, `calibrate.py` ([tools](../tools/README.md)). It also calls CMake and
  Ninja on the root `CMakeLists.txt` and `tools/vision` ([build](../build/README.md)), and finally
  `serve/server.py` ([serve](../serve/README.md)).

## Gotchas

- The module docstring (`setup.py:1-37`) is older than the code. It speaks of "four questions" and
  `--family qwen|swift`, while `main()` also offers `coder` and `unsloth`, the KV and speed-projection questions,
  and more flags.
- `explicit` (`setup.py:3650`) does not include `--update`, `--calibrate` or `--gpus`. A bare
  `START-HERE.bat --gpus 0,1` in a fresh copy with an earlier install elsewhere adopts that install, with its own
  `--gpus` taking priority.

## Tests

`tools/test_setup_golden.py` runs `main()` end to end on mocked PCs. Most other `test_setup_*` files run `main()`
through its `install()` harness ([tests.md](tests.md)).

## In UnbiasedStrata

Removed (commit `8894a81`). There is no install flow: `make build` builds the binary, and the pack and draft head
are made by hand (see the fork's `UPSTREAM.md`).

## See also

[hardware-detection.md](hardware-detection.md) · [config-file.md](config-file.md) · [risk-prompts.md](risk-prompts.md) ·
[INSTALL.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INSTALL.md)
