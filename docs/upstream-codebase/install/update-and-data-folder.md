[Upstream docs](../README.md) › [Install](README.md) › Update and data folder

# Updating, config upgrades, and the shared data folder

## Purpose

This page covers how an existing install picks up a new version: `UPDATE.bat` / `update.sh` (`setup.py --update`),
the engine check that runs before every start, and the config edits a start makes. It also covers how the model
files live outside the Strata folder so that a fresh copy of Strata finds and reuses them.

Files covered: [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py)
(`setup.py:2033-2121`, `setup.py:2335-2340`, `setup.py:2520-2830`, `setup.py:2987-3038`, `setup.py:3650-3675`),
[UPDATE.bat](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/UPDATE.bat),
[update.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/update.sh). Background:
[INSTALL.md "Updating"](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INSTALL.md#updating).

## `--update` (#475)

The launchers (`UPDATE.bat:10-36`, `update.sh:8-31`) work like this. In a git clone they run `git pull --ff-only`,
and stop if git is missing or the pull fails ("Files you changed here can stop it"). In a zip copy they print the zip
URL and say to unzip it anywhere: "it finds the model files in Strata-data". Either way they then run setup with
`--update` and the user's arguments. Both wrap everything in one block or function that is read before it runs,
because `git pull` can change the script itself.

`update_install(have, a)` (`setup.py:3012-3038`) asks nothing, never touches the model files and never starts a
model:

1. It skips any `strata-*.json` that is not a model config (#549). If none is left, it says to run
   `START-HERE.bat` / `./setup.sh` and returns 0.
2. `pip_install(requirements.txt)`.
3. Unless `--build` is given, `update_installed_engine(a.prebuilt)` runs (next section).
4. For each config: `upgrade_config()`, `refresh_draft_vocab()` for its `--mtp` folder and saved `draft_vocab`, and on
   Windows HIP `hip_runtime_beside_exe()`. It prints "<model>: up to date".
5. It finishes with "Strata is updated (engine X.Y.Z). Start the model with START-HERE.bat ...".

## The engine update on every start

`update_installed_engine(url_base, toolkit=None)` (`setup.py:2033-2121`) runs before every start (unless `--build`
is given), before `--calibrate`, and in `--update`. It checks `engine/`, and then `engine-cuda12/` if that has a
`BUILD.json`.

| Installed engine | Action |
| --- | --- |
| Windows HIP, ready-made, older than `WIN_HIP_MIN_ENGINE` | `get_prebuilt_hip(updating=True)` for the first supported AMD card. On failure it warns "starting the installed one" |
| HIP compiled here, `src` hash changed | `build_engine_hip()` for every arch it was built for, keeping its vision setting |
| CUDA compiled here (`source: local`), `src` or `vision_src` changed (a `git pull`) | `build_engine()` for its archs plus the main card's. A CUDA 12 build keeps its own archs |
| Ready-made, version >= `MIN_ENGINE` (0.1.39) | nothing |
| Ready-made, older | `get_prebuilt(updating=True)` (a CUDA 12 engine asks for its own old archs), then `pip_cuda_libs()` |

Rules that protect a working install:

- A running engine cannot be replaced. Setup opens the exe for writing to check; if that fails it prints "engine ... is
  in use: close the model window and run this again to update it" and returns (`setup.py:2087-2094`).
- A failed compile or download never stops the start. It warns, and the installed engine starts (#49). If
  `get_prebuilt` had deleted `BUILD.json` before downloading, the old text is written back.
- An engine version that is not published yet is reported as "the release may still be uploading" (#58).

## Config upgrades on every start

`upgrade_config(cfg_path, cfg)` (`setup.py:2987-3009`) runs in `start()` and `--update`:

- `--prefill 2048` with an engine of 0.1.13 or newer becomes `--prefill auto` ("about 2x faster on long prompts").
  The reverse, `auto` back to `2048`, happens when the engine is older than 0.1.13, for example an old engine kept
  after a failed update.
- Under WSL, `--kv-resident N` is removed, because the driver pins only about 1 GB there.

Other changes made at a start are listed in [config-file.md](config-file.md#changes-made-at-a-start).

## The data folder

The model files (GGUFs, packs and the MTP layer; 70-120 GB) live in a data folder next to the Strata folder,
`Strata-data`, and not inside it. The comment explains why: "updating Strata by unzipping a new copy used to give a
new, empty folder and a full download again" (`setup.py:2335-2340`).

`data_folder(requested)` (`setup.py:2630-2681`) runs at the top of every `main()`:

1. Destination: `--data-dir`, else `settings.json`'s `data_dir`, else `ROOT.parent / "Strata-data"`. If that folder
   cannot be created, `ROOT` is used with a warning.
2. Sources: this folder, every earlier install (`other_installs()`: the `installs` list in `settings.json` plus
   `Strata*` folders next to this one and one level down, each holding a `setup.py`), the previously remembered data
   folder, and a `Strata-data` nested in any of these (#198).
3. A source on the same drive (`same_drive()` compares `st_dev`) has its `models/` merged file by file
   (`move_into()`: a rename, so instant on one drive). Each pack and the `mtp/` folder move whole or not at all, so
   two copies are never mixed. Its `strata-*.json` files are re-pointed to the new place (`repoint_config()`,
   `setup.py:2602-2627`; on Windows paths compare case-insensitively).
4. A source on another drive is used where it is, with no 70 GB copy. Such folders are returned as `elsewhere`
   ("roots"), and steps 5-6 look there for shards, the `mmproj`, packs and the MTP layer (`find_in()`).
5. `settings.json` is saved with `data_dir` and `installs` (this folder first, up to 20).

`settings.json` (`settings_path()`, `setup.py:2520-2523`) is `%APPDATA%\Strata\settings.json` on Windows, and
`$XDG_CONFIG_HOME/strata/settings.json` or `~/.config/strata/settings.json` on Linux. It also holds the saved
calibrations ([calibration.md](calibration.md)).

## Adopting an earlier install in a new copy

When a fresh copy has no `strata-*.json` and the run names no setup flag, `main()` sets the copy up like the most
recent earlier install (`setup.py:3650-3675`):

- `previous_config()` (`setup.py:2788-2796`) picks the newest readable `strata-*.json` among `elsewhere` and the
  earlier installs. A cut-off or empty one is skipped with a warning (#459).
- `choices_from_config()` (`setup.py:2799-2821`) recovers the family, size, context, KV, vision, speed projection,
  host, API key, port, an explicit VRAM reserve (#493), the CUDA 12 choice, and the GPUs with their layer split.
- Setup prints "Found your earlier install in ... setting up this copy the same way - the model files are reused,
  nothing big is downloaded" and continues with `--yes`. The new config also inherits the earlier config's user keys
  (`write_setup_config(..., source=adopted)`, [config-file.md](config-file.md#writing-and-re-writing-459-629)).

## Gotchas

- `--update` does not migrate model files or change sizes. It only refreshes the program, the engine and the configs.
- On Linux a `git pull` changes the source hash, so the next start compiles a locally built engine again ("a minute or
  two for the changed files", per INSTALL.md). A ready-made engine is replaced only when `MIN_ENGINE` in `setup.py`
  moves up.
- `other_installs()` treats any `Strata*` folder with a `setup.py` near this one as an earlier install. A copy of a
  different fork in a sibling folder would be scanned too.

## Tests

`tools/test_setup_update.py` (`--update` refreshes without starting, `--build` keeps a compiled engine path, nothing
installed, foreign `strata-*.json` skipped, `main --update` never starts), `tools/test_setup_risk.py`
(`BrokenEarlierConfig`, `Context.test_the_earlier_install_keeps_its_context`), `tools/test_setup_older_gpus.py`
(`Start.test_update_looks_at_both_engines`), and `tools/test_setup_pins.py` (`Engine.test_an_installed_engine_is_kept`).

## In UnbiasedStrata

Removed. The fork has no updater: a new version is built with `make build`, which asks for
`git submodule update --init` when `third_party/llama.cpp` is empty. Model files are wherever the user put them and
are passed by path. There is no `settings.json` and no data-folder migration.

## See also

[engine.md](engine.md) · [launchers-and-chat.md](launchers-and-chat.md) · [config-file.md](config-file.md)
