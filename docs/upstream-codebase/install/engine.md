[Upstream docs](../README.md) › [Install](README.md) › Engine

# Getting the engine: ready-made or compiled here

## Purpose

This page covers step 4 of setup and the engine half of every start: deciding which engine a model runs (CUDA 13,
the experimental CUDA 12, or HIP), downloading a ready-made build or compiling one with CMake, recording what was
installed in `BUILD.json`, and compiling again when the source or the cards change.

Files covered: [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py)
(`setup.py:93-118`, `setup.py:557-627`, `setup.py:951-1010`, `setup.py:1254-1343`, `setup.py:1345-1926`,
`setup.py:1930-2332`, `setup.py:3388-3450`, `setup.py:4096-4129`). The CMake options themselves are covered in
[build](../build/README.md). Background:
[OLDER_GPUS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/OLDER_GPUS.md),
[AMD_HIP.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/AMD_HIP.md).

## Engine folders and BUILD.json

| Folder | Holds |
| --- | --- |
| `engine/` | The CUDA 13 engine (`strata(.exe)`, `strata-vision(.exe)`) or the HIP engine (plus `strata-device(.exe)` and `rocm/bin` on Windows) |
| `engine-cuda12/` (`ENGINE12_DIR`) | The experimental CUDA 12 engine for Pascal / Volta |
| `third_party/llama.cpp` | llama.cpp at `LLAMA_CPP_COMMIT = 3cf03257...`, from a GitHub zip (no git). The `tools/ui/` tree is left out because of Windows' 260-character path limit (#206) (`get_llama_cpp()`, `setup.py:1254-1284`) |
| `build/`, `build-vision/`, `build-cuda12/`, `build-vision-cuda12/`, `build-hip/` | CMake + Ninja build trees |

`engine/BUILD.json` records what is installed and decides whether it is current:

| Key | Ready-made (from the zip) | Compiled here (`build_engine`, `setup.py:2327-2330`) | HIP compiled here (`setup.py:1895-1896`) |
| --- | --- | --- | --- |
| `source` | not `local` (`prebuilt` for the Windows HIP zip) | `local` | `local-hip` |
| `version` | the release's | `source_version()`, from `CMakeLists.txt` (`project(strata VERSION 0.1.39)`) | same |
| `archs` | compute capabilities (`[75, 86, 89, 120]`) or gfx names | the cards compiled for | gfx names |
| `ptx`, `vision_archs`, `cuda` | PTX included; the encoder's archs; CUDA version | no | no |
| `backend` | `hip` for HIP zips | (absent) | `hip` |
| `lib_dirs` / `cuda_dirs` | HIP: `rocm/bin` | `cuda_dirs`: the toolkit's `bin`, `bin/x64`, `lib64` | `lib_dirs`: the ROCm library folders |
| `src`, `vision_src` | no | `source_hash()` of the engine and encoder sources | `src` (+ `vision_src` with the CPU encoder) |
| `vision`, `toolkit`, `isa_floor` | no | `gpu`/`cpu`/`none`; `12` for the CUDA 12 build; `avx`/`none` for an older CPU | `vision`, `isa_floor` |
| `rocm`, `hipblaslt_version` | Windows HIP zip | no | no |

`source_hash(parts)` (`setup.py:2235-2243`) is the first 16 hex digits of a SHA-256 over `LLAMA_CPP_COMMIT` and
every file under `CMakeLists.txt`, `src`, `include` and `third_party/ggml` (`ENGINE_SOURCES`), or under
`tools/vision` (`VISION_SOURCES`). CRLF is normalized to LF. When a `git pull` changes it, a locally compiled engine
is compiled again (#31).

`engine_version(exe)` (`setup.py:2860-2875`) reads `BUILD.json`'s version, or else searches the binary for
`engine=X.Y.Z\n`, which 0.1.13 and newer carry. Without either it assumes 0.1.12.

## CUDA 13 or CUDA 12

| Name | Lines | Rule |
| --- | --- | --- |
| `cuda_choice(archs, cuda)` | `setup.py:601-616` | `--cuda 13` gives 13 (warning that it cannot run on a card below sm_75). `--cuda 12` gives 12 (a note on RTX 50 + CUDA 12.8 crashes, #220). Otherwise any card below `CUDA13_MIN_ARCH = 75` gives 12, else 13. One engine runs per model, so its oldest card decides |
| `old_gpus_opt_in(found, named, cuda, other)` | `setup.py:571-588` | Why Pascal/Volta (sm 60-70) cards may be used in this run: `STRATA_EXPERIMENTAL_SM60=1`, `--cuda 12`, the user named such a card with `--gpu`/`--gpus`, or the PC has no newer NVIDIA card and no supported AMD card. Otherwise `None`, and those cards are "not supported" |
| `experimental_sm60()` | `setup.py:560-564` | `STRATA_EXPERIMENTAL_SM60=1` or `OLD_GPUS` set |
| `engine_dir(toolkit)`, `config_toolkit(cfg)` | `setup.py:619-626` | `engine/` or `engine-cuda12/`. A config runs CUDA 12 when it has `"cuda": 12` or its exe is in `engine-cuda12/` |
| `engine_defs(archs, toolkit)` | `setup.py:2254-2257` | `-DSTRATA_EXPERIMENTAL_SM60=ON` for any arch below 75 or for every CUDA 12 build |

Driver minimums: `MIN_DRIVER = 580` for CUDA 13, `CUDA12_MIN_DRIVER = 528` (Windows) / `525` (Linux) for CUDA 12.
The comment notes that drivers this old were not tested.

## The ready-made engine (NVIDIA)

`get_prebuilt(url_base, gpu, vision, updating, toolkit)` (`setup.py:1946-2030`):

1. An installed non-HIP engine is kept when it was not compiled here, has code for every chosen card (or PTX for a
   newer one) and is at least `MIN_ENGINE = 0.1.39`. A compiled-here engine returns `None`, so `build_engine()`
   checks it. A card without code means "compiling instead" (#128). An older version is replaced.
2. The URLs come from `prebuilt_bases()` (`setup.py:1937-1943`). With the default `--prebuilt`, the release of this
   checkout's version (`https://github.com/Niko1221/Strata/releases/download/v<version>/`) is tried first and
   `.../releases/latest/download/` second (#214). An explicit `--prebuilt` URL or local folder (or
   `STRATA_PREBUILT_URL`) is the only base.
3. The asset is `strata-windows-x64.zip` / `strata-linux-x64.zip`, or `...-cuda12.zip` for the CUDA 12 engine. A
   HEAD request checks it exists, `download()` fetches it, and it is unpacked to `engine/_unpack`.
4. It is refused (deleted with its mark, so it is not reused, PR #324) if its version is below `MIN_ENGINE` or it
   has no code for a card. Otherwise its files replace the folder's, and on Linux the binaries are made executable.
5. Then `pip_cuda_libs(toolkit)` (`setup.py:2124-2129`) installs NVIDIA's libraries from pip:
   `nvidia-cublas==13.0.2.14` and `nvidia-cuda-runtime==13.0.96` (~0.4 GB), or for CUDA 12
   `nvidia-cublas-cu12==12.9.1.4` and `nvidia-cuda-runtime-cu12==12.9.79`. `cuda_lib_dirs()`
   (`setup.py:1329-1342`) finds their folders for the config's `lib_dirs`.

The zips are produced by the release tooling. The comment at `setup.py:96-97` names `tools/make_release.py`, which
is not in this snapshot.

## Compiling here (NVIDIA)

`build_engine(gpu, vision, yes, llama, toolkit)` (`setup.py:2274-2332`) runs when `--build` is given, when no
ready-made engine fits, when the CPU has no AVX2, or when a card is new to a locally compiled engine:

- It skips the work when a local engine exists with the same `src` hash, all the archs and the same `isa_floor`
  (and, with images, the encoder with the same `vision_src`).
- `install_build_tools()` (`setup.py:2132-2205`), see below.
- The engine is built with `cmake_build(ROOT, build/, "strata", ...)` and these definitions:
  `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=<archs> -DCMAKE_CUDA_COMPILER=<nvcc>
  -DSTRATA_GGML_DIR=<third_party/llama.cpp>`, plus `engine_defs` and `isa_floor_defs`. The result is copied to
  `engine/strata`.
- The image encoder is built with `cmake_build(tools/vision, build-vision/, "strata-vision", ...)` using
  `-DLLAMA_DIR=<llama.cpp> -DSTRATA_VISION_CUDA=ON|OFF`, plus the CUDA archs for the GPU encoder.
- `isa_floor_defs(floor, bdir, meta)` (`setup.py:2246-2251`) adds `-DSTRATA_ISA_FLOOR=avx|none` for an older CPU.
  It deletes `CMakeCache.txt` when the floor changed, because ggml's CPU options are cached.

`cmake_build(src, bdir, target, defs, vcvars, bat_name)` (`setup.py:2208-2228`) runs
`cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ...` and then `cmake --build ... -j max(2, cpus // 2)`. A failed build is
tried once more, because CUDA 13.0's ptxas sometimes fails to parse a PTX file it just wrote (#45). On Windows both
commands go into a `.bat` (`build-strata.bat`, `build-vision.bat`, `...-cuda12.bat`) that first calls `vcvars64.bat`.

### Build tools

| Function | Lines | What it finds or installs |
| --- | --- | --- |
| `find_nvcc(below)` | `setup.py:951-985` | `STRATA_NVCC` alone if set (#601). Otherwise the newest of `nvcc` on PATH, `CUDA_PATH`/`CUDA_HOME`, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*`, `/usr/local/cuda*` and `/opt/cuda*` (Arch, #46). `below=(13,0)` restricts it to CUDA 12.x |
| `find_vcvars()` | `setup.py:988-997` | `vswhere` for Visual Studio 2019 or 2022 (`[16.0,18.0)`) with the VC tools. CUDA 13 rejects newer ones |
| `find_tool(name)` | `setup.py:1000-1010` | `cmake` / `ninja` on PATH or pip's `Scripts`/`bin` |
| `install_build_tools(gpu, yes)` | `setup.py:2132-2205` | The CUDA needed is 13.0 for sm_120, 12.0 otherwise; the CUDA 12 engine needs 12.8+ for sm_120, else 12.0+. If anything is missing it asks "Install them now?" (default y). Windows: `winget install Microsoft.VisualStudio.2022.BuildTools` (VCTools workload) and `Nvidia.CUDA --version 13.0`. Linux: `apt-get` only, `sudo apt-get install build-essential`, and on Ubuntu 22.04 / 24.04 only the `cuda-keyring` deb and `cuda-toolkit-13-0`. A CUDA 12 toolkit is never installed: setup stops with a link to the archive |

## AMD on Windows: the ready-made HIP engine

`get_prebuilt_hip(url_base, gpu, updating)` (`setup.py:1684-1750`) fetches `strata-windows-x64-hip.zip` (strata.exe,
strata-device.exe and the ROCm DLLs). It keeps an installed one when its `source` is `prebuilt`, it is HIP, its
version is at least `WIN_HIP_MIN_ENGINE` (the larger of 0.1.39 and 0.1.33), and it covers the card's arch. A zip that
is not HIP, is too old, or lacks the arch is refused. `hip_runtime_beside_exe()` (`setup.py:1638-1652`) copies
`amdhip64_*.dll` and `amd_comgr*.dll` from `rocm/bin` next to the exe. Without that, Windows loads a driver's
System32 copy first, and the bundled rocBLAS then crashed on the first prompt (#468 #461). The copy runs at install,
at every start and on `--update`. If no zip is published, setup stops and points to `tools\hip\build_windows.bat`. A
CPU without AVX2 also stops, since there is no Windows HIP compile.

## AMD on Linux: compiled here

`build_engine_hip(gpu, llama, vision)` (`setup.py:1852-1901`):

- It needs `c++`/`g++` and `git` ("sudo apt install build-essential git").
- `rocm_root(archs)` (`setup.py:1771-1816`) uses a system ROCm (`ROCM_PATH` or `/opt/rocm`) when it has `bin/hipcc`,
  `lib/libhipblas.so*`, a version of at least 7.0 (from `rocm_version.h`) and the HIP development files
  (`rocm_dev_missing()`, #446). Otherwise it pip-installs `rocm[libraries,devel]==7.10.0a20251120`
  (`STRATA_ROCM_VERSION`) from TheRock's per-family index (`ROCM_INDEXES`: gfx110X-dgpu, gfx120X-all, gfx103X-all;
  `STRATA_ROCM_INDEX` overrides it) into `.venv` ("~10 GB, no sudo"), stamped in `.venv/.strata-rocm.json`, and
  takes the root from `rocm-sdk path --root`. Cards of two families stop, because the wheels hold one family.
- CMake runs into `build-hip/` with `-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF
  -DSTRATA_PREFILL_MMQ=ON -DCMAKE_HIP_ARCHITECTURES=<gfx;...>`, the ROCm clang++ and paths, and
  `-DSTRATA_GGML_DIR`, plus the ISA floor.
- `--vision cpu` adds the CPU image encoder (`build_vision_cpu()`, `setup.py:1917-1926`, #304).
- `hipblaslt_table(arch, lib_dirs, ver)` (`setup.py:1833-1849`): when
  `tools/hip/<arch>-hipblaslt-<version>.txt` exists for the installed hipBLASLt version and its header reads
  `STRATA_HIPBLASLT_TUNING_V1 <arch> <ver>`, the config gets `env STRATA_HIPBLASLT_TUNING=<table>`. The comment
  cites "+40-60% prompt speed on the 7900 XTX". The snapshot ships tables for gfx1100 (100100, 100200), gfx1200
  (100202) and gfx1201 (100202, 100500).

## Images with a ready-made engine

If the zip has no `strata-vision`, images force a compile (`setup.py:4112-4114`). If its encoder lacks the card's arch
(`vision_archs`), `prebuilt_vision()` moves images to the CPU encoder instead of compiling (#331).

## Engine changes at a start

- `update_installed_engine()` ([update-and-data-folder.md](update-and-data-folder.md#the-engine-update-on-every-start)).
- `ensure_engine_for(cards, ...)` (`setup.py:3388-3412`): a card the engine has no code for gets a compile for all
  cards, and the config's `lib_dirs` is refreshed.
- `use_cuda12(cards, ...)` (`setup.py:3432-3450`) and `get_cuda12_engine()` (`setup.py:3415-3423`): a Pascal/Volta
  card added to a CUDA 13 model moves that model's config to the CUDA 12 engine (`"cuda": 12`, a new `exe` and
  `lib_dirs`). Other models keep their engine.

## Gotchas

- One engine per model: two installed models can run different toolkits, but one model cannot mix them.
- `get_prebuilt()` deletes `BUILD.json` before downloading an update. If the update fails, `update_installed_engine()`
  writes it back (`setup.py:2116-2118`).
- On Windows a running engine is locked. The update tests this by opening the exe for writing, and asks you to close
  the model window (`setup.py:2087-2094`).

## Tests

`tools/test_setup_pins.py` (`Engine`: release first, latest fallback, a refused archive not kept, an installed engine
kept), `tools/test_setup_older_gpus.py` (the CUDA 12 choice, opt-in, wheels, build tools, moving a model),
`tools/test_setup_amd.py` (ROCm fallback, build for every arch, the Windows HIP zip, the runtime DLL copy),
`tools/test_setup_oldcpu.py` (`isa_floor_defs`), `tools/test_setup_choices.py` (`PrebuiltVision`, `ExperimentalSm60`,
`HipVision`), and `tools/test_setup_prompts.py` (EOF never starts an installer).

## In UnbiasedStrata

Removed. `make build` replaces all of it: one CMake configure and build of the target `unbiased-strata` with
`-DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_MMQ_KQUANTS=ON` for archs
`61;70;75;86;89;120`, dropping any arch the installed nvcc cannot compile, and adding
`-DSTRATA_EXPERIMENTAL_SM60=ON` when an arch below 75 remains. ggml comes from the `third_party/llama.cpp` submodule.
There are no ready-made downloads, no HIP, no Windows and no `BUILD.json`
([makefile](../../codebase/build/makefile.md)).

## See also

[downloads-and-packs.md](downloads-and-packs.md) · [update-and-data-folder.md](update-and-data-folder.md) ·
[build](../build/README.md)
