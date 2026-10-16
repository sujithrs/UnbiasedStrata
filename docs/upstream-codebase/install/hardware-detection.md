[Upstream docs](../README.md) › [Install](README.md) › Hardware detection

# Hardware detection

## Purpose

This page covers how `setup.py` learns what the PC has: GPUs from three vendors, the driver version, RAM, the page
file, CPU features and core types, disk space, the disk type, WSL and a Linux desktop. All of it is read without
installing anything, and the step 1 checks and every later recommendation depend on it.

Files covered: [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py)
(`setup.py:300-1016`, `setup.py:1345-1681`, `setup.py:2878-2895`, `setup.py:3337-3343`).

## NVIDIA

| Function | Lines | Behaviour |
| --- | --- | --- |
| `gpus()` | `setup.py:529-541` | Runs `nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits`. Each card becomes `{index, name, vram_gb (MiB/1024), arch ("8.9" -> "89"), driver}`. The numbering is nvidia-smi's (PCI bus order) |
| `gpu_info(pick)` | `setup.py:935-948` | The main card: `pick`, else `GPU_PICK`, else the most VRAM (rounded GB; ties go to the lower index). Adds `count`. Stops on an unknown number |
| `gpu_problem(g, together)` | `setup.py:629-638` | Why a card cannot be used. Below compute capability 7.5 it is "not supported", unless it is Pascal/Volta (6.0-7.0) and opted in ([engine.md](engine.md#cuda-13-or-cuda-12)). With `together`, a card under `SPLIT_MIN_VRAM_GB - 0.5` (7.5 GB) cannot share a model |
| `gpu_table(found)` | `setup.py:651-655` | Prints "Your NVIDIA GPUs:" with each card's verdict |
| `driver_major(gpu)` | `setup.py:1930-1934` | The driver's major version. Step 1 stops below `MIN_DRIVER = 580` (CUDA 13.0), or below `CUDA12_MIN_DRIVER` (528 Windows / 525 Linux) for the CUDA 12 engine (`setup.py:3791-3796`) |

When `nvidia-smi` answers nothing, step 1 stops: "no NVIDIA GPU found". If AMD cards were seen, the hint mentions
`--backend hip` (`setup.py:3772-3775`). The multi-card rules (`together_ok`, `gpu_rank`, `split_short`) are in
[low-ram-and-multi-gpu.md](low-ram-and-multi-gpu.md#choosing-several-gpus).

## AMD on Linux: the KFD topology

`amd_gpus(sysfs="/sys")` (`setup.py:1375-1410`) needs only the amdgpu kernel driver, not ROCm:

- It walks `/sys/class/kfd/kfd/topology/nodes/<n>/properties` in numeric order. A node with
  `gfx_target_version` 0 or `simd_count` 0 is a CPU node and is skipped. The GPU nodes are numbered 0, 1, ... as HIP
  numbers them.
- The arch comes from `gfx_target_version`: `gfx{v//10000}{(v//100)%100:x}{v%100:x}`. For example 110000 gives
  `gfx1100` and 120001 gives `gfx1201`.
- VRAM comes from `/sys/class/drm/renderD<drm_render_minor>/device/mem_info_vram_total`, and the name from
  `product_name` there, or `AMD_NAMES[arch]` when the name is missing.
- `amd_problem(g)` (`setup.py:1413-1418`): a card is supported only when its arch is in
  `AMD_ARCHS = gfx1100, gfx1101, gfx1200, gfx1201, gfx1030, gfx1031`. Integrated GPUs are listed but unsupported.
  On Windows, the installed engine's own "cannot run:" verdict also counts.

## AMD on Windows

Windows has no KFD, so `amd_gpus()` calls `amd_gpus_win()` (`setup.py:1421-1424`). That returns the HIP runtime's
own list when a HIP engine is installed (`hip_devices()`), and otherwise `amd_gpus_windows()`:

- `_win_display_adapters()` (`setup.py:1480-1492`): PowerShell
  `Get-CimInstance Win32_VideoController` gives each adapter's name, PNP id and `AdapterRAM` (32-bit, capped at 4 GB).
- `_win_display_registry()` (`setup.py:1495-1524`): reads the display-class registry key
  (`{4d36e968-...}`), without admin rights, for each driver instance's `DriverDesc`, `MatchingDeviceId`,
  `HardwareInformation.qwMemorySize` (the 64-bit VRAM size) and `DriverVersion`.
- `amd_gpus_windows()` (`setup.py:1540-1573`) keeps only `VEN_1002` devices, matches each adapter to its registry
  entry for the VRAM, and maps the PCI device id (`_WIN_AMD_DID`), or failing that the name (`_WIN_AMD_NAME` regexes),
  to an arch. An unknown arch is listed as `unknown (PCI xxxx)` and is not supported.
- `hip_devices()` (`setup.py:1576-1618`) runs the installed engine's `strata-device(.exe) --list-devices`, with the
  bundled ROCm DLLs first on `PATH`. It parses `device N: name`, `arch gfxNNNN, X GiB` and `cannot run: ...` lines.
- After the ready-made HIP engine is installed, `hip_card()` (`setup.py:1664-1681`) re-numbers the chosen card as
  HIP numbers it. `hip_match()` matches it by its rank among cards of the same arch, because an integrated Radeon
  takes HIP device 0 (#325). It stops with a driver hint when HIP lists no GPU.

## Intel Arc

`setup.py` itself has no Intel detection. `--backend sycl` hands the run to `sycl/setup_intel.py`
(`setup.py:3507-3530`), which finds an `xe` or `i915` card in `/sys/class/drm` and reuses setup's AMD path. That
handoff file is documented with the SYCL port.

## Choosing NVIDIA or AMD

`setup.py:3721-3739`: HIP is used when `--backend hip` is given, or when `--backend` is not given, no NVIDIA card is
usable, and an AMD card is. With usable cards of both kinds and no `--backend`, setup asks "Which cards?"
(1 = NVIDIA, recommended).

## RAM and the page file

| Function | Lines | Behaviour |
| --- | --- | --- |
| `ram_gb()` | `setup.py:315-321` | Windows `GlobalMemoryStatusEx().ullTotalPhys`; Linux `MemTotal` from `/proc/meminfo`. In GiB (`/ 2**30`), so a "64 GB" PC reads about 63.7 (the golden test PCs use 63.7, 31.9, 47.0, 95.8, 127.8) |
| `page_file_gb()` | `setup.py:324-331` | Windows only: `ullTotalPageFile - ullTotalPhys`. Under 4 GB, step 1 warns, because under WDDM every VRAM allocation is also charged to the commit (#60) |

In a container, `/proc/meminfo` shows the host's RAM, not the container's limit. That is why the Docker entry point
always passes `--low-ram` (`docker-entrypoint.sh:36-39`).

## CPU

| Function | Lines | Behaviour |
| --- | --- | --- |
| `cpu_info()` | `setup.py:402-422` | `(name, avx2, avx512)`. Linux reads `/proc/cpuinfo` flags; `avx512` means all of `avx512f, avx512bw, avx512vl, avx512_vnni, avx512vbmi`. Windows uses `IsProcessorFeaturePresent(40/41)`, and for AVX2 a CPUID fallback `_cpuid_avx2()` (a Ryzen 9 3950X reported no AVX2 through the API, #159) |
| `_cpuid_avx512_full()`, `_cpuid_avx2()`, `_cpuid_floor()`, `_run_stub()` | `setup.py:425-526` | Windows: run a few bytes of x64 machine code from `VirtualAlloc`'d memory to execute CPUID / XGETBV, because Windows has no feature bit for VNNI / VBMI |
| `cpu_floor(avx2)` | `setup.py:446-464` | The older-CPU build needed: `""` (AVX2: the normal engine), `"avx"` (AVX with OS YMM support), `"none"` (SSE4.2 + POPCNT), or `"unsupported"`. `STRATA_ISA_FLOOR=avx` or `none` forces a floor on any PC |
| `cpu_cores()` | `setup.py:334-372` | `(performance, efficiency)` physical cores on a hybrid CPU. Windows uses `GetLogicalProcessorInformationEx`'s `EfficiencyClass`; Linux uses `cpu_capacity` per physical core. `None` when all cores are alike or the OS does not say |

What step 1 does with these (`setup.py:3822-3837`): it prints `CPU: <name> (AVX-512 | AVX2 | no AVX2)`. It stops on
"unsupported", which means neither AVX2 nor SSE4.2. Without AVX2 it warns that support is experimental and slow and
sets `a.build = True`, so the engine is compiled with `STRATA_ISA_FLOOR`. A Windows AMD install stops instead, since
setup compiles the AMD engine only on Linux. AVX-512 also selects the Q2_0 AVX-512 pack
([downloads-and-packs.md](downloads-and-packs.md#the-pack)). The hybrid cores feed `--pool-workers`
([calibration.md](calibration.md)).

## Disk and environment

| Function | Lines | Behaviour |
| --- | --- | --- |
| `free_gb(path)` | `setup.py:1013-1015` | Free space of the models folder's drive, in GB (`1e9`). Checked against the need at the end of step 2 |
| `rotational_disk(path)` | `setup.py:2882-2895` | Linux: the disk name when `/sys/dev/block/<maj>:<min>/queue/rotational` (or its parent disk's) is 1, else `None`. Used for `--ple-io ram` (#605). Skipped under WSL, whose virtual disk reports rotational |
| `is_wsl()` | `setup.py:2878-2879` | Linux with "microsoft" in the kernel release. Turns KV streaming off |
| `linux_desktop(env)` | `setup.py:3340-3343` | `WAYLAND_DISPLAY` or `DISPLAY` set. On an AMD install it prints the desktop VRAM reserve tip |
| `find_nvcc`, `find_vcvars`, `find_tool` | `setup.py:951-1010` | Build tools, see [engine.md](engine.md#build-tools) |

## Gotchas

- VRAM figures are in GiB (`MiB / 1024`) while disk figures are in GB (`1e9`). The thresholds in the code use these
  units: `vram_gb < 11` for the slow warning, `< 14` and `< 20` for the context recommendation.
- On Windows the AMD numbering changes once the HIP engine is installed: before, it follows the display-adapter
  order; after, it follows HIP's. `hip_card()` bridges the two during the first install.

## Tests

`tools/test_setup_amd.py` (mocked KFD sysfs, Windows adapters and registry, `--list-devices` output),
`tools/test_setup_oldcpu.py` (`cpu_floor` on mocked `/proc/cpuinfo`, the Windows probe, `STRATA_ISA_FLOOR`),
`tools/test_setup_hybrid.py`, and `tools/test_setup_choices.py` (`RotationalDisk`, `DesktopReserveTip`). See
[tests.md](tests.md).

## In UnbiasedStrata

Removed. The fork's front end does no hardware survey. It passes `--pool-workers` from `--threads`, or from half a
container's CPU quota (`default_threads()` in `frontend/main.cpp`), and leaves the rest to the engine.

## See also

[model-choice.md](model-choice.md) · [engine.md](engine.md) ·
[INSTALL.md "What you need"](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INSTALL.md#what-you-need) ·
[OLDER_GPUS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/OLDER_GPUS.md)
