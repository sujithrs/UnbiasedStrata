[Upstream docs](../README.md) › [Build](README.md) › Builds driven by the installer and scripts

# How the build is driven: installer, Docker, release scripts, hand builds

## Purpose

The exact CMake invocations that upstream's own scripts run, and the hand-build recipes its docs give. The installer's
logic (when it downloads a ready-made engine, how it picks GPUs, models and configs) is documented in
[../install/README.md](../install/README.md); this page quotes only its build commands.

Files covered:
[setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py) (build functions only),
[Dockerfile](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/Dockerfile) (lines 66-103),
[tools/hip/build_windows.bat](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/hip/build_windows.bat),
[tools/vision/CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/vision/CMakeLists.txt).

## When the installer compiles at all

By default `setup.py` downloads a ready-made engine zip (`strata-windows-x64.zip` / `strata-linux-x64.zip`, built with
CUDA 13.0; `PREBUILT_URL` and `PREBUILT_TAG_URL`, `setup.py:95-104`). It compiles when:

- `--build` is given, or no ready-made engine fits;
- the GPU is AMD on Linux (there is no ready-made Linux HIP engine; Windows HIP uses a ready-made zip only);
- the experimental CUDA 12 engine is needed on Linux (`get_cuda12_engine`, `setup.py:3413-3423`);
- a vision encoder is wanted and the ready-made engine has none;
- an engine that was compiled locally is older than the source (its `BUILD.json` `src` hash differs, e.g. after a
  `git pull`), or a newly added card has an arch the engine lacks (`setup.py:2087-2104`, `3395-3410`).

Intel (`--backend sycl`) hands over to `sycl/setup_intel.py` (`setup.py:3506-3530`); see
[../sycl/README.md](../sycl/README.md).

The comment at `setup.py:96-97` says the ready-made zips are built by `tools/make_release.py`; that script is not in
this snapshot, so the release build's exact flags are not determined from source (the `STRATA_PORTABLE` comment at
`CMakeLists.txt:31-33` describes its intent).

## Shared pieces

### `get_llama_cpp()` (`setup.py:1254-1283`)

Returns `third_party/llama.cpp` if it already has `ggml/CMakeLists.txt` and `gguf-py/`; otherwise downloads
`https://github.com/ggml-org/llama.cpp/archive/<LLAMA_CPP_COMMIT>.zip`, extracts it without `tools/ui/` (Windows path
length, #206), and moves it into place with retries for antivirus locks (PR #63). No git is needed. Every build below
passes this directory as `-DSTRATA_GGML_DIR`, so the CMake `FetchContent` path is never used by the installer.

### `cmake_build(src, bdir, target, defs, vcvars, bat_name)` (`setup.py:2208-2228`)

```
cmake -G Ninja -DCMAKE_MAKE_PROGRAM=<ninja> -S <src> -B <bdir> -DCMAKE_BUILD_TYPE=Release <defs...>
cmake --build <bdir> --target <target> -j <max(2, cpu_count // 2)>
```

- `cmake` and `ninja` come from PATH or from the `.venv`'s pip packages (both are in `PY_PACKAGES`, `setup.py:119`).
- A failed build is retried once: CUDA 13.0's ptxas sometimes fails to parse a PTX file it just wrote (issue #45).
- On Windows the commands are written to a `.bat` in the repo root (`build-strata.bat`, `build-strata-cuda12.bat`,
  `build-vision.bat`, ...) that first `call`s the Visual Studio `vcvars` script, then run with `cmd /c`.

### Rebuild detection: `source_hash()` and `ENGINE_SOURCES` (`setup.py:2231-2243`)

`ENGINE_SOURCES = ("CMakeLists.txt", "src", "include", "third_party/ggml")`, `VISION_SOURCES = ("tools/vision",)`.
The hash is SHA-256 over the llama.cpp commit and every file's relative path and content (CRLF normalised to LF),
truncated to 16 hex digits. It is stored as `src` / `vision_src` in `engine/BUILD.json`.

### Extra definitions

| Helper | Lines | Adds |
| --- | --- | --- |
| `engine_defs(archs, toolkit)` | `setup.py:2254-2257` | `-DSTRATA_EXPERIMENTAL_SM60=ON` if any arch is below 75 or the toolkit is CUDA 12 |
| `isa_floor_defs(floor, bdir, meta)` | `setup.py:2246-2251` | `-DSTRATA_ISA_FLOOR=avx` / `none` on a CPU without AVX2; deletes the build folder's `CMakeCache.txt` when the floor changed, because ggml's CPU options are cached |

## The NVIDIA engine: `build_engine()` (`setup.py:2274-2329`)

```
cmake_build(ROOT, build/ | build-cuda12/, "strata",
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF
  -DCMAKE_CUDA_ARCHITECTURES=<every card's arch, plus those already in BUILD.json, ';'-joined>
  -DCMAKE_CUDA_COMPILER=<nvcc>  -DSTRATA_GGML_DIR=<third_party/llama.cpp>
  [-DSTRATA_EXPERIMENTAL_SM60=ON] [-DSTRATA_ISA_FLOOR=avx|none])
```

- Toolkit: 12 when the oldest card is below sm_75 (`CUDA13_MIN_ARCH`), else 13; CUDA 12 builds go to
  `build-cuda12/` and `engine-cuda12/`, others to `build/` and `engine/`. `find_nvcc()` (`setup.py:951`) takes
  `STRATA_NVCC` if set, else the newest of PATH, `CUDA_PATH`, `CUDA_HOME`, `/usr/local/cuda*`, `/opt/cuda*` (Windows:
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*`). `install_build_tools()` installs a toolkit and compiler if
  missing (see [../install/README.md](../install/README.md)).
- The arch list is the GPUs' own compute capabilities, never `native`, so a model split across different cards gets
  code for each.
- `STRATA_NATIVE_EXPERTS` is left at its default (ON), `STRATA_PORTABLE` is not passed (ggml-cpu is built for this
  CPU), `STRATA_MMQ_KQUANTS` is not passed.
- Output: `build/strata` copied to `engine/strata`; `BUILD.json` = `{"source": "local", "version", "archs", "vision",
  ["toolkit": 12], "cuda_dirs", "src", "vision_src", ["isa_floor"]}`.

Image encoder (when `vision` is `gpu` or `cpu`):

```
cmake_build(ROOT/tools/vision, build-vision/ | build-vision-cuda12/, "strata-vision",
  -DLLAMA_DIR=<third_party/llama.cpp> -DSTRATA_VISION_CUDA=ON|OFF
  [-DCMAKE_CUDA_ARCHITECTURES=<archs> -DCMAKE_CUDA_COMPILER=<nvcc>])     # GPU encoder only
```

`build-vision/bin/strata-vision` is copied to `engine/`.

## The AMD engine on Linux: `build_engine_hip()` (`setup.py:1852-1900`)

Needs `c++`/`g++` and `git`; locates ROCm with `rocm_root(archs)` and exports `HIP_PLATFORM=amd`, `HIP_COMPILER=clang`,
`HIP_RUNTIME=rocclr`, `ROCM_PATH`, `HIP_PATH`, `LD_LIBRARY_PATH`, `PATH`.

```
cmake_build(ROOT, build-hip/, "strata",
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PREFILL_MMQ=ON
  -DCMAKE_HIP_ARCHITECTURES=<archs ';'-joined>
  -DCMAKE_HIP_COMPILER=<rocm>/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=<rocm>
  -DCMAKE_PREFIX_PATH=<rocm>;<lib parents>
  "-DCMAKE_HIP_FLAGS=--rocm-path=<rocm> --rocm-device-lib-path=<bitcode dir>"
  -DSTRATA_GGML_DIR=<third_party/llama.cpp> [-DSTRATA_ISA_FLOOR=...])
```

`BUILD.json` gets `"source": "local-hip", "backend": "hip"`. Images are CPU-only on AMD: `build_vision_cpu()`
(`setup.py:1917-1927`) builds `tools/vision` with `-DSTRATA_VISION_CUDA=OFF`.

## The Docker image (`Dockerfile:66-103`)

The image build runs a Python heredoc that imports `setup` and calls the same helpers, so the container engine is
"built exactly the way setup.py builds them":

```
setup.cmake_build(ROOT, ROOT/"build", "strata",
  ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF",
   f"-DCMAKE_CUDA_ARCHITECTURES={arch}", f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DSTRATA_GGML_DIR={llama}"], None, "build-strata.bat")
```

`arch` comes from the build arg `CUDA_ARCHITECTURES` (default `75;80;86;89;120`; commas accepted). Neither
`engine_defs` nor `isa_floor_defs` is applied, so an arch below 75 is refused by CMake and ggml-cpu is compiled for the
build host's CPU (`GGML_NATIVE=ON`). It writes a `BUILD.json` with `"source": "local"` and the current source hash, so
the container's first start reuses the engine. Details: [docker.md](docker.md).

## The Windows AMD release: `tools/hip/build_windows.bat`

```
cmake -G Ninja -S <repo> -B build-hip-win -DCMAKE_BUILD_TYPE=Release
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF|ON -DSTRATA_PREFILL_MMQ=ON
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON "-DCMAKE_HIP_ARCHITECTURES=%STRATA_HIP_ARCHS%"
  -DCMAKE_C_COMPILER=<rocm>/lib/llvm/bin/clang.exe -DCMAKE_CXX_COMPILER=<rocm>/lib/llvm/bin/clang++.exe
  -DCMAKE_HIP_COMPILER=<same clang++> -DCMAKE_HIP_COMPILER_ROCM_ROOT=<rocm> -DCMAKE_PREFIX_PATH=<rocm>
  "-DCMAKE_HIP_FLAGS=--rocm-path=<rocm> --rocm-device-lib-path=<bitcode>" [-DSTRATA_GGML_DIR=...]
cmake --build build-hip-win --target strata strata-device        (or everything with the "tests" argument)
```

Configure runs only when `build.ninja` does not exist yet (lines 66-74). ROCm comes from TheRock pip wheels
(`STRATA_ROCM_VERSION` default `10.2.0a20260930`); without `STRATA_GGML_DIR` CMake fetches llama.cpp. The result is
packaged by `tools/hip/package_windows.py` into `dist/strata-windows-x64-hip.zip`, which `setup.py` downloads for
Windows AMD users or takes from `--prebuilt <dist folder>`.

## Hand-build recipes in upstream's docs

| Doc | Command (abridged) |
| --- | --- |
| `docs/AMD_HIP.md` "Build" | `cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100` then `cmake --build build-hip --target strata -j2` |
| `docs/AMD_HIP.md` gfx906 | `-DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_GGML_DIR=... -DCMAKE_C(XX)_COMPILER=/opt/rocm/llvm/bin/clang(++)`, `ninja -C build-906 strata` |
| `docs/AMD_HIP.md` "Tuning table" | `cmake --build build-hip --target tune_hipblaslt` |
| `docs/OLDER_GPUS.md` | `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES="61;70" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc` |
| `docs/NVIDIA_V100.md` | `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON`; parity test via `-DSTRATA_PARITY_PROMPT_ATTN=ON ... --target qsa_prompt_attn_parity` |
| `docs/ORCA_Q4_K_S.md` | `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_ORCA_Q4KS_MMQ=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared`, targets `strata native_expert_parity ple_q5_parity` |
| `docs/INTEL_ARC.md` | `cmake -S . -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_SYCL=ON` (or `-S sycl`), `--target strata` |

## Gotchas

- The installer's build folders (`build/`, `build-cuda12/`, `build-hip/`, `build-vision*/`) and generated `.bat` files
  are in `.gitignore`; deleting `engine/BUILD.json` or changing any file under `ENGINE_SOURCES` triggers a recompile on
  the next start.
- The build uses half the logical CPUs (`-j max(2, n // 2)`), not all.
- The Docker build compiles ggml-cpu for the machine that runs `docker build`; an image moved to a host with an older
  CPU may hit illegal instructions in ggml-cpu (inferred from `GGML_NATIVE=ON`; not stated in upstream's docs).

## In UnbiasedStrata

All of this is removed with `setup.py`, `tools/hip/`, `tools/vision/`, `sycl/` and the `Dockerfile` (8894a81). The
fork's single entry point is `make build` (`Makefile`, 1b59e5e, 1e40b06):
`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON
-DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=<filtered list> [-DSTRATA_EXPERIMENTAL_SM60=ON]`,
then `cmake --build build --target unbiased-strata`. Unlike upstream's installer it uses `STRATA_PORTABLE` and the
K-quant MMQ kernels, and it takes ggml from the submodule. See [../../codebase/build/makefile.md](../../codebase/build/makefile.md).

## See also

[../install/README.md](../install/README.md), [docker.md](docker.md), [cmake-options.md](cmake-options.md),
[gpu-backends.md](gpu-backends.md)
