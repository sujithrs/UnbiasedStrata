[Upstream docs](../README.md) › Intel SYCL port

# The Intel SYCL port (`sycl/`)

`sycl/` is an experimental build of Strata's own engine for Intel GPUs (Arc, Xe2 "Battlemage" first), written in
SYCL with Intel oneAPI. It is not a new backend behind an interface: the CUDA sources in `src/` and `include/` were
translated mechanically by Intel's SYCLomatic tool (`dpct`), corrected by a script of fixups (`sycl/tools/fixups.py`),
and then changed by hand where the Intel GPU behaves differently. The port is a separate CMake project
(`sycl/CMakeLists.txt`) that compiles the migrated copy of a file when `sycl/` has one and upstream's original
otherwise, so no upstream file is edited. On top of the engine sit three Python wrappers that steer the unchanged
upstream installer and server from outside: `sycl/setup_intel.py` (wraps `setup.py`), `sycl/serve/server_intel.py`
(wraps `serve/server.py`) and `sycl/serve/strata-sycl.sh` (runs the engine binary inside a oneAPI Docker image). The
main installer reaches it through `setup.py --backend sycl`; the top-level CMake reaches it through
`-DSTRATA_ENABLE_SYCL=ON`. The port was contributed in upstream PR #423 and tracks engine 0.1.39; the upstream
maintainers state they have no Intel GPU and only compile it and run its kernel tests on a CPU device
(`docs/INTEL_ARC.md`). Terms such as MTP, QSA, PLE, GDN and "native pack" are defined in the
[glossary](../guides/glossary.md).

## Contents

| Page | What it covers |
|---|---|
| [migration.md](migration.md) | How the port was produced and is kept up to date: the dev image, `get-cuda-headers.sh`, `migrate.sh`, `normalize.sh`, every rule in `fixups.py`, `merge_upstream.py`, `build.sh` |
| [layout.md](layout.md) | The tree, the source-resolution rule, the vendored dpct helper headers, and a per-file inventory of `sycl/src`, `sycl/include/strata` and `sycl/third_party` (migrated, hand-changed, port-only, stub, not built) |
| [differences.md](differences.md) | What the SYCL engine does differently from the CUDA engine: host/device handshake, queues, sub-groups, copies, `--stream-experts`, the host mirror, verify-window changes, kernel speed work, XMX, what is disabled, port-only env vars |
| [build.md](build.md) | `sycl/CMakeLists.txt` target by target, the load-bearing compiler flags, oneAPI requirements, AOT targets, the top-level `STRATA_ENABLE_SYCL` option, the parity tests and benches |
| [serving.md](serving.md) | The installer hand-off (`setup.py --backend sycl`, `tools/test_setup_sycl.py`), `setup_intel.py`, `strata-sycl.sh`, `server_intel.py` (xe telemetry, model menu, logprobs), `gpustat.py` |
| [hardware.md](hardware.md) | Which Intel GPUs it targets, what has been run on which card, measured numbers as reported by upstream's docs, platform gotchas |

## Source map

Files outside `sycl/src`, `sycl/include/strata` and `sycl/third_party` (those 137 files are listed one by one in
[layout.md](layout.md#inventory)).

| File | Lines | Role |
|---|---|---|
| [sycl/CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/CMakeLists.txt) | 241 | The SYCL build: source resolution, compiler flags, engine libraries, `strata` binary, parity tests |
| [sycl/.gitignore](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/.gitignore) | 4 | Ignores dpct's reports, `.yaml` and `migrate.log` |
| [sycl/setup_intel.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/setup_intel.py) | 246 | `setup.py` with the GPU check, engine step, RAM rule and config swapped for the Intel path |
| [sycl/serve/server_intel.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/serve/server_intel.py) | 242 | `serve/server.py` plus xe telemetry, a model menu and OpenAI logprobs, patched in at run time |
| [sycl/serve/strata-sycl.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/serve/strata-sycl.sh) | 27 | The engine `exe`: `docker run` of the SYCL binary in the `strata-sycl-dev` image, pipes attached |
| [sycl/serve/xe_telemetry.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/serve/xe_telemetry.py) | 103 | `_XeGpu`: Arc readings for the Monitor tab from sysfs and `/run/gpustat.json` |
| [sycl/serve/web/switcher.js](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/serve/web/switcher.js) | 60 | The web app's Model menu (`model_switcher` hosts) |
| [sycl/serve/web/switcher.css](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/serve/web/switcher.css) | 7 | Style of the Model menu |
| [sycl/tools/Dockerfile](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/Dockerfile) | 9 | `strata-sycl-dev` image: a community llama.cpp SYCL image plus `dpct`, ninja, Python |
| [sycl/tools/Dockerfile.unitrace](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/Dockerfile.unitrace) | 8 | The dev image plus Intel's `unitrace` profiler |
| [sycl/tools/get-cuda-headers.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/get-cuda-headers.sh) | 28 | Extracts CUDA 12.8 headers from NVIDIA's pip wheels for dpct |
| [sycl/tools/migrate.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/migrate.sh) | 47 | Writes a compilation database and runs dpct over every CUDA-touching file |
| [sycl/tools/normalize.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/normalize.sh) | 26 | Puts a fresh dpct output into the port's shape and runs `fixups.py` |
| [sycl/tools/fixups.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/fixups.py) | 305 | Idempotent corrections of dpct's output, one reason per rule |
| [sycl/tools/merge_upstream.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/merge_upstream.py) | 37 | 3-way merge of a re-migration into `sycl/` for the files upstream changed |
| [sycl/tools/build.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/build.sh) | 12 | Configure and build with `icx`/`icpx` inside the dev image |
| [sycl/tools/gpustat.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/gpustat.py) | 182 | Root sampler writing `/run/gpustat.json` (load, VRAM, power, temperatures, PCIe link) |
| [sycl/tools/gpustat.service](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/tools/gpustat.service) | 10 | systemd unit for `gpustat.py` |
| `sycl/include/dpct/` (46 files) | 37,651 | Intel's dpct helper headers, vendored (Apache-2.0 with LLVM exception); see [layout.md](layout.md#the-vendored-dpct-helper-headers) |
| `sycl/src/`, `sycl/include/strata/`, `sycl/third_party/` (137 files) | 82,112 | Migrated, hand-changed and port-only engine sources; see [layout.md](layout.md#inventory) |

Files outside `sycl/` that belong to this area:

| File | Lines | Role |
|---|---|---|
| [CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/CMakeLists.txt#L12-L22) | 11 of the file | `option(STRATA_ENABLE_SYCL ... OFF)`: `add_subdirectory(sycl)` and `return()` |
| [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py#L3507-L3531) | `sycl_setup`, 25 of 4,422 | `--backend sycl`: warn, refuse Windows, hand over to `sycl/setup_intel.py` |
| [tools/test_setup_sycl.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_setup_sycl.py) | 75 | Unit tests of the hand-over and the CMake option |
| [docs/INTEL.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INTEL.md) | 652 | The port author's notes, measurements and maintenance procedure |
| [docs/INTEL_ARC.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INTEL_ARC.md) | 126 | Upstream's user page: requirements, build, what was tested by whom |

## Key concepts

- **Migration, not a backend.** The engine has no device-abstraction layer, so the port is a translated copy of
  every file that touches the CUDA runtime: 44 of 44 kernel files in `src/kernels/cuda/` plus host files that call
  CUDA. dpct turns `__global__` kernels into `parallel_for` lambdas over `sycl::nd_range<3>`, streams into
  `sycl::queue*` (`dpct::queue_ptr`), CUDA graphs into `sycl_ext_oneapi_graph` command graphs, cuBLAS into
  `dpct::blas` (oneMKL) and warp shuffles into sub-group operations. About 1,500 `DPCTnnnn` advisory comments
  remain in the code. See [migration.md](migration.md).
- **Mechanical vs hand-written.** Most of the 137 engine files are dpct output plus `fixups.py`; the hand-written
  parts are marked with `SYCL port` / `SYCL:` comments (about 150 lines carry one) and concentrate in `generate.cpp`,
  `prefill.cpp`, `verify.cpp`, `iq_kernels.dp.cpp`, `native_mmvq.dp.cpp`, `fused_gr.dp.cpp` and
  `verify_kernels.dp.cpp`. 15 files exist only in the port. See [layout.md](layout.md#inventory) and
  [differences.md](differences.md).
- **Source resolution.** `strata_resolve()` (`sycl/CMakeLists.txt:56-75`) maps `src/X.cu` to `sycl/src/X.dp.cpp`
  (a migrated copy is mandatory) and `src/X.cpp` to `sycl/src/X.cpp` if it exists, else upstream's `src/X.cpp`;
  `sycl/include` is searched before `include/`.
- **Everything in VRAM.** The CUDA engine keeps every expert in RAM and caches hot ones on the GPU. The port's
  preferred mode, `--stream-experts`, keeps no host copy: experts go from the GGUF straight into the VRAM cache,
  and those that do not fit go into a pinned host mirror read over PCIe.

```
            installer                                   engine build
 ./setup.sh --backend sycl                 src/**/*.cu, *.cpp   (upstream CUDA sources)
   setup.py sycl_setup()                       | sycl/tools/migrate.sh   (dpct + compile_commands.json)
   sycl/setup_intel.py                         | sycl/tools/normalize.sh (names, unchanged files, DPCT serials)
     imports setup.py, patches 8 steps,        | sycl/tools/fixups.py    (dpct mistakes)
     runs setup.main()                         | hand edits ("// SYCL port")
     writes strata-<model>.json                v
       {"backend":"sycl",                  sycl/src, sycl/include/strata  + upstream files not migrated
        "exe":".../strata-sycl.sh", ...}       | sycl/CMakeLists.txt (icpx -fsycl, oneMKL, ggml-cpu)
     writes run-<model>.sh                     v
                                           build-sycl-aot/strata  (or build-sycl/strata)
            run time                                 ^
 run-<model>.sh                                      |
   python sycl/serve/server_intel.py                 |
     = serve/server.py + xe telemetry + model menu + logprobs
     spawns exe: sycl/serve/strata-sycl.sh  --stdin/stdout line protocol-->  docker run strata-sycl-dev
                                                     (/dev/dri, data root mounted at /work)
```

## In UnbiasedStrata

The fork removed the whole port: all 201 files under `sycl/` were deleted in commit `8894a81` ("Remove installer,
Python, non-target platforms, benchmarks and marketing"), together with `setup.py`, `tools/` (including
`tools/test_setup_sycl.py`) and `docs/INTEL.md` / `docs/INTEL_ARC.md`. `git log 12a7825..HEAD -- sycl` shows no
other commit. The top-level `CMakeLists.txt` at the fork's HEAD still contains the `STRATA_ENABLE_SYCL` option and
its `add_subdirectory(sycl)`; with the directory gone, turning the option on fails at configure time. The fork's
own documentation is in [docs/codebase](../../codebase/README.md).
