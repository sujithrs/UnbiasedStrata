[Upstream docs](../README.md) › [Build](README.md) › Docker

# Docker image

## Purpose

Upstream ships a Linux container for NVIDIA cards: the engine (and optionally the image encoder) is compiled during
`docker build`, and the container's first start downloads the model into a volume and serves it. Files covered:

| File | Lines | Role |
| --- | --- | --- |
| [Dockerfile](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/Dockerfile) | 115 | Image definition and the build step |
| [docker-entrypoint.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docker-entrypoint.sh) | 65 | Start-up: setup pass on first start, then the server |
| [.dockerignore](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/.dockerignore) | 27 | Build context filter |

User-facing instructions: [docs/INSTALL.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INSTALL.md) ("Docker (Linux)").

## Image build (`Dockerfile`)

| Lines | Step |
| --- | --- |
| 41 | `FROM nvidia/cuda:13.0.0-devel-ubuntu24.04` |
| 46 | `ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 STRATA_EXECV=1` |
| 48-51 | apt: `build-essential ca-certificates curl git libatomic1 libgomp1 python3 python3-pip python3-venv unzip` |
| 53-54 | `WORKDIR /opt/strata`, `COPY . .` |
| 58-59 | `ARG CUDA_ARCHITECTURES=75;80;86;89;120`, `ARG BUILD_VISION=1` |
| 61-64 | `.venv` with `requirements.txt` (which brings `cmake` and `ninja` as pip packages) |
| 70-100 | Python heredoc: `setup.get_llama_cpp()` (downloads the pinned llama.cpp zip into `third_party/llama.cpp`), `setup.find_nvcc()`, `setup.cmake_build()` for `strata` and, unless `BUILD_VISION=0`, `tools/vision` with `-DSTRATA_VISION_CUDA=ON`; copies the binaries to `engine/` and writes `engine/BUILD.json` (`source: local`, `archs`, `vision`, `cuda_dirs`, `src` / `vision_src` hashes) |
| 103 | `rm -rf build build-vision` (the CMake trees are not kept; `third_party/llama.cpp` is) |
| 105-106 | `VOLUME ["/data"]`, `EXPOSE 8080` |
| 112-113 | `HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3`: `curl -fs http://127.0.0.1:${PORT:-8080}/health` |
| 115 | `ENTRYPOINT ["./docker-entrypoint.sh"]` |

The exact CMake definitions are listed in [installer-builds.md](installer-builds.md#the-docker-image-dockerfile66-103).
The arch list is fixed at build time because `docker build` has no GPU (comment `Dockerfile:7-11`): the engine is a fat
binary with one cubin per listed arch. A card outside the list needs a rebuild with
`--build-arg CUDA_ARCHITECTURES=<arch>`; an arch below 75 is refused by CMake (the Dockerfile does not add
`STRATA_EXPERIMENTAL_SM60`).

Because `BUILD.json` records `source: local` and the current source hash, `setup.py` treats the image's engine as
already compiled and does not recompile at container start.

Note: `.dockerignore` excludes `third_party/llama.cpp/` from the context, so the build always downloads llama.cpp
again; it also excludes `.git`, `.venv/`, `/engine/`, `/build*/`, `/dist/`, `/packs/`, `/mtp/`, `/models/`,
`/Strata-data/`, `third_party/*.zip`, logs and generated `strata-*.json` / `strata-*.txt` / `run-*` files.

## Start-up (`docker-entrypoint.sh`)

Environment variables and defaults (`docker-entrypoint.sh:9-21`):

| Variable | Default | Passed to `setup.py` as |
| --- | --- | --- |
| `STRATA_DATA` | `/data` | `--data-dir` |
| `FAMILY` | `qwen` | `--family` |
| `MODEL` | `IQ2_XS` | `--model` |
| `CONTEXT` | `32768` | `--context` |
| `VISION` | `no` (`yes`, `cpu`) | `--vision` |
| `HOST` | `0.0.0.0` | `--host` |
| `PORT` | `8080` | `--port` |
| `API_KEY` | empty | `--api-key` |
| `KV` | empty (setup's default, int8) | `--kv` (only if set) |
| `GPUS` | empty (`"0,2"` or `"all"`) | `--gpus` (only if set) |
| `GPU` | empty | `--gpu` (only if set) |
| `LAYER_SPLIT` | empty | `--layer-split` (only if set) |
| `LOW_RAM` | `auto` (`on`, `off`) | `--low-ram` (always) |
| `REINSTALL` | `0` | forces the setup pass |

Flow:

```
tag = [<family>-]<model lowercased>        (qwen has no prefix)
cfg = $STRATA_DATA/config/strata-<tag>.json
if REINSTALL=1 or cfg missing:
    .venv/bin/python setup.py --setup --yes --family .. --model .. --context .. --vision .. --data-dir ..
                              --host .. --api-key .. --port .. --no-start --low-ram .. [--kv ..] [--gpus ..] [--gpu ..] [--layer-split ..]
    copy /opt/strata/strata-<tag>.json to cfg if it differs
else:
    symlink cfg to /opt/strata/strata-<tag>.json if absent
exec .venv/bin/python setup.py --port .. [--gpus ..] [--gpu ..] [--layer-split ..]
```

`setup.py` without `--setup` finds the installed config and starts `serve/server.py`. With `STRATA_EXECV=1` it
replaces itself with the server (`os.execv`, `setup.py:3190-3194`), so the server is PID 1 and receives `docker stop`'s
SIGTERM. `GPUS` / `GPU` / `LAYER_SPLIT` are repeated at start so they pin the cards; otherwise a one-card config may be
offered to a two-card host once (`docker-entrypoint.sh:55-60`).

Run example from the Dockerfile header (`Dockerfile:17-23`):

```sh
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 -v strata-data:/data -e MODEL=IQ2_XS strata
```

Host requirements stated there: an NVIDIA driver 580 or newer and `nvidia-container-toolkit`.

## Platform notes

- NVIDIA only; no HIP or SYCL image. The `sycl/tools/Dockerfile` in the SYCL port is a separate thing (see
  [../sycl/README.md](../sycl/README.md)).
- `setup.py` reads RAM from `/proc/meminfo`, which in a container is the host's total, so a memory-capped container
  needs `-e LOW_RAM=on` (`Dockerfile:30-32`).
- The model (about 70 GB, per the header), packs, MTP layer and config live on `/data`; the engine is in the image.

## Gotchas

- `docs/INSTALL.md` says the default arch set covers 86, 89, 120 and 80; the Dockerfile's default also includes 75.
- The health check's 600 s start period is deliberate: the port opens only after the model loads, and a shorter one
  would let a restart policy kill a container mid-download (`Dockerfile:108-111`).
- ggml-cpu is compiled for the CPU of the machine running `docker build` (`GGML_NATIVE=ON`, no `STRATA_PORTABLE`).

## In UnbiasedStrata

Removed: `Dockerfile`, `docker-entrypoint.sh` and the installer they call were deleted in 8894a81. The fork has no
container build.

## See also

[installer-builds.md](installer-builds.md), [../install/README.md](../install/README.md), [gpu-backends.md](gpu-backends.md)
