[Upstream docs](../README.md) › [Tools](README.md) › Image encoder (`tools/vision/`)

# The image encoder: `strata-vision`

## Purpose

`tools/vision/` builds `strata-vision`, the image half of Strata's multimodal path. It turns a picture into the
embedding rows the text model reads in place of its `<|image_pad|>` tokens, using llama.cpp's `mtmd` library and the
model's `mmproj` file (vision encoder plus projector). The Python server starts it as a resident child process and
hands the resulting file to the engine, which places the rows at the image's pad tokens and gives them 2-D M-RoPE
positions (`--serve GENI`) (`tools/vision/strata_vision.cpp:1-13`). It is optional: setup builds it only when images
are chosen.

| File | Lines | Role |
| --- | ---: | --- |
| [tools/vision/CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/vision/CMakeLists.txt) | 44 | Standalone CMake project: llama.cpp (`mtmd`, `llama`) as a subdirectory, one executable |
| [tools/vision/strata_vision.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/vision/strata_vision.cpp) | 205 | The resident encoder process |

## Build

```
cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp
      [-DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native] [-DSTRATA_PORTABLE=ON]
cmake --build build-vision --target strata-vision          # -> build-vision/bin/strata-vision
```

| Option | Default | Effect |
| --- | --- | --- |
| `LLAMA_DIR` | `../../third_party/llama.cpp` | the llama.cpp checkout (setup passes the pinned one) |
| `STRATA_VISION_CUDA` | OFF | build ggml-cuda so `--gpu` can run the encoder on the GPU |
| `STRATA_PORTABLE` | OFF | AVX2 baseline (`GGML_NATIVE` off, AVX/AVX2/FMA/F16C/BMI2 on), no OpenMP, static MSVC runtime |

Everything else in llama.cpp is switched off (common, tools, tests, examples, server, curl, OpenSSL); shared
libraries are off, `GGML_BACKEND_DL` off, `LLAMA_BUILD_MTMD` on (`tools/vision/CMakeLists.txt:17-39`).

Who builds it: setup on NVIDIA (`STRATA_VISION_CUDA` ON for `--vision gpu`, `setup.py:2318-2324`), a CPU-only build
beside the HIP engine (`build_vision_cpu`, `setup.py:1917-1924`), the Docker image (`Dockerfile:83-86`), and the
ready-made engine zip, which ships `strata-vision` (`setup.py:96`).

## Protocol (stdin/stdout)

```
strata-vision --mmproj <mmproj.gguf> --model <text model shard 1> [--gpu] [--threads N]
              [--max-tokens N] [--flash-attn on|off|auto]
-> READY <n_embd>
<- ENC <image path> <output path>      -> OK <n_tokens> <nx> <ny> <ms>   |   ERR <message>
<- QUIT
```

- The text model is opened vocabulary-only (no weights); `n_embd` comes from the projector's
  `clip.vision.projection_dim` (`strata_vision.cpp:92-121`).
- Without `--gpu`, `CUDA_VISIBLE_DEVICES=-1` is set before anything touches CUDA, so a CUDA build does not take
  0.4-0.7 GB of VRAM (the comment's measurement) on a card the engine needs (`strata_vision.cpp:79-87`); threads
  default to half the hardware threads.
- With `--gpu`, it encodes a 2048x2048 grey image before printing READY, so the encoder's GPU buffers are allocated
  before the engine starts and sizes its expert cache (`strata_vision.cpp:122-149`).
- The image path ends at the last space before the output path, so image paths may contain spaces
  (`parse_enc`, `strata_vision.cpp:39-48`).
- Output file: `int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}` then `n_tokens x n_embd` float32, row i at grid
  (i % nx, i / nx); a non-rectangular grid is an error (`strata_vision.cpp:176-192`).

The server side is `serve/server.py`'s vision class (`serve/server.py:1263-1300`): it passes `--gpu`, `--threads`,
`--max-tokens` from the config's `"vision"` section, caches encodings by the image's hash, and can unload and restart
the process. See [../serve/README.md](../serve/README.md).

## Platform notes

- Windows and Linux: the environment variable is set with `_putenv_s` or `setenv`.
- GPU encoding needs a CUDA build; on AMD (HIP) setups the encoder is built for the CPU.
- Upstream's numbers (DETAILS.md, "Images (vision)"): the mmproj file is 0.9 GB (a 27-layer ViT plus projector); on
  the GPU 0.1-0.5 s per picture up to 1,024 image tokens with ~1.4 GB of VRAM kept free; on the CPU 10-30 s with
  pictures scaled to ~300 tokens. `--vision-tokens N` (0.1.39, #625) changes the cap.

## Tests

No unit test of `strata-vision` itself. `tools/conversation_cache_isolation.py --scenario image` writes synthetic
`.sve` files in this format and checks that cached state is keyed by image identity
([cache-and-batch-gates.md](cache-and-batch-gates.md)).

## In UnbiasedStrata

Removed in fork commit 8894a81 together with the Python server; the fork has no image path.

## See also

- [pack-format.md](pack-format.md#7-test-and-runtime-fixtures) (the `.sve` format)
- [../serve/README.md](../serve/README.md), [../install/README.md](../install/README.md)
