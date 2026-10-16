[Upstream docs](../README.md) › [Tools](README.md) › GGUF I/O and the tokenizer

# GGUF I/O, the gguf-py locator, the tokenizer and the i-quant fixtures

## Purpose

The small libraries the other tools (and the Python server and setup) share: a GGUF header reader with no
dependencies, a minimal GGUF writer with the Q2_0 codec, the function that finds llama.cpp's gguf-py, the
byte-level BPE tokenizer the server uses to turn text into token ids, and the generator of i-quant test fixtures.

| File | Lines | Role |
| --- | ---: | --- |
| [tools/gguf_reader.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/gguf_reader.py) | 142 | GGUF v3 header reader (metadata and tensor directory), knows type 42 = Q2_0 |
| [tools/gguf_writer.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/gguf_writer.py) | 234 | Minimal GGUF v3 writer (F32, F16, Q2_0) and the Q2_0 quantizer/dequantizer |
| [tools/_paths.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/_paths.py) | 28 | Finds llama.cpp's gguf-py |
| [tools/strata_tokenizer.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/strata_tokenizer.py) | 331 | The tokenizer (qwen35 byte-level BPE) and the `<pack>/tokenizer/` exporter |
| [tools/iq_fixture.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/iq_fixture.py) | 137 | Deterministic i-quant block fixtures for the `iq_parity` CTest |

## `tools/gguf_reader.py`

Written because the gguf PyPI package of the time could not open the model: its `GGMLQuantizationType` had no
member 42 (`Q2_0`, the GSQ-RCO routed-expert encoding) (`gguf_reader.py:1-7`). It is the reference for the C++
reader in `src/artifact/` (`include/strata/artifact/gguf_reader.hpp:8`).

- `GGML_TYPES` (`gguf_reader.py:20-28`): ggml type id -> name, 0..42.
- `BLOCK_GEOMETRY` (`gguf_reader.py:39-48`): (elements per block, bytes per block) for every type the models use,
  including Q2_0 (64, 18), MXFP4 and NVFP4.
- `TensorInfo` (`gguf_reader.py:51-73`): name, shape (GGUF order, dim 0 fastest), type id and name, offset;
  `expected_bytes()` returns None for an unknown type or a size that is not whole blocks.
- `GGUFFile(path)` (`gguf_reader.py:76-142`): parses the magic, accepts only version 3, reads every metadata value
  (scalars and arrays) and the tensor directory, honours `general.alignment` (default 32), and sets `data_start`.
  Tensor data is never read. Helpers: `by_type()`, `find(needle)`.

Users: `iq_pack.py`, `strata_pack.py`, `pack_layer.py`, `canonical_xcheck.py`, `strata_tokenizer.py`,
`gguf_writer.py`'s self-check, `ref/load.py`, `conversation_cache_isolation.py`, and `setup.py` (shard checks and
finding the PLE table's shard, `setup.py:4219-4220`).

## `tools/gguf_writer.py`

A GGUF v3 writer for synthetic files, written because gguf-py could not represent Q2_0 at the time
(`gguf_writer.py:1-19`).

- `quantize_q2_0(w)` / `dequantize_q2_0(raw)` (`gguf_writer.py:42-74`): the Q2_0 block is 64 elements, an FP16
  scale `d = amax`, 16 bytes of 2-bit codes, four per byte LSB-first; code = clamp(round(w/d) + 1, 0, 3), value =
  (code - 1) * d, so the grid is {-d, 0, d, 2d}. `dequantize_q2_0` is the project's Q2_0 reference used by
  `ref/load.py`, `canonical_xcheck.py` and `iq_fixture.py`.
- `GGUFWriter` (`gguf_writer.py:86-182`): `add(key, value, type_name=None)` (types inferred for bool, int, float,
  str and lists), `add_f32(name, arr)`, `add_q2_0(name, arr, shape)` (the full GGUF shape must be given, because a
  fused expert tensor is 3-D), `write(path)`. Supported tensor types: F32, F16, Q2_0 (`TYPE_IDS`,
  `gguf_writer.py:34`).
- `python tools/gguf_writer.py` runs `_selfcheck` (`gguf_writer.py:186-230`), which writes
  `<repo>/bench/tiny-selftest.gguf`, reads it back and checks that requantizing dequantized Q2_0 values is a fixed
  point.

`conversation_cache_isolation.py` uses `GGUFWriter` to write its control-vector fixture.

## `tools/_paths.py`

`gguf_py()` (`_paths.py:15-22`) returns the first directory containing a `gguf/` package among
`$STRATA_GGUF_PY`, `<repo>/third_party/llama.cpp/gguf-py` (what setup downloads) and the development tree's
`../../.ref/llama.cpp/gguf-py`; otherwise it exits with "llama.cpp's gguf-py was not found". `add_gguf_py()`
prepends it to `sys.path`. The pinned gguf-py knows Q2_0 (`mtp_pack.py:36`). Setup sets `STRATA_GGUF_PY` for every
tool it runs (`setup.py:4174`).

Users: `iq_pack.py`, `mtp_pack.py`, `mtp_rt.py`, `test_iq_pack.py`. `iq_fixture.py` inserts
`third_party/llama.cpp/gguf-py` itself and does not use `_paths.py`; `canonical_xcheck.py` and `ref/load.py` import
`gguf` without adding any path.

## `tools/strata_tokenizer.py`

The tokenizer of record. The engine has no tokenizer; it takes token ids (`src/program/generate.cpp:9-12`). The
Python server builds a `Tokenizer` from `<pack>/tokenizer/` at start (`serve/server.py:3952-3960`), so every prompt
the engine sees was tokenized here (see [../serve/README.md](../serve/README.md)).

| Piece | Lines | What it does |
| --- | --- | --- |
| `bytes_to_unicode` | 25-40 | the GPT-2 byte -> printable-character map, so any UTF-8 input is representable |
| `QWEN35_PATTERN` | 56-64 | the `qwen35` pre-tokenizer regex, transcribed from llama.cpp's `LLAMA_VOCAB_PRE_TYPE_QWEN35` (with `\p{M}`, which the QWEN3 pattern lacks) |
| `Tokenizer.__init__` | 68-112 | vocabulary (duplicates refused), merge ranks (a merge naming an unknown token refused), special literals: type 4 (USER_DEFINED, e.g. `<think>`) always matched, type 3 (CONTROL, e.g. `<|im_end|>`) only with `parse_special` |
| `from_gguf` | 121-137 | reads `tokenizer.ggml.*` metadata; requires model `gpt2` |
| `_bpe` / `_bpe_heap` | 140-203 | lowest-rank-first merges; words longer than `HEAP_MIN` = 64 symbols use an O(n log n) heap with the same order (#268) |
| `encode(text, parse_special=False, plain=())` | 239-247 | `plain` spans are tokenized as text even where they spell a literal (#537) |
| `token_bytes`, `decode` | 249-266 | ids -> bytes -> UTF-8 (`errors="replace"` by default) |
| `extract(gguf, out)` | 270-298 | writes `<out>/tokenizer/` (format on [pack-format.md](pack-format.md#17-tokenizer)) |
| `main` | 301-327 | `--gguf`, `--out`, `--check` (round-trips a 16-string corpus: CJK, Arabic, emoji, combining marks, control bytes, a 5,000-character run) |

The docstring records the vocabulary of the model: 248,320 tokens, 247,587 merges, `pre` = `qwen35`
(`strata_tokenizer.py:4`). Needs the `regex` package (pinned in `requirements.txt`).

Users: `serve/server.py`, `serve/test_detok.py`, `iq_pack.py` (subprocess), setup (subprocess for the canonical
pack, `setup.py:4182-4184`), `draft_vocab.py`, `calibrate.py`, `batch_test.py`, `conversation_cache_parity.py`,
and the community benchmark scripts under `bench/results/`.

## `tools/iq_fixture.py`

Writes, for each of ten types (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_M, IQ4_NL, IQ4_XS, Q2_0, Q3_K), a
`.bin` with an `int32 type, rows, cols` header and raw blocks, and a `.f32` reference
([pack-format.md](pack-format.md#7-test-and-runtime-fixtures)). The blocks are seeded random bytes whose FP16
scales are overwritten with 1.0 at each layout's scale offsets (`repair_scales`, `iq_fixture.py:68-83`), because
gguf-py has no quantizer for i-quants; the reference comes from gguf-py's dequantizer, or from `gguf_writer.py`'s
codec for Q2_0.

```
python tools/iq_fixture.py --out build/iq_fixture [--seed 7] [--rows 32] [--cols 256]
```

Exit code 3 when numpy or the vendored gguf-py is missing (`iq_fixture.py:37-47`), which CTest treats as a skip.
CMake registers `iq_parity_fixtures` (this script, `FIXTURES_SETUP`) and `iq_parity` (`FIXTURES_REQUIRED`) and
searches for a Python with numpy; `-DIQ_FIXTURE_PYTHON=...` overrides it (`CMakeLists.txt:593-630`). The consumer is
`src/kernels/iq_parity.cpp` (and the SYCL copy in `sycl/src/kernels/iq_parity.cpp`).

## Configuration

| Setting | Effect |
| --- | --- |
| `STRATA_GGUF_PY` | first place `_paths.gguf_py()` looks |
| `IQ_FIXTURE_PYTHON` (CMake) | the interpreter that runs `iq_fixture.py` under CTest |

## Tests

- `python tools/gguf_writer.py`: the self-check (see Gotchas).
- `python tools/strata_tokenizer.py --gguf <shard 1> --out <dir> --check`: round-trip check on a real GGUF.
- `serve/test_detok.py` and `serve/test_server.py` exercise the tokenizer through the server (another area).
- CTest `iq_parity_fixtures` + `iq_parity`.

## Gotchas

- `gguf_writer._selfcheck` calls `add_q2_0("blk.0.ffn_gate_exps.weight", A, 64)` with an integer where a shape list
  is expected (`gguf_writer.py:196`); `shape[0]` on an int raises `TypeError`, so the self-check fails as published.
  It also writes into `<repo>/bench/`, not a temporary directory.
- `GGUFFile` parses version 3 only.
- `Tokenizer` built from the exported files gets `pre` = `qwen35` by default and no `special_ids`; the pattern is
  always `QWEN35_PATTERN`, whatever `tokenizer.ggml.pre` says (no other pre-tokenizer is implemented).
- `iq_fixture.build` seeds `numpy.random.RandomState((seed, KERNEL_IDS[name]))` through an expression with a dead
  `if False` branch (`iq_fixture.py:93`); the effect is deterministic per type.

## In UnbiasedStrata

All five were removed in fork commit 8894a81 and restored unchanged in c00942d. `strata_tokenizer.py` was then
changed in e41dfc7 so that `regex` is optional (exporting `tokenizer/` works without it; encoding raises). The
others are unchanged at the fork's HEAD (`git diff 12a7825 HEAD` shows no change).

## See also

- [pack-format.md](pack-format.md), [pack-builders.md](pack-builders.md), [reference-loader.md](reference-loader.md)
- [../serve/README.md](../serve/README.md) (the server that tokenizes with this module)
- The fork's [pack-builders.md](../../codebase/tools/pack-builders.md)
