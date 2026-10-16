[Upstream docs](../README.md) › [Tools](README.md) › Pack builders

# Pack builders

## Purpose

These scripts turn a model's GGUF shards (or the BF16 checkpoint) into the files the engine loads with `--pack`,
`--embd-gguf` and the PLE GGUF. They run once per model, at install time (setup runs them, see
[../install/README.md](../install/README.md)) or by hand. None of them is compiled into the engine. The file
formats they write are on [pack-format.md](pack-format.md).

| File | Lines | Role |
| --- | ---: | --- |
| [tools/iq_pack.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/iq_pack.py) | 602 | Native pack for any model file: index, float tensors, expert map |
| [tools/strata_pack.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/strata_pack.py) | 423 | Canonical Q2_0 pack: `build`, `verify`, `info` |
| [tools/pack_layer.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/pack_layer.py) | 123 | The canonical expert arena (1,382,400-byte blobs) |
| [tools/pack_index.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/pack_index.py) | 260 | Canonical `manifest.json` -> the engine's `index.txt` |
| [tools/canonical_xcheck.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/canonical_xcheck.py) | 799 | The canonical-form encoders/decoders (`MAPPINGS`) and their bit-exactness check |
| [tools/embd_bf16_pack.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/embd_bf16_pack.py) | 87 | BF16 token embedding from the checkpoint as a one-tensor GGUF |
| [tools/ple_fp8_pack.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/ple_fp8_pack.py) | 123 | The FP8 PLE n-gram table from the checkpoint as a one-tensor GGUF |

## `tools/iq_pack.py` - the native pack

### Command line

```
python tools/iq_pack.py --gguf <model>-00001-of-0000N.gguf --out <pack dir>
                        [--compat-bf16] [--experts-bin] [--skip-experts] [--base <canonical pack>]
```

| Flag | Effect |
| --- | --- |
| `--gguf` | shard 1; the other shards are found by the `-0000N-of-0000M` suffix and must all exist (`Model.__init__`, `iq_pack.py:144-168`) |
| `--out` | the pack directory (created) |
| `--compat-bf16` | allow rounding conversions to the engine's float forms (nearest even); writes `compat-bf16.json`. Refused together with `--base` (`iq_pack.py:505-506`) |
| `--experts-bin` | also write `experts.bin` (+ sidecar); without it the engine reads the experts from the GGUF |
| `--skip-experts` | rewrite the index only; do not write `experts.bin` |
| `--base` | reuse a canonical Q2_0 pack's `dense.bin` (hard link) for a model whose float tensors are byte-identical; needs every tensor in shard 1 (`iq_pack.py:555-560`) |

The path is taken with `.absolute()`, not `.resolve()`, so a Hugging Face snapshot symlink keeps its shard name
(`iq_pack.py:507-509`).

### Control flow (`main`, `iq_pack.py:493-598`)

```
Model(src): open every shard, check split keys, map every tensor name -> (GGUFFile, TensorInfo, memmap, path)
expert_layout(): build native_experts.txt text in memory          <- a model that cannot be packed is refused here,
                                                                     before any file of the pack changes
decide whether an existing experts.bin can be reused (size + sidecar + six sampled blobs)
index_standalone()  (or index_from_base() with --base)
   classify every non-expert tensor, refuse if any needs --compat-bf16 and it was not given
   write dense.bin.tmp, index.txt.tmp, conversions.json.tmp
   delete native_experts.txt, then rename the three into place
strata_tokenizer.py --gguf --out   (if tokenizer/vocab.json or chat_template.jinja is missing)
write native_experts.txt.tmp -> native_experts.txt                 <- completion marker, last
[--experts-bin] experts.bin.tmp -> experts.bin, then experts.bin.src.json
```

`check_split` (`iq_pack.py:175-198`) enforces llama.cpp's split keys as the engine does: shard 1 carries
`general.architecture`, every shard declares `split.count`, `split.no` and `split.tensors.count` consistently, and
the shards together hold `split.tensors.count` tensors. A single file whose metadata says it is shard 1 of N but
whose name has no `-00001-of-0000N` is refused.

`expert_layout` (`iq_pack.py:429-473`) takes the layer count from the highest `blk.N.ffn_*_exps.weight`, and the
expert count from the router `blk.0.ffn_gate_inp.weight`'s second dimension (a pruned model ships fewer than 512);
all routers must agree. Every layer needs gate, up and down `[*, *, n_expert]` tensors of whole blocks, gate and up
of the same type. The constant `N_EXPERT = 512` (`iq_pack.py:202`) is not used for this.

### How each non-expert tensor is classified (`index_standalone`, `iq_pack.py:274-360`)

Skipped: expert tensors and `per_layer_token_embd.weight` (the 28.8 GB PLE table, which the engine reads from its
GGUF, `NOT_IN_PACK`, `iq_pack.py:60`). A tensor with more than two dimensions stops the build.

| Tensor | Result |
| --- | --- |
| quantized, not in `FORM` | served natively from the GGUF: a shape-only index row |
| `blk.1.ple_key.weight` in Q2_0 or Q8_0 | served natively (the engine's native PLE key, `NATIVE_PLE_KEY`, `iq_pack.py:89`) |
| quantized, in `FORM` | needs `--compat-bf16` (dequantized through gguf-py, then converted); refused otherwise, before anything is written |
| float, in `FORM` with the same type, or float not in `FORM` | copied as stored |
| float, in `FORM` with another type | converted by `convert` (below) |

<a id="conversions"></a>
### Conversions (`convert`, `iq_pack.py:226-271`)

| Source -> form | Rule | Default? |
| --- | --- | --- |
| F32 -> BF16, every value's low 16 bits zero | exact: the top halves are written | yes |
| F32 with other values, F16, or a quantized type -> BF16 | ggml's round-to-nearest-even via gguf-py `quants.quantize(..., BF16)` (`bf16_bytes`, `iq_pack.py:118-126`) | only `--compat-bf16` |
| F32 -> F16 (`ple_conv1d`) | stored as F32, index kind 3, the loader narrows | yes |
| other -> F16 | numpy nearest even, refused on overflow or non-finite (`f16_values`, `iq_pack.py:129-138`) | only `--compat-bf16` |
| F16 / BF16 -> F32 | exact widening | yes |
| quantized -> F32 | dequantized | only `--compat-bf16` |

Non-finite values are refused in every conversion. Each conversion produces a record for `conversions.json`
with its maximum absolute error and the source bytes' sha256.

### Model- and platform-specific behaviour

- **GSQ-RCO files (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, the Coder's IQ1_M, Swift 1.5):** the float tensors are already in
  the engine's forms; no conversion. Swift 1.5's routers are F32 with exactly-BF16 values and are written as BF16
  (docstring, `iq_pack.py:31-33`).
- **Unsloth UD-Q4_K_XL / UD-IQ4_XS:** need `--compat-bf16` (setup passes it through the family's `pack_args`,
  `setup.py:215`, `4188-4189`). Upstream's own measurement for UD-Q4_K_XL: 264 F32 tensors converted exactly, 195
  Q8_0 projections rounded to BF16 (largest error 0.0144), the PLE conv narrowed to F16 (largest error 3e-8)
  ([docs/UNSLOTH_Q4.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/UNSLOTH_Q4.md)).
  The four-shard file has a layer whose roles straddle shards 2 and 3: that layer gets a v4 line.
- **Third-party quants (OrcaRouter IQ3_XXS, Q4_K_S):** documented in
  [docs/ORCA.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/ORCA.md) and
  [docs/ORCA_Q4_K_S.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/ORCA_Q4_K_S.md),
  both with `--compat-bf16`.
- **Low-RAM mode:** setup reruns the tool with `--experts-bin` (`setup.py:4190-4193`).
- **AMD HIP:** the mmap path needs the pack's `experts.bin`, so `--experts-bin` is added there
  ([docs/AMD_HIP.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/AMD_HIP.md)).

## Canonical Q2_0 pack: `strata_pack.py`, `pack_layer.py`, `pack_index.py`

Setup builds this pack only for Q2_0 of the original model on a CPU with AVX-512, where the AVX-512 expert kernel
reads the canonical blob layout (`setup.py:4175-4184`): "a one-time ~40 GB conversion".

```
python tools/strata_pack.py build  --gguf <Q2_0 shard 1> --out <pack> [--layers N] [--skip-hash]
python tools/strata_pack.py verify --gguf <Q2_0 shard 1> --out <pack> [--limit N] [--expert-every 32]
python tools/strata_pack.py info   --out <pack>
python tools/pack_index.py --pack <pack> [-o <index.txt>]
python tools/pack_layer.py [n_layers]          # standalone: the arena only, into <repo>/pack/
```

- **`build`** (`strata_pack.py:206-279`): points `pack_layer.SHARD1` at `--gguf` so the arena and the dense
  tensors come from the same file, builds `experts.bin` through `pack_layer.build`, then encodes every other tensor
  with `tensor_entry` (`strata_pack.py:107-159`) into `dense.bin`, except `token_embd.weight` which goes to
  `embd.bin`. A tensor type without a mapping, or a short read, stops the build. Shard 2 (the PLE table) is not
  copied; the manifest records where its tensor is. `--skip-hash` leaves out the sha256 of the shards (setup's way).
- **`verify`** (`strata_pack.py:338-380`): "a pack that has not passed `verify` does not exist" (docstring). It
  re-reads the pack from disk, decodes each tensor using only the manifest (`decode_entry`,
  `strata_pack.py:162-192`) and compares with `reference_values` from `canonical_xcheck.py` bit for bit. With no
  `--limit` and a recorded `shard1_sha256`, it first checks the shard's hash. It then checks every Nth expert of
  every layer in the arena with `rebuild_role` (`strata_pack.py:282-304`), written as an independent inverse of
  `pack_layer`'s layout.
- **`pack_layer.build`** (`pack_layer.py:68-115`): vectorized per layer over 512 experts; asserts every role is
  Q2_0 and 460,800 bytes per expert.
- **`pack_index.py`** (`pack_index.py:99-256`): flattens `manifest.json` into the 19-column `index.txt`. It refuses
  planes that are not contiguous and in codes/scales/offsets order, a `has_offset` tensor without an `offsets`
  plane, plane sizes that disagree with the manifest, and an engine form that is not the pack form plus the scale
  widening. It prints the pool size, the bytes saved by re-rounding BF16/F16-in-F32 tensors, and the bytes added by
  widening FP16 scales.

`canonical_xcheck.py` holds the specification of the canonical forms: per source type an encoder to codes,
scales and offsets and a decoder transcribed from ggml's `dequantize_row_*` (for example Q4_0's low nibbles fill the
first half of a block, and the `-8` is applied before the multiply, `canonical_xcheck.py:86-110`). Run on its own it
checks `decode(encode(block)) == reference(block)` for a sample (`--tensors`, `--blocks`) or for every block of
every tensor (`--full`, optionally split across processes with `--split I/N`). The reference is gguf-py for every
type except Q2_0, whose reference is the project's own decoder in `gguf_writer.py`, and the tool says so
(`reference_name`, `canonical_xcheck.py:595-604`). The default shard path is `$STRATA_SHARD1` or the development
layout (`canonical_xcheck.py:45-47`).

## `tools/embd_bf16_pack.py` and `tools/ple_fp8_pack.py`

Both read the original BF16 checkpoint directory (`model.safetensors.index.json`) and copy one tensor's bytes, in
64 MiB chunks, into a hand-written one-tensor GGUF, via a `.part` file renamed at the end after a size check.

- `embd_bf16_pack.py --model <dir> --out <file.gguf>`: the text `embed_tokens.weight` (not the visual or MTP one),
  which must be 2-D BF16. The engine takes it with `--embd-gguf`; its docstring says the BF16 table costs 0.6 GB of
  RAM more than the Q8_0 table written by an `nvfp4_convert.py` (not in this repository) and no VRAM
  (`embd_bf16_pack.py:5-6`).
- `ple_fp8_pack.py --model <dir> --out <file.gguf>`: the 128 shards `...ngram_embedding.shard_{k}.weight` (FP8
  E4M3, `[2500012, 160]` each) in shard order, plus the BF16 `weight_scale` stored as `strata.ple.scale`. The
  docstring gives 320,001,536 rows of 160 values, 51.2 GB, and says ISTA-DASLab's IQ4_NL table is 8% off these
  values per row (`ple_fp8_pack.py:5-8`). The engine's `PleTable` reads it (`src/kernels/ngram.cpp:158`, `197`).

Setup does not run either of these two (no call in `setup.py`).

## Configuration

| Setting | Used by |
| --- | --- |
| `STRATA_GGUF_PY` | `iq_pack.py` (through `_paths.add_gguf_py`, only when dequantizing or converting to BF16) |
| `STRATA_SHARD1` | `canonical_xcheck.py` default `--gguf` |
| `PYTHONPATH` with gguf-py | `canonical_xcheck.py` and `strata_pack.py verify` import `gguf` directly for F32/F16/BF16 references and through `ref/load.py` for the rest; nothing adds it to `sys.path` |

## Tests

- `tools/test_iq_pack.py` (528 lines): synthetic GGUFs written with gguf-py's `GGUFWriter` (arch `qwen4exp`).
  Covers BF16 halfway rounding to even, refused non-finite conversions, projection conversions and native bytes,
  inexact F32 routers refused by default, a pack of a BF16 file staying byte-identical, quantized control weights
  needing the flag, Q2_0 PLE key staying native, `experts.bin` reuse only with this GGUF's blobs, snapshot
  symlinks, the conversion records, four shards with a per-role split layer (v4), unsplit layers staying v3,
  missing/duplicate/foreign/truncated shards, a wrong type leaving the pack untouched, an interrupted pack and a
  stale same-size `experts.bin`. Run:
  `python -m unittest discover -s tools -p test_iq_pack.py` (needs numpy and gguf-py).
- `tools/test_shards.py` (119 lines): `strata_pack.verify` reading back the source hash on a minimal hand-written
  GGUF (matching hash, wrong hash naming both, no hash or `--limit` skipping it), plus setup's `check_shards`.
  It imports `setup.py`. Run: `python -m unittest tools.test_shards`.
- CTest `expert_layout_test` reads the golden `native_experts.txt` copies in `tests/data/native_experts/`.
- No unit test covers `pack_layer.py`, `pack_index.py`, `canonical_xcheck.py`, `embd_bf16_pack.py` or
  `ple_fp8_pack.py`; `canonical_xcheck.py` and `strata_pack.py verify` are themselves checks against a real shard.

## Gotchas

- `setup.py:123-124` says Swift 1.5's Q2_0 files split one layer's experts across shards, "which the pack tool
  (tools/iq_pack.py) cannot prepare yet (#171)", so setup offers Q2_0 for the original model only; `iq_pack.py`'s
  docstring (lines 26-31) describes the v4 per-role layout that handles such a split.
- `pack_layer.py` hardcodes Flash-Next's geometry (H 2560, FF 640, 512 experts) and Q2_0; `strata_pack.build`
  defaults to 48 layers and derives shard 2's name by replacing `00001-of-00002` in shard 1's.
- `pack_index.py` starts with a UTF-8 byte-order mark (bytes EF BB BF).
- `strata_pack.py` and `pack_layer.py` keep default shard paths from the development layout (`<Strata>/Q2_0/...`);
  setup always passes `--gguf`.
- `pack_index.py`'s docstring mentions `tools/pack_budget.py`, which is not in the published tree.

## In UnbiasedStrata

`iq_pack.py` was removed in fork commit 8894a81, restored unchanged in c00942d, then changed in e41dfc7 for
Qwen3.6-35B-A3B (a `qwen35moe` FORM table, native-projection type checks, the `-MTP` draft block left out, the
profile copied in). `strata_pack.py`, `pack_layer.py` and `pack_index.py` were restored unchanged, but
`canonical_xcheck.py` was not, so the fork's `strata_pack.py` cannot import. `embd_bf16_pack.py`,
`ple_fp8_pack.py` and `test_shards.py` were removed. See the fork's [pack-builders.md](../../codebase/tools/pack-builders.md).

## See also

- [pack-format.md](pack-format.md), [gguf-and-tokenizer.md](gguf-and-tokenizer.md),
  [reference-loader.md](reference-loader.md), [python-tests.md](python-tests.md)
- [../install/README.md](../install/README.md) (setup's step 6, which runs these tools)
