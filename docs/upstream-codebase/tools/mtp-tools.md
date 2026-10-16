[Upstream docs](../README.md) › [Tools](README.md) › MTP draft-layer tools

# MTP draft-layer tools

## Purpose

The engine drafts tokens with the model's own MTP (multi-token prediction) layer and checks them in one pass of the
main model (speculative decoding; see the [glossary](../guides/glossary.md)). The GSQ-RCO GGUFs ship no MTP layer
(`tools/mtp_fetch.py:3`), so three tools build it from the original BF16 checkpoint, without downloading the
checkpoint:

```
huggingface.co/Qwen/Qwen3.8-Flash-Next  (BF16, 131 shards, 360 GB; 31 mtp.* tensors in 28 shards)
   │ mtp_fetch.py fetch --out <mtp>        HTTP range reads of the shard headers, then of the mtp.* tensors only
   ▼
<mtp>/tensors/*.bin + mtp-manifest.json   (raw BF16, sha256 checked)
   │ mtp_pack.py --src <mtp> --experts q2_0 --out <mtp>/mtp-q2_0.gguf
   ▼
MTP GGUF (arch qwen4exp-mtp): dense BF16/F32, routed experts Q2_0 | Q4_0 | Q8_0
   │ mtp_rt.py --gguf <mtp>/mtp-q2_0.gguf --out <mtp>/rt
   ▼
<mtp>/rt/{experts.bin, dense.bin, dense.txt}  + draft_vocab.bin copied from data/ by setup
   │ strata --mtp <mtp>/rt
   ▼
src/core/mtp.cpp
```

Setup runs exactly this chain with `--experts q2_0` (`setup.py:4202-4208`). The same draft layer serves every
model variant: the original's GSQ-RCO sizes, the Coder, Swift 1.5 and the Unsloth files (docs/ORCA.md and
docs/UNSLOTH_Q4.md say so for the third-party and Unsloth files: "the original model's draft head, as setup does
for Swift").

| File | Lines | Role |
| --- | ---: | --- |
| [tools/mtp_fetch.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/mtp_fetch.py) | 308 | `inventory`, `fetch`, `verify` of the `mtp.*` tensors, pinned revision and sha256 |
| [tools/mtp_pack.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/mtp_pack.py) | 193 | MTP tensors -> MTP GGUF, routed experts quantized |
| [tools/mtp_rt.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/mtp_rt.py) | 111 | MTP GGUF -> the engine's runtime files |
| [tools/test_mtp_fetch.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_mtp_fetch.py) | 201 | Tests of `mtp_fetch.py` with a mocked HTTP server |

## `tools/mtp_fetch.py`

```
python tools/mtp_fetch.py inventory --out DIR        # shard headers only
python tools/mtp_fetch.py fetch     --out DIR [--only SUBSTR]   # resumable
python tools/mtp_fetch.py verify    --out DIR        # offline; exit 3 if a tensor is missing or wrong
```

- **Pinned source.** `PINNED_REVISION` = `de4b8e4d43b917e7706784d8bb445c9af86a3540` (the checkpoint's commit on
  2026-09-30, #214, `mtp_fetch.py:29-33`). `STRATA_MTP_REVISION` overrides it; `HF_ENDPOINT` selects a mirror
  (#495). When the pinned revision is gone (a 404 on its index), `resolve_repo` falls back to `main` with a message
  (`mtp_fetch.py:153-167`).
- **Range reads must be honoured** (#327): `get` requires HTTP 206 and a `Content-Range` that starts with the asked
  range, and the exact length, else it retries (4 attempts, exponential back-off) and raises (`check_range`,
  `mtp_fetch.py:110-137`). A mirror or proxy that ignores `Range` would otherwise deliver the shard's start, and the
  drafter would accept nothing.
- **Per-tensor sha256.** `SHA256` holds all 31 tensors' hashes at the pinned revision (`mtp_fetch.py:44-107`).
  `fetch` keeps a partial file and resumes it with `ab`; after each tensor it hashes the file and, if wrong, deletes
  it and fetches it once more, then stops with "the download source sends wrong data" (`mtp_fetch.py:224-254`).
  An inventory from another repository URL or revision is read again, and tensors from there are kept only when
  they match the pinned hashes (`mtp_fetch.py:217-231`).
- **`verify`** (`mtp_fetch.py:259-289`) hashes `tensors/*.bin` against `SHA256`, caches verdicts by size and mtime
  in `tensors/verified.json`, and returns the bad names; nothing is checked when another revision is configured.
  Setup's `mtp_corrupt` runs it and rebuilds the draft layer on exit code 3 (`setup.py:3355-3364`, `4195-4208`).

## `tools/mtp_pack.py`

```
python tools/mtp_pack.py --src <fetch dir> --experts q2_0|q4_0|q8_0 --out <file.gguf> [--check-experts 8]
```

Reads `mtp-manifest.json` and writes a GGUF through llama.cpp's gguf-py (`gguf.GGUFWriter`, arch `qwen4exp-mtp`):

- 1-D tensors (norms) as F32, other dense tensors as BF16 (docstring: ~0.18 GB).
- The fused routed experts `mtp.layers.0.mlp.experts.gate_up_proj` [512, 1280, 2560] and `down_proj`
  [512, 2560, 640], quantized along the last (input) axis, streamed one expert at a time from a read-only memmap
  (the fused gate_up alone is 6.7 GB as float32) (`mtp_pack.py:154-172`):

| `--experts` | Quantizer | Size (docstring) |
| --- | --- | --- |
| `q2_0` (default) | per 64-element block, the scale that minimizes squared error over the grid {-1, 0, 1, 2} x d, searched over 17 steps from amax/2 to amax, codes computed against the stored FP16 scale (`mtp_pack.py:58-82`) | ~0.71 GB |
| `q4_0` | ggml's reference rounding (`mtp_pack.py:85-97`) | ~1.42 GB |
| `q8_0` | amax / 127 (`mtp_pack.py:100-108`) | ~2.67 GB |

It reports the relative RMS error of the first `--check-experts` experts per tensor and writes
`<out>.report.json`. This is round-to-nearest, not GSQ; the docstring says the format is chosen by measured draft
acceptance, not by this error (`mtp_pack.py:18-19`).

## `tools/mtp_rt.py`

```
python tools/mtp_rt.py --gguf <mtp-q2_0.gguf> --out <rt dir>
```

Reads the MTP GGUF with gguf-py's `GGUFReader` and writes:

- `experts.bin`: 512 blobs in the engine's expert blob layout - gate/up rows interleaved, then down rows, the Q2_0
  codes in one plane and the FP16 scales in another (`blob_of`, `mtp_rt.py:48-58`), the same 1,382,400-byte layout
  as the canonical pack ([pack-format.md](pack-format.md#21-expert-blob-pack_layerpy)). It asserts the expert
  tensors are Q2_0-shaped (`(512, 1280, 720)` and `(512, 2560, 180)` bytes, `mtp_rt.py:70-72`), so only a
  `--experts q2_0` GGUF can be converted.
- `dense.bin` and `dense.txt`: ten large projections (`fc_embedding`, `fc_hidden`, q/k/v/o, the indexer's
  `index_qk_proj`, the shared expert's three) quantized to Q8_0 for the engine's multi-column MMVQ kernel (`Q8`,
  `mtp_rt.py:29-31`); F32 norm weights written as `1 + w` (Gemma-style RMSNorm, as vLLM's `GemmaRMSNorm`); every
  other tensor BF16 (`mtp_rt.py:76-103`). Each tensor is 256-byte aligned.

The engine loads the directory in `src/core/mtp.cpp` (`dense.txt` at line 164, `experts.bin` at 194,
`draft_vocab.bin` at 437) and fails with "run tools/mtp_rt.py" when `dense.txt` is missing.

## Configuration

| Setting | Used by |
| --- | --- |
| `STRATA_MTP_REVISION` | `mtp_fetch.py`: another checkpoint revision (hash checks then off) |
| `HF_ENDPOINT` | `mtp_fetch.py`: a Hugging Face mirror; hashes still apply |
| `STRATA_GGUF_PY` | `mtp_pack.py`, `mtp_rt.py` (gguf-py through `_paths.add_gguf_py`) |
| `--draft-vocab cjk|en|cyrillic|fr` (setup) | which `data/draft_vocab*.bin` setup copies into `rt/` ([profile-and-draft-vocab.md](profile-and-draft-vocab.md)) |

## Model- and platform-specific notes

- Everything is Flash-Next-specific: tensor names, the 512-expert fused shapes, H = 2560 and FF = 640 are
  hardcoded (`mtp_pack.py:38`, `mtp_rt.py:27-31`).
- Measured by upstream, in the Intel port's notes: the draft layer built by these tools is 4.9 GB downloaded and
  809 MiB of VRAM, and drafts for the Coder fine-tune at 87% acceptance
  ([docs/INTEL.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INTEL.md#L172-L175)).
  The install requirements give ~6 GB of disk for the MTP layer ([docs/DETAILS.md](../../upstream/DETAILS.md),
  "Before you start").

## Tests

`python -m unittest tools.test_mtp_fetch` (from the repository root): a fake checkpoint behind a mocked
`urlopen`, nothing downloaded. Covers `check_range`, a mirror that ignores `Range` being refused, a fetch from an
honest mirror, wrong bytes fetched again once then refused, an inventory from another revision read again,
`verify`'s cached verdicts and its skip for other revisions, a hash for every tensor, and setup's
`mtp_corrupt` (which imports `setup.py`). `mtp_pack.py` and `mtp_rt.py` have no unit tests.

## Gotchas

- `mtp_fetch.py` hardcodes 31 hashes; a revision other than the pinned one gets no integrity check at all.
- `mtp_rt.py` accepts only a Q2_0 expert GGUF; the `q4_0` and `q8_0` arms of `mtp_pack.py` are for acceptance
  comparisons (`mtp_pack.py:7`).
- `mtp_pack.py` imports gguf-py at module import time, so even `--help` needs it.

## In UnbiasedStrata

All three tools and `test_mtp_fetch.py` were removed in fork commit 8894a81 and restored unchanged in c00942d; at
the fork's HEAD they are identical to upstream, but `test_mtp_fetch.py`'s setup test imports the removed
`setup.py`. The fork adds `tools/mtp_qwen36_pack.py` (commit 7f92fa8) for Qwen3.6's draft block. See the fork's
[mtp-tools.md](../../codebase/tools/mtp-tools.md).

## See also

- [pack-format.md](pack-format.md#3-mtp-draft-layer), [profile-and-draft-vocab.md](profile-and-draft-vocab.md)
- [../install/README.md](../install/README.md) (setup's step 6)
