[Upstream docs](../README.md) › [Tools](README.md) › Reference loader (`ref/`)

# The reference loader: `ref/load.py`

## Purpose

`ref/` is where upstream keeps its Python reference ("oracle") code. In the published tree it holds one file,
`ref/load.py`: a demand-paged reader that turns tensors of a (possibly multi-shard) GGUF into FP32 numpy arrays,
whole, per expert or per row, without materializing the model (`ref/load.py:1-16`). It is used by
`tools/canonical_xcheck.py` as the reference decoder for the canonical pack (and through it by
`tools/strata_pack.py verify`).

| File | Lines | Role |
| --- | ---: | --- |
| [ref/load.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/ref/load.py) | 212 | `Gguf`: lazy GGUF -> FP32 tensors, experts and rows |

## Main pieces

| Name | Lines | What it does |
| --- | --- | --- |
| `GEOM` | 35-38 | (elements, bytes) per block for the 13 types it decodes: F32, F16, BF16, Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, Q2_0 |
| `_dequant_flat(type, raw)` | 41-57 | one dispatch: Q2_0 through `tools/gguf_writer.dequantize_q2_0`, BF16/F32/F16 by hand, every other type through gguf-py's `quants.dequantize` |
| `_discover_shards(path)` | 64-105 | a list, a directory (the `-NNNNN-of-MMMMM` shard set, or a single `.gguf`; refuses to guess between several unrelated files such as an `mmproj`), or one shard whose siblings must all exist |
| `Gguf(path)` | 108-127 | opens every shard, maps tensor name -> shard; duplicates across shards refused; `g` is shard 1's `GGUFFile` (metadata) |
| `Gguf.tensor(name)` | 144-153 | whole tensor, shape reversed to (out, in[, experts]) |
| `Gguf.rows(name, idx)` | 156-183 | selected rows of a 2-D tensor, decoding only those rows; written for the PLE table (`[160, 320001536]` IQ4_NL, 26.82 GiB, read 16 rows per token) |
| `Gguf.expert(name, e)` | 186-208 | one expert of a fused `[in, out, experts]` tensor: a contiguous run, decoded alone, as (out, in) |
| `Gguf.close()` | 210-212 | closes the shard files |

The split between decoders is deliberate: Q2_0 cannot be represented by the gguf-py of the time, so it uses the
project's own decoder, and "a bug in `tools/gguf_writer.dequantize_q2_0` cannot hide behind gguf-py and vice versa"
(`ref/load.py:8-11`).

## Data flow

```
GGUF shards ──GGUFFile (tools/gguf_reader.py)──► tensor directory per shard
            ──seek + read only the needed bytes──► _dequant_flat ──► FP32 numpy
                                                     ├─ Q2_0: tools/gguf_writer.dequantize_q2_0
                                                     └─ others: gguf-py quants.dequantize
callers: tools/canonical_xcheck.py (reference_values) ◄── tools/strata_pack.py verify
```

## Configuration

`ref/load.py` adds `tools/` to `sys.path` (for `gguf_reader` and `gguf_writer`) but not gguf-py: `import gguf` must
resolve through `PYTHONPATH` (or an installed package) for any type except Q2_0, F32, F16 and BF16.

## What upstream does not publish

The engine's comments cite a larger Python oracle that is not in the tree: `ref/gdn.py`, `ref/qsa.py`,
`ref/moe.py`, `ref/model.py`, `ref/ngram.py`, `ref/quant.py` and `ref/gr.py` are each named several times in `src/`
and `include/` (for example the GDN parity test transcribes "ref/gdn.py"), but only `ref/load.py` exists at
6f32ec0. The parity tests therefore carry their own C++ transcriptions of the reference.

## Model-specific notes

- Built for Flash-Next's GGUFs: the PLE row path and the two-shard discovery are motivated by its 26.82 GiB
  `per_layer_token_embd.weight` in shard 2 (`ref/load.py:64-74`, `156-166`).
- `GEOM` lacks Q4_1, Q5_1, Q2_K and the IQ1/IQ2/IQ3 types, so it cannot decode the i-quant experts of the IQ2_XS,
  IQ3_XXS, IQ3_S or Coder files; it covers the canonical pack's types.

## Tests

No direct unit test. `canonical_xcheck.py` exercises `_dequant_flat` against a real shard.

## Gotchas

- `tensor()` on the PLE table would decode about 205 GB of FP32 (`ref/load.py:161-162`); use `rows()`.
- A directory argument with several unrelated `.gguf` files and no shard set raises instead of guessing.

## In UnbiasedStrata

Removed in fork commit 8894a81, restored in c00942d, and extended in e41dfc7 with the missing block types (Q4_1,
Q5_1, Q2_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M) for Qwen3.6's UD files. The fork also adds
`ref/qwen36.py`, a numpy oracle of the Qwen3.6 graph. See the fork's
[reference-model.md](../../codebase/tools/reference-model.md).

## See also

- [pack-builders.md](pack-builders.md) (`canonical_xcheck.py`, `strata_pack.py verify`),
  [gguf-and-tokenizer.md](gguf-and-tokenizer.md)
