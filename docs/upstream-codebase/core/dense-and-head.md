[Upstream docs](../README.md) › [Core](README.md) › Dense projections and head

# Native dense projections, the native head and the native embedding

## Purpose

The pack holds every dense tensor in Strata's canonical S-form ([weights-and-layout.md](weights-and-layout.md)).
With `--native SHARD1` the engine instead serves the big projections, the output head and the token embedding
straight from the model's GGUF blocks, unchanged, through ggml-derived kernels (`native_mmvq`, `iq_embed_rows`). The
canonical copies of those tensors are then skipped by the loader, which gives VRAM back to the expert cache (the
`generate.cpp:2183` comment says about 2.7 GB). Setup always starts the engine this way (`setup.py:4224`), for the
canonical Q2_0 pack as well as for native packs.

Files covered:
[include/strata/core/native_dense.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/native_dense.hpp),
[src/core/native_dense.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/native_dense.cpp),
[include/strata/core/native_head.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/native_head.hpp),
[src/core/native_head.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/native_head.cpp).

## `NativeDense` (GGUF-form projections)

| Function | Lines | What it does |
| --- | --- | --- |
| `eligible(tensor, include_ple_key)` | `native_dense.cpp:24-36` | A `blk.*` tensor ending in `.attn_qkv`, `.attn_gate`, `.ssm_out`, `.attn_q`, `.attn_k`, `.attn_v`, `.attn_output`, `.ffn_gate_shexp`, `.ffn_up_shexp` or `.ffn_down_shexp`. `blk.1.ple_key.weight` only with `include_ple_key` and type Q2_0 (42), IQ3_XXS (18), IQ4_XS (23) or Q8_0 (8, Unsloth's UD-Q4_K_XL) |
| `served_names(shards, include_ple_key, out, err)` | `:47-62` | Names `load` would serve (eligible, `native_mmvq_supported(type)`, 2-D), from the GGUF headers only, so the loader can skip their canonical bytes |
| `set_layer_range(lb, le)` | `:64` | Process-wide filter for the next `load`: a layer split's stage uploads only its own `blk.N.` projections (PLE tensors everywhere) |
| `keep_unquantized_ple_key(pack, skip, err)` | `:65-73` | #326: if the pack's `blk.1.ple_key.weight` row is unquantized (an `iq_pack --compat-bf16` key, e.g. OrcaRouter's IQ3_XXS), it is removed from `skip` so the arena keeps it |
| `load(shards, table, err, include_ple_key, layer_lo, layer_hi)` | `:80-209` | See below |

`load` checks every shard before uploading anything: the first shard's architecture (`check_architecture`), that
additional shards carry matching `split.count` / `split.tensors.count` metadata and distinct `split.no`, that no two
tensor payloads start at the same offset, and that every tensor's span (including tensors it will not upload) fits
the payload and does not overlap the next one (`:97-153`). Then for each eligible tensor in range it requires a
canonical row of the same shape (`ne0`, `ne1`) that is quantized, uploads the raw blocks with one `cudaMalloc` each,
and only after every check and allocation has succeeded publishes `native_data`, `native_type` and one shared Q8_1
scratch (`native_q8_1`) into the `WeightRef`s (`:194-201`). An unsupported type keeps the canonical path.

The shared scratch is why the header (`native_dense.hpp:11-15`) requires one ordered session stream: every native
projection quantizes its input into the same buffer. `gemv_quantized` in `layer.cpp:140-158` dispatches on
`native_data`: `native_quantize_q8_1` then `native_mmvq`, with an `x_q8_1_ready` shortcut when the previous native
projection already quantized the same `x` (attn_gate after attn_qkv, attn_q after attn_v).

## `NativeHead` (output projection)

* `load(shards, n_in, n_out, err)` (`native_head.cpp:19-69`): opens all shards as one `GgufModel`, checks the
  architecture on the metadata shard, and finds `output.weight` in whichever shard holds it (shard 2 of Unsloth's
  UD-Q4_K_XL, whose shard 1 holds no tensor, `:26-27`). The type must be one `native_mmvq` supports and the shape
  `[n_in, n_out]` with `n_in % 256 == 0`. One `cudaMalloc` for the blocks and one for a Q8_1 scratch.
* `run(mixed, logits, stream, err)` (`:71-93`): type 13 (Q5_K, the head of the Q2_0 and IQ3_XXS files) goes through
  `native_q5_k_f32`; any other type through `native_quantize_q8_1` + `native_mmvq`. The header names IQ4_XS for
  IQ2_XS (`native_head.hpp:57`).
* `weights()`, `type()`, `row_bytes()` let the verify window run a multi-column head and the MTP drafter gather a
  token subset (`draft_vocab.bin`) of it.

Callers: `generate.cpp:3030` loads it (on one GPU; a layer split loads it on its last stage) before the expert cache
is sized and before the host arena is registered (the comment at `:3019-3024` explains both: a head loaded after the
cache ate the reserve, and WDDM refused the allocation once tens of GiB were registered, #620). The plain decode loop
calls `run` at `generate.cpp:3872`; the verifier and the drafter use `weights()` directly.

## `NativeEmbed` (token embedding)

* `load(shards, n_embd, n_vocab, err)` (`native_head.cpp:108-188`): `token_embd.weight` from the model (or from a
  one-tensor `--embd-gguf` file whose architecture is `strata-embd`, written by `tools/embd_bf16_pack.py`). The type
  must have a GPU dequantizer (`embed_type_supported`), the payload must be complete. The table is copied into
  **mapped pinned host memory** (`cudaHostAllocMapped | Portable`), so each token reads one row over PCIe and the
  table costs no VRAM.
* Fallbacks: if the pinned allocation fails (the WSL2 pinned budget of about 1 GiB, `:134-149`) the table goes into
  VRAM instead. On HIP, if `cudaHostGetDevicePointer` gives no alias (#325, the Windows HIP stack), the table also
  goes to VRAM (`:153-171`).
* `gather_dev(tokens, n, out, stream)` (`:190-192`) for device token ids (verify window, MTP, prompt path);
  `gather_one(token, out, stream)` (`:194-196`) for a host id.
* `set_native_embed(e)` / `native_embed()` (`:100-101`): a process-wide pointer. `embed_row` in `layer.cpp:1033-1038`,
  the verify window and the drafter use it when it is set.

## Configuration

| Flag | Effect |
| --- | --- |
| `--native SHARD1` | Preset: native dense (every shard plus the PLE shard), native head from the model, native embedding, and the native kernel switches (`generate.cpp:1802-1822`) |
| `--native-head-gguf PATH`, `--native-dense-gguf PATH` (repeatable) | The individual switches the preset sets (A/B use) |
| `--embd-gguf PATH` | The embedding from another GGUF (BF16 as the checkpoint ships it) |
| `--keep-canonical` | Load canonical forms too (no skip set) |

## Model and platform notes

* Q2_0 / IQ3_XXS GGUFs: Q5_K head; IQ2_XS: IQ4_XS head (`native_head.hpp:57`).
* UD-Q4_K_XL: `output.weight` in shard 2; Q8_0 PLE key served natively; the pack converts its Q8_0
  hyper-connection projections to BF16 (`setup.py:4186`).
* OrcaRouter IQ3_XXS packed with `--compat-bf16`: the PLE key stays in the arena (#326).
* HIP on Windows: embedding table in VRAM when there is no mapped alias.

## Tests

`native_dense_ple_key_test` (`tests/core/native_dense_ple_key_test.cpp`, `CMakeLists.txt:446-449`, needs
`STRATA_NATIVE_EXPERTS` and a CUDA device) checks the #326 rule. Kernel parity for `native_mmvq` and friends belongs
to the kernels area (`native_mmvq_multi`, `mmvq_multi_parity`).

## Gotchas

* All native projections share one Q8_1 scratch; running two sessions' graphs on different streams with the same
  `NativeDense` would race.
* `NativeDense` must outlive every graph that references its weights (header comment).
* `NativeHead::load` refuses a second load; it does not reload on a model change.

## In UnbiasedStrata

Kept unchanged (`native_dense.*`, `native_head.*`). The fork's generate bounds the native projections to the trunk's
layers (commit e41dfc7) by calling `load` with `layer_hi`. Fork page:
[dense-and-head.md](../../codebase/core/dense-and-head.md).

## See also

[weights-and-layout.md](weights-and-layout.md) · [layer-forward.md](layer-forward.md) · [verify.md](verify.md) ·
[mtp-draft-head.md](mtp-draft-head.md)
