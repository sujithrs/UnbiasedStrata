[Upstream docs](../README.md) › [Core](README.md) › Weights and layout

# Model geometry, layer layout and the dense-weight loader

## Purpose

Three things every other core page relies on: the model's fixed geometry (`ModelGeometry`), the per-layer view of the
pack by tensor name with its shape checks (`LayerView`, `check_layer`, `check_all`), and the loader that puts the
pack's dense (non-expert) tensors into one VRAM arena in the form the kernels read (`WeightTable`).

Files covered:
[include/strata/core/layout.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/layout.hpp),
[src/core/layout.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/layout.cpp),
[include/strata/core/weights.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/weights.hpp),
[src/core/weights.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/weights.cpp).

## `ModelGeometry`: fixed Flash-Next constants

`struct ModelGeometry` (`layout.hpp:27-59`) is a plain struct of defaults, "taken from `docs/semantics.md` and the
artifact's own metadata". There is no reader that fills it from a GGUF.

| Group | Fields (default) |
| --- | --- |
| Model | `n_embd` 2560, `n_layers` 48, `qsa_interval` 4 |
| GDN (36 layers) | `ssm_state_size` 128, `ssm_k_heads` 16, `ssm_v_heads` 48, `ssm_d_conv` 4, `ssm_conv_channels` 10240 (2·128·16 + 128·48), `ssm_value_dim` 6144 |
| QSA (12 layers) | `n_head` 24, `n_head_kv` 2, `head_dim` 256, `idx_q_heads` 4, `idx_key_dim` 128 |
| Hyper-connections (every layer) | `hc` 4, `hc_lr` 320 |
| MoE (every layer) | `n_expert` 512, `n_ff` 640 |

Helpers: `hc_dim()` = `hc * n_embd` (10240); `n_qsa_layers()` = `n_layers / qsa_interval` (12); `n_gdn_layers()`
(36). `is_qsa_layer(g, l)` (`layout.hpp:63-65`) is `l % 4 == 3`: layers 3, 7, ..., 47 are QSA.

What does change at run time:

* `generate.cpp:2060` builds `g` from the defaults. With `--native`, `generate.cpp:2073-2079` reads the metadata shard
  and overrides `n_expert` from `qwen4exp.expert_count` and the router's `K` from `qwen4exp.expert_used_count` (the
  comment names the GSQ-RCO Coder, 256 experts). The rope keys are read there too, into the rope config rather than
  into `ModelGeometry`.
* Other widths the QSA kernels need (`n_rot` 64, `idx_block`, `idx_top_k`, `page_size`) come from
  `strata::kernels::qsa_real_shapes()`; `qsa_shapes(g)` in `layer.cpp` and `stage_token` in `session.cpp:177-189`
  copy the five `ModelGeometry` attention widths over it.
* The MTP drafter always gets a default-constructed `ModelGeometry` (`generate.cpp:3013`), so it keeps 512 experts
  even under the Coder.

The `k` the session uses is separate from the geometry (`SessionState::k`, default 10).

## `LayerView` and the shape checks

`LayerView(table, layer)` (`layout.hpp:72-83`) resolves `"blk.<layer>." + suffix` through `WeightTable::find`
(`layout.cpp:137-139`). A null result means the tensor is absent, which is expected (a GDN layer has no `attn_q`).

`check_layer(table, g, layer, err)` and `check_all(table, g, err)` (`layout.cpp:141-171`) assert, at load time,
every shape the kernels depend on and name the tensor on a mismatch:

* 2-D shapes (`layout.cpp:50-80`, `ne0` is the contiguous axis): the six hyper-connection tensors
  (`hc_attn_down` `[hc_dim, hc_lr]`, `hc_attn_up` `[hc_lr, hc_dim]`, `hc_attn_inject` `[hc_dim, hc]`, the same for
  `hc_ffn_*`), the router `ffn_gate_inp` `[2560, 512]`, the shared expert's gate/up/down, the six GDN projections,
  and on QSA layers `attn_q` `[2560, 2·24·256]`, `attn_k`, `attn_v`, `attn_output` and the two indexer projections.
* 1-D tensors with their **engine form** (`layout.cpp:96-107`): all F32 except `ffn_gate_inp_shexp.weight`, which
  is BF16 (`WeightKind::Bf16InF32`, 2 bytes per element in the arena). The comment at `layout.cpp:19-27` records
  the bug this catches: reading that tensor as f32 made every MoE output non-finite.
* `check_all` also counts the QSA and GDN layers against `n_qsa_layers()` / `n_gdn_layers()`.

The comments at `layout.hpp:7-11` and `layout.cpp:76-77` name the bug behind the indexer rows: the indexer key
store was once sized from the query head count (4), which made it 4 times too big.

Callers: in this snapshot `check_all` and `check_layer` have no caller in `src/program/` or `src/prefill/`; the only
`check_all` hits outside `layout.cpp` are an unrelated local function of the same name in
`src/kernels/s2_expert_grouped_parity.cpp`. The per-layer functions in `layer.cpp` repeat the essential checks at
use (for example the BF16 form of `ffn_gate_inp_shexp` in `moe_shared`).

## `WeightTable`: the dense-weight loader

### The index

`tools/pack_index.py` (canonical pack) or `tools/iq_pack.py` (native pack) writes `<pack>/index.txt`. A header
`# align A pool P tensors N`, then one space-separated row of 19 fields per tensor (parsed at
`weights.cpp:154-193`):

```
name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems codebook
has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind
```

A row with fewer than 19 fields is refused ("a wrong byte offset waiting to happen", `weights.cpp:172-180`).
`file` selects `dense.bin` (0), `embd.bin` (1), `experts.bin` (2) or `extra.bin` (3, a native pack's float
tensors; `weights.cpp:64-72`).

### Engine forms (`WeightKind`, `weights.hpp:41-46`)

| Index kind | Engine form | Transform in the loader |
| --- | --- | --- |
| 0 `Verbatim` | quantized planes or F16 | copied; a quantized tensor is split into codes / scales / offset segments, and an fp16 scale plane is **widened to f32** (`Conv::WidenF16`, exact) |
| 1 `Bf16InF32` | BF16 | the pack holds the value promoted to f32; the loader keeps the high 16 bits (exact) |
| 2 `F32` | F32 | copied |
| 3 `F16InF32` | F16 | real f32 -> f16 rounding |
| 4 / 5 (native packs) | BF16 / F16 | already 16-bit in the file (`raw16`), copied, then labelled as kinds 1 / 3 (`weights.cpp:182-185`) |

The header comment (`weights.hpp:3-19`) gives the budget this buys: the pack's 5.305 GiB of dense weights become
4.067 GiB in the arena, plus 0.444 GiB of embedding and 10.88 MiB of widened scales (4.522 GiB in all, for the
canonical Q2_0 pack). 90 of 303 quantized tensors carry fp16 scales in the pack.

### `WeightRef` (`weights.hpp:48-96`)

Device pointer `data`, `bytes`, shape (`ne0` contiguous, `ne1`), `elements`, `kind`; for quantized tensors the
canonical S-form attributes (`code_bits`, `code_bias`, `group_elems`, `codebook_iq4nl`, `has_offset`) and the
three plane sizes **as loaded** (`codes_bytes`, `scales_bytes` always 4 B per scale, `offset_bytes`); the source span
(`src_off`, `src_bytes`, `file_id`, `scales_fp16`) so tests can check the loader against the pack; `act_kind` (0 =
the weight wants a Q8_0 activation, 1 = Q8_K; the S-form cannot tell Q5_0 from Q5_K, `weights.hpp:80-84`); and the
native fields `native_data`, `native_q8_1`, `native_type` that `NativeDense` fills ([dense-and-head.md](dense-and-head.md)).
`resident` is false for a tensor the loader skipped because another form serves it.

### The functions

| Function | Lines | What it does |
| --- | --- | --- |
| `WeightTable::pool_bytes(pack, out, err, skip)` | `weights.cpp:89-119` | The arena size from the header without loading; with a `skip` set, the size of the compacted arena (kept rows re-aligned at the index's alignment) |
| `WeightTable::index_code_bits(pack, name, bits, err)` | `:121-142` | One row's `code_bits` (0 = stored unquantized, -1 = absent), for `NativeDense::keep_unquantized_ple_key` (#326) |
| `WeightTable::load(pack, arena, bytes, err, skip)` | `:144-472` | Parses the index, compacts it under `skip`, refuses an arena smaller than the pool, then streams every row through two pinned staging buffers (8 MiB in, 16 MiB out because widening doubles a plane) with `cudaMemcpy`; skipped rows keep their metadata with `data == nullptr`. A quantized row with `dst_bytes == 0` is refused: "this pack holds the tensor only in its GGUF form (run with --native SHARD1)" (`:269-274`) |
| `WeightTable::find(name)` | `:474-477` | Lookup |

The loader checks its own segment list against both recorded totals (`weights.cpp:331-342`) and refuses an fp16
plane of odd length. Files are read with `std::fopen` / `fseeko` / `_fseeki64` (`read_at`, `weights.cpp:77-85`).
`report()` returns tensors, bytes saved by re-rounding and read / upload time.

## Control flow

`generate.cpp:2185-2200` builds the skip set unless `--keep-canonical` is given: the names `NativeDense::served_names`
returns, `output.weight` when a native head is loaded, and on a native pack `token_embd.weight` (the native embedding
serves it) and, through `keep_unquantized_ple_key`, the PLE key unless the pack stores it unquantized. Then
`:2238` `pool_bytes`, a `cudaMalloc` of that size, and `:2258` `load`. A layer split loads one table per stage with
`NativeDense::set_layer_range` restricting the skip set (`:2573-2585`). The resulting table is passed by reference to
every layer function, the session, the verifier and the drafter.

## Configuration

`--pack DIR` names the pack; `--native SHARD1` (with the GGUF shards beside it) decides which tensors are skipped;
`--keep-canonical` loads the canonical forms as well (an A/B arm). No environment variables in these files.

## Model and platform notes

* Canonical Q2_0 pack: every dense tensor, the embedding and the head are in the arena in S-form.
* Native packs: projections that `NativeDense` serves have shape-only rows (`dst_bytes == 0`); BF16/F16 tensors are
  stored as kinds 4/5; `extra.bin` holds float tensors the base pack quantized.
* Windows: 64-bit seeks through `_fseeki64` (`weights.cpp:78-79`); the comment at `:74-76` notes that `std::fopen`
  takes an ANSI path there.

## Tests

* `native_dense_ple_key_test` (`tests/core/native_dense_ple_key_test.cpp`, needs `STRATA_NATIVE_EXPERTS` and CUDA,
  exit 77 skips): `index_code_bits` edge cases and that a BF16 PLE key row stays in the arena.
* There is no dedicated unit test of `check_layer` / `check_all` or of `load` in this snapshot.

## Gotchas

* `scales_bytes` is the arena's width (4 B per scale), not the pack's; `scales_fp16` records the pack's.
* `pool_bytes` without a skip set returns the header's pool; with one it recomputes the compacted size from each
  row's `dst_bytes`, so the two can differ.
* `ModelGeometry` changes are kernel-contract changes (header comment): nothing validates a GGUF against these
  constants except the shape checks and the kernels' own guards.

## In UnbiasedStrata

Changed. The fork replaced the fixed struct with `ModelArch` plus fields read from the GGUF (`model_geometry.hpp`,
`qsa_shapes.hpp`, `derive()`, `has_ple`, `has_indexer()`, `gdn_gate()`), extended `layout.cpp`'s checks for the
plain-residual / dense-attention model, and added a `WeightTable::set` test seam; `weights.cpp` is unchanged. See the
fork's [model-geometry.md](../../codebase/core/model-geometry.md) and
[weights-and-layout.md](../../codebase/core/weights-and-layout.md).

## See also

[dense-and-head.md](dense-and-head.md) · [layer-forward.md](layer-forward.md) ·
[experts/pack-format.md](experts/pack-format.md) · [glossary](../guides/glossary.md)
