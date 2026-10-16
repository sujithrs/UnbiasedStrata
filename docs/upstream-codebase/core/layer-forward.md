[Upstream docs](../README.md) › [Core](README.md) › One layer's forward pass

# One layer's decode forward pass

## Purpose

`layer.hpp` / `layer.cpp` compose the kernels of one decode step of one layer, for one token, in the order of the
numpy reference (`ref/`). Every function here only enqueues work on a stream, so it can be captured into a CUDA graph
([cuda-graphs.md](cuda-graphs.md)). The routed experts are deliberately outside: the layer publishes its routing
through the doorbell and combines a `parts` buffer that the expert side fills ([experts/](experts/README.md)). The
speculative verify window ([verify.md](verify.md)) re-implements the same arithmetic for T tokens with multi-column
kernels and calls a few of these functions directly.

Files covered:
[include/strata/core/layer.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/layer.hpp),
[src/core/layer.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/layer.cpp).

## The block (`block_layer_pre` / `block_layer_post`)

Every one of the 48 layers is a hyper-connection block: the residual `R` is `hc = 4` streams of `n_embd` floats
(`BlockBuffers::R`, `layer.hpp:436-457`), read into one mixer input and written back with per-stream weights.

```
block_layer_pre[l]   (layer.cpp:1170-1289)
  [l == 1 and the PLE is ready]  ple_block on R (in place) + ple_history_advance
  stage 0  gr_read(hc_attn_*)  -> bb.mixed, bb.inject          (fused_gr_read when fused GR is on)
  stage 1  qsa_layer | gdn_layer (bb.mixed -> bb.block_out)
  stage 2  gr_write(R, block_out, inject)                       (skipped when fused: folded into the next read)
  stage 3  gr_read(hc_ffn_*)   -> bb.mixed, bb.inject / inject2
  stage 4  moe_route(bb.mixed): router, top-k, DOORBELL publish + ring
           moe_shared (when "shared early", the default)        -> the GPU works while the host runs the pool
block_layer_post[l]  (layer.cpp:1319-1343)
  stage 5  moe_combine_parts(parts, weights, shared) -> bb.block_out   (or moe_finish when shared is late)
  stage 6  gr_write(R, block_out, inject) [+ control vector]   (fused: only on the last layer, to materialise R)
```

`block_layer` (`:1344-1349`) runs both halves back to back for the uncaptured `session_token`. The `half` and
`stage_prefix` parameters of `block_layer_pre` capture prefixes of the stages for measurement only
(`layer.hpp:600-616`).

**The split at the router is a correctness rule.** `post[l]` combines `parts` with the weights `moe_route` wrote for
the same layer, so the pool must answer for layer `l` between the two graphs; handing layer `l`'s experts to layer
`l+1` (the "L100 bug", `layer.hpp:144-165`) produced fluent, deterministic, wrong tokens. Nothing may run between
`pre[l]` and `post[l]` that touches `bb.mixed` / `bb.inject`, which is why the overlap window is only the shared
expert.

**Fused hyper-connections** (`layer_set_fused_gr`, `layer.hpp:461-466`): with the native MMVF contract the two reads
become `fused_gr_read` kernels that also apply the previous half's pending write; `R` then lags by one write between
halves, and is materialised before the layer-1 PLE (`:1193-1197`) and after the last layer (`:1331-1332`). A control
vector (`--control-vector-scaled`) after a layer forces the write to be applied (`cvec().covers`).

## Gated DeltaNet: `gdn_layer` (36 layers)

`layer.cpp:223-334`. The order is `ref/gdn.py::gdn_decode`, quoted in `layer.hpp:8-31`:

```
qkv  = attn_qkv @ x                     (10240 = q|k|v)
conv -> SiLU over the whole conv output (before the split)
q, k = l2_norm (v is not normalised); q scaled by 1/sqrt(128)
beta = sigmoid(ssm_beta @ x)
gate = softplus(ssm_alpha @ x + ssm_dt.bias) * ssm_a
recurrent step (state S x h_v x S per layer)
y    = rms_norm(o) * ssm_norm * sigmoid(z),  z = attn_gate @ x     (sigmoid for qwen4exp)
out  = ssm_out @ y
```

* Activation contract (`docs/activation-contract.md`, quoted at `layer.cpp:113-139`): a quantized weight with
  `code_bits == 2` reads a Q8_0 activation (`s2_gemv_q8`); any other quantized weight reads Q8_0 or Q8_K by its
  `act_kind`; a BF16 weight reads BF16 (or F32 under `--native-bf16`, `project_bf16`, `:94-100`). The pack mixes
  types per layer (`attn_qkv` is IQ4_XS on 13 layers, Q3_K on 18, Q4_K on 4, Q2_0 on 1), so both quantized images
  are produced. With a native projection (`WeightRef::native_data`), `gemv_quantized` (`:140-172`) uses
  `native_quantize_q8_1` + `native_mmvq` instead.
* Kernel variants: canonical (`gdn_conv_step`, `gdn_l2_norm`, `gdn_step`, `gdn_out_norm`), native
  (`native_gdn_*`, when `native_gdn_enabled()`), and fused (`fused_gdn_conv_l2`, `fused_gdn_ab`,
  `fused_gdn_step_norm`) when `layer_set_fused_gdn` (default on), the native GDN, native BF16 projections,
  `ssm_d_conv == 4` and `ssm_state_size == 128` all hold (`:247-248`, `:306`).
* The per-layer `state` / `conv_state` pointers are set by the session (`gdn_point_at`, [session.md](session.md));
  the GDN scratch (`GdnBuffers`, `layer.hpp:59-85`) is shared by all 36 layers.

## QSA sparse attention: `qsa_layer` (12 layers)

`layer.cpp:876-1021`, order `ref/model.py::_qsa` (`layer.hpp:367-384`). QSA is full attention over a **selected
subset** of cells: a small "lightning indexer" scores blocks of past keys and a top-k picks which cells the query
attends to.

1. Activations: Q8_K and Q8_0 images (canonical projections), BF16 for the indexer.
2. Step counts: `qsa_step_fill` writes every per-token count into pinned `st.host_step` / `host_pos`, and a captured
   copy (a kernel reading mapped memory under `layer_set_publish_kernel`, else a memcpy node) moves them to
   `st.step` / `st.pos_dev`. Nothing per token is a kernel argument (`layer.cpp:908-917`); the header explains why the
   staging lives at fixed pinned addresses (`layer.hpp:257-267`).
3. The indexer's raw key `indexer.k_proj @ x`, appended before any norm (pooled and rotated once per block).
4. `attn_k`, `attn_v`; K is normed and RoPE-rotated; K and V appended to the KV pools in the state's format.
5. `attn_q` (q | gate per head, split by halves), q normed and rotated; the indexer query normed and rotated.
6. Selection: `qsa_block_scores` + `qsa_block_topk` (FP32 block scores and radix top-k, `layer_set_fast_select`,
   default on) or the older `qsa_index_step` + `topk_512_step`. `qsa_kv_resolve` then makes streamed blocks resident.
7. Attention: `qsa_decode_attn_step` (split-K, reads the pools directly, `layer_set_fast_attn`, default on) or
   gather + `qsa_attend_step`; `native_flash_attn_short_step` is a diagnostic adapter for contexts of at most 256.
8. Output gate `attn * sigmoid(gate)` (`native_qsa_gate_apply` or `qsa_gate_apply_f32`), then `attn_output`.

A position outside the RoPE table is refused (`:876`). All grid sizes come from capacities (`max_cells`,
`qsa_selection_width(kTopkMaxCells, s)`), not from the token's position, so the graph replays at any position.

### The QSA state and KV formats

`QsaState` (`layer.hpp:206-268`) is one layer's persistent per-sequence state: K/V pools, page table, indexer
tail / dead row / pooled rows / block position, RoPE cos/sin tables, step buffers and their pinned staging.

| Format | Set by | Bytes per cell (both K and V, 2 KV heads x 256) | Notes |
| --- | --- | --- | --- |
| FP16 | `--kv fp16` (default) | 2,048 | identity page table |
| INT8 + FP16 scale per 64 | `--kv int8` (`qsa_set_kv_int8`) | 1,056 (`layer.hpp:291`) | `STRATA_KV_ROT=1` adds the Hadamard rotation (`qsa_set_kv_int8_rotate`) |
| Q4_0 after a Walsh-Hadamard rotation | `--kv q4_0` (PR #21) | 576 (`layer.hpp:291`) | queries rotated too, output rotated back |
| K8V4 hybrid | `--kv k8v4` | 816 (`layer.hpp:219`) | INT8 K unrotated, rotated Q4_0 V; mode 0 only |

`kv_mode` (`layer.hpp:230-237`): 0 = every page in VRAM; 1 = **KV streaming** (`--kv-resident N`): the
authoritative K/V in pinned host memory and `n_slots` pages in VRAM, resolved per selection (at least
`qsa_kv_resident_min()` = 20,480 cells, `layer.cpp:556`); 2 = a ring (the MTP drafter's window). `kv_plan`
(`layer.cpp:523-541`) decides the mode; `STRATA_KV_RING_OFF` / `STRATA_KV_MAIN_OFF` are A/B switches. The RoPE table
is built on the host in float64 once, in the session's primary QSA state, and the other layers borrow it
(`share_rope`, `layer.hpp:270-273`; 64 MiB at 262K per the comment). The process's rope scaling (`--rope-scaling`,
YaRN, linear) is inside the table (`layer.cpp:686-699`).

| Function | Lines | Role |
| --- | --- | --- |
| `qsa_state_bytes(g, max_cells, with_rope, ring_cells)` | `layer.cpp:559-575` | Size for the chosen format and mode |
| `qsa_state_init(...)` | `:577-714` | Carve, pin the staging (and the host copy in modes 1/2), build the RoPE table, set the page table; returns 0 when pinned RAM is short (the WSL note at `:664-668`) |
| `qsa_state_zero(st, g, stream)` | `:715-742` | Zero pools and indexer; a streamed state only resets its map |
| `qsa_attn_pools`, `qsa_kv_resolve` | `:744-759` | The pools a reader sees; make a selection's blocks resident (mode 1) |
| `qsa_kv_format(st)` | `layer.hpp:298-306` | Format id for the block movers; exits on a hybrid state |

## The MoE: route, shared expert, combine

| Function | Lines | What it does |
| --- | --- | --- |
| `moe_route` | `layer.cpp:355-392` | `ffn_gate_inp` (BF16) projection, then top-k: `native_router_top10` only for 512 experts and k = 10, else the generic `router_top10` (softmax over all experts, stable sort, renormalise). Then the **doorbell** publish |
| `moe_route_window` | `:393-417` | The verify window's n tokens routed with one multi-column BF16 projection and one top-k launch when that is bitwise the per-token result; else per token. `STRATA_ROUTE_PER_TOKEN`, `STRATA_ROUTE_PROJ_PER_TOKEN` force the per-token forms |
| `moe_shared` | `:418-452` | The shared expert (`ffn_{gate,up,down}_shexp` and the BF16 scalar gate `ffn_gate_inp_shexp`), canonical or native per tensor (`NativeSharedWeights`) |
| `moe_combine_parts` | `:453-469` | `out = sum_i w[i] * parts[i] + shared` (`native_moe_combine` or `moe_combine`) |
| `moe_finish`, `moe_layer` | `:470-475`, `:495-505` | Shared + combine; route + finish (uncaptured path) |

`layer_set_shared_early` (default on, `layer.hpp:191-196`) moves the shared expert to the end of `pre[l]`, after the
ring, so the GPU computes it while the host runs the CPU pool; the result is bitwise unchanged. The comment at
`layer.hpp:131-139` explains why the router sees a BF16 activation: an fp16 activation moved the logits by 8.1e-3,
enough to flip a top-10 selection.

## The doorbell

`struct Doorbell` (`layer.hpp:405-422`) is the handoff of one layer's routed-expert input to the host: `x_f`
(n_embd floats), `ids` and `weights` (k each), a sequence word `seq` and, for the token graph, a host-written `flag`.
Each field is **mapped pinned** memory, so the device writes it and the host reads it without a copy
(`doorbell_init`, `layer.cpp:1023-1026`).

* The write is a kernel: `doorbell_publish` (payload and ring in one kernel, `layer_set_publish_kernel`, default
  on) or three memcpy nodes followed by `doorbell_ring` (`layer.cpp:379-389`). The header cites the two measurements
  behind this: a `cudaEventRecord` inside a capture was silently dropped (round 199), and a host spin that only reads
  memory never let the kernel run on Windows (round 195), so the host also calls `cudaEventQuery`.
* The ring value is read from the device copy and incremented, so replays of the same graph ring new numbers.
* `doorbell_reset` zeroes `seq` and `flag` before a token.

## PLE at layer 1

The per-layer embedding (PLE) is layer 1's n-gram embedding: the token and its two predecessors are hashed into 16
rows of `per_layer_token_embd.weight`, a 28.8 GB IQ4_NL table in the model's second GGUF shard that is read from the
SSD, never from the pack (`layer.hpp:524-536`).

* `PleRun` (`layer.hpp:537-563`) holds the table, its six `blk.1.ple_*` weights (the key may be BF16, quantized
  canonical or native), the token window, the normalized history (row-fastest), the device and host row buffers and
  scratch. `ready()` gates the whole module.
* Host side, once per token, **outside** any capture: `ple_issue_token` hashes and starts the 16 row reads,
  `ple_finish_token` collects them and queues the upload (`layer.cpp:1290-1312`). Doing this inside a captured
  function would replay the capture-time token's rows forever (`layer.hpp:565-568`).
* Device side, inside `pre[1]` before layer 1's own read: `ple_block` updates `R` in place, then
  `ple_history_advance` shifts the history (`layer.cpp:1192-1227`).

## The two ends of a token

* `embed_row(tables, g, token, out, stream, err)` (`layer.cpp:1033-1087`): the GGUF-form table when
  `native_embed()` is set, else a column dequant of the canonical `token_embd.weight` (`embedding_gather`, bitwise
  the decoder of `s_gemv_q8k_split`). The session broadcasts the row to the four streams (`session_zero` with
  `R_init`).
* `lm_head_mix` (`:1088-1108`): the final `gr_read` with `output_hc_{norm,down,up}` into `bb.mixed`.
* `lm_head` (`:1109-1126`): `lm_head_mix`, Q8_K, then the canonical `output.weight` GEMV. With `--native` the
  caller runs `NativeHead::run` on `bb.mixed` instead ([dense-and-head.md](dense-and-head.md)).

## Switches and diagnostics

All are process-wide and must be set before capture.

| Function | Default | Set by `generate.cpp` from |
| --- | --- | --- |
| `layer_set_native_bf16` | off | `--native-bf16` (on with `--native`) |
| `layer_set_fused_gr` | off | `--gr-native-mmvf` and not `--no-fused-gr`, no layer split, no dump (`:2284`) |
| `layer_set_fused_gdn`, `layer_set_fast_attn`, `layer_set_fast_select`, `layer_set_publish_kernel` | on | `--no-fused-gdn`, `--no-fast-attn`, `--no-fast-select`, `--no-publish-kernel` |
| `layer_set_shared_early` | on | `--shared-late` |
| `layer_set_native_flash_attn_short` | off | diagnostic flag (`generate.cpp:2287`) |
| `qsa_set_kv_*` | FP16, all resident | `--kv`, `--kv-resident`, `STRATA_KV_ROT` (`generate.cpp:1751-1773`) |

`layer_verify_compatible(why)` (`layer.cpp:476-486`) is true only for the configuration the verify window
reproduces bit for bit: native BF16 projections, fused GR, fused native GDN, split-K attention without the short
adapter, block selection, native indexer. Stage timing (`stage_timing_enable` and friends, `layer.cpp:803-873`)
records event pairs per (layer, stage) on the `--no-capture` path only, since an event record inside a capture is
dropped. The half-level dump (`BlockBuffers::dump`, `dump_stride_floats`, `layer.cpp:1146-1169`) copies each half's
output into a host buffer as graph nodes, for layer-by-layer comparison with the reference.

## Model and platform notes

* Canonical Q2_0 pack: every projection canonical unless `--native` serves it; with `--native` the projections are
  native and the experts canonical.
* Coder: same layer, 256-expert router through the generic top-k.
* The code has no `#if` for HIP or Windows; HIP builds go through `include/strata/hip_compat/` and the kernel
  sources.

## Tests

The kernels each have parity tests in the kernels area (`gdn_parity`, `qsa_parity`, `gr_parity`, `ple_parity`,
`kv_q8_parity`, `kv_q4_parity`, `kv_hybrid_parity`, `kv_stream_parity`, `router_top10_parity`, `shared_expert_parity`;
`CMakeLists.txt:671-822`). The composed layer is checked end to end against the reference (Gate C1) through the dump
paths; there is no unit test of `layer.cpp` itself in `tests/`.

## Gotchas

* A per-token value passed as a kernel argument inside a captured function is replayed forever; per-token data must
  live in device buffers fed from fixed pinned addresses.
* `ffn_gate_inp_shexp` and the indexer projections are BF16 in the arena (2 B per element); reading them as f32 does
  not fault inside the arena.
* `qsa_kv_format` exits the process on a hybrid state rather than return a wrong layout.

## In UnbiasedStrata

Changed. The fork added the plain block for `hc == 1` (`plain_norm_rows`, `plain_residual_rows`), dense attention
without an indexer, the SiLU GDN gate (`g.gdn_gate()`), `qsa_kv_streaming_applies`, and a geometry-taking
`layer_verify_compatible`, all chosen from the GGUF-derived `ModelGeometry`. Flash-Next's path is unchanged. Fork page:
[layer-forward.md](../../codebase/core/layer-forward.md).

## See also

[session.md](session.md) · [cuda-graphs.md](cuda-graphs.md) · [verify.md](verify.md) ·
[experts/dispatch.md](experts/dispatch.md) · [glossary](../guides/glossary.md)
