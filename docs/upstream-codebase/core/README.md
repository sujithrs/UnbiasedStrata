[Upstream docs](../README.md) › Core

# Core engine runtime (upstream)

The core is upstream Strata's GPU-side inference engine for Qwen3.8-Flash-Next (`qwen4exp`) and its variants (the
256-expert Coder, Swift 1.5, Unsloth's UD-IQ4_XS and UD-Q4_K_XL). It loads the dense (non-expert) weights from a
*pack* into one VRAM arena, optionally attaches the GGUF-form ("native") projections, head and embedding, carves the
per-sequence state (GDN recurrences, QSA KV caches, scratch), and runs one token, or a speculative window of up to 8
tokens, through all 48 layers as captured CUDA (or HIP) graphs. It hosts the MTP draft layer. The routed experts are
**not** computed inside the layer: each layer publishes its router output (a "doorbell"), and the expert side
([experts/](experts/README.md)) answers with one output row per routed expert, from the VRAM expert cache, the GPU
over PCIe, another GPU, or the CPU pool. The program layer (`src/program/generate.cpp`, which runs the CLI and the
`--serve` engine that `serve/server.py` starts) drives the core; the prompt path (`src/prefill/`) and the
conversation cache ([conversation/](conversation/README.md)) reuse the same `SessionState`. Terms such as GDN, QSA,
PLE, hyper-connection and doorbell are defined in the [glossary](../guides/glossary.md) and on the pages below.

## Contents

| Page | What it covers |
| --- | --- |
| [weights-and-layout.md](weights-and-layout.md) | The fixed `ModelGeometry` (Flash-Next constants; the Coder's expert count from the GGUF), `LayerView` and the shape checks, and the dense-weight loader (`WeightTable`, `index.txt`, engine forms) |
| [dense-and-head.md](dense-and-head.md) | Native GGUF projections (`NativeDense`), the native output head (`NativeHead`) and the GGUF-form embedding (`NativeEmbed`) |
| [layer-forward.md](layer-forward.md) | One layer's decode pass: hyper-connection read/write, Gated DeltaNet, QSA sparse attention and the KV formats, PLE at layer 1, MoE route / shared expert / combine, the doorbell, embedding and head |
| [session.md](session.md) | `SessionState`, the session arena carve, the layer-range carve, and the decode drivers (`session_token`, `session_loop`, `session_run_token`) |
| [cuda-graphs.md](cuda-graphs.md) | Every captured graph (per-layer pre/post, measurement splits, the token graph, verify, commit, batch and MTP graphs), `GraphRegistry`, and the capture rules |
| [verify.md](verify.md) | The speculative verify window (`Verifier`): T tokens in one graph, commit, batch windows, layer-split stages |
| [mtp-draft-head.md](mtp-draft-head.md) | The MTP draft layer (`MtpDrafter`) and coupled draft sampling |
| [device-and-memory.md](device-and-memory.md) | Device checks (CUDA, HIP, gfx906), `DeviceArena`, `PinnedArena` and its loaders, `OnDevice`, `STRATA_EMULATE_CC`, the Windows HIP VRAM budget |
| [standalone-tools.md](standalone-tools.md) | `strata-device`, `strata-load`, `strata-concurrent`, `strata-overlap` |
| [experts/](experts/README.md) | Expert sources, VRAM cache and profile, dispatch and the PCIe share, resident RAM mode and adaptive swaps, multi-GPU tiers, pack formats |
| [conversation/](conversation/README.md) | Running-state checkpoints, K/V images, the conversation cache |

## Source map

Every file under `src/core/` and `include/strata/core/` (line counts are `wc -l` at 6f32ec0). Expert and
conversation files are listed again in their subfolders.

| File | Lines | Role |
| --- | ---: | --- |
| [include/strata/core/layout.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/layout.hpp) | 97 | `ModelGeometry` (fixed Flash-Next constants), `is_qsa_layer`, `LayerView`, `check_layer` / `check_all` |
| [src/core/layout.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/layout.cpp) | 173 | Per-layer tensor shape and engine-form checks |
| [include/strata/core/weights.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/weights.hpp) | 141 | `WeightKind`, `WeightRef`, `LoadReport`, `WeightTable` |
| [src/core/weights.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/weights.cpp) | 479 | `index.txt` parser, plane segments, staged upload |
| [include/strata/core/native_dense.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/native_dense.hpp) | 45 | `NativeDense` |
| [src/core/native_dense.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/native_dense.cpp) | 210 | Eligibility, GGUF directory validation, upload |
| [include/strata/core/native_head.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/native_head.hpp) | 68 | `NativeHead`, `NativeEmbed`, `set_native_embed` |
| [src/core/native_head.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/native_head.cpp) | 198 | Head upload and run; embedding table in mapped pinned memory (VRAM fallback) |
| [include/strata/core/layer.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/layer.hpp) | 627 | Buffers, states and functions of one layer: GDN, MoE, QSA, timing, doorbell, block, PLE, embedding, head |
| [src/core/layer.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/layer.cpp) | 1349 | Their implementation |
| [include/strata/core/session.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/session.hpp) | 436 | `SessionState`, `SessionGraphs`, `session_loop`, `TokenGraph` |
| [src/core/session.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/session.cpp) | 970 | Session carve, capture, replays, host loop, token graph |
| [include/strata/core/graph.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/graph.hpp) | 117 | `CapturedGraph`, `GraphRegistry` |
| [src/core/graph.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/graph.cpp) | 146 | Their implementation |
| [include/strata/core/hit_hook.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/hit_hook.hpp) | 41 | `HitPhase`, `HitFn`: the seam between `session_loop` and the VRAM hit path |
| [include/strata/core/progress.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/progress.hpp) | 59 | Process-wide heartbeat and stage for the `--serve` watchdog |
| [include/strata/core/verify.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/verify.hpp) | 364 | `Verifier` |
| [src/core/verify.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/verify.cpp) | 2158 | Window recording, run, commit, batch windows, layer split, diagnostics |
| [include/strata/core/mtp.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/mtp.hpp) | 188 | `MtpDrafter` |
| [src/core/mtp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/mtp.cpp) | 993 | Draft layer load, bind, forward recording, prefill, draft rounds |
| [include/strata/core/coupled_draft.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/coupled_draft.hpp) | 90 | Coupled draft sampling's counter and penalty-ring arithmetic |
| [src/core/coupled_draft_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/coupled_draft_test.cpp) | 115 | CPU test of that arithmetic |
| [include/strata/core/device.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/device.hpp) | 95 | `DeviceInfo`, device checks, `CudaError`, `DeviceArena` |
| [src/core/device.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/device.cu) | 344 | Device queries, CUDA / HIP / gfx906 checks, arena, Windows HIP free-VRAM budget |
| [include/strata/core/pinned.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/pinned.hpp) | 152 | `PinnedArena`, expert-file loaders, `stream_bandwidth` |
| [src/core/pinned.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/pinned.cu) | 725 | Large pages / THP, registration (whole or sliced), shared-file backing, readers |
| [src/core/pinned_shared_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/pinned_shared_test.cpp) | 61 | Linux test of the `MAP_SHARED` arena backing |
| [include/strata/core/on_device.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/on_device.hpp) | 22 | `OnDevice`: scoped device switch (multi-GPU) |
| [include/strata/core/emulate.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/emulate.hpp) | 36 | `STRATA_EMULATE_CC` |
| [src/core/device_main.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/device_main.cpp) | 122 | `strata-device` |
| [src/core/load_main.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/load_main.cpp) | 122 | `strata-load` |
| [src/core/concurrent_main.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/concurrent_main.cpp) | 197 | `strata-concurrent` |
| [src/core/overlap_main.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/overlap_main.cpp) | 169 | `strata-overlap` |
| [include/strata/core/expert_source.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/expert_source.hpp) | 720 | Expert sources, dispatch, GGUF expert loader ([experts/](experts/README.md)) |
| [src/core/expert_source.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/expert_source.cpp) | 2985 | Their implementation |
| [include/strata/core/expert_cache.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/expert_cache.hpp) | 227 | `ExpertCache`, expert profile I/O |
| [src/core/expert_cache.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/expert_cache.cpp) | 696 | Their implementation |
| [include/strata/core/peer_experts.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/peer_experts.hpp) | 106 | `PeerExperts` |
| [src/core/peer_experts.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/peer_experts.cpp) | 325 | Its implementation |
| [include/strata/core/remote_experts.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/remote_experts.hpp) | 89 | `RemoteExperts` |
| [src/core/remote_experts.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/remote_experts.cpp) | 341 | Its implementation |
| [include/strata/core/remote_expert_opt.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/remote_expert_opt.hpp) | 53 | `RemoteExpertOpt` |
| [src/core/remote_expert_opt.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/remote_expert_opt.cu) | 194 | Its implementation |
| [include/strata/core/conversation_cache.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/conversation_cache.hpp) | 283 | Conversation types and `ConversationCache` ([conversation/](conversation/README.md)) |
| [include/strata/core/conversation_snapshot.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/conversation_snapshot.hpp) | 96 | Save / validate / restore API |
| [include/strata/core/conversation_buffer.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/conversation_buffer.hpp) | 154 | Segmented host byte storage |
| [include/strata/core/conversation_memory.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/conversation_memory.hpp) | 19 | Physical-RAM admission |
| [src/core/conversation_state.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/conversation_state.cpp) | 332 | Running state and whole snapshots |
| [src/core/conversation_snapshot.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/conversation_snapshot.cpp) | 268 | Per-layer K/V images |
| [src/core/conversation_memory.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/conversation_memory.cpp) | 51 | `MemAvailable` / `GlobalMemoryStatusEx` |
| [src/core/conversation_checked.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/conversation_checked.hpp) | 24 | Overflow-checked `add` / `product` |
| [src/core/conversation_cache_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/conversation_cache_test.cpp) | 305 | CPU tests of the cache and buffer |
| [src/core/conversation_memory_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/conversation_memory_test.cpp) | 66 | CPU tests of the RAM probe |
| [src/core/conversation_snapshot_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/conversation_snapshot_test.cpp) | 324 | GPU round-trip tests |
| [src/core/conversation_validation_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/conversation_validation_test.cpp) | 239 | Validation before any CUDA call; fault-injection variant |

Build targets (`CMakeLists.txt:246-279`, `:412-482`): `strata_core` (static: `device.cu`, `pinned.cu`,
`src/platform/memory.cpp`, `graph.cpp`, `weights.cpp`, `layout.cpp`) and `strata_engine` (static: `layer.cpp`,
`session.cpp`, `expert_source.cpp`, `remote_experts.cpp`, `remote_expert_opt.cu`, `expert_cache.cpp`,
`native_head.cpp`, `native_dense.cpp`, `verify.cpp`, `mtp.cpp`, the three conversation sources, `peer_experts.cpp`).
The CMake comment above `strata_engine` gives the reason for the split: `layer.cpp` calls kernels, and
`strata_kernels` already links `strata_core`. Both libraries are built only with `STRATA_ENABLE_CUDA` or
`STRATA_ENABLE_HIP`. With HIP the same sources compile through `include/strata/hip_compat/` (see the build area);
the Intel port keeps its own copies under `sycl/` (SYCL area), not documented here.

## Key concepts

### One model, fixed geometry

Upstream has no geometry reader: `ModelGeometry` (`layout.hpp:27-59`) holds Flash-Next's numbers as defaults
(48 layers, every 4th one QSA, 36 GDN + 12 QSA, `n_embd` 2560, hyper-connections `hc = 4`, 512 experts, `n_ff` 640).
The only override is in `src/program/generate.cpp:2073-2079`: with `--native`, the GGUF's
`qwen4exp.expert_count` and `qwen4exp.expert_used_count` replace `n_expert` and `K` (the GSQ-RCO Coder keeps 256 of
512 experts, still 10 active). Everything else, including the MTP draft layer's geometry
(`generate.cpp:3013`, always the canonical 512 experts), is the constant set. See
[weights-and-layout.md](weights-and-layout.md).

### Model variants as the core sees them

| Variant | Pack | What changes in the core |
| --- | --- | --- |
| Q2_0 on an AVX-512 CPU | Canonical pack (`tools/strata_pack.py` + `pack_index.py`, `setup.py:4175-4181`): `experts.bin` of fixed 1,382,400-byte blobs, no `native_experts.txt` | Per-layer graphs and the token graph are captured (`generate.cpp:3822`, `:4333`); the single-token CPU pool path is available; default PCIe share 0.2 |
| Q2_0 without AVX-512, IQ2_XS, IQ3_XXS, IQ3_S, Swift 1.5, Coder (IQ1_M), Unsloth UD-IQ4_XS / UD-Q4_K_XL | Native pack (`tools/iq_pack.py`): `native_experts.txt`, experts read from the GGUF shards (or `experts.bin` in the low-RAM mode) | Experts run in verify windows only (`expert_source.cpp:1868-1874`); no per-layer or token graph (`generate.cpp:3822`); slot sizes vary per layer (`ExpertCache::open_sized`); default PCIe share 0.55, scaled by a link probe |
| Coder | Native, 256 experts | `n_expert` = 256 from the GGUF; the fused native router (512 x 10 only) is replaced by the generic top-k (`layer.cpp:369-372`); the drafter keeps 512 experts |
| Unsloth UD-Q4_K_XL | Native, v4 per-role shards | `output.weight` may live in a later shard (`native_head.cpp:26-33`); a Q8_0 PLE key is served natively (`native_dense.cpp:27-30`) |

Every pack is started with `--native SHARD1` by setup (`setup.py:4224-4227`), which turns on every native kernel
switch (`generate.cpp:1802-1806`). The canonical Q2_0 pack differs only in its experts.

### Start-up order (as `src/program/generate.cpp` does it)

```
qsa_set_kv_* (--kv, --kv-resident)                          generate.cpp:1756-1773
ModelGeometry g (defaults) [+ expert_count from the GGUF]   :2060-2079
NativeEmbed::load, set_native_embed                         :2172-2177
NativeDense::served_names (skip set), keep_unquantized_ple_key   :2188-2196
WeightTable::pool_bytes -> cudaMalloc(arena) -> WeightTable::load   :2238-2258
NativeDense::load                                           :2268
layer_set_* switches                                        :2277-2287
doorbell_init                                               :2370
session_bytes / session_init / session_zero                 :2873-2890
MtpDrafter::load (canonical geometry), NativeHead::load     :3016, :3030
expert tier (ExpertSource, ExpertCache, profile fill)       experts/
session_capture (canonical pack only)                       :3827
session_capture_token (canonical pack, one GPU)             :4335
Verifier::init + MtpDrafter::bind (--spec / --serve)        :5134-5135, :7831-7836
```

### One decode token (canonical pack, per-layer graph form, `session_loop`)

```
for each layer l:
  pre[l]  (captured): [PLE at l == 1] -> gr_read(attn) -> GDN | QSA -> gr_write(attn)
                      -> gr_read(ffn) -> router top-k -> DOORBELL RING (x, ids, weights to mapped memory)
                      -> shared expert (overlaps the host's pool)
  host:   poll ring (cudaEventQuery + volatile read) -> hits(Launch) -> pool() -> H2D misses -> hits(Combine)
  post[l] (captured): combine(parts, weights, shared) -> gr_write(ffn)
head: lm_head_mix (final hyper-connection read) -> output projection -> logits
```

The token graph (`session_capture_token`) puts all 48 layers in one graph and makes the GPU wait on a host flag
(`doorbell_wait`). The default serving path replaces both with the verify window (`Verifier`, up to 8 tokens per
graph) plus the MTP drafter, which is what native packs always use.

### Memory tiers handled here

* **VRAM:** the dense-weight arena, native projections, native head, session arena (GDN state, QSA KV, scratch), the
  MTP draft layer (about 0.9 GB per `mtp.hpp:44`), and the expert cache that takes what is left.
* **Pinned host:** the expert arena (`PinnedArena`), the doorbell and every mapped staging buffer, the KV streaming
  host copy (`--kv-resident`), the native embedding table.

## In UnbiasedStrata

The fork ([fork's own core docs](../../codebase/core/README.md)) changed 14 of these files and added two headers
(`git diff --stat 12a7825 HEAD -- src/core include/strata/core`); nothing in core was deleted. The Qwen3.6-35B-A3B
port (recorded in the fork's history under e41dfc7) replaced the fixed geometry with `ModelGeometry` read from the
GGUF (new `model_geometry.hpp`, `qsa_shapes.hpp`, `ModelArch` in `layout.hpp`), and added the plain residual (hc 1),
dense attention, the SiLU GDN gate, a geometry check in the verify window and a test seam in `weights.hpp`;
`session.cpp` carves the PLE history only for models with a PLE. Later commits: 44b419b (the Qwen3.6 drafter in
`mtp.cpp`), b4ee41e (verify commit without an indexer), 3a806b5 (conversation checkpoint and snapshot for a dense,
plain-residual model), fa69e82 (the multi-token expert dispatch at the layer's own width), and 9a53e42 (the GGUF
expert loader reads through the new `robust_read.hpp` instead of `fopen`/`fread`; plus `--ram-cap` and
`complement_count`). The fork's `UPSTREAM.md:39-40` says the AMD and Windows code paths are still in the source but
are not built, tested or supported there; the SYCL copy of core was removed with the rest of `sycl/` (8894a81).

## See also

* [docs/upstream/DETAILS.md](../../upstream/DETAILS.md), "How it works": upstream's summary of the tiers.
* [HOW_IT_WORKS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/HOW_IT_WORKS.md),
  [MODELS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MODELS.md),
  [MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MULTI_GPU.md),
  [AMD_HIP.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/AMD_HIP.md).
* [Glossary](../guides/glossary.md).
