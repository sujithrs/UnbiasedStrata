[Upstream docs](../README.md) › [Core](README.md) › CUDA graphs

# CUDA graphs in the core

## Purpose

Decode in upstream Strata is launch-bound, so nearly all GPU work of a token is recorded once with stream capture and
replayed. This page lists every graph the core captures, the handshake between a running graph and the host, and the
rules that keep replays correct. It also covers `GraphRegistry`, a generic capture cache that the engine itself does
not use. The same code runs on HIP through the `hip_compat` mapping (`cudaGraph*` -> `hipGraph*`).

Files covered:
[include/strata/core/graph.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/graph.hpp),
[src/core/graph.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/graph.cpp);
the captures themselves live in `session.cpp`, `verify.cpp` and `mtp.cpp`.

## The rules

From `graph.hpp:6-31`, `layer.hpp:386-404` and `session.hpp:127-136`:

1. **A graph re-reads its input buffers at replay, but not its kernel arguments.** Anything that changes per step
   (position, step counts, token ids, page tables) must be data in a device buffer, never an argument. The QSA layers
   therefore read `st.step` / `st.pos_dev`, refreshed from fixed pinned staging by a node inside the graph.
2. **Fixed addresses are the caller's obligation.** Every buffer a captured body touches is allocated before capture
   and never reallocated; staging must not be a stack array or a local vector (`layer.hpp:257-267`).
3. **A `cudaEventRecord` inside a capture is silently dropped** (round 199), so mid-graph signals are kernels that
   write mapped pinned memory (`doorbell_publish`, `doorbell_ring`, `gpu_stamp`).
4. **The host wait must enter the driver.** A memory-only spin never let the kernel start under Windows' batched
   submission (round 195), so host loops call `cudaEventQuery` or `cudaStreamQuery` while they spin.
5. **No synchronous call in a captured function.** The capture fails at `cudaStreamEndCapture`
   (`session.cpp` reports "a synchronous call in the layer?").
6. **Host-side work per token stays outside the capture.** The PLE hash and row gather run in the driver
   (`ple_issue_token` / `ple_finish_token`), and sampling with request parameters runs after the verify graph
   (`verify.cpp:1542-1546`).

Measured motivation (`session.hpp:132-136`): one block's 43-node graph replayed in 1.585 ms against 2.393 ms as
direct launches.

## Inventory

| Graph | Captured by | Count | Used for |
| --- | --- | --- | --- |
| `pre[l]` (`SessionGraphs::execs`) | `session_capture`, `session.cpp:191-276` | 48 | Block up to the router, the doorbell and the shared expert |
| `post[l]` (`posts`) | same | 48 | Combine and residual write |
| `preA`, `preB`, `preP[0..4]` | same, with `split = true` (`--gpu-stages`) | 7 x 48 | Measurement only: the mixer half, the FFN front, and stage prefixes |
| `TokenGraph::exec` | `session_capture_token`, `session.cpp:829-901` | 1 | The whole token, with `doorbell_wait` handshakes |
| `Verifier::exec_[T]` | `Verifier::capture`, `verify.cpp:1202-1264` | one per T in 1..8 | A T-token verify window through all of the stage's layers and the head |
| `Verifier::commit_exec_` | `capture_commit`, `:1265-1330` | 1 | Advance GDN and indexer state by the accepted tokens |
| `exec_bm_`, `commit_bm_` | `capture_batch`, `capture_commit_batch` (`:1780-1887`) | keyed by rows, S, hand-off base | Batch windows over independent sequences |
| `MtpDrafter::prefill_exec_[T]`, `prefill_dev_exec_[T]` | `capture_prefill`, `capture_prefill_dev` (`mtp.cpp:673-690`) | per T | The draft layer's prompt K/V |
| `round_exec_[T]`, `step_exec_[j]` (+ `_c_` coupled variants) | `capture_round`, `capture_step` (`mtp.cpp:691-741`) | per T, per step | A draft round: catch-up over T cells and the first draft; each further draft step |

The prompt path (`src/prefill/`) captures its own graphs (prefill area).

Which graphs exist in a run:

* **Canonical Q2_0 pack, plain decode:** the per-layer pair, and the token graph unless `--no-token-graph`, a dump,
  a layer split, or a hit hook that is not fully on the device (`generate.cpp:3822`, `:4333-4334`).
* **Native packs:** no per-layer or token graph (`generate.cpp:3822` comment: "a native pack runs verify windows
  only"); verify, commit and MTP graphs.
* **`--serve` with `--spec` / `--mtp`** (setup's default, `setup.py:4226`): verify, commit and MTP graphs. The
  per-layer capture condition (`generate.cpp:3822`) does not look at `--spec`, so a canonical pack captures those
  too; which serve paths replay them was not traced here.

All captures use `cudaStreamBeginCapture(..., cudaStreamCaptureModeThreadLocal)`. `Verifier::capture` also calls
`cudaGraphUpload` and prints "captured the T-token window"; `STRATA_VERIFY_NODES` lists the node types and kernel
names (CUDA only, names from CUDA 12.3, `verify.cpp:1221-1250`).

## Handshakes between a graph and the host

| Graph | GPU -> host | Host -> GPU |
| --- | --- | --- |
| `pre[l]` / `post[l]` | `h_seq` ring (`doorbell_ring` / `doorbell_publish`) | the host launches `post[l]` itself after copying `parts` |
| Token graph | `h_seq` ring per layer | `h_flag = ring` after the pool; the graph's `doorbell_wait` kernel spins on it, then `copy_from_mapped` reads `y_miss` |
| Verify window | ring per (layer, token group) in mapped `h_seq_` | `flagA` (the pool published the GPU plan), `flagB` (the PCIe share has landed; raised by a `cudaLaunchHostFunc` on the copy stream after DMA, `verify.cpp:1621-1632`), `flag` (the CPU's rows are in mapped memory); the graph waits with `wait_flag_ge` kernels (`:1042-1066`) |
| MTP round | each draft id written to mapped `h_out_` | none; the host reads ids as they appear and prefetches PLE rows |

`Verifier::release_gpu_waits` (#267) raises every flag past any ring so that a window the GPU cannot finish drains
instead of leaving a spin kernel resident; the serve watchdog calls it through `release_gpu_fn` before ending the
engine (on Windows a resident spin kernel had left the GPU "lost").

## `CapturedGraph` and `GraphRegistry` (`graph.hpp:51-115`, `graph.cpp`)

* `CapturedGraph` is move-only and owns a `cudaGraphExec_t` and a completion event. `begin` / `end` wrap
  `cudaStreamBeginCapture` / `EndCapture`; `end` refuses a capture with zero nodes (`graph.cpp:54-73`). `launch`
  records the event after the replay; `wait_ms(timeout)` polls it with `cudaEventQuery` against a real clock
  (`:84-102`).
* `GraphRegistry` caches graphs by `(LayerType, n_tokens)`; `record` runs the body only the first time.
* No engine code uses them in this snapshot: the only references outside the two files would be
  `bench/micro/graph_registry_test.cu`, which `CMakeLists.txt:861-866` registers only if it exists, and `bench/micro/`
  is not in the snapshot. `CapturedGraph::wait_ms` spins with `_mm_pause` between queries (`graph.cpp:100`, x86).

## Configuration

| Flag / env | Effect |
| --- | --- |
| `--no-capture` | Direct launches (`session_token`) |
| `--no-token-graph` | Per-layer graphs instead of the token graph |
| `--gpu-stages` | Capture the measurement splits |
| `STRATA_TG_FLUSH_US` | Token graph flush interval while a ring is late |
| `STRATA_VERIFY_NODES` | List the verify graph's nodes |
| `STRATA_VERIFY_TRACE`, `STRATA_VERIFY_PROFILE` | Breadcrumbs and per-stage stamps written by kernels inside the verify graph |

## Gotchas

* Everything a graph reads must outlive it: `NativeDense`, `NativeHead`, the session arena, `SessionLoopScratch`.
* `session_run_token` refuses a staging buffer other than the one baked into the graph.
* Process-wide switches (`layer_set_*`, `qsa_set_kv_*`, `set_native_embed`) are read at capture time; changing them
  later has no effect on existing graphs.

## In UnbiasedStrata

`graph.hpp` / `graph.cpp` are unchanged. The fork's graphs differ only through its layer, verify and MTP changes
(the plain block and dense attention for Qwen3.6). Fork page: [cuda-graphs.md](../../codebase/core/cuda-graphs.md).

## See also

[session.md](session.md) · [verify.md](verify.md) · [mtp-draft-head.md](mtp-draft-head.md) ·
[layer-forward.md](layer-forward.md)
