[Upstream docs](../README.md) › [Core](README.md) › Verify window

# The speculative verify window (`Verifier`)

## Purpose

Speculative decoding: a drafter (the MTP layer, [mtp-draft-head.md](mtp-draft-head.md), or prompt lookup from
`src/spec/`) proposes up to 7 tokens; the `Verifier` runs the last accepted token and the drafts, T <= 8 tokens at
consecutive positions, through all 48 layers in **one captured graph** and produces the head's pick for every row.
Row t's argmax is bit for bit what plain greedy decode would produce after token t, because every kernel is either the
single-token kernel applied per row or a multi-token kernel with the same per-row arithmetic (`verify.hpp:3-8`). The
program accepts the longest prefix of drafts that matches, then calls `commit(n_keep)`. This is the default serving
path (setup passes `--spec 4 --mtp ...`, `setup.py:4226`) and the only decode path for native packs.

Files covered:
[include/strata/core/verify.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/verify.hpp),
[src/core/verify.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/verify.cpp).

## Requirements (`Verifier::init`, `verify.cpp:301-531`)

* `2 <= max_t <= kVerifyMaxT (8)` and `<= cpu::MAXT`.
* A profile-filled VRAM expert tier: `VerifyHits` (`verify.hpp:49-56`) with the device and host residency tables,
  the slot arena base, per-slot offsets for sized (native) slots, and the blob size. Without it init fails with a
  message that names `--expert-profile` / `--expert-cache` and what to free (`:322-328`).
* `layer_verify_compatible` ([layer-forward.md](layer-forward.md)): native BF16 projections, fused GR, fused native
  GDN, split-K attention, block selection, native indexer.
* Flash-Next geometry: `fused_gr_supported(2560, 4, 320)`, `k == 10`, GDN state 128, conv 4 (`:335-339`). This holds
  for the Coder too (10 active experts).
* A layer range with hand-off buffers when it is a split stage (`:340-345`).

`init` allocates mapped staging (tokens, step records, positions, commit record, PLE rows, picks, the doorbell
payload and four handshake words, the miss rows) and one device arena (`Bump` carve, 256-byte aligned). It decides
`all_resident_` (every expert of the stage's layers in VRAM: a **zero-doorbell graph** with no CPU handshakes,
`STRATA_VERIFY_ALL_RESIDENT=0` disables) and `device_plan_` (E-6, `STRATA_VERIFY_DEVICE_PLAN=1`, off by default:
"exact, but neutral").

## One window: `run(T, tokens, pos0, pool, user, out, err)` (`verify.cpp:1390-1582`)

```
capture(T) (first use), capture_commit
stage_inputs: tokens, step records, positions into mapped staging
[PLE stage] hash T tokens from ss.ple_prev; prefetch their rows
launch exec_[T]
for each layer l in the stage, each token group:
    wait for the ring  -> pool(user, x_f[n_tok], ids, n_tok, k, out, l)   (expert_pool_dispatch_multi)
                          the pool fills the GPU plan (flag A) and starts DMA for the PCIe share (flag B)
                       -> raise flag (CPU rows in mapped memory)
    [layer 0] collect this window's PLE rows so the SSD reads overlap layer 0 and the drafter
cudaStreamSynchronize; [sampled request] sample rows host-side; out[t] = pick of row t
```

Inside the graph (`record_window`, `:545-1174`), per layer and token group:

* **pre** (`:619-994`): hyper-connection reads (fused, folding the previous layer's write), the PLE block at layer 1
  for each row, then GDN rows with the T-token GDN kernels or QSA rows (batched GEMVs, norms and RoPE over the
  window, `STRATA_DEC_BATCH=0` for token by token), the router for all rows (`moe_route_window`), the shared expert
  (on a second stream unless `STRATA_SH_STREAM=0`), and the doorbell publish of `x`, ids and weights (a variant that
  also publishes residency when no remote optimisation is attached).
* **post** (`:996-1118`): wait for the plan, run the VRAM groups with the grouped expert kernels
  (`native_expert_grouped` on a native pack, `moe_grouped_s2` on the canonical pack), wait for the PCIe share and run
  it (`pcie_mode` 0 DMA into staging, 1 read the mapped arena directly, 2 a copy kernel stages it), wait for the CPU's
  rows, add, combine with the router weights.
* **head** (`:1120-1172`): final hyper-connection read per row, then `native_mmvq` over T columns with the native
  head (or `lm_head` per row), and the greedy argmax recorded into mapped `h_out_`.

`set_split(true)` (`--spec-split`, off by default; the member comment says "exact but slower") splits a window into
two token groups so the CPU experts of one overlap the GPU work of the other.

### Sampling and penalties

Greedy is recorded in the graph. A sampled or penalised request is sampled again on the host side after the replay
with the request's `SamplerParams`; row t draws `Philox(seed, pos0 + t)`, so a seed reproduces the same text however
the windows were cut (`verify.cpp:1542-1546`). `set_history` supplies one penalty-history row per window row
(`verify.hpp:96-106`); `set_head_sampling(false)` skips it for windows whose picks are discarded (prompt reads).

## Commit (`commit(n_keep)`, `verify.cpp:1699-1728`; graph at `:1265-1330`)

The window appended K/V and indexer keys for all T positions and left the GDN state alone. The commit graph makes the
first `n_keep` tokens permanent:

* GDN: replays the accepted rows from inputs the window stored (`gdn_conv_commit`, `gdn_step_norm_multi` over
  `qkv_L_`, `h_L_`, `gate_L_`, `beta_L_`).
* QSA indexer: restores the key tail from its snapshot and re-appends the accepted keys (a rejected key could land in
  a slot the current block still needs).
* PLE: copies the history snapshot taken after token `n_keep - 1`.
* Rejected K/V cells are simply overwritten when those positions are processed again.

`set_commit_async(true)` (one GPU only, `generate.cpp:2554`; `STRATA_COMMIT_SYNC=1` keeps the wait) lets `commit`
return without waiting: the next window follows on the same stream, and anything else that touches the session
first calls `wait_commit`. On the host, `commit` advances `ss.ple_prev` once per accepted token on the stage that holds
layer 1.

## Batch windows (`init_slots`, `run_slots`, `run_slot_rows`, `commit_slots`; `verify.cpp:1743-2063`)

`--batch N` serves N independent sequences: a batch window holds S <= max_t rows, one token per slot, each reading and
writing its own `SessionState` (same layer range and `max_cells`). Weights are read once for all rows. Greedy, no
drafts; per-slot sampling is redrawn on the last stage (`set_slot_sampling`, penalties not applied). Graphs are cached
per row set (`batch_key`). `batch_launch` / `batch_poll` (`:2065-2149`) run the stages of a layer split as a
pipeline: stage k runs group g while stage k+1 runs group g-1, all served from one host thread.

## Layer split stages

`set_stage(lb, le, handoff_in, handoff_out)` and `set_next(next, user)` (`verify.hpp:112-126`) make one `Verifier`
per GPU. A later stage reads, per token, the residual, the last layer's pending write and its inject from the mapped
hand-off (`handoff_floats` = `hc*n_embd + n_embd + hc`), so the split is bit-exact; `run` and `commit` continue into
the next stage. Each object records its device and wraps public calls in `OnDevice`. The MTP drafter lives on the
last stage's device. See
[MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MULTI_GPU.md).

## Diagnostics

| Hook | Effect |
| --- | --- |
| `diag(f)` / `trace_dump(f)` | Watchdog view of the window in flight; `STRATA_VERIFY_TRACE=1` adds a host event ring and GPU breadcrumbs in mapped memory (#649) |
| `release_gpu_waits(timeout)` | #267: raise every flag, wait for the streams, refuse later windows |
| `STRATA_VERIFY_PROFILE=1`, `profile_report()` | Per-stage GPU stamps, one line per call |
| `window_logprobs(...)` (`STRATA_LOGPOS`) | Per-row log-probabilities for measurement |
| `copy_logits(t, host)` | Row t of the last window's head logits |
| `STRATA_VERIFY_DEBUG`, `STRATA_DBG_NAN`, `STRATA_VERIFY_NODES`, `STRATA_TEST_VERIFY_STALL=N` | Debug prints, first non-finite head, graph listing, a test hook that withholds a flag in window N |

## Configuration

| Flag / env | Effect |
| --- | --- |
| `--spec T` | Window size (drafts + 1); also `max_t` with `--batch` |
| `--batch N` | Batch windows |
| `--spec-split` | Two token groups per window |
| `--pcie-mode auto\|dma\|direct\|kernel` | `dma` = 0, `direct` = 1, anything else = 2 (`generate.cpp:5168`) |
| `--pcie-frac F` | The PCIe share the pool plans ([experts/dispatch.md](experts/dispatch.md)) |
| `STRATA_COMMIT_SYNC`, `STRATA_VERIFY_ALL_RESIDENT`, `STRATA_VERIFY_DEVICE_PLAN`, `STRATA_DEC_BATCH`, `STRATA_SH_STREAM`, `STRATA_PLE_BATCH`, `STRATA_FUSE_HEAD_GR` | A/B switches named above |

## Platform notes

* HIP (`STRATA_USE_HIP`): `STRATA_VERIFY_COHERENT=1` allocates the handshake words in explicitly coherent host memory
  and `STRATA_DOORBELL_STORE=1` stores each ring instead of incrementing it over PCIe (opt-in A/Bs for gfx1030 verify
  timeouts, #649; `verify.cpp:72-98`, `:947-951`). The trace prints `HIP_HOST_COHERENT`, `HSA_ENABLE_SDMA` and
  `GPU_MAX_HW_QUEUES`. The node listing is CUDA only.
* Windows: `raise_flag` uses `_InterlockedCompareExchange`, elsewhere `__atomic_compare_exchange_n` (`:1603-1616`).
* Multi-GPU: mapped buffers are portable only when a peer tier exists (`peer_portable()`), because the peer card
  writes its rows into them.

## Tests

No unit test of `verify.cpp` in `tests/` of this snapshot; correctness is the bitwise-equality argument above plus
the kernels' parity tests (`route_window_parity`, `mmvq_multi_parity`, `iq_multi_parity`, `gdn_rec_parity`).

## Gotchas

* A window past the context is refused (`pos0 + T > max_cells`).
* Once `release_gpu_waits` has run, the verifier refuses every later window ("restart the engine").
* With `set_commit_async`, a reader of the session from another stream or the host must call `wait_commit` first.

## In UnbiasedStrata

Changed: the fork added `verify_geometry_supported` (the plain block at hc 1, `k` equal to the file's
`expert_used_count`) and plain-block branches, and commit b4ee41e skips the indexer tail in the commit capture when
there is no indexer. Fork page: [verify.md](../../codebase/core/verify.md).

## See also

[mtp-draft-head.md](mtp-draft-head.md) · [cuda-graphs.md](cuda-graphs.md) · [experts/dispatch.md](experts/dispatch.md) ·
[session.md](session.md)
