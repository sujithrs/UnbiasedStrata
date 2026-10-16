[Upstream docs](../README.md) › [Program](README.md) › Start-up

# Start-up: from argv to "session is up"

**Purpose:** the order in which `main` checks its options, finds out what model and pack it was given, and allocates
everything, up to the log line `strata generate: session is up (engine X)` (`src/program/generate.cpp:3867`). The
order matters: VRAM taken early is not available to the expert cache, and some allocations must happen before tens of
GiB of host pages are registered with CUDA (Windows WDDM refuses them afterwards).

**Covers:** `src/program/generate.cpp:1228-3992`. The expert-cache and helper-tier part is described in more detail
in [expert-tiers.md](expert-tiers.md), and the layer-split part in [layer-split.md](layer-split.md).

## The sequence

| Lines | Step |
| --- | --- |
| 1232 | `stdout` unbuffered, so every line printed before a crash survives a pipe |
| 1233-1241 | AMD on Linux with `STRATA_ARENA_MMAP=1`: set `GPU_PINNED_MIN_XFER_SIZE=1000000` so ROCclr stages copies instead of locking the mapped `experts.bin` pages in place |
| 1245-1251 | `CUDA_MODULE_LOADING=EAGER` unless already set: every kernel loads before the expert cache takes the free VRAM (costs about 30 MB). With lazy loading, a kernel first used mid-prompt found no VRAM for its code |
| 1252-1531 | argv parsing ([cli-flags.md](cli-flags.md)) |
| 1532-1536 | `set_coupled_draft`; `set_peer_portable(--peer-device >= 1)` before any allocation; a warning when parking is configured but turned off by `--prompt-cache 0` or `--conversation-cache-slots 0` |
| 1537-1597 | `--layer-split` and `--split-device` parsing and checks ([layer-split.md](layer-split.md)) |
| 1600-1624 | Expert-source flag combinations (`--shared-expert-arena`, `--resident-*` with mmap, split and helper caches) |
| 1627-1706 | Multi-GPU: a split needs `--expert-profile`; the `STRATA_SPLIT_OWN` own-buffer rule; helper caches placed on the GPUs no stage uses |
| 1707-1722 | HIP: every GPU in use must be an architecture the binary has code for (`gpu_arch_problem`), else exit 1 |
| 1723-1738 | `--prefill auto` without a profile, or with `--no-prefill-borrow`, becomes a fixed 2048; serve without `--tokens` gets a placeholder prompt; otherwise `--tokens` is required |
| 1740-1773 | `--ple-io` and `--kv` checks; the `qsa_set_kv_*` switches; `STRATA_KV_ROT` |
| 1777-1780 | Suffix drafting widens the window: `--mtp-max-t = --spec`, `--spec = min(--spec + 2, 8)` |
| 1782-1825 | `--native SHARD1` expansion (shards, PLE shard by tensor name, the native switches, head and dense GGUFs) |
| 1826-1950 | Remaining option checks: logits stride, PLE, context, rope ranges, sampling, helper placement, peer, short attention, measurement modes with a split, native head shards, PLE key, CPU oracle |
| 1952-1985 | CPU ISA checks (below) |
| 1992-2016 | `expert_layout_load(pack)`, the GPU-kernel support check for each layer's expert types, and `native_pack` |
| 2017-2032 | Helper contexts: CUDA1's (and with `STRATA_EARLY_REMOTE_CONTEXTS=1` every helper's) is created **before** any CUDA0 weight or the host arena |
| 2038-2059 | The PCIe share (`--pcie-frac`): measured link for native packs; the AVX-512 requirement for the canonical pack |
| 2060-2162 | `ModelGeometry g` (canonical defaults) and `K = 10`; with `--native`, the GGUF's `qwen4exp.expert_count` / `expert_used_count` and rope keys; rope resolution and `rope_scaling_set` |
| 2163-2182 | Native pack: the refusals (below) and the token embedding in mapped host memory (`NativeEmbed`, vocabulary 248,320) |
| 2183-2236 | The `skip` set: canonical tensors not loaded because they are served natively; per-stage trimming |
| 2237-2264 | `WeightTable::pool_bytes`, one `cudaMalloc` for the weight arena, `wt.load`. On failure it prints the free VRAM and says another program holds it |
| 2266-2274 | `NativeDense::load`: the native GDN, QSA and shared-expert projections |
| 2276-2313 | Kernel switches (`layer_set_*`, `native_*_set_enabled`); with `--vision`, the identity M-RoPE table `max_context + 64` cells long |
| 2314-2327 | Control vectors loaded (before any graph capture) |
| 2329-2343 | `SessionState ss` declared; the session buffer itself comes later. The main stream is created **non-blocking**: on WDDM the legacy default stream cost about 8x on every launch (comment at `:2331-2337`) |
| 2345-2374 | Half-dump staging (must be set before capture); doorbell (`doorbell_init`) |
| 2376-2499 | PLE table and module wiring (below) |
| 2501-2506 | The `parts` buffer (K x n_embd f32) |
| 2508-2553 | `--split-skip-if-fits` |
| 2554 | `Verifier::set_commit_async(!multi_gpu)` |
| 2555-2640 | Later stages: weights, dense, streams, control vector, M-RoPE table, PCIe probe |
| 2643-2690 | `--expert-profile` read (`read_expert_profile`); `STRATA_PEER_HOT` reordering for a peer GPU; a copy for `--expert-profile-save` |
| 2691-2864 | Layer-split placement search and the profile divided by stage |
| 2866-2927 | Sessions sized to each device's own layer range (the carve); `session_zero`; PLE history wired |
| 2929-3000 | `--batch` slot sessions on every stage, before the cache is sized |
| 3002-3018 | `MtpDrafter::load` (on the last stage's device with a split) |
| 3019-3043 | Head (native Q5_K head unless split) and the logits buffer. Both come **before** the expert cache and the arena |
| 3044-3054 | Helper contexts CUDA2 and CUDA3 (when not created early) |
| 3076-3117 | GPU name, compute capability or gfx arch, CUDA runtime/header mismatch warning (#542), `device_code_error` |
| 3118-3189 | Expert source: `FileExpertSource` (mmap) or `ArenaExpertSource` (resident arena), see [expert-tiers.md](expert-tiers.md) |
| 3190-3197 | `ExpertPool` (workers, pinning, host thread, affinity) |
| 3198-3449 | Expert-cache sizing, sized slots, `--vram-elastic`, the open-and-shrink loop |
| 3450-3552 | Profile fill and slot verification; the later stages' caches |
| 3554-3659 | Helper caches on CUDA1..3; peer GPU |
| 3661-3746 | `Drive` wiring, routing-aware prefetch (`RouterLookahead`), R4 hit-path buffers |
| 3747-3864 | Stage timing, routing dump, per-layer graph capture (canonical pack only), the `--no-capture` refusals |
| 3866-3992 | "session is up"; `run_head`, `d_emb`, `d_next`, `put_input`; sampler parameters; dump files |

After that, the measurement modes run and exit, or the token graph and residency table are built (`:4236-4341`),
followed by either `--serve` or the one-shot path.

## CPU checks (`:1952-1985`, `:2053-2059`)

- No AVX2 with FMA and F16C and no older-CPU build: exit 2 with the CPU's name. Strata runs on Intel Haswell (2013),
  AMD Zen (2017) or newer.
- An older-CPU build (`isa_floor_build()` is `avx` or `none`, compiled with `STRATA_ISA_FLOOR`) checks for AVX or
  SSE4.2 + POPCNT. It says that it is experimental and slow (ggml-cpu kernels), or that the normal build would be
  faster on this CPU.
- Canonical pack: `cpu_require_expert_support()` (the Q2_0 CPU kernels are AVX-512 only). Native pack: without
  AVX-512 the log says the experts run on AVX2. With `STRATA_NO_IQ256` set, or without AVX2, they run on ggml-cpu
  `vec_dot`.

## Pack kinds and model variants

| What | How the driver sees it |
| --- | --- |
| Canonical Q2_0 pack (`pack/full`) | `expert_layout().native == false`. Experts from `experts.bin`, uniform slots of `max_blob` bytes, per-layer CUDA graphs captured (`:3823`), token loop used. Default `--pcie-frac` 0.2, never probed (`:2039-2043`). The PLE key is loaded from the arena (`skip.erase("blk.1.ple_key.weight")`, `:2194`) |
| Native IQ packs (`tools/iq_pack.py`: IQ2_XS, IQ3_XXS, IQ3_S; NVFP4 and Q8_0 are mentioned in comments) | `native == true`. Needs `--native SHARD1`, `--spec >= 2`, no `--keep-canonical`, and `--prefill CHUNK` when the prompt has more than one token (`:2164-2170`). No per-layer graphs; the last prompt token is the first verify window. The token embedding stays in mapped host memory and is skipped from the arena (`:2200`). Slots are sized per pair from the profile (`:3299-3314`). Default `--pcie-frac` 0.55, scaled by the probed link |
| A native pack's experts | `ArenaExpertSource::set_gguf(--native)`: a native pack may take its experts from shard 1 (`:3147`). With `--mmap-experts` and no `experts.bin`, the GGUF shards are mapped in place (`src.set_gguf`, `:3127`) |
| The Coder (GSQ-RCO, 256 of 512 experts) | `g.n_expert` from `qwen4exp.expert_count`, `K` from `qwen4exp.expert_used_count` (`:2074-2079`). The draft layer keeps the canonical 512-expert geometry (`static const ModelGeometry draft_geometry{}`, `:3013`). Each layer's types are checked by `native_expert_supported` (`:2005-2014`) |
| Swift 1.5 | The PLE table is in shard 1 and layers 13-47 are in shard 2, so `--native` adds every shard to `--native-dense-gguf` and finds the PLE shard by tensor name (`:1787-1793`, `:1810-1812`) |
| Unsloth UD-Q4_K_XL and UD-IQ4_XS | The PLE table is in shard 2. Its `blk.1.ple_conv1d.weight` is F32 in the checkpoint and must have been narrowed to F16 by `tools/iq_pack.py`, else exit 1 (`:2465-2478`, #255). The experts usually exceed RAM: `--resident-budget-gib` and the file tier ([expert-tiers.md](expert-tiers.md)) |
| OrcaRouter IQ3_XXS (`--compat-bf16` pack) | `NativeDense::keep_unquantized_ple_key` keeps its BF16 PLE key in the arena (`:2195-2199`, #326) |
| A PLE-only table (`tools/ple_fp8_pack.py`, architecture `strata-ple`) | Not added to `--native-dense-gguf`, because it holds no projections (`:1813-1821`) |
| `--embd-gguf` (`tools/embd_bf16_pack.py`) | The token embedding from that file instead (BF16 as shipped) |

A control vector whose `controlvector.model_hint` is not `qwen4exp` only gets a warning (`:1089-1091`).

## The PLE table (`:2376-2499`)

The per-layer token embedding (PLE) lives on layer 1. Its table is a mapping of the original GGUF shard that holds
`per_layer_token_embd.weight`, and is never loaded into VRAM. Its six weights `blk.1.ple_*` are in the arena.

- `PleIoOptions`: `--ple-io` mode (`direct` = unbuffered reads, `mmap`, `ram` = mmap plus `mlock`), in-flight reads,
  row cache, the I/O thread. The SSD keep-alive (`STRATA_SSD_KEEPALIVE`, default 100 ms;
  `STRATA_SSD_KEEPALIVE_WINDOW`, default 60 s) is active only in `direct` mode with the I/O thread.
- Linux, not WSL: a table on a rotational disk with `--ple-io direct` gets a warning (#605). Random reads can stall a
  prompt for minutes, so `--ple-io ram` or an SSD is recommended.
- Missing `blk.1.ple_*` tensors exit 1. Running without them is a different model (LEDGER L123).
- The key is used as BF16 when it is unquantized (the IQ files' `extra.bin`), else as S2 codes plus f32 scales. With
  `--native-ple-key`, the native form must be ggml type 42, 18, 23 or 8 with Q8_1 activations.
- The conv1d weight must be F16: index kind 5 or 3 in a native pack, kind 0 in the canonical pack.
- The history pointer is wired only after `session_init` carves it (`:2887`).

## Sessions, drafter, head (`:2866-3043`)

- `session_bytes(g, max_context, K, lo, hi)` sizes a session for a layer range, and `session_init` carves it.
  CUDA0's session covers `[0, split_at[0])` with a split, and the whole model otherwise. With KV streaming
  (`--kv-resident`), it logs how many cells per QSA layer stay in VRAM.
- `--batch`: `o.batch` extra sessions per stage, each the same size as the stage's own. Slots that do not fit lower
  the count; fewer than 2 turns batching off with a warning.
- `MtpDrafter::load(o.mtp, draft_geometry, ss, spec, err, mtp_window)`, allocated before the host arena is
  registered (WDDM refused the draft weights afterwards).
- The head: `wt.find("output.weight")` sets `n_vocab`. The native Q5_K head loads here, before the expert cache, so
  the cache is sized around it (#620: 128K IQ3_S once ended with 30 MiB free).

## Platform notes

- **Windows and WSL2 (WDDM):** `under_wddm()` is true on Windows and when `/dev/dxg` exists (`:110-117`). A
  multi-GPU run pins at most 8 GiB of the arena unless `STRATA_ARENA_PIN_GIB` says otherwise (`:3153-3161`). The
  expert cache is reopened smaller when the free figure after writing the slots is below the reserve. A Windows
  commit-limit failure retries at 3/4 size up to 8 times and points to the page-file setting (`:3376-3399`). A
  Task-Scheduler start that loads slower than 0.2 GiB/s gets a hint (`:3178-3187`).
- **HIP:** the GPU's `gcnArchName` is logged; on Windows, the loaded `amdhip64_7.dll` path is logged too
  (`:3084-3093`). PCIe bursts are timed on the host on Windows HIP (`:1172-1182`). `--vram-elastic` is off (NVIDIA
  only, `:3323-3325`). `pin_blobs` and `GPU_PINNED_MIN_XFER_SIZE` apply to file-backed arenas
  (`STRATA_ARENA_MMAP=1`).
- **CUDA:** a runtime/header major-version mismatch is warned about (#542). `STRATA_EMULATE_CC` marks an emulated
  compute capability in the GPU line.

## Gotchas

- `expert_layout_load` is called with the canonical `ModelGeometry` defaults (`:1996-1997`), before the GGUF's
  `expert_count` is read at `:2078`. How a pruned pack's layout reconciles the two is decided inside
  `expert_layout_load` and is not shown here.
- A refusal exits 2 (a usage problem) or 1 (a runtime failure). The server reads the log to explain a failed start.

## In UnbiasedStrata

The fork adds three steps:

- the whole `ModelGeometry` is read from the GGUF (`read_model_geometry`, `qwen4exp` or `qwen35moe`);
- the `--vram-cap` blocker and the `--ram-cap` clamp are set up;
- a Qwen3.6 VRAM plan can refuse the start or lower `--max-context`.

The fork makes the PLE optional for models that have none, and runs `check_all` on the layout after the native
projections. See [../../codebase/program/startup.md](../../codebase/program/startup.md).

## See also

- [expert-tiers.md](expert-tiers.md), [layer-split.md](layer-split.md), [cli-flags.md](cli-flags.md).
- Upstream [docs/MODELS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MODELS.md)
  for the variants and their sizes.
