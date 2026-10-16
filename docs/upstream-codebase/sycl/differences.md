[Upstream docs](../README.md) › [Intel SYCL port](README.md) › Differences from the CUDA engine

# What the SYCL engine does differently

## Purpose

The port computes the same model with the same algorithms as the CUDA engine, and most of its code is a mechanical
translation ([migration.md](migration.md)). This page lists what is not mechanical: the changes needed for
correctness on an Intel GPU, the changes made for speed on the Arc Pro B70, the features the port adds (expert
streaming from the GGUF, a pinned host mirror, logprobs), and the CUDA features it leaves out. Every item cites the
port's file; the same file without `sycl/` is upstream's CUDA version.

Files covered: the "dpct + hand", "dpct + fixups", "hand", "copy + hand", "port-only" and "stub" rows of the
[inventory](layout.md#inventory); in particular `sycl/src/program/generate.cpp`, `sycl/src/prefill/prefill.cpp`,
`sycl/src/core/verify.cpp`, `sycl/src/core/gguf_expert_source.cpp`, `sycl/src/kernels/cuda/{iq_kernels,
native_mmvq, fused_gr, verify_kernels, qsa_prompt_attn_xmx}.dp.cpp` and the port's headers
`sycl/include/strata/sycl_{queue,doorbell,math}.hpp`.

## 1. Translation layer: what dpct changed everywhere

| CUDA | In the port |
|---|---|
| `__global__` kernel, `<<<grid, block, smem, stream>>>` | a `__dpct_inline__` function launched with `q->parallel_for(sycl::nd_range<3>(...), [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {...})`; `threadIdx`/`blockIdx` read from `sycl::ext::oneapi::this_work_item::get_nd_item<3>()` |
| `cudaStream_t` (null = default stream) | `sycl::queue*`, always through `strata::q_of(stream)` (`sycl/include/strata/sycl_queue.hpp:55-57`), which maps null to `dpct::get_in_order_queue()`; 286 call sites in 51 files |
| CUDA graphs (stream capture) | `dpct::experimental::begin_recording` / `end_recording` producing `sycl_ext_oneapi_graph` command graphs, launched with `queue->ext_oneapi_graph(exec)` |
| `__shfl_xor_sync` and friends | `dpct::experimental::permute_sub_group_by_xor`, `select_from_sub_group`, `shift_sub_group_left/right` |
| `__dp4a` | `strata::dp4a` (`sycl/include/strata/sycl_math.hpp:41-43`), a wrapper of `dpct::dp4a` that IGC pattern-matches into the DP4A instruction |
| cuBLAS | `dpct::blas::gemm` over oneMKL (`sycl/src/prefill/gemm.dp.cpp:454`); link flag `-qmkl=sequential` |
| `cudaMemcpy` / `cudaMemset` | `dpct::get_in_order_queue().memcpy/memset(...).wait()` (the `.wait()` added by fixup 7d) |
| error codes, `cudaGetLastError` | SYCL exceptions; functions are wrapped in `try { ... } catch (sycl::exception const&)` and `DPCT_CHECK_ERROR` |
| `cudaStreamQuery` | `queue->ext_oneapi_empty()` (`sycl/src/core/verify.cpp:190`) |

## 2. Correctness fixes for the Intel GPU

- **Host-mapped flags (doorbells).** The decode loop is a handshake through host-mapped memory: the GPU rings a
  sequence number the host polls, and a one-thread kernel spins on a flag the host writes. CUDA uses `volatile`
  accesses, which bypass the caches on NVIDIA; on Intel a `volatile` device load read its first value from L3
  forever. `strata::sys_load` / `sys_store` (`sycl/include/strata/sycl_doorbell.hpp:16-22`) are relaxed
  `sycl::atomic_ref` accesses with `memory_scope::system`, the store followed by a system-scope release fence.
  Fixup 7c rewrites the doorbell sites in `elementwise.dp.cpp:377-383,599` and
  `verify_kernels.dp.cpp:914-1087` to use them.
- **Bounded spins.** An unbounded device spin that never sees its flag makes the xe driver time out the queue and
  reset the GT node by node; the card stayed wedged until a reboot (measured twice, per the header comment). Every
  spin is capped at `strata::kSpinMax` (`sycl_doorbell.hpp:28`), so the failure becomes a wrong window that the
  verifier's checks catch. Gotcha: the constant is `20u * 1000u` with the comment "experiment: 100x smaller", while
  the paragraph above it still speaks of "~2 M host-memory reads".
- **32-lane sub-groups.** Every kernel assumes 32-thread warps (`lane = tid & 31`, warp-synchronous reductions).
  dpct pinned the sub-group size only where it saw a shuffle (INTEL.md: 134 of 289 launches); the rest would get
  Xe2's default of 16 and race. `-fsycl-default-sub-group-size=32` (`sycl/CMakeLists.txt:46`) forces 32
  everywhere; 177 launches carry an explicit `reqd_sub_group_size`.
- **Synchronous copies.** See fixup 7d and the parity-test rule at `fixups.py:292-299`: dpct's copies did not wait,
  and the streaming expert source then filled the cache from reused buffers (non-deterministic residuals, NaN by
  layer 5, per INTEL.md).
- **Floating point.** `-fp-model=precise` for all code and correctly rounded fp32 divide and sqrt in device code
  ([build.md](build.md#compiler-flags)); without the latter `quantize_act_parity` had 303k mismatches against ggml.
- **Type punning.** ggml's `vec_dot_iq2_xs` reads four 16-bit codes through a `uint16_t*` cast to an `int2`, which
  the SYCL device compiler does not honour; the port extracts them with shifts (`iq_kernels.dp.cpp:160`, `:806`).
- **dpct mistranslations fixed by hand.** The s2 activation rounding lost its parentheses and rounded every value
  to 0 (`s2_expert_grouped.dp.cpp:650`); the others are scripted in `fixups.py` (the `__ldg` cast, `__fadd_rn`
  ternaries, the IQ4_XS table, the stage-room free memory).
- **Exit.** In serve mode the engine aborted at exit because destructors ran queue waits against a runtime already
  shutting down. After `QUIT` it now drains the device and calls `std::_Exit(0)` (`generate.cpp:7362-7367`), and
  `Verifier::~Verifier` is best-effort (`verify.cpp:278`).
- **Level Zero v2 event waits.** Under the Level Zero v2 adapter, a host thread's wait on a copy-queue event left
  that event unable to release another queue's barrier, and long prompts hung once the stager's ring wrapped
  (INTEL.md, "Prompt-slot borrowing: the hang"). The stager (`prefill.cpp:386-395`, `Stager::issued_one`) and the
  PLE upload (`prefill.cpp:566-570`, `ple_done_seq`) now have the copy queue write a sequence number into
  page-locked memory that the host polls; no host thread waits on a queue event there.

## 3. Expert storage: `--stream-experts` and the host mirror

The CUDA engine keeps every expert in host RAM (an arena, 23.4 GiB for the Coder) and caches the hot ones in VRAM.
The port adds a mode for a card that can hold (nearly) every expert:

- **`--stream-experts`** (`generate.cpp:1568`; requires a native pack and `--native SHARD1`,
  `generate.cpp:3303-3312`). The expert source is `strata::core::GgufExpertSource`
  (`sycl/include/strata/core/gguf_expert_source.hpp`, `sycl/src/core/gguf_expert_source.cpp`): `blob(layer, expert)`
  reads the expert's three tensor slices (gate, up, down) straight from the GGUF into one of 512 ring buffers
  (`kRing`, `gguf_expert_source.cpp:17`); a pointer stays valid until the ring wraps. `read_into()` lets the prompt
  path's stager threads read into their own buffers, because holding ring pointers across ~1,900 reads of a 512-slot
  ring returned overwritten blobs (the 80K-token run of 2026-09-30). `open()` refuses a canonical pack (it needs
  the GGUF offsets in `native_experts.txt`). The file of each slice is looked up per layer and role (`3 * layer +
  role`, `fd_of`), since a shard boundary can fall inside a layer; indexing by layer alone produced NaN on Swift
  1.5, whose layers 13-47 are in shard 2.
- **Cache fill pipeline** (`generate.cpp:3682-3690`). Slots are admitted in profile order, worker threads read in
  file order into page-locked batches, and a batch's copies overlap the next batch's reads. INTEL.md: a cold Coder
  start 82 s to 26 s. `STRATA_FILL_SERIAL=1` restores the serial fill.
- **Pinned host mirror** ("plan item 2", `generate.cpp:3785-3836`). Experts that get no VRAM slot are read once into
  pinned USM host memory (`GgufExpertSource::mirror`, up to `STRATA_MIRROR_MIB`, default `MemAvailable` minus 4 GiB;
  `0` turns it off). They then report `pinned()` with a `device_alias()`, and
  `resident_plan_set_mirror()` (`sycl/include/strata/kernels/resident_plan_mirror.hpp`) lets the device-built
  verify plan point the expert kernels straight at them over PCIe. Without the mirror each miss was an SSD read.
  With `STRATA_VERIFY_NO_HOST` set and experts neither resident nor mirrored, the engine prints a warning
  (`generate.cpp:3832`).
- **Prompt-slot borrowing default** (`generate.cpp:1796-1805`). Lending expert-cache slots to the prompt path
  costs about 1 s per prompt but keeps every expert in VRAM at long contexts, so the port borrows by default only
  above a 32K context; `--prefill-borrow` / `--no-prefill-borrow` (`generate.cpp:1468-1469`) decide explicitly.
- **`--vram-elastic`** and the segmented arena are compiled out (`expert_cache.cpp:157-419`, "CUDA-only (#533)").

## 4. Prompt path (`sycl/src/prefill/prefill.cpp`)

| Change | Where | Switch |
|---|---|---|
| Stager threads read streamed experts from the GGUF themselves (a job may name `(layer, expert)` instead of a host pointer) | `:236-244`, `:3077` | `STRATA_GGUF_PLAN_READ=1` takes plan pointers instead |
| Stager and PLE upload completion by polled sequence numbers (see section 2) | `:386`, `:566` | none |
| The stream-all walk (copy every non-resident expert of a layer ahead) runs only when fewer than 10% of (layer, expert) pairs are outside VRAM; the share comes from `set_nonresident_share()` called by `generate.cpp:4833-4836` | `:111`, `:155-175`, `:1418` | `STRATA_PREFILL_STREAM_ALL=1` / `=0` |
| A short first chunk so the GPU starts while the rest of the PLE rows are still read | `:1656-1662`, `:1820` | `STRATA_PREFILL_FIRST=<tokens>` (default 256; 0 off) |
| Debug: wait for the GPU at every phase mark and log it | `:1500-1503` | `STRATA_PREFILL_SYNC=1` |
| Debug: dump the selected cells of the last chunk's first QSA layer | `:2492-2497` | `STRATA_DUMP_SEL=<file>` |
| The key-head DeltaNet kernel's gate is an NVIDIA SM-count rule | `prefill/kernels.dp.cpp:1003` | `STRATA_GDN_KEYHEAD=1` (upstream variable) |
| Upstream's fused int8 prompt kernels (#136) are stubbed: `built()` and `available()` return false | `sycl/src/prefill/moe_fused_stub.cpp` | none |

Two non-CUDA host files are copied and changed for the prompt path's I/O: `sycl/src/platform/direct_file.cpp:314-317`
raises the default O_DIRECT reader count from 16 to 64 threads (`STRATA_IO_THREADS` still overrides), and
`sycl/src/ngram/ple_reader.cpp` adds a per-gather timing trace (`STRATA_PLE_TRACE=1`, `:95-99`).

## 5. Decode loop and verify window

| Change | Where | Switch |
|---|---|---|
| Every window graph and the drafter's graphs are captured at load (`Verifier::warm`, `Mtp` warm-up) instead of on first use | `generate.cpp:7971-7976`, `verify.hpp:129-131`, `mtp.hpp:52` | `STRATA_WARM_GRAPHS=0` |
| Deferred commit: `commit(n, err, false)` leaves the commit graph running while the drafter's round runs on its own queue; `commit_finish()` collects it. Upstream's own `set_commit_async` / `wait_commit` (0.1.32) are mapped onto it | `verify.hpp:125-128`, `verify.cpp:1742-1746`, `generate.cpp:8076` | `STRATA_COMMIT_SYNC=1` (upstream variable) |
| With every routed expert resident, the host waits for the whole window instead of per-layer rings, because a kernel's writes to host-mapped memory are not reliably visible while the graph runs | `verify.cpp:1357-1365`, `generate.cpp:3832` | `STRATA_VERIFY_NO_HOST=1` (set by `strata-sycl.sh`) |
| Eager mode: the window runs on the queue instead of as a graph, with per-stage host timestamps | `verify.cpp:580`, `:1104`, `:1332`, `:1568` | `STRATA_VERIFY_EAGER=1` |
| Per-layer residual ladder (token 0) for debugging | `verify.cpp:536`, `:1473`, `verify.hpp:224-225` | `STRATA_VERIFY_DEBUG=1` (upstream variable, extended) |
| `resident_plan` groups entries in parallel, one thread per entry (thread 0 alone took 73 us per layer) | `verify_kernels.dp.cpp:924-960` | `STRATA_PLAN_PARALLEL=0` |
| The plan reads the host-mirror table | `verify_kernels.dp.cpp:1124` | via `STRATA_MIRROR_MIB` |
| Graph-node kernel names are not available (`"kernel"`), the stage profiler's `%globaltimer` reads 0 | `fixups.py` rules 10 and 14 | none |
| The fused multi-token router and the native router also for 256 experts (the Coder) | `verify.cpp:877`, `layer.cpp:412`, `native_router.dp.cpp:70`, `native_router.hpp:19-20` | none |
| `mtp.cpp` takes the device's own free-memory figure | `mtp.cpp:469` | none |

## 6. Serve protocol addition: logprobs

The engine's serve mode accepts `logprobs=K` (clamped to -1..20) on a `GEN` line (`generate.cpp:6310-6313`,
`:6339`). After each `T <id>` line it then writes `LP <logprob> <id>:<logprob> ...` with the token's
log-probability and the K most likely tokens', computed from the verify window's head logits with a log-sum-exp
(before temperature, penalties and sampling), or `LP nan` when the logits row cannot be copied
(`generate.cpp:7046-7066`). A server that does not ask never sees an `LP` line; upstream's `serve/server.py` ignores
them and `sycl/serve/server_intel.py` turns them into OpenAI `logprobs` ([serving.md](serving.md)).

## 7. Kernel work for Xe2 (speed)

All of these keep the CUDA kernel's contract; INTEL.md reports output tokens identical or bitwise-equal sums unless
noted, measured on the Arc Pro B70.

| Area | Change | File | Switch to the old path |
|---|---|---|---|
| i-quant expert dots | SWAR sign compare/subtract instead of dpct's per-byte loops | `iq_kernels.dp.cpp:103` | none |
| | IQ4_NL codebook in registers, aligned weight loads | `iq_kernels.dp.cpp:333`, `:997`, `:1021` | none |
| | Grouped expert kernels with 8 lanes per row, 32 rows per group (port default for every format, also upstream's Q4_K/Q5_K/Q5_1/Q8_0 experts) | `iq_kernels.dp.cpp:676-684`, `:2015-2019` | `STRATA_EXPERT_SPLIT=1` runs upstream's multi kernels |
| | Lanes per row at run time (4/8/16/32) | `iq_kernels.dp.cpp:2161-2170` | `STRATA_GU_LANES`, `STRATA_DOWN_LANES` (compile-time default `STRATA_EXPERT_LANES` = 8) |
| | Codebook grid in local memory | `iq_kernels.dp.cpp:366-370` | compile-time `STRATA_GRID_SLM=1` (default 0) |
| | Expert dequant writes a run of FP16 values as one vector store | `iq_kernels.dp.cpp:1634`, `:1752` | none |
| Dense matvec (`native_mmvq`) | Q6_K 16-byte loads from 2-byte-aligned blocks as two aligned loads and a shift (`load16_a2`) | `native_mmvq.dp.cpp:1319-1349` | `STRATA_MMVQ_A2=0` |
| | Wide 16-byte-load kernels for Q4_K, Q5_K, IQ4_XS | `:1557`, `:1685` | `STRATA_MMVQ_WIDE_K=0` |
| | Wide kernels for Q8_0 and IQ4_NL | `:1704`, `:1774` | `STRATA_MMVQ_WIDE_32=0` |
| | Wide Q6_K kernel gate and width | `:1803`, `:2505-2512` | `STRATA_MMVQ_WIDE=0`; `STRATA_MMVQ_WIDE_MIN`, `STRATA_MMVQ_WIDE_MAXC`, `STRATA_MMVQ_WIDE_RPW` |
| | Software prefetch, row-blocked kernel (bench knobs, off) | `:1435`, `:1504`, `:1815` | `STRATA_MMVQ_PF`, `STRATA_MMVQ_RPW` |
| | SIMD16 variants (no gain in the engine per INTEL.md) | `:148-162` | `STRATA_MMVQ_SG` = 16, 1, 2 or 3 (default 32) |
| | Q6_K at a 224-byte stride (experiment) | `:1793-1826`, `native_mmvq.hpp:33-35` | bench only |
| Hyper-connection read (`fused_gr`) | Norm per (token, stream) | `fused_gr.dp.cpp:324`, `:375-379` | `STRATA_GR_NORM_SPLIT=0` |
| | Down kernel without the 60 KB shared tile | `:479`, `:597` | `STRATA_GR_DOWN_DIRECT=0` |
| | Down projection sliced by columns, fixed-order reduce | `:526`, `:583` | `STRATA_GR_DOWN_SLICED=0` |
| | Upstream's split/staged read variants and their self-test are not used (the self-test crashed on the B70) | `:2186-2190` | `STRATA_HC_CHECK=1` runs it |
| | Tile size from the device's local memory (no per-kernel opt-in) | `:1635`, `:1778` and fixup at `fixups.py:102-105` | none |

## 8. XMX (Intel's matrix engines)

Two `joint_matrix` kernels exist; neither is on by default because both lost to the existing paths on the B70
(INTEL.md "XMX"):

- `qsa_prompt_attn_xmx` (`sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp`, port of the `mma.sync` prompt
  attention, "v2"). `qsa_prompt_attn_batch` tries it first (`qsa_prompt_attn.dp.cpp:1578`); it returns false unless
  `STRATA_PROMPT_ATTN_XMX` is `1` (64-cell chunks) or `32` (32-cell chunks), and for Q4_0 KV pools or other head
  shapes (`:364-369`). INTEL.md: correct, about 2x slower than the FP32 fallback per chunk.
- `xmx_gemm_iq` (`iq_kernels.dp.cpp:2597-2720`, declared in `sycl/include/strata/kernels/xmx_gemm.hpp`): a fused
  dequant + GEMM from quantized rows. Only `xmx_gemm_bench` calls it; the engine's prompt path does not. INTEL.md:
  4-5x slower than dequant + oneMKL. oneMKL's FP16 GEMMs already use XMX.

INTEL.md also mentions `STRATA_EXPERT_XMX=1` (XMX expert dots, "parked"); that variable does not occur in this
snapshot's sources.

## 9. What the port leaves out

| CUDA feature | In the port |
|---|---|
| Tensor-core kernels with inline PTX (`mma.sync`, `ldmatrix`, `cp.async`) in `qsa_prompt_attn`, `qsa_select`'s block scores and `native_qsa_score` | compiled out by fixup 11; the older-card fallback kernels run |
| sm_90 thread-block-cluster greedy sampler and QSA top-k (0.1.38) | off (`sampler.dp.cpp:1392`, `qsa_select.dp.cpp:1565`) |
| `cp.async` in the key-head DeltaNet prompt kernel | plain copies (`prefill/kernels.dp.cpp:778`) |
| `--peer-device` (a second GPU as an expert-cache tier) | stub: `PeerExperts::open` fails with "not ported to SYCL yet" (`sycl/src/core/peer_experts.cpp`) |
| `--vram-elastic` | compiled out (`expert_cache.cpp`) |
| ggml MMQ prompt path (`moe_mmq.cu`) and fused int8 prompt kernels (#136) | not built; stub for the latter |
| `src/core/remote_expert_opt.cu`, `--remote-expert-opt` | not migrated; the port's `generate.cpp` does not parse the flag |
| `--batch` / `--slots`, `--batch-groups`, `--trim-stage-weights` (upstream 0.1.39 `generate.cpp:1375-1399`) | not parsed by the port's `generate.cpp`, which exits with `unknown argument` (`generate.cpp:1598-1604`). From reading the code: `setup.py --parallel N` writes `"parallel": N` to the config, `setup_intel.py` keeps that key, and `serve/server.py`'s `parallel_args` (`serve/server.py:1485-1505`) then passes `--batch N`, so a SYCL config with `"parallel"` of 2 or more would not start. Not tested |
| `cudaInitDevice` spin scheduling | dropped (fixup 15) |
| `%globaltimer` stage profiler | reads zeros (fixup 10) |
| Images (the vision path) | not wired; `setup_intel.py` adds `--vision none` unless given and answers "none" for `hip_vision` |

Port-only CLI flags in `generate.cpp`: `--stream-experts`, `--prefill-borrow` (and `--no-prefill-borrow`, which
upstream also has).

## Configuration: port-only environment variables

Variables that occur in `sycl/src` or `sycl/include/strata` but not in upstream's `src/` or `include/`:

| Variable | Default | Effect | Where |
|---|---|---|---|
| `STRATA_VERIFY_NO_HOST` | unset | host waits for the whole verify window (all-resident only) | `verify.cpp:1362`, `generate.cpp:3832` |
| `STRATA_VERIFY_EAGER` | unset | verify window without a graph, per-stage timing | `verify.cpp:582`, `:1104` |
| `STRATA_WARM_GRAPHS` | on | `0`: capture graphs on first use | `generate.cpp:7974` |
| `STRATA_MIRROR_MIB` | `MemAvailable` - 4 GiB | cap of the pinned host mirror; `0` off | `generate.cpp:3806` |
| `STRATA_FILL_SERIAL` | unset | serial cache fill from the GGUF | `generate.cpp:3687` |
| `STRATA_PREFILL_FIRST` | 256 | first prompt chunk length; `0` off | `prefill.cpp:1659` |
| `STRATA_PREFILL_STREAM_ALL` | auto (< 10% non-resident) | `1`/`0` force the stream-all walk | `prefill.cpp:167` |
| `STRATA_PREFILL_SYNC` | unset | debug sync per phase | `prefill.cpp:1502` |
| `STRATA_GGUF_PLAN_READ` | unset | stager uses plan pointers, not GGUF reads | `prefill.cpp:1923` |
| `STRATA_DUMP_SEL` | unset | dump QSA selections to a file | `prefill.cpp:2495` |
| `STRATA_PLE_TRACE` | unset | per-gather PLE timing trace | `ngram/ple_reader.cpp:97` |
| `STRATA_PLAN_PARALLEL` | on | `0`: `resident_plan` grouped by thread 0 | `verify_kernels.dp.cpp:926` |
| `STRATA_PROMPT_ATTN_XMX` | 0 | `1`/`32`: XMX prompt attention | `qsa_prompt_attn_xmx.dp.cpp:367` |
| `STRATA_EXPERT_SPLIT` | unset | upstream's multi expert kernels | `iq_kernels.dp.cpp:2019` |
| `STRATA_GU_LANES`, `STRATA_DOWN_LANES` | 8 | lanes per row of the port's expert kernels | `iq_kernels.dp.cpp:2169-2170` |
| `STRATA_MMVQ_A2`, `_WIDE`, `_WIDE_K`, `_WIDE_32` | on | `0`: older matvec paths | `native_mmvq.dp.cpp:1349`, `:1803`, `:1685`, `:1774` |
| `STRATA_MMVQ_WIDE_MIN`, `_WIDE_MAXC`, `_WIDE_RPW` | 0, 8, 1 (with A2) | wide Q6_K gate and shape | `native_mmvq.dp.cpp:2509-2510`, `:1508` |
| `STRATA_MMVQ_PF`, `STRATA_MMVQ_RPW` | 0 | bench knobs | `native_mmvq.dp.cpp:1504`, `:1815` |
| `STRATA_MMVQ_SG` | 32 | sub-group width of the decode matvecs | `native_mmvq.dp.cpp:159` |
| `STRATA_GR_NORM_SPLIT`, `STRATA_GR_DOWN_SLICED`, `STRATA_GR_DOWN_DIRECT` | on | `0`: older hyper-connection kernels | `fused_gr.dp.cpp:377`, `:583`, `:597` |
| `STRATA_HC_CHECK` | unset | run upstream's HC variant self-test | `fused_gr.dp.cpp:2188` |

Upstream variables the port's tooling sets: `STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_VERIFY_NO_HOST=1` and
`STRATA_STAGER_THREADS=12` (`sycl/serve/strata-sycl.sh`). SYCL runtime variables that matter:
`SYCL_CACHE_PERSISTENT=0` (the persistent JIT cache crashed on Xe2), `ONEAPI_DEVICE_SELECTOR`,
`SYCL_UR_USE_LEVEL_ZERO_V2=0` (the v1 adapter; used while diagnosing the stager hang),
`SYCL_PROGRAM_COMPILE_OPTIONS` (see [build.md](build.md)).

## Model-specific behaviour

- **Coder IQ1_M** (256 experts per layer): the reference model of the port; all 12,288 experts fit a 32 GB card
  with `--stream-experts`; the 256-expert router paths above exist for it.
- **Qwen3.8-Flash-Next IQ2_XS** and **Swift 1.5 IQ2_XS**: larger than VRAM on a 32 GB card; the remainder goes to the
  pinned mirror. Swift 1.5 exposed the per-role shard lookup bug in `GgufExpertSource`.
- **Q2_0 canonical pack**: `--stream-experts` refuses it (needs a native pack); the CUDA-style arena then applies.
  The s2 kernels are migrated and tested (`dequant_s2_parity`, `s2_gemv_parity`, ...). INTEL.md expects Q2_0 to
  behave like IQ2_XS but reports no run.
- IQ3_XXS / IQ3_S: INTEL.md estimates 19-26 GB mirrored on a 32 GB card, beyond its test machine's 23 GB of RAM;
  INTEL_ARC.md reports a community IQ3_XXS run on two B70s with `--layer-split`.

## Tests

The kernel changes are covered by the parity tests (`gr_parity`, `iq_multi_parity`, `qsa_parity`,
`qsa_prompt_attn_parity`, `kv_hybrid_parity`, ...) and by the port-only benches; the engine changes by comparing
greedy output tokens across runs, as INTEL.md describes for every step. See [build.md](build.md#tests).

## Gotchas

- `STRATA_VERIFY_NO_HOST=1` is only valid when every expert is resident (or mirrored); `strata-sycl.sh` always sets
  it. INTEL_ARC.md: on smaller cards this is the path that has hung, and upstream issue #667 suspected missing
  system fences for CPU flag stores.
- Host-to-device visibility of host-mapped memory during a running kernel is unreliable on this platform (INTEL.md);
  the all-resident path avoids it.
- The chunk boundary from `STRATA_PREFILL_FIRST` moves rounding in the expert GEMMs, so a long greedy continuation
  can diverge late (INTEL.md).
- INTEL.md refers to `sycl/probe/bw.cpp` and `sycl/probe/doorbell.cpp`, which are not in this snapshot.

## In UnbiasedStrata

Removed with `sycl/` in commit `8894a81`. None of these changes exist in the fork's CUDA engine.

## See also

- [layout.md](layout.md#inventory): which files carry these changes.
- [migration.md](migration.md): the scripted part of the changes (`fixups.py`).
- [serving.md](serving.md): how `strata-sycl.sh` and the wrappers set the switches above.
- [hardware.md](hardware.md): the measurements behind the defaults.
