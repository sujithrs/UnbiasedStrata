[Upstream docs](../README.md) › [Intel SYCL port](README.md) › Layout

# Layout of `sycl/` and the file inventory

## Purpose

Where everything in `sycl/` lives, how the SYCL build picks between a migrated copy and upstream's original, what the
vendored dpct headers are, and, for each of the 137 engine files under `sycl/src`, `sycl/include/strata` and
`sycl/third_party`, whether it is mechanical dpct output, dpct output changed by hand, a hand port, port-only, a
stub, or present but not built.

## The tree

```
sycl/
  CMakeLists.txt          the SYCL build (build.md)
  setup_intel.py          setup.py wrapper (serving.md)
  serve/                  server_intel.py, strata-sycl.sh, xe_telemetry.py, web/switcher.{js,css} (serving.md)
  tools/                  migrate.sh, normalize.sh, fixups.py, merge_upstream.py, build.sh, get-cuda-headers.sh,
                          Dockerfile, Dockerfile.unitrace (migration.md); gpustat.py, gpustat.service (serving.md)
  include/dpct/           Intel's dpct helper headers, vendored (46 files)
  include/strata/         migrated copies of upstream headers that touch CUDA, plus 7 port-only headers
  src/core/               migrated host code of the engine core, GgufExpertSource (port-only), peer_experts stub
  src/kernels/cuda/       the 44 migrated CUDA kernel files (X.cu -> X.dp.cpp) plus the XMX prompt attention
  src/kernels/            migrated parity tests, port-only benches
  src/prefill/            the migrated prompt path, the fused-kernel stub, two unbuilt MMQ files
  src/program/            generate.cpp, the `strata` binary's main (migrated, heavily hand-changed)
  src/ngram/, src/platform/   two non-CUDA upstream files copied and changed by hand
  third_party/ggml/       ggml-common.h in its migrated (SYCL) form
```

## Source resolution

`sycl/CMakeLists.txt` lists sources by their upstream names and resolves each one with `strata_resolve()`
(`sycl/CMakeLists.txt:56-75`):

| Upstream name | Resolved to |
|---|---|
| `src/X.cu` | `sycl/src/X.dp.cpp`; a missing copy is a `FATAL_ERROR` ("no migrated copy") |
| `src/X.cpp` (or any other) | `sycl/src/X.cpp` if it exists, else upstream's `src/X.cpp`; neither is a `FATAL_ERROR` |

Headers resolve the same way through the include order: `sycl/include` (the migrated headers and `dpct/`) comes
before upstream's `include/` (`sycl/CMakeLists.txt:77-81`). So a header the port carries shadows upstream's, and
every header it does not carry (for example `include/strata/kernels/qsa.hpp`) is upstream's. Files that dpct left
byte-identical are deleted by `normalize.sh` so this fallback picks the original. The CPU-side libraries
(`src/spec/*`, `src/kernels/cpu/*`), `src/platform/memory.cpp`, `src/core/layout.cpp`, `src/kernels/ngram.cpp` and
the speculative-decoding code always come from upstream, unchanged.

Upstream `.cu` files with no copy in `sycl/` and not built by the port: `src/core/remote_expert_opt.cu` (and its
`--remote-expert-opt` flag), `src/prefill/moe_fused.cu` and `moe_fused_iq.cu` (replaced by
`moe_fused_stub.cpp`), and the CUDA-only tests `src/prefill/gdn_rec_parity.cu` and `gemm_bf16_parity.cu`. Several
newer upstream tests in `src/kernels/` (for example `decode_cluster_parity.cpp`, `mmvq_multi_parity.cpp`,
`ple_fp8_parity.cpp`, `qsa_topk_parity.cpp`, `route_window_parity.cpp`) have no SYCL copy and no SYCL target.

## The vendored dpct helper headers

`sycl/include/dpct/` (46 files, 37,651 lines) is the helper library that `dpct --gen-helper-function` copies next
to its output, so the port builds without SYCLomatic installed. It is Intel's code under Apache-2.0 with the LLVM
exception (`include/dpct/LICENSE.TXT`). Only one edit is made to it: fixup rule 1 renames two non-uniform-group
functions in `util.hpp` for the 2026.1 compiler ([migration.md](migration.md#fixupspy-the-scripted-corrections)).

| Part | Files | What it provides |
|---|---|---|
| `dpct.hpp` | 1 | The umbrella header every migrated file includes (122 port files do). Pulls in `atomic.hpp`, `device.hpp`, `image.hpp`, `kernel.hpp`, `math.hpp`, `memory.hpp`, `util.hpp`, `bindless_images.hpp`, `graph.hpp`, `ze_utils.hpp`. Defines `__dpct_inline__` and `DPCT_COMPATIBILITY_TEMP` as 900 (`dpct.hpp:78`), the stand-in for `__CUDA_ARCH__` that fixup rule 11 works around |
| Runtime core | `device.hpp`, `memory.hpp`, `util.hpp`, `kernel.hpp`, `atomic.hpp`, `math.hpp`, `graph.hpp`, `group_utils.hpp`, `ze_utils.hpp`, `compat_service.hpp`, `shmem_utils.hpp`, `detail/memory_detail.hpp`, `detail/math_detail.hpp`, `detail/group_utils_detail.hpp` | `dpct::get_in_order_queue()`, `dpct::get_current_device()` (with `queues_wait_and_throw()`, `get_memory_info()`), `dpct::queue_ptr`, `dpct::global_memory` / `constant_memory`, `dpct::dp4a` and other CUDA intrinsics, graph wrappers, Level Zero interop |
| Library wrappers | `blas_utils.hpp`, `blas_gemm_utils.hpp`, `lapack_utils.hpp`, `fft_utils.hpp`, `sparse_utils.hpp`, `rng_utils.hpp`, `dnnl_utils.hpp`, `ccl_utils.hpp`, `lib_common_utils.hpp` and their `detail/` files | cuBLAS/cuSOLVER/cuFFT/... equivalents over oneMKL, oneDNN, oneCCL. The port uses only `dpct::blas` (`blas_utils.hpp`, included by `src/prefill/gemm.dp.cpp` and `src/kernels/xmx_gemm_bench.cpp`) |
| `image.hpp`, `bindless_images.hpp` | 2 | Texture equivalents; not used by Strata's kernels as far as a search of `sycl/src` shows |
| `dpl_extras/`, `dpl_utils.hpp` | 9 | Thrust-like algorithms over oneDPL |
| `codepin/` | 2 | dpct's CodePin debugging instrumentation |

## Inventory

Kinds:

| Kind | Meaning | Count |
|---|---|---|
| dpct | dpct output plus the general `fixups.py` rules (queue casts, waits, intrinsics); otherwise mechanical | 76 |
| dpct + fixups | as above plus a file-specific rule in `fixups.py` | 12 |
| dpct + hand | dpct output with hand-written changes marked `SYCL port` | 15 |
| dpct, hand-merged | dpct output whose upstream changes are merged by hand (`merge_upstream.py` `HAND`, INTEL.md) | 3 |
| hand | ported by hand (dpct never saw the file) | 2 |
| copy + hand | a non-CUDA upstream file (or header) copied into `sycl/` and changed | 4 |
| copy | unchanged copy | 1 |
| port-only | exists only in the port | 15 |
| stub | replaces an upstream CUDA-only feature with "not available" answers | 2 |
| dpct, not built | migrated but referenced by no CMake target | 7 |

"(ctest)" marks a test registered with `add_test`; see [build.md](build.md#tests). Line references in the Role
column are into the file of that row.

### `sycl/src/core`

| File | Lines | Kind | Role |
|---|---|---|---|
| [sycl/src/core/device.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/device.dp.cpp) | 454 | dpct | `device.cu`: device selection, memory and stream helpers of the runtime core |
| [sycl/src/core/pinned.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/pinned.dp.cpp) | 794 | dpct | `pinned.cu`: the pinned host arena and the parallel expert load |
| [sycl/src/core/graph.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/graph.cpp) | 249 | dpct | GraphRegistry (CUDA graphs become `sycl_ext_oneapi_graph` command graphs) |
| [sycl/src/core/weights.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/weights.cpp) | 527 | dpct | dense-weight loader |
| [sycl/src/core/session.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/session.cpp) | 1436 | dpct | one token through all 48 layers; the doorbell protocol |
| [sycl/src/core/layer.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/layer.cpp) | 1536 | dpct + hand | the GDN layer; native router also for 256 experts (`layer.cpp:412`) |
| [sycl/src/core/expert_cache.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/expert_cache.cpp) | 901 | dpct + hand | VRAM expert slots; `--vram-elastic` / segmented arena compiled out (`#if 0/1 // SYCL port`, #533) |
| [sycl/src/core/expert_source.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/expert_source.cpp) | 3088 | dpct | the expert-source adapter (arena, mmap, CPU pool) |
| [sycl/src/core/gguf_expert_source.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/gguf_expert_source.cpp) | 205 | port-only | `GgufExpertSource`: experts read from the GGUF on demand (`--stream-experts`), pinned host mirror |
| [sycl/src/core/remote_experts.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/remote_experts.cpp) | 477 | dpct, hand-merged | expert tier on another device; `cudaInitDevice` flags dropped (fixup 15) |
| [sycl/src/core/peer_experts.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/peer_experts.cpp) | 41 | stub | `--peer-device` answers "not ported to SYCL yet" |
| [sycl/src/core/native_head.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/native_head.cpp) | 286 | dpct | native embedding and output head |
| [sycl/src/core/native_dense.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/native_dense.cpp) | 257 | dpct | native dense projections |
| [sycl/src/core/verify.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/verify.cpp) | 1837 | dpct, hand-merged | the speculative verify window; eager mode, NO_HOST wait, deferred commit, debug ladder |
| [sycl/src/core/mtp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/mtp.cpp) | 1167 | dpct, hand-merged | the MTP draft layer on the GPU; graphs warmed at load |
| [sycl/src/core/conversation_snapshot.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/conversation_snapshot.cpp) | 293 | dpct | conversation snapshots (#216) |
| [sycl/src/core/conversation_state.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/conversation_state.cpp) | 316 | hand | two CUDA calls (device sync, copy) rewritten as SYCL queue operations |
| [sycl/src/core/conversation_snapshot_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/conversation_snapshot_test.cpp) | 337 | dpct | snapshot test against the engine (`ctest`, skip code 77) |
| [sycl/src/core/concurrent_main.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/concurrent_main.cpp) | 272 | dpct, not built | `strata-concurrent` diagnostic (CPU expert path vs GPU stream overlap); no CMake target |
| [sycl/src/core/overlap_main.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/core/overlap_main.cpp) | 239 | dpct, not built | `strata-overlap` diagnostic; no CMake target |

### `sycl/src/kernels/cuda` (kernels)

| File | Lines | Kind | Role |
|---|---|---|---|
| [sycl/src/kernels/cuda/bf16_gemv.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/bf16_gemv.dp.cpp) | 251 | dpct | BF16 GEMV |
| [sycl/src/kernels/cuda/cvec.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/cvec.dp.cpp) | 302 | dpct | control-vector kernel |
| [sycl/src/kernels/cuda/dequant_bf16.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/dequant_bf16.dp.cpp) | 318 | dpct | device dequantizers to BF16 |
| [sycl/src/kernels/cuda/dequant_s2.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/dequant_s2.dp.cpp) | 72 | dpct | S2 (Q2_0 pack) decode |
| [sycl/src/kernels/cuda/elementwise.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/elementwise.dp.cpp) | 765 | dpct + fixups | layer glue kernels; doorbell ring/wait through `sys_load`/`sys_store` (`:377-383`, `:599`) |
| [sycl/src/kernels/cuda/fused_gdn.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/fused_gdn.dp.cpp) | 341 | dpct | fused gated-delta-net step |
| [sycl/src/kernels/cuda/fused_gr.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/fused_gr.dp.cpp) | 2223 | dpct + hand | fused hyper-connection read; port's split norm, direct and column-sliced down kernels |
| [sycl/src/kernels/cuda/gdn.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/gdn.dp.cpp) | 401 | dpct | gated delta-net non-projection parts |
| [sycl/src/kernels/cuda/gr.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/gr.dp.cpp) | 735 | dpct | gated residual / hyper-connection read and write |
| [sycl/src/kernels/cuda/iq_kernels.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/iq_kernels.dp.cpp) | 2752 | dpct + hand | i-quant expert kernels; SYCL ggml tables, lane-count kernels, wide dequant stores, XMX GEMM (`:2597`) |
| [sycl/src/kernels/cuda/kv_q4.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/kv_q4.dp.cpp) | 353 | dpct | Q4_0 KV with Walsh-Hadamard rotation |
| [sycl/src/kernels/cuda/kv_q8.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/kv_q8.dp.cpp) | 204 | dpct | INT8 KV append/gather |
| [sycl/src/kernels/cuda/kv_stream.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/kv_stream.dp.cpp) | 490 | dpct | KV streaming (`--kv-resident`) window |
| [sycl/src/kernels/cuda/native_bf16.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/native_bf16.dp.cpp) | 345 | dpct | native BF16 matvec (from llama.cpp MMVF) |
| [sycl/src/kernels/cuda/native_flash_attn.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/native_flash_attn.dp.cpp) | 382 | dpct | native flash attention (from llama.cpp) |
| [sycl/src/kernels/cuda/native_gdn.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/native_gdn.dp.cpp) | 167 | dpct | native gated delta-net (from llama.cpp) |
| [sycl/src/kernels/cuda/native_gdn_preprocess.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/native_gdn_preprocess.dp.cpp) | 358 | dpct | native GDN preprocessing |
| [sycl/src/kernels/cuda/native_gr_norm.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/native_gr_norm.dp.cpp) | 165 | dpct | native hyper-connection norm |
| [sycl/src/kernels/cuda/native_gr_postops.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/native_gr_postops.dp.cpp) | 247 | dpct | native hyper-connection post-ops |
| [sycl/src/kernels/cuda/native_mmvq.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/native_mmvq.dp.cpp) | 2700 | dpct + hand | native GGUF matvec; SIMD16 option, row-blocked and wide 16-byte-load kernels, aligned Q6_K loads |
| [sycl/src/kernels/cuda/native_moe.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/native_moe.dp.cpp) | 156 | dpct | native MoE helpers |
| [sycl/src/kernels/cuda/native_ple_postops.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/native_ple_postops.dp.cpp) | 516 | dpct | native PLE post-ops |
| [sycl/src/kernels/cuda/native_qsa.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/native_qsa.dp.cpp) | 209 | dpct | native QSA attention pieces |
| [sycl/src/kernels/cuda/native_qsa_indexer.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/native_qsa_indexer.dp.cpp) | 547 | dpct + fixups | native QSA indexer (the `__fadd_rn` parenthesis bug fixed by fixup 3) |
| [sycl/src/kernels/cuda/native_qsa_score.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/native_qsa_score.dp.cpp) | 329 | dpct + fixups | native QSA block scores; PTX path compiled out (`:39`), scalar path on (`:95`) |
| [sycl/src/kernels/cuda/native_rope.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/native_rope.dp.cpp) | 213 | dpct | native RoPE (ggml rope_multi / rope_yarn contract) |
| [sycl/src/kernels/cuda/native_router.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/native_router.dp.cpp) | 244 | dpct + hand | native top-10 router; templated on 512 or 256 experts (`:70`) |
| [sycl/src/kernels/cuda/ple.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/ple.dp.cpp) | 676 | dpct | PLE block's GPU half |
| [sycl/src/kernels/cuda/qsa.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/qsa.dp.cpp) | 1277 | dpct | QSA cache, indexer and attention |
| [sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp) | 518 | dpct | QSA decode attention |
| [sycl/src/kernels/cuda/qsa_prompt_attn.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/qsa_prompt_attn.dp.cpp) | 1620 | dpct + fixups | prompt attention; tensor-core path off, XMX dispatch at `:1578` |
| [sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp) | 385 | port-only | prompt attention on XMX `joint_matrix` (opt-in `STRATA_PROMPT_ATTN_XMX`) |
| [sycl/src/kernels/cuda/qsa_select.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/qsa_select.dp.cpp) | 1772 | dpct + fixups | QSA block selection; tensor-core and cluster paths off (`:1410`, `:1565`) |
| [sycl/src/kernels/cuda/quantize_act.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/quantize_act.dp.cpp) | 424 | dpct | Q8 activation quantizer (byte-exact vs ggml) |
| [sycl/src/kernels/cuda/rope.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/rope.dp.cpp) | 165 | dpct | NEOX partial RoPE |
| [sycl/src/kernels/cuda/router_top10.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/router_top10.dp.cpp) | 569 | dpct | MoE router |
| [sycl/src/kernels/cuda/s2_expert_grouped.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/s2_expert_grouped.dp.cpp) | 1801 | dpct + hand | grouped GPU expert (rounding parenthesis fixed by hand, `:650`) |
| [sycl/src/kernels/cuda/s2_gemv.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/s2_gemv.dp.cpp) | 85 | dpct | S2 GEMV |
| [sycl/src/kernels/cuda/s2_gemv_fast.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/s2_gemv_fast.dp.cpp) | 250 | dpct + fixups | S2 GEMV experiments; constant table heap-allocated (fixup 7f) |
| [sycl/src/kernels/cuda/s2_gemv_q8.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/s2_gemv_q8.dp.cpp) | 141 | dpct | S2 GEMV on Q8_0 activations |
| [sycl/src/kernels/cuda/s2_gemv_quads.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/s2_gemv_quads.dp.cpp) | 138 | dpct | S2 GEMV, four elements per code load |
| [sycl/src/kernels/cuda/s_gemv.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/s_gemv.dp.cpp) | 1015 | dpct + fixups | S-family GEMV (`__ldg` cast restored, fixup 7b) |
| [sycl/src/kernels/cuda/sampler.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/sampler.dp.cpp) | 1772 | dpct + fixups | sampler chain; sm_90 cluster greedy kernel off (`:1392`) |
| [sycl/src/kernels/cuda/shared_expert.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/shared_expert.dp.cpp) | 596 | dpct | shared expert |
| [sycl/src/kernels/cuda/verify_kernels.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cuda/verify_kernels.dp.cpp) | 1302 | dpct + hand | verify-window kernels; doorbell waits, parallel `resident_plan`, host-mirror table |
| [sycl/src/kernels/iq2xs_host_tables.inc](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/iq2xs_host_tables.inc) | 138 | port-only | IQ2_XS tables as host arrays for `iq_multi_parity` |

### `sycl/src/kernels` (tests and benches)

| File | Lines | Kind | Role |
|---|---|---|---|
| [sycl/src/kernels/bf16_gemv_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/bf16_gemv_parity.cpp) | 191 | dpct | parity test (ctest) |
| [sycl/src/kernels/cvec_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/cvec_parity.cpp) | 242 | dpct | parity test (ctest) |
| [sycl/src/kernels/dequant_s2_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/dequant_s2_parity.cpp) | 165 | dpct | parity test (ctest) |
| [sycl/src/kernels/elementwise_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/elementwise_parity.cpp) | 717 | dpct | parity test (ctest) |
| [sycl/src/kernels/gdn_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/gdn_parity.cpp) | 594 | dpct | parity test (ctest) |
| [sycl/src/kernels/gr_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/gr_parity.cpp) | 1202 | dpct | parity test (ctest) |
| [sycl/src/kernels/iq_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/iq_parity.cpp) | 153 | dpct | parity test (ctest); needs oracle fixtures |
| [sycl/src/kernels/iq_multi_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/iq_multi_parity.cpp) | 607 | dpct + hand | multi-entry IQ kernels bitwise vs per-column ones (ctest) |
| [sycl/src/kernels/kv_q4_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/kv_q4_parity.cpp) | 429 | dpct | parity test (ctest) |
| [sycl/src/kernels/kv_q8_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/kv_q8_parity.cpp) | 230 | dpct | parity test (ctest) |
| [sycl/src/kernels/kv_stream_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/kv_stream_parity.cpp) | 344 | dpct | parity test (ctest) |
| [sycl/src/kernels/kv_hybrid_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/kv_hybrid_parity.cpp) | 482 | dpct + hand | hybrid K8V4 KV; built, not registered; turns on the XMX prompt attention |
| [sycl/src/kernels/native_expert_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/native_expert_parity.cpp) | 300 | hand | one native expert three ways on real GGUF rows (needs a shard) |
| [sycl/src/kernels/native_grouped_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/native_grouped_parity.cpp) | 458 | dpct | `native_expert_grouped` group stride and fused SwiGLU (ctest) |
| [sycl/src/kernels/ple_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/ple_parity.cpp) | 1569 | dpct | parity test (ctest); needs a Q2_0 shard |
| [sycl/src/kernels/qsa_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/qsa_parity.cpp) | 1684 | dpct + hand | parity test (ctest) |
| [sycl/src/kernels/qsa_prompt_attn_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/qsa_prompt_attn_parity.cpp) | 318 | dpct + fixups | prompt attention; built, not registered; sets `STRATA_PROMPT_ATTN_XMX=1` |
| [sycl/src/kernels/quantize_act_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/quantize_act_parity.cpp) | 450 | dpct | parity test (ctest) |
| [sycl/src/kernels/rope_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/rope_parity.cpp) | 614 | dpct | parity test (ctest) |
| [sycl/src/kernels/router_top10_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/router_top10_parity.cpp) | 186 | dpct | parity test (ctest) |
| [sycl/src/kernels/s2_expert_grouped_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/s2_expert_grouped_parity.cpp) | 790 | dpct | rewritten expert kernels vs previous ones (ctest) |
| [sycl/src/kernels/s2_gemv_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/s2_gemv_parity.cpp) | 205 | dpct | parity test (ctest) |
| [sycl/src/kernels/s2_gemv_q8_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/s2_gemv_q8_parity.cpp) | 236 | dpct | parity test (ctest) |
| [sycl/src/kernels/s_gemv_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/s_gemv_parity.cpp) | 754 | dpct | parity test (ctest) |
| [sycl/src/kernels/s_gemv_q8k_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/s_gemv_q8k_parity.cpp) | 500 | dpct | parity test (ctest) |
| [sycl/src/kernels/sampler_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/sampler_parity.cpp) | 1453 | dpct | parity test (ctest) |
| [sycl/src/kernels/shared_expert_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/shared_expert_parity.cpp) | 739 | dpct | parity test (ctest) |
| [sycl/src/kernels/dequant_bf16_test.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/dequant_bf16_test.cpp) | 116 | dpct, not built | dequantizer test; no CMake target |
| [sycl/src/kernels/q5_projection_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/q5_projection_parity.cpp) | 86 | dpct, not built | Q5_1 shared-expert projection test; no CMake target |
| [sycl/src/kernels/qsa_select_bench.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/qsa_select_bench.cpp) | 234 | dpct, not built | QSA selection timing; no CMake target |
| [sycl/src/kernels/mmvq_bench.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/mmvq_bench.cpp) | 51 | port-only | dense `native_mmvq` timing |
| [sycl/src/kernels/mmvq_sg_bench.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/mmvq_sg_bench.cpp) | 56 | port-only | SIMD16 vs SIMD32 decode mmvq timing |
| [sycl/src/kernels/q6k_align_bench.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/q6k_align_bench.cpp) | 59 | port-only | Q6_K 210-byte vs 224-byte block stride timing |
| [sycl/src/kernels/gr_bench.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/gr_bench.cpp) | 58 | port-only | `fused_gr_read_multi` timing for 1-6 tokens |
| [sycl/src/kernels/xmx_gemm_bench.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/kernels/xmx_gemm_bench.cpp) | 93 | port-only | XMX quantized GEMM vs dequant + oneMKL |

### `sycl/src/prefill`

| File | Lines | Kind | Role |
|---|---|---|---|
| [sycl/src/prefill/gemm.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/prefill/gemm.dp.cpp) | 506 | dpct | prompt GEMMs: cuBLAS calls become `dpct::blas::gemm` (oneMKL) |
| [sycl/src/prefill/kernels.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/prefill/kernels.dp.cpp) | 2082 | dpct + fixups | prompt-path kernels; `cp.async` replaced by plain copies (`:778`) |
| [sycl/src/prefill/prefill.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/prefill/prefill.cpp) | 3657 | dpct + hand | the prompt path; polled-sequence stager, stream-all gate, short first chunk, GGUF stager reads |
| [sycl/src/prefill/moe_fused_stub.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/prefill/moe_fused_stub.cpp) | 37 | stub | upstream's fused int8 prompt kernels (#136) answer "not built" |
| [sycl/src/prefill/moe_mmq.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/prefill/moe_mmq.dp.cpp) | 311 | dpct, not built | ggml MMQ prompt path (needs llama.cpp's ggml-cuda sources) |
| [sycl/src/prefill/ggml_cuda_host.dp.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/prefill/ggml_cuda_host.dp.cpp) | 104 | dpct, not built | host symbols for the MMQ path |

### `sycl/src/program`

| File | Lines | Kind | Role |
|---|---|---|---|
| [sycl/src/program/generate.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/program/generate.cpp) | 8380 | dpct + hand | `strata generate` / serve mode; `--stream-experts`, borrow default, mirror, logprobs, warm graphs, `_Exit` |

### `sycl/src/ngram`, `sycl/src/platform`

| File | Lines | Kind | Role |
|---|---|---|---|
| [sycl/src/ngram/ple_reader.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/ngram/ple_reader.cpp) | 570 | copy + hand | PLE row reader plus a per-gather trace (`STRATA_PLE_TRACE`) |
| [sycl/src/platform/direct_file.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/src/platform/direct_file.cpp) | 369 | copy + hand | O_DIRECT reader with 64 default I/O threads instead of 16 (`:314-317`) |

### `sycl/include/strata`

| File | Lines | Kind | Role |
|---|---|---|---|
| [sycl/include/strata/sycl_queue.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/sycl_queue.hpp) | 14 | port-only | `strata::q_of(stream)`: null stream means the default in-order queue |
| [sycl/include/strata/sycl_doorbell.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/sycl_doorbell.hpp) | 29 | port-only | `sys_load` / `sys_store` (system-scope atomics), `kSpinMax` |
| [sycl/include/strata/sycl_math.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/sycl_math.hpp) | 15 | port-only | `strata::dp4a`, the one place the DP4A form is chosen |
| [sycl/include/strata/core/gguf_expert_source.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/core/gguf_expert_source.hpp) | 78 | port-only | `GgufExpertSource` interface |
| [sycl/include/strata/kernels/xmx_gemm.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/kernels/xmx_gemm.hpp) | 11 | port-only | `xmx_gemm_iq` declaration |
| [sycl/include/strata/kernels/qsa_prompt_attn_xmx.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/kernels/qsa_prompt_attn_xmx.hpp) | 11 | port-only | `qsa_prompt_attn_xmx` declaration |
| [sycl/include/strata/kernels/resident_plan_mirror.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/kernels/resident_plan_mirror.hpp) | 10 | port-only | `resident_plan_set_mirror` declaration |
| [sycl/include/strata/kernels/native_router.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/kernels/native_router.hpp) | 21 | copy + hand | adds `native_router_top10_multi_ne` (256 or 512 experts) |
| [sycl/include/strata/kernels/native_mmvq.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/kernels/native_mmvq.hpp) | 115 | copy + hand | adds the Q6_K wide-kernel switch and the 224-byte-stride experiment |
| [sycl/include/strata/kernels/dp4a.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/kernels/dp4a.hpp) | 51 | dpct + fixups | no `__nanosleep` in the spin pause (fixup at `fixups.py:100`) |
| [sycl/include/strata/kernels/bf16_bits.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/kernels/bf16_bits.hpp) | 64 | dpct | bf16 conversions |
| [sycl/include/strata/kernels/f16_bits.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/kernels/f16_bits.hpp) | 107 | dpct | fp16 conversions |
| [sycl/include/strata/kernels/mrope.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/kernels/mrope.hpp) | 80 | dpct | multimodal rotary positions |
| [sycl/include/strata/kernels/q8_1_finite.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/kernels/q8_1_finite.hpp) | 41 | dpct | q8_1 scale and sum kept finite (#606) |
| [sycl/include/strata/kernels/rope.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/kernels/rope.hpp) | 67 | dpct | NEOX partial RoPE API |
| [sycl/include/strata/kernels/rope_scaling.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/kernels/rope_scaling.hpp) | 196 | dpct | RoPE scaling (linear, YaRN) |
| [sycl/include/strata/kernels/sampler.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/kernels/sampler.hpp) | 87 | dpct | sampler chain API |
| [sycl/include/strata/core/coupled_draft.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/core/coupled_draft.hpp) | 93 | dpct | coupled draft sampling |
| [sycl/include/strata/core/graph.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/core/graph.hpp) | 118 | dpct + fixups | GraphRegistry API (`mutable` timestamp, fixup 16) |
| [sycl/include/strata/core/mtp.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/core/mtp.hpp) | 203 | dpct + hand | MTP API plus `warm()` |
| [sycl/include/strata/core/on_device.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/core/on_device.hpp) | 38 | dpct | scoped current device |
| [sycl/include/strata/core/peer_experts.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/core/peer_experts.hpp) | 108 | dpct | `--peer-device` API (implemented by the stub) |
| [sycl/include/strata/core/remote_experts.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/core/remote_experts.hpp) | 85 | dpct | remote expert tier API |
| [sycl/include/strata/core/session.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/core/session.hpp) | 441 | dpct | Session API |
| [sycl/include/strata/core/verify.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/include/strata/core/verify.hpp) | 293 | dpct + hand | Verifier API plus `commit(wait=false)`, `warm()`, debug buffers |

### `sycl/third_party/ggml`

| File | Lines | Kind | Role |
|---|---|---|---|
| [sycl/third_party/ggml/ggml-common.h](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/third_party/ggml/ggml-common.h) | 2193 | dpct | ggml block structs and grids, SYCL declaration mode (`sycl::half`) |
| [sycl/third_party/ggml/LICENSE](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/sycl/third_party/ggml/LICENSE) | 21 | copy | ggml MIT license |

## In UnbiasedStrata

Removed: every file listed here was deleted in commit `8894a81`. The fork's engine (`src/`, `include/`) descends
from the same upstream sources but has no SYCL counterpart.

## See also

- [migration.md](migration.md): how the dpct files were produced.
- [differences.md](differences.md): what the "dpct + hand", "hand" and "port-only" changes do.
- [build.md](build.md): which of these files each CMake target compiles.
