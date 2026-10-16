[Upstream docs](../README.md) › [Prefill](README.md) › MoE half

# The MoE half of a prompt layer

## Purpose

Half 1 of every layer: the router picks 10 of 512 experts per token, the shared expert runs on every token, the
routed experts are grouped by expert and multiplied, and the results are combined into `bo`, which the
hyper-connection write adds to the residual. This page covers the router, the shared expert, the four routed-expert
paths and how one is chosen per layer.

Files: [src/prefill/prefill.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/prefill.cpp)
(`:2235-2887`),
[src/prefill/moe_fused.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/moe_fused.cu),
[src/prefill/moe_fused_iq.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/moe_fused_iq.cu),
[include/strata/prefill/moe_fused.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/prefill/moe_fused.hpp),
[include/strata/prefill/moe_fused_iq.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/prefill/moe_fused_iq.hpp),
`route` / `swiglu_*` / `moe_combine` in [src/prefill/kernels.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/kernels.cu).
The MMQ wrapper itself is on [ggml-bridge.md](ggml-bridge.md).

## Router and shared expert

| Step | Where | What |
| --- | --- | --- |
| Router logits | `src/prefill/prefill.cpp:2244` | BF16 GEMM of `ffn_gate_inp.weight` on `mixed_bf` -> `logits [T, 512]` |
| Top-k | `:2245`, `src/prefill/kernels.cu:617-655`, `:928-935` | `route`: one warp per token, softmax over 512 (`route_kernel<16>`; 256 experts `<8>`; else `router_top10`), top 10 with ties to the lower id, weights renormalized over the ten (sum clamped at 2^-14) |
| Shared expert | `:2247-2250` | `ffn_gate_shexp` and `ffn_up_shexp` (FP16 GEMMs of the native blocks), `swiglu_pair` (saturated FP16), `ffn_down_shexp` -> `shared` |
| Shared gate | `:2251-2253` | `ffn_gate_inp_shexp.weight` must be BF16; a `[T, 1]` BF16 GEMM -> `sg` |
| Combine | `:2857-2863`, `src/prefill/kernels.cu:984-988` | `bo[t] = shared[t] * sigmoid(sg[t]) + sum_k w[t,k] * Dm[slot[t,k]]`; with a peer the primary first waits for the peer's rows |

## The four routed-expert paths

| Path | Chosen when (per layer) | Grouping | Product |
| --- | --- | --- | --- |
| Fused Q2_0 int8 | Strata Q2_0 pack, MMQ planned for the layer, stream-all walk (chunk >= 1,024 tokens), no `--peer-device`, `fused::enabled()` (sm_80+, `STRATA_PF_FUSED` not 0) | on the GPU (`fused::group`), no host sync | `fused::experts`: one launch per batch of up to 128 experts, gate/up + SwiGLU + int8 H, then down |
| Fused native IQ int8 | native pack, same conditions, and `fused::native_supported(gu_type, d_type)`, which needs `STRATA_PF_FUSED=1` (opt-in) | on the GPU | `fused::experts_native` |
| llama.cpp MMQ | `mmq_plan().layer[l]` (both formats have an MMQ tile on every GPU) and not fused | on the host after one sync | per 16-expert group: gather, q8_1 activations, gate/up, SwiGLU, q8_1 of H, down |
| FP16 GEMM fallback | no MMQ for the layer (a build without MMQ, `STRATA_PREFILL_MMQ=0`, IQ1_M, a type `fits` refuses) | on the host | per expert: dequantize to FP16 (`blob_dequant_f16`, or `iq_dequant_gu_f16` / `iq_dequant_f16` for a native layer), two cuBLAS FP16 GEMMs |

The choice is made at `src/prefill/prefill.cpp:2261-2269`. `mmq_plan()` (`:669-690`) is computed once: MMQ is on
when the build has it and `STRATA_PREFILL_MMQ` is not 0; a layer is MMQ when `mmq::fits(gu_type, 1280)` and
`mmq::fits(d_type, 2560)`; the Strata Q2_0 pack counts as type 42 (Q2_0) for both, its blob converted to GGUF Q2_0
blocks on the gather. `fallback` records that some layer keeps the FP16 path, which makes `carve` allocate its FP16
buffers (`Xs`, `Hh`).

Pack by pack (from the code paths above and the header comments):

| Pack | Default prompt experts |
| --- | --- |
| Q2_0 (Strata pack) | fused int8 on sm_80+ for chunks of 1,024+ tokens; MMQ for a smaller last chunk, below sm_80, on HIP, or with `STRATA_PF_FUSED=0` |
| IQ2_XS, IQ3_XXS, IQ3_S (native) | MMQ; the fused IQ kernels with `STRATA_PF_FUSED=1` |
| Coder (IQ1_M layers) | IQ1_M layers take the FP16 path (MMQ and the fused kernels do not cover IQ1_M); its other layers as above |
| UD-Q4_K_XL (Q4_K / Q5_K gate-up, Q5_1 / Q8_0 down) | MMQ only in a CUDA build with `STRATA_MMQ_KQUANTS`; otherwise FP16 |

## The MMQ path in detail

1. **Host grouping** (`:2330-2402`): copy the router's `ids` to the mapped table (or the host), synchronize the
   compute stream (the only host sync per MoE layer), count rows per expert, assign offsets (local experts first,
   then the peer's), write `slot` (pair -> row) and `src` (row -> token) back through the mapped table.
2. **Activations** (`:2416`): `mmq::quantize(mixed, src_dev, Xq, ...)` writes the layer's q8_1 rows in expert
   order straight from `mixed`.
3. **Bounds** (`:2419-2434`): absolute row bounds per expert (`n + 1` entries), then per 16-expert group its relative
   bounds (17 entries) for the down product, which reads only the group's own H rows.
4. **Per expert** in id order (`compute`, `:2731-2797`): gather the expert into its group slot
   (`gather_native` for a native blob, `gather_strata_q2` for the Strata pack); with group gathers the whole group
   in one launch. When the group is complete: zero the 4,096-byte tail after the last expert (`MMQ_TAIL`: MMQ reads
   up to one tile past a matrix whose rows are not a multiple of 256, and the zero bytes must decode to finite
   values), gate/up `Product` (1,280 x 2,560 per expert) into `GU`, `mmq::swiglu` (interleaved rows for the Strata
   pack, split for GGUF) into `H`, `quantize` of the group's H rows, down `Product` (2,560 x 640) into `Dm`.
5. Experts arrive either from their cache slot or from the ring (see
   [expert-streaming.md](expert-streaming.md)).

## The fused int8 kernels (`moe_fused.cu`, #136)

The header (`include/strata/prefill/moe_fused.hpp:4-15`) states the difference from MMQ: activations are rounded to
int8 once per token (not per routed slot) with a scale per 32 values; grouping (counts, offsets, rows per expert, a
list of 64-row tiles) runs on the GPU from the router's ids; one launch per batch of up to `kMaxBatch` (128) experts
reads each expert where it already is (a pointer per expert, no gather), runs gate/up with SwiGLU and the int8
rounding of H in its epilogue, then down into the per-slot rows the combine reads.

- Arithmetic (`src/prefill/moe_fused.cu:4-19`): a Q2_0 weight is `d_w * (q - 1)`, so the offset is folded into the
  activation block's stored sum; the int32 dot becomes a float exactly via the `0x4B400000` (1.5 x 2^23) trick; codes
  are unpacked in registers and multiplied with `mma.sync m16n8k32` (s8 x s8 -> s32). Activations are stored in a
  permuted K order (`perm32`) so a lane's two B registers are one 8-byte load.
- Shape (`:41-56`): 512 threads (16 warps), work items of up to 64 routed rows x 256 weight rows, a 4-stage
  `cp.async` pipeline; the grid is persistent over the batch's items.
- `group` (`:431-447`): count (shared-memory histogram), scan (one block, one thread per expert, so at most 1,024
  experts), place (atomics: the order of rows within an expert is not fixed; no output depends on it).
- In the layer loop (`src/prefill/prefill.cpp:2272-2329`): `quantize_act` (or `quantize_act_native`) into `Xq`,
  `group` into `GU` / `slot_dev` / `src_dev`, then batches over the experts in id order with at most a third of the
  ring streamed per batch (so the next batch's blobs arrive while one computes); one wait on the batch's last copy,
  one `used` record per ring entry afterwards. The first use prints "prompt experts on the fused int8 kernels".
- `dev_info` (`src/prefill/moe_fused.cu:360-395`) makes the kernels available only on sm_80+, when the build's device
  code has them (PTX version >= 80) and their shared-memory occupancy fits; a gfx906 build never reports them.
- `enabled()` / `requested()` (`:404-417`): `enabled()` is on unless `STRATA_PF_FUSED=0` ("0.1.36: on by default for
  the Q2_0 pack"); `requested()` needs `STRATA_PF_FUSED=1` and gates the native kernels.

## The native IQ fused kernels (`moe_fused_iq.cu`)

Same grouping, batch shape and epilogues as the Q2_0 kernels; the load stage differs
(`src/prefill/moe_fused_iq.cu:1-20`): per 64 values of K a thread decodes one 32-value sub-block of i-quant blocks to
int8 in shared memory (codebooks in shared memory, up to 8 KB for IQ2_S), with the block's scale per 32 values or per
16 (IQ2_XS, IQ2_S, multiplied at m16n8k16). Activations and H are int8 per 32 values in natural order
(`quantize_act_native`). Covered: gate/up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS; down Q2_0, IQ4_NL
(`gu_covered` / `d_covered`, `:562-565`). `pick_ww` (`:576-591`) uses 128-row work items when an expert averages 56 to
112 routed rows, else 64 (`STRATA_PF_FUSED_TILE=64|128` forces one). `STRATA_PF_FUSED_NATIVE=0` keeps MMQ under
`STRATA_PF_FUSED=1`.

## Configuration

| Knob | Effect |
| --- | --- |
| `STRATA_PF_FUSED=0` | Q2_0 keeps MMQ (byte-identical answers to 0.1.35, per `docs/DETAILS.md`) |
| `STRATA_PF_FUSED=1` | also the native packs' fused kernels |
| `STRATA_PF_FUSED_NATIVE=0`, `STRATA_PF_FUSED_TILE=64|128` | native fused kernels off / tile shape |
| `STRATA_PREFILL_MMQ=0` | FP16 path everywhere (A/B) |
| `STRATA_PREFILL_GROUP_GATHER=0` | one gather per expert |
| `STRATA_DBG_NAN=1` | report the first layer of a chunk whose MoE produced non-finite values |
| CMake `STRATA_MMQ_KQUANTS`, `STRATA_ORCA_Q4KS_MMQ` | extra MMQ types; see [ggml-bridge.md](ggml-bridge.md) |

## Measured (from upstream docs and comments)

- `docs/DETAILS.md`, engine 0.1.36, RTX 5070: Q2_0 prompts 4K 1,294 -> 1,570, 32K 2,170 -> 2,653, 128K 2,123 -> 2,468
  tok/s with the fused kernels; "teacher-forced KL 0.009 vs 0.012" at 32K against an FP16 reference; native fused
  opt-in: "IQ2_XS prompts +12% at 4K, +3% at 32K, the IQ3 packs about even".
- `pick_ww` comment: per-layer milliseconds for 64- vs 128-row items on the RTX 5070 at 2,048 / 3,584 / 8,192 tokens.

## Tests

`prefill_fused_moe_test` and `prefill_fused_iq_test` (against a double-precision reference and the MMQ path),
`prefill_mmq_kquant_test`; see [tests.md](tests.md).

## Gotchas

- The header and the CMake comment of `moe_fused.hpp` still say the Q2_0 kernels are "opt-in with STRATA_PF_FUSED=1";
  the code has them on by default since 0.1.36 (`src/prefill/moe_fused.cu:401-410`).
- `--peer-device` keeps MMQ whether or not the peer took the prompt path, because `fused_ring()` sized the buffers
  for it (`src/prefill/prefill.cpp:2265-2267`).
- A native pack with any uncovered layer keeps MMQ's buffer sizes even for its covered layers (`fused_ring`, `:140-148`;
  the comment cites garbage or a hung prompt on gfx1151 before this rule).

## In UnbiasedStrata

Kept; changed in e41dfc7: widths come from the model (the fused IQ kernels read `n_embd` / `n_ff` from `NativeGeom`),
the router covers top-8 of 256 for Qwen3.6, and MMQ adds Q6_K under `STRATA_MMQ_KQUANTS`. See
[../../codebase/prefill/moe.md](../../codebase/prefill/moe.md).

## See also

[expert-streaming.md](expert-streaming.md), [ggml-bridge.md](ggml-bridge.md), [gemm.md](gemm.md),
[multi-gpu.md](multi-gpu.md), [tests.md](tests.md).
