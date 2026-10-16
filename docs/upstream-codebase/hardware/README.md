[Upstream docs](../README.md) › Hardware

# Hardware: Qwen3.8-Flash-Next on specific NVIDIA cards

This area follows the whole of upstream Strata (engine 0.1.39, commit `6f32ec0`) through two concrete hardware
targets: an NVIDIA RTX 50 card (Blackwell, `sm_120`, the architecture the engine was written for) and the RTX 3090
(Ampere, `sm_86`), alone and in pairs. It does not own any code. It explains how the parts documented elsewhere
fit together on one card: what the installer (`setup.py`, started by `START-HERE.bat` or `setup.sh`) chooses for a
card of a given VRAM and a PC of a given RAM, the engine command line that choice produces, how
`serve/server.py` starts the engine with it, and, part by part through the model's architecture, which code runs,
what it costs in VRAM and RAM and what keeps that cost down. Numbers are quoted with the machine they were
measured on; derived figures are marked as estimates. Terms such as "expert", "verify window" or "KV streaming" are
defined in the [glossary](../guides/glossary.md).

## Contents

| Page | What it covers |
| --- | --- |
| [rtx50-flash-next.md](rtx50-flash-next.md) | Qwen3.8-Flash-Next on an RTX 50 card: the architecture part by part with its code, VRAM and RAM cost; the `sm_90+` thread-block-cluster kernels and other RTX 50 paths; what the installer chooses for each RTX 50 card; sizing per card and model variant; build, run and check |
| [rtx3090-flash-next.md](rtx3090-flash-next.md) | Part A: one RTX 3090 (what upstream measured on it, what the installer chooses, what `sm_86` gets). Part B: two RTX 3090s (the installer's layer split, the engine's split search, helper caches and peer tier, measurements on 2x 3090) |

## Source map

The files below are the ones these pages cite most. Each is documented in its own area; this table is the index.

| File | Lines | Role on these pages |
| --- | ---: | --- |
| [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py) | 4,422 | the installer: finds the cards, recommends model, context, KV and GPUs, writes `strata-<model>.json` and starts the server |
| [setup.sh](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.sh) | 39 | Linux launcher: makes `.venv` and runs `setup.py` |
| [START-HERE.bat](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/START-HERE.bat) | 52 | Windows launcher for `setup.py` |
| [serve/server.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/serve/server.py) | 4,145 | the Python server: turns the config into the engine's arguments and environment (`engine_args`, `child_env`) and keeps one `strata --serve` process |
| [Dockerfile](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/Dockerfile) | 115 | container build: CUDA 13.0 image, engine for `75;80;86;89;120` |
| [CMakeLists.txt](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/CMakeLists.txt) | 1,254 | default arch `120`, the 7.5 floor, the CUDA 12.8 warning for `sm_120`, `decode_cluster_parity` |
| [include/strata/core/device.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/device.hpp) | 95 | `device_info`, `device_code_error` |
| [src/core/device.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/device.cu) | 344 | the run-time compute-capability check (7.5+, or 6.0+ in the experimental build) |
| [include/strata/core/layout.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/layout.hpp) | 97 | `ModelGeometry`: the model's fixed shape |
| [src/program/generate.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/program/generate.cpp) | 8,361 | the engine's `main`: options, PCIe probe, expert-cache sizing, the layer-split search, helper and peer tiers, the `INFO` line |
| [src/kernels/cuda/qsa_select.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/qsa_select.cu) | 1,237 | sparse-attention block scoring and top-k, including the `sm_90+` cluster top-k |
| [src/kernels/cuda/sampler.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cuda/sampler.cu) | 1,081 | sampling, including the `sm_90+` cluster greedy argmax |
| [src/kernels/decode_cluster_parity.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/decode_cluster_parity.cpp) | 424 | parity and timing program for the two cluster kernels |
| [src/prefill/moe_fused.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/moe_fused.cu) | 465 | the fused INT8 prompt-expert kernels and `STRATA_PF_FUSED` |
| [src/prefill/prefill.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/prefill/prefill.cpp) | 3,084 | the prompt path; the split's idle-stage help (`STRATA_PREFILL_HELP`) and the peer's prompt rows |
| [src/kernels/ngram.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/ngram.cpp) | 506 | the n-gram (PLE) table and its `--ple-io` modes |
| [src/core/session.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/session.cpp) | 970 | `session_bytes`: the per-layer-range session a split stage carves |
| [include/strata/core/verify.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/verify.hpp) | 364 | the verify window; `handoff_floats`, the residual one stage hands to the next |
| [src/core/remote_experts.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/remote_experts.cpp) | 341 | helper expert caches on CUDA1..3 (`--expert-cache-device1..3`) |
| [include/strata/core/remote_experts.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/remote_experts.hpp) | 89 | `RemoteExperts` |
| [src/core/remote_expert_opt.cu](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/remote_expert_opt.cu) | 194 | `--remote-expert-opt` for the helper caches |
| [include/strata/core/remote_expert_opt.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/remote_expert_opt.hpp) | 53 | `RemoteExpertOpt` |
| [src/core/peer_experts.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/core/peer_experts.cpp) | 325 | the adaptive peer tier (`--peer-device`) |
| [include/strata/core/peer_experts.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/core/peer_experts.hpp) | 106 | `PeerExperts` |

Upstream's own prose behind these pages:

| Document | Lines | What it gives |
| --- | ---: | --- |
| [docs/DETAILS.md](../../upstream/DETAILS.md) (copy kept in the fork) | 1,127 | every RTX 5070 measurement, the 3090 estimates, low-RAM modes, KV options |
| [docs/MODELS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MODELS.md) | 155 | the model variants and what fits which RAM |
| [docs/INSTALL.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INSTALL.md) | 260 | requirements, several cards, Docker |
| [docs/MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MULTI_GPU.md) | 174 | the layer split, its limits and measurements |
| [docs/SECOND_GPU.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/SECOND_GPU.md) | 162 | the helper caches, `--remote-expert-opt` and the peer tier |
| [bench/results/2026-09-29-rtx3090-epyc-milan/](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-29-rtx3090-epyc-milan/README.md) | - | one and two RTX 3090s, engine 0.1.26 |
| [bench/results/2026-09-29-layer-split/](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-29-layer-split/README.md) | - | RTX 5080 + RTX 3090 split sweep |
| [bench/results/2026-09-30-community-rtx-5090/](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/bench/results/2026-09-30-community-rtx-5090/README.md) | - | a community RTX 5090 run, engine 0.1.29 |

## Key concepts

**What a card changes, and what it does not.** The model's arithmetic is the same on every card. A card changes
three things:

1. **How many experts fit in VRAM.** About 96% of the model's bytes are routed experts (24,576 of them, 23-50 GB
   depending on the variant). Everything else of the model fits on a 12 GB card; what is left over becomes the expert
   cache. Every expert in VRAM is one the CPU does not compute. Upstream's rule of thumb is "every extra GB holds
   ~700 more experts" ([docs/HOW_IT_WORKS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/HOW_IT_WORKS.md)).
2. **Which kernels run.** `sm_90+` (RTX 50) gets two thread-block-cluster kernels in decode; `sm_80+` (RTX 30 and
   newer) gets tensor-core prompt kernels; `sm_75` (RTX 20) runs FP32 fallbacks.
3. **The host link.** Missed experts can be read over PCIe. The engine measures the link at start and scales the
   GPU's share of missed experts down below 20 GB/s.

The RAM decides which variant fits at all: by default every expert of the variant is held in pinned RAM, whatever
the card holds ([docs/MODELS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MODELS.md),
"Will it fit?").

**From card to engine command line.**

```
setup.py                                   serve/server.py                      strata --serve (engine)
 nvidia-smi: cards, VRAM, compute cap.  ->  reads strata-<model>.json     ->   generate.cpp main:
 RAM (GiB), CPU (AVX2/AVX-512), disk        engine_args(): args                  probe PCIe, size the expert
 recommends: family/size by RAM,              + --layer-split <auto|K>           cache from free VRAM, place a
   context by the smallest card's VRAM,       when "gpu" lists >1 card           layer split, fill caches from
   KV, KV streaming, low-RAM mode, GPUs     child_env(): CUDA_DEVICE_ORDER=       the expert profile, READY
 writes "args", "gpu", "layer_split"          PCI_BUS_ID, CUDA_VISIBLE_DEVICES
```

**The installer's per-card rules** (all in `setup.py`; details on the leaf pages):

| Decision | Rule | Code |
| --- | --- | --- |
| Usable card | compute capability 7.5+ (Pascal/Volta only in the experimental CUDA 12 engine); warns under 11 GB | `gpu_problem` `setup.py:629`, `setup.py:3797` |
| Several cards | offered when two or more can share (each 8 GB or more); the newest generation first; a pair whose second card is under 12 GB is not recommended | `together_ok` `:658`, `gpu_rank` `:641`, `split_short` `:664`, `choose_gpus` `:763` |
| Context | 32K under 14 GB, 64K under 20 GB, else 128K, by the smallest chosen card; capped by a RAM rule for IQ3_XXS / IQ3_S | `setup.py:3937-3942` |
| KV | FP16 up to 8K, else INT8 (or `q4_0` / `k8v4` when asked) | `setup.py:3986` |
| KV streaming | `--kv-resident 32768` from 64K when RAM holds the KV copy (not with `k8v4`, not under WSL) | `setup.py:4254-4290` |
| n-gram table | the engine's default `--ple-io direct` (SSD); `--ple-io ram` only for a model on a rotational disk on Linux with the RAM for it | `setup.py:4238-4250` |
| Engine | the ready-made engine, built with CUDA 13.0 (driver 580+), or a local compile for a card it has no code for | `setup.py:105-106`, `get_prebuilt` `:1946`, `build_engine` `:2274` |

## In UnbiasedStrata

| Part | Fork status |
| --- | --- |
| The RTX 50 kernels (`qsa_select.cu`, `sampler.cu`), `moe_fused.cu`, `decode_cluster_parity.cpp`, `device.cu`, the helper and peer tiers (`remote_experts.cpp`, `remote_expert_opt.cu`, `peer_experts.cpp`) | kept unchanged (`git diff 12a7825 HEAD` is empty for them) |
| `CMakeLists.txt`, `generate.cpp`, `prefill.cpp`, `session.cpp`, `verify.hpp`, `layout.hpp` | changed, mostly by the Qwen3.6-35B-A3B port (model geometry from the GGUF, a VRAM planner); the Flash-Next paths described here are upstream's |
| `src/kernels/ngram.cpp` | changed in `9a53e42`: `--ple-io ram` reads the table into anonymous memory instead of `mlock`ing the mapping |
| `setup.py`, `setup.sh`, `START-HERE.bat`, `serve/server.py`, `Dockerfile` | removed in `8894a81`; the fork has a C++ front end with fixed flags, one GPU only |

The fork's own versions of these two pages are
[../../codebase/guides/rtx50-flash-next.md](../../codebase/guides/rtx50-flash-next.md) and
[../../codebase/guides/rtx3090-flash-next.md](../../codebase/guides/rtx3090-flash-next.md); each leaf page here
ends with the differences.
