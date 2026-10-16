[Upstream docs](../README.md) › [Intel SYCL port](README.md) › Hardware and results

# Target GPUs and reported results

## Purpose

Which Intel GPUs the port is written for, which ones its setup recognises, what has actually been run on which
card, and the numbers upstream's documents report. Every number below is quoted from
[docs/INTEL.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INTEL.md)
(the port author's notes) or
[docs/INTEL_ARC.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INTEL_ARC.md)
(the maintainers' page), with the date and engine version given there; none was re-measured for this page.

Files covered: `sycl/setup_intel.py` (the card table), `sycl/CMakeLists.txt` (`STRATA_SYCL_AOT`), the two docs above.

## The target

The port was written for and measured on an **Intel Arc Pro B70** (32 GB, Xe2 "Battlemage", PCI id `8086:e223`,
AOT device `bmg-g31`) running the Coder (IQ1_M) on Ubuntu 24.04, in a PCIe 3.0 x8 slot (INTEL.md). Its design
choices assume a card large enough to hold every expert of the model in VRAM (`--stream-experts`); smaller cards
mirror the rest in pinned host RAM and are slower (INTEL_ARC.md). The GPU must run the `xe` or `i915` kernel driver
with Level Zero.

AOT device names given by INTEL_ARC.md for `-DSTRATA_SYCL_AOT`: `bmg-g31` (Arc Pro B70) and `bmg-g21` (Arc B580,
B570, Pro B60). Other cards use the default SPIR-V build, compiled at run time.

## Cards setup recognises

`INTEL_ARC` in `sycl/setup_intel.py:44-47` (PCI device id → name, VRAM in GB). An unknown id is still accepted under
the `xe` driver, named `Intel GPU <id> (xe)` and sized from its VRAM BAR; under `i915` it is skipped as an integrated
GPU.

| PCI id | Name | VRAM (GB) | Family |
|---|---|---|---|
| `e223` | Arc Pro B70 | 32 | Battlemage |
| `e221` | Arc Pro B60 | 24 | Battlemage |
| `e20b` | Arc B580 | 12 | Battlemage |
| `e20c` | Arc B570 | 10 | Battlemage |
| `e212` | Arc B50 | 16 | Battlemage |
| `56a0` | Arc A770 | 16 | Alchemist |
| `56a1` | Arc A750 | 8 | Alchemist |
| `56a2` | Arc A580 | 8 | Alchemist |
| `56a5` | Arc A380 | 6 | Alchemist |
| `56a6` | Arc A310 | 4 | Alchemist |
| `5690` | Arc A770M | 16 | Alchemist (mobile) |

`gpustat.py` has its own name table with only `e223` (`sycl/tools/gpustat.py:23`) and falls back to `lspci`.

## What has been run, and by whom

From INTEL_ARC.md "What has been run, and by whom" (release 0.1.39):

| Who | Hardware | Result |
|---|---|---|
| Port author | Arc Pro B70 32 GB, Ubuntu 24.04 | Coder IQ1_M: 70-78 tok/s decode, ~790 tok/s prompt; IQ2_XS: 51-64 tok/s; 256K context measured |
| Community | 2x Arc Pro B70, `--layer-split` | Flash-Next IQ3_XXS 66 tok/s decode, 394 tok/s prompt (with the `stage_room` fix now in 0.1.39) |
| Community | Arc Pro B50 16 GB | Coder IQ1_M ~23 tok/s, IQ2_XS ~25-27 tok/s, up to 128K |
| Community | Arc B580 12 GB, WSL2 | IQ2_XS ~21 tok/s, Coder ~15 tok/s; device loss also seen |
| Maintainers | no Arc | compile check and kernel tests on a CPU device only |

"Nobody has run the 0.1.39 port on an Arc yet"; the numbers were measured on earlier versions. Not tested by anyone
(INTEL_ARC.md): Windows, Arc on WSL2 for the 0.1.39 port, the Alchemist A-series, integrated Arc GPUs (the B390 /
Panther Lake of issue #515; "the shared-memory planning does not exist yet"), and images.

## Reported numbers on the Arc Pro B70

INTEL.md, "What you get (2026-10-01)", Coder IQ1_M, engine 0.1.31-sycl (0.1.32-0.1.38-sycl "reproduce them
exactly"), greedy runs of the engine with 256 new tokens, AOT build:

| | Strata SYCL port | llama.cpp SYCL on the same card | Strata CUDA engine, for comparison |
|---|---|---|---|
| decode | 78.2 tok/s (19-token prompt); 75.7 after 2,184 tokens; 69 after 40K; 66.6 after 128K; 55.9 after 256K; 48-72 through the API | 23-25 tok/s | 44-51 tok/s on an RTX 5070 |
| prompt | 790 tok/s at 2,184 tokens; 1,117 at 40K; 888 at 128K; 757 at 256K | ~150 tok/s (424 at 105K) | ~1,870 tok/s |
| speculative decoding | MTP draft layer of the base checkpoint, 70-85% of drafts accepted on code | no | yes |
| context | 256K measured (`--kv-resident`) | 131K measured ceiling | up to 262K |
| RAM | little (23 GB is fine): no host copy of the experts | little | 32-64 GB |

Other results in INTEL.md, same card:

| What | Result |
|---|---|
| Original Qwen3.8-Flash-Next IQ2_XS (bigger than VRAM) | 18,329 of 24,576 experts in VRAM, 6,247 (8.4 GiB) in the pinned host mirror; decode 50.8 tok/s (19-token prompt) and 60.6 after 2,184 tokens on 2026-10-01; after the 0.1.31 merge 58.6 / 64.2 |
| Swift 1.5 IQ2_XS | 48 tok/s raw decode after two port fixes (2026-10-01) |
| Long contexts, 0.1.32-sycl, `--kv-resident 32768 --vram-reserve-mib 2048 --prefill 4096` | 128K int8: 888 tok/s prompt, 66.6 tok/s decode, 29.9 GB peak VRAM; 256K int8: 757 / 55.9, 30.2 GB |
| An 80,000-token prompt | 101 s (790 tok/s) without slot borrowing; 75 s (1,062 tok/s) and 35-40 tok/s decode after it with borrowing |
| Load time (2026-10-02) | Coder start to first token 82 s → 26 s after the fill was pipelined; IQ2_XS 120 s → 41 s |
| Kernel bandwidth | dense decode kernels 100-280 GB/s of a 608 GB/s card (`mmvq_sg_bench`); Q6_K wide kernel up to 493 GB/s after the alignment fix |

INTEL.md's "Decode round 2" table traces Coder decode from 53.9 to 77.9 tok/s through individual kernel changes
(the parallel `resident_plan`, the direct and sliced GR down kernels, wide 16-byte-load kernels, the IQ4_NL expert
dots); [differences.md](differences.md#7-kernel-work-for-xe2-speed) lists those changes and their switches.

## Platform behaviour that matters

From INTEL.md and INTEL_ARC.md:

- **Do not over-commit VRAM.** On the `xe` driver a GPU allocation past VRAM does not fail: buffers are evicted to
  host RAM, the kernel runs out of memory and the machine livelocks until its watchdog resets it (a 262K-context
  llama.cpp request did this on 2026-09-29). Leave about 1.5 GB free. This is why `setup_intel.py` sizes
  `--vram-reserve-mib` explicitly.
- **`SYCL_CACHE_PERSISTENT=0`.** The persistent JIT cache segfaulted on Xe2 during the first compile; the dev image
  sets it.
- **A wedged GPU.** An unbounded device spin made the driver reset the GT and the card stayed unusable until a
  reboot (twice); hence `kSpinMax` ([differences.md](differences.md#2-correctness-fixes-for-the-intel-gpu)).
- **Host-mapped memory.** `volatile` device accesses are cached on Intel; host-to-device visibility while a kernel
  runs is unreliable. `STRATA_VERIFY_NO_HOST=1` (set by `strata-sycl.sh`) is valid only with every expert in VRAM;
  on smaller cards that path has hung, and issue #667 suspected the GPU does not see CPU flag stores without a system
  fence (INTEL_ARC.md).
- **Level Zero v2 adapter.** Host waits on queue events caused hangs; the port polls sequence numbers instead.
- **Two cards.** `ONEAPI_DEVICE_SELECTOR=level_zero:*` and `--layer-split`.
- **Measured slot.** The B70 trained at PCIe Gen3 x8 in the author's machine (it can do Gen5 x16), which affects the
  host-mirror results.
- **Speed depends on the text.** Decode with speculative decoding follows draft acceptance: 45-51 tok/s on code,
  30-39 on prose at 32K (INTEL.md).

## The llama.cpp half of INTEL.md

The first part of INTEL.md ("Things that matter on this GPU", "How much context fits") describes serving the same
GGUF through llama.cpp's SYCL backend, the comparison point; its context table (32,768 to 131,072 tokens measured,
262,144 does not fit on 32 GB) is about llama.cpp, not the port.

## In UnbiasedStrata

Not applicable: the fork removed the port (`8894a81`) and supports no Intel GPU.

## See also

- [build.md](build.md#options): `STRATA_SYCL_AOT`.
- [serving.md](serving.md#syclsetup_intelpy): how setup detects the card.
- [differences.md](differences.md): the engine changes behind these results.
