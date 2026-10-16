[Upstream docs](../README.md) › [Install](README.md) › Context, KV and options

# Context, KV cache, prefill, images, draft vocabulary and the other options

## Purpose

This page covers the choices step 2 and step 7 make after the model and size: how long a context, which rope
scaling, which KV cache precision, whether the KV cache streams to RAM, images, the draft head's token subset, VRAM
reserve, parallel slots, the experimental speed projection and the n-gram table's I/O. For each one it gives the rule
setup recommends and the engine flag it writes.

Files covered: [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py)
(`setup.py:3936-4060`, `setup.py:4217-4374`, and the helpers cited below).

## Fixed engine flags

Every config starts with the same core (`setup.py:4224-4227`):

```
--pack <pack> --native <shard 1> [--ple-gguf <shard with per_layer_token_embd>] --expert-profile <data/...bin>
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp <mtp/rt> --max-context <ctx>
```

- `--ple-gguf` is written only for files with 2 shards or fewer. The shard holding `per_layer_token_embd.weight` is
  found by reading each shard's tensor directory (shard 2 for the original, shard 1 for Swift). For 3- and 4-shard
  Unsloth files the engine finds it itself (`setup.py:4219-4224`). Setup stops if no shard has the tensor.
- **Prefill**: always `--prefill auto` (the engine picks its chunk, up to 8192, from free VRAM). Configs from before
  engine 0.1.13 had `--prefill 2048`, which `upgrade_config()` rewrites at the next start (`setup.py:2987-3009`).
- `--spec 4` and `--spec-min-p 0.5` are the MTP speculative-decoding defaults. A calibration may replace
  `--spec-min-p` ([calibration.md](calibration.md)).

## Context (`--context`)

- Choices: `CONTEXTS = 8192, 32768, 65536, 131072, 262144, 393216, 524288` (`setup.py:181`). `--context` takes any
  integer.
- **Recommended** (`setup.py:3937-3942`), from the smallest VRAM among the chosen cards (each card keeps its layers'
  KV for the whole context): under 14 GB 32K, under 20 GB 64K, else 128K. Budget models: under 14 GB 8K, else 32K.
- **RAM rule** (#406): `ctx_ram_need()` (`setup.py:2505-2511`) estimates `arena_gb + ctx x 13 x 1056 B + 24 GB` for
  IQ3_XXS and IQ3_S only, and not in the low-RAM mode. `ram_ctx()` (`setup.py:2514-2517`) is the longest context
  that fits, but never less than 131072, so it does not lower any of the recommendations above. Its job is the menu
  note "(needs ~N GB RAM, this PC has M: may run out of memory)" past 128K, and the warning when such a context is
  kept anyway (`setup.py:3960-3965`).
- 384K and 512K are marked "(experimental: setup adds rope scaling)".

## Rope scaling past 262144 (`--rope-scaling`, `--rope-scale`)

`resolve_rope(ctx, scaling, scale)` (`setup.py:3479-3503`) works on the final context:

| Context | `--rope-scaling` | `--rope-scale` | Result |
| --- | --- | --- | --- |
| <= 262144 | omitted or `none` | omitted | no rope flags (the stock model) |
| <= 262144 | omitted or `none` | given | error: "--rope-scale needs --rope-scaling linear or yarn" |
| <= 262144 | `linear` / `yarn` | omitted / given | that method, factor `max(1, ctx/262144)` = 1, or as given |
| > 262144 | omitted | omitted / given | `yarn`, factor `ctx/262144` (1.5 for 384K, 2 for 512K) or as given |
| > 262144 | `none` | any | refused: "the model has never seen those positions" |
| > 262144 | `linear` / `yarn` | omitted / given | that method, the derived or given factor |

When the run is interactive and the context is past 262144 with no method given, setup asks "RoPE extension method?"
(yarn or linear, default yarn). The result is written as `--rope-scaling <m> --rope-scale <f:g>` (`setup.py:4228-4229`).

## KV cache precision (`--kv`)

- Context <= 8192: the KV cache stays fp16 and no `--kv` flag is written. A given `--kv` is ignored
  (`setup.py:3986`).
- Above 8K: `--kv` as given (`int8`, `q4_0` or `k8v4`), else `int8`. Interactively, without `--kv` and without
  `--yes`, setup asks "KV cache?": 1) 8-bit (recommended, "what every published number was measured with"),
  2) 4-bit (Hadamard-rotated, half the memory). `k8v4` (INT8 K + 4-bit V, 816 B per cell, per the flag's help) is
  available only as a flag.
- Written as `--kv <kv>` (`setup.py:4230-4231`).

## KV streaming (`--kv-streaming auto|on|off`)

From a 64K context, the whole KV cache can live in RAM with only the attention's 32K-position window in VRAM, so more
experts fit on the GPU. The flag is `--kv-resident 32768` (`setup.py:4251-4292`).

- RAM cost: `kv_ram_gb = ctx x 13 x (576 for q4_0, else 1056) B` (12 attention layers plus the draft layer).
  `stream_fits = RAM >= ram_gb + kv_ram_gb + 1`.
- Decision order: `k8v4` never streams (the engine refuses the pair). WSL never streams (the driver pins only about
  1 GB). `off` keeps it in VRAM. With ctx >= 65536 and (`stream_fits` or `on`) it streams, warning when it does not
  fit. Otherwise, at 64K and above, it says why it is off (#620). `on` under 64K warns and stays off.
- With a budget model and no explicit `--resident-budget-gib`, the KV RAM is taken out of the budget:
  `resident_budget_gib(model, ram, kv_ram_gb)`. An explicit budget above that gets a warning and is kept.

## Images (`--vision yes|no|none|gpu|cpu`, `--vision-tokens N`)

- Interactive question "Do you want images?" (default n; "a 0.9 GB download and keeps ~1.4 GB of VRAM free").
  `--yes` leaves images off. `yes` means `gpu` (`setup.py:4000-4009`).
- Off for UD-Q4_K_XL. On AMD, `hip_vision()` (`setup.py:1904-1914`) gives the CPU encoder with `--vision cpu` on
  Linux and off otherwise, because there is no HIP encoder and the Windows HIP zip has no encoder.
- A ready-made engine whose encoder has no code for the card falls back to the CPU encoder (`prebuilt_vision()`,
  `setup.py:2260-2271`, #331).
- Written as `args += --vision --vram-reserve-mib 700` (the `VISION` table: 700 MiB for both devices), plus a
  `vision` block in the config: `exe` (`strata-vision`), `mmproj`, `model` (shard 1), `gpu` (bool), `max_tokens`, and
  `threads = cpu_count // 2` for the CPU encoder (`setup.py:4295-4300`, `setup.py:4366-4372`).
- `max_tokens`: 1024 on the GPU, 300 on the CPU. `--vision-tokens N` overrides it, or the earlier config's value for
  the same device is kept (`vision_tokens()`, `setup.py:3216-3238`, #625).
- On a GPU of 12.5 GB or less, images on the GPU print a tip to use `--vram-reserve-mib 1000`
  (`VISION_GPU_SMALL_RESERVE_MIB`).

## VRAM reserve (`--vram-reserve-mib N`)

VRAM the engine leaves free for other programs, 700 MiB by default in the engine. Given at setup, it replaces the
images' 700 or is added (`setup.py:4301-4311`). It warns when it is below what the GPU encoder needs. Given at a
start, it is saved into the config's `args` (`setup.py:3075-3083`, #493).

Tips that change nothing (recommendations only):

- **Card under 7.5 GB** (one GPU): `small_card_note()` (`setup.py:3263-3280`, #496) suggests an 8K context and
  `--draft-vocab en` if the start stops with "no VRAM is left for the expert cache".
- **AMD card driving a Linux desktop**: `desktop_reserve_note()` (`setup.py:3346-3352`, #560 #516) suggests
  `--vram-reserve-mib 3072`.

## Draft vocabulary (`--draft-vocab cjk|en|cyrillic|fr`)

The MTP draft head proposes tokens only from a subset of the vocabulary, stored in `mtp/rt/draft_vocab.bin`.

| Choice | Shipped file | Draft head VRAM (`DRAFT_VOCAB_MIB`, IQ3_S the largest) |
| --- | --- | ---: |
| `cjk` (default) | `data/draft_vocab.bin` (with Chinese, Japanese, Korean; since 0.1.27, #137) | 348 MiB |
| `cyrillic` | `data/draft_vocab_cyrillic.bin` | 193 MiB |
| `fr` | `data/draft_vocab_fr.bin` (#597) | 151 MiB |
| `en` | `data/draft_vocab_en.bin` (English and code) | 133 MiB |

- `refresh_draft_vocab(rt, choice)` (`setup.py:3366-3385`) copies the chosen file when `rt/draft_vocab.bin` is
  missing, or when it is one of the shipped subsets (or the pre-0.1.27 one in `OLD_DRAFT_VOCABS`) and differs. A
  subset made by hand is kept. It runs at setup, at every start and on `--update`.
- The choice is saved as `"draft_vocab"`. A re-run without the flag keeps it (`saved_draft_vocab()`,
  `setup.py:3206-3213`).
- Cards under 14 GB without a chosen subset get a tip to use `--draft-vocab en` (`draft_vocab_note()`,
  `setup.py:3245-3257`, #474).

## Several requests at once (`--parallel N`)

`"parallel": N` in the config makes the server pass `--batch N` to the engine (`serve/server.py:1485-1503`). Setup
writes it only when `--parallel` is given (`setup.py:4355-4365`). Otherwise it prints a recommendation at the end:

- `parallel_slot_gb(ctx, kv, streaming)` = `min(ctx, 32768 if streaming) x 12 x (576 | 1056) B + 0.17 GB`. The code
  comment gives a measured 0.56 GiB at 32K int8.
- `parallel_recommend()` (`setup.py:3297-3312`) takes `cache = sum(VRAM - 5)` over the chosen cards. The largest
  `n` in 2..4 with `n x slot <= 0.2 x cache` and `cache - n x slot >= 0.5 x arena_gb` is recommended, else 0 (one at
  a time).
- The note cites a measurement on a 12 GB RTX 5070 (Q2_0, 32K): "a request alone 11-24% slower with 2-4 slots"
  (`setup.py:3301-3303`). Values above 8 are noted as clamped by the engine.

## Experimental speed projection

Offered for `qwen` and `coder` only, off by default ("Turn on the experimental speed projection?", default n;
`setup.py:4041-4060`). The text says it "changes how the model answers" and describes the vector as "a
refusal-direction projection". `on` uses
`data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf`, and a path uses that GGUF.
Written as `--control-vector-scaled <gguf>:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir
per-layer` (`setup.py:4320-4323`). For Swift and Unsloth a request is ignored with a warning.

## The n-gram table on a spinning disk (`--ple-io ram`)

If the PLE shard sits on a rotational disk (Linux, not WSL; `rotational_disk()`), setup adds `--ple-io ram` when
`RAM >= ram_gb + table + 4`, where the table is the `per_layer_token_embd` tensor's size or 28.8 GB. Otherwise it
warns that prompts can stall for minutes (`setup.py:4238-4250`, #605).

## Other args

- Low-RAM `--resident-experts` / `--mmap-experts`, the budget `--resident-budget-gib`, and `--remote-expert-opt`:
  see [low-ram-and-multi-gpu.md](low-ram-and-multi-gpu.md).
- `--pool-workers` (hybrid CPUs) and calibrated `--pcie-frac`: see [calibration.md](calibration.md).

## Tests

`tools/test_setup_rope.py` (every row of the rope table), `tools/test_setup_risk.py` (`Context`, `KvStreaming`),
`tools/test_setup_parallel.py`, `tools/test_setup_draft_vocab.py`, `tools/test_setup_config.py` (`VisionTokens`),
`tools/test_setup_choices.py` (`PrebuiltVision`, `HipVision`, `RotationalDisk`, `VramReserve`, `SmallCardTip`,
`DesktopReserveTip`).

## In UnbiasedStrata

Removed as choices. The fork's front end fixes most of them (`frontend/main.cpp:676-692`): the same core flags,
`--max-context` from `--ctx` (default 32768), `--kv int8` above 8K, `--ple-gguf <shard 2> --ple-io ram` whenever there is a second shard
(with its own reader, see the fork's `UPSTREAM.md`), `--vision` when `--mmproj` is given, and no rope, KV streaming, speed projection or
batch flags unless passed after `--`. `--batch` is refused there.

## See also

[config-file.md](config-file.md) · [model-choice.md](model-choice.md) ·
[DETAILS.md "Images (vision)"](../../upstream/DETAILS.md#images-vision) ·
[INSTALL.md "Setup's questions"](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INSTALL.md#setups-questions)
