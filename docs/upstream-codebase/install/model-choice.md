[Upstream docs](../README.md) › [Install](README.md) › Model choice

# Model choice: families, sizes and the rules

## Purpose

This page covers which model files setup offers, the numbers it plans with, and how it picks the recommended answer
from the PC's RAM. It also covers the special rules for the Unsloth "budget" models.

Files covered: [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py)
(`MODELS` `setup.py:122-152`, `FAMILIES` `setup.py:183-217`, step 2 `setup.py:3857-3935`, `--check`
`setup.py:3838-3854`). User-facing background:
[MODELS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MODELS.md),
[UNSLOTH_Q4.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/UNSLOTH_Q4.md).

## The families (`FAMILIES`, `setup.py:183-217`)

A family is one Hugging Face repository layout of the same Qwen3.8-Flash-Next architecture.

| `--family` | Title | Repository (pinned in `HF_REVISIONS`) | Shard name pattern | Tag | Notes |
| --- | --- | --- | --- | --- | --- |
| `qwen` | Qwen3.8-Flash-Next | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, `<size>/` subfolder | `Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf` | (none) | The default (menu item 1) |
| `swift` | Swift 1.5 | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (no subfolder) | `Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-...-of-00002.gguf` | `swift-` | Prints its license line (Swift Open License 1.0). Own `mmproj` |
| `coder` | Qwen3.8-Flash-Next Coder | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`, `<size>/` | same pattern as `qwen` | `coder-` | Expert profile `data/expert-profile-coder.bin` |
| `unsloth` | Qwen3.8-Flash-Next (Unsloth) | `unsloth/Qwen3.8-Flash-Next-GGUF`, `<size>/` | `Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf` (4 shards; UD-IQ4_XS: 3) | `unsloth-` | `pack_args: --compat-bf16`; SHA-256 pinned per shard; `mmproj` from the ISTA-DASLab repo; images off by family default |

`model_name` in the config is `<family name>-<size lower case>`, for example `qwen3.8-flash-next-iq3_xxs` or
`swift-1.5-iq2_xs` (`setup.py:4325`).

## The sizes (`MODELS`, `setup.py:122-152`)

| `--model` | Families | Download (GB) | `ram_gb` (need) | `arena_gb` (experts) | Flags |
| --- | --- | ---: | ---: | ---: | --- |
| `Q2_0` | qwen | 66.4 | 48 | 34.0 | "2-bit, the fastest". Swift's Q2_0 is excluded because its shards split one layer's experts, which `iq_pack.py` cannot prepare (#171) |
| `IQ2_XS` | qwen, swift | 68.0 | 48 | 35.5 | |
| `IQ3_XXS` | qwen, swift | 75.8 | 60 | 42.9 | |
| `IQ3_S` | qwen | 83.6 | 62 | 50.3 | "needs a 64 GB PC with little else running" |
| `IQ1_M` | coder | 58.4 | 32 | 23.4 | The Coder's only size: 256 of 512 experts |
| `UD-IQ4_XS` | unsloth | 93.7 | 48 | 59.5 | `budget`, 3 shards, needs engine 0.1.38, images allowed. A regular choice from 0.1.39 (#621) |
| `UD-Q4_K_XL` | unsloth | 111.3 | 48 | 77.0 | `budget`, `nvidia_only`, `experimental`; needs engine 0.1.32 (`UNSLOTH_ENGINE`); images off |

- `ram_gb` is the RAM setup wants for the size. `arena_gb` is the size of its experts (the "arena", see the
  [glossary](../guides/glossary.md)). Both drive the RAM, low-RAM, context, disk and parallel rules.
- A size without `families` defaults to `("qwen", "swift")` (`setup.py:3871`).
- Per family, the menu shows: qwen Q2_0, IQ2_XS, IQ3_XXS, IQ3_S; swift IQ2_XS, IQ3_XXS; coder IQ1_M; unsloth
  UD-IQ4_XS, UD-Q4_K_XL. An experimental size is sorted last and is never the default (`setup.py:3872`).

## The recommended answers

| Question | Rule | Lines |
| --- | --- | --- |
| Which model? | Item 1, `qwen` | `setup.py:3859-3866` |
| Which size? | `IQ3_XXS` when RAM >= 60 (GiB as read) and the family has it, else item 1: Q2_0 for qwen, IQ2_XS for swift, IQ1_M for coder, UD-IQ4_XS for unsloth | `setup.py:3891-3892` |
| A `--model` the family does not have | Stops, naming the family's sizes and the `--family X --model Y` pairs that do have it. With `--gguf-dir` the hint adds the list of GGUFs Strata can run (#444) | `setup.py:3873-3879` |

The menu line for each size shows its download and RAM use. It adds "<- needs N GB RAM, you have M" when RAM is
short, or "<- fits in the low-RAM mode (the GPU holds ~X%, the rest in RAM/from the SSD)" where that applies. A budget
model shows the GiB of its experts that would stay in RAM (`setup.py:3880-3890`).

What `--yes` picks on the golden test's mocked PCs (`tools/test_setup_golden.json`, "recommended" rows):

| PC (RAM as read, cards) | Recommended size | Context | Other flags in `args` |
| --- | --- | --- | --- |
| 63.7 GB, 1x RTX 5090 32 GB | IQ3_XXS | 131072 | `--kv int8 --kv-resident 32768` |
| 31.9 GB, 2x RTX 3090 24 GB | Q2_0 | 131072 | `--resident-experts` (low-RAM, one GPU) |
| 47.0 GB, 2x RTX 5060 Ti 16 GB | Q2_0 | 65536 | layer split `[0, 1]`, `--remote-expert-opt` |
| 95.8 GB, 1x RTX 5080 16 GB | IQ3_XXS | 65536 | `--kv-resident 32768` |
| 127.8 GB, 1x RTX 4090 24 GB | IQ3_XXS | 131072 | `--kv-resident 32768` |

Note: upstream's prose docs recommend IQ2_XS for 48 and 64 GB
([MODELS.md "Pick by RAM"](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MODELS.md#pick-by-ram),
[AI_SETUP.md step 3](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/AI_SETUP.md#3-pick-the-model)).
The code's default is Q2_0 below 60 GB and IQ3_XXS from 60 GB, as the golden baseline shows. AI_SETUP.md also lists
Q2_0 among Swift 1.5's sizes, but `MODELS["Q2_0"]["families"]` is `("qwen",)` only.

## The RAM checks

| Check | Rule | Lines |
| --- | --- | --- |
| Too little RAM for any model | RAM < 32 - 4 = 28, not `--check`, and the low-RAM mode cannot help (`low_ram_fits("IQ1_M", ram, vram)` is false or `--low-ram off`): `confirm_risk` | `setup.py:3801-3816` |
| The chosen size pages | Not low-RAM, not a budget model, and RAM < `ram_gb` - 4: `confirm_paging()` asks "Install it anyway?". The default is n, or y with `--low-ram off` | `setup.py:3933-3934`, `setup.py:2486-2502` |
| Low-RAM mode | RAM < `arena_gb` + 10 (`LOW_RAM_HEADROOM_GB`) turns it on with `--low-ram auto` | [low-ram-and-multi-gpu.md](low-ram-and-multi-gpu.md) |

## The budget models (Unsloth UD-Q4_K_XL, UD-IQ4_XS)

These keep a RAM budget of their experts (`--resident-budget-gib N`) and read the rest from the GGUF on the SSD. They
never use the low-RAM mode or an `experts.bin` (`setup.py:3894-3927`):

1. UD-Q4_K_XL warns that it is EXPERIMENTAL and "several times slower than the 2-3-bit models".
2. On AMD, a model with `nvidia_only` (UD-Q4_K_XL) asks "Try it anyway?", default n, before the download (#429).
   UD-IQ4_XS is not asked.
3. RAM < 48: `confirm_risk` "Install it anyway?".
4. `budget_choice()` (`setup.py:1219-1230`) uses `--resident-budget-gib` as given, warning when it is above the
   recommendation, or else `resident_budget_gib()` (`setup.py:1211-1216`). The rule is
   `max(8, min(round(RAM) - 24 - ceil(KV RAM GB), arena_gb / 1.0737))`: 40 GiB on 64 GB, 71 GiB (all of UD-Q4_K_XL's
   experts) on 96 or 128 GB. `UNSLOTH_RAM_LEFT_GB = 24` is the room for the OS, the engine and the file cache.
5. Several GPUs: `unsloth_together()`, see [low-ram-and-multi-gpu.md](low-ram-and-multi-gpu.md#the-ram-budget-on-several-gpus).
6. `--low-ram resident|mmap|on` is ignored for them, with a warning. `--resident-budget-gib` on a non-budget size is
   ignored too, with a warning.
7. Step 4 stops if the engine is older than the model's `engine` (UD-IQ4_XS 0.1.38) or `UNSLOTH_ENGINE` (0.1.32),
   before the 94-111 GB download (`setup.py:4125-4128`).
8. Recommended context: 8K under 14 GB of VRAM, else 32K, because every GB of KV cache is a GB fewer of cached
   experts (`setup.py:3939-3940`).

## The `--check` verdicts

`--check` stops after step 1 and prints one line per size (`setup.py:3838-3854`):

- `fits` when RAM >= `ram_gb`; `tight` within 8 GB below; `does not fit` otherwise.
- Budget models: "[EXPERIMENTAL, ]fits with N GiB of its experts in RAM, the rest read from the SSD", or "does not
  fit". On AMD every budget model gets " - NVIDIA only so far, untested on AMD" appended, UD-IQ4_XS included, although
  the install itself does not ask for UD-IQ4_XS.
- Low-RAM sizes: "fits in the low-RAM mode (the GPU holds ~X% of its experts, the rest stays in RAM / is read from
  the SSD as needed)".

It ends with "This PC can run Strata. Run it again without --check to install." and exit code 0.

## Images, license and the speed projection per family

- Images are off for any size whose `vision` (or the family's) is `False`. UD-Q4_K_XL inherits `False` from the
  `unsloth` family and UD-IQ4_XS overrides it with `True` (`setup.py:3996-3999`).
- Swift 1.5 prints its license line after the family is chosen.
- The experimental speed projection is offered only for `qwen` and `coder` (`setup.py:4045-4060`)
  ([context-kv-and-options.md](context-kv-and-options.md#experimental-speed-projection)).

## Tests

`tools/test_setup_golden.py` (the recommended sizes on five PCs), `tools/test_setup_unsloth.py` (pins, budget, engine
version, AMD question, `--check` verdict, family default and menus), `tools/test_setup_choices.py`
(`GgufDirUnsupported`, the size of another family), and `tools/test_setup_risk.py` (`RamFloor`, paging).

## In UnbiasedStrata

Removed. The fork runs the model it is given (`--model SHARD1.gguf`): Qwen3.8-Flash-Next IQ3_XXS per its
`UPSTREAM.md`, and it added Qwen3.6-35B-A3B. No sizes are offered and there is no RAM rule.

## See also

[context-kv-and-options.md](context-kv-and-options.md) · [low-ram-and-multi-gpu.md](low-ram-and-multi-gpu.md) ·
[downloads-and-packs.md](downloads-and-packs.md) · [DETAILS.md "Which model?"](../../upstream/DETAILS.md#which-model)
