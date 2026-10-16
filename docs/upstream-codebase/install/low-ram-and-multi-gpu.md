[Upstream docs](../README.md) › [Install](README.md) › Low-RAM and multi-GPU

# The low-RAM mode, the RAM budget, and several GPUs

## Purpose

This page covers three things. First, how setup runs a model whose experts do not fit in RAM: the low-RAM mode for
the 2-3-bit sizes, and the RAM budget for the Unsloth sizes. Second, how it chooses one GPU or several (`--gpu`,
`--gpus`, `--layer-split`). Third, how the two interact, both at setup and later at each start.

Files covered: [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py)
(`setup.py:544-933`, `setup.py:2343-2471`, `setup.py:3928-3932`, `setup.py:4010-4040`, `setup.py:4339-4346`,
`setup.py:3095-3160`, `setup.py:3388-3450`). Background:
[MODELS.md "A big graphics card and little RAM"](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MODELS.md#a-big-graphics-card-and-little-ram),
[INSTALL.md "Two or three cards"](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/INSTALL.md#two-or-three-cards).

## The low-RAM mode (`--low-ram auto|on|off|resident|mmap`)

Normally the engine copies all of a model's experts (`arena_gb`) into RAM, and the GPU caches the most-used ones. In
the low-RAM mode the experts come instead from one file, `<pack>/experts.bin`, written once by
`tools/iq_pack.py --experts-bin` ([downloads-and-packs.md](downloads-and-packs.md#the-pack)).

| Function | Lines | Rule |
| --- | --- | --- |
| `low_ram_needed(model, ram)` | `setup.py:2347-2350` | `RAM < arena_gb + 10` (`LOW_RAM_HEADROOM_GB`: the OS, the engine's other buffers, the server) |
| `low_ram_gpu_gb(model, vram, ctx, kv)` | `setup.py:2353-2359` | GB of experts the GPU cache holds: `VRAM - 5 - (ctx - 32768) x 13 x (576 / 1056) B`, clamped to `[0, arena_gb]`. In this mode the KV cache stays in VRAM |
| `low_ram_resident(...)` | `setup.py:2367-2372` | Resident variant when `RAM >= (arena_gb - gpu_gb) + 10` |
| `low_ram_fits(model, ram, vram)` | `setup.py:2375-2378` | `RAM - 6 + max(0, VRAM - 5) >= arena_gb`. Used for the menu notes, the `--check` verdicts and the RAM floor |

Decision in `main()`:

1. `low_ram = no budget model and (--low-ram on|resident|mmap, or auto and low_ram_needed)` (`setup.py:3928-3929`).
2. Several GPUs chosen: `low_ram_together()` (below).
3. Once the context and images are known (`setup.py:4015-4040`), the VRAM is the main card's less the image encoder's
   reserve. The **resident** variant is used with `--low-ram resident`, or when it is not `mmap` and
   `low_ram_resident()` holds. Otherwise the **mapped** variant is used. Several GPUs always use mapped. Setup prints
   the GPU's share and warns under 60% that "most of the experts are read from the SSD".
4. An engine older than `RESIDENT_ENGINE = 0.1.30` gets mapped instead of resident (`setup.py:4232-4235`).
5. Written as `--resident-experts` or `--mmap-experts` (`setup.py:4236-4237`). On HIP the resident variant also gets
   `env STRATA_RESIDENT_PIN=0`, because large page-locked allocations can fail or be slow under ROCm
   (`setup.py:4337-4338`).
6. Disk: `arena_gb + 1` more for `experts.bin`, except for Q2_0 on an AVX-512 CPU, whose pack already has one.

| Variant | Engine flag | What happens to the experts the GPU does not hold |
| --- | --- | --- |
| resident | `--resident-experts` | Copied from `experts.bin` into RAM once, so nothing is read from the SSD while it answers. RAM use is steady. The engine falls back to mapped when they do not fit the free RAM (per the code comment) |
| mapped | `--mmap-experts` | Read through the OS file cache, so a PC short of RAM keeps re-reading them from the SSD |

## The RAM budget (Unsloth `UD-*`)

`--resident-budget-gib N`: the engine keeps N GiB of the most-used experts in RAM and reads the rest from the GGUF
in place. The rule and the questions are in [model-choice.md](model-choice.md#the-budget-models-unsloth-ud-q4_k_xl-ud-iq4_xs).

### The RAM budget on several GPUs

The engine refuses `--resident-budget-gib` with a layer split, exiting with code 2 (#498). So:

- `unsloth_split_need_gb(model) = download_gb + 24`: about 135 GB for UD-Q4_K_XL, 118 GB for UD-IQ4_XS
  (`setup.py:834-838`).
- At setup, `unsloth_together()` (`setup.py:2437-2471`) behaves as follows. With RAM below that need it uses one GPU.
  With an explicit `--resident-budget-gib` it uses one GPU. With `--gpus` it uses all the cards without a budget.
  Otherwise it asks "<model>: which GPUs?", where 1 is one GPU (recommended, "the tested setup") and 2 is all of them
  ("about twice as fast in #498 (2x RTX 3090: 31 -> 64-78 tokens/s)"). With `--yes` it uses one GPU.
- A split writes no `--resident-budget-gib` (`q4_split`, `setup.py:3920`, `setup.py:4293-4294`).
- At a start, `split_budget(cfg)` (`setup.py:841-859`) removes the budget from a config started on several GPUs. If
  RAM is short it stops before saving.

## Choosing several GPUs

### NVIDIA, at setup

| Function | Lines | Rule |
| --- | --- | --- |
| `together_ok(found)` | `setup.py:658-661` | Cards with no `gpu_problem(g, together=True)`: compute capability 7.5+ (or opted-in Pascal/Volta) and at least 7.5 GB (`SPLIT_MIN_VRAM_GB = 8`, minus 0.5). Empty if fewer than two |
| `gpu_rank(g)` | `setup.py:641-644` | Order: newest arch first (it gets the first layers and most of the work), then VRAM, then index |
| `split_short(cards)` | `setup.py:664-669` | #448: when the first card has 11.5 GB or more, the later cards under 11.5 GB (`SPLIT_PROMPT_VRAM_GB = 12`). Such a card cannot lend a 4096-token prompt chunk, and the code comment cites a 10 GB RTX 3080 beside a 32 GB card at "512 tokens, prompts 6.2x slower" |
| `parse_gpus(text, found)` | `setup.py:678-692` | `"0,2"` or `"all"` (`together_ok`). Needs two or more distinct numbers |
| `check_gpus(sel, found, ..., named)` | `setup.py:695-721` | Stops on a missing or unusable card, saying what can be used. A card named by the user that only lacks VRAM for sharing is a `confirm_risk` ("Use it anyway?") instead |
| `choose_gpus(a, found)` | `setup.py:763-798` | `--gpus`, then `--gpu`, else: with `together_ok` it asks "Which GPUs?" with options [the best pair together], [all of them] (with three or more), then each single card. The pair is recommended, or the first card alone when `split_short` flags the pair |

The prompt explains the trade: "each card holds the experts of its own layers, so together they hold about twice as
many, and prompts are read about 20% faster ... A much slower extra card can also make it slower." The first chosen
card is the main one: its VRAM sizes the defaults, and `gpu["archs"]` lists every chosen card's arch so the engine
gets code for all of them.

### AMD, at setup

There is no question. Setup uses one card (the most VRAM, or `--gpu N`) unless `--gpus` names several (Linux only;
Windows stops). `amd_parse_gpus()` (`setup.py:1427-1451`) accepts `"1,0"` or `"all"` (supported cards, most VRAM
first), numbered as HIP numbers them. The cards may be of different archs, and the engine is compiled for each. Cards
of two ROCm wheel families need a system ROCm 7 ([engine.md](engine.md#amd-on-linux-compiled-here)).

### Low-RAM mode with several GPUs

`low_ram_together()` (`setup.py:2402-2434`; #364 #384) handles this. The resident variant has no layer split, so all
the cards together use the mapped variant.

- With `--gpus` it keeps all the cards, with a warning that "RAM can fill up to 0 free during long prompts".
- Otherwise, unless `--low-ram resident` or `--yes`, it prints the RAM math (`low_ram_one_gpu_why()`,
  `setup.py:2381-2399`) and asks "Low-RAM mode: which GPUs?". Option 1 is one GPU (recommended, steady RAM use) and
  option 2 is all of them ("1.3-1.6x" faster in #364 #384, per the text).
- With `--yes` or `--low-ram resident` it uses one GPU.

### What is written

- One card: `"gpu": <index>` and `"gpus_asked": true`, when the PC has more than one card or `--gpu` was given
  (`setup.py:4339-4341`).
- Several: `"gpu": [main, ...]` and `"layer_split": <--layer-split or "auto">`. `recommend_remote_expert_opt()`
  (`setup.py:865-882`) appends `--remote-expert-opt` to `args` unless `--no-remote-expert-opt` is given or the config
  says `"remote_expert_opt": false`. The flag makes helper expert caches complementary to the main GPU's (the code
  comment cites "dual RTX 4090: +63% mixed, +132% code" for #578).
- The server turns `gpu` into `CUDA_DEVICE_ORDER=PCI_BUS_ID` plus `CUDA_VISIBLE_DEVICES`, or `HIP_VISIBLE_DEVICES`
  for AMD. A list also becomes `--layer-split <value>` (`serve/server.py:1382-1389`, `serve/server.py:1428-1479`,
  `serve/server.py:1560-1576`). `--layer-split` takes the first layer of each later GPU, one rising number per extra
  GPU, for example `"18"` or `"16,32"`. It is not a count of layers per card.

## At each start (`start()`, `setup.py:3095-3160`)

| Given at start | NVIDIA | AMD (HIP) |
| --- | --- | --- |
| `--gpus 0,2` / `all` | `check_gpus(named=True)`, saved as `gpu` + `layer_split` + `gpus_asked`, `split_budget`, `--remote-expert-opt` | `amd_parse_gpus`; stops if the engine has no code for a card ("set it up for these cards: ./setup.sh --setup --backend hip --gpus ..."), `split_budget`, saved |
| `--gpu N` | This start only: `check_gpus([N])`, server `--gpu N` | Server `--gpu N` |
| neither | `offer_together()` (`setup.py:885-932`): once per config (`gpus_asked`), when two cards could share it and, for a budget config, RAM holds the split. Asks "Use both from now on?", defaulting to y, but to n for a resident low-RAM config, a `split_short` pair or a budget config | The saved choice |

Then, for a list of cards:

- `split_mmap(cfg)` (`setup.py:801-814`) replaces `--resident-experts` with `--mmap-experts`, with a warning.
- `split_budget(cfg)`, as above.
- `ensure_engine_for(cards, ...)` (`setup.py:3388-3412`): a card the installed engine has no code for gets the engine
  compiled for all cards before the start (#128). A Pascal/Volta card added to a CUDA 13 model moves the model to the
  CUDA 12 engine (`use_cuda12()`, `setup.py:3432-3450`).

## Gotchas

- `"gpus_asked": true` is also written for a one-card choice made at setup, so a model set up on one card of a
  two-card PC is never offered the pair at start.
- `--gpu` at a start is not saved, while `--gpus` is saved. With `--setup`, `--gpu` is saved too (the `--gpu` help text,
  `setup.py:3558-3559`).
- The parallel-slot recommendation counts every card of a split ([context-kv-and-options.md](context-kv-and-options.md#several-requests-at-once---parallel-n)).

## Tests

`tools/test_setup_risk.py` (`LowRamGpus`, `StartOnSeveralGpus`, `SmallCard`, `SplitShortCard`),
`tools/test_setup_lowram.py`, `tools/test_setup_unsloth.py` (`LayerSplit`, split tests),
`tools/test_setup_remote_opt.py`, `tools/test_setup_amd.py` (`GpuLists`), and the golden cases
`32GB-2x24GB` (low-RAM, one GPU) and `47GB-2x16GB` (a split). See [tests.md](tests.md).

## In UnbiasedStrata

Removed. The fork runs one GPU only. Its front end refuses `--gpus`, `--layer-split`, `--split-device`,
`--peer-device`, `--peer-slots` and `--remote-expert*` after `--` (`frontend/main.cpp:629-640`). It offers
`--ram-cap GIB` and `--vram-cap MIB` instead of a low-RAM mode question.

## See also

[context-kv-and-options.md](context-kv-and-options.md) · [config-file.md](config-file.md) ·
[MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/MULTI_GPU.md) ·
[SECOND_GPU.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/SECOND_GPU.md)
