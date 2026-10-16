[Upstream docs](../README.md) › [Install](README.md) › Questions, risks and stops

# Questions, risk prompts and hard stops

## Purpose

Setup follows a rule its comments call "the owner's rule": setup recommends, it never forces (#403 #406 #364 #384).
This page lists every question setup can ask, what `--yes` does with each, which risky choices are kept with only a
warning, which ones need consent, and which conditions stop setup no matter what.

Files covered: [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py)
(`ask` `setup.py:256-269`, `confirm_risk` `setup.py:2474-2483`, `confirm_paging` `setup.py:2486-2502`, and the call
sites below). The module docstring states the rule (`setup.py:34-37`).

## The mechanics

- `ask(question, choices, default, yes)`: with `yes` it returns the default without reading input. Otherwise Enter
  means the default, an invalid answer asks again, and EOF stops setup ("input ended before a setup answer was
  received ... pass --yes"). A closed stdin can therefore never consent to anything, for example the build-tools
  install (#393).
- `confirm_risk(msg, explicit, yes, stop, hint, question, default="n")`: prints the risk as a warning, then:
  - with `yes` and `explicit` (the user named the risky choice with a flag such as `--model` or `--gpus`), it goes on;
  - otherwise it asks `question` (default n; with `yes` alone that default is taken), and a "no" calls
    `fail(stop, hint)`.
  So `--yes` alone is not consent. An explicit flag plus `--yes` is.
- Adopting an earlier install sets `a.yes = True` ([flow.md](flow.md)).

## Every question

| Question | Where | Default (`--yes`) | Skipped when |
| --- | --- | --- | --- |
| "Which one?" (installed models + "install another") | `setup.py:3709` | 1 | one config |
| "Tune which one?" | `setup.py:3690` | 1 | one config |
| "Which cards?" (NVIDIA or AMD) | `setup.py:3737` | 1 NVIDIA | `--backend`, or only one kind usable |
| "Which GPUs?" | `setup.py:797` | the best pair, or the first card when `split_short` | `--gpu`/`--gpus`, fewer than two usable together |
| "Which model?" | `setup.py:3865` | 1 (`qwen`) | `--family` |
| "Which size?" | `setup.py:3892` | IQ3_XXS from 60 GB, else item 1 | `--model` |
| "<model>: which GPUs?" (budget model split) | `setup.py:2466` | 1 one GPU | `--yes`, `--gpus`, `--resident-budget-gib`, RAM short |
| "Low-RAM mode: which GPUs?" | `setup.py:2426` | 1 one GPU | `--yes`, `--gpus`, `--low-ram resident` |
| "Context?" | `setup.py:3956` | by VRAM ([context-kv-and-options.md](context-kv-and-options.md)) | `--context` |
| "RoPE extension method?" | `setup.py:3975` | yarn | context <= 262144, `--rope-scaling`, `--yes` (yarn) |
| "KV cache?" | `setup.py:3993` | 1 (8-bit) | context <= 8192, `--kv`, `--yes` |
| "Do you want images?" | `setup.py:4008` | n | `--vision`, a model without images, AMD |
| "Turn on the experimental speed projection?" | `setup.py:4052` | n | `--experimental-speed-projection`, not qwen/coder |
| "Install them now?" (build tools) | `setup.py:2164` | y | nothing missing |
| "Tune Strata for this PC now?" | `setup.py:4387-4389` | (not asked with `--yes`) | `--yes`, `--no-start`, AMD, a saved calibration |
| "Use both from now on?" (at a start) | `setup.py:921-922` | y, or n for a resident low-RAM config, a `split_short` pair or a budget config | `gpus_asked`, a list already, fewer than two cards |

## Kept, with a warning (no question)

| Choice | What is said | Where |
| --- | --- | --- |
| A context above the RAM rule (past 128K with IQ3_XXS/IQ3_S) | "needs ~N GB of RAM by setup's estimate ... Kept as you chose" | `setup.py:3960-3965` |
| `--resident-budget-gib` above the recommendation | "Kept as you chose: the engine clamps it to the RAM it finds free at start (less 4 GB)" | `setup.py:1225-1229` |
| `--kv-streaming on` without the RAM | "Kept as you chose (--kv-streaming on): it may page or run out of RAM" | `setup.py:4275-4278` |
| A budget above the KV-reduced one | "kept as you chose" | `setup.py:4284-4286` |
| `--vram-reserve-mib` below what the GPU encoder needs | "kept as you chose (it may run out of VRAM when it reads a picture)" | `setup.py:4304-4306` |
| `--parallel` above the recommendation | "kept as you chose" | `parallel_note()`, `setup.py:3315-3334` |
| `--gpus` with the low-RAM mode | "as you chose (--gpus): ... RAM can fill up to 0 free" | `setup.py:2410-2417` |
| `--cuda 13` with a Pascal/Volta card, `--cuda 12` on newer cards | the reason from `cuda_choice()` | `setup.py:3788-3790` |

## Needs consent (`confirm_risk`): an explicit flag + `--yes`, or "y"

| Risk | Explicit flag that consents with `--yes` | Where |
| --- | --- | --- |
| RAM below 28 GB and the low-RAM mode cannot help | `--model` | `setup.py:3803-3814` |
| The chosen size pages (`confirm_paging`; "Install it anyway?", default n, but y with `--low-ram off`) | `--model`, or `--low-ram off` | `setup.py:2486-2502`, call `setup.py:3933-3934` |
| UD-Q4_K_XL on AMD ("Try it anyway?") | `--model` | `setup.py:3901-3908` |
| A budget model with RAM below 48 GB ("Install it anyway?") | `--model` | `setup.py:3909-3916` |
| A named card with too little VRAM to share a model ("Use it anyway?") | `--gpus` (naming it) | `check_gpus(named=True)`, `setup.py:705-712` |

`tools/test_setup_golden.py::test_the_refusals_with_yes_alone_stay` is meant to check that a size the baseline
refused with `--yes` and no `--model` still stops. In this snapshot's baseline the only refused rows are
`32GB-2x24GB unsloth UD-Q4_K_XL` and `47GB-2x16GB unsloth UD-Q4_K_XL`. Both name `--model`, so the test checks
nothing. The current code installs both: the explicit `--model` is consent, giving an 8 GiB and a 23 GiB budget when
run through the harness here. The `test_yes` cases skip rows whose baseline code is not 0.

`tools/test_setup_risk.py` (`RamFloor`, `ConfirmRisk`) covers the `--yes`-alone stop directly.

## Hard stops (`fail`)

None of these can be overridden by a flag. The fix is to change the PC or the choice:

- No NVIDIA GPU from `nvidia-smi` on the NVIDIA path, or no supported AMD card on the HIP path.
- NVIDIA driver below 580, or below 528/525 for CUDA 12.
- A chosen card missing or unusable (when not only short of VRAM), malformed `--gpu`/`--gpus`.
- A CPU without SSE4.2 + POPCNT. A CPU without AVX2 with an AMD card on Windows.
- `--gpus` with AMD on Windows. Cards from two ROCm wheel families without a system ROCm 7.
- `--model` that the family does not have. `--gguf-dir` holding an unsupported quant or another model's files.
- Not enough free disk. A short or corrupt shard. A wrong SHA-256 (the file is deleted). An unfinished download.
- A budget model with an engine older than it needs. No ready-made Windows HIP engine.
- `--rope-scaling none` past 262144, or `--rope-scale` without a method inside it.
- A missing speed-projection vector. A model with no `per_layer_token_embd` tensor.
- Build tools declined, or not installed (non-apt Linux, non-Ubuntu-22.04/24.04 CUDA, missing winget).
- At a start: a config pointing at missing files ("run it again with --setup to repair"). `--gpus` with a budget
  config when RAM is short (`split_budget`). An AMD `--gpus` card the engine has no code for.
- Any command run with `run()` that fails.

Every `fail` ends with "Setup stopped. Fix the item above and run it again - everything already done is kept and
skipped." and exit code 1. The launchers then pause (`START-HERE.bat:41`).

## Tests

`tools/test_setup_risk.py` (`ConfirmRisk`, `Context`, `LowRamGpus`, `SmallCard`, `SplitShortCard`, `RamFloor`,
`KvStreaming`), `tools/test_setup_prompts.py` (EOF, `--yes`, re-prompting, no installer on EOF),
`tools/test_setup_lowram.py` (`PagingQuestion`), and `tools/test_setup_golden.py` (`test_enter_for_every_question`:
Enter gives the same configs as `--yes`).

## In UnbiasedStrata

Removed. The fork asks nothing. Its front end exits with code 2 and a one-line message when a file is missing or an
option is refused (`frontend/main.cpp:629-673`).

## See also

[flow.md](flow.md) · [cli-flags.md](cli-flags.md) · [tests.md](tests.md)
