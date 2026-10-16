[Upstream docs](../README.md) › [Tools](README.md) › Calibration and benchmarks

# Calibration (`calibrate.py`) and the needle benchmark (`needle_bench.py`)

## Purpose

Two tools that run the real model to measure something:

- `tools/calibrate.py` tunes three hardware-dependent engine settings on the user's PC. It is setup's
  `--calibrate` (`setup.py:2936-2969`) and can be run alone on a run config.
- `tools/needle_bench.py` measures long-context recall through a running server's OpenAI-compatible API.

| File | Lines | Role |
| --- | ---: | --- |
| [tools/calibrate.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/calibrate.py) | 248 | Measures `--pcie-frac`, `--spec-min-p`, `--pool-workers` and keeps a value only when it beats the default by more than 3% |
| [tools/test_calibrate.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_calibrate.py) | 234 | Tests with a stand-in engine whose speed is a function of the settings |
| [tools/needle_bench.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/needle_bench.py) | 118 | Needle-in-a-haystack recall at chosen lengths and depths |

## `tools/calibrate.py`

### What it tunes

From its docstring (`calibrate.py:3-12`) and DETAILS.md ("Tuning for your PC"):

| Setting | Meaning | Candidates |
| --- | --- | --- |
| `--pcie-frac` | share of the experts missing from VRAM that are copied over PCIe and run on the GPU instead of the CPU | 0.0, 0.2, 0.35, 0.55, 0.75 plus the engine's default (`PCIE_FRACS`, `calibrate.py:33`) |
| `--spec-min-p` | how sure the draft layer must be to extend a verify window by another guess | 0.3, 0.5, 0.7 plus the default (`SPEC_MIN_PS`, `calibrate.py:34`) |
| `--pool-workers` | CPU threads computing experts | the engine's count, two thirds and half of it, at least 2 (`worker_candidates`, `calibrate.py:63-69`) |

The defaults were measured on a Ryzen 5 7600 with an RTX 5070 on PCIe 5 (`calibrate.py:4`). Decode speed only.

### How it measures (`run` and `measure`, `calibrate.py:105-215`)

```
run(cfg): tokenizer from cfg["tokenizer"] (vocab.json, merges.txt, token_type.json)
          3 fixed prompts, chat-templated with thinking off
          engine args = serve.server.engine_args(cfg)     (#447: exactly what the server starts)
measure:  start ONE engine with the product defaults (apply(args, {}))
          read its defaults from engine.info (pcie_frac, spec_min_p, pool_workers)
          warm up 2 rounds
          1. sweep pcie_frac at the default spec_min_p       } per-request "strata_tune" keys,
          2. sweep spec_min_p at the best pcie_frac          } no restart
          3. winner vs default, interleaved, 3 times each -> pick() keeps it only if > MIN_GAIN = 3% better
          close the engine (QUIT)
          4. for each worker count: restart the engine with --pool-workers, 1 warm-up, 2 measurements; pick()
returns {"settings": {flag: value}, "report": {...sweeps, confirm, workers, seconds, tok_s}}
```

Each measurement is the median over the three prompts of `generated / decode_ms` for 128 greedy tokens
(`Session.rate`, `calibrate.py:88-98`). The per-request tuning works because the server turns
`sampling["strata_tune"]` into `pcie_frac=` and `spec_min_p=` keys on the engine's GEN line (values in [0, 1],
`serve/server.py:692-699`).

`apply(args, settings)` (`calibrate.py:234-241`) writes the chosen values into the config's engine arguments and
puts every untouched setting back to the product default (`DEFAULTS`: `--spec-min-p 0.5`, the engine's own PCIe
share and worker count), so an older calibration's values never linger. `engine_error(log, since)` returns the
engine's own last message from its log for setup to show when a calibration fails (#447).

### Use

```
python tools/calibrate.py strata-<model>.json      # prints {"settings", "report"}; does not write the config
./setup.sh --calibrate   |   START-HERE.bat --calibrate   # measures, writes the config, remembers it per PC and model
```

Setup stores the result under `calibration` in its settings file keyed by `hardware_key(cfg)` (GPU, CPU, RAM, model,
context, images setting) and reapplies it on a reinstall (`setup.py:2955-2962`, `2972-2986`). Setup offers the
tuning itself on NVIDIA only; a saved calibration is reapplied on Linux HIP too (#566), not on Windows HIP
(`setup_calibration`, `setup.py:2977-2986`).

## `tools/needle_bench.py`

```
python tools/needle_bench.py [--url http://127.0.0.1:8080] [--api-key KEY] [--lengths 32k,128k,262k]
                             [--depths 10,50,90] [--timeout 1800] [--out results.json]
```

- The haystack is the repository's own text (`docs/*.md`, `src/*.cpp`, `src/*.cu`, `include/*.hpp`, `serve/*.py`,
  `tools/*.py`, and llama.cpp's docs under `third_party/`), in a fixed order, repeated as needed, cut by characters
  at 3.2 characters per token, with lengths 2% under target so "128k" fits a 128K context (`needle_bench.py:25-51`,
  `85-86`).
- The needle is `The secret code word for this text is: <word>-<word>-<NNN>.` inserted at a line start at each
  depth, words from a fixed list with `random.Random(7)` (`needle_bench.py:82-95`).
- Each request goes to `/v1/chat/completions` with `max_tokens` 40, temperature 0, thinking off
  (`chat_template_kwargs.enable_thinking = false`). A length the server's context (from `/metrics`
  `engine.max_context`) cannot hold is skipped.
- Prints one line per test (FOUND or missed, prompt tokens, seconds), exits 0 only when every test found its needle.

Upstream results produced with it: `bench/results/2026-09-28-rope-scaling/`, `2026-10-03-v100-prompt-attn/`, and
two community runs (RTX 5090, 2x MI50) which "found all six needles"
([docs/COMMUNITY_BENCHMARKS.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/COMMUNITY_BENCHMARKS.md)).

## Configuration

| Input | Used by |
| --- | --- |
| `strata-<model>.json` keys `exe`, `args`, `cwd`, `log`, `tokenizer`, `gpu`, `lib_dirs` (written by setup, `setup.py:4323-4325`) | `calibrate.py` (through `serve.server.StrataEngine` and `engine_args`) |
| `--url`, `--api-key` | `needle_bench.py` |

## Platform notes

- Calibration measures through `serve.server.StrataEngine`, so it works wherever the server does; setup only
  offers it on NVIDIA (see above).
- Both tools need the model loaded: the GPU is busy for 5-10 minutes during calibration (setup's message,
  `setup.py:2942-2943`).

## Tests

`python -m unittest tools.test_calibrate` (no GPU): a `FakeEngine` whose decode speed is a function of
`(pcie_frac, spec_min_p, workers)`. Covers defaults kept when the speed is flat, a small gain treated as noise,
finding the PCIe share and draft floor, fewer workers, an old calibration reset to the baseline, `apply`, that the
engine arguments are the server's, `engine_error`, `worker_candidates`, `pick`, and setup's `calibrate_config`
writing and remembering the result or keeping the defaults (with the engine's reason) on failure. It imports
`setup.py`. `needle_bench.py` has no test.

## Gotchas

- The calibration's three prompts and 128 tokens are fixed; results are noisy by a few percent, which is why a value
  must win by more than 3% in an interleaved re-measurement (`calibrate.py:14-15`).
- Run alone, `calibrate.py` only prints; it does not change the config.
- `needle_bench.py`'s haystack depends on what is checked out (it reads `third_party/llama.cpp/docs` when present), so
  the same `--lengths` can produce a different text on another install.

## In UnbiasedStrata

Removed in fork commit 8894a81 (with the Python server they depend on) and not restored. The fork's C++ front end has
no per-request tuning keys.

## See also

- [cache-and-batch-gates.md](cache-and-batch-gates.md) (the other scripts that drive a real engine)
- [../serve/README.md](../serve/README.md) (`StrataEngine`, the GEN line, `/metrics`), [../install/README.md](../install/README.md) (setup's `--calibrate`)
