[Upstream docs](../README.md) › [Tools](README.md) › Shipped data files

# Shipped data files (`data/`)

## Purpose

`data/` holds the small binary inputs the engine needs that do not come from a model download: the expert-cache
rankings, the draft-vocabulary subsets, and the experimental control vector. They were produced by the tools on
[profile-and-draft-vocab.md](profile-and-draft-vocab.md) (the control vector by a third party) and are copied or
referenced by setup when it writes a model's run config (see [../install/README.md](../install/README.md)).

## Inventory

All facts below were read from the files at commit 6f32ec0.

| File | Bytes | Content | Producer | Consumer |
| --- | ---: | --- | --- | --- |
| [data/expert-profile.bin](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/data/expert-profile.bin) | 196,632 | `STRP` v1, 48 layers x 512 experts, 24,576 pairs ranked | `tools/make_profile.py` | engine `--expert-profile` for the original model, Swift 1.5 and the Unsloth files (`setup.py:4225`) |
| [data/expert-profile-coder.bin](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/data/expert-profile-coder.bin) | 98,328 | `STRP` v1, 48 x 256, 12,288 pairs ranked | `tools/make_profile.py --n-expert 256` | the Coder (family key `profile`, `setup.py:205`) |
| [data/draft_vocab.bin](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/data/draft_vocab.bin) | 425,196 | 106,299 int32 ids: English/code plus CJK | `tools/draft_vocab.py --add cjk` | setup `--draft-vocab cjk` (default) |
| [data/draft_vocab_en.bin](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/data/draft_vocab_en.bin) | 162,100 | 40,525 ids: English/code | not determined from source | `--draft-vocab en` |
| [data/draft_vocab_cyrillic.bin](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/data/draft_vocab_cyrillic.bin) | 235,852 | 58,963 ids: English/code plus Cyrillic | `tools/draft_vocab.py --add cyrillic` | `--draft-vocab cyrillic` |
| [data/draft_vocab_fr.bin](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/data/draft_vocab_fr.bin) | 184,844 | 46,211 ids: English/code plus French corpus tokens | `tools/draft_vocab.py --corpus` (#597) | `--draft-vocab fr` (0.1.39) |
| [data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf) | 483,520 | GGUF `controlvector`, `model_hint` `qwen4exp`, `layer_count` 47, F32 tensors `direction.1` .. `direction.47` of 2,560 values | third party (see its README) | engine `--control-vector-scaled` when setup's `--experimental-speed-projection on` |
| [data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.json](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.json) | 797 | the package's selected configuration and scores (mode `project`, `per-layer`, layers 4-44, scale 1.0) | third party | documentation only |
| [data/experimental-speed-projection/README.md](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/data/experimental-speed-projection/README.md) | 1,755 | what the vector does, how to turn it on, origin and license | - | - |

Every draft subset starts with ids 0..4 and its largest id is 248,319 (the last token of the 248,320-token
vocabulary); no subset has duplicates. `draft_vocab.bin`, `draft_vocab_cyrillic.bin` and `draft_vocab_fr.bin` each
begin with exactly the 40,525 ids of `draft_vocab_en.bin`, followed by their additions in ascending id order - the
order `tools/draft_vocab.py` writes with `--base data/draft_vocab_en.bin` (checked on the files; the producer
column above follows the tool's docstring and DETAILS.md, the exact commands are not recorded).

## How they reach the engine

```
setup.py step 6/7
  data/draft_vocab*.bin ──refresh_draft_vocab()──► <Strata-data>/mtp/rt/draft_vocab.bin ──► src/core/mtp.cpp
  data/expert-profile[-coder].bin ──"--expert-profile" in strata-<model>.json──► src/core/expert_cache.cpp
  data/experimental-speed-projection/*.gguf ──"--control-vector-scaled F:1.0 --control-vector-layer-range 4 44
                                              --cvec-mode project --cvec-dir per-layer"──► the cvec kernels
```

- `refresh_draft_vocab` (`setup.py:3366-3385`) copies the chosen subset when the destination is missing or holds a
  different shipped subset (recognized by sha256, including one older subset in `OLD_DRAFT_VOCABS`,
  `setup.py:3199`); a subset made by hand is kept. A setup run without `--draft-vocab` keeps the subset the model's
  config chose before (`setup.py:4209-4211`).
- The profile is referenced in place, not copied (`setup.py:4225`).
- The control vector is referenced in place; the flags are written only for the original model, not Swift 1.5 or
  other models (`setup.py:4048-4060`, `4320-4323`). The server turns it off per request with
  `"experimental_speed_projection": false` (the engine key `cvec=0`, `serve/server.py:702-706`).

## Experimental speed projection

From its README and [docs/DETAILS.md](../../upstream/DETAILS.md) ("Experimental speed projection"): when loaded,
the engine removes each layer's direction from every hyper-connection stream after layers 4-44 (`h -= (h . v) v`).
The package describes it as a refusal-direction projection: the model declines far fewer requests (it reports 1 of
50 vs 50 of 50 on its test set), which removes a safety behaviour. Upstream measured on Q2_0 with fixed experts over
2,557 teacher-forced tokens: top-1 token changed at 10% of positions, mean KL 0.063 nats (max 4.1), perplexity +15%
on code, +2.3% on a document, +0.4% on a chat; 0.2-0.4% cost per token. Off by default. The package's JSON names
its base as Unsloth's UD-Q2_K_XL and a llama.cpp fork branch `cvec-hooks`. License: the Qwen Community License 1.0
of the model applies (README).

## Gotchas

- The profiles are tied to 48 layers and 512 or 256 experts; the engine refuses another shape
  (`src/core/expert_cache.cpp:42-49`).
- The draft subsets are token ids of this model's vocabulary; nothing in the file says which vocabulary.
- `tools/make_profile.py`'s default `--out` is `data/expert-profile.bin`, so a bare run overwrites the shipped
  ranking.

## In UnbiasedStrata

The fork keeps `draft_vocab.bin`, `draft_vocab_en.bin` and `expert-profile.bin` byte-identical to upstream, removes
`draft_vocab_cyrillic.bin`, `draft_vocab_fr.bin`, `expert-profile-coder.bin` and the whole
`experimental-speed-projection/` folder (commit 8894a81), and adds `data/qwen36-expert-profile.bin` for Qwen3.6
(40 x 256, e41dfc7). See the fork's [data-files.md](../../codebase/guides/data-files.md).

## See also

- [profile-and-draft-vocab.md](profile-and-draft-vocab.md), [pack-format.md](pack-format.md)
- [../install/README.md](../install/README.md), [../serve/README.md](../serve/README.md)
