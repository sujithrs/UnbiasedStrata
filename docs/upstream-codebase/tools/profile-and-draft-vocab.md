[Upstream docs](../README.md) › [Tools](README.md) › Expert profile and draft vocabulary

# Expert profile and draft vocabulary builders

## Purpose

Two tools that build small ranking/selection files the engine reads at start:

- `tools/make_profile.py` writes `expert-profile.bin`, a ranking of every (layer, expert) pair. With
  `--expert-cache auto`, the engine fills the VRAM expert slots it can afford with the profile's pairs in order, so
  the ranking decides which experts start resident; the adaptive tier then swaps in what a conversation routes most
  (`tools/make_profile.py:3-6`). Setup passes `--expert-profile data/expert-profile.bin` (or the Coder's file)
  (`setup.py:4225`).
- `tools/draft_vocab.py` writes `draft_vocab.bin`, the subset of the vocabulary the MTP draft head may propose. The
  draft logits are computed over those rows only: a smaller subset is faster and takes less VRAM, but a token
  outside it can never be drafted (`tools/draft_vocab.py:3-6`).

| File | Lines | Role |
| --- | ---: | --- |
| [tools/make_profile.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/make_profile.py) | 119 | Expert-cache ranking from a base profile and routing traces |
| [tools/draft_vocab.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/draft_vocab.py) | 146 | Extends a draft-vocabulary subset by script or by corpus coverage |
| [tools/test_make_profile.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_make_profile.py) | 104 | Tests of the ranking and the file round trip |
| [tools/test_draft_vocab.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_draft_vocab.py) | 47 | Tests of corpus coverage |

The files they produced ship in `data/`: see [data-files.md](data-files.md). Formats:
[pack-format.md](pack-format.md#4-expert-cache-profile-expert-profilebin-toolsmake_profilepy) and
[pack-format.md](pack-format.md#5-draft-vocabulary-subset-draft_vocabbin).

## `tools/make_profile.py`

```
python tools/make_profile.py [TRACE ...] [--base data/expert-profile.bin | --no-base] [--reorder]
                             [--out PATH] [--n-expert 256]
```

| Function | Lines | What it does |
| --- | --- | --- |
| `read_profile(path, n_expert)` | 30-37 | checks `STRP` and that the header is `N_LAYER` x `n_expert`; returns the ranked pairs |
| `read_trace(path, n_expert)` | 40-51 | counts (layer, expert) occurrences in a `--dump-routing` trace (`i32 layer, i32 k`, k ids, k weights) |
| `write_profile(path, ranked, n_expert)` | 54-63 | header, the ranked pairs, then each layer's slot table (-1 = unranked) |
| `rank_profile(base, freq, n_expert, no_base, reorder)` | 66-90 | the order (below) |
| `main` | 93-115 | sums the traces, ranks, writes, asserts the round trip |

The order (`rank_profile`): by default the base profile's pairs, then the traces' pairs most frequent first (ties:
lower layer, then lower expert), then every missing pair interleaved across layers (expert-major, layer-minor).
`--reorder` puts the traces' pairs before the base's; `--no-base` drops the base. A base is never reordered against
itself. Because the shipped profile already ranks all 24,576 pairs, a trace changes nothing without `--reorder`
(0.1.39, #587, `make_profile.py:8-12`; [docs/DETAILS.md](../../upstream/DETAILS.md), "Other GPUs").

To get a trace: run the engine once with `--dump-routing FILE` on a prompt typical of the use
(`src/program/generate.cpp:610`; it needs the expert pool, `generate.cpp:3784-3790`). The serve path writes the
same records with zero weights, which `read_trace` ignores anyway (it counts frequencies). The engine itself can save
what its adaptive tier learned in this format with `--expert-profile-save` (#477, `generate.cpp:663-665`).

## `tools/draft_vocab.py`

```
python tools/draft_vocab.py --gguf <model shard 1> [--base data/draft_vocab.bin] [--add han,kana,hangul,cjk_punct,cjk,cyrillic]
                            [--corpus f1.txt ...] [--coverage 0.99] [--out PATH] [--stats]
```

- Tokenizes with `strata_tokenizer.Tokenizer.from_gguf` (the vocabulary comes from the GGUF) and classifies each
  token's text by Unicode script (`SCRIPTS`, `draft_vocab.py:35-43`; a token that is not whole UTF-8 characters
  belongs to no script).
- `--add`: adds every token of the named scripts; `cjk` = han + kana + hangul + cjk_punct.
- `--corpus` (#597): tokenizes the files as plain text and adds the most frequent tokens that together cover
  `--coverage` of the occurrences (`covering`, `draft_vocab.py:72-81`; ties: the lower id first), then prints the
  share of the corpus outside the base and outside the new subset (`outside`, `draft_vocab.py:84-88`).
- The base ids keep their order; added ids follow in id order (`draft_vocab.py:121-122`). Base ids outside the
  vocabulary stop the tool.
- Without `--out` (or with `--stats`) it prints per-script counts for the base and the new subset.

How the shipped subsets relate (from the docstring's examples and DETAILS.md, "The draft layer's tokens"):
`draft_vocab.bin` is the English/code subset with the CJK scripts added (#137, the docstring's `--add cjk` example);
`draft_vocab_cyrillic.bin` is the English/code subset plus the Cyrillic script; `draft_vocab_fr.bin` is
`draft_vocab_en.bin` plus the tokens covering 99% of a French Wikipedia corpus (#597). The exact commands that
produced the shipped files are not recorded beyond these examples. Measured by upstream (DETAILS.md, "The draft layer's tokens"): with the CJK subset, answers in
those languages 15-38% faster (Q2_0, RTX 5070); with the French subset, drafts accepted in French answers 0.51 ->
0.60 (IQ3_XXS, RTX 5070, 8 prompts x 2 passes) and 141 -> 158 tok/s on an RTX 5090 (IQ3_S, reporter's
measurement).

Setup copies the chosen subset into `<mtp>/rt/draft_vocab.bin` (`refresh_draft_vocab`, `setup.py:3366-3385`;
`--draft-vocab cjk|en|cyrillic|fr`, default `cjk`), replacing a shipped subset but keeping one made by hand, and
suggests `en` on cards under 14 GB (`draft_vocab_note`, `setup.py:3245-3257`). Its VRAM estimates per subset for the draft head:
cjk 348, cyrillic 193, fr 151, en 133 MiB, "IQ3_S: the largest" (`DRAFT_VOCAB_MIB`, `setup.py:3241`).

## Configuration

| Setting | Effect |
| --- | --- |
| engine `--expert-profile FILE` | the ranking `--expert-cache auto` follows |
| engine `--dump-routing FILE` | write a routing trace |
| engine `--expert-profile-save FILE` | write the learned ranking in this format |
| setup `--draft-vocab cjk|en|cyrillic|fr` | which shipped subset goes into `rt/` |

## Model-specific behaviour

- `N_LAYER = 48` is fixed (`make_profile.py:26`); `--n-expert` changes the experts per layer: 512 for the original
  model and its fine-tunes, 256 for the Coder (`data/expert-profile-coder.bin`, chosen by the family's `profile`
  key, `setup.py:205`). The engine refuses a profile of another shape (`src/core/expert_cache.cpp:42-49`).
- The draft subsets are token ids of the Flash-Next vocabulary (248,320 tokens); every shipped file's largest id is
  248,319.

## Tests

- `python -m unittest tools.test_make_profile`: a complete base hides the trace, `--reorder` puts the trace first,
  frequency and tie order, the default only appends, `--no-base`, reorder without traces equals the default, and an
  end-to-end run of `main`. Uses a 4-expert geometry with the fixed 48 layers.
- `python -m unittest tools.test_draft_vocab`: coverage order, ties, the outside share, and reading every corpus
  file (no model needed).
- `serve/test_server.py` writes profiles with `make_profile.write_profile` (patching `N_LAYER`), and the CTest
  `expert_profile_save_test` compares the engine's writer with this format byte for byte.

## Gotchas

- `--out` defaults to `data/expert-profile.bin`, so a bare run overwrites the shipped profile (`make_profile.py:101`).
- `--base` defaults to the 512-expert shipped profile even with `--n-expert 256`; for the Coder pass
  `--base data/expert-profile-coder.bin` or `--no-base`, or `read_profile` exits on the shape mismatch.
- `draft_vocab.py` reads the whole vocabulary with the Python tokenizer and needs `regex`.

## In UnbiasedStrata

Both tools and their tests were removed in fork commit 8894a81 and restored in c00942d. `draft_vocab.py` and
`test_draft_vocab.py` are unchanged at the fork's HEAD. `make_profile.py` was changed for Qwen3.6 (40 layers x 256
experts): `--n-layer`, `--text-trace` for llama.cpp route traces (e41dfc7), and `--out` made required so a bare run
cannot overwrite the pinned profile (e6026d4). See the fork's
[profile-and-draft-vocab.md](../../codebase/tools/profile-and-draft-vocab.md).

## See also

- [data-files.md](data-files.md), [pack-format.md](pack-format.md), [mtp-tools.md](mtp-tools.md)
- [Upstream docs/DETAILS.md](../../upstream/DETAILS.md)
