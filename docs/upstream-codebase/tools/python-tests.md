[Upstream docs](../README.md) › [Tools](README.md) › Python tests

# Python tests in `tools/`

## Purpose

`tools/test_*.py` are `unittest` suites that run without a GPU, a model download or a network: they use synthetic
GGUFs, mocked HTTP, stand-in engines and evidence fixtures. Upstream has no CI workflow that runs them (`.github/`
holds only `FUNDING.yml`); they are run by hand. This page lists the suites for the tools in this area; the
installer's `test_setup_*.py` and `test_setup_golden.json` are on [../install/README.md](../install/README.md), and
`test_strata_mcp.py` on [../serve/README.md](../serve/README.md).

## Suites

| Suite | Lines | Tests | Needs | Covers |
| --- | ---: | ---: | --- | --- |
| [tools/test_iq_pack.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_iq_pack.py) | 528 | 23 | numpy, gguf-py | `iq_pack.py`: conversions, refusals, split shards, v3/v4 expert maps, interrupted packs, `experts.bin` reuse ([pack-builders.md](pack-builders.md#tests)) |
| [tools/test_shards.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_shards.py) | 119 | 7 | numpy, `setup.py` | setup's `check_shards` and `strata_pack.verify`'s source-hash check |
| [tools/test_make_profile.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_make_profile.py) | 104 | 7 | - | `rank_profile` orders and an end-to-end `main` |
| [tools/test_draft_vocab.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_draft_vocab.py) | 47 | 4 | `regex` (imports the tokenizer) | corpus coverage |
| [tools/test_mtp_fetch.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_mtp_fetch.py) | 201 | 8 | `setup.py` for one test | range checks, sha256 refetch, inventory revision, `verify`, setup's `mtp_corrupt` |
| [tools/test_calibrate.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_calibrate.py) | 234 | 14 | `setup.py`, `serve/server.py` | the calibration logic with a fake engine, and setup's use of it |
| [tools/test_conversation_cache_parity.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_conversation_cache_parity.py) | 170 | 12 | run from the repo root | the parity, pressure, oversized, exchange and admission verdicts |
| [tools/test_conversation_cache_soak.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_conversation_cache_soak.py) | 73 | 4 | " | the soak verdict and answer normalization |
| [tools/test_conversation_cache_disabled.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_conversation_cache_disabled.py) | 104 | 6 | " | the cache-disabled comparison |
| [tools/test_conversation_cache_growth.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_conversation_cache_growth.py) | 79 | 4 | " | full vs incremental capture verdict |
| [tools/test_conversation_cache_isolation.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_conversation_cache_isolation.py) | 82 | 4 | ", numpy | isolation verdict and the generated control-vector / `.sve` fixtures |
| [tools/test_conversation_cache_http_smoke.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_conversation_cache_http_smoke.py) | 71 | 4 | " | the HTTP smoke orchestration against a fake client |
| [tools/test_conversation_restore_speed.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/tools/test_conversation_restore_speed.py) | 34 | 2 | - | `override` and `compare` |

The test counts are the `def test_` methods in each file.

## Running them

From the repository root, with the `.venv` setup creates (numpy, regex and the rest pinned in
`requirements.txt`) and gguf-py available (setup's `third_party/llama.cpp/gguf-py`, or `STRATA_GGUF_PY`):

```
python -m unittest tools.test_make_profile tools.test_draft_vocab tools.test_mtp_fetch tools.test_shards \
                   tools.test_calibrate tools.test_conversation_restore_speed
python -m unittest discover -s tools -p test_iq_pack.py          # as docs/ORCA.md and docs/UNSLOTH_Q4.md give it
python -m unittest tools.test_conversation_cache_parity tools.test_conversation_cache_soak \
                   tools.test_conversation_cache_disabled tools.test_conversation_cache_growth \
                   tools.test_conversation_cache_isolation tools.test_conversation_cache_http_smoke
```

The conversation-cache suites import `tools.<module>` as a namespace package (there is no `tools/__init__.py`), so
they must run from the repository root. Several dry-run tests launch the gate scripts as subprocesses.

## Related checks outside `tools/`

| Check | What |
| --- | --- |
| CTest `iq_parity_fixtures` + `iq_parity` | `tools/iq_fixture.py` writes fixtures into the build directory, `src/kernels/iq_parity.cpp` compares the kernels (`CMakeLists.txt:593-630`) |
| CTest `expert_layout_test` | golden `native_experts.txt` files in `tests/data/native_experts/` |
| CTest `expert_profile_save_test` | the engine's profile writer against `make_profile.py`'s format |
| `serve/test_server.py`, `serve/test_detok.py` | use `make_profile.write_profile` and `strata_tokenizer` |
| `python tools/gguf_writer.py` | self-check (fails as published, see [gguf-and-tokenizer.md](gguf-and-tokenizer.md#gotchas)) |
| `python tools/strata_tokenizer.py --check`, `tools/canonical_xcheck.py [--full]`, `tools/strata_pack.py verify` | checks against a real model shard |

## Gotchas

- `test_shards.py`, `test_calibrate.py` and one test in `test_mtp_fetch.py` import `setup.py`; `test_calibrate.py`
  also needs `serve/` importable. They are not standalone tool tests.
- `test_iq_pack.py` calls `_paths.add_gguf_py()` at import, so a checkout without gguf-py exits at once with
  "llama.cpp's gguf-py was not found".

## In UnbiasedStrata

Kept (restored in c00942d after removal in 8894a81): `test_iq_pack.py` and `test_make_profile.py` (both extended for
Qwen3.6 in e41dfc7), `test_draft_vocab.py` and `test_mtp_fetch.py` (unchanged; the latter's setup test imports the
removed `setup.py`). Removed: `test_shards.py`, `test_calibrate.py`, all `test_conversation_*` suites. The fork adds
`test_qwen36_oracle.py`. See the fork's [python-tests.md](../../codebase/tools/python-tests.md).

## See also

- [README.md](README.md), [pack-builders.md](pack-builders.md), [cache-and-batch-gates.md](cache-and-batch-gates.md)
