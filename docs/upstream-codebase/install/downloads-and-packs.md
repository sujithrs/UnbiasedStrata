[Upstream docs](../README.md) › [Install](README.md) › Downloads and packs

# Downloads, checks, and building the pack and the draft head

## Purpose

This page covers steps 5 and 6. Step 5 fetches the model's GGUF shards and the vision encoder, resuming where it
stopped, from pinned Hugging Face revisions, and checks them. Step 6 runs the pack tools that turn the GGUF into
Strata's pack and builds the MTP draft head from the original Qwen checkpoint.

Files covered: [setup.py](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/setup.py)
(`setup.py:62-94`, `setup.py:153-173`, `setup.py:1019-1252`, `setup.py:3355-3385`, `setup.py:4061-4214`). The tools
it runs are covered in [tools](../tools/README.md).

## Sources and pinning (#214)

| What | Where |
| --- | --- |
| Model shards | `hf(repo)` gives `<endpoint>/<repo>/resolve/<pinned sha>/`, with the family's `<size>/` subfolder (`setup.py:82-84`). Pinned revisions (`HF_REVISIONS`, `setup.py:65-70`): ISTA-DASLab GSQ-RCO `ed59f920...` (2026-09-29), Swift 1.5 `b22d729e...` (2026-09-24), Coder `5348543e...` (2026-09-29), Unsloth `38bb39ee...` (2026-09-30) |
| Mirror | `HF_ENDPOINT` (for example `https://hf-mirror.com`) replaces `https://huggingface.co`. The pins and checks are the same (`hf_endpoint()`, `setup.py:76-79`, #495) |
| A pinned revision that is gone | A 404 on the HEAD request retries `.../resolve/main/` with a warning (`hf_unpinned()`, `setup.py:87-89`; `setup.py:1049-1053`) |
| llama.cpp | `https://github.com/ggml-org/llama.cpp/archive/3cf03257....zip` (`setup.py:93-94`). Where GitHub cannot be reached, a tree placed in `third_party/llama.cpp` by hand is used (#585, INSTALL.md) |
| Engine zips | GitHub releases, see [engine.md](engine.md#the-ready-made-engine-nvidia) |
| Python packages | `requirements.txt` pins ([launchers-and-chat.md](launchers-and-chat.md#requirementstxt)) |
| MTP tensors | Fetched by `tools/mtp_fetch.py` from the original Qwen checkpoint (only ~5 GB). It is pinned too and follows `HF_ENDPOINT` (tested in `test_setup_pins.py`) |

## download() (`setup.py:1026-1098`)

- A target with a `.done` mark is skipped without asking the server.
- `file://` URLs and plain paths are copied, which the tests and mirrors use.
- A HEAD request gets `Content-Length`, with up to 5 attempts 5 s apart, and the 404 fallback above. A file that
  already has that size but no mark (finished by an older setup) is just marked.
- Download into `<name>.part` with `Range: bytes=<have>-`. If the server ignores the range (no 206), it starts over.
  It reads 8 MiB at a time and prints a progress line every 2 s. Up to 30 attempts are made, with "download
  interrupted ... retrying in 10 s".
- A size mismatch at the end stops with "(the download resumes where it stopped)". On success `.part` is renamed and
  marked.
- The User-Agent is `strata-setup`.

## Which shards, and where (`setup.py:4061-4088`)

- Normal: `<--models-dir>/<family tag><SIZE>/` (the size's case kept: `models/IQ3_XXS/`, `models/swift-IQ2_XS/`),
  with the family's file names (`setup.py:3936`, `setup.py:4061-4063`). Shard count is 2, 4 for UD-Q4_K_XL and 3 for
  UD-IQ4_XS (`model_file()`, `model_shards()`, `setup.py:817-823`).
- `--gguf-dir DIR` uses the files where they are. `gguf_dir_shards()` (`setup.py:1116-1133`, #305) takes the
  published first-shard name, or else the single `*-00001-of-*.gguf` in the folder (preferring one containing the size
  name), and derives the other names from its `-of-0000N`. `gguf_dir_problem()` (`setup.py:1161-1183`, #444) stops when
  the file is a quant Strata cannot run (Unsloth UD-IQ3_XXS, K-quants; `GGUF_QUANT` / `SUPPORTED_GGUFS`), or when the
  chosen file is missing but other GGUFs are there. In that case it names the `--family/--model` pairs that match.
- Shards already downloaded in a Strata folder on another drive are used in place.
- A whole file copied in by hand without a mark is marked when its size matches its own tensor directory
  (`whole_shard()`, `setup.py:1101-1110`, #173).
- Step 5 prints the folder and the expected file names, so files fetched by hand can be put there (#495).
- **Shard 2 sharing**: for `qwen` and `coder`, shard 2 (the n-gram / PLE table) is the same file for every size and
  the Coder. A finished `Qwen3.8-Flash-Next-GSQ-RCO-*-00002-of-00002.gguf` in another size's folder is hard-linked
  instead of downloaded (`setup.py:4145-4154`).

## Checks

| Check | Lines | What |
| --- | --- | --- |
| `check_shards(shards)` | `setup.py:1233-1251` | Every shard exists, parses (`tools/gguf_reader.GGUFFile`) and is at least `data_start + max(offset + bytes)` long. It stops naming the bytes missing. Only the header is read |
| `verify_sha256(s, size, sha)` | `setup.py:1186-1208` | Unsloth shards only (`UNSLOTH_SHARDS`, `UNSLOTH_IQ4_XS_SHARDS`, `setup.py:155-173`): checks the exact size, then hashes 16 MiB at a time. A wrong hash deletes the file and its mark. A good one stores `sha256 <hex>` in the mark, so the "~5 minutes of hashing 111 GB" happen once |
| `mtp_corrupt(mtp)` | `setup.py:3355-3363` | Runs `tools/mtp_fetch.py verify --out <mtp>`. Exit code 3 means the tensors are not the pinned checkpoint's: a mirror that ignored range requests had left the shards' starts there, and the draft layer "accepted nothing - with no error anywhere" (#327). It forces a re-fetch and rebuild |

The GSQ-RCO files have no SHA-256 table in setup. They are checked by name, size and tensor directory only.

## Disk space (`setup.py:4078-4088`)

```
need = (download_gb - bytes already on disk incl. .part, unless --gguf-dir or complete) + 8
     + 40                 if Q2_0, AVX-512 CPU and family qwen   (the AVX-512 pack)
     + 1                  if images
     + arena_gb + 1       if the low-RAM mode (experts.bin), except the AVX-512 Q2_0 case
```

The free space of the models folder's drive must be at least `need` GB, or setup stops with "use --models-dir on a
bigger drive". A resumed download needs room only for what is still missing (#425).

## The vision encoder

When images are on, the family's `mmproj-*-BF16.gguf` (0.9 GB per the prompt) is taken from `--gguf-dir`, the data
folder or another root, or else downloaded from the family's `mmproj_hf` (`setup.py:4161-4169`). Unsloth uses the
ISTA-DASLab repository's file.

## The pack

The pack goes into `<data>/packs/<tag>`, or an existing one in another root is used. `STRATA_GGUF_PY` points the tools
at `third_party/llama.cpp/gguf-py` (`setup.py:4173-4174`).

| Case | Commands (`setup.py:4175-4193`) | Done when |
| --- | --- | --- |
| Q2_0, AVX-512 CPU, family `qwen` | `tools/strata_pack.py build --gguf <shard1> --out <pack> --skip-hash`, then `tools/pack_index.py --pack <pack>`, then `tools/strata_tokenizer.py --gguf <shard1> --out <pack>` | `index.txt` (written last) and `experts.bin` exist; `tokenizer/vocab.json` exists. The comment calls it "a one-time ~40 GB conversion" with the experts "repacked for the AVX-512 kernel" |
| Every other size | `tools/iq_pack.py --gguf <shard1> --out <pack> [--compat-bf16]`. Unsloth passes `--compat-bf16`, which makes its Q8_0 hyper-connection projections BF16. The experts stay in the GGUF and are read at start | `native_experts.txt` and `tokenizer/vocab.json` exist |
| Low-RAM mode | also `tools/iq_pack.py --gguf <shard1> --out <pack> --experts-bin` ("one time, <arena_gb> GB") | `experts.bin` exists |

## The MTP draft head

The draft layer drives speculative decoding ("~2x faster output", per the step's text). It lives in `<data>/mtp/`
and is shared by every model and family (`setup.py:4195-4212`):

```
tools/mtp_fetch.py fetch --out <mtp>                                   # ~5 GB of MTP tensors, original Qwen checkpoint
tools/mtp_pack.py --src <mtp> --experts q2_0 --out <mtp>/mtp-q2_0.gguf
tools/mtp_rt.py --gguf <mtp>/mtp-q2_0.gguf --out <mtp>/rt              # the engine's --mtp folder
```

It runs when `rt/experts.bin` is missing or `mtp_corrupt()` says so. Then `refresh_draft_vocab(rt, choice)` copies
the chosen `data/draft_vocab*.bin` to `rt/draft_vocab.bin`
([context-kv-and-options.md](context-kv-and-options.md#draft-vocabulary---draft-vocab-cjkencyrillicfr)).

## Idempotence

Every artifact has a "done" signal: `.done` marks for downloads, `index.txt` / `native_experts.txt` /
`tokenizer/vocab.json` / `experts.bin` for packs, and `rt/experts.bin` for the MTP layer. Running setup again redoes
only what is missing. The data folder is shared by every Strata copy
([update-and-data-folder.md](update-and-data-folder.md#the-data-folder)).

## Tests

`tools/test_setup_pins.py` (pinned URLs, gone-revision fallback, `HF_ENDPOINT` incl. `mtp_fetch`, step 5's folder
line), `tools/test_setup_unsloth.py` (`Pins`, `Sha256`, the resumed-download disk rule, `--compat-bf16`, never
`experts.bin`), `tools/test_setup_choices.py` (`GgufDirShards`, `GgufDirUnsupported`). `tools/test_setup_golden.py`
mocks `download`, `check_shards`, `run` and `mtp_corrupt`.

## In UnbiasedStrata

Removed. The fork downloads nothing (`frontend/main.cpp` checks that every file exists and exits otherwise). The pack
and draft head are made once by hand with upstream's `tools/iq_pack.py`, `mtp_fetch.py fetch/verify`, `mtp_pack.py`
and `mtp_rt.py` at `6f32ec0`, as listed in the fork's `UPSTREAM.md`, which also records the shards' revision
(`ed59f920...`) and the SHA-256 of the prepared files. The fork's working tree carries copies of some of these
tools again under `tools/` ([tools](../tools/README.md)).

## See also

[engine.md](engine.md) · [model-choice.md](model-choice.md) ·
[UNSLOTH_Q4.md "The files"](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/docs/UNSLOTH_Q4.md#the-files)
