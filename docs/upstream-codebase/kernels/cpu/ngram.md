[Upstream docs](../../README.md) › [Kernels](../README.md) › [CPU](README.md) › N-gram hash and PLE table

# The PLE n-gram hash and the per-layer token table (`PleTable`)

## Purpose

Qwen3.8-Flash-Next (`qwen4exp`) has a per-layer embedding (PLE) on layer 1 whose input is 16 rows gathered from a
hashed n-gram table, `per_layer_token_embd.weight`: 320,001,536 rows of 160 values (IQ4_NL: 90 B per row, 28.8 GB in
the original model's **second GGUF shard**). The table is not in the pack; it is the only tensor the engine reads
straight from a GGUF. This file is the host side of that module (`include/strata/kernels/ngram.hpp:1-10`):

- `ngram_rows`, the **host hash**: 16 row indices per token from the token and its two predecessors (64-bit multiply
  and XOR; `qwen4exp.cpp` L1092-1124 in llama.cpp, host-side there because "ggml has no int64 and no xor").
- `PleTable`, the **row gather**: opens the shard, validates the tensor, and returns the 16 rows as 2,560 floats
  (head-slowest), through one of three I/O modes (`--ple-io direct|mmap|ram`).

The GPU block that consumes the rows is `src/kernels/cuda/ple.cu` ([../cuda/README.md](../cuda/README.md)). The
unbuffered SSD reader used by the default mode is `strata::ngram::PleReader` (`src/ngram/ple_reader.cpp`, 547 lines).
Every upstream model (Flash-Next, Coder, Swift 1.5, Unsloth, OrcaRouter) has this layer.

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/kernels/ngram.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/ngram.hpp) | 216 | Geometry constants, `PleConsts`, hash, dequantizers, `PleIo`, `PleIoOptions`, `PleTable`, the prefetch switch |
| [src/kernels/ngram.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/ngram.cpp) | 506 | Implementation |

`ngram.cpp` is host-only but is compiled into the GPU library `strata_kernels` (with `src/ngram/ple_reader.cpp` and
`src/platform/direct_file.cpp`, `CMakeLists.txt:333-335`) because `ple.cu` shares its geometry constants
(`CMakeLists.txt:773-775`). It is therefore built only with a GPU backend (CUDA, either HIP build).

## Geometry constants (`ngram.hpp:31-49`)

`NG_N_EMBD = 2560`, `PLE_HEAD_DIM = 160`, `NGRAM_SIZE = 3`, `HEADS_PER_NGRAM = 8`, `PLE_N_HEADS = 16`,
`PLE_CONV_KERNEL = 4`, `PLE_EOS_TOKEN_ID = 248044`, `NG_HC = 4`, `NG_HC_DIM = 10240`, `NG_HIST = 9`,
`TOKEN_NULL = -1`, `NG_RMS_EPS = 1e-6`, `PLE_TABLE_ROWS = 320001536`, `PLE_ROW_BYTES = 90` (IQ4_NL),
`PLE_ROW_BYTES_FP8 = 160`.

## Main functions and types

| Symbol | Where | What it does |
|---|---|---|
| `ple_artifact_consts()` | `ngram.cpp:37` | The artifact's three layer multipliers, 16 head vocab sizes (20000003, 20000023, ... - not an arithmetic sequence) and their running-sum offsets, written out verbatim |
| `ngram_mixed(ctx, mult, n)` | `ngram.cpp:58` | `ctx[0]*m[0] ^ ctx[1]*m[1] ^ ...` in wrapping uint64 |
| `ngram_rows(tokens, prev, n_tokens, c, out)` | `ngram.cpp:67` | Per token: context = token + predecessors (`prev` oldest first); an EOS or missing (`< 0`) predecessor cuts and propagates to every older position; heads 0-7 hash the bigram, 8-15 the trigram; `row = mixed % vocab[h] + offset[h]` |
| `iq4nl_dequant_row(row, out160)` | `ngram.cpp:101` | 5 blocks of 32: fp16 `d`, then 16 bytes; byte j holds elements j and j+16 (split halves, not interleaved) |
| `fp8_e4m3_dequant_row(row, scale, out160)` | `ngram.cpp:133` | E4M3 "fn" table (0x7F / 0xFF NaN read as 0) times the table's single scale |
| `PleIo`, `PleIoOptions` | `ngram.hpp:113-128` | `Direct` (default) or `Mmap`; `max_inflight` 256, `cache_rows` 1<<20, `io_thread`, `lock` (ram mode), keep-alive settings |
| `PleTable::open(path, err, io)` | `ngram.cpp:176` | Parses the GGUF with `GgufFile`, finds `per_layer_token_embd.weight`, accepts IQ4_NL, Q5_0 (110 B rows, OrcaRouter #296) or I8 tagged `strata.ple.format = f8_e4m3` with a positive `strata.ple.scale` (#291); checks the tensor fills the file exactly (alone in its shard) or fits at its offset (Swift 1.5's shard 1 holds layers 0-12 and the table); then sets up the I/O mode |
| `issue(rows16)` / `collect(out2560, err)` | `ngram.cpp:358`, `:384` | The split API: issue as soon as the token is known, collect before layer 1. One token in flight per table |
| `prefetch_rows(rows16)` / `wait_prefetches()` | `ngram.cpp:290`, `:283` | Up to 16 non-blocking prefetch tickets (Direct), or `PrefetchVirtualMemory` (Windows, mapped mode) |
| `gather_batch(rows, n_tokens, out, err)` | `ngram.cpp:398` | Many tokens as one reader request (page dedupe and sort); reuses matching prefetch tickets directly |
| `gather(rows16, out2560)` | `ngram.cpp:462` | Synchronous `issue` + `collect` (Direct) or 16 row reads with a `PrefetchVirtualMemory` first (mapped) |
| `read_row`, `rows()`, `format()`, `locked()`, `bytes_read()`, `io_report()` | `ngram.cpp:330-460` | Diagnostics; `io_report` prints the reader's hit rate, SSD reads and latency percentiles |
| `ple_prefetch_enable(bool)` | `ngram.cpp:34` | A/B switch for the Windows `PrefetchVirtualMemory` hint (`--no-ple-prefetch`) |

## The three `--ple-io` modes

| Mode | `PleIoOptions` | What happens at `open` | Row reads |
|---|---|---|---|
| `direct` (default) | `mode = Direct` | The GGUF mapping is dropped after the header parse; `PleReader::open` at `data_start + tensor offset` (`ngram.cpp:246-261`) | Unbuffered 4 KiB reads on an I/O thread, bounded row cache; the table never enters RAM or the OS file cache |
| `mmap` | `mode = Mmap` | `GgufFile`'s read-only mapping is kept | Page faults into the mapping (Windows: `PrefetchVirtualMemory` hint first) |
| `ram` | `mode = Mmap, lock = true` | POSIX only. `madvise(MADV_WILLNEED)` then **`mlock` of the whole table inside the file mapping**; if `mlock` fails ("raise `ulimit -l`") every page is touched once instead (`ngram.cpp:262-278`) | Memory reads; `locked()` says whether `mlock` succeeded. Pages that were only touched can still be evicted and faulted back from the file |

Selected in `src/program/generate.cpp:2386-2432` from `--ple-io`, `--ple-inflight`, `--ple-row-cache`,
`--ple-sync-submit`, `--ple-delay-us`; `--ple-io ram` is refused on Windows ("no mlock", `generate.cpp:1745-1750`).
Upstream's installer adds `--ple-io ram` only when the model sits on a rotational disk and the RAM can hold the
table (`setup.py:4238-4250`, #605); the engine warns about a rotational disk in direct mode
(`generate.cpp:2410-2419`).

## Data flow

```
token t, prev[2] --ngram_rows--> rows16 --PleTable::issue--> (Direct: PleReader tickets on the I/O thread)
                                                  ...embedding, layer 0 on the GPU...
                         PleTable::collect --decode (IQ4_NL / Q5_0 / FP8)--> out2560 (host)
                         --cudaMemcpyAsync--> emb_dev --> ple.cu block on layer 1
```

Callers:
- Decode: `ple_issue_token` / `ple_finish_token` / `ple_stage_token` in `src/core/layer.cpp:1290-1315` (hash,
  `issue`, `collect`, H2D copy).
- Verify window: `src/core/verify.cpp:1410-1435` (hash and `prefetch_rows` per token, then `gather_batch`), also
  `:1515` and `:1924-1926`.
- MTP drafter: `src/core/mtp.cpp:890-893` (hash, `prefetch_rows`).
- Prompt path: `src/prefill/prefill.cpp:1607-1612` (`gather_batch` per chunk).
- The table is opened once at `src/program/generate.cpp:2405`.

## Invariants and rival readings (`ngram.hpp:12-21`)

The hash is "the one place in this model where a wrong bit is silently plausible"; each property has a rival that
still yields an in-range index, and each is asserted observable by `ple_parity`:

- XOR, not sum; `% vocab_size`, not a mask (the vocab sizes are not powers of two).
- The EOS cut is evaluated before storing and propagates to every older position; a missing predecessor is
  `TOKEN_NULL = -1` (`t < 0`); token id 0 is a real token; a token's own EOS does not cut its own context.
- `prev` is oldest first: the predecessor s positions back is `prev[i*2 + (2 - s)]` (`ngram.cpp:74-77`).
- The IQ4_NL codebook has no `-8` at the pinned ggml (3cf03257) (`ngram.hpp:84-89`).
- Rows are flattened head-slowest (`out[h*160 .. (h+1)*160)`), as `ggml_get_rows` does.
- The table offset comes from a real GGUF parse: the data section starts after the header (192 B in the original
  shard), and reading at file offset 0 would decode shifted but plausible rows (`ngram.hpp:157-161`,
  `ngram.cpp:223-245`).
- Never keep the shard mapped while reading it direct: a live mapping serializes the unbuffered reads ("311 -> 1,575
  us per token", `ngram.hpp:110-112`).

## Configuration

| Knob | Effect |
|---|---|
| `--ple-gguf PATH` | The shard holding `per_layer_token_embd.weight` (with `--native`, found in the model's shards when not given) |
| `--ple-io direct\|mmap\|ram` | I/O mode (above) |
| `--ple-row-cache N` | Row cache capacity in rows (default 1,048,576; 0 = off) |
| `--ple-inflight N` | Outstanding SSD reads (default 256, 1..1024) |
| `--ple-delay-us U` | Fault injection: every row read completes no earlier than U us |
| `--ple-sync-submit` | Submit reads on the token thread (A/B arm) |
| `--no-ple-prefetch` | Disable the Windows `PrefetchVirtualMemory` hint (`generate.cpp:3197`) |
| `STRATA_SSD_KEEPALIVE`, `STRATA_SSD_KEEPALIVE_WINDOW` | Direct mode with the I/O thread only: ms without a read before a keep-alive read (default 100, 0 = off) and seconds after the last request (default 60) (`generate.cpp:2394-2404`) |

## Platform-specific behaviour

- **Windows**: `PrefetchVirtualMemory` before the 16 page faults of a mapped gather (measured 2.10-2.61 ms per token
  for the serial faults, `ngram.hpp:191-202`); no `ram` mode.
- **Linux / macOS**: `ram` mode available; the `madvise(MADV_RANDOM)` the phase plan asks for is not applied
  (`ngram.hpp:204-214`).

## Tests

| Test | What |
|---|---|
| `ple_parity --selftest` (ctest `ple_parity`) | Hash vs `ref/ngram.py` via `ple_oracle_vectors.inc`; table rows vs numpy over the original GGUF; the PLE block vs a ggml capture. Needs the real shard, `pack/full/dense.bin` and `bench/micro/ple_{in,out}.bin` |
| `ple_fp8_parity` (no ctest) | FP8 table through `PleTable` in Direct and Mmap modes and `gather_batch`, bit for bit vs the checkpoint's decoded values |
| `ple_q5_parity` (no ctest) | Q5_0 rows vs ggml's dequantizer on a real file |
| `ple_reader_selftest` (`src/ngram/ple_reader_test.cpp`, `CMakeLists.txt:843-847`) | The reader, on a temporary file |

See [../parity/README.md](../parity/README.md).

## Gotchas

- `ngram.hpp:120-122` and `:165` describe `lock` as locking "the whole mapped table"; when `mlock` fails, the pages
  are only touched, and `locked()` is false.
- `issue` returns false if the previous token's rows were never collected; `gather_batch` refuses while a token is in
  flight.
- In Direct mode `gather` is `const` but mutates the reader through a `const_cast` (`ngram.cpp:465`).
- The `Impl` comment calls Q5_0 "the mapped reader only" (`ngram.cpp:142`), but `open` passes the 110 B row size to
  `PleReader` in Direct mode too; `ngram.cpp:217-219` says the earlier refusal was stale.
- `ple_parity`'s default GGUF path is `../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` relative to
  the source tree unless `STRATA_PLE_GGUF` or `--gguf` is given.

## In UnbiasedStrata

Changed by commit 9a53e42 ("n-gram table in its own memory"): `--ple-io ram` no longer `mlock`s the file mapping. It
copies the table into an anonymous private mapping with 16 parallel readers (`platform::read_range`), releases the
GGUF, and then attempts `mlock` as an extra; the memory is resident whether or not the lock succeeds. The hash, the
dequantizers and the Direct / Mmap modes are unchanged. The fork's page:
[../../../codebase/kernels/cpu/ngram.md](../../../codebase/kernels/cpu/ngram.md).

## See also

- [../cuda/README.md](../cuda/README.md) (the PLE block), [../parity/README.md](../parity/README.md)
- [docs/upstream/DETAILS.md](../../../upstream/DETAILS.md) (the SSD tier)
