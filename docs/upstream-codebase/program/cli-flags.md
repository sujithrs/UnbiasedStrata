[Upstream docs](../README.md) › [Program](README.md) › Command-line flags

# Command-line flags

**Purpose:** every flag of the `strata` engine, with its default, its effect and the checks that refuse it.

**Covers:** `struct Options` (`src/program/generate.cpp:243-536`), `usage()` (`:538-719`) and the argv loop
(`:1255-1531`). The loop is two `if` chains; the second starts at `:1364`. One chain of more than 120 `else if`
branches went past MSVC's limit of 128 nested blocks (C1061). The order of the tests and what each one does did not
change.

## Counting and parsing rules

- `main` recognises **148 `--` spellings**, `--help` included, plus `-h`. `--expert-cache-device1`, `2` and `3` count
  as three. Counted with `grep -o 'a == "--[a-z0-9-]*"'` over `:1255-1531`.
- An **unknown flag is an error**: `main` prints `unknown argument: X` and `usage()`, then exits 2 (`:1523-1529`). The
  comment says why: a mistyped `--max-neww` that quietly generated 16 tokens would look like a working run.
- A flag without its value exits 2 with `<flag> needs a value` (`next()`, `:1257-1260`).
- Numbers are read with `atoi`, `atoll` or `atof` unless noted, so a malformed number quietly becomes 0. These parse
  strictly: `--tokens`, `--tokens-file` and `--eos-ids` (`parse_i64_list`, `:721-739`), `--logits-stride`
  (`logits_selection::parse_stride`), the three `--conversation-cache-*` flags (`from_chars` with a range check,
  `:1417-1430`) and `--control-vector-scaled` (`strtof`, `:1439-1453`).
- These flags are parsed but **not mentioned anywhere in `usage()`**: `--adapt-decay`, `--adapt-every`,
  `--dump-final-r`, `--eos-ids`, `--mtp`, `--mtp-window`, `--no-fast-attn`, `--no-fast-select`, `--no-fused-gdn`,
  `--no-prefill-borrow`, `--no-publish-kernel`, `--no-spec-split`, `--no-split-rows`, `--pcie-frac`, `--pcie-mode`,
  `--prefill-until`, `--prompt-cache-root`, `--resident-budget-gib`, `--spec-corrupt`, `--spec-min-p`,
  `--spec-oracle`, `--spec-split`, `--split-device`, `--split-skip-if-fits`, `--stop-eos` and `--vram-reserve-mib`.
- These are mentioned only inside another flag's entry: `--spec`, `--serve`, `--layer-split`, `--adapt-swaps`,
  `--slots`, `--control-vector`, `--expert-profile-save-every`, `--vram-segment-mib`, `--top-p`, `--temperature` and
  the `--yarn-*` knobs.
- `usage()` lists `--ple-gguf` twice (`:545`, `:648`).
- `usage()` says `--native` also turns on "the CPU q8_0 contract unless the expert cache is on". The code does not:
  the `--native` block leaves `--cpu-oracle-q8-0` alone (comment at `:1823-1824`).

The setup's start script passes this set (`setup.py:4224-4227`):

```
--pack P --native SHARD1 [--ple-gguf PLE_SHARD] --expert-profile data/<profile>.bin --expert-cache auto
--prefill auto --spec 4 --spec-min-p 0.5 --mtp DIR --max-context CTX
```

It adds `--kv` above an 8K context, `--resident-experts` or `--mmap-experts` in the low-RAM modes, `--ple-io ram` on a
rotational disk, and rope flags when the context goes past the trained one. `serve/server.py` `engine_args`
(`:1461-1480`) adds `--layer-split`, `--split-skip-if-fits`, `--vram-elastic` and the parallel and learned-profile
arguments from the config.

## Input, output length, context

| Flag | Default | Effect |
| --- | --- | --- |
| `--help`, `-h` | | Print `usage()` and exit 0 |
| `--pack DIR` | `pack/full` | The pack directory: `index.txt`, the dense arena, `experts.bin` or `native_experts.txt` |
| `--tokens LIST` | required, except with `--serve` | The prompt as ids in `[0, 2^31)`, separated by commas or spaces. Give only one of `--tokens` and `--tokens-file` |
| `--tokens-file PATH` | | The same, read from a file; a NUL byte is refused (`:1270-1281`) |
| `--max-new N` | 16 | Tokens to generate. Must be `> 0` and `<= --max-context`, and the prompt plus N must fit (`:1847-1851`) |
| `--max-context N` | 4096 | KV and state capacity in cells; must be below `2^31 - 8` (`:1843`) |
| `--stop-eos` | off | Stop at an end-of-turn id |
| `--eos-ids LIST` | `248044,248046` | End-of-turn ids (`<\|endoftext\|>`, `<\|im_end\|>`); giving them also turns on `--stop-eos` |

With `--serve` and no `--tokens`, the prompt becomes `{248045}`, `--max-new` becomes 1 and `--stop-eos` is turned on
(`:1727-1732`). Each request then brings its own tokens.

## Sampling (one-shot mode)

`--serve` samples per request from the `GEN` keys instead ([serve-mode.md](serve-mode.md)).

| Flag | Default | Effect |
| --- | --- | --- |
| `--greedy` | on | Argmax |
| `--seed S` | 0 | Philox seed; **also turns greedy off** (`:1294`) |
| `--top-k N` | 20 | Top-k; the log line shows it capped at 64 (`:3925`) |
| `--top-p F` | 0.95 | Nucleus; must be in `(0, 1]` |
| `--temperature F` | 1.0 | Must be finite and `>= 0`; `<= 0` is reported as greedy |
| `--coupled-draft` / `--no-coupled-draft` | `STRATA_SPEC_COUPLED` (`coupled_draft_env()`) | Coupled draft sampling for the MTP drafter when sampling (`set_coupled_draft`, `:1532`) |

## Rope scaling (context extension)

The rope settings resolve once, before `session_init` (`:2062-2162`). Precedence: an explicit flag, then the model
file's `qwen4exp.rope.*` keys, then the defaults. `none` and `1` are explicit opt-outs. With type `none`, every scaling
knob is reset and only the frequency base is kept; a note is printed if any knob was given (`:2118-2130`).

| Flag | Default | Effect |
| --- | --- | --- |
| `--rope-scaling none\|linear\|yarn` | absent (the model file's, else none) | Scaling type, with llama.cpp's names |
| `--rope-scale F` | 0 = absent | Extension factor; must be `>= 1` when given |
| `--rope-freq-base N` | 0 = the model's (1e7) | Must be `> 1` when given |
| `--rope-freq-scale F` | 0 = `1/--rope-scale` | The raw ggml knob |
| `--yarn-orig-ctx N` | 0 = the model's, else 262144 | The trained context YaRN corrects toward |
| `--yarn-ext-factor F` | -1 = auto (1 for yarn, 0 otherwise) | |
| `--yarn-attn-factor F` | 1.0 | Must be `> 0` |
| `--yarn-beta-fast F` | 32.0 | Must be `> 0` |
| `--yarn-beta-slow F` | 1.0 | Must be `> 0` |

Every knob must be finite (`:1868-1874`), because `atof("nan")` passes every range check. After resolution,
`rope_scaling_invalid` checks the whole config (`:2134-2140`).

## Model files and the PLE table

| Flag | Default | Effect |
| --- | --- | --- |
| `--native SHARD1` | off | The whole native arithmetic set at once (`:1782-1825`). It lists every shard of the model (`gguf_split_paths`; a missing shard is an error). It sets `--ple-gguf` to the shard that holds `per_layer_token_embd.weight`. It turns on `--stream-token`, `--gr-native-mmvf`, `--native-bf16`, `--native-bf16-extra`, `--native-ple-key`, `--native-moe-combine`, `--native-gdn`, `--native-router`, `--native-qsa`, `--native-qsa-indexer`, `--native-rope` and `--native-ple-postops`. `--native-head-gguf` defaults to SHARD1, and `--native-dense-gguf` to every shard plus the PLE shard (unless that is a PLE-only `strata-ple` file). Required by a native (IQ) pack |
| `--native-head-gguf PATH` | SHARD1 with `--native` | The native Q5_K output head; requires `--stream-token` (`:1988-1991`) |
| `--native-dense-gguf PATH` | | Native GDN, QSA and shared-expert projections; repeat it for each shard |
| `--embd-gguf PATH` | | The token embedding from this GGUF instead of `--native`'s (from `tools/embd_bf16_pack.py`); mapped host memory |
| `--keep-canonical` | off | A/B: also load canonical copies of natively served tensors (more VRAM). A native pack refuses it (`:2165`) |
| `--ple-gguf PATH` | found by `--native` | The PLE (n-gram) table shard. Required unless `--no-ple` (`:1838-1841`) |
| `--no-ple` | off | Diagnostic ablation: the output is a different model's. Cannot be combined with `--ple-gguf` |
| `--ple-io direct\|mmap\|ram` | `direct` | How table rows are read: unbuffered SSD reads (`direct`), plain mmap, or mmap locked in RAM (`ram`, refused on Windows, `:1745-1750`) |
| `--ple-row-cache N` | 1048576 | Bounded cache of fetched rows (90 B each); 0 turns it off |
| `--ple-inflight N` | 256 | Outstanding reads, 1..1024 |
| `--ple-delay-us U` | 0 | Fault injection: every row read completes no earlier than U microseconds |
| `--ple-sync-submit` | off | A/B: submit table reads on the token thread instead of an I/O thread |
| `--no-ple-prefetch` | off | A/B: read the PLE table's sixteen rows one at a time (`ple_prefetch_enable(false)`, `:3197`) |

## Native arithmetic switches (each one is an A/B arm of `--native`)

| Flag | Effect |
| --- | --- |
| `--stream-token` | Queue token work on the session stream (the logits are read back only when needed) |
| `--check-logits` | Copy and check all logits in the stream-token path |
| `--gr-fp32-activations` | CUDA-oracle precision for the hyper-connection (GR) activations |
| `--gr-native-mmvf` | Pinned GR norm and projections; the fused GR read is used only with this flag, and only outside the per-stage and dump measurements (`:2284-2285`) |
| `--native-bf16` | BF16 projections for the SSM gates, the router and the indexer |
| `--native-bf16-extra` | BF16 for the PLE value and the shared expert's scalar gate |
| `--native-ple-key` | Native PLE key; requires PLE and `--native-dense-gguf` (`:1943-1946`) |
| `--native-moe-combine` | Pinned fused routed + shared combine |
| `--native-gdn` | Pinned GDN norms, gates and recurrence |
| `--native-flash-attn-short` | Diagnostic attention adapter; requires `--max-context <= 256` and the normal decode loop (no measurement mode, `:1918-1925`). The loop checks a status per QSA layer (`:7753-7761`) |
| `--native-qsa-indexer` | Pinned indexer key cache and pooling |
| `--native-qsa` | Pinned QSA norms and output gate |
| `--native-rope` | Pinned text-only rotary arithmetic |
| `--native-ple-postops` | Pinned PLE post-projection arithmetic; requires PLE |
| `--native-router` | Pinned fused 512-expert top-10 router |
| `--cpu-oracle-q8-0` | llama.cpp's CPU q8_0 expert contract. Cannot be combined with `--expert-cache` or `--expert-profile` (`:1947-1950`); it cost 27.0 against 17.2 ms/token of pool time (comment at `:1823-1824`) |

## KV cache

| Flag | Default | Effect |
| --- | --- | --- |
| `--kv fp16\|int8\|q4_0\|k8v4` | `fp16` | KV storage. `q4` is an alias of `q4_0`. `int8` is half of fp16's VRAM; `q4_0` is 4-bit after a Hadamard rotation; `k8v4` is INT8 K plus rotated Q4_0 V (816 B per cell). `STRATA_KV_ROT=1` sends int8 through the rotation too (`:1751-1764`) |
| `--kv-resident N` | 0 | KV streaming: keep N cells of each QSA layer in VRAM and the whole K/V in pinned RAM; `usage()` says the minimum is 20480. Must be `>= 0`; `k8v4` refuses it (`:1765-1773`) |

## Graph and kernel A/B arms

| Flag | Effect |
| --- | --- |
| `--no-capture` | Run the layers directly (`session_token`) instead of replaying graphs. That path has no CPU expert pool hook, so it **requires `--no-pool`**, and it refuses `--dump-layers` and `--dump-halves` (`:3843-3864`) |
| `--no-token-graph` | Two graphs per layer (the host launches each one) instead of one graph per token |
| `--no-fused-gr` | The six-kernel hyper-connection read and a separate write |
| `--no-fast-attn` | Gather plus one-block-per-head QSA attention |
| `--no-publish-kernel` | Memcpy nodes for the doorbell and the QSA step |
| `--no-fused-gdn` | llama.cpp-layout GDN step plus a separate out-norm |
| `--no-fast-select` | FP64 row scores and bit-serial cell top-k |
| `--shared-late` | The shared expert after the CPU pool (the old order) instead of overlapped with it (`layer_set_shared_early`, `:1781`) |
| `--sync-every-layer` | Debug: synchronise after every layer (`--no-capture` path) |

## CPU expert pool and expert source

| Flag | Default | Effect |
| --- | --- | --- |
| `--no-pool` | off | Skip the CPU expert pool: the GPU-only floor. Turns off the hit path and the routing dump |
| `--pool-workers N` | 0 | 0 = every physical core except the one the host loop runs on |
| `--pool-affinity all\|auto\|p-cores` | `all` | Worker placement; `pcores` is accepted too. `auto` and `p-cores` order P-cores first on hybrid CPUs. Anything else exits 2 |
| `--no-host-worker` | off | A/B: the host thread does not join the drain (five workers plus an idle core instead of six) |
| `--no-split-rows` | off | A/B: one whole expert per pool thread (`drive.d.split_rows`) |
| `--mmap-experts` | off | Read the experts through a mapping (`FileExpertSource`) instead of the resident arena. For a native pack without `experts.bin`, the GGUF shards are mapped in place |
| `--shared-expert-arena FILE` | | Linux: back the resident arena with one `MAP_SHARED` file (meant for `/dev/shm`). Cannot be combined with `--mmap-experts` (`:1600-1603`) |
| `--resident-cpu-experts` | off | With mmap and a static profile, copy the experts the GPU cache does not hold into ordinary RAM. Requires `--mmap-experts` and `--expert-profile` (`:1604-1607`) |
| `--resident-experts` | off | The low-RAM PC's resident mode, chosen by setup: `--mmap-experts --resident-cpu-experts` with the copy page-locked, 4 GiB of RAM headroom, and plain mmap with a warning when the copy does not fit. `STRATA_RESIDENT_PIN=0` and `STRATA_RESIDENT_HEADROOM_GIB` change it (`:1493-1502`) |
| `--resident-budget-gib N` | 0 | The resident mode with a RAM budget (N > 0): the hottest N GiB of non-GPU experts by profile rank go into RAM, and the rest are read from the files (`:1503-1513`) |

Combinations (`:1608-1624`): with a layer split and no helper caches, setup's `--resident-experts` falls back to plain
`--mmap-experts` with a warning. An explicit `--resident-cpu-experts` with a layer split or helper caches exits 2.

## VRAM expert cache and adaptive tier

| Flag | Default | Effect |
| --- | --- | --- |
| `--expert-cache N\|auto` | 0 (off) | Slots of VRAM-resident experts; `auto` is -1. With `--expert-profile`, 0 also means auto (`:2653`) |
| `--expert-cache-per-layer` | off | Each layer gets its own slot range. Sized slots are then not used (`:3296-3299`) |
| `--expert-cache-cpu-order` | off | GPU expert reduction in the CPU's order (`drive.d.hit_cpu_order`) |
| `--expert-profile P` | | A `profile.bin` (from `tools/make_profile.py`) that decides residency; the cache is filled from it once at start |
| `--expert-profile-save P` | | `--serve`: save what the adaptive tier learned as a profile, on `QUIT` and periodically |
| `--expert-profile-save-every MIN` | 10 | Minutes between saves, checked before each request; 0 = on `QUIT` only |
| `--no-hit-poke` | off | A/B: do not poke the driver right after the hit launch, so the GPU hit work does not overlap the CPU pool |
| `--vram-reserve-mib N` | 700 | VRAM left free after `--expert-cache auto`. If the default leaves too few slots, the engine shrinks it to as little as 300 MiB, but not when the flag was given (`:3236-3253`) |
| `--vram-elastic` | off | `--serve`, one NVIDIA GPU, no helper caches: the cache is built in physical segments so the `VRAM` command can give part of it back (`:3317-3332`) |
| `--vram-segment-mib N` | 512 | Segment size; below 64 turns `--vram-elastic` off |
| `--adapt-every N` | 4 | Every N verify rounds, swap often-routed missing experts into VRAM; 0 = static residency |
| `--adapt-swaps N` | 96 | At most N swaps per adaptive round |
| `--adapt-decay F` | 0.7 | Usage counts are multiplied by F after each round |
| `--pcie-frac F` | -1 = the pack's default | Share (0..1) of each layer's distinct missed experts that the GPU reads over PCIe while the CPU computes the rest. The default is 0.2 for the canonical pack and 0.55 for native packs, scaled down on a link measured below 20 GB/s (`:2038-2052`) |
| `--pcie-mode auto\|dma\|kernel\|direct` | `auto` | How that share is read: `dma` = 0, `direct` = 1, anything else (`auto`, `kernel`) = 2, the copy kernel (`:5168`, `:7848`). The string is not validated |

## Multi-GPU

| Flag | Default | Effect |
| --- | --- | --- |
| `--layer-split K[,K2..]\|auto` | off | `--serve` only. K is the first layer of each later GPU (rising, from 2), or `auto` for the placement search. See [layer-split.md](layer-split.md) |
| `--split-device D1[,D2..]` | the next visible GPUs | The later stages' devices; `0` with one K puts both stages on this GPU, sharing everything (the bit-exact A/B of the hand-off) |
| `--split-skip-if-fits` | off | With `auto`: run on CUDA0 alone when it holds every profiled pair plus the session, the drafter and the reserve |
| `--trim-stage-weights` | off | An explicit (not `auto`) split: each GPU loads only its own layers' dense weights. The same switch as `STRATA_STAGE_TRIM=1` |
| `--expert-cache-device1 N\|auto`, `--expert-cache-device2 ...`, `--expert-cache-device3 ...` | 0 | Helper expert caches on CUDA1..3 (with a layer split: the visible GPUs no stage uses). device2 needs device1 and device3 needs device2 (`:1896-1897`) |
| `--expert-cache-remote-placement stripe\|layer` | `stripe` | Stripe ranked experts across the helpers, or give each layer one helper |
| `--remote-expert-opt` | off | The helper decode optimisation (`RemoteExpertOpt`); requires `--serve` |
| `--peer-device N` | -1 (off) | A second GPU as an adaptive expert tier (rows over P2P). Not with `--layer-split` or the helper caches (`:1911-1916`); needs `--expert-profile` and the cache |
| `--peer-reserve-mib N` | 600 | VRAM the peer tier leaves free on its card |
| `--peer-slots N` | 0 = what fits | Expert slots on the peer |
| `--peer-adapt-swaps N` | -1 = `--adapt-swaps` | Peer swaps per adaptive round |
| `--peer-prefill-rows N` | -1 = `chunk * K / 2` | Prompt rows per layer the peer computes; 0 keeps the prompt path on the primary (`:4989-4995`) |

## Prompt path (batched prefill)

| Flag | Default | Effect |
| --- | --- | --- |
| `--prefill CHUNK\|auto\|auto:N` | 0 = the token path | Batched prompt processing in chunks. `auto` picks the largest chunk, up to 8192, whose buffers the expert cache can lend. `auto:16384` and `auto:32768`, or `STRATA_PREFILL_AUTO_MAX`, allow larger chunks (`:1379-1388`). Without a profile, or with `--no-prefill-borrow`, `auto` becomes a fixed 2048 (`:1723-1726`) |
| `--no-prefill-borrow` | off | Reserve the prompt path's buffers for the whole session instead of borrowing the cache's last slots |
| `--prefill-until P` | 0 | Validation: batch only positions `[0, P)` and run the rest of the prompt through the token path, teacher-forced |

## Speculative decoding

| Flag | Default | Effect |
| --- | --- | --- |
| `--spec T` | 0 = plain decode | Verify-window size: the last token plus T-1 drafts. With `--suffix-draft > 0` and no `--mtp-max-t`, `--mtp-max-t` becomes T and `--spec` becomes `min(T + 2, 8)` (`:1777-1780`) |
| `--mtp DIR` | | The MTP draft layer's runtime directory (from `tools/mtp_rt.py`); ignored with a warning when `--spec < 2` (`:3006-3009`) |
| `--mtp-window N` | 32768 | The draft layer attends to the last N cells (0 = every cell) |
| `--mtp-max-t M` | 0 = `--spec` | Cap on the MTP's own windows; longer windows come from suffix matches |
| `--suffix-draft N` | 3 | Prompt lookup: draft from an earlier repeat of the last N or more tokens when the draft policy expects it to pay; 0 = MTP only |
| `--spec-min-p P` | 0 | A draft enters the window only while every draft up to it has at least probability P under the draft layer |
| `--spec-split` / `--no-spec-split` | off | Opt-in split verify window (the overlap study: exact, about 7% slower, per the `Options` comment) |
| `--spec-oracle PATH` | | Without `--mtp`: draft from a token file (the expected continuation, for the exactness test) |
| `--spec-corrupt N` | 0 | Replace every Nth draft with a wrong token |

## Serve mode

| Flag | Default | Effect |
| --- | --- | --- |
| `--serve` | off | Stay resident and take requests on stdin. Needs `--spec >= 2`, `--mtp` and `--prefill` (`:4608-4613`) |
| `--vision` | off | Keep a per-cell (t, h, w) rotary position table so `GENI` requests (images) can run |
| `--batch N`, `--slots N` | 0 | Up to N requests decode together in batch slots. Needs `--serve` and at least 2 slots, and no split on one GPU; capped at 8 (`kVerifyMaxT`) and at what fits (`:2934-3000`) |
| `--batch-groups G` | 1 | With a layer split: the slots in G groups pipelined through the GPUs; G must divide `--batch` |
| `--prompt-cache N` | 6 | Conversation checkpoints kept between requests (about 118 MB of host RAM each); 0 = read every prompt from the start |
| `--prompt-cache-every N` | 16384 | Also checkpoint every N freshly read prompt tokens; 0 = off |
| `--prompt-cache-root N` | 2048 | A prompt read from token 0 also checkpoints at its first turn boundary when that is at least N tokens in; 0 = never |
| `--turn-token ID` | 248045 (`<\|im_start\|>`) | The token that opens a chat turn |
| `--tail-role-token ID` | -1 (off) | A short turn of this role right before the new assistant turn is left out of the checkpoint (the server's reasoning-effort turn) |
| `--short-read N` | 64 | A prompt part of at most N fresh tokens is read through the verify windows instead of the batched path |
| `--conversation-cache-mib N` | 0 (off) | Host RAM budget for parked conversations |
| `--conversation-cache-slots N` | 4 | At most N parked conversations |
| `--conversation-cache-min-free-mib N` | 2560 | Physical RAM floor when parking |

## Control vectors

| Flag | Default | Effect |
| --- | --- | --- |
| `--control-vector FILE` | | A llama.cpp control-vector GGUF at scale 1 |
| `--control-vector-scaled FILE:SCALE[,...]` | | Several files with their scales; the last colon splits, so a Windows `C:\...` path keeps its drive |
| `--control-vector-layer-range A B` | 1 .. the last layer | The layers it applies to (inclusive) |
| `--cvec-mode add\|project` | `add` | `h += s v`, or `h -= s (h.v) v` with v a unit vector |
| `--cvec-dir per-layer\|single:L` | `per-layer` | Each layer's own direction, or layer L's everywhere (project mode only) |

The vectors are loaded before any graph is captured (`load_control_vectors`, `:1074-1146`, called at `:2316-2322`).
`--serve` switches a loaded vector per request (`cvec=0|1`).

## Diagnostics and measurement

| Flag | Effect |
| --- | --- |
| `--dump-logits PATH` | One row of raw logits per position, with an int32 `{n_vocab, n_rows}` header written with the first row |
| `--logits-stride N` | Keep every Nth row plus the final input position (default 1); N > 1 requires `--max-new 1` and `--dump-logits`; give it once only |
| `--dump-residual PATH` | The final R (hc x n_embd f32, after an int32 header), for checking the head on its own |
| `--dump-mixed PATH` | The post-attention residual `bb.mixed` (n_embd f32) |
| `--dump-layers PATH` | R after every layer, per position (the C1 bisection ladder); needs the captured path |
| `--dump-halves PATH` | Both halves' block outputs and injections per layer; captured into the layer graphs |
| `--dump-routing PATH` | Routed expert ids and weights per layer per position; needs the pool |
| `--dump-final-r PATH` | The residual after the last layer per processed position (token path only) |
| `--stage-timing` | Per-stage CUDA-event shares; requires `--no-capture` (`:3753-3759`) |
| `--graph-only` | Measure: replay only the 48 `pre` graphs, then exit |
| `--gpu-only-full` | Measure: replay pre + post for all layers plus the LM head (the per-token GPU floor), then exit |
| `--gpu-stages` | Measure: capture each layer as three graphs and time them from outside the capture, then exit |
| `--stats` | Print the per-phase breakdown after a one-shot run |

`--gpu-only-full` and `--gpu-stages` are refused with a layer split across GPUs (`:1929-1933`). Details are in
[diagnostics.md](diagnostics.md).

## In UnbiasedStrata

The fork keeps all 148 spellings and adds three: `--vram-cap MIB`, `--ram-cap GIB` and `--teacher-force`. Its
`usage()` also documents them. See [../../codebase/program/cli-flags.md](../../codebase/program/cli-flags.md).

## See also

- [startup.md](startup.md): the order in which these flags take effect.
- [env-vars.md](env-vars.md): the environment variables that act as hidden flags.
