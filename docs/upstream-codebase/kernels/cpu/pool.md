[Upstream docs](../../README.md) › [Kernels](../README.md) › [CPU](README.md) › Expert pool

# The CPU expert pool (`ExpertPool`)

## Purpose

Every routed expert that is not resident in VRAM is computed on the CPU, from the expert bytes in host RAM, while the
GPU computes the resident ones. The pool's job "is not to be clever - it is to keep every physical core reading
expert bytes for the whole layer, and to be cheap enough that ten dispatches per layer cost less than one expert"
(`include/strata/kernels/cpu/pool.hpp:3-6`). It owns pinned worker threads (by default one per physical core except
the one the host loop spins on), a lock-free claim protocol, and four ways of splitting a layer's experts across the
threads.

## Files

| File | Lines | Role |
|---|---|---|
| [include/strata/kernels/cpu/pool.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/include/strata/kernels/cpu/pool.hpp) | 286 | `ExpertJob`, `ExpertJobMulti`, `PoolAffinity`, `CpuTopology`, `ThreadAffinity`, `ExpertPool`; protocol notes |
| [src/kernels/cpu/pool.cpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool.cpp) | 779 | Topology detection, pinning, the worker loop, claim / publish / wait, the run modes |
| [src/kernels/cpu/pool_affinity_linux.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_affinity_linux.hpp) | 69 | `detail::` helpers over `pthread_{get,set}affinity_np` with dynamically sized `CPU_ALLOC` masks (CPU ids above 1023 included) |
| [src/kernels/cpu/pool_affinity_win.hpp](https://github.com/Niko1221/Strata/blob/6f32ec070f23ced9f50e704d854d775da52591ab/src/kernels/cpu/pool_affinity_win.hpp) | 64 | `detail::` helpers: hard processor-group affinity for pool threads, CPU Sets for the host thread |

## Main types and functions

| Symbol | Where | What it does |
|---|---|---|
| `ExpertJob` | `pool.hpp:47` | One expert: `blob`, shared read-only `act` (`ActQ`), `out` (H floats), `weight` (never applied here), `slot` |
| `ExpertJobMulti` | `pool.hpp:57` | One expert for `nt <= MAXT` tokens of a verify window: per-token `act`, `out`, and native activations `nact` |
| `PoolAffinity` | `pool.hpp:68` | `All` (default: one worker per physical core), `Auto` (hybrid CPUs: P-cores, then their SMT siblings, then E-cores), `PCores` (P-cores and siblings only); #272 |
| `CpuTopology`, `detect_cpu_topology(skip_first, affinity)` | `pool.hpp:74`, `pool.cpp:32` | Windows: `GetLogicalProcessorInformationEx` with `EfficiencyClass`; Linux: the allowed CPU mask plus sysfs `topology/{physical_package_id,core_id}` and `cpu_capacity`. Keeps the first logical CPU of each physical core; on a hybrid CPU in `All` mode sorts P-cores first (#642). `skip_first` reserves the first core as `host_core` |
| `physical_cores(skip_first, affinity)` | `pool.cpp:276` | `detect_cpu_topology(...).worker_cores` |
| `pin_current_thread(core)` / `restore_thread_affinity(prev)` | `pool.cpp:297`, `:323` | Pin the calling (host) thread and restore it later. Windows: reversible CPU Set selection; Linux: saves the full mask |
| `ExpertPool(n_workers, pin, host_works, affinity)` | `pool.cpp:371` | `n_workers <= 0`: every physical core except the first (hybrid with `Auto` / `PCores`: P-cores - 1). Worker i is pinned to `topo_.worker_cores[i]` (`:392-398`); each owns an `ExpertScratch` |
| `run(jobs, n)` | `pool.cpp:720` | Whole experts, one per claim (mode 0) |
| `run_split(jobs, n)` | `pool.cpp:634` | At most `kMaxSplit = 16` experts split by rows across all threads: gate/up rows (mode 1), quantize each intermediate on the host, down rows (mode 2). Bitwise the same as `run` |
| `run_split_multi(jobs, n)` | `pool.cpp:650` | Canonical Q2_0, multi-token: `3 * threads` equal row ranges over all `n * FF` gate/up rows (mode 3), quantize, then over `n * H` down rows (mode 4); at most `kMaxSplitMulti = 96` experts |
| `run_split_multi_native(fmt, jobs, n)` | `pool.cpp:689` | Native packs: modes 5 and 6, in batches of 96 experts |
| `diag(FILE*)` | `pool.cpp:350` | Watchdog dump: epoch, claims, done, parked / sleeping counts, each thread's state (issue #31) |
| `phase_ms(...)` | `pool.hpp:188` | Host time waiting for park / drain / re-park; `ms_multi_gu/q/down`, `multi_bytes` for the multi paths |

Constants: `kSpinBeforeSleep = 20 ms` (`pool.hpp:201`), `kStall = 60 s` (`pool.hpp:204`).

## Platform: topology and pinning

| | Windows | Linux |
|---|---|---|
| Core list | `GetLogicalProcessorInformationEx(RelationProcessorCore)`; processor id = `group * 64 + bit` (`pool.cpp:38-153`). Without it, every active processor of every group (`:40-53`) | `pthread_getaffinity_np` mask, then sysfs per CPU (`:154-272`). Without sysfs every allowed CPU counts, SMT siblings included (`:155-158`) |
| Hybrid detection | `EfficiencyClass` differs between cores (`:86-105`) | `cpu_capacity` differs between CPUs (`:218-231`) |
| Worker pinning | `SetThreadGroupAffinity` to one processor (`pool_affinity_win.hpp:86-95`); hard affinity is used only for pool-owned threads | `pthread_setaffinity_np` with a `CPU_ALLOC` mask sized for the CPU id (`pool_affinity_linux.hpp:60-66`) |
| Host pinning | `SetThreadSelectedCpuSets` (soft); the CPU Set id is found by group and processor index (`pool_affinity_win.hpp:109-130`); restoring an empty selection restores Windows 11's implicit all-group eligibility (`pool.cpp:326-331`) | Saves the whole mask, pins, restores it (`pool.cpp:311-319`, `:333-335`) |

The hybrid ordering came from forks: #642 ("from Hardin22's fork") puts P-cores first so a pool smaller than the core
count (setup's recommended `--pool-workers` on a hybrid CPU) runs on P-cores and the first E-cores
(`pool.cpp:107-113`, `:233-236`).

## The claim / park protocol

```
host (run / run_phase)                            worker i
------------------------------------              ------------------------------------------
wait_parked(): parked_ == n_                      parked_++ (once, at start)
write jobs_/mode_/mrows_/...                      spin on epoch_ (_mm_pause); after 20 ms sleep on sleep_cv_
begin_batch(n):                                     (sleepers_++ then re-check epoch_)
  done_ = 0
  head_ = epoch+1 << 32 | n << 16 | 0  (release)
  publish(): epoch_++ ; if sleepers_ notify_all   sees new epoch -> seen = epoch_ ; parked_--
host_works: claim(e) and run jobs too              drain(seen): claim(seen) = CAS on head_ that only
wait_done(n): done_ >= n                             succeeds while head_'s epoch == seen and index < n;
wait_parked(): parked_ == n_                         run job ; done_++ (release)
                                                  parked_++ and spin again
```

- **Epoch-tagged claims (issue #29).** `head_` is one 64-bit word `epoch | njobs | index` (`pool.hpp:250`,
  `pool.cpp:27-29`). A worker that wakes late can only claim from the batch it woke for, so it can never run a job
  of the next batch twice or bump `done_` after the host reset it (`pool.hpp:19-27`, `pool.cpp:473-482`).
- **Waiting for park as well as done.** `run` waits for every worker to be parked before writing a batch and again
  after `done_ == n`, because a worker can still be inside its drain loop after its last increment
  (`pool.hpp:14-17`).
- **Lost wake-up.** `publish` and the sleeper both use seq_cst on `epoch_` / `sleepers_`, so one of them always sees
  the other's write (`pool.cpp:410-421`).
- **Each hot atomic on its own cache line** (`alignas(64)`), and nothing is incremented in the spin loop (review
  finding C3, `pool.hpp:231-242`); `pauses()` therefore always returns 0 (`pool.hpp:176-179`).
- **Sleeping workers (issue #4).** A parked worker spins for `kSpinBeforeSleep`, then blocks on `sleep_cv_`, so the
  engine does not hold every core at 100% between requests (`pool.hpp:194-201`, `pool.cpp:440-457`).
- **Stall watchdog.** A wait that sees no progress for `kStall` prints the state, calls
  `strata::core::release_gpu_waits` (so a GPU spinning on this layer's flag is released, #267) and aborts, so the
  server can restart the engine (`pool.cpp:498-540`).
- **The host drains too** (`host_works`, default on): during `run` the host claims jobs through the same `head_`
  instead of spinning on `done_`. Measured before the change: 33.7 GB/s with five workers and an idle host core,
  where 5/6 x 44.14 = 36.8 was expected (`pool.hpp:126-136`, `pool.cpp:739-763`). `--no-host-worker` is the A/B arm.
- With `n_ == 1` (one worker), `run` computes every job inline on the host (`pool.cpp:722-725`) and `run_split`
  falls back to `run` (`:636`); the multi-token paths have no such check.
- `begin_batch` aborts on more than 65,535 jobs (16-bit index field, `pool.cpp:485-488`).

## Row splitting

```
run_split_multi_native(fmt, jobs[0..n))           drain(), mode 5 / 6, per claimed task i
  mtasks = 3 * (workers + host)                    rows [mrows*i/mtasks, mrows*(i+1)/mtasks) over all experts
  mrows = nb * FF   -> run_phase(5)               mode 5: native_gu_rows (or q2_rows_any x2 + SwiGLU for Q2_0)
  host: native_quant_h / act_quant_any per token           -> SplitBufMulti[e].ff[t]
  mrows = nb * H    -> run_phase(6)               mode 6: native_down_rows (or q2_rows_any for Q2_0 down)
                                                           -> jobs[e].out[t]
```

A task's row range may span two experts (`pool.cpp:564-597`). Modes 5/6 use the compile-time `FF` / `H`
(`pool.cpp:563`, `:699`, `:705`, `:708`), which equal every upstream model's expert widths. A Q2_0 gate/up layer uses
`thread_local` gate and up buffers and computes SwiGLU itself (`pool.cpp:569-580`).

## How the CPU and the GPU overlap

The pool is driven from `src/core/expert_source.cpp`, which the session's layer loop (`src/core/session.cpp`) calls
once per layer:

```
GPU stream                               host thread (pinned to the reserved core)
pre[l] graph: mixer, router, top-k   -->  doorbell ring (mapped pinned memory: x_f, ids, weights)
                                          spin on the ring
hits: VRAM-resident experts          <--  enqueue the GPU's share (does not wait)
   computing on the GPU ...               expert_pool_dispatch: one ActQ for the layer,
                                            an ExpertJob per routed expert that is NOT in VRAM,
                                            ExpertPool::run_split / run   (all cores, host included)
H2D copy of y_miss  <------------------  the CPU's expert outputs (rows owned by the GPU written as zeros)
post[l] graph: moe_combine applies the router weights to GPU + CPU parts
```

- `expert_pool_dispatch` (`src/core/expert_source.cpp:1852`) builds one job per expert the CPU owns, sharing one
  `ActQ`, zero-fills rows owned by the GPU or a remote helper, and calls `run_split` (default) or `run`
  (`--no-split-rows`) (`expert_source.cpp:1961-1962`).
- In a speculative verify window, `expert_pool_dispatch_multi` (`expert_source.cpp:1982`) groups the window's tokens
  by distinct CPU expert (one `ExpertJobMulti` per expert), quantizes each token's activation once
  (`act_quant_any`, `native_quant_act` or `act_quant_q8_1`, `:2144-2149`) and calls `run_split_multi_native` or
  `run_split_multi` (`:2213-2214`).
- The host thread is pinned once per session to `physical_cores(false)[0]` - the core the pool keeps free - by
  `SessionLoopScratch` (`src/core/session.cpp:541-550`); the header records 36.32 GB/s with the host pinned vs
  26.9 GB/s unpinned (`pool.hpp:99-101`).
- The pool never applies the router weight: `moe_combine` on the device does (`expert_source.cpp:1854`).

## Callers and dependencies

- Constructed once in `src/program/generate.cpp:3190` as
  `ExpertPool pool(o.pool_workers, /*pin=*/true, /*host_works=*/!o.no_host_worker, o.pool_affinity)`.
- Kernels: [q2-expert.md](q2-expert.md) (modes 0-4, Q2_0 in 5/6) and [native-experts.md](native-experts.md) (5/6).
- `strata/core/progress.hpp`: `diag_pool_fn()` registration (`pool.cpp:388`) and `release_gpu_waits()`.
- `Threads::Threads` (`CMakeLists.txt:941-942`).

## Configuration

| Knob | Effect |
|---|---|
| `--pool-workers N` (`generate.cpp:1346`) | Worker count (0 = default). Upstream's `setup.py` adds a recommended value on hybrid CPUs (`setup.py:391-399`) |
| `--pool-affinity all\|auto\|p-cores` (`generate.cpp:1347-1356`) | `PoolAffinity` (`pcores` is accepted too) |
| `--no-host-worker` (`:1357`) | `host_works = false` (A/B arm) |
| `--no-split-rows` (`:1389`) | `expert_pool_dispatch` uses `run` instead of `run_split` (`generate.cpp:3667`) |
| `STRATA_POOL_SPIN_US` | Spin time before a parked worker sleeps (a test knob; default 20 ms, `pool.cpp:373-374`) |
| `--cpu-oracle-q8-0` | Disables the split / multi paths (they fall back to `run`, one token at a time) |

## Tests

| Test | What |
|---|---|
| `pool_test --selftest` | Bitwise vs serial, every job exactly once (sentinel outputs), repeated batches, batch sizes around the worker count; needs `pack/full/experts.bin` |
| `pool_stress 20` | 20 s of tiny batches (0-4 jobs, `run` and `run_split`) with sleeps long enough for workers to sleep; a watchdog turns a hang into a failure (issue #29) |
| `pool_affinity_test` | Linux: affinity round trips (full, sparse / nested, high CPU ids, invalid targets); Windows: CPU Set round trips and worker hard pins |

Run conditions: [tests.md](tests.md).

## Gotchas

- The pool reserves the first physical core only by convention: if the caller does not pin the host thread there,
  the reservation "is a fiction" (`pool.hpp:93-97`).
- `run_split_multi` with more than 96 experts, or with the oracle contract on, falls back to single-token
  `run_split` in chunks of 16 (`pool.cpp:652-666`).
- A worker that went to sleep counts as parked; this is why claims carry the epoch.
- In `Auto` / `PCores` mode the Windows and Linux hybrid branches list every logical CPU of the E-cores (SMT included),
  not one per core (`pool.cpp:137-139`, `:256-258`).

## In UnbiasedStrata

`pool.cpp` changed in 22d72e6: modes 5/6 and `run_split_multi_native` use the layer's `NativeFmt::n_ff` / `n_embd`
instead of `FF` / `H`, because with Qwen3.6-35B-A3B's 2048 x 512 the canonical widths wrote past every output row.
`pool.hpp` and the affinity helpers are unchanged. The fork's page:
[../../../codebase/kernels/cpu/pool.md](../../../codebase/kernels/cpu/pool.md).

## See also

- [q2-expert.md](q2-expert.md), [native-experts.md](native-experts.md), [tests.md](tests.md)
- [docs/upstream/DETAILS.md](../../../upstream/DETAILS.md) ("How it works")
