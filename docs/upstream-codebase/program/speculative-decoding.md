[Upstream docs](../README.md) › [Program](README.md) › Speculative decoding

# Speculative decoding: verify windows, MTP and suffix drafts

**Purpose:** decode several tokens per pass of the model. A **verify window** runs T tokens through all layers at
once: the last emitted token and T-1 drafts. The window's pick after token t is exactly what plain decode would emit
there (greedy, or the same Philox draw when sampling). So the first draft that differs ends the round, and the round
emits `accepted + 1` tokens. Drafts come from the **MTP draft layer** (`strata::core::MtpDrafter`) or from **prompt
lookup** (`strata::spec::SuffixDrafter`). The **draft policy** (`strata::spec::DraftPolicy`) picks between them. This
is the only decode path for native packs, and the only one in `--serve`.

**Covers:** `src/program/generate.cpp:1774-1780` (window widening), `:3002-3018` (drafter load), `:7799-8178` (the
one-shot loop and its report), `:7047-7192` (the serve loop's decode, which follows the same pattern).

## Set-up

- `--spec T` is the window capacity. `kVerifyMaxT = 8` (`include/strata/kernels/verify_kernels.hpp:21`).
- With `--suffix-draft > 0` and no `--mtp-max-t`, the MTP keeps the requested T (`--mtp-max-t = T`) and the window
  grows to `min(T + 2, 8)`, so a lookup window can be up to 2 tokens longer (`:1777-1780`). The comment's measurement:
  code edits +6-11%, ordinary text unchanged (the comment cites bench/results/2026-09-27-spec, which is not in this snapshot).
- `MtpDrafter::load(--mtp, draft_geometry, session, spec, err, --mtp-window)`. The draft layer is the canonical
  model's MTP head (512 experts) even for the pruned Coder, so `draft_geometry` is the default `ModelGeometry`
  (`:3011-3013`). It is loaded before the expert cache and on the last stage's device with a split. `--mtp` is
  dropped with `--spec < 2`.
- `Verifier::init(wt, g, ss, VerifyHits, head, spec)`. `VerifyHits` carries the device and host residency tables,
  the cache base, the blob size and the per-slot offsets (E-6), so the window computes resident experts on the GPU
  and the rest through the pool callback (`drive_pool_multi`, `:783-804`). In one-shot mode the window **requires**
  the device residency table: `--expert-profile`, `--expert-cache` and the token graph (`:7817-7821`).
- `mtp.bind(wt, head, ver.final_R_all())`: the drafter reads the window's final residuals.
- `ver.set_sampling(sp)` (one-shot: the CLI's sampling, no penalties) and `mtp.set_draft_sampling(sp)`, which only
  matters with coupled drafts. `ver.set_split(--spec-split)`. `ver.set_pcie_mode(dma=0 | direct=1 | else 2)`: the
  copy kernel is the default for every pack, because with DMA the host could block on a driver lock inside a window
  (issue #31). The comment gives its cost as about 1-3% of decode on IQ3_S.
- `drive.d.pcie_num = round(--pcie-frac * 256)`: the share of missed experts the GPU reads over PCIe.

## One round (`:8000-8108`; serve `:7087-7192`)

1. **Window size.** `T = S_mtp` (`--mtp-max-t`, else `--spec`). With `--spec-min-p P`, T grows from 1 while the
   drafter's probability for each draft so far is at least P (`dprob`). The first window of a native pack, and the
   first window of every serve request, is `T = 1` (the last prompt token alone).
2. **Lookup.** With `--suffix-draft > 0`, `sfx.propose(spec - 1)` looks for an earlier repeat of the context's last
   tokens (at least `--suffix-draft` long; the drafter's table holds up to `max_context + 4096` tokens). It is used
   only if its first token equals the MTP's first draft (in one-shot mode without `--mtp`, any match counts).
   `policy.choose(T, k, match)` then decides whether a lookup window of its length beats the MTP window in expected
   tokens per ms.
3. **Fill the window:** `window[0] = x`, then the suffix tokens, the MTP drafts, or (one-shot without `--mtp`) the
   `--spec-oracle` file's tokens. With `--spec-corrupt N`, every Nth draft is replaced by `(d + 1) % n_vocab`.
4. `apply_pending(!adapt_nowait())`: the previous adaptive round's copies land first (#463).
5. Serve only, with penalties: `penalty_rows` stages each row's history and uploads it.
6. **Verify:** `ver.run(T, window, p, pool, user, outv)`. `a` = the number of leading drafts equal to the window's
   picks.
7. The **adaptive tier** starts on its own thread every `--adapt-every` rounds ([expert-tiers.md](expert-tiers.md)).
8. **Commit:** `ver.commit(a + 1)` keeps the state of the emitted tokens only.
9. **Emit** `outv[0..a]`: one-shot appends to `produced`; serve prints `T <id>`. Each token goes into the suffix
   drafter's context. Stop on an end-of-turn id (with `--stop-eos` in one-shot mode; always in serve).
10. **Draft the next window:** `mtp.draft(T, outv, p, a, drafts, err, dprob, spec_min_p)`, unless the run has ended.
    In serve, coupled drafts with penalties get the row-0 history first (`set_draft_history`).
11. `policy.observe(from_sfx, T, a, match, round_ms)` learns acceptance and window costs. The first window is not
    timed.
12. `x = outv[a]`, `p += a + 1`.

In one-shot mode, the canonical pack first produces one token through the token loop, and then
`mtp.draft_first(spec, ss.R, x, p - 1, ...)` seeds the first drafts from the token loop's residual (`:7980-7984`).
After the loop, `ver.wait_commit` waits for the last asynchronous commit before anything reads the session.

## One-shot report (`:8114-8177`)

- `speculation`: rounds, drafts accepted of offered, tokens per round;
- `window sizes` (with `--spec-min-p`) and `suffix drafts`;
- `accepted per round`: a histogram;
- `verify window`: wait for rings, pool, host and commit ms per round, and CPU experts distinct and routed per layer;
- `pool multi`: gate/up, quantize and down ms per round, and GB/s;
- `dispatch`: plan, activation quantize, jobs and run;
- `adaptive tier`: swaps and ms per round;
- `resident RAM` and `expert tiers`: RAM and file blobs, MB read, read speed, the routing prefetch's warm hits;
- `pcie experts`: per layer;
- `mtp`: ms per round drafting, MTP prompt ms, VRAM.

## Exactness and its limits

- A window is exact against plain decode at the same position and seed. `--spec-oracle` with `--spec-corrupt` is the
  test arm: correct drafts are accepted and corrupted ones end the round.
- A swapped-in expert computed on the GPU instead of the CPU rounds differently. That is why the window waits for the
  adaptive copies (#463).
- `--spec-split` is exact and slower (the `Options` comment says about 7%).

## Model and platform notes

- **Native packs** always decode here. The first window holds the last prompt token alone and produces both the first
  token and the drafter's first cell.
- **Canonical pack** with `--spec 0`: plain decode through the token loop.
- **Layer split** (serve only): `ver` is a chain of per-stage verifiers ([layer-split.md](layer-split.md)). The
  drafter and the head are on the last stage.

## Tests

`src/spec/draft_policy_test.cpp` and `src/spec/controller_test.cpp` (in the spec library) test the policy. The engine
loop is tested end to end by the oracle and corrupt arms above and by `tools/batch_test.py` (solo against batched,
token for token).

## In UnbiasedStrata

Changed for Qwen3.6:

- the drafter's geometry follows the target for `qwen35moe` (6abffa7);
- without a drafter or an oracle, a window holds one token instead of padding with token 0 (1f5ef7d);
- `--teacher-force` commits every row along the oracle and refuses `--mtp` (c7825b4).

See [../../codebase/program/speculative-decoding.md](../../codebase/program/speculative-decoding.md).

## See also

[generation-loop.md](generation-loop.md), [serve-mode.md](serve-mode.md), [expert-tiers.md](expert-tiers.md).
