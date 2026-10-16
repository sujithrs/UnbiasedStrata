# Upstream Strata documentation (commit 6f32ec0)

This tree documents **upstream Strata** (https://github.com/Niko1221/Strata) at commit
`6f32ec070f23ced9f50e704d854d775da52591ab`, engine 0.1.39: the exact code this fork was imported from (the fork's
first commit, `12a7825`). Upstream runs Qwen3.8-Flash-Next and its variants (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, the
Coder, Swift 1.5, Unsloth UD-Q4_K_XL) on NVIDIA, AMD or Intel GPUs plus system RAM, on Windows or Linux, with a
one-click installer and a Python server.

Most of upstream's files are not in this repository any more, so source references link to GitHub at that commit.
Each page has an "In UnbiasedStrata" note on what the fork kept, changed or removed. The fork's own documentation
is in [../codebase/README.md](../codebase/README.md).

## Status

The writers for this tree were stopped part-way to save budget.

- **Complete:** build and program.
- **Draft:** every other area. Its pages were written from the source but not reviewed, so check line numbers
  before relying on them.
- **Placeholder:** a page marked "Not written yet" exists only so links resolve. Where the fork has a page on the
  same topic, the placeholder links it.

| Area | What it covers | Pages | Status |
| --- | --- | --- | --- |
| [build/](build/README.md) | CMake options and targets, GPU backends (CUDA archs, HIP, gfx906, SYCL hand-off), the llama.cpp fetch, Docker | 8 written, 0 placeholder | complete |
| [install/](install/README.md) | The one-click installer (setup.py), launchers, hardware detection, model choice, the strata-<model>.json config | 12 written, 3 placeholder | draft: unchecked |
| [serve/](serve/README.md) | The Python server: OpenAI and Anthropic APIs, engine supervision, structured output, MCP, the web app | 14 written, 1 placeholder | draft: unchecked |
| [program/](program/README.md) | generate.cpp: every flag, start-up, expert tiers, layer split, generation and speculative loops, serve protocol | 11 written, 0 placeholder | complete |
| [core/](core/README.md) | Layers, session, CUDA graphs, verify window, draft layer; experts/ and conversation/ subfolders | 7 written, 7 placeholder | draft: unchecked |
| [kernels/](kernels/README.md) | CUDA kernels (cuda/), CPU expert kernels and the n-gram table (cpu/), parity tests (parity/), AMD (hip/) | 14 written, 9 placeholder | draft: unchecked |
| [prefill/](prefill/README.md) | The batched prompt path: chunks, borrowed cache slots, the streamed expert ring, MMQ and fused kernels | 9 written, 0 placeholder | draft: unchecked |
| [libs/](libs/README.md) | Speculation policy, platform I/O, the planner, the GGUF reader, the PLE reader | 1 written, 5 placeholder | draft: unchecked |
| [sycl/](sycl/README.md) | The Intel GPU port: how it was migrated from CUDA, its layout, build and server wrapper | 7 written, 0 placeholder | draft: unchecked |
| [tools/](tools/README.md) | Python tools: pack builders, GGUF I/O, tokenizer, draft head, profile and draft vocabulary, test scripts | 11 written, 1 placeholder | draft: unchecked |
| [guides/](guides/README.md) | Testing index, benchmarks, models, prose-doc guide, glossary | 3 written, 3 placeholder | draft: unchecked |
| [hardware/](hardware/README.md) | Qwen3.8-Flash-Next on RTX 50 and on one or two RTX 3090s | 3 written, 0 placeholder | RTX 50 and RTX 3090 pages rechecked against source; overview unchecked |

## How upstream fits together

```
 START-HERE.bat / setup.sh -> setup.py          install/   detects GPUs and RAM, picks the model, downloads,
                                                           builds or fetches the engine, builds the pack,
                                                           writes strata-<model>.json
 serve/server.py (Python, OpenAI + Anthropic)   serve/     starts the engine from that config and speaks a
          | stdin/stdout line protocol                     line protocol with it: GEN ... / T ... DONE
          v
 strata --serve  (src/program/generate.cpp)     program/   engine main: weights, expert tiers, layer split
          |
          +-- core/      layers, session, verify window, MTP draft layer, expert cache and tiers, conversation state
          +-- kernels/   CUDA (and HIP) kernels; CPU expert pool; the PLE n-gram table
          +-- prefill/   the batched prompt path
          +-- libs/      speculation policy, file I/O, GGUF reader, PLE reader
 sycl/   the same engine migrated to Intel GPUs, with its own server wrapper
 tools/  Python tools that build packs, draft heads and profiles; test scripts
```

## Reading paths

- **Running upstream on specific cards:** [hardware/](hardware/README.md), then [install/](install/README.md).
- **How a request is served:** [serve/](serve/README.md), then [program/serve-mode.md](program/serve-mode.md).
- **How the engine fits a big model in limited memory:** [program/expert-tiers.md](program/expert-tiers.md),
  [program/layer-split.md](program/layer-split.md), and the fork's
  [core/experts/](../codebase/core/experts/README.md) (upstream's own page is a placeholder).
- **What the fork changed:** each page's "In UnbiasedStrata" note, and the fork's [UPSTREAM.md](../../UPSTREAM.md).
