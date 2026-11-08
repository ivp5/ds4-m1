# ds4-m1

**DeepSeek V4.1 Flash 1017 on one 64 GB M1 Max: 48 tok/s decode · 343 tok/s prefill in coding-agent work (one-shot prose: 27-32 tok/s).**

[antirez/ds4](https://github.com/antirez/ds4) with Metal kernels, Neural Engine programs, SSD expert streaming and a trained speculative drafter
for a 64-GB Apple Silicon Mac. The whole model runs: all 384 routed experts, the Engram tables, thinking and tool calls. The experts stream from
the SSD into a 38 GiB cache; everything else stays in memory. **Model files: [ivp5/DeepSeek-V4.1-Flash-1017-ds4m1](https://huggingface.co/ivp5/DeepSeek-V4.1-Flash-1017-ds4m1)** on Hugging Face.

## 1. Download the model (287 GB)

```sh
hf download ivp5/DeepSeek-V4.1-Flash-1017-ds4m1 --local-dir ds4m1
cd ds4m1 && shasum -a 256 -c MANIFEST.sha256 && cd ..
```

Four files; [the model card](https://huggingface.co/ivp5/DeepSeek-V4.1-Flash-1017-ds4m1) lists what each holds.

## 2. Raise the memory limits (once per boot)

```sh
sudo sysctl -w iogpu.wired_limit_mb=59808 vm.global_user_wire_limit=58497454571 vm.user_wire_limit=58497454571
```

The engine wires about 53 GiB. Every number on this page was measured with these limits. The expert cache (4,100 slots, 38.2 GiB) must
fit in the GPU working set, which is `iogpu.wired_limit_mb`, and is locked with `mlock`, which the two `vm` limits bound. With lower limits the
engine shrinks the cache (it prints `V4.1 SSD cache fitted from 4100 to N experts`): more experts are read from the SSD and decode is slower.
Close other large apps.

## 3. Build

```sh
make -j8 ds4 ds4-agent ds4-server
```

## 4. Run

```sh
./ds4-agent -m ds4m1/DeepSeek-V4.1-Flash-1017-ds4m1.gguf
```

`./ds4` (one prompt) and `./ds4-server` (an HTTP API) take the same `-m`. The first start unpacks 13 GB of small parts into
`~/Library/Caches/ds4m1/` (`DS4M1_CACHE` moves it). Sampling defaults: temperature 1.0, top-p 1.0, min-p 0.2.

## How it compares

| | **ds4m1 (this engine + the HF files)** | vcruz305 SAGE 1.55 bpw (EXL3 / REXL3) | Unsloth UD-IQ2_XXS (GGUF) |
|---|---|---|---|
| Runs on | **M1 Max, 64 GB** | 128 GB-class GPU (108.5 GiB resident) | llama.cpp, 300.5 GB of weights |
| Download | **287 GB** | 320 GB | 300.5 GB |
| Decode | **48.4 tok/s** | — | — |
| Prefill | **342.6 tok/s** | — | — |
| Routed experts vs official, same test (8 agent sessions) | **KL 0.047 · top-1 92.8%** | REXL3: KL 0.091 · top-1 89.9% | — |
| wiki1100 vs official, routed experts only (the same 1,099 positions) | **KL 0.324 · top-1 85.1%** | REXL3: KL 0.380 · top-1 83.9% | — |
| wiki1100 vs official, the whole model | KL 0.833 · top-1 73.6% | **EXL3: KL 0.088 · top-1 88.2% (its own slice)** | KL 0.203 · top-1 90.5% (its own slice) |
| KL vs official on its card's own text | — | EXL3: 0.088 wiki1100, 0.098 e2e_eval; REXL3: 0.051 e2e_eval | 0.203 wiki1100, 0.142 long17k |
| Top-1 on its card's own text | — | EXL3: 88.2% wiki1100, 94.5% e2e_eval | 90.5% wiki1100, 91.2% long17k |
| Whole model vs Q8_0 reference (4 agent sessions) | KL 0.118 · top-1 90.5% | — | — |
| Coding-agent tasks | **15/15 solved, 822 tokens and 20 s per solution** | — | — |

— = not stated on that card. The same-test row swaps only the routed experts (attention, dense layers, router and head are
the official weights in both), so it compares the two expert codecs directly.

## Speed (M1 Max, 64 GB)

| Workload | Decode | Prefill | GPU clock |
|---|---:|---:|---:|
| 16 coding-agent tasks back to back (one process, ~7 min) | 48.4 tok/s | 342.6 tok/s | 1,061 MHz |
| same, steady tasks 5-15, a second run | 48.6 tok/s | 342.2 tok/s | 1,073 MHz |
| same, the published engine (ds4-m1 main) on these four files | 49.5 tok/s | 350.0 tok/s | 1,073 MHz |
| `ds4` one-shot, WikiText prompt of 2,062 tokens + 256 generated (a fresh process) | 26.9 tok/s | 198.0 tok/s | |
| `ds4` one-shot, WikiText prompt of 8,172 tokens + 256 generated (a fresh process) | 31.8 tok/s | 324.9 tok/s | |
| `ds4` one-shot, WikiText prompt of 32,803 tokens + 256 generated (a fresh process) | 27.1 tok/s | 383.2 tok/s | |

The agent rows are agent work (code, tool calls), where the drafter's tokens are accepted 75-80 % of the time; on encyclopedic prose it accepts
25-70 %, so the one-shot rows generate slower. A fresh process also installs its first experts from the SSD (a few seconds).

## Fidelity

**Routed experts against vcruz305's SAGE REXL3 1.55 bpw** — official FP8/FP4 model as teacher, 11,263 positions of 8 agent sessions:

| Routed experts | KL mean | KL p50 | KL p90 | Top-1 |
|---|---:|---:|---:|---:|
| **ds4m1** (2-bit gate/up + 2-bit down) | **0.0467** | 0.00337 | **0.132** | **92.8%** |
| SAGE REXL3 1.55 bpw | 0.0906 | 0.00333 | 0.271 | 89.9% |

ds4m1 wins on all 8 sessions; the medians are equal and the gap is the tail.

**wiki1100 against the official model** — the first 1,100 tokens of the WikiText-2 raw test split (the official tokenizer, no BOS), 1,099 next-token positions; the official log-probabilities computed on the AMD box, the published engine's on this Mac:

| Model | KL mean | KL p50 | Top-1 | Top-5 | Perplexity (official 2.22) |
|---|---:|---:|---:|---:|---:|
| ds4m1, the whole model, shipped configuration | 0.8330 | 0.26375 | 73.6% | 58.1% | 4.47 |
| ds4m1, the whole model, deterministic routing | 0.5227 | 0.06636 | 80.0% | 63.7% | 3.32 |
| ds4m1 routed experts only (official everything else) | 0.3239 | | 85.1% | | |
| SAGE REXL3 1.55 routed experts only | 0.3801 | | 83.9% | | |

Two runs of the shipped configuration differ from each other by KL 0.1116 (mean): the router prefers experts already in the cache. The official model predicts most of this text with near certainty (top-1 probability of 0.99 or more at 603 of 1,099 positions, perplexity 2.22); at those positions the deterministic run reads KL 0.21, and at the positions where the official top-1 probability is under 0.5 it reads 1.31. The gap is not only the start of the text: the second half alone reads KL 0.54 shipped and 0.37 with deterministic routing. The shipped configuration's extra KL over deterministic routing is largest in the first ~330 positions.

vcruz305 and Unsloth do not say which 1,100 tokens of WikiText-2 they used or how they scored them. On this slice, against this reference, SAGE REXL3's routed experts alone read KL 0.380, while vcruz305's card reports 0.088 for its whole EXL3 model on its own slice: numbers from different cards are not on one scale. The routed-experts row compares like with like: both codecs on the same positions against the same official log-probabilities.

**The whole published model** against the Q8_0 GGUF, teacher-forced on four coding-agent sessions (thinking / answer tokens split):

| Session | Positions | KL mean | KL p50 | Top-1 | Top-5 | Thinking KL / top-1 | Answer KL / top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| calc | 818 | 0.118 | 0.00088 | 89.9% | 69.8% | 0.212 / 81.8% | 0.039 / 96.6% |
| textutil | 536 | 0.053 | 0.00006 | 97.4% | 63.8% | 0.255 / 86.7% | 0.020 / 99.1% |
| roman (long thinking) | 1,254 | 0.147 | 0.01430 | 88.4% | 72.2% | 0.181 / 85.3% | 0.082 / 94.6% |
| matrix (long thinking) | 777 | 0.116 | 0.00083 | 89.6% | 70.1% | 0.198 / 82.2% | 0.046 / 95.9% |
| **pooled** | **3,385** | **0.118** | | **90.5%** | **69.8%** | | |

## Tokens per correct solution

Every generated token of every attempt, over the correct solutions:

| Task set | Solved | Tokens per solution | Seconds per solution |
|---|---:|---:|---:|
| 15 coding-agent tasks (Python modules with pytest suites) | 15 / 15 | 822 | 20.1 |

DeepSeek's own benchmark suites (HLE, Terminal-Bench, SWE, ...) were not re-run here: each takes days at 48 tok/s.

## What is upstream's and what is ours

This tree is antirez/ds4 at [b3f5018](https://github.com/antirez/ds4/commit/b3f5018) (2026-11-03) merged with this fork. 3,491 of upstream's
3,513 files are upstream's byte for byte. The 22 that differ are the engine core (`ds4.c`, `ds4_metal.m` and four headers), the five frontends,
the Engram reader, the help and defaults file, five Metal kernel files (four gained this fork's kernels; `dsv4_kv.metal` rounds FP8 / FP4
values with bit arithmetic), the `Makefile` and this README. The fork adds 18 files: the shipped Metal kernel variants in `live/`, three
headers, two Metal files and `metal/README.md`.

The two core files are merged function by function: upstream's version of every function that only upstream changed, both sides' changes
where they merge cleanly, and every new upstream function. This fork's V4.1 path (the 2-bit routed experts, SSD streaming into a fixed expert
cache, the Neural Engine programs, the drafter) replaces upstream's on Apple Silicon. Two upstream GPU entry points that need upstream's
expert-streaming layer (Qwen MoE streaming and the GLM exact-typed indexed decode) refuse in this fork. Upstream's own V4.1 Metal path targets
128 GB-class machines. The merged engine computes exactly what was measured above: its deterministic dump on the published model files is
identical to the measured engine's, all 818 positions.

## Known issues

- The per-verify routing switches (the thinking-span expert cap and related step dials) are cleared when the asynchronous verify commits, so
  they reach only the first layers of a drafted verify. The configuration without this race measured 16% less useful generation, so the
  default keeps it.
- Outputs are not bit-repeatable run to run: the router prefers experts already in the cache.
- Only DeepSeek V4.1 Flash 1017 is supported by this fork's Metal path; upstream's other models are not tested here.

## License

MIT, as [antirez/ds4](https://github.com/antirez/ds4) (`LICENSE`). The evaluation cases' sources and licenses: `EVAL_DATA.md`, `licenses/`.
