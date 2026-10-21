# Performance and Benchmarking

[README](../README.md)

Compare the same checkpoint, quantization, context, and sampling settings.
Record the commit and whether weights were resident, streamed, or distributed.
Keep other GPU workloads idle and repeat in alternating order: one favorable
run is not a speed result.

## Context sweeps

`ds4-bench` measures prefill and generation at successive context frontiers:

```sh
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128
```

Each prefill number measures the newly added interval. Generation uses a fixed
greedy, non-EOS probe. The benchmark normally restores a memory snapshot after
each probe; network TP/pipeline runs and snapshots beyond its memory limit use
prefix replay instead. Do not interpret replay time as continued-prefill speed.

Use `--step-mul F` for exponential context spacing. Output is CSV, including
prefill throughput, generation throughput, and snapshot size when available.
The prompt is the cleaned public-domain *I Promessi Sposi* text in
[speed-bench](../speed-bench/README.md).

Prefill defaults are configuration-specific: ordinary DeepSeek uses 4096-token
chunks, CUDA TP uses 2048, and long PRO prompts may use 8192. GLM chooses its
own chunks and rejects `--prefill-chunk`. The strict DeepSeek API-vector test
pins 2048; do not generalize that setting to every benchmark.

## Recorded Flash Q2 baseline

These existing sweeps use 2048-token intervals and 128 generation tokens per
frontier. They are recorded baselines, not measurements of every subsequent
commit. Full data: [M5 Max](../speed-bench/m5_max.csv) and
[DGX Spark](../speed-bench/gb10.csv).

| Machine | Context | Prefill | Generation |
| --- | ---: | ---: | ---: |
| M5 Max, 128 GB | 2048 | 790.18 t/s | 39.35 t/s |
| M5 Max, 128 GB | 16384 | 572.53 t/s | 36.14 t/s |
| M5 Max, 128 GB | 32768 | 557.04 t/s | 34.36 t/s |
| M5 Max, 128 GB | 65536 | 398.50 t/s | 27.64 t/s |
| DGX Spark, 128 GB | 2048 | 825.76 t/s | 18.05 t/s |
| DGX Spark, 128 GB | 16384 | 872.44 t/s | 15.10 t/s |
| DGX Spark, 128 GB | 32768 | 855.94 t/s | 14.43 t/s |
| DGX Spark, 128 GB | 65536 | 822.98 t/s | 13.84 t/s |

![M5 Max Flash Q2 throughput](../speed-bench/m5_max_ts.svg)

Historical PRO Q2 measurements on the M3 Ultra are retained in this chart:

![PRO Q2 on M3 Ultra](../speed-bench/pro_model_m3_ultra_ts.svg)

## What to compare next

- For SSD streaming, record the effective cache and distinguish cold startup
  from a warm cache.
- For multiple sessions, report both individual latency and aggregate
  throughput. Ordered fallback is not native batching.
- For DSpark/MTP, compare plain decode too. A faster drafter path can still be
  slower than ordinary decoding on an unpredictable prompt.
- For TP, keep quantization and prompt equal; comparing resident Q4 on two
  machines with streamed Q4 on one measures capacity benefits as well as parallelism.

Recent focused TP and DSpark comparisons, including their limitations, live in
[QA_BEFORE_RELEASES.md](../QA_BEFORE_RELEASES.md). Use its speed-regression
procedure rather than accumulating one-off timings in the main README.

## Where the optimized profile can gain performance

These are the defaults used by the native binaries on Apple Silicon. Upstream already has SSD
streaming, DSpark and CED-aware prefill. The fork specializes their execution
for a single user on a memory-constrained Apple Silicon Mac.

| Mechanism | Work reduced / overlapped | Important qualification |
| --- | --- | --- |
| Dense Q4L overlays and specialized Metal kernels | Q8_0 dense parts go from 8.5 to 4.5 bits per weight including scales; paired-row paths reuse reads and reduce intermediate traffic | Lossy; requires calibrated side files |
| BS64 gate/up and G32 down | Replace expensive quantized-expert decoding with layouts and arithmetic suited to the small row counts used during generation | BS64 is a faster-to-compute representation of already low-bit weights, not a promise of fewer bytes than IQ2_XXS |
| Resident expert cache and learned lookahead | Start likely expert reads before demand, preserve useful cached experts, overlap SSD I/O with GPU work | Content-dependent; cold and warm sessions differ |
| Deferred decoder expert contributions | Eligible cache-missing experts in layers 20–34 execute two layers later; encode deferred work only when needed, keep demand reads out of prefetch calibration, and sleep with WFE while awaiting the decision | Approximate; enabled for optimized agent/CLI profiles. Set `LLMS_DS4_LAG=0` to compare without deferral |
| Specialized speculation | Compact draft support, expert bank, vocabulary shortlist and paired-token verification amortize target-model work over accepted tokens | Acceptance matters; upstream already supports speculation |
| Short decoder replay and phase-specific execution | Process the full encoder prompt but reduce upper decoder work to a bounded suffix; split calls at the boundary | The selected four-token replay is an approximation, not upstream's exact dependency suffix |
| Scheduling, fusion and power-aware waiting | Smaller intermediates, specialized reductions, BF16 routers, asynchronous handoffs and WFE waiting reduce overhead and wasted energy | Individual gains interact; they cannot be added as independent percentages |

Some speed also comes from deliberate **quality/performance tradeoffs**, including
routing policy, reduced precision, deferred expert contributions and short decoder replay. This profile is not
an exact-routing/exact-model baseline. These choices need end-to-end evaluation,
not just kernel timing.

**CPU power budgeting does not mean forcing the whole engine onto efficiency
cores.** The native binaries do not apply `taskpolicy -b` to themselves. Demand-side
threads retain responsive scheduling; the expert-dispatch waiter uses ARM WFE
instead of continuously spinning, while speculative read-ahead uses utility QoS.
The thermal gate disables GPU keep-busy filler when IOReport indicates the GPU's
thermal clamp. Backgrounding unrelated CPU work may free power and memory
bandwidth, but backgrounding latency-critical I/O can slow generation. macOS QoS
is a scheduling policy, not a portable hard core-affinity guarantee.

There is no published, matched whole-stack speedup attribution against the exact
upstream HEAD for this exported pack. The historical numbers below are from the
source runtime, not additive contributions or a fresh-clone performance guarantee.
