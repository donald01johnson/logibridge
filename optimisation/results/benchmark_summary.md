# F2 Five-Metric Benchmark Results

Benchmark configuration: 10 warm-up runs excluded, 200 measured runs, 1 TFLite thread(s), laptop TDP 15.0 W.

| Variant | Mean latency (ms) | p95 latency (ms) | Size (KB) | Accuracy (%) | Energy (mJ) |
|---|---:|---:|---:|---:|---:|
| M1 — FP32 Baseline | 0.002826 | 0.002909 | 5.477 | 100.00 | 0.042383 |
| M2 — PTQ Full INT8 | 0.012944 | 0.013160 | 4.781 | 100.00 | 0.194162 |
| M3 — 35% Pruned + Full INT8 PTQ | 0.013294 | 0.015203 | 4.781 | 100.00 | 0.199416 |

Energy values are estimates based on psutil process CPU utilisation, the documented laptop CPU TDP, and E = P × t.
