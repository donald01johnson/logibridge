# F3 — Deployment Recommendation

## Recommendation

Deploy M1, the FP32 baseline, to the 85-truck pilot fleet.

## SLA Translation

The end-to-end alert SLA is 90 seconds, or 90,000 milliseconds. The
preprocessing pipeline requires an initial 30-second feature window, leaving a
conservative maximum of 60 seconds, or 60,000 milliseconds, for model
inference, local messaging, alert generation, and safety margin.

M1 recorded a mean inference latency of 0.002826 ms and p95 latency of
0.002909 ms. Its p95 latency consumes approximately 0.00000485 percent of the
60,000 ms remainder and provides over 20 million times latency headroom.

All three variants satisfy the SLA, but M1 has the lowest measured mean and
p95 latency.

## Hardware Storage and Memory

The selected deployment target is a Raspberry Pi 5-class Linux edge gateway.
For this device, the assignment's Flash and SRAM constraints correspond to
persistent microSD or attached storage and LPDDR4X system RAM, respectively.

The measured model sizes are:

| Variant | Model size |
|---|---:|
| M1 FP32 | 5.477 KB |
| M2 Full INT8 | 4.781 KB |
| M3 Pruned plus INT8 | 4.781 KB |

M1 is only 0.696 KB larger than M2 and M3. This difference is negligible for
the gateway's persistent storage. The six-value input, small 6-32-16-3 Dense
architecture, and limited activation tensors are also negligible relative to
the device's system memory.

TFLite runtime memory is not identical to model-file size because the
interpreter also allocates tensors and runtime structures. Nevertheless, none
of the three variants creates a meaningful storage or memory constraint on
the selected Linux gateway.

## Critical Recall

The recommended M1 variant achieved 100 percent Class 2 Critical recall on
the held-out validation set. This exceeds the mandatory requirement of more
than 95 percent.

The baseline confusion matrix classified all 18 held-out Critical windows
correctly. No Critical false negatives were observed in this controlled
synthetic validation set.

The result should not be interpreted as independent field validation because
the dataset is synthetic and adjacent sliding windows may overlap.

## Variant Comparison

| Variant | Mean latency (ms) | p95 latency (ms) | Size (KB) | Accuracy | Critical recall | Estimated energy (mJ) |
|---|---:|---:|---:|---:|---:|---:|
| M1 FP32 | 0.002826 | 0.002909 | 5.477 | 100% | 100% | 0.042383 |
| M2 Full INT8 | 0.012944 | 0.013160 | 4.781 | 100% | 100% | 0.194162 |
| M3 Pruned plus INT8 | 0.013294 | 0.015203 | 4.781 | 100% | 100% | 0.199416 |

M1 provides the best measured latency and estimated energy. M2 provides the
smallest file size, but its absolute advantage over M1 is only 0.696 KB. M3
is dominated by M2 because both have the same file size, accuracy, and
Critical recall, while M3 has slightly worse latency and energy.

## Final Decision

M1 should be deployed because it has the best measured mean latency, p95
latency, and estimated energy while preserving 100 percent Critical recall.
Its small storage penalty is immaterial on the selected Raspberry Pi 5-class
gateway.

M2 should remain available as an alternative for validation on the actual
truck hardware. INT8 can perform differently on ARM CPUs and
integer-accelerated devices, so a physical-device benchmark may change the
runtime ordering.

Energy values are TDP-scaled estimates from the Intel laptop benchmark, not
direct measurements of Raspberry Pi or truck-device electrical power.
