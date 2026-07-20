# B2 — Arithmetic Intensity and Roofline Analysis

## 1. Given Values

The assignment provides the following approximate values for the Roofline analysis:

```text
Operations per inference = 45 MFLOPs
Data accessed per inference = 18 MB
Raspberry Pi 5 CPU peak compute = 16 GFLOP/s
Raspberry Pi 5 memory bandwidth = 12 GB/s
```

Decimal SI units are used consistently:

```text
1 MFLOP = 1,000,000 FLOPs
1 MB = 1,000,000 bytes
1 GFLOP/s = 1,000,000,000 FLOPs/s
1 GB/s = 1,000,000,000 bytes/s
```

## 2. Arithmetic Intensity

Arithmetic Intensity is the number of operations performed for each byte transferred:

```text
Arithmetic Intensity = FLOPs per inference / bytes accessed per inference
```

Substituting the supplied values:

```text
Arithmetic Intensity
= 45,000,000 FLOPs / 18,000,000 bytes
= 2.5 FLOP/byte
```

Therefore, the model's arithmetic intensity is:

```text
AI = 2.5 FLOP/byte
```

## 3. Ridge Point

The Roofline ridge point is where the compute-performance ceiling and memory-bandwidth ceiling intersect:

```text
Ridge point = peak compute throughput / peak memory bandwidth
```

Substituting the Raspberry Pi 5 values:

```text
Ridge point
= 16,000,000,000 FLOP/s / 12,000,000,000 byte/s
= 1.3333 FLOP/byte
```

Therefore:

```text
Ridge point ≈ 1.33 FLOP/byte
```

## 4. Roofline Classification

Compare the model's arithmetic intensity with the ridge point:

```text
Model AI = 2.5 FLOP/byte
Ridge point = 1.33 FLOP/byte
```

```text
2.5 > 1.33
```

The model lies to the right of the ridge point. Under the supplied Roofline assumptions, it is therefore **compute-bound**, not memory-bandwidth-bound.

The ratio is:

```text
2.5 / 1.3333 ≈ 1.875
```

Thus, the model's arithmetic intensity is approximately 1.875 times the ridge-point value.

## 5. Performance-Ceiling Check

The memory-bandwidth performance ceiling at the model's arithmetic intensity is:

```text
Memory ceiling
= memory bandwidth × arithmetic intensity
= 12 GB/s × 2.5 FLOP/byte
= 30 GFLOP/s
```

The compute ceiling is:

```text
Compute ceiling = 16 GFLOP/s
```

The Roofline performance is the lower of the two ceilings:

```text
Attainable performance ceiling
= min(16 GFLOP/s, 30 GFLOP/s)
= 16 GFLOP/s
```

Because the 16 GFLOP/s compute ceiling is reached before the 30 GFLOP/s memory ceiling, compute throughput limits performance.

## 6. Theoretical Lower-Bound Latency

The compute-time lower bound is:

```text
Compute time
= 45,000,000 FLOPs / 16,000,000,000 FLOP/s
= 0.0028125 seconds
= 2.8125 ms
```

The memory-transfer lower bound is:

```text
Memory time
= 18,000,000 bytes / 12,000,000,000 byte/s
= 0.0015 seconds
= 1.5 ms
```

The larger lower bound dominates:

```text
Roofline latency lower bound
= max(2.8125 ms, 1.5 ms)
= 2.8125 ms
```

This is a theoretical bound based on the assignment's approximate peak values. Real latency can be higher because of instruction efficiency, cache behavior, framework overhead, operating-system scheduling, and inability to sustain peak throughput.

## 7. Optimisation Implication

Because the model is compute-bound under the supplied assumptions, the most direct latency improvements are those that reduce arithmetic work or increase effective compute throughput:

1. Use NEON-vectorised and fused inference kernels so the Raspberry Pi CPU uses its SIMD units effectively.
2. Apply INT8 quantisation only when the selected Raspberry Pi runtime provides an efficient integer execution path; quantisation can reduce arithmetic cost, but the actual hardware/runtime must be benchmarked.
3. Reduce FLOPs through a smaller network, structured unit removal, operator simplification, or architecture redesign.
4. Offload compatible inference operations to the 13-TOPS Hailo-8L accelerator after converting the model to a supported Hailo deployment format.
5. Use operator fusion and compile-time optimisation to reduce instruction and dispatch overhead.

Increasing memory bandwidth alone is not the primary Roofline recommendation because the model's theoretical memory ceiling of 30 GFLOP/s is already above the CPU compute ceiling of 16 GFLOP/s. Reducing unnecessary data movement may still improve energy and practical efficiency, but it does not change the primary compute-bound classification unless Arithmetic Intensity or hardware balance changes substantially.

## 8. Conclusion

```text
Arithmetic Intensity = 2.5 FLOP/byte
Ridge point = 1.33 FLOP/byte
Model position = right of ridge point
Classification = compute-bound
Roofline performance ceiling = 16 GFLOP/s
Theoretical latency lower bound = 2.8125 ms
```

The Roofline model therefore indicates that LogiEdge latency should be improved primarily through compute-side optimisation: efficient SIMD or INT8 kernels, fewer operations, operator fusion, or accelerator offload. A memory-bandwidth upgrade alone would not address the dominant theoretical bottleneck.

## 9. Scope Note

This B2 analysis uses the assignment-prescribed approximations of 45 MFLOPs and 18 MB per inference. These values are not derived from the final small `6 → 32 → 16 → 3` MLP artifact benchmarked in Component F. Consequently, the 2.8125 ms Roofline lower bound must not be compared directly with the F2 laptop latency measurements; the two sections describe different stated workloads and hardware assumptions.
