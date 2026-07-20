# Component B — Hardware Selection and Justification

## 1. Selected Edge Platform

FreightBridge should deploy the Raspberry Pi 5, 8 GB, with the
13-TOPS AI HAT+ as the on-truck edge node.

The detailed Constraint Triangle comparison is documented in:

- constraint_triangle.md

The Arithmetic Intensity and Roofline analysis is documented in:

- [`roofline_analysis.md`](roofline_aint Triangle Conclusion

The dominant constraint vertex is power efficiency. The edge node must
operate from the truck's 12 V supply through a DC-DC converter while
remaining within the 10 W AI power budget.

The Raspberry Pi 5 with AI HAT+ consumes 7.5 W under the assignment
assumption:

    Power margin
    = 10 W - 7.5 W
    = 2.5 W within budget

The Jetson Orin Nano Super consumes 15 W at the stated moderate-load
operating point:

    Power overrun
    = 15 W - 10 W
    = 5 W over budget

The STM32H7 consumes only 0.4 W but would require the implemented Linux,
Docker, Mosquitto, TFLite, PSI, registry, and Ansible deployment stack to be
redesigned for an MCU environment.

## 3. Fleet Cost Summary

| Hardware | 85-truck pilot | 265-truck full scale |
|---|---:|---:|
| Raspberry Pi 5 + AI HAT+ | INR 1,275,000 | INR 3,975,000 |
| Jetson Orin Nano Super | INR 3,825,000 | INR 11,925,000 |
| STM32H7 custom MCU | INR 297,500 | INR 927,500 |

The Raspberry Pi option saves INR 2,550,000 relative to Jetson for the pilot
and INR 7,950,000 at full scale.

Although STM32H7 has the lowest hardware cost, its price does not include the
engineering, cybersecurity, safety-validation, and maintenance cost of
reimplementing the completed operational stack.

## 4. Latency Suitability

The system must detect and alert within 90 seconds. The initial feature
window consumes 30 seconds, leaving approximately 60 seconds for inference,
local messaging, alert generation, and safety margin.

The final model is small, and model execution is not expected to dominate the
end-to-end response time. The Raspberry Pi option therefore provides adequate
performance without the excessive compute capability, power consumption, and
cost of the Jetson platform.

The F2 laptop benchmark is evidence about the model variants on the Intel
test system and is not a direct Raspberry Pi benchmark. Physical-device
validation remains necessary before production deployment.

## 5. Roofline Conclusion

Using the assignment-prescribed workload:

    Arithmetic Intensity
    = 45 MFLOPs / 18 MB
    = 2.5 FLOP/byte

The Raspberry Pi 5 ridge point is:

    Ridge point
    = 16 GFLOP/s / 12 GB/s
    = 1.33 FLOP/byte

Because:

    2.5 FLOP/byte > 1.33 FLOP/byte

the workload is classified as compute-bound under the supplied Roofline
assumptions.

The memory-bandwidth ceiling is:

    12 GB/s x 2.5 FLOP/byte
    = 30 GFLOP/s

The compute ceiling is only 16 GFLOP/s, so compute throughput is reached
before memory bandwidth becomes limiting.

## 6. Optimization Implication

The Roofline result indicates that latency should primarily be improved by:

- Using efficient NEON SIMD kernels
- Reducing model FLOPs
- Applying operator fusion
- Using an optimized INT8 execution path where physical-device benchmarks
  show a benefit
- Converting compatible models for Hailo-8L accelerator execution

Increasing memory bandwidth alone is not the primary recommendation because
the theoretical memory ceiling already exceeds the CPU compute ceiling.

## 7. Final Hardware Decision

The Raspberry Pi 5 with the 13-TOPS AI HAT+ provides the best overall balance:

- It remains within the 10 W AI power budget.
- It offers sufficient performance for the 90-second safety requirement.
- It costs substantially less than Jetson at both fleet scales.
- It preserves the completed Linux-based MQTT, Docker, TFLite, PSI,
  registry, OTA, and Ansible architecture.
- It provides future accelerator capacity for more demanding models.

Jetson is rejected because its stated operating power exceeds the project
budget and its compute capacity is unnecessary for the current model.

STM32H7 is rejected for the pilot because the hardware savings would require
a substantial embedded-system redevelopment and revalidation effort.
