# B1 — Constraint Triangle Application

## 1. Decision Context

The three Constraint Triangle vertices are **performance**, **power**, and **cost**. For FreightBridge, the 90-second alert deadline is a mandatory safety gate. Once that gate is satisfied, the differentiating hard constraint is the **10 W AI power budget** supplied from the truck's 12 V electrical system through a DC-DC converter. Therefore, **power efficiency is the dominant constraint vertex**, while fleet cost and preservation of the completed software stack determine the preferred power-compliant option.

## 2. Given Hardware Data

| Option | Hardware | Unit price | TDP / stated operating power |
|---|---|---:|---:|
| 1 | Raspberry Pi 5, 8 GB, with 13-TOPS AI HAT+ | INR 15,000 | 7.5 W |
| 2 | Jetson Orin Nano Super Developer Kit, 67 TOPS | INR 45,000 | 15 W at moderate load |
| 3 | STM32H7 custom MCU with sensor ICs | INR 3,500 | 0.4 W |

The fleet sizes are:

```text
Pilot fleet = 85 trucks
Full-scale fleet = 265 trucks
AI power budget = 10 W per truck
Alert deadline = 90 seconds = 90,000 ms
```

## 3. Fleet-Cost Calculations

The fleet hardware cost is:

```text
Fleet cost = unit price × number of trucks
```

### 3.1 Raspberry Pi 5 + AI HAT+

```text
Pilot cost
= INR 15,000/truck × 85 trucks
= INR 1,275,000
= INR 12.75 lakh
```

```text
Full-scale cost
= INR 15,000/truck × 265 trucks
= INR 3,975,000
= INR 39.75 lakh
```

### 3.2 Jetson Orin Nano Super

```text
Pilot cost
= INR 45,000/truck × 85 trucks
= INR 3,825,000
= INR 38.25 lakh
```

```text
Full-scale cost
= INR 45,000/truck × 265 trucks
= INR 11,925,000
= INR 119.25 lakh
= INR 1.1925 crore
```

### 3.3 STM32H7 Custom MCU

```text
Pilot cost
= INR 3,500/truck × 85 trucks
= INR 297,500
= INR 2.975 lakh
```

```text
Full-scale cost
= INR 3,500/truck × 265 trucks
= INR 927,500
= INR 9.275 lakh
```

## 4. Power-Budget Calculations

The nominal power margin is:

```text
Power margin = 10 W AI budget − device TDP
```

### 4.1 Raspberry Pi 5 + AI HAT+

```text
Power margin
= 10 W − 7.5 W
= 2.5 W within budget
```

Percentage of AI budget used:

```text
(7.5 W ÷ 10 W) × 100
= 75%
```

The Raspberry Pi option therefore retains 25% of the stated AI power budget as nominal margin. The final electrical design must still account for DC-DC losses and any loads outside the quoted AI-node figure.

### 4.2 Jetson Orin Nano Super

```text
Power margin
= 10 W − 15 W
= −5 W
```

```text
Power-budget overrun
= 15 W − 10 W
= 5 W over budget
```

Percentage of AI budget used:

```text
(15 W ÷ 10 W) × 100
= 150%
```

At the assignment's stated moderate-load operating point, the Jetson exceeds the allowed AI power budget by 50%.

### 4.3 STM32H7 Custom MCU

```text
Power margin
= 10 W − 0.4 W
= 9.6 W within budget
```

Percentage of AI budget used:

```text
(0.4 W ÷ 10 W) × 100
= 4%
```

The STM32H7 has the strongest power position and retains 96% nominal margin.

## 5. Cost-Difference Calculations

### 5.1 Raspberry Pi Saving Relative to Jetson

Per-truck saving:

```text
INR 45,000 − INR 15,000
= INR 30,000 per truck
```

Pilot saving:

```text
INR 30,000 × 85
= INR 2,550,000
= INR 25.50 lakh
```

Full-scale saving:

```text
INR 30,000 × 265
= INR 7,950,000
= INR 79.50 lakh
```

### 5.2 Raspberry Pi Premium Relative to STM32H7

Per-truck premium:

```text
INR 15,000 − INR 3,500
= INR 11,500 per truck
```

Pilot premium:

```text
INR 11,500 × 85
= INR 977,500
= INR 9.775 lakh
```

Full-scale premium:

```text
INR 11,500 × 265
= INR 3,047,500
= INR 30.475 lakh
```

The STM32H7 has the lowest purchase cost, but this comparison excludes the engineering and assurance cost of replacing the completed Linux deployment stack with an MCU-specific design.

## 6. Constraint-Triangle Summary

| Option | Performance vertex | Power vertex | Cost vertex | Overall assessment |
|---|---|---|---|---|
| Raspberry Pi 5 + AI HAT+ | Sufficient for the 90-second SLA; 13-TOPS accelerator provides future headroom | 7.5 W, within the 10 W limit | Mid-range: INR 12.75 lakh pilot, INR 39.75 lakh full scale | Best balance |
| Jetson Orin Nano Super | Highest capability at 67 TOPS, but excessive for the present six-feature MLP | 15 W, 5 W over budget | Highest: INR 38.25 lakh pilot, INR 1.1925 crore full scale | Reject |
| STM32H7 custom MCU | Model may fit, but complete operational stack requires redesign | Best: 0.4 W | Lowest: INR 2.975 lakh pilot, INR 9.275 lakh full scale | Reject for pilot |

## 7. Option Evaluation

### 7.1 Raspberry Pi 5 + AI HAT+

The Raspberry Pi option provides the strongest balance across all three vertices. At 7.5 W it stays 2.5 W below the AI ceiling. Its Linux environment preserves the implemented Mosquitto, Docker, TFLite, PSI-monitoring, local-registry, and Ansible workflow. The current model is a small `6 → 32 → 16 → 3` MLP, and the end-to-end system has a 90,000 ms alert deadline. The first feature window consumes 30 seconds, leaving approximately 60,000 ms for inference, messaging, alert generation, and safety margin.

The F2 laptop benchmark measured an M1 p95 inference latency of 0.002909 ms. That value is not a Raspberry Pi benchmark and must not be presented as one; however, it confirms that model execution is tiny relative to the SLA. The 13-TOPS Hailo-8L accelerator provides future capacity, although the present TFLite artifact would require a Hailo-compatible conversion and runtime path to execute on the accelerator directly.

### 7.2 Jetson Orin Nano Super

The Jetson offers the highest compute capability, but the current six-feature model does not require 67 TOPS. More importantly, its stated 15 W moderate-load value exceeds the 10 W AI budget by 5 W. It also costs three times as much per truck as the Raspberry Pi option. The pilot would cost INR 38.25 lakh and the full deployment INR 1.1925 crore, without a meaningful benefit against the 90-second deadline. The Jetson is therefore rejected on power and cost.

### 7.3 STM32H7 Custom MCU

The STM32H7 is strongest on power and hardware price. Its 0.4 W consumption is comfortably inside the budget, and the 4.781–5.477 KB model files are not themselves prohibitive. However, the deployed system includes a local Mosquitto broker, Dockerized inference, runtime model switching, PSI monitoring, file-based evidence, a local Docker registry, and Ansible deployment. Reproducing these capabilities on an MCU would require an RTOS or bare-metal redesign, embedded MQTT, TFLite Micro or an equivalent runtime, static memory planning, secure boot, a custom OTA mechanism, and new safety and cybersecurity validation. Those engineering costs are not included in the INR 3,500 unit estimate.

## 8. Recommendation

FreightBridge should select the **Raspberry Pi 5, 8 GB, with the 13-TOPS AI HAT+** for the 85-truck pilot and the planned 265-truck deployment. Power is the dominant Constraint Triangle vertex because the device must remain inside the 10 W AI budget while operating from the truck's 12 V supply through a DC-DC converter. The Raspberry Pi option satisfies that limit at 7.5 W, provides ample performance for the 90-second alert SLA, and preserves the completed Linux deployment and MLOps toolchain.

The Jetson is rejected because its 15 W moderate-load value exceeds the power ceiling and its additional compute is unnecessary. The STM32H7 is rejected for the pilot because its attractive unit price and power consumption would be offset by a substantial software, safety, and security redesign. The Raspberry Pi therefore offers the lowest-risk balance of performance, power, fleet cost, and implementation continuity.
