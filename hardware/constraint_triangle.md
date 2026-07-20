# B1 — Constraint Triangle Application

## Decision Context

The hardware decision is evaluated against the three Constraint Triangle vertices: performance, power, and cost. Performance is a mandatory safety gate because the system must alert within 90 seconds. However, all viable options must also operate from the truck's 12 V supply through a DC-DC converter and remain within the 10 W AI power budget. For this deployment, **power efficiency is the dominant vertex**, with latency treated as a non-negotiable acceptance condition and cost used to distinguish fleet-scale alternatives.

## Fleet Cost and Power Comparison

| Option | Unit cost | 85-truck pilot | 265-truck full scale | TDP | Margin to 10 W budget |
|---|---:|---:|---:|---:|---:|
| Raspberry Pi 5 8 GB + AI HAT+ 13 TOPS | INR 15,000 | INR 1,275,000 | INR 3,975,000 | 7.5 W | 2.5 W within budget |
| Jetson Orin Nano Super | INR 45,000 | INR 3,825,000 | INR 11,925,000 | 15 W | 5 W over budget |
| STM32H7 custom MCU | INR 3,500 | INR 297,500 | INR 927,500 | 0.4 W | 9.6 W within budget |

## Option 1 — Raspberry Pi 5 + AI HAT+

The Raspberry Pi option provides the best balance across all three vertices. At 7.5 W it stays 2.5 W below the AI power ceiling, leaving some allowance within the specified budget for conversion losses and operating variation. Its Linux environment supports the implemented Mosquitto, Docker, TFLite, PSI-monitoring, local-registry, and Ansible workflow without a complete software rewrite. The measured M1 p95 inference latency was 0.002909 ms on the development laptop; actual Raspberry Pi latency will differ, but the 90-second end-to-end SLA is sufficiently large that the 30-second feature window, rather than model execution, is expected to dominate. The 13-TOPS Hailo-8L accelerator also provides capacity for future models, although the current TFLite MLP would require a Hailo-compatible deployment path to use that accelerator directly.

## Option 2 — Jetson Orin Nano Super

The Jetson offers the highest compute capability at 67 TOPS, but that performance is unnecessary for the current six-feature MLP. Its stated 15 W moderate-load value exceeds the 10 W AI budget by 5 W, making power the decisive rejection criterion. It is also three times the Raspberry Pi's unit cost: the pilot costs INR 3.825 million and full deployment INR 11.925 million. Compared with Raspberry Pi, this adds INR 2.55 million for the pilot and INR 7.95 million at full scale without providing a meaningful benefit against the 90-second SLA.

## Option 3 — STM32H7 Custom MCU

The STM32H7 is strongest on power and cost, consuming only 0.4 W and costing INR 0.2975 million for the pilot. Its MCU-class memory can support small embedded inference, and the 4.781–5.477 KB model files are not themselves prohibitive. However, the deployed solution is more than the model: it includes MQTT brokering, Dockerized inference, runtime model switching, PSI monitoring, local evidence storage, registry-based OTA, and Ansible deployment. Reproducing this Linux-based operational stack on a custom MCU would require a substantial redesign using an RTOS, embedded MQTT, TFLite Micro, and a separate secure update mechanism. That development, validation, cybersecurity, and maintenance burden is not reflected in the INR 3,500 hardware estimate.

## Recommendation

FreightBridge should deploy the **Raspberry Pi 5 8 GB with the 13-TOPS AI HAT+**. The platform satisfies the hard 10 W budget, offers ample performance for the 90-second alert SLA, and preserves the completed Linux deployment and MLOps toolchain. Jetson is rejected because it exceeds both the power ceiling and the justified compute requirement. STM32H7 is rejected for the pilot because its low hardware cost would be offset by a major software and assurance redesign. The Raspberry Pi therefore provides the lowest-risk balance of performance, power, fleet cost, and implementation continuity.
