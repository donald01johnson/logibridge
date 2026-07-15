# E3 — OTA Strategy Selection

## Inputs

- Update interval: every 6 weeks
- INT8 TFLite model size: 280 KB
- Fleet size: 85 trucks
- M2M tariff: INR 0.10 per MB
- Conversion: 1,000 KB equals 1 MB
- Model size per truck: 0.28 MB
- Transfer cost per truck: INR 0.028

## Full Replacement

The updated model is sent to all 85 trucks in one deployment.

    Bandwidth = 0.28 MB * 85 = 23.80 MB
    Cost = 23.80 MB * INR 0.10/MB = INR 2.38

Full replacement is simple and keeps the fleet on one version, but it exposes
the entire pilot fleet immediately if the model contains a regression. Its
fleet-wide blast radius is unsuitable as the default for safety-critical
pharmaceutical cold-chain monitoring.

## Canary Deployment

The model is first deployed to ten trucks.

    Canary bandwidth = 0.28 MB * 10 = 2.80 MB
    Canary cost = 2.80 MB * INR 0.10/MB = INR 0.28

If the canary succeeds, the model is deployed to the remaining 75 trucks.

    Expansion bandwidth = 0.28 MB * 75 = 21.00 MB
    Expansion cost = 21.00 MB * INR 0.10/MB = INR 2.10

The successful complete cycle is therefore:

    Total bandwidth = 2.80 MB + 21.00 MB = 23.80 MB
    Total cost = INR 0.28 + INR 2.10 = INR 2.38

If validation fails, the rollout can stop after 2.80 MB and INR 0.28 of
transfer, leaving the remaining 75 trucks on the validated production model.

## Shadow Mode

The new model is transferred to all 85 trucks but does not control alerts.
The existing production model remains authoritative while shadow predictions
are logged.

    Bandwidth = 0.28 MB * 85 = 23.80 MB
    Cost = 23.80 MB * INR 0.10/MB = INR 2.38

Shadow mode has no new-model actuation risk during validation, but it requires
the old and new models to run concurrently. This increases compute, memory,
energy, logging, and monitoring overhead on every truck.

## Comparison

| Strategy | Initial bandwidth | Initial cost | Full-cycle bandwidth | Full-cycle cost |
|---|---:|---:|---:|---:|
| Full replacement | 23.80 MB | INR 2.38 | 23.80 MB | INR 2.38 |
| Canary, 10 trucks first | 2.80 MB | INR 0.28 | 23.80 MB after promotion | INR 2.38 after promotion |
| Shadow mode | 23.80 MB | INR 2.38 | 23.80 MB | INR 2.38 |

All strategies use the same bandwidth when the new model eventually reaches
all 85 trucks. The deciding factors are therefore safety, blast radius,
runtime overhead, rollback scope, and connectivity behavior rather than the
small cellular-data charge.

## Recommendation

FreightBridge should use a ten-truck canary deployment for routine six-week
updates. It limits the initial safety blast radius, supports rollback before
fleet-wide promotion, avoids the dual-model runtime overhead of shadow mode,
and has no additional full-cycle data cost relative to full replacement.

This strategy also fits rural connectivity conditions. A truck that is
temporarily offline continues using the previous validated local model and
downloads the update when coverage returns. The rollout therefore does not
depend on simultaneous connectivity across the fleet.

Full replacement is rejected as the routine strategy because it exposes all
85 trucks simultaneously without production validation. Shadow mode is
rejected for ordinary same-architecture updates because executing two models
on every truck increases compute, memory, energy, and monitoring overhead.

Shadow mode should still be required for major architecture changes or models
with unknown failure modes, while full replacement remains suitable for first
installation or an already field-validated emergency fix.

## Calculation Limitation

The calculation covers the 280 KB model payload only. Actual cellular usage
can be higher because of transport headers, registry metadata, encryption,
retries, or interrupted downloads.
