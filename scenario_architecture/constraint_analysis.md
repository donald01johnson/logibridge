# A1 — Constraint Analysis

## Latency

A refrigeration failure raises cargo temperature by 1°C per minute, so a
90-second response permits a 1.5°C rise before alerting. India's May 2026
national mobile median latency was 26 ms; therefore, when a stable connection
exists, a single network round trip is far below the 90-second limit. However,
this national figure is not a measurement of the specific rural route, and a
cloud-only path also includes uplink transmission, server queuing, inference,
downlink delivery, and retries.

The specified 35–90-minute route outages equal 2,100–5,400 seconds, or
approximately 23–60 times the complete SLA. Cloud inference is therefore
feasible only while connected and cannot guarantee the safety deadline.
LogiEdge performs 30-second windowing, TFLite inference, and local alert
generation inside the truck, leaving approximately 60 seconds of the SLA
after the first feature window.

## Bandwidth

The calculation uses explicit binary-record assumptions: one float32
temperature value requires 4 bytes, while three float32 vibration axes require
12 bytes per 500 Hz sample. Each door event is represented by an 8-byte
timestamp and a 1-byte state.

Temperature produces:

    1 × 86,400 × 4 = 345,600 bytes/day = 0.3456 MB/day

Vibration produces:

    500 × 86,400 × 12 = 518,400,000 bytes/day = 518.4 MB/day

Door traffic is 9N bytes for N events. Assuming 20 events per day gives
180 bytes. Total raw payload is approximately 518.746 MB per truck per day,
costing:

    518.746 MB × INR 0.10/MB = INR 51.87 per truck per day

For comparison, a conservative edge scenario containing 100 one-kilobyte
alert summaries per day uses only 0.10 MB and costs INR 0.01, a 99.98 percent
payload reduction. The door-event count and alert volume are stated
assumptions because the assignment does not prescribe them. MQTT, TLS,
metadata, and retransmission overhead are excluded.

## Connectivity

During the seven documented 35–90-minute dead zones, a cloud-only system
cannot upload sensor data, receive predictions, or issue cloud-generated
alerts. LogiEdge continues local MQTT acquisition, filtering, feature
extraction, fixed-statistics normalization, inference, and alert logging
without cellular service. Predictions and summaries are buffered and
synchronized with the operations centre after connectivity returns.

## Privacy

On-device inference keeps raw pharmaceutical cargo telemetry inside the truck
instead of continuously transmitting temperature, vibration, and door
histories to external infrastructure. Only authorized alerts, health
summaries, and synchronization records need to leave the vehicle, supporting
data-minimization commitments and reducing third-party exposure.

Access-controlled local storage, encrypted communication, signed updates, and
auditable synchronization should complement on-device inference to provide
contractual evidence that cargo-condition data is handled only by authorized
parties.
