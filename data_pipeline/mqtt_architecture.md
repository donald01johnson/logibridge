# MQTT Architecture and Data-Fusion Justification

## 1. Purpose

The LogiBridge MQTT pipeline transports cold-chain sensor readings between
the truck sensor simulator, the edge preprocessing service, the inference
service, the drift monitor, and downstream alert consumers.

The transport layer is designed to support local operation on the truck.
Sensor collection, preprocessing, feature extraction, model inference, and
alert generation can continue while the cellular uplink is unavailable.

## 2. MQTT Components

The MQTT architecture contains the following logical components:

1. Sensor simulator or physical sensor gateway
2. Local Mosquitto broker
3. Preprocessing and feature-fusion service
4. Edge inference service
5. Drift-monitoring service
6. Local alert and event log
7. Optional cellular operations-centre uplink

The local broker decouples data producers from consumers. Each service can
publish or subscribe independently without requiring direct point-to-point
connections between every component.

## 3. Topic Hierarchy

The topic hierarchy is scoped by truck identifier.

    logibridge/trucks/{truck_id}/status
    logibridge/trucks/{truck_id}/sensors/temperature
    logibridge/trucks/{truck_id}/sensors/vibration
    logibridge/trucks/{truck_id}/sensors/door
    logibridge/trucks/{truck_id}/sensors/combined
    logibridge/trucks/{truck_id}/features
    logibridge/trucks/{truck_id}/inference
    logibridge/trucks/{truck_id}/alerts
    logibridge/trucks/{truck_id}/monitoring/psi

For example:

    logibridge/trucks/TRUCK-001/sensors/temperature
    logibridge/trucks/TRUCK-001/features
    logibridge/trucks/TRUCK-001/inference

Truck-specific topic roots prevent readings from different vehicles from
being mixed. A fleet-level subscriber can use the wildcard topic:

    logibridge/trucks/+/inference

A subscriber that requires all events from one truck can use:

    logibridge/trucks/TRUCK-001/#

## 4. Publisher and Subscriber Responsibilities

### Sensor simulator

The sensor simulator publishes:

- Temperature readings
- Vibration RMS readings
- Door-state events
- A synchronized combined sensor state
- Retained ONLINE and OFFLINE status

### Preprocessing service

The preprocessing service subscribes to:

    logibridge/trucks/{truck_id}/sensors/combined

It applies:

- Five-sample moving-average filtering
- Thirty-second windowing
- Ten-second window stepping
- Six-feature extraction
- Training-statistics normalization

It publishes processed windows to:

    logibridge/trucks/{truck_id}/features

### Inference service

The inference service will subscribe to the feature topic and publish class
probabilities and the predicted Normal, Warning, or Critical state to:

    logibridge/trucks/{truck_id}/inference

### Drift-monitoring service

The drift monitor will observe feature or inference messages and publish
Population Stability Index results to:

    logibridge/trucks/{truck_id}/monitoring/psi

### Alert consumer

The local alert service will consume Critical predictions and publish or log
alerts under:

    logibridge/trucks/{truck_id}/alerts

## 5. Payload Design

Messages use JSON because it is readable during development, simple to
inspect with Mosquitto command-line tools, and sufficiently expressive for
timestamps, identifiers, units, and numeric measurements.

A sensor message includes:

- Schema version
- Truck identifier
- Sensor name
- UTC timestamp
- Sequence number
- Anomaly mode
- Numeric value and unit

A feature-window message includes:

- Truck identifier
- Window sequence number
- Window start and end timestamps
- Sample count
- Feature names
- Six raw features
- Six normalized features when statistics are available

The schema-version field allows future payload revisions while retaining
compatibility checks.

## 6. Sampling and Windowing

Temperature is sampled at 1 Hz.

Vibration RMS is sampled at 0.5 Hz.

The simulator publishes the latest synchronized combined state at 1 Hz. The
preprocessor consumes this combined state so that temperature, vibration,
and door context are represented on a common timeline.

The preprocessor applies a five-sample moving average to temperature and
vibration. It then forms 30-second sliding windows with a 10-second step.

Each completed window produces one six-value feature vector.

## 7. Feature-Level Data Fusion

LogiBridge uses feature-level fusion rather than raw-sample concatenation or
decision-level fusion.

The fused feature vector is:

1. Mean filtered temperature
2. Maximum filtered temperature
3. Mean filtered vibration RMS
4. Maximum filtered vibration RMS
5. Door-open fraction
6. Door transition count

Feature-level fusion is appropriate because the three sensors provide
complementary evidence about one cargo condition.

Temperature features represent refrigeration performance and thermal drift.

Vibration features represent abnormal mechanical or vehicle motion.

Door features provide operational context. A temperature increase combined
with an open door can have a different interpretation from a temperature
increase while the door remains closed.

The fused vector is compact, fixed length, and suitable for the six-input
multilayer perceptron required by the project. It also reduces the volume of
data passed to the inference service compared with transmitting an entire
raw time window.

## 8. MQTT Quality of Service

QoS 1 is used for sensor, feature, inference, monitoring, and alert messages.

QoS 1 provides at-least-once delivery. This is preferred over QoS 0 because
sensor windows and Critical predictions should not be silently discarded
during brief local interruptions.

QoS 2 is not selected because its additional handshake and state-management
overhead is unnecessary for this telemetry pipeline.

Because QoS 1 can produce duplicate messages, consumers should use fields
such as truck identifier, timestamp, and sequence number to detect repeated
events when strict deduplication is required.

## 9. Retained Messages

Retained publication is suitable for truck status and latest door state.
A new subscriber can immediately learn the most recently published state.

High-rate temperature, vibration, combined-sensor, and feature messages are
not retained because consumers require the current stream rather than stale
historical windows.

## 10. Offline Operation

All safety-critical processing is local to the truck:

    Sensors
      |
      v
    Local Mosquitto Broker
      |
      v
    Preprocessing
      |
      v
    Local Inference
      |
      v
    Local Alert and Event Log

The cellular uplink is not required for sensor acquisition, preprocessing,
classification, or local alerts.

When connectivity is available, selected predictions, alerts, drift metrics,
and summaries can be forwarded to the operations centre.

When connectivity is unavailable, the edge pipeline continues operating.
Messages intended for the operations centre should be stored locally with
timestamps and sequence identifiers, then transmitted after connectivity is
restored.

## 11. Privacy and Data Minimisation

Raw high-frequency sensor data remains local unless operational policy
requires transmission.

The operations centre can receive compact feature vectors, predictions,
alerts, and monitoring summaries instead of every raw measurement.

This reduces transmitted data and limits unnecessary exposure of detailed
vehicle telemetry.

## 12. Failure Handling

The implementation includes the following controls:

- MQTT connection failures are logged.
- Invalid JSON messages are rejected.
- Missing or invalid sensor fields are rejected.
- Door state accepts only OPEN or CLOSE.
- Sensor timestamps must be non-decreasing.
- Feature vectors are checked for finite values.
- Saved training statistics are validated before normalization.
- Services handle SIGINT and SIGTERM for graceful shutdown.

Future deployment hardening will add persistent broker storage, bounded
offline queues, authentication, access-control lists, and encrypted uplink
communication.

## 13. Validated End-to-End Flow

The implemented validation flow is:

    simulator.py
        |
        | publishes combined JSON sensor state
        v
    Mosquitto
        |
        | topic:
        | logibridge/trucks/TRUCK-001/sensors/combined
        v
    preprocessing.py
        |
        | five-sample filtering
        | 30-second window
        | 10-second step
        | six-feature extraction
        | normalization
        v
    Mosquitto
        |
        | topic:
        | logibridge/trucks/TRUCK-001/features
        v
    Feature subscriber

The self-test also validates saved statistics and compares correct
normalization with deliberately three-standard-deviation-shifted means.

## 14. Component C Conclusion

The implemented sensor pipeline provides local MQTT transport, smoothing,
windowing, synchronized feature extraction, normalization, and feature-level
fusion.

The design supports offline edge inference while preserving a structured
path for later fleet-level monitoring and operations-centre synchronization.
