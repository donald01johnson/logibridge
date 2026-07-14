
# MQTT Architecture and Feature-Level Data Fusion

## 1. Purpose

LogiBridge uses a local MQTT architecture to connect the cold-chain sensor
simulator, preprocessing pipeline, inference service, drift monitor, and
alert-handling components.

The Mosquitto broker runs locally on the truck edge node. Temperature,
vibration, and door-event messages therefore remain available to local
services even when cellular connectivity to the operations centre is
unavailable.

## 2. Sensor Streams

The simulator produces the following three streams.

| Stream | Frequency | Normal behaviour | Anomaly behaviour |
|---|---:|---|---|
| Temperature | 1 Hz | Normal distribution with mean 4.0 degrees Celsius and standard deviation 0.3 | Linear increase of 0.08 degrees Celsius per reading |
| Vibration RMS | 0.5 Hz | Normal distribution with mean 0.45 g and standard deviation 0.05 | Step change to a normal distribution with mean 1.2 g and standard deviation 0.15 |
| Door event | Discrete | OPEN or CLOSE event with timestamp | Retained as operational context |

The supported simulator modes are:

- `none`
- `temp_drift`
- `vibration`
- `combined`

The `combined` mode simultaneously activates temperature drift and anomalous
vibration.

## 3. MQTT Topic Hierarchy

Topics are scoped by truck identifier so that messages from different
vehicles remain separated.

    logibridge/trucks/{truck_id}/status
    logibridge/trucks/{truck_id}/sensors/temperature
    logibridge/trucks/{truck_id}/sensors/vibration
    logibridge/trucks/{truck_id}/sensors/door
    logibridge/trucks/{truck_id}/sensors/combined
    logibridge/trucks/{truck_id}/features
    logibridge/trucks/{truck_id}/inference
    logibridge/trucks/{truck_id}/alerts
    logibridge/trucks/{truck_id}/monitoring/psi

Example topics for `TRUCK-001` are:

    logibridge/trucks/TRUCK-001/sensors/temperature
    logibridge/trucks/TRUCK-001/sensors/vibration
    logibridge/trucks/TRUCK-001/sensors/door
    logibridge/trucks/TRUCK-001/features
    logibridge/trucks/TRUCK-001/inference

A fleet-level consumer can subscribe using:

    logibridge/trucks/+/inference

A consumer requiring all messages for one truck can subscribe using:

    logibridge/trucks/TRUCK-001/#

## 4. MQTT Producers and Consumers

### Sensor simulator

The sensor simulator publishes temperature, vibration, door events, combined
sensor state, and simulator status.

### Preprocessing service

The preprocessing service consumes the sensor values, applies filtering,
extracts the six required features, loads the fixed normalisation statistics,
and prepares the model input.

### Inference service

The inference service consumes a six-value normalised feature vector, performs
local model inference, and publishes the predicted class and confidence values
to:

    logibridge/trucks/{truck_id}/inference

### Drift monitor

The drift-monitoring service observes model confidence scores and publishes
Population Stability Index results to:

    logibridge/trucks/{truck_id}/monitoring/psi

### Alert consumer

Critical predictions can be written to a local alert log and published to:

    logibridge/trucks/{truck_id}/alerts

## 5. Quality of Service

QoS 1 is selected for sensor, feature, inference, monitoring, and alert
messages.

QoS 1 provides at-least-once delivery. It is preferable to QoS 0 for this
application because model-input windows and Critical predictions should not
be silently lost during a temporary local communication interruption.

QoS 2 is not selected because its additional acknowledgement and
state-management overhead is unnecessary for this local telemetry pipeline.

Because QoS 1 can deliver a duplicate message, payloads include identifiers
such as truck ID, timestamp, and sequence number. Consumers can use these
fields for deduplication when required.

## 6. Retained Messages

Retained delivery is suitable for status and latest door-state messages. A
new subscriber can immediately discover the most recently published state.

High-frequency temperature, vibration, combined-sensor, feature, and
inference messages are not retained because consumers require current stream
data rather than a stale historical window.

## 7. Preprocessing Sequence

The required preprocessing sequence is:

1. Apply a five-sample moving average independently to the temperature stream.
2. Apply a five-sample moving average independently to the vibration stream.
3. Form 30-second sliding windows.
4. Advance each window by 10 seconds.
5. Extract three temperature features.
6. Extract three vibration features.
7. Concatenate both feature groups into one six-value vector.
8. Normalise the vector using fixed statistics loaded from
   `training_stats.npy`.

Temperature arrives at 1 Hz. Vibration arrives at 0.5 Hz. The filtering
implementation advances the temperature filter for every new temperature
reading and advances the vibration filter only when a new vibration reading
is available.

A carried-forward vibration value may be used to maintain a synchronised
sensor state, but it is not inserted into the vibration filter as a second
measurement.

## 8. Temperature Feature Extraction

The filtered temperature stream produces three features per window.

### Temperature mean

The mean represents the average cargo-compartment temperature during the
30-second window.

### Temperature standard deviation

The standard deviation measures short-term thermal variability. A stable
refrigeration system should normally show limited variation around the
setpoint.

### Temperature rate of change

A least-squares linear slope is fitted across the window and converted from
degrees Celsius per second to degrees Celsius per minute.

This feature represents the direction and speed of thermal deterioration.
It is especially useful for detecting the linear `temp_drift` anomaly.

## 9. Vibration Feature Extraction

The filtered 0.5 Hz vibration stream produces three features per window.

### Vibration RMS

The root-mean-square value represents overall mechanical vibration energy.

### Vibration peak

The maximum absolute filtered vibration value captures the strongest
vibration observed during the window.

### Vibration kurtosis

Pearson kurtosis is calculated from the second and fourth central moments.
It describes the shape of the vibration-value distribution and can identify
a distribution with unusually strong tails or peaks.

## 10. Six-Value Joint Feature Vector

The final model input uses this fixed order:

    [
        temperature_mean_c,
        temperature_std_c,
        temperature_rate_c_per_min,
        vibration_rms_g,
        vibration_peak_g,
        vibration_kurtosis
    ]

The temperature feature group is extracted independently from the filtered
temperature stream:

    temperature_features = [
        temperature_mean_c,
        temperature_std_c,
        temperature_rate_c_per_min
    ]

The vibration feature group is extracted independently from the filtered
vibration stream:

    vibration_features = [
        vibration_rms_g,
        vibration_peak_g,
        vibration_kurtosis
    ]

Feature-level fusion concatenates the two groups:

    fused_features =
        concatenate(
            temperature_features,
            vibration_features
        )

The resulting six-value vector is passed to one classification model.

## 11. Role of Door Events

Door events remain part of the MQTT architecture and operational record.
They provide useful context when interpreting a temperature change and can
support alert investigation or future model extensions.

Door-derived values are not included in the current model input because the
assignment defines an exact six-value vector containing three temperature
features and three vibration features.

## 12. Normalisation

Feature means and standard deviations are computed from exactly 10 minutes,
or 600 seconds, of clean Normal-class simulator output.

The values are saved to:

    data_pipeline/training_stats.npy

The saved artifact includes:

- Feature names and order
- Mean of every feature
- Standard deviation of every feature
- Source duration
- Moving-average length
- Sliding-window duration
- Sliding-window step

Training, validation, and runtime inference load these fixed values.

The runtime pipeline never recomputes normalisation statistics from live
data. Recomputing from live data could cause the reference distribution to
move toward an ongoing fault and reduce anomaly visibility.

Normalisation is performed using:

    normalised_feature =
        (raw_feature - training_mean)
        / training_standard_deviation

## 13. Shifted-Statistics Experiment

The mandatory experiment evaluates inference under two conditions.

### Correct statistics

    correct_input =
        (raw_features - stored_mean)
        / stored_standard_deviation

### Three-sigma-shifted statistics

Each stored mean is increased by three stored standard deviations:

    shifted_mean =
        stored_mean
        + 3 * stored_standard_deviation

The same raw validation features are then normalised using:

    shifted_input =
        (raw_features - shifted_mean)
        / stored_standard_deviation

The corrected baseline MLP produced the following held-out validation
results:

| Statistics used | Validation accuracy |
|---|---:|
| Correct stored statistics | 100.00% |
| Means shifted by 3 sigma | 100.00% |
| Accuracy change | 0.00 percentage points |

The shifted condition changed the normalised inputs but did not change the
predicted class labels in the controlled synthetic validation set. The three
synthetic classes remained sufficiently separated for the classifier to
retain the same accuracy.

This result does not mean that incorrect normalisation is safe in a real
deployment. Real sensor distributions are expected to overlap more than the
controlled synthetic scenarios, and incorrect statistics could alter model
confidence or move observations across decision boundaries.

## 14. Why Feature-Level Fusion Was Selected

Feature-level fusion is appropriate because temperature and vibration provide
different but complementary evidence about refrigeration health.

Temperature mean, variability, and rate of change represent the thermal
condition of the cargo compartment.

Vibration RMS, peak, and kurtosis represent mechanical behaviour of the
refrigeration unit.

Combining these descriptors before classification allows a single model to
learn relationships between thermal and mechanical behaviour. For example,
temperature drift accompanied by abnormal vibration provides stronger
evidence of refrigeration failure than either signal considered without the
other.

The six-value input is compact, fixed-length, and suitable for inference on
a constrained edge device.

## 15. Comparison with Data-Level Fusion

Data-level fusion would combine raw temperature and vibration samples before
feature extraction.

It is less suitable for LogiBridge for the following reasons:

- Temperature and vibration use different sampling frequencies.
- Temperature and vibration use different physical units and numeric scales.
- Raw fusion requires resampling or explicit temporal alignment.
- A raw fused window contains more values than the six-feature vector.
- A larger input increases model memory and computation requirements.
- Raw samples contain noise that the moving-average and feature-extraction
  stages are intended to reduce.

Feature-level fusion allows each stream to be processed using
signal-appropriate descriptors before combination.

## 16. Comparison with Decision-Level Fusion

Decision-level fusion would use separate temperature and vibration
classifiers and combine their final predictions.

It is less suitable for LogiBridge for the following reasons:

- It requires training and maintaining at least two models.
- It increases model storage and inference workload.
- It complicates Docker packaging and over-the-air model updates.
- It complicates performance monitoring and version management.
- Cross-sensor relationships are considered only after each model has made
  an independent decision.

The feature-level approach uses one classifier and allows interactions between
thermal and mechanical features to influence the class decision directly.

## 17. Fusion-Level Justification

Feature-level fusion provides the most appropriate balance for this
cold-chain edge application because it:

- Preserves meaningful information from each sensor stream.
- Handles different sensor frequencies before fusion.
- Produces a compact six-value vector.
- Requires only one classification model.
- Reduces model input size compared with raw data-level fusion.
- Captures cross-sensor relationships earlier than decision-level fusion.
- Simplifies deployment, monitoring, and over-the-air updates.
- Is computationally suitable for an on-truck edge node.

## 18. Offline Operation

The safety-critical path remains local:

    Sensors
        |
        v
    Local Mosquitto Broker
        |
        v
    Filtering and Feature Extraction
        |
        v
    Fixed Normalisation
        |
        v
    Local Model Inference
        |
        v
    Local Alert Log

Cellular connectivity is not required for local classification or alert
generation.

When connectivity is available, predictions, alerts, and monitoring summaries
can be synchronised with the operations centre. When connectivity is absent,
the local pipeline continues operating and stores events for later
synchronisation.

## 19. C3 Conclusion

LogiBridge implements feature-level fusion by independently extracting three
temperature features and three vibration features and concatenating them into
one six-value vector.

This design satisfies the required model interface while limiting memory,
computation, and deployment complexity. Compared with data-level fusion, it
avoids raw-stream alignment and unnecessarily large inputs. Compared with
decision-level fusion, it requires only one model and preserves cross-sensor
relationships before the final classification decision.
