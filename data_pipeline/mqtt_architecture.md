# MQTT Architecture and Feature-Level Data Fusion

## 1. Sensor and MQTT Architecture

LogiBridge collects three cold-chain data streams:

- Temperature at 1 Hz
- Vibration RMS at 0.5 Hz
- Discrete door OPEN and CLOSE events

The sensor simulator publishes these streams to a local Mosquitto broker.
Local MQTT communication decouples sensor production, preprocessing,
inference, drift monitoring, and alert handling.

Truck-scoped topics include:

    logibridge/trucks/{truck_id}/sensors/temperature
    logibridge/trucks/{truck_id}/sensors/vibration
    logibridge/trucks/{truck_id}/sensors/door
    logibridge/trucks/{truck_id}/sensors/combined
    logibridge/trucks/{truck_id}/features
    logibridge/trucks/{truck_id}/inference
    logibridge/trucks/{truck_id}/alerts

The MQTT broker and processing services run on the truck edge node. The
pipeline therefore continues to operate without a cellular connection.

## 2. Required Preprocessing Sequence

Temperature and vibration are first smoothed independently using
five-sample moving-average filters.

The filtered streams are then analysed using a 30-second sliding window with
a 10-second step.

The temperature stream produces:

1. Temperature mean
2. Temperature standard deviation
3. Temperature rate of change in degrees Celsius per minute

The vibration stream produces:

1. Vibration RMS
2. Vibration peak
3. Vibration kurtosis

The two three-value feature groups are concatenated in a fixed order:

    [
        temperature_mean_c,
        temperature_std_c,
        temperature_rate_c_per_min,
        vibration_rms_g,
        vibration_peak_g,
        vibration_kurtosis
    ]

This forms the joint six-value model input.

## 3. Door Events

Door events remain an important operational stream and are retained in the
MQTT architecture for event logging, alert interpretation, and future model
extensions.

Door-derived values are not added to the current model input because the
assignment specifies an exact six-value vector consisting of three
temperature features and three vibration features.

## 4. Normalisation

Feature means and standard deviations are calculated from ten minutes of
clean Normal-class output.

The resulting values are saved in:

    data_pipeline/training_stats.npy

The file records:

- Feature order
- Feature means
- Feature standard deviations
- Clean Normal source duration
- Moving-average size
- Window duration
- Window step

Training, validation, testing, and runtime inference load this fixed file.
Statistics are never recomputed from live data because doing so would allow
the reference distribution to drift with an ongoing fault.

## 5. Shifted-Statistics Experiment

The mandatory experiment uses two inference conditions:

### Correct condition

    normalized = (features - training_mean) / training_std

### Shifted condition

Each stored mean is shifted by three standard deviations:

    shifted_mean = training_mean + 3 * training_std

Inference is then repeated using:

    shifted_normalized =
        (features - shifted_mean) / training_std

The baseline model accuracy under correct statistics and shifted statistics
will be measured on the same held-out test set.

The exact accuracy values will be generated after the corrected D1 model is
trained and will be reported in the Phase 2 report.

## 6. Why Feature-Level Fusion Was Selected

Feature-level fusion is used because temperature and vibration describe
different but related aspects of refrigeration health.

Temperature mean, variability, and rate of change describe cargo thermal
condition and the direction of thermal deterioration.

Vibration RMS, peak, and kurtosis describe overall mechanical energy,
short-duration shocks, and changes in the vibration distribution associated
with mechanical abnormalities.

Concatenating these complementary descriptors allows one classifier to learn
relationships across thermal and mechanical behaviour.

The result is a compact six-value input, reducing the amount of information
that must be passed to the model compared with raw sensor windows.

## 7. Comparison with Data-Level Fusion

Data-level fusion would concatenate raw temperature and vibration samples
before feature extraction.

This option is less suitable because:

- Temperature and vibration have different sampling rates.
- Their units and numeric scales differ.
- Raw fusion requires resampling and longer model inputs.
- The resulting model would require more memory and computation.
- Raw windows are less appropriate for the constrained edge node.

Feature-level fusion handles each signal using meaningful signal-specific
statistics before combination.

## 8. Comparison with Decision-Level Fusion

Decision-level fusion would train one model for temperature and another for
vibration, then combine two predictions.

This option is less suitable because:

- It requires multiple models.
- It increases deployment and update complexity.
- It consumes more storage and inference resources.
- Cross-sensor relationships are observed only after each model has already
  made an independent decision.

A single fused model can learn interactions between thermal drift and
mechanical vibration before producing the final cargo-state classification.

## 9. Justification

Feature-level fusion provides the best balance for this specific edge
deployment:

- It preserves temperature-specific and vibration-specific information.
- It creates a small and fixed model input.
- It avoids raw-stream alignment complexity.
- It requires only one classifier.
- It is efficient for on-truck inference.
- It supports the exact six-feature model required by the assignment.

The design therefore offers more context than decision-level fusion while
requiring substantially fewer resources than raw data-level fusion.
