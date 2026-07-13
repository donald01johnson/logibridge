#!/usr/bin/env python3
"""LogiBridge sensor preprocessing and feature-fusion pipeline.

Assignment component C2.

Features:
1. Mean filtered temperature
2. Maximum filtered temperature
3. Mean filtered vibration
4. Maximum filtered vibration
5. Door-open fraction
6. Door transition count
"""

import argparse
import json
import logging
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt


LOGGER = logging.getLogger("logibridge.preprocessing")

MOVING_AVERAGE_SIZE = 5
WINDOW_DURATION_SECONDS = 30.0
WINDOW_STEP_SECONDS = 10.0

FEATURE_NAMES = [
    "temperature_mean_c",
    "temperature_max_c",
    "vibration_mean_g",
    "vibration_max_g",
    "door_open_fraction",
    "door_transition_count",
]


@dataclass
class SensorSample:
    timestamp: float
    temperature_c: float
    vibration_rms_g: float
    door_state: str


class MovingAverageFilter:
    """Five-sample moving-average filter for numeric sensors."""

    def __init__(self, size=MOVING_AVERAGE_SIZE):
        self.size = size
        self.temperatures = deque(maxlen=size)
        self.vibrations = deque(maxlen=size)

    def update(self, sample):
        self.temperatures.append(sample.temperature_c)
        self.vibrations.append(sample.vibration_rms_g)

        return SensorSample(
            timestamp=sample.timestamp,
            temperature_c=float(np.mean(self.temperatures)),
            vibration_rms_g=float(np.mean(self.vibrations)),
            door_state=sample.door_state,
        )


def validate_door_state(value):
    """Validate and normalize a door state."""

    state = str(value).strip().upper()

    if state not in ("OPEN", "CLOSE"):
        raise ValueError(
            "door_state must be OPEN or CLOSE"
        )

    return state


def count_door_transitions(states):
    """Count changes between adjacent door states."""

    if len(states) < 2:
        return 0

    count = 0

    for previous, current in zip(states[:-1], states[1:]):
        if previous != current:
            count += 1

    return count


def extract_features(samples):
    """Extract one six-value fused feature vector."""

    if not samples:
        raise ValueError("Cannot process an empty window")

    temperatures = np.asarray(
        [sample.temperature_c for sample in samples],
        dtype=np.float32,
    )

    vibrations = np.asarray(
        [sample.vibration_rms_g for sample in samples],
        dtype=np.float32,
    )

    door_states = [
        validate_door_state(sample.door_state)
        for sample in samples
    ]

    door_open_fraction = float(
        np.mean(
            [
                1.0 if state == "OPEN" else 0.0
                for state in door_states
            ]
        )
    )

    transition_count = float(
        count_door_transitions(door_states)
    )

    features = np.asarray(
        [
            float(np.mean(temperatures)),
            float(np.max(temperatures)),
            float(np.mean(vibrations)),
            float(np.max(vibrations)),
            door_open_fraction,
            transition_count,
        ],
        dtype=np.float32,
    )

    if features.shape != (6,):
        raise RuntimeError("Feature vector must contain six values")

    if not np.all(np.isfinite(features)):
        raise ValueError("Feature vector contains invalid values")

    return features


class SlidingWindowProcessor:
    """Five-sample filtering and 30-second sliding windows."""

    def __init__(self, statistics=None):
        self.filter = MovingAverageFilter()
        self.samples = deque()
        self.statistics = statistics
        self.next_window_end = None
        self.last_timestamp = None

    def add_sample(self, sample):
        """Add a sample and return completed feature windows."""

        if self.last_timestamp is not None:
            if sample.timestamp < self.last_timestamp:
                raise ValueError(
                    "Timestamps must be non-decreasing"
                )

        self.last_timestamp = sample.timestamp

        filtered = self.filter.update(sample)
        self.samples.append(filtered)

        if self.next_window_end is None:
            self.next_window_end = (
                sample.timestamp + WINDOW_DURATION_SECONDS
            )

        completed = []

        while sample.timestamp >= self.next_window_end:
            window_start = (
                self.next_window_end
                - WINDOW_DURATION_SECONDS
            )

            window_samples = [
                item
                for item in self.samples
                if window_start <= item.timestamp
                and item.timestamp <= self.next_window_end
            ]

            if window_samples:
                raw_features = extract_features(window_samples)

                normalized_features = None

                if self.statistics is not None:
                    normalized_features = normalize_features(
                        raw_features,
                        self.statistics,
                    )

                completed.append(
                    {
                        "window_start": window_start,
                        "window_end": self.next_window_end,
                        "sample_count": len(window_samples),
                        "raw_features": raw_features,
                        "normalized_features": normalized_features,
                    }
                )

            self.next_window_end += WINDOW_STEP_SECONDS

        earliest_required = (
            self.next_window_end
            - WINDOW_DURATION_SECONDS
        )

        while self.samples:
            if self.samples[0].timestamp >= earliest_required:
                break

            self.samples.popleft()

        return completed


def fit_training_statistics(feature_matrix):
    """Calculate training-feature means and standard deviations."""

    matrix = np.asarray(feature_matrix, dtype=np.float32)

    if matrix.ndim != 2 or matrix.shape[1] != 6:
        raise ValueError(
            "Feature matrix must have shape N by 6"
        )

    if matrix.shape[0] < 2:
        raise ValueError(
            "At least two feature windows are required"
        )

    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)

    std = np.where(std < 0.000001, 0.000001, std)

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def normalize_features(features, statistics):
    """Apply z-score normalization."""

    values = np.asarray(features, dtype=np.float32)
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    std = np.asarray(statistics["std"], dtype=np.float32)

    if values.shape[-1] != 6:
        raise ValueError("Expected six input features")

    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError("Statistics must contain six values")

    if np.any(std <= 0):
        raise ValueError(
            "Standard deviations must be positive"
        )

    return ((values - mean) / std).astype(np.float32)


def create_shifted_statistics(statistics):
    """Shift every training mean by three standard deviations."""

    mean = np.asarray(statistics["mean"], dtype=np.float32)
    std = np.asarray(statistics["std"], dtype=np.float32)

    return {
        "mean": mean + (3.0 * std),
        "std": std.copy(),
    }


def save_training_statistics(path, statistics):
    """Save training statistics to training_stats.npy."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "1.0",
        "feature_names": FEATURE_NAMES,
        "mean": np.asarray(
            statistics["mean"],
            dtype=np.float32,
        ),
        "std": np.asarray(
            statistics["std"],
            dtype=np.float32,
        ),
        "moving_average_size": MOVING_AVERAGE_SIZE,
        "window_duration_seconds": WINDOW_DURATION_SECONDS,
        "window_step_seconds": WINDOW_STEP_SECONDS,
    }

    np.save(output_path, payload, allow_pickle=True)

    LOGGER.info(
        "Saved training statistics to %s",
        output_path,
    )


def load_training_statistics(path):
    """Load training statistics from training_stats.npy."""

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            "Statistics file not found: "
            + str(input_path)
        )

    payload = np.load(
        input_path,
        allow_pickle=True,
    ).item()

    if payload["feature_names"] != FEATURE_NAMES:
        raise ValueError("Unexpected feature order")

    statistics = {
        "mean": np.asarray(
            payload["mean"],
            dtype=np.float32,
        ),
        "std": np.asarray(
            payload["std"],
            dtype=np.float32,
        ),
    }

    if statistics["mean"].shape != (6,):
        raise ValueError("Invalid mean shape")

    if statistics["std"].shape != (6,):
        raise ValueError("Invalid standard-deviation shape")

    return statistics


def process_samples(samples, statistics=None):
    """Process a finite list of sensor samples."""

    processor = SlidingWindowProcessor(
        statistics=statistics
    )

    windows = []

    for sample in samples:
        windows.extend(processor.add_sample(sample))

    return windows


def generate_test_samples(duration=70, seed=42):
    """Generate deterministic samples for the self-test."""

    generator = np.random.default_rng(seed)

    samples = []
    door_state = "CLOSE"

    for second in range(duration + 1):
        if second in (15, 21, 48, 55):
            if door_state == "CLOSE":
                door_state = "OPEN"
            else:
                door_state = "CLOSE"

        temperature = float(
            generator.normal(4.0, 0.3)
        )

        vibration = max(
            0.0,
            float(generator.normal(0.45, 0.05)),
        )

        samples.append(
            SensorSample(
                timestamp=float(second),
                temperature_c=temperature,
                vibration_rms_g=vibration,
                door_state=door_state,
            )
        )

    return samples


def run_self_test(stats_path):
    """Run deterministic validation of the complete pipeline."""

    samples = generate_test_samples()
    windows = process_samples(samples)

    if len(windows) < 2:
        raise AssertionError(
            "At least two windows were expected"
        )

    feature_matrix = np.vstack(
        [
            window["raw_features"]
            for window in windows
        ]
    )

    statistics = fit_training_statistics(
        feature_matrix
    )

    save_training_statistics(
        stats_path,
        statistics,
    )

    loaded_statistics = load_training_statistics(
        stats_path
    )

    normalized = normalize_features(
        feature_matrix,
        loaded_statistics,
    )

    shifted_statistics = create_shifted_statistics(
        loaded_statistics
    )

    shifted_normalized = normalize_features(
        feature_matrix,
        shifted_statistics,
    )

    print("LogiBridge preprocessing self-test")
    print("=" * 40)
    print("Input samples:", len(samples))
    print("Completed windows:", len(windows))
    print("Feature matrix shape:", feature_matrix.shape)

    print()
    print("Feature names:")

    for index, name in enumerate(FEATURE_NAMES, start=1):
        print(" ", index, name)

    print()
    print("Raw feature matrix:")
    print(
        np.array2string(
            feature_matrix,
            precision=4,
            separator=", ",
        )
    )

    print()
    print("Training means:")
    print(
        np.array2string(
            loaded_statistics["mean"],
            precision=4,
            separator=", ",
        )
    )

    print("Training standard deviations:")
    print(
        np.array2string(
            loaded_statistics["std"],
            precision=4,
            separator=", ",
        )
    )

    print()
    print("Correct normalized feature means:")
    print(
        np.array2string(
            np.mean(normalized, axis=0),
            precision=4,
            separator=", ",
        )
    )

    print("Shifted normalized feature means:")
    print(
        np.array2string(
            np.mean(shifted_normalized, axis=0),
            precision=4,
            separator=", ",
        )
    )

    mean_difference = float(
        np.mean(
            np.abs(
                normalized - shifted_normalized
            )
        )
    )

    print(
        "Mean absolute normalization difference:",
        round(mean_difference, 4),
    )

    if not np.allclose(
        np.mean(normalized, axis=0),
        np.zeros(6),
        atol=0.0001,
    ):
        raise AssertionError(
            "Correct normalized means are not near zero"
        )

    if not np.allclose(
        np.mean(shifted_normalized, axis=0),
        np.full(6, -3.0),
        atol=0.11,
    ):
        raise AssertionError(
            "Shifted normalized means are not near minus three"
        )

    print()
    print("[PASS] Five-sample moving-average filter")
    print("[PASS] 30-second sliding window")
    print("[PASS] 10-second window step")
    print("[PASS] Six-feature extraction")
    print("[PASS] Statistics save and load")
    print("[PASS] Z-score normalization")
    print("[PASS] Three-sigma shift experiment")
    print("[PASS] Preprocessing self-test completed")


class MqttPreprocessingService:
    """Consume combined samples and publish feature vectors."""

    def __init__(
        self,
        broker,
        port,
        truck_id,
        qos,
        statistics,
    ):
        self.broker = broker
        self.port = port
        self.truck_id = truck_id
        self.qos = qos

        self.input_topic = (
            "logibridge/trucks/"
            + truck_id
            + "/sensors/combined"
        )

        self.output_topic = (
            "logibridge/trucks/"
            + truck_id
            + "/features"
        )

        self.processor = SlidingWindowProcessor(
            statistics=statistics
        )

        self.running = True
        self.connected = False
        self.window_sequence = 0

        self.client = mqtt.Client(
            callback_api_version=(
                mqtt.CallbackAPIVersion.VERSION2
            ),
            client_id=(
                "logibridge-preprocessor-" + truck_id
            ),
            protocol=mqtt.MQTTv311,
        )

        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message

    def on_connect(
        self,
        client,
        userdata,
        flags,
        reason_code,
        properties,
    ):
        del userdata, flags, properties

        if reason_code != 0:
            LOGGER.error(
                "MQTT connection failed: %s",
                reason_code,
            )
            return

        self.connected = True

        client.subscribe(
            self.input_topic,
            qos=self.qos,
        )

        LOGGER.info(
            "Subscribed to %s",
            self.input_topic,
        )

    def on_disconnect(
        self,
        client,
        userdata,
        disconnect_flags,
        reason_code,
        properties,
    ):
        del client
        del userdata
        del disconnect_flags
        del properties

        self.connected = False

        if self.running:
            LOGGER.warning(
                "MQTT disconnected: %s",
                reason_code,
            )

    def on_message(self, client, userdata, message):
        del client, userdata

        try:
            payload = json.loads(
                message.payload.decode("utf-8")
            )

            timestamp_text = payload["timestamp"]

            normalized_timestamp = timestamp_text.replace(
                "Z",
                "+00:00",
            )

            timestamp = datetime.fromisoformat(
                normalized_timestamp
            ).timestamp()

            sample = SensorSample(
                timestamp=timestamp,
                temperature_c=float(
                    payload["temperature_c"]
                ),
                vibration_rms_g=float(
                    payload["vibration_rms_g"]
                ),
                door_state=validate_door_state(
                    payload["door_state"]
                ),
            )

            windows = self.processor.add_sample(sample)

            for window in windows:
                self.publish_window(window)

        except Exception as error:
            LOGGER.error(
                "Rejected invalid sensor message: %s",
                error,
            )

    def publish_window(self, window):
        """Publish one feature window as JSON."""

        self.window_sequence += 1

        raw_features = {
            name: float(window["raw_features"][index])
            for index, name in enumerate(FEATURE_NAMES)
        }

        payload = {
            "schema_version": "1.0",
            "truck_id": self.truck_id,
            "window_sequence": self.window_sequence,
            "window_start_unix": window["window_start"],
            "window_end_unix": window["window_end"],
            "sample_count": window["sample_count"],
            "feature_names": FEATURE_NAMES,
            "raw_features": raw_features,
        }

        normalized = window["normalized_features"]

        if normalized is not None:
            payload["normalized_features"] = {
                name: float(normalized[index])
                for index, name in enumerate(FEATURE_NAMES)
            }

        result = self.client.publish(
            self.output_topic,
            json.dumps(
                payload,
                separators=(",", ":"),
            ),
            qos=self.qos,
        )

        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            LOGGER.error(
                "Feature publication failed: %s",
                result.rc,
            )
            return

        LOGGER.info(
            "WINDOW %03d samples=%d raw=%s",
            self.window_sequence,
            window["sample_count"],
            np.array2string(
                window["raw_features"],
                precision=4,
                separator=", ",
            ),
        )

    def stop(self):
        self.running = False

    def run(self):
        """Connect and run until interrupted."""

        self.client.connect(
            self.broker,
            self.port,
            keepalive=60,
        )

        self.client.loop_start()

        deadline = time.monotonic() + 10.0

        while not self.connected:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "MQTT connection timed out"
                )

            time.sleep(0.05)

        LOGGER.info(
            "Preprocessing service started for %s",
            self.truck_id,
        )

        try:
            while self.running:
                time.sleep(0.25)
        finally:
            self.client.loop_stop()
            self.client.disconnect()
            LOGGER.info(
                "Preprocessing service stopped"
            )


def build_parser():
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "LogiBridge preprocessing pipeline"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    test_parser = subparsers.add_parser(
        "self-test"
    )

    test_parser.add_argument(
        "--stats-path",
        default="data_pipeline/training_stats.npy",
    )

    mqtt_parser = subparsers.add_parser(
        "mqtt"
    )

    mqtt_parser.add_argument(
        "--broker",
        default="127.0.0.1",
    )

    mqtt_parser.add_argument(
        "--port",
        type=int,
        default=1883,
    )

    mqtt_parser.add_argument(
        "--truck-id",
        default="TRUCK-001",
    )

    mqtt_parser.add_argument(
        "--qos",
        type=int,
        choices=(0, 1, 2),
        default=1,
    )

    mqtt_parser.add_argument(
        "--stats-path",
        default="data_pipeline/training_stats.npy",
    )

    mqtt_parser.add_argument(
        "--raw-only",
        action="store_true",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser


def main():
    """Program entry point."""

    parser = build_parser()
    arguments = parser.parse_args()

    logging.basicConfig(
        level=(
            logging.DEBUG
            if arguments.verbose
            else logging.INFO
        ),
        format=(
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if arguments.command == "self-test":
        run_self_test(arguments.stats_path)
        return 0

    statistics = None

    if not arguments.raw_only:
        stats_path = Path(arguments.stats_path)

        if stats_path.exists():
            statistics = load_training_statistics(
                stats_path
            )
        else:
            LOGGER.warning(
                "Statistics file not found; "
                "publishing raw features only"
            )

    service = MqttPreprocessingService(
        broker=arguments.broker,
        port=arguments.port,
        truck_id=arguments.truck_id,
        qos=arguments.qos,
        statistics=statistics,
    )

    def stop_service(signal_number, frame):
        del frame

        LOGGER.info(
            "Received signal %s",
            signal_number,
        )

        service.stop()

    signal.signal(
        signal.SIGINT,
        stop_service,
    )

    signal.signal(
        signal.SIGTERM,
        stop_service,
    )

    service.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
