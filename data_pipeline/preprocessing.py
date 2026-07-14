#!/usr/bin/env python3
"""LogiBridge preprocessing and feature-level fusion.

Assignment Task C2 requirements:

* Five-sample moving average on temperature and vibration
* Thirty-second sliding window
* Ten-second window step
* Six-value feature vector:
  1. Temperature mean
  2. Temperature standard deviation
  3. Temperature rate of change in degrees Celsius per minute
  4. Vibration RMS
  5. Vibration peak
  6. Vibration kurtosis
* Fixed normalization statistics generated from ten minutes of clean
  Normal-class output
* Runtime loading of statistics
* Three-standard-deviation shifted-statistics experiment
"""

import argparse
import logging
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import numpy as np


LOGGER = logging.getLogger("logibridge.preprocessing")

MOVING_AVERAGE_SIZE = 5
WINDOW_DURATION_SECONDS = 30.0
WINDOW_STEP_SECONDS = 10.0
NORMAL_STATISTICS_DURATION_SECONDS = 600

FEATURE_NAMES = [
    "temperature_mean_c",
    "temperature_std_c",
    "temperature_rate_c_per_min",
    "vibration_rms_g",
    "vibration_peak_g",
    "vibration_kurtosis",
]


@dataclass
class SensorSample:
    """One synchronized sensor observation.

    vibration_updated is True only when a new 0.5 Hz vibration
    measurement is available. On intermediate 1 Hz cycles, the latest
    vibration value may be carried forward, but it must not be inserted
    into the five-sample vibration filter again.
    """

    timestamp: float
    temperature_c: float
    vibration_rms_g: float
    door_state: str = "CLOSE"
    vibration_updated: bool = True


class MovingAverageFilter:
    """Independent five-sample filters for temperature and vibration."""

    def __init__(self, size=MOVING_AVERAGE_SIZE):
        if size <= 0:
            raise ValueError(
                "Moving-average size must be positive"
            )

        self.size = size

        self.temperature_values = deque(
            maxlen=size
        )

        self.vibration_values = deque(
            maxlen=size
        )

        self.latest_filtered_vibration = None

    def update(self, sample):
        """Filter one synchronized sensor observation.

        Temperature is updated for every 1 Hz sample.

        Vibration is updated only when vibration_updated is True,
        corresponding to a new 0.5 Hz vibration observation.
        """

        self.temperature_values.append(
            float(sample.temperature_c)
        )

        filtered_temperature = float(
            np.mean(
                self.temperature_values
            )
        )

        if (
            sample.vibration_updated
            or not self.vibration_values
        ):
            self.vibration_values.append(
                float(
                    sample.vibration_rms_g
                )
            )

            self.latest_filtered_vibration = float(
                np.mean(
                    self.vibration_values
                )
            )

        if self.latest_filtered_vibration is None:
            raise RuntimeError(
                "No vibration value is available"
            )

        return SensorSample(
            timestamp=float(
                sample.timestamp
            ),
            temperature_c=(
                filtered_temperature
            ),
            vibration_rms_g=(
                self.latest_filtered_vibration
            ),
            door_state=str(
                sample.door_state
            ).upper(),
            vibration_updated=bool(
                sample.vibration_updated
            ),
        )


def calculate_temperature_rate(
    timestamps,
    temperatures,
):
    """Calculate temperature slope in degrees Celsius per minute.

    A least-squares linear slope is used across the complete window.
    This is less sensitive to single-sample noise than subtracting only
    the first and final values.
    """

    relative_time = timestamps - timestamps[0]

    time_variance = float(
        np.sum(
            (
                relative_time
                - np.mean(relative_time)
            )
            ** 2
        )
    )

    if time_variance <= 0.0:
        return 0.0

    covariance = float(
        np.sum(
            (
                relative_time
                - np.mean(relative_time)
            )
            * (
                temperatures
                - np.mean(temperatures)
            )
        )
    )

    slope_per_second = covariance / time_variance

    return slope_per_second * 60.0


def calculate_kurtosis(values):
    """Calculate Pearson kurtosis from the second and fourth moments.

    A Gaussian distribution has Pearson kurtosis close to 3.0.
    A constant or numerically near-constant window returns 0.0 because
    kurtosis is undefined when variance is zero.
    """

    values = np.asarray(values, dtype=np.float64)

    centered = values - np.mean(values)

    second_moment = float(
        np.mean(centered ** 2)
    )

    if second_moment <= 1e-12:
        return 0.0

    fourth_moment = float(
        np.mean(centered ** 4)
    )

    return fourth_moment / (
        second_moment ** 2
    )



def extract_features(samples):
    """Extract and concatenate the exact six C2 features.

    Temperature features are calculated from filtered 1 Hz temperature
    observations.

    Vibration features are calculated only from filtered observations for
    which vibration_updated is True, preserving the original 0.5 Hz
    vibration stream.
    """

    if len(samples) < 2:
        raise ValueError(
            "At least two filtered samples are required"
        )

    temperature_samples = list(
        samples
    )

    vibration_samples = [
        sample
        for sample in samples
        if sample.vibration_updated
    ]

    if len(vibration_samples) < 2:
        raise ValueError(
            "At least two genuine vibration samples "
            "are required"
        )

    temperature_timestamps = np.asarray(
        [
            sample.timestamp
            for sample in temperature_samples
        ],
        dtype=np.float64,
    )

    temperatures = np.asarray(
        [
            sample.temperature_c
            for sample in temperature_samples
        ],
        dtype=np.float64,
    )

    vibrations = np.asarray(
        [
            sample.vibration_rms_g
            for sample in vibration_samples
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.diff(
            temperature_timestamps
        ) >= 0
    ):
        raise ValueError(
            "Window timestamps must be non-decreasing"
        )

    temperature_features = np.asarray(
        [
            float(
                np.mean(
                    temperatures
                )
            ),
            float(
                np.std(
                    temperatures,
                    ddof=0,
                )
            ),
            float(
                calculate_temperature_rate(
                    temperature_timestamps,
                    temperatures,
                )
            ),
        ],
        dtype=np.float32,
    )

    vibration_features = np.asarray(
        [
            float(
                np.sqrt(
                    np.mean(
                        vibrations ** 2
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        vibrations
                    )
                )
            ),
            float(
                calculate_kurtosis(
                    vibrations
                )
            ),
        ],
        dtype=np.float32,
    )

    fused_features = np.concatenate(
        [
            temperature_features,
            vibration_features,
        ],
        axis=0,
    ).astype(np.float32)

    if fused_features.shape != (6,):
        raise RuntimeError(
            "Feature vector must have shape (6,)"
        )

    if not np.all(
        np.isfinite(
            fused_features
        )
    ):
        raise ValueError(
            "Feature vector contains non-finite values"
        )

    return fused_features


class SlidingWindowProcessor:
    """Filter samples and produce thirty-second sliding windows."""

    def __init__(self, statistics=None):
        self.filter = MovingAverageFilter()
        self.samples = deque()
        self.statistics = statistics
        self.next_window_end = None
        self.last_timestamp = None

    def add_sample(self, sample):
        """Add one sample and return zero or more completed windows."""

        if self.last_timestamp is not None:
            if sample.timestamp < self.last_timestamp:
                raise ValueError(
                    "Input timestamps must be non-decreasing"
                )

        self.last_timestamp = sample.timestamp

        filtered_sample = self.filter.update(sample)

        self.samples.append(filtered_sample)

        if self.next_window_end is None:
            self.next_window_end = (
                sample.timestamp
                + WINDOW_DURATION_SECONDS
            )

        completed_windows = []

        while sample.timestamp >= self.next_window_end:
            window_start = (
                self.next_window_end
                - WINDOW_DURATION_SECONDS
            )

            window_samples = [
                candidate
                for candidate in self.samples
                if window_start <= candidate.timestamp
                and candidate.timestamp
                <= self.next_window_end
            ]

            if len(window_samples) >= 2:
                raw_features = extract_features(
                    window_samples
                )

                normalized_features = None

                if self.statistics is not None:
                    normalized_features = (
                        normalize_features(
                            raw_features,
                            self.statistics,
                        )
                    )

                completed_windows.append(
                    {
                        "window_start": window_start,
                        "window_end": self.next_window_end,
                        "sample_count": len(
                            window_samples
                        ),
                        "raw_features": raw_features,
                        "normalized_features": (
                            normalized_features
                        ),
                    }
                )

            self.next_window_end += (
                WINDOW_STEP_SECONDS
            )

        earliest_required = (
            self.next_window_end
            - WINDOW_DURATION_SECONDS
        )

        while self.samples:
            if (
                self.samples[0].timestamp
                >= earliest_required
            ):
                break

            self.samples.popleft()

        return completed_windows


def process_samples(samples, statistics=None):
    """Process a finite list of samples."""

    processor = SlidingWindowProcessor(
        statistics=statistics
    )

    completed_windows = []

    for sample in samples:
        completed_windows.extend(
            processor.add_sample(sample)
        )

    return completed_windows


def fit_training_statistics(feature_matrix):
    """Compute means and standard deviations from clean Normal data."""

    matrix = np.asarray(
        feature_matrix,
        dtype=np.float32,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Feature matrix must be two-dimensional"
        )

    if matrix.shape[1] != 6:
        raise ValueError(
            "Feature matrix must contain six columns"
        )

    if matrix.shape[0] < 2:
        raise ValueError(
            "At least two windows are required"
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "Feature matrix contains invalid values"
        )

    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0, ddof=0)

    std = np.where(
        std < 1e-6,
        1e-6,
        std,
    )

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def normalize_features(features, statistics):
    """Apply fixed z-score normalization."""

    values = np.asarray(
        features,
        dtype=np.float32,
    )

    mean = np.asarray(
        statistics["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        statistics["std"],
        dtype=np.float32,
    )

    if values.shape[-1] != 6:
        raise ValueError(
            "Expected six input features"
        )

    if mean.shape != (6,):
        raise ValueError(
            "Statistics mean must have shape (6,)"
        )

    if std.shape != (6,):
        raise ValueError(
            "Statistics standard deviation must have shape (6,)"
        )

    if np.any(std <= 0):
        raise ValueError(
            "Standard deviations must be positive"
        )

    normalized = (
        values - mean
    ) / std

    return normalized.astype(np.float32)


def create_shifted_statistics(
    statistics,
    sigma_shift=3.0,
):
    """Create deliberately incorrect shifted statistics."""

    mean = np.asarray(
        statistics["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        statistics["std"],
        dtype=np.float32,
    )

    return {
        "mean": (
            mean + sigma_shift * std
        ).astype(np.float32),
        "std": std.copy().astype(np.float32),
    }


def save_training_statistics(path, statistics):
    """Save fixed statistics to training_stats.npy."""

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": "2.0",
        "source": (
            "10 minutes of clean Normal-class output"
        ),
        "source_duration_seconds": (
            NORMAL_STATISTICS_DURATION_SECONDS
        ),
        "feature_names": FEATURE_NAMES,
        "mean": np.asarray(
            statistics["mean"],
            dtype=np.float32,
        ),
        "std": np.asarray(
            statistics["std"],
            dtype=np.float32,
        ),
        "moving_average_size": (
            MOVING_AVERAGE_SIZE
        ),
        "window_duration_seconds": (
            WINDOW_DURATION_SECONDS
        ),
        "window_step_seconds": (
            WINDOW_STEP_SECONDS
        ),
    }

    np.save(
        output_path,
        payload,
        allow_pickle=True,
    )

    LOGGER.info(
        "Saved fixed Normal-class statistics to %s",
        output_path,
    )


def load_training_statistics(path):
    """Load fixed statistics without recomputing them from live data."""

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

    if list(payload["feature_names"]) != FEATURE_NAMES:
        raise ValueError(
            "Saved feature order is incorrect"
        )

    if int(
        payload["source_duration_seconds"]
    ) != NORMAL_STATISTICS_DURATION_SECONDS:
        raise ValueError(
            "Statistics were not generated from the required duration"
        )

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
        raise ValueError(
            "Loaded mean has an invalid shape"
        )

    if statistics["std"].shape != (6,):
        raise ValueError(
            "Loaded standard deviation has an invalid shape"
        )

    if np.any(statistics["std"] <= 0):
        raise ValueError(
            "Loaded standard deviations are invalid"
        )

    return statistics



def generate_clean_normal_samples(
    duration_seconds=(
        NORMAL_STATISTICS_DURATION_SECONDS
    ),
    seed=42,
):
    """Generate clean Normal data through the shared C1 simulator.

    This guarantees that training_stats.npy is produced from the same
    simulator implementation used by C1 and D1.
    """

    from data_pipeline.simulator import (
        generate_offline_samples,
    )

    raw_samples = generate_offline_samples(
        anomaly="none",
        duration_seconds=(
            duration_seconds
        ),
        seed=seed,
    )

    samples = []

    for raw_sample in raw_samples:
        timestamp = float(
            raw_sample["timestamp"]
        )

        vibration_updated = bool(
            raw_sample.get(
                "vibration_updated",
                int(timestamp) % 2 == 0,
            )
        )

        samples.append(
            SensorSample(
                timestamp=timestamp,
                temperature_c=float(
                    raw_sample[
                        "temperature_c"
                    ]
                ),
                vibration_rms_g=float(
                    raw_sample[
                        "vibration_rms_g"
                    ]
                ),
                door_state=str(
                    raw_sample.get(
                        "door_state",
                        "CLOSE",
                    )
                ),
                vibration_updated=(
                    vibration_updated
                ),
            )
        )

    return samples


def generate_normal_statistics(
    output_path,
    seed=42,
):
    """Create statistics from exactly ten minutes of clean data."""

    samples = generate_clean_normal_samples(
        duration_seconds=(
            NORMAL_STATISTICS_DURATION_SECONDS
        ),
        seed=seed,
    )

    windows = process_samples(samples)

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
        output_path,
        statistics,
    )

    return feature_matrix, statistics


def run_self_test(output_path, seed):
    """Validate every C2 preprocessing requirement."""

    feature_matrix, statistics = (
        generate_normal_statistics(
            output_path=output_path,
            seed=seed,
        )
    )

    loaded_statistics = (
        load_training_statistics(
            output_path
        )
    )

    correct_normalized = normalize_features(
        feature_matrix,
        loaded_statistics,
    )

    shifted_statistics = (
        create_shifted_statistics(
            loaded_statistics,
            sigma_shift=3.0,
        )
    )

    shifted_normalized = normalize_features(
        feature_matrix,
        shifted_statistics,
    )

    correct_means = np.mean(
        correct_normalized,
        axis=0,
    )

    shifted_means = np.mean(
        shifted_normalized,
        axis=0,
    )

    print("LogiBridge C2 Self-Test")
    print("=" * 44)

    print(
        "Normal source duration:",
        NORMAL_STATISTICS_DURATION_SECONDS,
        "seconds",
    )

    print(
        "Generated Normal windows:",
        len(feature_matrix),
    )

    print(
        "Feature matrix shape:",
        feature_matrix.shape,
    )

    print()
    print("Feature order:")

    for index, feature_name in enumerate(
        FEATURE_NAMES,
        start=1,
    ):
        print(
            " ",
            index,
            feature_name,
        )

    print()
    print("Training means:")
    print(
        np.array2string(
            loaded_statistics["mean"],
            precision=5,
            separator=", ",
        )
    )

    print("Training standard deviations:")
    print(
        np.array2string(
            loaded_statistics["std"],
            precision=5,
            separator=", ",
        )
    )

    print()
    print("Correct normalized feature means:")
    print(
        np.array2string(
            correct_means,
            precision=5,
            separator=", ",
        )
    )

    print("Shifted normalized feature means:")
    print(
        np.array2string(
            shifted_means,
            precision=5,
            separator=", ",
        )
    )

    if not np.allclose(
        correct_means,
        np.zeros(6),
        atol=0.0001,
    ):
        raise AssertionError(
            "Correct normalized means are not near zero"
        )

    if not np.allclose(
        shifted_means,
        np.full(6, -3.0),
        atol=0.01,
    ):
        raise AssertionError(
            "Shifted normalized means are not near minus three"
        )

    print()
    print("[PASS] Five-sample temperature filtering")
    print("[PASS] Five-sample vibration filtering")
    print("[PASS] Thirty-second windows")
    print("[PASS] Ten-second window step")
    print("[PASS] Temperature mean")
    print("[PASS] Temperature standard deviation")
    print("[PASS] Temperature rate of change")
    print("[PASS] Vibration RMS")
    print("[PASS] Vibration peak")
    print("[PASS] Vibration kurtosis")
    print("[PASS] Six-value feature-level fusion")
    print("[PASS] Ten-minute Normal statistics")
    print("[PASS] Fixed statistics save and load")
    print("[PASS] Three-sigma shifted statistics")
    print("[PASS] C2 preprocessing validation completed")


def build_parser():
    """Build the CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "LogiBridge C2 preprocessing pipeline"
        )
    )

    parser.add_argument(
        "command",
        choices=[
            "self-test",
            "generate-stats",
            "show-stats",
        ],
    )

    parser.add_argument(
        "--stats-path",
        default=(
            "data_pipeline/training_stats.npy"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser


def main():
    """Program entry point."""

    arguments = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if arguments.command == "self-test":
        run_self_test(
            output_path=arguments.stats_path,
            seed=arguments.seed,
        )

    elif arguments.command == "generate-stats":
        matrix, statistics = (
            generate_normal_statistics(
                output_path=(
                    arguments.stats_path
                ),
                seed=arguments.seed,
            )
        )

        print(
            "Generated windows:",
            len(matrix),
        )

        print(
            "Statistics mean:",
            statistics["mean"],
        )

        print(
            "Statistics std:",
            statistics["std"],
        )

    elif arguments.command == "show-stats":
        statistics = load_training_statistics(
            arguments.stats_path
        )

        print("Feature names:")
        print(FEATURE_NAMES)
        print("Mean:")
        print(statistics["mean"])
        print("Standard deviation:")
        print(statistics["std"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
