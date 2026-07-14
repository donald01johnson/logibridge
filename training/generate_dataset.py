#!/usr/bin/env python3
"""Generate the LogiBridge three-class training dataset.

Assignment component D1.

Classes:

0 - Normal
1 - Warning
2 - Critical

The script generates 300 feature windows:

* 120 Normal windows
* 90 Warning windows
* 90 Critical windows

Training statistics are fitted using only the training split.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data_pipeline.preprocessing import (
    FEATURE_NAMES,
    SensorSample,
    SlidingWindowProcessor,
    fit_training_statistics,
    normalize_features,
    save_training_statistics,
)


LOGGER = logging.getLogger("logibridge.dataset")

CLASS_NAMES = {
    0: "Normal",
    1: "Warning",
    2: "Critical",
}

TARGET_WINDOWS = {
    0: 120,
    1: 90,
    2: 90,
}

WINDOW_DURATION_SECONDS = 30
WINDOW_STEP_SECONDS = 10


def required_duration_for_windows(window_count):
    """Return the duration needed to generate an exact window count."""

    return (
        WINDOW_DURATION_SECONDS
        + ((window_count - 1) * WINDOW_STEP_SECONDS)
    )


def clamp(value, minimum, maximum):
    """Restrict a numeric value to a specified range."""

    return max(minimum, min(maximum, value))


def maybe_toggle_door(
    generator,
    current_state,
    probability,
):
    """Randomly toggle the cargo-door state."""

    if generator.random() < probability:
        if current_state == "CLOSE":
            return "OPEN"

        return "CLOSE"

    return current_state


def generate_normal_sample(
    generator,
    second,
    door_state,
):
    """Generate one Normal-class sensor sample."""

    temperature = generator.normal(4.0, 0.30)
    vibration = generator.normal(0.45, 0.05)

    door_state = maybe_toggle_door(
        generator,
        door_state,
        probability=0.006,
    )

    sample = SensorSample(
        timestamp=float(second),
        temperature_c=float(
            clamp(temperature, 2.5, 5.5)
        ),
        vibration_rms_g=float(
            clamp(vibration, 0.20, 0.70)
        ),
        door_state=door_state,
    )

    return sample, door_state


def generate_warning_sample(
    generator,
    second,
    door_state,
    scenario,
):
    """Generate one Warning-class sensor sample."""

    cycle_position = second % 180
    gradual_offset = min(cycle_position / 180.0, 1.0)

    if scenario == 0:
        temperature_mean = 5.5 + (1.4 * gradual_offset)
        vibration_mean = 0.58
        door_probability = 0.012

    elif scenario == 1:
        temperature_mean = 4.8
        vibration_mean = 0.76
        door_probability = 0.014

    else:
        temperature_mean = 5.25 + (0.7 * gradual_offset)
        vibration_mean = 0.68
        door_probability = 0.020

    temperature = generator.normal(
        temperature_mean,
        0.38,
    )

    vibration = generator.normal(
        vibration_mean,
        0.08,
    )

    if generator.random() < 0.025:
        vibration += generator.uniform(0.10, 0.25)

    door_state = maybe_toggle_door(
        generator,
        door_state,
        probability=door_probability,
    )

    sample = SensorSample(
        timestamp=float(second),
        temperature_c=float(
            clamp(temperature, 3.2, 8.2)
        ),
        vibration_rms_g=float(
            clamp(vibration, 0.30, 1.15)
        ),
        door_state=door_state,
    )

    return sample, door_state


def generate_critical_sample(
    generator,
    second,
    door_state,
    scenario,
):
    """Generate one Critical-class sensor sample."""

    cycle_position = second % 150
    gradual_offset = min(cycle_position / 120.0, 1.0)

    if scenario == 0:
        temperature_mean = 8.3 + (2.0 * gradual_offset)
        vibration_mean = 0.95
        door_probability = 0.030

    elif scenario == 1:
        temperature_mean = 7.2
        vibration_mean = 1.28
        door_probability = 0.035

    else:
        temperature_mean = 8.8 + (1.5 * gradual_offset)
        vibration_mean = 1.20
        door_probability = 0.055

    temperature = generator.normal(
        temperature_mean,
        0.50,
    )

    vibration = generator.normal(
        vibration_mean,
        0.13,
    )

    if generator.random() < 0.10:
        vibration += generator.uniform(0.20, 0.55)

    door_state = maybe_toggle_door(
        generator,
        door_state,
        probability=door_probability,
    )

    sample = SensorSample(
        timestamp=float(second),
        temperature_c=float(
            clamp(temperature, 5.5, 13.0)
        ),
        vibration_rms_g=float(
            clamp(vibration, 0.65, 2.20)
        ),
        door_state=door_state,
    )

    return sample, door_state


def generate_class_windows(
    class_label,
    target_window_count,
    seed,
):
    """Generate raw feature windows for one class."""

    generator = np.random.default_rng(seed)

    duration = required_duration_for_windows(
        target_window_count
    )

    processor = SlidingWindowProcessor()

    windows = []
    door_state = "CLOSE"

    for second in range(duration + 1):
        scenario = (second // 300) % 3

        if class_label == 0:
            sample, door_state = generate_normal_sample(
                generator,
                second,
                door_state,
            )

        elif class_label == 1:
            sample, door_state = generate_warning_sample(
                generator,
                second,
                door_state,
                scenario,
            )

        elif class_label == 2:
            sample, door_state = generate_critical_sample(
                generator,
                second,
                door_state,
                scenario,
            )

        else:
            raise ValueError(
                "Unsupported class label: "
                + str(class_label)
            )

        completed_windows = processor.add_sample(sample)

        for completed_window in completed_windows:
            windows.append(
                completed_window["raw_features"]
            )

    feature_matrix = np.asarray(
        windows,
        dtype=np.float32,
    )

    if feature_matrix.shape != (
        target_window_count,
        len(FEATURE_NAMES),
    ):
        raise RuntimeError(
            "Generated feature matrix has unexpected shape. "
            "Expected "
            + str(
                (
                    target_window_count,
                    len(FEATURE_NAMES),
                )
            )
            + ", received "
            + str(feature_matrix.shape)
        )

    labels = np.full(
        target_window_count,
        class_label,
        dtype=np.int64,
    )

    LOGGER.info(
        "Generated %d %s windows",
        target_window_count,
        CLASS_NAMES[class_label],
    )

    return feature_matrix, labels


def split_one_class(
    features,
    labels,
    generator,
):
    """Split one class into train, validation, and test sets."""

    sample_count = len(labels)

    indices = generator.permutation(sample_count)

    training_count = int(
        np.floor(sample_count * 0.70)
    )

    validation_count = int(
        np.floor(sample_count * 0.15)
    )

    test_count = (
        sample_count
        - training_count
        - validation_count
    )

    training_indices = indices[:training_count]

    validation_start = training_count
    validation_end = (
        validation_start + validation_count
    )

    validation_indices = indices[
        validation_start:validation_end
    ]

    test_indices = indices[
        validation_end:
    ]

    if len(test_indices) != test_count:
        raise RuntimeError(
            "Incorrect test split size"
        )

    return {
        "train_features": features[training_indices],
        "train_labels": labels[training_indices],
        "validation_features": features[
            validation_indices
        ],
        "validation_labels": labels[
            validation_indices
        ],
        "test_features": features[test_indices],
        "test_labels": labels[test_indices],
    }


def combine_and_shuffle(
    feature_parts,
    label_parts,
    generator,
):
    """Combine class-specific arrays and shuffle them together."""

    features = np.concatenate(
        feature_parts,
        axis=0,
    )

    labels = np.concatenate(
        label_parts,
        axis=0,
    )

    indices = generator.permutation(len(labels))

    return features[indices], labels[indices]


def class_counts(labels):
    """Return class counts in a JSON-friendly dictionary."""

    counts = {}

    for class_label, class_name in CLASS_NAMES.items():
        counts[class_name] = int(
            np.sum(labels == class_label)
        )

    return counts


def validate_dataset(
    raw_train_features,
    train_labels,
    raw_validation_features,
    validation_labels,
    raw_test_features,
    test_labels,
):
    """Validate all dataset arrays before saving."""

    datasets = {
        "raw_train_features": raw_train_features,
        "raw_validation_features": (
            raw_validation_features
        ),
        "raw_test_features": raw_test_features,
    }

    for name, values in datasets.items():
        if values.ndim != 2:
            raise ValueError(
                name + " must be two-dimensional"
            )

        if values.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                name + " must contain six columns"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                name + " contains non-finite values"
            )

    label_sets = {
        "train_labels": train_labels,
        "validation_labels": validation_labels,
        "test_labels": test_labels,
    }

    for name, values in label_sets.items():
        unique_values = set(
            np.unique(values).tolist()
        )

        if unique_values != {0, 1, 2}:
            raise ValueError(
                name
                + " does not contain all three classes"
            )

    total_samples = (
        len(train_labels)
        + len(validation_labels)
        + len(test_labels)
    )

    if total_samples != 300:
        raise ValueError(
            "Expected 300 total windows, received "
            + str(total_samples)
        )


def print_feature_summary(
    split_name,
    features,
    labels,
):
    """Print a readable summary for one dataset split."""

    print()
    print(split_name)
    print("-" * len(split_name))
    print("Shape:", features.shape)
    print("Class counts:", class_counts(labels))

    print("Feature minimums:")
    print(
        np.array2string(
            np.min(features, axis=0),
            precision=4,
            separator=", ",
        )
    )

    print("Feature means:")
    print(
        np.array2string(
            np.mean(features, axis=0),
            precision=4,
            separator=", ",
        )
    )

    print("Feature maximums:")
    print(
        np.array2string(
            np.max(features, axis=0),
            precision=4,
            separator=", ",
        )
    )


def build_dataset(
    output_path,
    statistics_path,
    seed,
):
    """Generate, split, normalize, validate, and save the dataset."""

    class_feature_matrices = {}
    class_label_vectors = {}

    for class_label in (0, 1, 2):
        class_features, class_labels = (
            generate_class_windows(
                class_label=class_label,
                target_window_count=(
                    TARGET_WINDOWS[class_label]
                ),
                seed=seed + (class_label * 1000),
            )
        )

        class_feature_matrices[class_label] = (
            class_features
        )

        class_label_vectors[class_label] = (
            class_labels
        )

    split_generator = np.random.default_rng(
        seed + 9999
    )

    class_splits = {}

    for class_label in (0, 1, 2):
        class_splits[class_label] = split_one_class(
            class_feature_matrices[class_label],
            class_label_vectors[class_label],
            split_generator,
        )

    raw_train_features, train_labels = (
        combine_and_shuffle(
            [
                class_splits[class_label][
                    "train_features"
                ]
                for class_label in (0, 1, 2)
            ],
            [
                class_splits[class_label][
                    "train_labels"
                ]
                for class_label in (0, 1, 2)
            ],
            split_generator,
        )
    )

    raw_validation_features, validation_labels = (
        combine_and_shuffle(
            [
                class_splits[class_label][
                    "validation_features"
                ]
                for class_label in (0, 1, 2)
            ],
            [
                class_splits[class_label][
                    "validation_labels"
                ]
                for class_label in (0, 1, 2)
            ],
            split_generator,
        )
    )

    raw_test_features, test_labels = (
        combine_and_shuffle(
            [
                class_splits[class_label][
                    "test_features"
                ]
                for class_label in (0, 1, 2)
            ],
            [
                class_splits[class_label][
                    "test_labels"
                ]
                for class_label in (0, 1, 2)
            ],
            split_generator,
        )
    )

    validate_dataset(
        raw_train_features,
        train_labels,
        raw_validation_features,
        validation_labels,
        raw_test_features,
        test_labels,
    )

    statistics = fit_training_statistics(
        raw_train_features
    )

    save_training_statistics(
        statistics_path,
        statistics,
    )

    train_features = normalize_features(
        raw_train_features,
        statistics,
    )

    validation_features = normalize_features(
        raw_validation_features,
        statistics,
    )

    test_features = normalize_features(
        raw_test_features,
        statistics,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        feature_names=np.asarray(
            FEATURE_NAMES,
            dtype=str,
        ),
        class_names=np.asarray(
            [
                CLASS_NAMES[0],
                CLASS_NAMES[1],
                CLASS_NAMES[2],
            ],
            dtype=str,
        ),
        raw_train_features=raw_train_features,
        train_features=train_features,
        train_labels=train_labels,
        raw_validation_features=(
            raw_validation_features
        ),
        validation_features=validation_features,
        validation_labels=validation_labels,
        raw_test_features=raw_test_features,
        test_features=test_features,
        test_labels=test_labels,
        training_mean=np.asarray(
            statistics["mean"],
            dtype=np.float32,
        ),
        training_std=np.asarray(
            statistics["std"],
            dtype=np.float32,
        ),
        random_seed=np.asarray(seed),
    )

    metadata = {
        "schema_version": "1.0",
        "random_seed": seed,
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "target_windows": TARGET_WINDOWS,
        "split": {
            "training": class_counts(train_labels),
            "validation": class_counts(
                validation_labels
            ),
            "test": class_counts(test_labels),
        },
        "total_windows": int(
            len(train_labels)
            + len(validation_labels)
            + len(test_labels)
        ),
        "moving_average_size": 5,
        "window_duration_seconds": 30,
        "window_step_seconds": 10,
    }

    metadata_path = output_path.with_suffix(
        ".json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("LogiBridge Dataset Generation")
    print("=" * 40)
    print("Output:", output_path)
    print("Metadata:", metadata_path)
    print("Statistics:", statistics_path)
    print("Total feature windows: 300")
    print("Feature count:", len(FEATURE_NAMES))

    print_feature_summary(
        "Training split",
        raw_train_features,
        train_labels,
    )

    print_feature_summary(
        "Validation split",
        raw_validation_features,
        validation_labels,
    )

    print_feature_summary(
        "Test split",
        raw_test_features,
        test_labels,
    )

    print()
    print("Training normalization means:")
    print(
        np.array2string(
            np.mean(train_features, axis=0),
            precision=5,
            separator=", ",
        )
    )

    print("Training normalization standard deviations:")
    print(
        np.array2string(
            np.std(train_features, axis=0),
            precision=5,
            separator=", ",
        )
    )

    print()
    print("[PASS] Generated 120 Normal windows")
    print("[PASS] Generated 90 Warning windows")
    print("[PASS] Generated 90 Critical windows")
    print("[PASS] Created stratified splits")
    print("[PASS] Fitted statistics on training data only")
    print("[PASS] Normalized all dataset splits")
    print("[PASS] Saved compressed dataset")
    print("[PASS] Saved dataset metadata")
    print("[PASS] Dataset generation completed")


def build_parser():
    """Build command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate the LogiBridge training dataset"
        )
    )

    parser.add_argument(
        "--output",
        default="training/dataset.npz",
        help="Output compressed dataset path",
    )

    parser.add_argument(
        "--stats-path",
        default="data_pipeline/training_stats.npy",
        help="Output normalization statistics path",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser


def main():
    """Program entry point."""

    arguments = build_parser().parse_args()

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

    build_dataset(
        output_path=arguments.output,
        statistics_path=arguments.stats_path,
        seed=arguments.seed,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
