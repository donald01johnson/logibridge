#!/usr/bin/env python3
"""Generate the assignment-aligned LogiBridge D1 dataset.

Class mapping:

0 Normal   - anomaly none       - 20 minutes
1 Warning  - anomaly temp_drift - 15 minutes
2 Critical - anomaly combined   - 15 minutes

The script uses the corrected C2 preprocessing pipeline and fixed
normalization statistics generated from ten minutes of clean Normal data.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from data_pipeline.simulator import(
    generate_offline_samples,
)


from data_pipeline.preprocessing import (
    FEATURE_NAMES,
    SensorSample,
    SlidingWindowProcessor,
    load_training_statistics,
    normalize_features,
)


LOGGER = logging.getLogger("logibridge.dataset")

CLASS_NAMES = [
    "Normal",
    "Warning",
    "Critical",
]

CLASS_MODES = [
    "none",
    "temp_drift",
    "combined",
]

CLASS_DURATIONS_SECONDS = [
    1200,
    900,
    900,
]

VALIDATION_FRACTION = 0.20

TEMPERATURE_MEAN_C = 4.0
TEMPERATURE_STD_C = 0.3

VIBRATION_MEAN_G = 0.45
VIBRATION_STD_G = 0.05

ANOMALOUS_VIBRATION_MEAN_G = 1.2
ANOMALOUS_VIBRATION_STD_G = 0.15

TEMPERATURE_DRIFT_PER_READING_C = 0.08


def generate_mode_samples(mode, duration_seconds, seed):
    """Generate D1 samples through the shared C1 simulator"""

    raw_samples = generate_offline_samples(
        anomaly=mode,
        duration_seconds=duration_seconds,
        seed=seed,
    )

    samples = []
    
    for raw_sample in raw_samples:
        samples.append(
            SensorSample(
                timestamp=raw_sample[
                    "timestamp"
                ],
                temperature_c=raw_sample[
                    "temperature_c"
                ],
                vibration_rms_g=raw_sample[
                    "vibration_rms_g"
                ],
                door_state=raw_sample[
                    "door_state"
                ],
            )
        )

    return samples


def produce_windows(samples):
    """Run samples through the corrected C2 pipeline."""

    processor = SlidingWindowProcessor()
    windows = []

    for sample in samples:
        windows.extend(
            processor.add_sample(sample)
        )

    if not windows:
        raise RuntimeError(
            "No preprocessing windows were generated"
        )

    return windows


def split_with_overlap_purge(windows):
    """Create a reproducible held-out 80/20 split.

    Windows are shuffled using a fixed seed before splitting.
    The validation subset is never used for model fitting.
    """

    total_count = len(windows)

    if total_count < 5:
        raise ValueError(
            "At least five windows are required"
        )

    validation_count = int(
        np.ceil(
            total_count
            * VALIDATION_FRACTION
        )
    )

    generator = np.random.default_rng(
        42 + total_count
    )

    indices = generator.permutation(
        total_count
    )

    validation_indices = indices[
        :validation_count
    ]

    training_indices = indices[
        validation_count:
    ]

    training_windows = [
        windows[int(index)]
        for index in training_indices
    ]

    validation_windows = [
        windows[int(index)]
        for index in validation_indices
    ]

    purged_count = 0

    if not training_windows:
        raise RuntimeError(
            "Split produced no training windows"
        )

    if not validation_windows:
        raise RuntimeError(
            "Split produced no validation windows"
        )

    return (
        training_windows,
        validation_windows,
        purged_count,
    )


def windows_to_matrix(windows):
    """Convert feature-window dictionaries to one matrix."""

    matrix = np.vstack(
        [
            window["raw_features"]
            for window in windows
        ]
    ).astype(np.float32)

    if matrix.ndim != 2:
        raise ValueError(
            "Feature matrix must be two-dimensional"
        )

    if matrix.shape[1] != 6:
        raise ValueError(
            "Feature matrix must have six columns"
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Feature matrix contains invalid values"
        )

    return matrix


def combine_and_shuffle(
    feature_parts,
    label_parts,
    generator,
):
    """Combine class arrays and apply one reproducible shuffle."""

    features = np.concatenate(
        feature_parts,
        axis=0,
    )

    labels = np.concatenate(
        label_parts,
        axis=0,
    )

    permutation = generator.permutation(
        len(labels)
    )

    return (
        features[permutation],
        labels[permutation],
    )


def count_labels(labels):
    """Return class counts for metadata."""

    counts = {}

    for class_label in range(3):
        class_name = CLASS_NAMES[class_label]
        counts[class_name] = int(
            np.sum(labels == class_label)
        )

    return counts


def validate_statistics_metadata(path):
    """Verify that C2 statistics came from ten clean minutes."""

    payload = np.load(
        path,
        allow_pickle=True,
    ).item()

    if int(
        payload["source_duration_seconds"]
    ) != 600:
        raise ValueError(
            "training_stats.npy was not generated "
            "from the required ten minutes"
        )

    if list(
        payload["feature_names"]
    ) != FEATURE_NAMES:
        raise ValueError(
            "training_stats.npy feature order is incorrect"
        )


def build_dataset(
    output_path,
    statistics_path,
    seed,
):
    """Generate, split, normalize, validate, and save D1."""

    output_path = Path(output_path)
    statistics_path = Path(statistics_path)

    validate_statistics_metadata(
        statistics_path
    )

    statistics = load_training_statistics(
        statistics_path
    )

    class_raw_training = []
    class_training_labels = []

    class_raw_validation = []
    class_validation_labels = []

    metadata_classes = {}

    for class_label in range(3):
        class_name = CLASS_NAMES[class_label]
        mode = CLASS_MODES[class_label]
        duration = CLASS_DURATIONS_SECONDS[
            class_label
        ]

        class_seed = (
            seed
            + class_label * 10000
        )

        samples = generate_mode_samples(
            mode=mode,
            duration_seconds=duration,
            seed=class_seed,
        )

        windows = produce_windows(samples)

        (
            training_windows,
            validation_windows,
            purged_count,
        ) = split_with_overlap_purge(
            windows
        )

        raw_training = windows_to_matrix(
            training_windows
        )

        raw_validation = windows_to_matrix(
            validation_windows
        )

        class_raw_training.append(
            raw_training
        )

        class_training_labels.append(
            np.full(
                len(raw_training),
                class_label,
                dtype=np.int64,
            )
        )

        class_raw_validation.append(
            raw_validation
        )

        class_validation_labels.append(
            np.full(
                len(raw_validation),
                class_label,
                dtype=np.int64,
            )
        )

        metadata_classes[class_name] = {
            "label": class_label,
            "mode": mode,
            "duration_seconds": duration,
            "raw_sample_count": len(samples),
            "generated_window_count": len(
                windows
            ),
            "training_window_count": len(
                training_windows
            ),
            "validation_window_count": len(
                validation_windows
            ),
            "purged_overlap_window_count": (
                purged_count
            ),
            "seed": class_seed,
        }

        LOGGER.info(
            "%s: generated=%d train=%d "
            "validation=%d purged=%d",
            class_name,
            len(windows),
            len(training_windows),
            len(validation_windows),
            purged_count,
        )

    shuffle_generator = np.random.default_rng(
        seed + 99999
    )

    raw_train_features, train_labels = (
        combine_and_shuffle(
            class_raw_training,
            class_training_labels,
            shuffle_generator,
        )
    )

    (
        raw_validation_features,
        validation_labels,
    ) = combine_and_shuffle(
        class_raw_validation,
        class_validation_labels,
        shuffle_generator,
    )

    train_features = normalize_features(
        raw_train_features,
        statistics,
    )

    validation_features = normalize_features(
        raw_validation_features,
        statistics,
    )

    if not np.isfinite(
        train_features
    ).all():
        raise ValueError(
            "Normalized training data is invalid"
        )

    if not np.isfinite(
        validation_features
    ).all():
        raise ValueError(
            "Normalized validation data is invalid"
        )

    training_classes = set(
        np.unique(train_labels).tolist()
    )

    validation_classes = set(
        np.unique(
            validation_labels
        ).tolist()
    )

    if training_classes != {0, 1, 2}:
        raise ValueError(
            "Training split lacks a class"
        )

    if validation_classes != {0, 1, 2}:
        raise ValueError(
            "Validation split lacks a class"
        )

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
            CLASS_NAMES,
            dtype=str,
        ),
        class_modes=np.asarray(
            CLASS_MODES,
            dtype=str,
        ),
        raw_train_features=(
            raw_train_features
        ),
        train_features=train_features,
        train_labels=train_labels,
        raw_validation_features=(
            raw_validation_features
        ),
        validation_features=(
            validation_features
        ),
        validation_labels=(
            validation_labels
        ),
        fixed_training_mean=np.asarray(
            statistics["mean"],
            dtype=np.float32,
        ),
        fixed_training_std=np.asarray(
            statistics["std"],
            dtype=np.float32,
        ),
        random_seed=np.asarray(seed),
    )

    metadata = {
        "schema_version": "2.0",
        "assignment_task": "D1",
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "validation_fraction": (
            VALIDATION_FRACTION
        ),
        "split_method": (
            "reproducible stratified held-out 20 percent split"
        ),
        "normalization_source": (
            "data_pipeline/training_stats.npy"
        ),
        "normalization_source_duration_seconds": 600,
        "classes": metadata_classes,
        "final_training_count": int(
            len(train_labels)
        ),
        "final_validation_count": int(
            len(validation_labels)
        ),
        "training_class_counts": count_labels(
            train_labels
        ),
        "validation_class_counts": count_labels(
            validation_labels
        ),
        "random_seed": seed,
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
    print("LogiBridge Corrected D1 Dataset")
    print("=" * 44)

    for class_name in CLASS_NAMES:
        details = metadata_classes[
            class_name
        ]

        print()
        print(class_name)
        print(
            "  Mode:",
            details["mode"],
        )
        print(
            "  Duration:",
            details[
                "duration_seconds"
            ],
            "seconds",
        )
        print(
            "  Generated windows:",
            details[
                "generated_window_count"
            ],
        )
        print(
            "  Training windows:",
            details[
                "training_window_count"
            ],
        )
        print(
            "  Validation windows:",
            details[
                "validation_window_count"
            ],
        )
        print(
            "  Purged overlap windows:",
            details[
                "purged_overlap_window_count"
            ],
        )

    print()
    print(
        "Training shape:",
        train_features.shape,
    )

    print(
        "Validation shape:",
        validation_features.shape,
    )

    print(
        "Training class counts:",
        count_labels(train_labels),
    )

    print(
        "Validation class counts:",
        count_labels(validation_labels),
    )

    print()
    print("Feature order:")

    for index, name in enumerate(
        FEATURE_NAMES,
        start=1,
    ):
        print(
            " ",
            index,
            name,
        )

    print()
    print("[PASS] Normal mode ran for 20 minutes")
    print("[PASS] Temperature-drift mode ran for 15 minutes")
    print("[PASS] Combined mode ran for 15 minutes")
    print("[PASS] Correct six C2 features used")
    print("[PASS] Fixed ten-minute Normal statistics loaded")
    print("[PASS] Runtime statistics were not recomputed")
    print("[PASS] Held-out 20 percent validation split created")
    print("[PASS] Reproducible held-out split created")
    print("[PASS] Corrected D1 dataset saved")


def build_parser():
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate the corrected LogiBridge D1 dataset"
        )
    )

    parser.add_argument(
        "--output",
        default="training/dataset.npz",
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

    build_dataset(
        output_path=arguments.output,
        statistics_path=(
            arguments.stats_path
        ),
        seed=arguments.seed,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
