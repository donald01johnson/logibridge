#!/usr/bin/env python3
"""Create M2 using full INT8 post-training quantization."""

import hashlib
import os
import sys
from pathlib import Path


os.environ.setdefault(
    "TF_USE_LEGACY_KERAS",
    "1",
)

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "2",
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import numpy as np
import tf_keras

from optimisation.f1_common import (
    CALIBRATION_SAMPLE_COUNT,
    convert_full_int8,
    evaluate_tflite,
    load_dataset,
    select_calibration_samples,
    write_json,
)


DATASET_PATH = Path(
    "training/dataset.npz"
)

KERAS_MODEL_PATH = Path(
    "training/models/baseline_model.keras"
)

OUTPUT_PATH = Path(
    "optimisation/models/m2_int8.tflite"
)

METRICS_PATH = Path(
    "optimisation/results/m2_metrics.json"
)


def sha256(path):
    """Return the SHA-256 digest of a file."""

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def main():
    """Convert and validate M2."""

    dataset = load_dataset(
        DATASET_PATH
    )

    model = tf_keras.models.load_model(
        KERAS_MODEL_PATH
    )

    calibration_features = (
        select_calibration_samples(
            dataset["train_features"],
            sample_count=(
                CALIBRATION_SAMPLE_COUNT
            ),
            seed=42,
        )
    )

    if len(
        calibration_features
    ) != 200:
        raise RuntimeError(
            "Expected exactly 200 "
            "calibration samples"
        )

    converted_model = convert_full_int8(
        model,
        calibration_features,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_bytes(
        converted_model
    )

    evaluation = evaluate_tflite(
        OUTPUT_PATH,
        dataset[
            "validation_features"
        ],
        dataset[
            "validation_labels"
        ],
    )

    if evaluation[
        "input_dtype"
    ] != "<class 'numpy.int8'>":
        raise RuntimeError(
            "M2 input is not INT8"
        )

    if evaluation[
        "output_dtype"
    ] != "<class 'numpy.int8'>":
        raise RuntimeError(
            "M2 output is not INT8"
        )

    metrics = {
        "variant": "M2",
        "name": "PTQ Full INT8",
        "method": (
            "Post-training full integer "
            "quantization"
        ),
        "optimization": (
            "tf.lite.Optimize.DEFAULT"
        ),
        "calibration_sample_count": 200,
        "calibration_source": (
            "reproducible subset of "
            "normalized training features"
        ),
        "model_path": str(
            OUTPUT_PATH
        ),
        "file_size_bytes": int(
            OUTPUT_PATH.stat().st_size
        ),
        "sha256": sha256(
            OUTPUT_PATH
        ),
        "validation_accuracy": (
            evaluation["accuracy"]
        ),
        "confusion_matrix": (
            evaluation[
                "confusion_matrix"
            ].tolist()
        ),
        "per_class_recall": (
            evaluation[
                "per_class_recall"
            ]
        ),
        "input_dtype": (
            evaluation["input_dtype"]
        ),
        "output_dtype": (
            evaluation["output_dtype"]
        ),
        "input_quantization": (
            evaluation[
                "input_quantization"
            ]
        ),
        "output_quantization": (
            evaluation[
                "output_quantization"
            ]
        ),
    }

    write_json(
        METRICS_PATH,
        metrics,
    )

    print("M2 Full INT8 PTQ")
    print("=" * 42)

    print(
        "Calibration samples:",
        len(calibration_features),
    )

    print(
        "Input dtype:",
        metrics["input_dtype"],
    )

    print(
        "Output dtype:",
        metrics["output_dtype"],
    )

    print(
        "Model bytes:",
        metrics["file_size_bytes"],
    )

    print(
        "Validation accuracy:",
        format(
            metrics[
                "validation_accuracy"
            ] * 100.0,
            ".2f",
        )
        + "%",
    )

    print(
        "Confusion matrix:"
    )

    print(
        np.asarray(
            metrics[
                "confusion_matrix"
            ]
        )
    )

    print(
        "Per-class recall:",
        metrics[
            "per_class_recall"
        ],
    )

    print(
        "Saved:",
        OUTPUT_PATH,
    )

    print(
        "Saved:",
        METRICS_PATH,
    )

    print(
        "[PASS] M2 used 200 calibration samples"
    )

    print(
        "[PASS] M2 input and output are INT8"
    )

    print(
        "[PASS] M2 full INT8 PTQ completed"
    )


if __name__ == "__main__":
    raise SystemExit(main())
