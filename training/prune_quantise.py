#!/usr/bin/env python3
"""Create M3 using 35 percent pruning followed by full INT8 PTQ."""

import hashlib
import math
import os
import random
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
import tensorflow as tf
import tensorflow_model_optimization as tfmot
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

BASELINE_MODEL_PATH = Path(
    "training/models/baseline_model.keras"
)

PRUNED_KERAS_PATH = Path(
    "optimisation/models/m3_pruned.keras"
)

OUTPUT_PATH = Path(
    "optimisation/models/m3_pruned_int8.tflite"
)

METRICS_PATH = Path(
    "optimisation/results/m3_metrics.json"
)

FINAL_SPARSITY = 0.35
PRUNING_EPOCHS = 40
BATCH_SIZE = 16
RANDOM_SEED = 42


def configure_reproducibility():
    """Configure deterministic random seeds."""

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    tf.random.set_seed(
        RANDOM_SEED
    )


def sha256(path):
    """Return a file SHA-256 digest."""

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def calculate_kernel_sparsity(model):
    """Calculate zero fraction across Dense kernels."""

    zero_count = 0
    value_count = 0
    layer_sparsity = {}

    for layer in model.layers:
        if not isinstance(
            layer,
            tf_keras.layers.Dense,
        ):
            continue

        weights = layer.get_weights()

        if not weights:
            continue

        kernel = np.asarray(
            weights[0]
        )

        zeros = int(
            np.sum(
                kernel == 0.0
            )
        )

        total = int(
            kernel.size
        )

        zero_count += zeros
        value_count += total

        layer_sparsity[
            layer.name
        ] = {
            "zero_weights": zeros,
            "total_kernel_weights": (
                total
            ),
            "sparsity": (
                zeros / total
                if total
                else 0.0
            ),
        }

    if value_count == 0:
        raise RuntimeError(
            "No Dense kernels found"
        )

    return {
        "zero_weights": zero_count,
        "total_kernel_weights": (
            value_count
        ),
        "sparsity": (
            zero_count
            / value_count
        ),
        "layers": layer_sparsity,
    }


def main():
    """Prune, fine-tune, strip, quantize, and validate M3."""

    configure_reproducibility()

    dataset = load_dataset(
        DATASET_PATH
    )

    baseline_model = (
        tf_keras.models.load_model(
            BASELINE_MODEL_PATH
        )
    )

    steps_per_epoch = int(
        math.ceil(
            len(
                dataset[
                    "train_features"
                ]
            )
            / BATCH_SIZE
        )
    )

    end_step = (
        steps_per_epoch
        * PRUNING_EPOCHS
    )

    pruning_schedule = (
        tfmot.sparsity.keras
        .PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=(
                FINAL_SPARSITY
            ),
            begin_step=0,
            end_step=end_step,
            power=3,
            frequency=1,
        )
    )

    model_for_pruning = (
        tfmot.sparsity.keras
        .prune_low_magnitude(
            baseline_model,
            pruning_schedule=(
                pruning_schedule
            ),
        )
    )

    model_for_pruning.compile(
        optimizer=tf_keras.optimizers.Adam(
            learning_rate=0.0005
        ),
        loss=(
            "sparse_categorical_crossentropy"
        ),
        metrics=["accuracy"],
    )

    callbacks = [
        tfmot.sparsity.keras
        .UpdatePruningStep(),
    ]

    history = model_for_pruning.fit(
        dataset["train_features"],
        dataset["train_labels"],
        validation_data=(
            dataset[
                "validation_features"
            ],
            dataset[
                "validation_labels"
            ],
        ),
        epochs=PRUNING_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    epochs_completed = len(
        history.history["loss"]
    )

    if epochs_completed != PRUNING_EPOCHS:
        raise RuntimeError(
            "Pruning schedule did not complete. "
            "Expected "
            + str(PRUNING_EPOCHS)
            + " epochs, received "
            + str(epochs_completed)
        )

    stripped_model = (
        tfmot.sparsity.keras
        .strip_pruning(
            model_for_pruning
        )
    )

    sparsity = calculate_kernel_sparsity(
        stripped_model
    )

    if sparsity["sparsity"] < 0.34:
        raise RuntimeError(
            "M3 did not reach approximately "
            "35 percent sparsity"
        )

    PRUNED_KERAS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stripped_model.save(
        PRUNED_KERAS_PATH
    )

    calibration_features = (
        select_calibration_samples(
            dataset["train_features"],
            sample_count=(
                CALIBRATION_SAMPLE_COUNT
            ),
            seed=43,
        )
    )

    converted_model = convert_full_int8(
        stripped_model,
        calibration_features,
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
            "M3 input is not INT8"
        )

    if evaluation[
        "output_dtype"
    ] != "<class 'numpy.int8'>":
        raise RuntimeError(
            "M3 output is not INT8"
        )

    metrics = {
        "variant": "M3",
        "name": (
            "35 Percent Pruned "
            "plus Full INT8 PTQ"
        ),
        "pruning_method": (
            "TFMOT prune_low_magnitude "
            "on Dense kernels"
        ),
        "pruning_schedule": {
            "type": (
                "PolynomialDecay"
            ),
            "initial_sparsity": 0.0,
            "final_sparsity": (
                FINAL_SPARSITY
            ),
            "begin_step": 0,
            "end_step": end_step,
            "power": 3,
            "frequency": 1,
        },
        "pruning_epochs_completed": int(
            len(
                history.history["loss"]
            )
        ),
        "measured_kernel_sparsity": (
            sparsity
        ),
        "calibration_sample_count": 200,
        "quantization": (
            "Full INT8 PTQ"
        ),
        "model_path": str(
            OUTPUT_PATH
        ),
        "pruned_keras_path": str(
            PRUNED_KERAS_PATH
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

    print()
    print("M3 Pruning plus Full INT8 PTQ")
    print("=" * 42)

    print(
        "Pruning epochs completed:",
        metrics[
            "pruning_epochs_completed"
        ],
    )

    print(
        "Target sparsity:",
        format(
            FINAL_SPARSITY
            * 100.0,
            ".2f",
        )
        + "%",
    )

    print(
        "Measured kernel sparsity:",
        format(
            sparsity["sparsity"]
            * 100.0,
            ".2f",
        )
        + "%",
    )

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
        metrics[
            "file_size_bytes"
        ],
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
        "[PASS] M3 used PolynomialDecay"
    )

    print(
        "[PASS] M3 reached approximately 35 percent sparsity"
    )

    print(
        "[PASS] M3 used 200 calibration samples"
    )

    print(
        "[PASS] M3 input and output are INT8"
    )

    print(
        "[PASS] M3 pruning and INT8 PTQ completed"
    )


if __name__ == "__main__":
    raise SystemExit(main())
