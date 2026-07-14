#!/usr/bin/env python3
"""Create and validate a changed TFLite model for the D2 OTA demo."""

import hashlib
import os
from pathlib import Path

os.environ.setdefault(
    "TF_USE_LEGACY_KERAS",
    "1",
)

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "2",
)

import numpy as np
import tensorflow as tf
import tf_keras


KERAS_MODEL_PATH = Path(
    "training/models/baseline_model.keras"
)

DATASET_PATH = Path(
    "training/dataset.npz"
)

CURRENT_TFLITE_PATH = Path(
    "inference/model.tflite"
)

CANDIDATE_PATH = Path(
    "/tmp/logibridge-ota-candidate.tflite"
)


def sha256(path):
    """Return the SHA-256 hash of one file."""

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while True:
            block = file_handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def run_tflite(model_path, features):
    """Run TFLite inference over all feature rows."""

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    input_shape = list(
        input_details[0]["shape"]
    )

    output_shape = list(
        output_details[0]["shape"]
    )

    if input_shape != [1, 6]:
        raise ValueError(
            "Unexpected input shape: "
            + str(input_shape)
        )

    if output_shape != [1, 3]:
        raise ValueError(
            "Unexpected output shape: "
            + str(output_shape)
        )

    probabilities = []

    for row in features:
        input_value = np.asarray(
            row,
            dtype=np.float32,
        ).reshape(1, 6)

        interpreter.set_tensor(
            input_details[0]["index"],
            input_value,
        )

        interpreter.invoke()

        output_value = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        probabilities.append(
            output_value
        )

    return np.asarray(
        probabilities,
        dtype=np.float32,
    )


def create_candidate(delta):
    """Modify one trained weight and convert the model."""

    model = tf_keras.models.load_model(
        KERAS_MODEL_PATH
    )

    weights = model.get_weights()

    if not weights:
        raise RuntimeError(
            "The Keras model has no weights"
        )

    changed_weights = [
        np.array(
            weight,
            copy=True,
        )
        for weight in weights
    ]

    original_weight = float(
        changed_weights[0].flat[0]
    )

    changed_weights[0].flat[0] = (
        changed_weights[0].flat[0]
        + np.float32(delta)
    )

    changed_weight = float(
        changed_weights[0].flat[0]
    )

    model.set_weights(
        changed_weights
    )

    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(model)
    )

    candidate_bytes = converter.convert()

    CANDIDATE_PATH.write_bytes(
        candidate_bytes
    )

    return (
        original_weight,
        changed_weight,
    )


def main():
    """Generate a changed model and verify it before replacement."""

    for required_path in [
        KERAS_MODEL_PATH,
        DATASET_PATH,
        CURRENT_TFLITE_PATH,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                str(required_path)
            )

    baseline_hash = sha256(
        CURRENT_TFLITE_PATH
    )

    dataset = np.load(
        DATASET_PATH
    )

    validation_features = dataset[
        "validation_features"
    ].astype(np.float32)

    validation_labels = dataset[
        "validation_labels"
    ].astype(np.int64)

    candidate_deltas = [
        0.001,
        0.005,
        0.01,
    ]

    selected_delta = None
    original_weight = None
    changed_weight = None
    candidate_hash = None

    for delta in candidate_deltas:
        (
            original_weight,
            changed_weight,
        ) = create_candidate(delta)

        candidate_hash = sha256(
            CANDIDATE_PATH
        )

        print(
            "Tried delta:",
            delta,
        )

        print(
            "Candidate hash:",
            candidate_hash,
        )

        if candidate_hash != baseline_hash:
            selected_delta = delta
            break

    if selected_delta is None:
        raise RuntimeError(
            "TFLite output remained identical "
            "for every candidate delta"
        )

    candidate_probabilities = run_tflite(
        CANDIDATE_PATH,
        validation_features,
    )

    candidate_predictions = np.argmax(
        candidate_probabilities,
        axis=1,
    )

    candidate_accuracy = float(
        np.mean(
            candidate_predictions
            == validation_labels
        )
    )

    if candidate_accuracy <= 0.88:
        raise RuntimeError(
            "Candidate OTA model accuracy does "
            "not exceed 88 percent"
        )

    CURRENT_TFLITE_PATH.write_bytes(
        CANDIDATE_PATH.read_bytes()
    )

    final_hash = sha256(
        CURRENT_TFLITE_PATH
    )

    if final_hash == baseline_hash:
        raise RuntimeError(
            "Final model hash did not change"
        )

    print()
    print("OTA Demo Model")
    print("=" * 42)

    print(
        "Selected weight delta:",
        selected_delta,
    )

    print(
        "Original weight:",
        original_weight,
    )

    print(
        "Changed weight:",
        changed_weight,
    )

    print(
        "Baseline hash:",
        baseline_hash,
    )

    print(
        "Updated hash:",
        final_hash,
    )

    print(
        "Updated model size bytes:",
        CURRENT_TFLITE_PATH.stat().st_size,
    )

    print(
        "Updated validation accuracy:",
        format(
            candidate_accuracy * 100.0,
            ".2f",
        )
        + "%",
    )

    print(
        "[PASS] TFLite model bytes changed"
    )

    print(
        "[PASS] Updated TFLite model is valid"
    )

    print(
        "[PASS] Updated validation accuracy exceeds 88%"
    )


if __name__ == "__main__":
    raise SystemExit(main())
