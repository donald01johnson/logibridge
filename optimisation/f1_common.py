#!/usr/bin/env python3
"""Shared utilities for LogiBridge F1 model variants."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


CLASS_NAMES = [
    "Normal",
    "Warning",
    "Critical",
]

CALIBRATION_SAMPLE_COUNT = 200


def load_dataset(path):
    """Load arrays required by optimisation scripts."""

    source = np.load(path)

    required = [
        "train_features",
        "train_labels",
        "validation_features",
        "validation_labels",
        "feature_names",
    ]

    missing = [
        name
        for name in required
        if name not in source.files
    ]

    if missing:
        raise ValueError(
            "Dataset is missing: "
            + ", ".join(missing)
        )

    result = {
        "train_features": source[
            "train_features"
        ].astype(np.float32),
        "train_labels": source[
            "train_labels"
        ].astype(np.int64),
        "validation_features": source[
            "validation_features"
        ].astype(np.float32),
        "validation_labels": source[
            "validation_labels"
        ].astype(np.int64),
        "feature_names": source[
            "feature_names"
        ].astype(str).tolist(),
    }

    if len(
        result["train_features"]
    ) < CALIBRATION_SAMPLE_COUNT:
        raise ValueError(
            "At least 200 training samples "
            "are required for calibration"
        )

    if result[
        "train_features"
    ].shape[1] != 6:
        raise ValueError(
            "Expected six model features"
        )

    return result


def select_calibration_samples(
    training_features,
    sample_count=CALIBRATION_SAMPLE_COUNT,
    seed=42,
):
    """Select exactly 200 reproducible calibration rows."""

    features = np.asarray(
        training_features,
        dtype=np.float32,
    )

    if len(features) < sample_count:
        raise ValueError(
            "Insufficient calibration samples"
        )

    generator = np.random.default_rng(
        seed
    )

    selected_indices = generator.choice(
        len(features),
        size=sample_count,
        replace=False,
    )

    return features[
        selected_indices
    ]


def representative_dataset(
    calibration_features,
):
    """Yield float32 [1, 6] inputs for TFLite calibration."""

    features = np.asarray(
        calibration_features,
        dtype=np.float32,
    )

    def generator():
        for row in features:
            yield [
                np.asarray(
                    row,
                    dtype=np.float32,
                ).reshape(1, 6)
            ]

    return generator


def convert_full_int8(
    model,
    calibration_features,
):
    """Convert one Keras model to integer-only TFLite."""

    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(model)
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = (
        representative_dataset(
            calibration_features
        )
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet
        .TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = (
        tf.int8
    )

    converter.inference_output_type = (
        tf.int8
    )

    return converter.convert()


def quantize_input(
    float_input,
    input_details,
):
    """Convert one float32 row to the interpreter input dtype."""

    values = np.asarray(
        float_input,
        dtype=np.float32,
    ).reshape(1, 6)

    dtype = input_details["dtype"]

    if dtype == np.float32:
        return values

    scale, zero_point = (
        input_details["quantization"]
    )

    if scale <= 0:
        raise ValueError(
            "Input quantization scale is invalid"
        )

    quantized = np.round(
        values / scale + zero_point
    )

    limits = np.iinfo(dtype)

    quantized = np.clip(
        quantized,
        limits.min,
        limits.max,
    )

    return quantized.astype(dtype)


def dequantize_output(
    output_value,
    output_details,
):
    """Convert interpreter output to float probabilities."""

    values = np.asarray(
        output_value
    )

    if values.dtype == np.float32:
        return values.astype(
            np.float32
        )

    scale, zero_point = (
        output_details["quantization"]
    )

    if scale <= 0:
        raise ValueError(
            "Output quantization scale is invalid"
        )

    return (
        (
            values.astype(np.float32)
            - float(zero_point)
        )
        * float(scale)
    )


def evaluate_tflite(
    model_path,
    features,
    labels,
):
    """Evaluate a TFLite model on held-out validation data."""

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()[0]
    )

    output_details = (
        interpreter.get_output_details()[0]
    )

    predictions = []
    probabilities = []

    for row in features:
        input_value = quantize_input(
            row,
            input_details,
        )

        interpreter.set_tensor(
            input_details["index"],
            input_value,
        )

        interpreter.invoke()

        raw_output = interpreter.get_tensor(
            output_details["index"]
        )[0]

        float_output = (
            dequantize_output(
                raw_output,
                output_details,
            )
        )

        probabilities.append(
            float_output
        )

        predictions.append(
            int(
                np.argmax(
                    float_output
                )
            )
        )

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    accuracy = float(
        np.mean(
            predictions == labels
        )
    )

    confusion_matrix = np.zeros(
        (3, 3),
        dtype=np.int64,
    )

    for true_label, prediction in zip(
        labels,
        predictions,
    ):
        confusion_matrix[
            int(true_label),
            int(prediction),
        ] += 1

    recalls = {}

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        denominator = int(
            np.sum(
                confusion_matrix[
                    class_index,
                    :,
                ]
            )
        )

        if denominator == 0:
            recall = 0.0
        else:
            recall = float(
                confusion_matrix[
                    class_index,
                    class_index,
                ]
                / denominator
            )

        recalls[class_name] = recall

    return {
        "accuracy": accuracy,
        "predictions": (
            predictions
        ),
        "probabilities": np.asarray(
            probabilities,
            dtype=np.float32,
        ),
        "confusion_matrix": (
            confusion_matrix
        ),
        "per_class_recall": recalls,
        "input_dtype": str(
            input_details["dtype"]
        ),
        "output_dtype": str(
            output_details["dtype"]
        ),
        "input_quantization": [
            float(
                input_details[
                    "quantization"
                ][0]
            ),
            int(
                input_details[
                    "quantization"
                ][1]
            ),
        ],
        "output_quantization": [
            float(
                output_details[
                    "quantization"
                ][0]
            ),
            int(
                output_details[
                    "quantization"
                ][1]
            ),
        ],
    }


def write_json(path, payload):
    """Write one JSON evidence artifact."""

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
