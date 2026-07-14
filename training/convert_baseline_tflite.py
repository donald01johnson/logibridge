#!/usr/bin/env python3
"""Convert the corrected LogiBridge baseline model to FP32 TFLite."""

import json
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

TFLITE_MODEL_PATH = Path(
    "inference/model.tflite"
)

METADATA_PATH = Path(
    "inference/model_metadata.json"
)


def run_tflite(interpreter, features):
    """Run a float32 TFLite model over a feature matrix."""

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    outputs = []

    for row in features:
        input_value = np.asarray(
            row,
            dtype=np.float32,
        ).reshape(1, 6)

        interpreter.set_tensor(
            input_index,
            input_value,
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_index
        )[0]

        outputs.append(output)

    return np.asarray(
        outputs,
        dtype=np.float32,
    )


def main():
    """Convert, validate, and save the baseline TFLite model."""

    if not KERAS_MODEL_PATH.exists():
        raise FileNotFoundError(
            str(KERAS_MODEL_PATH)
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            str(DATASET_PATH)
        )

    model = tf_keras.models.load_model(
        KERAS_MODEL_PATH
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

    keras_probabilities = model.predict(
        validation_features,
        verbose=0,
    )

    keras_predictions = np.argmax(
        keras_probabilities,
        axis=1,
    )

    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(model)
    )

    tflite_bytes = converter.convert()

    TFLITE_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TFLITE_MODEL_PATH.write_bytes(
        tflite_bytes
    )

    interpreter = tf.lite.Interpreter(
        model_path=str(
            TFLITE_MODEL_PATH
        )
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if list(
        input_details[0]["shape"]
    ) != [1, 6]:
        raise ValueError(
            "Expected TFLite input shape [1, 6]"
        )

    if list(
        output_details[0]["shape"]
    ) != [1, 3]:
        raise ValueError(
            "Expected TFLite output shape [1, 3]"
        )

    tflite_probabilities = run_tflite(
        interpreter,
        validation_features,
    )

    tflite_predictions = np.argmax(
        tflite_probabilities,
        axis=1,
    )

    keras_accuracy = float(
        np.mean(
            keras_predictions
            == validation_labels
        )
    )

    tflite_accuracy = float(
        np.mean(
            tflite_predictions
            == validation_labels
        )
    )

    prediction_agreement = float(
        np.mean(
            keras_predictions
            == tflite_predictions
        )
    )

    maximum_probability_difference = float(
        np.max(
            np.abs(
                keras_probabilities
                - tflite_probabilities
            )
        )
    )

    if tflite_accuracy <= 0.88:
        raise RuntimeError(
            "TFLite validation accuracy "
            "does not exceed 88 percent"
        )

    if prediction_agreement < 1.0:
        raise RuntimeError(
            "Keras and TFLite class "
            "predictions differ"
        )

    metadata = {
        "schema_version": "1.0",
        "variant": "M1_FP32_TFLite",
        "source_model": str(
            KERAS_MODEL_PATH
        ),
        "tflite_model": str(
            TFLITE_MODEL_PATH
        ),
        "input_shape": [
            int(value)
            for value in input_details[0]["shape"]
        ],
        "output_shape": [
            int(value)
            for value in output_details[0]["shape"]
        ],
        "input_dtype": str(
            input_details[0]["dtype"]
        ),
        "output_dtype": str(
            output_details[0]["dtype"]
        ),
        "file_size_bytes": int(
            TFLITE_MODEL_PATH.stat().st_size
        ),
        "keras_validation_accuracy": (
            keras_accuracy
        ),
        "tflite_validation_accuracy": (
            tflite_accuracy
        ),
        "prediction_agreement": (
            prediction_agreement
        ),
        "maximum_probability_difference": (
            maximum_probability_difference
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Baseline TFLite Conversion")
    print("=" * 42)

    print(
        "Keras validation accuracy:",
        format(
            keras_accuracy * 100.0,
            ".2f",
        )
        + "%",
    )

    print(
        "TFLite validation accuracy:",
        format(
            tflite_accuracy * 100.0,
            ".2f",
        )
        + "%",
    )

    print(
        "Prediction agreement:",
        format(
            prediction_agreement * 100.0,
            ".2f",
        )
        + "%",
    )

    print(
        "Maximum probability difference:",
        format(
            maximum_probability_difference,
            ".8f",
        ),
    )

    print(
        "Input shape:",
        input_details[0]["shape"],
    )

    print(
        "Output shape:",
        output_details[0]["shape"],
    )

    print(
        "Model size bytes:",
        TFLITE_MODEL_PATH.stat().st_size,
    )

    print(
        "Saved model:",
        TFLITE_MODEL_PATH,
    )

    print(
        "Saved metadata:",
        METADATA_PATH,
    )

    print(
        "[PASS] Baseline TFLite conversion"
    )

    print(
        "[PASS] TFLite validation accuracy exceeds 88%"
    )

    print(
        "[PASS] Keras and TFLite predictions agree"
    )


if __name__ == "__main__":
    raise SystemExit(main())
