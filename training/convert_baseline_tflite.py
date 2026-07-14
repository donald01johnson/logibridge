#!/usr/bin/env python3
"""Convert the trained FP32 Keras baseline to FP32 TFLite."""

import os
from pathlib import Path

os.environ.setdefault(
    "TF_USE_LEGACY_KERAS",
    "1",
)

import numpy as np
import tensorflow as tf
import tf_keras


KERAS_MODEL = Path(
    "training/models/baseline_model.keras"
)

TFLITE_MODEL = Path(
    "inference/model.tflite"
)


def main():
    if not KERAS_MODEL.exists():
        raise FileNotFoundError(
            str(KERAS_MODEL)
        )

    model = tf_keras.models.load_model(
        KERAS_MODEL
    )

    converter = (
        tf.lite.TFLiteConverter
        .from_keras_model(model)
    )

    tflite_bytes = converter.convert()

    TFLITE_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TFLITE_MODEL.write_bytes(
        tflite_bytes
    )

    interpreter = tf.lite.Interpreter(
        model_path=str(TFLITE_MODEL)
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    if list(
        input_details[0]["shape"]
    ) != [1, 6]:
        raise ValueError(
            "Unexpected TFLite input shape"
        )

    sample = np.zeros(
        (1, 6),
        dtype=np.float32,
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        sample,
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    if list(output.shape) != [1, 3]:
        raise ValueError(
            "Unexpected TFLite output shape"
        )

    print(
        "TFLite model:",
        TFLITE_MODEL,
    )

    print(
        "File size bytes:",
        TFLITE_MODEL.stat().st_size,
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
        "[PASS] Baseline TFLite conversion and inference"
    )


if __name__ == "__main__":
    raise SystemExit(main())
