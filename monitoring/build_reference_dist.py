#!/usr/bin/env python3
"""Build the PSI reference distribution from 300 clean Normal windows."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


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

from data_pipeline.preprocessing import (
    FEATURE_NAMES,
    SensorSample,
    SlidingWindowProcessor,
    load_training_statistics,
)

from data_pipeline.simulator import (
    generate_offline_samples,
)

from monitoring.psi_common import (
    PSI_BINS,
    PSI_BIN_LABELS,
    confidence_distribution,
)


MODEL_PATH = Path(
    "inference/model.tflite"
)

STATISTICS_PATH = Path(
    "data_pipeline/training_stats.npy"
)

OUTPUT_PATH = Path(
    "monitoring/reference_dist.json"
)

REFERENCE_WINDOW_COUNT = 300
REFERENCE_DURATION_SECONDS = 3020
REFERENCE_SEED = 4200


def file_sha256(path):
    """Return the SHA-256 digest of one file."""

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


class TFLitePredictor:
    """Small TFLite prediction wrapper."""

    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_path)
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter
            .get_input_details()
        )

        self.output_details = (
            self.interpreter
            .get_output_details()
        )

        input_shape = list(
            self.input_details[0]["shape"]
        )

        output_shape = list(
            self.output_details[0]["shape"]
        )

        if input_shape != [1, 6]:
            raise ValueError(
                "Expected input shape [1, 6]"
            )

        if output_shape != [1, 3]:
            raise ValueError(
                "Expected output shape [1, 3]"
            )

    def probabilities(self, features):
        """Run one model inference."""

        input_value = np.asarray(
            features,
            dtype=np.float32,
        ).reshape(1, 6)

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            input_value,
        )

        self.interpreter.invoke()

        return self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )[0].astype(np.float64)


def generate_clean_windows():
    """Generate exactly 300 clean normalized feature windows."""

    raw_samples = generate_offline_samples(
        anomaly="none",
        duration_seconds=(
            REFERENCE_DURATION_SECONDS
        ),
        seed=REFERENCE_SEED,
    )

    statistics = load_training_statistics(
        STATISTICS_PATH
    )

    processor = SlidingWindowProcessor(
        statistics=statistics
    )

    windows = []

    for raw_sample in raw_samples:
        sample = SensorSample(
            timestamp=float(
                raw_sample["timestamp"]
            ),
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
            vibration_updated=bool(
                raw_sample[
                    "vibration_updated"
                ]
            ),
        )

        windows.extend(
            processor.add_sample(sample)
        )

    if len(windows) != REFERENCE_WINDOW_COUNT:
        raise RuntimeError(
            "Expected exactly 300 windows, received "
            + str(len(windows))
        )

    return windows


def main():
    """Generate and save the reference distribution."""

    for required_path in [
        MODEL_PATH,
        STATISTICS_PATH,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                str(required_path)
            )

    predictor = TFLitePredictor(
        MODEL_PATH
    )

    windows = generate_clean_windows()

    normal_class_confidences = []

    predicted_labels = []

    for window in windows:
        normalized_features = window[
            "normalized_features"
        ]

        probabilities = predictor.probabilities(
            normalized_features
        )

        predicted_labels.append(
            int(
                np.argmax(probabilities)
            )
        )

        normal_class_confidences.append(
            float(probabilities[0])
        )

    scores = np.asarray(
        normal_class_confidences,
        dtype=np.float64,
    )

    counts, proportions = (
        confidence_distribution(scores)
    )

    normal_prediction_rate = float(
        np.mean(
            np.asarray(
                predicted_labels
            ) == 0
        )
    )

    payload = {
        "schema_version": "1.0",
        "metric": (
            "model output Normal-class "
            "confidence probability"
        ),
        "score_definition": (
            "probabilities[0], representing P(Normal)"
        ),
        "reference_source": (
            "300 clean Normal-class windows "
            "from simulator anomaly mode none"
        ),
        "reference_window_count": (
            REFERENCE_WINDOW_COUNT
        ),
        "simulated_duration_seconds": (
            REFERENCE_DURATION_SECONDS
        ),
        "random_seed": REFERENCE_SEED,
        "feature_names": FEATURE_NAMES,
        "bins": PSI_BINS.tolist(),
        "bin_labels": PSI_BIN_LABELS,
        "counts": counts.tolist(),
        "proportions": proportions.tolist(),
        "minimum_confidence": float(
            np.min(scores)
        ),
        "maximum_confidence": float(
            np.max(scores)
        ),
        "mean_confidence": float(
            np.mean(scores)
        ),
        "normal_prediction_rate": (
            normal_prediction_rate
        ),
        "model_path": str(
            MODEL_PATH
        ),
        "model_sha256": file_sha256(
            MODEL_PATH
        ),
        "statistics_path": str(
            STATISTICS_PATH
        ),
        "statistics_sha256": file_sha256(
            STATISTICS_PATH
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("LogiEdge PSI Reference Distribution")
    print("=" * 46)

    print(
        "Reference windows:",
        len(windows),
    )

    print(
        "Score definition:",
        payload["score_definition"],
    )

    print(
        "Bins:",
        payload["bin_labels"],
    )

    print(
        "Counts:",
        payload["counts"],
    )

    print(
        "Proportions:",
        payload["proportions"],
    )

    print(
        "Mean Normal confidence:",
        format(
            payload["mean_confidence"],
            ".6f",
        ),
    )

    print(
        "Normal prediction rate:",
        format(
            normal_prediction_rate * 100.0,
            ".2f",
        )
        + "%",
    )

    print(
        "Saved:",
        OUTPUT_PATH,
    )

    print(
        "[PASS] Inference completed on 300 clean Normal windows"
    )

    print(
        "[PASS] Four-bin reference distribution saved"
    )


if __name__ == "__main__":
    raise SystemExit(main())
