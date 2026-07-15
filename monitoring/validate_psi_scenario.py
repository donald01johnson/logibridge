#!/usr/bin/env python3
"""Validate PSI drift and recovery using the trained LogiEdge model."""

import json
import sys
from collections import deque
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
    SensorSample,
    SlidingWindowProcessor,
    load_training_statistics,
)

from data_pipeline.simulator import (
    generate_offline_samples,
)

from monitoring.psi_common import (
    calculate_psi,
    confidence_distribution,
)


MODEL_PATH = Path(
    "inference/model.tflite"
)

STATISTICS_PATH = Path(
    "data_pipeline/training_stats.npy"
)

REFERENCE_PATH = Path(
    "monitoring/reference_dist.json"
)

OUTPUT_PATH = Path(
    "monitoring/psi_scenario_results.json"
)

ROLLING_WINDOW_SIZE = 100
DRIFT_THRESHOLD = 0.25
RECOVERY_THRESHOLD = 0.10


class TFLitePredictor:
    """Run the deployed TFLite classifier."""

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

    def normal_confidence(self, features):
        """Return model output P(Normal)."""

        input_value = np.asarray(
            features,
            dtype=np.float32,
        ).reshape(1, 6)

        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            input_value,
        )

        self.interpreter.invoke()

        probabilities = (
            self.interpreter.get_tensor(
                self.output_details[0][
                    "index"
                ]
            )[0]
        )

        return float(
            probabilities[0]
        )


def generate_confidences(
    predictor,
    mode,
    duration_seconds,
    seed,
):
    """Generate model Normal-class confidences for one mode."""

    raw_samples = generate_offline_samples(
        anomaly=mode,
        duration_seconds=duration_seconds,
        seed=seed,
    )

    statistics = load_training_statistics(
        STATISTICS_PATH
    )

    processor = SlidingWindowProcessor(
        statistics=statistics
    )

    confidence_values = []

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

        windows = processor.add_sample(
            sample
        )

        for window in windows:
            confidence_values.append(
                predictor.normal_confidence(
                    window[
                        "normalized_features"
                    ]
                )
            )

    return confidence_values


def rolling_psi(
    rolling_scores,
    reference_proportions,
):
    """Calculate PSI for a full rolling confidence window."""

    if len(
        rolling_scores
    ) != ROLLING_WINDOW_SIZE:
        raise ValueError(
            "Rolling window must contain exactly 100 scores"
        )

    _, current_proportions = (
        confidence_distribution(
            list(rolling_scores)
        )
    )

    psi_value, _ = calculate_psi(
        reference_proportions,
        current_proportions,
    )

    return (
        psi_value,
        current_proportions,
    )


def main():
    """Validate clean, drift, and recovery phases."""

    reference = json.loads(
        REFERENCE_PATH.read_text(
            encoding="utf-8"
        )
    )

    reference_proportions = np.asarray(
        reference["proportions"],
        dtype=np.float64,
    )

    predictor = TFLitePredictor(
        MODEL_PATH
    )

    clean_scores = generate_confidences(
        predictor=predictor,
        mode="none",
        duration_seconds=1020,
        seed=5100,
    )

    if len(clean_scores) != 100:
        raise RuntimeError(
            "Expected 100 initial clean windows, received "
            + str(len(clean_scores))
        )

    drift_scores = generate_confidences(
        predictor=predictor,
        mode="combined",
        duration_seconds=300,
        seed=5200,
    )

    recovery_scores = generate_confidences(
        predictor=predictor,
        mode="none",
        duration_seconds=1020,
        seed=5300,
    )

    if len(recovery_scores) != 100:
        raise RuntimeError(
            "Expected 100 recovery windows, received "
            + str(len(recovery_scores))
        )

    rolling_scores = deque(
        clean_scores,
        maxlen=ROLLING_WINDOW_SIZE,
    )

    clean_psi, clean_distribution = (
        rolling_psi(
            rolling_scores,
            reference_proportions,
        )
    )

    drift_events = []

    drift_crossed = False
    first_drift_cross_seconds = None

    for index, score in enumerate(
        drift_scores,
        start=1,
    ):
        rolling_scores.append(score)

        logical_seconds = (
            30
            + (index - 1) * 10
        )

        if logical_seconds % 60 == 0:
            (
                psi_value,
                current_distribution,
            ) = rolling_psi(
                rolling_scores,
                reference_proportions,
            )

            drift_events.append(
                {
                    "logical_seconds": (
                        logical_seconds
                    ),
                    "psi": psi_value,
                    "distribution": (
                        current_distribution.tolist()
                    ),
                }
            )

            print(
                "DRIFT "
                + str(logical_seconds)
                + "s PSI="
                + format(
                    psi_value,
                    ".3f",
                )
            )

            if (
                not drift_crossed
                and psi_value
                > DRIFT_THRESHOLD
            ):
                drift_crossed = True

                first_drift_cross_seconds = (
                    logical_seconds
                )

                print(
                    "[LOGIBRIDGE DRIFT ALERT] PSI="
                    + format(
                        psi_value,
                        ".3f",
                    )
                )

    recovery_events = []

    recovered = False
    first_recovery_seconds = None

    for index, score in enumerate(
        recovery_scores,
        start=1,
    ):
        rolling_scores.append(score)

        logical_seconds = (
            30
            + (index - 1) * 10
        )

        if logical_seconds % 60 == 0:
            (
                psi_value,
                current_distribution,
            ) = rolling_psi(
                rolling_scores,
                reference_proportions,
            )

            recovery_events.append(
                {
                    "logical_seconds": (
                        logical_seconds
                    ),
                    "psi": psi_value,
                    "distribution": (
                        current_distribution.tolist()
                    ),
                }
            )

            print(
                "RECOVERY "
                + str(logical_seconds)
                + "s PSI="
                + format(
                    psi_value,
                    ".3f",
                )
            )

            if (
                not recovered
                and psi_value
                < RECOVERY_THRESHOLD
            ):
                recovered = True

                first_recovery_seconds = (
                    logical_seconds
                )

                print(
                    "[LOGIBRIDGE RECOVERY] PSI="
                    + format(
                        psi_value,
                        ".3f",
                    )
                )

    results = {
        "schema_version": "1.0",
        "score_definition": (
            "probabilities[0], representing P(Normal)"
        ),
        "reference_window_count": 300,
        "rolling_window_size": (
            ROLLING_WINDOW_SIZE
        ),
        "monitor_interval_seconds": 60,
        "drift_threshold": (
            DRIFT_THRESHOLD
        ),
        "recovery_threshold": (
            RECOVERY_THRESHOLD
        ),
        "initial_clean_psi": (
            clean_psi
        ),
        "initial_clean_distribution": (
            clean_distribution.tolist()
        ),
        "drift_score_count": len(
            drift_scores
        ),
        "drift_events": drift_events,
        "drift_crossed_threshold": (
            drift_crossed
        ),
        "first_drift_cross_seconds": (
            first_drift_cross_seconds
        ),
        "recovery_score_count": len(
            recovery_scores
        ),
        "recovery_events": (
            recovery_events
        ),
        "recovered_below_threshold": (
            recovered
        ),
        "first_recovery_seconds": (
            first_recovery_seconds
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            results,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("PSI Scenario Summary")
    print("=" * 42)

    print(
        "Initial clean PSI:",
        format(
            clean_psi,
            ".3f",
        ),
    )

    print(
        "Drift crossed 0.25:",
        drift_crossed,
    )

    print(
        "First drift crossing:",
        first_drift_cross_seconds,
        "logical seconds",
    )

    print(
        "Recovery below 0.10:",
        recovered,
    )

    print(
        "First recovery:",
        first_recovery_seconds,
        "logical seconds",
    )

    if not drift_crossed:
        raise RuntimeError(
            "Combined anomaly did not cross PSI 0.25"
        )

    if (
        first_drift_cross_seconds is None
        or first_drift_cross_seconds > 300
    ):
        raise RuntimeError(
            "PSI did not cross 0.25 within five minutes"
        )

    if not recovered:
        raise RuntimeError(
            "PSI did not recover below 0.10"
        )

    print()
    print(
        "[PASS] Clean Normal rolling PSI established"
    )

    print(
        "[PASS] Combined anomaly crossed PSI 0.25 within five minutes"
    )

    print(
        "[PASS] Clean recovery returned PSI below 0.10"
    )

    print(
        "[PASS] E1 drift scenario validated"
    )


if __name__ == "__main__":
    raise SystemExit(main())
