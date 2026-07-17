#!/usr/bin/env python3
"""Benchmark LogiBridge M1, M2, and M3 using the F2 methodology.

Required metrics:
1. Mean inference latency over 200 measured runs
2. p95 inference latency
3. Model file size
4. Held-out validation accuracy
5. Estimated energy per inference using E = P * t

Ten warm-up runs are performed and excluded from latency measurements.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
import tensorflow as tf


WARMUP_RUNS = 10
BENCHMARK_RUNS = 200
DEFAULT_THREADS = 1
CPU_SAMPLE_SECONDS = 1.0

CLASS_NAMES = [
    "Normal",
    "Warning",
    "Critical",
]

VARIANTS = [
    {
        "variant": "M1",
        "name": "FP32 Baseline",
        "path": Path(
            "optimisation/models/"
            "m1_fp32.tflite"
        ),
    },
    {
        "variant": "M2",
        "name": "PTQ Full INT8",
        "path": Path(
            "optimisation/models/"
            "m2_int8.tflite"
        ),
    },
    {
        "variant": "M3",
        "name": (
            "35% Pruned + "
            "Full INT8 PTQ"
        ),
        "path": Path(
            "optimisation/models/"
            "m3_pruned_int8.tflite"
        ),
    },
]


def sha256(path):
    """Return a file SHA-256 digest."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_validation_dataset(path):
    """Load held-out validation features and labels."""

    source = np.load(path)

    required = [
        "validation_features",
        "validation_labels",
        "feature_names",
        "class_names",
    ]

    missing = [
        name
        for name in required
        if name not in source.files
    ]

    if missing:
        raise ValueError(
            "Dataset is missing arrays: "
            + ", ".join(missing)
        )

    features = source[
        "validation_features"
    ].astype(np.float32)

    labels = source[
        "validation_labels"
    ].astype(np.int64)

    feature_names = source[
        "feature_names"
    ].astype(str).tolist()

    class_names = source[
        "class_names"
    ].astype(str).tolist()

    if features.ndim != 2:
        raise ValueError(
            "Validation features must be 2D"
        )

    if features.shape[1] != 6:
        raise ValueError(
            "Validation data must contain "
            "six features"
        )

    if len(features) != len(labels):
        raise ValueError(
            "Feature and label counts differ"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Validation features contain "
            "invalid values"
        )

    return {
        "features": features,
        "labels": labels,
        "feature_names": feature_names,
        "class_names": class_names,
    }


def create_interpreter(
    model_path,
    num_threads,
):
    """Create and validate one TFLite interpreter."""

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=num_threads,
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()[0]
    )

    output_details = (
        interpreter.get_output_details()[0]
    )

    if list(
        input_details["shape"]
    ) != [1, 6]:
        raise ValueError(
            "Unexpected input shape for "
            + str(model_path)
            + ": "
            + str(
                input_details["shape"]
            )
        )

    if list(
        output_details["shape"]
    ) != [1, 3]:
        raise ValueError(
            "Unexpected output shape for "
            + str(model_path)
            + ": "
            + str(
                output_details["shape"]
            )
        )

    return (
        interpreter,
        input_details,
        output_details,
    )


def quantize_input(
    float_features,
    input_details,
):
    """Convert one normalized feature row to model input dtype."""

    values = np.asarray(
        float_features,
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
            "INT8 input has invalid "
            "quantization scale"
        )

    quantized = np.round(
        values / float(scale)
        + float(zero_point)
    )

    limits = np.iinfo(dtype)

    quantized = np.clip(
        quantized,
        limits.min,
        limits.max,
    )

    return quantized.astype(dtype)


def dequantize_output(
    raw_output,
    output_details,
):
    """Convert model output to float probabilities."""

    values = np.asarray(
        raw_output
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
            "INT8 output has invalid "
            "quantization scale"
        )

    return (
        values.astype(np.float32)
        - float(zero_point)
    ) * float(scale)


def run_one_inference(
    interpreter,
    input_details,
    output_details,
    feature_row,
):
    """Set the tensor, invoke, and return output probabilities."""

    input_value = quantize_input(
        feature_row,
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

    return dequantize_output(
        raw_output,
        output_details,
    )


def calculate_confusion_matrix(
    true_labels,
    predicted_labels,
):
    """Return a 3x3 confusion matrix."""

    matrix = np.zeros(
        (3, 3),
        dtype=np.int64,
    )

    for true_label, predicted_label in zip(
        true_labels,
        predicted_labels,
    ):
        matrix[
            int(true_label),
            int(predicted_label),
        ] += 1

    return matrix


def calculate_recalls(matrix):
    """Calculate recall for every class."""

    recalls = {}

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        support = int(
            np.sum(
                matrix[
                    class_index,
                    :,
                ]
            )
        )

        if support == 0:
            recall = 0.0
        else:
            recall = float(
                matrix[
                    class_index,
                    class_index,
                ]
                / support
            )

        recalls[class_name] = recall

    return recalls


def evaluate_accuracy(
    model_path,
    num_threads,
    features,
    labels,
):
    """Evaluate one variant on the complete held-out set."""

    (
        interpreter,
        input_details,
        output_details,
    ) = create_interpreter(
        model_path,
        num_threads,
    )

    predictions = []

    for row in features:
        probabilities = (
            run_one_inference(
                interpreter,
                input_details,
                output_details,
                row,
            )
        )

        predictions.append(
            int(
                np.argmax(
                    probabilities
                )
            )
        )

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    accuracy = float(
        np.mean(
            predictions == labels
        )
    )

    confusion_matrix = (
        calculate_confusion_matrix(
            labels,
            predictions,
        )
    )

    return {
        "accuracy": accuracy,
        "confusion_matrix": (
            confusion_matrix
        ),
        "per_class_recall": (
            calculate_recalls(
                confusion_matrix
            )
        ),
        "predictions": predictions,
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


def benchmark_latency(
    model_path,
    num_threads,
    features,
):
    """Run 10 excluded warm-ups and exactly 200 timed inferences."""

    (
        interpreter,
        input_details,
        output_details,
    ) = create_interpreter(
        model_path,
        num_threads,
    )

    for run_index in range(
        WARMUP_RUNS
    ):
        row = features[
            run_index
            % len(features)
        ]

        run_one_inference(
            interpreter,
            input_details,
            output_details,
            row,
        )

    latencies_ms = []

    for run_index in range(
        BENCHMARK_RUNS
    ):
        row = features[
            run_index
            % len(features)
        ]

        start = time.perf_counter_ns()

        run_one_inference(
            interpreter,
            input_details,
            output_details,
            row,
        )

        end = time.perf_counter_ns()

        latency_ms = (
            end - start
        ) / 1_000_000.0

        latencies_ms.append(
            float(latency_ms)
        )

    if len(
        latencies_ms
    ) != BENCHMARK_RUNS:
        raise RuntimeError(
            "Expected exactly 200 measured runs"
        )

    return {
        "latencies_ms": latencies_ms,
        "mean_ms": float(
            statistics.mean(
                latencies_ms
            )
        ),
        "median_ms": float(
            statistics.median(
                latencies_ms
            )
        ),
        "p95_ms": float(
            np.percentile(
                latencies_ms,
                95,
                method="linear",
            )
        ),
        "p99_ms": float(
            np.percentile(
                latencies_ms,
                99,
                method="linear",
            )
        ),
        "minimum_ms": float(
            min(latencies_ms)
        ),
        "maximum_ms": float(
            max(latencies_ms)
        ),
        "standard_deviation_ms": float(
            statistics.stdev(
                latencies_ms
            )
        ),
    }


def measure_cpu_utilisation(
    model_path,
    num_threads,
    features,
    duration_seconds,
):
    """Measure process CPU percent under sustained model inference.

    This is a separate sampling block. Its invocations are not included
    in the required 200-run latency distribution.
    """

    (
        interpreter,
        input_details,
        output_details,
    ) = create_interpreter(
        model_path,
        num_threads,
    )

    process = psutil.Process(
        os.getpid()
    )

    process.cpu_percent(
        interval=None
    )

    start = time.perf_counter()
    deadline = (
        start
        + duration_seconds
    )

    invocation_count = 0

    while time.perf_counter() < deadline:
        row = features[
            invocation_count
            % len(features)
        ]

        run_one_inference(
            interpreter,
            input_details,
            output_details,
            row,
        )

        invocation_count += 1

    elapsed_seconds = (
        time.perf_counter()
        - start
    )

    process_cpu_percent = float(
        process.cpu_percent(
            interval=None
        )
    )

    if invocation_count == 0:
        raise RuntimeError(
            "CPU sampling performed "
            "no inferences"
        )

    return {
        "process_cpu_percent": (
            process_cpu_percent
        ),
        "sampling_seconds": (
            elapsed_seconds
        ),
        "sampling_inferences": (
            invocation_count
        ),
    }


def benchmark_variant(
    variant,
    dataset,
    num_threads,
    tdp_watts,
    cpu_sample_seconds,
):
    """Measure all five required metrics for one model."""

    model_path = variant["path"]

    if not model_path.exists():
        raise FileNotFoundError(
            str(model_path)
        )

    accuracy_result = (
        evaluate_accuracy(
            model_path=model_path,
            num_threads=num_threads,
            features=dataset[
                "features"
            ],
            labels=dataset[
                "labels"
            ],
        )
    )

    latency_result = (
        benchmark_latency(
            model_path=model_path,
            num_threads=num_threads,
            features=dataset[
                "features"
            ],
        )
    )

    cpu_result = (
        measure_cpu_utilisation(
            model_path=model_path,
            num_threads=num_threads,
            features=dataset[
                "features"
            ],
            duration_seconds=(
                cpu_sample_seconds
            ),
        )
    )

    process_cpu_percent = (
        cpu_result[
            "process_cpu_percent"
        ]
    )

    cpu_fraction = min(
        max(
            process_cpu_percent
            / 100.0,
            0.0,
        ),
        1.0,
    )

    estimated_power_watts = (
        tdp_watts
        * cpu_fraction
    )

    estimated_energy_mj = (
        estimated_power_watts
        * latency_result[
            "mean_ms"
        ]
    )

    file_size_bytes = int(
        model_path.stat().st_size
    )

    file_size_kb = (
        file_size_bytes
        / 1024.0
    )

    return {
        "variant": variant["variant"],
        "name": variant["name"],
        "model_path": str(
            model_path
        ),
        "sha256": sha256(
            model_path
        ),
        "threads": num_threads,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": (
            BENCHMARK_RUNS
        ),
        "mean_latency_ms": (
            latency_result["mean_ms"]
        ),
        "median_latency_ms": (
            latency_result[
                "median_ms"
            ]
        ),
        "p95_latency_ms": (
            latency_result["p95_ms"]
        ),
        "p99_latency_ms": (
            latency_result["p99_ms"]
        ),
        "minimum_latency_ms": (
            latency_result[
                "minimum_ms"
            ]
        ),
        "maximum_latency_ms": (
            latency_result[
                "maximum_ms"
            ]
        ),
        "latency_std_ms": (
            latency_result[
                "standard_deviation_ms"
            ]
        ),
        "latencies_ms": (
            latency_result[
                "latencies_ms"
            ]
        ),
        "file_size_bytes": (
            file_size_bytes
        ),
        "file_size_kb": (
            file_size_kb
        ),
        "validation_samples": int(
            len(
                dataset[
                    "labels"
                ]
            )
        ),
        "validation_accuracy": (
            accuracy_result[
                "accuracy"
            ]
        ),
        "validation_accuracy_percent": (
            accuracy_result[
                "accuracy"
            ] * 100.0
        ),
        "confusion_matrix": (
            accuracy_result[
                "confusion_matrix"
            ].tolist()
        ),
        "per_class_recall": (
            accuracy_result[
                "per_class_recall"
            ]
        ),
        "input_dtype": (
            accuracy_result[
                "input_dtype"
            ]
        ),
        "output_dtype": (
            accuracy_result[
                "output_dtype"
            ]
        ),
        "input_quantization": (
            accuracy_result[
                "input_quantization"
            ]
        ),
        "output_quantization": (
            accuracy_result[
                "output_quantization"
            ]
        ),
        "laptop_tdp_watts": (
            tdp_watts
        ),
        "process_cpu_percent": (
            process_cpu_percent
        ),
        "cpu_utilisation_fraction": (
            cpu_fraction
        ),
        "estimated_power_watts": (
            estimated_power_watts
        ),
        "estimated_energy_mj": (
            estimated_energy_mj
        ),
        "cpu_sampling_seconds": (
            cpu_result[
                "sampling_seconds"
            ]
        ),
        "cpu_sampling_inferences": (
            cpu_result[
                "sampling_inferences"
            ]
        ),
        "energy_formula": (
            "energy_mJ = "
            "TDP_W * min(process_CPU_percent/100, 1) "
            "* mean_latency_ms"
        ),
    }


def write_csv(path, results):
    """Save a compact five-metric comparison table."""

    fieldnames = [
        "variant",
        "name",
        "mean_latency_ms",
        "p95_latency_ms",
        "file_size_kb",
        "validation_accuracy_percent",
        "estimated_energy_mj",
        "process_cpu_percent",
        "estimated_power_watts",
        "critical_recall_percent",
    ]

    with Path(path).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "variant": (
                        result["variant"]
                    ),
                    "name": result["name"],
                    "mean_latency_ms": (
                        format(
                            result[
                                "mean_latency_ms"
                            ],
                            ".9f",
                        )
                    ),
                    "p95_latency_ms": (
                        format(
                            result[
                                "p95_latency_ms"
                            ],
                            ".9f",
                        )
                    ),
                    "file_size_kb": (
                        format(
                            result[
                                "file_size_kb"
                            ],
                            ".6f",
                        )
                    ),
                    "validation_accuracy_percent": (
                        format(
                            result[
                                "validation_accuracy_percent"
                            ],
                            ".4f",
                        )
                    ),
                    "estimated_energy_mj": (
                        format(
                            result[
                                "estimated_energy_mj"
                            ],
                            ".9f",
                        )
                    ),
                    "process_cpu_percent": (
                        format(
                            result[
                                "process_cpu_percent"
                            ],
                            ".4f",
                        )
                    ),
                    "estimated_power_watts": (
                        format(
                            result[
                                "estimated_power_watts"
                            ],
                            ".6f",
                        )
                    ),
                    "critical_recall_percent": (
                        format(
                            result[
                                "per_class_recall"
                            ]["Critical"]
                            * 100.0,
                            ".4f",
                        )
                    ),
                }
            )


def print_summary(results):
    """Print the five required metrics."""

    print()
    print("F2 Five-Metric Benchmark Results")
    print("=" * 112)

    heading = (
        f'{"Variant":<8}'
        f'{"Mean ms":>14}'
        f'{"p95 ms":>14}'
        f'{"Size KB":>14}'
        f'{"Accuracy %":>16}'
        f'{"Energy mJ":>16}'
        f'{"CPU %":>12}'
    )

    print(heading)
    print("-" * 112)

    for result in results:
        print(
            f'{result["variant"]:<8}'
            f'{result["mean_latency_ms"]:>14.6f}'
            f'{result["p95_latency_ms"]:>14.6f}'
            f'{result["file_size_kb"]:>14.3f}'
            f'{result["validation_accuracy_percent"]:>16.2f}'
            f'{result["estimated_energy_mj"]:>16.6f}'
            f'{result["process_cpu_percent"]:>12.2f}'
        )

    print()


def build_parser():
    """Build command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LogiBridge M1, M2, "
            "and M3 on five F2 metrics"
        )
    )

    parser.add_argument(
        "--dataset",
        default="training/dataset.npz",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
    )

    parser.add_argument(
        "--tdp-watts",
        type=float,
        required=True,
        help=(
            "Documented laptop CPU TDP "
            "used for energy estimation"
        ),
    )

    parser.add_argument(
        "--cpu-sample-seconds",
        type=float,
        default=CPU_SAMPLE_SECONDS,
    )

    parser.add_argument(
        "--output-json",
        default=(
            "optimisation/results/"
            "benchmark_results.json"
        ),
    )

    parser.add_argument(
        "--output-csv",
        default=(
            "optimisation/results/"
            "benchmark_results.csv"
        ),
    )

    return parser


def main():
    """Benchmark all three variants."""

    arguments = (
        build_parser().parse_args()
    )

    if arguments.threads <= 0:
        raise ValueError(
            "Thread count must be positive"
        )

    if arguments.tdp_watts <= 0:
        raise ValueError(
            "TDP must be positive"
        )

    if (
        arguments.cpu_sample_seconds
        < 0.5
    ):
        raise ValueError(
            "CPU sample duration must "
            "be at least 0.5 seconds"
        )

    dataset = (
        load_validation_dataset(
            arguments.dataset
        )
    )

    output_json = Path(
        arguments.output_json
    )

    output_csv = Path(
        arguments.output_csv
    )

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    print("LogiBridge F2 Benchmark")
    print("=" * 48)

    print(
        "Warm-up runs:",
        WARMUP_RUNS,
        "(excluded)",
    )

    print(
        "Measured latency runs:",
        BENCHMARK_RUNS,
    )

    print(
        "TFLite threads:",
        arguments.threads,
    )

    print(
        "Laptop TDP assumption:",
        arguments.tdp_watts,
        "W",
    )

    print(
        "Validation samples:",
        len(
            dataset["labels"]
        ),
    )

    for variant in VARIANTS:
        print()
        print(
            "Benchmarking",
            variant["variant"],
            "-",
            variant["name"],
        )

        result = benchmark_variant(
            variant=variant,
            dataset=dataset,
            num_threads=(
                arguments.threads
            ),
            tdp_watts=(
                arguments.tdp_watts
            ),
            cpu_sample_seconds=(
                arguments.cpu_sample_seconds
            ),
        )

        results.append(result)

        print(
            " Mean latency:",
            format(
                result[
                    "mean_latency_ms"
                ],
                ".6f",
            ),
            "ms",
        )

        print(
            " p95 latency:",
            format(
                result[
                    "p95_latency_ms"
                ],
                ".6f",
            ),
            "ms",
        )

        print(
            " File size:",
            format(
                result[
                    "file_size_kb"
                ],
                ".3f",
            ),
            "KB",
        )

        print(
            " Accuracy:",
            format(
                result[
                    "validation_accuracy_percent"
                ],
                ".2f",
            ),
            "%",
        )

        print(
            " Estimated energy:",
            format(
                result[
                    "estimated_energy_mj"
                ],
                ".6f",
            ),
            "mJ",
        )

    payload = {
        "schema_version": "1.0",
        "methodology": {
            "warmup_runs": WARMUP_RUNS,
            "warmup_excluded": True,
            "measured_runs": (
                BENCHMARK_RUNS
            ),
            "latency_timer": (
                "time.perf_counter_ns"
            ),
            "p95_method": (
                "numpy percentile, "
                "linear interpolation"
            ),
            "file_size_unit": (
                "KB using 1024 bytes"
            ),
            "accuracy_source": (
                "complete held-out "
                "validation split"
            ),
            "energy_estimation": (
                "E=P*t using psutil "
                "process CPU percent and "
                "documented laptop TDP"
            ),
            "threads": (
                arguments.threads
            ),
            "laptop_tdp_watts": (
                arguments.tdp_watts
            ),
            "cpu_sampling_seconds": (
                arguments.cpu_sample_seconds
            ),
            "energy_is_estimate": True,
        },
        "hardware": {
            "platform": (
                platform.platform()
            ),
            "processor": (
                platform.processor()
            ),
            "machine": (
                platform.machine()
            ),
            "physical_cpu_cores": (
                psutil.cpu_count(
                    logical=False
                )
            ),
            "logical_cpu_count": (
                psutil.cpu_count(
                    logical=True
                )
            ),
        },
        "feature_names": (
            dataset["feature_names"]
        ),
        "class_names": (
            dataset["class_names"]
        ),
        "results": results,
    }

    output_json.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(
        output_csv,
        results,
    )

    print_summary(
        results
    )

    print(
        "Saved JSON:",
        output_json,
    )

    print(
        "Saved CSV:",
        output_csv,
    )

    print()
    print(
        "[PASS] Ten warm-up runs excluded"
    )

    print(
        "[PASS] Exactly 200 latency "
        "runs measured per variant"
    )

    print(
        "[PASS] All five F2 metrics "
        "measured for M1, M2, and M3"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
