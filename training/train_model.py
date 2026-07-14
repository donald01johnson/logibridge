#!/usr/bin/env python3
"""Train and evaluate the assignment-aligned LogiBridge MLP.

Assignment Task D1:

* Six input features
* Hidden layers containing 32 and 16 ReLU units
* Three-class softmax output
* Held-out 20 percent validation evaluation
* Validation accuracy must exceed 88 percent

The script also performs the Task C2 mandatory experiment by comparing
validation accuracy using correct fixed statistics and means shifted by
three standard deviations.
"""

import argparse
import csv
import json
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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_keras

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


CLASS_NAMES = [
    "Normal",
    "Warning",
    "Critical",
]

EXPECTED_FEATURE_NAMES = [
    "temperature_mean_c",
    "temperature_std_c",
    "temperature_rate_c_per_min",
    "vibration_rms_g",
    "vibration_peak_g",
    "vibration_kurtosis",
]

VALIDATION_ACCURACY_TARGET = 0.88


def configure_reproducibility(seed):
    """Configure deterministic pseudo-random behaviour."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_dataset(path):
    """Load and validate the corrected D1 dataset."""

    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset does not exist: "
            + str(dataset_path)
        )

    source = np.load(dataset_path)

    required_arrays = [
        "feature_names",
        "class_names",
        "raw_train_features",
        "train_features",
        "train_labels",
        "raw_validation_features",
        "validation_features",
        "validation_labels",
        "fixed_training_mean",
        "fixed_training_std",
    ]

    missing = [
        name
        for name in required_arrays
        if name not in source.files
    ]

    if missing:
        raise ValueError(
            "Dataset is missing arrays: "
            + ", ".join(missing)
        )

    dataset = {
        "feature_names": source[
            "feature_names"
        ].astype(str).tolist(),
        "class_names": source[
            "class_names"
        ].astype(str).tolist(),
        "raw_train_features": source[
            "raw_train_features"
        ].astype(np.float32),
        "train_features": source[
            "train_features"
        ].astype(np.float32),
        "train_labels": source[
            "train_labels"
        ].astype(np.int64),
        "raw_validation_features": source[
            "raw_validation_features"
        ].astype(np.float32),
        "validation_features": source[
            "validation_features"
        ].astype(np.float32),
        "validation_labels": source[
            "validation_labels"
        ].astype(np.int64),
        "fixed_training_mean": source[
            "fixed_training_mean"
        ].astype(np.float32),
        "fixed_training_std": source[
            "fixed_training_std"
        ].astype(np.float32),
    }

    validate_dataset(dataset)

    return dataset


def validate_dataset(dataset):
    """Validate feature order, dimensions, labels, and values."""

    if dataset["feature_names"] != EXPECTED_FEATURE_NAMES:
        raise ValueError(
            "Dataset feature order does not match corrected C2"
        )

    if dataset["class_names"] != CLASS_NAMES:
        raise ValueError(
            "Dataset class names are incorrect"
        )

    feature_arrays = [
        "raw_train_features",
        "train_features",
        "raw_validation_features",
        "validation_features",
    ]

    for name in feature_arrays:
        values = dataset[name]

        if values.ndim != 2:
            raise ValueError(
                name + " must be two-dimensional"
            )

        if values.shape[1] != 6:
            raise ValueError(
                name + " must contain six features"
            )

        if not np.isfinite(values).all():
            raise ValueError(
                name + " contains invalid values"
            )

    if len(
        dataset["train_features"]
    ) != len(
        dataset["train_labels"]
    ):
        raise ValueError(
            "Training feature and label counts differ"
        )

    if len(
        dataset["validation_features"]
    ) != len(
        dataset["validation_labels"]
    ):
        raise ValueError(
            "Validation feature and label counts differ"
        )

    for label_name in [
        "train_labels",
        "validation_labels",
    ]:
        unique_labels = set(
            np.unique(
                dataset[label_name]
            ).tolist()
        )

        if unique_labels != {0, 1, 2}:
            raise ValueError(
                label_name
                + " must contain all three classes"
            )

    if dataset[
        "fixed_training_mean"
    ].shape != (6,):
        raise ValueError(
            "Fixed mean must have shape (6,)"
        )

    if dataset[
        "fixed_training_std"
    ].shape != (6,):
        raise ValueError(
            "Fixed standard deviation must have shape (6,)"
        )

    if np.any(
        dataset["fixed_training_std"] <= 0
    ):
        raise ValueError(
            "Fixed standard deviations must be positive"
        )


def build_model(learning_rate):
    """Build the recommended 32-16 ReLU MLP."""

    model = tf_keras.Sequential(
        [
            tf_keras.layers.Input(
                shape=(6,),
                name="sensor_features",
            ),
            tf_keras.layers.Dense(
                32,
                activation="relu",
                name="hidden_32",
            ),
            tf_keras.layers.Dense(
                16,
                activation="relu",
                name="hidden_16",
            ),
            tf_keras.layers.Dense(
                3,
                activation="softmax",
                name="cargo_state",
            ),
        ],
        name="logibridge_baseline_mlp",
    )

    model.compile(
        optimizer=tf_keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=(
            "sparse_categorical_crossentropy"
        ),
        metrics=["accuracy"],
    )

    return model


def calculate_class_weights(labels):
    """Calculate balanced training-class weights."""

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    total = len(labels)
    class_total = len(CLASS_NAMES)

    weights = {}

    for label in range(class_total):
        count = int(
            np.sum(labels == label)
        )

        if count == 0:
            raise ValueError(
                "Training class is empty"
            )

        weights[label] = (
            total
            / float(class_total * count)
        )

    return weights


def predict_labels(model, features, batch_size):
    """Return probabilities and predicted labels."""

    probabilities = model.predict(
        features,
        batch_size=batch_size,
        verbose=0,
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1,
    )

    return probabilities, predicted_labels


def evaluate_predictions(true_labels, predicted_labels):
    """Calculate accuracy and per-class metrics."""

    accuracy = float(
        accuracy_score(
            true_labels,
            predicted_labels,
        )
    )

    precision, recall, f1_score, support = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=[0, 1, 2],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
    )

    per_class = {}

    for label in range(3):
        class_name = CLASS_NAMES[label]

        per_class[class_name] = {
            "label": label,
            "precision": float(
                precision[label]
            ),
            "recall": float(
                recall[label]
            ),
            "f1_score": float(
                f1_score[label]
            ),
            "support": int(
                support[label]
            ),
        }

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def normalize_with_statistics(
    raw_features,
    mean,
    standard_deviation,
):
    """Normalize raw values with supplied fixed statistics."""

    raw_values = np.asarray(
        raw_features,
        dtype=np.float32,
    )

    mean_values = np.asarray(
        mean,
        dtype=np.float32,
    )

    std_values = np.asarray(
        standard_deviation,
        dtype=np.float32,
    )

    normalized = (
        raw_values - mean_values
    ) / std_values

    return normalized.astype(np.float32)


def run_shifted_statistics_experiment(
    model,
    dataset,
    batch_size,
):
    """Measure accuracy using correct and 3-sigma shifted means."""

    mean = dataset[
        "fixed_training_mean"
    ]

    standard_deviation = dataset[
        "fixed_training_std"
    ]

    raw_validation = dataset[
        "raw_validation_features"
    ]

    validation_labels = dataset[
        "validation_labels"
    ]

    correct_features = (
        normalize_with_statistics(
            raw_features=raw_validation,
            mean=mean,
            standard_deviation=standard_deviation,
        )
    )

    shifted_mean = (
        mean
        + 3.0 * standard_deviation
    )

    shifted_features = (
        normalize_with_statistics(
            raw_features=raw_validation,
            mean=shifted_mean,
            standard_deviation=standard_deviation,
        )
    )

    _, correct_predictions = predict_labels(
        model,
        correct_features,
        batch_size,
    )

    _, shifted_predictions = predict_labels(
        model,
        shifted_features,
        batch_size,
    )

    correct_accuracy = float(
        accuracy_score(
            validation_labels,
            correct_predictions,
        )
    )

    shifted_accuracy = float(
        accuracy_score(
            validation_labels,
            shifted_predictions,
        )
    )

    accuracy_change = (
        shifted_accuracy
        - correct_accuracy
    )

    return {
        "correct_accuracy": correct_accuracy,
        "shifted_accuracy": shifted_accuracy,
        "accuracy_change": accuracy_change,
        "accuracy_drop": (
            correct_accuracy
            - shifted_accuracy
        ),
        "sigma_shift": 3.0,
        "shifted_mean": shifted_mean,
        "correct_predictions": (
            correct_predictions
        ),
        "shifted_predictions": (
            shifted_predictions
        ),
    }


def save_history_csv(path, history):
    """Save epoch-level learning history."""

    output_path = Path(path)

    keys = list(
        history.history.keys()
    )

    epoch_count = len(
        history.history[keys[0]]
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file_handle:
        writer = csv.writer(file_handle)

        writer.writerow(
            ["epoch"] + keys
        )

        for epoch_index in range(
            epoch_count
        ):
            row = [epoch_index + 1]

            for key in keys:
                row.append(
                    float(
                        history.history[key][
                            epoch_index
                        ]
                    )
                )

            writer.writerow(row)


def plot_training_history(path, history):
    """Save training and validation curves."""

    epochs = np.arange(
        1,
        len(history.history["loss"]) + 1,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
    )

    axes[0].plot(
        epochs,
        history.history["loss"],
        label="Training loss",
    )

    axes[0].plot(
        epochs,
        history.history["val_loss"],
        label="Validation loss",
    )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        history.history["accuracy"],
        label="Training accuracy",
    )

    axes[1].plot(
        epochs,
        history.history[
            "val_accuracy"
        ],
        label="Validation accuracy",
    )

    axes[1].axhline(
        VALIDATION_ACCURACY_TARGET,
        color="red",
        linestyle="--",
        label="88% requirement",
    )

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_confusion_matrix(path, matrix):
    """Save held-out validation confusion matrix."""

    figure, axis = plt.subplots(
        figsize=(6.5, 5.5)
    )

    image = axis.imshow(
        matrix,
        cmap="Blues",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set_title(
        "Held-Out Validation Confusion Matrix"
    )

    axis.set_xlabel(
        "Predicted class"
    )

    axis.set_ylabel(
        "True class"
    )

    axis.set_xticks(
        np.arange(3)
    )

    axis.set_yticks(
        np.arange(3)
    )

    axis.set_xticklabels(
        CLASS_NAMES
    )

    axis.set_yticklabels(
        CLASS_NAMES
    )

    threshold = (
        float(np.max(matrix)) / 2.0
    )

    for row in range(3):
        for column in range(3):
            value = int(
                matrix[row, column]
            )

            axis.text(
                column,
                row,
                str(value),
                horizontalalignment="center",
                verticalalignment="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_metrics(
    path,
    model,
    dataset,
    history,
    validation_loss,
    evaluation,
    experiment,
    seed,
):
    """Save model and experiment metrics to JSON."""

    metrics = {
        "schema_version": "2.0",
        "assignment_task": "D1",
        "model_name": model.name,
        "architecture": {
            "input_features": 6,
            "hidden_layers": [32, 16],
            "hidden_activation": "relu",
            "output_classes": 3,
            "output_activation": "softmax",
            "parameter_count": int(
                model.count_params()
            ),
        },
        "feature_names": (
            dataset["feature_names"]
        ),
        "class_names": CLASS_NAMES,
        "random_seed": seed,
        "dataset": {
            "training_samples": int(
                len(
                    dataset[
                        "train_labels"
                    ]
                )
            ),
            "validation_samples": int(
                len(
                    dataset[
                        "validation_labels"
                    ]
                )
            ),
            "held_out_validation_fraction": (
                0.20
            ),
        },
        "training": {
            "epochs_completed": int(
                len(
                    history.history[
                        "loss"
                    ]
                )
            ),
            "best_validation_accuracy": float(
                max(
                    history.history[
                        "val_accuracy"
                    ]
                )
            ),
            "best_validation_loss": float(
                min(
                    history.history[
                        "val_loss"
                    ]
                )
            ),
            "restored_validation_loss": float(
                validation_loss
            ),
        },
        "validation": {
            "accuracy": evaluation[
                "accuracy"
            ],
            "per_class": evaluation[
                "per_class"
            ],
            "confusion_matrix": evaluation[
                "confusion_matrix"
            ].tolist(),
            "accuracy_requirement": (
                VALIDATION_ACCURACY_TARGET
            ),
            "requirement_passed": bool(
                evaluation["accuracy"]
                > VALIDATION_ACCURACY_TARGET
            ),
        },
        "shifted_statistics_experiment": {
            "sigma_shift": experiment[
                "sigma_shift"
            ],
            "correct_stats_accuracy": (
                experiment[
                    "correct_accuracy"
                ]
            ),
            "shifted_stats_accuracy": (
                experiment[
                    "shifted_accuracy"
                ]
            ),
            "accuracy_change": (
                experiment[
                    "accuracy_change"
                ]
            ),
            "accuracy_drop": (
                experiment[
                    "accuracy_drop"
                ]
            ),
            "shifted_mean": experiment[
                "shifted_mean"
            ].tolist(),
        },
    }

    Path(path).write_text(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def train(arguments):
    """Train, evaluate, and save the corrected D1 model."""

    configure_reproducibility(
        arguments.seed
    )

    dataset = load_dataset(
        arguments.dataset
    )

    output_directory = Path(
        arguments.output_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_directory
        / "baseline_model.keras"
    )

    best_model_path = (
        output_directory
        / "baseline_best.keras"
    )

    metrics_path = (
        output_directory
        / "baseline_metrics.json"
    )

    report_path = (
        output_directory
        / "classification_report.txt"
    )

    history_csv_path = (
        output_directory
        / "training_history.csv"
    )

    history_plot_path = (
        output_directory
        / "training_history.png"
    )

    confusion_matrix_path = (
        output_directory
        / "confusion_matrix.png"
    )

    print("Corrected LogiBridge D1 Training")
    print("=" * 46)

    print(
        "TensorFlow:",
        tf.__version__,
    )

    print(
        "tf_keras:",
        tf_keras.__version__,
    )

    print(
        "Training shape:",
        dataset["train_features"].shape,
    )

    print(
        "Validation shape:",
        dataset[
            "validation_features"
        ].shape,
    )

    print("Feature order:")

    for index, name in enumerate(
        dataset["feature_names"],
        start=1,
    ):
        print(
            " ",
            index,
            name,
        )

    model = build_model(
        arguments.learning_rate
    )

    print()
    model.summary()

    class_weights = (
        calculate_class_weights(
            dataset["train_labels"]
        )
    )

    print()
    print(
        "Class weights:",
        class_weights,
    )

    callbacks = [
        tf_keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=arguments.patience,
            min_delta=0.0001,
            restore_best_weights=True,
            verbose=1,
        ),
        tf_keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(
                2,
                arguments.patience // 3,
            ),
            min_lr=0.00001,
            verbose=1,
        ),
        tf_keras.callbacks.ModelCheckpoint(
            filepath=str(
                best_model_path
            ),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
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
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    validation_loss, keras_accuracy = (
        model.evaluate(
            dataset[
                "validation_features"
            ],
            dataset[
                "validation_labels"
            ],
            verbose=0,
        )
    )

    _, predicted_labels = predict_labels(
        model,
        dataset[
            "validation_features"
        ],
        arguments.batch_size,
    )

    evaluation = evaluate_predictions(
        dataset[
            "validation_labels"
        ],
        predicted_labels,
    )

    if not np.isclose(
        keras_accuracy,
        evaluation["accuracy"],
        atol=0.000001,
    ):
        raise RuntimeError(
            "Keras and sklearn validation accuracies differ"
        )

    experiment = (
        run_shifted_statistics_experiment(
            model=model,
            dataset=dataset,
            batch_size=arguments.batch_size,
        )
    )

    report = classification_report(
        dataset["validation_labels"],
        predicted_labels,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    model.save(model_path)

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    save_history_csv(
        history_csv_path,
        history,
    )

    plot_training_history(
        history_plot_path,
        history,
    )

    plot_confusion_matrix(
        confusion_matrix_path,
        evaluation[
            "confusion_matrix"
        ],
    )

    
    save_metrics(
        path=metrics_path,
        model=model,
        dataset=dataset,
        history=history,
        validation_loss=validation_loss,
        evaluation=evaluation,
        experiment=experiment,
        seed=arguments.seed,
    )

    print()
    print("Held-Out Validation Results")
    print("=" * 46)

    print(
        "Validation loss:",
        format(
            float(validation_loss),
            ".6f",
        ),
    )

    print(
        "Validation accuracy:",
        format(
            evaluation["accuracy"]
            * 100.0,
            ".2f",
        )
        + "%",
    )

    print()
    print("Classification report:")
    print(report)

    print("Confusion matrix:")
    print(
        evaluation[
            "confusion_matrix"
        ]
    )

    print()
    print("Mandatory Shifted-Statistics Experiment")
    print("=" * 46)

    print(
        "Correct-statistics accuracy:",
        format(
            experiment[
                "correct_accuracy"
            ] * 100.0,
            ".2f",
        )
        + "%",
    )

    print(
        "Shifted-statistics accuracy:",
        format(
            experiment[
                "shifted_accuracy"
            ] * 100.0,
            ".2f",
        )
        + "%",
    )

    print(
        "Accuracy change:",
        format(
            experiment[
                "accuracy_change"
            ] * 100.0,
            ".2f",
        ),
        "percentage points",
    )

    print(
        "Accuracy drop:",
        format(
            experiment[
                "accuracy_drop"
            ] * 100.0,
            ".2f",
        ),
        "percentage points",
    )

    print()
    print("Saved model:", model_path)
    print("Saved best model:", best_model_path)
    print("Saved metrics:", metrics_path)
    print("Saved report:", report_path)
    print(
        "Saved confusion matrix:",
        confusion_matrix_path,
    )

    if (
        evaluation["accuracy"]
        > VALIDATION_ACCURACY_TARGET
    ):
        print(
            "[PASS] Held-out validation accuracy exceeds 88%"
        )
    else:
        print(
            "[FAIL] Held-out validation accuracy does not exceed 88%"
        )

        raise RuntimeError(
            "Assignment validation-accuracy requirement was not met"
        )

    print(
        "[PASS] Correct-versus-shifted statistics experiment completed"
    )

    print(
        "[PASS] Corrected D1 model training completed"
    )


def build_parser():
    """Build command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the corrected LogiBridge D1 MLP"
        )
    )

    parser.add_argument(
        "--dataset",
        default="training/dataset.npz",
    )

    parser.add_argument(
        "--output-dir",
        default="training/models",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
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

    train(arguments)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
