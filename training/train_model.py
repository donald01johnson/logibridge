#!/usr/bin/env python3
"""Train and evaluate the LogiBridge baseline MLP classifier.

Assignment component D1.

Architecture:

* Six input features
* Dense layer with 32 ReLU units
* Dense layer with 16 ReLU units
* Three-class softmax output

Classes:

0 - Normal
1 - Warning
2 - Critical
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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

EXPECTED_FEATURE_COUNT = 6
VALIDATION_ACCURACY_TARGET = 0.88
CRITICAL_RECALL_TARGET = 0.95


def configure_reproducibility(seed):
    """Configure reproducible pseudo-random behavior."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def load_dataset(dataset_path):
    """Load and validate the generated dataset."""

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            "Dataset file does not exist: " + str(path)
        )

    dataset = np.load(path)

    required_arrays = [
        "feature_names",
        "class_names",
        "train_features",
        "train_labels",
        "validation_features",
        "validation_labels",
        "test_features",
        "test_labels",
    ]

    missing_arrays = [
        array_name
        for array_name in required_arrays
        if array_name not in dataset.files
    ]

    if missing_arrays:
        raise ValueError(
            "Dataset is missing arrays: "
            + ", ".join(missing_arrays)
        )

    result = {
        "feature_names": dataset[
            "feature_names"
        ].astype(str),
        "class_names": dataset[
            "class_names"
        ].astype(str),
        "train_features": dataset[
            "train_features"
        ].astype(np.float32),
        "train_labels": dataset[
            "train_labels"
        ].astype(np.int64),
        "validation_features": dataset[
            "validation_features"
        ].astype(np.float32),
        "validation_labels": dataset[
            "validation_labels"
        ].astype(np.int64),
        "test_features": dataset[
            "test_features"
        ].astype(np.float32),
        "test_labels": dataset[
            "test_labels"
        ].astype(np.int64),
    }

    validate_dataset(result)

    return result


def validate_dataset(dataset):
    """Validate dataset dimensions, labels, and numeric values."""

    feature_arrays = [
        "train_features",
        "validation_features",
        "test_features",
    ]

    label_arrays = [
        "train_labels",
        "validation_labels",
        "test_labels",
    ]

    for array_name in feature_arrays:
        values = dataset[array_name]

        if values.ndim != 2:
            raise ValueError(
                array_name
                + " must be a two-dimensional matrix"
            )

        if values.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                array_name
                + " must contain exactly six features"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                array_name
                + " contains non-finite values"
            )

    for array_name in label_arrays:
        values = dataset[array_name]

        if values.ndim != 1:
            raise ValueError(
                array_name
                + " must be one-dimensional"
            )

        unique_labels = set(
            np.unique(values).tolist()
        )

        if unique_labels != {0, 1, 2}:
            raise ValueError(
                array_name
                + " must contain classes 0, 1, and 2"
            )

    matching_pairs = [
        ("train_features", "train_labels"),
        (
            "validation_features",
            "validation_labels",
        ),
        ("test_features", "test_labels"),
    ]

    for feature_name, label_name in matching_pairs:
        if len(dataset[feature_name]) != len(
            dataset[label_name]
        ):
            raise ValueError(
                feature_name
                + " and "
                + label_name
                + " contain different sample counts"
            )

    if len(dataset["feature_names"]) != 6:
        raise ValueError(
            "Dataset feature-name list must contain six values"
        )


def build_model(learning_rate):
    """Build the assignment-specified baseline MLP."""

    model = tf_keras.Sequential(
        [
            tf_keras.layers.Input(
                shape=(EXPECTED_FEATURE_COUNT,),
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

    optimizer = tf_keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
        ],
    )

    return model


def calculate_class_weights(labels):
    """Calculate balanced weights from the training labels."""

    labels = np.asarray(labels, dtype=np.int64)

    total_count = len(labels)
    class_count = len(CLASS_NAMES)

    weights = {}

    for class_label in range(class_count):
        label_count = int(
            np.sum(labels == class_label)
        )

        if label_count == 0:
            raise ValueError(
                "Training split has an empty class"
            )

        weights[class_label] = (
            total_count
            / (class_count * label_count)
        )

    return weights


def save_history_csv(path, history):
    """Save epoch-level training history."""

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_keys = list(history.history.keys())
    epoch_count = len(
        history.history[history_keys[0]]
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_handle:
        writer = csv.writer(file_handle)

        writer.writerow(
            ["epoch"] + history_keys
        )

        for epoch_index in range(epoch_count):
            writer.writerow(
                [epoch_index + 1]
                + [
                    float(
                        history.history[key][
                            epoch_index
                        ]
                    )
                    for key in history_keys
                ]
            )


def plot_training_history(path, history):
    """Save loss and accuracy curves."""

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    axes[0].set_title("Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Sparse categorical cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs,
        history.history["accuracy"],
        label="Training accuracy",
    )

    axes[1].plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation accuracy",
    )

    axes[1].axhline(
        VALIDATION_ACCURACY_TARGET,
        color="red",
        linestyle="--",
        label="88% target",
    )

    axes[1].set_title("Model Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_confusion_matrix(path, matrix):
    """Save the test-set confusion matrix."""

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(6.5, 5.5)
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(image, ax=axis)

    axis.set(
        title="LogiBridge Test Confusion Matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )

    threshold = matrix.max() / 2.0

    for row_index in range(matrix.shape[0]):
        for column_index in range(
            matrix.shape[1]
        ):
            value = int(
                matrix[row_index, column_index]
            )

            axis.text(
                column_index,
                row_index,
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
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def evaluate_predictions(
    true_labels,
    predicted_labels,
):
    """Calculate classification metrics."""

    overall_accuracy = float(
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

    for class_label, class_name in enumerate(
        CLASS_NAMES
    ):
        per_class[class_name] = {
            "label": class_label,
            "precision": float(
                precision[class_label]
            ),
            "recall": float(
                recall[class_label]
            ),
            "f1_score": float(
                f1_score[class_label]
            ),
            "support": int(
                support[class_label]
            ),
        }

    return {
        "accuracy": overall_accuracy,
        "critical_recall": float(recall[2]),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def save_metrics_json(
    path,
    model,
    dataset,
    training_history,
    validation_loss,
    validation_accuracy,
    test_loss,
    evaluation,
    seed,
):
    """Save machine-readable training and evaluation results."""

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_accuracy = float(
        max(training_history.history["val_accuracy"])
    )

    best_validation_loss = float(
        min(training_history.history["val_loss"])
    )

    parameter_count = int(
        model.count_params()
    )

    metrics = {
        "schema_version": "1.0",
        "model_name": model.name,
        "architecture": {
            "input_features": 6,
            "hidden_layers": [32, 16],
            "hidden_activation": "relu",
            "output_classes": 3,
            "output_activation": "softmax",
            "parameter_count": parameter_count,
        },
        "feature_names": [
            str(value)
            for value in dataset["feature_names"]
        ],
        "class_names": CLASS_NAMES,
        "random_seed": seed,
        "dataset": {
            "training_samples": int(
                len(dataset["train_labels"])
            ),
            "validation_samples": int(
                len(dataset["validation_labels"])
            ),
            "test_samples": int(
                len(dataset["test_labels"])
            ),
        },
        "training": {
            "epochs_completed": int(
                len(
                    training_history.history[
                        "loss"
                    ]
                )
            ),
            "best_validation_accuracy": (
                best_validation_accuracy
            ),
            "best_validation_loss": (
                best_validation_loss
            ),
            "restored_validation_loss": float(
                validation_loss
            ),
            "restored_validation_accuracy": float(
                validation_accuracy
            ),
        },
        "test": {
            "loss": float(test_loss),
            "accuracy": evaluation["accuracy"],
            "critical_recall": evaluation[
                "critical_recall"
            ],
            "per_class": evaluation[
                "per_class"
            ],
            "confusion_matrix": evaluation[
                "confusion_matrix"
            ].tolist(),
        },
        "thresholds": {
            "validation_accuracy_target": (
                VALIDATION_ACCURACY_TARGET
            ),
            "critical_recall_target": (
                CRITICAL_RECALL_TARGET
            ),
            "validation_accuracy_passed": bool(
                validation_accuracy
                >= VALIDATION_ACCURACY_TARGET
            ),
            "critical_recall_passed": bool(
                evaluation["critical_recall"]
                >= CRITICAL_RECALL_TARGET
            ),
        },
    }

    output_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def train_model(arguments):
    """Train, evaluate, save, and validate the baseline model."""

    configure_reproducibility(arguments.seed)

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

    checkpoint_path = (
        output_directory
        / "baseline_best.keras"
    )

    final_model_path = (
        output_directory
        / "baseline_model.keras"
    )

    history_csv_path = (
        output_directory
        / "training_history.csv"
    )

    history_plot_path = (
        output_directory
        / "training_history.png"
    )

    confusion_plot_path = (
        output_directory
        / "confusion_matrix.png"
    )

    metrics_path = (
        output_directory
        / "baseline_metrics.json"
    )

    print("LogiBridge Baseline MLP Training")
    print("=" * 42)

    print(
        "TensorFlow version:",
        tf.__version__,
    )

    print(
        "tf_keras version:",
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

    print(
        "Test shape:",
        dataset["test_features"].shape,
    )

    model = build_model(
        learning_rate=arguments.learning_rate
    )

    print()
    model.summary()

    class_weights = calculate_class_weights(
        dataset["train_labels"]
    )

    print()
    print("Class weights:", class_weights)

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
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        dataset["train_features"],
        dataset["train_labels"],
        validation_data=(
            dataset["validation_features"],
            dataset["validation_labels"],
        ),
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )

    validation_loss, validation_accuracy = (
        model.evaluate(
            dataset["validation_features"],
            dataset["validation_labels"],
            verbose=0,
        )
    )

    test_loss, model_test_accuracy = (
        model.evaluate(
            dataset["test_features"],
            dataset["test_labels"],
            verbose=0,
        )
    )

    test_probabilities = model.predict(
        dataset["test_features"],
        batch_size=arguments.batch_size,
        verbose=0,
    )

    predicted_labels = np.argmax(
        test_probabilities,
        axis=1,
    )

    evaluation = evaluate_predictions(
        dataset["test_labels"],
        predicted_labels,
    )

    if not np.isclose(
        model_test_accuracy,
        evaluation["accuracy"],
        atol=0.000001,
    ):
        raise RuntimeError(
            "Keras and sklearn test accuracies differ"
        )

    report_text = classification_report(
        dataset["test_labels"],
        predicted_labels,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    model.save(final_model_path)

    save_history_csv(
        history_csv_path,
        history,
    )

    plot_training_history(
        history_plot_path,
        history,
    )

    plot_confusion_matrix(
        confusion_plot_path,
        evaluation["confusion_matrix"],
    )

    save_metrics_json(
        path=metrics_path,
        model=model,
        dataset=dataset,
        training_history=history,
        validation_loss=validation_loss,
        validation_accuracy=validation_accuracy,
        test_loss=test_loss,
        evaluation=evaluation,
        seed=arguments.seed,
    )

    report_path = (
        output_directory
        / "classification_report.txt"
    )

    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    print()
    print("Evaluation Results")
    print("=" * 42)

    print(
        "Validation loss:",
        format(float(validation_loss), ".6f"),
    )

    print(
        "Validation accuracy:",
        format(
            float(validation_accuracy) * 100.0,
            ".2f",
        )
        + "%",
    )

    print(
        "Test loss:",
        format(float(test_loss), ".6f"),
    )

    print(
        "Test accuracy:",
        format(
            evaluation["accuracy"] * 100.0,
            ".2f",
        )
        + "%",
    )

    print(
        "Critical recall:",
        format(
            evaluation["critical_recall"]
            * 100.0,
            ".2f",
        )
        + "%",
    )

    print()
    print("Classification report:")
    print(report_text)

    print("Confusion matrix:")
    print(evaluation["confusion_matrix"])

    print()
    print("Saved model:", final_model_path)
    print("Best checkpoint:", checkpoint_path)
    print("Saved metrics:", metrics_path)
    print("Saved history:", history_csv_path)
    print("Saved history plot:", history_plot_path)
    print(
        "Saved confusion matrix:",
        confusion_plot_path,
    )

    validation_passed = (
        validation_accuracy
        >= VALIDATION_ACCURACY_TARGET
    )

    critical_recall_passed = (
        evaluation["critical_recall"]
        >= CRITICAL_RECALL_TARGET
    )

    print()

    if validation_passed:
        print(
            "[PASS] Validation accuracy is at least 88%"
        )
    else:
        print(
            "[FAIL] Validation accuracy is below 88%"
        )

    if critical_recall_passed:
        print(
            "[PASS] Critical-class recall is at least 95%"
        )
    else:
        print(
            "[FAIL] Critical-class recall is below 95%"
        )

    if not validation_passed:
        raise RuntimeError(
            "Validation-accuracy requirement was not met"
        )

    if not critical_recall_passed:
        raise RuntimeError(
            "Critical-recall requirement was not met"
        )

    print("[PASS] Baseline model training completed")

    return {
        "model_path": final_model_path,
        "metrics_path": metrics_path,
        "validation_accuracy": float(
            validation_accuracy
        ),
        "test_accuracy": evaluation["accuracy"],
        "critical_recall": evaluation[
            "critical_recall"
        ],
    }


def build_parser():
    """Build command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the LogiBridge baseline MLP"
        )
    )

    parser.add_argument(
        "--dataset",
        default="training/dataset.npz",
        help="Generated dataset path",
    )

    parser.add_argument(
        "--output-dir",
        default="training/models",
        help="Model and evaluation output directory",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Maximum training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial Adam learning rate",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser


def main():
    """Program entry point."""

    arguments = build_parser().parse_args()

    train_model(arguments)

    return 0


if __name__ == "__main__":
    sys.exit(main())
