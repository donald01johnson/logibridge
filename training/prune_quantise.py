#!/usr/bin/env python3
"""Create M3 with scheduled structured Dense-unit pruning + full INT8 PTQ.

The required baseline is a Dense MLP, not a CNN. Therefore complete hidden
units are the Dense equivalent of convolutional filters. TFMOT PolynomialDecay
sets the target pruning fraction over 40 epochs. A custom Keras callback ranks
whole hidden units and masks complete incoming/outgoing connections. The final
masked network is physically compacted before full INT8 PTQ.
"""

import hashlib
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

DATASET_PATH = Path("training/dataset.npz")
BASELINE_MODEL_PATH = Path("training/models/baseline_model.keras")
STRUCTURED_KERAS_PATH = Path("optimisation/models/m3_structured.keras")
OUTPUT_PATH = Path("optimisation/models/m3_pruned_int8.tflite")
METRICS_PATH = Path("optimisation/results/m3_metrics.json")

TARGET_STRUCTURED_SPARSITY = 0.35
PRUNING_EPOCHS = 40
COMPACT_FINETUNE_EPOCHS = 15
BATCH_SIZE = 16
RANDOM_SEED = 42


def configure_reproducibility():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_dense_layers(model):
    layers = [
        layer for layer in model.layers
        if isinstance(layer, tf_keras.layers.Dense)
    ]
    if len(layers) != 3:
        raise RuntimeError(
            "Expected exactly three Dense layers for 6->32->16->3; "
            f"found {len(layers)}"
        )
    if [layer.units for layer in layers] != [32, 16, 3]:
        raise RuntimeError(
            "Expected Dense units [32, 16, 3], received "
            f"{[layer.units for layer in layers]}"
        )
    return layers


def parameter_count(model):
    return int(sum(np.prod(weight.shape) for weight in model.get_weights()))


class StructuredUnitPruning(tf_keras.callbacks.Callback):
    """Apply PolynomialDecay to complete Dense hidden units.

    A hidden unit is removed structurally by zeroing its bias, all incoming
    weights, and all outgoing weights. The lowest-importance whole units are
    selected. Masks are re-applied after every optimizer batch so removed
    units cannot regrow.
    """

    def __init__(self, schedule, final_sparsity, end_step):
        super().__init__()
        self.schedule = schedule
        self.final_sparsity = float(final_sparsity)
        self.end_step = int(end_step)
        self.removed_first = np.asarray([], dtype=np.int64)
        self.removed_second = np.asarray([], dtype=np.int64)
        self.last_sparsity = 0.0

    @staticmethod
    def _lowest_indices(values, count):
        count = int(max(0, min(count, len(values) - 1)))
        if count == 0:
            return np.asarray([], dtype=np.int64)
        return np.argsort(values, kind="stable")[:count].astype(np.int64)

    def _scheduled_sparsity(self, step):
        should_prune, sparsity = self.schedule(
            tf.constant(int(step), dtype=tf.int64)
        )
        should_value = bool(should_prune.numpy())
        sparsity_value = float(sparsity.numpy())
        return should_value, sparsity_value

    def _apply(self, sparsity):
        first, second, output = get_dense_layers(self.model)
        w1, b1 = [np.array(value, copy=True) for value in first.get_weights()]
        w2, b2 = [np.array(value, copy=True) for value in second.get_weights()]
        w3, b3 = [np.array(value, copy=True) for value in output.get_weights()]

        remove_first_count = int(round(first.units * float(sparsity)))
        remove_second_count = int(round(second.units * float(sparsity)))

        # Whole-unit importance uses both incoming and outgoing connections.
        importance_first = (
            np.linalg.norm(w1, axis=0)
            + np.linalg.norm(w2, axis=1)
            + np.abs(b1)
        )
        importance_second = (
            np.linalg.norm(w2, axis=0)
            + np.linalg.norm(w3, axis=1)
            + np.abs(b2)
        )

        self.removed_first = self._lowest_indices(
            importance_first, remove_first_count
        )
        self.removed_second = self._lowest_indices(
            importance_second, remove_second_count
        )

        # First hidden units: incoming column, bias, outgoing row.
        if self.removed_first.size:
            w1[:, self.removed_first] = 0.0
            b1[self.removed_first] = 0.0
            w2[self.removed_first, :] = 0.0

        # Second hidden units: incoming column, bias, outgoing row.
        if self.removed_second.size:
            w2[:, self.removed_second] = 0.0
            b2[self.removed_second] = 0.0
            w3[self.removed_second, :] = 0.0

        first.set_weights([w1, b1])
        second.set_weights([w2, b2])
        output.set_weights([w3, b3])
        self.last_sparsity = float(sparsity)

    def on_train_batch_end(self, batch, logs=None):
        del batch, logs
        step = int(self.model.optimizer.iterations.numpy())
        should_prune, sparsity = self._scheduled_sparsity(step)
        if should_prune or step >= self.end_step:
            self._apply(min(sparsity, self.final_sparsity))

    def on_epoch_end(self, epoch, logs=None):
        del logs
        step = int(self.model.optimizer.iterations.numpy())
        _, sparsity = self._scheduled_sparsity(step)
        if step >= self.end_step:
            sparsity = self.final_sparsity
        self._apply(min(sparsity, self.final_sparsity))
        print(
            "Structured pruning epoch "
            f"{epoch + 1}: target={self.last_sparsity:.6f}, "
            f"removed=[{len(self.removed_first)}, "
            f"{len(self.removed_second)}]"
        )

    def force_final_mask(self):
        self._apply(self.final_sparsity)


def retained_indices(unit_count, removed):
    removed_set = set(int(value) for value in removed)
    return np.asarray(
        [index for index in range(unit_count) if index not in removed_set],
        dtype=np.int64,
    )


def build_compact_model(masked_model, removed_first, removed_second):
    first, second, output = get_dense_layers(masked_model)
    w1, b1 = first.get_weights()
    w2, b2 = second.get_weights()
    w3, b3 = output.get_weights()

    keep_first = retained_indices(first.units, removed_first)
    keep_second = retained_indices(second.units, removed_second)

    if len(keep_first) >= first.units or len(keep_second) >= second.units:
        raise RuntimeError("No complete hidden units were removed")

    inputs = tf_keras.Input(shape=(6,), name="features")
    x = tf_keras.layers.Dense(
        len(keep_first), activation="relu", name="structured_dense_1"
    )(inputs)
    x = tf_keras.layers.Dense(
        len(keep_second), activation="relu", name="structured_dense_2"
    )(x)
    outputs = tf_keras.layers.Dense(
        3, activation="softmax", name="class_probabilities"
    )(x)
    compact = tf_keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="logibridge_m3_structured_compact",
    )

    compact.layers[1].set_weights([
        w1[:, keep_first],
        b1[keep_first],
    ])
    compact.layers[2].set_weights([
        w2[np.ix_(keep_first, keep_second)],
        b2[keep_second],
    ])
    compact.layers[3].set_weights([
        w3[keep_second, :],
        b3,
    ])

    structure = {
        "original_hidden_units": [int(first.units), int(second.units)],
        "removed_hidden_units": [
            int(len(removed_first)), int(len(removed_second))
        ],
        "retained_hidden_units": [
            int(len(keep_first)), int(len(keep_second))
        ],
        "removed_indices": {
            "hidden_1": np.asarray(removed_first, dtype=int).tolist(),
            "hidden_2": np.asarray(removed_second, dtype=int).tolist(),
        },
        "layer_1_structured_sparsity": float(
            len(removed_first) / first.units
        ),
        "layer_2_structured_sparsity": float(
            len(removed_second) / second.units
        ),
        "overall_hidden_unit_reduction": float(
            (len(removed_first) + len(removed_second))
            / (first.units + second.units)
        ),
    }
    return compact, structure


def main():
    configure_reproducibility()
    dataset = load_dataset(DATASET_PATH)
    baseline = tf_keras.models.load_model(BASELINE_MODEL_PATH)
    get_dense_layers(baseline)

    # Clone so the saved baseline is never modified.
    pruning_model = tf_keras.models.clone_model(baseline)
    pruning_model.set_weights(baseline.get_weights())

    steps_per_epoch = int(math.ceil(
        len(dataset["train_features"]) / BATCH_SIZE
    ))
    end_step = steps_per_epoch * PRUNING_EPOCHS

    schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=TARGET_STRUCTURED_SPARSITY,
        begin_step=0,
        end_step=end_step,
        power=3,
        frequency=1,
    )
    structured_callback = StructuredUnitPruning(
        schedule=schedule,
        final_sparsity=TARGET_STRUCTURED_SPARSITY,
        end_step=end_step,
    )

    pruning_model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = pruning_model.fit(
        dataset["train_features"],
        dataset["train_labels"],
        validation_data=(
            dataset["validation_features"],
            dataset["validation_labels"],
        ),
        epochs=PRUNING_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[structured_callback],
        shuffle=True,
        verbose=2,
    )

    epochs_completed = len(history.history["loss"])
    if epochs_completed != PRUNING_EPOCHS:
        raise RuntimeError(
            f"PolynomialDecay did not complete: {epochs_completed}/"
            f"{PRUNING_EPOCHS} epochs"
        )

    structured_callback.force_final_mask()
    compact, structure = build_compact_model(
        pruning_model,
        structured_callback.removed_first,
        structured_callback.removed_second,
    )

    overall = structure["overall_hidden_unit_reduction"]
    if not 0.34 <= overall <= 0.37:
        raise RuntimeError(
            "Structured hidden-unit reduction is outside the discrete range "
            f"around 35%: {overall:.6f}"
        )

    compact.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=0.0002),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    finetune_history = compact.fit(
        dataset["train_features"],
        dataset["train_labels"],
        validation_data=(
            dataset["validation_features"],
            dataset["validation_labels"],
        ),
        epochs=COMPACT_FINETUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=2,
    )

    STRUCTURED_KERAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    compact.save(STRUCTURED_KERAS_PATH)

    calibration = select_calibration_samples(
        dataset["train_features"],
        sample_count=CALIBRATION_SAMPLE_COUNT,
        seed=43,
    )
    if len(calibration) != 200:
        raise RuntimeError("Expected exactly 200 calibration samples")

    OUTPUT_PATH.write_bytes(convert_full_int8(compact, calibration))
    evaluation = evaluate_tflite(
        OUTPUT_PATH,
        dataset["validation_features"],
        dataset["validation_labels"],
    )

    if evaluation["input_dtype"] != "<class 'numpy.int8'>":
        raise RuntimeError("M3 input is not INT8")
    if evaluation["output_dtype"] != "<class 'numpy.int8'>":
        raise RuntimeError("M3 output is not INT8")
    if evaluation["per_class_recall"]["Critical"] <= 0.95:
        raise RuntimeError("M3 Critical recall does not exceed 95%")

    baseline_parameters = parameter_count(baseline)
    compact_parameters = parameter_count(compact)

    metrics = {
        "variant": "M3",
        "name": "35 Percent Structured Dense-Unit Pruning plus Full INT8 PTQ",
        "assignment_mapping": (
            "The baseline is a Dense MLP with no convolutional filters. "
            "Complete hidden units are pruned as the Dense equivalent of "
            "structured filters."
        ),
        "pruning_method": (
            "TFMOT PolynomialDecay schedule with complete Dense-unit masks, "
            "followed by physical architecture compaction"
        ),
        "pruning_schedule": {
            "type": "PolynomialDecay",
            "initial_sparsity": 0.0,
            "final_sparsity": TARGET_STRUCTURED_SPARSITY,
            "begin_step": 0,
            "end_step": end_step,
            "power": 3,
            "frequency": 1,
        },
        "pruning_epochs_completed": epochs_completed,
        "compact_finetune_epochs_completed": int(
            len(finetune_history.history["loss"])
        ),
        "structured_pruning": structure,
        "baseline_parameter_count": baseline_parameters,
        "compact_parameter_count": compact_parameters,
        "parameter_reduction_fraction": float(
            1.0 - compact_parameters / baseline_parameters
        ),
        "calibration_sample_count": 200,
        "quantization": "Full INT8 PTQ",
        "model_path": str(OUTPUT_PATH),
        "structured_keras_path": str(STRUCTURED_KERAS_PATH),
        "file_size_bytes": int(OUTPUT_PATH.stat().st_size),
        "sha256": sha256(OUTPUT_PATH),
        "validation_accuracy": evaluation["accuracy"],
        "confusion_matrix": evaluation["confusion_matrix"].tolist(),
        "per_class_recall": evaluation["per_class_recall"],
        "input_dtype": evaluation["input_dtype"],
        "output_dtype": evaluation["output_dtype"],
        "input_quantization": evaluation["input_quantization"],
        "output_quantization": evaluation["output_quantization"],
    }
    write_json(METRICS_PATH, metrics)

    print()
    print("M3 Structured Dense-Unit Pruning + Full INT8 PTQ")
    print("=" * 62)
    print("Schedule: PolynomialDecay")
    print("Target structured sparsity:", TARGET_STRUCTURED_SPARSITY)
    print("Original hidden units:", structure["original_hidden_units"])
    print("Removed hidden units:", structure["removed_hidden_units"])
    print("Retained hidden units:", structure["retained_hidden_units"])
    print("Overall hidden-unit reduction:", overall)
    print("Baseline parameters:", baseline_parameters)
    print("Compact parameters:", compact_parameters)
    print("Parameter reduction:", metrics["parameter_reduction_fraction"])
    print("Pruning epochs completed:", epochs_completed)
    print("Calibration samples:", len(calibration))
    print("Input dtype:", metrics["input_dtype"])
    print("Output dtype:", metrics["output_dtype"])
    print("Validation accuracy:", metrics["validation_accuracy"])
    print("Critical recall:", metrics["per_class_recall"]["Critical"])
    print("Model size bytes:", metrics["file_size_bytes"])
    print()
    print("[PASS] M3 used TFMOT PolynomialDecay")
    print("[PASS] M3 pruned complete Dense hidden units")
    print("[PASS] M3 physically compacted the pruned architecture")
    print("[PASS] M3 used exactly 200 calibration samples")
    print("[PASS] M3 input and output are INT8")
    print("[PASS] M3 Critical recall exceeds 95%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
