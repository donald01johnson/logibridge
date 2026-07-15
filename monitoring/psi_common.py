#!/usr/bin/env python3
"""Shared Population Stability Index utilities for LogiEdge."""

import numpy as np


PSI_BINS = np.asarray(
    [
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
    ],
    dtype=np.float64,
)

PSI_BIN_LABELS = [
    "[0.00, 0.25)",
    "[0.25, 0.50)",
    "[0.50, 0.75)",
    "[0.75, 1.00]",
]

PSI_EPSILON = 1e-6


def confidence_distribution(scores):
    """Return counts and proportions across the four required bins."""

    values = np.asarray(
        scores,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "Confidence scores must be one-dimensional"
        )

    if len(values) == 0:
        raise ValueError(
            "At least one confidence score is required"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Confidence scores contain non-finite values"
        )

    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(
            "Confidence scores must be between zero and one"
        )

    counts, _ = np.histogram(
        values,
        bins=PSI_BINS,
    )

    proportions = (
        counts.astype(np.float64)
        / float(len(values))
    )

    return (
        counts.astype(np.int64),
        proportions,
    )


def smooth_distribution(proportions):
    """Apply epsilon smoothing and renormalize one distribution."""

    values = np.asarray(
        proportions,
        dtype=np.float64,
    )

    if values.shape != (4,):
        raise ValueError(
            "PSI distribution must contain four values"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "PSI distribution contains invalid values"
        )

    if np.any(values < 0.0):
        raise ValueError(
            "PSI proportions cannot be negative"
        )

    smoothed = np.maximum(
        values,
        PSI_EPSILON,
    )

    smoothed = (
        smoothed
        / np.sum(smoothed)
    )

    return smoothed


def calculate_psi(
    reference_proportions,
    current_proportions,
):
    """Calculate PSI between reference and current distributions."""

    reference = smooth_distribution(
        reference_proportions
    )

    current = smooth_distribution(
        current_proportions
    )

    contributions = (
        current - reference
    ) * np.log(
        current / reference
    )

    psi_value = float(
        np.sum(contributions)
    )

    return (
        psi_value,
        contributions,
    )
