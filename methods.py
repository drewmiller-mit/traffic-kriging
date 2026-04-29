"""Experiment utilities for predefined row-masking and imputation workflows."""

from __future__ import annotations

from typing import Union

import numpy as np


ROW_MASK_FRACTIONS_BY_RESOLUTION = {
    200: 0.76,
    400: 0.38,
    600: 0.25,
    800: 0.19,
}


def _normalize_resolution(resolution: Union[str, int, float]) -> int:
    """Convert inputs like `400`, `400.0`, or `'400m'` to an integer meter value."""
    if isinstance(resolution, str):
        cleaned = resolution.strip().lower().removesuffix("m")
        try:
            return int(float(cleaned))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported resolution value '{resolution}'. "
                f"Expected one of {sorted(ROW_MASK_FRACTIONS_BY_RESOLUTION)} meters."
            ) from exc

    try:
        return int(resolution)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported resolution value '{resolution}'. "
            f"Expected one of {sorted(ROW_MASK_FRACTIONS_BY_RESOLUTION)} meters."
        ) from exc


def apply_row_masks(
    matrix: np.ndarray,
    mask_indices: list[np.ndarray],
) -> list[np.ndarray]:
    """Apply precomputed row-index masks to a `(space, time)` matrix."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2D with shape (space, time); got {arr.shape}.")

    masked_arrays = []
    for rows in mask_indices:
        masked = arr.copy()
        row_idx = np.asarray(rows, dtype=int)
        if row_idx.size > 0:
            if np.any(row_idx < 0) or np.any(row_idx >= arr.shape[0]):
                raise IndexError(
                    f"Mask rows must lie in [0, {arr.shape[0] - 1}], got {row_idx}."
                )
            masked[row_idx, :] = np.nan
        masked_arrays.append(masked)

    return masked_arrays


def _get_masked_entry_values(
    ground_truth: np.ndarray,
    imputed: np.ndarray,
    masked: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return `(truth, estimate)` values only at experimentally masked entries.

    For this workflow:
    - `ground_truth` is the original full matrix.
    - `masked` is the matrix passed into the imputation method.
    - `imputed` is the reconstructed matrix to evaluate.
    """
    truth = np.asarray(ground_truth, dtype=float)
    estimate = np.asarray(imputed, dtype=float)
    masked_input = np.asarray(masked, dtype=float)

    if truth.shape != estimate.shape or truth.shape != masked_input.shape:
        raise ValueError(
            "ground_truth, imputed, and masked must all have the same shape."
        )
    if truth.ndim != 2:
        raise ValueError(
            f"Expected 2D matrices with shape (space, time); got {truth.shape}."
        )

    evaluation_mask = np.isnan(masked_input) & ~np.isnan(truth) & ~np.isnan(estimate)
    if not np.any(evaluation_mask):
        raise ValueError(
            "No valid experimentally masked entries were found to evaluate."
        )

    return truth[evaluation_mask], estimate[evaluation_mask]


def masked_mae(
    ground_truth: np.ndarray,
    imputed: np.ndarray,
    masked: np.ndarray,
) -> float:
    """Mean absolute error on the entries hidden by the masking step."""
    truth, estimate = _get_masked_entry_values(ground_truth, imputed, masked)
    return float(np.mean(np.abs(estimate - truth)))


def masked_mape(
    ground_truth: np.ndarray,
    imputed: np.ndarray,
    masked: np.ndarray,
) -> float:
    """
    Mean absolute percentage error on the entries hidden by masking.

    Entries with zero ground-truth value are excluded to avoid division by zero.
    Returned values are percentages, e.g. `12.5` means `12.5%`.
    """
    truth, estimate = _get_masked_entry_values(ground_truth, imputed, masked)
    nonzero_truth = truth != 0
    if not np.any(nonzero_truth):
        raise ValueError("MAPE is undefined because all evaluated ground-truth values are zero.")

    percentage_errors = np.abs((estimate[nonzero_truth] - truth[nonzero_truth]) / truth[nonzero_truth])
    return float(np.mean(percentage_errors) * 100.0)


def masked_rmse(
    ground_truth: np.ndarray,
    imputed: np.ndarray,
    masked: np.ndarray,
) -> float:
    """Root mean squared error on the entries hidden by the masking step."""
    truth, estimate = _get_masked_entry_values(ground_truth, imputed, masked)
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))
