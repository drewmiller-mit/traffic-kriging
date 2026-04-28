"""Experiment utilities for row-masking and imputation workflows."""

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


def generate_masked_row_arrays(
    matrix: np.ndarray,
    resolution: Union[str, int, float],
    num_masks: int = 5,
    rng: Union[np.random.Generator, int, None] = None,
) -> list[np.ndarray]:
    """
    Create multiple masked copies of a `(space, time)` matrix.

    Each output masks a different random subset of full rows by replacing those
    rows with `np.nan`. The fraction of rows masked depends on the matrix's
    spatial resolution:

    - 200 m: 76%
    - 400 m: 38%
    - 600 m: 25%
    - 800 m: 19%

    Only rows without pre-existing `NaN` values are eligible for masking so the
    requested mask percentage is applied to rows with known ground truth.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2D with shape (space, time); got {arr.shape}.")
    if num_masks <= 0:
        raise ValueError("num_masks must be a positive integer.")

    dx_meters = _normalize_resolution(resolution)
    if dx_meters not in ROW_MASK_FRACTIONS_BY_RESOLUTION:
        raise ValueError(
            f"Unsupported resolution '{resolution}'. "
            f"Expected one of {sorted(ROW_MASK_FRACTIONS_BY_RESOLUTION)} meters."
        )

    eligible_rows = np.flatnonzero(~np.isnan(arr).any(axis=1))
    mask_indices = generate_row_mask_indices(
        num_rows=arr.shape[0],
        resolution=dx_meters,
        eligible_rows=eligible_rows,
        num_masks=num_masks,
        rng=rng,
    )
    return apply_row_masks(arr, mask_indices)


def generate_row_mask_indices(
    num_rows: int,
    resolution: Union[str, int, float],
    eligible_rows: np.ndarray | None = None,
    num_masks: int = 5,
    rng: Union[np.random.Generator, int, None] = None,
) -> list[np.ndarray]:
    """
    Generate reusable row-index masks for a given spatial resolution.

    Use this when the same masking combinations must be reused across multiple
    metrics or temporal resolutions with the same spatial resolution.
    """
    if num_rows <= 0:
        raise ValueError("num_rows must be positive.")
    if num_masks <= 0:
        raise ValueError("num_masks must be a positive integer.")

    dx_meters = _normalize_resolution(resolution)
    if dx_meters not in ROW_MASK_FRACTIONS_BY_RESOLUTION:
        raise ValueError(
            f"Unsupported resolution '{resolution}'. "
            f"Expected one of {sorted(ROW_MASK_FRACTIONS_BY_RESOLUTION)} meters."
        )

    if eligible_rows is None:
        arr_eligible = np.arange(num_rows, dtype=int)
    else:
        arr_eligible = np.asarray(eligible_rows, dtype=int)

    mask_fraction = ROW_MASK_FRACTIONS_BY_RESOLUTION[dx_meters]
    num_rows_to_mask = int(round(num_rows * mask_fraction))

    if num_rows_to_mask > arr_eligible.size:
        raise ValueError(
            f"Need {num_rows_to_mask} fully observed rows to mask at {dx_meters} m, "
            f"but only {arr_eligible.size} eligible rows were found."
        )

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    mask_indices = []

    for _ in range(num_masks):
        if num_rows_to_mask > 0:
            masked_rows = generator.choice(
                arr_eligible,
                size=num_rows_to_mask,
                replace=False,
            )
            mask_indices.append(np.sort(masked_rows.astype(int, copy=False)))
        else:
            mask_indices.append(np.array([], dtype=int))

    return mask_indices


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


def find_common_fully_observed_rows(matrices: list[np.ndarray]) -> np.ndarray:
    """
    Return row indices that are fully observed in every matrix in the list.

    This is the row pool you want when masks must be held constant across
    multiple metrics or temporal resolutions at the same spatial resolution.
    """
    if not matrices:
        raise ValueError("matrices must be a non-empty list.")

    common_rows = None
    expected_rows = None
    for matrix in matrices:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Each matrix must be 2D; got {arr.shape}.")
        if expected_rows is None:
            expected_rows = arr.shape[0]
        elif arr.shape[0] != expected_rows:
            raise ValueError(
                "All matrices must have the same number of rows to compare common eligible rows."
            )

        eligible_rows = np.flatnonzero(~np.isnan(arr).any(axis=1))
        if common_rows is None:
            common_rows = eligible_rows
        else:
            common_rows = np.intersect1d(common_rows, eligible_rows)

    return np.asarray(common_rows, dtype=int)


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
