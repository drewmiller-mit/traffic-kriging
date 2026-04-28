"""Spatiotemporal KNN baselines for matrix imputation experiments."""

from __future__ import annotations

import numpy as np

from methods import masked_mae, masked_mape, masked_rmse


def _as_float_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2D with shape (space, time); got {arr.shape}.")
    return arr


def spatiotemporal_knn_distance(
    target_row: int,
    target_col: int,
    candidate_rows: np.ndarray,
    candidate_cols: np.ndarray,
    space_scale: float = 1.0,
    time_scale: float = 1.0,
) -> np.ndarray:
    """
    Compute normalized Euclidean distance in space-time coordinates.

    The default distance is:

        sqrt((delta_row / space_scale)^2 + (delta_time / time_scale)^2)

    `space_scale` and `time_scale` let you control how much one row-step should
    matter relative to one time-step.
    """
    if space_scale <= 0:
        raise ValueError("space_scale must be positive.")
    if time_scale <= 0:
        raise ValueError("time_scale must be positive.")

    row_offsets = (np.asarray(candidate_rows, dtype=float) - float(target_row)) / float(space_scale)
    col_offsets = (np.asarray(candidate_cols, dtype=float) - float(target_col)) / float(time_scale)
    return np.sqrt(row_offsets**2 + col_offsets**2)


def find_nearest_observed_neighbors(
    matrix: np.ndarray,
    target_row: int,
    target_col: int,
    k: int = 8,
    max_space_distance: int | None = 3,
    max_time_distance: int | None = 6,
    space_scale: float = 1.0,
    time_scale: float = 3.0,
    allow_global_fallback: bool = True,
) -> dict[str, np.ndarray]:
    """
    Find the nearest observed entries to a target cell in space-time.

    Parameters
    ----------
    matrix
        Matrix with shape `(space, time)`.
    target_row, target_col
        Coordinates of the missing entry to impute.
    k
        Number of nearest observed neighbors to return.
    max_space_distance, max_time_distance
        Optional local search windows in row and column index units.
    space_scale, time_scale
        Distance normalization terms for the combined space-time metric.
    allow_global_fallback
        If the local window finds no candidates, retry on all observed cells.

    Returns
    -------
    dict
        Keys: `rows`, `cols`, `values`, `distances`.
    """
    arr = _as_float_matrix(matrix)
    n_rows, n_cols = arr.shape

    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not (0 <= target_row < n_rows and 0 <= target_col < n_cols):
        raise IndexError(
            f"Target coordinate ({target_row}, {target_col}) is outside matrix shape {arr.shape}."
        )

    observed_rows, observed_cols = np.nonzero(~np.isnan(arr))
    if observed_rows.size == 0:
        return {
            "rows": np.array([], dtype=int),
            "cols": np.array([], dtype=int),
            "values": np.array([], dtype=float),
            "distances": np.array([], dtype=float),
        }

    # Never use the target cell itself as its own neighbor.
    not_target = ~((observed_rows == target_row) & (observed_cols == target_col))
    observed_rows = observed_rows[not_target]
    observed_cols = observed_cols[not_target]

    if observed_rows.size == 0:
        return {
            "rows": np.array([], dtype=int),
            "cols": np.array([], dtype=int),
            "values": np.array([], dtype=float),
            "distances": np.array([], dtype=float),
        }

    def _apply_window_filter(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = np.ones(rows.shape[0], dtype=bool)
        if max_space_distance is not None:
            keep &= np.abs(rows - target_row) <= int(max_space_distance)
        if max_time_distance is not None:
            keep &= np.abs(cols - target_col) <= int(max_time_distance)
        return rows[keep], cols[keep]

    candidate_rows, candidate_cols = _apply_window_filter(observed_rows, observed_cols)
    if candidate_rows.size == 0 and allow_global_fallback:
        candidate_rows, candidate_cols = observed_rows, observed_cols

    if candidate_rows.size == 0:
        return {
            "rows": np.array([], dtype=int),
            "cols": np.array([], dtype=int),
            "values": np.array([], dtype=float),
            "distances": np.array([], dtype=float),
        }

    distances = spatiotemporal_knn_distance(
        target_row=target_row,
        target_col=target_col,
        candidate_rows=candidate_rows,
        candidate_cols=candidate_cols,
        space_scale=space_scale,
        time_scale=time_scale,
    )

    values = arr[candidate_rows, candidate_cols]
    order = np.argsort(distances, kind="stable")
    top_k = order[: min(k, order.size)]

    return {
        "rows": candidate_rows[top_k].astype(int, copy=False),
        "cols": candidate_cols[top_k].astype(int, copy=False),
        "values": values[top_k].astype(float, copy=False),
        "distances": distances[top_k].astype(float, copy=False),
    }


def impute_missing_entry_with_knn(
    matrix: np.ndarray,
    target_row: int,
    target_col: int,
    k: int = 8,
    max_space_distance: int | None = 3,
    max_time_distance: int | None = 6,
    space_scale: float = 1.0,
    time_scale: float = 3.0,
    weight_power: float = 1.0,
    eps: float = 1e-8,
    allow_global_fallback: bool = True,
) -> tuple[float, dict[str, np.ndarray]]:
    """
    Impute one missing cell from its nearest observed space-time neighbors.

    Neighbor values are aggregated with inverse-distance weights:

        weight = 1 / (distance + eps)^weight_power
    """
    if weight_power <= 0:
        raise ValueError("weight_power must be positive.")

    arr = _as_float_matrix(matrix)
    neighbors = find_nearest_observed_neighbors(
        matrix=arr,
        target_row=target_row,
        target_col=target_col,
        k=k,
        max_space_distance=max_space_distance,
        max_time_distance=max_time_distance,
        space_scale=space_scale,
        time_scale=time_scale,
        allow_global_fallback=allow_global_fallback,
    )

    if neighbors["values"].size == 0:
        return np.nan, neighbors

    weights = 1.0 / np.power(neighbors["distances"] + eps, weight_power)
    estimate = float(np.average(neighbors["values"], weights=weights))
    neighbors["weights"] = weights.astype(float, copy=False)
    return estimate, neighbors


def spatiotemporal_knn_impute(
    matrix: np.ndarray,
    k: int = 8,
    max_space_distance: int | None = 3,
    max_time_distance: int | None = 6,
    space_scale: float = 1.0,
    time_scale: float = 3.0,
    weight_power: float = 1.0,
    eps: float = 1e-8,
    allow_global_fallback: bool = True,
    update_source: bool = False,
    return_neighbor_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[tuple[int, int], dict[str, np.ndarray]]]:
    """
    Impute all `NaN` cells in a `(space, time)` matrix using spatiotemporal KNN.

    Parameters
    ----------
    update_source
        If `False`, each imputation only uses originally observed values.
        If `True`, newly imputed values can be reused for later missing cells.
    return_neighbor_details
        If `True`, also return a dictionary keyed by `(row, col)` containing the
        neighbors used for each imputed entry.
    """
    arr = _as_float_matrix(matrix)
    source = arr.copy()
    imputed = arr.copy()
    missing_coords = np.argwhere(np.isnan(imputed))
    neighbor_details = {}

    for row, col in missing_coords:
        estimate, neighbors = impute_missing_entry_with_knn(
            matrix=source,
            target_row=int(row),
            target_col=int(col),
            k=k,
            max_space_distance=max_space_distance,
            max_time_distance=max_time_distance,
            space_scale=space_scale,
            time_scale=time_scale,
            weight_power=weight_power,
            eps=eps,
            allow_global_fallback=allow_global_fallback,
        )
        imputed[row, col] = estimate
        if update_source and not np.isnan(estimate):
            source[row, col] = estimate
        if return_neighbor_details:
            neighbor_details[(int(row), int(col))] = neighbors

    if return_neighbor_details:
        return imputed, neighbor_details
    return imputed


def evaluate_spatiotemporal_knn(
    ground_truth: np.ndarray,
    masked_matrix: np.ndarray,
    k: int = 8,
    max_space_distance: int | None = 3,
    max_time_distance: int | None = 6,
    space_scale: float = 1.0,
    time_scale: float = 3.0,
    weight_power: float = 1.0,
    eps: float = 1e-8,
    allow_global_fallback: bool = True,
    update_source: bool = False,
    return_imputed_matrix: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
    """
    Run spatiotemporal KNN imputation on one masked matrix and score it.

    Parameters
    ----------
    ground_truth
        Original unmasked `(space, time)` matrix.
    masked_matrix
        Input matrix with experimentally hidden values set to `np.nan`.

    Returns
    -------
    dict or `(dict, np.ndarray)`
        Metrics dictionary with `mae`, `mape`, and `rmse`. If
        `return_imputed_matrix=True`, also returns the imputed matrix.
    """
    imputed_matrix = spatiotemporal_knn_impute(
        matrix=masked_matrix,
        k=k,
        max_space_distance=max_space_distance,
        max_time_distance=max_time_distance,
        space_scale=space_scale,
        time_scale=time_scale,
        weight_power=weight_power,
        eps=eps,
        allow_global_fallback=allow_global_fallback,
        update_source=update_source,
        return_neighbor_details=False,
    )

    metrics = {
        "mae": masked_mae(ground_truth, imputed_matrix, masked_matrix),
        "mape": masked_mape(ground_truth, imputed_matrix, masked_matrix),
        "rmse": masked_rmse(ground_truth, imputed_matrix, masked_matrix),
    }

    if return_imputed_matrix:
        return metrics, imputed_matrix
    return metrics


def evaluate_spatiotemporal_knn_on_masks(
    ground_truth: np.ndarray,
    masked_matrices: list[np.ndarray],
    k: int = 8,
    max_space_distance: int | None = 3,
    max_time_distance: int | None = 6,
    space_scale: float = 1.0,
    time_scale: float = 3.0,
    weight_power: float = 1.0,
    eps: float = 1e-8,
    allow_global_fallback: bool = True,
    update_source: bool = False,
    return_imputed_matrices: bool = False,
) -> list[dict[str, float]] | tuple[list[dict[str, float]], list[np.ndarray]]:
    """
    Run and score spatiotemporal KNN over a collection of masked matrices.

    This is the convenience wrapper for the 5 random-mask experiment setup.
    """
    all_metrics = []
    all_imputed = []

    for mask_idx, masked_matrix in enumerate(masked_matrices):
        result = evaluate_spatiotemporal_knn(
            ground_truth=ground_truth,
            masked_matrix=masked_matrix,
            k=k,
            max_space_distance=max_space_distance,
            max_time_distance=max_time_distance,
            space_scale=space_scale,
            time_scale=time_scale,
            weight_power=weight_power,
            eps=eps,
            allow_global_fallback=allow_global_fallback,
            update_source=update_source,
            return_imputed_matrix=return_imputed_matrices,
        )

        if return_imputed_matrices:
            metrics, imputed_matrix = result
            all_imputed.append(imputed_matrix)
        else:
            metrics = result

        metrics = {
            "mask_index": int(mask_idx),
            **metrics,
        }
        all_metrics.append(metrics)

    if return_imputed_matrices:
        return all_metrics, all_imputed
    return all_metrics
