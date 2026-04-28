"""Phase 1 regression kriging with local space-time residual interpolation."""

from __future__ import annotations

from typing import Any

import numpy as np

from kriging import (
    DEFAULT_I24_SPACE_BIN_CSV_DIR,
    exponential_semivariogram,
    fit_exponential_semivariogram,
    resolve_space_coords,
)
from methods import masked_mae, masked_mape, masked_rmse


def _as_float_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2D with shape (space, time); got {arr.shape}.")
    return arr


def _normalize_axis(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    arr = np.asarray(values, dtype=float)
    center = float(np.mean(arr))
    scale = float(np.max(np.abs(arr - center)))
    if scale <= 0.0:
        scale = 1.0
    normalized = (arr - center) / scale
    return normalized, {"center": center, "scale": scale}


def build_trend_design_matrix(
    space_coords: np.ndarray,
    num_time_steps: int,
    include_space_quadratic: bool = True,
    include_time_quadratic: bool = True,
    include_interaction: bool = True,
) -> dict[str, Any]:
    """
    Build a polynomial feature design for a `(space, time)` grid.

    Features are built on normalized coordinates to keep the least-squares
    problem numerically stable.
    """
    coords = np.asarray(space_coords, dtype=float)
    if coords.ndim != 1:
        raise ValueError("space_coords must be 1D.")
    if num_time_steps <= 0:
        raise ValueError("num_time_steps must be positive.")

    s_norm, s_stats = _normalize_axis(coords)
    t_norm, t_stats = _normalize_axis(np.arange(num_time_steps, dtype=float))

    s_grid = np.repeat(s_norm[:, None], num_time_steps, axis=1)
    t_grid = np.repeat(t_norm[None, :], coords.shape[0], axis=0)

    feature_names = ["intercept", "space_linear", "time_linear"]
    feature_columns = [
        np.ones(s_grid.size, dtype=float),
        s_grid.ravel(),
        t_grid.ravel(),
    ]

    if include_space_quadratic:
        feature_names.append("space_quadratic")
        feature_columns.append((s_grid**2).ravel())
    if include_time_quadratic:
        feature_names.append("time_quadratic")
        feature_columns.append((t_grid**2).ravel())
    if include_interaction:
        feature_names.append("space_time_interaction")
        feature_columns.append((s_grid * t_grid).ravel())

    design_matrix = np.column_stack(feature_columns)
    return {
        "design_matrix": design_matrix,
        "feature_names": feature_names,
        "normalized_space_coords": s_norm.astype(float, copy=False),
        "normalized_time_coords": t_norm.astype(float, copy=False),
        "space_stats": s_stats,
        "time_stats": t_stats,
    }


def fit_trend_surface(
    matrix: np.ndarray,
    space_coords: np.ndarray,
    include_space_quadratic: bool = True,
    include_time_quadratic: bool = True,
    include_interaction: bool = True,
    ridge: float = 1e-8,
) -> dict[str, Any]:
    """
    Fit a smooth polynomial trend surface to all observed cells in the matrix.

    The fitted mean model is:

        m(s, t) = beta_0 + beta_1 s + beta_2 t + beta_3 s^2 + beta_4 t^2 + beta_5 s t

    with optional omission of the quadratic and interaction terms.
    """
    arr = _as_float_matrix(matrix)
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative.")

    design = build_trend_design_matrix(
        space_coords=space_coords,
        num_time_steps=arr.shape[1],
        include_space_quadratic=include_space_quadratic,
        include_time_quadratic=include_time_quadratic,
        include_interaction=include_interaction,
    )

    y = arr.ravel()
    observed_mask = np.isfinite(y)
    if not np.any(observed_mask):
        raise ValueError("Need at least one observed cell to fit a trend surface.")

    X_obs = design["design_matrix"][observed_mask]
    y_obs = y[observed_mask]

    if ridge > 0.0:
        lhs = X_obs.T @ X_obs + ridge * np.eye(X_obs.shape[1], dtype=float)
        rhs = X_obs.T @ y_obs
        coefficients = np.linalg.solve(lhs, rhs)
    else:
        coefficients, *_ = np.linalg.lstsq(X_obs, y_obs, rcond=None)

    trend_surface = (design["design_matrix"] @ coefficients).reshape(arr.shape)
    residual_matrix = arr - trend_surface
    residual_matrix[~np.isfinite(arr)] = np.nan

    return {
        "trend_surface": trend_surface.astype(float, copy=False),
        "residual_matrix": residual_matrix.astype(float, copy=False),
        "coefficients": coefficients.astype(float, copy=False),
        "feature_names": design["feature_names"],
        "normalized_space_coords": design["normalized_space_coords"],
        "normalized_time_coords": design["normalized_time_coords"],
        "space_stats": design["space_stats"],
        "time_stats": design["time_stats"],
        "ridge": float(ridge),
    }


def build_empirical_spatiotemporal_semivariogram(
    residual_matrix: np.ndarray,
    space_coords: np.ndarray,
    space_scale: float = 1.0,
    time_scale: float = 3.0,
    max_pairs: int = 20000,
    num_lag_bins: int = 20,
    rng: np.random.Generator | int | None = None,
) -> dict[str, np.ndarray]:
    """
    Estimate an isotropic space-time empirical semivariogram from sampled pairs.

    Distances use the normalized metric:

        d = sqrt((delta_space / space_scale)^2 + (delta_time / time_scale)^2)
    """
    residuals = _as_float_matrix(residual_matrix)
    coords = np.asarray(space_coords, dtype=float)
    if coords.ndim != 1 or coords.shape[0] != residuals.shape[0]:
        raise ValueError("space_coords must align with the matrix rows.")
    if space_scale <= 0.0:
        raise ValueError("space_scale must be positive.")
    if time_scale <= 0.0:
        raise ValueError("time_scale must be positive.")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive.")
    if num_lag_bins <= 0:
        raise ValueError("num_lag_bins must be positive.")

    obs_rows, obs_cols = np.nonzero(~np.isnan(residuals))
    obs_values = residuals[obs_rows, obs_cols]
    num_obs = obs_values.size
    if num_obs < 2:
        raise ValueError("Need at least two observed residual cells to fit a semivariogram.")

    max_possible_pairs = num_obs * (num_obs - 1) // 2
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    if max_possible_pairs <= max_pairs:
        left_indices, right_indices = np.triu_indices(num_obs, k=1)
    else:
        sample_size = int(max_pairs)
        left_indices = generator.integers(0, num_obs, size=sample_size)
        right_indices = generator.integers(0, num_obs - 1, size=sample_size)
        right_indices = right_indices + (right_indices >= left_indices)
        keep = left_indices != right_indices
        left_indices = left_indices[keep]
        right_indices = right_indices[keep]

    delta_space = coords[obs_rows[left_indices]] - coords[obs_rows[right_indices]]
    delta_time = obs_cols[left_indices].astype(float) - obs_cols[right_indices].astype(float)
    pair_distances = np.sqrt((delta_space / space_scale) ** 2 + (delta_time / time_scale) ** 2)
    pair_semivariances = 0.5 * (obs_values[left_indices] - obs_values[right_indices]) ** 2

    valid = np.isfinite(pair_distances) & np.isfinite(pair_semivariances) & (pair_distances > 0.0)
    pair_distances = pair_distances[valid]
    pair_semivariances = pair_semivariances[valid]

    if pair_distances.size == 0:
        raise ValueError("Could not compute any finite pairwise space-time semivariances.")

    bin_edges = np.linspace(
        float(np.min(pair_distances)),
        float(np.max(pair_distances)),
        int(num_lag_bins) + 1,
    )
    if np.allclose(bin_edges[0], bin_edges[-1]):
        bin_edges = np.array([bin_edges[0], bin_edges[0] + 1.0], dtype=float)

    bin_index = np.digitize(pair_distances, bin_edges, right=True) - 1
    bin_index = np.clip(bin_index, 0, bin_edges.size - 2)

    empirical_lags = []
    empirical_semivariance = []
    pair_counts = []

    for idx in range(bin_edges.size - 1):
        members = bin_index == idx
        if not np.any(members):
            continue
        empirical_lags.append(float(np.mean(pair_distances[members])))
        empirical_semivariance.append(float(np.mean(pair_semivariances[members])))
        pair_counts.append(int(np.sum(members)))

    return {
        "lags": np.asarray(empirical_lags, dtype=float),
        "semivariance": np.asarray(empirical_semivariance, dtype=float),
        "pair_counts": np.asarray(pair_counts, dtype=int),
        "pair_distances": pair_distances.astype(float, copy=False),
        "pair_semivariances": pair_semivariances.astype(float, copy=False),
    }


def fit_spatiotemporal_residual_variogram(
    residual_matrix: np.ndarray,
    space_coords: np.ndarray,
    space_scale: float = 1.0,
    time_scale: float = 3.0,
    max_pairs: int = 20000,
    num_lag_bins: int = 20,
    fit_nugget: bool = False,
    rng: np.random.Generator | int | None = None,
) -> dict[str, Any]:
    """Fit one pooled isotropic space-time variogram on trend residuals."""
    empirical = build_empirical_spatiotemporal_semivariogram(
        residual_matrix=residual_matrix,
        space_coords=space_coords,
        space_scale=space_scale,
        time_scale=time_scale,
        max_pairs=max_pairs,
        num_lag_bins=num_lag_bins,
        rng=rng,
    )
    fitted = fit_exponential_semivariogram(
        empirical_lags=empirical["lags"],
        empirical_semivariance=empirical["semivariance"],
        pair_counts=empirical["pair_counts"],
        fit_nugget=fit_nugget,
    )
    return {
        **fitted,
        "space_scale": float(space_scale),
        "time_scale": float(time_scale),
        "empirical_lags": empirical["lags"],
        "empirical_semivariance": empirical["semivariance"],
        "pair_counts": empirical["pair_counts"],
        "pair_distances": empirical["pair_distances"],
        "pair_semivariances": empirical["pair_semivariances"],
    }


def local_spatiotemporal_distance(
    target_space: float,
    target_time: int,
    candidate_spaces: np.ndarray,
    candidate_times: np.ndarray,
    space_scale: float = 1.0,
    time_scale: float = 3.0,
) -> np.ndarray:
    """Compute isotropic normalized space-time distances."""
    if space_scale <= 0.0:
        raise ValueError("space_scale must be positive.")
    if time_scale <= 0.0:
        raise ValueError("time_scale must be positive.")

    delta_space = (np.asarray(candidate_spaces, dtype=float) - float(target_space)) / float(space_scale)
    delta_time = (np.asarray(candidate_times, dtype=float) - float(target_time)) / float(time_scale)
    return np.sqrt(delta_space**2 + delta_time**2)


def select_local_residual_neighbors(
    observed_rows: np.ndarray,
    observed_cols: np.ndarray,
    observed_values: np.ndarray,
    target_row: int,
    target_col: int,
    space_coords: np.ndarray,
    max_space_distance: int | None = 3,
    max_time_distance: int | None = 6,
    min_neighbors: int = 12,
    max_neighbors: int = 40,
    space_scale: float = 1.0,
    time_scale: float = 3.0,
    allow_global_fallback: bool = True,
) -> dict[str, np.ndarray]:
    """
    Select a local observed space-time neighborhood for residual kriging.

    The local window is expressed in row/column index units, while the ranking
    distance is computed in normalized physical space-time units.
    """
    if min_neighbors <= 0:
        raise ValueError("min_neighbors must be positive.")
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive.")
    if min_neighbors > max_neighbors:
        raise ValueError("min_neighbors cannot exceed max_neighbors.")

    rows = np.asarray(observed_rows, dtype=int)
    cols = np.asarray(observed_cols, dtype=int)
    values = np.asarray(observed_values, dtype=float)
    coords = np.asarray(space_coords, dtype=float)

    if rows.shape != cols.shape or rows.shape != values.shape:
        raise ValueError("Observed rows, cols, and values must have aligned shapes.")

    local_mask = np.ones(rows.shape[0], dtype=bool)
    if max_space_distance is not None:
        local_mask &= np.abs(rows - int(target_row)) <= int(max_space_distance)
    if max_time_distance is not None:
        local_mask &= np.abs(cols - int(target_col)) <= int(max_time_distance)

    candidate_rows = rows[local_mask]
    candidate_cols = cols[local_mask]
    candidate_values = values[local_mask]

    if candidate_rows.size < min_neighbors and allow_global_fallback:
        candidate_rows = rows
        candidate_cols = cols
        candidate_values = values

    if candidate_rows.size == 0:
        return {
            "rows": np.array([], dtype=int),
            "cols": np.array([], dtype=int),
            "values": np.array([], dtype=float),
            "distances": np.array([], dtype=float),
        }

    distances = local_spatiotemporal_distance(
        target_space=float(coords[target_row]),
        target_time=int(target_col),
        candidate_spaces=coords[candidate_rows],
        candidate_times=candidate_cols,
        space_scale=space_scale,
        time_scale=time_scale,
    )
    order = np.argsort(distances, kind="stable")
    keep = order[: min(max_neighbors, order.size)]

    return {
        "rows": candidate_rows[keep].astype(int, copy=False),
        "cols": candidate_cols[keep].astype(int, copy=False),
        "values": candidate_values[keep].astype(float, copy=False),
        "distances": distances[keep].astype(float, copy=False),
    }


def local_regression_kriging_weights(
    neighbor_rows: np.ndarray,
    neighbor_cols: np.ndarray,
    target_row: int,
    target_col: int,
    space_coords: np.ndarray,
    variogram_params: dict[str, float],
    solver_jitter: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Solve the ordinary kriging system for one local space-time neighborhood."""
    rows = np.asarray(neighbor_rows, dtype=int)
    cols = np.asarray(neighbor_cols, dtype=int)
    coords = np.asarray(space_coords, dtype=float)
    if rows.size == 0:
        raise ValueError("Need at least one neighbor to solve local kriging weights.")
    if solver_jitter < 0.0:
        raise ValueError("solver_jitter must be non-negative.")

    pair_space = coords[rows][:, None] - coords[rows][None, :]
    pair_time = cols.astype(float)[:, None] - cols.astype(float)[None, :]
    pair_distances = np.sqrt(
        (pair_space / float(variogram_params["space_scale"])) ** 2
        + (pair_time / float(variogram_params["time_scale"])) ** 2
    )

    gamma_obs = exponential_semivariogram(
        pair_distances,
        partial_sill=float(variogram_params["partial_sill"]),
        range_param=float(variogram_params["range"]),
        nugget=float(variogram_params["nugget"]),
    )
    if solver_jitter > 0.0:
        gamma_obs = gamma_obs.copy()
        diag = np.diag_indices_from(gamma_obs)
        gamma_obs[diag] += solver_jitter

    target_distances = local_spatiotemporal_distance(
        target_space=float(coords[target_row]),
        target_time=int(target_col),
        candidate_spaces=coords[rows],
        candidate_times=cols,
        space_scale=float(variogram_params["space_scale"]),
        time_scale=float(variogram_params["time_scale"]),
    )
    gamma_target = exponential_semivariogram(
        target_distances,
        partial_sill=float(variogram_params["partial_sill"]),
        range_param=float(variogram_params["range"]),
        nugget=float(variogram_params["nugget"]),
    )

    system = np.zeros((rows.size + 1, rows.size + 1), dtype=float)
    rhs = np.zeros(rows.size + 1, dtype=float)
    system[:-1, :-1] = gamma_obs
    system[:-1, -1] = 1.0
    system[-1, :-1] = 1.0
    rhs[:-1] = gamma_target
    rhs[-1] = 1.0

    solution = np.linalg.solve(system, rhs)
    return solution[:-1].astype(float, copy=False), float(solution[-1])


def regression_kriging_impute(
    matrix: np.ndarray,
    space_coords: np.ndarray | None = None,
    dx_meters: int | None = None,
    space_bin_csv_dir: str = DEFAULT_I24_SPACE_BIN_CSV_DIR,
    include_space_quadratic: bool = True,
    include_time_quadratic: bool = True,
    include_interaction: bool = True,
    trend_ridge: float = 1e-8,
    max_space_distance: int | None = 3,
    max_time_distance: int | None = 6,
    min_neighbors: int = 12,
    max_neighbors: int = 40,
    space_scale: float | None = None,
    time_scale: float = 6.0,
    variogram_max_pairs: int = 20000,
    variogram_num_lag_bins: int = 20,
    fit_nugget: bool = False,
    solver_jitter: float = 1e-8,
    allow_global_fallback: bool = True,
    min_value: float | None = 0.0,
    max_value: float | None = None,
    rng: np.random.Generator | int | None = None,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """
    Impute missing cells with Phase 1 regression kriging.

    This method:
    1. fits a smooth space-time trend surface on observed cells
    2. computes residuals around that trend
    3. fits one pooled isotropic space-time variogram on the residual field
    4. kriges each missing cell from a local observed neighborhood
    """
    arr = _as_float_matrix(matrix)
    coords = resolve_space_coords(
        num_rows=arr.shape[0],
        space_coords=space_coords,
        dx_meters=dx_meters,
        space_bin_csv_dir=space_bin_csv_dir,
    )
    if arr.shape[0] >= 2:
        default_space_scale = float(np.median(np.diff(np.sort(coords))))
        if default_space_scale <= 0.0:
            default_space_scale = 1.0
    else:
        default_space_scale = 1.0
    resolved_space_scale = default_space_scale if space_scale is None else float(space_scale)
    if resolved_space_scale <= 0.0:
        raise ValueError("space_scale must be positive.")

    trend_details = fit_trend_surface(
        matrix=arr,
        space_coords=coords,
        include_space_quadratic=include_space_quadratic,
        include_time_quadratic=include_time_quadratic,
        include_interaction=include_interaction,
        ridge=trend_ridge,
    )
    trend_surface = trend_details["trend_surface"]
    residual_matrix = trend_details["residual_matrix"]

    imputed = arr.copy()
    missing_coords = np.argwhere(np.isnan(imputed))
    if missing_coords.size == 0:
        if return_details:
            return imputed, {
                **trend_details,
                "variogram": None,
                "weights_by_target": {},
            }
        return imputed

    observed_rows, observed_cols = np.nonzero(~np.isnan(residual_matrix))
    observed_residuals = residual_matrix[observed_rows, observed_cols]
    if observed_residuals.size < 2:
        imputed[np.isnan(imputed)] = trend_surface[np.isnan(imputed)]
        if min_value is not None or max_value is not None:
            imputed = np.clip(imputed, min_value, max_value)
        if return_details:
            return imputed, {
                **trend_details,
                "variogram": None,
                "weights_by_target": {},
            }
        return imputed

    variogram_details = fit_spatiotemporal_residual_variogram(
        residual_matrix=residual_matrix,
        space_coords=coords,
        space_scale=resolved_space_scale,
        time_scale=time_scale,
        max_pairs=variogram_max_pairs,
        num_lag_bins=variogram_num_lag_bins,
        fit_nugget=fit_nugget,
        rng=rng,
    )

    weights_by_target: dict[tuple[int, int], dict[str, Any]] = {}
    for row, col in missing_coords:
        neighbor_info = select_local_residual_neighbors(
            observed_rows=observed_rows,
            observed_cols=observed_cols,
            observed_values=observed_residuals,
            target_row=int(row),
            target_col=int(col),
            space_coords=coords,
            max_space_distance=max_space_distance,
            max_time_distance=max_time_distance,
            min_neighbors=min_neighbors,
            max_neighbors=max_neighbors,
            space_scale=resolved_space_scale,
            time_scale=time_scale,
            allow_global_fallback=allow_global_fallback,
        )

        if neighbor_info["values"].size == 0:
            estimate = float(trend_surface[row, col])
            mu = np.nan
            weights = np.array([], dtype=float)
        else:
            weights, mu = local_regression_kriging_weights(
                neighbor_rows=neighbor_info["rows"],
                neighbor_cols=neighbor_info["cols"],
                target_row=int(row),
                target_col=int(col),
                space_coords=coords,
                variogram_params=variogram_details,
                solver_jitter=solver_jitter,
            )
            residual_estimate = float(np.dot(weights, neighbor_info["values"]))
            estimate = float(trend_surface[row, col] + residual_estimate)

        imputed[row, col] = estimate
        if return_details:
            weights_by_target[(int(row), int(col))] = {
                "neighbor_rows": neighbor_info["rows"],
                "neighbor_cols": neighbor_info["cols"],
                "neighbor_distances": neighbor_info["distances"],
                "weights": weights,
                "mu": float(mu) if np.isfinite(mu) else np.nan,
            }

    if min_value is not None or max_value is not None:
        imputed = np.clip(imputed, min_value, max_value)

    if return_details:
        return imputed, {
            **trend_details,
            "variogram": variogram_details,
            "space_coords": coords.astype(float, copy=False),
            "space_scale": float(resolved_space_scale),
            "time_scale": float(time_scale),
            "weights_by_target": weights_by_target,
        }
    return imputed


def evaluate_regression_kriging(
    ground_truth: np.ndarray,
    masked_matrix: np.ndarray,
    space_coords: np.ndarray | None = None,
    dx_meters: int | None = None,
    space_bin_csv_dir: str = DEFAULT_I24_SPACE_BIN_CSV_DIR,
    include_space_quadratic: bool = True,
    include_time_quadratic: bool = True,
    include_interaction: bool = True,
    trend_ridge: float = 1e-8,
    max_space_distance: int | None = 3,
    max_time_distance: int | None = 6,
    min_neighbors: int = 12,
    max_neighbors: int = 40,
    space_scale: float | None = None,
    time_scale: float = 6.0,
    variogram_max_pairs: int = 20000,
    variogram_num_lag_bins: int = 20,
    fit_nugget: bool = False,
    solver_jitter: float = 1e-8,
    allow_global_fallback: bool = True,
    min_value: float | None = 0.0,
    max_value: float | None = None,
    rng: np.random.Generator | int | None = None,
    return_imputed_matrix: bool = False,
    return_details: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray] | tuple[dict[str, float], np.ndarray, dict[str, Any]]:
    """Run Phase 1 regression kriging on one masked matrix and score it."""
    result = regression_kriging_impute(
        matrix=masked_matrix,
        space_coords=space_coords,
        dx_meters=dx_meters,
        space_bin_csv_dir=space_bin_csv_dir,
        include_space_quadratic=include_space_quadratic,
        include_time_quadratic=include_time_quadratic,
        include_interaction=include_interaction,
        trend_ridge=trend_ridge,
        max_space_distance=max_space_distance,
        max_time_distance=max_time_distance,
        min_neighbors=min_neighbors,
        max_neighbors=max_neighbors,
        space_scale=space_scale,
        time_scale=time_scale,
        variogram_max_pairs=variogram_max_pairs,
        variogram_num_lag_bins=variogram_num_lag_bins,
        fit_nugget=fit_nugget,
        solver_jitter=solver_jitter,
        allow_global_fallback=allow_global_fallback,
        min_value=min_value,
        max_value=max_value,
        rng=rng,
        return_details=return_details,
    )

    if return_details:
        imputed_matrix, details = result
    else:
        imputed_matrix = result
        details = None

    metrics = {
        "mae": masked_mae(ground_truth, imputed_matrix, masked_matrix),
        "mape": masked_mape(ground_truth, imputed_matrix, masked_matrix),
        "rmse": masked_rmse(ground_truth, imputed_matrix, masked_matrix),
    }

    if return_imputed_matrix and return_details:
        return metrics, imputed_matrix, details
    if return_imputed_matrix:
        return metrics, imputed_matrix
    return metrics


def evaluate_regression_kriging_on_masks(
    ground_truth: np.ndarray,
    masked_matrices: list[np.ndarray],
    space_coords: np.ndarray | None = None,
    dx_meters: int | None = None,
    space_bin_csv_dir: str = DEFAULT_I24_SPACE_BIN_CSV_DIR,
    include_space_quadratic: bool = True,
    include_time_quadratic: bool = True,
    include_interaction: bool = True,
    trend_ridge: float = 1e-8,
    max_space_distance: int | None = 3,
    max_time_distance: int | None = 6,
    min_neighbors: int = 12,
    max_neighbors: int = 40,
    space_scale: float | None = None,
    time_scale: float = 6.0,
    variogram_max_pairs: int = 20000,
    variogram_num_lag_bins: int = 20,
    fit_nugget: bool = False,
    solver_jitter: float = 1e-8,
    allow_global_fallback: bool = True,
    min_value: float | None = 0.0,
    max_value: float | None = None,
    rng: np.random.Generator | int | None = None,
    return_imputed_matrices: bool = False,
) -> list[dict[str, float]] | tuple[list[dict[str, float]], list[np.ndarray]]:
    """Run Phase 1 regression kriging over a collection of masked matrices."""
    all_metrics = []
    all_imputed = []
    base_rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    for mask_idx, masked_matrix in enumerate(masked_matrices):
        per_mask_seed = int(base_rng.integers(0, np.iinfo(np.int32).max))
        result = evaluate_regression_kriging(
            ground_truth=ground_truth,
            masked_matrix=masked_matrix,
            space_coords=space_coords,
            dx_meters=dx_meters,
            space_bin_csv_dir=space_bin_csv_dir,
            include_space_quadratic=include_space_quadratic,
            include_time_quadratic=include_time_quadratic,
            include_interaction=include_interaction,
            trend_ridge=trend_ridge,
            max_space_distance=max_space_distance,
            max_time_distance=max_time_distance,
            min_neighbors=min_neighbors,
            max_neighbors=max_neighbors,
            space_scale=space_scale,
            time_scale=time_scale,
            variogram_max_pairs=variogram_max_pairs,
            variogram_num_lag_bins=variogram_num_lag_bins,
            fit_nugget=fit_nugget,
            solver_jitter=solver_jitter,
            allow_global_fallback=allow_global_fallback,
            min_value=min_value,
            max_value=max_value,
            rng=per_mask_seed,
            return_imputed_matrix=return_imputed_matrices,
            return_details=False,
        )

        if return_imputed_matrices:
            metrics, imputed_matrix = result
            all_imputed.append(imputed_matrix)
        else:
            metrics = result

        all_metrics.append({"mask_index": int(mask_idx), **metrics})

    if return_imputed_matrices:
        return all_metrics, all_imputed
    return all_metrics
