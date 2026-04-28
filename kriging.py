"""Spatial ordinary kriging baselines for row-masking matrix imputation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from methods import masked_mae, masked_mape, masked_rmse


DEFAULT_I24_SPACE_BIN_CSV_DIR = (
    "data/i24/matrix_sweeps/daily_combined_repaired/space_bin_csvs"
)


def _as_float_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2D with shape (space, time); got {arr.shape}.")
    return arr


def load_i24_space_bin_centers(
    dx_meters: int,
    space_bin_csv_dir: str = DEFAULT_I24_SPACE_BIN_CSV_DIR,
) -> np.ndarray:
    """Load I-24 space-bin center coordinates in miles for a given spatial resolution."""
    csv_path = Path(space_bin_csv_dir) / f"space_bins_dx_{int(dx_meters)}m.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find space-bin CSV for {dx_meters} m at '{csv_path}'."
        )

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.size == 0 or "center_miles" not in data.dtype.names:
        raise ValueError(
            f"Expected a 'center_miles' column in '{csv_path}', but could not parse it."
        )

    return np.asarray(data["center_miles"], dtype=float)


def resolve_space_coords(
    num_rows: int,
    space_coords: np.ndarray | None = None,
    dx_meters: int | None = None,
    space_bin_csv_dir: str = DEFAULT_I24_SPACE_BIN_CSV_DIR,
) -> np.ndarray:
    """
    Resolve 1D spatial coordinates for the matrix rows.

    Priority:
    1. explicit `space_coords`
    2. I-24 space-bin centers if `dx_meters` is provided
    3. row indices `0, 1, ..., num_rows - 1`
    """
    if num_rows <= 0:
        raise ValueError("num_rows must be positive.")

    if space_coords is not None:
        coords = np.asarray(space_coords, dtype=float)
    elif dx_meters is not None:
        coords = load_i24_space_bin_centers(
            dx_meters=dx_meters,
            space_bin_csv_dir=space_bin_csv_dir,
        )
    else:
        coords = np.arange(num_rows, dtype=float)

    if coords.ndim != 1 or coords.shape[0] != num_rows:
        raise ValueError(
            f"space_coords must be 1D with length {num_rows}; got {coords.shape}."
        )

    if not np.all(np.isfinite(coords)):
        raise ValueError("space_coords must be finite.")

    return coords


def exponential_semivariogram(
    distances: np.ndarray,
    partial_sill: float,
    range_param: float,
    nugget: float = 0.0,
) -> np.ndarray:
    """
    Exponential semivariogram model.

    The diagonal is forced to zero so same-location semivariances stay exact in
    the kriging system even when a nugget is present.
    """
    distances_arr = np.asarray(distances, dtype=float)
    if partial_sill < 0:
        raise ValueError("partial_sill must be non-negative.")
    if range_param <= 0:
        raise ValueError("range_param must be positive.")
    if nugget < 0:
        raise ValueError("nugget must be non-negative.")

    gamma = nugget + partial_sill * (1.0 - np.exp(-distances_arr / range_param))
    return np.where(distances_arr == 0.0, 0.0, gamma)


def build_empirical_spatial_semivariogram(
    residual_matrix: np.ndarray,
    space_coords: np.ndarray,
    lag_round_decimals: int = 12,
) -> dict[str, np.ndarray]:
    """
    Estimate a pooled empirical spatial semivariogram from residuals.

    Parameters
    ----------
    residual_matrix
        Residual matrix with shape `(observed_space, time)`.
    space_coords
        Coordinates aligned to the residual rows.
    """
    residuals = _as_float_matrix(residual_matrix)
    coords = np.asarray(space_coords, dtype=float)
    if coords.ndim != 1 or coords.shape[0] != residuals.shape[0]:
        raise ValueError(
            "space_coords must be 1D and aligned with the residual matrix rows."
        )

    if residuals.shape[0] < 2:
        raise ValueError("Need at least two observed rows to estimate a semivariogram.")

    pair_distances = []
    pair_semivariances = []

    for i in range(residuals.shape[0] - 1):
        deltas = residuals[i + 1 :] - residuals[i]
        semivars = 0.5 * np.nanmean(deltas**2, axis=1)
        distances = np.abs(coords[i + 1 :] - coords[i])
        valid = np.isfinite(semivars) & np.isfinite(distances) & (distances > 0.0)
        if np.any(valid):
            pair_distances.append(distances[valid])
            pair_semivariances.append(semivars[valid])

    if not pair_distances:
        raise ValueError("Could not compute any finite pairwise semivariances.")

    all_distances = np.concatenate(pair_distances)
    all_semivariances = np.concatenate(pair_semivariances)
    rounded_distances = np.round(all_distances, decimals=lag_round_decimals)
    unique_lags, inverse = np.unique(rounded_distances, return_inverse=True)

    lag_semivariances = np.zeros(unique_lags.shape[0], dtype=float)
    pair_counts = np.zeros(unique_lags.shape[0], dtype=int)

    for lag_idx in range(unique_lags.shape[0]):
        member_mask = inverse == lag_idx
        lag_semivariances[lag_idx] = float(np.mean(all_semivariances[member_mask]))
        pair_counts[lag_idx] = int(np.sum(member_mask))

    return {
        "lags": unique_lags.astype(float, copy=False),
        "semivariance": lag_semivariances,
        "pair_counts": pair_counts,
        "pair_distances": all_distances.astype(float, copy=False),
        "pair_semivariances": all_semivariances.astype(float, copy=False),
    }


def fit_exponential_semivariogram(
    empirical_lags: np.ndarray,
    empirical_semivariance: np.ndarray,
    pair_counts: np.ndarray | None = None,
    fit_nugget: bool = False,
) -> dict[str, float]:
    """Fit an exponential semivariogram model to empirical lag estimates."""
    lags = np.asarray(empirical_lags, dtype=float)
    semivars = np.asarray(empirical_semivariance, dtype=float)

    if lags.ndim != 1 or semivars.ndim != 1 or lags.shape != semivars.shape:
        raise ValueError("empirical_lags and empirical_semivariance must be aligned 1D arrays.")
    if lags.size == 0:
        raise ValueError("Need at least one lag to fit a semivariogram.")
    if np.any(lags <= 0.0):
        raise ValueError("All empirical lags must be strictly positive.")

    counts = None if pair_counts is None else np.asarray(pair_counts, dtype=float)
    if counts is not None and counts.shape != lags.shape:
        raise ValueError("pair_counts must match empirical_lags if provided.")

    max_semivar = float(np.nanmax(semivars))
    default_partial_sill = max(max_semivar, 1e-6)
    default_range = max(float(np.median(lags)), 1e-6)

    # With very few lag bins, a heuristic fit is safer than overfitting.
    min_lags_needed = 3 if fit_nugget else 2
    if lags.size < min_lags_needed:
        nugget = float(max(min(float(semivars[0]) * 0.25, default_partial_sill * 0.5), 0.0))
        if not fit_nugget:
            nugget = 0.0
        partial_sill = max(default_partial_sill - nugget, 1e-6)
        return {
            "model": "exponential",
            "nugget": nugget,
            "partial_sill": float(partial_sill),
            "range": float(default_range),
            "sill": float(nugget + partial_sill),
        }

    sigma = None
    if counts is not None:
        sigma = 1.0 / np.sqrt(np.maximum(counts, 1.0))

    if fit_nugget:
        initial_params = np.array(
            [
                max(min(float(semivars.min()) * 0.25, default_partial_sill * 0.5), 0.0),
                max(default_partial_sill * 0.75, 1e-6),
                default_range,
            ],
            dtype=float,
        )
        bounds = ([0.0, 1e-9, 1e-9], [np.inf, np.inf, np.inf])

        def _fit_func(h: np.ndarray, nugget: float, partial_sill: float, range_param: float) -> np.ndarray:
            return exponential_semivariogram(
                h,
                partial_sill=partial_sill,
                range_param=range_param,
                nugget=nugget,
            )

        params, _ = curve_fit(
            _fit_func,
            lags,
            semivars,
            p0=initial_params,
            bounds=bounds,
            sigma=sigma,
            absolute_sigma=False,
            maxfev=20000,
        )
        nugget, partial_sill, range_param = params
    else:
        initial_params = np.array(
            [
                default_partial_sill,
                default_range,
            ],
            dtype=float,
        )
        bounds = ([1e-9, 1e-9], [np.inf, np.inf])

        def _fit_func(h: np.ndarray, partial_sill: float, range_param: float) -> np.ndarray:
            return exponential_semivariogram(
                h,
                partial_sill=partial_sill,
                range_param=range_param,
                nugget=0.0,
            )

        params, _ = curve_fit(
            _fit_func,
            lags,
            semivars,
            p0=initial_params,
            bounds=bounds,
            sigma=sigma,
            absolute_sigma=False,
            maxfev=20000,
        )
        partial_sill, range_param = params
        nugget = 0.0

    return {
        "model": "exponential",
        "nugget": float(nugget),
        "partial_sill": float(partial_sill),
        "range": float(range_param),
        "sill": float(nugget + partial_sill),
    }


def fit_pooled_residual_semivariogram(
    matrix: np.ndarray,
    support_rows: np.ndarray,
    space_coords: np.ndarray,
    fit_nugget: bool = False,
) -> dict[str, Any]:
    """
    Fit one pooled spatial semivariogram from column-demeaned observed rows.

    The mean is removed per time column so the variogram models spatial
    structure rather than the global traffic level.
    """
    arr = _as_float_matrix(matrix)
    support = np.asarray(support_rows, dtype=int)
    coords = np.asarray(space_coords, dtype=float)

    if support.ndim != 1 or support.size == 0:
        raise ValueError("support_rows must be a non-empty 1D array.")
    if coords.ndim != 1 or coords.shape[0] != arr.shape[0]:
        raise ValueError("space_coords must be aligned with the matrix rows.")

    support_values = arr[support]
    column_means = np.nanmean(support_values, axis=0)
    residuals = support_values - column_means[None, :]

    empirical = build_empirical_spatial_semivariogram(
        residual_matrix=residuals,
        space_coords=coords[support],
    )
    fitted = fit_exponential_semivariogram(
        empirical_lags=empirical["lags"],
        empirical_semivariance=empirical["semivariance"],
        pair_counts=empirical["pair_counts"],
        fit_nugget=fit_nugget,
    )

    return {
        **fitted,
        "column_means": column_means.astype(float, copy=False),
        "support_rows": support.astype(int, copy=False),
        "support_coords": coords[support].astype(float, copy=False),
        "empirical_lags": empirical["lags"],
        "empirical_semivariance": empirical["semivariance"],
        "pair_counts": empirical["pair_counts"],
        "pair_distances": empirical["pair_distances"],
        "pair_semivariances": empirical["pair_semivariances"],
        "residual_matrix": residuals.astype(float, copy=False),
    }


def get_semivariogram_diagnostics(
    matrix: np.ndarray,
    space_coords: np.ndarray | None = None,
    dx_meters: int | None = None,
    space_bin_csv_dir: str = DEFAULT_I24_SPACE_BIN_CSV_DIR,
    fit_nugget: bool = False,
    num_curve_points: int = 200,
) -> dict[str, Any]:
    """
    Return the empirical semivariogram points and fitted curve for one matrix.

    This is the inspection helper to use when you want to visualize what data
    the variogram fit is using for a specific masked or partially observed
    matrix.
    """
    arr = _as_float_matrix(matrix)
    coords = resolve_space_coords(
        num_rows=arr.shape[0],
        space_coords=space_coords,
        dx_meters=dx_meters,
        space_bin_csv_dir=space_bin_csv_dir,
    )

    support_rows = np.flatnonzero(~np.isnan(arr).any(axis=1))
    if support_rows.size < 2:
        raise ValueError(
            "Need at least two fully observed rows in the input matrix to inspect a semivariogram."
        )

    details = fit_pooled_residual_semivariogram(
        matrix=arr,
        support_rows=support_rows,
        space_coords=coords,
        fit_nugget=fit_nugget,
    )

    max_distance = float(np.nanmax(details["pair_distances"]))
    curve_distances = np.linspace(0.0, max(max_distance, 1e-9), int(max(num_curve_points, 2)))
    fitted_curve = exponential_semivariogram(
        curve_distances,
        partial_sill=float(details["partial_sill"]),
        range_param=float(details["range"]),
        nugget=float(details["nugget"]),
    )

    return {
        **details,
        "space_coords": coords.astype(float, copy=False),
        "curve_distances": curve_distances.astype(float, copy=False),
        "fitted_curve": fitted_curve.astype(float, copy=False),
    }


def plot_fitted_semivariogram(
    semivariogram_details: dict[str, Any],
    ax: Any = None,
    show_pair_cloud: bool = True,
    show_lag_points: bool = True,
    pair_alpha: float = 0.2,
    pair_size: float = 16.0,
    lag_marker_size: float = 48.0,
):
    """
    Plot the raw pair cloud, lag-averaged empirical points, and fitted curve.

    Pass the output of `get_semivariogram_diagnostics(...)`.
    """
    if pair_alpha < 0.0 or pair_alpha > 1.0:
        raise ValueError("pair_alpha must lie in [0, 1].")
    if pair_size <= 0.0:
        raise ValueError("pair_size must be positive.")
    if lag_marker_size <= 0.0:
        raise ValueError("lag_marker_size must be positive.")

    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    if show_pair_cloud:
        ax.scatter(
            semivariogram_details["pair_distances"],
            semivariogram_details["pair_semivariances"],
            s=pair_size,
            alpha=pair_alpha,
            color="0.65",
            label="Pair cloud",
        )

    if show_lag_points:
        ax.scatter(
            semivariogram_details["empirical_lags"],
            semivariogram_details["empirical_semivariance"],
            s=lag_marker_size,
            color="tab:blue",
            label="Lag averages",
            zorder=3,
        )

    ax.plot(
        semivariogram_details["curve_distances"],
        semivariogram_details["fitted_curve"],
        color="tab:red",
        linewidth=2.0,
        label="Fitted exponential",
    )
    ax.set_xlabel("Spatial Lag Distance")
    ax.set_ylabel("Semivariance")
    ax.set_title("Fitted Spatial Semivariogram")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig, ax


def ordinary_kriging_weights(
    observed_coords: np.ndarray,
    target_coord: float,
    variogram_params: dict[str, float],
    solver_jitter: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Solve the ordinary kriging system for one target coordinate."""
    coords = np.asarray(observed_coords, dtype=float)
    if coords.ndim != 1 or coords.size == 0:
        raise ValueError("observed_coords must be a non-empty 1D array.")
    if solver_jitter < 0:
        raise ValueError("solver_jitter must be non-negative.")

    pairwise_distances = np.abs(coords[:, None] - coords[None, :])
    gamma_obs = exponential_semivariogram(
        pairwise_distances,
        partial_sill=float(variogram_params["partial_sill"]),
        range_param=float(variogram_params["range"]),
        nugget=float(variogram_params["nugget"]),
    )
    if solver_jitter > 0.0:
        gamma_obs = gamma_obs.copy()
        diag_idx = np.diag_indices_from(gamma_obs)
        gamma_obs[diag_idx] += solver_jitter

    gamma_target = exponential_semivariogram(
        np.abs(coords - float(target_coord)),
        partial_sill=float(variogram_params["partial_sill"]),
        range_param=float(variogram_params["range"]),
        nugget=float(variogram_params["nugget"]),
    )

    system = np.zeros((coords.size + 1, coords.size + 1), dtype=float)
    rhs = np.zeros(coords.size + 1, dtype=float)

    system[:-1, :-1] = gamma_obs
    system[:-1, -1] = 1.0
    system[-1, :-1] = 1.0
    rhs[:-1] = gamma_target
    rhs[-1] = 1.0

    solution = np.linalg.solve(system, rhs)
    weights = solution[:-1]
    mu = float(solution[-1])
    return weights.astype(float, copy=False), mu


def spatial_ordinary_kriging_impute(
    matrix: np.ndarray,
    space_coords: np.ndarray | None = None,
    dx_meters: int | None = None,
    space_bin_csv_dir: str = DEFAULT_I24_SPACE_BIN_CSV_DIR,
    fit_nugget: bool = False,
    min_value: float | None = 0.0,
    max_value: float | None = None,
    solver_jitter: float = 1e-8,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """
    Impute missing matrix entries using spatial ordinary kriging on residuals.

    The variogram is fitted once from the fully observed support rows, and the
    resulting kriging weights are reused across all time columns.
    """
    arr = _as_float_matrix(matrix)
    coords = resolve_space_coords(
        num_rows=arr.shape[0],
        space_coords=space_coords,
        dx_meters=dx_meters,
        space_bin_csv_dir=space_bin_csv_dir,
    )

    imputed = arr.copy()
    missing_mask = np.isnan(imputed)
    if not np.any(missing_mask):
        if return_details:
            return imputed, {"support_rows": np.arange(arr.shape[0], dtype=int)}
        return imputed

    support_rows = np.flatnonzero(~np.isnan(arr).any(axis=1))
    if support_rows.size == 0:
        if return_details:
            return imputed, {"support_rows": np.array([], dtype=int)}
        return imputed
    if support_rows.size == 1:
        fill_values = np.repeat(arr[support_rows[0]][None, :], arr.shape[0], axis=0)
        imputed[missing_mask] = fill_values[missing_mask]
        if min_value is not None or max_value is not None:
            imputed = np.clip(imputed, min_value, max_value)
        details = {
            "support_rows": support_rows.astype(int, copy=False),
            "column_means": arr[support_rows[0]].astype(float, copy=False),
            "weights_by_row": {},
        }
        if return_details:
            return imputed, details
        return imputed

    variogram_details = fit_pooled_residual_semivariogram(
        matrix=arr,
        support_rows=support_rows,
        space_coords=coords,
        fit_nugget=fit_nugget,
    )

    support_values = arr[support_rows]
    column_means = variogram_details["column_means"]
    support_residuals = support_values - column_means[None, :]

    weights_by_row: dict[int, dict[str, np.ndarray | float]] = {}
    target_rows = np.flatnonzero(np.isnan(imputed).any(axis=1))
    for target_row in target_rows:
        weights, mu = ordinary_kriging_weights(
            observed_coords=coords[support_rows],
            target_coord=float(coords[target_row]),
            variogram_params=variogram_details,
            solver_jitter=solver_jitter,
        )
        predicted_residuals = weights @ support_residuals
        row_missing_mask = np.isnan(imputed[target_row])
        imputed[target_row, row_missing_mask] = (
            column_means[row_missing_mask] + predicted_residuals[row_missing_mask]
        )
        weights_by_row[int(target_row)] = {
            "weights": weights.astype(float, copy=False),
            "mu": float(mu),
        }

    if min_value is not None or max_value is not None:
        imputed = np.clip(imputed, min_value, max_value)

    if return_details:
        return imputed, {
            **variogram_details,
            "weights_by_row": weights_by_row,
            "space_coords": coords.astype(float, copy=False),
        }
    return imputed


def evaluate_spatial_ordinary_kriging(
    ground_truth: np.ndarray,
    masked_matrix: np.ndarray,
    space_coords: np.ndarray | None = None,
    dx_meters: int | None = None,
    space_bin_csv_dir: str = DEFAULT_I24_SPACE_BIN_CSV_DIR,
    fit_nugget: bool = False,
    min_value: float | None = 0.0,
    max_value: float | None = None,
    solver_jitter: float = 1e-8,
    return_imputed_matrix: bool = False,
    return_details: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray] | tuple[dict[str, float], np.ndarray, dict[str, Any]]:
    """Run spatial ordinary kriging on one masked matrix and score it."""
    result = spatial_ordinary_kriging_impute(
        matrix=masked_matrix,
        space_coords=space_coords,
        dx_meters=dx_meters,
        space_bin_csv_dir=space_bin_csv_dir,
        fit_nugget=fit_nugget,
        min_value=min_value,
        max_value=max_value,
        solver_jitter=solver_jitter,
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


def evaluate_spatial_ordinary_kriging_on_masks(
    ground_truth: np.ndarray,
    masked_matrices: list[np.ndarray],
    space_coords: np.ndarray | None = None,
    dx_meters: int | None = None,
    space_bin_csv_dir: str = DEFAULT_I24_SPACE_BIN_CSV_DIR,
    fit_nugget: bool = False,
    min_value: float | None = 0.0,
    max_value: float | None = None,
    solver_jitter: float = 1e-8,
    return_imputed_matrices: bool = False,
) -> list[dict[str, float]] | tuple[list[dict[str, float]], list[np.ndarray]]:
    """Run and score spatial ordinary kriging over a collection of masked matrices."""
    all_metrics = []
    all_imputed = []

    for mask_idx, masked_matrix in enumerate(masked_matrices):
        result = evaluate_spatial_ordinary_kriging(
            ground_truth=ground_truth,
            masked_matrix=masked_matrix,
            space_coords=space_coords,
            dx_meters=dx_meters,
            space_bin_csv_dir=space_bin_csv_dir,
            fit_nugget=fit_nugget,
            min_value=min_value,
            max_value=max_value,
            solver_jitter=solver_jitter,
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
