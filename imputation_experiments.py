"""Reusable experiment runners and artifact writers for matrix imputation methods."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from advanced_kriging import evaluate_regression_kriging_on_masks
from asmx import DEFAULT_ASM_PARAMS, METERS_PER_MILE, evaluate_asm_on_masks
from kriging import evaluate_spatial_ordinary_kriging_on_masks
from knn import evaluate_spatiotemporal_knn_on_masks
from methods import (
    apply_row_masks,
    find_common_fully_observed_rows,
    generate_masked_row_arrays,
    generate_row_mask_indices,
)


I24_REPAIRED_MATRIX_DIR = "data/i24/matrix_sweeps/daily_combined_repaired"
I24_MATRIX_STEM_PATTERN = re.compile(
    r"^(?P<day>[a-z0-9]+)_(?P<direction>[a-z]+)_(?P<start>\d+)_(?P<end>\d+)"
    r"_dt_(?P<dt>[^_]+)_dx_(?P<dx_meters>\d+)m_(?P<metric>[a-z_]+)$"
)


ScalarMap = dict[str, Any]
EvaluateOnMasksFn = Callable[..., tuple[list[dict[str, float]], list[np.ndarray]]]
PerMatrixKwargsFn = Callable[[pd.Series], dict[str, Any]]


def _as_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _extract_scalar_fields(mapping: dict[str, Any] | None) -> ScalarMap:
    if mapping is None:
        return {}

    scalar_types = (str, int, float, bool, np.integer, np.floating, np.bool_)
    return {
        key: _as_python_scalar(value)
        for key, value in mapping.items()
        if isinstance(value, scalar_types)
    }


def _parse_dt_label_to_seconds(dt_label: str) -> float:
    """Convert labels like `10s`, `30s`, `1min`, `5min` to seconds."""
    label = str(dt_label).strip().lower()
    if label.endswith("min"):
        return float(label[:-3]) * 60.0
    if label.endswith("s"):
        return float(label[:-1])
    raise ValueError(f"Unsupported dt label '{dt_label}'.")


def _scale_positive_count(base_value: int | None, factor: float, minimum: int = 1) -> int | None:
    if base_value is None:
        return None
    if base_value <= 0:
        raise ValueError(f"base_value must be positive when provided; got {base_value}.")
    if factor <= 0.0:
        raise ValueError(f"factor must be positive; got {factor}.")
    return max(int(minimum), int(round(float(base_value) * float(factor))))


def build_resolution_scaled_window_kwargs(
    manifest_row: pd.Series,
    base_dx_meters: int = 400,
    base_dt_label: str = "30s",
    base_max_space_distance: int | None = 3,
    base_max_time_distance: int | None = 6,
) -> dict[str, int | None]:
    """
    Scale row/column neighborhood counts by matrix resolution.

    The intent is to preserve an approximately fixed physical neighborhood size:

    - finer resolution -> more row/column neighbors
    - coarser resolution -> fewer row/column neighbors
    """
    dx_meters = int(manifest_row["dx_meters"])
    dt_seconds = _parse_dt_label_to_seconds(str(manifest_row["dt"]))
    base_dt_seconds = _parse_dt_label_to_seconds(base_dt_label)

    space_factor = float(base_dx_meters) / float(dx_meters)
    time_factor = float(base_dt_seconds) / float(dt_seconds)

    return {
        "max_space_distance": _scale_positive_count(
            base_value=base_max_space_distance,
            factor=space_factor,
            minimum=1,
        ),
        "max_time_distance": _scale_positive_count(
            base_value=base_max_time_distance,
            factor=time_factor,
            minimum=1,
        ),
    }


def _compose_per_matrix_kwargs(
    *funcs: PerMatrixKwargsFn | None,
) -> PerMatrixKwargsFn | None:
    active_funcs = [func for func in funcs if func is not None]
    if not active_funcs:
        return None

    def _composed(manifest_row: pd.Series) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for func in active_funcs:
            merged.update(func(manifest_row))
        return merged

    return _composed


def parse_i24_matrix_metadata(file_path: str | Path) -> dict[str, Any]:
    """Parse day, metric, dt, and dx metadata from an I-24 matrix filename."""
    path = Path(file_path)
    match = I24_MATRIX_STEM_PATTERN.match(path.stem)
    if match is None:
        raise ValueError(
            f"Could not parse I-24 matrix metadata from '{path.name}'."
        )

    parsed = match.groupdict()
    return {
        "file": path.name,
        "matrix_path": str(path),
        "day": parsed["day"],
        "direction": parsed["direction"],
        "start_label": parsed["start"],
        "end_label": parsed["end"],
        "dt": parsed["dt"],
        "dx_meters": int(parsed["dx_meters"]),
        "metric": parsed["metric"],
    }


def load_i24_experiment_manifest(
    metrics: tuple[str, ...] = ("velocity", "flow_per_lane"),
    input_dir: str = I24_REPAIRED_MATRIX_DIR,
) -> pd.DataFrame:
    """Load experiment rows for the requested I-24 repaired metrics."""
    input_path = Path(input_dir)
    manifest_specs = {
        "velocity": ("velocity_manifest.csv", "velocity_path"),
        "flow_per_lane": ("flow_per_lane_manifest.csv", "flow_per_lane_path"),
    }

    rows = []
    repaired_manifest_path = input_path / "repaired_manifest.csv"
    repaired_manifest = None
    if any(metric in {"flow", "density"} for metric in metrics):
        if not repaired_manifest_path.exists():
            raise FileNotFoundError(
                f"Missing repaired manifest for flow/density metrics: "
                f"'{repaired_manifest_path}'."
            )
        repaired_manifest = pd.read_csv(repaired_manifest_path)

    for metric in metrics:
        if metric in {"flow", "density"}:
            metric_rows = repaired_manifest[repaired_manifest["metric"] == metric]
            for _, record in metric_rows.iterrows():
                metadata = parse_i24_matrix_metadata(record["output_path"])
                rows.append(
                    {
                        **metadata,
                        "shape": record.get("repaired_shape"),
                    }
                )
            continue

        if metric not in manifest_specs:
            raise ValueError(
                f"Unsupported metric '{metric}'. Expected one of "
                f"{sorted([*manifest_specs, 'density', 'flow'])}."
            )

        manifest_name, path_column = manifest_specs[metric]
        manifest_path = input_path / manifest_name
        manifest_df = pd.read_csv(manifest_path)

        for _, record in manifest_df.iterrows():
            metadata = parse_i24_matrix_metadata(record[path_column])
            rows.append(
                {
                    **metadata,
                    "shape": record.get("shape"),
                }
            )

    manifest = pd.DataFrame(rows)
    return manifest.sort_values(
        ["day", "metric", "dt", "dx_meters"], kind="stable"
    ).reset_index(drop=True)


def build_i24_mask_index_map(
    metrics: tuple[str, ...] = ("velocity", "flow_per_lane"),
    day_names: tuple[str, ...] | None = None,
    dt_values: tuple[str, ...] | None = None,
    dx_values: tuple[int, ...] | None = None,
    input_dir: str = I24_REPAIRED_MATRIX_DIR,
    num_masks: int = 5,
    rng: np.random.Generator | int | None = None,
) -> dict[str, list[np.ndarray]]:
    """
    Precompute reusable row-mask index sets keyed by matrix path.

    Masks are generated once per `(day, direction, start, end, dt, dx)` group and
    assigned to every requested metric in that group. This keeps the hidden rows
    aligned across flow, density, velocity, and flow-per-lane matrices.
    """
    manifest = load_i24_experiment_manifest(metrics=metrics, input_dir=input_dir)
    if day_names is not None:
        manifest = manifest[manifest["day"].isin(day_names)]
    if dt_values is not None:
        manifest = manifest[manifest["dt"].isin(dt_values)]
    if dx_values is not None:
        manifest = manifest[manifest["dx_meters"].isin(dx_values)]
    manifest = manifest.reset_index(drop=True)

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    mask_indices_by_file: dict[str, list[np.ndarray]] = {}

    group_columns = [
        "day",
        "direction",
        "start_label",
        "end_label",
        "dt",
        "dx_meters",
    ]
    for _, group in manifest.groupby(group_columns, sort=False):
        matrices = [np.load(path) for path in group["matrix_path"]]
        first_shape = matrices[0].shape
        if any(matrix.shape != first_shape for matrix in matrices):
            shapes = {
                Path(path).name: matrix.shape
                for path, matrix in zip(group["matrix_path"], matrices, strict=True)
            }
            raise ValueError(f"Metric matrix shape mismatch within mask group: {shapes}.")

        eligible_rows = find_common_fully_observed_rows(matrices)
        shared_mask_indices = generate_row_mask_indices(
            num_rows=first_shape[0],
            resolution=int(group.iloc[0]["dx_meters"]),
            eligible_rows=eligible_rows,
            num_masks=num_masks,
            rng=generator,
        )
        for matrix_path in group["matrix_path"]:
            mask_indices_by_file[matrix_path] = [
                rows.astype(int, copy=False) for rows in shared_mask_indices
            ]

    return mask_indices_by_file


def save_mask_index_map(
    mask_indices_by_file: dict[str, list[np.ndarray]],
    output_path: str | Path,
) -> Path:
    """Persist a precomputed mask map to a pickle file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(mask_indices_by_file, f)
    return path


def load_mask_index_map(
    input_path: str | Path,
) -> dict[str, list[np.ndarray]]:
    """Load a precomputed mask map from a pickle file."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Mask map pickle not found at '{path}'.")
    with path.open("rb") as f:
        return pickle.load(f)


def get_i24_mask_indices_by_dx(
    mask_indices_by_file: dict[str, list[np.ndarray]],
    day_name: str,
    metric: str,
    dt_label: str,
    direction: str = "west",
    start_label: str = "1200",
    end_label: str = "1600",
) -> dict[int, list[np.ndarray]]:
    """
    Convert the full-path mask map into the `{dx_meters: [row-index arrays]}` shape
    expected by the existing figure helpers.
    """
    selected: dict[int, list[np.ndarray]] = {}

    for matrix_path, mask_list in mask_indices_by_file.items():
        metadata = parse_i24_matrix_metadata(matrix_path)
        if (
            metadata["day"] == day_name
            and metadata["metric"] == metric
            and metadata["dt"] == dt_label
            and metadata["direction"] == direction
            and metadata["start_label"] == start_label
            and metadata["end_label"] == end_label
        ):
            selected[int(metadata["dx_meters"])] = [
                np.asarray(mask_rows, dtype=int) for mask_rows in mask_list
            ]

    if not selected:
        raise ValueError(
            "No matching masks found for "
            f"day={day_name}, metric={metric}, dt={dt_label}, "
            f"direction={direction}, start={start_label}, end={end_label}."
        )

    return dict(sorted(selected.items()))


def select_artifact_mask_index(
    metrics_for_case: list[dict[str, Any]],
    artifact_mask_index: int = 0,
) -> int:
    """Choose which of the saved mask realizations should be persisted as the artifact."""
    available_indices = {int(record["mask_index"]) for record in metrics_for_case}
    if artifact_mask_index not in available_indices:
        raise ValueError(
            f"artifact_mask_index={artifact_mask_index} is not available. "
            f"Found mask indices {sorted(available_indices)}."
        )
    return int(artifact_mask_index)


def _save_metric_matrices(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    matrices_dir = output_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    if summary_df.empty:
        return

    for (day, metric), group in summary_df.groupby(["day", "metric"], sort=True):
        for score_name in ("mae", "mape", "rmse"):
            matrix = (
                group.pivot(index="dt", columns="dx_meters", values=f"{score_name}_mean")
                .sort_index(axis=0)
                .sort_index(axis=1)
            )
            matrix.to_csv(matrices_dir / f"{day}_{metric}_{score_name}_matrix.csv")


def _build_summary_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    metric_columns = {"mask_index", "mae", "mape", "rmse"}
    group_columns = [column for column in raw_df.columns if column not in metric_columns]

    summary = (
        raw_df.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
        .sort_values(["day", "metric", "dt", "dx_meters"], kind="stable")
        .reset_index(drop=True)
    )
    return summary


def run_i24_imputation_experiment(
    method_name: str,
    evaluate_on_masks_fn: EvaluateOnMasksFn,
    method_kwargs: dict[str, Any] | None = None,
    per_matrix_kwargs_fn: PerMatrixKwargsFn | None = None,
    metrics: tuple[str, ...] = ("velocity", "flow_per_lane"),
    day_names: tuple[str, ...] | None = None,
    dt_values: tuple[str, ...] | None = None,
    dx_values: tuple[int, ...] | None = None,
    input_dir: str = I24_REPAIRED_MATRIX_DIR,
    results_root: str = "results",
    num_masks: int = 5,
    artifact_mask_index: int = 0,
    rng: np.random.Generator | int | None = None,
    mask_indices_by_file: dict[str, list[np.ndarray]] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run an imputation experiment across the repaired I-24 matrix sweep.

    The evaluator must accept:
    - `ground_truth`
    - `masked_matrices`
    - `return_imputed_matrices=True`

    It must return `(all_metrics, all_imputed_matrices)` where `all_metrics`
    contains one record per mask with a `mask_index` key.
    """
    if num_masks <= 0:
        raise ValueError("num_masks must be positive.")
    if artifact_mask_index < 0:
        raise ValueError("artifact_mask_index must be non-negative.")

    base_method_kwargs = {} if method_kwargs is None else dict(method_kwargs)
    result_config_fields = _extract_scalar_fields(base_method_kwargs)

    manifest = load_i24_experiment_manifest(metrics=metrics, input_dir=input_dir)
    if day_names is not None:
        manifest = manifest[manifest["day"].isin(day_names)]
    if dt_values is not None:
        manifest = manifest[manifest["dt"].isin(dt_values)]
    if dx_values is not None:
        manifest = manifest[manifest["dx_meters"].isin(dx_values)]
    manifest = manifest.reset_index(drop=True)

    output_dir = Path(results_root) / method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    imputed_dir = output_dir / "imputed_matrices"
    imputed_dir.mkdir(parents=True, exist_ok=True)

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    if mask_indices_by_file is None:
        mask_indices_by_file = build_i24_mask_index_map(
            metrics=metrics,
            day_names=day_names,
            dt_values=dt_values,
            dx_values=dx_values,
            input_dir=input_dir,
            num_masks=num_masks,
            rng=generator,
        )

    raw_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []

    for _, manifest_row in manifest.iterrows():
        matrix_path = manifest_row["matrix_path"]
        ground_truth = np.load(matrix_path)

        if matrix_path in mask_indices_by_file:
            masked_matrices = apply_row_masks(
                matrix=ground_truth,
                mask_indices=mask_indices_by_file[matrix_path],
            )
        else:
            masked_matrices = generate_masked_row_arrays(
                matrix=ground_truth,
                resolution=int(manifest_row["dx_meters"]),
                num_masks=num_masks,
                rng=generator,
            )

        dynamic_kwargs = {} if per_matrix_kwargs_fn is None else dict(per_matrix_kwargs_fn(manifest_row))
        call_kwargs = {
            **base_method_kwargs,
            **dynamic_kwargs,
            "ground_truth": ground_truth,
            "masked_matrices": masked_matrices,
            "return_imputed_matrices": True,
        }

        all_metrics, all_imputed_matrices = evaluate_on_masks_fn(**call_kwargs)
        if len(all_metrics) != len(all_imputed_matrices):
            raise ValueError(
                "Expected the evaluator to return one imputed matrix per metrics record."
            )

        per_case_config = {
            **result_config_fields,
            **_extract_scalar_fields(dynamic_kwargs),
        }

        for metric_record in all_metrics:
            raw_rows.append(
                {
                    "file": manifest_row["file"],
                    "day": manifest_row["day"],
                    "metric": manifest_row["metric"],
                    "dt": manifest_row["dt"],
                    "dx_meters": int(manifest_row["dx_meters"]),
                    **metric_record,
                    **per_case_config,
                }
            )

        selected_index = select_artifact_mask_index(
            metrics_for_case=all_metrics,
            artifact_mask_index=artifact_mask_index,
        )
        selected_position = next(
            idx for idx, record in enumerate(all_metrics)
            if int(record["mask_index"]) == selected_index
        )
        imputed_matrix = np.asarray(all_imputed_matrices[selected_position], dtype=float)
        artifact_filename = (
            f"{manifest_row['day']}_{manifest_row['metric']}_dt_{manifest_row['dt']}"
            f"_dx_{int(manifest_row['dx_meters'])}m_mask_{selected_index}_imputed.npy"
        )
        artifact_path = imputed_dir / artifact_filename
        np.save(artifact_path, imputed_matrix)

        artifact_rows.append(
            {
                "method": method_name,
                "file": manifest_row["file"],
                "matrix_path": matrix_path,
                "day": manifest_row["day"],
                "metric": manifest_row["metric"],
                "dt": manifest_row["dt"],
                "dx_meters": int(manifest_row["dx_meters"]),
                "artifact_mask_index": selected_index,
                "artifact_path": str(artifact_path),
                **per_case_config,
            }
        )

    raw_df = (
        pd.DataFrame(raw_rows)
        .sort_values(["day", "metric", "dt", "dx_meters", "mask_index"], kind="stable")
        .reset_index(drop=True)
    )
    summary_df = _build_summary_table(raw_df)
    artifact_df = (
        pd.DataFrame(artifact_rows)
        .sort_values(["day", "metric", "dt", "dx_meters"], kind="stable")
        .reset_index(drop=True)
    )

    raw_df.to_csv(output_dir / "raw_results.csv", index=False)
    summary_df.to_csv(output_dir / "summary_results.csv", index=False)
    artifact_df.to_csv(output_dir / "imputed_matrix_manifest.csv", index=False)
    _save_metric_matrices(summary_df=summary_df, output_dir=output_dir)

    return {
        "manifest": manifest,
        "raw_results": raw_df,
        "summary_results": summary_df,
        "artifact_manifest": artifact_df,
    }


def run_i24_knn_experiment(
    method_kwargs: dict[str, Any] | None = None,
    scale_neighborhood_with_resolution: bool = True,
    base_dx_meters: int = 400,
    base_dt_label: str = "30s",
    **experiment_kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Convenience wrapper for the spatiotemporal KNN baseline."""
    base_method_kwargs = {} if method_kwargs is None else dict(method_kwargs)
    user_per_matrix_kwargs_fn = experiment_kwargs.pop("per_matrix_kwargs_fn", None)

    resolution_scaled_kwargs_fn = None
    if scale_neighborhood_with_resolution:
        resolution_scaled_kwargs_fn = lambda row: build_resolution_scaled_window_kwargs(
            manifest_row=row,
            base_dx_meters=base_dx_meters,
            base_dt_label=base_dt_label,
            base_max_space_distance=base_method_kwargs.get("max_space_distance", 3),
            base_max_time_distance=base_method_kwargs.get("max_time_distance", 6),
        )

    return run_i24_imputation_experiment(
        method_name="knn",
        evaluate_on_masks_fn=evaluate_spatiotemporal_knn_on_masks,
        method_kwargs=base_method_kwargs,
        per_matrix_kwargs_fn=_compose_per_matrix_kwargs(
            resolution_scaled_kwargs_fn,
            user_per_matrix_kwargs_fn,
        ),
        **experiment_kwargs,
    )


def run_i24_kriging_experiment(
    method_kwargs: dict[str, Any] | None = None,
    **experiment_kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Convenience wrapper for the spatial ordinary kriging baseline."""
    base_method_kwargs = {} if method_kwargs is None else dict(method_kwargs)

    def _per_matrix_kwargs(row: pd.Series) -> dict[str, Any]:
        kwargs = {"dx_meters": int(row["dx_meters"])}
        if "per_matrix_kwargs_fn" in experiment_kwargs and experiment_kwargs["per_matrix_kwargs_fn"] is not None:
            kwargs.update(experiment_kwargs["per_matrix_kwargs_fn"](row))
        return kwargs

    forwarded_kwargs = dict(experiment_kwargs)
    forwarded_kwargs["per_matrix_kwargs_fn"] = _per_matrix_kwargs

    return run_i24_imputation_experiment(
        method_name="kriging",
        evaluate_on_masks_fn=evaluate_spatial_ordinary_kriging_on_masks,
        method_kwargs=base_method_kwargs,
        **forwarded_kwargs,
    )


def run_i24_advanced_kriging_experiment(
    method_kwargs: dict[str, Any] | None = None,
    scale_neighborhood_with_resolution: bool = True,
    base_dx_meters: int = 400,
    base_dt_label: str = "30s",
    **experiment_kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper for the Phase 1 regression-kriging baseline.

    By default, the spatial and temporal neighborhood sizes are scaled by the
    matrix resolution so the local window spans roughly the same physical extent
    across `dx` and `dt`.
    """
    base_method_kwargs = {} if method_kwargs is None else dict(method_kwargs)
    user_per_matrix_kwargs_fn = experiment_kwargs.pop("per_matrix_kwargs_fn", None)

    def _inject_dx_meters(row: pd.Series) -> dict[str, Any]:
        return {"dx_meters": int(row["dx_meters"])}

    resolution_scaled_kwargs_fn = None
    if scale_neighborhood_with_resolution:
        resolution_scaled_kwargs_fn = lambda row: build_resolution_scaled_window_kwargs(
            manifest_row=row,
            base_dx_meters=base_dx_meters,
            base_dt_label=base_dt_label,
            base_max_space_distance=base_method_kwargs.get("max_space_distance", 3),
            base_max_time_distance=base_method_kwargs.get("max_time_distance", 6),
        )

    return run_i24_imputation_experiment(
        method_name="advanced_kriging",
        evaluate_on_masks_fn=evaluate_regression_kriging_on_masks,
        method_kwargs=base_method_kwargs,
        per_matrix_kwargs_fn=_compose_per_matrix_kwargs(
            _inject_dx_meters,
            resolution_scaled_kwargs_fn,
            user_per_matrix_kwargs_fn,
        ),
        **experiment_kwargs,
    )


def run_i24_asm_experiment(
    method_kwargs: dict[str, Any] | None = None,
    **experiment_kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper for Adaptive Smoothing Method speed imputation.

    ASM expects I-24 matrices as `(space, time)` speed fields in mph, with
    missing values encoded as `np.nan`. This wrapper converts the experiment
    manifest schema into the ASM schema:

    - `dx_meters` -> `dx` in miles
    - `dt` labels like `10s`, `1min` -> `dt` in seconds

    The default experiment scope is velocity only because ASM reconstructs
    speeds, not flow or density.
    """
    base_method_kwargs = {**DEFAULT_ASM_PARAMS}
    if method_kwargs is not None:
        base_method_kwargs.update(method_kwargs)

    user_per_matrix_kwargs_fn = experiment_kwargs.pop("per_matrix_kwargs_fn", None)
    experiment_kwargs.setdefault("metrics", ("velocity",))

    def _inject_asm_schema(row: pd.Series) -> dict[str, Any]:
        kwargs = {
            "dx_miles": float(row["dx_meters"]) / METERS_PER_MILE,
            "dt_seconds": _parse_dt_label_to_seconds(str(row["dt"])),
        }
        if user_per_matrix_kwargs_fn is not None:
            kwargs.update(user_per_matrix_kwargs_fn(row))
        return kwargs

    return run_i24_imputation_experiment(
        method_name="asm",
        evaluate_on_masks_fn=evaluate_asm_on_masks,
        method_kwargs=base_method_kwargs,
        per_matrix_kwargs_fn=_inject_asm_schema,
        **experiment_kwargs,
    )
