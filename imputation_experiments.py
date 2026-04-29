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
from gnn_kriging import evaluate_gnn_kriging_on_masks
from kriging import evaluate_spatial_ordinary_kriging_on_masks
from knn import evaluate_spatiotemporal_knn_on_masks
from metanet_imputation import evaluate_metanet_on_masks
from methods import (
    ROW_MASK_FRACTIONS_BY_RESOLUTION,
    _normalize_resolution,
    apply_row_masks,
)


I24_REPAIRED_MATRIX_DIR = "data/i24/matrix_sweeps/daily_combined_repaired"
DEFAULT_I24_MASK_INDEX_PATH = "results/masks/i24_row_mask_indices.pkl"
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


def _coerce_resolution_mask_map(
    mask_data: Any,
) -> dict[int, list[np.ndarray]]:
    """Normalize supported saved-mask payloads to `{dx_meters: [row arrays]}`."""
    if isinstance(mask_data, dict) and "masks_by_dx" in mask_data:
        raw_masks = mask_data["masks_by_dx"]
    else:
        raw_masks = mask_data

    if not isinstance(raw_masks, dict):
        raise TypeError("Saved mask data must be a dictionary keyed by spatial resolution.")

    masks_by_dx: dict[int, list[np.ndarray]] = {}
    for raw_dx, raw_mask_list in raw_masks.items():
        dx_meters = _normalize_resolution(raw_dx)
        if dx_meters not in ROW_MASK_FRACTIONS_BY_RESOLUTION:
            raise ValueError(
                f"Unsupported mask resolution {raw_dx!r}. Expected one of "
                f"{sorted(ROW_MASK_FRACTIONS_BY_RESOLUTION)} meters."
            )
        if not isinstance(raw_mask_list, (list, tuple)):
            raise TypeError(f"Masks for {dx_meters}m must be a list of row-index arrays.")
        masks_by_dx[dx_meters] = [
            np.asarray(mask_rows, dtype=int) for mask_rows in raw_mask_list
        ]

    return masks_by_dx


def _validate_predefined_masks_for_manifest(
    masks_by_dx: dict[int, list[np.ndarray]],
    manifest: pd.DataFrame,
) -> None:
    """Validate saved row masks against the current I-24 matrix shapes."""
    for dx_meters, group in manifest.groupby("dx_meters", sort=False):
        dx_meters = int(dx_meters)
        if dx_meters not in masks_by_dx:
            raise ValueError(f"Saved mask map is missing {dx_meters}m masks.")

        row_counts = set()
        for shape in group["shape"]:
            row_count = int(str(shape).split("x", maxsplit=1)[0])
            row_counts.add(row_count)
        if len(row_counts) != 1:
            raise ValueError(f"Expected one row count for {dx_meters}m; got {row_counts}.")

        num_rows = row_counts.pop()
        expected_masked_rows = int(
            round(num_rows * ROW_MASK_FRACTIONS_BY_RESOLUTION[dx_meters])
        )
        mask_list = masks_by_dx[dx_meters]
        if len(mask_list) != 5:
            raise ValueError(f"Expected 5 saved masks for {dx_meters}m; got {len(mask_list)}.")

        for mask_index, mask_rows in enumerate(mask_list):
            if mask_rows.ndim != 1:
                raise ValueError(f"{dx_meters}m mask {mask_index} must be one-dimensional.")
            if mask_rows.size != expected_masked_rows:
                raise ValueError(
                    f"{dx_meters}m mask {mask_index} has {mask_rows.size} rows; "
                    f"expected {expected_masked_rows} from the configured sparsity ratio."
                )
            if np.unique(mask_rows).size != mask_rows.size:
                raise ValueError(f"{dx_meters}m mask {mask_index} contains duplicate rows.")
            if mask_rows.size and (mask_rows.min() <= 0 or mask_rows.max() >= num_rows - 1):
                raise ValueError(
                    f"{dx_meters}m mask {mask_index} includes a boundary row. "
                    "The first and last spatial segments must remain observed."
                )


def expand_i24_predefined_mask_index_map(
    masks_by_dx: dict[int, list[np.ndarray]],
    metrics: tuple[str, ...] = ("velocity", "flow_per_lane"),
    day_names: tuple[str, ...] | None = None,
    dt_values: tuple[str, ...] | None = None,
    dx_values: tuple[int, ...] | None = None,
    input_dir: str = I24_REPAIRED_MATRIX_DIR,
) -> dict[str, list[np.ndarray]]:
    """
    Expand saved resolution-level masks to the path-keyed map used by evaluators.

    The same five masks for each spatial resolution are applied to every day,
    temporal resolution, and metric in the selected manifest.
    """
    manifest = load_i24_experiment_manifest(metrics=metrics, input_dir=input_dir)
    if day_names is not None:
        manifest = manifest[manifest["day"].isin(day_names)]
    if dt_values is not None:
        manifest = manifest[manifest["dt"].isin(dt_values)]
    if dx_values is not None:
        manifest = manifest[manifest["dx_meters"].isin(dx_values)]
    manifest = manifest.reset_index(drop=True)

    _validate_predefined_masks_for_manifest(masks_by_dx, manifest)
    return {
        str(row["matrix_path"]): [
            mask_rows.astype(int, copy=False)
            for mask_rows in masks_by_dx[int(row["dx_meters"])]
        ]
        for _, row in manifest.iterrows()
    }


def build_i24_mask_index_map(
    metrics: tuple[str, ...] = ("velocity", "flow_per_lane"),
    day_names: tuple[str, ...] | None = None,
    dt_values: tuple[str, ...] | None = None,
    dx_values: tuple[int, ...] | None = None,
    input_dir: str = I24_REPAIRED_MATRIX_DIR,
    mask_path: str | Path = DEFAULT_I24_MASK_INDEX_PATH,
) -> dict[str, list[np.ndarray]]:
    """
    Load predefined row masks and return them keyed by matrix path.

    This function no longer generates masks. The saved mask artifact is the
    single source of truth, with one set of five masks per spatial resolution.
    """
    masks_by_dx = _coerce_resolution_mask_map(load_mask_index_map(mask_path))
    return expand_i24_predefined_mask_index_map(
        masks_by_dx=masks_by_dx,
        metrics=metrics,
        day_names=day_names,
        dt_values=dt_values,
        dx_values=dx_values,
        input_dir=input_dir,
    )


def _find_i24_metric_matrix_path(
    manifest_row: pd.Series,
    *,
    metric: str,
    input_dir: str = I24_REPAIRED_MATRIX_DIR,
) -> str:
    """Return the repaired matrix path for another metric in the same I-24 case."""
    current_metric = str(manifest_row["metric"])
    if current_metric == metric:
        return str(manifest_row["matrix_path"])

    current_path = Path(str(manifest_row["matrix_path"]))
    candidate = current_path.with_name(
        current_path.name.replace(f"_{current_metric}.npy", f"_{metric}.npy")
    )
    if candidate.exists():
        return str(candidate)

    manifest = load_i24_experiment_manifest(metrics=(metric,), input_dir=input_dir)
    match = manifest[
        (manifest["day"] == manifest_row["day"])
        & (manifest["direction"] == manifest_row["direction"])
        & (manifest["start_label"] == manifest_row["start_label"])
        & (manifest["end_label"] == manifest_row["end_label"])
        & (manifest["dt"] == manifest_row["dt"])
        & (manifest["dx_meters"].astype(int) == int(manifest_row["dx_meters"]))
    ]
    if match.empty:
        raise FileNotFoundError(
            f"Could not find companion {metric} matrix for "
            f"{manifest_row['file']}."
        )
    return str(match.iloc[0]["matrix_path"])


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
    artifact_mask_index: int = 0,
    mask_indices_by_file: dict[str, list[np.ndarray]] | None = None,
    mask_path: str | Path = DEFAULT_I24_MASK_INDEX_PATH,
    mask_indices_to_run: tuple[int, ...] | list[int] | None = None,
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
    if artifact_mask_index < 0:
        raise ValueError("artifact_mask_index must be non-negative.")
    selected_mask_indices = (
        None
        if mask_indices_to_run is None
        else tuple(int(index) for index in mask_indices_to_run)
    )
    if selected_mask_indices is not None:
        if not selected_mask_indices:
            raise ValueError("mask_indices_to_run must not be empty when provided.")
        if any(index < 0 for index in selected_mask_indices):
            raise ValueError("mask_indices_to_run values must be non-negative.")

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

    if mask_indices_by_file is None:
        mask_indices_by_file = build_i24_mask_index_map(
            metrics=metrics,
            day_names=day_names,
            dt_values=dt_values,
            dx_values=dx_values,
            input_dir=input_dir,
            mask_path=mask_path,
        )

    raw_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []

    for _, manifest_row in manifest.iterrows():
        matrix_path = manifest_row["matrix_path"]
        ground_truth = np.load(matrix_path)

        if matrix_path not in mask_indices_by_file:
            raise ValueError(
                f"No predefined mask indices found for '{matrix_path}'. "
                f"Check the saved mask artifact at '{mask_path}'."
            )
        full_mask_list = mask_indices_by_file[matrix_path]
        if selected_mask_indices is None:
            mask_list_for_case = full_mask_list
            original_mask_indices = tuple(range(len(full_mask_list)))
        else:
            if max(selected_mask_indices) >= len(full_mask_list):
                raise ValueError(
                    f"Requested mask index {max(selected_mask_indices)} for "
                    f"'{matrix_path}', but only {len(full_mask_list)} masks are available."
                )
            mask_list_for_case = [full_mask_list[index] for index in selected_mask_indices]
            original_mask_indices = selected_mask_indices

        masked_matrices = apply_row_masks(
            matrix=ground_truth,
            mask_indices=mask_list_for_case,
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
        if len(all_metrics) != len(original_mask_indices):
            raise ValueError(
                "Expected the evaluator to return one metrics record per requested mask."
            )

        per_case_config = {
            **result_config_fields,
            **_extract_scalar_fields(dynamic_kwargs),
        }

        remapped_metrics = []
        for metric_record, original_mask_index in zip(
            all_metrics,
            original_mask_indices,
            strict=True,
        ):
            remapped_record = {
                **metric_record,
                "mask_index": int(original_mask_index),
            }
            remapped_metrics.append(remapped_record)
            raw_rows.append(
                {
                    "file": manifest_row["file"],
                    "day": manifest_row["day"],
                    "metric": manifest_row["metric"],
                    "dt": manifest_row["dt"],
                    "dx_meters": int(manifest_row["dx_meters"]),
                    **remapped_record,
                    **per_case_config,
                }
            )

        selected_index = select_artifact_mask_index(
            metrics_for_case=remapped_metrics,
            artifact_mask_index=artifact_mask_index,
        )
        selected_position = next(
            idx for idx, record in enumerate(remapped_metrics)
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


def run_i24_gnn_kriging_experiment(
    method_kwargs: dict[str, Any] | None = None,
    **experiment_kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper for STCAGCN-based GNN kriging.

    The GNN evaluator uses the matrix under evaluation as the reconstruction
    target and the companion `velocity` matrix as speed-pattern side
    information for the adaptive graph.
    """
    base_method_kwargs = {} if method_kwargs is None else dict(method_kwargs)
    user_per_matrix_kwargs_fn = experiment_kwargs.pop("per_matrix_kwargs_fn", None)
    input_dir = experiment_kwargs.get("input_dir", I24_REPAIRED_MATRIX_DIR)
    results_root = experiment_kwargs.get("results_root", "results")
    base_method_kwargs.setdefault(
        "training_history_output_dir",
        str(Path(results_root) / "gnn_kriging" / "training_loss"),
    )

    def _inject_velocity_matrix(row: pd.Series) -> dict[str, Any]:
        velocity_path = _find_i24_metric_matrix_path(
            row,
            metric="velocity",
            input_dir=input_dir,
        )
        kwargs = {
            "velocity_matrix": np.load(velocity_path),
            "training_history_prefix": Path(str(row["file"])).stem,
        }
        if base_method_kwargs.get("mask_fraction") is None:
            kwargs["mask_fraction"] = ROW_MASK_FRACTIONS_BY_RESOLUTION[
                int(row["dx_meters"])
            ]
        if user_per_matrix_kwargs_fn is not None:
            kwargs.update(user_per_matrix_kwargs_fn(row))
        return kwargs

    return run_i24_imputation_experiment(
        method_name="gnn_kriging",
        evaluate_on_masks_fn=evaluate_gnn_kriging_on_masks,
        method_kwargs=base_method_kwargs,
        per_matrix_kwargs_fn=_inject_velocity_matrix,
        **experiment_kwargs,
    )


def run_i24_gnn_no_spam_experiment(
    method_kwargs: dict[str, Any] | None = None,
    **experiment_kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper for the STCAGCN ablation without SPAM/velocity input.

    This uses the same static chain adjacency and target-row holdout protocol as
    the main GNN experiment, but disables adaptive speed-pattern adjacency and
    does not pass velocity data to training or inference.
    """
    base_method_kwargs = {"use_spam": False}
    if method_kwargs is not None:
        base_method_kwargs.update(method_kwargs)

    user_per_matrix_kwargs_fn = experiment_kwargs.pop("per_matrix_kwargs_fn", None)
    results_root = experiment_kwargs.get("results_root", "results")
    base_method_kwargs.setdefault(
        "training_history_output_dir",
        str(Path(results_root) / "gnn_no_spam" / "training_loss"),
    )

    def _inject_no_spam_schema(row: pd.Series) -> dict[str, Any]:
        kwargs = {
            "training_history_prefix": Path(str(row["file"])).stem,
        }
        if base_method_kwargs.get("mask_fraction") is None:
            kwargs["mask_fraction"] = ROW_MASK_FRACTIONS_BY_RESOLUTION[
                int(row["dx_meters"])
            ]
        if user_per_matrix_kwargs_fn is not None:
            kwargs.update(user_per_matrix_kwargs_fn(row))
        return kwargs

    return run_i24_imputation_experiment(
        method_name="gnn_no_spam",
        evaluate_on_masks_fn=evaluate_gnn_kriging_on_masks,
        method_kwargs=base_method_kwargs,
        per_matrix_kwargs_fn=_inject_no_spam_schema,
        **experiment_kwargs,
    )


def run_i24_asm_experiment(
    method_kwargs: dict[str, Any] | None = None,
    scale_delta_tau_to_resolution: bool = False,
    **experiment_kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper for Adaptive Smoothing Method speed imputation.

    ASM expects I-24 matrices as `(space, time)` speed fields in mph, with
    missing values encoded as `np.nan`. This wrapper converts the experiment
    manifest schema into the ASM schema:

    - `dx_meters` -> `dx` in miles
    - `dt` labels like `10s`, `1min` -> `dt` in seconds

    Set `scale_delta_tau_to_resolution=True` to use `delta=dx_miles` and
    `tau=dt_seconds` for each matrix, i.e. smoothing scales of one neighboring
    cell in space and time.
    """
    base_method_kwargs = {
        **DEFAULT_ASM_PARAMS,
        # I-24 westbound matrices are indexed by increasing postmile, while ASM's
        # wave-speed convention is easier to interpret on the downstream axis.
        "space_axis_sign": -1,
    }
    if method_kwargs is not None:
        base_method_kwargs.update(method_kwargs)

    user_per_matrix_kwargs_fn = experiment_kwargs.pop("per_matrix_kwargs_fn", None)
    experiment_kwargs.setdefault("metrics", ("velocity",))

    def _inject_asm_schema(row: pd.Series) -> dict[str, Any]:
        dx_miles = float(row["dx_meters"]) / METERS_PER_MILE
        dt_seconds = _parse_dt_label_to_seconds(str(row["dt"]))
        kwargs = {
            "dx_miles": dx_miles,
            "dt_seconds": dt_seconds,
        }
        if scale_delta_tau_to_resolution:
            kwargs.update(
                {
                    "delta": dx_miles,
                    "tau": dt_seconds,
                }
            )
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


def plot_i24_imputation_summary_matrices(
    summary_results: pd.DataFrame | str | Path,
    output_dir: str | Path | None = None,
    score_names: tuple[str, ...] = ("mae", "mape", "rmse"),
    metrics: tuple[str, ...] | None = None,
    method_name: str | None = None,
    value_suffix: str = "_mean",
    cmap: str = "viridis",
    annotate: bool = True,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> dict[tuple[str, str], Any]:
    """
    Plot resolution-sweep heatmaps from an imputation `summary_results` table.

    Rows are temporal resolutions (`dt`), columns are spatial resolutions
    (`dx_meters`), and values are score columns such as `mae_mean`.
    """
    import matplotlib.pyplot as plt

    if isinstance(summary_results, (str, Path)):
        summary_df = pd.read_csv(summary_results)
    else:
        summary_df = summary_results.copy()

    if summary_df.empty:
        raise ValueError("summary_results is empty.")

    selected_metrics = (
        tuple(metrics)
        if metrics is not None
        else tuple(summary_df["metric"].drop_duplicates())
    )
    output_path = None if output_dir is None else Path(output_dir)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    figures: dict[tuple[str, str], Any] = {}
    dt_order = ["10s", "30s", "1min", "5min"]

    for metric in selected_metrics:
        metric_df = summary_df[summary_df["metric"] == metric]
        if metric_df.empty:
            continue

        for score_name in score_names:
            value_column = f"{score_name}{value_suffix}"
            if value_column not in metric_df.columns:
                raise ValueError(
                    f"Missing score column '{value_column}' in summary_results."
                )

            matrix = (
                metric_df.pivot(
                    index="dt",
                    columns="dx_meters",
                    values=value_column,
                )
                .reindex([dt for dt in dt_order if dt in set(metric_df["dt"])])
                .sort_index(axis=1)
            )

            fig, ax = plt.subplots(figsize=figsize)
            image = ax.imshow(matrix.values, aspect="auto", cmap=cmap)
            ax.set_title(
                f"{method_name or 'imputation'} {metric} {score_name.upper()}"
            )
            ax.set_xlabel("Spatial resolution (m)")
            ax.set_ylabel("Temporal resolution")
            ax.set_xticks(np.arange(matrix.shape[1]))
            ax.set_xticklabels(matrix.columns.astype(int))
            ax.set_yticks(np.arange(matrix.shape[0]))
            ax.set_yticklabels(matrix.index.astype(str))
            fig.colorbar(image, ax=ax, label=value_column)

            if annotate:
                finite_values = matrix.values[np.isfinite(matrix.values)]
                threshold = (
                    float(np.nanmin(finite_values) + np.nanmax(finite_values)) / 2.0
                    if finite_values.size
                    else 0.0
                )
                for row_idx in range(matrix.shape[0]):
                    for col_idx in range(matrix.shape[1]):
                        value = matrix.iat[row_idx, col_idx]
                        if np.isfinite(value):
                            color = "white" if value > threshold else "black"
                            ax.text(
                                col_idx,
                                row_idx,
                                f"{value:.2g}",
                                ha="center",
                                va="center",
                                color=color,
                                fontsize=8,
                            )

            fig.tight_layout()
            figures[(metric, score_name)] = fig

            if output_path is not None:
                method_prefix = f"{method_name}_" if method_name else ""
                fig.savefig(
                    output_path / f"{method_prefix}{metric}_{score_name}_heatmap.png",
                    dpi=200,
                    bbox_inches="tight",
                )

    return figures


def plot_i24_imputed_matrix_artifacts(
    artifact_manifest: pd.DataFrame | str | Path,
    mask_indices_by_file: dict[str, list[np.ndarray]] | None = None,
    input_dir: str = I24_REPAIRED_MATRIX_DIR,
    output_dir: str | Path | None = None,
    mask_path: str | Path = DEFAULT_I24_MASK_INDEX_PATH,
    metrics: tuple[str, ...] | None = None,
    day_names: tuple[str, ...] | None = None,
    dt_values: tuple[str, ...] | None = None,
    dx_values: tuple[int, ...] | None = None,
    max_figures: int | None = None,
    include_error: bool = False,
) -> dict[str, Any]:
    """
    Plot original, masked, and reconstructed matrices from saved imputation artifacts.

    If `mask_indices_by_file` is not passed, masks are loaded from the
    predefined mask artifact at `mask_path`.
    """
    import matplotlib.pyplot as plt
    from data_utils import (
        _format_dt_resolution_label,
        _format_dx_resolution_label,
        _get_i24_plot_bounds,
        _get_i24_time_bounds,
        _plot_matrix_on_ax,
    )

    if isinstance(artifact_manifest, (str, Path)):
        artifact_df = pd.read_csv(artifact_manifest)
    else:
        artifact_df = artifact_manifest.copy()

    if artifact_df.empty:
        raise ValueError("artifact_manifest is empty.")

    if metrics is not None:
        artifact_df = artifact_df[artifact_df["metric"].isin(metrics)]
    if day_names is not None:
        artifact_df = artifact_df[artifact_df["day"].isin(day_names)]
    if dt_values is not None:
        artifact_df = artifact_df[artifact_df["dt"].isin(dt_values)]
    if dx_values is not None:
        artifact_df = artifact_df[artifact_df["dx_meters"].astype(int).isin(dx_values)]
    artifact_df = artifact_df.reset_index(drop=True)

    if artifact_df.empty:
        raise ValueError("No artifact rows remain after filtering.")

    if mask_indices_by_file is None:
        mask_indices_by_file = build_i24_mask_index_map(
            metrics=tuple(artifact_df["metric"].drop_duplicates()),
            day_names=tuple(artifact_df["day"].drop_duplicates()),
            dt_values=tuple(artifact_df["dt"].drop_duplicates()),
            dx_values=tuple(int(value) for value in artifact_df["dx_meters"].drop_duplicates()),
            input_dir=input_dir,
            mask_path=mask_path,
        )

    output_path = None if output_dir is None else Path(output_dir)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    rows_to_plot = artifact_df
    if max_figures is not None:
        rows_to_plot = rows_to_plot.head(max_figures)

    figures: dict[str, Any] = {}
    for _, row in rows_to_plot.iterrows():
        matrix_path = str(row["matrix_path"])
        ground_truth = np.load(matrix_path)
        imputed = np.load(row["artifact_path"])
        artifact_mask_index = int(row["artifact_mask_index"])

        if matrix_path not in mask_indices_by_file:
            raise ValueError(f"No mask indices found for '{matrix_path}'.")
        mask_list = mask_indices_by_file[matrix_path]
        if artifact_mask_index >= len(mask_list):
            raise ValueError(
                f"artifact_mask_index={artifact_mask_index} is not available for "
                f"'{matrix_path}'."
            )

        masked = apply_row_masks(ground_truth, [mask_list[artifact_mask_index]])[0]
        error = imputed - ground_truth

        panels = [
            ("Original", ground_truth, None),
            ("Masked", masked, None),
            ("Reconstructed", imputed, None),
        ]
        if include_error:
            max_abs_error = float(np.nanmax(np.abs(error)))
            panels.append(("Reconstruction Error", error, (-max_abs_error, max_abs_error)))

        finite_values = np.concatenate(
            [
                np.asarray(matrix, dtype=float)[np.isfinite(matrix)]
                for _, matrix, colorbar_range in panels
                if colorbar_range is None and np.any(np.isfinite(matrix))
            ]
        )
        shared_range = (
            (float(np.nanmin(finite_values)), float(np.nanmax(finite_values)))
            if finite_values.size
            else None
        )

        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(5.2 * len(panels), 4.5),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)

        dx_meters = int(row["dx_meters"])
        t_min, t_max = _get_i24_time_bounds()
        start_pm, end_pm = _get_i24_plot_bounds(dx_meters)
        time_resolution_label = _format_dt_resolution_label(str(row["dt"]))
        space_resolution_label = _format_dx_resolution_label(dx_meters)

        last_pcm = None
        for ax, (panel_title, matrix, panel_range) in zip(axes, panels, strict=True):
            colorbar_range = panel_range if panel_range is not None else shared_range
            last_pcm = _plot_matrix_on_ax(
                ax=ax,
                matrix=matrix,
                title=panel_title,
                colorbar_label=str(row["metric"]),
                colorbar_range=colorbar_range,
                t_min=t_min,
                t_max=t_max,
                start_pm=start_pm,
                end_pm=end_pm,
                add_colorbar=False,
                show_ylabel=ax is axes[0],
                title_prefix="",
                time_resolution_label=time_resolution_label,
                space_resolution_label=space_resolution_label,
            )

        if last_pcm is not None:
            fig.colorbar(last_pcm, ax=axes.ravel().tolist(), shrink=0.85)

        figure_key = (
            f"{row['method']}_{row['day']}_{row['metric']}_dt_{row['dt']}"
            f"_dx_{dx_meters}m_mask_{artifact_mask_index}"
        )
        fig.suptitle(figure_key)
        figures[figure_key] = fig

        if output_path is not None:
            fig.savefig(
                output_path / f"{figure_key}.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)
        elif plt.get_backend().lower() != "agg":
            plt.show()

    return figures


def run_i24_metanet_experiment(
    method_kwargs: dict[str, Any] | None = None,
    **experiment_kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper for METANET physics-based imputation.

    METANET calibrates on total flow and speed, so companion `flow` and
    `velocity` matrices are loaded for every evaluated metric. The default
    experiment scope is `velocity`, `flow`, and `density`; `flow_per_lane` is
    also supported when explicitly requested.
    """
    base_method_kwargs = {} if method_kwargs is None else dict(method_kwargs)
    user_per_matrix_kwargs_fn = experiment_kwargs.pop("per_matrix_kwargs_fn", None)
    input_dir = experiment_kwargs.get("input_dir", I24_REPAIRED_MATRIX_DIR)
    metrics = tuple(experiment_kwargs.get("metrics", ("velocity", "flow", "density")))
    experiment_kwargs["metrics"] = metrics
    calibration_cache = base_method_kwargs.setdefault("calibration_cache", {})

    required_mask_metrics = tuple(dict.fromkeys((*metrics, "flow", "velocity")))
    if experiment_kwargs.get("mask_indices_by_file") is None:
        experiment_kwargs["mask_indices_by_file"] = build_i24_mask_index_map(
            metrics=required_mask_metrics,
            day_names=experiment_kwargs.get("day_names"),
            dt_values=experiment_kwargs.get("dt_values"),
            dx_values=experiment_kwargs.get("dx_values"),
            input_dir=input_dir,
            mask_path=experiment_kwargs.get("mask_path", DEFAULT_I24_MASK_INDEX_PATH),
        )

    def _inject_metanet_schema(row: pd.Series) -> dict[str, Any]:
        flow_path = _find_i24_metric_matrix_path(
            row,
            metric="flow",
            input_dir=input_dir,
        )
        velocity_path = _find_i24_metric_matrix_path(
            row,
            metric="velocity",
            input_dir=input_dir,
        )
        kwargs = {
            "metric": str(row["metric"]),
            "dx_meters": int(row["dx_meters"]),
            "dt_seconds": _parse_dt_label_to_seconds(str(row["dt"])),
            "total_flow_matrix": np.load(flow_path),
            "velocity_matrix": np.load(velocity_path),
            "case_key": (
                row["day"],
                row["direction"],
                row["start_label"],
                row["end_label"],
                row["dt"],
                int(row["dx_meters"]),
            ),
            "calibration_cache": calibration_cache,
        }
        if user_per_matrix_kwargs_fn is not None:
            kwargs.update(user_per_matrix_kwargs_fn(row))
        return kwargs

    return run_i24_imputation_experiment(
        method_name="metanet",
        evaluate_on_masks_fn=evaluate_metanet_on_masks,
        method_kwargs=base_method_kwargs,
        per_matrix_kwargs_fn=_inject_metanet_schema,
        **experiment_kwargs,
    )
