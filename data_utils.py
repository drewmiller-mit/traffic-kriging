import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ijson
import zipfile
import re
from contextlib import contextmanager
from io import TextIOWrapper
from pathlib import Path
from typing import Optional, Tuple


def _normalize_time_bound(value):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.timestamp()
    return pd.Timestamp(value).timestamp()

def _candidate_i24_paths(file_path, data_dir="data/i24"):
    if os.path.exists(file_path):
        return [file_path]

    base_name = os.path.basename(str(file_path))
    stem, ext = os.path.splitext(base_name)
    day_name = stem if ext in {".json", ".zip"} else base_name

    candidates = [
        os.path.join(data_dir, f"{day_name}.json"),
        os.path.join(data_dir, f"{day_name}.zip"),
        os.path.join(data_dir, f"{day_name}.zip.download", f"{day_name}.zip"),
    ]
    return candidates

def resolve_i24_file_path(file_path, data_dir="data/i24"):
    for candidate in _candidate_i24_paths(file_path, data_dir=data_dir):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not resolve input '{file_path}' in '{data_dir}'."
    )

@contextmanager
def open_i24_trajectory_stream(file_path, data_dir="data/i24"):
    resolved_path = resolve_i24_file_path(file_path, data_dir=data_dir)

    if resolved_path.endswith(".json"):
        with open(resolved_path, "r") as f:
            yield f
        return

    if resolved_path.endswith(".zip"):
        try:
            with zipfile.ZipFile(resolved_path) as zf:
                json_members = [name for name in zf.namelist() if name.endswith(".json")]
                if not json_members:
                    raise ValueError(
                        f"No JSON file found inside archive '{resolved_path}'."
                    )
                with zf.open(json_members[0], "r") as raw_file:
                    with TextIOWrapper(raw_file, encoding="utf-8") as f:
                        yield f
        except zipfile.BadZipFile as exc:
            raise ValueError(
                f"Archive '{resolved_path}' is not a readable zip file. "
                "If it came from a browser download, it may still be incomplete."
            ) from exc
        return

    raise ValueError(f"Unsupported input format for '{resolved_path}'.")

def resolve_i24_day_paths(day_names, data_dir="data/i24"):
    return [resolve_i24_file_path(day_name, data_dir=data_dir) for day_name in day_names]

def load_trajectories(
    file_path,
    trajectory_timeframe=pd.Timedelta(minutes=10),
    min_time=None,
    max_time=None,
    direction_str="west",
    data_dir="data/i24",
    min_mile_marker=58.7,
    max_mile_marker=62.9,
):
    """
    Loads trajectories from a given file path, filters by direction and time range.

    Args:
        file_path (str): Path to the file containing the trajectories.
        trajectory_timeframe (pd.Timedelta): Time range for which to load trajectories. Default is 10 minutes.
        min_time (pd.Timestamp): Minimum time for which to load trajectories. Default is None.
        max_time (pd.Timestamp): Maximum time for which to load trajectories. Default is None.
        direction_str (str): Direction for which to load trajectories. Default is "west".
        data_dir (str): Base directory used to resolve shorthand day names like "nov21".
        min_mile_marker (float): Lower corridor bound in mile markers.
        max_mile_marker (float): Upper corridor bound in mile markers.

    Returns:
        pd.DataFrame: DataFrame containing the loaded trajectories, with columns "trajectory_id", "timestamp", "x_position", "y_position", and "speed".
    """
    if direction_str == "west":
        direction_num = -1
    elif direction_str == "east":
        direction_num = 1
    else:
        raise ValueError("direction_str must be either 'west' or 'east'.")

    min_timestamp = _normalize_time_bound(min_time)
    max_timestamp = _normalize_time_bound(max_time)
    if (
        min_timestamp is not None
        and max_timestamp is not None
        and min_timestamp > max_timestamp
    ):
        raise ValueError("min_time must be earlier than or equal to max_time.")
    selected_trajectories = []
    t_min = None
    t_max = None
    min_position_m = min_mile_marker * 5280 * 0.3048
    max_position_m = max_mile_marker * 5280 * 0.3048
    # Open file and stream data
    with open_i24_trajectory_stream(file_path, data_dir=data_dir) as f:
        trajectory_iterator = ijson.items(f, "item")

        for traj in trajectory_iterator:
            # Mile marker 61 is 322080 feet or 98170 m
            # Mile marker 62 is 327360 feet or 99779.3 m
            x_positions = (
                np.array(traj.get("x_position", []), dtype=np.float32) * 0.3048
            )  # Convert feet to meters
            y_positions = (
                np.array(traj.get("y_position", []), dtype=np.float32) * 0.3048
            )  # Convert feet to meters
            direction = traj.get("direction")

            if len(x_positions) > 1 and direction == direction_num:
                timestamps = np.array(traj.get("timestamp", []), dtype=np.float64)
                if len(timestamps) == 0:
                    continue

                if min_timestamp is not None and timestamps[-1] < min_timestamp:
                    continue
                if max_timestamp is not None and timestamps[0] > max_timestamp:
                    break

                selected_trajectories.append(
                    {
                        "trajectory": traj,
                        "timestamps": timestamps,
                        "x_positions": x_positions,
                        "y_positions": y_positions,
                    }
                )

                # Efficient min/max tracking
                t_min = timestamps[0] if t_min is None else min(t_min, timestamps[0])
                t_max = timestamps[-1] if t_max is None else max(t_max, timestamps[-1])

                if (
                    t_max is not None
                    and t_min is not None
                    and (t_max - t_min) > trajectory_timeframe.total_seconds()
                ):
                    break

    print(f"Loaded {len(selected_trajectories)} {direction_str}bound trajectories.")

    if not selected_trajectories:
        return pd.DataFrame(
            columns=["trajectory_id", "timestamp", "x_position", "speed"]
        )

    # Vectorized DataFrame creation
    all_trajectory_ids = []
    all_timestamps = []
    all_x_positions = []
    all_y_positions = []

    for idx, traj in enumerate(selected_trajectories):
        mask = (traj["x_positions"] >= min_position_m) & (
            traj["x_positions"] <= max_position_m
        )
        if min_timestamp is not None:
            mask &= traj["timestamps"] >= min_timestamp
        if max_timestamp is not None:
            mask &= traj["timestamps"] <= max_timestamp

        filtered_timestamps = traj["timestamps"][mask]
        filtered_x_positions = traj["x_positions"][mask]
        filtered_y_positions = traj["y_positions"][mask]
        if len(filtered_timestamps) == 0:
            continue

        num_points = len(filtered_timestamps)
        all_trajectory_ids.extend([idx] * num_points)
        all_timestamps.extend(filtered_timestamps)
        all_x_positions.extend(filtered_x_positions)
        all_y_positions.extend(filtered_y_positions)
    df = pd.DataFrame(
        {
            "trajectory_id": np.array(all_trajectory_ids, dtype=np.int32),
            "timestamp": pd.to_datetime(all_timestamps, unit="s"),
            "x_position": np.array(all_x_positions, dtype=np.float32),
            "y_position": np.array(all_y_positions, dtype=np.float32),
        }
    )

    print(df.columns.tolist())
    print(df)

    return df


def _build_space_bins_in_travel_direction(
    x_min,
    x_max,
    space_interval,
    x_increases_in_travel_direction,
):
    """
    Build full-width space-bin edges anchored in travel direction.

    The trailing partial bin in the direction of travel is dropped. Because
    `pd.cut` requires increasing bin edges, this helper returns both travel-
    direction edges and ascending cut edges.
    """
    if space_interval <= 0:
        raise ValueError("space_interval must be positive.")
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min.")

    num_full_bins = int(np.floor((x_max - x_min) / space_interval))
    if num_full_bins < 1:
        raise ValueError(
            "The x-position range is smaller than one full space_interval; "
            "no full space bins remain after trimming the trailing partial bin."
        )

    if x_increases_in_travel_direction:
        space_bins = x_min + np.arange(num_full_bins + 1) * space_interval
        cut_bins = space_bins
    else:
        space_bins = x_max - np.arange(num_full_bins + 1) * space_interval
        cut_bins = space_bins[::-1]

    return space_bins, cut_bins


def _detect_bad_spatial_segments(
    flow_matrix,
    density_matrix,
    ratio_threshold=0.6,
    min_persistent_fraction=0.6,
    neighbor_similarity_threshold=0.3,
    eps=1e-6,
):
    """
    Flag spatial bins that are persistently suppressed relative to both
    immediate spatial neighbors.

    To reduce false positives from genuine traffic transitions, a row is only
    flagged when the left/right neighbors remain reasonably similar over time in
    both flow and density.
    """
    if flow_matrix.shape != density_matrix.shape:
        raise ValueError("flow_matrix and density_matrix must have matching shapes.")

    _, num_space_bins = flow_matrix.shape
    flagged_bins = []
    diagnostics = []

    for space_bin in range(1, num_space_bins - 1):
        flow_prev = flow_matrix[:, space_bin - 1]
        flow_curr = flow_matrix[:, space_bin]
        flow_next = flow_matrix[:, space_bin + 1]

        density_prev = density_matrix[:, space_bin - 1]
        density_curr = density_matrix[:, space_bin]
        density_next = density_matrix[:, space_bin + 1]

        flow_neighbor_mean = 0.5 * (flow_prev + flow_next)
        density_neighbor_mean = 0.5 * (density_prev + density_next)

        flow_valid = flow_neighbor_mean > eps
        density_valid = density_neighbor_mean > eps
        if not flow_valid.any() or not density_valid.any():
            continue

        flow_ratio = np.full_like(flow_curr, np.nan, dtype=float)
        density_ratio = np.full_like(density_curr, np.nan, dtype=float)
        flow_ratio[flow_valid] = flow_curr[flow_valid] / flow_neighbor_mean[flow_valid]
        density_ratio[density_valid] = (
            density_curr[density_valid] / density_neighbor_mean[density_valid]
        )

        flow_neighbor_gap = np.abs(flow_prev - flow_next) / np.maximum(
            flow_neighbor_mean, eps
        )
        density_neighbor_gap = np.abs(density_prev - density_next) / np.maximum(
            density_neighbor_mean, eps
        )

        flow_low_fraction = np.mean(flow_ratio[flow_valid] <= ratio_threshold)
        density_low_fraction = np.mean(density_ratio[density_valid] <= ratio_threshold)
        flow_neighbor_gap_median = float(np.nanmedian(flow_neighbor_gap[flow_valid]))
        density_neighbor_gap_median = float(
            np.nanmedian(density_neighbor_gap[density_valid])
        )

        is_flagged = (
            flow_low_fraction >= min_persistent_fraction
            and density_low_fraction >= min_persistent_fraction
            and flow_neighbor_gap_median <= neighbor_similarity_threshold
            and density_neighbor_gap_median <= neighbor_similarity_threshold
        )

        if is_flagged:
            flagged_bins.append(space_bin)

        diagnostics.append(
            {
                "space_bin": int(space_bin),
                "flow_low_fraction": float(flow_low_fraction),
                "density_low_fraction": float(density_low_fraction),
                "flow_median_ratio": float(np.nanmedian(flow_ratio[flow_valid])),
                "density_median_ratio": float(np.nanmedian(density_ratio[density_valid])),
                "flow_neighbor_gap_median": flow_neighbor_gap_median,
                "density_neighbor_gap_median": density_neighbor_gap_median,
                "flagged": bool(is_flagged),
            }
        )

    return flagged_bins, diagnostics


def _interpolate_bad_spatial_segments(matrix, bad_space_bins):
    """
    Replace flagged spatial bins using adjacent spatial neighbors plus diagonal
    spatiotemporal support from those adjacent rows.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D with shape (time, space).")

    source = matrix.copy()
    corrected = matrix.copy()
    num_time_bins, num_space_bins = source.shape

    for space_bin in sorted(set(int(idx) for idx in bad_space_bins)):
        if space_bin <= 0 or space_bin >= num_space_bins - 1:
            continue

        for time_bin in range(num_time_bins):
            spatial_neighbors = [
                source[time_bin, space_bin - 1],
                source[time_bin, space_bin + 1],
            ]

            diagonal_neighbors = []
            if time_bin > 0:
                diagonal_neighbors.extend(
                    [
                        source[time_bin - 1, space_bin - 1],
                        source[time_bin - 1, space_bin + 1],
                    ]
                )
            if time_bin < num_time_bins - 1:
                diagonal_neighbors.extend(
                    [
                        source[time_bin + 1, space_bin - 1],
                        source[time_bin + 1, space_bin + 1],
                    ]
                )

            values = spatial_neighbors.copy()
            if diagonal_neighbors:
                values.append(float(np.mean(diagonal_neighbors)))

            corrected[time_bin, space_bin] = float(np.mean(values))

    return corrected


def _interpolate_rows_in_space_time_matrix(matrix, rows_to_interpolate):
    """
    Interpolate specific spatial rows in a `(space, time)` matrix.

    Uses the same scheme as `_interpolate_bad_spatial_segments`: immediate
    spatial neighbors plus diagonal spatiotemporal support from those adjacent
    rows, while avoiding temporal neighbors from the bad row itself.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D with shape (space, time).")

    repaired_time_space = _interpolate_bad_spatial_segments(
        matrix.T, rows_to_interpolate
    )
    return repaired_time_space.T

def load_trajectories_for_days(
    day_names,
    trajectory_timeframe=pd.Timedelta(minutes=10),
    min_time=None,
    max_time=None,
    direction_str="west",
    data_dir="data/i24",
):
    frames = []
    trajectory_offset = 0
    for day_name in day_names:
        df = load_trajectories(
            day_name,
            trajectory_timeframe=trajectory_timeframe,
            min_time=min_time,
            max_time=max_time,
            direction_str=direction_str,
            data_dir=data_dir,
        ).copy()
        if not df.empty:
            df["trajectory_id"] += trajectory_offset
            trajectory_offset = int(df["trajectory_id"].max()) + 1
        df["source_day"] = os.path.splitext(os.path.basename(str(day_name)))[0]
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "trajectory_id",
                "timestamp",
                "x_position",
                "y_position",
                "source_day",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def compute_flow_density_matrices_from_trajectories(
    df,
    x_increases_in_travel_direction,
    time_interval=pd.Timedelta(minutes=1),
    space_interval=100,
    fill_value=0.0,
    interpolate_bad_segments=True,
    bad_segment_ratio_threshold=0.6,
    bad_segment_min_persistent_fraction=0.6,
    bad_segment_neighbor_similarity_threshold=0.3,
):
    """
    Compute flow and density matrices from a trajectory dataframe without
    writing CSVs or histogram images.

    Returns matrices with shape `(num_time_bins, num_space_bins)` and metadata
    describing the inferred extents and bins.

    When `interpolate_bad_segments` is enabled, interior spatial bins that are
    persistently suppressed relative to both neighbors in both flow and density
    are flagged and replaced using adjacent spatial rows plus diagonal
    spatiotemporal support.
    """
    if not isinstance(x_increases_in_travel_direction, bool):
        raise ValueError("x_increases_in_travel_direction must be passed as a bool.")

    required_columns = ["trajectory_id", "timestamp", "x_position"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    working = df.copy()
    if working.empty:
        raise ValueError("Input trajectory dataframe is empty.")

    t_min = working["timestamp"].min()
    t_max = working["timestamp"].max()
    x_min = working["x_position"].min()
    x_max = working["x_position"].max()
    if x_min == x_max:
        raise ValueError(
            "x_min and x_max are identical, meaning no variation in x_position."
        )

    time_bins = pd.date_range(start=t_min, end=t_max, freq=time_interval)
    space_bins, cut_bins = _build_space_bins_in_travel_direction(
        x_min,
        x_max,
        space_interval,
        x_increases_in_travel_direction,
    )
    x_start = float(space_bins[0])
    x_end = float(space_bins[-1])

    if len(time_bins) < 2:
        raise ValueError("Not enough time range for the requested time_interval.")
    if len(space_bins) < 2:
        raise ValueError(
            "space_bins array is empty or too small; adjust space_interval."
        )

    working["time_bin"] = pd.cut(
        working["timestamp"], bins=time_bins, labels=False, include_lowest=True
    )
    working["space_bin_raw"] = pd.cut(
        working["x_position"], bins=cut_bins, labels=False, include_lowest=True
    )
    working = working.dropna(subset=["time_bin", "space_bin_raw"]).astype(
        {"time_bin": int, "space_bin_raw": int}
    )
    num_space_bins = len(space_bins) - 1
    if x_increases_in_travel_direction:
        working["space_bin"] = working["space_bin_raw"].astype(int)
    else:
        working["space_bin"] = (
            num_space_bins - 1 - working["space_bin_raw"]
        ).astype(int)

    flow_matrix = np.full(
        (len(time_bins) - 1, num_space_bins), fill_value, dtype=float
    )
    density_matrix = np.full_like(flow_matrix, fill_value)

    grouped = working.groupby(["time_bin", "space_bin"])
    area_bin = (space_interval / 1000.0) * time_interval.total_seconds() / 3600.0

    for (time_bin, space_bin), group in grouped:
        traj_group = group.groupby("trajectory_id")
        total_distance = sum(
            traj_group["x_position"].apply(lambda x: x.max() - x.min())
        )
        total_time = sum(
            traj_group["timestamp"].apply(lambda x: (x.max() - x.min()).total_seconds())
        )
        flow_matrix[time_bin, space_bin] = (total_distance / 1000.0) / area_bin
        density_matrix[time_bin, space_bin] = (total_time / 3600.0) / area_bin

    bad_space_bins = []
    bad_segment_diagnostics = []
    if interpolate_bad_segments and num_space_bins >= 3:
        bad_space_bins, bad_segment_diagnostics = _detect_bad_spatial_segments(
            flow_matrix,
            density_matrix,
            ratio_threshold=bad_segment_ratio_threshold,
            min_persistent_fraction=bad_segment_min_persistent_fraction,
            neighbor_similarity_threshold=bad_segment_neighbor_similarity_threshold,
        )
        if bad_space_bins:
            flow_matrix = _interpolate_bad_spatial_segments(flow_matrix, bad_space_bins)
            density_matrix = _interpolate_bad_spatial_segments(
                density_matrix, bad_space_bins
            )

    metadata = {
        "t_min": t_min,
        "t_max": t_max,
        "x_min": x_min,
        "x_max": x_max,
        "x_start": x_start,
        "x_end": x_end,
        "x_increases_in_travel_direction": x_increases_in_travel_direction,
        "time_bins": time_bins,
        "space_bins": space_bins,
        "interpolated_bad_space_bins": bad_space_bins,
        "bad_segment_diagnostics": bad_segment_diagnostics,
    }
    return flow_matrix, density_matrix, metadata


def _format_timedelta_label(delta):
    total_seconds = int(pd.Timedelta(delta).total_seconds())
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}min"
    return f"{total_seconds}s"


def _parse_timedelta_label(label):
    if label.endswith("h"):
        return pd.Timedelta(hours=int(label[:-1]))
    if label.endswith("min"):
        return pd.Timedelta(minutes=int(label[:-3]))
    if label.endswith("s"):
        return pd.Timedelta(seconds=int(label[:-1]))
    raise ValueError(f"Unsupported timedelta label '{label}'.")


_I24_MATRIX_SWEEP_PATTERN = re.compile(
    r"^(?P<day_name>[^_]+)_(?P<direction>[^_]+)_(?P<start>\d{4})_(?P<end>\d{4})"
    r"_dt_(?P<dt>[^_]+)_dx_(?P<dx>\d+)m_"
    r"(?P<metric>flow_per_lane|flow|density|velocity)\.npy$"
)

def _parse_i24_matrix_sweep_path(path):
    match = _I24_MATRIX_SWEEP_PATTERN.match(Path(path).name)
    if match is None:
        return None

    parsed = match.groupdict()
    parsed["dx"] = int(parsed["dx"])
    parsed["path"] = str(path)
    return parsed

def save_i24_hourly_trajectory_batches(
    day_windows,
    batch_hours=1,
    direction_str="west",
    data_dir="data/i24",
    output_dir="data/i24/parsed_trajectories",
):
    """
    Parse I-24 trajectories in hourly batches and save each batch to disk.

    Parameters
    ----------
    day_windows : dict[str, iterable]
        Mapping from day key to iterable batch start times.
    """
    batch_duration = pd.Timedelta(hours=batch_hours)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for day_name, start_times in day_windows.items():
        for start_time in start_times:
            batch_start = pd.Timestamp(start_time)
            batch_end = batch_start + batch_duration
            batch_end_inclusive = batch_end - pd.Timedelta(microseconds=1)

            trajectories = load_trajectories(
                day_name,
                trajectory_timeframe=batch_duration,
                min_time=batch_start,
                max_time=batch_end_inclusive,
                direction_str=direction_str,
                data_dir=data_dir,
            )

            batch_label = f"{batch_start:%H%M}_{batch_end:%H%M}"
            output_path = (
                output_dir / f"{day_name}_{direction_str}_{batch_label}_trajectories.csv.gz"
            )
            trajectories.to_csv(output_path, index=False)

            manifest_rows.append(
                {
                    "day_name": day_name,
                    "direction": direction_str,
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "trajectory_path": str(output_path),
                    "n_rows": int(len(trajectories)),
                    "n_trajectories": int(
                        trajectories["trajectory_id"].nunique()
                        if "trajectory_id" in trajectories.columns
                        else 0
                    ),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "trajectory_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest

def combine_i24_daily_matrix_sweeps(
    input_dir="data/i24/matrix_sweeps",
    output_dir=None,
    require_consecutive_hours=True,
):
    """
    Combine hourly I-24 sweep matrices into daily matrices.

    Hourly sweep files are expected to match the naming scheme created by
    `sweep_i24_flow_density_matrices`. Inputs are loaded in chronological
    order, concatenated along the time axis, then transposed before saving so
    the combined output uses shape `(space, time)`.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir / "daily_combined"
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_rows = []
    for path in sorted(input_dir.glob("*.npy")):
        parsed = _parse_i24_matrix_sweep_path(path)
        if parsed is None:
            continue
        parsed_rows.append(parsed)

    if not parsed_rows:
        raise ValueError(f"No sweep matrices matching the expected pattern were found in '{input_dir}'.")

    manifest_df = pd.DataFrame(parsed_rows)
    saved_rows = []

    group_columns = ["day_name", "direction", "dt", "dx", "metric"]
    for keys, group in manifest_df.groupby(group_columns, sort=True):
        day_name, direction, dt_label, dx_meters, metric = keys
        ordered = group.sort_values(["start", "end"]).reset_index(drop=True)

        if require_consecutive_hours:
            for idx in range(len(ordered) - 1):
                if ordered.loc[idx, "end"] != ordered.loc[idx + 1, "start"]:
                    raise ValueError(
                        "Encountered a gap or overlap while combining "
                        f"{day_name} {direction} dt={dt_label} dx={dx_meters}m {metric}: "
                        f"{ordered.loc[idx, 'end']} followed by {ordered.loc[idx + 1, 'start']}."
                    )

        hourly_matrices = []
        expected_space_bins = None
        for _, row in ordered.iterrows():
            matrix = np.load(row["path"])
            if matrix.ndim != 2:
                raise ValueError(f"Expected a 2D matrix in '{row['path']}', got shape {matrix.shape}.")

            if expected_space_bins is None:
                expected_space_bins = matrix.shape[1]
            elif matrix.shape[1] != expected_space_bins:
                raise ValueError(
                    f"Space-bin mismatch while combining '{day_name}' {metric} matrices: "
                    f"expected {expected_space_bins}, got {matrix.shape[1]} in '{row['path']}'."
                )

            hourly_matrices.append(matrix)

        combined_time_space = np.concatenate(hourly_matrices, axis=0)
        combined_space_time = combined_time_space.T

        start_label = ordered.loc[0, "start"]
        end_label = ordered.loc[len(ordered) - 1, "end"]
        output_name = (
            f"{day_name}_{direction}_{start_label}_{end_label}"
            f"_dt_{dt_label}_dx_{dx_meters}m_{metric}.npy"
        )
        output_path = output_dir / output_name
        np.save(output_path, combined_space_time)

        saved_rows.append(
            {
                "day_name": day_name,
                "direction": direction,
                "dt": dt_label,
                "dx_meters": dx_meters,
                "metric": metric,
                "hour_count": int(len(hourly_matrices)),
                "time_bins": int(combined_space_time.shape[1]),
                "space_bins": int(combined_space_time.shape[0]),
                "input_paths": "|".join(ordered["path"].tolist()),
                "output_path": str(output_path),
            }
        )

    combined_manifest = pd.DataFrame(saved_rows)
    combined_manifest_path = output_dir / "daily_combined_manifest.csv"
    combined_manifest.to_csv(combined_manifest_path, index=False)
    return combined_manifest


def repair_i24_daily_combined_matrices(
    input_dir="data/i24/matrix_sweeps/daily_combined",
    output_dir=None,
    drop_first_row=True,
    bad_rows_by_dx=None,
    trim_initial_time=pd.Timedelta(minutes=30),
):
    """
    Repair already-saved daily combined I-24 matrices.

    Expected input matrices use shape `(space, time)`.

    Processing order:
    1. Drop row 0, if requested.
    2. Interpolate configured bad rows on the resulting matrix.
    3. Remove the first `trim_initial_time` worth of time columns.

    Row indices in `bad_rows_by_dx` are interpreted in the original matrix
    indexing, before row 0 is removed.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir / "repaired"
    output_dir.mkdir(parents=True, exist_ok=True)

    if bad_rows_by_dx is None:
        bad_rows_by_dx = {
            200: [13],
            400: [7],
            600: [],
            800: [],
        }

    saved_rows = []
    for path in sorted(input_dir.glob("*.npy")):
        parsed = _parse_i24_matrix_sweep_path(path)
        if parsed is None:
            continue

        matrix = np.load(path)
        if matrix.ndim != 2:
            raise ValueError(f"Expected a 2D matrix in '{path}', got shape {matrix.shape}.")

        repaired = matrix.copy()
        removed_rows = []
        if drop_first_row:
            if repaired.shape[0] < 2:
                raise ValueError(f"Cannot drop row 0 from '{path}' with shape {repaired.shape}.")
            repaired = repaired[1:, :]
            removed_rows.append(0)

        dx_meters = int(parsed["dx"])
        requested_rows = []
        for row in bad_rows_by_dx.get(dx_meters, []):
            row = int(row)
            if drop_first_row:
                if row == 0:
                    continue
                row -= 1
            requested_rows.append(row)
        interpolated_rows = [
            row for row in requested_rows if 0 < row < repaired.shape[0] - 1
        ]
        if interpolated_rows:
            repaired = _interpolate_rows_in_space_time_matrix(repaired, interpolated_rows)

        dt_delta = _parse_timedelta_label(parsed["dt"])
        trim_initial_columns = int(trim_initial_time / dt_delta)
        if trim_initial_columns > 0:
            if trim_initial_columns >= repaired.shape[1]:
                raise ValueError(
                    f"Cannot trim {trim_initial_columns} time columns from '{path}' "
                    f"with shape {repaired.shape}."
                )
            repaired = repaired[:, trim_initial_columns:]

        output_path = output_dir / path.name
        np.save(output_path, repaired)

        saved_rows.append(
            {
                "source_path": str(path),
                "output_path": str(output_path),
                "metric": parsed["metric"],
                "dx_meters": dx_meters,
                "original_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
                "repaired_shape": f"{repaired.shape[0]}x{repaired.shape[1]}",
                "removed_rows": "|".join(str(row) for row in removed_rows),
                "interpolated_rows": "|".join(str(row) for row in interpolated_rows),
                "trimmed_initial_time_minutes": float(pd.Timedelta(trim_initial_time).total_seconds() / 60.0),
                "trimmed_initial_columns": int(trim_initial_columns),
            }
        )

    repaired_manifest = pd.DataFrame(saved_rows)
    repaired_manifest_path = output_dir / "repaired_manifest.csv"
    repaired_manifest.to_csv(repaired_manifest_path, index=False)
    return repaired_manifest


def save_repaired_i24_velocity_matrices(
    input_dir="data/i24/matrix_sweeps/daily_combined_repaired",
    output_dir=None,
):
    """
    Build repaired velocity matrices as `flow / density`.

    Outputs are saved alongside the repaired matrices using the same naming
    scheme, with metric suffix `velocity.npy`.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_rows = []
    for flow_path in sorted(input_dir.glob("*_flow.npy")):
        parsed = _parse_i24_matrix_sweep_path(flow_path)
        if parsed is None or parsed["metric"] != "flow":
            continue

        density_name = flow_path.name.replace("_flow.npy", "_density.npy")
        density_path = input_dir / density_name
        if not density_path.exists():
            raise FileNotFoundError(
                f"Missing density matrix for flow matrix '{flow_path}'. "
                f"Expected '{density_path}'."
            )

        flow = np.load(flow_path).astype(float, copy=False)
        density = np.load(density_path).astype(float, copy=False)
        if flow.shape != density.shape:
            raise ValueError(
                f"Flow/density shape mismatch for '{flow_path.name}': "
                f"{flow.shape} vs {density.shape}."
            )

        velocity = np.full(flow.shape, np.nan, dtype=float)
        valid = (~np.isnan(flow)) & (~np.isnan(density)) & (density != 0)
        np.divide(flow, density, out=velocity, where=valid)

        output_name = flow_path.name.replace("_flow.npy", "_velocity.npy")
        output_path = output_dir / output_name
        np.save(output_path, velocity)

        saved_rows.append(
            {
                "flow_path": str(flow_path),
                "density_path": str(density_path),
                "velocity_path": str(output_path),
                "dx_meters": int(parsed["dx"]),
                "dt": parsed["dt"],
                "shape": f"{velocity.shape[0]}x{velocity.shape[1]}",
            }
        )

    if not saved_rows:
        raise ValueError(f"No repaired flow matrices were found in '{input_dir}'.")

    manifest = pd.DataFrame(saved_rows)
    manifest_path = output_dir / "velocity_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest


def save_repaired_i24_flow_per_lane_matrices(
    input_dir="data/i24/matrix_sweeps/daily_combined_repaired",
    lane_mapping_dir="data/i24/segment_mappings",
    output_dir=None,
):
    """
    Normalize repaired flow matrices by lane count for each spatial row.

    Each row `i` is divided by the scalar lane count from
    `lane_mapping_dx_<resolution>m.npy` at index `i`. Outputs are saved
    alongside the repaired matrices with metric suffix `flow_per_lane.npy`.
    """
    input_dir = Path(input_dir)
    lane_mapping_dir = Path(lane_mapping_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_rows = []
    lane_cache = {}

    for flow_path in sorted(input_dir.glob("*_flow.npy")):
        parsed = _parse_i24_matrix_sweep_path(flow_path)
        if parsed is None or parsed["metric"] != "flow":
            continue

        dx_meters = int(parsed["dx"])
        if dx_meters not in lane_cache:
            lane_path = lane_mapping_dir / f"lane_mapping_dx_{dx_meters}m.npy"
            if not lane_path.exists():
                raise FileNotFoundError(
                    f"Missing lane mapping for {dx_meters}m resolution: '{lane_path}'."
                )
            lane_cache[dx_meters] = np.load(lane_path).astype(float, copy=False)

        flow = np.load(flow_path).astype(float, copy=False)
        lane_mapping = lane_cache[dx_meters]
        if flow.ndim != 2:
            raise ValueError(f"Expected 2D flow matrix in '{flow_path}', got {flow.shape}.")
        if flow.shape[0] != lane_mapping.shape[0]:
            raise ValueError(
                f"Lane mapping length mismatch for '{flow_path.name}': "
                f"matrix has {flow.shape[0]} rows, lane mapping has {lane_mapping.shape[0]}."
            )

        lane_counts = lane_mapping[:, np.newaxis]
        flow_per_lane = np.full(flow.shape, np.nan, dtype=float)
        valid = (~np.isnan(lane_counts)) & (lane_counts != 0)
        np.divide(flow, lane_counts, out=flow_per_lane, where=valid)

        output_name = flow_path.name.replace("_flow.npy", "_flow_per_lane.npy")
        output_path = output_dir / output_name
        np.save(output_path, flow_per_lane)

        saved_rows.append(
            {
                "flow_path": str(flow_path),
                "lane_mapping_path": str(
                    lane_mapping_dir / f"lane_mapping_dx_{dx_meters}m.npy"
                ),
                "flow_per_lane_path": str(output_path),
                "dx_meters": dx_meters,
                "dt": parsed["dt"],
                "shape": f"{flow_per_lane.shape[0]}x{flow_per_lane.shape[1]}",
            }
        )

    if not saved_rows:
        raise ValueError(f"No repaired flow matrices were found in '{input_dir}'.")

    manifest = pd.DataFrame(saved_rows)
    manifest_path = output_dir / "flow_per_lane_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest


def print_i24_space_bin_miles(
    matrix_dir,
    min_postmile=58.8,
    output_dir=None,
):
    """
    Print legacy I-24 space-bin edges and centers in miles for each spatial
    resolution found in `matrix_dir`.

    This matches the pre-patch binning convention: start from the minimum
    postmile and step forward uniformly by `dx`, without re-deriving bins from
    the newer travel-direction logic.
    """
    matrix_dir = Path(matrix_dir)
    output_dir = Path(output_dir) if output_dir is not None else matrix_dir / "space_bin_csvs"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_dx = {}
    for path in sorted(matrix_dir.glob("*.npy")):
        parsed = _parse_i24_matrix_sweep_path(path)
        if parsed is None:
            continue

        dx_meters = int(parsed["dx"])
        matrix = np.load(path, mmap_mode="r")
        num_space_bins = int(matrix.shape[0])
        per_dx.setdefault(dx_meters, num_space_bins)

    if not per_dx:
        raise ValueError(f"No matrix files matching the expected pattern were found in '{matrix_dir}'.")

    saved_rows = []
    for dx_meters in sorted(per_dx):
        num_space_bins = per_dx[dx_meters]
        dx_miles = dx_meters / 1609.34
        edges = min_postmile + dx_miles * np.arange(num_space_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        bin_df = pd.DataFrame(
            {
                "space_bin": np.arange(num_space_bins, dtype=int),
                "left_edge_miles": edges[:-1],
                "center_miles": centers,
                "right_edge_miles": edges[1:],
                "dx_meters": dx_meters,
                "dx_miles": dx_miles,
            }
        )
        csv_path = output_dir / f"space_bins_dx_{dx_meters}m.csv"
        bin_df.to_csv(csv_path, index=False)
        saved_rows.append(
            {
                "dx_meters": dx_meters,
                "dx_miles": dx_miles,
                "num_space_bins": num_space_bins,
                "csv_path": str(csv_path),
            }
        )

        print(f"\ndx = {dx_meters} m ({dx_miles:.6f} mi), bins = {num_space_bins}")
        print("edges_miles =")
        print(np.round(edges, 6).tolist())
        print("centers_miles =")
        print(np.round(centers, 6).tolist())

    manifest_df = pd.DataFrame(saved_rows)
    manifest_path = output_dir / "space_bin_csv_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_df

def sweep_i24_flow_density_matrices(
    trajectory_manifest,
    x_increases_in_travel_direction,
    time_intervals,
    space_intervals,
    output_dir="data/i24/matrix_sweeps",
):
    """
    Build flow and density matrices for each saved trajectory batch across a
    time/space sweep, and save them as `.npy` files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(trajectory_manifest, (str, os.PathLike, Path)):
        manifest_df = pd.read_csv(
            trajectory_manifest, parse_dates=["batch_start", "batch_end"]
        )
    else:
        manifest_df = trajectory_manifest.copy()

    saved_rows = []
    for _, batch in manifest_df.iterrows():
        trajectory_path = Path(batch["trajectory_path"])
        trajectories = pd.read_csv(trajectory_path, parse_dates=["timestamp"])
        if trajectories.empty:
            continue

        batch_start = pd.Timestamp(batch["batch_start"])
        batch_end = pd.Timestamp(batch["batch_end"])
        batch_label = f"{batch_start:%H%M}_{batch_end:%H%M}"

        for time_interval in time_intervals:
            time_delta = pd.Timedelta(minutes=time_interval)
            time_label = _format_timedelta_label(time_delta)

            for space_interval in space_intervals:
                flow_matrix, density_matrix, _ = compute_flow_density_matrices_from_trajectories(
                    trajectories,
                    x_increases_in_travel_direction=x_increases_in_travel_direction,
                    time_interval=time_delta,
                    space_interval=space_interval,
                    fill_value=0.0,
                )

                stem = (
                    f"{batch['day_name']}_{batch['direction']}_{batch_label}"
                    f"_dt_{time_label}_dx_{int(space_interval)}m"
                )
                flow_path = output_dir / f"{stem}_flow.npy"
                density_path = output_dir / f"{stem}_density.npy"
                np.save(flow_path, flow_matrix)
                np.save(density_path, density_matrix)

                saved_rows.append(
                    {
                        "day_name": batch["day_name"],
                        "direction": batch["direction"],
                        "batch_start": batch_start,
                        "batch_end": batch_end,
                        "time_interval_minutes": time_interval,
                        "space_interval_meters": space_interval,
                        "flow_path": str(flow_path),
                        "density_path": str(density_path),
                    }
                )

    saved_manifest = pd.DataFrame(saved_rows)
    saved_manifest_path = output_dir / "matrix_sweep_manifest.csv"
    saved_manifest.to_csv(saved_manifest_path, index=False)
    return saved_manifest


def summarize_sparsity(
    frame: pd.DataFrame,
    label: str,
    mode: str = "by_station",
    group_col: str = "id",
    id_col: Optional[str] = None,
    top_n: int = 25,
    figsize: Tuple[int, int] = (10, 4),
) -> pd.DataFrame:
    """
    Unified sparsity summary utility with plotting + clean tabular output.

    Modes:
    - by_station: null percentage by column within each group_col value.
    - column_nulls: null counts and percentages by column.
    - lane_all_null_by_id: rows where all lane columns are null, grouped by station ID.
    """
    if frame.empty:
        print(f"{label}: frame is empty.")
        return pd.DataFrame()

    def _finalize_plot(title: str, xlabel: str, ylabel: str) -> None:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    if mode == "by_station":
        if group_col not in frame.columns:
            raise ValueError(f"Group column '{group_col}' not found in frame.")

        cols_to_score = [c for c in frame.columns if c != group_col]
        if not cols_to_score:
            raise ValueError(
                f"No columns available to summarize after excluding group column '{group_col}'."
            )

        summary = (
            frame.groupby(group_col)[cols_to_score]
            .apply(lambda x: x.isnull().mean())
            .reset_index()
            .melt(id_vars=[group_col], var_name="column", value_name="percent_null")
            .sort_values([group_col, "percent_null"], ascending=[True, False])
            .reset_index(drop=True)
        )
        summary["percent_null"] = summary["percent_null"].round(3)

        print(f"{label} null summary (by {group_col}):")
        print(f"Total rows: {len(frame):,}")

        heatmap_df = summary.pivot(index=group_col, columns="column", values="percent_null")
        dynamic_height = max(figsize[1], 0.35 * len(heatmap_df.index) + 2)
        plt.figure(figsize=(figsize[0], dynamic_height))
        plt.imshow(heatmap_df.values, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        plt.colorbar(label="Percent Null")
        plt.xticks(np.arange(len(heatmap_df.columns)), heatmap_df.columns, rotation=45, ha="right")
        plt.yticks(np.arange(len(heatmap_df.index)), heatmap_df.index.astype(str))
        _finalize_plot(
            title=f"{label}: Null Share by {group_col} and column",
            xlabel="Column",
            ylabel=group_col,
        )
        return summary

    if mode == "column_nulls":
        null_count = frame.isnull().sum()
        summary = pd.DataFrame(
            {
                "column": null_count.index,
                "null_count": null_count.values.astype(int),
            }
        )
        summary["percent_null"] = (summary["null_count"] / len(frame)).round(3)
        summary = summary[summary["null_count"] > 0].sort_values(
            "null_count", ascending=False
        )

        print(f"{label} sensor readings: {len(frame):,}")

        if not summary.empty:
            plot_df = summary.head(top_n)
            plt.figure(figsize=figsize)
            plt.bar(plot_df["column"], plot_df["percent_null"])
            plt.xticks(rotation=45, ha="right")
            _finalize_plot(
                title=f"{label}: Null Share by Column (top {len(plot_df)})",
                xlabel="Column",
                ylabel="Percent Null",
            )
        return summary.reset_index(drop=True)

    if mode == "lane_all_null_by_id":
        lane_cols = [col for col in frame.columns if "lane" in col.lower()]
        if not lane_cols:
            print(f"{label}: no columns containing 'lane' found.")
            return pd.DataFrame()

        resolved_id_col = id_col or ("ID" if "ID" in frame.columns else "id")
        if resolved_id_col not in frame.columns:
            raise ValueError(
                f"ID column '{resolved_id_col}' not found. Pass id_col explicitly."
            )

        all_lane_null = frame[lane_cols].isna().all(axis=1)
        total_rows = frame.groupby(resolved_id_col).size().rename("total_rows")
        null_rows = (
            frame.loc[all_lane_null]
            .groupby(resolved_id_col)
            .size()
            .rename("all_lane_null_rows")
        )
        summary = (
            total_rows.to_frame()
            .join(null_rows, how="left")
            .fillna(0)
            .astype({"all_lane_null_rows": int})
            .reset_index()
        )
        summary["pct_all_lane_null"] = (
            summary["all_lane_null_rows"] / summary["total_rows"]
        ).round(3)
        summary = summary.sort_values("all_lane_null_rows", ascending=False)

        print(f"{label} lane-null summary (per {resolved_id_col}):")
        print(f"Total rows: {len(frame):,}")

        plot_df = summary.head(top_n)
        plt.figure(figsize=figsize)
        plt.bar(plot_df[resolved_id_col].astype(str), plot_df["pct_all_lane_null"])
        plt.xticks(rotation=45, ha="right")
        _finalize_plot(
            title=f"{label}: All-Lane-Null Share by {resolved_id_col} (top {len(plot_df)})",
            xlabel=resolved_id_col,
            ylabel="Percent All-Lane-Null",
        )
        return summary.reset_index(drop=True)

    raise ValueError(
        "Unsupported mode. Use one of: 'by_station', 'column_nulls', 'lane_all_null_by_id'."
    )

def increase_resolution(matrix: np.ndarray, space_factor, time_factor) -> np.ndarray:
    return np.repeat(matrix, space_factor, axis=0).repeat(time_factor, axis=1)

def subdivide_space_bins(space_bins, factor):
    """
    Split each interval in `space_bins` into `factor` equal parts.

    Example: space_bins=[83, 85] with factor=4 → [83, 83.5, 84, 84.5, 85]
    """
    if not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer")
    edges = np.asarray(space_bins, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("space_bins must be a 1D sequence with ≥2 edges")

    refined = [edges[0]]
    for start, end in zip(edges[:-1], edges[1:]):
        segment_edges = np.linspace(start, end, factor + 1)[1:]
        refined.extend(segment_edges.tolist())
    return np.array(refined)

def df_to_matrix(df, time_column, space_column, value_column):
    """Convert long-form data to a matrix with shape (space, time)."""
    required = {time_column, space_column, value_column}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    t = df[time_column].to_numpy()
    s = df[space_column].to_numpy()

    # Enforce integer index columns
    if not np.all(np.equal(t, t.astype(int))):
        raise ValueError(f"{time_column} must contain integer indices")
    if not np.all(np.equal(s, s.astype(int))):
        raise ValueError(f"{space_column} must contain integer indices")

    t_idx = t.astype(int)
    s_idx = s.astype(int)
    values = df[value_column].to_numpy()

    num_time_bins = t_idx.max() + 1
    num_space_bins = s_idx.max() + 1

    # rows = space, columns = time
    matrix = np.full((num_space_bins, num_time_bins), np.nan)
    matrix[s_idx, t_idx] = values

    return matrix

def average_neighbors_y(matrix, num_neighbors=3):
    smoothed = matrix.astype(float, copy=True)
    weight_template = np.arange(num_neighbors + 1, 0, -1)
    full_weights = np.concatenate([weight_template[1:], weight_template])
    for i in range(num_neighbors, matrix.shape[0] - num_neighbors):
        for j in range(matrix.shape[1]):
            window = matrix[i - num_neighbors : i + num_neighbors + 1, j]
            valid_mask = ~np.isnan(window)
            if not valid_mask.any():
                smoothed[i, j] = np.nan
                continue

            smoothed[i, j] = np.average(
                window[valid_mask],
                weights=full_weights[valid_mask],
            )
    return smoothed

def process_pems(
    df,
    time_col,
    postmile_col,
    value_col,
    start_pm,
    end_pm,
    time_interval,
    space_interval,
    t_min = None,
    t_max = None
):
    """
    Bin values into a time-space matrix.

    Output orientation:
    - rows: space (bottom row corresponds to start_pm)
    - cols: time (left to right)
    """
    df = df.copy()

    # Validate inputs
    required_columns = [time_col, postmile_col, value_col]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            "Input DataFrame must contain the following columns: {required_columns}"
        )
    if start_pm == end_pm:
        raise ValueError("start_pm and end_pm must differ.")

    # Build time bins
    if t_min is not None and t_max is not None:
        time_bins = pd.date_range(start=t_min, end=t_max, freq=time_interval)
    else:
        time_bins = pd.date_range(
            start=df[time_col].min(),
            end=df[time_col].max(),
            freq=time_interval,
        )

    if len(time_bins) < 2:
        raise ValueError("Not enough time range for the requested time_interval.")

    # Build uniform space bins and use ascending edges for pd.cut
    travel_bins = _build_space_bins(start_pm, end_pm, space_interval)
    cut_bins = np.sort(travel_bins)
    num_space_bins = len(cut_bins) - 1

    # Assign bin indices
    df["time_bin"] = pd.cut(
        df[time_col], bins=time_bins, labels=False, include_lowest=True
    )
    df["space_bin_asc"] = pd.cut(
        df[postmile_col], bins=cut_bins, labels=False, include_lowest=True
    )

    # Drop out-of-range points and cast bin indices
    df = df.dropna(subset=["time_bin", "space_bin_asc"]).astype(
        {"time_bin": int, "space_bin_asc": int}
    )

    # Row mapping so TOP row corresponds to start_pm and BOTTOM row to end_pm
    if start_pm < end_pm:
        # Ascending PM: ascending bins already put start_pm at top (row 0)
        df["space_bin"] = df["space_bin_asc"].astype(int)
    else:
        # Descending PM: reverse so start_pm maps to top row
        df["space_bin"] = (num_space_bins - 1 - df["space_bin_asc"]).astype(int)

    # Build matrix with rows=space, cols=time
    num_time_bins = len(time_bins) - 1
    value_matrix = np.full((num_space_bins, num_time_bins), np.nan)
    grouped = df.groupby(["space_bin", "time_bin"])[value_col].mean()
    for (space_bin, time_bin), value in grouped.items():
        value_matrix[space_bin, time_bin] = value

    # Convert to df
    value_df = matrix_to_df(value_matrix)
    return value_matrix, value_df

def matrix_to_df(value_matrix: np.ndarray) -> pd.DataFrame:
    """
    Convert a value matrix to long format with integer indices.

    Returns columns: space_index, time_index, value
    """
    if value_matrix.ndim != 2:
        raise ValueError("value_matrix must be 2D.")

    num_space_bins, num_time_bins = value_matrix.shape
    space_idx, time_idx = np.meshgrid(
        np.arange(num_space_bins), np.arange(num_time_bins), indexing="ij"
    )
    return pd.DataFrame(
        {
            "space_index": space_idx.ravel(),
            "time_index": time_idx.ravel(),
            "value": value_matrix.ravel(),
        }
    )

def plot_matrix(
    matrix,
    title,
    colorbar_label=None,
    colorbar_range=None,
    t_min=None,
    t_max=None,
    start_pm=None,
    end_pm=None,
):
    """
    Plot a single heatmap with optional inferred time/space resolution.
    """
    fig, ax = plt.subplots()
    _plot_matrix_on_ax(
        ax=ax,
        matrix=matrix,
        title=title,
        colorbar_label=colorbar_label,
        colorbar_range=colorbar_range,
        t_min=t_min,
        t_max=t_max,
        start_pm=start_pm,
        end_pm=end_pm,
        add_colorbar=True,
        scale_figure_size=True,
    )
    fig.tight_layout()
    plt.show()
    return fig, ax


def _plot_matrix_on_ax(
    ax,
    matrix,
    title,
    colorbar_label=None,
    colorbar_range=None,
    t_min=None,
    t_max=None,
    start_pm=None,
    end_pm=None,
    add_colorbar=False,
    show_xlabel=True,
    show_ylabel=True,
    title_prefix="Time-Space Diagram for",
    scale_figure_size=False,
    time_resolution_label=None,
    space_resolution_label=None,
):
    if (t_min is None) != (t_max is None):
        raise ValueError("Pass both t_min and t_max, or neither.")
    if (start_pm is None) != (end_pm is None):
        raise ValueError("Pass both start_pm and end_pm, or neither.")

    num_space_bins, num_time_bins = matrix.shape
    time_ticks = np.linspace(0, num_time_bins - 1, min(10, num_time_bins)).astype(int)
    space_ticks = np.linspace(0, num_space_bins - 1, min(10, num_space_bins)).astype(int)

    colorbar_label = colorbar_label or title

    if colorbar_range is not None:
        if len(colorbar_range) != 2:
            raise ValueError("colorbar_range must be a two-value (vmin, vmax) tuple.")
        vmin, vmax = colorbar_range
        if vmin >= vmax:
            raise ValueError("colorbar_range must satisfy vmin < vmax.")
    else:
        vmin = vmax = None

    inferred_time_increment = None
    inferred_space_increment = None

    if t_min is not None:
        if t_max <= t_min:
            raise ValueError("t_max must be greater than t_min to infer time spacing.")
        inferred_time_increment = (t_max - t_min) / num_time_bins

    if start_pm is not None:
        if start_pm == end_pm:
            raise ValueError("start_pm must differ from end_pm to infer spacing.")
        inferred_space_increment = (end_pm - start_pm) / num_space_bins
        space_edges = start_pm + np.arange(num_space_bins + 1) * inferred_space_increment
    else:
        space_edges = None

    # Scale figure dimensions with actual axis ranges when available.
    if inferred_time_increment is not None:
        time_range_hours = max((t_max - t_min).total_seconds() / 3600.0, 1e-6)
    else:
        time_range_hours = max(num_time_bins / 12.0, 1e-6)

    if space_edges is not None:
        space_extent_miles = abs(space_edges[-1] - space_edges[0])
        space_range_miles = max(space_extent_miles, 1e-6)
    else:
        space_range_miles = max(num_space_bins / 10.0, 1e-6)

    width_per_hour = 5
    height_per_mile = 1
    fig_width = np.clip(time_range_hours * width_per_hour, 3.0, 20.0)
    fig_height = np.clip(space_range_miles * height_per_mile, 3.0, 20.0)
    if scale_figure_size:
        ax.figure.set_size_inches(fig_width, fig_height, forward=True)

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="lightgray")

    pcm = ax.imshow(
        matrix,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    if add_colorbar:
        cbar = ax.figure.colorbar(pcm, ax=ax)
        cbar.set_label(colorbar_label)

    ax.set_xticks(time_ticks)
    if inferred_time_increment is not None:
        time_tick_times = [t_min + int(i) * inferred_time_increment for i in time_ticks]
        if show_xlabel:
            ax.set_xticklabels([ts.strftime("%H:%M") for ts in time_tick_times], rotation=45)
        else:
            ax.set_xticklabels([])
    else:
        ax.set_xticklabels(time_ticks if show_xlabel else [])
    if show_xlabel:
        ax.set_xlabel("Time" if inferred_time_increment is not None else "Time Index")
    else:
        ax.set_xlabel("")

    ax.set_yticks(space_ticks)
    if inferred_space_increment is not None:
        # Use evenly spaced tick positions, but label them with rounded integer PM values.
        pm_by_row = np.linspace(start_pm, end_pm, num_space_bins)
        space_labels = [int(round(float(pm_by_row[idx]))) for idx in space_ticks]
        ax.set_yticklabels(space_labels if show_ylabel else [])
    else:
        ax.set_yticklabels(space_ticks if show_ylabel else [])
    if show_ylabel:
        ax.set_ylabel("Space (Postmile Abs)" if inferred_space_increment is not None else "Space Index")
    else:
        ax.set_ylabel("")

    full_title = f"{title_prefix} {title}".strip() if title_prefix else title
    title_lines = [full_title]
    resolution_parts = []

    if time_resolution_label is not None:
        resolution_parts.append(time_resolution_label)
    elif inferred_time_increment is not None:
        time_res_sec = int(round(inferred_time_increment.total_seconds()))
        resolution_parts.append(f"{time_res_sec} sec")

    if space_resolution_label is not None:
        resolution_parts.append(space_resolution_label)
    elif inferred_space_increment is not None:
        space_res_miles = abs(round(inferred_space_increment, 2))
        resolution_parts.append(f"{space_res_miles} miles")

    if resolution_parts:
        title_lines.append("Resolution: " + " x ".join(resolution_parts))

    ax.set_title("\n".join(title_lines))
    return pcm


def _load_i24_repaired_matrix(
    day_name,
    metric,
    dt_label,
    dx_meters,
    input_dir="data/i24/matrix_sweeps/daily_combined_repaired",
):
    input_dir = Path(input_dir)
    path = input_dir / (
        f"{day_name}_west_1200_1600_dt_{dt_label}_dx_{dx_meters}m_{metric}.npy"
    )
    if not path.exists():
        raise FileNotFoundError(f"Could not find repaired matrix '{path}'.")
    return np.load(path), path


def _get_i24_plot_bounds(
    dx_meters,
    space_bin_csv_dir="data/i24/matrix_sweeps/daily_combined_repaired/space_bin_csvs",
):
    csv_path = Path(space_bin_csv_dir) / f"space_bins_dx_{dx_meters}m.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find space-bin CSV '{csv_path}'.")

    bins_df = pd.read_csv(csv_path)
    start_pm = float(
        max(
            bins_df["left_edge_miles"].max(),
            bins_df["right_edge_miles"].max(),
        )
    )
    end_pm = float(
        min(
            bins_df["left_edge_miles"].min(),
            bins_df["right_edge_miles"].min(),
        )
    )
    return start_pm, end_pm


def _get_i24_time_bounds(start_label="1200", end_label="1600"):
    base_date = pd.Timestamp("2000-01-01")
    t_min = base_date + pd.Timedelta(hours=int(start_label[:2]), minutes=int(start_label[2:]))
    t_max = base_date + pd.Timedelta(hours=int(end_label[:2]), minutes=int(end_label[2:]))
    return t_min, t_max


def _format_dt_resolution_label(dt_label):
    if dt_label.endswith("min"):
        return f"{int(dt_label[:-3])} min"
    if dt_label.endswith("s"):
        return f"{int(dt_label[:-1])} sec"
    return dt_label


def _format_dx_resolution_label(dx_meters):
    return f"{int(dx_meters)} m"


def _save_or_show_figure(fig, output_path=None, dpi=200):
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_i24_figure_1_speed_and_flow_per_lane_by_day(
    days=("nov21", "nov23", "nov30"),
    dt_label="10s",
    dx_meters=400,
    input_dir="data/i24/matrix_sweeps/daily_combined_repaired",
    space_bin_csv_dir="data/i24/matrix_sweeps/daily_combined_repaired/space_bin_csvs",
    output_path=None,
):
    """Figure 1: speed and flow-per-lane at a fixed resolution for each day."""
    metrics = [
        ("velocity", "Speed", "Speed (mph)"),
        ("flow_per_lane", "Flow Per Lane", "Flow Per Lane"),
    ]
    t_min, t_max = _get_i24_time_bounds(start_label="0630", end_label="1000")
    start_pm, end_pm = _get_i24_plot_bounds(dx_meters, space_bin_csv_dir=space_bin_csv_dir)
    time_resolution_label = _format_dt_resolution_label(dt_label)
    space_resolution_label = _format_dx_resolution_label(dx_meters)

    loaded = {}
    color_ranges = {}
    for metric, _, _ in metrics:
        matrices = []
        for day_name in days:
            matrix, _ = _load_i24_repaired_matrix(
                day_name=day_name,
                metric=metric,
                dt_label=dt_label,
                dx_meters=dx_meters,
                input_dir=input_dir,
            )
            loaded[(day_name, metric)] = matrix
            matrices.append(matrix)
        stacked = np.concatenate([m[np.isfinite(m)] for m in matrices if np.isfinite(m).any()])
        color_ranges[metric] = (float(np.nanmin(stacked)), float(np.nanmax(stacked)))

    fig, axes = plt.subplots(
        len(days),
        len(metrics),
        figsize=(12, max(8, 3 * len(days))),
        squeeze=False,
        layout="constrained",
    )

    pcm_by_metric = {}
    for row_idx, day_name in enumerate(days):
        for col_idx, (metric, metric_title, colorbar_label) in enumerate(metrics):
            pcm = _plot_matrix_on_ax(
                ax=axes[row_idx, col_idx],
                matrix=loaded[(day_name, metric)],
                title=f"{day_name.upper()} {metric_title}",
                colorbar_label=colorbar_label,
                colorbar_range=color_ranges[metric],
                t_min=t_min,
                t_max=t_max,
                start_pm=start_pm,
                end_pm=end_pm,
                add_colorbar=False,
                show_xlabel=row_idx == len(days) - 1,
                show_ylabel=col_idx == 0,
                time_resolution_label=time_resolution_label,
                space_resolution_label=space_resolution_label,
            )
            pcm_by_metric[metric] = pcm

    for col_idx, (metric, _, colorbar_label) in enumerate(metrics):
        cbar = fig.colorbar(
            pcm_by_metric[metric],
            ax=axes[:, col_idx],
            shrink=0.95,
            pad=0.03,
        )
        cbar.set_label(colorbar_label)

    fig.suptitle(
        f"Figure 1: Speed and Flow Per Lane at {time_resolution_label} x {space_resolution_label}"
    )
    return _save_or_show_figure(fig, output_path=output_path)


def plot_i24_figure_2_flow_per_lane_resolution_sweep(
    day_name="nov21",
    dt_labels=("10s", "30s", "1min", "5min"),
    dx_values=(200, 400, 600, 800),
    input_dir="data/i24/matrix_sweeps/daily_combined_repaired",
    space_bin_csv_dir="data/i24/matrix_sweeps/daily_combined_repaired/space_bin_csvs",
    output_path=None,
):
    """Figure 2: flow-per-lane sweep across temporal and spatial resolutions."""
    t_min, t_max = _get_i24_time_bounds(start_label="0630", end_label="1000")
    loaded = {}
    finite_values = []

    for dt_label in dt_labels:
        for dx_meters in dx_values:
            matrix, _ = _load_i24_repaired_matrix(
                day_name=day_name,
                metric="flow_per_lane",
                dt_label=dt_label,
                dx_meters=dx_meters,
                input_dir=input_dir,
            )
            loaded[(dt_label, dx_meters)] = matrix
            if np.isfinite(matrix).any():
                finite_values.append(matrix[np.isfinite(matrix)])

    stacked = np.concatenate(finite_values)
    colorbar_range = (float(np.nanmin(stacked)), float(np.nanmax(stacked)))

    fig, axes = plt.subplots(
        len(dt_labels),
        len(dx_values),
        figsize=(4 * len(dx_values), 2.8 * len(dt_labels)),
        squeeze=False,
        layout="constrained",
    )

    last_pcm = None
    for row_idx, dt_label in enumerate(dt_labels):
        for col_idx, dx_meters in enumerate(dx_values):
            start_pm, end_pm = _get_i24_plot_bounds(
                dx_meters,
                space_bin_csv_dir=space_bin_csv_dir,
            )
            last_pcm = _plot_matrix_on_ax(
                ax=axes[row_idx, col_idx],
                matrix=loaded[(dt_label, dx_meters)],
                title=f"{day_name.upper()} {dt_label} x {dx_meters}m",
                colorbar_label="Flow Per Lane",
                colorbar_range=colorbar_range,
                t_min=t_min,
                t_max=t_max,
                start_pm=start_pm,
                end_pm=end_pm,
                add_colorbar=False,
                show_xlabel=row_idx == len(dt_labels) - 1,
                show_ylabel=col_idx == 0,
                time_resolution_label=_format_dt_resolution_label(dt_label),
                space_resolution_label=_format_dx_resolution_label(dx_meters),
            )

    cbar = fig.colorbar(last_pcm, ax=axes, shrink=0.95, pad=0.03)
    cbar.set_label("Flow Per Lane")
    fig.suptitle(f"Figure 2: {day_name.upper()} Flow Per Lane Resolution Sweep")
    return _save_or_show_figure(fig, output_path=output_path)


def plot_i24_figure_3_flow_per_lane_masking_grid(
    day_name="nov21",
    dt_label="10s",
    dx_values=(200, 400, 600, 800),
    num_masks=5,
    rng=None,
    mask_indices_by_dx=None,
    input_dir="data/i24/matrix_sweeps/daily_combined_repaired",
    space_bin_csv_dir="data/i24/matrix_sweeps/daily_combined_repaired/space_bin_csvs",
    output_path=None,
):
    """Figure 3: grid of random masked flow-per-lane matrices across spatial resolutions."""
    from methods import apply_row_masks, generate_masked_row_arrays

    t_min, t_max = _get_i24_time_bounds(start_label="0630", end_label="1000")
    masked_by_dx = {}
    finite_values = []

    for dx_meters in dx_values:
        matrix, _ = _load_i24_repaired_matrix(
            day_name=day_name,
            metric="flow_per_lane",
            dt_label=dt_label,
            dx_meters=dx_meters,
            input_dir=input_dir,
        )
        if mask_indices_by_dx is not None:
            if dx_meters not in mask_indices_by_dx:
                raise KeyError(f"mask_indices_by_dx is missing resolution {dx_meters}m.")
            masked_matrices = apply_row_masks(matrix, mask_indices_by_dx[dx_meters])
        else:
            masked_matrices = generate_masked_row_arrays(
                matrix=matrix,
                resolution=dx_meters,
                num_masks=num_masks,
                rng=rng,
            )
        if len(masked_matrices) != num_masks:
            raise ValueError(
                f"Expected {num_masks} masked matrices for {dx_meters}m, got {len(masked_matrices)}."
            )
        masked_by_dx[dx_meters] = masked_matrices
        if np.isfinite(matrix).any():
            finite_values.append(matrix[np.isfinite(matrix)])

    stacked = np.concatenate(finite_values)
    colorbar_range = (float(np.nanmin(stacked)), float(np.nanmax(stacked)))

    fig, axes = plt.subplots(
        num_masks,
        len(dx_values),
        figsize=(4 * len(dx_values), 2.4 * num_masks),
        squeeze=False,
        layout="constrained",
    )

    last_pcm = None
    for row_idx in range(num_masks):
        for col_idx, dx_meters in enumerate(dx_values):
            start_pm, end_pm = _get_i24_plot_bounds(
                dx_meters,
                space_bin_csv_dir=space_bin_csv_dir,
            )
            last_pcm = _plot_matrix_on_ax(
                ax=axes[row_idx, col_idx],
                matrix=masked_by_dx[dx_meters][row_idx],
                title=f"Mask {row_idx + 1} | {dx_meters}m",
                colorbar_label="Flow Per Lane",
                colorbar_range=colorbar_range,
                t_min=t_min,
                t_max=t_max,
                start_pm=start_pm,
                end_pm=end_pm,
                add_colorbar=False,
                show_xlabel=row_idx == num_masks - 1,
                show_ylabel=col_idx == 0,
                title_prefix="Masked",
                time_resolution_label=_format_dt_resolution_label(dt_label),
                space_resolution_label=_format_dx_resolution_label(dx_meters),
            )

    cbar = fig.colorbar(last_pcm, ax=axes, shrink=0.95, pad=0.03)
    cbar.set_label("Flow Per Lane")
    fig.suptitle(
        f"Figure 3: {day_name.upper()} Flow Per Lane Random Row Masking at "
        f"{_format_dt_resolution_label(dt_label)}",
    )
    return _save_or_show_figure(fig, output_path=output_path)

def _build_space_bins(x_start, x_end, spacing_mi):
    """Return monotonic bin edges between x_start and x_end with |spacing_mi| steps."""
    if spacing_mi == 0:
        raise ValueError("spacing_mi must be non-zero.")
    if x_start == x_end:
        raise ValueError("x_start and x_end must differ.")

    distance = x_end - x_start
    direction = np.sign(distance)
    step = abs(spacing_mi)
    num_segments = max(int(np.ceil(abs(distance) / step)), 1)

    edges = x_start + direction * step * np.arange(num_segments + 1)
    edges[-1] = x_end  # ensure exact end value
    return edges

def get_ramps_per_segment(ramps_path, x_start, x_end, spacing_mi):
    """Return boolean on/off-ramp flags per spatial bin."""
    ramps_df = pd.read_csv(ramps_path).copy()
    ramps_df["x"] = ramps_df["x_rcs_miles"]

    edges = _build_space_bins(x_start, x_end, spacing_mi)
    num_bins = len(edges) - 1
    on_ramp = np.zeros(num_bins, dtype=bool)
    off_ramp = np.zeros(num_bins, dtype=bool)

    x_vals = ramps_df["x"].to_numpy()
    entry_vals = ramps_df["entry_node"].astype(str).str.upper().eq("TRUE").to_numpy()
    exit_vals = ramps_df["exit_node"].astype(str).str.upper().eq("TRUE").to_numpy()

    asc_edges = edges if edges[0] < edges[-1] else edges[::-1]
    bin_idx = np.searchsorted(asc_edges, x_vals, side="right") - 1
    valid = (bin_idx >= 0) & (bin_idx < num_bins) & (x_vals >= asc_edges[0]) & (x_vals < asc_edges[-1])

    if edges[0] > edges[-1]:
        bin_idx = num_bins - 1 - bin_idx

    valid_bins = bin_idx[valid]
    on_ramp_counts = np.bincount(
        valid_bins,
        weights=entry_vals[valid].astype(np.uint8),
        minlength=num_bins,
    ).astype(int)
    off_ramp_counts = np.bincount(
        valid_bins,
        weights=exit_vals[valid].astype(np.uint8),
        minlength=num_bins,
    ).astype(int)
    on_ramp = on_ramp_counts > 0
    off_ramp = off_ramp_counts > 0

    for i, (on_count, off_count) in enumerate(zip(on_ramp_counts, off_ramp_counts)):
        low, high = sorted((edges[i], edges[i + 1]))
        print(
            f"Ramp bin {i} [{low}, {high}): "
            f"{on_count} on-ramps, {off_count} off-ramps"
        )

    return on_ramp, off_ramp, edges

def get_lanes_per_segment(lanes_path, x_start, x_end, spacing_mi, debug=False):
    """Return weighted-average lane counts per spatial bin."""
    lanes_df = pd.read_csv(lanes_path).copy()
    lanes_df["x_min"] = lanes_df[["x_start_mile", "x_end_mile"]].min(axis=1)
    lanes_df["x_max"] = lanes_df[["x_start_mile", "x_end_mile"]].max(axis=1)

    edges = _build_space_bins(x_start, x_end, spacing_mi)
    lane_vals = np.zeros(len(edges) - 1)
    if debug:
        direction = "ascending" if x_end > x_start else "descending"
        print(
            f"[lanes] x_start={x_start:.3f}, x_end={x_end:.3f}, spacing={abs(spacing_mi):.3f} mi, "
            f"direction={direction}, bins={len(edges)-1}"
        )
        print(f"[lanes] first 3 edges: {np.round(edges[:3], 4)}")
        print(f"[lanes] last  3 edges: {np.round(edges[-3:], 4)}")

    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        low, high = (a, b) if a < b else (b, a)
        overlapping = lanes_df[(lanes_df["x_max"] > low) & (lanes_df["x_min"] < high)]

        if overlapping.empty:
            lane_vals[i] = lane_vals[i - 1] if i > 0 else np.nan
            if debug:
                print(
                    f"[lanes][bin {i:02d}] range=({low:.3f}, {high:.3f}) overlap=0 -> "
                    f"lane={lane_vals[i]:.3f}"
                )
            continue

        overlap_len = np.minimum(overlapping["x_max"], high) - np.maximum(overlapping["x_min"], low)
        overlap_len = np.clip(overlap_len, 0, None)

        if np.any(overlap_len > 0):
            lane_vals[i] = np.average(overlapping["lanes"], weights=overlap_len)
            if debug:
                print(
                    f"[lanes][bin {i:02d}] range=({low:.3f}, {high:.3f}) overlap={len(overlapping)} "
                    f"weighted_lane={lane_vals[i]:.3f}"
                )
        else:
            lane_vals[i] = overlapping["lanes"].mean()
            if debug:
                print(
                    f"[lanes][bin {i:02d}] range=({low:.3f}, {high:.3f}) overlap={len(overlapping)} "
                    f"mean_lane={lane_vals[i]:.3f} (zero overlap lengths fallback)"
                )

    if debug:
        nan_count = int(np.isnan(lane_vals).sum())
        print(
            f"[lanes] done: min={np.nanmin(lane_vals):.3f}, max={np.nanmax(lane_vals):.3f}, "
            f"nan_count={nan_count}, first5={np.round(lane_vals[:5], 3)}"
        )
    return lane_vals


def save_i24_lane_and_ramp_mappings(
    space_bin_csv_dir="data/i24/matrix_sweeps/daily_combined_repaired/space_bin_csvs",
    lanes_path="data/i24/lanes.csv",
    ramps_path="data/i24/ramps.csv",
    output_dir="data/i24/segment_mappings",
):
    """
    Build and save per-resolution lane and ramp mappings aligned to matrix space indices.

    For each `space_bins_dx_*m.csv` file:
    - `x_start` is taken from the CSV's global spatial maximum
    - `x_end` is taken from the CSV's global spatial minimum
    - `spacing_mi` is taken from `dx_miles`

    Saved outputs for each resolution:
    - `lane_mapping_dx_<resolution>m.npy`
    - `on_ramp_mapping_dx_<resolution>m.npy`
    - `off_ramp_mapping_dx_<resolution>m.npy`
    """
    space_bin_csv_dir = Path(space_bin_csv_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _trim_degenerate_last_bin(values, trim_last_bin, expected_bins, label, dx_meters):
        if len(values) == expected_bins:
            return values
        if len(values) == expected_bins + 1 and trim_last_bin:
            return values[:-1]
        raise ValueError(
            f"{label} mapping length mismatch for {dx_meters}m: "
            f"expected {expected_bins}, got {len(values)}."
        )

    records = []

    for csv_path in sorted(space_bin_csv_dir.glob("space_bins_dx_*m.csv")):
        bins_df = pd.read_csv(csv_path)
        if bins_df.empty:
            raise ValueError(f"Space-bin CSV '{csv_path}' is empty.")

        required_columns = {
            "left_edge_miles",
            "right_edge_miles",
            "dx_meters",
            "dx_miles",
        }
        missing_columns = required_columns.difference(bins_df.columns)
        if missing_columns:
            raise ValueError(
                f"Space-bin CSV '{csv_path}' is missing required columns: "
                f"{sorted(missing_columns)}."
            )

        edge_min = float(
            min(
                bins_df["left_edge_miles"].min(),
                bins_df["right_edge_miles"].min(),
            )
        )
        edge_max = float(
            max(
                bins_df["left_edge_miles"].max(),
                bins_df["right_edge_miles"].max(),
            )
        )
        x_start = edge_max
        x_end = edge_min
        spacing_mi = float(bins_df["dx_miles"].iloc[0])
        dx_meters = int(round(float(bins_df["dx_meters"].iloc[0])))

        lane_mapping = get_lanes_per_segment(
            lanes_path=lanes_path,
            x_start=x_start,
            x_end=x_end,
            spacing_mi=spacing_mi,
        )
        on_ramp_mapping, off_ramp_mapping, edges = get_ramps_per_segment(
            ramps_path=ramps_path,
            x_start=x_start,
            x_end=x_end,
            spacing_mi=spacing_mi,
        )

        expected_bins = len(bins_df)
        trim_last_bin = (
            len(edges) == expected_bins + 2 and np.isclose(edges[-1], edges[-2])
        )

        lane_mapping = _trim_degenerate_last_bin(
            lane_mapping,
            trim_last_bin,
            expected_bins,
            "Lane",
            dx_meters,
        )
        on_ramp_mapping = _trim_degenerate_last_bin(
            on_ramp_mapping,
            trim_last_bin,
            expected_bins,
            "On-ramp",
            dx_meters,
        )
        off_ramp_mapping = _trim_degenerate_last_bin(
            off_ramp_mapping,
            trim_last_bin,
            expected_bins,
            "Off-ramp",
            dx_meters,
        )
        if trim_last_bin:
            edges = edges[:-1]

        np.save(output_dir / f"lane_mapping_dx_{dx_meters}m.npy", lane_mapping)
        np.save(output_dir / f"on_ramp_mapping_dx_{dx_meters}m.npy", on_ramp_mapping)
        np.save(output_dir / f"off_ramp_mapping_dx_{dx_meters}m.npy", off_ramp_mapping)

        records.append(
            {
                "dx_meters": dx_meters,
                "space_bins": expected_bins,
                "x_start": x_start,
                "x_end": x_end,
                "spacing_mi": spacing_mi,
                "edges_count": len(edges),
                "lane_path": str(output_dir / f"lane_mapping_dx_{dx_meters}m.npy"),
                "on_ramp_path": str(output_dir / f"on_ramp_mapping_dx_{dx_meters}m.npy"),
                "off_ramp_path": str(output_dir / f"off_ramp_mapping_dx_{dx_meters}m.npy"),
            }
        )

    if not records:
        raise ValueError(
            f"No space-bin CSVs matching 'space_bins_dx_*m.csv' were found in '{space_bin_csv_dir}'."
        )

    manifest_df = pd.DataFrame(records).sort_values("dx_meters").reset_index(drop=True)
    manifest_df.to_csv(output_dir / "segment_mapping_manifest.csv", index=False)
    return manifest_df


def y_weighted_fill_or_smooth(
    matrix,
    mode="impute",                 # "impute" or "smooth"
    num_neighbors=3,
    max_passes=5,
    include_center=True
):
    """
    mode="impute": only fill NaN cells using weighted y-neighbors.
    mode="smooth": update all cells using weighted y-neighbors.
    """
    if mode not in {"impute", "smooth"}:
        raise ValueError("mode must be either 'impute' or 'smooth'")

    # Force numeric ndarray with real np.nan
    arr = np.asarray(matrix)
    if not np.issubdtype(arr.dtype, np.number):
        arr = pd.DataFrame(arr).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    else:
        arr = arr.astype(float, copy=True)

    out = arr.copy()
    n_rows, n_cols = out.shape

    for _ in range(max_passes):
        prev_nan = np.isnan(out).sum()
        src = out.copy()  # read from previous pass only

        for i in range(n_rows):
            for j in range(n_cols):
                # In impute mode, skip non-NaN cells
                if mode == "impute" and not np.isnan(src[i, j]):
                    continue

                vals, wts = [], []

                # Only relevant for smooth mode
                if mode == "smooth" and include_center and not np.isnan(src[i, j]):
                    vals.append(src[i, j])
                    wts.append(num_neighbors + 1)

                for d in range(1, num_neighbors + 1):
                    w = num_neighbors - d + 1  # e.g., 3,2,1
                    up = i - d
                    dn = i + d

                    if up >= 0 and not np.isnan(src[up, j]):
                        vals.append(src[up, j]); wts.append(w)
                    if dn < n_rows and not np.isnan(src[dn, j]):
                        vals.append(src[dn, j]); wts.append(w)

                if vals:
                    out[i, j] = np.average(vals, weights=wts)
                elif mode == "smooth":
                    out[i, j] = np.nan  # smooth mode overwrites all cells

        # early stop for impute mode when no NaN count change
        if mode == "impute":
            new_nan = np.isnan(out).sum()
            if new_nan == prev_nan:
                break

    return out
