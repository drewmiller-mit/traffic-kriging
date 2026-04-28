"""METANET-based preparation utilities for physics-based imputation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


KILOMETERS_PER_MILE = 1.609344
SEGMENT_MAPPING_DIR = Path("data/i24/segment_mappings")
SEGMENT_MAPPING_MANIFEST = SEGMENT_MAPPING_DIR / "segment_mapping_manifest.csv"


MatrixOrientation = Literal["space_time", "time_space"]
VelocityUnits = Literal["mph", "kmh"]


@dataclass(frozen=True)
class I24SegmentMappings:
    """Lane and ramp indicator arrays aligned to I-24 matrix rows."""

    lane_mapping: np.ndarray
    on_ramp_mapping: np.ndarray
    off_ramp_mapping: np.ndarray


@dataclass(frozen=True)
class MetanetReadyInputs:
    """Arrays and metadata in the units expected by METANET calibration."""

    rho_hat: np.ndarray
    q_hat: np.ndarray
    v_hat: np.ndarray
    T: float
    l: float
    lane_mapping: np.ndarray
    on_ramp_mapping: np.ndarray
    off_ramp_mapping: np.ndarray
    observed_mask: np.ndarray
    total_flow_space_time: np.ndarray
    avg_velocity_space_time: np.ndarray


def parse_dt_label_to_seconds(dt_label: str) -> float:
    """Convert labels like `10s`, `30s`, `1min`, or `5min` to seconds."""
    label = str(dt_label).strip().lower()
    if label.endswith("min"):
        return float(label[:-3]) * 60.0
    if label.endswith("s"):
        return float(label[:-1])
    raise ValueError(f"Unsupported dt label '{dt_label}'.")


def _as_space_time_matrix(
    matrix: np.ndarray,
    *,
    name: str,
    orientation: MatrixOrientation,
) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D; got shape {arr.shape}.")
    if orientation == "space_time":
        return arr
    if orientation == "time_space":
        return arr.T
    raise ValueError(
        f"Unsupported orientation '{orientation}'. Expected 'space_time' or 'time_space'."
    )


def load_i24_segment_mappings(
    dx_meters: int,
    *,
    mapping_dir: str | Path = SEGMENT_MAPPING_DIR,
    expected_segments: int | None = None,
) -> I24SegmentMappings:
    """Load lane, on-ramp, and off-ramp mappings for an I-24 spatial resolution."""
    mapping_dir = Path(mapping_dir)
    manifest_path = mapping_dir / SEGMENT_MAPPING_MANIFEST.name
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        matches = manifest[manifest["dx_meters"].astype(int) == int(dx_meters)]
        if matches.empty:
            raise ValueError(f"No segment mapping manifest row found for dx={dx_meters}m.")
        row = matches.iloc[0]
        lane_path = Path(row["lane_path"])
        on_ramp_path = Path(row["on_ramp_path"])
        off_ramp_path = Path(row["off_ramp_path"])
    else:
        lane_path = mapping_dir / f"lane_mapping_dx_{int(dx_meters)}m.npy"
        on_ramp_path = mapping_dir / f"on_ramp_mapping_dx_{int(dx_meters)}m.npy"
        off_ramp_path = mapping_dir / f"off_ramp_mapping_dx_{int(dx_meters)}m.npy"

    lane_mapping = np.load(lane_path).astype(float, copy=False)
    on_ramp_mapping = np.load(on_ramp_path).astype(float, copy=False)
    off_ramp_mapping = np.load(off_ramp_path).astype(float, copy=False)

    lengths = {
        "lane_mapping": lane_mapping.shape[0],
        "on_ramp_mapping": on_ramp_mapping.shape[0],
        "off_ramp_mapping": off_ramp_mapping.shape[0],
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Segment mapping length mismatch: {lengths}.")
    if expected_segments is not None and lane_mapping.shape[0] != expected_segments:
        raise ValueError(
            f"Mapping length mismatch for dx={dx_meters}m: matrix has "
            f"{expected_segments} rows, mapping has {lane_mapping.shape[0]}."
        )

    return I24SegmentMappings(
        lane_mapping=lane_mapping,
        on_ramp_mapping=on_ramp_mapping,
        off_ramp_mapping=off_ramp_mapping,
    )


def prepare_metanet_calibration_inputs(
    total_flow_matrix: np.ndarray,
    avg_velocity_matrix: np.ndarray,
    *,
    dx_meters: int,
    dt_seconds: float | None = None,
    dt_label: str | None = None,
    velocity_units: VelocityUnits = "mph",
    orientation: MatrixOrientation = "space_time",
    mapping_dir: str | Path = SEGMENT_MAPPING_DIR,
) -> MetanetReadyInputs:
    """
    Convert masked flow and speed matrices into METANET-ready calibration arrays.

    Inputs are expected as total flow, not flow-per-lane. The returned `q_hat`,
    `rho_hat`, and `v_hat` are shaped `(time, segment)`, with `q_hat` in veh/hr,
    `rho_hat` in veh/km, `v_hat` in km/hr, `T` in hours, and `l` in km. Missing
    values are preserved as `NaN` so calibration/simulation can perform the
    imputation rather than receiving pre-filled data.
    """
    if dt_seconds is None:
        if dt_label is None:
            raise ValueError("Provide either `dt_seconds` or `dt_label`.")
        dt_seconds = parse_dt_label_to_seconds(dt_label)
    if dt_seconds <= 0.0:
        raise ValueError(f"dt_seconds must be positive; got {dt_seconds}.")
    if dx_meters <= 0:
        raise ValueError(f"dx_meters must be positive; got {dx_meters}.")

    flow_st = _as_space_time_matrix(
        total_flow_matrix,
        name="total_flow_matrix",
        orientation=orientation,
    )
    velocity_st = _as_space_time_matrix(
        avg_velocity_matrix,
        name="avg_velocity_matrix",
        orientation=orientation,
    )
    if flow_st.shape != velocity_st.shape:
        raise ValueError(
            "total_flow_matrix and avg_velocity_matrix must have matching shapes; "
            f"got {flow_st.shape} and {velocity_st.shape}."
        )

    observed_mask = np.isfinite(flow_st) & np.isfinite(velocity_st)
    mappings = load_i24_segment_mappings(
        dx_meters=dx_meters,
        mapping_dir=mapping_dir,
        expected_segments=flow_st.shape[0],
    )

    q_st = np.where(np.isfinite(flow_st), np.maximum(flow_st, 1e-4), np.nan)
    velocity_st = np.where(
        np.isfinite(velocity_st),
        np.maximum(velocity_st, 1e-4),
        np.nan,
    )

    if velocity_units == "mph":
        v_kmh_st = velocity_st * KILOMETERS_PER_MILE
    elif velocity_units == "kmh":
        v_kmh_st = velocity_st
    else:
        raise ValueError("velocity_units must be either 'mph' or 'kmh'.")

    rho_veh_per_km_st = np.divide(
        q_st,
        v_kmh_st,
        out=np.full_like(q_st, np.nan, dtype=float),
        where=np.isfinite(q_st) & np.isfinite(v_kmh_st) & (v_kmh_st > 0.0),
    )
    rho_veh_per_km_st = np.where(
        np.isfinite(rho_veh_per_km_st),
        np.maximum(rho_veh_per_km_st, 1e-4),
        np.nan,
    )

    return MetanetReadyInputs(
        rho_hat=rho_veh_per_km_st.T,
        q_hat=q_st.T,
        v_hat=v_kmh_st.T,
        T=float(dt_seconds) / 3600.0,
        l=float(dx_meters) / 1000.0,
        lane_mapping=mappings.lane_mapping,
        on_ramp_mapping=mappings.on_ramp_mapping,
        off_ramp_mapping=mappings.off_ramp_mapping,
        observed_mask=observed_mask,
        total_flow_space_time=q_st,
        avg_velocity_space_time=velocity_st,
    )


def run_metanet_calibration_on_matrices(
    total_flow_matrix: np.ndarray,
    avg_velocity_matrix: np.ndarray,
    *,
    dx_meters: int,
    dt_seconds: float | None = None,
    dt_label: str | None = None,
    velocity_units: VelocityUnits = "mph",
    orientation: MatrixOrientation = "space_time",
    mapping_dir: str | Path = SEGMENT_MAPPING_DIR,
    calibration_kwargs: dict | None = None,
) -> tuple[dict, MetanetReadyInputs]:
    """Prepare masked matrices and call the METANET calibration routine."""
    from metanet_calibration.metanet_calibration.ipopt_optimization import run_calibration

    prepared = prepare_metanet_calibration_inputs(
        total_flow_matrix=total_flow_matrix,
        avg_velocity_matrix=avg_velocity_matrix,
        dx_meters=dx_meters,
        dt_seconds=dt_seconds,
        dt_label=dt_label,
        velocity_units=velocity_units,
        orientation=orientation,
        mapping_dir=mapping_dir,
    )
    kwargs = {
        "num_calibrated_segments": prepared.q_hat.shape[1] - 2,
        "include_ramping": True,
        "time_varying_ramping": True,
        "smoothing": True,
        "varylanes": False,
        "lane_mapping": prepared.lane_mapping,
        "on_ramp_mapping": prepared.on_ramp_mapping,
        "off_ramp_mapping": prepared.off_ramp_mapping,
    }
    if calibration_kwargs is not None:
        kwargs.update(calibration_kwargs)

    results = run_calibration(
        rho_hat=prepared.rho_hat,
        q_hat=prepared.q_hat,
        T=prepared.T,
        l=prepared.l,
        **kwargs,
    )
    return results, prepared
