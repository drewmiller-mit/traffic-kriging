"""Adaptive Smoothing Method wrappers for speed-field imputation experiments."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods import masked_mae, masked_mape, masked_rmse


METERS_PER_MILE = 1609.344
DEFAULT_ASM_PARAMS = {
    "c_cong": 12.26,
    "c_free": -50.40,
    "v_thr": 49.57,
    "v_delta": 10.11,
}


def _as_float_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2D with shape (space, time); got {arr.shape}.")
    return arr


def _resolve_grid_spacing(
    dx: float | None = None,
    dt: float | None = None,
    dx_miles: float | None = None,
    dt_seconds: float | None = None,
) -> tuple[float, float]:
    resolved_dx = dx if dx is not None else dx_miles
    resolved_dt = dt if dt is not None else dt_seconds
    if resolved_dx is None:
        raise ValueError("ASM requires `dx` or `dx_miles`.")
    if resolved_dt is None:
        raise ValueError("ASM requires `dt` or `dt_seconds`.")
    return float(resolved_dx), float(resolved_dt)


def fft_four_convs(Dp, Mp, k_cong, k_free, eps=1e-6, use_ortho=True):
    """
    FFT-accelerated 2D convolution of data and mask with two kernels.

    Parameters
    ----------
    Dp : Tensor (B, C, H, W) — data (NaN replaced with 0)
    Mp : Tensor (B, C, H, W) — binary mask (1 = observed, 0 = missing)
    k_cong, k_free : Tensor (F, C, Kh, Kw) — congestion / free-flow kernels
    eps : float — small constant added to denominator for numerical stability

    Returns
    -------
    sum_cong, N_cong, sum_free, N_free : Tensors (B, F, oh, ow)
        Weighted speed sums and weight counts for each kernel.
    """
    Dp = torch.nan_to_num(Dp, nan=0.0, posinf=0.0, neginf=0.0)
    Mp = torch.nan_to_num(Mp, nan=0.0, posinf=0.0, neginf=0.0)

    B, C, H, W = Dp.shape
    F_out, _, Kh, Kw = k_cong.shape
    Fh, Fw = H + Kh - 1, W + Kw - 1
    device, dtype = Dp.device, Dp.dtype

    Dp_pad = torch.zeros(B, C, Fh, Fw, device=device, dtype=dtype)
    Mp_pad = torch.zeros(B, C, Fh, Fw, device=device, dtype=dtype)
    Dp_pad[..., :H, :W] = Dp
    Mp_pad[..., :H, :W] = Mp

    k1_pad = torch.zeros(F_out, C, Fh, Fw, device=device, dtype=dtype)
    k2_pad = torch.zeros(F_out, C, Fh, Fw, device=device, dtype=dtype)
    k1_pad[..., :Kh, :Kw] = k_cong
    k2_pad[..., :Kh, :Kw] = k_free

    norm = "ortho" if use_ortho else None

    Df  = torch.fft.rfftn(Dp_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    Mf  = torch.fft.rfftn(Mp_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    Kf1 = torch.fft.rfftn(k1_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    Kf2 = torch.fft.rfftn(k2_pad, dim=(-2, -1), s=(Fh, Fw), norm=norm)

    y1 = torch.fft.irfftn(Df * Kf1, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    y2 = torch.fft.irfftn(Df * Kf2, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    z1 = torch.fft.irfftn(Mf * Kf1, dim=(-2, -1), s=(Fh, Fw), norm=norm)
    z2 = torch.fft.irfftn(Mf * Kf2, dim=(-2, -1), s=(Fh, Fw), norm=norm)

    oh, ow = H - Kh + 1, W - Kw + 1
    sum_cong = y1[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]
    sum_free = y2[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow]
    N_cong   = z1[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow] + eps
    N_free   = z2[..., Kh-1:Kh-1+oh, Kw-1:Kw-1+ow] + eps

    return sum_cong, N_cong, sum_free, N_free


def _resolve_space_axis_sign(space_axis_sign: int | float | str) -> float:
    """
    Resolve the sign convention for the ASM space axis.

    `1` means matrix row index increases in the positive physical x direction.
    `-1` means matrix row index increases opposite to the positive physical x
    direction, which flips the sign of the wave speeds in row coordinates.
    """
    if isinstance(space_axis_sign, str):
        normalized = space_axis_sign.strip().lower()
        aliases = {
            "row_increases_with_x": 1.0,
            "postmile": 1.0,
            "milepost": 1.0,
            "same": 1.0,
            "normal": 1.0,
            "row_decreases_with_x": -1.0,
            "reverse": -1.0,
            "reversed": -1.0,
            "flip": -1.0,
            "flipped": -1.0,
        }
        if normalized not in aliases:
            raise ValueError(
                "space_axis_sign must be 1, -1, or a known alias such as "
                "'same' or 'reverse'."
            )
        return aliases[normalized]

    sign = float(space_axis_sign)
    if sign not in {-1.0, 1.0}:
        raise ValueError("space_axis_sign must be either 1 or -1.")
    return sign


class AdaptiveSmoothing(nn.Module):
    """
    ASMx: Adaptive Smoothing Method for Traffic State Estimation.

    Reconstructs a dense speed field from sparse detector observations by convolving
    with two physics-informed anisotropic kernels (congested and free-flow), blended
    via a smooth sigmoid transition.

    Parameters
    ----------
    kernel_time_window : float — half-extent of the kernel in time (seconds)
    kernel_space_window : float — half-extent of the kernel in space (miles)
    dx : float — spatial grid cell size (miles)
    dt : float — temporal grid cell size (seconds)
    init_delta : float — initial spatial smoothing scale (miles)
    init_tau : float — initial temporal smoothing scale (seconds)
    init_c_cong : float — congestion wave speed (mph)
    init_c_free : float — free-flow wave speed (mph)
    init_v_thr : float — speed threshold for blending (mph)
    init_v_delta : float — transition width for blending (mph)
    """
    def __init__(self,
                 kernel_time_window: float,
                 kernel_space_window: float,
                 dx: float,
                 dt: float,
                 init_delta: float = 0.09,
                 init_tau: float = 9.27,
                 init_c_cong: float = 12.26,
                 init_c_free: float = -50.40,
                 init_v_thr: float = 49.57,
                 init_v_delta: float = 10.11,
                 space_axis_sign: int | float | str = 1):
        super().__init__()
        self.size_t = int(kernel_time_window / dt)
        self.size_x = int(kernel_space_window / dx)
        self.dt = dt
        self.dx = dx
        self.space_axis_sign = _resolve_space_axis_sign(space_axis_sign)

        t_offs = torch.arange(-self.size_t, self.size_t + 1) * dt
        x_offs = torch.arange(-self.size_x, self.size_x + 1) * dx * self.space_axis_sign
        X, T = torch.meshgrid(x_offs, t_offs, indexing='ij')
        self.register_buffer('T_offsets', T.float())
        self.register_buffer('X_offsets', X.float())

        self.delta   = nn.Parameter(torch.tensor(init_delta, dtype=torch.float32))
        self.tau     = nn.Parameter(torch.tensor(init_tau, dtype=torch.float32))
        self.c_cong  = nn.Parameter(torch.tensor(init_c_cong, dtype=torch.float32))
        self.c_free  = nn.Parameter(torch.tensor(init_c_free, dtype=torch.float32))
        self.v_thr   = nn.Parameter(torch.tensor(init_v_thr, dtype=torch.float32))
        self.v_delta = nn.Parameter(torch.tensor(init_v_delta, dtype=torch.float32))

    def forward(self, raw_data: torch.Tensor):
        """
        Run ASMx on a speed field.

        Parameters
        ----------
        raw_data : Tensor — 2D (space, time), 3D (batch, space, time),
                   or 4D (batch, channel, space, time). NaN = missing.

        Returns
        -------
        Tensor — reconstructed speed field (batch, space, time), no NaN.
        """
        if raw_data.ndim == 2:
            raw_data = raw_data.unsqueeze(0).unsqueeze(0)
        elif raw_data.ndim == 3:
            raw_data = raw_data.unsqueeze(1)

        mask = (~raw_data.isnan()).float()
        data = torch.nan_to_num(raw_data, nan=0.0)

        c_cong_s = self.c_cong / 3600.0
        c_free_s = self.c_free / 3600.0

        t_cong = self.T_offsets - self.X_offsets / c_cong_s
        t_free = self.T_offsets - self.X_offsets / c_free_s

        k_cong = torch.exp(-(t_cong.abs() / self.tau + self.X_offsets.abs() / self.delta))
        k_free = torch.exp(-(t_free.abs() / self.tau + self.X_offsets.abs() / self.delta))

        k_cong = k_cong.unsqueeze(0).unsqueeze(0)
        k_free = k_free.unsqueeze(0).unsqueeze(0)

        pad = (self.size_t, self.size_t, self.size_x, self.size_x)
        Dp = F.pad(data, pad, value=0.0)
        Mp = F.pad(mask, pad, value=0.0)

        sum_cong, N_cong, sum_free, N_free = fft_four_convs(Dp, Mp, k_cong, k_free)

        v_cong = sum_cong / N_cong
        v_free = sum_free / N_free

        v_min = torch.min(v_cong, v_free)
        w = 0.5 * (1 + torch.tanh((self.v_thr - v_min) / self.v_delta))
        v = w * v_cong + (1 - w) * v_free

        valid_cong = (N_cong > 0).float()
        valid_free = (N_free > 0).float()
        v = valid_cong * valid_free * v + (1 - valid_cong) * v_free + (1 - valid_free) * v_cong

        return v.squeeze(1)


def run_asmx(speed_matrix, dx, dt, delta, tau,
             c_cong=DEFAULT_ASM_PARAMS["c_cong"],
             c_free=DEFAULT_ASM_PARAMS["c_free"],
             v_thr=DEFAULT_ASM_PARAMS["v_thr"],
             v_delta=DEFAULT_ASM_PARAMS["v_delta"],
             min_value: float | None = 0.0,
             max_value: float | None = None,
             preserve_observed: bool = False,
             space_axis_sign: int | float | str = 1):
    """
    Convenience wrapper: run ASMx on a numpy speed matrix.

    Parameters
    ----------
    speed_matrix : ndarray (space, time) — speed in mph, NaN = missing
    dx, dt : float — spatial (mi) and temporal (s) resolution
    delta, tau : float — spatial and temporal smoothing scales
    c_cong, c_free, v_thr, v_delta : float — physics parameters

    Returns
    -------
    ndarray (space, time) — reconstructed speed field
    """
    speed_matrix = _as_float_matrix(speed_matrix)
    if dx <= 0.0:
        raise ValueError("dx must be positive miles.")
    if dt <= 0.0:
        raise ValueError("dt must be positive seconds.")
    if delta <= 0.0:
        raise ValueError("delta must be positive miles.")
    if tau <= 0.0:
        raise ValueError("tau must be positive seconds.")

    space_size, time_size = speed_matrix.shape
    kernel_time_window = time_size * dt
    kernel_space_window = space_size * dx

    model = AdaptiveSmoothing(
        kernel_time_window=kernel_time_window,
        kernel_space_window=kernel_space_window,
        dx=dx, dt=dt,
        init_delta=delta, init_tau=tau,
        init_c_cong=c_cong, init_c_free=c_free,
        init_v_thr=v_thr, init_v_delta=v_delta,
        space_axis_sign=space_axis_sign,
    )
    model.eval()

    speed_matrix = np.ascontiguousarray(speed_matrix)
    raw_tensor = torch.from_numpy(speed_matrix).float()
    with torch.no_grad():
        smoothed = model(raw_tensor)
    imputed = smoothed[0].cpu().numpy().astype(speed_matrix.dtype, copy=False)
    if min_value is not None or max_value is not None:
        imputed = np.clip(imputed, min_value, max_value)
    if preserve_observed:
        observed = np.isfinite(speed_matrix)
        imputed[observed] = speed_matrix[observed]
    return imputed


def asm_impute(
    matrix: np.ndarray,
    dx: float,
    dt: float,
    delta: float = 0.09,
    tau: float = 9.27,
    c_cong: float = DEFAULT_ASM_PARAMS["c_cong"],
    c_free: float = DEFAULT_ASM_PARAMS["c_free"],
    v_thr: float = DEFAULT_ASM_PARAMS["v_thr"],
    v_delta: float = DEFAULT_ASM_PARAMS["v_delta"],
    min_value: float | None = 0.0,
    max_value: float | None = None,
    preserve_observed: bool = True,
    space_axis_sign: int | float | str = 1,
) -> np.ndarray:
    """
    Impute a `(space, time)` speed matrix with ASM.

    `dx` is in miles, `dt` is in seconds, and speeds/wave speeds are in mph.
    Missing cells must be encoded as `np.nan`.
    """
    return run_asmx(
        speed_matrix=matrix,
        dx=dx,
        dt=dt,
        delta=delta,
        tau=tau,
        c_cong=c_cong,
        c_free=c_free,
        v_thr=v_thr,
        v_delta=v_delta,
        min_value=min_value,
        max_value=max_value,
        preserve_observed=preserve_observed,
        space_axis_sign=space_axis_sign,
    )


def evaluate_asm(
    ground_truth: np.ndarray,
    masked_matrix: np.ndarray,
    dx: float | None = None,
    dt: float | None = None,
    dx_miles: float | None = None,
    dt_seconds: float | None = None,
    delta: float = 0.09,
    tau: float = 9.27,
    c_cong: float = DEFAULT_ASM_PARAMS["c_cong"],
    c_free: float = DEFAULT_ASM_PARAMS["c_free"],
    v_thr: float = DEFAULT_ASM_PARAMS["v_thr"],
    v_delta: float = DEFAULT_ASM_PARAMS["v_delta"],
    min_value: float | None = 0.0,
    max_value: float | None = None,
    preserve_observed: bool = True,
    space_axis_sign: int | float | str = 1,
    return_imputed_matrix: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
    """Run ASM on one masked speed matrix and score experimentally hidden cells."""
    resolved_dx, resolved_dt = _resolve_grid_spacing(
        dx=dx,
        dt=dt,
        dx_miles=dx_miles,
        dt_seconds=dt_seconds,
    )
    imputed_matrix = asm_impute(
        matrix=masked_matrix,
        dx=resolved_dx,
        dt=resolved_dt,
        delta=delta,
        tau=tau,
        c_cong=c_cong,
        c_free=c_free,
        v_thr=v_thr,
        v_delta=v_delta,
        min_value=min_value,
        max_value=max_value,
        preserve_observed=preserve_observed,
        space_axis_sign=space_axis_sign,
    )

    metrics = {
        "mae": masked_mae(ground_truth, imputed_matrix, masked_matrix),
        "mape": masked_mape(ground_truth, imputed_matrix, masked_matrix),
        "rmse": masked_rmse(ground_truth, imputed_matrix, masked_matrix),
    }

    if return_imputed_matrix:
        return metrics, imputed_matrix
    return metrics


def evaluate_asm_on_masks(
    ground_truth: np.ndarray,
    masked_matrices: list[np.ndarray],
    dx: float | None = None,
    dt: float | None = None,
    dx_miles: float | None = None,
    dt_seconds: float | None = None,
    delta: float = 0.09,
    tau: float = 9.27,
    c_cong: float = DEFAULT_ASM_PARAMS["c_cong"],
    c_free: float = DEFAULT_ASM_PARAMS["c_free"],
    v_thr: float = DEFAULT_ASM_PARAMS["v_thr"],
    v_delta: float = DEFAULT_ASM_PARAMS["v_delta"],
    min_value: float | None = 0.0,
    max_value: float | None = None,
    preserve_observed: bool = True,
    space_axis_sign: int | float | str = 1,
    return_imputed_matrices: bool = False,
) -> list[dict[str, float]] | tuple[list[dict[str, float]], list[np.ndarray]]:
    """Run and score ASM over a collection of masked speed matrices."""
    all_metrics = []
    all_imputed = []
    resolved_dx, resolved_dt = _resolve_grid_spacing(
        dx=dx,
        dt=dt,
        dx_miles=dx_miles,
        dt_seconds=dt_seconds,
    )

    for mask_idx, masked_matrix in enumerate(masked_matrices):
        result = evaluate_asm(
            ground_truth=ground_truth,
            masked_matrix=masked_matrix,
            dx=resolved_dx,
            dt=resolved_dt,
            delta=delta,
            tau=tau,
            c_cong=c_cong,
            c_free=c_free,
            v_thr=v_thr,
            v_delta=v_delta,
            min_value=min_value,
            max_value=max_value,
            preserve_observed=preserve_observed,
            space_axis_sign=space_axis_sign,
            return_imputed_matrix=return_imputed_matrices,
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
