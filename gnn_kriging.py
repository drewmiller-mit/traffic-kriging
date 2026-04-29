"""GNN building blocks for traffic kriging experiments.

This module is a local adaptation of the STCAGCN implementation from
`GNN4Flow-main`.  It keeps the borrowed architecture isolated from the
experiment runner so later stages can add I-24-specific training and evaluation
without modifying the upstream code.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F

from methods import masked_mae, masked_mape, masked_rmse


def calculate_asymmetric_random_walk(adj: np.ndarray) -> np.ndarray:
    """Return row-normalized random-walk adjacency as `float32`."""
    adj_mx = sp.coo_matrix(np.asarray(adj, dtype=np.float32))
    rowsum = np.asarray(adj_mx.sum(1)).flatten()
    with np.errstate(divide="ignore"):
        d_inv = np.power(rowsum, -1.0).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj_mx).astype(np.float32).toarray()


def build_directed_chain_adjacency(
    num_nodes: int,
    self_loops: bool = True,
    neighbor_weight: float = 1.0,
) -> np.ndarray:
    """Build a simple bidirectional freeway-chain adjacency matrix."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if neighbor_weight <= 0.0:
        raise ValueError("neighbor_weight must be positive.")

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes - 1):
        adj[i, i + 1] = neighbor_weight
        adj[i + 1, i] = neighbor_weight
    if self_loops:
        np.fill_diagonal(adj, 1.0)
    return adj


class DiffusionGraphConvolution(nn.Module):
    """Diffusion graph convolution with optional adaptive adjacency fusion."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        order: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        if order <= 0:
            raise ValueError("order must be positive.")

        self.order = int(order)
        self.activation = activation
        self.num_matrices = 2 * self.order + 1
        self.theta_static = nn.Parameter(
            torch.empty(in_features * self.num_matrices, out_features)
        )
        self.bias_static = nn.Parameter(torch.empty(out_features))
        self.theta_adaptive = nn.Parameter(
            torch.empty(in_features * self.num_matrices, out_features)
        )
        self.bias_adaptive = nn.Parameter(torch.empty(out_features))
        self.fusion = nn.Linear(3 * out_features, out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        static_std = 1.0 / math.sqrt(self.theta_static.shape[1])
        self.theta_static.data.uniform_(-static_std, static_std)
        self.bias_static.data.uniform_(
            -1.0 / math.sqrt(self.bias_static.shape[0]),
            1.0 / math.sqrt(self.bias_static.shape[0]),
        )

        adaptive_std = 1.0 / math.sqrt(self.theta_adaptive.shape[1])
        self.theta_adaptive.data.uniform_(-adaptive_std, adaptive_std)
        self.bias_adaptive.data.uniform_(
            -1.0 / math.sqrt(self.bias_adaptive.shape[0]),
            1.0 / math.sqrt(self.bias_adaptive.shape[0]),
        )

    @staticmethod
    def _diffuse(
        x: torch.Tensor,
        supports: list[torch.Tensor],
        order: int,
    ) -> torch.Tensor:
        batch_size, num_features, num_times, num_nodes = x.shape
        x0 = x.permute(0, 3, 1, 2).reshape(
            batch_size,
            num_nodes,
            num_features * num_times,
        )

        diffused = [x0]
        for support in supports:
            x_k = torch.bmm(support, x0)
            diffused.append(x_k)
            for _ in range(2, order + 1):
                x_k = torch.bmm(support, x_k)
                diffused.append(x_k)

        stacked = torch.stack(diffused, dim=0)
        stacked = stacked.reshape(
            2 * order + 1,
            batch_size,
            num_nodes,
            num_features,
            num_times,
        )
        stacked = stacked.permute(1, 4, 2, 3, 0)
        return stacked.reshape(
            batch_size,
            num_times,
            num_nodes,
            num_features * (2 * order + 1),
        )

    def _project(
        self,
        features: torch.Tensor,
        theta: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        projected = torch.einsum("btni,io->btno", features, theta)
        return projected + bias

    def forward(
        self,
        x: torch.Tensor,
        forward_adj: torch.Tensor,
        backward_adj: torch.Tensor,
        adaptive_adj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        static_features = self._diffuse(
            x=x,
            supports=[forward_adj, backward_adj],
            order=self.order,
        )
        static_out = self._project(
            static_features,
            self.theta_static,
            self.bias_static,
        )

        if adaptive_adj is not None:
            adaptive_features = self._diffuse(
                x=x,
                supports=[adaptive_adj, adaptive_adj],
                order=self.order,
            )
            adaptive_out_1 = self._project(
                adaptive_features,
                self.theta_static,
                self.bias_static,
            )
            adaptive_out_2 = self._project(
                adaptive_features,
                self.theta_adaptive,
                self.bias_adaptive,
            )
            out = self.fusion(torch.cat([static_out, adaptive_out_1, adaptive_out_2], dim=-1))
        else:
            out = static_out

        if self.activation == "relu":
            out = F.relu(out)
        elif self.activation == "selu":
            out = F.selu(out)
        elif self.activation != "linear":
            raise ValueError(f"Unsupported activation '{self.activation}'.")

        return out.permute(0, 3, 1, 2)


class ChannelAlign(nn.Module):
    """Align channel counts for residual temporal convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1x1 = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels > out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv1x1 is not None:
            return self.conv1x1(x)
        if self.in_channels < self.out_channels:
            return F.pad(x, [0, 0, 0, 0, 0, self.out_channels - self.in_channels, 0, 0])
        return x


class TemporalConvolution(nn.Module):
    """Same-length temporal convolution over `(batch, channels, time, nodes)`."""

    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        activation: str = "linear",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        self.kernel_size = int(kernel_size)
        self.activation = activation
        self.out_channels = out_channels
        self.align = ChannelAlign(in_channels, out_channels)
        conv_out_channels = out_channels * 2 if activation == "glu" else out_channels
        self.conv = nn.Conv2d(
            in_channels,
            conv_out_channels,
            kernel_size=(self.kernel_size, 1),
            padding=(self.kernel_size // 2, 0),
        )
        self.dropout = float(dropout)

    def _same_time_length(self, x: torch.Tensor, target_time: int) -> torch.Tensor:
        if x.shape[2] > target_time:
            return x[:, :, :target_time, :]
        if x.shape[2] < target_time:
            return F.pad(x, [0, 0, 0, target_time - x.shape[2], 0, 0, 0, 0])
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.align(x)
        conv = self._same_time_length(self.conv(x), residual.shape[2])

        if self.activation == "glu":
            linear_part = conv[:, : self.out_channels, :, :]
            gate = conv[:, self.out_channels :, :, :]
            out = (linear_part + residual) * torch.sigmoid(gate)
        elif self.activation == "sigmoid":
            out = torch.sigmoid(conv + residual)
        elif self.activation == "linear":
            out = conv + residual
        else:
            raise ValueError(f"Unsupported temporal activation '{self.activation}'.")

        return F.dropout(out, self.dropout, training=self.training)


class PatternAdaptiveAdjacency(nn.Module):
    """Learn a batch-specific adjacency from speed patterns."""

    def __init__(self, time_len: int, hidden_channels: int, epsilon: float = 0.1) -> None:
        super().__init__()
        if time_len <= 0 or hidden_channels <= 0:
            raise ValueError("time_len and hidden_channels must be positive.")
        self.epsilon = float(epsilon)
        self.linear1 = nn.Linear(time_len, hidden_channels, bias=False)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, speed_data: torch.Tensor, zero_diagonal: bool = False) -> torch.Tensor:
        if speed_data.ndim != 3:
            raise ValueError(
                "speed_data must have shape (batch, time, nodes); "
                f"got {tuple(speed_data.shape)}."
            )

        speed_features = speed_data.transpose(1, 2)
        speed_features = self.activation(self.linear1(speed_features))
        speed_features = self.activation(self.linear2(speed_features))

        numerator = torch.bmm(speed_features, speed_features.transpose(1, 2))
        norms = torch.linalg.norm(speed_features, dim=2, keepdim=True).clamp_min(1e-8)
        denominator = torch.bmm(norms, norms.transpose(1, 2))
        adjacency = numerator / denominator

        if zero_diagonal:
            adjacency = adjacency - torch.diag_embed(
                torch.diagonal(adjacency, dim1=-2, dim2=-1)
            )
        adjacency = torch.where(
            adjacency > self.epsilon,
            adjacency,
            torch.zeros_like(adjacency),
        )
        return adjacency


class STCAGCN(nn.Module):
    """Spatiotemporal correlation-adaptive GCN for volume reconstruction."""

    def __init__(
        self,
        time_len: int,
        order: int = 1,
        in_channels: int = 1,
        hidden_channels: int = 128,
        temporal_kernel_size: int = 3,
        adaptive_type: str = "pam",
        use_spam: bool = True,
    ) -> None:
        super().__init__()
        self.time_dimension = int(time_len)
        self.hidden_dimension = int(hidden_channels)
        self.order = int(order)
        self.in_channels = int(in_channels)
        self.adaptive_type = adaptive_type
        self.use_spam = bool(use_spam)

        self.gnn1 = DiffusionGraphConvolution(in_channels, hidden_channels, order)
        self.temporal = TemporalConvolution(
            temporal_kernel_size,
            hidden_channels,
            hidden_channels,
            dropout=0.0,
        )
        self.gnn2 = DiffusionGraphConvolution(hidden_channels, hidden_channels, order)
        self.gnn3 = DiffusionGraphConvolution(hidden_channels, hidden_channels, order)
        self.gnn4 = DiffusionGraphConvolution(hidden_channels, hidden_channels, order)
        self.gnn5 = DiffusionGraphConvolution(hidden_channels, hidden_channels, order)
        self.readout = nn.Linear(hidden_channels * 5, in_channels)

        if self.use_spam:
            if adaptive_type != "pam":
                raise ValueError("Only adaptive_type='pam' is currently supported.")
            self.pattern_adjacency = PatternAdaptiveAdjacency(
                time_len=time_len,
                hidden_channels=hidden_channels,
            )
        else:
            self.pattern_adjacency = None

    @staticmethod
    def _expand_adjacency(adj: torch.Tensor, batch_size: int) -> torch.Tensor:
        if adj.ndim == 2:
            return adj.unsqueeze(0).expand(batch_size, -1, -1)
        if adj.ndim == 3:
            if adj.shape[0] != batch_size:
                raise ValueError(
                    f"Batch adjacency has batch size {adj.shape[0]}, expected {batch_size}."
                )
            return adj
        raise ValueError(f"Adjacency must be 2D or 3D; got {tuple(adj.shape)}.")

    def forward(
        self,
        x: torch.Tensor,
        forward_adj: torch.Tensor,
        backward_adj: torch.Tensor,
        first_forward_adj: torch.Tensor,
        first_backward_adj: torch.Tensor,
        speed_data: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if x.ndim != 4:
            raise ValueError(
                "x must have shape (batch, channels, time, nodes); "
                f"got {tuple(x.shape)}."
            )
        if x.shape[2] != self.time_dimension:
            raise ValueError(
                f"x has time dimension {x.shape[2]}, expected {self.time_dimension}."
            )

        batch_size = x.shape[0]
        first_forward = self._expand_adjacency(first_forward_adj, batch_size)
        first_backward = self._expand_adjacency(first_backward_adj, batch_size)
        forward = self._expand_adjacency(forward_adj, batch_size)
        backward = self._expand_adjacency(backward_adj, batch_size)

        if self.use_spam:
            if speed_data is None:
                raise ValueError("speed_data is required when use_spam=True.")
            adaptive_zero_diag = F.softmax(
                self.pattern_adjacency(speed_data, zero_diagonal=True),
                dim=-1,
            )
            adaptive_with_diag_raw = self.pattern_adjacency(speed_data, zero_diagonal=False)
            adaptive_with_diag = F.softmax(adaptive_with_diag_raw, dim=-1)
        else:
            adaptive_zero_diag = None
            adaptive_with_diag_raw = None
            adaptive_with_diag = None

        x_s = self.gnn1(x, first_forward, first_backward, adaptive_zero_diag)
        x_t = self.temporal(x_s)
        x_2 = self.gnn2(x_t, forward, backward, adaptive_with_diag) + x_t
        x_3 = self.gnn3(x_2, forward, backward, adaptive_with_diag) + x_2
        x_4 = self.gnn4(x_3, forward, backward, adaptive_with_diag) + x_3
        x_5 = self.gnn5(x_4, forward, backward, adaptive_with_diag) + x_4

        stacked = torch.stack([x_s, x_2, x_3, x_4, x_5], dim=-1)
        stacked = stacked.permute(0, 2, 3, 1, 4).reshape(
            batch_size,
            self.time_dimension,
            x.shape[-1],
            self.hidden_dimension * 5,
        )
        reconstructed = self.readout(stacked).permute(0, 3, 1, 2)
        return reconstructed, adaptive_with_diag_raw, adaptive_zero_diag


def _as_space_time_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D with shape (space, time); got {arr.shape}.")
    return arr


def _fill_missing_with_profile(matrix: np.ndarray) -> np.ndarray:
    """Fill NaNs with row means, then column means, then a global mean."""
    arr = np.asarray(matrix, dtype=float)
    filled = arr.copy()
    if np.all(np.isnan(filled)):
        raise ValueError("Cannot fill a matrix that is entirely NaN.")

    with np.errstate(invalid="ignore"):
        row_means = np.nanmean(filled, axis=1)
    nan_rows, nan_cols = np.where(np.isnan(filled))
    if nan_rows.size:
        row_values = row_means[nan_rows]
        row_valid = np.isfinite(row_values)
        filled[nan_rows[row_valid], nan_cols[row_valid]] = row_values[row_valid]

    with np.errstate(invalid="ignore"):
        col_means = np.nanmean(filled, axis=0)
    nan_rows, nan_cols = np.where(np.isnan(filled))
    if nan_rows.size:
        col_values = col_means[nan_cols]
        col_valid = np.isfinite(col_values)
        filled[nan_rows[col_valid], nan_cols[col_valid]] = col_values[col_valid]

    if np.isnan(filled).any():
        filled[np.isnan(filled)] = float(np.nanmean(arr))
    return filled


def _normalization_scale(matrix: np.ndarray, explicit_scale: float | None, name: str) -> float:
    if explicit_scale is not None:
        scale = float(explicit_scale)
    else:
        finite = np.asarray(matrix, dtype=float)[np.isfinite(matrix)]
        if finite.size == 0:
            raise ValueError(f"Cannot infer {name}; matrix has no finite values.")
        scale = float(np.nanmax(np.abs(finite)))

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} must be positive and finite; got {scale}.")
    return scale


def prepare_gnn_inputs(
    target_window: np.ndarray,
    speed_window: np.ndarray,
    observed_mask: np.ndarray | None = None,
    target_scale: float = 1.0,
    speed_scale: float = 1.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert `(time, nodes)` numpy windows into STCAGCN tensors.

    Returns `(x, speed, label)` where:
    - `x`: `(1, 1, time, nodes)`, normalized and zeroed where unobserved
    - `speed`: `(1, time, nodes)`, normalized side information
    - `label`: `(1, 1, time, nodes)`, normalized target values
    """
    target = np.asarray(target_window, dtype=np.float32)
    speed = np.asarray(speed_window, dtype=np.float32)
    if target.ndim != 2:
        raise ValueError(f"target_window must be 2D with shape (time, nodes); got {target.shape}.")
    if speed.shape != target.shape:
        raise ValueError(
            "speed_window must have the same shape as target_window; "
            f"got {speed.shape} and {target.shape}."
        )

    if observed_mask is None:
        mask = np.isfinite(target)
    else:
        mask = np.asarray(observed_mask, dtype=bool)
        if mask.shape != target.shape:
            raise ValueError(
                "observed_mask must have the same shape as target_window; "
                f"got {mask.shape} and {target.shape}."
            )
        mask = mask & np.isfinite(target)

    filled_target = _fill_missing_with_profile(target)
    filled_speed = _fill_missing_with_profile(speed)
    model_input = np.where(mask, filled_target, 0.0) / float(target_scale)
    labels = filled_target / float(target_scale)
    speed_input = filled_speed / float(speed_scale)

    torch_device = torch.device("cpu") if device is None else torch.device(device)
    x_tensor = torch.from_numpy(model_input[None, None, :, :].astype(np.float32)).to(torch_device)
    speed_tensor = torch.from_numpy(speed_input[None, :, :].astype(np.float32)).to(torch_device)
    label_tensor = torch.from_numpy(labels[None, None, :, :].astype(np.float32)).to(torch_device)
    return x_tensor, speed_tensor, label_tensor


def _prepare_adjacency_tensors(
    num_nodes: int,
    device: torch.device,
    adjacency: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if adjacency is None:
        adj = build_directed_chain_adjacency(num_nodes=num_nodes, self_loops=True)
    else:
        adj = np.asarray(adjacency, dtype=np.float32)
        if adj.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"adjacency must have shape {(num_nodes, num_nodes)}; got {adj.shape}."
            )

    first_adj = adj.copy()
    np.fill_diagonal(first_adj, 0.0)

    forward = torch.from_numpy(calculate_asymmetric_random_walk(adj)).to(device)
    backward = torch.from_numpy(calculate_asymmetric_random_walk(adj.T)).to(device)
    first_forward = torch.from_numpy(calculate_asymmetric_random_walk(first_adj)).to(device)
    first_backward = torch.from_numpy(calculate_asymmetric_random_walk(first_adj.T)).to(device)
    return forward, backward, first_forward, first_backward


def _sample_training_batch(
    target_matrix: np.ndarray,
    speed_matrix: np.ndarray,
    seq_length: int,
    batch_size: int,
    mask_fraction: float,
    target_scale: float,
    speed_scale: float,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes, num_times = target_matrix.shape
    if seq_length > num_times:
        raise ValueError(
            f"seq_length={seq_length} cannot exceed matrix time dimension {num_times}."
        )

    max_start = num_times - seq_length
    starts = rng.integers(0, max_start + 1, size=batch_size)
    x_batch = []
    speed_batch = []
    label_batch = []
    hidden_loss_mask_batch = []

    for start in starts:
        stop = int(start) + seq_length
        target_window = target_matrix[:, start:stop].T
        speed_window = speed_matrix[:, start:stop].T
        finite_mask = np.isfinite(target_window)

        observed_mask = finite_mask.copy()
        hidden_training_mask = np.zeros_like(finite_mask, dtype=bool)
        eligible_nodes = np.flatnonzero(np.any(finite_mask, axis=0))
        num_masked = int(round(eligible_nodes.size * mask_fraction))
        if num_masked > 0:
            masked_nodes = rng.choice(eligible_nodes, size=num_masked, replace=False)
            observed_mask[:, masked_nodes] = False
            hidden_training_mask[:, masked_nodes] = finite_mask[:, masked_nodes]
        else:
            hidden_training_mask = finite_mask

        x_tensor, speed_tensor, label_tensor = prepare_gnn_inputs(
            target_window=target_window,
            speed_window=speed_window,
            observed_mask=observed_mask,
            target_scale=target_scale,
            speed_scale=speed_scale,
            device=device,
        )
        loss_mask = torch.from_numpy(
            hidden_training_mask[None, None, :, :].astype(np.float32)
        ).to(device)
        x_batch.append(x_tensor)
        speed_batch.append(speed_tensor)
        label_batch.append(label_tensor)
        hidden_loss_mask_batch.append(loss_mask)

    return (
        torch.cat(x_batch, dim=0),
        torch.cat(speed_batch, dim=0),
        torch.cat(label_batch, dim=0),
        torch.cat(hidden_loss_mask_batch, dim=0),
    )


def train_stcagcn_for_matrix(
    target_matrix: np.ndarray,
    velocity_matrix: np.ndarray | None = None,
    seq_length: int = 24,
    hidden_channels: int = 128,
    order: int = 1,
    temporal_kernel_size: int = 3,
    batch_size: int = 32,
    max_epochs: int = 300,
    batches_per_epoch: int | None = None,
    learning_rate: float = 5e-4,
    mask_fraction: float = 0.5,
    target_scale: float | None = None,
    speed_scale: float | None = None,
    adjacency: np.ndarray | None = None,
    min_value: float | None = 0.0,
    max_value: float | None = None,
    device: torch.device | str | None = None,
    rng: np.random.Generator | int | None = None,
    verbose: bool = False,
    log_every_epochs: int = 1,
    use_spam: bool = True,
) -> tuple[STCAGCN, dict[str, object]]:
    """Train one STCAGCN model for one `(space, time)` target matrix."""
    target = _as_space_time_matrix(target_matrix, "target_matrix")
    if use_spam:
        if velocity_matrix is None:
            raise ValueError("velocity_matrix is required when use_spam=True.")
        speed = _as_space_time_matrix(velocity_matrix, "velocity_matrix")
        if speed.shape != target.shape:
            raise ValueError(
                "velocity_matrix must have the same shape as target_matrix; "
                f"got {speed.shape} and {target.shape}."
            )
    else:
        speed = np.zeros_like(target, dtype=float)
    if not 0.0 <= mask_fraction < 1.0:
        raise ValueError("mask_fraction must be in [0, 1).")
    if batch_size <= 0 or max_epochs <= 0:
        raise ValueError("batch_size and max_epochs must be positive.")
    if log_every_epochs <= 0:
        raise ValueError("log_every_epochs must be positive.")

    target_scale_value = _normalization_scale(target, target_scale, "target_scale")
    speed_scale_value = (
        _normalization_scale(speed, speed_scale, "speed_scale") if use_spam else 1.0
    )
    torch_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    if batches_per_epoch is None:
        batches_per_epoch = max(1, target.shape[1] // max(seq_length * batch_size, 1))

    model = STCAGCN(
        time_len=seq_length,
        order=order,
        in_channels=1,
        hidden_channels=hidden_channels,
        temporal_kernel_size=temporal_kernel_size,
        use_spam=use_spam,
    ).to(torch_device)
    forward, backward, first_forward, first_backward = _prepare_adjacency_tensors(
        num_nodes=target.shape[0],
        device=torch_device,
        adjacency=adjacency,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    last_loss = float("nan")
    loss_history: list[dict[str, float | int]] = []

    model.train()
    step = 0
    for epoch in range(max_epochs):
        epoch_loss_sum = 0.0
        for batch_idx in range(batches_per_epoch):
            x_batch, speed_batch, label_batch, loss_mask = _sample_training_batch(
                target_matrix=target,
                speed_matrix=speed,
                seq_length=seq_length,
                batch_size=batch_size,
                mask_fraction=mask_fraction,
                target_scale=target_scale_value,
                speed_scale=speed_scale_value,
                rng=generator,
                device=torch_device,
            )
            optimizer.zero_grad()
            prediction, _, _ = model(
                x_batch,
                forward,
                backward,
                first_forward,
                first_backward,
                speed_batch,
            )
            absolute_error = torch.abs(prediction - label_batch) * loss_mask
            denominator = torch.clamp(loss_mask.sum(), min=1.0)
            loss = absolute_error.sum() / denominator
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            epoch_loss_sum += last_loss
            loss_history.append(
                {
                    "epoch": int(epoch),
                    "batch": int(batch_idx),
                    "step": int(step),
                    "loss": last_loss,
                }
            )
            step += 1
        if verbose and (
            epoch == 0
            or (epoch + 1) % log_every_epochs == 0
            or epoch == max_epochs - 1
        ):
            mean_epoch_loss = epoch_loss_sum / float(batches_per_epoch)
            print(
                "[STCAGCN] "
                f"epoch {epoch + 1}/{max_epochs} "
                f"mean_loss={mean_epoch_loss:.6g} "
                f"last_loss={last_loss:.6g}",
                flush=True,
            )

    training_info = {
        "target_scale": target_scale_value,
        "speed_scale": speed_scale_value,
        "training_loss": last_loss,
        "seq_length": int(seq_length),
        "loss_history": loss_history,
    }
    return model, training_info


def impute_with_stcagcn(
    model: STCAGCN,
    masked_matrix: np.ndarray,
    velocity_matrix: np.ndarray | None,
    target_scale: float,
    speed_scale: float,
    adjacency: np.ndarray | None = None,
    min_value: float | None = 0.0,
    max_value: float | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Impute a masked `(space, time)` matrix with a trained STCAGCN model."""
    masked = _as_space_time_matrix(masked_matrix, "masked_matrix")
    if model.use_spam:
        if velocity_matrix is None:
            raise ValueError("velocity_matrix is required when model.use_spam=True.")
        speed = _as_space_time_matrix(velocity_matrix, "velocity_matrix")
        if speed.shape != masked.shape:
            raise ValueError(
                "velocity_matrix must have the same shape as masked_matrix; "
                f"got {speed.shape} and {masked.shape}."
            )
    else:
        speed = np.zeros_like(masked, dtype=float)

    seq_length = model.time_dimension
    num_nodes, num_times = masked.shape
    if seq_length > num_times:
        raise ValueError(
            f"Model seq_length={seq_length} cannot exceed matrix time dimension {num_times}."
        )
    torch_device = next(model.parameters()).device if device is None else torch.device(device)
    model.to(torch_device)
    forward, backward, first_forward, first_backward = _prepare_adjacency_tensors(
        num_nodes=num_nodes,
        device=torch_device,
        adjacency=adjacency,
    )

    prediction_sum = np.zeros_like(masked, dtype=float)
    prediction_count = np.zeros_like(masked, dtype=float)
    starts = list(range(0, max(num_times - seq_length + 1, 1), seq_length))
    if starts[-1] != num_times - seq_length:
        starts.append(num_times - seq_length)
    starts = [start for start in starts if start >= 0]

    model.eval()
    with torch.no_grad():
        for start in starts:
            stop = start + seq_length
            target_window = masked[:, start:stop].T
            speed_window = speed[:, start:stop].T
            observed_mask = np.isfinite(target_window)
            x_tensor, speed_tensor, _ = prepare_gnn_inputs(
                target_window=target_window,
                speed_window=speed_window,
                observed_mask=observed_mask,
                target_scale=target_scale,
                speed_scale=speed_scale,
                device=torch_device,
            )
            prediction, _, _ = model(
                x_tensor,
                forward,
                backward,
                first_forward,
                first_backward,
                speed_tensor,
            )
            pred_window = prediction.squeeze(0).squeeze(0).cpu().numpy().T * target_scale
            prediction_sum[:, start:stop] += pred_window
            prediction_count[:, start:stop] += 1.0

    imputed = prediction_sum / np.maximum(prediction_count, 1.0)
    observed = np.isfinite(masked)
    imputed[observed] = masked[observed]
    if min_value is not None or max_value is not None:
        imputed = np.clip(
            imputed,
            -np.inf if min_value is None else float(min_value),
            np.inf if max_value is None else float(max_value),
        )
    return imputed


def save_gnn_training_loss_history(
    loss_history: list[dict[str, float | int]],
    output_path: str | Path,
    mask_index: int | None = None,
) -> Path:
    """Save a STCAGCN training loss history to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["epoch", "batch", "step", "loss"]
    if mask_index is not None:
        fieldnames = ["mask_index", *fieldnames]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in loss_history:
            output_row = dict(row)
            if mask_index is not None:
                output_row = {"mask_index": int(mask_index), **output_row}
            writer.writerow(output_row)
    return path


def evaluate_gnn_kriging_on_masks(
    ground_truth: np.ndarray,
    masked_matrices: list[np.ndarray],
    velocity_matrix: np.ndarray | None = None,
    seq_length: int = 24,
    hidden_channels: int = 128,
    order: int = 1,
    temporal_kernel_size: int = 3,
    batch_size: int = 32,
    max_epochs: int = 300,
    batches_per_epoch: int | None = None,
    learning_rate: float = 5e-4,
    mask_fraction: float | None = None,
    adjacency: np.ndarray | None = None,
    min_value: float | None = 0.0,
    max_value: float | None = None,
    device: torch.device | str | None = None,
    rng: np.random.Generator | int | None = None,
    training_history_output_dir: str | Path | None = None,
    training_history_prefix: str = "gnn_training_loss",
    verbose: bool = False,
    log_every_epochs: int = 1,
    use_spam: bool = True,
    return_imputed_matrices: bool = False,
) -> list[dict[str, float]] | tuple[list[dict[str, float]], list[np.ndarray]]:
    """
    Train and score STCAGCN over masked matrices.

    A fresh model is trained for each mask realization. During training, rows
    that are hidden in that realization are removed from the target matrix, so
    the model cannot learn their target values before inference. The complete
    velocity matrix is still used as side information for adaptive adjacency.
    """
    truth = _as_space_time_matrix(ground_truth, "ground_truth")
    if use_spam:
        if velocity_matrix is None:
            raise ValueError("velocity_matrix is required when use_spam=True.")
        speed = _as_space_time_matrix(velocity_matrix, "velocity_matrix")
        if speed.shape != truth.shape:
            raise ValueError(
                "velocity_matrix must have the same shape as ground_truth; "
                f"got {speed.shape} and {truth.shape}."
            )
    else:
        speed = None

    master_rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    all_metrics = []
    all_imputed = []
    for mask_idx, masked_matrix in enumerate(masked_matrices):
        masked = _as_space_time_matrix(masked_matrix, f"masked_matrices[{mask_idx}]")
        if masked.shape != truth.shape:
            raise ValueError(
                f"masked_matrices[{mask_idx}] must have shape {truth.shape}; "
                f"got {masked.shape}."
            )

        heldout_entries = np.isnan(masked) & np.isfinite(truth)
        heldout_rows = np.any(heldout_entries, axis=1)
        training_mask_fraction = (
            float(np.mean(heldout_rows)) if mask_fraction is None else float(mask_fraction)
        )
        training_target = truth.copy()
        training_target[heldout_rows, :] = np.nan
        training_speed = None
        if use_spam:
            training_speed = speed.copy()
            training_speed[heldout_rows, :] = np.nan
        if not np.any(np.isfinite(training_target)):
            raise ValueError(
                f"Mask {mask_idx} leaves no finite target values for GNN training."
            )
        if use_spam and not np.any(np.isfinite(training_speed)):
            raise ValueError(
                f"Mask {mask_idx} leaves no finite speed values for GNN training."
            )

        if verbose:
            print(
                "[STCAGCN] "
                f"training mask {mask_idx + 1}/{len(masked_matrices)} "
                f"with train_mask_fraction={training_mask_fraction:.3f}",
                flush=True,
            )
        model, training_info = train_stcagcn_for_matrix(
            target_matrix=training_target,
            velocity_matrix=training_speed,
            seq_length=seq_length,
            hidden_channels=hidden_channels,
            order=order,
            temporal_kernel_size=temporal_kernel_size,
            batch_size=batch_size,
            max_epochs=max_epochs,
            batches_per_epoch=batches_per_epoch,
            learning_rate=learning_rate,
            mask_fraction=training_mask_fraction,
            adjacency=adjacency,
            min_value=min_value,
            max_value=max_value,
            device=device,
            rng=master_rng,
            verbose=verbose,
            log_every_epochs=log_every_epochs,
            use_spam=use_spam,
        )
        if training_history_output_dir is not None:
            save_gnn_training_loss_history(
                loss_history=training_info["loss_history"],  # type: ignore[arg-type]
                output_path=(
                    Path(training_history_output_dir)
                    / f"{training_history_prefix}_mask_{mask_idx}.csv"
                ),
                mask_index=mask_idx,
            )
        imputed = impute_with_stcagcn(
            model=model,
            masked_matrix=masked,
            velocity_matrix=speed,
            target_scale=float(training_info["target_scale"]),
            speed_scale=float(training_info["speed_scale"]),
            adjacency=adjacency,
            min_value=min_value,
            max_value=max_value,
            device=device,
        )
        metrics = {
            "mask_index": int(mask_idx),
            "mae": masked_mae(truth, imputed, masked),
            "mape": masked_mape(truth, imputed, masked),
            "rmse": masked_rmse(truth, imputed, masked),
        }
        all_metrics.append(metrics)
        if return_imputed_matrices:
            all_imputed.append(imputed)

    if return_imputed_matrices:
        return all_metrics, all_imputed
    return all_metrics


def _load_loss_history_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if "step" not in row or "loss" not in row:
                raise ValueError(f"Expected 'step' and 'loss' columns in '{path}'.")
            parsed = {
                "step": float(row["step"]),
                "loss": float(row["loss"]),
            }
            if "mask_index" in row and row["mask_index"] != "":
                parsed["mask_index"] = float(row["mask_index"])
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No loss rows found in '{path}'.")
    return rows


def plot_gnn_training_loss_curve(
    loss_history: list[dict[str, float | int]] | str | Path,
    ax: object | None = None,
    title: str = "GNN Training Loss",
    label: str | None = None,
    log_y: bool = False,
) -> object:
    """
    Plot a STCAGCN training loss curve.

    `loss_history` can be either the `training_info["loss_history"]` returned by
    `train_stcagcn_for_matrix(...)` or a CSV path written by
    `save_gnn_training_loss_history(...)` / `evaluate_gnn_kriging_on_masks(...)`.
    """
    import matplotlib.pyplot as plt

    if isinstance(loss_history, (str, Path)):
        history_rows = _load_loss_history_csv(Path(loss_history))
    else:
        history_rows = [dict(row) for row in loss_history]
        if not history_rows:
            raise ValueError("loss_history is empty.")

    steps = np.asarray([float(row["step"]) for row in history_rows], dtype=float)
    losses = np.asarray([float(row["loss"]) for row in history_rows], dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.plot(steps, losses, linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("MAE loss")
    if log_y:
        ax.set_yscale("log")
    if label is not None:
        ax.legend()
    ax.grid(True, alpha=0.25)
    return ax
