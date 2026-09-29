"""Five unlabeled tabular anomaly scorers used by the submitted experiment.

The classes in this module deliberately expose the same compact interface as the
classical scorers in :mod:`early_prediction.models`::

    fit(X) -> self
    score_samples(X) -> one anomaly/risk score per row

No class accepts failure labels.  Larger scores always mean more anomalous and,
in the downstream experiment, higher future-failure risk.

The formal model set is Isolation Forest, PCA, AE, Deep SVDD and NeuTraL-AD.
All consume N x D feature matrices; labels are never accepted by fitting APIs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised only without torch
    torch = None
    class _NNFallback:
        Module = object
    nn = _NNFallback()
    DataLoader = None
    TensorDataset = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


EPS = 1e-12


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "Deep learning anomaly models require PyTorch. Install it before requesting these models."
        ) from _TORCH_IMPORT_ERROR


def _as_2d_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"features must be 2D, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return arr


def _sanitize_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if np.all(np.isfinite(values)):
        return values
    finite = values[np.isfinite(values)]
    high = float(np.max(finite)) if finite.size else 0.0
    low = float(np.min(finite)) if finite.size else 0.0
    return np.nan_to_num(values, nan=high, posinf=high, neginf=low)


def _resolve_device(requested: str) -> "torch.device":
    _require_torch()
    name = str(requested).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={requested!r} requested but CUDA is unavailable")
    return torch.device(name)


def _set_torch_seed(seed: int, torch_num_threads: int = 0) -> None:
    _require_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if int(torch_num_threads) > 0:
        torch.set_num_threads(int(torch_num_threads))


def _activation(name: str) -> "nn.Module":
    _require_torch()
    kind = str(name).lower()
    if kind == "relu":
        return nn.ReLU()
    if kind == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    if kind == "tanh":
        return nn.Tanh()
    if kind == "gelu":
        return nn.GELU()
    raise ValueError("activation must be one of: relu, leaky_relu, tanh, gelu")


def _normalise_hidden_dims(
    hidden_dims: Optional[Sequence[int]],
    n_features: int,
) -> list[int]:
    if hidden_dims is None:
        first = max(8, min(128, int(n_features) * 2))
        second = max(4, min(64, int(n_features)))
        dims = [first, second]
    else:
        dims = [int(v) for v in hidden_dims]
    if not dims or any(v <= 0 for v in dims):
        raise ValueError("hidden_dims must contain positive integers")
    return dims


def _make_loader(
    x: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> "DataLoader":
    _require_torch()
    tensor = torch.from_numpy(_as_2d_float(x))
    dataset = TensorDataset(tensor)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=max(1, min(int(batch_size), len(dataset))),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        generator=generator if shuffle else None,
    )


class _MLPAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(latent_dim) <= 0:
            raise ValueError("latent_dim must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        encoder_layers: list[nn.Module] = []
        previous = int(n_features)
        for width in hidden_dims:
            encoder_layers.append(nn.Linear(previous, int(width)))
            encoder_layers.append(_activation(activation))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(float(dropout)))
            previous = int(width)
        encoder_layers.append(nn.Linear(previous, int(latent_dim)))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        previous = int(latent_dim)
        reversed_hidden = list(reversed([int(v) for v in hidden_dims]))
        for width in reversed_hidden:
            decoder_layers.append(nn.Linear(previous, width))
            decoder_layers.append(_activation(activation))
            if dropout > 0:
                decoder_layers.append(nn.Dropout(float(dropout)))
            previous = width
        decoder_layers.append(nn.Linear(previous, int(n_features)))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z, reconstruction


class AutoencoderScorer:
    """Feed-forward autoencoder anomaly scorer for N x D data.

    Failure labels are never used.  The network is fitted on the unlabeled
    reference pool and each sample is scored by its reconstruction residual.
    """

    def __init__(
        self,
        hidden_dims: Optional[Sequence[int]] = None,
        latent_dim: int = 8,
        activation: str = "relu",
        dropout: float = 0.0,
        epochs: int = 30,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        gradient_clip: float = 5.0,
        score_mode: str = "mse",
        device: str = "auto",
        random_state: int = 0,
        torch_num_threads: int = 0,
        verbose: bool = False,
    ) -> None:
        _require_torch()
        self.hidden_dims = None if hidden_dims is None else tuple(int(v) for v in hidden_dims)
        self.latent_dim = int(latent_dim)
        self.activation = str(activation)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = float(gradient_clip)
        self.score_mode = str(score_mode).lower()
        self.device_name = str(device)
        self.random_state = int(random_state)
        self.torch_num_threads = int(torch_num_threads)
        self.verbose = bool(verbose)

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

            raise ValueError("learning_rate must be positive")
        if self.score_mode not in {"mse", "standardized_mse", "max_residual"}:
            raise ValueError(
                "AE score_mode must be one of: mse, standardized_mse, max_residual"
            )

        self.network: Optional[_MLPAutoencoder] = None
        self.device_: Optional[torch.device] = None
        self.residual_variance_: Optional[np.ndarray] = None
        self.training_history_: list[dict[str, float]] = []
        self.n_features_in_: Optional[int] = None

    def fit(self, x: np.ndarray) -> "AutoencoderScorer":
        values = _as_2d_float(x)
        if len(values) < 2:
            raise ValueError("AE requires at least 2 training samples")
        _set_torch_seed(self.random_state, self.torch_num_threads)
        self.device_ = _resolve_device(self.device_name)
        self.n_features_in_ = int(values.shape[1])
        hidden = _normalise_hidden_dims(self.hidden_dims, self.n_features_in_)
        self.network = _MLPAutoencoder(
            n_features=self.n_features_in_,
            hidden_dims=hidden,
            latent_dim=min(self.latent_dim, max(1, hidden[-1])),
            activation=self.activation,
            dropout=self.dropout,
        ).to(self.device_)

        optimiser = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loader = _make_loader(values, self.batch_size, shuffle=True, seed=self.random_state)
        self.training_history_ = []
        self.network.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            seen = 0
            for (batch_cpu,) in loader:
                batch = batch_cpu.to(self.device_, dtype=torch.float32)
                optimiser.zero_grad(set_to_none=True)
                _, reconstruction = self.network(batch)
                loss = torch.mean((batch - reconstruction) ** 2)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"AE produced non-finite loss at epoch {epoch + 1}")
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.gradient_clip
                    )
                optimiser.step()
                total_loss += float(loss.detach().cpu()) * len(batch)
                seen += len(batch)
            mean_loss = total_loss / max(1, seen)
            self.training_history_.append(
                {
                    "epoch": float(epoch + 1),
                    "total_loss": float(mean_loss),
                    "reconstruction_loss": float(mean_loss),
                    "learning_rate": float(optimiser.param_groups[0]["lr"]),
                }
            )
            if self.verbose:
                print(f"[AE] epoch={epoch + 1}/{self.epochs} loss={mean_loss:.6g}")

        # Fit residual standardisation only from the same unlabeled training data.
        residual_sq = self._residual_squared(values)
        variance = np.mean(residual_sq, axis=0, dtype=np.float64)
        positive = variance[np.isfinite(variance) & (variance > 0.0)]
        reference = float(np.median(positive)) if positive.size else 1.0
        floor = max(reference * 1e-6, 1e-12)
        self.residual_variance_ = np.maximum(
            np.nan_to_num(variance, nan=reference, posinf=reference, neginf=reference),
            floor,
        )
        self.network.eval()
        return self

    def _residual_squared(self, x: np.ndarray) -> np.ndarray:
        if self.network is None or self.device_ is None:
            raise RuntimeError("AE has not been fitted")
        values = _as_2d_float(x)
        output: list[np.ndarray] = []
        loader = _make_loader(values, self.batch_size, shuffle=False, seed=self.random_state)
        self.network.eval()
        with torch.no_grad():
            for (batch_cpu,) in loader:
                batch = batch_cpu.to(self.device_, dtype=torch.float32)
                _, reconstruction = self.network(batch)
                residual = (batch - reconstruction) ** 2
                output.append(residual.detach().cpu().numpy().astype(np.float64))
        return np.vstack(output) if output else np.empty((0, values.shape[1]), dtype=np.float64)

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        residual_sq = self._residual_squared(x)
        if self.score_mode == "mse":
            scores = np.mean(residual_sq, axis=1)
        else:
            if self.residual_variance_ is None:
                raise RuntimeError("AE residual variance is unavailable")
            standardized = residual_sq / self.residual_variance_[None, :]
            if self.score_mode == "standardized_mse":
                scores = np.mean(standardized, axis=1)
            else:
                scores = np.max(standardized, axis=1)
        return _sanitize_scores(scores)



def _loader(x: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> "DataLoader":
    _require_torch()
    tensor = torch.from_numpy(_as_2d_float(x))
    ds = TensorDataset(tensor)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=max(1, min(int(batch_size), len(ds))),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        generator=generator if shuffle else None,
    )


class _SVDDEncoder(nn.Module):
    """Bias-free MLP encoder used to reduce trivial constant solutions."""

    def __init__(
        self,
        n_features: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        dims = [int(n_features), *[int(v) for v in hidden_dims], int(latent_dim)]
        if any(v <= 0 for v in dims):
            raise ValueError("Deep SVDD dimensions must be positive")
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if i < len(dims) - 2:
                layers.append(_activation(activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class _SVDDPretrainAE(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.encoder = _SVDDEncoder(n_features, hidden_dims, latent_dim, activation)
        decoder_dims = [int(latent_dim), *list(reversed([int(v) for v in hidden_dims])), int(n_features)]
        layers: list[nn.Module] = []
        for i in range(len(decoder_dims) - 1):
            layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1], bias=False))
            if i < len(decoder_dims) - 2:
                layers.append(_activation(activation))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        z = self.encoder(x)
        return z, self.decoder(z)


class DeepSVDDScorer:
    """Deep Support Vector Data Description scorer for N x D tabular data."""

    def __init__(
        self,
        hidden_dims: Optional[Sequence[int]] = None,
        latent_dim: int = 16,
        activation: str = "leaky_relu",
        pretrain_epochs: int = 10,
        epochs: int = 30,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        pretrain_learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        gradient_clip: float = 5.0,
        center_epsilon: float = 0.1,
        device: str = "auto",
        random_state: int = 0,
        torch_num_threads: int = 0,
        verbose: bool = False,
    ) -> None:
        _require_torch()
        self.hidden_dims = None if hidden_dims is None else tuple(int(v) for v in hidden_dims)
        self.latent_dim = int(latent_dim)
        self.activation = str(activation)
        self.pretrain_epochs = int(pretrain_epochs)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.pretrain_learning_rate = float(pretrain_learning_rate)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = float(gradient_clip)
        self.center_epsilon = float(center_epsilon)
        self.device_name = str(device)
        self.random_state = int(random_state)
        self.torch_num_threads = int(torch_num_threads)
        self.verbose = bool(verbose)
        if self.latent_dim <= 0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("latent_dim, epochs and batch_size must be positive")
        if self.pretrain_epochs < 0:
            raise ValueError("pretrain_epochs must be >= 0")
        if self.learning_rate <= 0 or self.pretrain_learning_rate <= 0:
            raise ValueError("learning rates must be positive")

        self.network: Optional[_SVDDEncoder] = None
        self.center_: Optional[np.ndarray] = None
        self.device_: Optional[torch.device] = None
        self.n_features_in_: Optional[int] = None
        self.training_history_: list[dict[str, float | str]] = []

    def _resolve_hidden(self, n_features: int) -> list[int]:
        if self.hidden_dims is not None:
            dims = [int(v) for v in self.hidden_dims]
        else:
            dims = [max(16, min(128, n_features * 2)), max(8, min(64, n_features))]
        if not dims or any(v <= 0 for v in dims):
            raise ValueError("hidden_dims must contain positive integers")
        return dims

    def _pretrain(self, values: np.ndarray, hidden: Sequence[int]) -> _SVDDEncoder:
        ae = _SVDDPretrainAE(
            self.n_features_in_, hidden, self.latent_dim, self.activation
        ).to(self.device_)
        opt = torch.optim.Adam(
            ae.parameters(),
            lr=self.pretrain_learning_rate,
            weight_decay=self.weight_decay,
        )
        loader = _loader(values, self.batch_size, True, self.random_state)
        for epoch in range(self.pretrain_epochs):
            ae.train()
            total = 0.0
            seen = 0
            for (cpu_batch,) in loader:
                batch = cpu_batch.to(self.device_, dtype=torch.float32)
                opt.zero_grad(set_to_none=True)
                _, recon = ae(batch)
                loss = torch.mean((recon - batch) ** 2)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Deep SVDD pretraining produced non-finite loss at epoch {epoch + 1}")
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ae.parameters(), self.gradient_clip)
                opt.step()
                total += float(loss.detach().cpu()) * len(batch)
                seen += len(batch)
            mean_loss = total / max(1, seen)
            self.training_history_.append({
                "phase": "pretrain",
                "epoch": float(epoch + 1),
                "total_loss": float(mean_loss),
                "reconstruction_loss": float(mean_loss),
                "learning_rate": float(opt.param_groups[0]["lr"]),
            })
            if self.verbose:
                print(f"[DeepSVDD pretrain] epoch={epoch + 1}/{self.pretrain_epochs} loss={mean_loss:.6g}")
        return ae.encoder

    def _initialize_center(self, values: np.ndarray) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("Deep SVDD encoder is unavailable")
        loader = _loader(values, self.batch_size, False, self.random_state)
        total: Optional[torch.Tensor] = None
        seen = 0
        self.network.eval()
        with torch.no_grad():
            for (cpu_batch,) in loader:
                batch = cpu_batch.to(self.device_, dtype=torch.float32)
                z = self.network(batch)
                batch_sum = z.sum(dim=0)
                total = batch_sum if total is None else total + batch_sum
                seen += len(batch)
        if total is None or seen == 0:
            raise ValueError("Cannot initialize Deep SVDD center from empty data")
        center = (total / float(seen)).detach().cpu().numpy().astype(np.float32)
        eps = abs(float(self.center_epsilon))
        if eps > 0:
            mask = np.abs(center) < eps
            signs = np.where(center < 0.0, -1.0, 1.0).astype(np.float32)
            center[mask] = signs[mask] * eps
        return center

    def fit(self, x: np.ndarray) -> "DeepSVDDScorer":
        values = _as_2d_float(x)
        if len(values) < 2:
            raise ValueError("Deep SVDD requires at least two training samples")
        _set_torch_seed(self.random_state, self.torch_num_threads)
        self.device_ = _resolve_device(self.device_name)
        self.n_features_in_ = int(values.shape[1])
        hidden = self._resolve_hidden(self.n_features_in_)
        self.training_history_ = []

        if self.pretrain_epochs > 0:
            self.network = self._pretrain(values, hidden).to(self.device_)
        else:
            self.network = _SVDDEncoder(
                self.n_features_in_, hidden, self.latent_dim, self.activation
            ).to(self.device_)

        self.center_ = self._initialize_center(values)
        center_t = torch.from_numpy(self.center_).to(self.device_, dtype=torch.float32)
        opt = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loader = _loader(values, self.batch_size, True, self.random_state + 17)
        for epoch in range(self.epochs):
            self.network.train()
            total = 0.0
            seen = 0
            for (cpu_batch,) in loader:
                batch = cpu_batch.to(self.device_, dtype=torch.float32)
                opt.zero_grad(set_to_none=True)
                z = self.network(batch)
                distances = torch.sum((z - center_t) ** 2, dim=1)
                loss = torch.mean(distances)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Deep SVDD produced non-finite loss at epoch {epoch + 1}")
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
                opt.step()
                total += float(loss.detach().cpu()) * len(batch)
                seen += len(batch)
            mean_loss = total / max(1, seen)
            self.training_history_.append({
                "phase": "svdd",
                "epoch": float(epoch + 1),
                "total_loss": float(mean_loss),
                "svdd_distance_loss": float(mean_loss),
                "learning_rate": float(opt.param_groups[0]["lr"]),
            })
            if self.verbose:
                print(f"[DeepSVDD] epoch={epoch + 1}/{self.epochs} loss={mean_loss:.6g}")
        self.network.eval()
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        if self.network is None or self.center_ is None or self.device_ is None:
            raise RuntimeError("Deep SVDD has not been fitted")
        values = _as_2d_float(x)
        center_t = torch.from_numpy(self.center_).to(self.device_, dtype=torch.float32)
        loader = _loader(values, self.batch_size, False, self.random_state)
        out: list[np.ndarray] = []
        self.network.eval()
        with torch.no_grad():
            for (cpu_batch,) in loader:
                batch = cpu_batch.to(self.device_, dtype=torch.float32)
                z = self.network(batch)
                d = torch.sum((z - center_t) ** 2, dim=1)
                out.append(d.detach().cpu().numpy().astype(np.float64))
        values_out = np.concatenate(out) if out else np.empty(0, dtype=np.float64)
        return _sanitize_scores(values_out)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("NeuTraL AD requires PyTorch") from _TORCH_IMPORT_ERROR


def _as_2d_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"features must be N x D, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return arr


def _sanitize_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if np.all(np.isfinite(values)):
        return values
    finite = values[np.isfinite(values)]
    hi = float(np.max(finite)) if finite.size else 0.0
    lo = float(np.min(finite)) if finite.size else 0.0
    return np.nan_to_num(values, nan=hi, posinf=hi, neginf=lo)


def _resolve_device(requested: str) -> "torch.device":
    _require_torch()
    name = str(requested).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device={requested!r} requested but CUDA is unavailable")
    return torch.device(name)


def _set_seed(seed: int, torch_num_threads: int = 0) -> None:
    _require_torch()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if int(torch_num_threads) > 0:
        torch.set_num_threads(int(torch_num_threads))


def _activation(name: str) -> "nn.Module":
    kind = str(name).lower()
    if kind == "relu":
        return nn.ReLU()
    if kind == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if kind == "gelu":
        return nn.GELU()
    if kind == "tanh":
        return nn.Tanh()
    raise ValueError("activation must be relu, leaky_relu, gelu, or tanh")


def _loader(x: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> "DataLoader":
    tensor = torch.from_numpy(_as_2d_float(x))
    ds = TensorDataset(tensor)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=max(1, min(int(batch_size), len(ds))),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        generator=generator if shuffle else None,
    )


class _MaskNet(nn.Module):
    """Bias-free multiplicative mask M_k(x); T_k(x)=M_k(x)*x."""

    def __init__(self, n_features: int, hidden_dim: int, depth: int = 3) -> None:
        super().__init__()
        depth = int(depth)
        if depth < 2:
            raise ValueError("transform_depth must be >= 2")
        dims = [int(n_features)] + [int(hidden_dim)] * (depth - 1) + [int(n_features)]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            layers.append(nn.ReLU() if i < len(dims) - 2 else nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class _Encoder(nn.Module):
    def __init__(self, n_features: int, hidden_dims: Sequence[int], latent_dim: int, activation: str) -> None:
        super().__init__()
        dims = [int(n_features), *[int(v) for v in hidden_dims], int(latent_dim)]
        if any(v <= 0 for v in dims):
            raise ValueError("encoder dimensions must be positive")
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if i < len(dims) - 2:
                layers.append(_activation(activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class _NeuTraLNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        activation: str,
        num_transforms: int,
        transform_hidden_dim: int,
        transform_depth: int,
    ) -> None:
        super().__init__()
        self.encoder = _Encoder(n_features, hidden_dims, latent_dim, activation)
        self.transforms = nn.ModuleList([
            _MaskNet(n_features, transform_hidden_dim, transform_depth)
            for _ in range(int(num_transforms))
        ])

    def views(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        z0 = self.encoder(x)
        transformed = [mask(x) * x for mask in self.transforms]
        zk = torch.stack([self.encoder(view) for view in transformed], dim=1)
        return z0, zk


class NeuTraLADScorer:
    """NeuTraL AD deterministic contrastive anomaly scorer for N x D data."""

    def __init__(
        self,
        hidden_dims: Optional[Sequence[int]] = None,
        latent_dim: int = 32,
        activation: str = "relu",
        num_transforms: int = 7,
        transform_hidden_dim: int = 64,
        transform_depth: int = 3,
        temperature: float = 0.1,
        epochs: int = 20,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        gradient_clip: float = 5.0,
        device: str = "auto",
        random_state: int = 0,
        torch_num_threads: int = 0,
        verbose: bool = False,
    ) -> None:
        _require_torch()
        self.hidden_dims = None if hidden_dims is None else tuple(int(v) for v in hidden_dims)
        self.latent_dim = int(latent_dim)
        self.activation = str(activation)
        self.num_transforms = int(num_transforms)
        self.transform_hidden_dim = int(transform_hidden_dim)
        self.transform_depth = int(transform_depth)
        self.temperature = float(temperature)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = float(gradient_clip)
        self.device_name = str(device)
        self.random_state = int(random_state)
        self.torch_num_threads = int(torch_num_threads)
        self.verbose = bool(verbose)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.num_transforms < 2:
            raise ValueError("num_transforms must be >= 2")
        if self.transform_hidden_dim <= 0 or self.transform_depth < 2:
            raise ValueError("invalid transformation architecture")
        if self.temperature <= 0 or self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("temperature/epochs/batch_size/learning_rate must be positive")
        self.network: Optional[_NeuTraLNet] = None
        self.device_: Optional[torch.device] = None
        self.n_features_in_: Optional[int] = None
        self.training_history_: list[dict[str, float]] = []

    def _resolve_hidden(self, n_features: int) -> list[int]:
        if self.hidden_dims is not None:
            dims = [int(v) for v in self.hidden_dims]
        else:
            dims = [max(32, min(128, n_features * 2)), max(16, min(64, n_features))]
        if not dims or any(v <= 0 for v in dims):
            raise ValueError("hidden_dims must contain positive integers")
        return dims

    def _sample_dcl(self, z0: "torch.Tensor", zk: "torch.Tensor") -> "torch.Tensor":
        """Per-sample DCL matching Eq. (2)/(3) of Qiu et al. (2021)."""
        z0 = F.normalize(z0, p=2, dim=-1, eps=1e-12)
        zk = F.normalize(zk, p=2, dim=-1, eps=1e-12)
        pos_logits = torch.sum(zk * z0[:, None, :], dim=-1) / self.temperature
        pair_logits = torch.matmul(zk, zk.transpose(1, 2)) / self.temperature
        k = int(zk.shape[1])
        eye = torch.eye(k, device=zk.device, dtype=torch.bool)[None, :, :]
        pair_logits = pair_logits.masked_fill(eye, float("-inf"))
        denominator_logits = torch.cat([pos_logits[:, :, None], pair_logits], dim=2)
        log_denom = torch.logsumexp(denominator_logits, dim=2)
        return torch.sum(-pos_logits + log_denom, dim=1)

    def _diagnostics(self, x: "torch.Tensor", z0: "torch.Tensor", zk: "torch.Tensor") -> dict[str, float]:
        with torch.no_grad():
            masks = torch.stack([m(x) for m in self.network.transforms], dim=1)
            latent_std = torch.std(z0, dim=0, unbiased=False)
            return {
                "mask_mean": float(torch.mean(masks).cpu()),
                "mask_std": float(torch.std(masks, unbiased=False).cpu()),
                "latent_std_mean": float(torch.mean(latent_std).cpu()),
                "latent_norm_mean": float(torch.mean(torch.linalg.vector_norm(z0, dim=1)).cpu()),
                "transformed_latent_norm_mean": float(torch.mean(torch.linalg.vector_norm(zk, dim=2)).cpu()),
            }

    def fit(self, x: np.ndarray) -> "NeuTraLADScorer":
        values = _as_2d_float(x)
        if len(values) < 2:
            raise ValueError("NeuTraL AD requires at least two training samples")
        _set_seed(self.random_state, self.torch_num_threads)
        self.device_ = _resolve_device(self.device_name)
        self.n_features_in_ = int(values.shape[1])
        self.network = _NeuTraLNet(
            self.n_features_in_, self._resolve_hidden(self.n_features_in_), self.latent_dim,
            self.activation, self.num_transforms, self.transform_hidden_dim, self.transform_depth,
        ).to(self.device_)
        opt = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loader = _loader(values, self.batch_size, True, self.random_state)
        self.training_history_ = []
        for epoch in range(self.epochs):
            self.network.train()
            total = 0.0
            seen = 0
            last_diag: dict[str, float] = {}
            for (cpu_batch,) in loader:
                batch = cpu_batch.to(self.device_, dtype=torch.float32)
                opt.zero_grad(set_to_none=True)
                z0, zk = self.network.views(batch)
                sample_loss = self._sample_dcl(z0, zk)
                loss = torch.mean(sample_loss)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"NeuTraL AD produced non-finite DCL at epoch {epoch + 1}")
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
                opt.step()
                total += float(loss.detach().cpu()) * len(batch)
                seen += len(batch)
                last_diag = self._diagnostics(batch, z0, zk)
            record = {
                "epoch": float(epoch + 1),
                "total_loss": float(total / max(1, seen)),
                "dcl_loss": float(total / max(1, seen)),
                "learning_rate": float(opt.param_groups[0]["lr"]),
                **last_diag,
            }
            self.training_history_.append(record)
            if self.verbose:
                print(f"[NeuTraLAD] epoch={epoch + 1}/{self.epochs} dcl={record['dcl_loss']:.6g}")
        self.network.eval()
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        if self.network is None or self.device_ is None:
            raise RuntimeError("NeuTraL AD has not been fitted")
        values = _as_2d_float(x)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(f"feature dimension mismatch: fitted={self.n_features_in_}, received={values.shape[1]}")
        loader = _loader(values, self.batch_size, False, self.random_state)
        out: list[np.ndarray] = []
        self.network.eval()
        with torch.no_grad():
            for (cpu_batch,) in loader:
                batch = cpu_batch.to(self.device_, dtype=torch.float32)
                z0, zk = self.network.views(batch)
                out.append(self._sample_dcl(z0, zk).cpu().numpy().astype(np.float64))
        return _sanitize_scores(np.concatenate(out) if out else np.empty(0, dtype=np.float64))


from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
import random

try:
    from pyod.models.copod import COPOD as PyODCOPOD
except ImportError:
    PyODCOPOD = None

ScalerType = Optional[Union[StandardScaler, RobustScaler]]

def _as_2d_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"features must be 2D, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return arr

def _sanitize_scores(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if np.all(np.isfinite(s)):
        return s
    finite = s[np.isfinite(s)]
    hi = float(np.max(finite)) if finite.size else 0.0
    lo = float(np.min(finite)) if finite.size else 0.0
    return np.nan_to_num(s, nan=hi, posinf=hi, neginf=lo)

SEQUENCE_MODELS = ()
DEEP_MODELS = ("ae", "deep_svdd", "neutral_ad")

def _as_3d_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"sequence features must be 3D, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return arr

def _input_len(x: Any) -> int:
    return int(len(x))

def _materialize_input(x: Any, local_indices: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if local_indices is None:
        return arr
    return arr[np.asarray(local_indices, dtype=np.int64)]

def _scale_input_fit(x: np.ndarray, scaler: ScalerType) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if scaler is None:
        return arr
    if arr.ndim == 2:
        out = scaler.fit_transform(arr).astype(np.float32)
    elif arr.ndim == 3:
        n, t, d = arr.shape
        flat = arr.reshape(n * t, d)
        out = scaler.fit_transform(flat).astype(np.float32).reshape(n, t, d)
    else:
        raise ValueError(f"Unsupported input rank for scaling: {arr.ndim}")
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

def _scale_input_transform(x: np.ndarray, scaler: ScalerType) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if scaler is None:
        return arr
    if arr.ndim == 2:
        out = scaler.transform(arr).astype(np.float32)
    elif arr.ndim == 3:
        n, t, d = arr.shape
        out = scaler.transform(arr.reshape(n * t, d)).astype(np.float32).reshape(n, t, d)
    else:
        raise ValueError(f"Unsupported input rank for scaling: {arr.ndim}")
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
class PCAReconstructionScorer:
    """PCA anomaly scorer with selectable residual aggregation.

    ``mse`` preserves the original implementation. ``standardized_mse`` divides
    each squared residual by its training residual variance, which prevents a
    few high-variance features from dominating the score and makes the scorer
    more sensitive to changes in previously stable features. ``max_residual``
    uses the largest standardized squared residual and is intentionally more
    tail-sensitive. These modes are tuned only on development periods and must
    be held fixed before final evaluation.
    """

    def __init__(
        self,
        n_components: Any = 0.95,
        random_state: int = 0,
        score_mode: str = "mse",
    ):
        self.n_components = n_components
        self.random_state = int(random_state)
        self.score_mode = str(score_mode).lower()
        if self.score_mode not in {"mse", "standardized_mse", "max_residual"}:
            raise ValueError(
                "PCA score_mode must be one of: mse, standardized_mse, max_residual"
            )
        self.pca: Optional[PCA] = None
        self.residual_variance_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "PCAReconstructionScorer":
        x = _as_2d_float(x)
        n_components = self.n_components
        if isinstance(n_components, (int, np.integer)):
            n_components = max(1, min(int(n_components), x.shape[0], x.shape[1]))
        self.pca = PCA(
            n_components=n_components,
            svd_solver="full",
            random_state=self.random_state,
        )
        self.pca.fit(x)

        projection = self.pca.transform(x)
        reconstruction = self.pca.inverse_transform(projection)
        residual_sq = (x - reconstruction) ** 2
        variance = np.mean(residual_sq, axis=0, dtype=np.float64)
        positive = variance[np.isfinite(variance) & (variance > 0.0)]
        reference = float(np.median(positive)) if positive.size else 1.0
        floor = max(reference * 1e-6, 1e-12)
        self.residual_variance_ = np.maximum(
            np.nan_to_num(variance, nan=reference, posinf=reference, neginf=reference),
            floor,
        )
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCAReconstructionScorer has not been fitted")
        x = _as_2d_float(x)
        projection = self.pca.transform(x)
        reconstruction = self.pca.inverse_transform(projection)
        residual_sq = (x - reconstruction) ** 2
        if self.score_mode == "mse":
            scores = np.mean(residual_sq, axis=1)
        else:
            if self.residual_variance_ is None:
                raise RuntimeError("PCA residual variance is unavailable")
            standardized = residual_sq / self.residual_variance_
            if self.score_mode == "standardized_mse":
                scores = np.mean(standardized, axis=1)
            else:
                scores = np.max(standardized, axis=1)
        return _sanitize_scores(scores)

class InductiveFeatureBaggedECODScorer:
    """ECOD using fixed training ECDFs and a feature-bagging ensemble."""

    def __init__(
        self,
        n_estimators: int = 1,
        max_features: Union[int, float] = 1.0,
        bootstrap_features: bool = False,
        combination: str = "average",
        smoothing: float = 0.5,
        random_state: int = 0,
    ):
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.bootstrap_features = bool(bootstrap_features)
        self.combination = str(combination).lower()
        self.smoothing = float(smoothing)
        self.random_state = int(random_state)

        if self.n_estimators < 1:
            raise ValueError("ECOD n_estimators must be >= 1")
        if self.combination not in {"average", "max"}:
            raise ValueError(
                "ECOD combination must be one of: average, max"
            )
        if self.smoothing <= 0:
            raise ValueError("ECOD smoothing must be positive")

        self.estimators_: list[Dict[str, np.ndarray]] = []
        self.n_features_in_: Optional[int] = None

    def _resolve_feature_count(self, n_features: int) -> int:
        if isinstance(self.max_features, (float, np.floating)):
            fraction = float(self.max_features)

            if not 0.0 < fraction <= 1.0:
                raise ValueError(
                    "Float ECOD max_features must be in (0, 1]"
                )

            return max(
                1,
                min(
                    n_features,
                    int(np.ceil(fraction * n_features)),
                ),
            )

        count = int(self.max_features)

        if count < 1:
            raise ValueError(
                "Integer ECOD max_features must be >= 1"
            )

        return min(count, n_features)

    @staticmethod
    def _skew_sign(x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)

        centered = values - np.mean(
            values,
            axis=0,
            keepdims=True,
        )

        std = np.std(values, axis=0, ddof=0)
        denominator = np.maximum(std ** 3, 1e-12)

        skewness = (
            np.mean(centered ** 3, axis=0)
            / denominator
        )

        return np.sign(
            np.nan_to_num(skewness, nan=0.0)
        ).astype(np.int8)

    def _score_one(
        self,
        x: np.ndarray,
        features: np.ndarray,
        sorted_reference: np.ndarray,
        skew_sign: np.ndarray,
    ) -> np.ndarray:
        x_sub = np.asarray(
            x[:, features],
            dtype=np.float64,
        )

        n_reference = int(sorted_reference.shape[0])
        n_samples, n_features = x_sub.shape

        p_left = np.empty(
            (n_samples, n_features),
            dtype=np.float64,
        )

        p_right = np.empty(
            (n_samples, n_features),
            dtype=np.float64,
        )

        denominator = (
            n_reference + 2.0 * self.smoothing
        )

        for j in range(n_features):
            reference_column = sorted_reference[:, j]
            values = x_sub[:, j]

            left_count = np.searchsorted(
                reference_column,
                values,
                side="right",
            )

            right_count = (
                n_reference
                - np.searchsorted(
                    reference_column,
                    values,
                    side="left",
                )
            )

            p_left[:, j] = (
                left_count + self.smoothing
            ) / denominator

            p_right[:, j] = (
                right_count + self.smoothing
            ) / denominator

        p_left = np.clip(p_left, 1e-12, 1.0)
        p_right = np.clip(p_right, 1e-12, 1.0)

        u_left = -np.log(p_left)
        u_right = -np.log(p_right)

        # ECOD skew-aware tail.
        u_skew = np.where(
            skew_sign[None, :] < 0,
            u_left,
            np.where(
                skew_sign[None, :] > 0,
                u_right,
                u_left + u_right,
            ),
        )

        dimensional_scores = np.maximum(
            np.maximum(u_left, u_right),
            u_skew,
        )

        return np.sum(
            dimensional_scores,
            axis=1,
            dtype=np.float64,
        )

    def fit(
        self,
        x: np.ndarray,
    ) -> "InductiveFeatureBaggedECODScorer":
        x = _as_2d_float(x)

        if len(x) < 2:
            raise ValueError(
                "Inductive ECOD requires at least 2 samples"
            )

        self.n_features_in_ = int(x.shape[1])

        feature_count = self._resolve_feature_count(
            self.n_features_in_
        )

        rng = np.random.default_rng(
            self.random_state
        )

        self.estimators_ = []

        for _ in range(self.n_estimators):
            if (
                not self.bootstrap_features
                and feature_count == self.n_features_in_
            ):
                features = np.arange(
                    self.n_features_in_,
                    dtype=np.int64,
                )
            else:
                features = rng.choice(
                    self.n_features_in_,
                    size=feature_count,
                    replace=self.bootstrap_features,
                ).astype(np.int64)

            x_sub = np.asarray(
                x[:, features],
                dtype=np.float64,
            )

            self.estimators_.append(
                {
                    "features": features,
                    "sorted_reference": np.sort(
                        x_sub,
                        axis=0,
                    ),
                    "skew_sign": self._skew_sign(x_sub),
                }
            )

        return self

    def score_samples(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError(
                "Inductive ECOD has not been fitted"
            )

        x = _as_2d_float(x)

        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                "ECOD feature dimension mismatch: "
                f"fitted={self.n_features_in_}, "
                f"received={x.shape[1]}"
            )

        estimator_scores = np.column_stack(
            [
                self._score_one(
                    x=x,
                    features=item["features"],
                    sorted_reference=item[
                        "sorted_reference"
                    ],
                    skew_sign=item["skew_sign"],
                )
                for item in self.estimators_
            ]
        )

        if self.combination == "average":
            scores = np.mean(
                estimator_scores,
                axis=1,
            )
        else:
            scores = np.max(
                estimator_scores,
                axis=1,
            )

        return _sanitize_scores(scores)

class KMeansDistanceScorer:
    def __init__(
        self,
        n_clusters: int = 50,
        batch_size: int = 2048,
        max_iter: int = 100,
        n_init: int = 10,
        random_state: int = 0,
        normalize_distance: bool = False,
        radius_quantile: float = 0.95,
    ):
        self.n_clusters = int(n_clusters)
        self.batch_size = int(batch_size)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.random_state = int(random_state)
        self.normalize_distance = bool(normalize_distance)
        self.radius_quantile = float(radius_quantile)
        if not (0.0 < self.radius_quantile < 1.0):
            raise ValueError("radius_quantile must be in (0, 1)")
        self.model: Optional[MiniBatchKMeans] = None
        self.cluster_radius_: Optional[np.ndarray] = None
        self.cluster_counts_: Optional[np.ndarray] = None
        self.n_iter_: Optional[int] = None
        self.inertia_: Optional[float] = None

    def fit(self, x: np.ndarray) -> "KMeansDistanceScorer":
        x = _as_2d_float(x)
        if len(x) == 0:
            raise ValueError("KMeans received empty training data")
        n_clusters = max(1, min(self.n_clusters, len(x)))
        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self.model.fit(x)
        self.n_iter_ = int(getattr(self.model, "n_iter_", 0))
        self.inertia_ = float(getattr(self.model, "inertia_", float("nan")))

        distances = self.model.transform(x)
        labels = np.argmin(distances, axis=1)
        nearest = distances[np.arange(len(x)), labels]
        self.cluster_counts_ = np.bincount(labels, minlength=int(n_clusters)).astype(np.int64)
        fallback = float(np.quantile(nearest, self.radius_quantile)) if len(nearest) else 1.0
        fallback = max(fallback, 1e-12)
        radii = np.full(int(n_clusters), fallback, dtype=np.float64)
        for cluster_id in range(int(n_clusters)):
            cluster_dist = nearest[labels == cluster_id]
            if cluster_dist.size:
                radii[cluster_id] = max(float(np.quantile(cluster_dist, self.radius_quantile)), 1e-12)
        self.cluster_radius_ = radii
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("KMeansDistanceScorer has not been fitted")
        distances = self.model.transform(_as_2d_float(x))
        labels = np.argmin(distances, axis=1)
        nearest = distances[np.arange(distances.shape[0]), labels]
        if self.normalize_distance:
            if self.cluster_radius_ is None:
                raise RuntimeError("KMeansDistanceScorer cluster radii are unavailable")
            nearest = nearest / self.cluster_radius_[labels]
        return _sanitize_scores(nearest)

class COPODScorer:
    """Optional PyOD COPOD adapter.

    Note: PyOD COPOD decision_function has batch-transductive ECDF semantics.
    It is supported for compatibility but is not part of the default early-prediction baseline.
    """
    def __init__(self, contamination: float = 0.1, n_jobs: int = 1):
        if PyODCOPOD is None:
            raise RuntimeError("COPOD requires pyod: pip install pyod")
        self.contamination = float(contamination)
        self.n_jobs = int(n_jobs)
        self.model = None

    def fit(self, x: np.ndarray) -> "COPODScorer":
        x = _as_2d_float(x)
        self.model = PyODCOPOD(contamination=self.contamination, n_jobs=self.n_jobs)
        self.model.fit(x)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("COPOD has not been fitted")
        return _sanitize_scores(self.model.decision_function(_as_2d_float(x)))

@dataclass
class ModelState:
    model_name: str
    base_model: str
    model: Any
    scaler: ScalerType
    train_samples: int
    calibration_samples: int
    sampling_pool_samples: int
    reference_samples: int
    detector_implementation: str
    config: Dict[str, Any]

def _make_scaler(name: str) -> ScalerType:
    kind = str(name).lower()
    if kind == "none":
        return None
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler(quantile_range=(25.0, 75.0))
    raise ValueError("scaler must be one of: none, standard, robust")

def _balanced_reference_sample(
    periods: Sequence[Any],
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    if not periods or any(_input_len(x) == 0 for x in periods):
        raise ValueError("Reference periods must be non-empty")
    ranks = set()
    trailing_shapes = set()
    for x in periods:
        shape = getattr(x, "shape", None)
        if shape is None:
            shape = np.asarray(x).shape
        ranks.add(len(shape))
        trailing_shapes.add(tuple(shape[1:]))
    if len(ranks) != 1 or len(trailing_shapes) != 1:
        raise ValueError("Reference periods have inconsistent input shapes")
    rank = next(iter(ranks))
    if rank not in {2, 3}:
        raise ValueError(f"Reference inputs must be N x D or N x T x D, got rank={rank}")

    lengths = np.asarray([_input_len(x) for x in periods], dtype=np.int64)
    total = int(np.sum(lengths))
    requested = min(max(1, int(max_samples)), total)
    quotas = np.minimum(lengths, requested // len(periods))
    remaining = requested - int(np.sum(quotas))
    while remaining > 0:
        capacity = lengths - quotas
        ids = np.flatnonzero(capacity > 0)
        if ids.size == 0:
            break
        take = min(remaining, int(ids.size))
        quotas[ids[:take]] += 1
        remaining -= take

    rng = np.random.default_rng(int(seed))
    parts: list[np.ndarray] = []
    for x, q in zip(periods, quotas):
        q = int(q)
        n = _input_len(x)
        ids = np.arange(n, dtype=np.int64) if q >= n else rng.choice(n, size=q, replace=False)
        part = _materialize_input(x, ids)
        parts.append(np.asarray(part, dtype=np.float32))
    out = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    return out[rng.permutation(len(out))], total

def _normalise_if_max_samples(value: Any, n_train: int) -> Any:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return "auto"
        value = float(text) if "." in text else int(text)
    if isinstance(value, float):
        if not 0.0 < value <= 1.0:
            raise ValueError("Float IsolationForest max_samples must be in (0, 1]")
        return value
    return max(1, min(int(value), int(n_train)))

def build_model(base_model: str, config: Dict[str, Any], seed: int, n_train: int) -> Any:
    name = str(base_model).lower()
    if name not in {"if", "pca", "ae", "deep_svdd", "neutral_ad"}:
        raise ValueError(f"Unsupported final model: {base_model}")
    if name == "if":
        return IsolationForest(
            n_estimators=int(config.get("n_estimators", 200)),
            max_samples=_normalise_if_max_samples(config.get("max_samples", 1024), n_train),
            contamination=config.get("contamination", "auto"),
            max_features=float(config.get("max_features", 1.0)),
            bootstrap=bool(config.get("bootstrap", False)),
            n_jobs=int(config.get("n_jobs", 1)),
            random_state=int(seed),
        )
    if name == "pca":
        return PCAReconstructionScorer(
            n_components=config.get("n_components", 0.95),
            random_state=int(seed),
            score_mode=str(config.get("score_mode", "mse")),
        )
    if name == "ae":
        return AutoencoderScorer(
            hidden_dims=config.get("hidden_dims"),
            latent_dim=int(config.get("latent_dim", 8)),
            activation=str(config.get("activation", "relu")),
            dropout=float(config.get("dropout", 0.0)),
            epochs=int(config.get("epochs", 30)),
            batch_size=int(config.get("batch_size", 1024)),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            weight_decay=float(config.get("weight_decay", 1e-5)),
            gradient_clip=float(config.get("gradient_clip", 5.0)),
            score_mode=str(config.get("score_mode", "mse")),
            device=str(config.get("device", "auto")),
            random_state=int(seed),
            torch_num_threads=int(config.get("torch_num_threads", 0)),
            verbose=bool(config.get("verbose", False)),
        )
    if name == "deep_svdd":
        return DeepSVDDScorer(
            hidden_dims=config.get("hidden_dims"),
            latent_dim=int(config.get("latent_dim", 16)),
            activation=str(config.get("activation", "leaky_relu")),
            pretrain_epochs=int(config.get("pretrain_epochs", 10)),
            epochs=int(config.get("epochs", 30)),
            batch_size=int(config.get("batch_size", 1024)),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            pretrain_learning_rate=float(config.get("pretrain_learning_rate", 1e-3)),
            weight_decay=float(config.get("weight_decay", 1e-6)),
            gradient_clip=float(config.get("gradient_clip", 5.0)),
            center_epsilon=float(config.get("center_epsilon", 0.1)),
            device=str(config.get("device", "auto")),
            random_state=int(seed),
            torch_num_threads=int(config.get("torch_num_threads", 0)),
            verbose=bool(config.get("verbose", False)),
        )
    if name == "neutral_ad":
        return NeuTraLADScorer(
            hidden_dims=config.get("hidden_dims"),
            latent_dim=int(config.get("latent_dim", 32)),
            activation=str(config.get("activation", "relu")),
            num_transforms=int(config.get("num_transforms", 7)),
            transform_hidden_dim=int(config.get("transform_hidden_dim", 64)),
            transform_depth=int(config.get("transform_depth", 3)),
            temperature=float(config.get("temperature", 0.1)),
            epochs=int(config.get("epochs", 20)),
            batch_size=int(config.get("batch_size", 512)),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            weight_decay=float(config.get("weight_decay", 1e-5)),
            gradient_clip=float(config.get("gradient_clip", 5.0)),
            device=str(config.get("device", "auto")),
            random_state=int(seed),
            torch_num_threads=int(config.get("torch_num_threads", 0)),
            verbose=bool(config.get("verbose", False)),
        )
    raise ValueError(f"Unsupported model: {base_model}")

def fit_model(
    model_name: str,
    base_model: str,
    reference_periods: Sequence[np.ndarray],
    config: Dict[str, Any],
    model_seed: int,
    sample_seed: int,
    calibration_samples: int = 20_000,
    sampling_pool_size: int = 120_000,
) -> ModelState:
    """Fit from unlabeled reference features using the predeclared sampling protocol.

    The held-out calibration subset is retained solely to preserve the training-sample
    semantics under which the predeclared Google/Backblaze hyperparameters were selected.
    No labels are accepted and no alert threshold is computed here.
    """
    random.seed(int(model_seed))
    np.random.seed(int(model_seed))
    cfg = dict(config)
    max_train = int(cfg.get("max_train_samples", 100_000))
    calibration_samples = max(1, int(calibration_samples))
    desired_pool = int(sampling_pool_size) if int(sampling_pool_size) > 0 else max_train + calibration_samples
    desired_pool = max(desired_pool, max_train + calibration_samples)

    pool, reference_samples = _balanced_reference_sample(
        reference_periods, desired_pool, sample_seed
    )
    if len(pool) < 4:
        raise ValueError("Reference pool is too small to split into fit and calibration subsets")

    minimum_fit = min(max_train, max(1, len(pool) - 1))
    calibration_n = min(calibration_samples, max(1, len(pool) - minimum_fit))
    fit_candidates = pool[:-calibration_n]
    train_n = min(max_train, len(fit_candidates))
    if train_n <= 0:
        raise ValueError("No training samples remain after calibration holdout")
    x_train = fit_candidates[:train_n]

    scaler = _make_scaler(str(cfg.get("scaler", "standard")))
    x_fit = _scale_input_fit(x_train, scaler)
    model = build_model(base_model, cfg, model_seed, len(x_fit))
    model.fit(x_fit)
    implementation = {
        "if": "sklearn.ensemble.IsolationForest",
        "pca": "sklearn.decomposition.PCA reconstruction residual",
        "ae": "PyTorch feed-forward autoencoder reconstruction residual",
        "deep_svdd": "PyTorch Deep SVDD one-class latent hypersphere distance",
        "neutral_ad": "PyTorch NeuTraL AD learnable transformations + deterministic contrastive score",
    }[str(base_model).lower()]
    return ModelState(
        model_name=str(model_name),
        base_model=str(base_model).lower(),
        model=model,
        scaler=scaler,
        train_samples=int(len(x_fit)),
        calibration_samples=int(calibration_n),
        sampling_pool_samples=int(len(pool)),
        reference_samples=int(reference_samples),
        detector_implementation=implementation,
        config=cfg,
    )

def score_model(state: ModelState, x: Any, sequence_chunk_size: int = 50_000) -> np.ndarray:
    arr = _as_2d_float(x)
    arr = _scale_input_transform(arr, state.scaler)
    if state.base_model == "if":
        return _sanitize_scores(-state.model.score_samples(arr))
    return _sanitize_scores(state.model.score_samples(arr))

SUPPORTED_MODELS = (
    "if", "pca", "ae", "deep_svdd", "neutral_ad",
)
