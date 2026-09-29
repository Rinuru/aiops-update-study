"""Five predeclared upstream drift detectors used by the submitted experiment.

Method policy
-------------
KSWIN, LSDD and KDQTree are the final statistical/batch-family detectors.\nKSWIN replaces the screened-out batch-wise KSDrift baseline while retaining\nthe Kolmogorov-Smirnov test in a native windowed streaming formulation:
  * KSWIN: validated C01 configuration (alpha=.005, window=100, stat=30),\n    preceded only by a warm-up-fitted unsupervised multivariate-to-univariate adapter.
  * LSDD: Alibi Detect ``LSDDDrift`` (LSDD statistic + permutation test).
  * KDQTree: Menelaus ``KdqTreeBatch`` (kdq-tree partition + KL divergence +
    bootstrap critical value).

LSDD/KDQTree retain their maintenance-aligned model-reference adapters.\nD3 and DAWIDD retain the already validated sample-stream adapters used in the
AIOps end-to-end experiments. This file deliberately does not add new drift
statistics or fusion rules.

Important: row downsampling is still the current random AIOps adapter in this
version. A later data-interface patch may replace only that external sampling
policy with deterministic sampling; detector algorithms must remain unchanged.
"""
from __future__ import annotations

import hashlib

from contextlib import contextmanager
from dataclasses import dataclass
import random
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import multiscale_graphcorr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

EPS = 1e-12


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    return np.where(np.isfinite(X), X, np.nan).astype(np.float32)


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def sample_rows(X: np.ndarray, max_samples: Optional[int], seed: int = 42) -> np.ndarray:
    """AIOps size adapter: random downsampling without replacement.

    This is intentionally kept separate from detector logic. It will be the
    only function replaced when the project moves to deterministic period
    sampling.
    """
    X = _as_2d_float(X)
    if max_samples is None or int(max_samples) <= 0 or len(X) <= int(max_samples):
        return X
    idx = _rng(seed).choice(len(X), size=int(max_samples), replace=False)
    return X[np.sort(idx)]


class ReferencePreprocessor:
    """Reference-only median imputation and standardization.

    The current/test batch never contributes to fitted preprocessing statistics,
    which avoids look-ahead leakage in offline AIOps evaluation.
    """

    def __init__(self) -> None:
        self.median_: Optional[np.ndarray] = None
        self.scaler_: Optional[StandardScaler] = None

    def fit(self, X_ref: np.ndarray) -> "ReferencePreprocessor":
        X_ref = _as_2d_float(X_ref)
        med = np.nanmedian(X_ref, axis=0)
        med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
        X_i = np.where(np.isfinite(X_ref), X_ref, med)
        scaler = StandardScaler().fit(X_i)
        self.median_ = med
        self.scaler_ = scaler
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.scaler_ is None:
            raise RuntimeError("ReferencePreprocessor must be fitted first")
        X = _as_2d_float(X)
        X_i = np.where(np.isfinite(X), X, self.median_)
        Z = self.scaler_.transform(X_i)
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def prepare_pair(
    X_ref: np.ndarray,
    X_cur: np.ndarray,
    max_samples: Optional[int],
    random_state: int,
    pair_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit preprocessing on the reference and then cap both batches."""
    X_ref = _as_2d_float(X_ref)
    X_cur = _as_2d_float(X_cur)
    prep = ReferencePreprocessor().fit(X_ref)
    A = prep.transform(X_ref)
    B = prep.transform(X_cur)
    A = sample_rows(A, max_samples, int(random_state) + int(pair_seed))
    B = sample_rows(B, max_samples, int(random_state) + 10_000 + int(pair_seed))
    return A, B


def _scalar(x, default=np.nan) -> float:
    if x is None:
        return float(default)
    a = np.asarray(x)
    if a.size == 0:
        return float(default)
    try:
        return float(a.reshape(-1)[0])
    except Exception:
        return float(default)


def _base_result(
    detector: str,
    raw_drift: bool,
    score: float,
    threshold: float,
    detector_valid: bool = True,
    invalid_reason: str = "",
) -> Dict[str, object]:
    raw_drift = bool(raw_drift and detector_valid)
    return {
        "detector": detector,
        "raw_drift": raw_drift,
        "confirmed_drift": raw_drift,
        "update_trigger": raw_drift,
        "is_drift": raw_drift,
        "drift_pred": raw_drift,
        "score": float(score) if np.isfinite(score) else np.nan,
        "threshold": float(threshold) if np.isfinite(threshold) else np.nan,
        "detector_valid": bool(detector_valid),
        "invalid_reason": str(invalid_reason),
    }


@contextmanager
def _temporary_external_seed(seed: int):
    """Seed external mature libraries without contaminating downstream model RNG."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch = None
    torch_cpu_state = None
    torch_cuda_state = None
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32 - 1))
        try:
            import torch as _torch
            torch = _torch
            torch_cpu_state = torch.random.get_rng_state()
            if torch.cuda.is_available():
                torch_cuda_state = torch.cuda.get_rng_state_all()
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        except Exception:
            torch = None
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch is not None and torch_cpu_state is not None:
            try:
                torch.random.set_rng_state(torch_cpu_state)
                if torch_cuda_state is not None:
                    torch.cuda.set_rng_state_all(torch_cuda_state)
            except Exception:
                pass


def _require_alibi():
    try:
        from alibi_detect.cd import KSDrift, LSDDDrift
        import alibi_detect
        return KSDrift, LSDDDrift, getattr(alibi_detect, "__version__", "unknown")
    except Exception as exc:
        raise ImportError(
            "KS/LSDD mature adapters require alibi-detect. "
            "Install the pinned detector dependencies before running these branches."
        ) from exc


def _require_menelaus():
    try:
        from menelaus.data_drift import KdqTreeBatch
        import menelaus
        return KdqTreeBatch, getattr(menelaus, "__version__", "0.2.0")
    except Exception as exc:
        raise ImportError(
            "KDQTree mature adapter requires menelaus==0.2.0."
        ) from exc



# ---------------------------------------------------------------------
# Final KS-family detector: KSWIN C01
# ---------------------------------------------------------------------

@dataclass
class KSWINConfig:
    """Predeclared KSWIN C01 configuration validated on Backblaze development data.

    The KSWIN statistic/window mechanics are not modified.  The only AIOps
    adapter maps multivariate rows to the one-dimensional stream required by
    KSWIN using StandardScaler + PCA(>=95% warm-up variance) + L2 norm of the
    retained PC-score vector.  The projection is fitted on unlabeled warm-up
    samples only and then held fixed.
    """

    alpha: float = 0.005
    window_size: int = 100
    stat_size: int = 30
    samples_per_period: int = 30
    pca_variance: float = 0.95
    random_state: int = 42


class KSWINDriftDetector:
    """Windowed two-sample KS detector using the validated C01 protocol."""

    @staticmethod
    def _kswin_stable_seed(*parts: object) -> int:
        payload = "|".join(str(x) for x in parts).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "little") % (2**31 - 1)

    name = "kswin"
    stream_based = True
    maintenance_aligned = False

    def __init__(self, config: KSWINConfig = KSWINConfig()):
        self.config = config
        if not (0.0 < float(config.alpha) < 1.0):
            raise ValueError("KSWIN alpha must be in (0,1)")
        if not (2 <= int(config.stat_size) <= int(config.window_size) // 2):
            raise ValueError("KSWIN requires 2 <= stat_size <= window_size/2")
        if int(config.samples_per_period) <= 0:
            raise ValueError("KSWIN samples_per_period must be positive")
        if not (0.0 < float(config.pca_variance) <= 1.0):
            raise ValueError("KSWIN pca_variance must be in (0,1]")

        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._window = np.empty(0, dtype=np.float64)
        self._rng_stream = _rng(self._kswin_stable_seed(int(config.random_state), "kswin_c01"))
        self._checks = 0
        self._alerts = 0
        self._period_ids: list[int] = []
        self._pca_components = 0
        self._explained_variance = np.nan

    def _sample_period(self, X: np.ndarray, period_id: int, stage: str) -> np.ndarray:
        X = _as_2d_float(X)
        n = int(self.config.samples_per_period)
        if len(X) <= n:
            return X.astype(np.float64, copy=False)
        stage_key = "warm" if stage == "warmup" else "dev"
        seed = self._kswin_stable_seed(int(self.config.random_state), stage_key, int(period_id))
        idx = np.sort(_rng(seed).choice(len(X), size=n, replace=False))
        return X[idx].astype(np.float64, copy=False)

    def _fit_projection(self, sampled_periods: Sequence[np.ndarray]) -> np.ndarray:
        clean = [np.asarray(x, dtype=np.float64) for x in sampled_periods if x is not None and len(x) > 0]
        if not clean:
            raise ValueError("KSWIN warm-up contains no samples")
        X = np.vstack(clean)
        if not np.all(np.isfinite(X)):
            # Dataset preprocessing should already produce finite values.
            # This guard is deterministic and fitted only on warm-up data.
            med = np.nanmedian(X, axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            X = np.where(np.isfinite(X), X, med)

        self._scaler = StandardScaler().fit(X)
        Z = self._scaler.transform(X)
        self._pca = PCA(
            n_components=float(self.config.pca_variance),
            svd_solver="full",
        ).fit(Z)
        PC = self._pca.transform(Z)
        self._pca_components = int(self._pca.n_components_)
        self._explained_variance = float(np.sum(self._pca.explained_variance_ratio_))
        return np.linalg.norm(PC, axis=1)

    def _project(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is None or self._pca is None:
            raise RuntimeError("KSWIN must be initialized before processing periods")
        A = np.asarray(X, dtype=np.float64)
        if not np.all(np.isfinite(A)):
            A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        PC = self._pca.transform(self._scaler.transform(A))
        return np.linalg.norm(PC, axis=1)

    def _add_scalar(self, value: float) -> tuple[bool, float | None, float | None]:
        """One native KSWIN stream update.

        Test the full window before dropping the oldest point.  This is the
        ordering used in the validated confirmatory script and ensures the old
        pool and recent block are both available.
        """
        raw = False
        statistic = None
        p_value = None

        if len(self._window) == int(self.config.window_size):
            stat_n = int(self.config.stat_size)
            old_pool = self._window[:-stat_n]
            recent = self._window[-stat_n:]
            if len(old_pool) < stat_n:
                raise RuntimeError(
                    f"KSWIN invalid window state: old_pool={len(old_pool)}, stat_size={stat_n}"
                )

            old_sample = self._rng_stream.choice(
                old_pool,
                size=stat_n,
                replace=False,
            )
            from scipy import stats as _stats
            ks, p = _stats.ks_2samp(
                old_sample,
                recent,
                alternative="two-sided",
                method="auto",
            )
            statistic = float(ks)
            p_value = float(p)
            self._checks += 1

            if p_value <= float(self.config.alpha) and statistic > 0.1:
                raw = True
                self._alerts += 1
                self._window = recent.copy()
            else:
                self._window = self._window[1:].copy()

        self._window = np.concatenate(
            [self._window, np.asarray([float(value)], dtype=np.float64)]
        )
        return raw, statistic, p_value

    def initialize_stream(self, periods: Sequence[np.ndarray], start_period: int = 1) -> None:
        sampled = []
        pids = []
        for offset, X in enumerate(periods):
            pid = int(start_period) + offset
            A = self._sample_period(X, pid, "warmup")
            sampled.append(A)
            pids.extend([pid] * len(A))

        warm_stream = self._fit_projection(sampled)
        self._window = warm_stream[-int(self.config.window_size):].astype(np.float64, copy=True)
        self._checks = 0
        self._alerts = 0
        self._rng_stream = _rng(self._kswin_stable_seed(int(self.config.random_state), "kswin_c01"))
        # Reference metadata describes the warm-up periods that fitted the
        # fixed projection and seeded the initial KSWIN stream.
        self._period_ids = [int(start_period) + i for i in range(len(periods))]

    def process_period(self, X: np.ndarray, period_id: int) -> Dict[str, object]:
        sampled = self._sample_period(X, int(period_id), "evaluation")
        stream = self._project(sampled)

        checks_before = int(self._checks)
        alerts_before = int(self._alerts)
        window_before = int(len(self._window))
        max_stat = np.nan
        min_p = np.nan

        for value in stream:
            raw_i, stat_i, p_i = self._add_scalar(float(value))
            if stat_i is not None:
                max_stat = float(stat_i) if not np.isfinite(max_stat) else max(float(max_stat), float(stat_i))
            if p_i is not None:
                min_p = float(p_i) if not np.isfinite(min_p) else min(float(min_p), float(p_i))

        period_alerts = int(self._alerts - alerts_before)
        raw = bool(period_alerts > 0)

        # Use max KS statistic as the score.  The complete native decision also
        # requires p<=alpha; both are exported explicitly.
        out = _base_result(
            self.name,
            raw,
            float(max_stat) if np.isfinite(max_stat) else np.nan,
            0.1,
        )
        ref_start = min(self._period_ids) if self._period_ids else max(1, int(period_id) - 1)
        ref_end = max(self._period_ids) if self._period_ids else max(ref_start, int(period_id) - 1)
        out.update({
            "stream_based": True,
            "implementation": "validated_KSWIN_C01",
            "implementation_version": "c01-stableseed-v2-20260829",
            "score_semantics": "max_sample_level_ks_statistic_with_native_p_gate",
            "threshold_method": "native_kswin_p_and_statistic",
            "decision_mode": "period_drift_if_any_native_sample_alert",
            "decision_reason": "native_kswin_alert" if raw else "no_native_kswin_alert",
            "p_value": float(min_p) if np.isfinite(min_p) else np.nan,
            "p_value_threshold": float(self.config.alpha),
            "max_ks_stat": float(max_stat) if np.isfinite(max_stat) else np.nan,
            "stream_checks": int(self._checks - checks_before),
            "stream_drift_alerts": int(period_alerts),
            "window_size_before": int(window_before),
            "window_size_after": int(len(self._window)),
            "reference_start_period": int(ref_start),
            "reference_end_period": int(ref_end),
            "kswin_alpha": float(self.config.alpha),
            "kswin_window_size": int(self.config.window_size),
            "kswin_stat_size": int(self.config.stat_size),
            "kswin_samples_per_period": int(self.config.samples_per_period),
            "kswin_pca_variance_target": float(self.config.pca_variance),
            "kswin_pca_components": int(self._pca_components),
            "kswin_explained_variance": float(self._explained_variance),
            "kswin_projection": "StandardScaler+PCA95+PC_score_L2_norm",
            "kswin_sampling_seed_rule": "stable_seed(seed,warm/dev,period)",
            "kswin_internal_rng_rule": "stable_seed(seed,kswin_c01)",
        })
        return out

    def detect_pair(self, *args, **kwargs):
        raise RuntimeError("KSWIN is stream-based; use initialize_stream/process_period")


# ---------------------------------------------------------------------
# 1. Mature KS: Alibi Detect KSDrift
# ---------------------------------------------------------------------

@dataclass
class KSConfig:
    p_val: float = 0.05
    correction: str = "fdr"
    max_samples: int = 5000
    alternative: str = "two-sided"
    random_state: int = 42


class KSDriftDetector:
    """Thin adapter over Alibi Detect ``KSDrift`` with optional fixed-reference state.

    ``reset_reference`` / ``process_period`` are maintenance-framework adapters.
    They do not alter KS statistics, significance correction, or the Alibi decision rule.
    The reference is changed only when the external maintenance policy explicitly resets it.
    """

    name = "ks"
    stream_based = False
    maintenance_aligned = True

    def __init__(self, config: KSConfig = KSConfig()):
        self.config = config
        if self.config.correction not in {"fdr", "bonferroni"}:
            raise ValueError("KS correction must be 'fdr' or 'bonferroni'")
        if not (0.0 < float(self.config.p_val) < 1.0):
            raise ValueError("KS p_val must be in (0,1)")
        self._prep = None
        self._detector = None
        self._version = "unknown"
        self._reference_start_period = None
        self._reference_end_period = None
        self._reference_reset_count = 0

    def reset_reference(self, X_ref: np.ndarray, reference_start_period: int | None = None,
                        reference_end_period: int | None = None, reset_seed: int = 0) -> None:
        KSDrift, _, version = _require_alibi()
        X_ref = _as_2d_float(X_ref)
        prep = ReferencePreprocessor().fit(X_ref)
        A = prep.transform(X_ref)
        A = sample_rows(A, self.config.max_samples, int(self.config.random_state) + int(reset_seed))
        if len(A) < 2:
            raise ValueError("KS reference needs at least 2 samples")
        self._prep = prep
        self._detector = KSDrift(
            A, p_val=float(self.config.p_val), correction=str(self.config.correction),
            alternative=str(self.config.alternative), data_type="tabular", update_x_ref=None,
        )
        self._version = version
        self._reference_start_period = reference_start_period
        self._reference_end_period = reference_end_period
        self._reference_reset_count += 1

    def process_period(self, X_cur: np.ndarray, period_id: int) -> Dict[str, object]:
        if self._detector is None or self._prep is None:
            raise RuntimeError("KS reference not initialized; call reset_reference first")
        B = self._prep.transform(X_cur)
        B = sample_rows(B, self.config.max_samples, int(self.config.random_state) + 10_000 + int(period_id))
        if len(B) < 2:
            return _base_result(self.name, False, np.nan, np.nan, False, "ks_insufficient_samples")
        pred = self._detector.predict(B, drift_type="batch", return_p_val=True, return_distance=True)
        data = pred["data"]
        p_values = np.asarray(data.get("p_val", []), dtype=float).reshape(-1)
        distances = np.asarray(data.get("distance", []), dtype=float).reshape(-1)
        p_threshold = _scalar(data.get("threshold"), self.config.p_val)
        min_p = float(np.nanmin(p_values)) if p_values.size else np.nan
        max_ks = float(np.nanmax(distances)) if distances.size else np.nan
        score = -np.log10(max(min_p, EPS)) if np.isfinite(min_p) else np.nan
        threshold = -np.log10(max(p_threshold, EPS)) if np.isfinite(p_threshold) else np.nan
        raw = bool(int(_scalar(data.get("is_drift"), 0)) == 1)
        out = _base_result(self.name, raw, score, threshold)
        out.update({
            "implementation": "alibi_detect.KSDrift", "implementation_version": self._version,
            "score_semantics": "neg_log10_min_feature_p",
            "threshold_method": f"alibi_{self.config.correction}",
            "p_value": min_p, "p_value_threshold": p_threshold,
            "max_ks_stat": max_ks,
            "mean_ks_stat": float(np.nanmean(distances)) if distances.size else np.nan,
            "n_features": int(B.shape[1]), "ks_correction": str(self.config.correction),
            "ks_alternative": str(self.config.alternative), "max_samples": int(self.config.max_samples),
            "maintenance_aligned_reference": True,
            "reference_start_period": self._reference_start_period,
            "reference_end_period": self._reference_end_period,
            "reference_reset_count": int(self._reference_reset_count),
        })
        return out

    def detect_pair(self, X_ref: np.ndarray, X_cur: np.ndarray, pair_seed: int = 0,
                    X_ref_periods: Optional[Sequence[np.ndarray]] = None,
                    X_calibration_periods: Optional[Sequence[np.ndarray]] = None,
                    calibration_ref_periods: int = 1) -> Dict[str, object]:
        # Backward-compatible stateless path used by legacy code.
        tmp = KSDriftDetector(self.config)
        tmp.reset_reference(X_ref, reset_seed=pair_seed)
        return tmp.process_period(X_cur, pair_seed)


# ---------------------------------------------------------------------
# 2. Mature LSDD: Alibi Detect LSDDDrift
# ---------------------------------------------------------------------

@dataclass
class LSDDConfig:
    p_val: float = 0.05
    max_samples: int = 3000
    backend: str = "pytorch"
    n_permutations: int = 100
    n_kernel_centers: Optional[int] = 100
    lambda_rd_max: float = 0.2
    device: str = "cpu"
    random_state: int = 42


class LSDDDriftDetector:
    """Thin adapter over Alibi Detect ``LSDDDrift`` with fixed-reference state."""

    name = "lsdd"
    stream_based = False
    maintenance_aligned = True

    def __init__(self, config: LSDDConfig = LSDDConfig()):
        self.config = config
        if not (0.0 < float(self.config.p_val) < 1.0):
            raise ValueError("LSDD p_val must be in (0,1)")
        if int(self.config.n_permutations) < 20:
            raise ValueError("LSDD n_permutations must be >=20")
        self._prep = None
        self._detector = None
        self._version = "unknown"
        self._reference_start_period = None
        self._reference_end_period = None
        self._reference_reset_count = 0

    def reset_reference(self, X_ref: np.ndarray, reference_start_period: int | None = None,
                        reference_end_period: int | None = None, reset_seed: int = 0) -> None:
        _, LSDDDrift, version = _require_alibi()
        X_ref = _as_2d_float(X_ref)
        prep = ReferencePreprocessor().fit(X_ref)
        A = prep.transform(X_ref)
        A = sample_rows(A, self.config.max_samples, int(self.config.random_state) + int(reset_seed))
        if len(A) < 4:
            raise ValueError("LSDD reference needs at least 4 samples")
        seed = int(self.config.random_state) + 20_003 * int(reset_seed)
        with _temporary_external_seed(seed):
            detector = LSDDDrift(
                A, backend=str(self.config.backend), p_val=float(self.config.p_val),
                n_permutations=int(self.config.n_permutations),
                n_kernel_centers=(None if self.config.n_kernel_centers is None else int(self.config.n_kernel_centers)),
                lambda_rd_max=float(self.config.lambda_rd_max), device=str(self.config.device),
                data_type="tabular", update_x_ref=None,
            )
        self._prep = prep
        self._detector = detector
        self._version = version
        self._reference_start_period = reference_start_period
        self._reference_end_period = reference_end_period
        self._reference_reset_count += 1

    def process_period(self, X_cur: np.ndarray, period_id: int) -> Dict[str, object]:
        if self._detector is None or self._prep is None:
            raise RuntimeError("LSDD reference not initialized; call reset_reference first")
        B = self._prep.transform(X_cur)
        B = sample_rows(B, self.config.max_samples, int(self.config.random_state) + 10_000 + int(period_id))
        if len(B) < 4:
            return _base_result(self.name, False, np.nan, np.nan, False, "lsdd_insufficient_samples")
        seed = int(self.config.random_state) + 20_003 * int(period_id)
        try:
            with _temporary_external_seed(seed):
                pred = self._detector.predict(B, return_p_val=True, return_distance=True)
        except ValueError as exc:
            # Alibi Detect LSDD can occasionally fail during kernel regularization
            # selection with the explicit message below. Our Backblaze locator
            # confirmed that this is a kernel-representation numerical failure,
            # not duplicate input rows. Treat only this known failure as an
            # invalid detector decision; all other ValueError exceptions still
            # propagate normally.
            message = str(exc)
            if "Too many repeat instances for LSDD-based detection" not in message:
                raise

            out = _base_result(
                self.name,
                False,
                np.nan,
                np.nan,
                detector_valid=False,
                invalid_reason="lsdd_kernel_repeat_numerical_failure",
            )
            out.update({
                "implementation": "alibi_detect.LSDDDrift",
                "implementation_version": self._version,
                "score_semantics": "invalid_lsdd_numerical_failure",
                "threshold_method": "alibi_permutation_test",
                "decision_reason": "lsdd_kernel_repeat_numerical_failure",
                "p_value": np.nan,
                "p_value_threshold": float(self.config.p_val),
                "distance": np.nan,
                "distance_threshold": np.nan,
                "lsdd_backend": str(self.config.backend),
                "lsdd_n_permutations": int(self.config.n_permutations),
                "lsdd_n_kernel_centers": self.config.n_kernel_centers,
                "lsdd_lambda_rd_max": float(self.config.lambda_rd_max),
                "max_samples": int(self.config.max_samples),
                "n_features": int(B.shape[1]),
                "maintenance_aligned_reference": True,
                "reference_start_period": self._reference_start_period,
                "reference_end_period": self._reference_end_period,
                "reference_reset_count": int(self._reference_reset_count),
                "numerical_failure": True,
                "exception_type": type(exc).__name__,
                "exception_message": message,
            })
            return out

        data = pred["data"]
        raw = bool(int(_scalar(data.get("is_drift"), 0)) == 1)
        p_value = _scalar(data.get("p_val"))
        p_threshold = _scalar(data.get("threshold"), self.config.p_val)
        distance = _scalar(data.get("distance"))
        distance_threshold = _scalar(data.get("distance_threshold"))
        if np.isfinite(distance) and np.isfinite(distance_threshold):
            score, threshold, semantics = distance, distance_threshold, "alibi_lsdd_distance"
        else:
            score = -np.log10(max(p_value, EPS)) if np.isfinite(p_value) else np.nan
            threshold = -np.log10(max(p_threshold, EPS)) if np.isfinite(p_threshold) else np.nan
            semantics = "neg_log10_permutation_p"
        out = _base_result(self.name, raw, score, threshold)
        out.update({
            "implementation": "alibi_detect.LSDDDrift", "implementation_version": self._version,
            "score_semantics": semantics, "threshold_method": "alibi_permutation_test",
            "p_value": p_value, "p_value_threshold": p_threshold, "distance": distance,
            "distance_threshold": distance_threshold, "lsdd_backend": str(self.config.backend),
            "lsdd_n_permutations": int(self.config.n_permutations),
            "lsdd_n_kernel_centers": self.config.n_kernel_centers,
            "lsdd_lambda_rd_max": float(self.config.lambda_rd_max),
            "max_samples": int(self.config.max_samples), "n_features": int(B.shape[1]),
            "maintenance_aligned_reference": True,
            "reference_start_period": self._reference_start_period,
            "reference_end_period": self._reference_end_period,
            "reference_reset_count": int(self._reference_reset_count),
        })
        return out

    def detect_pair(self, X_ref: np.ndarray, X_cur: np.ndarray, pair_seed: int = 0,
                    X_ref_periods: Optional[Sequence[np.ndarray]] = None,
                    X_calibration_periods: Optional[Sequence[np.ndarray]] = None,
                    calibration_ref_periods: int = 1) -> Dict[str, object]:
        tmp = LSDDDriftDetector(self.config)
        tmp.reset_reference(X_ref, reset_seed=pair_seed)
        return tmp.process_period(X_cur, pair_seed)


# ---------------------------------------------------------------------
# 3. Mature KDQTree: Menelaus KdqTreeBatch
# ---------------------------------------------------------------------

@dataclass
class KDQTreeConfig:
    alpha: float = 0.01
    bootstrap_samples: int = 500
    count_ubound: int = 100
    cutpoint_proportion_lbound: float = 2e-10
    max_samples: int = 5000
    random_state: int = 42


class KDQTreeDriftDetector:
    """Thin adapter over Menelaus ``KdqTreeBatch`` with explicit reference control.

    Menelaus' kdqTree keeps the reference fixed across ``update`` calls. The
    maintenance framework changes it only through ``reset_reference`` after a
    model update, using the public ``set_reference`` API.
    """

    name = "kdqtree"
    stream_based = False
    maintenance_aligned = True

    def __init__(self, config: KDQTreeConfig = KDQTreeConfig()):
        self.config = config
        if not (0.0 < float(self.config.alpha) < 1.0):
            raise ValueError("KDQTree alpha must be in (0,1)")
        if int(self.config.bootstrap_samples) < 20:
            raise ValueError("KDQTree bootstrap_samples must be >=20")
        self._prep = None
        self._detector = None
        self._version = "unknown"
        self._reference_start_period = None
        self._reference_end_period = None
        self._reference_reset_count = 0

    def reset_reference(self, X_ref: np.ndarray, reference_start_period: int | None = None,
                        reference_end_period: int | None = None, reset_seed: int = 0) -> None:
        KdqTreeBatch, version = _require_menelaus()
        X_ref = _as_2d_float(X_ref)
        prep = ReferencePreprocessor().fit(X_ref)
        A = prep.transform(X_ref)
        A = sample_rows(A, self.config.max_samples, int(self.config.random_state) + int(reset_seed))
        if len(A) < 4:
            raise ValueError("KDQTree reference needs at least 4 samples")
        seed = int(self.config.random_state) + 30_007 * int(reset_seed)
        with _temporary_external_seed(seed):
            detector = KdqTreeBatch(
                alpha=float(self.config.alpha),
                bootstrap_samples=int(self.config.bootstrap_samples),
                count_ubound=int(self.config.count_ubound),
                cutpoint_proportion_lbound=float(self.config.cutpoint_proportion_lbound),
            )
            detector.set_reference(A)
        self._prep = prep
        self._detector = detector
        self._version = version
        self._reference_start_period = reference_start_period
        self._reference_end_period = reference_end_period
        self._reference_reset_count += 1

    def process_period(self, X_cur: np.ndarray, period_id: int) -> Dict[str, object]:
        if self._detector is None or self._prep is None:
            raise RuntimeError("KDQTree reference not initialized; call reset_reference first")
        B = self._prep.transform(X_cur)
        B = sample_rows(B, self.config.max_samples, int(self.config.random_state) + 10_000 + int(period_id))
        if len(B) < 4:
            return _base_result(self.name, False, np.nan, np.nan, False, "kdqtree_insufficient_samples")
        seed = int(self.config.random_state) + 30_007 * int(period_id)
        with _temporary_external_seed(seed):
            self._detector.update(B)
            raw = str(getattr(self._detector, "drift_state", None)).lower() == "drift"
            critical = _scalar(getattr(self._detector, "_critical_dist", np.nan))
            try:
                score = float(self._detector._kdqtree.kl_distance(tree_id1="build", tree_id2="test"))
                n_leaves = int(len(self._detector._kdqtree.leaf_counts("build")))
            except Exception:
                score, n_leaves = np.nan, 0
        valid = bool(np.isfinite(score) and np.isfinite(critical) and n_leaves > 0)
        out = _base_result(self.name, raw, score, critical, detector_valid=valid,
                           invalid_reason="" if valid else "kdqtree_menelaus_score_unavailable")
        out.update({
            "implementation": "menelaus.KdqTreeBatch", "implementation_version": self._version,
            "score_semantics": "kdqtree_kl_divergence",
            "threshold_method": "menelaus_bootstrap_critical_kl",
            "kdq_alpha": float(self.config.alpha),
            "kdq_bootstrap_samples": int(self.config.bootstrap_samples),
            "kdq_count_ubound": int(self.config.count_ubound),
            "kdq_cutpoint_proportion_lbound": float(self.config.cutpoint_proportion_lbound),
            "n_leaves": int(n_leaves), "max_samples": int(self.config.max_samples),
            "n_features": int(B.shape[1]), "maintenance_aligned_reference": True,
            "reference_start_period": self._reference_start_period,
            "reference_end_period": self._reference_end_period,
            "reference_reset_count": int(self._reference_reset_count),
        })
        return out

    def detect_pair(self, X_ref: np.ndarray, X_cur: np.ndarray, pair_seed: int = 0,
                    X_ref_periods: Optional[Sequence[np.ndarray]] = None,
                    X_calibration_periods: Optional[Sequence[np.ndarray]] = None,
                    calibration_ref_periods: int = 1) -> Dict[str, object]:
        tmp = KDQTreeDriftDetector(self.config)
        tmp.reset_reference(X_ref, reset_seed=pair_seed)
        return tmp.process_period(X_cur, pair_seed)

@dataclass
class D3Config:
    """Original-style D3 stream parameters plus an AIOps batch adapter.

    ``window_size``, ``rho`` and ``auc_threshold`` correspond to the original
    D3 parameters w, rho and tau. ``samples_per_period`` is not a D3 model
    parameter; it only bounds how many rows from a very large AIOps period are
    fed to the sample-level stream detector.
    """

    window_size: int = 100
    rho: float = 0.10
    auc_threshold: float = 0.70
    cv_folds: int = 2
    C: float = 1.0
    max_iter: int = 300
    samples_per_period: int = 30
    random_state: int = 42


class D3DriftDetector:
    """Sample-stream D3 following the original w/rho/tau window mechanics.

    AIOps data arrive as very large period batches. Each period is uniformly
    downsampled to ``samples_per_period`` rows and those rows are fed in stream
    order. A period is marked as drifted when at least one original-style D3
    window check inside that period exceeds tau.
    """

    name = "d3"
    stream_based = True

    def __init__(self, config: D3Config = D3Config()):
        self.config = config
        self._validate_config()
        self._median = None
        self._scaler = None
        self._X: list[np.ndarray] = []
        self._period_ids: list[int] = []
        self._check_count = 0

    def _validate_config(self) -> None:
        if int(self.config.window_size) < 20:
            raise ValueError("D3 window_size must be >= 20")
        if not (0.0 < float(self.config.rho) <= 1.0):
            raise ValueError("D3 rho must be in (0, 1]")
        if not (0.5 <= float(self.config.auc_threshold) <= 1.0):
            raise ValueError("D3 auc_threshold must be in [0.5, 1]")
        if int(self.config.samples_per_period) <= 0:
            raise ValueError("D3 samples_per_period must be positive")

    @property
    def _new_size(self) -> int:
        return max(1, int(round(int(self.config.window_size) * float(self.config.rho))))

    @property
    def _buffer_size(self) -> int:
        return int(self.config.window_size) + self._new_size

    def reset(self) -> None:
        self._X = []
        self._period_ids = []
        self._check_count = 0

    def _sample_period(self, X: np.ndarray, period_id: int) -> np.ndarray:
        X = _as_2d_float(X)
        n = int(self.config.samples_per_period)
        if len(X) <= n:
            return X
        rng = np.random.default_rng(int(self.config.random_state) + 10_003 * int(period_id))
        idx = np.sort(rng.choice(len(X), size=n, replace=False))
        return X[idx]

    def _fit_preprocessor(self, sampled_periods: Sequence[np.ndarray]) -> None:
        clean = [_as_2d_float(x) for x in sampled_periods if x is not None and len(x) > 0]
        if not clean:
            raise ValueError("D3 warm-up contains no samples")
        X = np.vstack(clean)
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
        X_i = np.where(np.isfinite(X), X, med)
        scaler = MinMaxScaler()
        scaler.fit(X_i)
        self._median = med
        self._scaler = scaler

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._median is None or self._scaler is None:
            raise RuntimeError("D3 must be initialized before processing evaluation periods")
        X = _as_2d_float(X)
        X_i = np.where(np.isfinite(X), X, self._median)
        Z = self._scaler.transform(X_i)
        return np.nan_to_num(Z, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    def _domain_auc(self, old: np.ndarray, new: np.ndarray, seed: int) -> float:
        X = np.vstack([old, new])
        y = np.concatenate([
            np.ones(len(old), dtype=np.int8),
            np.zeros(len(new), dtype=np.int8),
        ])
        n_splits = min(
            int(self.config.cv_folds),
            int(np.sum(y == 0)),
            int(np.sum(y == 1)),
        )
        if n_splits < 2:
            return np.nan
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pred = np.zeros(len(y), dtype=float)
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            clf = LogisticRegression(
                C=float(self.config.C),
                solver="liblinear",
                max_iter=int(self.config.max_iter),
                random_state=seed + fold,
            )
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict_proba(X[te])[:, 1]
        return float(roc_auc_score(y, pred))

    def _check_and_shift(self) -> dict:
        w = int(self.config.window_size)
        n_new = self._new_size
        X = np.vstack(self._X[: self._buffer_size])
        period_ids = self._period_ids[: self._buffer_size]
        old = X[:w]
        new = X[w:w + n_new]
        seed = int(self.config.random_state) + 50_000 + self._check_count * 1009
        score = self._domain_auc(old, new, seed)
        drift = bool(np.isfinite(score) and score > float(self.config.auc_threshold))
        ref_pids = period_ids[:w]
        target_pids = period_ids[w:w + n_new]
        info = {
            "score": score,
            "raw_drift": drift,
            "reference_start_period": int(min(ref_pids)) if ref_pids else None,
            "reference_end_period": int(max(ref_pids)) if ref_pids else None,
            "target_start_period": int(min(target_pids)) if target_pids else None,
            "target_end_period": int(max(target_pids)) if target_pids else None,
        }
        self._check_count += 1

        # Original D3 window mechanics: after drift retain only the rho*w new
        # samples; otherwise slide forward by rho*w and retain w samples.
        keep = n_new if drift else w
        self._X = self._X[self._buffer_size - keep:self._buffer_size]
        self._period_ids = self._period_ids[self._buffer_size - keep:self._buffer_size]
        return info

    def _process_transformed(self, Z: np.ndarray, period_id: int) -> list[dict]:
        checks: list[dict] = []
        for row in Z:
            self._X.append(np.asarray(row, dtype=np.float32))
            self._period_ids.append(int(period_id))
            if len(self._X) >= self._buffer_size:
                checks.append(self._check_and_shift())
        return checks

    def initialize_stream(self, periods: Sequence[np.ndarray], start_period: int = 1) -> None:
        sampled = []
        period_ids = []
        for offset, X in enumerate(periods):
            pid = int(start_period) + offset
            Xs = self._sample_period(X, pid)
            sampled.append(Xs)
            period_ids.extend([pid] * len(Xs))
        self._fit_preprocessor(sampled)
        self.reset()
        # Deployment initialization: seed the original D3 source window with
        # the most recent unlabeled reference samples, but do not emit or adapt
        # to development-stage drift alarms. Evaluation starts from this fixed
        # reference state.
        Z = np.vstack([self._transform(x) for x in sampled if len(x) > 0])
        w = int(self.config.window_size)
        if len(Z) < w:
            raise ValueError(f"D3 warm-up needs at least {w} sampled rows, got {len(Z)}")
        self._X = [np.asarray(row, dtype=np.float32) for row in Z[-w:]]
        self._period_ids = [int(x) for x in period_ids[-w:]]

    def process_period(self, X: np.ndarray, period_id: int) -> dict:
        Xs = self._sample_period(X, period_id)
        checks = self._process_transformed(self._transform(Xs), int(period_id))
        valid_scores = [c for c in checks if np.isfinite(c.get("score", np.nan))]
        if not valid_scores:
            return _base_result(
                self.name, False, np.nan, float(self.config.auc_threshold),
                detector_valid=False,
                invalid_reason="d3_stream_buffer_not_ready",
            ) | {
                "stream_based": True,
                "stream_checks": 0,
                "stream_drift_alerts": 0,
                "d3_window_size": int(self.config.window_size),
                "d3_rho": float(self.config.rho),
                "d3_auc_threshold": float(self.config.auc_threshold),
                "d3_samples_per_period": int(self.config.samples_per_period),
            }

        best = max(valid_scores, key=lambda c: float(c["score"]))
        n_alerts = int(sum(bool(c["raw_drift"]) for c in valid_scores))
        raw = n_alerts > 0
        out = _base_result(
            self.name, raw, float(best["score"]), float(self.config.auc_threshold)
        )
        out.update({
            "stream_based": True,
            "score_semantics": "original_d3_domain_auc",
            "threshold_method": "fixed_tau",
            "decision_mode": "fixed_tau",
            "decision_reason": "domain_auc_exceeded" if raw else "domain_auc_below_tau",
            "stream_checks": int(len(valid_scores)),
            "stream_drift_alerts": n_alerts,
            "d3_window_size": int(self.config.window_size),
            "d3_rho": float(self.config.rho),
            "d3_auc_threshold": float(self.config.auc_threshold),
            "d3_cv_folds": int(self.config.cv_folds),
            "d3_C": float(self.config.C),
            "d3_samples_per_period": int(self.config.samples_per_period),
            "reference_start_period": best.get("reference_start_period"),
            "reference_end_period": best.get("reference_end_period"),
            "target_start_period": best.get("target_start_period"),
            "target_end_period": best.get("target_end_period"),
        })
        return out

    # Kept only to make accidental pairwise use fail loudly instead of silently
    # running a non-original period-pair approximation.
    def detect_pair(self, *args, **kwargs):
        raise RuntimeError("D3 is stream-based in this framework; use initialize_stream/process_period")


def _build_d3(**cfg):
    return D3DriftDetector(D3Config(
        window_size=int(cfg.get("d3_window_size", 100)),
        rho=float(cfg.get("d3_rho", 0.10)),
        auc_threshold=float(cfg.get("d3_auc_threshold", 0.70)),
        cv_folds=int(cfg.get("d3_cv_folds", 2)),
        C=float(cfg.get("d3_C", 1.0)),
        max_iter=int(cfg.get("d3_max_iter", 300)),
        samples_per_period=int(cfg.get("d3_samples_per_period", 30)),
        random_state=int(cfg.get("random_state", 42)),
    ))


@dataclass
class DAWIDDConfig:
    """DAWIDD dynamic-window parameters plus AIOps batch sampling.

    The DAWIDD paper allows the independence test to be replaced. Here we use
    SciPy's mature non-parametric Multiscale Graph Correlation (MGC) test for
    X ⟂ T, while retaining DAWIDD's dynamic adapting-window logic.
    """

    max_window_size: int = 90
    min_window_size: int = 70
    p_value_threshold: float = 0.005
    samples_per_period: int = 30
    permutations: int = 1000
    workers: int = 1
    random_state: int = 42


class DAWIDDPeriodDetector:
    name = "dawidd"
    stream_based = True

    def __init__(self, config: DAWIDDConfig = DAWIDDConfig()):
        self.config = config
        self._validate_config()
        self._prep: Optional[ReferencePreprocessor] = None
        self._X: list[np.ndarray] = []
        self._time: list[float] = []
        self._period_ids: list[int] = []
        self._test_count = 0

    def _validate_config(self) -> None:
        if int(self.config.min_window_size) < 10:
            raise ValueError("DAWIDD min_window_size must be >= 10")
        if int(self.config.max_window_size) < int(self.config.min_window_size):
            raise ValueError("DAWIDD max_window_size must be >= min_window_size")
        if not (0.0 < float(self.config.p_value_threshold) < 1.0):
            raise ValueError("DAWIDD p_value_threshold must be in (0, 1)")
        if int(self.config.samples_per_period) <= 0:
            raise ValueError("DAWIDD samples_per_period must be positive")
        if int(self.config.permutations) < 19:
            raise ValueError("DAWIDD permutations must be >= 19")

    def reset(self) -> None:
        self._X = []
        self._time = []
        self._period_ids = []
        self._test_count = 0

    def _sample_period(self, X: np.ndarray, period_id: int) -> np.ndarray:
        X = _as_2d_float(X)
        n = int(self.config.samples_per_period)
        if len(X) <= n:
            return X
        rng = np.random.default_rng(int(self.config.random_state) + 20_011 * int(period_id))
        idx = np.sort(rng.choice(len(X), size=n, replace=False))
        return X[idx]

    def _fit_preprocessor(self, sampled_periods: Sequence[np.ndarray]) -> None:
        clean = [_as_2d_float(x) for x in sampled_periods if x is not None and len(x) > 0]
        if not clean:
            raise ValueError("DAWIDD warm-up contains no samples")
        self._prep = ReferencePreprocessor().fit(np.vstack(clean))

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._prep is None:
            raise RuntimeError("DAWIDD must be initialized before processing evaluation periods")
        return self._prep.transform(X)

    def _random_trim_to_max(self, seed: int) -> int:
        n = len(self._X)
        max_n = int(self.config.max_window_size)
        if n <= max_n:
            return 0
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(n, size=max_n, replace=False))
        self._X = [self._X[i] for i in keep]
        self._time = [self._time[i] for i in keep]
        self._period_ids = [self._period_ids[i] for i in keep]
        return int(n - max_n)

    def _test_independence(self, seed: int) -> tuple[float, float]:
        X = np.vstack(self._X)
        t = np.asarray(self._time, dtype=float).reshape(-1, 1)
        # Normalize time to avoid numerical scale artifacts. MGC is a mature
        # non-parametric independence test; DAWIDD itself only requires a valid
        # test of X and T independence.
        if np.std(t) <= EPS:
            return 0.0, 1.0
        t = (t - np.mean(t)) / max(float(np.std(t)), EPS)
        try:
            res = multiscale_graphcorr(
                X,
                t,
                reps=int(self.config.permutations),
                workers=int(self.config.workers),
                random_state=int(seed),
            )
            return float(res.statistic), float(res.pvalue)
        except Exception:
            return np.nan, np.nan

    def _append_period(self, Z: np.ndarray, period_id: int) -> None:
        n = len(Z)
        if n <= 0:
            return
        # AIOps data are evaluated at period granularity; do not invent a finer
        # timestamp that the source data do not provide.
        for row in Z:
            self._X.append(np.asarray(row, dtype=np.float32))
            self._time.append(float(period_id))
            self._period_ids.append(int(period_id))

    def _process_period_internal(self, X: np.ndarray, period_id: int, evaluate: bool) -> dict:
        Xs = self._sample_period(X, period_id)
        self._append_period(self._transform(Xs), int(period_id))
        removed_random = self._random_trim_to_max(
            int(self.config.random_state) + 30_013 * int(period_id)
        )
        before = len(self._X)
        history_pids = [p for p in self._period_ids if p < int(period_id)]
        ref_start = min(history_pids) if history_pids else (min(self._period_ids) if self._period_ids else int(period_id))
        ref_end = max(history_pids) if history_pids else max(ref_start, int(period_id) - 1)

        if before < int(self.config.min_window_size):
            return _base_result(
                self.name, False, np.nan,
                -np.log10(float(self.config.p_value_threshold)),
                detector_valid=False,
                invalid_reason="dawidd_stream_window_not_ready",
            ) | {
                "stream_based": True,
                "window_size_before": before,
                "window_size_after": before,
                "randomly_removed": removed_random,
                "reference_start_period": int(ref_start),
                "reference_end_period": int(ref_end),
            }

        statistic, p_value = self._test_independence(
            int(self.config.random_state) + 40_009 * int(period_id) + self._test_count
        )
        self._test_count += 1
        if not np.isfinite(p_value):
            return _base_result(
                self.name, False, np.nan,
                -np.log10(float(self.config.p_value_threshold)),
                detector_valid=False,
                invalid_reason="dawidd_independence_test_failed",
            ) | {
                "stream_based": True,
                "window_size_before": before,
                "window_size_after": before,
                "randomly_removed": removed_random,
                "reference_start_period": int(ref_start),
                "reference_end_period": int(ref_end),
            }

        raw = bool(p_value <= float(self.config.p_value_threshold))
        after = before
        if raw:
            # DAWIDD adapts the window after a drift. Keep the most recent
            # n_min samples so the detector can continue on the new regime.
            keep = int(self.config.min_window_size)
            if len(self._X) > keep:
                self._X = self._X[-keep:]
                self._time = self._time[-keep:]
                self._period_ids = self._period_ids[-keep:]
            after = len(self._X)

        eps_p = max(float(p_value), 1.0 / (int(self.config.permutations) + 1.0), 1e-12)
        score = float(-np.log10(eps_p))
        threshold = float(-np.log10(float(self.config.p_value_threshold)))
        out = _base_result(self.name, raw, score, threshold)
        out.update({
            "stream_based": True,
            "score_semantics": "dawidd_mgc_neg_log10_p",
            "threshold_method": "p_value",
            "decision_reason": "x_time_dependence_rejected" if raw else "x_time_independence_not_rejected",
            "p_value": float(p_value),
            "dependence_statistic": float(statistic),
            "dawidd_adapter": "dynamic_window_mgc_independence",
            "dawidd_max_window_size": int(self.config.max_window_size),
            "dawidd_min_window_size": int(self.config.min_window_size),
            "dawidd_p_value_threshold": float(self.config.p_value_threshold),
            "dawidd_samples_per_period": int(self.config.samples_per_period),
            "dawidd_permutations": int(self.config.permutations),
            "window_size_before": int(before),
            "window_size_after": int(after),
            "randomly_removed": int(removed_random),
            "reference_start_period": int(ref_start),
            "reference_end_period": int(ref_end),
        })
        return out

    def initialize_stream(self, periods: Sequence[np.ndarray], start_period: int = 1) -> None:
        sampled = []
        pids = []
        for offset, X in enumerate(periods):
            pid = int(start_period) + offset
            Xs = self._sample_period(X, pid)
            sampled.append(Xs)
            pids.extend([pid] * len(Xs))
        self._fit_preprocessor(sampled)
        self.reset()
        # Deployment initialization: populate DAWIDD's dynamic window with the
        # most recent unlabeled reference samples. No development-stage alarm
        # is exposed or used to shrink the window before evaluation.
        Z = np.vstack([self._transform(x) for x in sampled if len(x) > 0])
        max_n = int(self.config.max_window_size)
        if len(Z) < int(self.config.min_window_size):
            raise ValueError(
                f"DAWIDD warm-up needs at least {self.config.min_window_size} sampled rows, got {len(Z)}"
            )
        Z = Z[-max_n:]
        pids = pids[-len(Z):]
        self._X = [np.asarray(row, dtype=np.float32) for row in Z]
        self._time = [float(pid) for pid in pids]
        self._period_ids = [int(pid) for pid in pids]

    def process_period(self, X: np.ndarray, period_id: int) -> dict:
        return self._process_period_internal(X, int(period_id), evaluate=True)

    def detect_pair(self, *args, **kwargs):
        raise RuntimeError("DAWIDD is stream-based in this framework; use initialize_stream/process_period")


def _build_dawidd(**cfg):
    return DAWIDDPeriodDetector(DAWIDDConfig(
        max_window_size=int(cfg.get("dawidd_max_window_size", 90)),
        min_window_size=int(cfg.get("dawidd_min_window_size", 70)),
        p_value_threshold=float(cfg.get("dawidd_p_value_threshold", 0.005)),
        samples_per_period=int(cfg.get("dawidd_samples_per_period", 30)),
        permutations=int(cfg.get("dawidd_permutations", 1000)),
        workers=int(cfg.get("dawidd_workers", 1)),
        random_state=int(cfg.get("random_state", 42)),
    ))

# ---------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------

def build_detector(name: str, **config):
    """Build one of the five final upstream detectors.

    Configuration values come from the submitted upstream detector JSON.
    """
    key = str(name).lower()
    random_state = int(config.get("random_state", 42))
    max_samples = int(config.get("max_samples", 5000))

    if key == "kswin":
        return KSWINDriftDetector(KSWINConfig(
            alpha=float(config.get("kswin_alpha", 0.005)),
            window_size=int(config.get("kswin_window_size", 100)),
            stat_size=int(config.get("kswin_stat_size", 30)),
            samples_per_period=int(config.get("kswin_samples_per_period", 30)),
            pca_variance=float(config.get("kswin_pca_variance", 0.95)),
            random_state=random_state,
        ))

    if key == "lsdd":
        centers = config.get("lsdd_n_kernel_centers", config.get("lsdd_n_centers", 100))
        if centers is not None:
            centers = int(centers)
            if centers <= 0:
                centers = None
        return LSDDDriftDetector(LSDDConfig(
            p_val=float(config.get("lsdd_p_val", config.get("alpha", 0.05))),
            max_samples=int(config.get("lsdd_max_samples", min(max_samples, 3000))),
            backend=str(config.get("lsdd_backend", "pytorch")),
            n_permutations=int(config.get("lsdd_n_permutations", config.get("n_bootstrap", 100))),
            n_kernel_centers=centers,
            lambda_rd_max=float(config.get("lsdd_lambda_rd_max", 0.2)),
            device=str(config.get("lsdd_device", "cpu")),
            random_state=random_state,
        ))

    if key in {"kdq", "kdqtree", "kdq_tree"}:
        return KDQTreeDriftDetector(KDQTreeConfig(
            alpha=float(config.get("kdq_alpha", 0.01)),
            bootstrap_samples=int(config.get("kdq_bootstrap_samples", 500)),
            count_ubound=int(config.get("kdq_count_ubound", 100)),
            cutpoint_proportion_lbound=float(config.get("kdq_cutpoint_proportion_lbound", 2e-10)),
            max_samples=int(config.get("kdq_max_samples", max_samples)),
            random_state=random_state,
        ))

    if key in {"d3", "discriminative_drift"}:
        return _build_d3(**config)

    if key in {"dawidd", "dawidd_period"}:
        return _build_dawidd(**config)

    raise ValueError(f"Unsupported detector: {name}")


SUPPORTED_DETECTORS = ("kswin", "lsdd", "kdqtree", "d3", "dawidd")
