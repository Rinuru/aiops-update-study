#!/usr/bin/env python3
"""Load early-fault-prediction datasets into natural periods and feature views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .sequence_data import SequencePeriod, build_entity_sequence_periods

import numpy as np
import pandas as pd

GOOGLE_ORIGIN_US = 604_046_279
GOOGLE_DAY_US = 86_400_000_000
ALIBABA_ORIGIN_S = 494_319 + 3 * 86_400
ALIBABA_DAY_S = 86_400

GOOGLE_NUMERIC = [
    "Num Tasks", "CPU Requested", "Mem Requested", "Disk Requested",
    "Avg CPU", "Avg Mem", "Avg Disk", "Std CPU", "Std Mem", "Std Disk",
]
GOOGLE_RATIO_FEATURES = [
    "CPU Usage / Requested", "Mem Usage / Requested", "Disk Usage / Requested",
    "CPU CV", "Mem CV", "Disk CV",
]
GOOGLE_CATEGORICAL = [
    ("User ID", 32), ("Job Name", 32), ("Scheduling Class", 8),
    ("Priority", 16), ("Diff Machine", 2),
]
ALIBABA_NUMERIC = ["inst_num", "plan_cpu", "plan_mem", "plan_gpu"]
ALIBABA_CATEGORICAL = [("user", 32), ("task_name", 32)]

BACKBLAZE_RAW = [
    "smart_1_raw", "smart_4_raw", "smart_5_raw", "smart_7_raw",
    "smart_9_raw", "smart_12_raw", "smart_187_raw", "smart_193_raw",
    "smart_194_raw", "smart_197_raw", "smart_199_raw",
]
BACKBLAZE_DIFF = [
    "smart_4_raw_diff", "smart_5_raw_diff", "smart_9_raw_diff",
    "smart_12_raw_diff", "smart_187_raw_diff", "smart_193_raw_diff",
    "smart_197_raw_diff", "smart_199_raw_diff",
]
BACKBLAZE_SECTOR_PROXY = [
    "smart_5_raw", "smart_187_raw", "smart_197_raw",
    "smart_5_raw_diff", "smart_187_raw_diff", "smart_197_raw_diff",
]
BACKBLAZE_FEATURE_GROUPS = ("full", "sector_proxy_only", "sector_proxy_removed")


def _backblaze_feature_names(group: str) -> list[str]:
    """Return one predeclared feature group; never select features from labels."""
    key = str(group).strip().lower()
    all_features = list(BACKBLAZE_RAW + BACKBLAZE_DIFF)
    if key == "full":
        return all_features
    if key == "sector_proxy_only":
        return list(BACKBLAZE_SECTOR_PROXY)
    if key == "sector_proxy_removed":
        proxy = set(BACKBLAZE_SECTOR_PROXY)
        return [name for name in all_features if name not in proxy]
    raise ValueError(
        f"Unknown Backblaze feature group {group!r}; "
        f"choose one of {BACKBLAZE_FEATURE_GROUPS}"
    )


@dataclass
class PeriodDataset:
    dataset: str
    protocol: str
    period_features: list[np.ndarray]
    period_labels: list[np.ndarray]
    period_names: list[str]
    feature_names: list[str]
    preprocessing: dict
    period_features_by_view: dict[str, list[np.ndarray]]
    feature_names_by_view: dict[str, list[str]]
    period_sequences_by_view: dict[str, list[SequencePeriod]] | None = None
    sequence_metadata: dict | None = None

    @property
    def num_periods(self) -> int:
        return len(self.period_features)

    def feature_view(self, name: str) -> list[np.ndarray]:
        key = str(name).lower()
        if key not in self.period_features_by_view:
            raise ValueError(
                f"Unknown feature view {name!r}; available={sorted(self.period_features_by_view)}"
            )
        return self.period_features_by_view[key]

    def model_view(self, model_name: str, feature_view: str):
        """Return model-aligned period inputs without touching labels.

        Tabular models receive the historical N x D matrices. Sequence models
        receive lazy entity-consistent SequencePeriod objects.
        """
        key = str(model_name).lower()
        view = str(feature_view).lower()
        if key in {"deepant", "lstm_ed"}:
            if not self.period_sequences_by_view:
                raise ValueError(
                    f"Model {model_name!r} requires entity-consistent sequence data, "
                    "but no sequence view was built for this dataset."
                )
            if view not in self.period_sequences_by_view:
                raise ValueError(
                    f"Sequence feature view {view!r} unavailable; "
                    f"available={sorted(self.period_sequences_by_view)}"
                )
            return self.period_sequences_by_view[view]
        return self.feature_view(view)

    def labels_for_model(self, model_name: str) -> list[np.ndarray]:
        key = str(model_name).lower()
        if key in {"deepant", "lstm_ed"}:
            if not self.period_sequences_by_view:
                raise ValueError(f"No sequence labels available for model {model_name!r}")
            # All feature views share the same target rows and labels.
            periods = next(iter(self.period_sequences_by_view.values()))
            return [period.labels for period in periods]
        return self.period_labels


def _numeric(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return (
        frame[list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def _hashed_one_hot(
    values: pd.Series,
    name: str,
    buckets: int,
) -> tuple[np.ndarray, list[str]]:
    buckets = max(2, int(buckets))
    text = values.fillna("<NA>").astype(str)
    hashes = pd.util.hash_pandas_object(text, index=False).to_numpy(dtype=np.uint64)
    index = (hashes % np.uint64(buckets)).astype(np.int64)
    encoded = np.zeros((len(text), buckets), dtype=np.float32)
    encoded[np.arange(len(text)), index] = 1.0
    return encoded, [f"{name}__hash_{i}" for i in range(buckets)]


def _assemble(
    frame: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_specs: Sequence[tuple[str, int]],
    feature_mode: str,
    categorical_hash_buckets: int,
) -> tuple[np.ndarray, list[str]]:
    mode = str(feature_mode).lower()
    if mode not in {"full", "numeric_only", "group_weighted"}:
        raise ValueError("feature_mode must be full, numeric_only, or group_weighted")

    parts = [_numeric(frame, numeric_columns)]
    names = list(numeric_columns)
    if mode != "numeric_only":
        for name, default_buckets in categorical_specs:
            buckets = categorical_hash_buckets if default_buckets == 32 else default_buckets
            encoded, encoded_names = _hashed_one_hot(frame[name], name, buckets)
            # The historical 'group_weighted' protocol intentionally kept the
            # one-hot groups at unit norm, so it is numerically identical here to full.
            parts.append(encoded)
            names.extend(encoded_names)

    features = np.hstack(parts).astype(np.float32, copy=False)
    features = np.nan_to_num(
        features, nan=0.0, posinf=1e6, neginf=-1e6
    ).astype(np.float32, copy=False)
    return features, names


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-6) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    den = pd.to_numeric(denominator, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out = np.zeros_like(num, dtype=np.float64)
    valid = np.abs(den) > float(eps)
    out[valid] = num[valid] / np.abs(den[valid])
    out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
    out = np.clip(out, -1e6, 1e6)
    return pd.Series(out, index=numerator.index)


def _add_google_ratio_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Predeclared Google augmented view used in the final V2 protocol."""
    out = frame.copy()
    out["CPU Usage / Requested"] = _safe_ratio(out["Avg CPU"], out["CPU Requested"])
    out["Mem Usage / Requested"] = _safe_ratio(out["Avg Mem"], out["Mem Requested"])
    out["Disk Usage / Requested"] = _safe_ratio(out["Avg Disk"], out["Disk Requested"])
    out["CPU CV"] = _safe_ratio(out["Std CPU"], out["Avg CPU"])
    out["Mem CV"] = _safe_ratio(out["Std Mem"], out["Avg Mem"])
    out["Disk CV"] = _safe_ratio(out["Std Disk"], out["Avg Disk"])
    return out


def _split_features(
    features: np.ndarray,
    period_index: np.ndarray,
    count: int,
    prefix: str,
) -> list[np.ndarray]:
    xs: list[np.ndarray] = []
    for period in range(1, count + 1):
        mask = period_index == period
        x = features[mask].astype(np.float32, copy=False)
        if len(x) == 0:
            raise ValueError(f"Empty natural period: {prefix}{period}")
        xs.append(x)
    return xs


def _split_labels(
    labels: np.ndarray,
    period_index: np.ndarray,
    count: int,
    prefix: str,
) -> tuple[list[np.ndarray], list[str]]:
    ys: list[np.ndarray] = []
    names: list[str] = []
    for period in range(1, count + 1):
        mask = period_index == period
        y = labels[mask].astype(np.int8, copy=False)
        if len(y) == 0:
            raise ValueError(f"Empty natural period: {prefix}{period}")
        ys.append(y)
        names.append(f"{prefix}{period}")
    return ys, names


def _same_view_dataset(
    dataset: str,
    protocol: str,
    features: np.ndarray,
    labels: np.ndarray,
    periods: np.ndarray,
    count: int,
    prefix: str,
    feature_names: list[str],
    preprocessing: dict,
) -> PeriodDataset:
    xs = _split_features(features, periods, count, prefix)
    ys, names = _split_labels(labels, periods, count, prefix)
    return PeriodDataset(
        dataset=dataset,
        protocol=protocol,
        period_features=xs,
        period_labels=ys,
        period_names=names,
        feature_names=list(feature_names),
        preprocessing=preprocessing,
        period_features_by_view={"base": xs, "augmented": xs},
        feature_names_by_view={"base": list(feature_names), "augmented": list(feature_names)},
    )


def load_google_g2(
    path: Path,
    feature_mode: str,
    categorical_hash_buckets: int,
    add_ratio_features: bool = False,
) -> PeriodDataset:
    frame = pd.read_csv(path, low_memory=False)
    required = [
        "Status", "Start Time", *GOOGLE_NUMERIC,
        *[name for name, _ in GOOGLE_CATEGORICAL],
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Google G2 missing columns: {missing}")

    start = pd.to_numeric(frame["Start Time"], errors="coerce")
    if start.isna().any():
        raise ValueError(f"Google G2 invalid Start Time rows: {int(start.isna().sum())}")
    periods = (
        np.floor((start.to_numpy(dtype=np.float64) - GOOGLE_ORIGIN_US) / GOOGLE_DAY_US)
        .astype(np.int64)
        + 1
    )
    outside = int(np.sum((periods < 1) | (periods > 28)))
    if outside:
        raise ValueError(f"Google G2 rows outside P1-P28: {outside}")

    base_frame = frame.copy()
    base_features, base_names = _assemble(
        base_frame, GOOGLE_NUMERIC, GOOGLE_CATEGORICAL,
        feature_mode, categorical_hash_buckets,
    )
    if add_ratio_features:
        augmented_frame = frame.copy()
        numeric = _add_google_ratio_features(augmented_frame[GOOGLE_NUMERIC].copy())
        for column in GOOGLE_RATIO_FEATURES:
            augmented_frame[column] = numeric[column]
        augmented_features, augmented_names = _assemble(
            augmented_frame,
            [*GOOGLE_NUMERIC, *GOOGLE_RATIO_FEATURES],
            GOOGLE_CATEGORICAL,
            feature_mode,
            categorical_hash_buckets,
        )
    else:
        augmented_features, augmented_names = base_features, base_names

    labels = (
        pd.to_numeric(frame["Status"], errors="coerce").fillna(-1).to_numpy() == 3
    ).astype(np.int8)
    base_xs = _split_features(base_features, periods, 28, "D")
    augmented_xs = _split_features(augmented_features, periods, 28, "D")
    ys, names = _split_labels(labels, periods, 28, "D")
    return PeriodDataset(
        dataset="g",
        protocol="google_g2",
        # Backward-compatible default is augmented; model runs choose explicitly.
        period_features=augmented_xs,
        period_labels=ys,
        period_names=names,
        feature_names=list(augmented_names),
        preprocessing={
            "file": str(path),
            "feature_mode": feature_mode,
            "categorical_hash_buckets": categorical_hash_buckets,
            "add_ratio_features": bool(add_ratio_features),
            "ratio_features": list(GOOGLE_RATIO_FEATURES) if add_ratio_features else [],
            "time_excluded_from_model": True,
            "label_excluded_from_model": True,
        },
        period_features_by_view={"base": base_xs, "augmented": augmented_xs},
        feature_names_by_view={"base": list(base_names), "augmented": list(augmented_names)},
    )


def load_alibaba_g1(
    path: Path,
    feature_mode: str,
    categorical_hash_buckets: int,
) -> PeriodDataset:
    frame = pd.read_csv(path, low_memory=False)
    required = [
        "start_time", "status", *ALIBABA_NUMERIC,
        *[name for name, _ in ALIBABA_CATEGORICAL],
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Alibaba G1 missing columns: {missing}")

    runtime_fields = [
        "cpu_usage", "gpu_wrk_util", "avg_mem", "max_mem",
        "avg_gpu_wrk_mem", "max_gpu_wrk_mem",
    ]
    present_runtime = [name for name in runtime_fields if name in frame.columns]
    if present_runtime:
        raise ValueError(
            "This is not the prepared Alibaba G1 file; runtime fields remain: "
            f"{present_runtime}"
        )

    start = pd.to_numeric(frame["start_time"], errors="coerce")
    if start.isna().any():
        raise ValueError(f"Alibaba G1 invalid start_time rows: {int(start.isna().sum())}")
    periods = (
        np.floor((start.to_numpy(dtype=np.float64) - ALIBABA_ORIGIN_S) / ALIBABA_DAY_S)
        .astype(np.int64)
        + 1
    )
    outside = int(np.sum((periods < 1) | (periods > 56)))
    if outside:
        raise ValueError(f"Alibaba G1 rows outside P1-P56: {outside}")

    features, feature_names = _assemble(
        frame, ALIBABA_NUMERIC, ALIBABA_CATEGORICAL,
        feature_mode, categorical_hash_buckets,
    )
    labels = (frame["status"].fillna("").astype(str).to_numpy() == "Failed").astype(np.int8)
    return _same_view_dataset(
        "a", "alibaba_g1", features, labels, periods, 56, "D", list(feature_names),
        {
            "file": str(path),
            "feature_mode": feature_mode,
            "categorical_hash_buckets": categorical_hash_buckets,
            "runtime_features_excluded": True,
            "time_excluded_from_model": True,
            "label_excluded_from_model": True,
        },
    )


def _signed_log1p(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return (np.sign(values) * np.log1p(np.abs(values))).astype(np.float32)


def load_backblaze(
    path: Path,
    log_transform: bool,
    feature_group: str = "full",
    build_sequences: bool = False,
    sequence_entity_col: str = "serial_number",
    sequence_window: int = 8,
    sequence_require_consecutive: bool = True,
) -> PeriodDataset:
    frame = pd.read_csv(path, low_memory=False)
    required = ["date", *BACKBLAZE_RAW, *BACKBLAZE_DIFF, "label"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Backblaze missing columns: {missing}")

    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Backblaze invalid dates: {int(dates.isna().sum())}")
    absolute_month = dates.dt.year * 12 + dates.dt.month
    periods = (absolute_month - int(absolute_month.min()) + 1).to_numpy(dtype=np.int64)
    if int(np.max(periods)) != 36:
        raise ValueError(f"Expected 36 Backblaze months, found {int(np.max(periods))}")

    selected_features = _backblaze_feature_names(feature_group)
    selected = set(selected_features)
    if log_transform:
        columns: list[np.ndarray] = []
        for name in BACKBLAZE_RAW:
            if name not in selected:
                continue
            values = pd.to_numeric(frame[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            if name == "smart_194_raw":
                columns.append(values.astype(np.float32))
            else:
                columns.append(np.log1p(np.clip(values, 0.0, None)).astype(np.float32))
        for name in BACKBLAZE_DIFF:
            if name in selected:
                columns.append(_signed_log1p(frame[name]))
        features = np.column_stack(columns).astype(np.float32, copy=False)
    else:
        features = _numeric(frame, selected_features)

    labels = pd.to_numeric(frame["label"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    dataset = _same_view_dataset(
        "b", "backblaze_original_future_7d", features, labels, periods, 36, "M",
        selected_features,
        {
            "file": str(path),
            "feature_mode": "numeric_only",
            "backblaze_log_transform": bool(log_transform),
            "backblaze_feature_group": str(feature_group).lower(),
            "selected_features": selected_features,
            "time_excluded_from_model": True,
            "label_excluded_from_model": True,
        },
    )
    if build_sequences:
        entity_col = str(sequence_entity_col).strip()
        if not entity_col:
            raise ValueError(
                "Sequence models require --sequence-entity-col, e.g. serial_number for Backblaze"
            )
        if entity_col not in frame.columns:
            raise ValueError(
                f"Backblaze sequence model requested entity column {entity_col!r}, "
                "but it is absent from the prepared CSV. Rebuild/retain a file with "
                "persistent disk identity (normally serial_number); do not fabricate "
                "sequences from unrelated rows."
            )
        # Pandas datetime64[ns] is represented as integer nanoseconds.
        one_day_ns = 86_400_000_000_000
        seq_views, seq_meta = build_entity_sequence_periods(
            features_by_view={"base": features, "augmented": features},
            labels=labels,
            periods=periods,
            entity_values=frame[entity_col],
            time_values=dates,
            period_count=36,
            period_prefix="M",
            window_size=int(sequence_window),
            require_consecutive=bool(sequence_require_consecutive),
            expected_step=one_day_ns if sequence_require_consecutive else None,
        )
        seq_meta.update({
            "entity_column": entity_col,
            "time_column": "date",
            "target_semantics": "window ends at current disk-day; label remains future-7d failure",
            "cross_entity_windows_forbidden": True,
        })
        dataset.period_sequences_by_view = seq_views
        dataset.sequence_metadata = seq_meta
        dataset.preprocessing["sequence_adapter"] = seq_meta
    return dataset


def load_dataset(
    dataset: str,
    data_dir: Path,
    google_file: str,
    alibaba_file: str,
    backblaze_file: str,
    feature_mode: str,
    categorical_hash_buckets: int,
    backblaze_log_transform: bool,
    backblaze_feature_group: str = "full",
    add_ratio_features: bool = False,
    build_sequences: bool = False,
    sequence_entity_col: str = "serial_number",
    sequence_window: int = 8,
    sequence_require_consecutive: bool = True,
) -> PeriodDataset:
    key = str(dataset).strip().lower()
    if key in {"g", "google"}:
        if build_sequences:
            raise ValueError(
                "Google G2 sequence models are intentionally disabled: the prepared G2 file "
                "contains one aggregated job-level vector and Job ID was removed. Recover a "
                "true within-job time series before using DeepAnT/LSTM-ED; do not slide across jobs."
            )
        return load_google_g2(
            data_dir / google_file,
            feature_mode,
            categorical_hash_buckets,
            add_ratio_features=add_ratio_features,
        )
    if key in {"a", "alibaba"}:
        if build_sequences:
            raise ValueError(
                "Alibaba G1 sequence models are not enabled for the prepared planning-time file. "
                "A verified persistent entity/time sequence must be defined first."
            )
        return load_alibaba_g1(
            data_dir / alibaba_file, feature_mode, categorical_hash_buckets
        )
    if key in {"b", "backblaze"}:
        return load_backblaze(
            data_dir / backblaze_file,
            backblaze_log_transform,
            feature_group=backblaze_feature_group,
            build_sequences=build_sequences,
            sequence_entity_col=sequence_entity_col,
            sequence_window=sequence_window,
            sequence_require_consecutive=sequence_require_consecutive,
        )
    raise ValueError("dataset must be g/google, a/alibaba, or b/backblaze")


def protocol_defaults(dataset: str) -> dict:
    """Canonical early-failure-prediction protocol boundaries (1-based periods)."""
    key = str(dataset).strip().lower()
    if key in {"g", "google"}:
        return {
            "reference_start": 1,
            "reference_end": 14,
            "eval_start": 15,
            "eval_end": 28,
            "top_ratios": [0.001, 0.01],
            "drift_ref_periods": 3,
        }
    if key in {"a", "alibaba"}:
        return {
            "reference_start": 1,
            "reference_end": 28,
            "eval_start": 29,
            "eval_end": 56,
            "top_ratios": [0.01, 0.05, 0.10],
            "drift_ref_periods": 14,
        }
    if key in {"b", "backblaze"}:
        return {
            "reference_start": 1,
            "reference_end": 18,
            "eval_start": 19,
            "eval_end": 36,
            "top_ratios": [0.001, 0.01],
            "drift_ref_periods": 3,
        }
    raise ValueError(f"Unknown dataset: {dataset}")
