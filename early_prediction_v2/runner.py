from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from .data import PeriodDataset, load_dataset, protocol_defaults
from .metrics import period_metrics, summarize_periods
from .models import (
    DEEP_MODELS, SEQUENCE_MODELS, ModelState, SUPPORTED_MODELS, fit_model, score_model,
)
from .strategies import (
    SUPPORTED_DRIFT_DETECTORS,
    SUPPORTED_STRATEGIES,
    build_drift_schedules,
    expand_strategies,
)

CODE_VERSION = "google-backblaze-unified-final-v1-20260926"


def canonical_json_hash(payload: Any) -> str:
    """Stable SHA-256 identifier for configs and run signatures."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    base_model: str
    config: Dict[str, Any]
    update_config: Dict[str, Any] | None = None
    initial_candidate_id: str = "predeclared_default"
    update_candidate_id: str = "predeclared_default"


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_UPSTREAM_CONFIG = CONFIG_DIR / "drift_detector_config_v1.json"
DEFAULT_GOOGLE_MODEL_CONFIG = CONFIG_DIR / "google_model_config_v2.json"
DEFAULT_BACKBLAZE_MODEL_CONFIG = CONFIG_DIR / "backblaze_maintenance_v1.json"

PARAMETER_POLICIES = ("fixed", "initial_to_maintenance")


def _canonical_dataset_name(name: str) -> str:
    value = str(name).lower()
    if value in {"g", "google"}:
        return "google"
    if value in {"b", "backblaze"}:
        return "backblaze"
    if value in {"a", "alibaba"}:
        return "alibaba"
    raise ValueError(f"Unsupported dataset: {name}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _resolve_model_config_path(args: argparse.Namespace) -> Path:
    if args.model_config_file is not None:
        return Path(args.model_config_file).resolve()
    dataset = _canonical_dataset_name(args.dataset)
    if dataset == "google":
        return Path(args.google_model_config_file).resolve()
    if dataset == "backblaze":
        return Path(args.backblaze_model_config_file).resolve()
    raise ValueError(
        "No predeclared downstream configuration is defined for Alibaba in the current "
        "three-file formal configuration. Supply --model-config-file explicitly."
    )


def _validate_downstream_specs(
    specs: Sequence[ModelSpec],
    dataset: str,
) -> None:
    """Formal-run guardrails for predeclared downstream configurations."""
    missing_names = [s.base_model for s in specs if s.model_name == s.base_model]
    if missing_names:
        raise ValueError(
            "Formal downstream config must preserve full model_name for: "
            + ", ".join(missing_names)
        )
    missing_views = [s.base_model for s in specs if "feature_view" not in s.config]
    if missing_views:
        raise ValueError(
            "Formal downstream config must explicitly define feature_view for: "
            + ", ".join(missing_views)
        )
    for spec in specs:
        view = str(spec.config["feature_view"]).lower()
        if view not in {"base", "augmented"}:
            raise ValueError(
                f"Invalid feature_view for {dataset}/{spec.base_model}: {view}"
            )
        if spec.update_config is not None:
            update_view = str(spec.update_config.get("feature_view", "")).lower()
            if update_view != view:
                raise ValueError(
                    f"Initial/update feature_view mismatch for {dataset}/{spec.base_model}: "
                    f"{view} != {update_view}"
                )


def _resolve_parameter_policy(
    requested: str,
    config_payload: dict[str, Any],
) -> str:
    """Resolve and validate the policy encoded by a downstream config file."""
    declared = config_payload.get("parameter_policy")
    inferred = (
        str(declared)
        if declared is not None
        else "initial_to_maintenance"
        if any(
            isinstance(item, dict) and "update_config" in item
            for item in config_payload.get("models", {}).values()
        )
        else "fixed"
    )
    if inferred not in PARAMETER_POLICIES:
        raise ValueError(f"Unsupported parameter policy in model config: {inferred!r}")
    if requested != "auto" and requested != inferred:
        raise ValueError(
            f"--parameter-policy={requested!r} conflicts with model config policy "
            f"{inferred!r}"
        )
    return inferred


def _load_upstream_config(path: Path, requested: Sequence[str]) -> dict[str, Any]:
    payload = _read_json(path, "Upstream detector config")
    detectors = payload.get("detectors")
    if not isinstance(detectors, dict):
        raise ValueError("Upstream config must contain a 'detectors' mapping")
    wanted = list(dict.fromkeys(str(x).lower() for x in requested))
    missing = [name for name in wanted if name not in detectors]
    if missing:
        raise ValueError(
            "Upstream detector config does not define requested detector(s): "
            + ", ".join(missing)
        )
    return payload


def _detector_kwargs_from_config(
    upstream: dict[str, Any],
    detector_seed: int,
) -> dict[str, Any]:
    d = upstream["detectors"]
    kwargs: dict[str, Any] = {"random_state": int(detector_seed)}

    if "kswin" in d:
        c = d["kswin"]
        kwargs.update({
            "kswin_alpha": float(c["alpha"]),
            "kswin_window_size": int(c["window_size"]),
            "kswin_stat_size": int(c["stat_size"]),
            "kswin_samples_per_period": int(c["samples_per_period"]),
            "kswin_pca_variance": float(c["pca_variance"]),
        })

    if "lsdd" in d:
        c = d["lsdd"]
        kwargs.update({
            "lsdd_p_val": float(c["p_val"]),
            "lsdd_backend": str(c.get("backend", "pytorch")),
            "lsdd_n_permutations": int(c.get("n_permutations", 100)),
            "lsdd_n_kernel_centers": int(c["n_kernel_centers"]),
            "lsdd_lambda_rd_max": float(c["lambda_rd_max"]),
            "lsdd_max_samples": int(c["max_samples"]),
            "lsdd_device": str(c.get("device", "cpu")),
        })

    if "kdqtree" in d:
        c = d["kdqtree"]
        kwargs.update({
            "kdq_alpha": float(c["alpha"]),
            "kdq_bootstrap_samples": int(c["bootstrap_samples"]),
            "kdq_count_ubound": int(c["count_ubound"]),
            "kdq_cutpoint_proportion_lbound": float(
                c.get("cutpoint_proportion_lbound", 2e-10)
            ),
            "kdq_max_samples": int(c["max_samples"]),
        })

    if "d3" in d:
        c = d["d3"]
        kwargs.update({
            "d3_window_size": int(c["window_size"]),
            "d3_rho": float(c["rho"]),
            "d3_auc_threshold": float(c["auc_threshold"]),
            "d3_cv_folds": int(c["cv_folds"]),
            "d3_C": float(c["C"]),
            "d3_max_iter": int(c["max_iter"]),
            "d3_samples_per_period": int(c["samples_per_period"]),
        })

    if "dawidd" in d:
        c = d["dawidd"]
        kwargs.update({
            "dawidd_max_window_size": int(c["max_window_size"]),
            "dawidd_min_window_size": int(c["min_window_size"]),
            "dawidd_p_value_threshold": float(c["p_value_threshold"]),
            "dawidd_samples_per_period": int(c["samples_per_period"]),
            "dawidd_permutations": int(c["permutations"]),
            "dawidd_workers": int(c["workers"]),
        })

    return kwargs


def _formal_policy_from_upstream(
    upstream: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    scope = upstream.get("scope", {})
    if not isinstance(scope, dict):
        raise ValueError("Upstream config 'scope' must be a mapping")
    args.drift_confirm_consecutive = int(scope.get("confirm_consecutive", 1))
    args.drift_cooldown_periods = int(scope.get("cooldown_periods", 0))
    args.drift_seed_mode = str(scope.get("seed_mode", "experiment"))
    args.drift_feature_view = str(scope.get("drift_feature_view", "augmented"))
    if args.drift_confirm_consecutive != 1:
        raise ValueError("Formal config requires confirm_consecutive=1")
    if args.drift_cooldown_periods != 0:
        raise ValueError("Formal config requires cooldown_periods=0")
    if args.drift_seed_mode != "experiment":
        raise ValueError("Formal config requires detector_seed=experiment_seed")


@dataclass
class Runtime:
    strategy: str
    detector: str | None
    state: ModelState
    current_ref_start: int
    current_ref_end: int
    model_version: int = 0
    retrain_count: int = 0
    initial_train_seconds: float = 0.0
    retrain_seconds: float = 0.0
    score_seconds: float = 0.0
    updates: list[dict[str, Any]] = field(default_factory=list)


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        tmp.write_text("", encoding="utf-8")
        tmp.replace(path)
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    tmp.replace(path)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    frame = pd.read_csv(path)
    return frame.where(pd.notna(frame), None).to_dict("records")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_model_specs(path: Path, requested: Sequence[str]) -> list[ModelSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("models", payload.get("selected_models", payload))
    if not isinstance(selected, dict):
        raise ValueError("Model config must contain a 'models' mapping")
    wanted = list(dict.fromkeys(str(x).lower() for x in requested))
    specs: list[ModelSpec] = []
    for base in wanted:
        if base not in selected:
            raise ValueError(f"Config does not define requested model: {base}")
        item = selected[base]
        if "initial_config" in item or "update_config" in item:
            if "initial_config" not in item or "update_config" not in item:
                raise ValueError(
                    f"Transition config for {base} requires both initial_config and update_config"
                )
            config = dict(item["initial_config"])
            update_config = dict(item["update_config"])
            model_name = str(item.get("model_name", base))
            initial_candidate_id = str(item.get("initial_candidate_id", "initial_config"))
            update_candidate_id = str(item.get("update_candidate_id", "update_config"))
        elif "config" in item:
            config = dict(item["config"])
            update_config = None
            model_name = str(item.get("model_name", base))
            initial_candidate_id = "predeclared_default"
            update_candidate_id = "predeclared_default"
        else:
            config = dict(item)
            model_name = str(config.pop("model_name", base))
            update_config = None
            initial_candidate_id = "predeclared_default"
            update_candidate_id = "predeclared_default"
        for phase, phase_config in (("initial", config), ("update", update_config)):
            if phase_config is None:
                continue
            if str(phase_config.get("base_model", base)).lower() != base:
                raise ValueError(f"{phase} base_model mismatch for {base}")
            if str(phase_config.get("feature_view", "base")).lower() != str(
                config.get("feature_view", "base")
            ).lower():
                raise ValueError(f"Transition may not change feature_view for {base}")
        specs.append(ModelSpec(
            model_name, base, config, update_config,
            initial_candidate_id, update_candidate_id,
        ))
    return specs


def _validate_protocol(data: PeriodDataset, ref_start: int, ref_end: int, eval_start: int, eval_end: int) -> None:
    if not (1 <= ref_start <= ref_end < eval_start <= eval_end <= data.num_periods):
        raise ValueError(
            f"Invalid protocol: reference=P{ref_start}-P{ref_end}, "
            f"evaluation=P{eval_start}-P{eval_end}, available={data.num_periods}"
        )


def _resolve_feature_view(spec: ModelSpec, args: argparse.Namespace) -> str:
    explicit = spec.config.get("feature_view")
    if explicit is not None:
        return str(explicit).lower()
    ratio_models = {str(name).lower() for name in getattr(args, "ratio_feature_models", [])}
    if bool(getattr(args, "add_ratio_features", False)) and spec.base_model in ratio_models:
        return "augmented"
    return "base"


def _fit_for_reference(
    spec: ModelSpec,
    period_features: Sequence[np.ndarray],
    ref_start: int,
    ref_end: int,
    experiment_seed: int,
    calibration_samples: int,
    sampling_pool_size: int,
    parameter_policy: str,
    fit_stage: str = "initial",
) -> ModelState:
    win_size = int(ref_end - ref_start + 1)
    # Match the historical formal runner: model seed is constant within one
    # model/experiment, while reference sampling changes with the rolling window.
    model_seed = stable_seed(experiment_seed, spec.base_model, "model")
    sample_seed = stable_seed(
        experiment_seed, spec.base_model, "sample", int(ref_start - 1), win_size
    )
    if parameter_policy == "fixed":
        if spec.update_config is not None:
            raise ValueError(
                f"fixed policy does not accept update_config for {spec.base_model}"
            )
        config = dict(spec.config)
        candidate_id = spec.initial_candidate_id
        selection_auc = None
        selection_gain = None
        selection_period = None
        selection_reason = "predeclared_fixed_config"
    elif parameter_policy == "initial_to_maintenance":
        if spec.update_config is None:
            raise ValueError(
                f"{parameter_policy} requires initial_config/update_config for {spec.base_model}"
            )
        is_update = str(fit_stage).lower() == "update"
        config = dict(spec.update_config if is_update else spec.config)
        candidate_id = (
            spec.update_candidate_id if is_update else spec.initial_candidate_id
        )
        selection_auc = None
        selection_gain = None
        selection_period = None
        selection_reason = (
            "predeclared_maintenance_config"
            if is_update else "predeclared_initial_config"
        )
    else:
        raise ValueError(f"Unsupported parameter policy: {parameter_policy!r}")
    state = fit_model(
        model_name=spec.model_name,
        base_model=spec.base_model,
        reference_periods=period_features[ref_start - 1: ref_end],
        config=config,
        model_seed=model_seed,
        sample_seed=sample_seed,
        calibration_samples=int(calibration_samples),
        sampling_pool_size=int(sampling_pool_size),
    )
    state.parameter_policy = str(parameter_policy)
    state.parameter_candidate_id = candidate_id
    state.selected_config_hash = canonical_json_hash(config)
    state.selection_auc = selection_auc
    state.selection_gain_over_default = selection_gain
    state.selection_period = selection_period
    state.selection_reason = selection_reason
    return state


def _training_history_rows(
    state: ModelState,
    *,
    data: PeriodDataset,
    spec: ModelSpec,
    feature_view: str,
    experiment: int,
    experiment_seed: int,
    strategy: str,
    detector: str | None,
    fit_stage: str,
    model_version_after_fit: int,
    reference_start: int,
    reference_end: int,
    trigger_period: int | None = None,
    effective_period: int | None = None,
) -> list[dict[str, Any]]:
    """Flatten optional model training diagnostics into CSV-ready rows.

    Classical models simply return no rows.  Deep models expose an unlabeled
    ``training_history_`` sequence; labels are never read here.
    """
    history = getattr(state.model, "training_history_", None)
    if not history:
        return []

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(history, start=1):
        metrics = dict(item) if isinstance(item, dict) else {"total_loss": item}
        metrics.setdefault("epoch", float(index))
        row: dict[str, Any] = {
            "code_version": CODE_VERSION,
            "dataset": data.dataset,
            "protocol": data.protocol,
            "experiment": int(experiment),
            "seed": int(experiment_seed),
            "model": spec.base_model,
            "model_name": spec.model_name,
            "feature_view": feature_view,
            "strategy": strategy,
            "drift_detector": detector,
            "fit_stage": fit_stage,
            "model_version_after_fit": int(model_version_after_fit),
            "trigger_period": trigger_period,
            "effective_period": effective_period,
            "reference_start": int(reference_start),
            "reference_end": int(reference_end),
            "train_samples": int(state.train_samples),
            "calibration_samples": int(state.calibration_samples),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _run_model(
    data: PeriodDataset,
    spec: ModelSpec,
    strategies: Sequence[str],
    detectors: Sequence[str],
    schedules: dict,
    ref_start: int,
    ref_end: int,
    eval_start: int,
    eval_end: int,
    periodic_interval: int,
    top_ratios: Sequence[float],
    experiment: int,
    experiment_seed: int,
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    win_size = ref_end - ref_start + 1
    feature_view = _resolve_feature_view(spec, args)
    period_features = data.model_view(spec.base_model, feature_view)
    period_labels = data.labels_for_model(spec.base_model)
    started = time.perf_counter()
    initial_state = _fit_for_reference(
        spec, period_features, ref_start, ref_end, experiment_seed,
        args.calibration_samples, args.sampling_pool_size,
        args.parameter_policy,
        fit_stage="initial",
    )
    initial_train_seconds = time.perf_counter() - started
    training_rows: list[dict[str, Any]] = _training_history_rows(
        initial_state,
        data=data,
        spec=spec,
        feature_view=feature_view,
        experiment=experiment,
        experiment_seed=experiment_seed,
        strategy="shared_initial",
        detector=None,
        fit_stage="initial",
        model_version_after_fit=0,
        reference_start=ref_start,
        reference_end=ref_end,
    )

    runtimes: dict[str, Runtime] = {}
    for strategy, detector in expand_strategies(strategies, detectors):
        # Fitted model states are immutable during scoring.  Deep PyTorch models
        # are therefore shared across branches at cold start to avoid copying
        # the same network several times onto GPU memory.  A branch receives a
        # new independent state as soon as that branch retrains.  Classical
        # models retain the historical deepcopy behaviour.
        runtime_state = (
            initial_state
            if spec.base_model in set(DEEP_MODELS)
            else copy.deepcopy(initial_state)
        )
        runtimes[strategy] = Runtime(
            strategy=strategy,
            detector=detector,
            state=runtime_state,
            current_ref_start=ref_start,
            current_ref_end=ref_end,
            initial_train_seconds=float(initial_train_seconds),
        )

    period_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []

    # Periodic maintenance occurs at the start of each scheduled evaluation
    # period.  The first P19 update uses only P1-P18 and is therefore causal.
    # Drift-adaptive branches cannot update before observing a detector alarm,
    # so their P_t alarm still becomes effective in P_{t+1}.
    if (
        "periodic" in runtimes
        and args.parameter_policy == "initial_to_maintenance"
    ):
        runtime = runtimes["periodic"]
        old_candidate_id = runtime.state.parameter_candidate_id
        started = time.perf_counter()
        runtime.state = _fit_for_reference(
            spec, period_features, ref_start, ref_end, experiment_seed,
            args.calibration_samples, args.sampling_pool_size,
            args.parameter_policy,
            fit_stage="update",
        )
        elapsed = time.perf_counter() - started
        runtime.model_version = 1
        runtime.retrain_count = 1
        runtime.retrain_seconds = elapsed
        training_rows.extend(
            _training_history_rows(
                runtime.state,
                data=data,
                spec=spec,
                feature_view=feature_view,
                experiment=experiment,
                experiment_seed=experiment_seed,
                strategy="periodic",
                detector=None,
                fit_stage="retrain",
                model_version_after_fit=runtime.model_version,
                reference_start=ref_start,
                reference_end=ref_end,
                trigger_period=eval_start - 1,
                effective_period=eval_start,
            )
        )
        update_rows.append({
            "dataset": data.dataset,
            "experiment": experiment,
            "model": spec.base_model,
            "strategy": "periodic",
            "detector": None,
            "trigger_period": eval_start - 1,
            "effective_period": eval_start,
            "reference_start": ref_start,
            "reference_end": ref_end,
            "retrain_seconds": elapsed,
            "parameter_policy": runtime.state.parameter_policy,
            "old_parameter_candidate_id": old_candidate_id,
            "parameter_candidate_id": runtime.state.parameter_candidate_id,
            "parameter_changed": old_candidate_id != runtime.state.parameter_candidate_id,
            "selected_config_hash": runtime.state.selected_config_hash,
            "selection_auc": runtime.state.selection_auc,
            "selection_gain_over_default": runtime.state.selection_gain_over_default,
            "selection_period": runtime.state.selection_period,
            "selection_reason": runtime.state.selection_reason,
        })

    for period in range(eval_start, eval_end + 1):
        idx = period - 1
        x = period_features[idx]
        y = period_labels[idx]
        has_next = period < eval_end
        step = period - eval_start + 1
        rows_by_strategy: dict[str, dict[str, Any]] = {}

        for strategy, runtime in runtimes.items():
            score_started = time.perf_counter()
            scores = score_model(
                runtime.state, x, sequence_chunk_size=args.sequence_score_chunk_size
            )
            score_seconds = time.perf_counter() - score_started
            runtime.score_seconds += score_seconds
            metrics = period_metrics(y, scores, top_ratios)
            drift_decision = schedules.get(runtime.detector, {}).get(period) if runtime.detector else None
            row = {
                "code_version": CODE_VERSION,
                "dataset": data.dataset,
                "protocol": data.protocol,
                "experiment": experiment,
                "seed": experiment_seed,
                "model": spec.base_model,
                "model_name": spec.model_name,
                "feature_view": feature_view,
                "feature_dim_used": int(getattr(x, "feature_dim", x.shape[-1])),
                "input_kind": "sequence" if spec.base_model in set(SEQUENCE_MODELS) else "tabular",
                "sequence_window": int(getattr(x, "window_size", 0)) or None,
                "strategy": strategy,
                "drift_detector": runtime.detector,
                "period": period,
                "period_name": data.period_names[idx],
                "model_version_used": runtime.model_version,
                "parameter_policy": runtime.state.parameter_policy,
                "parameter_candidate_id_used": runtime.state.parameter_candidate_id,
                "selected_config_hash": runtime.state.selected_config_hash,
                "selection_auc_used": runtime.state.selection_auc,
                "selection_gain_over_default_used": runtime.state.selection_gain_over_default,
                "selection_period_used": runtime.state.selection_period,
                "reference_start": runtime.current_ref_start,
                "reference_end": runtime.current_ref_end,
                "train_samples": runtime.state.train_samples,
                "calibration_samples": runtime.state.calibration_samples,
                "sampling_pool_samples": runtime.state.sampling_pool_samples,
                "reference_samples": runtime.state.reference_samples,
                "detector_implementation": runtime.state.detector_implementation,
                "score_seconds": float(score_seconds),
                "retrained_after_period": False,
                "update_reason": "none",
                "new_reference_start": None,
                "new_reference_end": None,
                "drift_confirmed": None if drift_decision is None else drift_decision.confirmed_drift,
                "drift_score": None if drift_decision is None else drift_decision.score,
                "drift_threshold": None if drift_decision is None else drift_decision.threshold,
                **metrics,
            }
            rows_by_strategy[strategy] = row

        # Update only after evaluating P_t; the new model first scores P_{t+1}.
        if has_next:
            new_ref_end = period
            new_ref_start = new_ref_end - win_size + 1
            if "periodic" in runtimes and step % int(periodic_interval) == 0:
                runtime = runtimes["periodic"]
                old_candidate_id = runtime.state.parameter_candidate_id
                started = time.perf_counter()
                runtime.state = _fit_for_reference(
                    spec, period_features, new_ref_start, new_ref_end, experiment_seed,
                    args.calibration_samples, args.sampling_pool_size,
                    args.parameter_policy,
                    fit_stage="update",
                )
                elapsed = time.perf_counter() - started
                runtime.current_ref_start, runtime.current_ref_end = new_ref_start, new_ref_end
                runtime.model_version += 1
                runtime.retrain_count += 1
                runtime.retrain_seconds += elapsed
                training_rows.extend(
                    _training_history_rows(
                        runtime.state,
                        data=data,
                        spec=spec,
                        feature_view=feature_view,
                        experiment=experiment,
                        experiment_seed=experiment_seed,
                        strategy="periodic",
                        detector=None,
                        fit_stage="retrain",
                        model_version_after_fit=runtime.model_version,
                        reference_start=new_ref_start,
                        reference_end=new_ref_end,
                        trigger_period=period,
                        effective_period=period + 1,
                    )
                )
                rows_by_strategy["periodic"].update({
                    "retrained_after_period": True,
                    "update_reason": "periodic_schedule",
                    "new_reference_start": new_ref_start,
                    "new_reference_end": new_ref_end,
                    "next_parameter_candidate_id": runtime.state.parameter_candidate_id,
                })
                update_rows.append({
                    "dataset": data.dataset, "experiment": experiment, "model": spec.base_model,
                    "strategy": "periodic", "detector": None, "trigger_period": period,
                    "effective_period": period + 1, "reference_start": new_ref_start,
                    "reference_end": new_ref_end, "retrain_seconds": elapsed,
                    "parameter_policy": runtime.state.parameter_policy,
                    "old_parameter_candidate_id": old_candidate_id,
                    "parameter_candidate_id": runtime.state.parameter_candidate_id,
                    "parameter_changed": old_candidate_id != runtime.state.parameter_candidate_id,
                    "selected_config_hash": runtime.state.selected_config_hash,
                    "selection_auc": runtime.state.selection_auc,
                    "selection_gain_over_default": runtime.state.selection_gain_over_default,
                    "selection_period": runtime.state.selection_period,
                    "selection_reason": runtime.state.selection_reason,
                })

            for strategy, runtime in runtimes.items():
                if runtime.detector is None:
                    continue
                decision = schedules.get(runtime.detector, {}).get(period)
                if decision is None or not decision.update_trigger:
                    continue
                old_candidate_id = runtime.state.parameter_candidate_id
                started = time.perf_counter()
                runtime.state = _fit_for_reference(
                    spec, period_features, new_ref_start, new_ref_end, experiment_seed,
                    args.calibration_samples, args.sampling_pool_size,
                    args.parameter_policy,
                    fit_stage="update",
                )
                elapsed = time.perf_counter() - started
                runtime.current_ref_start, runtime.current_ref_end = new_ref_start, new_ref_end
                runtime.model_version += 1
                runtime.retrain_count += 1
                runtime.retrain_seconds += elapsed
                training_rows.extend(
                    _training_history_rows(
                        runtime.state,
                        data=data,
                        spec=spec,
                        feature_view=feature_view,
                        experiment=experiment,
                        experiment_seed=experiment_seed,
                        strategy=strategy,
                        detector=runtime.detector,
                        fit_stage="retrain",
                        model_version_after_fit=runtime.model_version,
                        reference_start=new_ref_start,
                        reference_end=new_ref_end,
                        trigger_period=period,
                        effective_period=period + 1,
                    )
                )
                reason = f"drift_{runtime.detector}"
                rows_by_strategy[strategy].update({
                    "retrained_after_period": True,
                    "update_reason": reason,
                    "new_reference_start": new_ref_start,
                    "new_reference_end": new_ref_end,
                    "next_parameter_candidate_id": runtime.state.parameter_candidate_id,
                })
                update_rows.append({
                    "dataset": data.dataset, "experiment": experiment, "model": spec.base_model,
                    "strategy": strategy, "detector": runtime.detector, "trigger_period": period,
                    "effective_period": period + 1, "reference_start": new_ref_start,
                    "reference_end": new_ref_end, "retrain_seconds": elapsed,
                    "parameter_policy": runtime.state.parameter_policy,
                    "old_parameter_candidate_id": old_candidate_id,
                    "parameter_candidate_id": runtime.state.parameter_candidate_id,
                    "parameter_changed": old_candidate_id != runtime.state.parameter_candidate_id,
                    "selected_config_hash": runtime.state.selected_config_hash,
                    "selection_auc": runtime.state.selection_auc,
                    "selection_gain_over_default": runtime.state.selection_gain_over_default,
                    "selection_period": runtime.state.selection_period,
                    "selection_reason": runtime.state.selection_reason,
                })

        period_rows.extend(rows_by_strategy.values())

    summary_rows: list[dict[str, Any]] = []
    for strategy, runtime in runtimes.items():
        rows = [row for row in period_rows if row["strategy"] == strategy]
        summary_rows.append({
            "code_version": CODE_VERSION,
            "dataset": data.dataset,
            "protocol": data.protocol,
            "experiment": experiment,
            "seed": experiment_seed,
            "model": spec.base_model,
            "model_name": spec.model_name,
            "feature_view": feature_view,
            "strategy": strategy,
            "drift_detector": runtime.detector,
            "parameter_policy": args.parameter_policy,
            "initial_parameter_candidate_id": initial_state.parameter_candidate_id,
            "distinct_parameter_groups_used": len({
                str(row["parameter_candidate_id_used"]) for row in rows
            }),
            "model_params": json.dumps({
                "initial_config": spec.config,
                "update_config": spec.update_config,
            }, sort_keys=True),
            "initial_train_seconds": runtime.initial_train_seconds,
            "retrain_count": runtime.retrain_count,
            "retrain_seconds": runtime.retrain_seconds,
            "score_seconds": runtime.score_seconds,
            **summarize_periods(rows, top_ratios),
        })

    # Framework-level comparison used by the paper: how much maintenance is
    # saved relative to Periodic, and how much predictive performance is retained.
    periodic_summary = next(
        (row for row in summary_rows if row["strategy"] == "periodic"),
        None,
    )
    if periodic_summary is not None:
        periodic_updates = int(periodic_summary.get("retrain_count", 0) or 0)
        for row in summary_rows:
            updates = int(row.get("retrain_count", 0) or 0)
            if periodic_updates > 0:
                row["update_ratio_vs_periodic"] = float(updates / periodic_updates)
                row["update_reduction_vs_periodic"] = float(
                    1.0 - updates / periodic_updates
                )
            else:
                row["update_ratio_vs_periodic"] = None
                row["update_reduction_vs_periodic"] = None

            for metric in ("mean_period_roc_auc", "mean_period_pr_auc"):
                periodic_value = periodic_summary.get(metric)
                value = row.get(metric)
                if periodic_value is None or value is None:
                    row[f"{metric}_gap_vs_periodic"] = None
                    row[f"{metric}_retention_vs_periodic"] = None
                    continue
                periodic_value = float(periodic_value)
                value = float(value)
                row[f"{metric}_gap_vs_periodic"] = float(value - periodic_value)
                row[f"{metric}_retention_vs_periodic"] = (
                    None
                    if abs(periodic_value) <= 1e-12
                    else float(value / periodic_value)
                )

    return period_rows, summary_rows, update_rows, training_rows


def run(args: argparse.Namespace) -> dict[str, Path]:
    defaults = protocol_defaults(args.dataset)
    ref_start = args.reference_start or defaults["reference_start"]
    ref_end = args.reference_end or defaults["reference_end"]
    eval_start = args.eval_start or defaults["eval_start"]
    eval_end = args.eval_end or defaults["eval_end"]
    top_ratios = args.top_ratios or defaults["top_ratios"]
    # Formal maintenance protocol:
    # batch detectors compare against the same period span used to train the
    # currently deployed downstream model.  This removes the legacy 3-period
    # reference ambiguity.  KSWIN/D3/DAWIDD keep their own native stream windows.
    drift_ref_periods = int(ref_end - ref_start + 1)

    model_config_path = _resolve_model_config_path(args)
    upstream_config_path = Path(args.upstream_config_file).resolve()
    upstream_config = _load_upstream_config(upstream_config_path, args.drift_detectors)
    _formal_policy_from_upstream(upstream_config, args)

    model_payload = _read_json(model_config_path, "Downstream model config")
    config_dataset = _canonical_dataset_name(model_payload.get("dataset", args.dataset))
    if config_dataset != _canonical_dataset_name(args.dataset):
        raise ValueError(
            f"Model config dataset {config_dataset!r} does not match "
            f"--dataset {_canonical_dataset_name(args.dataset)!r}"
        )
    args.parameter_policy = _resolve_parameter_policy(
        args.parameter_policy,
        model_payload,
    )

    specs = load_model_specs(model_config_path, args.models)
    _validate_downstream_specs(specs, _canonical_dataset_name(args.dataset))

    if args.parameter_policy == "initial_to_maintenance":
        missing = [spec.base_model for spec in specs if spec.update_config is None]
        if missing:
            raise ValueError(
                "Transition config lacks update_config for: " + ", ".join(missing)
            )
    else:
        unexpected = [spec.base_model for spec in specs if spec.update_config is not None]
        if unexpected:
            raise ValueError(
                "Fixed config unexpectedly defines update_config for: "
                + ", ".join(unexpected)
            )

    if any(str(spec.config.get("feature_view", "")).lower() == "augmented" for spec in specs):
        args.add_ratio_features = True

    sequence_specs = [spec for spec in specs if spec.base_model in set(SEQUENCE_MODELS)]
    requested_windows = {
        int(spec.config.get("sequence_window", args.sequence_window))
        for spec in sequence_specs
    }
    if len(requested_windows) > 1:
        raise ValueError(
            "All sequence models in one run must use the same sequence_window so they "
            "share one causally aligned dataset view. Run separate experiments otherwise."
        )
    sequence_window = next(iter(requested_windows), int(args.sequence_window))

    data = load_dataset(
        dataset=args.dataset,
        data_dir=args.data_dir,
        google_file=args.google_file,
        alibaba_file=args.alibaba_file,
        backblaze_file=args.backblaze_file,
        feature_mode=args.feature_mode,
        categorical_hash_buckets=args.categorical_hash_buckets,
        backblaze_log_transform=args.backblaze_log_transform,
        backblaze_feature_group=args.backblaze_feature_group,
        add_ratio_features=args.add_ratio_features,
        build_sequences=bool(sequence_specs),
        sequence_entity_col=args.sequence_entity_col,
        sequence_window=sequence_window,
        sequence_require_consecutive=args.sequence_require_consecutive,
    )
    _validate_protocol(data, ref_start, ref_end, eval_start, eval_end)
    for spec in specs:
        view = _resolve_feature_view(spec, args)
        # model_view performs the correct tabular/sequence validation.
        data.model_view(spec.base_model, view)

    # Drift schedules are built inside each experiment so stochastic detector
    # sampling/testing uses the same seed as that experiment.  This is essential
    # for genuine multi-seed stability evaluation; otherwise every experiment
    # would reuse one fixed detector schedule.
    drift_features = data.feature_view(args.drift_feature_view) if "drift_adaptive" in args.strategies else None

    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{data.dataset}_{args.run_name}"
    paths = {
        "period_metrics": result_dir / f"{prefix}_period_metrics.csv",
        "summary_by_run": result_dir / f"{prefix}_summary_by_run.csv",
        "drift_decisions": result_dir / f"{prefix}_drift_decisions.csv",
        "model_updates": result_dir / f"{prefix}_model_updates.csv",
        "training_history": result_dir / f"{prefix}_training_history.csv",
        "manifest": result_dir / f"{prefix}_manifest.json",
        "checkpoint": result_dir / f"{prefix}_checkpoint.json",
    }
    run_signature = canonical_json_hash({
        "dataset": _canonical_dataset_name(args.dataset),
        "models": list(args.models),
        "strategies": list(args.strategies),
        "detectors": list(args.drift_detectors),
        "reference": [ref_start, ref_end],
        "evaluation": [eval_start, eval_end],
        "periodic_interval": int(args.periodic_interval),
        "backblaze_feature_group": str(args.backblaze_feature_group),
        "seeds": [int(args.seed + i * args.seed_step) for i in range(args.n_experiments)],
        "parameter_policy": args.parameter_policy,
        "model_config_hash": canonical_json_hash(json.loads(model_config_path.read_text(encoding="utf-8"))),
        "upstream_config_hash": canonical_json_hash(upstream_config),
        "feature_mode": args.feature_mode,
        "backblaze_log_transform": bool(args.backblaze_log_transform),
        "add_ratio_features": bool(args.add_ratio_features),
    })

    all_period: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_updates: list[dict[str, Any]] = []
    all_training: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    completed: set[tuple[int, str]] = set()
    if args.resume and paths["checkpoint"].exists():
        checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
        if checkpoint.get("run_signature") != run_signature:
            raise ValueError(
                "Existing checkpoint belongs to a different experiment. "
                "Use another --run-name or disable --resume."
            )
        completed = {(int(x[0]), str(x[1])) for x in checkpoint.get("completed", [])}
        all_period = _read_csv_rows(paths["period_metrics"])
        all_summary = _read_csv_rows(paths["summary_by_run"])
        all_updates = _read_csv_rows(paths["model_updates"])
        all_training = _read_csv_rows(paths["training_history"])
        drift_rows = _read_csv_rows(paths["drift_decisions"])
        # If interruption happened after CSV replacement but before checkpoint
        # commit, discard that incomplete unit before recomputing it.
        def committed(row: dict[str, Any]) -> bool:
            try:
                return (int(row["experiment"]), str(row["model"])) in completed
            except (KeyError, TypeError, ValueError):
                return False
        all_period = [row for row in all_period if committed(row)]
        all_summary = [row for row in all_summary if committed(row)]
        all_updates = [row for row in all_updates if committed(row)]
        all_training = [row for row in all_training if committed(row)]

    for experiment_index in range(args.n_experiments):
        experiment = experiment_index + 1
        seed = args.seed + experiment_index * args.seed_step
        random.seed(seed)
        np.random.seed(seed)

        schedules: dict = {}
        if "drift_adaptive" in args.strategies:
            detector_seed = seed if args.drift_seed_mode == "experiment" else int(args.drift_seed)
            schedules, experiment_drift_rows = build_drift_schedules(
                period_features=drift_features,
                eval_start=eval_start,
                eval_end=eval_end,
                detectors=args.drift_detectors,
                drift_ref_periods=drift_ref_periods,
                calibration_start=ref_start,
                calibration_end=ref_end,
                confirm_consecutive=args.drift_confirm_consecutive,
                cooldown_periods=args.drift_cooldown_periods,
                model_reference_periods=int(ref_end - ref_start + 1),
                detector_kwargs=_detector_kwargs_from_config(
                    upstream_config,
                    detector_seed,
                ),
            )
            for row in experiment_drift_rows:
                row["experiment"] = int(experiment)
                row["seed"] = int(seed)
                row["drift_seed"] = int(detector_seed)
                row["drift_seed_mode"] = args.drift_seed_mode
            existing_drift = any(
                int(row.get("experiment", -1)) == experiment for row in drift_rows
            )
            if not existing_drift:
                drift_rows.extend(experiment_drift_rows)

        for spec in specs:
            unit = (int(experiment), spec.base_model)
            if unit in completed:
                print(f"[resume] skip experiment={experiment} model={spec.base_model}")
                continue
            periods, summaries, updates, training = _run_model(
                data=data, spec=spec, strategies=args.strategies,
                detectors=args.drift_detectors, schedules=schedules,
                ref_start=ref_start, ref_end=ref_end, eval_start=eval_start, eval_end=eval_end,
                periodic_interval=args.periodic_interval, top_ratios=top_ratios,
                experiment=experiment, experiment_seed=seed, args=args,
            )
            all_period.extend(periods)
            all_summary.extend(summaries)
            all_updates.extend(updates)
            all_training.extend(training)
            # One model/seed is the recovery unit. All files are replaced
            # atomically before the checkpoint marks the unit complete.
            _write_csv(paths["period_metrics"], all_period)
            _write_csv(paths["summary_by_run"], all_summary)
            _write_csv(paths["drift_decisions"], drift_rows)
            _write_csv(paths["model_updates"], all_updates)
            _write_csv(paths["training_history"], all_training)
            completed.add(unit)
            _write_json_atomic(paths["checkpoint"], {
                "run_signature": run_signature,
                "completed": sorted([list(x) for x in completed]),
                "updated_at": datetime.now().isoformat(),
            })
    _write_csv(paths["period_metrics"], all_period)
    _write_csv(paths["summary_by_run"], all_summary)
    _write_csv(paths["drift_decisions"], drift_rows)
    _write_csv(paths["model_updates"], all_updates)
    _write_csv(paths["training_history"], all_training)
    manifest = {
        "code_version": CODE_VERSION,
        "created_at": datetime.now().isoformat(),
        "task": (
            "unsupervised retraining with one predeclared fixed configuration"
            if args.parameter_policy == "fixed"
            else "unsupervised retraining with predeclared stage-specific configurations"
        ),
        "label_usage": (
            "labels are excluded from training, drift detection, retraining and "
            "scoring; they are used only for offline evaluation"
        ),
        "protocol": {
            "reference_start": ref_start, "reference_end": ref_end,
            "eval_start": eval_start, "eval_end": eval_end,
            "periodic_interval": args.periodic_interval,
            "drift_ref_periods": drift_ref_periods,
            "batch_reference_policy": "model_aligned_full_training_window",
            "drift_ref_periods_applies_to": ["lsdd", "kdqtree"],
            "stream_detector_reference": "KSWIN/D3/DAWIDD use native sample-stream windows; drift_ref_periods does not define their reference window",
            "top_ratios": list(top_ratios),
        },
        "dataset": data.dataset,
        "dataset_protocol": data.protocol,
        "feature_names": data.feature_names,
        "feature_names_by_view": data.feature_names_by_view,
        "feature_dim_by_view": {
            name: int(periods[0].shape[1])
            for name, periods in data.period_features_by_view.items()
        },
        "preprocessing": data.preprocessing,
        "sequence_metadata": data.sequence_metadata,
        "models": [spec.__dict__ for spec in specs],
        "parameter_policy": args.parameter_policy,
        "strategies": list(args.strategies),
        "experiment_seeds": [int(args.seed + i * args.seed_step) for i in range(args.n_experiments)],
        "drift_detectors": list(args.drift_detectors),
        "configuration_files": {
            "upstream": str(upstream_config_path),
            "downstream": str(model_config_path),
        },
        "formal_config_status": {
            "upstream_config_version": upstream_config.get("config_version"),
            "upstream_status": upstream_config.get("status"),
        },
        "drift_detector_config": {
            "drift_feature_view": args.drift_feature_view,
            "drift_ref_periods": drift_ref_periods,
            "model_reference_periods": int(ref_end - ref_start + 1),
            "drift_confirm_consecutive": args.drift_confirm_consecutive,
            "drift_cooldown_periods": args.drift_cooldown_periods,
            "drift_seed_mode": args.drift_seed_mode,
            "experiment_seed_rule": "detector_seed = experiment_seed",
            "detectors": {
                name: upstream_config["detectors"][name]
                for name in args.drift_detectors
            },
        },
        "output_files": {k: str(v) for k, v in paths.items()},
    }
    _write_json_atomic(paths["manifest"], manifest)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Google/Backblaze unsupervised maintenance experiment",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", required=True, choices=["g", "google", "a", "alibaba", "b", "backblaze"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--google-file", default="google_job_failure_g2.csv")
    parser.add_argument("--alibaba-file", default="alibaba_job_data_g1.csv")
    parser.add_argument("--backblaze-file", default="disk_failure_v2.csv")
    parser.add_argument("--feature-mode", choices=["full", "numeric_only", "group_weighted"], default="group_weighted")
    parser.add_argument("--categorical-hash-buckets", type=int, default=32)
    parser.add_argument("--backblaze-log-transform", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--backblaze-feature-group",
        choices=["full", "sector_proxy_only", "sector_proxy_removed"],
        default="full",
        help="Predeclared Backblaze feature ablation; ignored by other datasets.",
    )
    parser.add_argument("--add-ratio-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--sequence-entity-col", default="serial_number",
        help="Persistent entity key used only by sequence models (Backblaze normally serial_number).",
    )
    parser.add_argument(
        "--sequence-window", type=int, default=8,
        help="Entity-consistent sequence length including the current target row.",
    )
    parser.add_argument(
        "--sequence-require-consecutive", action=argparse.BooleanOptionalAction, default=True,
        help="For Backblaze, require every adjacent observation in a window to be exactly one day apart.",
    )
    parser.add_argument(
        "--sequence-score-chunk-size", type=int, default=50000,
        help="Lazy sequence materialization chunk size during scoring.",
    )
    parser.add_argument(
        "--ratio-feature-models", nargs="*", choices=list(SUPPORTED_MODELS), default=[]
    )

    parser.add_argument(
        "--upstream-config-file",
        type=Path,
        default=DEFAULT_UPSTREAM_CONFIG,
        help="Authoritative predeclared upstream detector configuration.",
    )
    parser.add_argument(
        "--google-model-config-file",
        type=Path,
        default=DEFAULT_GOOGLE_MODEL_CONFIG,
        help="Predeclared Google downstream-model configuration.",
    )
    parser.add_argument(
        "--backblaze-model-config-file",
        type=Path,
        default=DEFAULT_BACKBLAZE_MODEL_CONFIG,
        help="Predeclared Backblaze downstream-model configuration.",
    )
    parser.add_argument(
        "--model-config-file",
        type=Path,
        default=None,
        help="Optional explicit downstream config override for non-standard runs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(SUPPORTED_MODELS),
        default=["if", "pca", "ae", "deep_svdd", "neutral_ad"],
    )
    parser.add_argument("--strategies", nargs="+", choices=list(SUPPORTED_STRATEGIES), default=list(SUPPORTED_STRATEGIES))
    parser.add_argument("--drift-detectors", nargs="+", choices=list(SUPPORTED_DRIFT_DETECTORS), default=list(SUPPORTED_DRIFT_DETECTORS))

    parser.add_argument("--reference-start", type=int)
    parser.add_argument("--reference-end", type=int)
    parser.add_argument("--eval-start", type=int)
    parser.add_argument("--eval-end", type=int)
    parser.add_argument("--periodic-interval", type=int, default=1)
    parser.add_argument("--top-ratios", nargs="+", type=float)
    parser.add_argument("--n-experiments", type=int, default=3)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--seed-step", type=int, default=1000)
    parser.add_argument("--result-dir", type=Path, default=Path("results/early_prediction"))
    parser.add_argument(
        "--run-name", default="adaptive_experiment",
        help="Stable output prefix. Reuse it with --resume after interruption.",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume completed seed/model units from atomic checkpoints.",
    )
    parser.add_argument("--calibration-samples", type=int, default=20000)
    parser.add_argument("--sampling-pool-size", type=int, default=120000)
    parser.add_argument(
        "--parameter-policy",
        choices=["auto", *PARAMETER_POLICIES],
        default="auto",
        help=(
            "Infer the policy from the model config (recommended), use one fixed "
            "config, or use the predeclared initial configuration followed by the maintenance configuration."
        ),
    )

    parser.add_argument("--drift-feature-view", choices=["base", "augmented"], default="augmented")
    parser.add_argument("--drift-ref-periods", type=int, default=0)
    parser.add_argument("--drift-confirm-consecutive", type=int, default=1)
    parser.add_argument("--drift-cooldown-periods", type=int, default=0)
    parser.add_argument("--drift-seed", type=int, default=42, help="Fixed detector seed used only when --drift-seed-mode=fixed")
    parser.add_argument(
        "--drift-seed-mode", choices=["experiment", "fixed"], default="experiment",
        help="Use each experiment seed for drift detectors (recommended for multi-seed runs), or reuse --drift-seed.",
    )
    # Detector hyperparameters are intentionally not exposed in formal mode.
    # They are read from --upstream-config-file.
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.models = list(dict.fromkeys(str(x).lower() for x in args.models))
    args.strategies = list(dict.fromkeys(str(x).lower() for x in args.strategies))
    args.drift_detectors = list(dict.fromkeys(str(x).lower() for x in args.drift_detectors))
    args.ratio_feature_models = list(dict.fromkeys(str(x).lower() for x in args.ratio_feature_models))
    if args.add_ratio_features and not args.ratio_feature_models:
        args.ratio_feature_models = list(args.models)
    if args.periodic_interval <= 0 or args.n_experiments <= 0:
        raise ValueError("periodic-interval and n-experiments must be positive")
    if args.calibration_samples <= 0 or args.sampling_pool_size <= 0:
        raise ValueError("calibration-samples and sampling-pool-size must be positive")
    if args.sequence_window < 2 or args.sequence_score_chunk_size <= 0:
        raise ValueError("sequence-window must be >=2 and sequence-score-chunk-size must be positive")
    if args.top_ratios is not None and any(not 0.0 < float(x) <= 1.0 for x in args.top_ratios):
        raise ValueError("Every top ratio must be in (0, 1]")
    paths = run(args)
    print("Saved:")
    for name, path in paths.items():
        print(f"  {name:20s}: {path}")

if __name__ == "__main__":
    main()
