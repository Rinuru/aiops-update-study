from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence
import numpy as np

from .detectors import build_detector

SUPPORTED_STRATEGIES = ("stationary", "periodic", "drift_adaptive")
SUPPORTED_DRIFT_DETECTORS = ("kswin", "lsdd", "kdqtree", "d3", "dawidd")


@dataclass(frozen=True)
class DriftDecision:
    detector: str
    period: int
    valid: bool
    raw_drift: bool
    confirmed_drift: bool
    update_trigger: bool
    score: float | None
    threshold: float | None
    reference_start: int
    reference_end: int
    affects_period: int | None
    reason: str


def expand_strategies(strategies: Sequence[str], detectors: Sequence[str]) -> list[tuple[str, str | None]]:
    requested = list(dict.fromkeys(str(x).lower() for x in strategies))
    detector_names = list(dict.fromkeys(str(x).lower() for x in detectors))
    out: list[tuple[str, str | None]] = []
    for name in requested:
        if name == "drift_adaptive":
            out.extend((f"drift_{detector}", detector) for detector in detector_names)
        else:
            out.append((name, None))
    return out


def build_drift_schedules(
    period_features: Sequence[np.ndarray],
    eval_start: int,
    eval_end: int,
    detectors: Sequence[str],
    drift_ref_periods: int,
    calibration_start: int | None = None,
    calibration_end: int | None = None,
    confirm_consecutive: int = 1,
    cooldown_periods: int = 0,
    detector_kwargs: Dict[str, Any] | None = None,
    model_reference_periods: int | None = None,
) -> tuple[dict[str, dict[int, DriftDecision]], list[dict[str, Any]]]:
    """Build label-free update schedules.

    Maintenance alignment:
    - KS/LSDD/KDQTree keep their mature statistics unchanged.
    - Their reference is initialized from the model's initial training reference.
    - The reference remains fixed while the deployed model remains unchanged.
    - After a confirmed drift at period t triggers model retraining for t+1,
      the detector reference is synchronously reset to the SAME rolling period
      window that the predictor uses for retraining.
    - D3/DAWIDD retain their own native stream/adaptive-memory mechanics.

    No labels are used. A drift found in P_t can only affect P_{t+1}.
    """
    kwargs = dict(detector_kwargs or {})
    schedules: dict[str, dict[int, DriftDecision]] = {}
    rows: list[dict[str, Any]] = []
    n_periods = len(period_features)
    drift_ref_periods = max(1, int(drift_ref_periods))

    cal_start = 1 if calibration_start is None else int(calibration_start)
    cal_end = int(eval_start) - 1 if calibration_end is None else int(calibration_end)
    if not (1 <= cal_start <= cal_end < int(eval_start) <= n_periods):
        raise ValueError(
            "Invalid drift warm-up/calibration range: "
            f"P{cal_start}-P{cal_end}, eval_start=P{eval_start}, available={n_periods}"
        )

    warmup_periods = [period_features[i] for i in range(cal_start - 1, cal_end)]
    initial_ref_len = int(cal_end - cal_start + 1)
    model_ref_len = (
        initial_ref_len
        if model_reference_periods is None
        else max(1, int(model_reference_periods))
    )

    for detector_index, detector_name in enumerate(detectors):
        name = str(detector_name).lower()
        local_kwargs = dict(kwargs)
        local_kwargs["random_state"] = (
            int(local_kwargs.get("random_state", 42)) + detector_index * 100_000
        )
        detector = build_detector(name=name, **local_kwargs)
        is_stream = bool(getattr(detector, "stream_based", False))
        is_maintenance_batch = (
            not is_stream
            and bool(getattr(detector, "maintenance_aligned", False))
            and hasattr(detector, "reset_reference")
            and hasattr(detector, "process_period")
        )

        # State held separately for each detector schedule.
        current_ref_start = cal_start
        current_ref_end = cal_end
        reference_reset_count = 0

        if is_stream:
            detector.initialize_stream(warmup_periods, start_period=cal_start)
        elif is_maintenance_batch:
            x_initial_ref = np.vstack(warmup_periods)
            detector.reset_reference(
                x_initial_ref,
                reference_start_period=cal_start,
                reference_end_period=cal_end,
                reset_seed=0,
            )
            reference_reset_count = 1

        streak = 0
        last_trigger: int | None = None
        period_map: dict[int, DriftDecision] = {}

        for period in range(int(eval_start), int(eval_end) + 1):
            idx = period - 1

            if is_stream:
                result = detector.process_period(period_features[idx], period_id=period)
                ref_start = result.get("reference_start_period")
                ref_end = result.get("reference_end_period")
                if ref_start is None:
                    ref_start = max(1, period - drift_ref_periods)
                if ref_end is None:
                    ref_end = period - 1

            elif is_maintenance_batch:
                # Compare the incoming batch against the training distribution
                # represented by the currently deployed model.  Do NOT roll the
                # reference unless this detector actually triggers maintenance.
                ref_start = int(current_ref_start)
                ref_end = int(current_ref_end)
                result = detector.process_period(
                    period_features[idx],
                    period_id=period,
                )

            else:
                # Backward-compatible fallback for a non-stream detector that
                # does not expose the maintenance-aligned adapter.
                ref_start_idx = max(0, idx - drift_ref_periods)
                ref_end_idx = idx - 1
                if ref_end_idx < ref_start_idx:
                    continue
                ref_periods = [
                    period_features[i] for i in range(ref_start_idx, ref_end_idx + 1)
                ]
                x_ref = np.vstack(ref_periods)
                x_cur = period_features[idx]
                result = detector.detect_pair(
                    x_ref,
                    x_cur,
                    pair_seed=period,
                    X_ref_periods=ref_periods,
                    X_calibration_periods=warmup_periods,
                    calibration_ref_periods=drift_ref_periods,
                )
                ref_start = ref_start_idx + 1
                ref_end = ref_end_idx + 1

            detector_valid = bool(result.get("detector_valid", True))
            raw = bool(
                result.get("raw_drift", result.get("is_drift", False))
            ) and detector_valid
            streak = streak + 1 if raw else 0
            confirmed = raw and streak >= max(1, int(confirm_consecutive))
            has_next = period < int(eval_end)
            trigger = bool(confirmed and has_next)

            if (
                trigger
                and last_trigger is not None
                and period - last_trigger <= int(cooldown_periods)
            ):
                trigger = False
                reason = "cooldown_suppressed"
            elif trigger:
                last_trigger = period
                reason = f"{name}_triggered_update"
            elif not detector_valid:
                reason = "invalid_detector"
            elif not confirmed:
                reason = "no_confirmed_drift"
            else:
                reason = "no_future_evaluation_period"

            score = result.get("score", np.nan)
            threshold = result.get("threshold", np.nan)
            decision = DriftDecision(
                detector=name,
                period=period,
                valid=detector_valid,
                raw_drift=raw,
                confirmed_drift=confirmed,
                update_trigger=trigger,
                score=float(score) if np.isfinite(score) else None,
                threshold=float(threshold) if np.isfinite(threshold) else None,
                reference_start=int(ref_start),
                reference_end=int(ref_end),
                affects_period=period + 1 if trigger else None,
                reason=reason,
            )
            period_map[period] = decision

            row = {
                **decision.__dict__,
                "detector_valid": detector_valid,
                "stream_based": bool(result.get("stream_based", False)),
                "maintenance_aligned_reference": bool(
                    result.get("maintenance_aligned_reference", is_maintenance_batch)
                ),
                "reference_reset_count_before_decision": int(
                    result.get("reference_reset_count", reference_reset_count)
                ),
                "model_reference_periods": (
                    int(model_ref_len) if is_maintenance_batch else None
                ),
                "threshold_method": result.get("threshold_method"),
                "decision_mode": result.get("decision_mode"),
                "score_semantics": result.get("score_semantics"),
                "decision_reason": result.get("decision_reason"),
                "implementation": result.get("implementation"),
                "implementation_version": result.get("implementation_version"),
                "p_value": result.get("p_value"),
                "p_value_threshold": result.get("p_value_threshold"),
                "distance": result.get("distance"),
                "distance_threshold": result.get("distance_threshold"),
                "max_ks_stat": result.get("max_ks_stat"),
                "mean_ks_stat": result.get("mean_ks_stat"),
                "n_features": result.get("n_features"),
                "n_leaves": result.get("n_leaves"),
                "max_samples": result.get("max_samples"),
                "ks_correction": result.get("ks_correction"),
                "ks_alternative": result.get("ks_alternative"),
                "kswin_alpha": result.get("kswin_alpha"),
                "kswin_window_size": result.get("kswin_window_size"),
                "kswin_stat_size": result.get("kswin_stat_size"),
                "kswin_samples_per_period": result.get("kswin_samples_per_period"),
                "kswin_pca_variance_target": result.get("kswin_pca_variance_target"),
                "kswin_pca_components": result.get("kswin_pca_components"),
                "kswin_explained_variance": result.get("kswin_explained_variance"),
                "kswin_projection": result.get("kswin_projection"),
                "lsdd_backend": result.get("lsdd_backend"),
                "lsdd_n_permutations": result.get("lsdd_n_permutations"),
                "lsdd_n_kernel_centers": result.get("lsdd_n_kernel_centers"),
                "kdq_alpha": result.get("kdq_alpha"),
                "kdq_bootstrap_samples": result.get("kdq_bootstrap_samples"),
                "kdq_count_ubound": result.get("kdq_count_ubound"),
                "dependence_statistic": result.get("dependence_statistic"),
                "stream_checks": result.get("stream_checks"),
                "stream_drift_alerts": result.get("stream_drift_alerts"),
                "window_size_before": result.get("window_size_before"),
                "window_size_after": result.get("window_size_after"),
                "randomly_removed": result.get("randomly_removed"),
                "d3_window_size": result.get("d3_window_size"),
                "d3_rho": result.get("d3_rho"),
                "d3_auc_threshold": result.get("d3_auc_threshold"),
                "d3_cv_folds": result.get("d3_cv_folds"),
                "d3_C": result.get("d3_C"),
                "d3_samples_per_period": result.get("d3_samples_per_period"),
                "dawidd_adapter": result.get("dawidd_adapter"),
                "dawidd_max_window_size": result.get("dawidd_max_window_size"),
                "dawidd_min_window_size": result.get("dawidd_min_window_size"),
                "dawidd_p_value_threshold": result.get("dawidd_p_value_threshold"),
                "dawidd_samples_per_period": result.get("dawidd_samples_per_period"),
                "dawidd_permutations": result.get("dawidd_permutations"),
                "warmup_start": cal_start,
                "warmup_end": cal_end,
                "reference_reset_after_period": False,
                "new_reference_start": None,
                "new_reference_end": None,
            }

            # Synchronize the batch detector reference with the retrained model.
            # The update becomes effective only for period t+1, matching runner.py.
            if trigger and is_maintenance_batch:
                new_ref_end = int(period)
                new_ref_start = max(1, new_ref_end - model_ref_len + 1)
                new_ref_periods = [
                    period_features[i]
                    for i in range(new_ref_start - 1, new_ref_end)
                ]
                detector.reset_reference(
                    np.vstack(new_ref_periods),
                    reference_start_period=new_ref_start,
                    reference_end_period=new_ref_end,
                    reset_seed=period,
                )
                current_ref_start = new_ref_start
                current_ref_end = new_ref_end
                reference_reset_count += 1
                row["reference_reset_after_period"] = True
                row["new_reference_start"] = int(new_ref_start)
                row["new_reference_end"] = int(new_ref_end)

            rows.append(row)

        schedules[name] = period_map

    return schedules, rows
