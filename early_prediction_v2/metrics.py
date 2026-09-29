from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def finite_scores(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if np.all(np.isfinite(s)):
        return s
    finite = s[np.isfinite(s)]
    hi = float(np.max(finite)) if finite.size else 0.0
    lo = float(np.min(finite)) if finite.size else 0.0
    return np.nan_to_num(s, nan=hi, posinf=hi, neginf=lo)


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    s = finite_scores(scores)
    if len(y) == 0 or len(y) != len(s) or np.unique(y).size < 2:
        return None
    return float(roc_auc_score(y, s))


def safe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    s = finite_scores(scores)
    if len(y) == 0 or len(y) != len(s) or int(np.sum(y == 1)) == 0:
        return None
    return float(average_precision_score(y, s))


def top_budget_metrics(y_true: np.ndarray, scores: np.ndarray, ratio: float) -> Dict[str, Any]:
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("top ratio must be in (0, 1]")
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    s = finite_scores(scores)
    if len(y) != len(s):
        raise ValueError("y_true and scores length mismatch")
    n = int(len(y))
    positives = int(np.sum(y == 1))
    if n == 0:
        return {"ratio": ratio, "k": 0, "captured": 0, "recall": None, "precision": None, "lift": None}
    k = max(1, min(n, int(np.ceil(float(ratio) * n))))
    order = np.argsort(-s, kind="mergesort")
    captured = int(np.sum(y[order[:k]] == 1))
    precision = float(captured / k)
    recall = None if positives == 0 else float(captured / positives)
    prevalence = float(positives / n)
    lift = None if positives == 0 else float(precision / prevalence)
    return {
        "ratio": float(ratio), "k": k, "captured": captured,
        "recall": recall, "precision": precision, "lift": lift,
    }


def period_metrics(y_true: np.ndarray, scores: np.ndarray, top_ratios: Sequence[float]) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    s = finite_scores(scores)
    out: Dict[str, Any] = {
        "samples": int(len(y)),
        "positives": int(np.sum(y == 1)),
        "positive_rate": None if len(y) == 0 else float(np.mean(y == 1)),
        "roc_auc": safe_roc_auc(y, s),
        "pr_auc": safe_pr_auc(y, s),
        "score_min": None if len(s) == 0 else float(np.min(s)),
        "score_mean": None if len(s) == 0 else float(np.mean(s)),
        "score_std": None if len(s) == 0 else float(np.std(s, ddof=0)),
        "score_max": None if len(s) == 0 else float(np.max(s)),
    }
    for ratio in top_ratios:
        item = top_budget_metrics(y, s, ratio)
        token = ratio_token(ratio)
        out[f"top_{token}_ratio"] = float(ratio)
        out[f"top_{token}_k"] = item["k"]
        out[f"top_{token}_captured"] = item["captured"]
        out[f"top_{token}_recall"] = item["recall"]
        out[f"top_{token}_precision"] = item["precision"]
        out[f"top_{token}_lift"] = item["lift"]
    return out


def ratio_token(ratio: float) -> str:
    return f"{float(ratio):.12g}".replace("-", "m").replace("+", "").replace(".", "p")


def _finite(values: Iterable[Any]) -> np.ndarray:
    parsed = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x):
            parsed.append(x)
    return np.asarray(parsed, dtype=np.float64)


def summarize_periods(rows: Sequence[Dict[str, Any]], top_ratios: Sequence[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"num_eval_periods": int(len(rows))}
    for metric in ("roc_auc", "pr_auc"):
        values = _finite(row.get(metric) for row in rows)
        out[f"mean_period_{metric}"] = None if len(values) == 0 else float(np.mean(values))
        out[f"std_period_{metric}"] = None if len(values) == 0 else float(np.std(values, ddof=0))
        out[f"min_period_{metric}"] = None if len(values) == 0 else float(np.min(values))
        out[f"max_period_{metric}"] = None if len(values) == 0 else float(np.max(values))
    total_positives = int(sum(int(row.get("positives", 0) or 0) for row in rows))
    for ratio in top_ratios:
        token = ratio_token(ratio)
        for metric in ("recall", "precision", "lift"):
            values = _finite(row.get(f"top_{token}_{metric}") for row in rows)
            out[f"mean_period_top_{token}_{metric}"] = None if len(values) == 0 else float(np.mean(values))
        total_k = int(sum(int(row.get(f"top_{token}_k", 0) or 0) for row in rows))
        captured = int(sum(int(row.get(f"top_{token}_captured", 0) or 0) for row in rows))
        out[f"top_{token}_ratio"] = float(ratio)
        out[f"total_top_{token}_budget_k"] = total_k
        out[f"total_top_{token}_captured"] = captured
        out[f"pooled_top_{token}_precision"] = None if total_k == 0 else float(captured / total_k)
        out[f"pooled_top_{token}_recall"] = None if total_positives == 0 else float(captured / total_positives)
    return out
