"""
Score evaluation: computes thresholds and metrics from scored traces.
Handles both labeled and unlabeled data.
"""

from __future__ import annotations


def evaluate_scores(scored: list[dict]) -> dict:
    labeled = [row for row in scored if row.get("label") in {0, 1}]

    if not labeled:
        scores = [row["score"] for row in scored]
        threshold = _percentile(scores, 0.90) if scores else 0.0
        return {
            "mode": "unlabeled",
            "threshold_p90": threshold,
            "n_flagged": sum(1 for s in scores if s >= threshold),
            "message": "No binary labels found; reported p90 threshold and flagged count only.",
        }

    labels = [int(row["label"]) for row in labeled]
    scores = [float(row["score"]) for row in labeled]
    threshold = _percentile(scores, 0.90)
    return {
        "mode": "labeled",
        "average_precision": _average_precision(labels, scores),
        "fpr_at_90_recall": _fpr_at_recall(labels, scores, target_recall=0.90),
        "threshold_p90": threshold,
        **_precision_recall_f1(labels, scores, threshold=threshold),
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def _average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / positives


def _fpr_at_recall(labels: list[int], scores: list[float], *,
                   target_recall: float = 0.95) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = 0
    fp = 0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        if tp / positives >= target_recall:
            return fp / negatives
    return None


def _precision_recall_f1(labels: list[int], scores: list[float], *,
                         threshold: float) -> dict[str, float]:
    tp = fp = fn = 0
    for label, score in zip(labels, scores):
        pred = 1 if score >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}