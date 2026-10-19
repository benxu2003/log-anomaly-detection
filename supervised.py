"""
Supervised Log Anomaly Detection — LogMine Clustered Data
=========================================================

Three first-tier methods for LogMine clustered data (parsed_events.csv):

  1. Bag-of-Templates + XGBoost
  2. Template N-gram + XGBoost
  3. Markov Transition Probability anomaly detector

Usage:
  python supervised.py                    # run all methods on parsed_events.csv
  python supervised.py --csv-dir /path/to/processed
"""

from __future__ import annotations

import csv
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── config ──────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
DEFAULT_CSV_DIR = HERE / "data" / "processed"
WINDOW_SIZE = 20          # events per window
ANOMALY_RATIO = 0.15      # fraction of windows to inject anomalies into
ANOMALY_INJECT_WINDOWS = 0.3  # fraction of windows in a day that get injected
MARKOV_THRESHOLD_PERCENTILE = 95.0  # percentile for the Markov anomaly threshold


# ============================================================
# 0. Data Loading & Windowing
# ============================================================

def load_parsed_events(csv_dir: str | Path | None = None) -> list[dict]:
    """Load all _parsed_events.csv files into a flat list of event dicts."""
    csv_dir = Path(csv_dir) if csv_dir else DEFAULT_CSV_DIR
    events = []
    for fpath in sorted(csv_dir.glob("*_parsed_events.csv")):
        dataset = fpath.name.replace("_parsed_events.csv", "")
        with open(fpath, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["source_dataset"] = dataset
                events.append(row)
    return events


def segment_into_windows(
    events: list[dict],
    window_size: int = WINDOW_SIZE,
) -> tuple[list[list[str]], list[str]]:
    """
    Segment events into fixed-size windows. Each window is a list of template_ids.
    Returns (windows, window_ids) where window_id = "{dataset}_{start_idx}".
    """
    windows = []
    window_ids = []

    for dataset in sorted({e["source_dataset"] for e in events}):
        day_events = [e for e in events if e["source_dataset"] == dataset]
        day_events.sort(key=lambda e: int(e.get("line_number", 0)))

        for i in range(0, len(day_events) - window_size + 1, window_size):
            chunk = day_events[i : i + window_size]
            template_ids = [e["template_id"] for e in chunk]
            windows.append(template_ids)
            window_ids.append(f"{dataset}_{i:06d}")

    return windows, window_ids


def inject_anomalies(
    windows: list[list[str]],
    anomaly_ratio: float = ANOMALY_RATIO,
    seed: int = 42,
) -> tuple[list[list[str]], np.ndarray]:
    """
    Create a labeled dataset by injecting synthetic anomalies into some windows.

    Anomaly types:
      - shuffle:     randomly reorder 30% of templates in the window
      - repeat:      repeat one template 5 times in a row
      - insert:      insert a rare/unknown template
      - drop:        remove the middle 20% of templates

    Returns (augmented_windows, labels) where label=1 means anomaly.
    """
    rng = np.random.RandomState(seed)
    labels = np.zeros(len(windows), dtype=int)
    augmented = [list(w) for w in windows]

    n_anomaly = max(1, int(len(windows) * anomaly_ratio))
    anomaly_indices = rng.choice(len(windows), size=n_anomaly, replace=False)

    all_templates = sorted({t for w in windows for t in w})

    for idx in anomaly_indices:
        labels[idx] = 1
        w = augmented[idx]
        anomaly_type = rng.choice(["shuffle", "repeat", "insert", "drop"])

        if anomaly_type == "shuffle":
            n = max(2, len(w) // 3)
            start = rng.randint(0, len(w) - n)
            segment = w[start : start + n]
            rng.shuffle(segment)
            w[start : start + n] = segment

        elif anomaly_type == "repeat":
            t = rng.choice(w)
            pos = rng.randint(0, max(1, len(w) - 5))
            for j in range(5):
                if pos + j < len(w):
                    w[pos + j] = t

        elif anomaly_type == "insert":
            fake = f"ANOMALY_INJECTED_{rng.randint(0, 1000)}"
            pos = rng.randint(0, len(w))
            w.insert(pos, fake)
            w.pop()  # keep same length

        elif anomaly_type == "drop":
            n = max(1, len(w) // 5)
            start = len(w) // 2 - n // 2
            end = start + n
            del w[start:end]
            while len(w) < len(windows[0]):
                w.append(w[-1] if w else all_templates[0])

    return augmented, labels


# ============================================================
# 1. Bag-of-Templates + XGBoost
# ============================================================

def build_bag_of_templates(
    windows: list[list[str]],
) -> tuple[np.ndarray, DictVectorizer]:
    """Count template occurrences per window → sparse feature matrix."""
    counters = []
    for w in windows:
        counters.append(dict(Counter(w)))

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(counters)
    return X, vec


def build_template_features(
    windows: list[list[str]],
) -> np.ndarray:
    """Handcrafted features per window: length, unique templates, entropy."""
    feats = np.zeros((len(windows), 3), dtype=np.float32)
    for i, w in enumerate(windows):
        counter = Counter(w)
        total = len(w)
        unique = len(counter)
        entropy = 0.0
        for c in counter.values():
            p = c / total
            if p > 0:
                entropy -= p * math.log(p)
        max_entropy = math.log(unique) if unique > 1 else 1.0
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        feats[i, 0] = total
        feats[i, 1] = unique
        feats[i, 2] = norm_entropy
    return feats


def train_xgboost_bow(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Bag-of-Templates + XGBoost. Returns (y_pred, y_prob)."""
    try:
        import xgboost as xgb
    except ImportError:
        print("[XGBoost] xgboost not installed. pip install xgboost")
        return None, None

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    _print_eval("Bag-of-Templates + XGBoost", y_test, y_pred, y_prob)
    return y_pred, y_prob


# ============================================================
# 2. Template N-gram + XGBoost
# ============================================================

def build_ngram_features(
    windows: list[list[str]],
    n: int = 2,
) -> tuple[np.ndarray, DictVectorizer]:
    """Count n-gram occurrences per window → feature matrix."""
    ngram_counters = []
    for w in windows:
        ngrams = Counter()
        for i in range(len(w) - n + 1):
            ngrams[tuple(w[i : i + n])] += 1
        ngram_counters.append(dict(ngrams))

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(ngram_counters)
    return X, vec


def train_xgboost_ngram(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Template N-gram + XGBoost. Returns (y_pred, y_prob)."""
    try:
        import xgboost as xgb
    except ImportError:
        print("[XGBoost] xgboost not installed. pip install xgboost")
        return None, None

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    _print_eval("Template Bigram + XGBoost", y_test, y_pred, y_prob)
    return y_pred, y_prob


# ============================================================
# 3. Markov Transition Probability
# ============================================================

class MarkovAnomalyDetector:
    """
    Learn first-order Markov transition probabilities from normal windows.
    Score a window by its average negative log transition probability.
    Lower probability → higher anomaly score.
    """

    def __init__(self, smoothing: float = 1e-6):
        self.smoothing = smoothing
        self.transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.state_counts: dict[str, int] = defaultdict(int)
        self._threshold: float | None = None

    def fit(self, windows: list[list[str]]):
        """Learn transition counts from normal windows."""
        for w in windows:
            for i in range(len(w) - 1):
                a, b = w[i], w[i + 1]
                self.transitions[a][b] += 1
                self.state_counts[a] += 1

    def score(self, window: list[str]) -> float:
        """Anomaly score: average negative log-likelihood of transitions."""
        if len(window) < 2:
            return 0.0
        log_prob = 0.0
        n_transitions = 0
        vocab_size = len(self.state_counts)

        for i in range(len(window) - 1):
            a, b = window[i], window[i + 1]
            count = self.transitions[a].get(b, 0) + self.smoothing
            total = self.state_counts.get(a, 0) + self.smoothing * max(vocab_size, 1)
            prob = count / total if total > 0 else self.smoothing
            log_prob += math.log(max(prob, 1e-12))
            n_transitions += 1

        return -log_prob / max(n_transitions, 1)

    def predict(self, windows: list[list[str]], threshold: float | None = None) -> np.ndarray:
        """Predict labels: 1 if score >= threshold, else 0."""
        if threshold is None:
            threshold = self._threshold or 0.0
        scores = np.array([self.score(w) for w in windows])
        return (scores >= threshold).astype(int)

    def predict_proba(self, windows: list[list[str]]) -> np.ndarray:
        """Return anomaly scores as pseudo-probabilities (sigmoid-normalized)."""
        scores = np.array([self.score(w) for w in windows])
        if scores.max() == scores.min():
            return np.full_like(scores, 0.5)
        normalized = (scores - scores.min()) / (scores.max() - scores.min())
        return normalized

    def fit_threshold(self, windows: list[list[str]], percentile: float = 95.0):
        """Set threshold at given percentile of scores on normal data."""
        scores = [self.score(w) for w in windows]
        self._threshold = np.percentile(scores, percentile)


def train_markov(
    train_windows: list[list[str]],
    train_labels: np.ndarray,
    test_windows: list[list[str]],
    test_labels: np.ndarray,
    threshold_percentile: float = MARKOV_THRESHOLD_PERCENTILE,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Markov transition probability anomaly detector. Returns (y_pred, y_prob)."""
    normal_windows = [w for w, l in zip(train_windows, train_labels) if l == 0]

    if not normal_windows:
        print("[Markov] No normal windows in training set, skipping.")
        return None, None

    detector = MarkovAnomalyDetector(smoothing=1e-6)
    detector.fit(normal_windows)
    detector.fit_threshold(normal_windows, percentile=threshold_percentile)

    y_score = detector.predict_proba(test_windows)
    y_pred = detector.predict(test_windows)

    _print_eval("Markov Transition Probability", test_labels, y_pred, y_score)
    return y_pred, y_score


# ============================================================
# 4. Ensemble (simple average of all three)
# ============================================================

def ensemble_predict(proba_list: list[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Average probabilities and threshold."""
    valid = [p for p in proba_list if p is not None]
    if not valid:
        return None
    avg = np.mean(valid, axis=0)
    return (avg >= threshold).astype(int)


# ============================================================
# Helpers
# ============================================================

def _print_eval(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    if y_pred is None:
        return
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  Precision:     {prec:.4f}")
    print(f"  Recall:        {rec:.4f}")
    print(f"  F1-Score:      {f1:.4f}")
    print(f"  ROC-AUC:       {auc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"], zero_division=0))


def _print_feature_importance(model, feature_names: list[str] | None = None, top_n: int = 15):
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        names = feature_names or model.get_booster().feature_names
        print(f"\n  Top {top_n} Features:")
        for rank, idx in enumerate(indices, 1):
            name = names[idx] if idx < len(names) else f"f{idx}"
            print(f"    {rank:2d}. {name:<50s} {importances[idx]:.4f}")
    except Exception:
        pass


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Supervised log anomaly detection on LogMine clustered data"
    )
    parser.add_argument(
        "--csv-dir", default=str(DEFAULT_CSV_DIR),
        help="Directory containing *_parsed_events.csv files"
    )
    parser.add_argument(
        "--window-size", type=int, default=WINDOW_SIZE,
        help="Events per window"
    )
    parser.add_argument(
        "--anomaly-ratio", type=float, default=ANOMALY_RATIO,
        help="Fraction of windows to inject anomalies into"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data for test set"
    )
    parser.add_argument(
        "--startup-windows", type=int, default=3,
        help="Skip first N windows per day (startup/initialization noise)"
    )
    parser.add_argument(
        "--markov-threshold", type=float, default=MARKOV_THRESHOLD_PERCENTILE,
        help="Percentile threshold for Markov anomaly flagging (default: 95)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Supervised Log Anomaly Detection — LogMine Clustered Data")
    print("=" * 60)

    # ── Load data ──
    print(f"\n[1/5] Loading parsed events from: {args.csv_dir}")
    events = load_parsed_events(args.csv_dir)
    print(f"  Total events: {len(events)}")

    unique_templates = sorted({e["template_id"] for e in events})
    unique_datasets = sorted({e["source_dataset"] for e in events})
    print(f"  Unique templates: {len(unique_templates)}")
    print(f"  Datasets (days):  {len(unique_datasets)}")
    print(f"  Datasets: {', '.join(unique_datasets[:5])}...")

    # ── Windowing ──
    print(f"\n[2/5] Segmenting into windows (size={args.window_size})...")
    windows, window_ids = segment_into_windows(events, args.window_size)
    print(f"  Total windows: {len(windows)}")
    print(f"  Example window[0] templates: {windows[0][:5]}...")

    # ── Inject anomalies ──
    print(f"\n[3/5] Injecting anomalies (ratio={args.anomaly_ratio})...")
    augmented, labels = inject_anomalies(windows, args.anomaly_ratio)
    print(f"  Normal windows:  {(labels == 0).sum()}")
    print(f"  Anomaly windows: {(labels == 1).sum()}")
    print(f"  Anomaly ratio:   {labels.mean():.2%}")

    # ── Train/Test split ──
    print(f"\n[4/5] Train/Test split (test_size={args.test_size})...")
    indices = np.arange(len(augmented))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=42, stratify=labels
    )

    train_windows = [augmented[i] for i in train_idx]
    test_windows = [augmented[i] for i in test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    print(f"  Train: {len(train_windows)}, Test: {len(test_windows)}")

    # ── Train & Evaluate ──
    print(f"\n[5/5] Training models...")

    # Method 1: Bag-of-Templates + XGBoost
    print("\n--- Method 1: Bag-of-Templates + XGBoost ---")
    X_bow, bow_vec = build_bag_of_templates(train_windows + test_windows)
    X_bow_train = X_bow[:len(train_windows)]
    X_bow_test = X_bow[len(train_windows):]
    y_pred_bow, y_prob_bow = train_xgboost_bow(X_bow_train, y_train, X_bow_test, y_test)

    # Method 2: Template N-gram + XGBoost
    print("\n--- Method 2: Template Bigram + XGBoost ---")
    X_ngram, ngram_vec = build_ngram_features(train_windows + test_windows, n=2)
    X_ngram_train = X_ngram[:len(train_windows)]
    X_ngram_test = X_ngram[len(train_windows):]
    y_pred_ngram, y_prob_ngram = train_xgboost_ngram(X_ngram_train, y_train, X_ngram_test, y_test)

    # Method 3: Markov Transition
    print("\n--- Method 3: Markov Transition Probability ---")
    y_pred_markov, y_prob_markov = train_markov(
        train_windows, y_train, test_windows, y_test,
        threshold_percentile=args.markov_threshold,
    )

    # ── Ensemble ──
    print("\n--- Ensemble (Average of all 3 methods) ---")
    proba_list = [y_prob_bow, y_prob_ngram, y_prob_markov]
    y_prob_ensemble = np.mean([p for p in proba_list if p is not None], axis=0)
    y_pred_ensemble = (y_prob_ensemble >= 0.5).astype(int)
    _print_eval("Ensemble (Avg Probability)", y_test, y_pred_ensemble, y_prob_ensemble)

    # ── 6. Anomaly Report: score ALL windows, output top anomalies ──
    print(f"\n[6/6] Generating anomaly report on all {len(windows)} windows...")

    original_windows = windows
    original_ids = window_ids
    normal_windows = [w for w, l in zip(original_windows, labels) if l == 0]

    # ── Markov: train on normal windows, score all ──
    detector = MarkovAnomalyDetector(smoothing=1e-6)
    detector.fit(normal_windows)
    detector.fit_threshold(normal_windows, percentile=args.markov_threshold)
    markov_scores = np.array([detector.score(w) for w in original_windows])
    markov_threshold = detector._threshold
    print(f"  Markov threshold (p{args.markov_threshold}): {markov_threshold:.4f}")

    # ── XGBoost with Bigram features (NOT BoW — Bigrams capture order) ──
    X_ngram_all, _ = build_ngram_features(original_windows, n=2)
    try:
        import xgboost as xgb
        scale_pos_weight = (labels == 0).sum() / max((labels == 1).sum(), 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        xgb_model.fit(X_ngram_all, labels)
        xgb_scores = xgb_model.predict_proba(X_ngram_all)[:, 1]
    except ImportError:
        xgb_scores = np.zeros(len(original_windows))

    # ── Combined: Markov (normalized) + Bigram XGBoost ──
    markov_norm = (markov_scores - markov_scores.min()) / (markov_scores.max() - markov_scores.min() + 1e-10)
    combined_scores = (markov_norm + xgb_scores) / 2.0

    # ── Mark startup windows (first N per day) ──
    startup_set: set[str] = set()
    day_windows: dict[str, list[tuple[str, int]]] = {}
    for wid in original_ids:
        day = wid.rsplit("_", 1)[0]  # e.g. "iawork_2026-01-23"
        day_windows.setdefault(day, []).append(wid)
    for day, wids in day_windows.items():
        sorted_wids = sorted(wids, key=lambda x: int(x.rsplit("_", 1)[-1]))
        for wid in sorted_wids[:args.startup_windows]:
            startup_set.add(wid)

    # ── Rank windows by combined score ──
    ranked = sorted(
        zip(original_ids, original_windows, markov_scores, markov_norm, xgb_scores, combined_scores),
        key=lambda x: -x[5],
    )

    # ── Output top anomalies ──
    top_n = min(30, len(ranked))
    print(f"\n  Top {top_n} Most Anomalous Windows:")
    header = f"  {'Rank':<5} {'Window ID':<40} {'Markov':>8} {'XGB(Ngram)':>11} {'Combined':>8} {'N':>3} {'Markov':>8} {'Comb':>6}  {'Sample Templates'}"
    print(header)
    print(f"  {'-'*5} {'-'*40} {'-'*8} {'-'*11} {'-'*8} {'-'*3} {'-'*8} {'-'*6}  {'-'*50}")
    for rank, (wid, w, ms, mn, xs, cs) in enumerate(ranked[:top_n], 1):
        unique_templates = list(dict.fromkeys(w))
        sample = " -> ".join(unique_templates[:5])
        if len(unique_templates) > 5:
            sample += " -> ..."
        is_startup = "START" if wid in startup_set else ""
        m_flag = "ANOM" if ms > markov_threshold else ""
        c_flag = "ANOM" if cs > 0.5 else ""
        print(f"  {rank:<5} {wid:<40} {ms:8.4f} {xs:11.4f} {cs:8.4f} {len(unique_templates):>3} {m_flag:>8} {c_flag:>6}  {sample} {is_startup}")

    # ── Save full report to CSV ──
    report_path = Path(args.csv_dir).parent / "reports" / "anomaly_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "window_id", "n_events", "n_unique_templates",
                         "markov_score", "xgb_ngram_score", "combined_score",
                         "markov_flagged", "combined_flagged", "is_startup",
                         "template_sequence"])
        for rank, (wid, w, ms, mn, xs, cs) in enumerate(ranked, 1):
            unique = list(dict.fromkeys(w))
            writer.writerow([
                rank, wid, len(w), len(unique),
                round(ms, 4), round(xs, 4), round(cs, 4),
                1 if ms > markov_threshold else 0,
                1 if cs > 0.5 else 0,
                1 if wid in startup_set else 0,
                " -> ".join(unique),
            ])

    # ── Summary stats ──
    markov_flagged = sum(1 for ms in markov_scores if ms > markov_threshold)
    combined_flagged = sum(1 for cs in combined_scores if cs > 0.5)
    markov_flagged_no_startup = sum(
        1 for wid, ms in zip(original_ids, markov_scores)
        if ms > markov_threshold and wid not in startup_set
    )
    combined_flagged_no_startup = sum(
        1 for wid, cs in zip(original_ids, combined_scores)
        if cs > 0.5 and wid not in startup_set
    )

    print(f"\n  Full anomaly report saved to: {report_path}")
    print(f"  Total windows:          {len(ranked)}")
    print(f"  Startup windows (skipped per day): {args.startup_windows}")
    print(f"  Markov flagged (p{args.markov_threshold}):     {markov_flagged} ({markov_flagged/len(ranked):.2%})")
    print(f"  Markov flagged (excl startup): {markov_flagged_no_startup} ({markov_flagged_no_startup/len(ranked):.2%})")
    print(f"  Combined flagged (>0.5): {combined_flagged} ({combined_flagged/len(ranked):.2%})")
    print(f"  Combined flagged (excl startup): {combined_flagged_no_startup} ({combined_flagged_no_startup/len(ranked):.2%})")

if __name__ == "__main__":
    main()