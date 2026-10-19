#!/usr/bin/env python3
"""
Synthetic log generator for the log-anomaly-detection demo.

Emits realistic multi-day log files that the pipeline ingests directly:
  data/raw/<name>.log                  -- "YYYY-MM-DD HH:MM:SS LEVEL UUID message"
  data/clustered/<name>_clustered.log  -- one line per template, LogMine format

Each simulated session (one UUID = one trace) follows a regular workflow
(INIT -> CONNECT -> TRANSFER -> WRITE -> IDLE). A fraction of sessions are
anomalous (shuffled, repeated, or dropped steps, or novel error templates),
which the Markov scorer flags via low-likelihood transitions.

Usage:
  python scripts/generate_synthetic_logs.py --days 5 --sessions-per-day 40
  python scripts/generate_synthetic_logs.py --out data --seed 7
"""

from __future__ import annotations

import argparse
import random
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

PLACEHOLDERS = {
    "sid": lambda: f"sess-{random.randint(1000, 9999)}",
    "host": lambda: f"10.0.{random.randint(0, 9)}.{random.randint(2, 254)}",
    "duration": lambda: str(random.randint(2, 900)),
    "db": lambda: random.choice(["orders", "users", "inventory", "metrics"]),
    "bytes": lambda: str(random.randint(1024, 8388608)),
    "key": lambda: f"cache:{random.randint(1, 5000)}",
    "file": lambda: random.choice(["snapshot.dat", "wal_0042.log", "index.bin"]),
    "pct": lambda: str(random.randint(20, 85)),
    "user": lambda: f"svc_{random.randint(1, 20)}",
    "device": lambda: random.choice(["/dev/sda1", "/dev/nvme0n1"]),
}

NORMAL_TEMPLATES = [
    ("INFO", "Session {sid} initialized"),
    ("INFO", "Connection established to {host}"),
    ("DEBUG", "Query executed on {db} in {duration}ms"),
    ("INFO", "Data transfer started: {bytes} bytes to {host}"),
    ("INFO", "Data transfer completed in {duration}ms"),
    ("DEBUG", "Cache hit for key {key}"),
    ("INFO", "File {file} written successfully"),
    ("DEBUG", "Heartbeat received from {host}"),
    ("INFO", "Health check passed"),
    ("INFO", "Configuration reloaded"),
    ("INFO", "Scheduled backup completed"),
    ("DEBUG", "Memory usage: {pct}%"),
    ("INFO", "User {user} authenticated"),
    ("INFO", "Connection closed for {host}"),
    ("INFO", "Session {sid} closed normally"),
]

ANOMALY_TEMPLATES = [
    ("ERROR", "Connection timeout to {host}"),
    ("ERROR", "Disk I/O failure on {device}"),
    ("FATAL", "Out of memory: {pct}% used"),
    ("ERROR", "Authentication failed for user {user}"),
]

WORKFLOW = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14]


def fill(template: str) -> str:
    for key, gen in PLACEHOLDERS.items():
        token = "{" + key + "}"
        if token in template:
            template = template.replace(token, gen())
    return template


def to_cluster_template(template: str) -> str:
    return " ".join("---" if tok.startswith("{") and tok.endswith("}")
                      else tok for tok in template.split())


def simulate_session(rng: random.Random, anomalous: bool) -> list[tuple[str, str]]:
    seq = [NORMAL_TEMPLATES[0]]
    for _ in range(rng.randint(2, 4)):
        for t in rng.sample(WORKFLOW[1:-2], rng.randint(3, 6)):
            seq.append(NORMAL_TEMPLATES[t])
    seq += [NORMAL_TEMPLATES[13], NORMAL_TEMPLATES[14]]

    if anomalous and len(seq) > 8:
        kind = rng.choice(["shuffle", "repeat", "drop", "novel"])
        if kind == "shuffle":
            i = rng.randint(1, len(seq) - 5)
            seg = seq[i:i + 5]
            rng.shuffle(seg)
            seq[i:i + 5] = seg
        elif kind == "repeat":
            i = rng.randint(1, len(seq) - 3)
            seq[i:i] = [seq[i]] * 4
        elif kind == "drop":
            i = rng.randint(2, len(seq) - 5)
            del seq[i:i + 4]
        else:
            seq.insert(rng.randint(1, len(seq) - 1), rng.choice(ANOMALY_TEMPLATES))
    return seq


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic demo logs.")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--sessions-per-day", type=int, default=40)
    parser.add_argument("--anomaly-ratio", type=float, default=0.15)
    parser.add_argument("--out", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default="2026-01-22")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    raw_dir = Path(args.out) / "raw"
    clustered_dir = Path(args.out) / "clustered"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clustered_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(args.start, "%Y-%m-%d")
    total_events = 0

    for d in range(args.days):
        day = start + timedelta(days=d)
        day_str = day.strftime("%Y-%m-%d")
        name = f"synth_{day_str}"
        events: list[tuple[datetime, str, str, tuple[str, str], str]] = []
        counts: Counter[tuple[str, str]] = Counter()

        for _ in range(args.sessions_per_day):
            trace_id = str(uuid.uuid4())
            seq = simulate_session(rng, rng.random() < args.anomaly_ratio)
            t = day + timedelta(seconds=rng.randint(0, 80000))
            for level, template in seq:
                events.append((t, level, trace_id, (level, template), fill(template)))
                counts[(level, template)] += 1
                t += timedelta(seconds=rng.randint(1, 6))

        events.sort(key=lambda e: e[0])
        with open(raw_dir / f"{name}.log", "w", encoding="utf-8") as f:
            for t, level, trace_id, _tpl, msg in events:
                f.write(f"{t:%Y-%m-%d %H:%M:%S} {level} {trace_id} {msg}\n")
        total_events += len(events)

        with open(clustered_dir / f"{name}_clustered.log", "w", encoding="utf-8") as f:
            for (level, template), c in sorted(counts.items(), key=lambda kv: -kv[1]):
                f.write(f"{c} {day_str} --- {level} --- {to_cluster_template(template)}\n")

    n_sessions = args.days * args.sessions_per_day
    print(f"Wrote {args.days} days, {total_events} events, {n_sessions} sessions "
          f"({args.anomaly_ratio:.0%} anomalous) -> {args.out}/")
    print(f"Next: python pipeline.py convert   # or: python supervised.py --csv-dir {args.out}/processed")


if __name__ == "__main__":
    main()
