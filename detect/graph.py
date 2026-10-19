"""
Graph builder: converts parsed_events.csv + template_state_map.csv into
per-trace state transition graphs without any LLM calls.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def build_graphs(
    events_csv: str | Path,
    template_state_map_csv: str | Path,
) -> list[dict]:
    template_to_state: dict[str, str] = {}
    with open(template_state_map_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            template_to_state[row["template_id"]] = row["state_id"]

    events_by_trace: dict[str, list[dict]] = defaultdict(list)
    with open(events_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            events_by_trace[row["trace_id"]].append(row)

    graphs = []
    for trace_id, events in sorted(events_by_trace.items()):
        ordered = sorted(events, key=lambda e: int(e.get("line_number") or 0))
        n = len(ordered)

        nodes = []
        for i, event in enumerate(ordered):
            tid = event.get("template_id", "")
            nodes.append({
                "id": event["event_id"],
                "features": {
                    "template_id": tid,
                    "state_id": template_to_state.get(tid, ""),
                    "component": event.get("component", ""),
                    "level": event.get("level", ""),
                    "position": i / max(n - 1, 1),
                },
            })

        edges = []
        for i in range(1, n):
            edges.append({
                "src": ordered[i - 1]["event_id"],
                "dst": ordered[i]["event_id"],
                "type": "temporal",
            })

        graphs.append({
            "trace_id": trace_id,
            "split": "real",
            "label": None,
            "nodes": nodes,
            "edges": edges,
            "summary_features": {},
        })

    return graphs


def write_graphs_jsonl(graphs: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for g in graphs:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")