"""
Anomaly scorer: zero-shot graph structure scoring with optional baseline layers.

Four scoring dimensions (Layer 1, always active):
  1. Entropy anomaly    — state distribution concentration
  2. Complexity anomaly — edge/node ratio (deadlock signal at ratio=1.0)
  3. Density anomaly    — event count per state
  4. Dominance anomaly  — single state occupying >50% of events

Optional layers:
  Layer 2: Missing critical states (requires required_states)
  Layer 3: Novel states + illegal transitions (requires baseline)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from detect.graph import build_graphs, write_graphs_jsonl


def score_trace(graph: dict, *, baseline: dict | None = None,
                required_states: list[str] | None = None,
                legal_transitions: set[tuple[str, str]] | None = None) -> float:
    nodes = graph.get("nodes", [])
    if not nodes:
        return 0.0

    state_counter: Counter[str] = Counter()
    edges: set[tuple[str, str]] = set()
    prev_state: str | None = None

    for node in nodes:
        sid = node["features"].get("state_id", "")
        if sid:
            state_counter[sid] += 1
        if prev_state and sid:
            edges.add((prev_state, sid))
        prev_state = sid if sid else prev_state

    if not state_counter:
        return 0.0

    counts = list(state_counter.values())
    total = sum(counts)
    n_states = len(counts)

    entropy = 0.0
    for c in counts:
        p = c / total
        if p > 0:
            entropy -= p * math.log(p)
    max_entropy = math.log(n_states) if n_states > 1 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    edge_node_ratio = len(edges) / max(n_states, 1)

    entropy_anomaly = (1.0 - norm_entropy) * math.log(total + 1) if n_states > 1 else 0.0
    complexity_anomaly = (1.0 / max(edge_node_ratio, 0.01)) * math.log(total + 1) * 2.0
    density_anomaly = math.log(total / max(n_states, 1) + 1) * (1.0 - norm_entropy) if n_states > 1 else 0.0
    top1_ratio = max(counts) / total
    dominance_anomaly = top1_ratio * math.log(total + 1) if top1_ratio > 0.5 and n_states > 1 else 0.0

    missing_penalty = 0.0
    if required_states:
        present = set(state_counter.keys())
        missing = set(required_states) - present
        missing_penalty = len(missing) * math.log(total + 1) * 1.5

    illegal_penalty = 0.0
    if legal_transitions:
        illegal = edges - legal_transitions
        illegal_penalty = len(illegal) * math.log(total + 1) * 0.8

    novel_penalty = 0.0
    if baseline:
        known = set(baseline.get("known_states", []))
        novel = set(state_counter.keys()) - known
        if novel:
            novel_penalty = len(novel) * math.log(total + 1) * 2.0

    return (entropy_anomaly + complexity_anomaly + density_anomaly
            + dominance_anomaly + missing_penalty + illegal_penalty + novel_penalty)


def score_traces(graphs: list[dict], *, baseline: dict | None = None,
                 required_states: list[str] | None = None,
                 legal_transitions: set[tuple[str, str]] | None = None) -> list[dict]:
    results = []
    for g in graphs:
        nodes = g.get("nodes", [])
        if not nodes:
            continue

        state_counter: Counter[str] = Counter()
        edges: set[tuple[str, str]] = set()
        prev_state: str | None = None

        for node in nodes:
            sid = node["features"].get("state_id", "")
            if sid:
                state_counter[sid] += 1
            if prev_state and sid:
                edges.add((prev_state, sid))
            prev_state = sid if sid else prev_state

        if not state_counter:
            continue

        counts = list(state_counter.values())
        total = sum(counts)
        n_states = len(counts)
        n_edges = len(edges)

        entropy = 0.0
        for c in counts:
            p = c / total
            if p > 0:
                entropy -= p * math.log(p)
        max_entropy = math.log(n_states) if n_states > 1 else 1.0
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        edge_node_ratio = n_edges / max(n_states, 1)
        top1_ratio = max(counts) / total

        e_anom = (1.0 - norm_entropy) * math.log(total + 1) if n_states > 1 else 0.0
        c_anom = (1.0 / max(edge_node_ratio, 0.01)) * math.log(total + 1) * 2.0
        d_anom = math.log(total / max(n_states, 1) + 1) * (1.0 - norm_entropy) if n_states > 1 else 0.0
        dom_anom = top1_ratio * math.log(total + 1) if top1_ratio > 0.5 and n_states > 1 else 0.0

        breakdown = {
            "entropy_anomaly": round(e_anom, 4),
            "complexity_anomaly": round(c_anom, 4),
            "density_anomaly": round(d_anom, 4),
            "dominance_anomaly": round(dom_anom, 4),
        }

        missing_penalty = 0.0
        missing_list: list[str] = []
        if required_states:
            present = set(state_counter.keys())
            missing_list = sorted(set(required_states) - present)
            missing_penalty = len(missing_list) * math.log(total + 1) * 1.5
            if missing_penalty > 0:
                breakdown["missing_penalty"] = round(missing_penalty, 4)

        illegal_penalty = 0.0
        illegal_count = 0
        if legal_transitions:
            illegal = edges - legal_transitions
            illegal_count = len(illegal)
            illegal_penalty = illegal_count * math.log(total + 1) * 0.8
            if illegal_penalty > 0:
                breakdown["illegal_penalty"] = round(illegal_penalty, 4)

        novel_penalty = 0.0
        novel_states: list[str] = []
        if baseline:
            known = set(baseline.get("known_states", []))
            novel_states = sorted(set(state_counter.keys()) - known)
            if novel_states:
                novel_penalty = len(novel_states) * math.log(total + 1) * 2.0
                breakdown["novel_penalty"] = round(novel_penalty, 4)

        result = {
            "trace_id": g["trace_id"],
            "score": round(sum(breakdown.values()), 4),
            "n_events": total,
            "n_states": n_states,
            "n_edges": n_edges,
            "edge_node_ratio": round(edge_node_ratio, 2),
            "top_states": {k: v for k, v in state_counter.most_common(3)},
            "breakdown": breakdown,
        }

        extra = {}
        if missing_list:
            extra["missing_states"] = missing_list
        if illegal_count:
            extra["illegal_transitions"] = illegal_count
        if novel_states:
            extra["novel_states"] = novel_states
        if extra:
            result["extra"] = extra

        results.append(result)

    results.sort(key=lambda r: -r["score"])
    return results


def score_traces_from_jsonl(jsonl_path: str | Path, **kwargs) -> list[dict]:
    graphs = []
    with open(jsonl_path) as f:
        for line in f:
            graphs.append(json.loads(line))
    return score_traces(graphs, **kwargs)


def score_lines(
    graphs: list[dict],
    *,
    baseline: dict | None = None,
    legal_transitions: set[tuple[str, str]] | None = None,
) -> list[dict]:
    known_states = set(baseline.get("known_states", [])) if baseline else set()

    line_results = []
    for g in graphs:
        nodes = g.get("nodes", [])
        if len(nodes) < 2:
            continue

        line_scores = []
        prev_node = nodes[0]
        prev_sid = prev_node["features"].get("state_id", "")

        for i in range(1, len(nodes)):
            curr_node = nodes[i]
            curr_sid = curr_node["features"].get("state_id", "")
            curr_tid = curr_node["features"].get("template_id", "")

            transition = (prev_sid, curr_sid) if prev_sid and curr_sid else None

            score = 0.0
            reasons: list[str] = []

            if transition and legal_transitions and transition not in legal_transitions:
                score += 1.0
                reasons.append("illegal_transition")

            if baseline and curr_sid and curr_sid not in known_states:
                score += 1.0
                reasons.append("novel_state")

            if not curr_sid:
                score += 0.5
                reasons.append("unmapped_template")

            if score > 0:
                line_scores.append({
                    "event_id": curr_node["id"],
                    "template_id": curr_tid,
                    "transition": f"{prev_sid} → {curr_sid}" if transition else "?",
                    "score": round(score, 2),
                    "reason": "+".join(reasons),
                })

            prev_node = curr_node
            prev_sid = curr_sid

        if line_scores:
            line_results.append({
                "trace_id": g["trace_id"],
                "n_events": len(nodes),
                "n_anomalous_lines": len(line_scores),
                "anomalous_lines": line_scores,
            })

    line_results.sort(key=lambda r: -r["n_anomalous_lines"])
    return line_results


def pipeline_score(
    events_csv: str | Path,
    template_state_map_csv: str | Path,
    *,
    graph_output: str | Path | None = None,
    baseline: dict | None = None,
    required_states: list[str] | None = None,
    legal_transitions: set[tuple[str, str]] | None = None,
    line_report_output: str | Path | None = None,
) -> list[dict]:
    graphs = build_graphs(events_csv, template_state_map_csv)
    if graph_output:
        write_graphs_jsonl(graphs, graph_output)

    trace_scores = score_traces(graphs, baseline=baseline, required_states=required_states,
                                legal_transitions=legal_transitions)

    if line_report_output:
        line_results = score_lines(graphs, baseline=baseline, legal_transitions=legal_transitions)
        report = {
            "summary": {
                "n_traces_with_anomalies": len(line_results),
                "n_total_anomalous_lines": sum(r["n_anomalous_lines"] for r in line_results),
            },
            "anomalous_lines_by_trace": line_results,
        }
        Path(line_report_output).parent.mkdir(parents=True, exist_ok=True)
        with open(line_report_output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return trace_scores