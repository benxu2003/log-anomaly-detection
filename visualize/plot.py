"""
Graph visualization: renders state transition graphs using networkx + matplotlib.
Deadlocks (edge/node ratio = 1.0) use circular layout; normal graphs use spring layout.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def load_graph(jsonl_path: str | Path, trace_id: str | None = None) -> dict:
    with open(jsonl_path) as f:
        for line in f:
            g = json.loads(line)
            if trace_id is None or g["trace_id"] == trace_id:
                return g
    raise ValueError(f"Trace '{trace_id}' not found in {jsonl_path}")


def plot_graph(graph: dict, output_path: str | Path | None = None):
    nodes = graph.get("nodes", [])
    if not nodes:
        print("Empty graph, nothing to plot.")
        return

    trace_id = graph["trace_id"]
    state_counter: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    prev_state: str | None = None

    for node in nodes:
        sid = node["features"].get("state_id", "")
        if sid:
            state_counter[sid] += 1
        if prev_state and sid and prev_state != sid:
            edge_counts[(prev_state, sid)] += 1
        prev_state = sid if sid else prev_state

    total_events = sum(state_counter.values())
    n_states = len(state_counter)
    n_edges = len(edge_counts)
    ratio = n_edges / max(n_states, 1)
    is_deadlock = ratio == 1.0 and n_states > 1

    G = nx.DiGraph()
    for state, count in state_counter.items():
        G.add_node(state, count=count)
    for (src, dst), count in edge_counts.items():
        G.add_edge(src, dst, weight=count)

    if is_deadlock:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=2.5, iterations=100)

    fig, ax = plt.subplots(figsize=(12, 8))
    max_count = max(state_counter.values())
    node_sizes = [2000 + 4000 * (state_counter[n] / max_count) for n in G.nodes()]
    node_colors = plt.cm.Set3([i / max(len(G.nodes()), 1) for i in range(len(G.nodes()))])

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           edgecolors="#333", linewidths=2, ax=ax)

    max_edge = max(edge_counts.values()) if edge_counts else 1
    edge_widths = [1 + 5 * (edge_counts[(u, v)] / max_edge) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color="#666", arrows=True,
                           arrowsize=20, width=edge_widths, ax=ax,
                           connectionstyle="arc3,rad=0.1")

    labels = {n: f"{n}\n({state_counter[n]})" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight="bold", ax=ax)

    edge_labels = {(u, v): f"x{edge_counts[(u, v)]}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)

    status = "DEADLOCK (ratio=1.0)" if is_deadlock else "NORMAL"
    ax.set_title(
        f"{trace_id}\n"
        f"{total_events} events | {n_states} states | {n_edges} edges | "
        f"ratio={ratio:.2f} | {status}",
        fontsize=12, fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_all(jsonl_path: str | Path, output_dir: str | Path,
             top_n: int = 30) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= top_n:
                break
            g = json.loads(line)
            path = output_dir / f"{g['trace_id']}.png"
            plot_graph(g, path)
            saved.append(path)
    return saved