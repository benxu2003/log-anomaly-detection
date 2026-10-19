#!/usr/bin/env python3
"""
Log Anomaly Detection Pipeline
==============================

Single entry point for the full pipeline:

  python pipeline.py all          # run everything
  python pipeline.py cluster      # Step 1: logmine clustering
  python pipeline.py convert      # Step 2: format conversion
  python pipeline.py discover     # Step 3: LLM state discovery
  python pipeline.py score        # Step 4: anomaly scoring
  python pipeline.py plot         # Step 5: visualize graphs

Author: Yiming Xu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from config import (
    CLUSTERED_DIR,
    GRAPHS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    REPORTS_DIR,
    ensure_dirs,
    graphs_path,
    parsed_events_path,
    report_path,
    state_map_path,
)
from preprocess import cluster_logs, convert_clustered_to_events
from states import LLMClient, discover_states, save_state_map
from detect import pipeline_score, evaluate_scores
from visualize import plot_graph, load_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("pipeline")


# ── Step functions ──────────────────────────────────────────────────────────

def step_cluster(args: argparse.Namespace) -> None:
    ensure_dirs()
    log_files = sorted(Path(args.raw_dir).glob("*.log"))
    if not log_files:
        print(f"No .log files found in {args.raw_dir}")
        return
    for lf in log_files:
        print(f"Clustering {lf.name} ...")
        result = cluster_logs(lf)
        if result:
            print(f"  -> {result}")
        else:
            print(f"  FAILED")


def step_convert(args: argparse.Namespace) -> None:
    ensure_dirs()
    clustered = sorted(Path(args.clustered_dir).glob("*_clustered.log"))
    if not clustered:
        print(f"No *_clustered.log files in {args.clustered_dir}")
        return
    for cp in clustered:
        convert_clustered_to_events(cp, log_dir=args.raw_dir, output_dir=args.processed_dir)


def step_discover(args: argparse.Namespace) -> None:
    ensure_dirs()
    client = LLMClient(model=args.model)
    parsed = sorted(Path(args.processed_dir).glob("*_parsed_events.csv"))
    if not parsed:
        print(f"No *_parsed_events.csv in {args.processed_dir}")
        return
    for pp in parsed:
        dataset = pp.name.replace("_parsed_events.csv", "")
        out = state_map_path(dataset)
        if out.exists() and not args.force:
            print(f"  Skip {dataset} (already exists, use --force to redo)")
            continue
        print(f"Discovering states for {dataset} ...")
        state_map = discover_states(pp, client, max_templates=args.max_templates)
        save_state_map(state_map, out)
        n_states = len(set(state_map.values()))
        print(f"  -> {n_states} states, {len(state_map)} templates -> {out}")


def step_score(args: argparse.Namespace) -> None:
    ensure_dirs()
    processed_dir = Path(args.processed_dir)
    fsm_dir = processed_dir / "fsm"
    parsed = sorted(processed_dir.glob("*_parsed_events.csv"))
    if not parsed:
        print(f"No *_parsed_events.csv in {processed_dir}")
        return

    all_results = []
    all_line_reports = []
    for events_csv in parsed:
        dataset = events_csv.name.replace("_parsed_events.csv", "")
        smap = fsm_dir / f"{dataset}_template_state_map.csv"
        if not smap.exists():
            print(f"  Skip {dataset}: no state map at {smap}")
            continue

        line_report = REPORTS_DIR / f"{dataset}_line_anomalies.json"
        results = pipeline_score(
            str(events_csv), str(smap),
            graph_output=str(graphs_path(dataset)),
            line_report_output=str(line_report),
        )
        for r in results:
            r["date"] = dataset
        all_results.extend(results)
        if results:
            print(f"  {dataset}: {len(results)} traces, top={results[0]['score']:.2f}")

        if line_report.exists():
            with open(line_report) as f:
                lr = json.load(f)
            all_line_reports.append(lr)
            n_anom = lr["summary"]["n_total_anomalous_lines"]
            n_traces = lr["summary"]["n_traces_with_anomalies"]
            print(f"    -> line anomalies: {n_anom} lines in {n_traces} traces -> {line_report}")

    evaluation = evaluate_scores(all_results)
    report = {"evaluation": evaluation, "sorted_scores": all_results}
    rp = report_path(args.report_name)
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {len(all_results)} traces")
    print(f"  p90: {evaluation.get('threshold_p90', 'N/A')}")
    print(f"  flagged: {evaluation.get('n_flagged', 'N/A')}")
    print(f"  report: {rp}")

    if all_line_reports:
        total_lines = sum(lr["summary"]["n_total_anomalous_lines"] for lr in all_line_reports)
        total_traces = sum(lr["summary"]["n_traces_with_anomalies"] for lr in all_line_reports)
        print(f"  line anomalies: {total_lines} lines in {total_traces} traces")

    print(f"\nTop 10:")
    print(f"{'#':<4} {'date':<20} {'trace_id':<45} {'score':>8} {'states':>6} {'edges':>5} {'ratio':>6}")
    print("-" * 100)
    for i, r in enumerate(all_results[:10], 1):
        deadlock = " DEADLOCK" if r["edge_node_ratio"] == 1.0 and r["n_states"] > 1 else ""
        print(f"{i:<4} {r['date']:<20} {r['trace_id']:<45} {r['score']:8.2f} {r['n_states']:>6} {r['n_edges']:>5} {r['edge_node_ratio']:>6.2f}{deadlock}")


def step_plot(args: argparse.Namespace) -> None:
    graph_file = Path(args.graph_file)
    if not graph_file.exists():
        print(f"Graph file not found: {graph_file}")
        return
    g = load_graph(graph_file, args.trace)
    out = args.output or f"{g['trace_id']}.png"
    plot_graph(g, out)


def step_all(args: argparse.Namespace) -> None:
    step_cluster(args)
    step_convert(args)
    step_discover(args)
    step_score(args)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log Anomaly Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py all
  python pipeline.py cluster --raw-dir data/raw
  python pipeline.py score --report-name my_report
        """,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_all = sub.add_parser("all", help="Run the full pipeline")
    _add_path_args(p_all)
    _add_discover_args(p_all)

    p_cluster = sub.add_parser("cluster", help="Step 1: Logmine clustering")
    _add_path_args(p_cluster)

    p_convert = sub.add_parser("convert", help="Step 2: Format conversion")
    _add_path_args(p_convert)

    p_discover = sub.add_parser("discover", help="Step 3: LLM state discovery")
    _add_path_args(p_discover)
    _add_discover_args(p_discover)

    p_score = sub.add_parser("score", help="Step 4: Anomaly scoring")
    _add_path_args(p_score)
    p_score.add_argument("--report-name", default="score_report", help="Report name")

    p_plot = sub.add_parser("plot", help="Step 5: Visualize graphs")
    p_plot.add_argument("graph_file", help="Graph JSONL file")
    p_plot.add_argument("--trace", "-t", default=None, help="Specific trace_id")
    p_plot.add_argument("--output", "-o", default=None, help="Output image path")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    commands = {
        "all": step_all,
        "cluster": step_cluster,
        "convert": step_convert,
        "discover": step_discover,
        "score": step_score,
        "plot": step_plot,
    }
    commands[args.command](args)


def _add_path_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--raw-dir", default=str(RAW_DIR), help="Raw logs directory")
    p.add_argument("--clustered-dir", default=str(CLUSTERED_DIR), help="Clustered logs directory")
    p.add_argument("--processed-dir", default=str(PROCESSED_DIR), help="Processed CSV directory")


def _add_discover_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, help="LLM model name")
    p.add_argument("--max-templates", type=int, default=500)
    p.add_argument("--force", action="store_true", help="Redo existing state maps")


if __name__ == "__main__":
    main()
