"""
Format converter: transforms Logmine clustered output into structured
parsed_events.csv compatible with the scoring pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from config import CLUSTERED_DIR, PROCESSED_DIR, RAW_DIR, parsed_events_path


# ── Token helpers ──────────────────────────────────────────────────────────

def _stable_tid(text: str) -> str:
    return "TMP_" + hashlib.md5(text.encode()).hexdigest().upper()[:10]


def _is_variable(token: str) -> bool:
    t = token.strip("(),;:[]")
    if re.match(r"^-?\d+$", t):
        return True
    if re.match(r"^-?\d+\.\d+$", t):
        return True
    if re.match(r"^0x[0-9A-Fa-f]+$", t):
        return True
    if re.match(r"^[A-Z]+_\d+$", t):
        return True
    if re.match(r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$", t):
        return True
    if re.match(r"^\d{10,}$", t):
        return True
    if re.match(r"^\d+\.\d+\.\d+\.\d+$", t):
        return True
    if re.match(r"^\[\d+/\d+\]$", t):
        return True
    return False


# ── Cluster line parser ─────────────────────────────────────────────────────

_CLUSTER_RE = re.compile(
    r"^(\d+)\s+"
    r"(\d{4}-\d{2}-\d{2})\s+"
    r"(\d{2}:\d{2}:\d{2}|-{3})\s+"
    r"(\w+)\s+"
    r"([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}|-{3})\s+"
    r"(.*)$"
)

_ORIG_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})\s+"
    r"(\d{2}:\d{2}:\d{2})\s+"
    r"(\w+)\s+"
    r"([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})\s+"
    r"(.*)$"
)


def _parse_cluster(line: str) -> dict | None:
    m = _CLUSTER_RE.match(line.strip())
    if not m:
        return None
    message = m.group(6)
    tokens = message.split()
    fixed_tokens = []
    tmpl_parts = []
    for tok in tokens:
        if tok == "---" or _is_variable(tok):
            tmpl_parts.append("<*>")
        else:
            fixed_tokens.append(tok)
            tmpl_parts.append(tok)
    if not fixed_tokens:
        return None
    template_text = " ".join(tmpl_parts)
    return {
        "count": int(m.group(1)),
        "fixed_tokens": fixed_tokens,
        "fixed_count": len(fixed_tokens),
        "template_text": template_text,
        "template_id": _stable_tid(template_text),
    }


def _subsequence_match(fixed: list[str], msg_tokens: list[str]) -> bool:
    if not fixed:
        return False
    mi = 0
    for ft in fixed:
        while mi < len(msg_tokens) and msg_tokens[mi] != ft:
            mi += 1
        if mi >= len(msg_tokens):
            return False
        mi += 1
    return True


def _extract_params(message: str, template_text: str) -> list[dict]:
    params = []
    mt = message.split()
    idx = 0
    for tp in template_text.split():
        if tp == "<*>" and idx < len(mt):
            val = mt[idx]
            if re.match(r"^-?\d+$", val):
                vt = "number"
            elif re.match(r"^[0-9A-Fa-f]{8}-", val):
                vt = "uuid"
            else:
                vt = "string"
            params.append({"key": f"p{idx}", "value": val, "value_type": vt})
        idx += 1
    return params


# ── Main converter ──────────────────────────────────────────────────────────

def convert_clustered_to_events(
    clustered_path: Path | str,
    log_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> Path:
    """
    Convert one day's clustered log to parsed_events.csv.

    Returns the path to the generated CSV.
    """
    clustered_path = Path(clustered_path)
    log_dir = Path(log_dir) if log_dir else RAW_DIR
    output_dir = Path(output_dir) if output_dir else PROCESSED_DIR

    dataset = clustered_path.name.replace("_clustered.log", "")
    log_path = log_dir / f"{dataset}.log"

    if not log_path.exists():
        raise FileNotFoundError(f"Original log not found: {log_path}")

    clusters = []
    for line in clustered_path.read_text(encoding="utf-8").splitlines():
        c = _parse_cluster(line)
        if c:
            clusters.append(c)
    clusters.sort(key=lambda c: c["fixed_count"], reverse=True)

    events = []
    unmatched = 0

    for lnum, raw in enumerate(log_path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        m = _ORIG_RE.match(raw)
        if not m:
            continue
        date_s, time_s, level, uuid, msg = m.groups()
        msg_tokens = msg.split()

        matched = None
        for c in clusters:
            if _subsequence_match(c["fixed_tokens"], msg_tokens):
                matched = c
                break

        if matched is None:
            unmatched += 1
            continue

        trace_id = uuid if uuid != "00000000-0000-0000-0000-000000000000" else f"SYSTEM_{date_s}"
        params = _extract_params(msg, matched["template_text"])

        events.append({
            "trace_id": trace_id,
            "event_id": f"{dataset.upper()}_{lnum:08d}",
            "timestamp": f"{date_s} {time_s}",
            "level": level,
            "component": "",
            "message": msg,
            "template_id": matched["template_id"],
            "template_text": matched["template_text"],
            "parameters": json.dumps(params),
            "source_dataset": dataset,
            "source_path": str(log_path),
            "line_number": str(lnum),
            "raw_text": raw,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = parsed_events_path(dataset)

    fields = [
        "source_dataset", "trace_id", "event_id", "timestamp", "level",
        "component", "message", "template_id", "template_text", "parameters",
        "source_path", "line_number", "raw_text",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(events)

    total = len(events) + unmatched
    pct = unmatched / total * 100 if total else 0
    print(f"  {dataset}: {len(clusters)} clusters, {len(events)} events, "
          f"{unmatched} unmatched ({pct:.1f}%) -> {out_path.name}")

    return out_path


def convert_all(
    clustered_dir: Path | str | None = None,
    log_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> list[Path]:
    """Convert all *_clustered.log files in clustered_dir."""
    src = Path(clustered_dir) if clustered_dir else CLUSTERED_DIR
    outputs = []
    for fpath in sorted(src.glob("*_clustered.log")):
        outputs.append(convert_clustered_to_events(fpath, log_dir=log_dir, output_dir=output_dir))
    return outputs