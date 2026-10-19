"""
Central configuration for the log anomaly detection pipeline.

All paths are relative to the project root. Override via environment variables
or by editing this file directly.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Data directories ---
RAW_DIR = Path(os.environ.get("LAD_RAW_DIR", ROOT / "data" / "raw"))
CLUSTERED_DIR = Path(os.environ.get("LAD_CLUSTERED_DIR", ROOT / "data" / "clustered"))
PROCESSED_DIR = Path(os.environ.get("LAD_PROCESSED_DIR", ROOT / "data" / "processed"))
FSM_DIR = PROCESSED_DIR / "fsm"
GRAPHS_DIR = Path(os.environ.get("LAD_GRAPHS_DIR", ROOT / "data" / "graphs"))
REPORTS_DIR = Path(os.environ.get("LAD_REPORTS_DIR", ROOT / "data" / "reports"))

# --- Logmine parameters ---
LOGMINE_BIN = os.environ.get("LOGMINE_BIN", str(ROOT.parent / "Logmine_" / "logmine" / "logmine"))
LOGMINE_MAX_DISTANCE = float(os.environ.get("LOGMINE_M", "0.3"))
LOGMINE_MIN_MEMBERS = int(os.environ.get("LOGMINE_I", "2"))
LOGMINE_SEPARATOR = os.environ.get("LOGMINE_P", "---")

# --- LLM parameters ---
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# --- State discovery parameters ---
MAX_TEMPLATES = int(os.environ.get("LAD_MAX_TEMPLATES", "500"))
MAX_TRACES = int(os.environ.get("LAD_MAX_TRACES", "80"))
MAX_TRACE_LENGTH = int(os.environ.get("LAD_MAX_TRACE_LENGTH", "80"))
MAX_ITERATIONS = int(os.environ.get("LAD_MAX_ITERATIONS", "3"))


def ensure_dirs() -> None:
    for d in [RAW_DIR, CLUSTERED_DIR, PROCESSED_DIR, FSM_DIR, GRAPHS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def parsed_events_path(dataset: str) -> Path:
    return PROCESSED_DIR / f"{dataset}_parsed_events.csv"


def state_map_path(dataset: str) -> Path:
    return FSM_DIR / f"{dataset}_template_state_map.csv"


def graphs_path(dataset: str) -> Path:
    return GRAPHS_DIR / f"{dataset}_graphs.jsonl"


def report_path(name: str = "score_report") -> Path:
    return REPORTS_DIR / f"{name}.json"