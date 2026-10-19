"""
Logmine clustering: wraps the external logmine binary to cluster raw log files
into template patterns.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config import (
    CLUSTERED_DIR,
    LOGMINE_BIN,
    LOGMINE_MAX_DISTANCE,
    LOGMINE_MIN_MEMBERS,
    LOGMINE_SEPARATOR,
    RAW_DIR,
)


def _clean_encoding(input_path: Path, output_path: Path) -> bool:
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    for enc in encodings:
        try:
            content = input_path.read_text(encoding=enc, errors="ignore")
            output_path.write_text(content, encoding="utf-8")
            return True
        except Exception:
            continue
    return False


def cluster_logs(log_file: Path | str) -> Path | None:
    """
    Run logmine on a single log file. Returns the path to the clustered output,
    or None on failure.
    """
    log_file = Path(log_file)
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    dataset = log_file.name.replace(".log", "")
    output_path = CLUSTERED_DIR / f"{dataset}_clustered.log"
    tmp = Path(f"/tmp/lad_{log_file.name}")

    try:
        if not _clean_encoding(log_file, tmp):
            return None

        cmd = [
            sys.executable, LOGMINE_BIN, str(tmp),
            "-m", str(LOGMINE_MAX_DISTANCE),
            "-i", str(LOGMINE_MIN_MEMBERS),
            f"-p={LOGMINE_SEPARATOR}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"Logmine error: {result.stderr}")
            return None

        output_path.write_text(result.stdout, encoding="utf-8")
        return output_path

    except subprocess.TimeoutExpired:
        print(f"Timeout: {log_file.name}")
        return None
    finally:
        if tmp.exists():
            tmp.unlink()


def cluster_all(raw_dir: Path | str | None = None) -> list[Path]:
    """Cluster all .log files in raw_dir."""
    src = Path(raw_dir) if raw_dir else RAW_DIR
    CLUSTERED_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []
    for log_file in sorted(src.glob("*.log")):
        result = cluster_logs(log_file)
        if result:
            outputs.append(result)
    return outputs